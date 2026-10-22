#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Build/query the pinned C++ CLI with full warmup, independent validation and retained raw results."""
from __future__ import annotations
import argparse
from array import array
import bisect
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import time


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def execute(command, environment, log):
    started = time.monotonic()
    with log.open('w') as stream:
        result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    record = {'command': command, 'elapsed_seconds': time.monotonic() - started,
              'exit_code': result.returncode, 'log': str(log)}
    if result.returncode:
        raise RuntimeError(json.dumps(record))
    return record


def uint32_bin(path):
    data = path.read_bytes()
    n, k = struct.unpack_from('<II', data)
    values = array('I')
    values.frombytes(data[8:8 + n * k * 4])
    if len(values) != n * k:
        raise ValueError(f'truncated uint32 array: {path}')
    return n, k, values


class Validator:
    def __init__(self, data, points, k):
        self.points, self.k = points, k
        self.n, gt_k, self.truth = uint32_bin(data / 'groundtruth.bin')
        if gt_k != k:
            raise ValueError('K must equal exact ground-truth width for matched strict ID recall')
        self.labels = [int(line) for line in (data / 'query_labels.txt').read_text().splitlines()]
        if len(self.labels) != self.n:
            raise ValueError('query label count differs from ground truth')
        self.postings = {}
        for label in set(self.labels):
            count, width, values = uint32_bin(data / f'label_{label}.u32bin')
            if width != 1 or len(values) != count:
                raise ValueError('invalid posting-list dimensions')
            self.postings[label] = values
        manifest = json.loads((data / 'manifest.json').read_text())
        self.identity = ':'.join(manifest['artifacts'][name]['sha256'] for name in
                                 ['query.u8bin', 'query_labels.u32bin', 'groundtruth.bin', 'labels.txt'])

    def validate(self, path, query_metrics=None):
        n, k, ids = uint32_bin(path)
        if (n, k) != (self.n, self.k):
            raise ValueError('returned ID file dimensions differ from query contract')
        violations = short = duplicates = 0
        counts_known = bool(query_metrics) and all("returned_results" in q for q in query_metrics)
        recalls = []
        grouped = collections.defaultdict(list)
        for i, label in enumerate(self.labels):
            available = query_metrics[i].get('returned_results', k) if query_metrics else k
            if type(available) is not int or not 0 <= available <= k:
                raise ValueError('returned_results must be an integer in [0, K]')
            row = ids[i * k:i * k + available]
            unique = set(row)
            duplicates += available - len(unique)
            short += int(available < k)
            posting = self.postings[label]
            for result in row:
                offset = bisect.bisect_left(posting, result)
                if result >= self.points or offset == len(posting) or posting[offset] != result:
                    violations += 1
            recall = len(unique.intersection(self.truth[i * k:(i + 1) * k])) / k
            recalls.append(recall)
            grouped[label].append(recall)
        slices = [{'label': label, 'eligible': len(self.postings[label]), 'n': len(values),
                   'recall': sum(values) / len(values)} for label, values in sorted(grouped.items())]
        if query_metrics:
            for entry in slices:
                latencies = sorted(query_metrics[i]['latency_us'] for i, label in enumerate(self.labels)
                                   if label == entry['label'])
                entry['mean_us'] = sum(latencies) / len(latencies)
                entry['p50_us'] = latencies[max(0, math.ceil(len(latencies) * .5) - 1)]
                entry['p95_us'] = latencies[max(0, math.ceil(len(latencies) * .95) - 1)]
                entry['p99_us'] = latencies[max(0, math.ceil(len(latencies) * .99) - 1)]
        return {'recall': sum(recalls) / n, 'filter_violations': violations, 'short_results': short if counts_known else None,
                'duplicate_results': duplicates, 'labels': slices, 'strict_id_recall': True}


def normalize(args, prefix, tag, threads, l, repeat, record, validator, output):
    stats_path = output / (tag + '.jsonl')
    raw = [json.loads(line) for line in stats_path.read_text().splitlines()] if stats_path.exists() else []
    summaries = [row for row in raw if row['kind'] == 'summary']
    query_metrics = sorted((row for row in raw if row['kind'] == 'query'), key=lambda row: row['query_index'])
    if query_metrics and (len(query_metrics) != validator.n or
                          [r['query_index'] for r in query_metrics] != list(range(validator.n))):
        raise ValueError('raw query metrics are incomplete')
    legacy = None
    if not summaries:
        for line in Path(record['log']).read_text().splitlines():
            parts = line.split()
            if len(parts) == 9 and parts[0] == str(l):
                legacy = list(map(float, parts))
        if legacy is None:
            raise ValueError('could not find C++ CLI measurement row')
    result_file = output / f'{tag}_{l}_idx_uint32.bin'
    result = validator.validate(result_file, query_metrics or None)
    row = {'implementation': 'cpp_macos_reference' if platform.system() == 'Darwin' else 'cpp_native_linux',
           'index_prefix': str(prefix), 'prepared_identity': validator.identity,
           'prepared': str(args.prepared_data.resolve()), 'mode': 'single', 'starts': 'label',
           'n_base': validator.points, 'queries': validator.n, 'k': args.k, 'l': l,
           'beam': args.beam, 'threads': threads, 'repeat': repeat,
           'timing_authoritative': args.timing_authoritative,
           'cache_bypass': True, 'shared_cache_nodes': args.cache_nodes,
           'storage': 'positional_pread' if platform.system() == 'Darwin' else 'linux_aio_direct',
           'warmup_queries': summaries[0].get('warmup_queries', 0) if summaries else 0,
           'raw_stats': str(stats_path) if raw else None,
           'result_ids': str(result_file), 'result_ids_sha256': sha(result_file), **result}
    if summaries:
        row['qps'] = summaries[0]['qps']
        row['latency_contract'] = summaries[0].get('latency_contract', 'legacy_inner_search')
        row['cpp_legacy_tie_aware_recall'] = summaries[0]['recall_percent'] / 100
        latencies = sorted(q['latency_us'] for q in query_metrics)
        row['mean_us'] = sum(latencies) / len(latencies)
        for name, percentile in [('p50_us', .5), ('p95_us', .95), ('p99_us', .99), ('p999_us', .999)]:
            row[name] = latencies[max(0, math.ceil(len(latencies) * percentile) - 1)]
        row['reader_requests'] = sum(q['read_requests'] for q in query_metrics)
        row['reader_bytes'] = sum(q['requested_read_bytes'] for q in query_metrics)
        row['mean_cmps'] = sum(q['pq_distance_comparisons'] for q in query_metrics) / validator.n
        row['mean_full_cmps'] = sum(q['full_distance_comparisons'] for q in query_metrics) / validator.n
        if all('inner_search_latency_us' in q for q in query_metrics):
            row['inner_mean_us'] = sum(q['inner_search_latency_us'] for q in query_metrics) / validator.n
        row['mean_hops'] = sum(q['hops'] for q in query_metrics) / validator.n
        row['label_checks'] = sum(q['label_checks'] for q in query_metrics)
        if all('reader_time_us' in q for q in query_metrics):
            row['reader_time_us'] = sum(q['reader_time_us'] for q in query_metrics)
    else:
        row.update({'qps': legacy[2], 'mean_us': legacy[3], 'p50_us': None, 'p95_us': None,
                    'p99_us': None, 'p999_us': legacy[4], 'cpp_legacy_tie_aware_recall': legacy[8] / 100,
                    'latency_contract': 'legacy_inner_search'})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--prepared-data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--name', default='yfcc_R64_L100')
    parser.add_argument('--index-prefix', type=Path, help='Existing index prefix, independently of output names')
    parser.add_argument('--build-threads', type=int, default=8)
    parser.add_argument('--build-dram-gib', type=float, default=16)
    parser.add_argument('--search-threads', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--l', type=int, nargs='+', default=[20, 40, 80, 160])
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--beam', type=int, default=4)
    parser.add_argument('--cache-nodes', type=int, default=0)
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--query-only', action='store_true')
    parser.add_argument('--timing-authoritative', action='store_true',
                        help='Mark a deliberately quiescent run as eligible for performance comparisons')
    args = parser.parse_args()
    if args.build_only and args.query_only:
        parser.error('--build-only and --query-only are mutually exclusive')
    if args.reps < 1 or any(t < 1 for t in args.search_threads) or any(l < args.k for l in args.l):
        parser.error('positive threads/reps and L >= K are required')
    if Path(args.name).name != args.name:
        parser.error('--name must be a file basename; pass index paths with --index-prefix')
    data = args.prepared_data.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = args.index_prefix.resolve() if args.index_prefix else output / args.name
    bin_dir = args.bin_dir.resolve()
    base = data / 'base.u8bin'
    with base.open('rb') as stream:
        points, dimensions = struct.unpack('<II', stream.read(8))
    if dimensions < 32:
        parser.error('this shared PQ32 experiment requires dimensions >= 32')
    environment = os.environ.copy()
    environment.update({'VECLIB_MAXIMUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OMP_TOOL': 'disabled'})
    environment.pop('DISKANN_REFERENCE_STATS_JSONL', None)
    environment.pop('DISKANN_REFERENCE_WARMUP_QUERIES', None)
    manifest = {'reference_kind': 'macos_portable_reference' if platform.system() == 'Darwin' else 'native_upstream_cli',
                'binaries': {name: sha(bin_dir / name) for name in ['build_disk_index', 'search_disk_index']},
                'input_points': points, 'dimensions': dimensions, 'metric': 'uint8_l2',
                'pq_bytes': 32, 'dummy_replica_threshold': 0,
                'timing_authoritative': args.timing_authoritative,
                'process_cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                'in_process_warmup': 'all queries, matching L/K/threads/filter, before each measured repeat',
                'environment': {k: environment[k] for k in
                                ['VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'OMP_TOOL', 'LD_LIBRARY_PATH']
                                if k in environment}, 'runs': []}
    if not args.query_only:
        if Path(str(prefix) + '_disk.index').exists():
            parser.error('index exists; use a different --name or --query-only')
        command = [str(bin_dir / 'build_disk_index'), '--data_type', 'uint8', '--dist_fn', 'l2',
                   '--data_path', str(base), '--index_path_prefix', str(prefix),
                   '--label_file', str(data / 'labels.txt'), '--label_type', 'uint',
                   '--FilteredLbuild', '100', '--filter_threshold', '0', '--QD', '32',
                   '-R', '64', '-L', '100', '-T', str(args.build_threads),
                   '-B', str(32 * points / 1024**3), '-M', str(args.build_dram_gib),
                   '--PQ_disk_bytes', '0', '--build_PQ_bytes', '0']
        environment['OMP_NUM_THREADS'] = str(args.build_threads)
        manifest['build'] = execute(command, environment, output / (args.name + '.build.log'))
        with Path(str(prefix) + '_pq_compressed.bin').open('rb') as stream:
            code_points, code_bytes = struct.unpack('<II', stream.read(8))
        if (code_points, code_bytes) != (points, 32):
            raise RuntimeError(f'unexpected PQ code dimensions: {(code_points, code_bytes)}')
    if not args.build_only:
        normalized_path = output / (args.name + '.normalized.jsonl')
        if normalized_path.exists():
            parser.error(f'normalized results already exist: {normalized_path}')
        validator = Validator(data, points, args.k)
        with normalized_path.open('w') as normalized:
            for threads in args.search_threads:
                for l in args.l:
                    config = f'{args.name}.t{threads}.L{l}'
                    environment['OMP_NUM_THREADS'] = str(threads)
                    base_command = [str(bin_dir / 'search_disk_index'), '--data_type', 'uint8', '--dist_fn', 'l2',
                                    '--index_path_prefix', str(prefix), '--query_file', str(data / 'query.u8bin'),
                                    '--query_filters_file', str(data / 'query_labels.txt'), '--label_type', 'uint',
                                    '--gt_file', str(data / 'groundtruth.bin'), '-K', str(args.k), '-L', str(l),
                                    '-W', str(args.beam), '-T', str(threads), '--num_nodes_to_cache', str(args.cache_nodes)]
                    warmup_log = output / (config + '.warmup.log')
                    if warmup_log.exists():
                        parser.error(f'warmup output exists: {warmup_log}')
                    warmup = execute(base_command + ['--result_path', str(output / (config + '.warmup'))],
                                     environment, warmup_log)
                    for repeat in range(args.reps):
                        tag = f'{config}.r{repeat}'
                        stats, log = output / (tag + '.jsonl'), output / (tag + '.log')
                        if stats.exists() or log.exists():
                            parser.error(f'measurement output exists for: {tag}')
                        measured_environment = environment.copy()
                        measured_environment['DISKANN_REFERENCE_STATS_JSONL'] = str(stats)
                        measured_environment['DISKANN_REFERENCE_WARMUP_QUERIES'] = '1'
                        record = execute(base_command + ['--result_path', str(output / tag)], measured_environment, log)
                        row = normalize(args, prefix, tag, threads, l, repeat, record, validator, output)
                        normalized.write(json.dumps(row) + '\n')
                        normalized.flush()
                        record.update({'warmup': warmup, 'normalized': row})
                        manifest['runs'].append(record)
                        print(json.dumps({key: row[key] for key in ['implementation', 'threads', 'l', 'repeat',
                                                                   'recall', 'filter_violations', 'short_results', 'qps', 'p99_us']}), flush=True)
        manifest['normalized_results'] = str(normalized_path)
    path = output / (args.name + ('.query' if args.query_only else '') + '.manifest.json')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'index_prefix': str(prefix), 'manifest': str(path)}))


if __name__ == '__main__':
    main()
