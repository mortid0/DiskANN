#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Run alternating frozen-policy/C++ held-out pairs on the existing Linux indexes.

Raw outputs are never edited. Derived merged rows receive global paired-repeat IDs,
with original repeat IDs and aggregate-file hashes retained for independent checks.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

FROZEN_PLAN = 'df42e65440af0a891185913e91f5f22cb6808c9251b2c1d8f6ca968d5bda2e59'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'prepared', 'out', 'native-binary', 'source-archive', 'plan'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=6)
    parser.add_argument('--fixed-control', action='store_true')
    parser.add_argument('--timing-authoritative', action='store_true')
    args = parser.parse_args()
    root, prepared, out = args.root.resolve(), args.prepared.resolve(), args.out.resolve()
    native, archive, plan = args.native_binary.resolve(), args.source_archive.resolve(), args.plan.resolve()
    if args.repeats < 3 or not args.timing_authoritative:
        parser.error('At least three pairs and explicit authoritative timing are required')
    affinity = sorted(os.sched_getaffinity(0))
    if affinity != list(range(8)) or sha(plan) != FROZEN_PLAN:
        parser.error('Requires physical cores0..7 and the unchanged original policy')
    query_count, dimensions = struct.unpack('<II', (prepared / 'query.u8bin').read_bytes()[:8])
    if query_count <= 2000 or dimensions != 192:
        parser.error('Expected larger192-dimensional held-out corpus')
    cpp_prefix = root / 'indexes/yfcc10m_stitched_R32_R128_L100'
    native_root = root / 'indexes/rust-native-10m-frozen'
    generation = native_root / 'generation-00000000000000000001'
    cpp_bin = root / 'cpp/build-native/apps'
    runner = root / 'rust/tools/single-hop-benchmark/cpp-port/run_reference.py'
    native_assets = {'current_sha256': native_root / 'CURRENT',
                     'catalog_sha256': generation / 'CATALOG',
                     'journal_sha256': native_root / 'updates.wal'}
    suffixes = ('_disk.index', '_disk.index_labels.txt', '_disk.index_labels_map.txt',
                '_disk.index_labels_to_medoids.txt', '_pq_pivots.bin', '_pq_compressed.bin')
    cpp_assets = {suffix: Path(str(cpp_prefix) + suffix) for suffix in suffixes}
    index_native_before = {key: sha(path) for key, path in native_assets.items()}
    index_cpp_before = {key: sha(path) for key, path in cpp_assets.items()}
    binaries_before = {'native': sha(native), 'cpp': sha(cpp_bin / 'search_disk_index')}
    archive_before = sha(archive)
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = out / 'paired-run-manifest.json'
    manifest = {'schema_version': 1, 'status': 'running', 'started': now(),
                'host': 'alex@100.83.201.102', 'output_root': str(out), 'cpu_affinity': affinity,
                'prepared': str(prepared), 'prepared_manifest_sha256': sha(prepared / 'manifest.json'),
                'queries': query_count, 'k': 10, 'nominal_l': 64, 'pairs': args.repeats,
                'order_protocol': 'CPP first in even pairs; Rust first in odd pairs; serial jobs with full same-process warmup',
                'plan_before_sha256': sha(plan), 'runs': [], 'fixed_control': None}
    environment = os.environ.copy()
    environment.update(OMP_TOOL='disabled', MKL_NUM_THREADS='1',
                       LD_LIBRARY_PATH=str(root / 'cpp-deps/root/usr/lib/x86_64-linux-gnu') + ':/usr/lib/llvm-18/lib')
    dump(manifest_path, manifest)
    native_rows, cpp_rows = [], []
    native_header = None

    def execute(engine, pair, order, control=False):
        nonlocal native_header
        directory = out / ('fixed-control' if control else f'pair-{pair:02d}')
        directory.mkdir(parents=True, exist_ok=True)
        if engine == 'cpp':
            cpp_out = directory / 'cpp'
            name = f'cpp-heldout-p{pair:02d}'
            command = [sys.executable, str(runner), '--bin-dir', str(cpp_bin),
                       '--prepared-data', str(prepared), '--output-dir', str(cpp_out),
                       '--name', name, '--index-prefix', str(cpp_prefix), '--query-only',
                       '--search-threads', '8', '--l', '64', '--reps', '1', '--k', '10',
                       '--beam', '4', '--cache-nodes', '0', '--timing-authoritative']
            aggregate = cpp_out / (name + '.normalized.jsonl')
        else:
            aggregate = directory / 'rust.jsonl'
            command = [str(native), '--action', 'query', '--root', str(native_root),
                       '--prepared', str(prepared), '--out', str(aggregate), '--threads', '8',
                       '--query-threads', '8', '--l', '64', '--reps', '1', '--limit', str(query_count),
                       '--timing-authoritative', 'false' if control else 'true',
                       '--query-details-dir', str(directory / 'rust-details'),
                       '--evaluation-scope', 'frozen_policy_heldout']
            if not control:
                command += ['--label-l', str(plan), '--allow-unused-plan-labels', 'true']
        record = {'implementation': engine, 'paired_repeat': pair, 'local_repeat': 0,
                  'order_in_pair': order, 'command': command, 'start': now(),
                  'aggregate_path': str(aggregate), 'stdout_log': str(directory / (engine + '.log'))}
        begin = time.monotonic()
        with Path(record['stdout_log']).open('x') as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        record.update(end=now(), elapsed_seconds=time.monotonic() - begin, exit_code=result.returncode)
        if result.returncode:
            manifest['status'] = 'failed'
            manifest['failed_run'] = record
            dump(manifest_path, manifest)
            raise RuntimeError(f'{engine} pair{pair} failed; see {record["stdout_log"]}')
        record['aggregate_sha256'] = sha(aggregate)
        raw = [json.loads(line) for line in aggregate.read_text().splitlines()]
        if engine == 'rust' and not control and native_header is None:
            headers = [row for row in raw if row.get('kind') == 'experiment']
            if len(headers) != 1:
                raise ValueError('Missing native experiment provenance')
            native_header = headers[0]
        rows = [row for row in raw if row.get('kind') == 'query'] if engine == 'rust' else raw
        if len(rows) != 1 or rows[0]['queries'] != query_count or rows[0]['repeat'] != 0:
            raise ValueError('Incomplete or unexpected one-repeat measurement')
        row = dict(rows[0])
        row.update(repeat=pair, paired_repeat=pair, local_repeat=0,
                   original_aggregate=str(aggregate), original_aggregate_sha256=record['aggregate_sha256'])
        if engine == 'rust':
            row['local_raw_queries'] = row['raw_queries']['path']
            row['raw_queries_relative_path'] = str(Path(row['raw_queries']['path']).relative_to(out))
        else:
            row['local_raw_stats'] = row['raw_stats']
            row['local_result_ids'] = row['result_ids']
            row['raw_stats_relative_path'] = str(Path(row['raw_stats']).relative_to(out))
            row['result_ids_relative_path'] = str(Path(row['result_ids']).relative_to(out))
        if control:
            manifest['fixed_control'] = record
            dump(out / 'fixed-control.normalized.json', row)
        else:
            target = native_rows if engine == 'rust' else cpp_rows
            target.append(row)
            with (out / (engine + '.normalized.jsonl')).open('w') as stream:
                if engine == 'rust':
                    stream.write(json.dumps(native_header) + '\n')
                for item in sorted(target, key=lambda item: item['repeat']):
                    stream.write(json.dumps(item) + '\n')
            manifest['runs'].append(record)
        dump(manifest_path, manifest)
        print(json.dumps({'engine': engine, 'paired_repeat': pair, 'control': control,
                          'queries': query_count, 'recall': row['recall'], 'qps': row['qps'],
                          'p99_us': row['p99_us']}), flush=True)

    for pair in range(args.repeats):
        for order, engine in enumerate(('cpp', 'rust') if pair % 2 == 0 else ('rust', 'cpp')):
            execute(engine, pair, order)
    if args.fixed_control:
        execute('rust', args.repeats, 0, control=True)
    native_after = {key: sha(path) for key, path in native_assets.items()}
    cpp_after = {key: sha(path) for key, path in cpp_assets.items()}
    identity = {'schema_version': 1, 'host': manifest['host'], 'output_root': str(out),
                'cpu_affinity': affinity, 'plan_before_sha256': manifest['plan_before_sha256'],
                'plan_after_sha256': sha(plan),
                'native': {'binary_sha256': sha(native), 'binary_before_sha256': binaries_before['native'], 'log_sha256': sha(out / 'rust.normalized.jsonl'),
                           'source_archive': {'path': str(archive), 'sha256': sha(archive), 'before_sha256': archive_before},
                           'index_before': index_native_before, 'index_after': native_after},
                'cpp': {'binary_sha256': sha(cpp_bin / 'search_disk_index'), 'binary_before_sha256': binaries_before['cpp'],
                        'log_sha256': sha(out / 'cpp.normalized.jsonl'),
                        'index_before': index_cpp_before, 'index_after': cpp_after},
                'runner_sha256': sha(Path(__file__)), 'cpp_runner_sha256': sha(runner),
                'prepared_manifest_sha256': manifest['prepared_manifest_sha256']}
    if (native_after != index_native_before or cpp_after != index_cpp_before or sha(plan) != FROZEN_PLAN
            or sha(native) != binaries_before['native']
            or sha(cpp_bin / 'search_disk_index') != binaries_before['cpp']
            or sha(archive) != archive_before):
        manifest['status'] = 'failed_identity_check'
        dump(manifest_path, manifest)
        raise ValueError('Index or policy changed during held-out validation')
    manifest.update(status='complete', ended=now(), identity=identity)
    dump(out / 'identity.json', identity)
    dump(manifest_path, manifest)


if __name__ == '__main__':
    main()
