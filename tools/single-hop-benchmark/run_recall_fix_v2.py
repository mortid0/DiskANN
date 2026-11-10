#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Evaluate the second generic builder candidate with explicit development and fresh validation scopes."""
import argparse
import itertools
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

from run_heldout import FROZEN_PLAN, dump, now, sha

ENGINES = ('cpp', 'baseline', 'candidate')
CPP_SUFFIXES = ('_disk.index', '_disk.index_labels.txt', '_disk.index_labels_map.txt',
                '_disk.index_labels_to_medoids.txt', '_pq_pivots.bin', '_pq_compressed.bin')


def native_identity(root):
    identity = {name: sha(path) for name, path in {
        'current_sha256': root / 'CURRENT',
        'catalog_sha256': root / 'generation-00000000000000000001/CATALOG',
        'journal_sha256': root / 'updates.wal'}.items()}
    generation = root / 'generation-00000000000000000001'
    identity['files_sha256'] = {str(path.relative_to(generation)): sha(path)
                               for path in sorted(generation.rglob('*')) if path.is_file()}
    return identity


def validate_fresh(proof_path, prepared, phase, count):
    proof = json.loads(proof_path.read_text())
    manifest = json.loads((prepared / 'manifest.json').read_text())
    if phase == 'development-recheck':
        # Primary natural queries have already been exposed to candidate 1.
        from run_recall_fix import validate_fresh as validate_primary
        return validate_primary(proof_path, prepared, 'validation', count)
    expected = proof.get('partitions', {}).get(phase, {})
    split = phase.replace('-', '_')
    suffix = 'calibration' if phase == 'secondary-calibration' else 'validation'
    if (proof.get('scope') != 'supplemental_shifted_atomic_label_v2_split'
            or proof.get('policy_sha256') != FROZEN_PLAN
            or manifest.get('source_query_namespace') != 'mixed_public_and_official_private_2727415019_100K'
            or manifest.get('split') != split
            or manifest.get('validation_scope') != f'recall_fix_supplemental_shifted_atomic_label_{suffix}_v2'
            or expected.get('queries') != count
            or expected.get('manifest_sha256') != sha(prepared / 'manifest.json')
            or proof.get('split_protocol_sha256') != manifest.get('split_protocol_sha256')
            or proof.get('original_pool_manifest_sha256') != manifest.get('source_pool_manifest', {}).get('sha256')):
        raise ValueError('Second-candidate split commitment does not match phase, namespace or prepared manifest')
    for name, expected_file in expected.get('artifacts', {}).items():
        path = prepared / name
        if sha(path) != expected_file.get('sha256') or path.stat().st_size != expected_file.get('bytes'):
            raise ValueError(f'Prepared artifact differs from sealed second-candidate split: {name}')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'prepared', 'out', 'plan', 'fresh-proof', 'candidate-source-archive',
                'candidate-build-log'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path)
    parser.add_argument('--phase', choices=('development-recheck', 'secondary-calibration', 'secondary-validation'), required=True)
    parser.add_argument('--candidate-binary', type=Path, required=True)
    parser.add_argument('--quality-only', action='store_true')
    args = parser.parse_args()
    if ((args.phase == 'secondary-calibration' and not args.quality_only)
            or (args.phase == 'secondary-validation' and args.quality_only)):
        parser.error('Calibration is quality-only and fresh validation requires authoritative timings')
    affinity = sorted(os.sched_getaffinity(0))
    if affinity != list(range(8)) or sha(args.plan) != FROZEN_PLAN:
        parser.error('Requires physical cores 0..7 and the unchanged original L policy')
    root, prepared, out = (getattr(args, name).resolve() for name in ('root', 'prepared', 'out'))
    count, dim = struct.unpack('<II', (prepared / 'query.u8bin').read_bytes()[:8])
    if count <= 0 or dim != 192:
        parser.error('Expected a nonempty 192-dimensional query corpus')
    validate_fresh(args.fresh_proof, prepared, args.phase, count)
    baseline_root = root / 'indexes/rust-native-10m-frozen'
    candidate_root = (args.candidate_root.resolve() if args.candidate_root else
                      root / 'indexes/rust-native-10m-recall-preserve')
    cpp_prefix = root / 'indexes/yfcc10m_stitched_R32_R128_L100'
    cpp_bin = root / 'cpp/build-native/apps'
    binaries = {'cpp': cpp_bin / 'search_disk_index',
                'baseline': root / 'artifacts/heldout/filtered_lifecycle.v5-heldout',
                'candidate': args.candidate_binary.resolve()}
    archives = {'baseline': root / 'artifacts/heldout/lifecycle-v5-heldout-complete-sources.tar.gz',
                'candidate': args.candidate_source_archive.resolve()}
    native_roots = {'baseline': baseline_root, 'candidate': candidate_root}
    cpp_runner = root / 'rust/tools/single-hop-benchmark/cpp-port/run_reference.py'
    index_before = {engine: native_identity(path) for engine, path in native_roots.items()}
    index_before['cpp'] = {suffix: sha(Path(str(cpp_prefix) + suffix)) for suffix in CPP_SUFFIXES}
    binary_before = {engine: sha(path) for engine, path in binaries.items()}
    archive_before = {engine: sha(path) for engine, path in archives.items()}
    if args.candidate_root is not None:
        for name in ('index_pq_pivots.bin', 'index_pq_compressed.bin'):
            if index_before['baseline']['files_sha256'][name] != index_before['candidate']['files_sha256'][name]:
                parser.error('Corrected candidate must retain both frozen baseline PQ artifacts')
        if index_before['baseline']['files_sha256']['vectors.bin'] != index_before['candidate']['files_sha256']['vectors.bin']:
            parser.error('Baseline and corrected candidate vectors/order must be identical')
    out.mkdir(parents=True, exist_ok=False)
    freeze = {'schema_version': 1, 'timestamp_utc': now(), 'phase': args.phase,
              'statement': ('Previously exposed primary natural queries: development recheck only; candidate 2 cannot claim these queries as fresh validation' if args.phase == 'development-recheck' else 'Candidate, breadth and split identities fixed before this evaluation; this partition has not been searched before this candidate freeze'),
              'query_freshness': 'previously_exposed_development' if args.phase == 'development-recheck' else 'previously_unsearched_partition',
              'plan_sha256': sha(args.plan), 'fresh_proof_sha256': sha(args.fresh_proof),
              'prepared_manifest_sha256': sha(prepared / 'manifest.json'),
              'candidate_build_log_sha256': sha(args.candidate_build_log),
              'binary_sha256': binary_before, 'source_archive_sha256': archive_before,
              'index_identity': index_before, 'candidate_root': str(candidate_root)}
    dump(out / 'candidate-freeze.json', freeze)
    manifest = {'schema_version': 1, 'status': 'running', 'started': now(),
                'host': 'alex@100.83.201.102', 'output_root': str(out), 'cpu_affinity': affinity,
                'phase': args.phase, 'timing_authoritative': not args.quality_only,
                'queries': count, 'k': 10, 'nominal_l': 64,
                'query_freshness': freeze['query_freshness'],
                'prepared': str(prepared), 'prepared_manifest_sha256': freeze['prepared_manifest_sha256'],
                'fresh_proof_sha256': freeze['fresh_proof_sha256'],
                'candidate_freeze_sha256': sha(out / 'candidate-freeze.json'),
                'plan_before_sha256': sha(args.plan), 'runs': [],
                'order_protocol': 'All six engine permutations, serial jobs, full same-process eight-worker warmup'}
    path = out / 'triple-run-manifest.json'
    dump(path, manifest)
    environment = os.environ.copy()
    environment.update(OMP_TOOL='disabled', MKL_NUM_THREADS='1',
                       LD_LIBRARY_PATH=str(root / 'cpp-deps/root/usr/lib/x86_64-linux-gnu') + ':/usr/lib/llvm-18/lib')
    merged = {engine: [] for engine in ENGINES}
    headers = {}
    orders = [ENGINES] if args.quality_only else list(itertools.permutations(ENGINES))
    for repeat, order in enumerate(orders):
        directory = out / f'triple-{repeat:02d}'
        directory.mkdir()
        for position, engine in enumerate(order):
            if engine == 'cpp':
                name = f'cpp-recall-{repeat:02d}'
                command = [sys.executable, str(cpp_runner), '--bin-dir', str(cpp_bin),
                           '--prepared-data', str(prepared), '--output-dir', str(directory / 'cpp'),
                           '--name', name, '--index-prefix', str(cpp_prefix), '--query-only',
                           '--search-threads', '8', '--l', '64', '--reps', '1', '--k', '10',
                           '--beam', '4', '--cache-nodes', '0']
                if not args.quality_only:
                    command.append('--timing-authoritative')
                aggregate = directory / 'cpp' / (name + '.normalized.jsonl')
            else:
                aggregate = directory / (engine + '.jsonl')
                command = [str(binaries[engine]), '--action', 'query', '--root', str(native_roots[engine]),
                           '--prepared', str(prepared), '--out', str(aggregate), '--threads', '8',
                           '--query-threads', '8', '--l', '64', '--reps', '1', '--limit', str(count),
                           '--timing-authoritative', str(not args.quality_only).lower(),
                           '--query-details-dir', str(directory / (engine + '-details')),
                           '--evaluation-scope', 'frozen_policy_heldout', '--label-l', str(args.plan.resolve()),
                           '--allow-unused-plan-labels', 'true']
            record = {'implementation': engine, 'paired_repeat': repeat, 'local_repeat': 0,
                      'order_in_pair': position, 'command': command, 'start': now(),
                      'aggregate_path': str(aggregate), 'stdout_log': str(directory / (engine + '.log'))}
            begin = time.monotonic()
            with Path(record['stdout_log']).open('x') as stream:
                process = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            record.update(end=now(), elapsed_seconds=time.monotonic()-begin, exit_code=process.returncode)
            if process.returncode:
                manifest.update(status='failed', failed_run=record)
                dump(path, manifest)
                raise RuntimeError(f'{engine} failed; see {record["stdout_log"]}')
            record['aggregate_sha256'] = sha(aggregate)
            rows = [json.loads(line) for line in aggregate.read_text().splitlines()]
            if engine != 'cpp':
                experiment, = [row for row in rows if row.get('kind') == 'experiment']
                if engine in headers and headers[engine] != experiment:
                    raise ValueError('Native experiment identity changed across repeats')
                headers[engine] = experiment
                rows = [row for row in rows if row.get('kind') == 'query']
            row, = rows
            if row['queries'] != count or row['repeat'] != 0:
                raise ValueError('Incomplete raw observation')
            row = dict(row, repeat=repeat, paired_repeat=repeat, local_repeat=0,
                       original_aggregate=str(aggregate), original_aggregate_sha256=record['aggregate_sha256'])
            if engine == 'cpp':
                row.update(raw_stats_relative_path=str(Path(row['raw_stats']).relative_to(out)),
                           result_ids_relative_path=str(Path(row['result_ids']).relative_to(out)))
            else:
                row['raw_queries_relative_path'] = str(Path(row['raw_queries']['path']).relative_to(out))
            merged[engine].append(row)
            with (out / (engine + '.normalized.jsonl')).open('w') as stream:
                if engine in headers:
                    stream.write(json.dumps(headers[engine]) + '\n')
                for item in merged[engine]:
                    stream.write(json.dumps(item) + '\n')
            manifest['runs'].append(record)
            dump(path, manifest)
            print(json.dumps({'engine': engine, 'repeat': repeat, 'recall': row['recall'],
                              'qps': row['qps'], 'p99_us': row['p99_us'], 'quality_only': args.quality_only}), flush=True)
    identity = {'schema_version': 1, 'host': manifest['host'], 'output_root': str(out),
                'cpu_affinity': affinity, 'phase': args.phase,
                'plan_before_sha256': manifest['plan_before_sha256'], 'plan_after_sha256': sha(args.plan),
                'prepared_manifest_sha256': sha(prepared / 'manifest.json'),
                'fresh_proof_sha256': sha(args.fresh_proof),
                'candidate_freeze_sha256': sha(out / 'candidate-freeze.json'),
                'query_freshness': freeze['query_freshness'],
                'runner_sha256': sha(Path(__file__)), 'cpp_runner_sha256': sha(cpp_runner), 'engines': {}}
    def identity_failure(message):
        manifest.update(status='failed_identity_check', failure=message, ended=now())
        dump(path, manifest)
        raise ValueError(message)

    for engine in ENGINES:
        after = (native_identity(native_roots[engine]) if engine != 'cpp' else
                 {suffix: sha(Path(str(cpp_prefix) + suffix)) for suffix in CPP_SUFFIXES})
        item = {'binary_sha256': sha(binaries[engine]), 'binary_before_sha256': binary_before[engine],
                'log_sha256': sha(out / (engine + '.normalized.jsonl')),
                'index_before': index_before[engine], 'index_after': after}
        if engine in archives:
            item['source_archive'] = {'path': str(archives[engine]), 'sha256': sha(archives[engine]),
                                      'before_sha256': archive_before[engine]}
        identity['engines'][engine] = item
        if after != index_before[engine] or item['binary_sha256'] != binary_before[engine]:
            identity_failure('Index/binary changed during comparison')
        if engine in archives and sha(archives[engine]) != archive_before[engine]:
            identity_failure('Source archive changed during comparison')
    if (sha(args.plan) != FROZEN_PLAN or identity['prepared_manifest_sha256'] != freeze['prepared_manifest_sha256']
            or identity['fresh_proof_sha256'] != freeze['fresh_proof_sha256']):
        identity_failure('Policy/data identity changed during comparison')
    try:
        validate_fresh(args.fresh_proof, prepared, args.phase, count)
    except ValueError as error:
        identity_failure(str(error))
    manifest.update(status='complete', ended=now(), identity=identity)
    dump(out / 'identity.json', identity)
    dump(path, manifest)


if __name__ == '__main__':
    main()
