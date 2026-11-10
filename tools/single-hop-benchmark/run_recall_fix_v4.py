#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Evaluate fixed-L centroid fallback plus nearest-PQ entry routing with exposed development and sealed validation scopes."""
import argparse
import datetime
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


PROTOCOL_SCOPE = 'dual_entry_v4_protocol_source_freeze'
SPLIT_SHA = '34965480cdef01db2fc9ecc757b48173efd2afb11bfb095c1422bfb9a4316af5'
RUNTIME_CORE_SHA256 = {
    'diskann-disk/src/build/filtered.rs': 'ba2357904399bacd035b05aec58b3b7e0f1f5afbd1b0616ffaf00f82b753d965',
    'diskann-disk/src/filtered/view.rs': 'e67b32e2126f884a2123f1027a2aff1d0703be637358bf472e27a7418071c4a2',
    'diskann-disk/src/search/provider/disk_provider.rs': '939ff3e60f363d2db8834d024243db6dc788f8aa012288a0f15997ed8dd52daf',
    'diskann-disk/src/search/search_mode.rs': '3430d87756a31427baa04302a82f556cf98afa2c0ca7168ec72da33bbedb721c',
}
RUNTIME_BINARY_SHA = 'ddd4277e9bb52f4cc42b4b9b66c6927a841ad6a96b534926389c7eb91a86625f'
RUNTIME_ARCHIVE_SHA = '0d19211cb5ff2e65a559520414760a49fb0cef742ec53e9335c6df755ad3e2e0'
REQUIRED_PROTOCOL_FILES = set(RUNTIME_CORE_SHA256) | {
    'tools/single-hop-benchmark/run_recall_fix_v4.py',
    'tools/single-hop-benchmark/analyze_recall_fix_v4.py',
    'tools/single-hop-benchmark/run_recall_fix.py',
    'tools/single-hop-benchmark/run_heldout.py',
    'tools/single-hop-benchmark/analyze_recall_fix.py',
    'tools/single-hop-benchmark/plan_label_l.py',
    'tools/single-hop-benchmark/summarize.py',
    'tools/single-hop-benchmark/summarize_delta.py',
    'tools/single-hop-benchmark/summarize_heldout.py',
    'tools/single-hop-benchmark/cpp-port/run_reference.py',
    'tools/single-hop-benchmark/prepare_yfcc.py',
    'tools/single-hop-benchmark/summarize_lifecycle.py',
    'tools/single-hop-benchmark/summarize_migration.py',
    'tools/single-hop-benchmark/summarize_native_build.py',
}
ENTRY_POLICY = {
    'mode': 'global_centroid_plus_nearest_pq',
    'maximum_initial_seeds': 2,
    'deduplicate_identical_seeds': True,
    'original_search_l_unchanged': True,
    'maximum_additional_ssd_seed_reads': 1,
}


def validate_freeze_order(protocol_time, candidate_time, first_query_time=None):
    values = [protocol_time, candidate_time]
    if first_query_time is not None:
        values.append(first_query_time)
    try:
        times = [datetime.datetime.fromisoformat(value) for value in values]
        if any(value.utcoffset() != datetime.timedelta(0) for value in times):
            raise ValueError('Timestamps must be UTC')
        if times != sorted(times):
            raise ValueError('Source freeze must precede candidate freeze and queries')
    except (TypeError, ValueError) as error:
        raise ValueError('V4 source/candidate/query freeze timestamp order is invalid') from error


def validate_amendment(amendment):
    if (amendment.get('secondary_validation_unsearched') is not True
            or amendment.get('secondary_split_sha256') != SPLIT_SHA
            or amendment.get('entry_policy') != ENTRY_POLICY
            or amendment.get('final_variant') is not True):
        raise ValueError('Requires the final recorded dual-entry policy with fixed L, at most two seeds and reserved validation')


def validate_protocol_freeze(path):
    import tarfile
    root = Path(__file__).resolve().parents[2]
    proof = json.loads(path.read_text())
    if proof.get('scope') != PROTOCOL_SCOPE or proof.get('schema_version') != 1:
        raise ValueError('Invalid V4 protocol source freeze')
    validate_freeze_order(proof.get('timestamp_utc'), now())
    files = proof.get('files', {})
    if not REQUIRED_PROTOCOL_FILES <= set(files):
        raise ValueError('V4 source freeze omits required runtime or statistical helper sources')
    if (proof.get('entry_policy') != ENTRY_POLICY
            or proof.get('runtime_core_sha256') != RUNTIME_CORE_SHA256
            or proof.get('runtime_source_archive', {}).get('sha256') != RUNTIME_ARCHIVE_SHA
            or proof.get('runtime_candidate_binary_sha256') != RUNTIME_BINARY_SHA):
        raise ValueError('V4 protocol does not bind the frozen V9 runtime and entry policy')
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha(root / relative) != expected:
            raise ValueError(f'Frozen protocol source changed: {name}')
    archive = proof.get('archive', {})
    name = archive.get('name', '')
    if not name or Path(name).name != name:
        raise ValueError('Invalid protocol archive name')
    archive_path = path.parent / name
    if sha(archive_path) != archive.get('sha256') or archive_path.stat().st_size != archive.get('bytes'):
        raise ValueError('Protocol source archive changed')
    import hashlib
    with tarfile.open(archive_path, 'r:gz') as source:
        contents = {member.name: hashlib.sha256(source.extractfile(member).read()).hexdigest()
                    for member in source.getmembers() if member.isfile()}
    if contents != files:
        raise ValueError('Protocol archive contents differ from frozen source declarations')
    return proof



def validate_runtime_archive(path, protocol):
    import hashlib
    import tarfile
    if sha(path) != RUNTIME_ARCHIVE_SHA or protocol.get('runtime_source_archive', {}).get('sha256') != RUNTIME_ARCHIVE_SHA:
        raise ValueError('Candidate runtime source archive is not the frozen V9 source')
    with tarfile.open(path, 'r:gz') as source:
        contents = {member.name.removeprefix('./'): hashlib.sha256(source.extractfile(member).read()).hexdigest()
                    for member in source.getmembers() if member.isfile()}
    if any(contents.get(name) != digest for name, digest in RUNTIME_CORE_SHA256.items()):
        raise ValueError('V9 runtime archive changes the frozen selector or fallback implementation')


def validate_pool_identity(identity, candidate_root, publication):
    base = candidate_root.parent.parent
    def canonical(value):
        path = Path(value)
        return Path(os.path.normpath(str(path if path.is_absolute() else base / path)))
    if canonical(publication.get('output_root', '')) != candidate_root:
        raise ValueError('Candidate must use the unchanged published spatial-entry root')
    hashes = {str(canonical(path)): value for path, value in publication.get('output_files_sha256', {}).items()}
    generation = candidate_root / 'generation-00000000000000000001'
    expected_names = {str(Path(name).relative_to(generation)) for name in hashes
                      if Path(name).parent == generation}
    if set(identity['files_sha256']) != expected_names:
        raise ValueError('Published spatial-entry inventory differs')
    for name, value in identity['files_sha256'].items():
        if hashes.get(str(generation / name)) != value:
            raise ValueError(f'Published spatial-entry artifact changed: {name}')
    for key, name in [('current_sha256', 'CURRENT'), ('journal_sha256', 'updates.wal')]:
        if hashes.get(str(candidate_root / name)) != identity[key]:
            raise ValueError(f'Published spatial-entry envelope changed: {name}')


def validate_calibration(calibration, indexes, binaries, archives, protocol_sha, amendment_sha):
    if (calibration.get('protocol_version') != 4 or calibration.get('phase') != 'secondary-development'
            or calibration.get('query_freshness') != 'previously_exposed_development'
            or calibration.get('entry_policy') != ENTRY_POLICY
            or calibration.get('protocol_source_freeze_sha256') != protocol_sha
            or calibration.get('development_amendment_sha256') != amendment_sha):
        raise ValueError('Fresh validation requires the same V9 dual-entry development protocol and policy')
    for name in ENGINES:
        row = calibration.get('engines', {}).get(name, {})
        if row.get('index_after') != indexes[name] or row.get('binary_sha256') != binaries[name]:
            raise ValueError(f'{name} assets differ from frozen V9 development')
        if name in archives and row.get('source_archive', {}).get('sha256') != archives[name]:
            raise ValueError(f'{name} sources differ from frozen V9 development')


def claim_final_validation(path, claim):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x') as output:
            output.write(json.dumps(claim, indent=2) + '\n')
    except FileExistsError as error:
        raise ValueError('Final fresh validation has already been claimed; it cannot be launched a second time') from error
    return {'path': str(path), 'sha256': sha(path), 'retained_copy': 'final-validation-claim.json'}


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
    if sha(proof_path) != SPLIT_SHA:
        raise ValueError('Final dual-entry evaluation requires the original sealed secondary split')
    partition = 'secondary-calibration' if phase == 'secondary-development' else phase
    expected = proof.get('partitions', {}).get(partition, {})
    split = partition.replace('-', '_')
    suffix = 'calibration' if phase == 'secondary-development' else 'validation'
    if (proof.get('scope') != 'supplemental_shifted_atomic_label_v2_split'
            or proof.get('policy_sha256') != FROZEN_PLAN
            or manifest.get('source_query_namespace') != 'mixed_public_and_official_private_2727415019_100K'
            or manifest.get('split') != split
            or manifest.get('validation_scope') != f'recall_fix_supplemental_shifted_atomic_label_{suffix}_v2'
            or expected.get('queries') != count
            or expected.get('manifest_sha256') != sha(prepared / 'manifest.json')
            or proof.get('split_protocol_sha256') != manifest.get('split_protocol_sha256')
            or proof.get('original_pool_manifest_sha256') != manifest.get('source_pool_manifest', {}).get('sha256')):
        raise ValueError('Spatial-routing split commitment does not match phase, namespace or prepared manifest')
    for name, expected_file in expected.get('artifacts', {}).items():
        path = prepared / name
        if sha(path) != expected_file.get('sha256') or path.stat().st_size != expected_file.get('bytes'):
            raise ValueError(f'Prepared artifact differs from sealed dual-entry split: {name}')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'prepared', 'out', 'plan', 'fresh-proof', 'candidate-source-archive',
                'candidate-build-log', 'development-amendment', 'protocol-freeze'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--calibration-identity', type=Path)
    parser.add_argument('--phase', choices=('development-recheck', 'secondary-development', 'secondary-validation'), required=True)
    parser.add_argument('--candidate-binary', type=Path, required=True)
    parser.add_argument('--quality-only', action='store_true')
    args = parser.parse_args()
    if args.phase == 'secondary-validation' and args.quality_only:
        parser.error('Fresh final validation requires authoritative timings')
    affinity = sorted(os.sched_getaffinity(0))
    if affinity != list(range(8)) or sha(args.plan) != FROZEN_PLAN:
        parser.error('Requires physical cores 0..7 and the unchanged original L policy')
    root, prepared, out = (getattr(args, name).resolve() for name in ('root', 'prepared', 'out'))
    count, dim = struct.unpack('<II', (prepared / 'query.u8bin').read_bytes()[:8])
    if count <= 0 or dim != 192:
        parser.error('Expected a nonempty 192-dimensional query corpus')
    validate_fresh(args.fresh_proof, prepared, args.phase, count)
    amendment = json.loads(args.development_amendment.read_text())
    validate_amendment(amendment)
    protocol = validate_protocol_freeze(args.protocol_freeze)
    if args.phase == 'secondary-validation' and args.calibration_identity is None:
        parser.error('Fresh validation requires --calibration-identity for the unchanged V9 development candidate')
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
    validate_runtime_archive(archives['candidate'], protocol)
    if (binary_before['candidate'] != protocol['runtime_candidate_binary_sha256']
            or sha(args.development_amendment) != protocol.get('development_amendment_sha256')
            or str(candidate_root) != amendment.get('candidate_root')
            or sha(args.candidate_build_log) != amendment.get('entry_publication_sha256')):
        raise ValueError('Candidate binary, source, amendment or spatial-entry root differs from the frozen V9 policy')
    if args.candidate_root is not None:
        for name in ('index_pq_pivots.bin', 'index_pq_compressed.bin'):
            if index_before['baseline']['files_sha256'][name] != index_before['candidate']['files_sha256'][name]:
                parser.error('Corrected candidate must retain both frozen baseline PQ artifacts')
        if index_before['baseline']['files_sha256']['vectors.bin'] != index_before['candidate']['files_sha256']['vectors.bin']:
            parser.error('Baseline and corrected candidate vectors/order must be identical')
    validate_pool_identity(index_before['candidate'], candidate_root, json.loads(args.candidate_build_log.read_text()))
    if out.exists():
        parser.error('Refusing to overwrite an existing evaluation output')
    final_claim = None
    claim_data = None
    if args.phase == 'secondary-validation':
        validate_calibration(json.loads(args.calibration_identity.read_text()), index_before, binary_before,
                             archive_before, sha(args.protocol_freeze), sha(args.development_amendment))
        claim_data = {'protocol_version': 4, 'phase': args.phase, 'timestamp_utc': now(),
                      'statement': 'One final frozen V9 evaluation of the original unsearched4316-query partition; no later candidate selection from this partition',
                      'output_root': str(out), 'plan_sha256': sha(args.plan), 'entry_policy': ENTRY_POLICY,
                      'fresh_proof_sha256': sha(args.fresh_proof),
                      'prepared_manifest_sha256': sha(prepared / 'manifest.json'),
                      'calibration_identity_sha256': sha(args.calibration_identity),
                      'protocol_source_freeze_sha256': sha(args.protocol_freeze),
                      'development_amendment_sha256': sha(args.development_amendment),
                      'binary_sha256': binary_before, 'source_archive_sha256': archive_before,
                      'index_identity': index_before}
        final_claim = claim_final_validation(root / 'artifacts/recall-fix/v4-final-validation-claim.json', claim_data)
    out.mkdir(parents=True, exist_ok=False)
    if final_claim:
        dump(out / final_claim['retained_copy'], claim_data)
    freeze = {'schema_version': 1, 'protocol_version': 4, 'entry_policy': ENTRY_POLICY,
              'protocol_source_freeze_sha256': sha(args.protocol_freeze),
              'protocol_source_archive_sha256': protocol['archive']['sha256'],
              'calibration_identity_sha256': sha(args.calibration_identity) if args.calibration_identity else None,
              'final_validation_claim': final_claim, 'timestamp_utc': now(), 'phase': args.phase,
              'statement': ('Previously exposed queries: development recheck only; candidate 4 cannot claim these queries as fresh validation' if args.phase != 'secondary-validation' else 'Candidate, breadth and split identities fixed before this evaluation; this partition has not been searched before this candidate freeze'),
              'query_freshness': 'previously_exposed_development' if args.phase != 'secondary-validation' else 'previously_unsearched_partition',
              'plan_sha256': sha(args.plan), 'fresh_proof_sha256': sha(args.fresh_proof),
              'prepared_manifest_sha256': sha(prepared / 'manifest.json'),
              'candidate_build_log_sha256': sha(args.candidate_build_log),
              'development_amendment_sha256': sha(args.development_amendment),
              'binary_sha256': binary_before, 'source_archive_sha256': archive_before,
              'index_identity': index_before, 'candidate_root': str(candidate_root)}
    dump(out / 'candidate-freeze.json', freeze)
    manifest = {'schema_version': 1, 'protocol_version': 4, 'entry_policy': ENTRY_POLICY,
                'protocol_source_freeze_sha256': freeze['protocol_source_freeze_sha256'],
                'protocol_source_archive_sha256': freeze['protocol_source_archive_sha256'],
                'calibration_identity_sha256': freeze['calibration_identity_sha256'],
                'final_validation_claim': final_claim, 'status': 'running', 'started': now(),
                'host': 'alex@100.83.201.102', 'output_root': str(out), 'cpu_affinity': affinity,
                'phase': args.phase, 'timing_authoritative': not args.quality_only,
                'queries': count, 'k': 10, 'nominal_l': 64,
                'query_freshness': freeze['query_freshness'],
                'development_amendment_sha256': freeze['development_amendment_sha256'],
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
    identity = {'schema_version': 1, 'protocol_version': 4, 'entry_policy': ENTRY_POLICY,
                'protocol_source_freeze_sha256': freeze['protocol_source_freeze_sha256'],
                'protocol_source_archive_sha256': freeze['protocol_source_archive_sha256'],
                'calibration_identity_sha256': freeze['calibration_identity_sha256'],
                'final_validation_claim': final_claim, 'host': manifest['host'], 'output_root': str(out),
                'cpu_affinity': affinity, 'phase': args.phase,
                'plan_before_sha256': manifest['plan_before_sha256'], 'plan_after_sha256': sha(args.plan),
                'prepared_manifest_sha256': sha(prepared / 'manifest.json'),
                'fresh_proof_sha256': sha(args.fresh_proof),
                'candidate_freeze_sha256': sha(out / 'candidate-freeze.json'),
                'query_freshness': freeze['query_freshness'],
                'development_amendment_sha256': freeze['development_amendment_sha256'],
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
        validate_protocol_freeze(args.protocol_freeze)
        if sha(args.protocol_freeze) != freeze['protocol_source_freeze_sha256']:
            raise ValueError('Protocol freeze changed during comparison')
        if sha(args.development_amendment) != freeze['development_amendment_sha256']:
            raise ValueError('Development amendment changed during comparison')
        validate_amendment(json.loads(args.development_amendment.read_text()))
        if final_claim and (sha(Path(final_claim['path'])) != final_claim['sha256']
                            or sha(out / final_claim['retained_copy']) != final_claim['sha256']):
            raise ValueError('Final validation claim changed')
        if args.calibration_identity and sha(args.calibration_identity) != freeze['calibration_identity_sha256']:
            raise ValueError('Calibration identity changed during comparison')
        validate_fresh(args.fresh_proof, prepared, args.phase, count)
    except ValueError as error:
        identity_failure(str(error))
    manifest.update(status='complete', ended=now(), identity=identity)
    dump(out / 'identity.json', identity)
    dump(path, manifest)


if __name__ == '__main__':
    main()
