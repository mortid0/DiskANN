#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Audit construction candidate 2 with explicit development and fresh split scopes.

The natural primary query set has already influenced development. Only the
precommitted secondary validation partition is fresh for this candidate. Its
atomic predicates are derived from two-tag vectors and remain a separate workload.
The frozen v1 analyzer is imported without modification for raw/statistical checks.
"""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_recall_fix as v1
from summarize_heldout import cpp_observations, native_observations

SPLIT_SHA = '34965480cdef01db2fc9ecc757b48173efd2afb11bfb095c1422bfb9a4316af5'
HOLD_SHA = '435a95faeb3dbf095e8ee0319f3416f7c824038d8140d7bc5e0ba9771a4b2198'
PHASES = ('development-recheck', 'secondary-calibration', 'secondary-validation')


def reconstruct_split(pool_labels):
    result = {'calibration': [], 'validation': []}
    for label in sorted(set(map(int, pool_labels))):
        positions = [int(value) for value in np.flatnonzero(pool_labels == label)]
        positions.sort(key=lambda position: hashlib.sha256(f'20261025:split:{position}:{label}'.encode()).digest())
        cut = len(positions) // 4
        result['calibration'].extend((position, label) for position in positions[:cut])
        result['validation'].extend((position, label) for position in positions[cut:])
    for name, rows in result.items():
        rows.sort(key=lambda item: hashlib.sha256(
            f'20261025:order-{name}:{item[0]}:{item[1]}'.encode()).digest())
    return {name: [position for position, _ in rows] for name, rows in result.items()}


def secondary_split_proof(args):
    v1.require(v1.sha(args.split_commitment) == SPLIT_SHA, 'Secondary v2 split commitment changed')
    declaration = json.loads(args.split_commitment.read_text())
    v1.require(declaration.get('scope') == 'supplemental_shifted_atomic_label_v2_split'
               and declaration.get('policy_sha256') == v1.FROZEN_SHA
               and declaration.get('primary_commitment_sha256') == v1.COMMITMENT_SHA,
               'Secondary v2 scope or policy binding differs')
    pool = v1.supplemental_prepared(args.original_pool)
    pool_args = copy.copy(args)
    pool_args.supplemental_prepared = args.original_pool
    pool_proof = v1.supplemental_source_proof(pool, pool_args)
    v1.require(v1.sha(args.original_pool / 'manifest.json') == declaration['original_pool_manifest_sha256']
               and declaration['original_selection_commitment_sha256'] == v1.SECONDARY_SELECTION_SHA
               and declaration['original_evaluation_commitment_sha256'] == v1.SECONDARY_EVALUATION_SHA,
               'Original secondary pool was replaced or modified')
    hold_path = Path(declaration['hold_amendment']['path'])
    v1.require(hold_path.is_file() and v1.sha(hold_path) == HOLD_SHA == declaration['hold_amendment']['sha256'],
               'Unsearched secondary hold amendment changed')
    for name, expected in declaration['source_pool_artifacts'].items():
        v1.require(v1.sha(args.original_pool / name) == expected['sha256']
                   and (args.original_pool / name).stat().st_size == expected['bytes'], 'Original secondary pool artifact changed')
    reconstructed = reconstruct_split(pool['labels'])
    pool_rows = v1.read_rows(args.original_pool / 'source-query-rows.jsonl')
    partitions = {}
    selected = {}
    for short, directory in [('calibration', args.calibration_prepared), ('validation', args.validation_prepared)]:
        phase = 'secondary-' + short
        expected = declaration['partitions'][phase]
        v1.require(v1.sha(directory / 'manifest.json') == expected['manifest_sha256'], 'V2 partition manifest changed')
        for name, record in expected['artifacts'].items():
            v1.require(v1.sha(directory / name) == record['sha256']
                       and (directory / name).stat().st_size == record['bytes'], 'V2 partition artifact changed')
        workload = v1.supplemental_prepared(directory)
        manifest = workload['manifest']
        v1.require(manifest.get('split') == phase.replace('-', '_')
                   and manifest.get('split_protocol_sha256') == declaration['split_protocol_sha256']
                   and manifest.get('source_pool_manifest', {}).get('sha256') == declaration['original_pool_manifest_sha256']
                   and manifest.get('validation_scope') == f'recall_fix_supplemental_shifted_atomic_label_{short}_v2',
                   'V2 partition namespace/scope/protocol differs')
        positions, _ = v1.matrix(directory / 'source_pool_indices.u32bin', '<u4')
        v1.require(positions.shape == (len(workload['queries']), 1)
                   and positions[:, 0].tolist() == reconstructed[short], 'V2 split/order differs from independent hash reconstruction')
        positions = positions[:, 0]
        for field in ('queries', 'labels', 'ids', 'namespace', 'truth', 'distances'):
            v1.require(np.array_equal(workload[field], pool[field][positions]), f'V2 partition {field} differs from immutable original pool')
        rows = v1.read_rows(directory / 'source-query-rows.jsonl')
        expected_rows = [dict(pool_rows[int(position)], query_index=index, source_pool_query_index=int(position))
                         for index, position in enumerate(positions)]
        v1.require(rows == expected_rows and len(rows) == expected['queries'], 'V2 source-row provenance differs')
        partitions[phase] = workload
        selected[phase] = set(map(int, positions))
    v1.require(not selected['secondary-calibration'] & selected['secondary-validation']
               and selected['secondary-calibration'] | selected['secondary-validation'] == set(range(len(pool['queries']))),
               'V2 partitions overlap or omit original pool rows')
    return partitions, {'source': v1.asset(args.split_commitment), 'hold_amendment': v1.asset(hold_path),
        'original_pool_proof': pool_proof, 'all_pool_rows_partitioned_once': True,
        'calibration_validation_embedding_overlap': 0,
        'partitions': {name: {'queries': len(workload['queries']), 'labels': dict(sorted(Counter(map(int, workload['labels'])).items()))}
                       for name, workload in partitions.items()},
        'scope': 'Freshly reserved atomic-derived secondary split; natural primary queries are exposed development data.'}


def same_candidate(previous, current):
    for name in v1.ENGINES:
        v1.require(previous['engines'][name]['binary_sha256'] == current['engines'][name]['binary_sha256']
                   and previous['engines'][name]['index_after'] == current['engines'][name]['index_before'],
                   f'{name} assets changed between calibration and fresh validation')
    for name in ('baseline', 'candidate'):
        v1.require(previous['engines'][name]['source_archive']['sha256']
                   == current['engines'][name]['source_archive']['sha256'],
                   f'{name} source archive changed after calibration')


def identity_proof(args, logs, manifest):
    evidence = json.loads(args.identity.read_text())
    prior = json.loads(args.prior_identity.read_text())
    freeze = json.loads(args.candidate_freeze.read_text())
    quality_only = args.quality_only
    fresh = 'previously_exposed_development' if args.phase == 'development-recheck' else 'previously_unsearched_partition'
    proof_path = args.primary_commitment if args.phase == 'development-recheck' else args.split_commitment
    v1.require(evidence.get('schema_version') == 1 and evidence.get('host') == prior.get('host')
               and evidence.get('cpu_affinity') == manifest.get('cpu_affinity') == list(range(8)), 'V2 host/core affinity differs')
    v1.require(manifest.get('identity') == evidence and evidence.get('output_root') == manifest.get('output_root')
               and evidence.get('phase') == manifest.get('phase') == freeze.get('phase') == args.phase
               and evidence.get('query_freshness') == manifest.get('query_freshness') == freeze.get('query_freshness') == fresh
               and manifest.get('timing_authoritative') is (not quality_only), 'V2 phase/freshness declaration differs')
    v1.require(evidence.get('plan_before_sha256') == evidence.get('plan_after_sha256') == v1.FROZEN_SHA
               and freeze.get('plan_sha256') == v1.FROZEN_SHA,
               'V2 changes the original frozen L policy')
    v1.require(evidence.get('prepared_manifest_sha256') == manifest.get('prepared_manifest_sha256')
               == freeze.get('prepared_manifest_sha256') == v1.sha(args.prepared / 'manifest.json')
               and evidence.get('fresh_proof_sha256') == manifest.get('fresh_proof_sha256')
               == freeze.get('fresh_proof_sha256') == v1.sha(proof_path)
               and evidence.get('candidate_freeze_sha256') == manifest.get('candidate_freeze_sha256') == v1.sha(args.candidate_freeze)
               and freeze.get('candidate_build_log_sha256') == v1.sha(args.candidate_build_log),
               'V2 data/build/source freeze binding differs')
    engines = evidence.get('engines', {})
    v1.require(set(engines) == set(v1.ENGINES), 'Incomplete V2 engine identities')
    for name in v1.ENGINES:
        item = engines[name]
        v1.require(v1.digest(item.get('binary_sha256')) and item.get('binary_before_sha256') == item['binary_sha256']
                   == freeze.get('binary_sha256', {}).get(name) and item.get('log_sha256') == v1.sha(logs[name]),
                   f'{name} V2 binary/log identity changed')
        before, after = item.get('index_before', {}), item.get('index_after', {})
        v1.require(before and before == after == freeze.get('index_identity', {}).get(name), f'{name} V2 index changed')
        for key, value in before.items():
            values = value.values() if key == 'files_sha256' else [value]
            v1.require(all(v1.digest(digest) for digest in values), f'{name} V2 index SHA declaration invalid')
        if name != 'cpp':
            v1.require(before.get('files_sha256'), f'{name} V2 full generation inventory absent')
    v1.require({key: engines['baseline']['index_before'][key] for key in prior['native']['index_before']}
               == prior['native']['index_before']
               and engines['baseline']['binary_sha256'] == prior['native']['binary_sha256'], 'Original Rust baseline changed')
    v1.require(engines['cpp']['index_before'] == prior['cpp']['index_before']
               and engines['cpp']['binary_sha256'] == prior['cpp']['binary_sha256'], 'Original C++ baseline changed')
    for filename in ('vectors.bin', 'index_pq_pivots.bin', 'index_pq_compressed.bin'):
        v1.require(v1.digest(engines['candidate']['index_before']['files_sha256'].get(filename))
                   and engines['candidate']['index_before']['files_sha256'].get(filename)
                   == engines['baseline']['index_before']['files_sha256'].get(filename),
                   f'V2 candidate changes the frozen vector order or PQ artifacts: {filename}')
    archived = {}
    for name, path in [('baseline', args.baseline_source_archive), ('candidate', args.candidate_source_archive)]:
        source = engines[name].get('source_archive', {})
        v1.require(source.get('sha256') == source.get('before_sha256')
                   == freeze.get('source_archive_sha256', {}).get(name) == v1.sha(path), f'{name} V2 source archive changed')
        archived[name] = v1.archived_files(path)
    v1.require(set(archived['baseline']) == set(archived['candidate']), 'V2 runtime source sets differ')
    changed = sorted(name for name in archived['baseline'] if archived['baseline'][name] != archived['candidate'][name])
    v1.require(changed == ['diskann-disk/src/build/filtered.rs'], 'V2 source changes extend beyond generic construction')
    v1.require(v1.sha(args.baseline_source_archive) == prior['native']['source_archive']['sha256'], 'V2 baseline sources changed')
    candidate_root = Path(freeze['candidate_root'])
    roots = {'candidate': str(candidate_root), 'baseline': str(candidate_root.parent / 'rust-native-10m-frozen')}
    cpp_prefix = str(candidate_root.parent / 'yfcc10m_stitched_R32_R128_L100')
    for name in ('baseline', 'candidate'):
        exp = v1.one([row for row in v1.read_rows(logs[name]) if row.get('kind') == 'experiment'], f'{name} V2 experiment')
        v1.require(exp.get('root') == roots[name] and exp.get('prepared') == manifest.get('prepared')
                   and exp.get('action') == 'query' and exp.get('n_base') == 10000000 and exp.get('dimensions') == 192
                   and exp.get('index_threads') == 8 and exp.get('beam') == 4 and exp.get('shared_cache_nodes') == 0
                   and exp.get('cache_bypass') is True and exp.get('storage') == 'native_io_uring'
                   and exp.get('warmup') == 'full_same_process_same_worker_count'
                   and exp.get('query_scheduling') == 'dynamic_one', f'{name} V2 index/query/I/O protocol differs')
    v1.require(all(row.get('index_prefix') == cpp_prefix and row.get('prepared') == manifest.get('prepared')
                   for row in v1.read_rows(logs['cpp'])), 'V2 C++ index/query differs')
    for job in manifest['runs']:
        command, name = job.get('command', []), job['implementation']
        option = '--index-prefix' if name == 'cpp' else '--root'
        v1.require(option in command and command[command.index(option) + 1] == (cpp_prefix if name == 'cpp' else roots[name]),
                   f'{name} V2 retained job command index differs')
    v1.require(type(freeze.get('timestamp_utc')) is str and freeze['timestamp_utc'] <= manifest['runs'][0]['start'],
               'V2 candidate freeze recorded after queries began')
    runner = Path(__file__).with_name('run_recall_fix_v2.py')
    cpp_runner = Path(__file__).parent / 'cpp-port/run_reference.py'
    v1.require(evidence.get('runner_sha256') == v1.sha(runner) and evidence.get('cpp_runner_sha256') == v1.sha(cpp_runner),
               'V2 runner or C++ normalizer changed')
    calibrated = None
    if args.phase == 'secondary-validation':
        v1.require(args.calibration_identity is not None, 'Fresh V2 validation must bind frozen calibration assets')
        calibrated = json.loads(args.calibration_identity.read_text())
        v1.require(calibrated.get('phase') == 'secondary-calibration', 'V2 validation calibration identity phase differs')
        same_candidate(calibrated, evidence)
    return {'source': v1.asset(args.identity), 'candidate_freeze': v1.asset(args.candidate_freeze),
            'changed_archived_source_files': changed, 'declarations': evidence,
            'calibration_identity': v1.asset(args.calibration_identity) if calibrated else None,
            'source_archives': {'baseline': v1.asset(args.baseline_source_archive), 'candidate': v1.asset(args.candidate_source_archive)},
            'candidate_build_log': v1.asset(args.candidate_build_log), 'query_freshness': fresh}


def summarize(args):
    v1.require(v1.sha(args.plan) == v1.FROZEN_SHA, 'Original per-label L policy changed')
    plan = v1.load_plan(args.plan)
    partitions, split_proof = secondary_split_proof(args)
    if args.phase == 'development-recheck':
        v1.require(v1.sha(args.primary_commitment) == v1.COMMITMENT_SHA, 'Primary commitment changed')
        workload = v1.prepared(args.prepared)
        primary = json.loads(args.primary_summary.read_text())
        v1.require(workload['identity'] == primary.get('prepared_identity'), 'Development recheck differs from exposed primary workload')
        freshness = 'previously_exposed_development'
    else:
        workload = partitions[args.phase]
        source = args.calibration_prepared if args.phase == 'secondary-calibration' else args.validation_prepared
        v1.require(v1.sha(args.prepared / 'manifest.json') == v1.sha(source / 'manifest.json'), 'Selected V2 phase prepared path differs')
        freshness = 'previously_unsearched_partition'
    quality_only = args.quality_only
    postings = {}
    for label in sorted(set(map(int, workload['labels']))):
        values, _ = v1.matrix(args.prepared / f'label_{label}.u32bin', '<u4')
        v1.require(values.shape[1] == 1 and np.all(np.diff(values[:, 0].astype(np.int64)) > 0), 'Invalid V2 posting')
        postings[label] = values[:, 0]
    v1.valid_result_ids(workload['truth'], workload, postings)
    logs = {'cpp': args.cpp, 'baseline': args.baseline, 'candidate': args.candidate}
    manifest = json.loads(args.jobs_manifest.read_text())
    root, remote = args.artifact_root or args.jobs_manifest.parent, Path(manifest['output_root'])
    observations, inputs = {}, {}
    for name in v1.ENGINES:
        if quality_only:
            observations[name], inputs[name] = v1.quality_observations(logs[name], workload, plan, postings, root, remote, name != 'cpp')
        elif name == 'cpp':
            observations[name], inputs[name] = cpp_observations(logs[name], None, workload, plan, postings, root, remote)
        else:
            _, observations[name], inputs[name] = native_observations(logs[name], workload, plan, postings, root, remote)
    jobs = v1.triple_jobs_proof(args.jobs_manifest, logs, root, quality_only=quality_only)
    identity = identity_proof(args, logs, manifest)
    result = v1.compare_engines(observations, workload, plan, postings, args.bootstrap, args.seed, quality_only=quality_only)
    cutoff = v1.cutoff_diagnostic(args.base, workload, {name: rows[0] for name, rows in observations.items()})
    for row in result['per_label']:
        mask = workload['labels'] == row['label']
        for name in v1.ENGINES:
            row[f'{name}_cutoff_recall_percent'] = float(cutoff['engines'][name]['hits'][mask].mean()) * 10
        for positive, reference in v1.CONTRASTS:
            row[f'{positive}_minus_{reference}_cutoff_pp'] = row[f'{positive}_cutoff_recall_percent'] - row[f'{reference}_cutoff_recall_percent']
    cutoff['engines'] = {name: {key: value for key, value in row.items() if key != 'hits'} for name, row in cutoff['engines'].items()}
    result.update({'schema_version': 2, 'phase': args.phase, 'query_freshness': freshness, 'timing_excluded': quality_only,
        'scope': 'Exposed natural-query development recheck' if args.phase == 'development-recheck' else
                 'Fresh secondary atomic-derived calibration' if quality_only else 'Fresh secondary atomic-derived validation; separate five-label shifted workload',
        'prepared_identity': workload['identity'], 'split_proof': split_proof, 'jobs_proof': jobs, 'identity_proof': identity,
        'cutoff_diagnostic': cutoff, 'inputs': inputs, 'targets': list(v1.TARGETS), 'acceptance_threshold': None,
        'bootstrap': {'replications': args.bootstrap, 'seed': args.seed, 'unit': 'paired query',
                      'per_label_simultaneous_family': 3 * len(result['per_label']), 'aggregate_simultaneous_family': 3},
        'limitations': [
            'The natural primary query set influenced candidate development and is not fresh validation for candidate 2.',
            'Secondary predicates are derived atomic labels from two-tag vectors and cannot establish natural one-tag recall or workload throughput.',
            'Reported validation performance describes this five-label mixture on the benchmark host, not production advertising traffic.',
            'Six timing repeats reuse queries; bootstrap units are paired queries, not neighbors or timing repetitions.',
            'Zero observed recall difference or an interval containing zero does not prove non-inferiority.',
            'All labels are disclosed; no acceptance threshold was chosen after reading these outcomes.'
        ]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=PHASES, required=True)
    for name in ('prepared', 'original-pool', 'split-commitment', 'calibration-prepared', 'validation-prepared',
                 'primary-commitment', 'primary-summary', 'public-queries', 'private-queries', 'private-metadata',
                 'base', 'plan', 'cpp', 'baseline', 'candidate', 'jobs-manifest', 'identity', 'prior-identity',
                 'baseline-source-archive', 'candidate-source-archive', 'candidate-freeze', 'candidate-build-log', 'out', 'csv'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path)
    parser.add_argument('--quality-only', action='store_true', help='Exclude all performance claims; secondary calibration requires this flag')
    parser.add_argument('--calibration-identity', type=Path)
    parser.add_argument('--bootstrap', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=20261025)
    args = parser.parse_args()
    v1.require(args.phase == 'development-recheck' or args.quality_only == (args.phase == 'secondary-calibration'),
               'Secondary calibration is quality-only and fresh secondary validation requires authoritative timings')
    v1.require(args.bootstrap >= 10000, 'At least10K query bootstrap replications required')
    v1.require(not args.out.exists() and not args.csv.exists(), 'V2 analysis outputs already exist')
    result = summarize(args)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    with args.csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['per_label'][0]))
        writer.writeheader()
        writer.writerows(result['per_label'])
    print(json.dumps({'summary': str(args.out), 'recall': {name: item['strict_id_recall'] for name, item in result['engines'].items()},
                      'label_comparison': result['label_comparison']}))


if __name__ == '__main__':
    main()
