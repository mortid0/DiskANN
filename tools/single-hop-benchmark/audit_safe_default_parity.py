#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Audit rollback output parity on exposed queries only; no fresh recall/performance claim."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import struct

from plan_label_l import load_plan
from run_safe_default_parity import archive_core, CORE
from run_recall_fix_v4 import validate_pool_identity
from summarize_heldout import asset, FROZEN_SHA, local_path, sha

V7_ARCHIVE_SHA = 'b05537b48619cf79f8d41684b40bd6f6aefb0aa66f6b1f25c6c557d6074ae843'
V7_BINARY_SHA = '9c35ea0a1b84f100416a9a824490313c817f893f841074bacd2bdbfdff84ca55'
V10_ARCHIVE_SHA = 'bca55c75b2158f5060e05b5ebad6bf3159e212952276b702a0cd2851a32db56f'
V10_BINARY_SHA = '0ffdb1ce816150d32238096264d7fe3f8394a1954157dd35530d9bbf80cfb557'
REFERENCE_IDENTITY_SHA = {
    'natural': '497e2ad73bfd46b7fefad123ab7a87653a22e6357c9fa861777a40a0a5a1948b',
    'calibration': '12ca7aaf3208254c405f8761a5568569b9708f1d19b512e04808e4a56a800e9b',
}
COUNTS = {'natural': 16600, 'calibration': 1436}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def one(values, message):
    require(len(values) == 1, message)
    return values[0]


def digest(value):
    return type(value) is str and len(value) == 64 and all(x in '0123456789abcdef' for x in value)


def ordered_result_parity(observed, reference, labels, plan):
    require(len(observed) == len(reference) == len(labels), 'Query row count differs')
    fields = ('ids', 'actual_l', 'label', 'query_index', 'k', 'l', 'repeat', 'phase', 'sequence',
              'base_searches', 'delta_candidates', 'exact_fallback', 'strict_id_hits')
    for index, (current, previous) in enumerate(zip(observed, reference)):
        label = labels[index]
        require(all(current.get(key) == previous.get(key) for key in fields),
                f'Ordered output/query identity differs at query {index}')
        require(current.get('query_index') == index and current.get('label') == label
                and current.get('actual_l') == plan[label] and current.get('l') == 64
                and current.get('repeat') == 0 and current.get('phase') == 'query'
                and current.get('k') == 10 and current.get('sequence') == 0
                and current.get('kind') == 'query_detail' and current.get('base_searches') == 1
                and current.get('delta_candidates') == 0 and current.get('exact_fallback') is False,
                f'Frozen L/snapshot/query contract differs at query {index}')
        ids, distances, old_distances = current.get('ids'), current.get('distances'), previous.get('distances')
        require(type(ids) is list and len(ids) == len(set(ids)) == 10
                and all(type(value) is int and 0 <= value < 10000000 for value in ids), 'Invalid result IDs')
        require(type(distances) is list and type(old_distances) is list
                and len(distances) == len(old_distances) == 10
                and all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                        for value in distances + old_distances), 'Invalid result distances')
        require(struct.pack('<10d', *distances) == struct.pack('<10d', *old_distances),
                f'Exact returned distance bits differ at query {index}')
    return {'queries': len(labels), 'ordered_ids_equal': True, 'returned_distance_bits_equal': True,
            'actual_frozen_l_equal': True, 'snapshot_and_result_contract_equal': True}


def workload(directory, count):
    manifest = json.loads((directory / 'manifest.json').read_text())
    require(manifest.get('query_count') == count and manifest.get('base_vectors') == 10000000
            and manifest.get('dimension') == 192 and manifest.get('k') == 10,
            'Exposed workload manifest shape differs')
    hashes = {}
    for name in ('query.u8bin', 'query_ids.u32bin', 'query_labels.u32bin'):
        path = directory / name
        expected = manifest['artifacts'][name]
        hashes[name] = sha(path)
        require(hashes[name] == expected['sha256'] and path.stat().st_size == expected['bytes'],
                f'Prepared query artifact changed: {name}')
    raw = (directory / 'query_labels.u32bin').read_bytes()
    require(struct.unpack('<II', raw[:8]) == (count, 1) and len(raw) == 8 + 4 * count,
            'Prepared label rows differ')
    labels = list(struct.unpack('<' + str(count) + 'I', raw[8:]))
    identity = ':'.join(manifest['artifacts'][name]['sha256'] for name in
                        ('query.u8bin', 'query_labels.u32bin', 'groundtruth.bin', 'labels.txt'))
    return labels, hashes, identity


def observation(aggregate_path, details_path, labels, plan, identity, *, authoritative):
    contents = rows(aggregate_path)
    experiment = one([row for row in contents if row.get('kind') == 'experiment'], 'Missing experiment')
    aggregate = one([row for row in contents if row.get('kind') == 'query'], 'Missing aggregate')
    require(len(contents) == 2 and aggregate.get('queries') == len(labels)
            and aggregate.get('warmup_queries') == len(labels) and aggregate.get('repeat') == 0
            and aggregate.get('prepared_identity') == identity and aggregate.get('timing_authoritative') is authoritative
            and aggregate.get('threads') == 8 and aggregate.get('l') == 64 and aggregate.get('k') == 10
            and aggregate.get('sequence') == 0 and aggregate.get('base_searches') == len(labels)
            and aggregate.get('delta_candidates') == 0 and aggregate.get('exact_fallback_queries') == 0,
            'Aggregate exposed query contract differs')
    require(experiment.get('action') == 'query' and experiment.get('n_base') == 10000000
            and experiment.get('dimensions') == 192 and experiment.get('index_threads') == 8
            and experiment.get('beam') == 4 and experiment.get('shared_cache_nodes') == 0
            and experiment.get('cache_bypass') is True and experiment.get('storage') == 'native_io_uring'
            and experiment.get('warmup') == 'full_same_process_same_worker_count'
            and experiment.get('query_scheduling') == 'dynamic_one', 'Experiment I/O/query protocol differs')
    require(aggregate.get('label_l_plan', {}).get('content_sha256') == FROZEN_SHA
            and aggregate['label_l_plan'].get('entries') == [{'label': label, 'l': plan[label]} for label in sorted(plan)]
            and all(plan.get(row.get('label')) == row.get('actual_l') for row in aggregate.get('labels', [])),
            'Aggregate actual L plan differs')
    require(aggregate.get('raw_queries', {}).get('sha256') == sha(details_path)
            and aggregate['raw_queries'].get('queries') == len(labels), 'Raw output binding differs')
    require(all(aggregate.get(key) == 0 for key in ('filter_violations', 'duplicate_results', 'short_results')),
            'Aggregate output errors')
    return experiment, aggregate, rows(details_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence', 'artifact-root', 'source-archive', 'pool-publication', 'plan', 'natural-prepared',
                 'calibration-prepared', 'reference-natural-identity', 'reference-calibration-identity',
                 'reference-source-archive', 'rollback-decision', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Refusing to overwrite parity audit')
    evidence = json.loads(args.evidence.read_text())
    require(evidence.get('schema_version') == 1 and evidence.get('status') == 'complete'
            and evidence.get('scope') == 'exposed_output_parity_only'
            and evidence.get('cpu_affinity') == list(range(8)), 'Parity run scope/status differs')
    require(sha(args.plan) == evidence['plan']['before_sha256'] == evidence['plan']['after_sha256'] == FROZEN_SHA,
            'Frozen L policy changed')
    require(sha(args.source_archive) == evidence['source_archive']['before_sha256']
            == evidence['source_archive']['after_sha256'] == V10_ARCHIVE_SHA, 'V10 source archive changed')
    require(digest(evidence['binary']['before_sha256'])
            and evidence['binary']['before_sha256'] == evidence['binary']['after_sha256'] == V10_BINARY_SHA, 'V10 binary changed')
    require(evidence['live_core_before_sha256'] == evidence['live_core_after_sha256']
            == evidence['archive_core_sha256'] == archive_core(args.source_archive), 'V10 live/archive core sources differ')
    require(sha(args.reference_source_archive) == V7_ARCHIVE_SHA, 'V7 reference source archive changed')
    require(sha(args.pool_publication) == evidence['pool_publication_sha256'], 'Pool publication changed')
    require(evidence['index_before'] == evidence['index_after'], 'Pool index changed during controls')
    validate_pool_identity(evidence['index_before'], Path(evidence['index_root']),
                           json.loads(args.pool_publication.read_text()))
    require(sha(Path(__file__).with_name('run_safe_default_parity.py')) == evidence.get('runner_sha256'),
            'Parity runner source changed')
    require([job.get('workload') for job in evidence['jobs']] == ['calibration', 'natural'],
            'Only the two exposed workloads may be executed')
    plan, outcomes, inputs = load_plan(args.plan), {}, []
    previous_end = None
    for job in evidence['jobs']:
        name, count = job['workload'], COUNTS[job['workload']]
        require(job.get('query_freshness') == 'previously_exposed_development'
                and job.get('timing_authoritative') is False and job.get('exit_code') == 0,
                'Exposed controls cannot claim fresh quality or authoritative timings')
        start, end = datetime.fromisoformat(job['start']), datetime.fromisoformat(job['end'])
        require(start.tzinfo is not None and end.tzinfo is not None and start <= end
                and (previous_end is None or previous_end <= start), 'Parity controls overlap or have invalid dates')
        previous_end = end
        directory = getattr(args, name + '_prepared')
        labels, query_hashes, identity = workload(directory, count)
        require(job['prepared_manifest_sha256'] == job['prepared_manifest_after_sha256'] == sha(directory / 'manifest.json')
                and job['prepared_artifacts_sha256'] == job['prepared_artifacts_after_sha256'] == query_hashes,
                'Exposed source queries changed')
        command = job['command']
        expected = {'--action': 'query', '--root': evidence['index_root'], '--prepared': job['prepared'],
                    '--out': job['aggregate_path'], '--threads': '8', '--query-threads': '8', '--l': '64',
                    '--reps': '1', '--limit': str(count), '--timing-authoritative': 'false',
                    '--evaluation-scope': 'frozen_policy_heldout', '--label-l': evidence['plan']['path'],
                    '--allow-unused-plan-labels': 'true', '--query-details-dir': str(Path(job['details_path']).parent)}
        require(command[0] == evidence['binary']['path'] and len(command) == 1 + 2 * len(expected)
                and {command[i]: command[i + 1] for i in range(1, len(command), 2)} == expected,
                'Retained query job command differs from the exposed-only fixed-L contract')
        current_aggregate = local_path(job['aggregate_path'], args.artifact_root, Path(evidence['output_root']))
        current_details = local_path(job['details_path'], args.artifact_root, Path(evidence['output_root']))
        require(sha(current_aggregate) == job['aggregate_sha256'] and sha(current_details) == job['details_sha256'],
                'New raw job files changed')
        reference_identity_path = getattr(args, 'reference_' + name + '_identity')
        require(sha(reference_identity_path) == REFERENCE_IDENTITY_SHA[name], 'Pinned V7 reference identity changed')
        reference_identity = json.loads(reference_identity_path.read_text())
        reference_engine = reference_identity['engines']['candidate']
        require(reference_engine['binary_sha256'] == V7_BINARY_SHA
                and reference_engine['source_archive']['sha256'] == V7_ARCHIVE_SHA,
                'Reference is not the preserved V7 candidate')
        reference_root = reference_identity_path.parent
        normalized = reference_root / 'candidate.normalized.jsonl'
        require(sha(normalized) == reference_engine['log_sha256'], 'V7 normalized source log changed')
        normalized_row = one([row for row in rows(normalized) if row.get('kind') == 'query' and row.get('repeat') == 0],
                             'Missing V7 first repeat')
        reference_aggregate = local_path(normalized_row['original_aggregate'], reference_root,
                                         Path(reference_identity['output_root']))
        require(sha(reference_aggregate) == normalized_row['original_aggregate_sha256'], 'V7 raw source job changed')
        reference_meta = one([row for row in rows(reference_aggregate) if row.get('kind') == 'query'], 'Missing V7 aggregate')
        reference_details = local_path(reference_meta['raw_queries']['path'], reference_root,
                                       Path(reference_identity['output_root']))
        current_exp, _, current_rows = observation(current_aggregate, current_details, labels, plan, identity, authoritative=False)
        reference_exp, _, reference_rows = observation(reference_aggregate, reference_details, labels, plan, identity,
                                                       authoritative=name == 'natural')
        require(current_exp.get('root') == evidence['index_root'] and current_exp.get('prepared') == job['prepared']
                and current_exp.get('prepared') == reference_exp.get('prepared'), 'Query or pool root declaration differs')
        require(reference_identity['prepared_manifest_sha256'] == sha(directory / 'manifest.json'),
                'Prepared partition differs from exposed V7 reference')
        outcomes[name] = ordered_result_parity(current_rows, reference_rows, labels, plan)
        inputs.extend(asset(path) for path in (current_aggregate, current_details, normalized, reference_aggregate,
                                               reference_details, reference_identity_path, directory / 'manifest.json'))
    result = {'schema_version': 1, 'status': 'All exposed output parity and local provenance checks passed',
              'scope': 'Conservative rollback regression control only; no new fresh validation, recall acceptance or performance claim',
              'workloads': outcomes, 'evidence': asset(args.evidence), 'rollback_decision': asset(args.rollback_decision),
              'plan': asset(args.plan), 'v10_source_archive': asset(args.source_archive),
              'v7_source_archive': asset(args.reference_source_archive), 'inputs': inputs,
              'binary_before_after_sha256': evidence['binary']['after_sha256'],
              'archive_core_sha256': evidence['archive_core_sha256'],
              'pool_index_unchanged': True, 'no_final_validation_partition_opened': True,
              'limitations': ['Binary/index before/after hashes are retained host declarations; raw outputs and source archives are locally rehashed.',
                             'Exact parity on these previously exposed queries does not establish equality for unseen queries.',
                             'V9 fresh validation does not validate the restored default; no fresh4316 queries were rerun.',
                             'Non-authoritative control latencies are not used for throughput or p99 claims.']}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'audit': str(args.out), 'workloads': outcomes}))


if __name__ == '__main__':
    main()
