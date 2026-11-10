#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Audit a fixed-L graph-builder correction against C++ and the previous Rust index.

The sealed formerly-private query split is never used to select a graph or L.
Bootstrap units are paired queries, not individual neighbors or timing repeats.
All labels are reported, including new regressions; this tool has no acceptance
threshold and never changes a frozen policy.
"""
import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics

import numpy as np

from plan_label_l import archived_files, load_plan
from summarize_delta import digest, integer, one, read_rows, require
from summarize_heldout import (
    FROZEN_SHA, METRICS, EXPECTED_SHA256, asset, cpp_observations, cutoff_diagnostic,
    local_path, matrix, native_observations, paired_bootstrap, prepared,
    repeat_summary, sha, sparse_labels, valid_result_ids, checked_query, validate_observations,
)

ENGINES = ('cpp', 'baseline', 'candidate')
TARGETS = (9, 24, 89, 98, 2495)
CONTRASTS = (('candidate', 'cpp'), ('candidate', 'baseline'), ('baseline', 'cpp'))
PRIVATE_SHA = {
    'query.private.2727415019.100K.u8bin': '0e06b358d41c2b9f4c7f5c2bc81f80061bc4d4b38d67b3c0b23d1700a3d8734d',
    'query.metadata.private.2727415019.100K.spmat': '174f0e04871223f8273b9516aff7a422c91ffb01c98d573c6746cf9718c70884',
    'GT.private.2727415019.ibin': '8d5628ce7a706d1ee346518515552ff8ee17de52daae3206be07e0273b932eae',
}
COMMITMENT_SHA = '3633c03931652bfb1f2cba352b2475732fac82c0c358899c510558985aae52c2'


def private_source_proof(validation, calibration, plan, args):
    """Independently verify source rows, namespace and global vector exclusions."""
    commitment = json.loads(args.commitment.read_text())
    evidence = json.loads(args.exclusion_proof.read_text())
    require(sha(args.commitment) == COMMITMENT_SHA, 'Fresh split commitment changed')
    require(commitment['policy_sha256'] == FROZEN_SHA
            and commitment['exclusion_evidence_sha256'] == sha(args.exclusion_proof),
            'Fresh split policy or exclusion evidence changed')
    sources = ((args.private_queries, 'query.private.2727415019.100K.u8bin'),
               (args.private_metadata, 'query.metadata.private.2727415019.100K.spmat'),
               (args.private_truth, 'GT.private.2727415019.ibin'))
    for path, name in sources:
        require(sha(path) == PRIVATE_SHA[name], f'Official private source changed: {name}')
    require(sha(args.public_queries) == EXPECTED_SHA256['query.public.100K.u8bin'],
            'Public exclusion source changed')
    source, _ = matrix(args.private_queries, 'u1')
    truth, distances = matrix(args.private_truth, '<u4', truth=True)
    ptr, labels = sparse_labels(args.private_metadata)
    public, _ = matrix(args.public_queries, 'u1')
    require(source.shape == (100000, 192) and len(ptr) == 100001 and truth.shape == (100000, 10),
            'Official private vector/metadata/truth shapes differ')
    excluded_embeddings = {row.tobytes() for row in public}
    seen_embeddings = set()
    eligible_ids = set()
    excluded_ids = []
    duplicate_ids = []
    for query_id in np.flatnonzero(np.diff(ptr) == 1):
        query_id = int(query_id)
        label = int(labels[ptr[query_id]])
        if label not in plan:
            continue
        vector = source[query_id].tobytes()
        if vector in excluded_embeddings:
            excluded_ids.append(query_id)
        elif vector in seen_embeddings:
            duplicate_ids.append(query_id)
        else:
            seen_embeddings.add(vector)
            eligible_ids.add(query_id)
    # Independently reconstruct the precommitted split and final query order.
    split_ids = {'calibration': [], 'validation': []}
    for label in sorted(plan):
        members = [i for i in eligible_ids if int(labels[ptr[i]]) == label]
        members.sort(key=lambda i: hashlib.sha256(f'20261023:split:{i}:{label}'.encode()).digest())
        cut = len(members) // 4
        split_ids['calibration'].extend((i, label) for i in members[:cut])
        split_ids['validation'].extend((i, label) for i in members[cut:])
    for name, rows in split_ids.items():
        rows.sort(key=lambda item: hashlib.sha256(
            f'20261023:order-{name}:{item[0]}:{item[1]}'.encode()).digest())
    partitions = {'validation': validation, 'calibration': calibration}
    observed_sets = {}
    for name, workload in partitions.items():
        meta = workload['manifest']
        require(meta.get('source_query_namespace') == 'official_private_2727415019_100K',
                'Private/public numeric query-ID namespaces must be distinguished')
        require(meta.get('split_exclusion_evidence_sha256') == sha(args.exclusion_proof),
                'Prepared exclusion proof binding differs')
        declared = commitment['partitions'][name]
        directory = args.prepared if name == 'validation' else args.calibration_prepared
        require(sha(directory / 'manifest.json') == declared['manifest_sha256']
                and len(workload['queries']) == declared['queries'], 'Committed fresh split assets/count changed')
        for filename, expected in declared['artifacts'].items():
            require(expected['sha256'] == sha(directory / filename)
                    and expected['bytes'] == (directory / filename).stat().st_size,
                    f'Committed fresh split artifact changed: {name}/{filename}')
        ids = workload['ids']
        require(ids.tolist() == [i for i, _ in split_ids[name]]
                and workload['labels'].tolist() == [label for _, label in split_ids[name]],
                'Prepared split or query order differs from the independently reconstructed frozen selection')
        require(np.all(ids < len(source)) and np.all(np.diff(ptr)[ids] == 1)
                and np.array_equal(source[ids], workload['queries'])
                and np.array_equal(labels[ptr[ids]], workload['labels'])
                and np.array_equal(truth[ids], workload['truth'])
                and np.array_equal(distances[ids], workload['distances']),
                'Fresh prepared query/filter/truth differs from official original row')
        require(set(map(int, workload['labels'])) <= set(plan), 'Fresh query label absent from frozen L plan')
        vectors = [row.tobytes() for row in workload['queries']]
        require(len(set(vectors)) == len(vectors) and not set(vectors) & excluded_embeddings,
                'Fresh vectors duplicate each other or overlap any public vector')
        observed_sets[name] = set(map(int, ids))
    require(not observed_sets['validation'] & observed_sets['calibration']
            and observed_sets['validation'] | observed_sets['calibration'] == eligible_ids,
            'Calibration/validation IDs overlap or do not exhaust the fresh eligible natural pool')
    require(not {row.tobytes() for row in validation['queries']} &
            {row.tobytes() for row in calibration['queries']}, 'Fresh calibration and validation embeddings overlap')
    require(sorted(excluded_ids) == sorted(row['query_id'] for row in evidence['excluded_query_rows'])
            and not duplicate_ids and not evidence['excluded_duplicates'], 'Private exclusion reconstruction differs')
    counts = Counter(map(int, validation['labels']))
    require(set(counts) == set(plan), 'Fresh validation does not cover all frozen-plan labels')
    return {'commitment': asset(args.commitment), 'exclusion_evidence': asset(args.exclusion_proof),
            'query_id_namespace': 'official_private_2727415019_100K',
            'source_assets': [asset(path) for path, _ in sources],
            'public_exclusion_source': asset(args.public_queries),
            'validation_queries': len(validation['queries']), 'calibration_queries': len(calibration['queries']),
            'validation_labels': len(counts), 'query_embedding_overlap_with_all_public_queries': 0,
            'calibration_validation_query_id_overlap': 0, 'calibration_validation_embedding_overlap': 0,
            'duplicate_fresh_embeddings': 0, 'excluded_official_private_query_ids': excluded_ids,
            'all_remaining_eligible_natural_private_queries_partitioned': True,
            'target_validation_counts': {str(label): counts[label] for label in TARGETS},
            'labels_with_fewer_than_100_queries': sorted(label for label, count in counts.items() if count < 100),
            'query_weighted_mean_l': sum(count * plan[label] for label, count in counts.items()) / len(validation['queries']),
            'truth_validation_scope': 'Published full-10M exact top10 source rows copied; membership and integer distances checked independently; no exhaustive nearest-neighbor recomputation.'}


SECONDARY_SELECTION_SHA = 'ea9289d1a23b3a6b17e9aee552b4c0d194d122c06d4a144918444f6bc5c26321'
SECONDARY_EVALUATION_SHA = '56a072bf269ee742cadc70c13858d3947b95ba6a3e4057f8a40c857a9cc88bdf'


def supplemental_prepared(directory):
    """Numeric source IDs may repeat across the two explicitly recorded namespaces."""
    manifest = json.loads((directory / 'manifest.json').read_text())
    require(manifest.get('base_vectors') == 10000000 and manifest.get('dimension') == 192
            and manifest.get('k') == 10 and manifest.get('dtype') == 'uint8'
            and manifest.get('metric') == 'L2', 'Unexpected supplemental corpus/metric')
    for name in ('query.u8bin', 'query_labels.u32bin', 'query_ids.u32bin', 'groundtruth.bin',
                 'labels.txt', 'query_labels.txt', 'source_namespace.u8bin', 'source-query-rows.jsonl'):
        expected = manifest.get('artifacts', {}).get(name, {})
        # Source row mapping may be separately bound by the source_query_rows field.
        if name == 'source-query-rows.jsonl':
            expected = manifest['source_query_rows']
        require(expected.get('sha256') == sha(directory / name)
                and expected.get('bytes') == (directory / name).stat().st_size,
                f'Supplemental prepared asset changed: {name}')
    queries, _ = matrix(directory / 'query.u8bin', 'u1')
    labels, _ = matrix(directory / 'query_labels.u32bin', '<u4')
    ids, _ = matrix(directory / 'query_ids.u32bin', '<u4')
    namespace, _ = matrix(directory / 'source_namespace.u8bin', 'u1')
    truth, distances = matrix(directory / 'groundtruth.bin', '<u4', truth=True)
    count = len(queries)
    require(queries.shape == (count, 192) and labels.shape == (count, 1) and ids.shape == (count, 1)
            and namespace.shape == (count, 1) and truth.shape == (count, 10)
            and manifest.get('query_count') == count and np.all(namespace <= 1)
            and np.all(ids < 100000) and np.all(truth < 10000000)
            and np.all(np.isfinite(distances)) and np.all(distances >= 0), 'Supplemental matrix shapes/domain differ')
    require(len(set(zip(namespace[:, 0].tolist(), ids[:, 0].tolist()))) == count,
            'Supplemental source namespace/ID pairs repeat')
    require([int(value) for value in (directory / 'query_labels.txt').read_text().splitlines()]
            == labels[:, 0].tolist(), 'Supplemental C++/Rust predicates differ')
    return {'manifest': manifest, 'queries': queries, 'labels': labels[:, 0], 'ids': ids[:, 0],
            'namespace': namespace[:, 0], 'truth': truth, 'distances': distances,
            'identity': ':'.join(manifest['artifacts'][name]['sha256'] for name in
                                ('query.u8bin', 'query_labels.u32bin', 'groundtruth.bin', 'labels.txt'))}


def supplemental_source_proof(workload, args):
    directory = args.supplemental_prepared
    selection_path, evaluation_path = directory / 'selection-commitment.json', directory / 'evaluation-commitment.json'
    require(sha(selection_path) == SECONDARY_SELECTION_SHA and sha(evaluation_path) == SECONDARY_EVALUATION_SHA,
            'Supplemental selection/evaluation commitment changed')
    selection, evaluation = json.loads(selection_path.read_text()), json.loads(evaluation_path.read_text())
    require(evaluation.get('scope') == 'supplemental_shifted_atomic_label_validation_only'
            and evaluation.get('selection_commitment_sha256') == sha(selection_path)
            and evaluation.get('primary_commitment_sha256') == COMMITMENT_SHA
            and evaluation.get('policy_sha256') == FROZEN_SHA,
            'Supplemental scope/primary split/L policy binding differs')
    expected = evaluation['partitions']['secondary-validation']
    require(expected['queries'] == len(workload['queries'])
            and expected['manifest_sha256'] == sha(directory / 'manifest.json'),
            'Supplemental committed workload differs')
    for name, declared in expected['artifacts'].items():
        require(declared['sha256'] == sha(directory / name)
                and declared['bytes'] == (directory / name).stat().st_size,
                f'Committed supplemental artifact changed: {name}')
    require(workload['manifest'].get('split') == 'secondary_validation'
            and workload['manifest'].get('source_query_namespace') == 'mixed_public_and_official_private_2727415019_100K'
            and workload['manifest'].get('validation_scope') == 'recall_fix_supplemental_shifted_atomic_label_validation_only',
            'Supplemental distribution-shift namespace/scope missing')
    excluded = set()
    for declared in selection['excluded_query_artifacts']:
        path = Path(declared['path'])
        require(path.is_file() and declared['sha256'] == sha(path)
                and declared['bytes'] == path.stat().st_size, 'Supplemental exposed/reserved-query exclusion source changed')
        vectors, _ = matrix(path, 'u1')
        excluded.update(row.tobytes() for row in vectors)
    require(len(excluded) == selection['excluded_embeddings'], 'Supplemental exclusion pool size differs')
    def rank(scope, *parts):
        return hashlib.sha256(':'.join(map(str, (20261024, scope, *parts))).encode()).digest()
    sources = {}
    known = dict(PRIVATE_SHA)
    known.update(EXPECTED_SHA256)
    for declared in selection['source_records']:
        path = Path(declared['path'])
        require(known.get(path.name) == declared['sha256'] == sha(path)
                and declared['bytes'] == path.stat().st_size, 'Supplemental original source changed')
    source_specs = [('public_100K', args.public_queries, args.public_queries.with_name('query.metadata.public.100K.spmat')),
                    ('official_private_2727415019_100K', args.private_queries, args.private_metadata)]
    reconstructed, excluded_rows, duplicate_rows, available = {}, [], [], Counter()
    for namespace, query_path, metadata_path in source_specs:
        vectors, _ = matrix(query_path, 'u1')
        pointers, tags = sparse_labels(metadata_path)
        sources[namespace] = vectors
        for query_id in np.flatnonzero(np.diff(pointers) == 2):
            query_id = int(query_id)
            original = list(map(int, tags[pointers[query_id]:pointers[query_id + 1]]))
            eligible = set(original) & set(TARGETS)
            if not eligible:
                continue
            available[namespace] += 1
            vector = vectors[query_id].tobytes()
            vector_sha = hashlib.sha256(vector).hexdigest()
            row = {'source_query_namespace': namespace, 'source_query_id': query_id,
                   'original_and_tags': original, 'embedding_sha256': vector_sha}
            if vector in excluded:
                excluded_rows.append(row)
                continue
            row['atomic_label'] = min(eligible, key=lambda label: rank('owner', vector_sha, label))
            old = reconstructed.get(vector)
            if old is None:
                reconstructed[vector] = row
            elif rank('representative', namespace, query_id) < rank('representative', old['source_query_namespace'], old['source_query_id']):
                duplicate_rows.append(old)
                reconstructed[vector] = row
            else:
                duplicate_rows.append(row)
    rows = sorted(reconstructed.values(), key=lambda row: rank('order', row['embedding_sha256']))
    for index, row in enumerate(rows):
        row['query_index'] = index
    retained = read_rows(directory / 'source-query-rows.jsonl')
    require(rows == retained and len(rows) == len(workload['queries'])
            and excluded_rows == selection['excluded_rows'] and duplicate_rows == selection['duplicate_rows_removed']
            and dict(available) == selection['eligible_two_tag_source_rows_before_exclusions'],
            'Supplemental candidate exclusion/ownership/order reconstruction differs')
    for index, row in enumerate(rows):
        namespace, source_id = row['source_query_namespace'], row['source_query_id']
        require(np.array_equal(workload['queries'][index], sources[namespace][source_id])
                and int(workload['ids'][index]) == source_id
                and int(workload['namespace'][index]) == int(namespace != 'public_100K')
                and int(workload['labels'][index]) == row['atomic_label'],
                'Supplemental original vector/namespace/atomic filter differs')
    counts = Counter(map(int, workload['labels']))
    require(set(counts) == set(TARGETS), 'Supplemental validation target coverage incomplete')
    return {'selection_commitment': asset(selection_path), 'evaluation_commitment': asset(evaluation_path),
            'source_query_rows': asset(directory / 'source-query-rows.jsonl'), 'queries': len(rows),
            'per_label_queries': dict(sorted(counts.items())), 'source_namespace_counts': dict(Counter(row['source_query_namespace'] for row in rows)),
            'embedding_overlap_with_exposed_or_reserved_queries': 0, 'duplicate_validation_embeddings': 0,
            'original_predicate': 'AND of two source tags', 'evaluated_predicate': 'One constituent target label chosen by frozen hash rank',
            'distribution_shift': 'Derived atomic predicates from untouched two-tag vectors; keep separate from natural one-tag results.',
            'truth': 'New exhaustive full10M atomic truth artifact; original AND truth is not reused; exact distances/membership independently checked during this analysis.',
            'validation_only': True, 'calibration_queries': 0}


def quality_observations(path, workload, plan, postings, root, remote_root, native):
    """Use identical quality checks for calibration while excluding its timings."""
    rows = read_rows(path)
    if native:
        rows = [row for row in rows if row.get('kind') == 'query']
    require(len(rows) == 1, 'Calibration requires exactly one quality-only observation per engine')
    row = rows[0]
    checked_query(row, workload, plan, native, policy=True, authoritative=False)
    if native:
        raw = local_path(row['raw_queries_relative_path'], root, remote_root)
        require(row['raw_queries']['sha256'] == sha(raw)
                and row['raw_queries']['queries'] == len(workload['queries']), 'Calibration raw detail hash/count differs')
        selected = sorted(read_rows(raw), key=lambda item: item.get('query_index', -1))
        require([item.get('query_index') for item in selected] == list(range(row['queries'])),
                'Calibration raw query indices incomplete/repeated')
        for index, item in enumerate(selected):
            require(item.get('label') == int(workload['labels'][index]) and item.get('l') == 64
                    and item.get('actual_l') == plan[item['label']] and item.get('k') == 10
                    and item.get('repeat') == 0 and item.get('phase') == 'query'
                    and item.get('sequence') == 0 and item.get('exact_fallback') is False
                    and item.get('base_searches') == 1 and item.get('delta_candidates') == 0,
                    'Calibration raw policy/snapshot differs')
        ids = np.asarray([item['ids'] for item in selected], dtype=np.int64)
        inputs = {'raw_queries': asset(raw)}
    else:
        raw = local_path(row['raw_stats_relative_path'], root, remote_root)
        ids_path = local_path(row['result_ids_relative_path'], root, remote_root)
        require(row.get('result_ids_sha256') == sha(ids_path), 'Calibration C++ ID hash differs')
        raw_rows = read_rows(raw)
        summary = one([item for item in raw_rows if item.get('kind') == 'summary'], 'calibration C++ raw summary')
        require(summary.get('query_count') == row['queries'] and summary.get('warmup_queries') == row['queries']
                and summary.get('threads') == 8 and summary.get('k') == 10 and summary.get('l') == 64
                and summary.get('beam_width') == 4 and summary.get('latency_contract') == 'public_search_call',
                'Calibration C++ raw protocol differs')
        selected = sorted([item for item in raw_rows if item.get('kind') == 'query'],
                          key=lambda item: item.get('query_index', -1))
        require(len(raw_rows) == len(selected) + 1
                and [item.get('query_index') for item in selected] == list(range(row['queries'])),
                'Calibration C++ raw query indices incomplete/repeated')
        require(all(item.get('returned_results') == 10 and item.get('threads') == 8
                    and item.get('k') == 10 and item.get('l') == 64 and item.get('beam_width') == 4
                    for item in selected), 'Calibration C++ raw results/protocol differ')
        ids, _ = matrix(ids_path, '<u4')
        inputs = {'raw_queries': asset(raw), 'result_ids': asset(ids_path)}
    observation = validate_observations(row, ids, [item['latency_us'] for item in selected], workload, postings, native)
    if native:
        require(all(item.get('strict_id_hits') == int(hits) for item, hits in zip(selected, observation['hits'])),
                'Calibration raw declared strict-ID hits differ')
    return [observation], [dict(inputs, repeat=0)]


def triple_jobs_proof(path, logs, artifact_root, quality_only=False):
    manifest = json.loads(path.read_text())
    jobs = manifest.get('runs', [])
    repeats = 1 if quality_only else 6
    require(manifest.get('schema_version') == 1 and manifest.get('status') == 'complete'
            and len(jobs) == 3 * repeats, 'Incomplete serial three-engine evaluation blocks')
    remote_root = Path(manifest['output_root'])
    merged = {name: (read_rows(log) if name == 'cpp' else
                     [row for row in read_rows(log) if row.get('kind') == 'query']) for name, log in logs.items()}
    allowed = {'repeat', 'paired_repeat', 'local_repeat', 'local_raw_queries', 'local_raw_stats', 'local_result_ids',
               'raw_queries_relative_path', 'raw_stats_relative_path', 'result_ids_relative_path',
               'original_aggregate', 'original_aggregate_sha256'}
    seen = set()
    experiment_rows = {'baseline': [], 'candidate': []}
    for index, job in enumerate(jobs):
        name, repeat = job.get('implementation'), job.get('paired_repeat')
        require(name in ENGINES and integer(repeat) and job.get('local_repeat') == 0 and job.get('exit_code') == 0,
                'Invalid timing job engine/repeat/status')
        require((name, repeat) not in seen, 'Repeated three-engine timing job')
        seen.add((name, repeat))
        source = local_path(job['aggregate_path'], artifact_root, remote_root)
        require(sha(source) == job['aggregate_sha256'], 'Original timing aggregate bytes changed')
        original_rows = read_rows(source)
        original = one(original_rows if name == 'cpp' else
                       [row for row in original_rows if row.get('kind') == 'query'], 'original timing aggregate')
        row = one([row for row in merged[name] if row.get('repeat') == repeat], 'merged timing aggregate')
        require(original.get('repeat') == 0 and row.get('paired_repeat') == repeat and row.get('local_repeat') == 0
                and {key: value for key, value in original.items() if key not in allowed}
                == {key: value for key, value in row.items() if key not in allowed},
                'Merged timing row changed beyond repeat and retained-path annotations')
        require(row.get('original_aggregate_sha256') == sha(source), 'Merged original aggregate binding differs')
        start, end = job.get('start'), job.get('end')
        require(type(start) is str and type(end) is str and start < end, 'Invalid timing interval')
        if index:
            require(jobs[index - 1]['end'] <= start, 'Timing jobs overlap')
        if name != 'cpp':
            experiment_rows[name].extend(row for row in original_rows if row.get('kind') == 'experiment')
    orders = []
    for repeat in range(repeats):
        block = jobs[repeat * 3: repeat * 3 + 3]
        order = tuple(job['implementation'] for job in block)
        require(set(order) == set(ENGINES) and all(job['paired_repeat'] == repeat for job in block),
                'Each serial block must contain all three engines once')
        require([job.get('order_in_pair') for job in block] == list(range(3)), 'Timing order annotations differ')
        orders.append(order)
    require(quality_only or set(orders) == set(itertools.permutations(ENGINES)), 'All six timing orders were not represented')
    require(seen == {(name, repeat) for name in ENGINES for repeat in range(repeats)}, 'Timing repeat matrix incomplete')
    for name, rows in experiment_rows.items():
        require(len(rows) == repeats and all(row == rows[0] for row in rows), f'{name} experiments changed across repeats')
    return {'source': asset(path), 'blocks': repeats, 'orders': [list(order) for order in orders],
            'merged_rows_match_original_aggregates': True, 'declared_timing_jobs_do_not_overlap': True}


def contrast_summary(positive, reference, labels, repetitions, seed):
    differences = positive['hits'] - reference['hits']
    summary = paired_bootstrap(differences, 10, repetitions, seed, families=3)
    summary['positive_wins_queries'] = summary.pop('rust_wins_queries')
    summary['reference_wins_queries'] = summary.pop('cpp_wins_queries')
    summary['macro_label_difference_percentage_points'] = float(np.mean([
        differences[labels == label].mean() for label in sorted(set(map(int, labels)))])) / 10 * 100
    return summary


def compare_engines(observations, workload, plan, postings, repetitions, seed, quality_only=False):
    summaries = ({name: {'repetitions': 1, 'strict_id_recall': float(rows[0]['hits'].mean()) / 10}
                  for name, rows in observations.items()} if quality_only else
                 {name: repeat_summary(rows) for name, rows in observations.items()})
    require(all(len(rows) == (1 if quality_only else 6) for rows in observations.values()),
            'Evaluation repetition count differs')
    for name, rows in observations.items():
        require(all(np.array_equal(row['ids'], rows[0]['ids']) for row in rows),
                f'{name} ordered result IDs differ across timing repeats')
    labels = sorted(set(map(int, workload['labels'])))
    aggregate_contrasts = {}
    for index, (positive, reference) in enumerate(CONTRASTS):
        key = f'{positive}_minus_{reference}'
        result = contrast_summary(observations[positive][0], observations[reference][0], workload['labels'],
                                  repetitions, seed + index)
        result['timing_ratios'] = {}
        for metric in (() if quality_only else METRICS):
            ratios = [a[metric] / b[metric] for a, b in zip(observations[positive], observations[reference])]
            result['timing_ratios'][metric] = {'median_paired_ratio': statistics.median(ratios),
                'range_paired_ratio': [min(ratios), max(ratios)], 'paired_ratios': ratios,
                'ratio_of_medians': summaries[positive][metric] / summaries[reference][metric]}
        aggregate_contrasts[key] = result
    per_label = []
    for label in labels:
        mask = workload['labels'] == label
        row = {'label': label, 'queries': int(mask.sum()), 'eligible_vectors': len(postings[label]),
               'match_fraction': len(postings[label]) / 10_000_000, 'cpp_l': 64, 'rust_l': plan[label],
               'target_label': label in TARGETS, 'p99_fewer_than_100_queries': int(mask.sum()) < 100}
        for name in ENGINES:
            series = observations[name]
            row[f'{name}_recall_percent'] = float(series[0]['hits'][mask].mean()) / 10 * 100
            if quality_only:
                continue
            row[f'{name}_mean_ms'] = statistics.median(item['labels'][label]['mean_us'] for item in series) / 1000
            row[f'{name}_p99_ms'] = statistics.median(item['labels'][label]['p99_us'] for item in series) / 1000
            row[f'{name}_p99_range_min_ms'] = min(item['labels'][label]['p99_us'] for item in series) / 1000
            row[f'{name}_p99_range_max_ms'] = max(item['labels'][label]['p99_us'] for item in series) / 1000
        for index, (positive, reference) in enumerate(CONTRASTS):
            key = f'{positive}_minus_{reference}'
            difference = observations[positive][0]['hits'][mask] - observations[reference][0]['hits'][mask]
            bootstrap = paired_bootstrap(difference, 10, repetitions, seed + label * 3 + index,
                                         families=3 * len(labels))
            row[f'{key}_pp'] = bootstrap['difference_percentage_points']
            for scope, field in [('marginal', 'paired_query_bootstrap_95_percent_interval_pp'),
                                 ('simultaneous', 'paired_query_bootstrap_simultaneous_95_percent_interval_pp')]:
                row[f'{key}_{scope}_95_lower_pp'], row[f'{key}_{scope}_95_upper_pp'] = bootstrap[field]
            row[f'{key}_empirical_variance_is_zero'] = bootstrap['bootstrap_empirical_difference_variance_is_zero']
            row[f'{key}_wins_queries'] = bootstrap['rust_wins_queries']
            row[f'{key}_losses_queries'] = bootstrap['cpp_wins_queries']
        per_label.append(row)
    counts = {}
    for positive, reference in CONTRASTS:
        key = f'{positive}_minus_{reference}'
        counts[key] = {'higher_labels': [row['label'] for row in per_label if row[f'{key}_pp'] > 0],
                       'equal_labels': [row['label'] for row in per_label if row[f'{key}_pp'] == 0],
                       'lower_labels': [row['label'] for row in per_label if row[f'{key}_pp'] < 0]}
    for name in ENGINES:
        summaries[name]['macro_label_recall'] = statistics.mean(row[f'{name}_recall_percent'] for row in per_label) / 100
    return {'engines': summaries, 'contrasts': aggregate_contrasts, 'per_label': per_label,
            'label_comparison': counts}


def clone_proof(args, engines, candidate_root):
    """Bind the corrected artifact intervention without rewriting build history."""
    paths = (getattr(args, 'clone_result', None), getattr(args, 'clone_provenance', None),
             getattr(args, 'original_graph_build_log', None))
    if not any(paths):
        return None
    require(all(paths), 'PQ correction audit requires clone result, checksummed provenance, and original graph build log')
    result_path, provenance_path, build_path = paths
    result, provenance = json.loads(result_path.read_text()), json.loads(provenance_path.read_text())
    require(sha(result_path) == sha(args.candidate_build_log),
            'Freeze must bind the PQ clone result as candidate artifact construction evidence')
    host_base = candidate_root.parent.parent
    def canonical(value):
        path = Path(value)
        return Path(os.path.normpath(str(path if path.is_absolute() else host_base / path)))
    source_root = canonical(result['source_root'])
    donor_root = candidate_root.parent / 'rust-native-10m-frozen'
    source_directory = source_root / 'generation-00000000000000000001'
    output_directory = candidate_root / 'generation-00000000000000000001'
    donor_directory = donor_root / 'generation-00000000000000000001'
    require(result.get('schema_version') == provenance.get('schema_version') == 1
            and result.get('status') == 'published and reopened successfully'
            and canonical(result.get('output_root', '')) == candidate_root
            and canonical(result.get('generation_directory', '')) == output_directory
            and result.get('generation') == 1 and result.get('validated_sequence') == 0
            and canonical(result.get('pq_donor_root', '')) == canonical(provenance.get('pq_donor_root', '')) == donor_root
            and canonical(provenance.get('source_root', '')) == source_root, 'PQ clone root/publication identity differs')
    require(result.get('source_manifest_preserved_verbatim') is True
            and provenance.get('source_manifest_preserved_verbatim') is True
            and provenance.get('replaced_files') == ['index_pq_pivots.bin', 'index_pq_compressed.bin']
            and result.get('input_sha256') == provenance.get('input_sha256'),
            'PQ clone changed build history or replaced files beyond the complete PQ pair')
    binding = result.get('bindings', {})
    require(binding == provenance.get('bindings') and binding.get('rows') == 10000000
            and binding.get('dimensions') == 192 and binding.get('postings_equal') is True
            and binding.get('stable_ids') == 'identity 0..N in both generations'
            and binding.get('source_entries_preserved') is True,
            'PQ clone vector/ordinal/membership bindings differ')
    def canonical_hashes(declarations):
        normalized = {}
        for path, value in declarations.items():
            key = str(canonical(path))
            require(key not in normalized or normalized[key] == value, 'Conflicting PQ clone path aliases')
            normalized[key] = value
        return normalized
    incoming, outgoing = canonical_hashes(result['input_sha256']), canonical_hashes(result['output_sha256'])
    require(all(digest(value) for value in incoming.values()) and all(digest(value) for value in outgoing.values()),
            'Invalid PQ clone SHA256 declaration')
    installed = engines['candidate']['index_before']['files_sha256']
    donor = engines['baseline']['index_before']['files_sha256']
    require(installed.get('clone-provenance.json') == sha(provenance_path),
            'PQ clone provenance is not the checksum-bound installed artifact')
    for name, expected in installed.items():
        require(outgoing.get(str(output_directory / name)) == expected,
                f'Installed candidate file differs from published PQ clone: {name}')
    for key, name in [('current_sha256', 'CURRENT'), ('journal_sha256', 'updates.wal')]:
        require(outgoing.get(str(candidate_root / name)) == engines['candidate']['index_before'][key]
                and incoming.get(str(donor_root / name)) == engines['baseline']['index_before'][key],
                f'PQ clone publication/donor envelope differs: {name}')
    require(outgoing.get(str(output_directory / 'CATALOG')) == engines['candidate']['index_before']['catalog_sha256']
            and incoming.get(str(donor_directory / 'CATALOG')) == engines['baseline']['index_before']['catalog_sha256'],
            'PQ clone output/donor catalog differs')
    for name in ('index_pq_pivots.bin', 'index_pq_compressed.bin'):
        require(installed.get(name) == donor.get(name) == incoming.get(str(donor_directory / name)),
                f'Corrected candidate PQ file differs from frozen baseline: {name}')
    preserved = [name for name in installed if name not in
                 ('CATALOG', 'clone-provenance.json', 'index_pq_pivots.bin', 'index_pq_compressed.bin')]
    require({'vectors.bin', 'labels.bin', 'index_disk.index'} <= set(preserved),
            'PQ clone is missing source graph/vector/label artifacts')
    for name in preserved:
        require(installed[name] == incoming.get(str(source_directory / name)),
                f'PQ clone changed source geometry or metadata: {name}')
    require(installed['vectors.bin'] == donor.get('vectors.bin')
            == incoming.get(str(donor_directory / 'vectors.bin')), 'PQ clone donor vector ordinal bytes differ')
    builds = [row for row in read_rows(build_path) if row.get('kind') == 'build']
    build = one(builds, 'original native graph build')
    require(build.get('implementation') == 'rust_native_lifecycle' and build.get('rows') == 10000000
            and build.get('dimensions') == 192
            and canonical(build.get('root', '')) == source_root,
            'Original graph build does not identify the cloned source geometry')
    require(provenance.get('original_source_manifest', {}).get('encoding') == 'u8'
            and provenance['original_source_manifest'].get('base_sequence') == 0,
            'PQ clone original manifest domain/sequence differs')
    return {'clone_result': asset(result_path), 'checksummed_clone_provenance': asset(provenance_path),
            'original_graph_build_log': asset(build_path),
            'original_build_initial_pq_codebook': build.get('config', {}).get('initial_pq_codebook'),
            'installed_pq_files_equal_frozen_baseline': True, 'candidate_graph_vectors_labels_preserved': True,
            'complete_vectors_equal_pq_donor': True,
            'scope': 'Explicit artifact intervention after original graph construction; original manifest/configuration preserved, not rewritten to claim codebook reuse.'}


def identity_proof(args, logs, manifest):
    evidence = json.loads(args.identity.read_text())
    prior = json.loads(args.prior_identity.read_text())
    require(evidence.get('schema_version') == 1 and evidence.get('cpu_affinity') == list(range(8))
            and evidence.get('host') == prior.get('host'), 'Host or physical-core affinity changed')
    require(evidence.get('plan_before_sha256') == evidence.get('plan_after_sha256') == FROZEN_SHA,
            'L policy changed during fresh validation')
    supplemental = getattr(args, 'supplemental', False)
    selected_prepared = (args.supplemental_prepared if supplemental else
                         args.calibration_prepared if args.quality_only else args.prepared)
    fresh_commitment = selected_prepared / 'evaluation-commitment.json' if supplemental else args.commitment
    require(evidence.get('prepared_manifest_sha256') == sha(selected_prepared / 'manifest.json')
            and manifest.get('prepared_manifest_sha256') == sha(selected_prepared / 'manifest.json')
            and evidence.get('output_root') == manifest.get('output_root'), 'Fresh workload or output-root binding differs')
    require(manifest.get('identity') == evidence, 'Run manifest and identity evidence differ')
    engines = evidence.get('engines', {})
    require(set(engines) == set(ENGINES), 'Missing three-engine identity')
    for name in ENGINES:
        item = engines[name]
        require(digest(item.get('binary_sha256')) and item.get('binary_before_sha256') == item['binary_sha256']
                and item.get('log_sha256') == sha(logs[name]), f'{name} binary/log identity changed')
        before, after = item.get('index_before', {}), item.get('index_after', {})
        require(before and before == after, f'{name} index identity changed during evaluation')
        flat = {key: value for key, value in before.items() if key != 'files_sha256'}
        require(all(digest(value) for value in flat.values()), f'{name} index SHA256 declarations invalid')
        if name != 'cpp':
            files = before.get('files_sha256', {})
            require(files and all(type(path) is str and digest(value) for path, value in files.items()),
                    f'{name} immutable generation file hash declarations invalid')
    require({key: engines['baseline']['index_before'][key] for key in prior['native']['index_before']} == prior['native']['index_before']
            and engines['baseline']['binary_sha256'] == prior['native']['binary_sha256'],
            'Baseline Rust graph/runtime differs from the prior held-out evaluation')
    require(engines['cpp']['index_before'] == prior['cpp']['index_before']
            and engines['cpp']['binary_sha256'] == prior['cpp']['binary_sha256'], 'C++ baseline assets changed')
    archived = {}
    for name, path in [('baseline', args.baseline_source_archive), ('candidate', args.candidate_source_archive)]:
        item = engines[name].get('source_archive', {})
        require(item.get('sha256') == item.get('before_sha256') == sha(path), f'{name} source archive changed')
        archived[name] = archived_files(path)
    require(set(archived['baseline']) == set(archived['candidate']), 'Candidate source archive file set differs')
    changed = sorted(name for name in archived['baseline'] if archived['baseline'][name] != archived['candidate'][name])
    require(changed == ['diskann-disk/src/build/filtered.rs'], 'Changes exceed the frozen graph-builder correction')
    require(sha(args.baseline_source_archive) == prior['native']['source_archive']['sha256'],
            'Baseline source archive differs from prior held-out evaluation')
    freeze = json.loads(args.candidate_freeze.read_text())
    phase = 'secondary-validation' if supplemental else 'calibration' if args.quality_only else 'validation'
    require(freeze.get('schema_version') == 1 and freeze.get('phase') == phase
            and manifest.get('phase') == phase and manifest.get('timing_authoritative') is (not args.quality_only),
            'Evaluation phase or authoritative-timing declaration differs')
    require(freeze.get('plan_sha256') == FROZEN_SHA
            and freeze.get('fresh_proof_sha256') == sha(fresh_commitment)
            and freeze.get('prepared_manifest_sha256') == sha(selected_prepared / 'manifest.json')
            and freeze.get('candidate_build_log_sha256') == sha(args.candidate_build_log),
            'Pre-evaluation freeze does not bind the original policy, fresh split, or build log')
    require(manifest.get('candidate_freeze_sha256') == sha(args.candidate_freeze)
            == evidence.get('candidate_freeze_sha256')
            and evidence.get('fresh_proof_sha256') == manifest.get('fresh_proof_sha256') == sha(fresh_commitment),
            'Run manifest does not bind the candidate freeze and fresh split')
    for name in ENGINES:
        require(freeze.get('binary_sha256', {}).get(name) == engines[name]['binary_sha256']
                and freeze.get('index_identity', {}).get(name) == engines[name]['index_before'],
                f'{name} assets changed between freeze and validation')
    for name in ('baseline', 'candidate'):
        require(freeze.get('source_archive_sha256', {}).get(name) == engines[name]['source_archive']['sha256'],
                f'{name} archived sources changed between freeze and validation')
    require(type(freeze.get('timestamp_utc')) is str
            and freeze['timestamp_utc'] <= manifest['runs'][0]['start'],
            'Candidate freeze was not recorded before all fresh validation searches')
    candidate_root = Path(freeze['candidate_root'])
    expected_roots = {'candidate': str(candidate_root),
                      'baseline': str(candidate_root.parent / 'rust-native-10m-frozen')}
    expected_cpp_prefix = str(candidate_root.parent / 'yfcc10m_stitched_R32_R128_L100')
    for name in ('baseline', 'candidate'):
        experiment = one([row for row in read_rows(logs[name]) if row.get('kind') == 'experiment'],
                         f'{name} retained experiment header')
        require(experiment.get('root') == expected_roots[name]
                and experiment.get('prepared') == manifest.get('prepared')
                and experiment.get('action') == 'query' and experiment.get('n_base') == 10000000
                and experiment.get('dimensions') == 192 and experiment.get('index_threads') == 8
                and experiment.get('beam') == 4 and experiment.get('shared_cache_nodes') == 0
                and experiment.get('cache_bypass') is True and experiment.get('storage') == 'native_io_uring'
                and experiment.get('warmup') == 'full_same_process_same_worker_count'
                and experiment.get('query_scheduling') == 'dynamic_one',
                f'{name} experiment index/workload/I/O protocol differs')
    require(all(row.get('index_prefix') == expected_cpp_prefix and row.get('prepared') == manifest.get('prepared')
                for row in read_rows(logs['cpp'])), 'C++ experiment index/workload differs')
    for job in manifest['runs']:
        command = job.get('command', [])
        name = job['implementation']
        option = '--index-prefix' if name == 'cpp' else '--root'
        require(option in command and command[command.index(option) + 1] ==
                (expected_cpp_prefix if name == 'cpp' else expected_roots[name]),
                f'{name} retained job command index differs')
    if supplemental:
        primary = json.loads(args.primary_identity.read_text())
        require(primary.get('phase') == 'validation', 'Supplemental result must bind a primary validation run')
        for name in ENGINES:
            require(engines[name]['binary_sha256'] == primary['engines'][name]['binary_sha256']
                    and engines[name]['index_before'] == primary['engines'][name]['index_after'],
                    f'{name} binary/index changed between primary and supplemental validation')
        for name in ('baseline', 'candidate'):
            require(engines[name]['source_archive']['sha256'] == primary['engines'][name]['source_archive']['sha256'],
                    f'{name} sources changed between primary and supplemental validation')
    clone = clone_proof(args, engines, candidate_root)
    runner = Path(__file__).with_name('run_recall_fix.py')
    cpp_runner = Path(__file__).parent / 'cpp-port/run_reference.py'
    require(evidence.get('runner_sha256') == sha(runner)
            and evidence.get('cpp_runner_sha256') == sha(cpp_runner), 'Timing runner sources changed')
    return {'source': asset(args.identity), 'previous_heldout_identity': asset(args.prior_identity),
            'declared_identity': evidence, 'candidate_freeze': asset(args.candidate_freeze),
            'candidate_freeze_declaration': freeze, 'changed_archived_source_files': changed,
            'pq_artifact_intervention': clone,
            'baseline_source_archive': asset(args.baseline_source_archive),
            'candidate_source_archive': asset(args.candidate_source_archive),
            'verification_scope': 'Retained logs and source archives are independently hashed; remote immutable index/binary declarations are checked consistently before and after serial timings.'}


def summarize(args):
    require(sha(args.plan) == FROZEN_SHA, 'Original per-label L policy changed')
    plan = load_plan(args.plan)
    validation, calibration = prepared(args.prepared), prepared(args.calibration_prepared)
    proof = private_source_proof(validation, calibration, plan, args)
    workload = calibration if args.quality_only else validation
    if args.supplemental:
        workload = supplemental_prepared(args.supplemental_prepared)
        proof['supplemental'] = supplemental_source_proof(workload, args)
    postings = {}
    for label in sorted(plan):
        values, _ = matrix(args.prepared / f'label_{label}.u32bin', '<u4')
        require(values.shape[1] == 1 and np.all(np.diff(values[:, 0].astype(np.int64)) > 0), 'Invalid label posting')
        postings[label] = values[:, 0]
    valid_result_ids(workload['truth'], workload, postings)
    manifest = json.loads(args.jobs_manifest.read_text())
    root = args.artifact_root or args.jobs_manifest.parent
    remote = Path(manifest['output_root'])
    logs = {'cpp': args.cpp, 'baseline': args.baseline, 'candidate': args.candidate}
    observations, inputs, experiments = {}, {}, {}
    if args.quality_only:
        for name in ENGINES:
            observations[name], inputs[name] = quality_observations(logs[name], workload, plan, postings,
                                                                  root, remote, native=name != 'cpp')
    else:
        observations['cpp'], inputs['cpp'] = cpp_observations(args.cpp, None, workload, plan, postings, root, remote)
        for name in ('baseline', 'candidate'):
            experiments[name], observations[name], inputs[name] = native_observations(logs[name], workload, plan,
                                                                                   postings, root, remote)
    jobs = triple_jobs_proof(args.jobs_manifest, logs, root, quality_only=args.quality_only)
    identity = identity_proof(args, logs, manifest)
    result = compare_engines(observations, workload, plan, postings, args.bootstrap, args.seed, quality_only=args.quality_only)
    cutoff = cutoff_diagnostic(args.base, workload, {name: rows[0] for name, rows in observations.items()})
    if cutoff:
        for row in result['per_label']:
            mask = workload['labels'] == row['label']
            for name in ENGINES:
                row[f'{name}_cutoff_recall_percent'] = float(cutoff['engines'][name]['hits'][mask].mean()) / 10 * 100
            for positive, reference in CONTRASTS:
                row[f'{positive}_minus_{reference}_cutoff_pp'] = (
                    row[f'{positive}_cutoff_recall_percent'] - row[f'{reference}_cutoff_recall_percent'])
        cutoff['engines'] = {name: {key: value for key, value in row.items() if key != 'hits'}
                             for name, row in cutoff['engines'].items()}
    result.update({'schema_version': 1, 'scope': ('Supplemental shifted atomic-label validation; separate from natural one-tag workload' if args.supplemental else 'Calibration quality-only diagnosis; validation quality remains sealed' if args.quality_only else 'Fresh sealed natural query validation; original L policy unchanged'),
                   'timing_excluded': args.quality_only,
                   'prepared_identity': workload['identity'], 'fresh_query_proof': proof, 'jobs_proof': jobs,
                   'identity_proof': identity, 'inputs': inputs, 'cutoff_diagnostic': cutoff,
                   'targets': list(TARGETS), 'bootstrap': {'replications': args.bootstrap, 'seed': args.seed,
                       'unit': 'paired query', 'per_label_simultaneous_family': 3 * len(result['per_label']),
                       'aggregate_simultaneous_family': 3},
                   'acceptance_threshold': None,
                   'limitations': [
                       'No production traffic, arrival-rate saturation, SSD cold-start or advertising embedding/SLO claim follows from this dataset.',
                       'Six timing repetitions reuse identical queries; they are performance observations, not independent recall samples.',
                       'Zero observed recall difference or an interval containing zero does not establish non-inferiority.',
                       'Sparse label p99 and empirical bootstrap intervals are weak; a degenerate interval cannot guarantee population recall parity.',
                       'Per-label latencies are service times within the mixed workload; no isolated per-label QPS is inferred.',
                       'Concurrent independent Vamana builds are not bit-reproducible; the new graph differs beyond a deterministic final-prune replay.',
                       'Fresh results must not be used to retune this candidate and then described as untouched validation.'
                   ]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'calibration-prepared', 'commitment', 'exclusion-proof', 'private-queries',
                 'private-metadata', 'private-truth', 'public-queries', 'base', 'plan', 'cpp', 'baseline', 'candidate',
                 'jobs-manifest', 'identity', 'prior-identity', 'baseline-source-archive',
                 'candidate-source-archive', 'candidate-freeze', 'candidate-build-log', 'out', 'csv'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path)
    parser.add_argument('--quality-only', action='store_true', help='Audit a one-triple quality-only diagnostic without reporting performance')
    parser.add_argument('--supplemental', action='store_true', help='Explicitly separate shifted atomic-label validation from natural queries')
    parser.add_argument('--supplemental-prepared', type=Path)
    parser.add_argument('--primary-identity', type=Path)
    parser.add_argument('--clone-result', type=Path)
    parser.add_argument('--clone-provenance', type=Path)
    parser.add_argument('--original-graph-build-log', type=Path)
    parser.add_argument('--bootstrap', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=20261023)
    args = parser.parse_args()
    require(not args.supplemental or (args.quality_only and args.supplemental_prepared and args.primary_identity),
            'Supplemental mode requires --quality-only, --supplemental-prepared and --primary-identity')
    require(args.bootstrap >= 10000, 'At least10K paired-query bootstrap replications required')
    require(not args.out.exists() and not args.csv.exists(), 'Output already exists')
    result = summarize(args)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    with args.csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['per_label'][0]))
        writer.writeheader()
        writer.writerows(result['per_label'])
    print(json.dumps({'summary': str(args.out), 'per_label': str(args.csv),
                      'recall': {name: item['strict_id_recall'] for name, item in result['engines'].items()},
                      'label_comparison': result['label_comparison']}))


if __name__ == '__main__':
    main()
