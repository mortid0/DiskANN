#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Audit paired same-construction legacy-prune versus retained-union results.

Reports quality and measured work counters only. L64/L80 diagnostic sweeps cannot
reconstruct a frozen policy that also requires unmeasured L68/L72/L96 values.
"""
import argparse
from array import array
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import mmap
from pathlib import Path
import struct


FROZEN_PLAN_SHA256 = 'df42e65440af0a891185913e91f5f22cb6808c9251b2c1d8f6ca968d5bda2e59'
TARGETS = [9, 24, 89, 98, 2495]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def u32_matrix(path, truth=False):
    data = Path(path).read_bytes()
    require(len(data) >= 8, f'short matrix: {path}')
    rows, columns = struct.unpack_from('<II', data)
    size = rows * columns
    require(len(data) == 8 + size * (8 if truth else 4), f'matrix length: {path}')
    values = array('I', struct.unpack_from(f'<{size}I', data, 8))
    distances = struct.unpack_from(f'<{size}f', data, 8 + size * 4) if truth else None
    return rows, columns, values, distances


def validate_dump(path, labels, k, breadths, postings, exact):
    result = {}
    counts = Counter()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            key = (row['l'], row['query'])
            require(key not in result, 'duplicate query at the same L')
            require(row['l'] in breadths and 0 <= row['query'] < len(labels), 'unexpected L or query')
            label = labels[row['query']]
            require(row['label'] == label, 'query-label mismatch')
            require(row['repeat'] == 0 and row['mode'] == 'single' and row['threads'] == 8,
                    'unexpected repeat, mode or worker count')
            require(row['returned'] == k and len(row['ids']) == k and len(row['distances']) == k,
                    'short or padded result')
            require(len(set(row['ids'])) == k, 'duplicate result IDs')
            pairs = list(zip(row['distances'], row['ids']))
            require(pairs == sorted(pairs), 'results are not sorted by exact distance and ID')
            for distance, ident in pairs:
                posting = postings[label]
                where = bisect_left(posting, ident)
                require(where < len(posting) and posting[where] == ident, 'filter-invalid result')
                require(math.isfinite(distance) and distance == exact(row['query'], ident),
                        'result distance differs from independent integer squared L2')
            result[key] = row
            counts[row['l']] += 1
    require(counts == Counter({l: len(labels) for l in breadths}), 'incomplete query/L sweep')
    return result


def compare(old, new, query_ids, l_for_query, truth, k):
    hist = Counter()
    old_hits = new_hits = changed = 0
    exact_sum_change = 0.
    for query in query_ids:
        key = (l_for_query(query), query)
        before, after = old[key], new[key]
        gold = set(truth[query * k:(query + 1) * k])
        a = len(gold.intersection(before['ids']))
        b = len(gold.intersection(after['ids']))
        old_hits += a
        new_hits += b
        hist[b - a] += 1
        changed += before['ids'] != after['ids']
        exact_sum_change += sum(after['distances']) - sum(before['distances'])
    n = sum(hist.values())
    return {'queries': n, 'k': k, 'legacy_hits': old_hits, 'preserved_hits': new_hits,
            'hit_difference': new_hits - old_hits,
            'legacy_recall_percent': 100 * old_hits / (n * k),
            'preserved_recall_percent': 100 * new_hits / (n * k),
            'difference_pp': 100 * (new_hits - old_hits) / (n * k),
            'queries_improved': sum(n for delta, n in hist.items() if delta > 0),
            'queries_worsened': sum(n for delta, n in hist.items() if delta < 0),
            'queries_equal_hits': hist[0], 'queries_changed_ordered_ids': changed,
            'paired_query_hit_difference_histogram': {str(d): n for d, n in sorted(hist.items())},
            'mean_returned_distance_sum_change': exact_sum_change / n}


def flags(command):
    result = {}
    for i, item in enumerate(command[:-1]):
        if item.startswith('--'):
            require(item not in result, 'duplicate command flag')
            result[item] = command[i + 1]
    return result


def analyze(args):
    results = args.results_dir
    prepared = args.prepared
    identity = json.loads((args.primary_run / 'identity.json').read_text())
    require(sha(args.plan) == FROZEN_PLAN_SHA256, 'frozen plan bytes changed')
    plan = {int(label): value for label, value in json.loads(args.plan.read_text()).items()}
    require(identity['plan_before_sha256'] == identity['plan_after_sha256'] == sha(args.plan), 'plan identity')
    manifest = json.loads((prepared / 'manifest.json').read_text())
    require(identity['prepared_manifest_sha256'] == sha(prepared / 'manifest.json'), 'prepared manifest identity')
    for name, artifact in manifest['artifacts'].items():
        require(sha(prepared / name) == artifact['sha256'], f'prepared artifact SHA: {name}')
    n, label_columns, labels, _ = u32_matrix(prepared / 'query_labels.u32bin')
    gn, k, truth, truth_distances = u32_matrix(prepared / 'groundtruth.bin', truth=True)
    require(label_columns == 1 and gn == n and k == 10, 'prepared query/truth shape')
    groups = defaultdict(list)
    postings = {}
    for query, label in enumerate(labels):
        groups[label].append(query)
    require(set(groups) == set(plan), 'prepared label coverage differs from frozen 64-label plan')
    for label in groups:
        rows, columns, members, _ = u32_matrix(prepared / f'label_{label}.u32bin')
        require(columns == 1 and rows and all(a < b for a, b in zip(members, members[1:])), 'invalid posting')
        postings[label] = members
    jobs = json.loads((results / 'causal-and-centroid-jobs.json').read_text())
    jobs = {row['name']: row for row in jobs if row['name'] in ['causal-new', 'causal-old-prune']}
    require(set(jobs) == {'causal-new', 'causal-old-prune'}, 'missing completed causal jobs')
    calls = {key: flags(row['command']) for key, row in jobs.items()}
    for key, row in jobs.items():
        require(row['exit_code'] == 0 and calls[key]['--timing-authoritative'] == 'false', 'invalid causal job')
        require(row['log_sha256'] == sha(results / f'{key}.log'), 'job log hash')
    new_flags, old_flags = calls['causal-new'], calls['causal-old-prune']
    for key in ['--pq-prefix', '--label-metadata', '--prepared', '--limit', '--ls', '--modes', '--starts', '--threads', '--reps', '--header']:
        require(new_flags[key] == old_flags[key], f'causal command mismatch: {key}')
    require(new_flags['--ls'] == '64,80' and new_flags['--limit'] == str(n), 'unexpected causal sweep')
    breadths = [64, 80]
    legacy = json.loads((results / 'legacy-reprune.json').read_text())
    rewrite_command = json.loads((results / 'causal-reprune-command.json').read_text())
    rewrite_flags = flags(rewrite_command['command'])
    require(legacy['input_graph'] == new_flags['--prefix'] + '_disk.index', 'input graph path')
    require(legacy['output_graph'] == old_flags['--prefix'] + '_disk.index', 'counterfactual graph path')
    require(legacy['labels'] == new_flags['--label-metadata'] == rewrite_flags['--labels'], 'label metadata path')
    require(rewrite_flags['--input-graph'] == legacy['input_graph'], 'rewrite source')
    require(legacy['degree_bound'] == 128 and abs(legacy['alpha'] - 1.2) < 1e-6, 'legacy build settings')
    candidate = identity['engines']['candidate']
    require(candidate['index_before'] == candidate['index_after'], 'primary candidate metadata changed')
    frozen = candidate['index_before']['files_sha256']
    require(new_flags['--pq-prefix'] == new_flags['--prefix'], 'PQ pair is not the candidate committed pair')
    for name in ['index_pq_pivots.bin', 'index_pq_compressed.bin', 'vectors.bin', 'labels.bin']:
        require(frozen[name] == identity['engines']['baseline']['index_before']['files_sha256'][name],
                f'frozen baseline pair/binding changed: {name}')
    require(sha(prepared / 'base.u8bin') == frozen['vectors.bin'], 'complete base-vector SHA differs from frozen graph binding')

    aggregates = {}
    for name in ['causal-new', 'causal-old-prune']:
        rows = [json.loads(line) for line in (results / f'{name}.jsonl').read_text().splitlines()]
        require(len(rows) == len(breadths) and {row['l'] for row in rows} == set(breadths), 'aggregate sweep')
        aggregates[name] = {row['l']: row for row in rows}
        for row in rows:
            require(row['queries'] == n and row['k'] == k and row['mode'] == 'single' and not row['timing_authoritative'], 'aggregate query settings')
            require(row['filter_violations'] == row['duplicate_results'] == row['short_results'] == 0, 'invalid aggregate results')
            require(row['label_metadata_identity'] == rows[0]['label_metadata_identity'], 'metadata changed across L')
            require(row['pq_prefix'] == new_flags['--pq-prefix'] and row['index_prefix'] == calls[name]['--prefix'], 'aggregate artifact paths')
    require(aggregates['causal-new'][64]['label_metadata_identity'] == aggregates['causal-old-prune'][64]['label_metadata_identity'], 'metadata fingerprints differ')

    query_bytes = (prepared / 'query.u8bin').read_bytes()
    qn, dimensions = struct.unpack_from('<II', query_bytes)
    require(qn == n and len(query_bytes) == 8 + n * dimensions, 'query vector shape')
    cache = {}
    with (prepared / 'base.u8bin').open('rb') as base_file:
        with mmap.mmap(base_file.fileno(), 0, access=mmap.ACCESS_READ) as base:
            base_rows, base_dimensions = struct.unpack_from('<II', base)
            require(dimensions == base_dimensions and len(base) == 8 + base_rows * dimensions, 'base vector shape')
            def exact(query, ident):
                require(0 <= ident < base_rows, 'base ordinal out of bounds')
                key = (query, ident)
                if key not in cache:
                    a = query_bytes[8 + query * dimensions:8 + (query + 1) * dimensions]
                    b = base[8 + ident * dimensions:8 + (ident + 1) * dimensions]
                    cache[key] = sum((x - y) ** 2 for x, y in zip(a, b))
                return cache[key]
            for query, label in enumerate(labels):
                for position in range(k):
                    slot = query * k + position
                    ident = truth[slot]
                    posting = postings[label]
                    where = bisect_left(posting, ident)
                    require(where < len(posting) and posting[where] == ident, 'truth filter violation')
                    require(truth_distances[slot] == exact(query, ident), 'truth distance mismatch')
            dumps = {name: validate_dump(results / f'{name}.queries.jsonl', labels, k, breadths, postings, exact)
                     for name in ['causal-new', 'causal-old-prune']}
    old, new = dumps['causal-old-prune'], dumps['causal-new']
    label_rows = []
    fixed_views = []
    for breadth in breadths:
        combined = compare(old, new, range(n), lambda query: breadth, truth, k)
        combined['l'] = breadth
        fixed_views.append(combined)
        for label, queries in sorted(groups.items()):
            row = {'view': f'all_labels_fixed_L{breadth}', 'label': label, 'l': breadth,
                   **compare(old, new, queries, lambda query: breadth, truth, k)}
            label_rows.append(row)
        for name, dump in dumps.items():
            hits = sum(len(set(truth[q*k:(q+1)*k]).intersection(dump[(breadth, q)]['ids'])) for q in range(n))
            require(abs(hits / (n*k) - aggregates[name][breadth]['recall']) < 1e-10, 'aggregate recall mismatch')
            recorded = {row['label']: row for row in aggregates[name][breadth]['labels']}
            for label, queries in groups.items():
                hits = sum(len(set(truth[q*k:(q+1)*k]).intersection(dump[(breadth, q)]['ids'])) for q in queries)
                require(recorded[label]['n'] == len(queries) and abs(recorded[label]['recall'] - hits/(len(queries)*k)) < 1e-10,
                        'per-label aggregate mismatch')
    target_queries = [q for q, label in enumerate(labels) if label in TARGETS]
    target_view = compare(old, new, target_queries, lambda q: plan[labels[q]], truth, k)
    for label in TARGETS:
        require(plan[label] in breadths, 'missing target policy breadth')
        label_rows.append({'view': 'five_target_frozen_policy', 'label': label, 'l': plan[label],
                           **compare(old, new, groups[label], lambda q: plan[label], truth, k)})
    native_rows = [json.loads(line) for line in (args.primary_run / 'triple-00/candidate-details/query.t8.L64.r0.jsonl').read_text().splitlines()]
    require(len(native_rows) == n and {row['query_index'] for row in native_rows} == set(range(n)),
            'native parity reference must contain each query exactly once')
    parity = Counter()
    for row in native_rows:
        query, label = row['query_index'], row['label']
        require(row['actual_l'] == plan[label] and labels[query] == label, 'native policy identity')
        require(row['sequence'] == 0 and row['base_searches'] == 1 and not row['exact_fallback']
                and row['delta_candidates'] == 0, 'native parity reference contains updates or fallback')
        if row['actual_l'] not in breadths:
            continue
        diagnostic = new[(row['actual_l'], query)]
        require(row['ids'] == diagnostic['ids'] and row['distances'] == diagnostic['distances'], 'same-graph native/diagnostic ordered output differs')
        parity['all_matching_breadths'] += 1
        if label in TARGETS:
            parity['target_queries'] += 1
    require(parity['target_queries'] == len(target_queries), 'native target parity coverage')
    metrics = {}
    for breadth in breadths:
        metrics[str(breadth)] = {name: {key: rows[breadth][key] for key in
            ['mean_cmps', 'mean_hops', 'reader_requests', 'reader_bytes']} for name, rows in aggregates.items()}
    inputs = [args.plan, prepared/'manifest.json', args.primary_run/'identity.json',
              args.primary_run/'triple-00/candidate-details/query.t8.L64.r0.jsonl']
    inputs.extend(results/name for name in ['causal-new.jsonl', 'causal-new.queries.jsonl', 'causal-old-prune.jsonl',
        'causal-old-prune.queries.jsonl', 'legacy-reprune.json', 'causal-reprune-command.json', 'causal-and-centroid-jobs.json'])
    return {'schema_version': 1, 'scope': 'quality-only causal ablation; no timing or throughput claims',
            'query_count': n, 'covered_labels': len(groups), 'k': k,
            'input_sha256': {str(path): sha(path) for path in inputs},
            'graph_intervention': legacy, 'frozen_native_candidate_files_sha256': frozen,
            'source_control': {'same_per_label_construction': True, 'same_PQ_pair': True,
                'same_entries_and_memberships': True, 'primary_candidate_identity_before_after_equal': True,
                'legacy_graph_content_sha256': None,
                'audit_limit': 'Counterfactual graph file is remote and was not independently rehashed here; the recorded exporter command/source and unchanged input identity bind its construction.'},
            'native_ordered_output_parity': dict(parity),
            'five_target_frozen_policy': target_view,
            'all_labels_fixed_L_exploratory': fixed_views, 'per_label': label_rows,
            'full_64_label_frozen_policy': {'status': 'not reconstructible from measured breadths',
                'missing_label_L': {str(label): l for label, l in plan.items() if l not in breadths}},
            'aggregate_work_counters_fixed_L_only': metrics,
            'per_query_work_counters': 'Unavailable: result dumps contain IDs/distances, not PQ comparisons, I/O or hops.',
            'interpretation': [
                'The legacy counterfactual reprunes the same constructed graph, so differences isolate retained stitching edges rather than independent concurrent Vamana builds.',
                'More retained edges can change fixed-L traversal non-monotonically; per-label and paired-query hit deltas may have either sign.',
                'Primary queries already informed this investigation; this mechanistic experiment supplies causal diagnosis, not a fresh generalization test.',
                'All timing fields were deliberately excluded. Work counters describe complete fixed-L sweeps, not a mixed policy or isolated target labels.',
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--primary-run', type=Path, required=True)
    parser.add_argument('--out-prefix', type=Path, required=True)
    args = parser.parse_args()
    outputs = [Path(str(args.out_prefix) + suffix) for suffix in ['.json', '.csv']]
    require(not any(path.exists() for path in outputs), 'outputs must be fresh')
    value = analyze(args)
    outputs[0].write_text(json.dumps(value, indent=2) + '\n')
    fields = ['view', 'label', 'l', 'queries', 'legacy_hits', 'preserved_hits', 'hit_difference',
              'legacy_recall_percent', 'preserved_recall_percent', 'difference_pp', 'queries_improved',
              'queries_worsened', 'queries_equal_hits', 'queries_changed_ordered_ids', 'mean_returned_distance_sum_change']
    with outputs[1].open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(value['per_label'])
    print(json.dumps({key: value[key] for key in ['scope', 'five_target_frozen_policy', 'all_labels_fixed_L_exploratory', 'native_ordered_output_parity']}))


if __name__ == '__main__':
    main()
