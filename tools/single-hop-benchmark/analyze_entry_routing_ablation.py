#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Quality-only entry-routing ablation on exposed supplemental calibration.

Require V8 with the original one-entry metadata to reproduce V7 ordered results,
then compare V8's spatial entry pools while graph/PQ/vector bytes remain fixed.
Never consumes the reserved secondary validation partition or reports throughput.
"""
import argparse
from bisect import bisect_left
from collections import Counter
import csv
import hashlib
import json
import math
import mmap
from pathlib import Path
import struct

from analyze_causal_prune import (FROZEN_PLAN_SHA256, TARGETS, compare, flags,
                                  require, sha, u32_matrix)


def require_calibration_manifest(manifest):
    # Check this before opening query assets, so this diagnostic cannot unseal validation.
    require(manifest.get('split') == 'secondary_calibration'
            and manifest.get('query_count') == 1436,
            'only the exposed 1436-query calibration partition is allowed')


def verify_query_provenance(prepared, manifest, labels, query_bytes):
    n, dimensions = struct.unpack_from('<II', query_bytes)
    require(n == len(labels) and len(query_bytes) == 8+n*dimensions, 'query vector shape')
    namespaces = (prepared/'source_namespace.u8bin').read_bytes()
    require(struct.unpack_from('<II', namespaces) == (n, 1) and len(namespaces) == n+8,
            'namespace matrix shape')
    id_rows, id_columns, source_ids, _ = u32_matrix(prepared/'query_ids.u32bin')
    pool_rows, pool_columns, pool_ids, _ = u32_matrix(prepared/'source_pool_indices.u32bin')
    require(id_rows == pool_rows == n and id_columns == pool_columns == 1, 'source ID matrix shape')
    namespace_files = {0: ('public_100K', 'query.public.100K.u8bin'),
                       1: ('official_private_2727415019_100K', 'query.private.2727415019.100K.u8bin')}
    source_records = {record['file']: record for record in manifest['sources']}
    original_vectors = {}
    for code, (_, name) in namespace_files.items():
        record = source_records[name]
        path = Path(record['path'])
        require(sha(path) == record['sha256'], 'original query corpus hash differs')
        data = path.read_bytes()
        rows, columns = struct.unpack_from('<II', data)
        require(columns == dimensions and len(data) == 8+rows*dimensions, 'original query corpus shape')
        original_vectors[code] = (rows, data)
    records = [json.loads(line) for line in (prepared/'source-query-rows.jsonl').read_text().splitlines()]
    require(len(records) == n, 'query provenance row count differs')
    seen_sources, seen_pool = set(), set()
    for query, row in enumerate(records):
        code = namespaces[8+query]
        require(code in namespace_files, 'unknown query namespace')
        source = (namespace_files[code][0], source_ids[query])
        require(source not in seen_sources and pool_ids[query] not in seen_pool, 'duplicate source query')
        seen_sources.add(source)
        seen_pool.add(pool_ids[query])
        require(row['query_index'] == query and row['atomic_label'] == labels[query]
                and row['source_query_id'] == source[1] and row['source_query_namespace'] == source[0]
                and row['source_pool_query_index'] == pool_ids[query]
                and labels[query] in row['original_and_tags'], 'source query ID/label/namespace differs')
        vector = query_bytes[8+query*dimensions:8+(query+1)*dimensions]
        source_rows, source_data = original_vectors[code]
        ident = source[1]
        require(0 <= ident < source_rows
                and vector == source_data[8+ident*dimensions:8+(ident+1)*dimensions]
                and hashlib.sha256(vector).hexdigest() == row['embedding_sha256'],
                'source query embedding differs')
    return {'verified_query_ids': n, 'unique_source_queries': len(seen_sources),
            'verified_source_vectors_sha256': {record['path']: record['sha256']
                for name, record in source_records.items() if name in {x[1] for x in namespace_files.values()}}}


def compare_entries(one, pool, indexes, breadth, truth, k):
    value = compare(one, pool, indexes, breadth, truth, k)
    for old, new in [('legacy_hits', 'one_entry_hits'), ('preserved_hits', 'pool_hits'),
                     ('legacy_recall_percent', 'one_entry_recall_percent'),
                     ('preserved_recall_percent', 'pool_recall_percent')]:
        value[new] = value.pop(old)
    return value


def verify_entry_metadata(publication, provenance, postings):
    require(publication['manifest_payload_preserved_verbatim'] is True, 'manifest payload changed')
    require(publication['source_manifest_before'] == publication['source_manifest_after'], 'manifest history changed')
    crcs = publication['manifest_payload_crc64_ecma182']
    require(crcs['source'] == crcs['installed'], 'manifest bytes changed')
    require(all(publication['verified_bindings'].get(key) is True for key in
                ['point_count_equal', 'point_labels_equal', 'postings_equal', 'stable_ids_equal', 'global_centroid_first']),
            'metadata binding checks missing')
    policy = publication['entry_policy']
    require(policy['spatial_clusters'] == 4 and policy['maximum_entries'] == 5
            and policy['maximum_fit_members'] == 4096 and policy['lloyd_iterations'] == 8
            and policy['sample_seed_base'] == 0x6c6162656c5f7633
            and policy['global_centroid_first'] is True, 'unexpected corpus-only pool policy')
    before = provenance['entry_selection']['old_entry_points']
    after = publication['entry_points']
    require(after == provenance['entry_selection']['new_entry_points'], 'entry provenance differs')
    require(set(before) == set(after) == set(map(str, postings)), 'entry-pool label coverage differs')
    for label, entries in after.items():
        require(1 <= len(entries) <= 5 and len(entries) == len(set(entries)), 'invalid entry pool')
        require(len(before[label]) == 1 and entries[0] == before[label][0], 'old centroid not preserved first')
        members = postings[int(label)]
        for ident in entries:
            where = bisect_left(members, ident)
            require(where < len(members) and members[where] == ident, 'entry is not an eligible actual member')


def validate_details(path, labels, truth, k, plan, postings, exact):
    result = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            require(row.get('kind') == 'query_detail', 'unexpected raw detail kind')
            query = row['query_index']
            require(0 <= query < len(labels) and query not in result, 'duplicate or invalid query index')
            label = labels[query]
            require(row['label'] == label and row['actual_l'] == plan[label] and row['l'] == 64,
                    'label or frozen query breadth differs')
            require(row['repeat'] == 0 and row['phase'] == 'query' and row['sequence'] == 0
                    and row['base_searches'] == 1 and row['delta_candidates'] == 0
                    and not row['exact_fallback'], 'unexpected repeats, updates, retries or fallback')
            require(row['k'] == k and len(row['ids']) == len(row['distances']) == k
                    and len(set(row['ids'])) == k, 'short, padded or duplicated result')
            pairs = list(zip(row['distances'], row['ids']))
            require(pairs == sorted(pairs), 'exact result ordering differs')
            for distance, ident in pairs:
                members = postings[label]
                where = bisect_left(members, ident)
                require(where < len(members) and members[where] == ident, 'filter-invalid result')
                require(math.isfinite(distance) and distance == exact(query, ident), 'result exact distance mismatch')
            hits = len(set(truth[query*k:(query+1)*k]).intersection(row['ids']))
            require(row['strict_id_hits'] == hits, 'raw strict ID hit counter mismatch')
            result[query] = row
    require(set(result) == set(range(len(labels))), 'incomplete raw query coverage')
    return result


def exact_parity(left, right):
    require(set(left) == set(right), 'ordered parity query coverage differs')
    for query in left:
        require(left[query]['ids'] == right[query]['ids']
                and left[query]['distances'] == right[query]['distances'],
                f'one-entry V8 differs from V7 at query {query}')
    return len(left)


def load_observation(aggregate, details, labels, truth, k, plan, postings, exact):
    rows = [json.loads(line) for line in Path(aggregate).read_text().splitlines()]
    experiments = [row for row in rows if row.get('kind') == 'experiment']
    queries = [row for row in rows if row.get('kind') == 'query']
    require(len(experiments) == len(queries) == 1, 'expected one experiment and one measured query repeat')
    experiment, row = experiments[0], queries[0]
    require(row['queries'] == len(labels) and row['k'] == k and row['repeat'] == 0
            and not row['timing_authoritative'], 'observation is not the quality-only calibration repeat')
    require(row['filter_violations'] == row['duplicate_results'] == row['short_results'] == 0,
            'invalid aggregate results')
    require(row['raw_queries']['sha256'] == sha(details) and row['raw_queries']['queries'] == len(labels),
            'raw detail hash/count differs')
    require(row['label_l_plan']['content_sha256'] == FROZEN_PLAN_SHA256, 'aggregate frozen plan differs')
    require({item['label']: item['l'] for item in row['label_l_plan']['entries']} == plan, 'aggregate plan contents differ')
    observations = validate_details(details, labels, truth, k, plan, postings, exact)
    grouped = {label: [q for q, value in enumerate(labels) if value == label] for label in set(labels)}
    recorded = {item['label']: item for item in row['labels']}
    require(set(recorded) == set(grouped), 'aggregate label coverage differs')
    total_hits = 0
    for label, indexes in grouped.items():
        hits = sum(observations[q]['strict_id_hits'] for q in indexes)
        require(recorded[label]['queries'] == len(indexes) and recorded[label]['actual_l'] == plan[label]
                and abs(recorded[label]['recall'] - hits/(len(indexes)*k)) < 1e-10,
                'aggregate per-label recall differs')
        total_hits += hits
    require(abs(row['recall'] - total_hits/(len(labels)*k)) < 1e-10, 'aggregate recall differs')
    return observations, experiment


def generation_files(index_identity, field):
    require(index_identity['index_before'] == index_identity['index_after'], f'{field} assets changed during calibration')
    return index_identity['index_before']['files_sha256']


def verify_candidate_job(directory, phase, manifest_sha256, aggregate):
    value = json.loads((directory/'triple-run-manifest.json').read_text())
    require(value['status'] == 'complete' and value['phase'] == phase
            and value['prepared_manifest_sha256'] == manifest_sha256
            and value['queries'] == 1436 and value['k'] == 10
            and value['timing_authoritative'] is False, 'candidate job scope differs')
    jobs = [row for row in value['runs'] if row['implementation'] == 'candidate']
    require(len(jobs) == 1 and jobs[0]['exit_code'] == 0
            and jobs[0]['aggregate_sha256'] == sha(aggregate), 'candidate aggregate job binding differs')
    call = flags(jobs[0]['command'])
    require(call['--timing-authoritative'] == 'false' and call['--limit'] == '1436'
            and call['--l'] == '64' and call['--reps'] == '1'
            and call['--prepared'] == value['prepared'], 'candidate quality command differs')
    return call


def analyze(args):
    manifest_path = args.prepared / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    require_calibration_manifest(manifest)
    for name, expected in manifest['artifacts'].items():
        require(sha(args.prepared/name) == expected['sha256'], f'prepared artifact differs: {name}')
    require(sha(args.plan) == FROZEN_PLAN_SHA256, 'plan bytes changed')
    plan = {int(label): value for label, value in json.loads(args.plan.read_text()).items()}
    n, columns, labels, _ = u32_matrix(args.prepared/'query_labels.u32bin')
    tn, k, truth, truth_distances = u32_matrix(args.prepared/'groundtruth.bin', truth=True)
    require(n == tn == 1436 and columns == 1 and k == 10 and set(labels) == set(TARGETS), 'calibration shapes or labels')
    postings = {label: u32_matrix(args.prepared/f'label_{label}.u32bin')[2] for label in plan}
    publication = json.loads(args.publication.read_text())
    provenance = json.loads(args.provenance.read_text())
    verify_entry_metadata(publication, provenance, postings)
    old_identity = json.loads((args.v7_calibration/'identity.json').read_text())
    new_identity = json.loads((args.v8_calibration/'identity.json').read_text())
    for identity, phase in [(old_identity, 'secondary-calibration'), (new_identity, 'secondary-development')]:
        require(identity['phase'] == phase and identity['prepared_manifest_sha256'] == sha(manifest_path), 'calibration identity scope')
        require(identity['plan_before_sha256'] == identity['plan_after_sha256'] == FROZEN_PLAN_SHA256, 'identity plan')
    old_engine, new_engine = old_identity['engines']['candidate'], new_identity['engines']['candidate']
    old_files = generation_files(old_engine, 'V7')
    new_files = generation_files(new_engine, 'V8')
    for name in ['index_disk.index', 'index_pq_pivots.bin', 'index_pq_compressed.bin', 'vectors.bin']:
        require(old_files[name] == new_files[name], f'graph/PQ/vector intervention is confounded: {name}')
    require(old_files['labels.bin'] != new_files['labels.bin'], 'entry metadata did not change')
    old_root, new_root = publication['input_root'], publication['output_root']
    source_directory = old_root + '/generation-00000000000000000001/'
    output_directory = publication['generation_directory'] + '/'
    for name, digest in old_files.items():
        require(publication['source_files_sha256'][source_directory+name] == digest, 'source publication binding differs')
    for name, digest in new_files.items():
        require(publication['output_files_sha256'][output_directory+name] == digest, 'installed publication binding differs')
    require(sha(args.provenance) == new_files['entry-pool-provenance.json'], 'checksummed entry provenance differs')
    require(provenance['input_root'] == old_root and provenance['output_root'] == new_root
            and provenance['source_files_sha256'] == publication['source_files_sha256'], 'provenance root/binding differs')
    control = json.loads(args.single_command.read_text())
    require(control['binary_sha256'] == new_engine['binary_sha256'], 'control is not the V8 binary')
    require(control['prepared_manifest_sha256'] == sha(manifest_path)
            and control['plan_sha256'] == FROZEN_PLAN_SHA256
            and control['parent_identity_sha256'] == sha(args.v7_calibration/'identity.json')
            and control['source_archive_sha256'] == new_engine['source_archive']['sha256'], 'single-entry control provenance differs')
    require(control['index_before'] == control['index_after'] == old_engine['index_after'], 'one-entry control graph/metadata differs')
    call = flags(control['command'])
    require(call['--root'] == old_root and call['--timing-authoritative'] == 'false'
            and call['--limit'] == str(n), 'single-entry control arguments differ')
    require(sha(args.prepared/'base.u8bin') == old_files['vectors.bin'], 'base vector binding differs')
    query_bytes = (args.prepared/'query.u8bin').read_bytes()
    qn, dimensions = struct.unpack_from('<II', query_bytes)
    require(qn == n and len(query_bytes) == 8+n*dimensions, 'query vector shape')
    source_query_verification = verify_query_provenance(args.prepared, manifest, labels, query_bytes)
    cache = {}
    paths = {'v7_one': (args.v7_calibration/'triple-00/candidate.jsonl', args.v7_calibration/'triple-00/candidate-details/query.t8.L64.r0.jsonl'),
             'v8_one': (args.v8_single_aggregate, args.v8_single_details),
             'v8_pool': (args.v8_calibration/'triple-00/candidate.jsonl', args.v8_calibration/'triple-00/candidate-details/query.t8.L64.r0.jsonl')}
    for directory, phase, name in [(args.v7_calibration, 'secondary-calibration', 'v7_one'),
                                   (args.v8_calibration, 'secondary-development', 'v8_pool')]:
        job_call = verify_candidate_job(directory, phase, sha(manifest_path), paths[name][0])
        require(job_call['--root'] == (new_root if name == 'v8_pool' else old_root)
                and job_call['--prepared'] == call['--prepared'], 'candidate root/prepared differs from control')
    observations = {}
    with (args.prepared/'base.u8bin').open('rb') as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as base:
            base_rows, base_dimensions = struct.unpack_from('<II', base)
            require(base_dimensions == dimensions and len(base) == 8+base_rows*dimensions, 'base matrix shape')
            def exact(query, ident):
                require(0 <= ident < base_rows, 'result ordinal exceeds base')
                if (query, ident) not in cache:
                    a = query_bytes[8+query*dimensions:8+(query+1)*dimensions]
                    b = base[8+ident*dimensions:8+(ident+1)*dimensions]
                    cache[(query, ident)] = sum((x-y)**2 for x,y in zip(a,b))
                return cache[(query, ident)]
            for query, label in enumerate(labels):
                for position in range(k):
                    slot = query*k+position
                    ident = truth[slot]
                    members = postings[label]
                    where = bisect_left(members, ident)
                    require(where < len(members) and members[where] == ident
                            and truth_distances[slot] == exact(query, ident), 'groundtruth membership/distance differs')
            for name, (aggregate, details) in paths.items():
                rows, experiment = load_observation(aggregate, details, labels, truth, k, plan, postings, exact)
                require(experiment['root'] == (new_root if name == 'v8_pool' else old_root), 'experiment root differs')
                observations[name] = rows
    parity = exact_parity(observations['v7_one'], observations['v8_one'])
    keyed = {name: {(row['actual_l'], query): row for query, row in rows.items()} for name, rows in observations.items()}
    per_label = []
    for label in sorted(set(labels)):
        indexes = [q for q, value in enumerate(labels) if value == label]
        per_label.append({'label': label, 'l': plan[label],
            **compare_entries(keyed['v8_one'], keyed['v8_pool'], indexes, lambda q: plan[labels[q]], truth, k)})
    combined = compare_entries(keyed['v8_one'], keyed['v8_pool'], range(n), lambda q: plan[labels[q]], truth, k)
    inputs = [args.publication, args.provenance, args.single_command, args.plan, manifest_path,
              args.v7_calibration/'identity.json', args.v8_calibration/'identity.json']
    inputs.extend(directory/'triple-run-manifest.json' for directory in [args.v7_calibration, args.v8_calibration])
    inputs.extend(path for pair in paths.values() for path in pair)
    return {'schema_version': 1, 'scope': 'quality-only causal entry-routing ablation on already-exposed supplemental calibration',
            'query_count': n, 'k': k, 'query_label_counts': dict(sorted(Counter(labels).items())),
            'source_query_verification': source_query_verification,
            'input_sha256': {str(path): sha(path) for path in inputs},
            'same_graph_PQ_vectors_sha256': {name: old_files[name] for name in
                ['index_disk.index', 'index_pq_pivots.bin', 'index_pq_compressed.bin', 'vectors.bin']},
            'unchanged_source_manifest_verified': True, 'all64_entry_pools_eligible_and_centroid_first': True,
            'old_V7_vs_V8_single_entry_exact_ordered_parity_queries': parity,
            'aggregate_V8_pool_minus_one_entry': combined, 'per_label_V8_pool_minus_one_entry': per_label,
            'interpretation': [
                'V8 with one-entry metadata exactly reproduces V7 ordered IDs and distances; the runtime change preserves this control.',
                'The spatial-pool comparison holds the actual graph, vectors and PQ pair fixed. It isolates corpus-derived entry metadata plus nearest-PQ-entry selection.',
                'The 1436-query partition was already exposed during development. These observations are mechanistic calibration evidence, not fresh generalization validation.',
                'No QPS, service-time, p99 or throughput claims are made. No per-query PQ/I/O/hop counters were recorded in these details.',
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['v7-calibration', 'v8-calibration', 'v8-single-aggregate', 'v8-single-details',
                'single-command', 'publication', 'provenance', 'prepared', 'plan', 'out-prefix']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    outputs = [Path(str(args.out_prefix)+suffix) for suffix in ['.json','.csv']]
    require(not any(path.exists() for path in outputs), 'outputs must be fresh')
    value = analyze(args)
    outputs[0].write_text(json.dumps(value, indent=2)+'\n')
    fields = ['label','l','queries','one_entry_hits','pool_hits','hit_difference','one_entry_recall_percent',
              'pool_recall_percent','difference_pp','queries_improved','queries_worsened',
              'queries_equal_hits','queries_changed_ordered_ids','mean_returned_distance_sum_change']
    with outputs[1].open('w',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(value['per_label_V8_pool_minus_one_entry'])
    print(json.dumps({key:value[key] for key in ['scope','old_V7_vs_V8_single_entry_exact_ordered_parity_queries',
                                               'aggregate_V8_pool_minus_one_entry','per_label_V8_pool_minus_one_entry']}))


if __name__ == '__main__':
    main()
