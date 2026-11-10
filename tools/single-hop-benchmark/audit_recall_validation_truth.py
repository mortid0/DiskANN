#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Exhaustively audit fresh natural target-label truth without any search results."""

import argparse
from collections import Counter
import json
from pathlib import Path
import struct
import time

import numpy as np

from prepare_recall_secondary import exact_truth, PRIMARY_COMMITMENT
from prepare_recall_validation import TARGETS, matrix, provenance, require, sha256


def audit(data, primary, raw, output):
    require(not output.exists(), f"Refusing to overwrite truth audit:{output}")
    proof_path = primary / 'split-commitment.json'
    require(sha256(proof_path) == PRIMARY_COMMITMENT, 'Primary fresh commitment changed')
    proof = json.loads(proof_path.read_text())
    prepared = primary / 'prepared-private-natural-validation'
    partition = proof['partitions']['validation']
    require(sha256(prepared / 'manifest.json') == partition['manifest_sha256'], 'Validation manifest changed')
    for name, expected in partition['artifacts'].items():
        path = prepared / name
        require(path.stat().st_size == expected['bytes'] and sha256(path) == expected['sha256'], f'Sealed validation artifact changed:{name}')
    sources = {record['file']: record for record in proof['sources']}
    base_path = data / 'base.10M.u8bin'
    official_truth_path = raw / 'GT.private.2727415019.ibin'
    for path in (base_path, official_truth_path):
        require(path.stat().st_size == sources[path.name]['bytes'] and sha256(path) == sources[path.name]['sha256'], f'Original source changed:{path}')
    all_queries = matrix(prepared / 'query.u8bin', 'u1')
    all_labels = matrix(prepared / 'query_labels.u32bin', '<u4')[:, 0]
    all_source_ids = matrix(prepared / 'query_ids.u32bin', '<u4')[:, 0]
    selected = np.flatnonzero(np.isin(all_labels, sorted(TARGETS)))
    queries = np.asarray(all_queries[selected])
    labels = np.asarray(all_labels[selected])
    source_ids = np.asarray(all_source_ids[selected])
    require(len(selected) == 1278, 'Unexpected target validation size')
    base = matrix(base_path, 'u1')
    postings = {label: matrix(data / 'prepared-10m' / f'label_{label}.u32bin', '<u4')[:, 0] for label in TARGETS}
    original_postings = json.loads((primary / 'exclusion-evidence.json').read_text())['posting_artifacts']
    for label in TARGETS:
        require(sha256(data / 'prepared-10m' / f'label_{label}.u32bin') == original_postings[str(label)]['sha256'], f'Posting source changed:{label}')
    n, k = struct.unpack('<II', (prepared / 'groundtruth.bin').read_bytes()[:8])
    require((n, k) == (16600, 10), 'Unexpected sealed ground truth shape')
    truth_ids = np.memmap(prepared / 'groundtruth.bin', mode='r', dtype='<i4', offset=8, shape=(n, k))
    truth_distances = np.memmap(prepared / 'groundtruth.bin', mode='r', dtype='<f4', offset=8 + 4*n*k, shape=(n, k))
    published_n, published_k = struct.unpack('<II', official_truth_path.read_bytes()[:8])
    require((published_n, published_k) == (100000, 10), 'Unexpected official truth shape')
    published_ids = np.memmap(official_truth_path, mode='r', dtype='<i4', offset=8, shape=(published_n, k))
    published_distances = np.memmap(official_truth_path, mode='r', dtype='<f4', offset=8 + 4*published_n*k, shape=(published_n, k))
    expected_ids = np.asarray(truth_ids[selected])
    expected_distances = np.asarray(truth_distances[selected])
    require(np.array_equal(expected_ids, published_ids[source_ids]) and np.array_equal(expected_distances, published_distances[source_ids]), 'Sealed target truth differs from original official private rows')
    started = time.monotonic()
    exact_ids, exact_distances, evaluations = exact_truth(base, queries, labels, postings, query_block=256, point_block=8192,
                                                       progress=lambda row: print(json.dumps(row), flush=True))
    require(np.all(exact_distances == np.floor(exact_distances)), 'Exact scan produced fractional distances')
    distance_matches = np.all(exact_distances == expected_distances, axis=1)
    # Independent scalar arithmetic checks every official and newly computed pair.
    scalar_checks = 0
    for row in range(len(queries)):
        members = postings[int(labels[row])]
        for ids, distances in ((expected_ids, expected_distances), (exact_ids, exact_distances)):
            require(len(set(map(int, ids[row]))) == k, 'Repeated neighbor ID')
            for point_id, expected_distance in zip(ids[row], distances[row]):
                position = int(np.searchsorted(members, point_id))
                require(position < len(members) and members[position] == point_id, 'Truth ID violates label predicate')
                actual = sum((int(a) - int(b)) ** 2 for a, b in zip(queries[row], base[point_id]))
                require(actual == expected_distance, 'Scalar integer squared-L2 check failed')
                scalar_checks += 1
    counts = Counter(map(int, labels))
    records = []
    for label in sorted(TARGETS):
        rows = np.flatnonzero(labels == label)
        records.append({'label': label, 'queries': len(rows), 'matching_base_vectors': len(postings[label]),
                        'exact_point_query_distance_evaluations': evaluations[str(label)],
                        'all_sorted_top10_distances_match_official': bool(np.all(distance_matches[rows])),
                        'queries_with_distance_vector_match': int(distance_matches[rows].sum()),
                        'queries_with_identical_ordered_ids': int(np.all(exact_ids[rows] == expected_ids[rows], axis=1).sum()),
                        'queries_with_different_ids_but_equal_distances': int((np.any(exact_ids[rows] != expected_ids[rows], axis=1) & distance_matches[rows]).sum())})
    result = {
        'schema_version': 1, 'scope': 'exhaustive_exact_truth_audit_fresh_natural_validation_five_target_labels',
        'queries': len(selected), 'k': k, 'base_vectors': len(base), 'dimension': base.shape[1],
        'primary_commitment_sha256': PRIMARY_COMMITMENT,
        'prepared_manifest': provenance(prepared / 'manifest.json'),
        'prepared_query_vectors': provenance(prepared / 'query.u8bin'),
        'prepared_query_ids': provenance(prepared / 'query_ids.u32bin'),
        'prepared_query_labels': provenance(prepared / 'query_labels.u32bin'),
        'prepared_groundtruth': provenance(prepared / 'groundtruth.bin'),
        'official_private_groundtruth': provenance(official_truth_path),
        'base': provenance(base_path),
        'postings': {str(label): provenance(data / 'prepared-10m' / f'label_{label}.u32bin') for label in sorted(TARGETS)},
        'auditor': provenance(Path(__file__).resolve()),
        'blocked_exact_truth_helper': provenance(Path(__file__).with_name('prepare_recall_secondary.py').resolve()),
        'labels': records,
        'total_exact_point_query_distance_evaluations': sum(evaluations.values()),
        'all_sorted_top10_distances_match_official': bool(np.all(distance_matches)),
        'queries_with_distance_vector_match': int(distance_matches.sum()),
        'official_truth_rows_copied_exactly_by_original_private_query_id': True,
        'independent_scalar_distance_and_label_membership_checks': scalar_checks,
        'tie_policy': 'Compare all ten sorted exact distances, allowing alternative IDs at equal distances; recomputed IDs use ascending point ID as deterministic tie-break',
        'arithmetic': 'uint8 192D integer squared L2; blocked float64 dot/norm scan is exact for all intermediate integers; every official and recomputed result independently checked with scalar Python integer arithmetic',
        'search_outputs_read': False, 'primary_files_modified': False,
        'seconds': time.monotonic() - started,
    }
    # Recheck sealed identities at the end too; the audit does not write primary files.
    require(sha256(proof_path) == PRIMARY_COMMITMENT, 'Primary commitment changed during audit')
    for name, expected in partition['artifacts'].items():
        require(sha256(prepared / name) == expected['sha256'], f'Primary artifact changed during audit:{name}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'out': str(output), 'sha256': sha256(output), 'queries': len(selected),
                      'all_match': result['all_sorted_top10_distances_match_official'],
                      'exact_distance_evaluations': result['total_exact_point_query_distance_evaluations'],
                      'scalar_checks': scalar_checks, 'seconds': result['seconds']}), flush=True)
    require(result['all_sorted_top10_distances_match_official'], 'Exhaustive exact truth differs from official target truth')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('benchmark-results/single-hop/data'))
    parser.add_argument('--primary', type=Path, default=Path('benchmark-results/single-hop/recall-fix/data/natural-private'))
    parser.add_argument('--raw', type=Path, default=Path('benchmark-results/single-hop/recall-fix/data/raw'))
    parser.add_argument('--out', type=Path, default=Path('benchmark-results/single-hop/recall-fix/results/primary-target-truth-audit.json'))
    args = parser.parse_args()
    audit(args.data.resolve(), args.primary.resolve(), args.raw.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
