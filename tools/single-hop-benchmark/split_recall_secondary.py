#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Precommit an unsearched secondary pool into calibration and validation.

The original pool, exact truth and selection commitment are immutable. Preparation
writes new subsets only; sealing subsequently binds the final coordinator-hold
amendment, before either new partition may be searched.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from prepare_recall_validation import POLICY_SHA256, TARGETS, matrix, provenance, require, sha256, write_matrix

SEED = 20261025
POOL_MANIFEST = 'e185982543b7f0be3d0d313944163d0804408df6cf1243fefca7a87cd5cb6b6f'
POOL_EVALUATION = '56a072bf269ee742cadc70c13858d3947b95ba6a3e4057f8a40c857a9cc88bdf'
POOL_SELECTION = 'ea9289d1a23b3a6b17e9aee552b4c0d194d122c06d4a144918444f6bc5c26321'
PRIMARY_COMMITMENT = '3633c03931652bfb1f2cba352b2475732fac82c0c358899c510558985aae52c2'
PROTOCOL_NAME = 'secondary-v2-split-protocol.json'
READY_NAME = 'secondary-v2-partitions-ready.json'
PROOF_NAME = 'secondary-v2-split-commitment.json'


def rank(scope, pool_index, label):
    return hashlib.sha256(f'{SEED}:{scope}:{pool_index}:{label}'.encode()).digest()


def split_indices(labels):
    partitions = {'calibration': [], 'validation': []}
    for label in sorted(set(map(int, labels))):
        indices = sorted((index for index, value in enumerate(labels) if int(value) == label),
                         key=lambda index: rank('split', index, label))
        cut = len(indices) // 4
        partitions['calibration'].extend(indices[:cut])
        partitions['validation'].extend(indices[cut:])
    for name, indices in partitions.items():
        indices.sort(key=lambda index: rank('order-' + name, index, int(labels[index])))
    return partitions


def check_artifacts(directory, expected):
    for name, record in expected.items():
        path = directory / name
        require(path.stat().st_size == record['bytes'] and sha256(path) == record['sha256'], f'Artifact identity changed:{path}')


def prepare(source, output):
    require(not (output / READY_NAME).exists() and not (output / PROTOCOL_NAME).exists()
            and not (output / PROOF_NAME).exists(), 'Refusing to overwrite prepared or sealed split')
    require(sha256(source / 'manifest.json') == POOL_MANIFEST
            and sha256(source / 'evaluation-commitment.json') == POOL_EVALUATION
            and sha256(source / 'selection-commitment.json') == POOL_SELECTION, 'Original unsearched pool changed')
    original = json.loads((source / 'manifest.json').read_text())
    selection = json.loads((source / 'selection-commitment.json').read_text())
    require(selection['policy']['sha256'] == POLICY_SHA256
            and selection['primary_commitment_sha256'] == PRIMARY_COMMITMENT, 'Original policy or natural-query exclusion changed')
    check_artifacts(source, original['artifacts'])
    labels = matrix(source / 'query_labels.u32bin', '<u4')[:, 0]
    queries = matrix(source / 'query.u8bin', 'u1')
    ids = matrix(source / 'query_ids.u32bin', '<u4')[:, 0]
    namespaces = matrix(source / 'source_namespace.u8bin', 'u1')[:, 0]
    source_rows = [json.loads(line) for line in (source / 'source-query-rows.jsonl').read_text().splitlines()]
    require(len(queries) == len(labels) == len(ids) == len(namespaces) == len(source_rows) == 5752
            and set(map(int, labels)) == TARGETS, 'Unexpected source pool coverage')
    embeddings = [row.tobytes() for row in queries]
    require(len(set(embeddings)) == len(embeddings), 'Duplicate source embeddings')
    require(len(set(zip(map(int, namespaces), map(int, ids)))) == len(ids), 'Repeated source namespace/query ID')
    truth_shape = struct.unpack('<II', (source / 'groundtruth.bin').read_bytes()[:8])
    require(truth_shape == (5752, 10), 'Unexpected exact source truth shape')
    truth_ids = np.memmap(source / 'groundtruth.bin', mode='r', dtype='<i4', offset=8, shape=truth_shape)
    truth_distances = np.memmap(source / 'groundtruth.bin', mode='r', dtype='<f4', offset=8 + 4*5752*10, shape=truth_shape)
    indices = split_indices(labels)
    cal, val = set(indices['calibration']), set(indices['validation'])
    require(not cal & val and cal | val == set(range(5752)), 'Split does not partition the source pool')
    require(not {embeddings[index] for index in cal} & {embeddings[index] for index in val}, 'Split embeddings overlap')
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        'schema_version': 1, 'scope': 'supplemental_shifted_atomic_label_v2_split', 'seed': SEED,
        'original_pool_manifest_sha256': POOL_MANIFEST,
        'original_selection_commitment_sha256': POOL_SELECTION,
        'original_evaluation_commitment_sha256': POOL_EVALUATION,
        'policy_sha256': POLICY_SHA256,
        'split_rank': "SHA256('20261025:split:{original_pool_query_index}:{atomic_label}')",
        'calibration_membership': 'For each atomic label take the first floor(n/4) indices ordered by split_rank',
        'validation_membership': 'All remaining indices per atomic label',
        'partition_order': "SHA256('20261025:order-{calibration|validation}:{original_pool_query_index}:{atomic_label}')",
        'source_query_namespace': original['source_query_namespace'],
        'distribution_shift': original['distribution_shift'],
        'original_pool_and_truth_remain_immutable': True,
        'secondary_quality_not_used_to_choose_split': True,
        'validation_is_reserved_for_one_frozen_candidate': True,
        'preparer': provenance(Path(__file__).resolve()),
    }
    protocol_path = output / PROTOCOL_NAME
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    protocol_sha = sha256(protocol_path)
    descriptions = {}
    for name, positions in indices.items():
        directory = output / ('prepared-secondary-atomic-' + name + '-v2')
        directory.mkdir(exist_ok=False)
        for filename in ('base.u8bin', 'labels.txt', *(f'label_{label}.u32bin' for label in
                          sorted(record['original_label'] for record in original['labels']))):
            (directory / filename).symlink_to(os.path.relpath(source / filename, directory))
        write_matrix(directory / 'query.u8bin', queries[positions], 'u1')
        write_matrix(directory / 'query_ids.u32bin', ids[positions], '<u4')
        write_matrix(directory / 'query_labels.u32bin', labels[positions], '<u4')
        write_matrix(directory / 'source_namespace.u8bin', namespaces[positions], 'u1')
        write_matrix(directory / 'source_pool_indices.u32bin', positions, '<u4')
        with (directory / 'groundtruth.bin').open('xb') as target:
            target.write(struct.pack('<II', len(positions), 10))
            np.asarray(truth_ids[positions], dtype='<i4').tofile(target)
            np.asarray(truth_distances[positions], dtype='<f4').tofile(target)
        (directory / 'query_labels.txt').write_text(''.join(f'{int(labels[index])}\n' for index in positions))
        rows = []
        for partition_index, pool_index in enumerate(positions):
            row = dict(source_rows[pool_index], source_pool_query_index=pool_index, query_index=partition_index)
            require(row['atomic_label'] == labels[pool_index] and row['source_query_id'] == ids[pool_index]
                    and row['embedding_sha256'] == hashlib.sha256(embeddings[pool_index]).hexdigest(), 'Source mapping mismatch')
            rows.append(row)
        row_path = directory / 'source-query-rows.jsonl'
        row_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
        counts = Counter(int(labels[index]) for index in positions)
        manifest = copy.deepcopy(original)
        manifest.update({
            'seed': SEED, 'query_count': len(positions), 'split': 'secondary_' + name,
            'validation_scope': 'recall_fix_supplemental_shifted_atomic_label_' + name + '_v2',
            'source_pool_manifest': provenance(source / 'manifest.json'),
            'source_pool_evaluation_commitment': provenance(source / 'evaluation-commitment.json'),
            'source_query_rows': provenance(row_path),
            'split_protocol': provenance(protocol_path), 'split_protocol_sha256': protocol_sha,
            'query_selection': protocol['calibration_membership' if name == 'calibration' else 'validation_membership'] + '; ' + protocol['partition_order'],
            'groundtruth': 'An exact row subset of the immutable full10M exhaustive atomic-label truth from the original secondary pool; source AND truth is never used',
            'verification': {'queries': len(positions), 'base_vectors': 10000000, 'k': 10,
                             'source_exhaustive_groundtruth_sha256': original['artifacts']['groundtruth.bin']['sha256'],
                             'all_subset_vectors_labels_ids_namespaces_and_truth_copied_exactly': True,
                             'all_source_pool_indices_unique': True,
                             'all_calibration_validation_embeddings_and_source_pairs_disjoint': True},
        })
        for record in manifest['labels']:
            label = record['original_label']
            record['selected_queries'] = counts[label]
            record['available_source_pool_queries'] = int(np.count_nonzero(labels == label))
            record.pop('available_one_tag_queries', None)
        filenames = ('query.u8bin', 'query_ids.u32bin', 'query_labels.u32bin', 'groundtruth.bin',
                     'labels.txt', 'query_labels.txt', 'source_namespace.u8bin', 'source-query-rows.jsonl', 'source_pool_indices.u32bin')
        manifest['artifacts'] = {filename: {'bytes': (directory / filename).stat().st_size,
                                          'sha256': sha256(directory / filename)} for filename in filenames}
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (directory / 'label_counts.json').write_text(json.dumps(manifest['labels'], indent=2) + '\n')
        descriptions['secondary-' + name] = {'path': str(directory), 'queries': len(positions), 'labels': len(counts),
                                             'per_label_queries': dict(sorted(counts.items())),
                                             'manifest_sha256': sha256(directory / 'manifest.json'), 'artifacts': manifest['artifacts']}
    ready = {'schema_version': 1, 'scope': protocol['scope'], 'protocol': protocol,
             'split_protocol_sha256': protocol_sha, 'source_pool': str(source),
             'original_pool_manifest_sha256': POOL_MANIFEST,
             'original_selection_commitment_sha256': POOL_SELECTION,
             'original_evaluation_commitment_sha256': POOL_EVALUATION,
             'source_pool_artifacts': original['artifacts'], 'policy_sha256': POLICY_SHA256,
             'primary_commitment_sha256': PRIMARY_COMMITMENT, 'partitions': descriptions,
             'calibration_validation_pool_index_overlap': 0, 'calibration_validation_embedding_overlap': 0,
             'calibration_validation_source_namespace_and_id_overlap': 0,
             'all_original_pool_rows_partitioned_exactly_once': True,
             'original_pool_queries': 5752, 'secondary_pool_never_searched_before_split': True}
    (output / READY_NAME).write_text(json.dumps(ready, indent=2) + '\n')
    # Re-open every file after writing and independently verify direct subset rows.
    for name, description in descriptions.items():
        directory = Path(description['path'])
        check_artifacts(directory, description['artifacts'])
        positions = matrix(directory / 'source_pool_indices.u32bin', '<u4')[:, 0]
        require(np.array_equal(matrix(directory / 'query.u8bin', 'u1'), queries[positions])
                and np.array_equal(matrix(directory / 'query_ids.u32bin', '<u4')[:, 0], ids[positions])
                and np.array_equal(matrix(directory / 'query_labels.u32bin', '<u4')[:, 0], labels[positions])
                and np.array_equal(matrix(directory / 'source_namespace.u8bin', 'u1')[:, 0], namespaces[positions]), 'Written subset differs from original source rows')
        count = len(positions)
        got_ids = np.memmap(directory / 'groundtruth.bin', mode='r', dtype='<i4', offset=8, shape=(count, 10))
        got_distances = np.memmap(directory / 'groundtruth.bin', mode='r', dtype='<f4', offset=8 + 4*count*10, shape=(count, 10))
        require(np.array_equal(got_ids, truth_ids[positions]) and np.array_equal(got_distances, truth_distances[positions]), 'Written exact truth subset differs')
    check_artifacts(source, original['artifacts'])
    require(sha256(source / 'manifest.json') == POOL_MANIFEST, 'Original pool manifest changed')
    print(json.dumps({'prepared': True, 'sealed': False, 'ready_path': str(output / READY_NAME),
                      'ready_sha256': sha256(output / READY_NAME), 'split_protocol_sha256': protocol_sha,
                      'partitions': {name: {key: value for key, value in item.items() if key != 'artifacts'}
                                     for name, item in descriptions.items()}}, indent=2))
    return ready


def seal(output, amendment):
    require(not (output / PROOF_NAME).exists(), 'Refusing to overwrite sealed split')
    ready_path = output / READY_NAME
    ready = json.loads(ready_path.read_text())
    require(ready['scope'] == 'supplemental_shifted_atomic_label_v2_split'
            and ready['policy_sha256'] == POLICY_SHA256
            and sha256(output / PROTOCOL_NAME) == ready['split_protocol_sha256'], 'Split preparation identity changed')
    source = Path(ready['source_pool'])
    require(sha256(source / 'manifest.json') == POOL_MANIFEST
            and sha256(source / 'evaluation-commitment.json') == POOL_EVALUATION
            and sha256(source / 'selection-commitment.json') == POOL_SELECTION, 'Original source commitments changed')
    check_artifacts(source, ready['source_pool_artifacts'])
    for record in ready['partitions'].values():
        directory = Path(record['path'])
        require(sha256(directory / 'manifest.json') == record['manifest_sha256'], 'Prepared split manifest changed')
        check_artifacts(directory, record['artifacts'])
    require(sha256(Path(ready['protocol']['preparer']['path'])) == ready['protocol']['preparer']['sha256'], 'Preparer changed before sealing')
    hold = json.loads(amendment.read_text())
    require(hold.get('secondary_exposure') == 'no secondary search has run', 'Hold amendment does not confirm unused secondary queries')
    require(hold.get('secondary_path_exists') is False, 'Hold amendment reports a secondary run directory')
    saved_amendment = output / 'secondary-v2-hold-amendment.json'
    saved_amendment.write_bytes(amendment.read_bytes())
    proof = {**ready, 'partitions_ready_sha256': sha256(ready_path),
             'selection_commitment_sha256': POOL_SELECTION,
             'hold_amendment': provenance(saved_amendment),
             'validation_quality_not_examined_before_candidate_frozen': True,
             'secondary_validation_is_a_distinct_atomic_derived_workload': True,
             'no_original_primary_or_secondary_artifact_modified': True}
    (output / PROOF_NAME).write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({'sealed': True, 'proof': str(output / PROOF_NAME), 'sha256': sha256(output / PROOF_NAME),
                      'hold_amendment_sha256': sha256(saved_amendment),
                      'partitions': {name: {'queries': item['queries'], 'per_label_queries': item['per_label_queries'],
                                            'manifest_sha256': item['manifest_sha256']} for name, item in ready['partitions'].items()}}, indent=2))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('benchmark-results/single-hop/recall-fix/data/secondary-two-tag/prepared-secondary-atomic-validation'))
    parser.add_argument('--out', type=Path, default=Path('benchmark-results/single-hop/recall-fix/data/secondary-two-tag'))
    parser.add_argument('--seal-only', action='store_true')
    parser.add_argument('--hold-amendment', type=Path)
    args = parser.parse_args()
    if args.seal_only:
        require(args.hold_amendment is not None, '--seal-only requires the final --hold-amendment')
        seal(args.out.resolve(), args.hold_amendment.resolve())
    else:
        prepare(args.source.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
