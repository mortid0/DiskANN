#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Audit centroid fallback plus nearest-PQ corpus entry routing with exposed development and sealed final validation.

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
import math
import os
from pathlib import Path

import numpy as np

import analyze_recall_fix as v1
import run_recall_fix_v4 as protocol
from summarize_heldout import cpp_observations, native_observations

SPLIT_SHA = '34965480cdef01db2fc9ecc757b48173efd2afb11bfb095c1422bfb9a4316af5'
HOLD_SHA = '435a95faeb3dbf095e8ee0319f3416f7c824038d8140d7bc5e0ba9771a4b2198'
PHASES = ('development-recheck', 'secondary-development', 'secondary-validation')
MEMBERSHIP_MANIFEST_SHA = 'a59365f315a91763f5cbd39025fcc2e6b331e0d8b3dfad5724dcbe31bd3a4e94'
MEMBERSHIP_PROOF_SHA = 'c8420e5b60a4a41bf1571b9e7507c2f7f4759c8b4709c84990c35fa3af853335'


def check_phase_declarations(phase, quality_only, evidence, manifest, freeze):
    fresh = 'previously_unsearched_partition' if phase == 'secondary-validation' else 'previously_exposed_development'
    v1.require(phase in PHASES and (phase != 'secondary-validation' or not quality_only)
               and evidence.get('phase') == manifest.get('phase') == freeze.get('phase') == phase
               and evidence.get('query_freshness') == manifest.get('query_freshness') == freeze.get('query_freshness') == fresh
               and manifest.get('timing_authoritative') is (not quality_only),
               'V4 phase/freshness declaration differs')
    return fresh


def load_memberships(directory):
    v1.require(v1.sha(directory / 'manifest.json') == MEMBERSHIP_MANIFEST_SHA,
               'Entry membership corpus differs from the pinned full64-label natural validation manifest')
    proof_path = directory.parent / 'exclusion-evidence.json'
    v1.require(v1.sha(proof_path) == MEMBERSHIP_PROOF_SHA, 'Immutable full64 posting proof changed')
    manifest, proof = json.loads((directory / 'manifest.json').read_text()), json.loads(proof_path.read_text())
    v1.require(len(manifest['labels']) == 64 and len(proof['posting_artifacts']) == 64,
               'Entry membership proof must cover the original64 labels')
    postings = {}
    for item in manifest['labels']:
        label = item['original_label']
        path = directory / f'label_{label}.u32bin'
        expected = proof['posting_artifacts'][str(label)]
        v1.require(v1.sha(path) == expected['sha256'] and path.stat().st_size == expected['bytes'],
                   f'Immutable entry membership posting changed: {label}')
        values, _ = v1.matrix(path, '<u4')
        v1.require(values.shape == (item['full_10m_count'], 1) and np.all(values < 10000000)
                   and np.all(np.diff(values[:, 0].astype(np.int64)) > 0), 'Malformed immutable posting')
        postings[label] = values[:, 0]
    return postings


def crc64_ecma182(data):
    value = 0
    for byte in data:
        value ^= byte << 56
        for _ in range(8):
            value = ((value << 1) ^ (0x42f0e1eba9ea3693 if value & (1 << 63) else 0)) & ((1 << 64) - 1)
    return value



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
        v1.require(v1.sha(directory / 'manifest.json') == expected['manifest_sha256'], 'V4 partition manifest changed')
        for name, record in expected['artifacts'].items():
            v1.require(v1.sha(directory / name) == record['sha256']
                       and (directory / name).stat().st_size == record['bytes'], 'V4 partition artifact changed')
        workload = v1.supplemental_prepared(directory)
        manifest = workload['manifest']
        v1.require(manifest.get('split') == phase.replace('-', '_')
                   and manifest.get('split_protocol_sha256') == declaration['split_protocol_sha256']
                   and manifest.get('source_pool_manifest', {}).get('sha256') == declaration['original_pool_manifest_sha256']
                   and manifest.get('validation_scope') == f'recall_fix_supplemental_shifted_atomic_label_{short}_v2',
                   'V4 partition namespace/scope/protocol differs')
        positions, _ = v1.matrix(directory / 'source_pool_indices.u32bin', '<u4')
        v1.require(positions.shape == (len(workload['queries']), 1)
                   and positions[:, 0].tolist() == reconstructed[short], 'V4 split/order differs from independent hash reconstruction')
        positions = positions[:, 0]
        for field in ('queries', 'labels', 'ids', 'namespace', 'truth', 'distances'):
            v1.require(np.array_equal(workload[field], pool[field][positions]), f'V4 partition {field} differs from immutable original pool')
        rows = v1.read_rows(directory / 'source-query-rows.jsonl')
        expected_rows = [dict(pool_rows[int(position)], query_index=index, source_pool_query_index=int(position))
                         for index, position in enumerate(positions)]
        v1.require(rows == expected_rows and len(rows) == expected['queries'], 'V4 source-row provenance differs')
        partitions[phase] = workload
        selected[phase] = set(map(int, positions))
    v1.require(not selected['secondary-calibration'] & selected['secondary-validation']
               and selected['secondary-calibration'] | selected['secondary-validation'] == set(range(len(pool['queries']))),
               'V4 partitions overlap or omit original pool rows')
    return partitions, {'source': v1.asset(args.split_commitment), 'hold_amendment': v1.asset(hold_path),
        'original_pool_proof': pool_proof, 'all_pool_rows_partitioned_once': True,
        'calibration_validation_embedding_overlap': 0,
        'partitions': {name: {'queries': len(workload['queries']), 'labels': dict(sorted(Counter(map(int, workload['labels'])).items()))}
                       for name, workload in partitions.items()},
        'scope': 'Original secondary split retained; natural and secondary calibration queries are exposed development; final validation remains separate.'}


def same_candidate(previous, current):
    for name in v1.ENGINES:
        v1.require(previous['engines'][name]['binary_sha256'] == current['engines'][name]['binary_sha256']
                   and previous['engines'][name]['index_after'] == current['engines'][name]['index_before'],
                   f'{name} assets changed between calibration and fresh validation')
    for name in ('baseline', 'candidate'):
        v1.require(previous['engines'][name]['source_archive']['sha256']
                   == current['engines'][name]['source_archive']['sha256'],
                   f'{name} source archive changed after calibration')


def check_entry_map(entries, old_entries, postings):
    v1.require(all(type(key) is str and key.isdecimal() and str(int(key)) == key for key in entries)
               and len(entries) == len(postings) and set(map(int, entries)) == set(postings),
               'Entry pool label coverage differs from immutable postings')
    v1.require(set(old_entries) == set(entries), 'Old centroid metadata label coverage differs')
    for key, ids in entries.items():
        label = int(key)
        v1.require(type(ids) is list and 1 <= len(ids) <= 5
                   and all(type(value) is int for value in ids) and len(set(ids)) == len(ids),
                   'Entry pool is empty, duplicated or oversized')
        v1.require(type(old_entries.get(key)) is list and old_entries[key]
                   and all(type(value) is int for value in old_entries[key]) and ids[0] == old_entries[key][0],
                   'Entry pool lost its global-centroid fallback')
        posting = postings[label]
        positions = np.searchsorted(posting, ids)
        v1.require(all(position < len(posting) and int(posting[position]) == value
                       for position, value in zip(positions, ids)),
                   'Entry pool contains a point outside its immutable label')


def entry_proof(args, candidate, parent, candidate_root):
    publication = json.loads(args.candidate_build_log.read_text())
    provenance = json.loads(args.entry_provenance.read_text())
    host_base = candidate_root.parent.parent
    def canonical(value):
        path = Path(value)
        return Path(os.path.normpath(str(path if path.is_absolute() else host_base / path)))
    parent_root = candidate_root.parent / 'rust-native-10m-centroid-build'
    input_directory = parent_root / 'generation-00000000000000000001'
    output_directory = candidate_root / 'generation-00000000000000000001'
    v1.require(publication.get('schema_version') == 1
               and publication.get('status') == 'published and reopened successfully'
               and canonical(publication.get('input_root', '')) == parent_root
               and canonical(publication.get('output_root', '')) == candidate_root
               and canonical(publication.get('generation_directory', '')) == output_directory
               and publication.get('validated_sequence') == 0,
               'Spatial-entry publication root or sequence differs')
    manifest = publication.get('source_manifest_before')
    v1.require(type(manifest) is dict and manifest == publication.get('source_manifest_after')
               == provenance.get('original_construction_manifest')
               and publication.get('manifest_payload_preserved_verbatim') is True
               and provenance.get('manifest_payload_preserved_verbatim') is True,
               'Spatial-entry intervention rewrites graph construction history')
    v1.require(manifest.get('l_build') == 100 and manifest.get('label_degree') == 32
               and manifest.get('max_degree') == 128 and type(manifest.get('alpha')) in (int, float)
               and math.isclose(manifest['alpha'], 1.2, rel_tol=0, abs_tol=1e-6),
               'Spatial entry changes graph construction breadth or degree')
    v1.require(provenance.get('only_replaced_artifact') == 'labels.bin'
               and provenance.get('added_artifact') == 'entry-pool-provenance.json'
               and provenance.get('query_workload_not_read') is True
               and provenance.get('graph_vectors_PQ_preserved_exactly') is True,
               'Spatial-entry provenance declares another intervention')
    payload_checksums = publication.get('manifest_payload_crc64_ecma182', {})
    v1.require(type(payload_checksums.get('source')) is int
               and payload_checksums['source'] == payload_checksums.get('installed'),
               'Spatial-entry intervention changed the serialized construction manifest payload')
    current = candidate['index_before']
    old = parent['engines']['candidate']['index_after']
    raw_source_hashes = publication.get('source_files_sha256', {})
    v1.require(provenance.get('source_files_sha256') == raw_source_hashes,
               'Spatial-entry source inventory differs from installed provenance')
    def normalized_hashes(declared):
        result = {}
        for name, digest in declared.items():
            key = str(canonical(name))
            v1.require(v1.digest(digest) and (key not in result or result[key] == digest),
                       'Invalid or conflicting spatial-entry file hash')
            result[key] = digest
        return result
    source_hashes = normalized_hashes(raw_source_hashes)
    output_hashes = normalized_hashes(publication.get('output_files_sha256', {}))
    v1.require(set(current['files_sha256']) == set(old['files_sha256']) | {'entry-pool-provenance.json'},
               'Spatial-entry publication adds or removes unexpected generation files')
    for name, digest in old['files_sha256'].items():
        if name not in ('labels.bin', 'CATALOG'):
            v1.require(current['files_sha256'].get(name) == digest,
                       f'Spatial-entry intervention changed an immutable parent artifact: {name}')
    for name, digest in old['files_sha256'].items():
        v1.require(source_hashes.get(str(input_directory / name)) == digest,
                   f'Spatial-entry parent file changed: {name}')
    for name, digest in current['files_sha256'].items():
        v1.require(output_hashes.get(str(output_directory / name)) == digest,
                   f'Spatial-entry installed artifact differs from publication: {name}')
    for key, filename in [('current_sha256', 'CURRENT'), ('journal_sha256', 'updates.wal')]:
        v1.require(source_hashes.get(str(parent_root / filename)) == old[key]
                   and output_hashes.get(str(candidate_root / filename)) == current[key],
                   f'Spatial-entry publication envelope differs: {filename}')
    v1.require(source_hashes.get(str(input_directory / 'CATALOG')) == old['catalog_sha256']
               and output_hashes.get(str(output_directory / 'CATALOG')) == current['catalog_sha256']
               and current['files_sha256'].get('entry-pool-provenance.json') == v1.sha(args.entry_provenance),
               'Spatial-entry catalog/provenance binding differs')
    provenance_record = publication.get('entry_provenance_artifact', {})
    v1.require(provenance_record.get('name') == 'entry-pool-provenance.json'
               and provenance_record.get('bytes') == args.entry_provenance.stat().st_size
               and provenance_record.get('checksum') == crc64_ecma182(args.entry_provenance.read_bytes()),
               'Spatial-entry provenance artifact CRC or length differs')
    selection = publication.get('entry_selection', {})
    v1.require(selection == provenance.get('entry_selection') and selection.get('corpus_only') is True
               and selection.get('query_results_used') is False,
               'Spatial representatives used query outcomes')
    policy = publication.get('entry_policy', {})
    v1.require(policy == selection.get('policy') and policy.get('spatial_clusters') == 4
               and policy.get('maximum_entries') == 5 and policy.get('maximum_fit_members') == 4096
               and policy.get('lloyd_iterations') == 8
               and policy.get('sample_seed_base') == 0x6c6162656c5f7633
               and policy.get('sample_seed_mixing') == 'XOR u64(label)'
               and policy.get('global_centroid_first') is True,
               'Spatial representative policy differs from precommitted corpus-only design')
    bindings = publication.get('verified_bindings', {})
    v1.require(all(bindings.get(key) is True for key in
                   ('point_count_equal', 'point_labels_equal', 'postings_equal', 'stable_ids_equal', 'global_centroid_first')),
               'Spatial-entry publication changed membership or stable IDs')
    entries = publication.get('entry_points', {})
    v1.require(entries == selection.get('new_entry_points'), 'Published entry points differ from selection audit')
    postings = load_memberships(args.membership_prepared)
    check_entry_map(entries, selection.get('old_entry_points', {}), postings)
    return {'publication': v1.asset(args.candidate_build_log),
            'checksummed_provenance': v1.asset(args.entry_provenance),
            'parent_graph_and_PQ_preserved': True, 'source_membership_preserved': True,
            'all_entry_memberships_independently_checked': True,
            'membership_manifest': v1.asset(args.membership_prepared / 'manifest.json'), 'labels': len(entries),
            'entry_count': sum(map(len, entries.values())), 'policy': policy,
            'scope': 'Metadata-only corpus entry intervention; public helper and source inputs retained. Sample clustering is not independently recomputed by this Python analyzer.'}


def identity_proof(args, logs, manifest):
    evidence = json.loads(args.identity.read_text())
    prior = json.loads(args.prior_identity.read_text())
    freeze = json.loads(args.candidate_freeze.read_text())
    quality_only = args.quality_only
    frozen_protocol = protocol.validate_protocol_freeze(args.protocol_freeze)
    protocol.validate_runtime_archive(args.candidate_source_archive, frozen_protocol)
    fresh = 'previously_exposed_development' if args.phase != 'secondary-validation' else 'previously_unsearched_partition'
    proof_path = args.primary_commitment if args.phase == 'development-recheck' else args.split_commitment
    v1.require(evidence.get('schema_version') == 1 and evidence.get('host') == prior.get('host')
               and evidence.get('cpu_affinity') == manifest.get('cpu_affinity') == list(range(8)), 'V4 host/core affinity differs')
    v1.require(manifest.get('identity') == evidence and evidence.get('output_root') == manifest.get('output_root'),
               'V4 run manifest/identity binding differs')
    check_phase_declarations(args.phase, quality_only, evidence, manifest, freeze)
    for declaration in (evidence, manifest, freeze):
        v1.require(declaration.get('protocol_version') == 4
                   and declaration.get('entry_policy') == protocol.ENTRY_POLICY
                   and declaration.get('protocol_source_freeze_sha256') == v1.sha(args.protocol_freeze)
                   and declaration.get('protocol_source_archive_sha256') == frozen_protocol['archive']['sha256'],
                   'V4 protocol source freeze or dual-entry policy differs')
    v1.require(evidence.get('plan_before_sha256') == evidence.get('plan_after_sha256') == v1.FROZEN_SHA
               and freeze.get('plan_sha256') == v1.FROZEN_SHA,
               'V4 changes the original frozen L policy')
    v1.require(evidence.get('prepared_manifest_sha256') == manifest.get('prepared_manifest_sha256')
               == freeze.get('prepared_manifest_sha256') == v1.sha(args.prepared / 'manifest.json')
               and evidence.get('fresh_proof_sha256') == manifest.get('fresh_proof_sha256')
               == freeze.get('fresh_proof_sha256') == v1.sha(proof_path)
               and evidence.get('candidate_freeze_sha256') == manifest.get('candidate_freeze_sha256') == v1.sha(args.candidate_freeze)
               and freeze.get('candidate_build_log_sha256') == v1.sha(args.candidate_build_log),
               'V4 data/build/source freeze binding differs')
    engines = evidence.get('engines', {})
    v1.require(set(engines) == set(v1.ENGINES), 'Incomplete V4 engine identities')
    for name in v1.ENGINES:
        item = engines[name]
        v1.require(v1.digest(item.get('binary_sha256')) and item.get('binary_before_sha256') == item['binary_sha256']
                   == freeze.get('binary_sha256', {}).get(name) and item.get('log_sha256') == v1.sha(logs[name]),
                   f'{name} V4 binary/log identity changed')
        before, after = item.get('index_before', {}), item.get('index_after', {})
        v1.require(before and before == after == freeze.get('index_identity', {}).get(name), f'{name} V4 index changed')
        for key, value in before.items():
            values = value.values() if key == 'files_sha256' else [value]
            v1.require(all(v1.digest(digest) for digest in values), f'{name} V4 index SHA declaration invalid')
        if name != 'cpp':
            v1.require(before.get('files_sha256'), f'{name} V4 full generation inventory absent')
    v1.require({key: engines['baseline']['index_before'][key] for key in prior['native']['index_before']}
               == prior['native']['index_before']
               and engines['baseline']['binary_sha256'] == prior['native']['binary_sha256'], 'Original Rust baseline changed')
    v1.require(engines['cpp']['index_before'] == prior['cpp']['index_before']
               and engines['cpp']['binary_sha256'] == prior['cpp']['binary_sha256'], 'Original C++ baseline changed')
    for filename in ('vectors.bin', 'index_pq_pivots.bin', 'index_pq_compressed.bin'):
        v1.require(v1.digest(engines['candidate']['index_before']['files_sha256'].get(filename))
                   and engines['candidate']['index_before']['files_sha256'].get(filename)
                   == engines['baseline']['index_before']['files_sha256'].get(filename),
                   f'V4 candidate changes the frozen vector order or PQ artifacts: {filename}')
    archived = {}
    for name, path in [('baseline', args.baseline_source_archive), ('candidate', args.candidate_source_archive)]:
        source = engines[name].get('source_archive', {})
        v1.require(source.get('sha256') == source.get('before_sha256')
                   == freeze.get('source_archive_sha256', {}).get(name) == v1.sha(path), f'{name} V4 source archive changed')
        archived[name] = v1.archived_files(path)
    v1.require(set(archived['baseline']) == set(archived['candidate']), 'V4 runtime source sets differ')
    changed = sorted(name for name in archived['baseline'] if archived['baseline'][name] != archived['candidate'][name])
    v1.require(all(archived['candidate'].get(name) == digest for name, digest in protocol.RUNTIME_CORE_SHA256.items()),
               'V4 changes the frozen corpus selector or V9 fallback runtime')
    v1.require(engines['candidate']['binary_sha256'] == protocol.RUNTIME_BINARY_SHA
               == frozen_protocol['runtime_candidate_binary_sha256'], 'V4 candidate binary is not the frozen V9 runtime')
    v1.require(changed == ['diskann-disk/src/build/filtered.rs', 'diskann-disk/src/filtered/view.rs', 'diskann-disk/src/search/provider/disk_provider.rs', 'diskann-disk/src/search/search_mode.rs'], 'V4 source changes extend beyond generic construction')
    v1.require(v1.sha(args.baseline_source_archive) == prior['native']['source_archive']['sha256'], 'V4 baseline sources changed')
    candidate_root = Path(freeze['candidate_root'])
    roots = {'candidate': str(candidate_root), 'baseline': str(candidate_root.parent / 'rust-native-10m-frozen')}
    cpp_prefix = str(candidate_root.parent / 'yfcc10m_stitched_R32_R128_L100')
    for name in ('baseline', 'candidate'):
        exp = v1.one([row for row in v1.read_rows(logs[name]) if row.get('kind') == 'experiment'], f'{name} V4 experiment')
        v1.require(exp.get('root') == roots[name] and exp.get('prepared') == manifest.get('prepared')
                   and exp.get('action') == 'query' and exp.get('n_base') == 10000000 and exp.get('dimensions') == 192
                   and exp.get('index_threads') == 8 and exp.get('beam') == 4 and exp.get('shared_cache_nodes') == 0
                   and exp.get('cache_bypass') is True and exp.get('storage') == 'native_io_uring'
                   and exp.get('warmup') == 'full_same_process_same_worker_count'
                   and exp.get('query_scheduling') == 'dynamic_one', f'{name} V4 index/query/I/O protocol differs')
    v1.require(all(row.get('index_prefix') == cpp_prefix and row.get('prepared') == manifest.get('prepared')
                   for row in v1.read_rows(logs['cpp'])), 'V4 C++ index/query differs')
    for job in manifest['runs']:
        command, name = job.get('command', []), job['implementation']
        option = '--index-prefix' if name == 'cpp' else '--root'
        v1.require(option in command and command[command.index(option) + 1] == (cpp_prefix if name == 'cpp' else roots[name]),
                   f'{name} V4 retained job command index differs')
    protocol.validate_freeze_order(frozen_protocol.get('timestamp_utc'), freeze.get('timestamp_utc'),
                                   manifest['runs'][0]['start'])
    runner = Path(__file__).with_name('run_recall_fix_v4.py')
    cpp_runner = Path(__file__).parent / 'cpp-port/run_reference.py'
    v1.require(evidence.get('runner_sha256') == v1.sha(runner) and evidence.get('cpp_runner_sha256') == v1.sha(cpp_runner),
               'V4 runner or C++ normalizer changed')
    amendment = json.loads(args.development_amendment.read_text())
    amendment_sha = v1.sha(args.development_amendment)
    protocol.validate_amendment(amendment)
    v1.require(amendment_sha == frozen_protocol.get('development_amendment_sha256')
               and amendment.get('candidate_root') == str(candidate_root)
               and amendment.get('entry_publication_sha256') == v1.sha(args.candidate_build_log),
               'V4 amendment differs from frozen protocol or unchanged pool publication')
    v1.require(amendment.get('secondary_validation_unsearched') is True
               and amendment.get('secondary_split_sha256') == SPLIT_SHA
               and evidence.get('development_amendment_sha256')
               == manifest.get('development_amendment_sha256')
               == freeze.get('development_amendment_sha256') == amendment_sha,
               'Spatial-routing amendment or exposed-query declaration changed')
    parent = json.loads(args.parent_identity.read_text())
    v1.require(amendment.get('candidate2_calibration_identity_sha256') == v1.sha(args.parent_identity),
               'Spatial-routing amendment binds another parent graph')
    for filename in ('index_disk.index', 'vectors.bin', 'index_pq_pivots.bin', 'index_pq_compressed.bin'):
        v1.require(engines['candidate']['index_before']['files_sha256'].get(filename)
                   == parent['engines']['candidate']['index_after']['files_sha256'].get(filename),
                   f'Spatial routing changed the frozen parent graph/vector/PQ bytes: {filename}')
    entry_audit = entry_proof(args, engines['candidate'], parent, candidate_root)
    calibrated = None
    if args.phase == 'secondary-validation':
        v1.require(args.calibration_identity is not None, 'Fresh V4 validation must bind frozen calibration assets')
        calibrated = json.loads(args.calibration_identity.read_text())
        v1.require(calibrated.get('phase') == 'secondary-development', 'V4 validation calibration identity phase differs')
        same_candidate(calibrated, evidence)
        protocol.validate_calibration(calibrated,
            {name: item['index_before'] for name, item in engines.items()},
            {name: item['binary_sha256'] for name, item in engines.items()},
            {name: engines[name]['source_archive']['sha256'] for name in ('baseline', 'candidate')},
            v1.sha(args.protocol_freeze), amendment_sha)
        v1.require(all(declaration.get('calibration_identity_sha256') == v1.sha(args.calibration_identity)
                       for declaration in (evidence, manifest, freeze)), 'V4 calibration identity binding differs')
        claim = evidence.get('final_validation_claim')
        v1.require(type(claim) is dict and claim.get('retained_copy') == 'final-validation-claim.json'
                   and claim == manifest.get('final_validation_claim') == freeze.get('final_validation_claim'),
                   'V4 final validation lacks its exclusive launch claim')
        local_claim = (args.artifact_root or args.jobs_manifest.parent) / claim['retained_copy']
        v1.require(v1.sha(local_claim) == claim.get('sha256'), 'Final validation claim bytes differ')
        claimed = json.loads(local_claim.read_text())
        v1.require(claimed.get('protocol_version') == 4 and claimed.get('phase') == 'secondary-validation'
                   and claimed.get('entry_policy') == protocol.ENTRY_POLICY
                   and claimed.get('calibration_identity_sha256') == v1.sha(args.calibration_identity)
                   and claimed.get('protocol_source_freeze_sha256') == v1.sha(args.protocol_freeze)
                   and claimed.get('development_amendment_sha256') == amendment_sha
                   and claimed.get('prepared_manifest_sha256') == evidence['prepared_manifest_sha256']
                   and claimed.get('fresh_proof_sha256') == evidence['fresh_proof_sha256']
                   and claimed.get('index_identity') == freeze['index_identity']
                   and claimed.get('binary_sha256') == freeze['binary_sha256']
                   and claimed.get('source_archive_sha256') == freeze['source_archive_sha256']
                   and claimed.get('timestamp_utc', '') <= freeze['timestamp_utc'],
                   'V4 exclusive final claim does not match the frozen candidate and partition')
    else:
        v1.require(all(declaration.get('final_validation_claim') is None for declaration in (evidence, manifest, freeze)),
                   'Exposed development cannot consume a fresh-validation launch claim')
    return {'source': v1.asset(args.identity), 'candidate_freeze': v1.asset(args.candidate_freeze),
            'changed_archived_source_files': changed, 'declarations': evidence,
            'calibration_identity': v1.asset(args.calibration_identity) if calibrated else None,
            'source_archives': {'baseline': v1.asset(args.baseline_source_archive), 'candidate': v1.asset(args.candidate_source_archive)},
            'candidate_build_log': v1.asset(args.candidate_build_log), 'query_freshness': fresh,
            'development_amendment': v1.asset(args.development_amendment),
            'parent_identity': v1.asset(args.parent_identity), 'spatial_entry_proof': entry_audit,
            'protocol_source_freeze': v1.asset(args.protocol_freeze), 'entry_policy': protocol.ENTRY_POLICY,
            'final_validation_claim': v1.asset(local_claim) if args.phase == 'secondary-validation' else None}


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
        workload = partitions['secondary-calibration' if args.phase == 'secondary-development' else args.phase]
        source = args.calibration_prepared if args.phase == 'secondary-development' else args.validation_prepared
        v1.require(v1.sha(args.prepared / 'manifest.json') == v1.sha(source / 'manifest.json'), 'Selected V4 phase prepared path differs')
        freshness = 'previously_exposed_development' if args.phase == 'secondary-development' else 'previously_unsearched_partition'
    quality_only = args.quality_only
    postings = {}
    for label in sorted(set(map(int, workload['labels']))):
        values, _ = v1.matrix(args.prepared / f'label_{label}.u32bin', '<u4')
        v1.require(values.shape[1] == 1 and np.all(np.diff(values[:, 0].astype(np.int64)) > 0), 'Invalid V4 posting')
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
    result.update({'schema_version': 4, 'phase': args.phase, 'query_freshness': freshness, 'timing_excluded': quality_only,
        'scope': 'Exposed natural-query development recheck' if args.phase == 'development-recheck' else
                 'Exposed atomic-query development recheck' if args.phase == 'secondary-development' else 'Fresh dual-entry atomic-label validation; separate five-label shifted workload',
        'prepared_identity': workload['identity'], 'split_proof': split_proof, 'jobs_proof': jobs, 'identity_proof': identity,
        'cutoff_diagnostic': cutoff, 'inputs': inputs, 'targets': list(v1.TARGETS), 'acceptance_threshold': None,
        'bootstrap': {'replications': args.bootstrap, 'seed': args.seed, 'unit': 'paired query',
                      'per_label_simultaneous_family': 3 * len(result['per_label']), 'aggregate_simultaneous_family': 3},
        'limitations': [
            'The natural primary query set influenced candidate development and is not fresh validation for candidate 4.',
            'Secondary predicates are derived atomic labels from two-tag vectors and cannot establish natural one-tag recall or workload throughput.',
            'At most two initial seeds are retained within the unchanged search L; the extra initial seed may alter the whole traversal and total SSD reads, which must be measured.',
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
                 'baseline-source-archive', 'candidate-source-archive', 'candidate-freeze', 'candidate-build-log', 'development-amendment', 'parent-identity', 'entry-provenance', 'membership-prepared', 'protocol-freeze', 'out', 'csv'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path)
    parser.add_argument('--quality-only', action='store_true', help='Exclude all performance claims; final validation always requires authoritative timings')
    parser.add_argument('--calibration-identity', type=Path)
    parser.add_argument('--bootstrap', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=20261025)
    args = parser.parse_args()
    v1.require(args.phase != 'secondary-validation' or not args.quality_only,
               'Fresh secondary validation requires authoritative timings')
    v1.require(args.bootstrap >= 10000, 'At least10K query bootstrap replications required')
    v1.require(not args.out.exists() and not args.csv.exists(), 'V4 analysis outputs already exist')
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
