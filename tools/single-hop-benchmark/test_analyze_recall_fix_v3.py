#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Guard corpus-only entry metadata and exposed/fresh query declarations."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import analyze_recall_fix_v3 as analysis


class EntryMaps(unittest.TestCase):
    def setUp(self):
        self.postings = {9: np.asarray([1, 3, 5, 7], dtype='<u4'), 24: np.asarray([2, 4, 6], dtype='<u4')}
        self.old = {'9': [3], '24': [4]}
        self.entries = {'9': [3, 1, 5, 7], '24': [4, 2, 6]}

    def test_preserves_centroid_first_while_other_pool_ids_need_not_be_first_posting(self):
        analysis.check_entry_map(self.entries, self.old, self.postings)

    def test_rejects_duplicate_entry(self):
        self.entries['9'] = [3, 3]
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            analysis.check_entry_map(self.entries, self.old, self.postings)

    def test_rejects_point_belonging_only_to_another_label(self):
        self.entries['9'] = [3, 2]
        with self.assertRaisesRegex(ValueError, 'outside its immutable label'):
            analysis.check_entry_map(self.entries, self.old, self.postings)

    def test_rejects_reset_global_centroid_even_when_replacement_matches_label(self):
        self.entries['9'] = [1, 3, 5]
        with self.assertRaisesRegex(ValueError, 'global-centroid fallback'):
            analysis.check_entry_map(self.entries, self.old, self.postings)

    def test_rejects_noncanonical_label_aliases(self):
        self.entries['09'] = self.entries.pop('9')
        with self.assertRaisesRegex(ValueError, 'coverage differs'):
            analysis.check_entry_map(self.entries, self.old, self.postings)

    def test_rejects_missing_label(self):
        del self.entries['24']
        with self.assertRaisesRegex(ValueError, 'coverage differs'):
            analysis.check_entry_map(self.entries, self.old, self.postings)


class PhaseFreshness(unittest.TestCase):
    def declarations(self, phase, quality_only):
        value = {'phase': phase, 'query_freshness': 'previously_unsearched_partition'
                 if phase == 'secondary-validation' else 'previously_exposed_development'}
        return copy.deepcopy(value), dict(value, timing_authoritative=not quality_only), copy.deepcopy(value)

    def test_secondary_development_stays_exposed_even_with_authoritative_timing(self):
        rows = self.declarations('secondary-development', False)
        self.assertEqual(analysis.check_phase_declarations('secondary-development', False, *rows),
                         'previously_exposed_development')

    def test_rejects_each_relabeled_exposed_declaration(self):
        for changed in range(3):
            rows = self.declarations('secondary-development', True)
            rows[changed]['query_freshness'] = 'previously_unsearched_partition'
            with self.assertRaisesRegex(ValueError, 'freshness declaration differs'):
                analysis.check_phase_declarations('secondary-development', True, *rows)

    def test_final_validation_requires_authoritative_timing(self):
        rows = self.declarations('secondary-validation', True)
        with self.assertRaisesRegex(ValueError, 'freshness declaration differs'):
            analysis.check_phase_declarations('secondary-validation', True, *rows)


class PublicationProof(unittest.TestCase):
    def fixture(self, root):
        candidate_root = Path('/remote/indexes/rust-native-10m-spatial-entries')
        parent_root = candidate_root.parent / 'rust-native-10m-centroid-build'
        generation = 'generation-00000000000000000001'
        old_files = {'index_disk.index': '1' * 64, 'vectors.bin': '2' * 64, 'labels.bin': '3' * 64,
                     'index_pq_pivots.bin': '4' * 64, 'index_pq_compressed.bin': '5' * 64, 'CATALOG': '6' * 64}
        old = {'files_sha256': old_files, 'current_sha256': '7' * 64,
               'catalog_sha256': '6' * 64, 'journal_sha256': '8' * 64}
        inputs = {str(parent_root / generation / name): digest for name, digest in old_files.items()}
        inputs.update({'indexes/rust-native-10m-centroid-build/CURRENT': '7' * 64,
                       'indexes/rust-native-10m-centroid-build/updates.wal': '8' * 64})
        policy = {'spatial_clusters': 4, 'maximum_entries': 5, 'maximum_fit_members': 4096,
                  'lloyd_iterations': 8, 'sample_seed_base': 0x6c6162656c5f7633,
                  'sample_seed_mixing': 'XOR u64(label)', 'global_centroid_first': True}
        selection = {'corpus_only': True, 'query_results_used': False, 'policy': policy,
                     'old_entry_points': {'9': [3]}, 'new_entry_points': {'9': [3, 1, 5]}}
        manifest = {'l_build': 100, 'label_degree': 32, 'max_degree': 128, 'alpha': 1.2}
        provenance = {'original_construction_manifest': manifest, 'manifest_payload_preserved_verbatim': True,
                      'only_replaced_artifact': 'labels.bin', 'added_artifact': 'entry-pool-provenance.json',
                      'query_workload_not_read': True, 'graph_vectors_PQ_preserved_exactly': True,
                      'source_files_sha256': inputs, 'entry_selection': selection}
        provenance_path = root / 'entry-pool-provenance.json'
        provenance_path.write_text(json.dumps(provenance))
        installed = dict(old_files, **{'labels.bin': '9' * 64, 'CATALOG': 'a' * 64,
                                       'entry-pool-provenance.json': analysis.v1.sha(provenance_path)})
        outputs = {str(candidate_root / generation / name): digest for name, digest in installed.items()}
        outputs.update({'indexes/rust-native-10m-spatial-entries/CURRENT': 'b' * 64,
                        'indexes/rust-native-10m-spatial-entries/updates.wal': '8' * 64})
        publication = {'schema_version': 1, 'status': 'published and reopened successfully',
            'input_root': 'indexes/rust-native-10m-centroid-build', 'output_root': 'indexes/rust-native-10m-spatial-entries',
            'generation_directory': str(candidate_root / generation), 'validated_sequence': 0,
            'source_manifest_before': manifest, 'source_manifest_after': manifest,
            'manifest_payload_preserved_verbatim': True, 'manifest_payload_crc64_ecma182': {'source': 123, 'installed': 123},
            'source_files_sha256': inputs, 'output_files_sha256': outputs,
            'entry_provenance_artifact': {'name': 'entry-pool-provenance.json', 'bytes': provenance_path.stat().st_size,
                                          'checksum': analysis.crc64_ecma182(provenance_path.read_bytes())},
            'entry_policy': policy, 'entry_selection': selection, 'entry_points': {'9': [3, 1, 5]},
            'verified_bindings': {name: True for name in ('point_count_equal', 'point_labels_equal', 'postings_equal',
                                                        'stable_ids_equal', 'global_centroid_first')}}
        publication_path = root / 'entry-pool-result.json'
        publication_path.write_text(json.dumps(publication))
        membership = root / 'full64-membership'
        membership.mkdir()
        (membership / 'manifest.json').write_text('{}')
        secondary = root / 'only-five-query-labels'
        secondary.mkdir()
        args = argparse.Namespace(candidate_build_log=publication_path, entry_provenance=provenance_path,
                                  membership_prepared=membership, prepared=secondary)
        candidate = {'index_before': {'files_sha256': installed, 'current_sha256': 'b' * 64,
                                     'catalog_sha256': 'a' * 64, 'journal_sha256': '8' * 64}}
        parent = {'engines': {'candidate': {'index_after': old}}}
        return args, candidate, parent, candidate_root, publication

    def test_uses_separate_full_membership_corpus_and_accepts_f32_alpha_serialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, candidate, parent, candidate_root, _ = self.fixture(Path(temporary))
            with patch.object(analysis, 'load_memberships', return_value={9: np.asarray([1, 3, 5], dtype='<u4')}) as loaded:
                result = analysis.entry_proof(args, candidate, parent, candidate_root)
            loaded.assert_called_once_with(args.membership_prepared)
            self.assertEqual(result['entry_count'], 3)

    def test_rejects_graph_change_during_metadata_only_intervention(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, candidate, parent, candidate_root, publication = self.fixture(Path(temporary))
            candidate['index_before']['files_sha256']['index_disk.index'] = 'f' * 64
            publication['output_files_sha256'][str(candidate_root / 'generation-00000000000000000001/index_disk.index')] = 'f' * 64
            args.candidate_build_log.write_text(json.dumps(publication))
            with self.assertRaisesRegex(ValueError, 'changed an immutable parent artifact'):
                analysis.entry_proof(args, candidate, parent, candidate_root)

    def test_crc64_matches_independent_standard_known_vector(self):
        self.assertEqual(analysis.crc64_ecma182(b'123456789'), 0x6c40df5f0b497347)


if __name__ == '__main__':
    unittest.main()
