#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Refuse mixed breadths, invalid metadata, accidental validation and false parity."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import analyze_entry_routing_ablation as module


class EntryRoutingGuards(unittest.TestCase):
    def rows(self):
        return [{'kind': 'query_detail', 'query_index': query, 'label': label,
                 'actual_l': breadth, 'l': 64, 'repeat': 0, 'phase': 'query', 'sequence': 0,
                 'base_searches': 1, 'delta_candidates': 0, 'exact_fallback': False,
                 'k': 1, 'ids': [ident], 'distances': [float(ident)], 'strict_id_hits': 1}
                for query, label, breadth, ident in [(0, 9, 64, 1), (1, 98, 80, 2)]]

    def validate(self, rows):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'details.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            return module.validate_details(path, [9, 98], [1, 2], 1, {9: 64, 98: 80},
                                           {9: [1, 2, 3], 98: [1, 2, 3]}, lambda query, ident: ident)

    def test_actual_frozen_breadth_and_unique_coverage_are_required(self):
        self.assertEqual(set(self.validate(self.rows())), {0, 1})
        wrong_l = self.rows()
        wrong_l[1]['actual_l'] = 64
        with self.assertRaisesRegex(ValueError, 'breadth'):
            self.validate(wrong_l)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.validate(self.rows()+self.rows()[:1])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.validate(self.rows()[:1])

    def test_filter_distance_and_hit_counter_are_checked_independently(self):
        for key, value, message in [('ids', [99], 'filter-invalid'),
                                    ('distances', [99.], 'exact distance'),
                                    ('strict_id_hits', 0, 'hit counter')]:
            rows = self.rows()
            rows[0][key] = value
            with self.assertRaisesRegex(ValueError, message):
                self.validate(rows)

    def test_ordered_control_parity_is_stronger_than_equal_recall(self):
        before = self.validate(self.rows())
        self.assertEqual(module.exact_parity(before, copy.deepcopy(before)), 2)
        after = copy.deepcopy(before)
        after[0]['ids'] = [3]
        with self.assertRaisesRegex(ValueError, 'one-entry V8 differs'):
            module.exact_parity(before, after)
        after = copy.deepcopy(before)
        after[0]['distances'] = [3.]
        with self.assertRaisesRegex(ValueError, 'one-entry V8 differs'):
            module.exact_parity(before, after)
        keyed_before = {(row['actual_l'], query): row for query, row in before.items()}
        keyed_after = copy.deepcopy(keyed_before)
        keyed_after[(80, 1)]['ids'] = [3]
        summary = module.compare_entries(keyed_before, keyed_after, [0, 1],
                                         lambda q: [64, 80][q], [1, 2], 1)
        self.assertEqual((summary['one_entry_hits'], summary['pool_hits']), (2, 1))
        self.assertNotIn('legacy_hits', summary)

    def metadata(self):
        before, after = {'9': [1], '98': [2]}, {'9': [1, 3], '98': [2, 3]}
        publication = {'manifest_payload_preserved_verbatim': True,
                       'source_manifest_before': {'historical_build': 'v7'},
                       'source_manifest_after': {'historical_build': 'v7'},
                       'manifest_payload_crc64_ecma182': {'source': 10, 'installed': 10},
                       'verified_bindings': {key: True for key in ['point_count_equal', 'point_labels_equal',
                           'postings_equal', 'stable_ids_equal', 'global_centroid_first']},
                       'entry_policy': {'spatial_clusters': 4, 'maximum_entries': 5,
                                        'maximum_fit_members': 4096, 'lloyd_iterations': 8,
                                        'sample_seed_base': 0x6c6162656c5f7633, 'global_centroid_first': True},
                       'entry_points': after}
        provenance = {'entry_selection': {'old_entry_points': before, 'new_entry_points': after}}
        return publication, provenance

    def test_entry_pool_requires_actual_members_dedup_and_original_centroid(self):
        publication, provenance = self.metadata()
        module.verify_entry_metadata(publication, provenance, {9: [1, 3], 98: [2, 3]})
        for entries, message in [([1, 99], 'eligible'), ([1, 1], 'invalid entry'),
                                 ([3, 1], 'centroid')]:
            publication, provenance = self.metadata()
            publication['entry_points']['9'] = entries
            provenance['entry_selection']['new_entry_points'] = publication['entry_points']
            with self.assertRaisesRegex(ValueError, message):
                module.verify_entry_metadata(publication, provenance, {9: [1, 3], 98: [2, 3]})

    def test_held_out_validation_manifest_is_rejected_before_assets(self):
        module.require_calibration_manifest({'split': 'secondary_calibration', 'query_count': 1436})
        for split, count in [('secondary_validation', 4316), ('secondary_calibration', 4316)]:
            with self.assertRaisesRegex(ValueError, 'only the exposed'):
                module.require_calibration_manifest({'split': split, 'query_count': count})

    def test_query_namespaces_ids_and_embeddings_cannot_be_silently_mixed(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary)
            public, private = bytes([7, 9]), bytes([8, 10])
            files = [('query.public.100K.u8bin', public), ('query.private.2727415019.100K.u8bin', private)]
            sources = []
            for name, data in files:
                path = prepared/name
                path.write_bytes(struct.pack('<II', 1, 2)+data)
                sources.append({'file': name, 'path': str(path), 'sha256': module.sha(path)})
            (prepared/'source_namespace.u8bin').write_bytes(struct.pack('<II', 2, 1)+bytes([0, 1]))
            (prepared/'query_ids.u32bin').write_bytes(struct.pack('<IIII', 2, 1, 0, 0))
            (prepared/'source_pool_indices.u32bin').write_bytes(struct.pack('<IIII', 2, 1, 4, 5))
            records = [{'query_index': q, 'atomic_label': label, 'source_query_id': 0,
                        'source_query_namespace': namespace, 'source_pool_query_index': q+4,
                        'original_and_tags': [label, 1000], 'embedding_sha256': hashlib.sha256(data).hexdigest()}
                       for q, label, namespace, data in [(0, 9, 'public_100K', public),
                           (1, 98, 'official_private_2727415019_100K', private)]]
            path = prepared/'source-query-rows.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in records))
            query = struct.pack('<II', 2, 2)+public+private
            value = module.verify_query_provenance(prepared, {'sources': sources}, [9, 98], query)
            self.assertEqual(value['verified_query_ids'], 2)
            # Identical numerical IDs in the two namespaces are legitimate, but their vectors are distinct.
            swapped = struct.pack('<II', 2, 2)+private+public
            with self.assertRaisesRegex(ValueError, 'embedding differs'):
                module.verify_query_provenance(prepared, {'sources': sources}, [9, 98], swapped)


if __name__ == '__main__':
    unittest.main()
