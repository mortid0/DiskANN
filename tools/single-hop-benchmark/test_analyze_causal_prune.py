#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Guard against mixing breadths, accepting duplicated dumps or invalid results."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('analyze_causal_prune', Path(__file__).with_name('analyze_causal_prune.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PairedBreadths(unittest.TestCase):
    def rows(self):
        return [{'query': query, 'label': label, 'l': breadth, 'repeat': 0,
                 'mode': 'single', 'threads': 8, 'returned': 1,
                 'ids': [ident], 'distances': [float(ident)]}
                for breadth in [64, 80] for query, label, ident in [(0, 9, 1), (1, 98, 2)]]

    def validate(self, rows):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'dump.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            return module.validate_dump(path, [9, 98], 1, [64, 80],
                                        {9: [1, 2, 3], 98: [1, 2, 3]}, lambda query, ident: ident)

    def test_two_records_for_a_query_are_valid_only_at_distinct_breadths(self):
        result = self.validate(self.rows())
        self.assertEqual(set(result), {(64, 0), (64, 1), (80, 0), (80, 1)})
        with self.assertRaisesRegex(ValueError, 'duplicate query'):
            self.validate(self.rows() + self.rows()[:1])
        with self.assertRaisesRegex(ValueError, 'incomplete query/L'):
            self.validate(self.rows()[:-1])

    def test_distances_and_filter_membership_are_independently_checked(self):
        bad_distance = self.rows()
        bad_distance[0]['distances'] = [99.]
        with self.assertRaisesRegex(ValueError, 'integer squared L2'):
            self.validate(bad_distance)
        bad_filter = self.rows()
        bad_filter[0]['ids'] = [99]
        with self.assertRaisesRegex(ValueError, 'filter-invalid'):
            self.validate(bad_filter)

    def test_policy_comparison_selects_the_correct_query_breadth(self):
        old = self.validate(self.rows())
        new = copy.deepcopy(old)
        new[(64, 0)]['ids'] = [3]
        new[(80, 1)]['ids'] = [3]
        # Results at the unused breadths must not conceal the two policy losses.
        summary = module.compare(old, new, [0, 1], lambda query: [64, 80][query], [1, 2], 1)
        self.assertEqual(summary['legacy_hits'], 2)
        self.assertEqual(summary['preserved_hits'], 0)
        self.assertEqual(summary['queries_worsened'], 2)
        self.assertEqual(summary['hit_difference'], -2)


if __name__ == '__main__':
    unittest.main()
