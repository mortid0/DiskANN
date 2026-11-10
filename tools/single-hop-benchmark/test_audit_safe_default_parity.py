# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Guards for exact exposed output parity, without any quality threshold."""
import copy
import unittest

from audit_safe_default_parity import ordered_result_parity


class ParityGuards(unittest.TestCase):
    def setUp(self):
        self.row = {'query_index': 0, 'label': 9, 'actual_l': 64, 'l': 64, 'k': 10,
                    'repeat': 0, 'phase': 'query', 'sequence': 0, 'kind': 'query_detail',
                    'base_searches': 1, 'delta_candidates': 0, 'exact_fallback': False,
                    'strict_id_hits': 5, 'ids': list(range(10)), 'distances': list(range(10))}

    def check(self, current):
        return ordered_result_parity([current], [self.row], [9], {9: 64})

    def test_exact_parity_ignores_non_authoritative_latency(self):
        current = dict(self.row, latency_us=999999)
        self.assertTrue(self.check(current)['returned_distance_bits_equal'])

    def test_reordered_ids_are_rejected(self):
        current = copy.deepcopy(self.row)
        current['ids'][0], current['ids'][1] = current['ids'][1], current['ids'][0]
        with self.assertRaisesRegex(ValueError, 'Ordered output'):
            self.check(current)

    def test_exact_distance_change_is_rejected(self):
        current = copy.deepcopy(self.row)
        current['distances'][0] = 0.00000001
        with self.assertRaisesRegex(ValueError, 'distance bits'):
            self.check(current)

    def test_query_label_and_actual_l_are_checked_against_policy(self):
        for field, value in [('query_index', 1), ('label', 24), ('actual_l', 68)]:
            with self.subTest(field=field):
                current = dict(self.row, **{field: value})
                with self.assertRaises(ValueError):
                    self.check(current)


if __name__ == '__main__':
    unittest.main()
