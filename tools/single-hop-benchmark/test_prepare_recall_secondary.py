#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Compare blocked exhaustive truth against independent scalar fixtures."""

import unittest

import numpy as np

import prepare_recall_secondary as secondary


class SecondaryPreparationTests(unittest.TestCase):
    def test_exact_truth_matches_scalar_scan_including_cross_block_ties(self):
        rng = np.random.default_rng(492)
        base = rng.integers(0, 256, (73, 17), dtype=np.uint8)
        base[0:7] = 11
        base[15:22] = 11
        base[40:47] = 11
        queries = np.vstack((np.full((1, 17), 11, dtype=np.uint8), rng.integers(0, 256, (6, 17), dtype=np.uint8)))
        labels = np.array([9, 9, 24, 24, 9, 24, 9], dtype=np.uint32)
        postings = {9: np.arange(73, dtype=np.uint32), 24: np.arange(0, 73, 2, dtype=np.uint32)}
        ids, distances, comparisons = secondary.exact_truth(base, queries, labels, postings, k=5, point_block=9, query_block=2)
        for row, (query, label) in enumerate(zip(queries, labels)):
            expected = sorted((sum((int(a) - int(b)) ** 2 for a, b in zip(query, base[point])), int(point))
                              for point in postings[int(label)])[:5]
            self.assertEqual(ids[row].tolist(), [item[1] for item in expected])
            self.assertEqual(distances[row].tolist(), [item[0] for item in expected])
        self.assertEqual(ids[0].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(comparisons, {'9': 4 * 73, '24': 3 * 37})
        self.assertEqual(secondary.scalar_verify(base, queries, labels, ids, distances, postings), 35)

    def test_maximum_uint8_squared_distance_is_exact(self):
        base = np.vstack((np.zeros((11, 192), dtype=np.uint8), np.full((11, 192), 255, dtype=np.uint8)))
        queries = np.array([np.zeros(192, dtype=np.uint8), np.full(192, 255, dtype=np.uint8)])
        labels = np.array([9, 24])
        postings = {9: np.arange(11, 22), 24: np.arange(11)}
        ids, distances, _ = secondary.exact_truth(base, queries, labels, postings, k=10, point_block=10, query_block=1)
        self.assertEqual(ids[0].tolist(), list(range(11, 21)))
        self.assertEqual(ids[1].tolist(), list(range(10)))
        self.assertTrue(np.all(distances == 192 * 255 * 255))

    def test_selection_is_unique_across_namespaces_and_unchanged_by_source_order(self):
        queries1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.uint8)
        queries2 = np.array([[1, 2], [9, 10]], dtype=np.uint8)
        sources = [('public_100K', queries1, np.array([0, 2, 4, 5, 7]), np.array([9, 24, 9, 98, 9, 9, 89])),
                   ('official_private_2727415019_100K', queries2, np.array([0, 2, 4]), np.array([9, 24, 24, 98]))]
        rows, excluded, duplicates, _ = secondary.select_candidates(sources, {bytes([3, 4])}, {9, 24, 89, 98})
        other, *_ = secondary.select_candidates(list(reversed(sources)), {bytes([3, 4])}, {9, 24, 89, 98})
        self.assertEqual(rows, other)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(set(row['embedding_sha256'] for row in rows)), 3)
        for row in rows:
            self.assertIn(row['atomic_label'], row['original_and_tags'])
            self.assertEqual(len(row['original_and_tags']), 2)


if __name__ == '__main__':
    unittest.main()
