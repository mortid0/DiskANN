#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Checks for fresh-query exclusions, splitting, and independent truth validation."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("prepare_recall_validation", Path(__file__).with_name("prepare_recall_validation.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def test_excludes_old_embeddings_across_labels_and_collapses_new_duplicates(self):
        queries = np.array([[1, 2], [1, 2], [3, 4], [3, 4], [5, 6], [7, 8]], dtype="u1")
        ptr = np.array([0, 1, 2, 3, 4, 6, 7])
        labels = np.array([9, 24, 9, 24, 9, 24, 89])
        selected, excluded, duplicates, available = prepare.select_fresh(
            queries, ptr, labels, {9, 24}, {bytes([1, 2])})
        self.assertEqual(selected, [(2, 9)])
        self.assertEqual(excluded, [(0, 9), (1, 24)])
        self.assertEqual(duplicates, [(3, 24)])
        self.assertEqual(dict(available), {9: 2, 24: 2})

    def test_split_is_order_independent_and_floor_stratified(self):
        rows = [(row, 9) for row in range(19)] + [(row, 24) for row in range(100, 103)]
        calibration, validation = prepare.split_queries(rows)
        self.assertEqual(prepare.split_queries(list(reversed(rows))), (calibration, validation))
        self.assertEqual(len(calibration), 4)
        self.assertEqual(sum(label == 9 for _, label in validation), 15)
        self.assertEqual(sum(label == 24 for _, label in validation), 3)
        self.assertTrue(set(calibration).isdisjoint(validation))
        self.assertEqual(set(calibration) | set(validation), set(rows))

    def test_rejects_truth_distance_and_predicate_corruption(self):
        base = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype="u1")
        queries = np.array([[1, 0], [2, 1]], dtype="u1")
        ids = np.array([[0, 1], [1, 2]], dtype="<i4")
        distances = np.array([[1, 1], [1, 1]], dtype="<f4")
        postings = {9: np.array([0, 1, 2], dtype="<u4"), 24: np.array([1, 2, 3], dtype="<u4")}
        self.assertEqual(prepare.verify_truth(base, queries, [(0, 9), (1, 24)], ids, distances, postings, 2), 4)
        bad_distances = distances.copy()
        bad_distances[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "squared L2"):
            prepare.verify_truth(base, queries, [(0, 9), (1, 24)], ids, bad_distances, postings, 2)
        bad_postings = {9: np.array([1, 2, 3], dtype="<u4"), 24: postings[24]}
        with self.assertRaisesRegex(ValueError, "atomic predicate"):
            prepare.verify_truth(base, queries, [(0, 9), (1, 24)], ids, distances, bad_postings, 2)

    def test_malformed_matrix_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.u8bin"
            prepare.write_matrix(path, [[1, 2], [3, 4]], "u1")
            self.assertEqual(prepare.matrix(path, "u1").tolist(), [[1, 2], [3, 4]])
            with path.open("ab") as output:
                output.write(b"x")
            with self.assertRaisesRegex(ValueError, "Malformed matrix"):
                prepare.matrix(path, "u1")


if __name__ == "__main__":
    unittest.main()
