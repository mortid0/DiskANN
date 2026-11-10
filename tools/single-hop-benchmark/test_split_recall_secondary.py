#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Verify label-stratified secondary splitting and sealed-artifact checks."""

from collections import Counter
from pathlib import Path
import tempfile
import unittest

import numpy as np

import split_recall_secondary as splitter


class SecondarySplitTests(unittest.TestCase):
    def test_split_partitions_each_label_with_exact_floor_quarter_calibration(self):
        labels = np.array([9] * 23 + [24] * 11 + [89] * 3 + [98] * 8 + [2495] * 4)
        before = labels.copy()
        split = splitter.split_indices(labels)
        self.assertTrue(np.array_equal(labels, before))
        calibration, validation = split['calibration'], split['validation']
        self.assertEqual(len(set(calibration)), len(calibration))
        self.assertEqual(len(set(validation)), len(validation))
        self.assertTrue(set(calibration).isdisjoint(validation))
        self.assertEqual(set(calibration) | set(validation), set(range(len(labels))))
        self.assertEqual(Counter(map(int, labels[calibration])), {9: 5, 24: 2, 98: 2, 2495: 1})
        self.assertEqual(Counter(map(int, labels[validation])), {9: 18, 24: 9, 89: 3, 98: 6, 2495: 3})
        self.assertEqual(splitter.split_indices(labels), split)

    def test_sparse_labels_remain_in_validation_and_boundary_allocation_is_correct(self):
        labels = [9, 24, 24, 89, 89, 89, 98, 98, 98, 98]
        split = splitter.split_indices(labels)
        self.assertEqual(Counter(labels[index] for index in split['calibration']), {98: 1})
        self.assertEqual(Counter(labels[index] for index in split['validation']), {9: 1, 24: 2, 89: 3, 98: 3})

    def test_sealed_artifact_rejects_same_size_content_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'query.u8bin'
            path.write_bytes(b'query bytes')
            expected = {'query.u8bin': {'bytes': path.stat().st_size, 'sha256': splitter.sha256(path)}}
            splitter.check_artifacts(root, expected)
            path.write_bytes(b'wrong bytes')
            with self.assertRaisesRegex(ValueError, 'Artifact identity changed'):
                splitter.check_artifacts(root, expected)


if __name__ == '__main__':
    unittest.main()
