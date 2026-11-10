#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Independent membership boundary and freshness-audit failure checks."""

import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("audit_books_predicates", Path(__file__).with_name("audit_books_predicates.py"))
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AuditBooksTests(unittest.TestCase):
    def test_publisher_boundaries_rating_and_vote_semantics(self):
        parameters = [1500000000, 100, 10000000, 5, 19]
        inside_first = [1499395200, 70, -97321744, 4, 1]
        inside_last = [1500000000, 130, 117321744, 5, 2147483647]
        records = [inside_first, inside_last]
        for column in range(5):
            low = inside_first.copy()
            low[column] -= 1
            records.append(low)
            if column != 4:  # The original vote predicate has no upper bound.
                high = inside_last.copy()
                high[column] += 1
                records.append(high)
        np.testing.assert_array_equal(AUDIT.direct_members(np.asarray(records, dtype=np.int64), parameters), [0, 1])
        clipped_asin = np.array([[1500000000, 100, 2147483647, 4, 9]], dtype="<i4")
        np.testing.assert_array_equal(AUDIT.direct_members(clipped_asin, [1500000000, 100, 2147483647, 4, 9]), [0])
        with self.assertRaisesRegex(ValueError, "rating"):
            AUDIT.direct_members(clipped_asin, [0, 0, 0, 7, 0])

    def test_original_sample_stratification_is_order_independent(self):
        entries = [{"label": row + 1, "base_count": row // 2} for row in range(128)]
        first = AUDIT.stratified_original(entries, 64)
        self.assertEqual(first, AUDIT.stratified_original(list(reversed(entries)), 64))
        self.assertEqual(len({row["label"] for row in first}), 64)
        self.assertEqual({row["base_count"] for row in first}, set(range(64)))

    def test_posting_hash_and_full_membership_corruption_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "posting.u32bin"
            path.write_bytes(struct.pack("<II3I", 3, 1, 0, 4, 9))
            identity = AUDIT.record(path)
            AUDIT.compare_posting(path, identity, np.array([0, 4, 9], dtype="<u4"), 3)
            with self.assertRaisesRegex(ValueError, "membership mismatch"):
                AUDIT.compare_posting(path, identity, np.array([0, 5, 9], dtype="<u4"), 3)
            path.write_bytes(struct.pack("<II3I", 3, 1, 0, 5, 9))
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                AUDIT.compare_posting(path, identity, np.array([0, 5, 9], dtype="<u4"), 3)

    def test_query_family_embedding_and_id_leakage_fail_separately(self):
        row = {"query_id": 1, "embedding": "a", "group": 10, "asin": 100}
        other = {"query_id": 2, "embedding": "b", "group": 20, "asin": 200}
        AUDIT.verify_group_separation({"calibration": [row], "validation": [other]})
        for field in row:
            leaked = {**other, field: row[field]}
            with self.assertRaisesRegex(ValueError, field):
                AUDIT.verify_group_separation({"calibration": [row], "validation": [leaked]})


if __name__ == "__main__":
    unittest.main()
