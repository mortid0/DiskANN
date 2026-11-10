#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Predicate boundaries, leakage guards, and exhaustive Books truth checks."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("prepare_bigvector_books", Path(__file__).with_name("prepare_bigvector_books.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class BooksPreparationTests(unittest.TestCase):
    def test_exact_inclusive_bounds_and_integer_truncation(self):
        p = prepare.predicate([1500000000, 100, 10000000, 5, 19])
        self.assertEqual(p["low_inclusive"], [1499395200, 70, -97321744, 4, 1])
        self.assertEqual(p["high_inclusive"], [1500000000, 130, 117321744, 5, 2147483647])
        low, high = np.array(p["low_inclusive"]), np.array(p["high_inclusive"])
        rows = [low.copy(), high.copy()]
        for column in range(5):
            bad_low, bad_high = low.copy(), high.copy()
            bad_low[column] -= 1
            bad_high[column] += 1
            rows.extend((bad_low, bad_high))
        attributes = np.array(rows, dtype=np.int64)
        self.assertEqual(prepare.PredicateIndex(attributes).members(p).tolist(), [0, 1])
        for rating, bounds in {1: (1, 2), 2: (1, 2), 3: (3, 4), 4: (3, 5), 5: (4, 5)}.items():
            q = prepare.predicate([0, 0, 0, rating, 0])
            self.assertEqual((q["low_inclusive"][3], q["high_inclusive"][3]), bounds)
        with self.assertRaisesRegex(ValueError, "rating"):
            prepare.predicate([0, 0, 0, 6, 0])

    def test_index_matches_full_mask_for_random_predicates(self):
        rng = np.random.default_rng(42)
        labels = np.column_stack((rng.integers(1400000000, 1500000000, 5000),
                                  rng.integers(1, 512, 5000), rng.integers(1, 2147483647, 5000),
                                  rng.integers(1, 6, 5000), rng.integers(0, 998, 5000)))
        index = prepare.PredicateIndex(labels)
        for row in labels[:50]:
            p = prepare.predicate(row)
            expected = np.flatnonzero(np.all((labels >= p["low_inclusive"]) & (labels <= p["high_inclusive"]), axis=1))
            self.assertTrue(np.array_equal(index.members(p), expected))

    def test_schema_python_is_never_executed_and_mutations_are_rejected(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py not installed")
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "fixture.hdf5"
            with h5py.File(p, "w") as target:
                target.attrs["type"] = "filter-ann"
                target.attrs["distance"] = "euclidean"
                target.attrs["label_names"] = prepare.NAMES
                target.attrs["label_types"] = ["int32"] * 5
                target.attrs["filter_expr_func"] = prepare.OFFICIAL_EXPRESSION + "\n# harmless comment\n"
                for name, shape, dtype in (("train_vec", (11, 3), "f4"), ("test_vec", (2, 3), "f4"),
                                          ("train_label", (11, 5), "i4"), ("test_label", (2, 5), "i4"),
                                          ("neighbors", (2, 100), "i4"), ("distances", (2, 100), "f4")):
                    target.create_dataset(name, shape, dtype=dtype)
                prepare.validate_schema(target, strict=False)
                with self.assertRaisesRegex(ValueError, "full AmazonBooks release"):
                    prepare.validate_schema(target)
                target.attrs["filter_expr_func"] = prepare.OFFICIAL_EXPRESSION + "\nraise RuntimeError('should never execute')\n"
                with self.assertRaisesRegex(ValueError, "expression changed"):
                    prepare.validate_schema(target, strict=False)

    def test_family_and_duplicate_components_do_not_cross_splits(self):
        queries = np.array([[1, 2], [2, 3], [2, 3], [4, 5], [6, 7], [8, 9]], dtype="<f4")
        attributes = np.array([[0, 10, asin, 3, 0] for asin in (100, 100, 200, 200, 300, 400)], dtype="<i4")
        groups, split, duplicates = prepare.grouped_split(queries, attributes)
        self.assertEqual(groups[:4], [0, 0, 0, 0])
        self.assertEqual(len(set(split[:4])), 1)
        self.assertEqual([(x["query_id"], x["retained_query_id"]) for x in duplicates], [(2, 1)])
        self.assertEqual(prepare.grouped_split(queries, attributes), (groups, split, duplicates))

    def test_bank_uses_base_only_deterministic_distinct_predicates(self):
        rng = np.random.default_rng(1)
        labels = np.column_stack((rng.integers(1500000000, 1500200000, 128),
                                  rng.integers(1, 20, 128), rng.integers(10000000, 10001000, 128),
                                  rng.integers(1, 6, 128), rng.integers(0, 20, 128)))
        first = prepare.fixed_bank(labels, count=4, candidates=64, minimum=10)
        second = prepare.fixed_bank(labels, count=4, candidates=64, minimum=10)
        self.assertEqual([x["predicate"] for x in first], [x["predicate"] for x in second])
        self.assertEqual(len({tuple(x["predicate"]["parameters"]) for x in first}), 4)
        for entry in first:
            self.assertEqual(entry["predicate"], prepare.predicate(labels[entry["source_base_row"]]))
            np.testing.assert_array_equal(entry["members"], prepare.PredicateIndex(labels).members(entry["predicate"]))

    def test_sparse_point_labels_preserve_exact_union_and_sentinel(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "labels.txt"
            postings = {9: np.array([0, 3, 7]), 2: np.array([1, 3, 5]), 20: np.array([3])}
            stats = prepare.write_point_labels(p, postings, 9, chunk=3)
            self.assertEqual(p.read_text().splitlines(), ["9", "2", "200386", "2,9,20", "200386", "2", "200386", "9", "200386"])
            self.assertEqual(stats["memberships"], 7)
            self.assertEqual(stats["max_row_memberships"], 3)
            self.assertEqual(stats["unqueried_sentinel_rows"], 4)

    def test_original_seal_preserves_predicates_excludes_small_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exported = root / "export"
            exported.mkdir()
            base = np.arange(30, dtype="<f4").reshape(15, 2)
            queries = np.array([[1, 2], [1, 2], [7, 8], [9, 10]], dtype="<f4")
            attrs = np.array([[1500000000, 100, 10000000, 3, 0]] * 15, dtype="<i4")
            qattrs = np.array([attrs[0], attrs[0], [1400000000, 100, 10000000, 3, 0], attrs[0]], dtype="<i4")
            for name, values, dtype in (("base.fbin", base, "<f4"), ("query.fbin", queries, "<f4"),
                                         ("base_attributes.i32bin", attrs, "<i4"), ("query_attributes.i32bin", qattrs, "<i4")):
                prepare.write_matrix(exported / name, values, dtype)
            prepare.write_json(exported / "export-identity.json", {"artifacts": {name: prepare.record(exported / name) for name in ("base.fbin", "query.fbin", "base_attributes.i32bin", "query_attributes.i32bin")}})
            prepare.seal(root, "original")
            sealed = root / "sealed-original"
            proof = json.loads((sealed / "split-commitment.json").read_text())
            self.assertEqual([x["label"] for x in proof["labels"]], [1, 4])
            self.assertEqual(json.loads((sealed / "metadata-exclusions.json").read_text()), [{"query_id": 2, "reason": "fewer_than_k_eligible_points", "eligible_points": 0}])
            self.assertEqual(sum(x["query_count"] for x in proof["outputs"].values()), 2)
            self.assertFalse(proof["fresh_validation_queries_searched"])
            self.assertFalse(proof["source_truth_used_to_select_queries"])
            self.assertIn("not arbitrary range", proof["filter_semantics"])
            for label in (1, 4):
                self.assertEqual(prepare.matrix(sealed / f"label_{label}.u32bin", "<u4").ravel().tolist(), list(range(15)))
            with (exported / "query.fbin").open("ab") as target:
                target.write(b"x")
            with self.assertRaisesRegex(ValueError, "Exported source changed"):
                prepare.seal(root, "original")

    def test_shared_exhaustive_oracle_matches_direct_full_filter(self):
        from prepare_amazon_c4 import exact_truth
        rng = np.random.default_rng(101)
        base = rng.standard_normal((37, 7)).astype("<f4")
        base[12] = base[2]
        queries = np.vstack((base[2], rng.standard_normal(7))).astype("<f4")
        postings = {1: np.arange(0, 37, 2, dtype="<u4"), 2: np.arange(1, 37, 2, dtype="<u4")}
        labels = np.array([1, 2])
        ids, distances, pairs = exact_truth(base, queries, labels, postings, 4, query_block=1, base_block=5)
        for row, label in enumerate(labels):
            members = postings[label]
            difference = base[members].astype(np.float64) - queries[row].astype(np.float64)
            truth = np.einsum("ij,ij->i", difference, difference)
            order = np.lexsort((members, truth))[:4]
            np.testing.assert_array_equal(ids[row], members[order])
            np.testing.assert_array_equal(distances[row], truth[order])
        self.assertEqual(pairs, 37)


if __name__ == "__main__":
    unittest.main()
