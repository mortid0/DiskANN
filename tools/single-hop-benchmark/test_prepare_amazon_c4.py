#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Guard source identities, family exclusion, exact predicates, and distance ties."""

import csv
import hashlib
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SPEC = importlib.util.spec_from_file_location("amazon_c4", Path(__file__).with_name("prepare_amazon_c4.py"))
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def test_download_rejects_bad_lfs_digest_and_git_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object"
            path.write_bytes(b"abc")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                prepare.checked_download("unused", path, (3, "0" * 64, None))
            with self.assertRaisesRegex(ValueError, "git-blob"):
                prepare.checked_download("unused", path, (3, None, "0" * 40))
            digest = hashlib.sha256(b"abc").hexdigest()
            blob = hashlib.sha1(b"blob 3\0abc").hexdigest()
            self.assertEqual(prepare.checked_download("unused", path, (3, digest, blob))["sha256"], digest)

    def test_canonical_query_equivalence(self):
        self.assertEqual(prepare.canonical_text("  ＦＯＯ\nBar "), "foo bar")

    def test_connected_families_cannot_cross_holdout(self):
        queries = [{"qid": i, "positive_item_id": f"item{i}", "user_hash": f"user{i}", "text_hash": f"text{i}"} for i in range(40)]
        queries[1]["positive_item_id"] = queries[0]["positive_item_id"]
        queries[2]["user_hash"] = queries[1]["user_hash"]
        queries[3]["text_hash"] = queries[2]["text_hash"]
        vectors = np.arange(120, dtype="<f4").reshape(40, 3)
        vectors[4] = vectors[3]
        split, groups, duplicates = prepare.grouped_split(queries, vectors)
        family = next(group for group in groups if 0 in group["source_query_rows"])
        self.assertEqual(family["source_query_rows"], [0, 1, 2, 3, 4])
        self.assertEqual(family["retained_query_rows"], [0, 1, 2])
        self.assertEqual({x["row"] for x in duplicates}, {3, 4})
        self.assertTrue(set(split["calibration"]).isdisjoint(split["validation"]))
        self.assertEqual(len(split["calibration"]) + len(split["validation"]), 38)
        self.assertEqual(prepare.grouped_split(queries, vectors), (split, groups, duplicates))

    def test_matrix_rejects_truncation_and_extra_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.fbin"
            prepare.write_matrix(path, [[1, 2], [3, 4]], "<f4")
            self.assertEqual(prepare.matrix(path).tolist(), [[1, 2], [3, 4]])
            with path.open("ab") as target:
                target.write(b"x")
            with self.assertRaisesRegex(ValueError, "Malformed"):
                prepare.matrix(path)

    def test_exhaustive_truth_matches_direct_distances_with_ties(self):
        rng = np.random.default_rng(21)
        base = rng.normal(size=(41, 7)).astype("<f4")
        base[2] = base[1]
        base[3] = base[1]
        queries = np.asarray([base[1], base[17], rng.normal(size=7)], dtype="<f4")
        labels = np.array([1, 2, 1], dtype="<u4")
        postings = {1: np.array([1, 2, 3, *range(20, 41)], dtype="<u4"), 2: np.arange(20, dtype="<u4")}
        ids, distances, pairs = prepare.exact_truth(base, queries, labels, postings, 3, query_block=1, base_block=4)
        for i, label in enumerate(labels):
            members = postings[int(label)]
            difference = base[members].astype("<f8") - queries[i].astype("<f8")
            exact = np.einsum("ij,ij->i", difference, difference)
            order = np.lexsort((members, exact))[:3]
            np.testing.assert_array_equal(ids[i], members[order])
            np.testing.assert_array_equal(distances[i], exact[order])
        self.assertEqual(ids[0].tolist(), [1, 2, 3])
        self.assertEqual(pairs, 2 * len(postings[1]) + len(postings[2]))

    def test_empty_product_policy_remaps_and_records_query_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            products = [{"item_id": "a", "category": "Books", "metadata": "Good book"},
                        {"item_id": "b", "category": "Books", "metadata": " \n"},
                        {"item_id": "c", "category": "Care", "metadata": "Good soap"}]
            raw = root / "raw" / "sampled_item_metadata_1M.jsonl"
            raw.write_text("".join(json.dumps(row) + "\n" for row in products))
            queries = root / "raw" / "test.csv"
            with queries.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=["qid", "query", "item_id", "user_id"])
                writer.writeheader()
                writer.writerows({"qid": i, "query": f"find {i}", "item_id": key, "user_id": "u" + key} for i, key in enumerate("abc"))
            (root / "download-identity.json").write_text(json.dumps({"raw": {name: prepare.record(root / "raw" / name) for name in prepare.RAW_FILES}}))
            (root / "metadata").mkdir()  # Previous failed validation left only an empty directory.
            with patch.object(prepare, "SOURCE_BASE_COUNT", 3), patch.object(prepare, "SOURCE_QUERY_COUNT", 3):
                prepare.metadata(root)
            meta = json.loads((root / "metadata" / "metadata.json").read_text())
            self.assertEqual((meta["base_count"], meta["retained_source_query_count"]), (2, 2))
            corpus = list(prepare.json_rows(root / "metadata" / "products.jsonl"))
            self.assertEqual([(p["id"], p["source_id"], p["item_id"]) for p in corpus], [(0, 0, "a"), (1, 2, "c")])
            kept = list(prepare.json_rows(root / "metadata" / "queries.jsonl"))
            self.assertEqual([(q["qid"], q["positive_base_id"]) for q in kept], [(0, 0), (2, 1)])
            self.assertEqual(json.loads((root / "metadata" / "excluded-queries.json").read_text())[0]["qid"], 1)
            self.assertFalse(any(root.glob("metadata.staging-*")))

    def test_independent_truth_audit_rejects_wrong_neighbor(self):
        base = np.arange(24, dtype="<f4").reshape(8, 3)
        q = base[[0, 2]]
        labels = np.array([1, 1])
        postings = {1: np.arange(8, dtype="<u4")}
        ids, distances, _ = prepare.exact_truth(base, q, labels, postings, 3, base_block=4)
        audit = prepare.audit_truth_direct(base, q, labels, postings, ids, distances, per_label=2)
        self.assertTrue(audit["all_checked_ids_and_distances_identical"])
        ids[0, 0] = 7
        with self.assertRaisesRegex(ValueError, "disagreement"):
            prepare.audit_truth_direct(base, q, labels, postings, ids, distances, per_label=2)

    def test_truth_rejects_undersized_predicate(self):
        with self.assertRaisesRegex(ValueError, "fewer than k"):
            prepare.exact_truth(np.zeros((2, 3)), np.zeros((1, 3)), np.array([1]), {1: np.array([0])}, 2)

    def test_screen_refines_nearly_identical_float_vectors(self):
        base = np.ones((12, 384), dtype="<f4")
        for i in range(12):
            base[i, 0] = np.nextafter(np.float32(1), np.float32(2), dtype=np.float32) if i >= 5 else 1
        query = np.ones((1, 384), dtype="<f4")
        ids, d, _ = prepare.exact_truth(base, query, np.array([1]), {1: np.arange(12, dtype="<u4")}, 8, base_block=4)
        self.assertEqual(ids.tolist(), [list(range(8))])
        self.assertEqual(d[0, :5].tolist(), [0] * 5)
        self.assertTrue(np.all(d[0, 5:] > 0))


if __name__ == "__main__":
    unittest.main()
