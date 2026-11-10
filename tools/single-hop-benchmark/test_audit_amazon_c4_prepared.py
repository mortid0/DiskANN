#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Independent membership, finite-distance, and query-family audit guards."""

import importlib.util
import json
import struct
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("audit_c4", Path(__file__).with_name("audit_amazon_c4_prepared.py"))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class PreparedAuditTests(unittest.TestCase):
    def test_memberships_cover_corpus_exactly_and_reject_wrong_posting(self):
        labels = np.array([1] * 10 + [2] * 10)
        records = [{"label": 1, "base_count": 10}, {"label": 2, "base_count": 10}]
        postings = {1: np.arange(10), 2: np.arange(10, 20)}
        audit.verify_postings(labels, records, postings)
        postings[1] = np.array([*range(9), 10])
        with self.assertRaisesRegex(ValueError, "differs"):
            audit.verify_postings(labels, records, postings)

    def test_truth_pair_and_tie_order_corruption_rejected(self):
        base = np.zeros((20, 3), dtype="<f4")
        base[10:] = 1
        queries = np.zeros((1, 3), dtype="<f4")
        ids = np.arange(10, dtype="<i4")[None, :]
        labels = np.array([1])
        postings = {1: np.arange(10)}
        d64 = np.zeros((1, 10), dtype="<f8")
        d32 = d64.astype("<f4")
        self.assertEqual(audit.verify_truth(base, queries, labels, postings, ids, d32, d64), 10)
        ids[0, :2] = [1, 0]
        with self.assertRaisesRegex(ValueError, "not sorted"):
            audit.verify_truth(base, queries, labels, postings, ids, d32, d64)
        ids[0, :2] = [0, 1]
        d32[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "Rounded"):
            audit.verify_truth(base, queries, labels, postings, ids, d32, d64)
        d32[0, 0] = 0
        d64[0, 0] = 1e-4
        d32[0, 0] = d64[0, 0]
        with self.assertRaisesRegex(ValueError, "coordinate"):
            audit.verify_truth(base, queries, labels, postings, ids, d32, d64)

    def full_fixture(self, root, shared_user=False):
        def write_json(path, value):
            path.write_text(json.dumps(value))
        def write_matrix(path, values, dtype):
            values = np.asarray(values, dtype=dtype)
            if values.ndim == 1:
                values = values[:, None]
            path.write_bytes(struct.pack("<II", *values.shape) + values.tobytes())
        for name in ("metadata", "embedding", "model", "sealed"):
            (root / name).mkdir()
        base = np.zeros((20, 384), dtype="<f4")
        base[10:] = 1
        write_matrix(root / "embedding/base.fbin", base, "<f4")
        labels_path = root / "metadata/labels.txt"
        labels_path.write_text("1\n" * 10 + "2\n" * 10)
        metadata_path = root / "metadata/metadata.json"
        write_json(metadata_path, {"retained_source_query_count": 2, "artifacts": {"labels.txt": audit.record(labels_path)}})
        configuration = root / "embedding/configuration.json"
        write_json(configuration, {"metadata": audit.record(metadata_path), "model_files": {}, "source": audit.record(Path(__file__))})
        embedding = root / "embedding/embedding-identity.json"
        write_json(embedding, {"configuration": audit.record(configuration), "outputs": {"base": audit.record(root / "embedding/base.fbin")}})
        sealed = root / "sealed"
        (sealed / "base.fbin").symlink_to(root / "embedding/base.fbin")
        (sealed / "labels.txt").symlink_to(labels_path)
        label_records = [{"label": 1, "base_count": 10}, {"label": 2, "base_count": 10}]
        for label in (1, 2):
            write_matrix(sealed / f"label_{label}.u32bin", np.arange((label - 1) * 10, label * 10), "<u4")
        outputs, manifests = {}, {}
        for split, label in (("calibration", 1), ("validation", 2)):
            prepared = sealed / ("prepared-" + split)
            prepared.mkdir()
            for name in ("base.fbin", "labels.txt", "label_1.u32bin", "label_2.u32bin"):
                (prepared / name).symlink_to(sealed / name)
            write_matrix(prepared / "query.fbin", base[[(label - 1) * 10]], "<f4")
            write_matrix(prepared / "query_labels.u32bin", [label], "<u4")
            write_matrix(prepared / "query_ids.u32bin", [label], "<u4")
            (prepared / "query_labels.txt").write_text(f"{label}\n")
            query = {"id": label-1, "qid": label, "positive_item_id": str(label), "user_hash": "same" if shared_user else str(label), "text_hash": str(label)}
            (prepared / "query_metadata.jsonl").write_text(json.dumps(query) + "\n")
            ids = np.arange((label - 1) * 10, label * 10, dtype="<i4")[None, :]
            d64 = np.zeros((1, 10), dtype="<f8")
            (prepared / "groundtruth.bin").write_bytes(struct.pack("<II", 1, 10) + ids.tobytes() + d64.astype("<f4").tobytes())
            write_matrix(prepared / "groundtruth_distances.f64bin", d64, "<f8")
            write_json(prepared / "truth-audit.json", {"pass": True})
            names = ["base.fbin", "labels.txt", "query.fbin", "query_labels.u32bin", "query_ids.u32bin", "query_labels.txt", "query_metadata.jsonl", "groundtruth.bin"]
            artifacts = {name: audit.record(prepared / name) for name in names}
            outputs[split] = {"artifacts": artifacts}
            manifests[split] = {"artifacts": artifacts.copy(), "labels": [{"original_label": x, "selected_queries": int(x == label)} for x in (1, 2)]}
        assignments = sealed / "group-assignments.json"
        write_json(assignments, [{"split": split, "source_query_rows": [i], "retained_query_rows": [i]} for i, split in enumerate(("calibration", "validation"))])
        commitment = sealed / "split-commitment.json"
        write_json(commitment, {"group_assignments": audit.record(assignments), "embedding_identity": audit.record(embedding), "base_vectors": 20, "dimension": 384, "labels": label_records, "outputs": outputs})
        for split, manifest in manifests.items():
            manifest["split_commitment"] = audit.record(commitment)
            write_json(sealed / ("prepared-" + split) / "manifest.json", manifest)
        return sealed

    def test_full_audit_adds_posting_closure_and_preserves_original_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed = self.full_fixture(root)
            result = audit.audit(root)
            self.assertEqual(result["all_category_postings_verified"], 2)
            self.assertEqual(result["truth_pairs_checked"], 20)
            for split in ("calibration", "validation"):
                prepared = sealed / ("prepared-" + split)
                self.assertTrue((prepared / "manifest.before-postings-audit.json").exists())
                manifest = json.loads((prepared / "manifest.json").read_text())
                self.assertIn("label_1.u32bin", manifest["artifacts"])
                self.assertIn("label_2.u32bin", manifest["artifacts"])
            with self.assertRaisesRegex(ValueError, "already sealed"):
                audit.audit(root)

    def test_cross_split_user_overlap_rejected_before_manifest_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed = self.full_fixture(root, shared_user=True)
            with self.assertRaisesRegex(ValueError, "overlap: user_hash"):
                audit.audit(root)
            self.assertFalse((sealed / "prepared-audit.json").exists())
            self.assertFalse(any(sealed.glob("prepared-*/manifest.before-postings-audit.json")))

    def test_truncated_postings_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "postings"
            path.write_bytes(b"\x01\0\0\0\x01\0\0\0")
            with self.assertRaisesRegex(ValueError, "Malformed"):
                audit.matrix(path, "<u4")


if __name__ == "__main__":
    unittest.main()
