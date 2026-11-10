#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

import run_f32_benchmark_v3 as bench


def row(engine, l, recall, short=0):
    return {"engine": engine, "l": l, "recall": recall, "queries": 10,
            "short_results": short, "filter_violations": 0, "duplicate_results": 0,
            "strict_id_recall": True}


class PolicyTests(unittest.TestCase):
    def test_selects_lowest_passing_l_without_assuming_monotonic_recall(self):
        rows = [row(engine, l, quality) for engine in bench.ENGINES
                for l, quality in [(10, .7), (16, .92), (24, .89), (32, .95)]]
        self.assertEqual({engine: value["l"] for engine, value in bench.choose_target(rows).items()},
                         {engine: 16 for engine in bench.ENGINES})

    def test_short_result_rows_are_reportable_but_not_selected_for_target90(self):
        rows = [item for engine in bench.ENGINES for item in
                (row(engine, 10, .91, short=1), row(engine, 16, .93))]
        self.assertTrue(all(value["l"] == 16 for value in bench.choose_target(rows).values()))

    def test_rejects_invalid_filter_result_and_unreached_target(self):
        rows = [row(engine, 16, .92) for engine in bench.ENGINES]
        rows[0]["filter_violations"] = 1
        with self.assertRaisesRegex(ValueError, "Invalid"):
            bench.choose_target(rows)
        missing = bench.choose_target([row(engine, 16, .89) for engine in bench.ENGINES])
        self.assertTrue(all(value["status"] == "unreached_within_grid" and value["l"] is None for value in missing.values()))

    def test_ninety_percent_boundary_uses_integer_hits(self):
        boundary = row("cpp", 16, .8999999999999999)
        boundary["raw_audit"] = {"strict_id_hits": 90, "total_truth_ids": 100}
        self.assertTrue(bench.reaches_target(boundary))
        boundary["raw_audit"]["strict_id_hits"] = 89
        boundary["recall"] = .89
        self.assertFalse(bench.reaches_target(boundary))

    def test_four_engine_orders_balance_position_and_pair_precedence(self):
        engines = [*bench.ENGINES, "rust_inline_global"]
        orders = bench.engine_orders(engines)
        self.assertEqual(len(orders), 8)
        for engine in engines:
            self.assertEqual([sum(order[position] == engine for order in orders) for position in range(4)], [2]*4)
            for other in engines:
                if other != engine:
                    self.assertEqual(sum(order.index(engine) < order.index(other) for order in orders), 4)

    def test_duplicate_l_is_rejected(self):
        rows = [row(engine, 16, .92) for engine in bench.ENGINES]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            bench.choose_target(rows + [copy.deepcopy(rows[0])])


class RawAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.matrix("query_labels.u32bin", 2, 1, [7, 7])
        self.matrix("label_7.u32bin", 4, 1, [1, 2, 3, 4])
        self.matrix("groundtruth.bin", 2, 2, [1, 2, 3, 4])
        self.raw = [{"query": 0, "label": 7, "l": 64, "returned": 2, "ids": [1, 2], "distances": [0., 1.], "micros": 10.},
                    {"query": 1, "label": 7, "l": 64, "returned": 1, "ids": [3], "distances": [0.], "micros": 30.}]
        self.path = self.root / "raw.jsonl"
        self.row = {"l": 64, "recall": .75, "short_results": 1, "p99_us": 30., "wall_seconds": .1, "qps": 20.,
                    "labels": [{"label": 7, "n": 2, "eligible": 4, "recall": .75, "p99_us": 30.}],
                    "query_details": {"path": str(self.path)}}
        self.write_raw()
        for name, values in (("base.fbin", [0., 1., 2., 3., 4.]), ("query.fbin", [1., 3.])):
            (self.root/name).write_bytes(struct.pack("<II", len(values), 1) + struct.pack(f"<{len(values)}f", *values))

    def matrix(self, name, n, d, values):
        (self.root / name).write_bytes(struct.pack("<II", n, d) + struct.pack(f"<{len(values)}I", *values))

    def write_raw(self):
        self.path.write_text("".join(json.dumps(item) + "\n" for item in self.raw))

    def test_recomputes_hits_and_p99_and_handles_short_results(self):
        audit = bench.audit_output(self.row, self.root, "rust_single")
        self.assertEqual(audit["strict_id_hits"], 3)
        self.assertEqual(audit["unique_queries"], 2)

    def test_rejects_wrong_returned_id_and_percentile(self):
        self.raw[1]["ids"] = [9]
        self.write_raw()
        with self.assertRaisesRegex(ValueError, "filter"):
            bench.audit_output(self.row, self.root, "rust_single")
        self.raw[1]["ids"] = [3]
        self.write_raw()
        self.row["p99_us"] = 10
        with self.assertRaisesRegex(ValueError, "p99"):
            bench.audit_output(self.row, self.root, "rust_single")

    def test_rejects_wrong_returned_distance_or_qps(self):
        self.raw[0]["distances"] = [0., 8.]
        self.write_raw()
        with self.assertRaisesRegex(ValueError, "distance disagrees"):
            bench.audit_output(self.row, self.root, "rust_single")
        self.raw[0]["distances"] = [0., 1.]
        self.write_raw()
        self.row["qps"] = 200.
        with self.assertRaisesRegex(ValueError, "QPS"):
            bench.audit_output(self.row, self.root, "rust_single")

    def test_rejects_repeated_or_unordered_queries(self):
        self.raw[1]["query"] = 0
        self.write_raw()
        with self.assertRaisesRegex(ValueError, "exactly once"):
            bench.audit_output(self.row, self.root, "rust_single")


class SplitBindingTests(unittest.TestCase):
    def fixture(self, calibration_family="a", validation_family="b", duplicate_vector=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        outputs = {}
        for split, qid, family, value in (("calibration", 1, calibration_family, 0.),
                                         ("validation", 2, validation_family, 0. if duplicate_vector else 1.)):
            directory = root / split
            directory.mkdir()
            (directory / "query_ids.u32bin").write_bytes(struct.pack("<III", 1, 1, qid))
            (directory / "query.fbin").write_bytes(struct.pack("<II", 1, 1) + struct.pack("<f", value))
            (directory / "query_metadata.jsonl").write_text(json.dumps({"qid": qid, "positive_item_id": family}) + "\n")
            outputs[split] = {"query_count": 1, "artifacts": {name: bench.record(directory / name)
                              for name in ("query_ids.u32bin", "query.fbin", "query_metadata.jsonl")}}
        proof = root / "split.json"
        proof.write_text(json.dumps({"outputs": outputs, "ann_results_used": False,
                                    "fresh_validation_queries_searched": False}))
        for split in outputs:
            (root / split / "manifest.json").write_text(json.dumps({"split": split, "query_count": 1,
                 "split_commitment": bench.record(proof), "artifacts": outputs[split]["artifacts"]}))
        return root, proof

    def test_accepts_bound_disjoint_partitions(self):
        root, proof = self.fixture()
        audit = bench.audit_split(root / "calibration", root / "validation", bench.record(proof))
        self.assertTrue(audit["query_embeddings_disjoint"])

    def test_rejects_embedding_and_family_leakage(self):
        root, proof = self.fixture(duplicate_vector=True)
        with self.assertRaisesRegex(ValueError, "embeddings leak"):
            bench.audit_split(root / "calibration", root / "validation", bench.record(proof))
        root, proof = self.fixture(validation_family="a")
        with self.assertRaisesRegex(ValueError, "family leaks"):
            bench.audit_split(root / "calibration", root / "validation", bench.record(proof))

    def test_rejects_another_commitment_or_wrong_phase(self):
        root, proof = self.fixture()
        manifest_path = root / "validation/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["split"] = "calibration"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "phase"):
            bench.audit_split(root / "calibration", root / "validation", bench.record(proof))


if __name__ == "__main__":
    unittest.main()
