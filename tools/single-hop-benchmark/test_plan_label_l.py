#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Regression coverage for label-floor selection and mixed-plan validation."""

import copy
import io
import json
from pathlib import Path
import tempfile
import tarfile
import unittest

import plan_label_l as planner


def query_row(l, repeat=0, cpp=False):
    counts = {label: 32 if label < 16 else 31 for label in range(64)}
    labels = [{"label": label, "queries": count, "eligible": 1000,
               "recall": (count * 9 - int(label == 1 and l == 64 and not cpp)) / (count * 10),
               "mean_us": 4000, "p50_us": 4000, "p95_us": 4100, "actual_l": l}
              for label, count in counts.items()]
    row = {"kind": "query", "phase": "query", "implementation": "rust_native_lifecycle",
           "l": l, "repeat": repeat, "k": 10, "threads": 8, "queries": 2000,
           "warmup_queries": 2000, "sequence": 0, "base_searches": 2000, "delta_candidates": 0,
           "strict_id_recall": True, "timing_authoritative": False,
           "latency_contract": "public_snapshot_search_call", "labels": labels,
           "recall": sum(item["queries"] * item["recall"] for item in labels) / 2000,
           "qps": 2000, "wall_seconds": 1, "mean_us": 4000, "p50_us": 4000,
           "p95_us": 4100, "p99_us": 4200,
           "prepared_identity": ":".join(["0" * 64] * 4), "label_l_plan": None,
           **{name: 0 for name in planner.VALIDITY}}
    if cpp:
        row.update({"implementation": "cpp_native_linux", "mode": "single", "starts": "label",
                    "index_prefix": "/ssd/cpp", "n_base": 10000000, "beam": 4,
                    "cache_bypass": True, "shared_cache_nodes": 0, "storage": "linux_aio_direct",
                    "timing_authoritative": True, "latency_contract": "public_search_call"})
        row["labels"] = [{**item, "n": item["queries"]} for item in labels]
    return row


def experiment():
    return {"kind": "experiment", "action": "query", "n_base": 10000000,
            "dimensions": 192, "index_threads": 8, "beam": 4, "shared_cache_nodes": 0,
            "cache_bypass": True, "storage": "native_io_uring", "query_scheduling": "dynamic_one",
            "warmup": "full_same_process_same_worker_count", "root": "/ssd/native", "prepared": "/data/prepared"}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class LabelPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.grid = self.root / "grid.jsonl"
        self.cpp = self.root / "cpp.jsonl"
        self.plan = self.root / "plan.json"
        self.evidence = self.root / "evidence.json"
        self.query = self.root / "mixed.jsonl"
        self.grid_rows = [experiment()] + [query_row(l) for l in planner.GRID]
        write_rows(self.grid, self.grid_rows)
        write_rows(self.cpp, [query_row(64, repeat, cpp=True) for repeat in range(3)])

    def make_plan(self):
        data, evidence = planner.planning(self.grid, self.cpp, self.plan)
        self.plan.write_bytes(data)
        self.evidence.write_text(json.dumps(evidence))
        plan = planner.load_plan(self.plan)
        rows = [experiment()]
        for repeat in range(3):
            row = query_row(64, repeat)
            row["timing_authoritative"] = True
            row["label_l_tuning_scope"] = "empirical_on_existing_prepared_workload_not_held_out"
            for item in row["labels"]:
                item["actual_l"] = plan[item["label"]]
                item["recall"] = 0.9
            row["recall"] = .9
            row["label_l_plan"] = {"path": str(self.plan), "content_sha256": planner.file_sha(self.plan),
                                   "entries": [{"label": label, "l": l} for label, l in sorted(plan.items())]}
            rows.append(row)
        write_rows(self.query, rows)
        return plan, rows

    def test_selects_smallest_qualifying_measured_breadth_and_validates_three_repeats(self):
        plan, _ = self.make_plan()
        self.assertEqual(plan[1], 68)
        self.assertEqual(sum(value == 64 for value in plan.values()), 63)
        result = planner.evaluate(self.query, self.plan, self.evidence, self.cpp)
        self.assertEqual(result["native"]["recall"], .9)
        self.assertEqual(result["qps_ratio_native_over_CPP"], 1)

    def test_nonmonotone_grid_does_not_select_later_breadth(self):
        grid = {l: {7: (10, 89 if l in (64, 72, 144) else 90)} for l in planner.GRID}
        selected, _ = planner.select_plan(grid, {7: (10, 90)})
        self.assertEqual(selected, {7: 68})

    def test_rejects_unreachable_label_floor(self):
        grid = {l: {7: (10, 89)} for l in planner.GRID}
        with self.assertRaisesRegex(ValueError, "Label 7 does not reach"):
            planner.select_plan(grid, {7: (10, 90)})

    def test_rejects_invalid_grid_counter(self):
        self.grid_rows[1]["short_results"] = 1
        write_rows(self.grid, self.grid_rows)
        with self.assertRaisesRegex(ValueError, "Invalid, short"):
            planner.planning(self.grid, self.cpp, self.plan)

    def test_rejects_authoritative_grid_timing(self):
        self.grid_rows[1]["timing_authoritative"] = True
        write_rows(self.grid, self.grid_rows)
        with self.assertRaisesRegex(ValueError, "grid observations must be diagnostic"):
            planner.planning(self.grid, self.cpp, self.plan)

    def test_rejects_missing_grid_breadth(self):
        write_rows(self.grid, self.grid_rows[:-1])
        with self.assertRaisesRegex(ValueError, "every fixed grid L"):
            planner.planning(self.grid, self.cpp, self.plan)

    def test_rejects_bad_actual_breadth_and_plan_hash(self):
        _, rows = self.make_plan()
        for mutate in (lambda row: row["labels"][0].update(actual_l=144),
                       lambda row: row["label_l_plan"].update(content_sha256="f" * 64)):
            changed = copy.deepcopy(rows)
            mutate(changed[1])
            write_rows(self.query, changed)
            with self.assertRaises(ValueError):
                planner.evaluate(self.query, self.plan, self.evidence, self.cpp)

    def test_rejects_mixed_repeat_label_shortfall(self):
        _, rows = self.make_plan()
        for row in rows[1:]:
            row["labels"][0]["recall"] = 287 / 320
            row["recall"] = 17999 / 20000
        write_rows(self.query, rows)
        with self.assertRaisesRegex(ValueError, r"falls below C\+\+ L64"):
            planner.evaluate(self.query, self.plan, self.evidence, self.cpp)

    def test_rejects_edited_plan_evidence(self):
        self.make_plan()
        evidence = json.loads(self.evidence.read_text())
        evidence["labels"][0]["target_CPP_strict_id_hits"] -= 1
        self.evidence.write_text(json.dumps(evidence))
        with self.assertRaisesRegex(ValueError, "independently recomputed"):
            planner.evaluate(self.query, self.plan, self.evidence, self.cpp)

    def test_rejects_duplicate_and_missing_mixed_repeats(self):
        _, rows = self.make_plan()
        for changed in (rows[:-1], rows[:-1] + [rows[1]]):
            write_rows(self.query, changed)
            with self.assertRaisesRegex(ValueError, "three unique"):
                planner.evaluate(self.query, self.plan, self.evidence, self.cpp)

    def test_identity_binds_logs_archives_and_unchanged_index(self):
        self.make_plan()
        required = ["diskann-disk/src/filtered/" + name + ".rs" for name in
                    ("native", "index", "view", "updates", "persistence", "metadata")]
        required += ["diskann-disk/src/build/filtered.rs", "diskann/src/neighbor/queue.rs",
                     "diskann/src/graph/search/single_hop_filter_search.rs",
                     "diskann-disk/examples/filtered_lifecycle.rs"]
        archives = []
        for version in ("grid", "mixed"):
            path = self.root / (version + ".tar.gz")
            with tarfile.open(path, "w:gz") as archive:
                for name in required:
                    content = (version if name.endswith("examples/filtered_lifecycle.rs") else "unchanged").encode()
                    item = tarfile.TarInfo(name)
                    item.size = len(content)
                    archive.addfile(item, io.BytesIO(content))
            archives.append(path)
        sidecar = {"schema_version": 1, "shared_index": {"host": "test", "root": "/ssd/native"}}
        for name, path, log in zip(("grid", "mixed"), archives, (self.grid, self.query)):
            sidecar[name] = {"binary_sha256": "1" * 64, "log_sha256": planner.file_sha(log),
                             "source_archive": {"path": str(path), "sha256": planner.file_sha(path)},
                             "index_identity": {key: "2" * 64 for key in
                                                ("current_sha256", "catalog_sha256", "journal_sha256")}}
        identity = self.root / "identity.json"
        identity.write_text(json.dumps(sidecar))
        result = planner.evaluate(self.query, self.plan, self.evidence, self.cpp, identity)
        self.assertTrue(result["identity"]["available"])
        sidecar["mixed"]["index_identity"]["current_sha256"] = "3" * 64
        identity.write_text(json.dumps(sidecar))
        with self.assertRaisesRegex(ValueError, "committed index or journal identity changed"):
            planner.evaluate(self.query, self.plan, self.evidence, self.cpp, identity)


if __name__ == "__main__":
    unittest.main()
