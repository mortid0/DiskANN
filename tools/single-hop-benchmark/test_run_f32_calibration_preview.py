#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Guard tests for calibration-only orchestration; no ANN subprocesses run."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("preview", Path(__file__).with_name("run_f32_calibration_preview.py"))
PREVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREVIEW)


class PreviewTests(unittest.TestCase):
    def test_freeze_retains_actual_holdout_identity_without_evaluate(self):
        run = Path("/example/run")
        command = PREVIEW.freeze_command(run, PREVIEW.layout(run), [Path("/frozen/source")])
        self.assertIn("--holdout-prepared", command)
        self.assertNotIn("evaluate", command)
        self.assertNotIn("--inline-prefix", command)
        self.assertNotIn("--functional-smoke", command)
        self.assertIn("/example/amazon-c4/data/sealed/prepared-calibration", command)

    def test_books_uses_original_prepared_bank_and_shared_filtered_graph(self):
        run = Path("/example/run")
        paths = PREVIEW.layout(run, "books-bank")
        self.assertEqual(paths["prepared"], Path("/example/amazon-books/sealed-bank/prepared-calibration"))
        self.assertEqual(paths["prefix"].name, "books_bank_filtered_R32_R128_L100")
        self.assertEqual(paths["job"].name, "books-bank-calibration-preview")

    def test_preview_outputs_are_exclusive_and_failed_evidence_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            PREVIEW.save(path, {"status": "failed"})
            with self.assertRaises(FileExistsError):
                PREVIEW.save(path, {"status": "complete"})
            self.assertEqual(json.loads(path.read_text()), {"status": "failed"})

    def test_mutable_job_status_replaces_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            PREVIEW.save(path, {"status": "running"})
            PREVIEW.save(path, {"status": "complete"}, replace=True)
            self.assertEqual(json.loads(path.read_text())["status"], "complete")

    def test_holdout_claim_has_original_runner_path(self):
        self.assertEqual(PREVIEW.claim_path(Path("/data/split-commitment.json")),
                         Path("/data/split-commitment.f32-evaluation-claim.json"))

    def test_summary_never_counts_repeated_timings_as_independent_queries(self):
        protocol = {"engines": ["cpp"], "evaluation_orders": [["cpp"], ["cpp"]]}
        rows = [{"scope": scope, "engine": "cpp", "recall": .9, "l": 64, "qps": qps,
                 "p99_us": 12, "labels": [{"label": 1, "n": 10, "eligible": 100, "recall": .9, "p99_us": 12}]}
                for scope in ("fixed_l", "target90") for qps in (100, 200)]
        summary = PREVIEW.summarize(rows, protocol, 10)
        self.assertEqual(summary["quality_queries"], 10)
        self.assertEqual(summary["timing_repeats"], 2)
        self.assertEqual(summary["holdout_ann_queries_run"], 0)
        self.assertEqual(summary["summary"][0]["qps_median"], 150)
        rows[1]["recall"] = .8
        with self.assertRaisesRegex(ValueError, "repeated quality"):
            PREVIEW.summarize(rows, protocol, 10)

    def test_live_dependencies_must_match_successful_build_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "native-helpers-v2").mkdir()
            library = run / "libtest.so"
            library.write_bytes(b"original")
            original = PREVIEW.record(library)
            (run / "native-helpers-v2/build-result.json").write_text(json.dumps({
                "runtime_dependencies": {"helper": {"resolved_dependencies": {str(library): original}}}}))
            (run / "native-helpers-v2/build-plan.json").write_text(json.dumps({
                "linked_libraries": [], "runtime_environment": {}}))
            with patch.object(PREVIEW.subprocess, "check_output", return_value=""):
                self.assertIn(library.resolve(), PREVIEW.runtime_source_paths(run))
            library.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "dependency changed"):
                PREVIEW.runtime_source_paths(run)


if __name__ == "__main__":
    unittest.main()
