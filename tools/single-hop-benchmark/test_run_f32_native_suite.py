#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Test orchestration safety and corpus bindings without running ANN workloads."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


SUITE = module("suite", "run_f32_native_suite.py")
PREVIEW = module("preview", "run_f32_calibration_preview.py")


class SuiteTests(unittest.TestCase):
    def test_exact_remaining_build_order_and_pq_reuse(self):
        run = Path("/benchmark/run")
        builds = SUITE.build_commands(run)
        self.assertEqual([item["name"] for item in builds], SUITE.BUILD_ORDER[1:])
        for item in builds:
            command = item["command"]
            self.assertIn(str(run / "native-source-correction/cpp-port/build_f32_stitched.py"), command)
            self.assertEqual(command[command.index("--threads")+1], "8")
            self.assertEqual(command[command.index("--build-l")+1], "100")
        self.assertNotIn("--pq-prefix", builds[1]["command"])
        self.assertEqual(builds[1]["command"][builds[1]["command"].index("--training-points")+1], "100000")
        for item in (builds[2], builds[3]):
            self.assertEqual(item["command"][item["command"].index("--pq-prefix")+1], str(run / "indexes" / SUITE.BUILD_ORDER[2]))

    def test_global_entry_and_books_sentinel_have_distinct_scope(self):
        builds = SUITE.build_commands(Path("/benchmark/run"))
        for item, entry in ((builds[0], 301656), (builds[3], 6481001)):
            command = item["command"]
            self.assertEqual(command[command.index("--global-entry-point")+1], str(entry))
            self.assertEqual(command[command.index("--label-degree")+1], "64")
            self.assertEqual(command[command.index("--stitched-degree")+1], "64")
            self.assertNotIn("--skip-label-build", command)
        for item in (builds[1], builds[2]):
            self.assertEqual(item["command"][item["command"].index("--skip-label-build")+1], "200386")

    def test_failed_prerequisite_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            PREVIEW.save(path, {"status": "failed", "exit_code": 1})
            with patch.object(SUITE.time, "sleep") as sleep:
                with self.assertRaisesRegex(ValueError, "Prerequisite failed"):
                    SUITE.wait_completed(path, PREVIEW, 30)
                sleep.assert_not_called()

    def test_completion_requires_successful_exit_and_unchanged_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "done.json"
            artifact.write_text("{}")
            status = root / "status.json"
            PREVIEW.save(status, {"status": "complete", "exit_code": 1})
            with self.assertRaisesRegex(ValueError, "successfully"):
                SUITE.wait_completed(status, PREVIEW, 30)
            PREVIEW.save(status, {"status": "complete", "exit_code": 0, "artifacts": [PREVIEW.record(artifact)]}, replace=True)
            artifact.write_text("changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                SUITE.wait_completed(status, PREVIEW, 30)

    def test_existing_suite_claim_prevents_any_new_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "native-helpers-v2").mkdir()
            (run / "native-helpers-v2/build-plan.json").write_text(json.dumps({"runtime_environment": {}}))
            (run / "jobs/amazon-f32-suite").mkdir(parents=True)
            args = SimpleNamespace(run_root=run, preview_script=Path("/preview.py"), analyzer=Path("/analyzer.py"))
            with patch.object(SUITE, "load_preview", return_value=PREVIEW), patch.object(SUITE.subprocess, "run") as execute:
                with self.assertRaises(FileExistsError):
                    SUITE.run_suite(args)
                execute.assert_not_called()

    def test_source_preflight_waits_until_authoritative_preview_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "native-helpers-v2").mkdir()
            (run / "native-helpers-v2/build-plan.json").write_text(json.dumps({"runtime_environment": {}}))
            for path in (run / "indexes" / (SUITE.BUILD_ORDER[0]+".manifest.json"),
                         run / "calibration-preview/c4/preview.json"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            args = SimpleNamespace(run_root=run, preview_script=Path("/preview.py"), analyzer=Path("/analyzer.py"),
                                   poll_seconds=30, books_preview_amendment=Path("/amendment.json"),
                                   books_preview_amendment_sha256="sha")
            order = []
            def wait(path, *_):
                order.append(Path(path).parent.name)
            def preflight(*_):
                order.append("heavy-preflight")
                raise ValueError("Intentional test stop before heavy work")
            with patch.object(SUITE, "load_preview", return_value=PREVIEW), patch.object(SUITE, "wait_completed", side_effect=wait), \
                 patch.object(SUITE, "preflight", side_effect=preflight), patch.object(SUITE.subprocess, "run") as execute:
                with self.assertRaisesRegex(ValueError, "Intentional test stop"):
                    SUITE.run_suite(args)
                execute.assert_not_called()
            self.assertEqual(order, ["c4-filtered-build", "c4-calibration-preview", "heavy-preflight"])

    def fixture_policies(self, run):
        for workload in SUITE.workloads(run):
            name = workload["name"]
            protocol = {"engines": SUITE.ENGINES, "cpu_affinity": list(range(8)), "fixed_l": 64,
                        "functional_only": False, "calibration_grid": [10, 64, 128]}
            protocol_path = run / "protocols" / name / "protocol.json"
            protocol_path.parent.mkdir(parents=True)
            PREVIEW.save(protocol_path, protocol)
            policy = {engine: {"status": "calibrated", "l": 64} for engine in SUITE.ENGINES}
            policy["rust_inline_global"] = {"status": "unreached_within_grid", "l": None}
            calibration_path = run / "calibration" / name / "calibration.json"
            calibration_path.parent.mkdir(parents=True)
            PREVIEW.save(calibration_path, {"protocol": PREVIEW.record(protocol_path), "heldout_quality_read": False,
                         "target90": policy, "rows": [{"target90": policy}]})

    def test_all_three_policies_are_required_before_any_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.fixture_policies(run)
            (run / "calibration/books-original/calibration.json").unlink()
            fake = SimpleNamespace(choose_target=lambda rows, engines: rows[0]["target90"], check_grid=lambda *_: None)
            with patch.object(PREVIEW, "load_runner", return_value=fake):
                with self.assertRaises(FileNotFoundError):
                    SUITE.freeze_policies(run, PREVIEW, [])
            self.assertFalse((run / "all-native-L-policies.json").exists())

    def test_checkpoint_binds_all_policies_and_rejects_later_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.fixture_policies(run)
            fake = SimpleNamespace(choose_target=lambda rows, engines: rows[0]["target90"], check_grid=lambda *_: None)
            with patch.object(PREVIEW, "load_runner", return_value=fake):
                checkpoint = SUITE.freeze_policies(run, PREVIEW, [])
            self.assertEqual(SUITE.verify_policy_checkpoint(run, PREVIEW), checkpoint)
            with self.assertRaises(FileExistsError), patch.object(PREVIEW, "load_runner", return_value=fake):
                SUITE.freeze_policies(run, PREVIEW, [])
            calibration = run / "calibration/books-original/calibration.json"
            calibration.write_text(calibration.read_text()+" ")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                SUITE.verify_policy_checkpoint(run, PREVIEW)

    def test_prior_holdout_claim_blocks_policy_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            claim = PREVIEW.claim_path(SUITE.workloads(run)[0]["prepared_root"] / "split-commitment.json")
            claim.parent.mkdir(parents=True)
            claim.write_text("{}")
            with self.assertRaisesRegex(ValueError, "already claimed"):
                SUITE.freeze_policies(run, PREVIEW, [])

    def test_incorrect_global_seed_rejected_before_more_work(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            prefix = run / SUITE.BUILD_ORDER[1]
            PREVIEW.save(Path(str(prefix)+".manifest.json"), {"status": "complete", "label_degree": 64,
                         "stitched_degree": 64, "build_l": 100, "disk_metadata": {"medoid": 0}})
            with self.assertRaisesRegex(ValueError, "entry differs"):
                SUITE.validate_index_manifest(run, prefix, PREVIEW)


if __name__ == "__main__":
    unittest.main()
