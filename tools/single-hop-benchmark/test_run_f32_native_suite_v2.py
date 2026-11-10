#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Exercise actual continuation dispatch/serialization without ANN processes."""

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


V2 = module("suite_v2", "run_f32_native_suite_v2.py")
V1 = module("suite_v1", "run_f32_native_suite.py")
PREVIEW = module("preview_v1", "run_f32_calibration_preview.py")
SUFFIXES = ("_disk.index", "_disk.index_labels.txt", "_disk.index_labels_map.txt",
            "_disk.index_labels_to_medoids.txt", "_pq_pivots.bin", "_pq_compressed.bin")
ENV = {"LD_LIBRARY_PATH": "/libraries", "OMP_TOOL": "disabled", "OMP_NUM_THREADS": "8",
       "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def args_for(run):
    return SimpleNamespace(run_root=run, previous_supervisor=Path(V1.__file__), preview_script=Path(PREVIEW.__file__),
                           analyzer=Path(__file__).with_name("analyze_f32_results_v3.py"),
                           books_preview_amendment=run / "books-preview-schedule-amendment-20261023.json",
                           books_preview_amendment_sha256="amendment-sha", infrastructure_amendment=run / "infrastructure-amendment-20261024.json",
                           infrastructure_amendment_sha256="infrastructure-sha")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    V2.save(path, value)


def failed_fixture(run):
    trace = V2.FAILED_STAGE + "\njson.dump\n" + V2.ERROR
    status = {"status": "failed", "exit_code": 1, "stage": V2.BANK_STAGE, "traceback": trace, "sources": []}
    write(run / "jobs/amazon-f32-suite/status.json", status)
    write(run / "jobs/amazon-f32-suite/stages" / V2.FAILED_STAGE / "status.json",
          {"status": "failed", "exit_code": 1, "traceback": trace})
    return status


class ContinuationTests(unittest.TestCase):
    def test_actual_stage_execution_serializes_every_command_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory).resolve() / "run"
            run.mkdir()
            executor = V2.StageExecutor(run, PREVIEW, [], ENV)
            stages = V2.planned_stages(run, args_for(run), V1, [{"path": "/frozen/source"}])
            observed = []
            for stage in stages:
                if not stage.get("command"):
                    continue
                # Even originally all-string build commands exercise Path conversion.
                stage["command"][0] = Path(stage["command"][0])
                def execute(command, **_):
                    state = json.loads((executor.current / "status.json").read_text())
                    persisted = json.loads((executor.current / "command.json").read_text())
                    self.assertTrue(all(isinstance(token, str) for token in state["command"]))
                    self.assertTrue(all(isinstance(token, str) for token in persisted["command"]))
                    self.assertEqual(command, persisted["command"])
                    self.assertEqual(state["command"], V2.normalize_command(stage["command"]))
                    self.assertTrue(str(executor.current / "resources.txt") in command)
                    stage["success"].parent.mkdir(parents=True, exist_ok=True)
                    stage["success"].write_text("{}")
                    observed.append(stage["name"])
                    return SimpleNamespace(returncode=0)
                with patch.object(V2.subprocess, "run", side_effect=execute):
                    executor.execute(stage)
                self.assertEqual(json.loads((executor.current / "status.json").read_text())["status"], "complete")
            self.assertEqual(len(observed), 15)
            self.assertTrue(observed[0].endswith("calibration-preview"))
            self.assertEqual({name.split("-")[0] for name in observed if name.startswith(("freeze-", "calibrate-", "evaluate-", "analyze-"))},
                             {"freeze", "calibrate", "evaluate", "analyze"})

    def test_execution_plan_round_trips_paths_before_any_file_is_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            stages = V2.planned_stages(run, args_for(run), V1, [])
            path = run / "execution-plan.json"
            V2.save(path, {"stages": stages, "nested": {"path": run}})
            value = json.loads(path.read_text())
            self.assertEqual(value["nested"]["path"], str(run))
            self.assertTrue(all(isinstance(token, str) for stage in value["stages"] for token in stage.get("command", [])))
            invalid = run / "invalid.json"
            with self.assertRaises(TypeError):
                V2.save(invalid, {"unknown": object()})
            self.assertFalse(invalid.exists())

    def test_recipe_preserves_build_parameters_and_all_calibration_gate(self):
        run = Path("/benchmark/run")
        stages = V2.planned_stages(run, args_for(run), V1, [])
        builds = [stage for stage in stages if "prefix" in stage]
        self.assertEqual([stage["command"] for stage in builds], [item["command"] for item in V1.build_commands(run)[2:]])
        self.assertEqual([stage["prefix"].name for stage in builds], V1.BUILD_ORDER[3:])
        gate = next(index for index, stage in enumerate(stages) if stage.get("action") == "freeze_policies")
        self.assertEqual(sum(stage["name"].startswith("calibrate-") for stage in stages[:gate]), 3)
        self.assertFalse(any(stage["name"].startswith("evaluate-") for stage in stages[:gate]))
        self.assertTrue(all(stage.get("policy_gate") for stage in stages if stage["name"].startswith("evaluate-")))
        self.assertEqual(stages[0]["command"][stages[0]["command"].index("--completed-build-status")+1],
                         run / "jobs/amazon-f32-suite/stages" / V2.BANK_STAGE / "status.json")

    def test_duplicate_continuation_job_never_overwrites_first_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            first = V2.StageExecutor(run, PREVIEW, [], ENV)
            before = (first.job / "status.json").read_bytes()
            with patch.object(V2.subprocess, "run") as execute:
                with self.assertRaises(FileExistsError):
                    V2.StageExecutor(run, PREVIEW, [], ENV)
                execute.assert_not_called()
            self.assertEqual((first.job / "status.json").read_bytes(), before)

    def test_source_or_command_failure_preserves_previously_completed_stage(self):
        for kind in ("source", "command"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                executor = V2.StageExecutor(Path(directory), PREVIEW, [], ENV)
                executor.execute({"name": "completed"})
                previous = executor.current / "status.json"
                before = previous.read_bytes()
                try:
                    if kind == "source":
                        executor.sources.append({"path": "/missing", "bytes": 1, "sha256": "bad"})
                        executor.execute({"name": "new"})
                    else:
                        executor.execute({"name": "new", "command": [12]})
                except (FileNotFoundError, ValueError):
                    executor.fail()
                else:
                    self.fail("Expected guard failure")
                self.assertEqual(previous.read_bytes(), before)
                self.assertEqual(json.loads((executor.job / "status.json").read_text())["status"], "failed")

    def test_policy_gate_failure_launches_nothing_and_preserves_prior_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = V2.StageExecutor(Path(directory), PREVIEW, [], ENV)
            executor.execute({"name": "calibrated"})
            previous = executor.current / "status.json"
            before = previous.read_bytes()
            def denied():
                raise ValueError("Missing all-policy checkpoint")
            with patch.object(V2.subprocess, "run") as execute:
                try:
                    executor.execute({"name": "evaluate", "command": [Path("/query")]}, gate=denied)
                except ValueError:
                    executor.fail()
                execute.assert_not_called()
            self.assertEqual(previous.read_bytes(), before)
            self.assertEqual(json.loads((executor.current / "status.json").read_text())["status"], "failed")
            self.assertFalse((executor.current / "process.log").exists())

    def test_expected_failure_is_required_before_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            status = failed_fixture(run)
            self.assertEqual(V2.verify_failure(run, V1, PREVIEW), status)
            status["traceback"] = "An unrelated failure"
            V2.save(run / "jobs/amazon-f32-suite/status.json", status, replace=True)
            with self.assertRaisesRegex(ValueError, "recorded command-serialization"):
                V2.verify_failure(run, V1, PREVIEW)

    def test_unexpected_failure_is_rejected_before_new_job_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            status = failed_fixture(run)
            status["exit_code"] = 2
            V2.save(run / "jobs/amazon-f32-suite/status.json", status, replace=True)
            before = (run / "jobs/amazon-f32-suite/status.json").read_bytes()
            with patch.object(V2.subprocess, "run") as execute:
                with self.assertRaisesRegex(ValueError, "expected failed run"):
                    V2.run_suite(args_for(run))
                execute.assert_not_called()
            self.assertFalse((run / "jobs" / V2.JOB_NAME).exists())
            self.assertEqual((run / "jobs/amazon-f32-suite/status.json").read_bytes(), before)

    def test_amendment_binds_this_source_and_exactly_three_reused_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory).resolve() / "run"
            failed_fixture(run)
            args = args_for(run)
            write(run / "experiment-plan.json", {})
            allowed = []
            digests = {}
            for name in V2.REUSED_MANIFESTS:
                path = run / "indexes" / (name+".manifest.json")
                write(path, {"fixture": name})
                allowed.append(PREVIEW.record(path))
                digests[name] = PREVIEW.record(path)["sha256"]
            amendment = {"schema_version": 1, "status": "authorized_continuation",
                         "original_plan": PREVIEW.record(run / "experiment-plan.json"),
                         "failed_suite_status": PREVIEW.record(run / "jobs/amazon-f32-suite/status.json"),
                         "previous_supervisor": PREVIEW.record(args.previous_supervisor),
                         "continuation_source": PREVIEW.record(Path(V2.__file__)),
                         "allowed_reused_index_manifests": allowed}
            write(args.infrastructure_amendment, amendment)
            args.infrastructure_amendment_sha256 = PREVIEW.record(args.infrastructure_amendment)["sha256"]
            with patch.dict(V2.REUSED_MANIFESTS, digests, clear=True):
                self.assertEqual(V2.verify_amendment(args, PREVIEW), amendment)
                amendment["continuation_source"]["sha256"] = "wrong-source"
                V2.save(args.infrastructure_amendment, amendment, replace=True)
                args.infrastructure_amendment_sha256 = PREVIEW.record(args.infrastructure_amendment)["sha256"]
                with self.assertRaisesRegex(ValueError, "continuation_source"):
                    V2.verify_amendment(args, PREVIEW)
                amendment["continuation_source"] = PREVIEW.record(Path(V2.__file__))
                amendment["allowed_reused_index_manifests"][1] = allowed[0]
                V2.save(args.infrastructure_amendment, amendment, replace=True)
                args.infrastructure_amendment_sha256 = PREVIEW.record(args.infrastructure_amendment)["sha256"]
                with self.assertRaisesRegex(ValueError, "exactly the three"):
                    V2.verify_amendment(args, PREVIEW)

    def test_prior_books_preview_or_real_holdout_blocks_continuation(self):
        for kind in ("preview", "holdout", "protocol", "partial-index", "launched-process"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                run = Path(directory) / "run"
                failed_fixture(run)
                if kind == "preview":
                    path = run / "calibration-preview/books-bank"
                elif kind == "holdout":
                    path = PREVIEW.claim_path(V1.workloads(run)[0]["prepared_root"] / "split-commitment.json")
                elif kind == "protocol":
                    path = run / "protocols"
                elif kind == "partial-index":
                    path = run / "indexes" / (V1.BUILD_ORDER[3] + ".work")
                else:
                    path = run / "jobs/amazon-f32-suite/stages" / V2.FAILED_STAGE / "process.log"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("retained evidence")
                with self.assertRaises(ValueError):
                    V2.verify_failure(run, V1, PREVIEW)
                self.assertEqual(path.read_text(), "retained evidence")

    def test_existing_stage_or_output_never_launches_duplicate_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            executor = V2.StageExecutor(run, PREVIEW, [], ENV)
            executor.execute({"name": "first"})
            previous = executor.current / "status.json"
            before = previous.read_bytes()
            with patch.object(V2.subprocess, "run") as execute:
                with self.assertRaises(FileExistsError):
                    executor.execute({"name": "first", "command": ["/query"]})
                output = run / "existing-output"
                output.mkdir()
                with self.assertRaisesRegex(ValueError, "already exists"):
                    executor.execute({"name": "other", "command": ["/query"], "absent": output})
                execute.assert_not_called()
            self.assertEqual(previous.read_bytes(), before)

    def reused_fixture(self, run):
        (run / "indexes").mkdir(parents=True)
        build_path = run / "native-helpers-v2/build-result.json"
        write(build_path, {"binaries": {"build_stitched_f32": {"sha256": "builder"}, "create_disk_layout": {"sha256": "layout"}}})
        write(run / "experiment-plan.json", {"workloads": {"amazon_c4": {"base_vectors": 4}, "books_bank": {"base_vectors": 8}}})
        inputs = {}
        for corpus, prepared, n in (("c4", V1.workloads(run)[0]["prepared_root"], 4), ("books", V1.workloads(run)[1]["prepared_root"], 8)):
            base, labels = run / (corpus+".base"), run / (corpus+".labels")
            base.write_bytes(corpus.encode())
            labels.write_bytes((corpus+"labels").encode())
            inputs[corpus] = {"base": PREVIEW.record(base), "labels": PREVIEW.record(labels), "source:build_evidence": PREVIEW.record(build_path)}
            write(prepared / "prepared-calibration/manifest.json", {"artifacts": {"base.fbin": PREVIEW.record(base), "labels.txt": PREVIEW.record(labels)}})
        digests = {}
        for name in list(V2.REUSED_MANIFESTS):
            prefix = run / "indexes" / name
            corpus = "c4" if name.startswith("c4") else "books"
            artifacts = {}
            for suffix in SUFFIXES:
                path = Path(str(prefix)+suffix)
                path.write_bytes((corpus+suffix).encode())
                artifacts[suffix] = PREVIEW.record(path)
            global_graph = name == V1.BUILD_ORDER[1]
            manifest = {"status": "complete", "label_degree": 64 if global_graph else 32,
                        "stitched_degree": 64 if global_graph else 128, "build_l": 100,
                        "disk_metadata": {"medoid": 301656 if global_graph else 0, "rows": 4 if corpus=="c4" else 8},
                        "unqueried_label_graph_omitted": 200386 if corpus=="books" else None,
                        "build_binaries_sha256": {"build_stitched_f32": "builder", "utils/create_disk_layout": "layout"},
                        "inputs_before": inputs[corpus], "inputs_after_encoding": inputs[corpus], "inputs_after": inputs[corpus],
                        "base_sha256": inputs[corpus]["base"]["sha256"], "labels_sha256": inputs[corpus]["labels"]["sha256"], "artifacts": artifacts}
            manifest_path = Path(str(prefix)+".manifest.json")
            write(manifest_path, manifest)
            digests[name] = PREVIEW.record(manifest_path)["sha256"]
        preview = run / "calibration-preview/c4/preview.json"
        write(preview, {})
        previous = run / "jobs/amazon-f32-suite/stages"
        for stage_name, path in (("00-wait-c4-build", run / "indexes" / (V1.BUILD_ORDER[0]+".manifest.json")),
                                 ("01-wait-c4-preview", preview), ("02-c4_global_R64_L100", run / "indexes" / (V1.BUILD_ORDER[1]+".manifest.json")),
                                 (V2.BANK_STAGE, run / "indexes" / (V1.BUILD_ORDER[2]+".manifest.json"))):
            state = {"status": "complete", "exit_code": 0, "artifacts": [PREVIEW.record(path)]}
            if stage_name in ("02-c4_global_R64_L100", V2.BANK_STAGE):
                state["command"] = V1.build_commands(run)[0 if stage_name.startswith("02-") else 1]["command"]
            write(previous / stage_name / "status.json", state)
        return digests

    def test_reuse_rehashes_every_final_artifact_and_input(self):
        for change in (None, "artifact", "input"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                run = Path(directory).resolve() / "run"
                digests = self.reused_fixture(run)
                if change == "artifact":
                    (run / "indexes" / (V1.BUILD_ORDER[2]+"_disk.index")).write_bytes(b"changed")
                elif change == "input":
                    (run / "books.base").write_bytes(b"changed")
                with patch.dict(V2.REUSED_MANIFESTS, digests, clear=True), patch.object(PREVIEW, "load_runner", return_value=SimpleNamespace(SUFFIXES=SUFFIXES)):
                    if change:
                        with self.assertRaisesRegex(ValueError, "artifact or input changed"):
                            V2.verify_reused_indexes(run, V1, PREVIEW, {"sources": []})
                    else:
                        proof = V2.verify_reused_indexes(run, V1, PREVIEW, {"sources": []})
                        self.assertEqual(len(proof["reused_manifests"]), 3)


if __name__ == "__main__":
    unittest.main()
