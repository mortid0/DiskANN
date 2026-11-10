#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Explicit continuation of the recorded Oct 23 orchestration failure.

Reuse only the three verified completed indexes. Preserve every original output,
start with the unlaunched Books calibration preview, and retain the frozen
all-calibrations-before-holdout gate. This is an infrastructure correction;
neither search nor construction parameters change. Launch under taskset/nohup.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import traceback

V1_SHA = "cc0911c4e649cc88eb89bc83a7edd0eac827e4008ba527ce701f29f98b7238d8"
JOB_NAME = "amazon-f32-suite-continuation-20261024"
FAILED_STAGE = "03b-books-bank-calibration-preview"
BANK_STAGE = "03-books_bank_filtered_R32_R128_L100"
ERROR = "TypeError: Object of type PosixPath is not JSON serializable"
REUSED_MANIFESTS = {
    "c4_filtered_R32_R128_L100": "350c938fba915bfefaa6e5594b03060f33be25e447e81953f8c1501f4928689e",
    "c4_global_R64_L100": "df03cae6a9bb313dc36a041b0d9cca6c67cd4e8174d44173fe9df781fe403429",
    "books_bank_filtered_R32_R128_L100": "9e34de1ff9ddc1ff1ee8a0a58958d45ae2ba9aacd74c1bb35eb5c7fd00fe151f",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize_command(command):
    if command is None:
        return None
    require(isinstance(command, (list, tuple)) and command, "Command must be a nonempty token list")
    tokens = []
    for token in command:
        require(isinstance(token, (str, os.PathLike)), "Command tokens must be strings or filesystem paths")
        token = os.fspath(token)
        require(isinstance(token, str) and "\0" not in token, "Command tokens must be valid text")
        tokens.append(token)
    return tokens


def json_value(value):
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), "JSON object keys must be strings")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def save(path, value, replace=False):
    """Validate/encode the whole document before opening any output file."""
    encoded = json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}") if replace else path
    with temporary.open("x") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)


def load_v1(path):
    import hashlib
    require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == V1_SHA, "The frozen original supervisor changed")
    spec = importlib.util.spec_from_file_location("frozen_native_suite_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_failure(run, v1, preview):
    """Read only small controls before claiming the new continuation job."""
    previous = run / "jobs/amazon-f32-suite"
    status = json.loads((previous / "status.json").read_text())
    failed = json.loads((previous / "stages" / FAILED_STAGE / "status.json").read_text())
    require(status.get("status") == "failed" and status.get("exit_code") == 1 and status.get("stage") == BANK_STAGE,
            "The original suite is not the expected failed run")
    require(failed.get("status") == "failed" and failed.get("exit_code") == 1,
            "The unlaunched Books preview does not have the expected failed stage")
    require(status.get("traceback") == failed.get("traceback") and ERROR in status["traceback"]
            and FAILED_STAGE in status["traceback"] and "json.dump" in status["traceback"],
            "The original failure is not the recorded command-serialization error")
    failed_dir = previous / "stages" / FAILED_STAGE
    require(not any((failed_dir / name).exists() for name in ("command.json", "process.log", "resources.txt")),
            "The failed preview may already have launched a subprocess")
    v1.assert_no_holdout_claims(run, preview)
    for path in [run / "jobs/books-bank-calibration-preview", run / "calibration-preview/books-bank",
                 *(run / name for name in ("protocols", "calibration", "validation", "analysis", "all-native-L-policies.json"))]:
        require(not path.exists(), f"A pending stage already has output; preserve it rather than retry: {path}")
    for name in v1.BUILD_ORDER[3:]:
        prefix = run / "indexes" / name
        require(not any(prefix.parent.glob(prefix.name + "*")), f"A pending index already has output: {prefix}")
    return status


def verify_amendment(args, preview):
    path = args.infrastructure_amendment.resolve()
    require(preview.record(path)["sha256"] == args.infrastructure_amendment_sha256,
            "Infrastructure amendment changed")
    amendment = json.loads(path.read_text())
    require(amendment.get("schema_version") == 1 and amendment.get("status") == "authorized_continuation",
            "Missing explicit authorization for this continuation")
    run = args.run_root.resolve()
    expected = {"original_plan": run / "experiment-plan.json",
                "failed_suite_status": run / "jobs/amazon-f32-suite/status.json",
                "previous_supervisor": args.previous_supervisor.resolve(),
                "continuation_source": Path(__file__).resolve()}
    for key, actual in expected.items():
        require(amendment[key] == preview.record(actual), f"Continuation amendment does not bind {key}")
    expected_manifests = {str((run / "indexes" / (name + ".manifest.json")).resolve()): digest
                          for name, digest in REUSED_MANIFESTS.items()}
    allowed = amendment["allowed_reused_index_manifests"]
    require(len(allowed) == 3 and {item["path"]: item["sha256"] for item in allowed} == expected_manifests,
            "Continuation must authorize exactly the three recorded completed indexes")
    for item in allowed:
        require(preview.record(item["path"]) == item, "An authorized reused manifest changed")
    return amendment


def verify_reused_indexes(run, v1, preview, original_status):
    """Hash each distinct source/input once and every final reused artifact."""
    checked = {}

    def check(item):
        path = Path(item["path"]).resolve()
        if str(path) not in checked:
            checked[str(path)] = preview.record(path)
        actual = checked[str(path)]
        require(all(actual[key] == item[key] for key in ("path", "bytes", "sha256")), f"Reused artifact or input changed: {path}")
        return actual

    for item in original_status["sources"]:
        check(item)
    previous = run / "jobs/amazon-f32-suite/stages"
    for name in ("00-wait-c4-build", "01-wait-c4-preview", "02-c4_global_R64_L100", BANK_STAGE):
        completed = json.loads((previous / name / "status.json").read_text())
        require(completed.get("status") == "complete" and completed.get("exit_code") == 0,
                f"A prerequisite stage is incomplete: {name}")
        require(len(completed.get("artifacts", [])) == 1, "Missing completed prerequisite artifact identity")
        for item in completed["artifacts"]:
            check(item)
        if name in ("02-c4_global_R64_L100", BANK_STAGE):
            recipe = v1.build_commands(run)[0 if name.startswith("02-") else 1]
            require(completed["command"] == normalize_command(recipe["command"]), "Completed build command differs from the frozen recipe")
    manifests = {}
    expected_suffixes = set(preview.load_runner(run / "source/tools/single-hop-benchmark/run_f32_benchmark_v3.py").SUFFIXES)
    for name, digest in REUSED_MANIFESTS.items():
        prefix = run / "indexes" / name
        manifest_path = Path(str(prefix) + ".manifest.json")
        require(preview.record(manifest_path)["sha256"] == digest, "A recorded completed index manifest changed")
        manifest = v1.validate_index_manifest(run, prefix, preview)
        require(set(manifest["artifacts"]) == expected_suffixes, "Reused index has an incomplete final artifact set")
        for suffix, item in manifest["artifacts"].items():
            require(Path(item["path"]).resolve() == Path(str(prefix) + suffix).resolve(), "Reused artifact points outside its declared index")
            check(item)
        require(manifest["inputs_before"] == manifest["inputs_after_encoding"] == manifest["inputs_after"],
                "Construction inputs changed during a completed build")
        for item in manifest["inputs_before"].values():
            check(item)
        manifests[name] = manifest
    filtered, global_graph = (manifests[name] for name in v1.BUILD_ORDER[:2])
    for suffix in ("_pq_pivots.bin", "_pq_compressed.bin"):
        require(filtered["artifacts"][suffix]["sha256"] == global_graph["artifacts"][suffix]["sha256"],
                "C4 filtered/global graphs no longer share their exact PQ pair")
    design = json.loads((run / "experiment-plan.json").read_text())
    for name, key in ((v1.BUILD_ORDER[0], "amazon_c4"), (v1.BUILD_ORDER[2], "books_bank")):
        workload = next(item for item in v1.workloads(run) if item["plan_key"] == key)
        prepared = json.loads((workload["prepared_root"] / "prepared-calibration/manifest.json").read_text())
        manifest = manifests[name]
        require(manifest["base_sha256"] == prepared["artifacts"]["base.fbin"]["sha256"]
                and manifest["labels_sha256"] == prepared["artifacts"]["labels.txt"]["sha256"]
                and manifest["disk_metadata"]["rows"] == design["workloads"][key]["base_vectors"],
                "Reused graph does not bind the planned corpus and predicates")
    return {"status": "complete", "checked_records": list(checked.values()),
            "reused_manifests": [preview.record(run / "indexes" / (name + ".manifest.json")) for name in REUSED_MANIFESTS]}


def planned_stages(run, args, v1, sources):
    """One command recipe for production dispatch and serialization tests."""
    previous = run / "jobs/amazon-f32-suite/stages" / BANK_STAGE / "status.json"
    stages = [{"name": "01-books-bank-calibration-preview", "command": [run / ".venv/bin/python", args.preview_script,
               "--run-root", run, "--workload", "books-bank", "--completed-build-status", previous,
               "--schedule-amendment", args.books_preview_amendment, "--schedule-amendment-sha256", args.books_preview_amendment_sha256],
               "absent": run / "calibration-preview/books-bank", "success": run / "calibration-preview/books-bank/preview.json"}]
    for index, item in enumerate(v1.build_commands(run)[2:], start=2):
        stages.append({"name": f"{index:02d}-{item['name']}", "command": item["command"], "prefix": item["prefix"],
                       "success": Path(str(item["prefix"]) + ".manifest.json")})
    python = run / ".venv/bin/python"
    runner = run / "source/tools/single-hop-benchmark/run_f32_benchmark_v3.py"
    final_sources = [item["path"] for item in sources]
    final_sources += [str(run / "calibration-preview" / name / file) for name in ("c4", "books-bank") for file in ("preview.json", "summary.json")]
    for workload in v1.workloads(run):
        name, prepared = workload["name"], workload["prepared_root"]
        output = run / "protocols" / name
        stages.append({"name": f"freeze-{name}", "command": [python, runner, "--phase", "freeze", "--dataset", name, "--out", output,
               "--prepared", prepared / "prepared-calibration", "--holdout-prepared", prepared / "prepared-validation",
               "--split-commitment", prepared / "split-commitment.json", "--cpp-bin-dir", run.parent / "cpp/build-native/apps",
               "--rust-binary", run / "native-rust/filtered_ssd_f32_v3", "--index-prefix", workload["prefix"],
               "--inline-prefix", workload["global_prefix"], "--sources", *final_sources], "absent": output, "success": output / "protocol.json"})
    for workload in v1.workloads(run):
        name = workload["name"]
        output = run / "calibration" / name
        stages.append({"name": f"calibrate-{name}", "command": [python, runner, "--phase", "calibrate",
               "--protocol", run / "protocols" / name / "protocol.json", "--out", output], "absent": output, "success": output / "calibration.json"})
    stages.append({"name": "freeze-all-L-policies", "action": "freeze_policies", "absent": run / "all-native-L-policies.json",
                   "success": run / "all-native-L-policies.json"})
    for workload in v1.workloads(run):
        name = workload["name"]
        output = run / "validation" / name
        stages.append({"name": f"evaluate-{name}", "command": [python, runner, "--phase", "evaluate",
               "--protocol", run / "protocols" / name / "protocol.json", "--calibration", run / "calibration" / name / "calibration.json",
               "--out", output], "absent": output, "success": output / "evaluation.json", "policy_gate": True})
    for workload in v1.workloads(run):
        name = workload["name"]
        output = run / "analysis" / name
        stages.append({"name": f"analyze-{name}", "command": [python, args.analyzer, "--evaluation", run / "validation" / name / "evaluation.json",
               "--out", output, "--relocate-prefix", f"{v1.MAC_PREFIX}={run.parent}"], "absent": output, "success": output / "summary.json"})
    return stages


class StageExecutor:
    def __init__(self, run, preview, sources, environment):
        self.run, self.preview, self.sources, self.environment = run, preview, sources, environment
        self.job = run / "jobs" / JOB_NAME
        self.job.parent.mkdir(parents=True, exist_ok=True)
        self.job.mkdir()
        self.status = {"status": "running", "started_utc": preview.now(), "pid": os.getpid(), "sources": sources,
                       "stage": "prerequisites", "automatic_retry": False, "continuation_of": str(run / "jobs/amazon-f32-suite/status.json")}
        self.current = None
        self.current_state = None
        save(self.job / "status.json", self.status)

    def check_sources(self):
        for item in self.sources:
            require(self.preview.record(item["path"]) == item, f"Continuation source or runtime dependency changed: {item['path']}")

    def execute(self, stage, action=None, gate=None):
        # A pre-dispatch failure must never relabel a completed previous stage.
        self.current = None
        self.current_state = None
        self.status["stage"] = stage["name"]
        # Normalize BEFORE either state or command JSON is serialized.
        command = normalize_command(stage.get("command"))
        self.check_sources()
        directory = self.job / "stages" / stage["name"]
        directory.mkdir(parents=True)
        self.current = directory
        self.current_state = {"status": "running", "started_utc": self.preview.now(), "command": command}
        save(directory / "status.json", self.current_state)
        self.status.update(stage=stage["name"], stage_started_utc=self.current_state["started_utc"])
        save(self.job / "status.json", self.status, replace=True)
        absent, prefix = stage.get("absent"), stage.get("prefix")
        require(absent is None or not absent.exists(), f"Stage output already exists: {absent}")
        require(prefix is None or not any(prefix.parent.glob(prefix.name + "*")), f"Index prefix already exists: {prefix}")
        if gate:
            gate()
        print(json.dumps({"stage": stage["name"], "status": "running", "started_utc": self.current_state["started_utc"]}), flush=True)
        if command:
            wrapped = ["/usr/bin/time", "-v", "-o", str(directory / "resources.txt"), "/usr/bin/taskset", "-c", "0-7", *command]
            save(directory / "command.json", {"command": wrapped, "environment": {key: self.environment[key] for key in
                 ("LD_LIBRARY_PATH", "OMP_TOOL", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}})
            with (directory / "process.log").open("x") as log:
                process = subprocess.run(wrapped, env=self.environment, cwd=self.run, stdout=log, stderr=subprocess.STDOUT)
            require(process.returncode == 0, f"Stage failed with exit {process.returncode}: {directory}")
        if action:
            action()
        success = stage.get("success")
        require(success is None or success.is_file(), f"Stage did not publish its completion artifact: {success}")
        self.check_sources()
        self.current_state.update(status="complete", exit_code=0, completed_utc=self.preview.now(),
                                  artifacts=[] if success is None else [self.preview.record(success)])
        save(directory / "status.json", self.current_state, replace=True)
        print(json.dumps({"stage": stage["name"], "status": "complete", "completed_utc": self.current_state["completed_utc"]}), flush=True)

    def fail(self):
        self.status.update(status="failed", exit_code=1, completed_utc=self.preview.now(), traceback=traceback.format_exc())
        if self.current is not None:
            self.current_state.update(status="failed", exit_code=1, completed_utc=self.status["completed_utc"], traceback=self.status["traceback"])
            save(self.current / "status.json", self.current_state, replace=True)
        save(self.job / "status.json", self.status, replace=True)


def run_suite(args):
    args.run_root = args.run_root.resolve()
    for key in ("previous_supervisor", "preview_script", "analyzer", "books_preview_amendment", "infrastructure_amendment"):
        setattr(args, key, getattr(args, key).resolve())
    run = args.run_root
    v1 = load_v1(args.previous_supervisor)
    preview = v1.load_preview(args.preview_script)
    original = verify_failure(run, v1, preview)  # Required before claiming the new job.
    verify_amendment(args, preview)
    environment = os.environ.copy()
    environment.update(json.loads((run / "native-helpers-v2/build-plan.json").read_text())["runtime_environment"])
    environment.update(OMP_NUM_THREADS="8", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_TOOL="disabled")
    sources = []
    executor = StageExecutor(run, preview, sources, environment)
    try:
        def preflight():
            v1.preflight(run, preview, args.analyzer, args.books_preview_amendment, args.books_preview_amendment_sha256)
            verification = verify_reused_indexes(run, v1, preview, original)
            save(executor.job / "reused-index-verification.json", verification)
            paths = preview.source_paths(run, args.books_preview_amendment)
            paths += [Path(__file__).resolve(), args.previous_supervisor, args.analyzer, args.infrastructure_amendment,
                      run / "jobs/amazon-f32-suite/status.json", run / "jobs/amazon-f32-suite/stages" / FAILED_STAGE / "status.json",
                      executor.job / "reused-index-verification.json",
                      run / "native-source-correction/cpp-port/build_f32_stitched.py",
                      run / "native-source-correction/cpp-port/build_stitched_f32.cpp",
                      run / "native-source-correction/cpp-port/generate_pq_f32.cpp"]
            sources.extend(preview.record(path) for path in dict.fromkeys(paths))
        executor.execute({"name": "00-verify-continuation-prerequisites", "success": executor.job / "reused-index-verification.json"}, action=preflight)
        stages = planned_stages(run, args, v1, sources)
        save(executor.job / "execution-plan.json", {"stages": stages, "sources": sources,
             "reused_indexes": list(REUSED_MANIFESTS), "holdout_ann_queries_run": 0})
        for stage in stages:
            action = (lambda: v1.freeze_policies(run, preview, sources)) if stage.get("action") == "freeze_policies" else None
            gate = (lambda: v1.verify_policy_checkpoint(run, preview)) if stage.get("policy_gate") else None
            executor.execute(stage, action, gate)
            if stage["name"].startswith(("02-books_original", "03-books_global")):
                v1.validate_index_manifest(run, stage["prefix"], preview)
        executor.status.update(status="complete", exit_code=0, completed_utc=preview.now(),
             results={workload["name"]: preview.record(run / "analysis" / workload["name"] / "summary.json") for workload in v1.workloads(run)})
        save(executor.job / "status.json", executor.status, replace=True)
    except BaseException:
        executor.fail()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--previous-supervisor", type=Path, default=Path(__file__).with_name("run_f32_native_suite.py"))
    parser.add_argument("--preview-script", type=Path, default=Path(__file__).with_name("run_f32_calibration_preview.py"))
    parser.add_argument("--analyzer", type=Path, default=Path(__file__).with_name("analyze_f32_results_v3.py"))
    parser.add_argument("--books-preview-amendment", type=Path, required=True)
    parser.add_argument("--books-preview-amendment-sha256", required=True)
    parser.add_argument("--infrastructure-amendment", type=Path, required=True)
    parser.add_argument("--infrastructure-amendment-sha256", required=True)
    run_suite(parser.parse_args())


if __name__ == "__main__":
    main()
