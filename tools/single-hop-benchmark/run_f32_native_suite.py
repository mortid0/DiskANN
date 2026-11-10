#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Serial native Amazon supervisor; launch under nohup and taskset -c 0-7.

The root-owned C4 build and calibration-only preview must complete first. This
process owns all remaining stages through independent analysis. Each stage has
an exclusive output and retained logs. Failures stop the suite; held-out runs
are never retried, resumed, or deleted automatically.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

PREVIEW_SHA = "627012b78c262ca1b2cb1c7c92ce5538280320ad2e7faf6ca87fd1cc15ac8593"
ANALYZER_SHA = "3955651f31d4c2ebe74712ce0b9fc2644fc7e8ac2430d7ba84ca0ef811e3b14f"
ENGINES = ["cpp", "rust_single", "rust_inline", "rust_inline_global"]
BUILD_ORDER = ["c4_filtered_R32_R128_L100", "c4_global_R64_L100",
               "books_bank_filtered_R32_R128_L100", "books_original_filtered_R32_R128_L100",
               "books_global_R64_L100"]
MAC_PREFIX = "/Users/mortid0/repos/github/DiskANN/benchmark-results/single-hop"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_preview(path):
    spec = importlib.util.spec_from_file_location("frozen_calibration_preview", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.record(path)["sha256"] == PREVIEW_SHA, "Approved preview source changed")
    return module


def workloads(run):
    return [{"name": name, "plan_key": key, "prepared_root": run.parent / prepared,
             "prefix": run / "indexes" / prefix, "global_prefix": run / "indexes" / global_prefix}
            for name, key, prepared, prefix, global_prefix in (
                ("amazon-c4", "amazon_c4", "amazon-c4/data/sealed", BUILD_ORDER[0], BUILD_ORDER[1]),
                ("books-bank", "books_bank", "amazon-books/sealed-bank", BUILD_ORDER[2], BUILD_ORDER[4]),
                ("books-original", "books_original", "amazon-books/sealed-original", BUILD_ORDER[3], BUILD_ORDER[4]))]


def build_commands(run):
    common = [str(run / ".venv/bin/python"), str(run / "native-source-correction/cpp-port/build_f32_stitched.py"),
              "--bin-dir", str(run / "native-helpers-v2/bin"), "--output-dir", str(run / "indexes"),
              "--threads", "8", "--build-l", "100", "--build-evidence", str(run / "native-helpers-v2/build-result.json")]
    commands = []
    for name, prepared, degree, stitched, pq, entry in (
            (BUILD_ORDER[1], run / "input-global-c4", 64, 64, BUILD_ORDER[0], 301656),
            (BUILD_ORDER[2], run.parent / "amazon-books/sealed-bank/prepared-calibration", 32, 128, None, None),
            (BUILD_ORDER[3], run.parent / "amazon-books/sealed-original/prepared-calibration", 32, 128, BUILD_ORDER[2], None),
            (BUILD_ORDER[4], run / "input-global-books", 64, 64, BUILD_ORDER[2], 6481001)):
        command = [*common, "--prepared-data", str(prepared), "--name", name,
                   "--label-degree", str(degree), "--stitched-degree", str(stitched)]
        if pq:
            command += ["--pq-prefix", str(run / "indexes" / pq)]
        else:
            command += ["--training-points", "100000"]
        if entry is not None:
            command += ["--global-entry-point", str(entry), "--global-entry-provenance", str(prepared / "global-entry-proof.json")]
        else:
            command += ["--skip-label-build", "200386"]
        commands.append({"name": name, "command": command, "prefix": run / "indexes" / name})
    return commands


def assert_no_holdout_claims(run, preview):
    for workload in workloads(run):
        claim = preview.claim_path(workload["prepared_root"] / "split-commitment.json")
        require(not claim.exists(), f"A real held-out workload was already claimed: {claim}")


def validate_policy(calibration, protocol, protocol_record):
    require(calibration["protocol"] == protocol_record and calibration["heldout_quality_read"] is False,
            "Calibration does not bind its untouched held-out protocol")
    require(protocol["engines"] == ENGINES and protocol["cpu_affinity"] == list(range(8))
            and protocol["fixed_l"] == 64 and not protocol["functional_only"], "Final protocol differs from the fixed design")
    policy = calibration["target90"]
    require(set(policy) == set(ENGINES), "A workload is missing a calibrated engine policy")
    for engine, selected in policy.items():
        if selected["status"] == "calibrated":
            require(type(selected["l"]) is int and selected["l"] in protocol["calibration_grid"],
                    f"Invalid selected L for {engine}")
        else:
            require(selected["status"] == "unreached_within_grid" and selected["l"] is None,
                    f"Unknown target-90 outcome for {engine}")


def freeze_policies(run, preview, sources):
    assert_no_holdout_claims(run, preview)
    records = {}
    for workload in workloads(run):
        name = workload["name"]
        protocol_path = run / "protocols" / name / "protocol.json"
        calibration_path = run / "calibration" / name / "calibration.json"
        protocol = json.loads(protocol_path.read_text())
        calibration = json.loads(calibration_path.read_text())
        validate_policy(calibration, protocol, preview.record(protocol_path))
        runner = preview.load_runner(run / "source/tools/single-hop-benchmark/run_f32_benchmark_v3.py")
        require(calibration["target90"] == runner.choose_target(calibration["rows"], engines=protocol["engines"]),
                "A calibration no longer selects its minimum passing L")
        runner.check_grid(protocol, calibration["rows"], calibration["target90"])
        records[name] = {"protocol": preview.record(protocol_path), "calibration": preview.record(calibration_path),
                         "target90": calibration["target90"]}
    checkpoint = run / "all-native-L-policies.json"
    preview.save(checkpoint, {"schema_version": 1, "status": "complete", "created_utc": preview.now(),
                 "holdout_ann_queries_run": 0, "all_workloads_calibrated_before_any_holdout": True,
                 "workloads": records, "sources": sources})
    return checkpoint


def verify_policy_checkpoint(run, preview):
    checkpoint = run / "all-native-L-policies.json"
    frozen = json.loads(checkpoint.read_text())
    require(frozen["all_workloads_calibrated_before_any_holdout"] is True and frozen["holdout_ann_queries_run"] == 0,
            "Missing all-workload calibration gate")
    require(set(frozen["workloads"]) == {workload["name"] for workload in workloads(run)}, "Incomplete calibration checkpoint")
    for workload in frozen["workloads"].values():
        for key in ("protocol", "calibration"):
            require(preview.record(workload[key]["path"]) == workload[key], "Frozen L-policy artifact changed")
        require(json.loads(Path(workload["calibration"]["path"]).read_text())["target90"] == workload["target90"],
                "Frozen L values changed")
    for item in frozen["sources"]:
        require(preview.record(item["path"]) == item, "A frozen suite source or dependency changed")
    return checkpoint


def wait_completed(path, preview, poll_seconds):
    while True:
        if Path(path).exists():
            status = json.loads(Path(path).read_text())
            require(status.get("status") in ("running", "complete", "failed"), f"Unknown prerequisite status: {path}")
            require(status["status"] != "failed", f"Prerequisite failed; its evidence is retained: {path}")
            if status["status"] == "complete":
                require(status.get("exit_code") == 0, f"Prerequisite did not exit successfully: {path}")
                for item in status.get("artifacts", []):
                    require(preview.record(item["path"]) == item, "Prerequisite completion artifact changed")
                return status
        time.sleep(poll_seconds)


def preflight(run, preview, analyzer, amendment, amendment_sha):
    require(preview.record(run / "experiment-plan.json")["sha256"] == preview.PLAN_SHA, "Preregistered plan changed")
    require(preview.record(run / "execution-amendment-20261023.json")["sha256"] == preview.AMENDMENT_SHA,
            "C4 preview amendment changed")
    require(preview.record(amendment)["sha256"] == amendment_sha, "Books preview schedule amendment changed")
    plan = json.loads((run / "experiment-plan.json").read_text())
    require(plan["build_order"] == BUILD_ORDER and plan["query"]["cores"] == list(range(8)), "Preregistered execution design differs")
    require(analyzer.is_file(), "The independently reviewed analyzer has not been deployed")
    require(preview.record(analyzer)["sha256"] == ANALYZER_SHA, "The independently reviewed V3 analyzer changed")
    for workload in workloads(run):
        design = plan["workloads"][workload["plan_key"]]
        for key, name in (("calibration_manifest", "prepared-calibration/manifest.json"),
                          ("holdout_manifest", "prepared-validation/manifest.json"), ("split_commitment", "split-commitment.json")):
            actual = preview.record(workload["prepared_root"] / name)
            require(all(actual[field] == design[key][field] for field in ("bytes", "sha256")),
                    f"Prepared workload no longer matches the preregistration: {workload['name']} {key}")
    preview.load_runner(run / "source/tools/single-hop-benchmark/run_f32_benchmark_v3.py")
    assert_no_holdout_claims(run, preview)
    return plan


def validate_index_manifest(run, prefix, preview):
    manifest = json.loads(Path(str(prefix) + ".manifest.json").read_text())
    name = prefix.name
    require(name in BUILD_ORDER and manifest.get("status") == "complete", "Incomplete index construction manifest")
    global_graph = name in (BUILD_ORDER[1], BUILD_ORDER[4])
    require((manifest["label_degree"], manifest["stitched_degree"], manifest["build_l"]) ==
            ((64, 64, 100) if global_graph else (32, 128, 100)), "Index construction parameters differ")
    expected_entry = 301656 if name == BUILD_ORDER[1] else 6481001 if name == BUILD_ORDER[4] else 0
    require(manifest["disk_metadata"]["medoid"] == expected_entry, "Index entry differs from the frozen base-only policy")
    expected_skip = 200386 if name in (BUILD_ORDER[2], BUILD_ORDER[3]) else None
    require(manifest["unqueried_label_graph_omitted"] == expected_skip, "Unexpected predicate graph omission")
    build = json.loads((run / "native-helpers-v2/build-result.json").read_text())
    for executable, key in (("build_stitched_f32", "build_stitched_f32"), ("utils/create_disk_layout", "create_disk_layout")):
        require(manifest["build_binaries_sha256"][executable] == build["binaries"][key]["sha256"],
                "Index construction used an unexpected binary")
    require(manifest["inputs_before"]["source:build_evidence"] == preview.record(run / "native-helpers-v2/build-result.json")
            or all(manifest["inputs_before"]["source:build_evidence"][key] == preview.record(run / "native-helpers-v2/build-result.json")[key]
                   for key in ("path", "bytes", "sha256")), "Index construction evidence differs")
    return manifest


def run_suite(args):
    run = args.run_root.resolve()
    preview_path = args.preview_script.resolve()
    analyzer = args.analyzer.resolve()
    preview = load_preview(preview_path)
    sources = []  # Heavy provenance hashing starts only after the C4 preview ends.
    environment = os.environ.copy()
    environment.update(json.loads((run / "native-helpers-v2/build-plan.json").read_text())["runtime_environment"])
    environment.update(OMP_NUM_THREADS="8", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_TOOL="disabled")
    job = run / "jobs/amazon-f32-suite"
    job.parent.mkdir(parents=True, exist_ok=True)
    job.mkdir()  # Exclusive suite claim. Existing failed or completed jobs are never reused.
    status = {"status": "running", "started_utc": preview.now(), "pid": os.getpid(), "sources": sources,
              "stage": "prerequisites", "automatic_retry": False}
    preview.save(job / "status.json", status)
    current = None

    def check_sources():
        for item in sources:
            require(preview.record(item["path"]) == item, f"Suite source or runtime dependency changed: {item['path']}")

    def stage(name, command=None, action=None, absent=None, prefix=None, success=None):
        nonlocal current
        current = job / "stages" / name
        check_sources()
        current.mkdir(parents=True)
        state = {"status": "running", "started_utc": preview.now(), "command": command}
        preview.save(current / "status.json", state)
        status.update(stage=name, stage_started_utc=state["started_utc"])
        preview.save(job / "status.json", status, replace=True)
        require(absent is None or not absent.exists(), f"Stage output already exists: {absent}")
        require(prefix is None or not any(prefix.parent.glob(prefix.name + "*")), f"Index prefix already exists: {prefix}")
        print(json.dumps({"stage": name, "status": "running", "started_utc": state["started_utc"]}), flush=True)
        if command:
            wrapped = ["/usr/bin/time", "-v", "-o", str(current / "resources.txt"),
                       "/usr/bin/taskset", "-c", "0-7", *map(str, command)]
            preview.save(current / "command.json", {"command": wrapped, "environment": {
                         key: environment[key] for key in ("LD_LIBRARY_PATH", "OMP_TOOL", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}})
            with (current / "process.log").open("x") as log:
                process = subprocess.run(wrapped, env=environment, cwd=run, stdout=log, stderr=subprocess.STDOUT)
            require(process.returncode == 0, f"Stage failed with exit {process.returncode}: {current}")
        if action:
            action()
        require(success is None or success.is_file(), f"Stage did not publish its completion artifact: {success}")
        check_sources()
        state.update(status="complete", exit_code=0, completed_utc=preview.now(),
                     artifacts=[] if success is None else [preview.record(success)])
        preview.save(current / "status.json", state, replace=True)
        print(json.dumps({"stage": name, "status": "complete", "completed_utc": state["completed_utc"]}), flush=True)

    try:
        stage("00-wait-c4-build", action=lambda: wait_completed(run / "jobs/c4-filtered-build/status.json", preview, args.poll_seconds),
              success=Path(str(run / "indexes" / BUILD_ORDER[0]) + ".manifest.json"))
        stage("01-wait-c4-preview", action=lambda: wait_completed(run / "jobs/c4-calibration-preview/status.json", preview, args.poll_seconds),
              success=run / "calibration-preview/c4/preview.json")
        preflight(run, preview, analyzer, args.books_preview_amendment, args.books_preview_amendment_sha256)
        paths = preview.source_paths(run, args.books_preview_amendment)
        paths += [Path(__file__).resolve(), analyzer,
                  run / "native-source-correction/cpp-port/build_f32_stitched.py",
                  run / "native-source-correction/cpp-port/build_stitched_f32.cpp",
                  run / "native-source-correction/cpp-port/generate_pq_f32.cpp"]
        sources.extend(preview.record(path) for path in dict.fromkeys(paths))
        validate_index_manifest(run, run / "indexes" / BUILD_ORDER[0], preview)
        preview.save(job / "execution-plan.json", {"builds": [{**item, "prefix": str(item["prefix"])} for item in build_commands(run)],
                     "workloads": [item["name"] for item in workloads(run)], "sources": sources,
                     "order": "C4 preview; C4 global; Books bank; bank preview; Books original; Books global; freeze all; calibrate all; freeze all policies; evaluate all; analyze all"})
        for index, item in enumerate(build_commands(run), start=2):
            stage(f"{index:02d}-{item['name']}", command=item["command"], prefix=item["prefix"],
                  success=Path(str(item["prefix"]) + ".manifest.json"))
            validate_index_manifest(run, item["prefix"], preview)
            if item["name"] == BUILD_ORDER[2]:
                bank_status = current / "status.json"
                stage("03b-books-bank-calibration-preview", command=[run / ".venv/bin/python", preview_path,
                      "--run-root", run, "--workload", "books-bank", "--completed-build-status", bank_status,
                      "--schedule-amendment", args.books_preview_amendment,
                      "--schedule-amendment-sha256", args.books_preview_amendment_sha256],
                      absent=run / "calibration-preview/books-bank", success=run / "calibration-preview/books-bank/preview.json")
        assert_no_holdout_claims(run, preview)
        full_sources = [item["path"] for item in sources]
        full_sources += [str(run / "calibration-preview" / name / file) for name in ("c4", "books-bank")
                         for file in ("preview.json", "summary.json")]
        python = run / ".venv/bin/python"
        runner = run / "source/tools/single-hop-benchmark/run_f32_benchmark_v3.py"
        for workload in workloads(run):
            name, prepared = workload["name"], workload["prepared_root"]
            output = run / "protocols" / name
            stage(f"freeze-{name}", command=[python, runner, "--phase", "freeze", "--dataset", name, "--out", output,
                  "--prepared", prepared / "prepared-calibration", "--holdout-prepared", prepared / "prepared-validation",
                  "--split-commitment", prepared / "split-commitment.json", "--cpp-bin-dir", run.parent / "cpp/build-native/apps",
                  "--rust-binary", run / "native-rust/filtered_ssd_f32_v3", "--index-prefix", workload["prefix"],
                  "--inline-prefix", workload["global_prefix"], "--sources", *full_sources], absent=output,
                  success=output / "protocol.json")
        for workload in workloads(run):
            name = workload["name"]
            output = run / "calibration" / name
            stage(f"calibrate-{name}", command=[python, runner, "--phase", "calibrate",
                  "--protocol", run / "protocols" / name / "protocol.json", "--out", output],
                  absent=output, success=output / "calibration.json")
        stage("freeze-all-L-policies", action=lambda: freeze_policies(run, preview, sources),
              absent=run / "all-native-L-policies.json", success=run / "all-native-L-policies.json")
        for workload in workloads(run):
            verify_policy_checkpoint(run, preview)
            name = workload["name"]
            output = run / "validation" / name
            stage(f"evaluate-{name}", command=[python, runner, "--phase", "evaluate",
                  "--protocol", run / "protocols" / name / "protocol.json", "--calibration", run / "calibration" / name / "calibration.json",
                  "--out", output], absent=output, success=output / "evaluation.json")
        for workload in workloads(run):
            name = workload["name"]
            output = run / "analysis" / name
            stage(f"analyze-{name}", command=[python, analyzer, "--evaluation", run / "validation" / name / "evaluation.json",
                  "--out", output, "--relocate-prefix", f"{MAC_PREFIX}={run.parent}"], absent=output,
                  success=output / "summary.json")
        status.update(status="complete", exit_code=0, completed_utc=preview.now(),
                      results={workload["name"]: preview.record(run / "analysis" / workload["name"] / "summary.json")
                               for workload in workloads(run)})
    except BaseException:
        status.update(status="failed", exit_code=1, completed_utc=preview.now(), traceback=traceback.format_exc())
        if current and current.is_dir():
            preview.save(current / "status.json", {"status": "failed", "exit_code": 1,
                         "completed_utc": preview.now(), "traceback": status["traceback"]}, replace=True)
        raise
    finally:
        preview.save(job / "status.json", status, replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--preview-script", type=Path, default=Path(__file__).with_name("run_f32_calibration_preview.py"))
    parser.add_argument("--analyzer", type=Path, default=Path(__file__).with_name("analyze_f32_results_v3.py"))
    parser.add_argument("--books-preview-amendment", type=Path, required=True)
    parser.add_argument("--books-preview-amendment-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.)
    args = parser.parse_args()
    require(1 <= args.poll_seconds <= 60, "Polling must stay between 1 and 60 seconds")
    run_suite(args)


if __name__ == "__main__":
    main()
