#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Run the approved C4 calibration-only preview; never invoke held-out evaluation.

Launch this foreground process under nohup (or another detached job manager).
Its exclusive job directory prevents duplicate runs. Failed output is retained;
there is deliberately no resume, deletion, or automatic retry mechanism.
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import traceback

PLAN_SHA = "5272163d12d87ef9bdc8d2076a4362c0c7af38cb04ebe3d6d01d86df4cd30452"
AMENDMENT_SHA = "e8e7696c29b005ce8a07ae5e5a991abd2efa647585c668110717c45f94a9aeef"
RUNNER_SHA = "9c33f3c59685e2f9b1fd04a130610097af393b6798a4d404da8fc580ff84e62e"
ENGINES = ("cpp", "rust_single", "rust_inline")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def save(path, value, replace=False):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}") if replace else path
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)


def runtime_source_paths(run):
    proof = json.loads((run / "native-helpers-v2/build-result.json").read_text())
    expected = {}
    for component in proof["runtime_dependencies"].values():
        for item in component["resolved_dependencies"].values():
            expected[item["path"]] = item
    plan = json.loads((run / "native-helpers-v2/build-plan.json").read_text())
    for item in plan["linked_libraries"]:
        expected[item["path"]] = item
    for path, item in expected.items():
        require(record(path) == item, f"Native dependency changed since the successful smoke: {path}")
    environment = os.environ.copy()
    environment.update(plan["runtime_environment"])
    for binary in (run.parent / "cpp/build-native/apps/search_disk_index", run / "native-rust/filtered_ssd_f32_v3"):
        linked = subprocess.check_output(["ldd", str(binary)], env=environment, text=True)
        require("not found" not in linked, f"A native query dependency is missing: {binary}")
        for path in re.findall(r"(?:=>\s+|^\s*)(/\S+)", linked, re.MULTILINE):
            item = record(path)
            if item["path"] in expected:
                require(item == expected[item["path"]], f"Live query dependency differs: {path}")
            expected[item["path"]] = item
    return [Path(path) for path in sorted(expected)]


def source_paths(run, schedule_amendment=None):
    paths = [run / "source/amazon-f32-suite-source.tar.gz",
            run / "native-source-correction/source-correction.tar.gz",
            run / "native-source-correction/amendment.json",
            run / "experiment-plan.json", run / "execution-amendment-20261023.json",
            run / "native-helpers-v2/build-plan.json", run / "native-helpers-v2/build-result.json",
            run / "native-rust/build-result.json", run / "python-environment.json",
            run / "native-functional-proof.json", Path(__file__).resolve()]
    if schedule_amendment:
        paths.append(Path(schedule_amendment))
    return paths + runtime_source_paths(run)


def layout(run, workload="c4"):
    require(workload in ("c4", "books-bank"), "Only the approved C4 and Books bank previews are supported")
    base = run.parent / ("amazon-c4/data/sealed" if workload == "c4" else "amazon-books/sealed-bank")
    prefix = "c4_filtered_R32_R128_L100" if workload == "c4" else "books_bank_filtered_R32_R128_L100"
    return {"runner": run / "source/tools/single-hop-benchmark/run_f32_benchmark_v3.py",
            "python": run / ".venv/bin/python",
            "prepared": base / "prepared-calibration", "holdout": base / "prepared-validation",
            "commitment": base / "split-commitment.json", "prefix": run / "indexes" / prefix,
            "job": run / "jobs" / f"{workload}-calibration-preview", "output": run / "calibration-preview" / workload}


def claim_path(commitment):
    return Path(commitment).with_suffix(".f32-evaluation-claim.json")


def freeze_command(run, paths, sources, workload="c4"):
    return [str(paths["python"]), str(paths["runner"]), "--phase", "freeze",
            "--dataset", f"amazon-{workload}-calibration-preview", "--out", str(paths["output"] / "protocol"),
            "--prepared", str(paths["prepared"]), "--holdout-prepared", str(paths["holdout"]),
            "--split-commitment", str(paths["commitment"]), "--cpp-bin-dir", str(run.parent / "cpp/build-native/apps"),
            "--rust-binary", str(run / "native-rust/filtered_ssd_f32_v3"),
            "--index-prefix", str(paths["prefix"]), "--sources", *map(str, sources)]


def load_runner(path):
    require(record(path)["sha256"] == RUNNER_SHA, "The frozen V3 runner changed")
    spec = importlib.util.spec_from_file_location("frozen_f32_preview_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(rows, protocol, count):
    summary = []
    for scope in ("fixed_l", "target90"):
        for engine in protocol["engines"]:
            group = [row for row in rows if row["scope"] == scope and row["engine"] == engine]
            if not group:
                require(scope == "target90", "Missing fixed-L preview engine")
                summary.append({"scope": scope, "engine": engine, "status": "unreached_within_calibration_grid"})
                continue
            require(len(group) == len(protocol["evaluation_orders"]), "Incomplete balanced preview repeats")
            require(len({row["recall"] for row in group}) == 1, "Preview repeated quality changed")
            labels = []
            for label in sorted(item["label"] for item in group[0]["labels"]):
                slices = [next(item for item in row["labels"] if item["label"] == label) for row in group]
                require(len({item["recall"] for item in slices}) == 1, "Per-label preview quality changed")
                labels.append({"label": label, "queries": slices[0]["n"], "eligible": slices[0]["eligible"],
                               "recall": slices[0]["recall"],
                               "p99_us_median": statistics.median(item["p99_us"] for item in slices),
                               "p99_us_range": [min(item["p99_us"] for item in slices), max(item["p99_us"] for item in slices)]})
            summary.append({"scope": scope, "engine": engine, "l": group[0]["l"], "recall": group[0]["recall"],
                            "qps_median": statistics.median(row["qps"] for row in group),
                            "qps_range": [min(row["qps"] for row in group), max(row["qps"] for row in group)],
                            "p99_us_median": statistics.median(row["p99_us"] for row in group),
                            "p99_us_range": [min(row["p99_us"] for row in group), max(row["p99_us"] for row in group)], "labels": labels})
    return {"phase": "calibration_preview", "data_split": "calibration", "tuning_data": True,
            "holdout_ann_queries_run": 0, "quality_queries": count,
            "timing_repeats": len(protocol["evaluation_orders"]),
            "timing_authoritative": True, "closed_loop_no_queue_or_network": True, "summary": summary}


def run_preview(run, workload="c4", completed_build_status=None, schedule_amendment=None, schedule_amendment_sha=None):
    run = Path(run).resolve()
    paths = layout(run, workload)
    require(record(run / "experiment-plan.json")["sha256"] == PLAN_SHA, "Experiment plan changed")
    require(record(run / "execution-amendment-20261023.json")["sha256"] == AMENDMENT_SHA,
            "Approved preview execution amendment changed")
    if workload == "books-bank":
        require(completed_build_status and schedule_amendment and schedule_amendment_sha,
                "Books preview requires its completed build status and separately frozen schedule amendment")
    if schedule_amendment:
        require(record(schedule_amendment)["sha256"] == schedule_amendment_sha, "Preview schedule amendment changed")
    require(not claim_path(paths["commitment"]).exists(), "The real workload holdout was already claimed")
    first = json.loads(Path(completed_build_status or run / "jobs/c4-filtered-build/status.json").read_text())
    require(first.get("status") == "complete" and first.get("exit_code") == 0, "The filtered build is incomplete or failed")
    manifest = json.loads(Path(str(paths["prefix"]) + ".manifest.json").read_text())
    require(manifest.get("status") == "complete", "The first C4 index lacks a complete construction manifest")
    runner = load_runner(paths["runner"])
    sources = [record(path) for path in source_paths(run, schedule_amendment)]
    paths["job"].parent.mkdir(parents=True, exist_ok=True)
    paths["job"].mkdir()  # Exclusive claim, including after an unsuccessful attempt.
    status = {"status": "running", "phase": "calibration_preview", "started_utc": now(), "pid": os.getpid(),
              "holdout_ann_queries_run": 0, "sources": sources}
    save(paths["job"] / "status.json", status)
    try:
        require(not paths["output"].exists(), "Preview output already exists")
        paths["output"].mkdir(parents=True)
        environment = os.environ.copy()
        environment.update(json.loads((run / "native-helpers-v2/build-plan.json").read_text())["runtime_environment"])
        environment.update(OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        os.environ.update({key: value for key, value in environment.items() if key in
                           ("LD_LIBRARY_PATH", "OMP_TOOL", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
        commands = [freeze_command(run, paths, [item["path"] for item in sources], workload),
                    [str(paths["python"]), str(paths["runner"]), "--phase", "calibrate",
                     "--protocol", str(paths["output"] / "protocol/protocol.json"),
                     "--out", str(paths["output"] / "calibration")]]
        for index, command in enumerate(commands):
            stage = paths["job"] / ("freeze" if index == 0 else "calibrate")
            stage.mkdir()
            save(stage / "command.json", {"command": command, "started_utc": now()})
            with (stage / "process.log").open("x") as log:
                process = subprocess.run(["/usr/bin/time", "-v", "-o", str(stage / "resources.txt"), *command],
                                         env=environment, stdout=log, stderr=subprocess.STDOUT, cwd=run)
            require(process.returncode == 0, f"Preview {stage.name} failed; see {stage / 'process.log'}")
        protocol_path = paths["output"] / "protocol/protocol.json"
        calibration_path = paths["output"] / "calibration/calibration.json"
        protocol = json.loads(protocol_path.read_text())
        calibration = json.loads(calibration_path.read_text())
        require(protocol["engines"] == list(ENGINES) and len(protocol["evaluation_orders"]) == 6,
                "Preview must retain the three shared-graph controls and all six orders")
        require(calibration["protocol"] == record(protocol_path) and calibration["heldout_quality_read"] is False,
                "Preview calibration identity differs")
        require(calibration["target90"] == runner.choose_target(calibration["rows"], engines=protocol["engines"]),
                "Preview L policy changed")
        runner.check_grid(protocol, calibration["rows"], calibration["target90"])
        runner.check_frozen(protocol)
        runner.affinity(protocol)
        n, _, _ = runner.verify_prepared(paths["prepared"])
        rows = []
        for repeat, order in enumerate(protocol["evaluation_orders"]):
            for scope in ("fixed_l", "target90"):
                for engine in order:
                    breadth = protocol["fixed_l"] if scope == "fixed_l" else calibration["target90"][engine]["l"]
                    if breadth is None:
                        continue
                    row = runner.run_engine(protocol, paths["prepared"],
                                            paths["output"] / f"timings/r{repeat}/{scope}/{engine}", engine, breadth, True)
                    require(row["queries"] == n and row["timing_authoritative"] is True, "Incomplete preview timing")
                    row.update(phase="calibration_preview", scope=scope, paired_repeat=repeat,
                               order_in_repeat=order.index(engine), data_split="calibration")
                    rows.append(row)
                    print(json.dumps({"phase": "calibration_preview", "repeat": repeat, "scope": scope,
                                      "engine": engine, "l": breadth, "recall": row["recall"],
                                      "qps": row["qps"], "p99_us": row["p99_us"]}), flush=True)
        runner.check_frozen(protocol)
        runner.verify_prepared(paths["prepared"])
        require(not claim_path(paths["commitment"]).exists(), "Preview unexpectedly encountered a holdout claim")
        save(paths["output"] / "preview.json", {"phase": "calibration_preview", "data_split": "calibration",
             "tuning_data": True, "holdout_ann_queries_run": 0, "protocol": record(protocol_path),
             "calibration": record(calibration_path), "queries": n, "rows": rows, "completed_utc": now(),
             "timing_authoritative": True})
        save(paths["output"] / "summary.json", summarize(rows, protocol, n))
        status.update(status="complete", exit_code=0, completed_utc=now(),
                      artifacts=[record(paths["output"] / name) for name in ("preview.json", "summary.json")])
    except BaseException:
        status.update(status="failed", exit_code=1, completed_utc=now(), traceback=traceback.format_exc())
        raise
    finally:
        save(paths["job"] / "status.json", status, replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workload", choices=("c4", "books-bank"), default="c4")
    parser.add_argument("--completed-build-status", type=Path)
    parser.add_argument("--schedule-amendment", type=Path)
    parser.add_argument("--schedule-amendment-sha256")
    args = parser.parse_args()
    run_preview(args.run_root, args.workload, args.completed_build_status,
                args.schedule_amendment, args.schedule_amendment_sha256)


if __name__ == "__main__":
    main()
