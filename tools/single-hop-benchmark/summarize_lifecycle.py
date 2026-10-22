#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Validate complete native Rust build/update/compact/reopen experiment logs.

Each input remains an independent run, allowing 1M and 10M evidence in one
summary without pooling distinct datasets. Each run must contain four natural
query phases, three search breadths and three timing repeats per configuration.
An adjacent .resources.txt file is included when available.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


PHASES = ("before_updates", "after_updates", "after_compact", "after_reopen")
PROBES = ("after_updates", "old_updated_snapshot_after_compact", "after_compact", "after_reopen")
VALIDITY = ("filter_violations", "duplicate_results", "short_results")
METRICS = ("recall", "qps", "mean_us", "p50_us", "p95_us", "p99_us", "wall_seconds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right):
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)


def provenance(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def one(rows, description):
    require(len(rows) == 1, f"Exactly one {description} required")
    return rows[0]


def integer(value):
    return type(value) is int and value >= 0


def zero_validity(row):
    require(all(type(row.get(name)) is int and row[name] == 0 for name in VALIDITY),
            "Invalid, duplicate, short or unvalidated results")


def phase_summary(rows):
    first = rows[0]
    result = {name: first[name] for name in ("phase", "l", "k", "threads", "queries", "sequence", "prepared_identity")}
    result.update({"repetitions": 3, "warmup_queries_per_repeat": first["warmup_queries"],
                   "timing_authoritative": all(row["timing_authoritative"] for row in rows),
                   "validation": {name: 0 for name in VALIDITY}})
    for metric in METRICS:
        values = [row[metric] for row in rows]
        result[metric] = statistics.median(values)
        result[metric + "_range"] = [min(values), max(values)]
    for metric in ("exact_fallback_queries", "base_searches", "delta_candidates"):
        values = [row[metric] for row in rows]
        result[metric + "_per_repeat"] = values
        result[metric + "_median"] = statistics.median(values)
    result["labels"] = []
    for label in sorted(item["label"] for item in first["labels"]):
        items = [next(item for item in row["labels"] if item["label"] == label) for row in rows]
        result["labels"].append({"label": label, "queries": items[0]["queries"],
            **{metric: statistics.median(item[metric] for item in items) for metric in ("recall", "mean_us", "p95_us")}})
    return result


def quality_difference(before, after):
    require((before["l"], before["k"], before["threads"], before["queries"], before["prepared_identity"])
            == (after["l"], after["k"], after["threads"], after["queries"], after["prepared_identity"]),
            "Quality comparison workloads differ")
    before_labels = {item["label"]: item for item in before["labels"]}
    require(before_labels.keys() == {item["label"] for item in after["labels"]}, "Quality comparison labels differ")
    differences = []
    for item in after["labels"]:
        old = before_labels[item["label"]]
        require(old["queries"] == item["queries"], "Quality comparison label sample sizes differ")
        differences.append({"label": item["label"], "queries": item["queries"],
            "before_strict_id_recall": old["recall"], "after_strict_id_recall": item["recall"],
            "difference_percentage_points": 100 * (item["recall"] - old["recall"])})
    return {"l": before["l"], "before_phase": before["phase"], "after_phase": after["phase"],
            "before_strict_id_recall": before["recall"], "after_strict_id_recall": after["recall"],
            "difference_percentage_points": 100 * (after["recall"] - before["recall"]),
            "worst_label_decline": min(differences, key=lambda item: item["difference_percentage_points"]),
            "largest_label_increase": max(differences, key=lambda item: item["difference_percentage_points"]),
            "labels_with_lower_recall": sum(item["difference_percentage_points"] < -1e-10 for item in differences),
            "labels_with_higher_recall": sum(item["difference_percentage_points"] > 1e-10 for item in differences),
            "labels": differences}


def resources(path):
    if not path.exists():
        return {"available": False}
    fields = {}
    for line in path.read_text().splitlines():
        if ": " in line:
            key, value = line.strip().rsplit(": ", 1)
            fields[key] = value
    require(fields.get("Exit status") == "0", "Resource wrapper reports failed process")
    elapsed = fields.get("Elapsed (wall clock) time (h:mm:ss or m:ss)")
    require(elapsed is not None, "Resource wrapper wall clock unavailable")
    wall = 0.0
    for component in elapsed.split(":"):
        wall = 60 * wall + float(component)
    rss = int(fields["Maximum resident set size (kbytes)"])
    return {"available": True, "source": provenance(path), "wall_seconds": wall,
            "maximum_rss_kib": rss, "maximum_rss_gib": rss / 1024**2,
            "user_seconds": float(fields["User time (seconds)"]),
            "system_seconds": float(fields["System time (seconds)"]),
            "scope": "whole lifecycle process: input load, build, queries, updates, retained old snapshots, compaction, reopen and validation; not search-only RSS"}


def summarize_run(path, allow_diagnostic=False):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(all(row.get("kind") in ("experiment", "build", "query", "lifecycle", "probe") for row in rows),
            "Unknown or missing row kind")
    experiment = one([row for row in rows if row["kind"] == "experiment"], "experiment row")
    require(experiment.get("action") == "all", "A complete --action all run is required")
    require(experiment.get("cache_bypass") is True and experiment.get("shared_cache_nodes") == 0,
            "Unexpected lifecycle graph cache protocol")
    require(experiment.get("storage") in ("native_io_uring", "positional_pread")
            and experiment.get("warmup") == "full_same_process_same_worker_count", "Missing native storage/warmup protocol")
    build = one([row for row in rows if row["kind"] == "build"], "build row")
    require((build["rows"], build["dimensions"]) == (experiment["n_base"], experiment["dimensions"]), "Build and experiment shapes differ")
    actions = {}
    for row in rows:
        if row["kind"] == "lifecycle":
            require(row.get("action") not in actions, "Repeated lifecycle action")
            actions[row["action"]] = row
    require(actions.keys() == {"apply", "compact", "reopen", "collect_garbage"}, "Incomplete lifecycle actions")
    sequence = actions["apply"]["sequence"]
    require(type(sequence) is int and sequence == 1 and type(actions["reopen"]["sequence"]) is int
            and actions["reopen"]["sequence"] == sequence,
            "Mutation sequence was not durably reopened")
    require(integer(actions["compact"]["generation"]) and actions["compact"]["generation"] > 1,
            "Compaction did not publish a new generation")
    gc = actions["collect_garbage"]
    require(gc.get("durable_reopen_verified") is True and gc.get("old_snapshots_consistent") is True
            and integer(gc.get("removed")), "Missing old-snapshot/reopen/GC checks")
    require(experiment["update_scope"] == "unqueried_natural_rows_and_fresh_label; supplied_natural_GT_remains_current"
            and actions["apply"]["scope"] == "rows_outside_queried_natural_labels_plus_new_probe_label",
            "Scenario does not establish current natural ground truth")
    mutation_count = actions["apply"]["mutations"]
    require(type(mutation_count) is int and mutation_count >= 4 and mutation_count % 4 == 0, "Unexpected mutation scenario size")
    probes = [row for row in rows if row["kind"] == "probe"]
    require(len(probes) == 4 and {row["phase"] for row in probes} == set(PROBES), "Incomplete exact probes")
    for row in probes:
        zero_validity(row)
        require(row["sequence"] == sequence and row["recall"] == 1.0 and row["point_record_checks"] is True,
                "Changed-record probe failed")
        require(row["truth"] == "independent_integer_squared_l2_then_stable_id", "Probe truth is not independent integer L2")
        require(row["live_records"] == mutation_count * 3 // 4 and row["deleted_records"] == mutation_count // 4
                and row["queries"] == row["live_records"] and row["dimensions"] == experiment["dimensions"], "Probe mutation/shape counts differ")
        require(row["l"] == row["live_records"] and 0 < row["k"] <= row["live_records"]
                and type(row["label"]) is int and 0 <= row["label"] <= 2**32-1,
                "Probe did not use exhaustive label breadth or a valid stable label")
        require(all(row[name] == probes[0][name] for name in ("label", "k", "l", "live_records", "deleted_records")),
                "Probe workload changed between lifecycle stages")
        require(integer(row["exact_fallback_queries"]) and row["exact_fallback_queries"] <= row["queries"], "Invalid probe fallback count")
    query_rows = [row for row in rows if row["kind"] == "query"]
    require(len(query_rows) == 36, "Expected 36 query rows: 4 phases * 3 breadths * 3 repeats")
    first = query_rows[0]
    fields = ("implementation", "k", "threads", "queries", "prepared_identity", "latency_contract", "strict_id_recall")
    groups = defaultdict(list)
    for row in query_rows:
        require(row.get("label_l_plan") is None, "Fixed-L lifecycle summary rejects per-label breadth plans")
        require(all(item.get("actual_l", row["l"]) == row["l"] for item in row["labels"]),
                "Fixed-L lifecycle row contains mixed per-label breadths")
        zero_validity(row)
        require(row["phase"] in PHASES and all(row.get(key) == first.get(key) for key in fields), "Natural query protocol differs")
        require(row["implementation"] == "rust_native_lifecycle" and row["strict_id_recall"] is True
                and row["latency_contract"] == "public_snapshot_search_call", "Unexpected lifecycle search metric")
        require(type(row["threads"]) is int and 0 < row["threads"] <= experiment["index_threads"], "Invalid query concurrency")
        require(type(row.get("timing_authoritative")) is bool
                and (row["timing_authoritative"] or allow_diagnostic), "Non-authoritative or missing lifecycle timing status")
        require(type(row["queries"]) is int and row["queries"] > 0 and row["warmup_queries"] == row["queries"], "Incomplete same-process warmup")
        require(type(row["k"]) is int and type(row["l"]) is int and 0 < row["k"] <= row["l"] < 2**32,
                "Invalid K or breadth")
        require(type(row["repeat"]) is int and type(row["sequence"]) is int, "Missing integer repeat or sequence")
        expected_sequence = 0 if row["phase"] == "before_updates" else sequence
        require(row["sequence"] == expected_sequence, "Query sequence does not match its phase")
        require(all(type(row[name]) in (int, float) and math.isfinite(row[name]) for name in METRICS)
                and 0 <= row["recall"] <= 1 and row["qps"] > 0 and row["wall_seconds"] > 0,
                "Invalid natural query metrics")
        require(0 < row["p50_us"] <= row["p95_us"] <= row["p99_us"] and row["mean_us"] > 0,
                "Invalid latency percentiles")
        require(math.isclose(row["qps"], row["queries"] / row["wall_seconds"], rel_tol=1e-12), "QPS/wall clock differ")
        require(all(integer(row[name]) for name in ("exact_fallback_queries", "base_searches", "delta_candidates"))
                and row["exact_fallback_queries"] <= row["queries"], "Invalid query work counters")
        require(row["delta_candidates"] == 0, "The declared natural workload unexpectedly has matching delta records")
        labels = row["labels"]
        require(len({item["label"] for item in labels}) == len(labels) and sum(item["queries"] for item in labels) == row["queries"],
                "Per-label query sample is incomplete")
        require(all(type(item["queries"]) is int and item["queries"] > 0 and 0 <= item["recall"] <= 1 for item in labels), "Invalid label metrics")
        require(all(all(type(item[metric]) in (int, float) and math.isfinite(item[metric]) and item[metric] > 0
                        for metric in ("mean_us", "p95_us")) for item in labels), "Invalid per-label latencies")
        require(close(sum(item["recall"] * item["queries"] for item in labels) / row["queries"], row["recall"]), "Label/aggregate recall differs")
        require(not {item["label"] for item in labels}.intersection(probe["label"] for probe in probes), "Probe label overlaps natural labels")
        groups[(row["phase"], row["l"])].append(row)
    breadths = sorted({row["l"] for row in query_rows})
    require(len(breadths) == 3 and set(groups) == {(phase, l) for phase in PHASES for l in breadths}, "Phase/breadth matrix is incomplete")
    for observations in groups.values():
        require(len(observations) == 3 and sorted(row["repeat"] for row in observations) == [0, 1, 2], "Exactly 3 unique repeats required per phase/L")
        label_reference = {item["label"]: item for item in observations[0]["labels"]}
        for row in observations:
            labels = {item["label"]: item for item in row["labels"]}
            require(label_reference.keys() == labels.keys(), "Label set changed across repeats")
            require(all(item["queries"] == label_reference[label]["queries"]
                        and close(item["recall"], label_reference[label]["recall"]) for label, item in labels.items()),
                    "Repeated per-label recall or sample differs")
    summaries = {(phase, l): phase_summary(groups[phase, l]) for phase in PHASES for l in breadths}
    compaction_quality = [quality_difference(summaries["after_updates", l], summaries["after_compact", l]) for l in breadths]
    update_quality = [quality_difference(summaries["before_updates", l], summaries["after_updates", l]) for l in breadths]
    reopen_quality = [quality_difference(summaries["after_compact", l], summaries["after_reopen", l]) for l in breadths]
    require(all(close(item["difference_percentage_points"], 0) and all(close(label["difference_percentage_points"], 0)
                for label in item["labels"]) for item in reopen_quality), "Persisted graph recall differs after reopen")
    worst = min(({"l": item["l"], **item["worst_label_decline"]} for item in compaction_quality), key=lambda item: item["difference_percentage_points"])
    times = {"initial_build_seconds_including_input_load": build["seconds_including_input_load"],
             "durable_apply_seconds": actions["apply"]["seconds"], "compaction_seconds": actions["compact"]["seconds"],
             "reopen_seconds": actions["reopen"]["seconds"]}
    require(all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in times.values()), "Invalid lifecycle durations")
    return {"status": "all assertions passed", "source": provenance(path), "experiment": experiment, "build": build,
            "timings": times, "whole_process_resources": resources(path.with_suffix(".resources.txt")),
            "validation": {"natural_query_rows": 36, "phases": list(PHASES), "breadths": breadths, "repeats_per_configuration": 3,
                "all_validity_counters_zero": True, "probe_exact_recall": 1.0, "probe_phases": list(PROBES),
                "old_snapshots_consistent": True, "durable_reopen_verified": True, "reopened_strict_recall_matches_compacted_per_label": True,
                "initial_sequence": 0, "committed_and_reopened_sequence": sequence, "compacted_generation": actions["compact"]["generation"],
                "collected_generations": gc["removed"], "authoritative_query_timing": all(row["timing_authoritative"] for row in query_rows)},
            "updates": {"mutations": mutation_count, "existing_vector_replacements": mutation_count // 4,
                "existing_label_replacements": mutation_count // 4, "deletes": mutation_count // 4, "inserts": mutation_count // 4,
                "scope": actions["apply"]["scope"], "natural_ground_truth_remains_current": True,
                "scope_limit": "natural workload has no matching delta records; fresh-label probes verify changed records, but this is not a natural-label churn or sustained-update-rate benchmark"},
            "configurations": [summaries[phase, l] for phase in PHASES for l in breadths],
            "probes": probes, "delta_update_quality_observations": update_quality,
            "compaction_quality": compaction_quality, "worst_per_label_compaction_quality_difference": worst,
            "reopen_quality": reopen_quality,
            "interpretation": ["Compaction builds new approximate graphs; PQ reuse or retraining depends on the archived implementation/configuration. Strict-ID recall may increase or decrease even when natural exact truth is unchanged.",
                "Per-label differences describe this finite query sample and include ID tie choices; no independent per-query distance-tie analysis was performed for these newly built generations.",
                "Latency summaries are medians of three repeat percentiles, not pooled tails or confidence intervals.",
                "The same natural query sample is reused across breadths, phases and repeats; these are not additional independent quality samples.",
                "Fallback counts identify exact scans; a fast configuration with fallbacks should not be described as graph-only performance."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-diagnostic", action="store_true", help="Accept non-authoritative timings while retaining that status")
    args = parser.parse_args()
    require(len({path.resolve() for path in args.inputs}) == len(args.inputs), "Repeated input file")
    runs = [summarize_run(path, args.allow_diagnostic) for path in args.inputs]
    result = {"schema_version": 1, "status": "all assertions passed", "runs": runs,
              "aggregation": "each dataset/run remains separate; no pooling across corpus sizes or phases"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for run in runs:
        print(json.dumps({"source": run["source"]["path"], "rows": run["build"]["rows"], **run["timings"],
                          "worst_per_label_compaction_difference": run["worst_per_label_compaction_quality_difference"]}))
    print(json.dumps({"output": str(args.output), "runs": len(runs), "status": result["status"]}))


if __name__ == "__main__":
    main()
