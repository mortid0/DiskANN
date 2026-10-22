#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Validate and compare the synthetic matching-delta top-k experiment.

Requires the complete unpruned --action delta log, the optimized --action query
log on its unchanged final index, two frozen source archives, and a provenance
sidecar. No process-wide resource or compilation times enter query comparisons.

The sidecar has schema_version=1, shared_index with host and absolute root, and
unpruned/optimized objects, each containing log_sha256, binary_sha256,
source_archive {path, sha256}, and index_identity. Both index_identity objects
must match exactly and contain generation, base_sequence, sequence,
current_sha256, catalog_sha256, and journal_sha256. Binary/index hashes are
declared by separately collected host evidence; this script verifies their
consistency and hashes the local logs/source archives, including each view.rs.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import tarfile


SIZES = (0, 100, 1000, 10000)
BREADTHS = (60, 64, 144)
VALIDITY = ("filter_violations", "duplicate_results", "short_results", "exact_fallback_queries")
METRICS = ("qps", "mean_us", "p50_us", "p95_us", "p99_us", "wall_seconds")
SCOPE = "synthetic_worst_case_all_queried_natural_labels_match_every_insert"
VIEW = "diskann-disk/src/filtered/view.rs"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value, minimum=0, strict=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > minimum if strict else value >= minimum))


def digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def provenance(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def one(rows, description):
    require(len(rows) == 1, f"Exactly one {description} required")
    return rows[0]


def read_rows(path):
    rows = []
    for number, text in enumerate(path.read_text().splitlines(), 1):
        if not text.strip():
            continue
        row = json.loads(text, parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"Nonfinite JSON {token} in {path}:{number}")))
        require(type(row) is dict, f"Object required at {path}:{number}")
        rows.append(row)
    return rows


def strict_hits(recall, denominator):
    require(finite(recall) and recall <= 1, "Invalid strict-ID recall")
    count = round(recall * denominator)
    require(abs(recall * denominator - count) < 1e-7,
            "Strict recall is not an integer hit count")
    return count


def label_quality(row):
    labels = row.get("labels")
    require(type(labels) is list and labels, "Missing per-label quality")
    quality = {}
    for item in labels:
        require(type(item) is dict and integer(item.get("label")) and item["label"] < 2**32
                and integer(item.get("queries"), 1), "Invalid per-label identity/sample")
        require(item["label"] not in quality, "Repeated label")
        require(all(finite(item.get(name), strict=True) for name in ("mean_us", "p95_us")),
                "Invalid per-label latency")
        quality[item["label"]] = (item["queries"], strict_hits(item.get("recall"), item["queries"] * row["k"]))
    require(sum(sample for sample, _ in quality.values()) == row["queries"],
            "Per-label sample count differs")
    require(sum(hits for _, hits in quality.values()) == strict_hits(row["recall"], row["queries"] * row["k"]),
            "Per-label and aggregate recall differ")
    return quality


def experiment(rows, action):
    item = one([row for row in rows if row.get("kind") == "experiment"], "experiment row")
    require(item.get("action") == action, f"Expected --action {action}")
    require(item.get("cache_bypass") is True and item.get("shared_cache_nodes") == 0
            and item.get("beam") == 4 and item.get("storage") == "native_io_uring"
            and item.get("query_scheduling") == "dynamic_one"
            and item.get("warmup") == "full_same_process_same_worker_count",
            "Unexpected native I/O, cache, scheduling or warmup protocol")
    require(integer(item.get("n_base"), 256) and integer(item.get("dimensions"), 1)
            and integer(item.get("index_threads"), 1)
            and type(item.get("root")) is str and item["root"], "Invalid experiment shape/identity")
    if action == "delta":
        require(item.get("update_scope") ==
                "synthetic_far_inserts_matching_every_queried_natural_label; exact_distance_proof_required",
                "Missing matching-far-insert scenario")
    return item


def query_groups(rows, exp, phases, sequences, reference=None):
    queries = [row for row in rows if row.get("kind") == "query"]
    require(len(queries) == len(phases) * len(BREADTHS) * 3, "Incomplete query row matrix")
    first = queries[0]
    protocol = ("implementation", "queries", "threads", "k", "prepared_identity",
                "latency_contract", "strict_id_recall")
    if reference is not None:
        require(all(first.get(key) == reference.get(key) for key in protocol),
                "Compared query workloads/protocols differ")
    groups = defaultdict(list)
    for row in queries:
        require(row.get("label_l_plan") is None, "Fixed-L delta summary rejects per-label breadth plans")
        require(all(item.get("actual_l", row["l"]) == row["l"] for item in row["labels"]),
                "Fixed-L delta row contains mixed per-label breadths")
        require(all(row.get(key) == first.get(key) for key in protocol), "Within-run query protocol differs")
        require(row.get("implementation") == "rust_native_lifecycle"
                and row.get("latency_contract") == "public_snapshot_search_call"
                and row.get("strict_id_recall") is True and row.get("timing_authoritative") is True,
                "Unexpected or non-authoritative query measurement")
        require(integer(row.get("queries"), 1) and integer(row.get("warmup_queries"), 1)
                and row["warmup_queries"] == row["queries"]
                and integer(row.get("threads"), 1) and row["threads"] <= exp["index_threads"]
                and integer(row.get("k"), 1) and integer(row.get("l"), 1)
                and row["k"] <= row["l"] < 2**32, "Invalid query shape or incomplete warmup")
        require(type(row.get("prepared_identity")) is str
                and len(row["prepared_identity"].split(":")) == 4
                and all(digest(part) for part in row["prepared_identity"].split(":")),
                "Prepared workload identity is not four verified asset SHA256s")
        require(row.get("phase") in phases and row["l"] in BREADTHS
                and integer(row.get("repeat")) and row["repeat"] < 3
                and integer(row.get("sequence")) and row["sequence"] == sequences[row["phase"]],
                "Unexpected phase, breadth, repeat or snapshot sequence")
        require(all(type(row.get(name)) is int and row[name] == 0 for name in VALIDITY),
                "Invalid, duplicate, short, fallback or unvalidated result")
        require(integer(row.get("base_searches"), row["queries"])
                and integer(row.get("delta_candidates")), "Invalid query work counters")
        size = phases[row["phase"]]
        require(row["delta_candidates"] == size * row["queries"], "Delta candidate count differs from all matching inserts")
        if row["phase"].startswith("delta_"):
            require(row.get("synthetic_delta_records") == size
                    and row.get("stress_scope") == "new_far_vectors_matching_every_queried_natural_label"
                    and row.get("groundtruth_unchanged_by_inserts") is True,
                    "Query phase is missing unchanged-truth stress metadata")
        require(all(finite(row.get(metric), strict=True) for metric in METRICS)
                and row["p50_us"] <= row["p95_us"] <= row["p99_us"], "Invalid query timing")
        require(math.isclose(row["qps"], row["queries"] / row["wall_seconds"], rel_tol=1e-12),
                "QPS and query wall clock differ")
        label_quality(row)
        groups[row["phase"], row["l"]].append(row)
    require(set(groups) == {(phase, breadth) for phase in phases for breadth in BREADTHS},
            "Missing phase/breadth configuration")
    for observations in groups.values():
        require(len(observations) == 3 and sorted(row["repeat"] for row in observations) == [0, 1, 2],
                "Exactly three unique repeats required")
        quality = label_quality(observations[0])
        require(all(label_quality(row) == quality
                    and row["base_searches"] == observations[0]["base_searches"] for row in observations),
                "Repeated per-label recall, samples or base work differs")
    return first, groups


def stress_metadata(rows, exp):
    metadata = one([row for row in rows if row.get("kind") == "delta_stress"], "delta-stress proof")
    require(metadata.get("action") == "delta" and metadata.get("sizes") == list(SIZES[1:])
            and metadata.get("scope") == SCOPE and metadata.get("far_vector_value") == 255
            and metadata.get("dimensions") == exp["dimensions"]
            and metadata.get("groundtruth_unchanged_by_inserts") is True
            and metadata.get("proof") == "every exact integer squared-L2 far distance strictly exceeds that query's published kth distance"
            and metadata.get("compaction_performed") is False and metadata.get("graph_unchanged") is True,
            "Missing or invalid unchanged-ground-truth proof")
    require(integer(metadata.get("minimum_exact_integer_far_distance"), 1)
            and metadata["minimum_exact_integer_far_distance"] <= exp["dimensions"] * 255**2
            and finite(metadata.get("minimum_far_minus_published_kth"), strict=True)
            and metadata["minimum_far_minus_published_kth"] <= metadata["minimum_exact_integer_far_distance"],
            "Invalid exact-distance proof margin")
    require(integer(metadata.get("generation"), 1) and integer(metadata.get("initial_sequence"))
            and integer(metadata.get("base_sequence"))
            and metadata["initial_sequence"] == metadata["base_sequence"]
            and integer(metadata.get("first_stable_id"), exp["n_base"])
            and metadata["first_stable_id"] + SIZES[-1] <= 2**32, "Invalid base sequence or stable-ID watermark")
    require(integer(metadata.get("max_delta_records"), SIZES[-1])
            and integer(metadata.get("largest_phase_accounted_delta_bytes"), 1)
            and integer(metadata.get("max_delta_bytes"), 1)
            and metadata["largest_phase_accounted_delta_bytes"] <= metadata["max_delta_bytes"]
            and integer(metadata.get("max_frame_bytes"), 1)
            and metadata.get("effective_frame_cap") == min(metadata["max_frame_bytes"], 16 * 1024**2)
            and integer(metadata.get("max_batch_records"), 1) and metadata["max_batch_records"] <= 1000
            and integer(metadata.get("conservative_final_journal_bytes"), 1)
            and integer(metadata.get("max_journal_bytes"), 1)
            and metadata["conservative_final_journal_bytes"] <= metadata["max_journal_bytes"],
            "Invalid durable delta admission/accounting metadata")
    applies = [row for row in rows if row.get("kind") == "delta_apply"]
    require(len(applies) == 3 and {row.get("phase") for row in applies} == {f"delta_{n}" for n in SIZES[1:]},
            "Incomplete or duplicate durable-apply phases")
    by_size = {row["cumulative_delta_records"]: row for row in applies}
    require(set(by_size) == set(SIZES[1:]), "Unexpected cumulative delta sizes")
    sequence = metadata["initial_sequence"]
    sequences = {"delta_0": sequence}
    previous = 0
    journal = 0
    for size in SIZES[1:]:
        row = by_size[size]
        expected_batches = ((size - previous - int(previous == 0) + metadata["max_batch_records"] - 1)
                            // metadata["max_batch_records"] + int(previous == 0))
        require(row.get("phase") == f"delta_{size}" and row.get("scope") == SCOPE
                and row.get("compaction_performed") is False
                and row.get("inserted_this_phase") == size - previous
                and type(row.get("durable_batches")) is int and row["durable_batches"] == expected_batches,
                "Invalid durable insert/batch accounting")
        sequence += expected_batches
        require(integer(row.get("sequence")) and row["sequence"] == sequence
                and integer(row.get("base_sequence")) and row["base_sequence"] == metadata["base_sequence"]
                and integer(row.get("journal_bytes"), journal + 1)
                and row["journal_bytes"] <= metadata["conservative_final_journal_bytes"],
                "Invalid sequence, base sequence or journal growth")
        require(finite(row.get("seconds"), strict=True)
                and finite(row.get("mutations_per_second"), strict=True)
                and math.isclose(row["mutations_per_second"], (size - previous) / row["seconds"], rel_tol=1e-12),
                "Invalid apply timing/accounting")
        sequences[row["phase"]] = sequence
        previous, journal = size, row["journal_bytes"]
    return metadata, sequences, [by_size[size] for size in SIZES[1:]]


def frozen_view(path):
    with tarfile.open(path, "r:*") as archive:
        matches = [member for member in archive.getmembers() if member.isfile()
                   and (member.name == VIEW or member.name.endswith("/" + VIEW))]
        member = one(matches, "frozen view source")
        return hashlib.sha256(archive.extractfile(member).read()).hexdigest()


def identities(evidence_path, paths, sources, exps, metadata, final_sequence):
    evidence = json.loads(evidence_path.read_text())
    require(evidence.get("schema_version") == 1, "Unsupported identity evidence schema")
    shared = evidence.get("shared_index", {})
    require(type(shared.get("host")) is str and shared["host"]
            and type(shared.get("root")) is str and PurePosixPath(shared["root"]).is_absolute(),
            "Missing native host or absolute shared index path")
    validated = {}
    for name in ("unpruned", "optimized"):
        item = evidence.get(name, {})
        source = item.get("source_archive", {})
        require(item.get("log_sha256") == provenance(paths[name])["sha256"]
                and source.get("sha256") == provenance(sources[name])["sha256"]
                and type(source.get("path")) is str and Path(source["path"]).name == sources[name].name
                and digest(item.get("binary_sha256")), "Result/source/build evidence does not match local assets")
        logged_root = PurePosixPath(exps[name]["root"])
        shared_root = PurePosixPath(shared["root"])
        require(shared_root == logged_root if logged_root.is_absolute()
                else shared_root.parts[-len(logged_root.parts):] == logged_root.parts,
                "Logged index does not match the absolute shared index identity")
        index = item.get("index_identity", {})
        require(all(integer(index.get(key)) for key in ("generation", "base_sequence", "sequence"))
                and index["generation"] == metadata["generation"]
                and index.get("base_sequence") == metadata["base_sequence"]
                and index.get("sequence") == final_sequence
                and all(digest(index.get(key)) for key in
                        ("current_sha256", "catalog_sha256", "journal_sha256")),
                "Invalid final index identity evidence")
        require(all(index[key] == shared[key] for key in index if key in shared),
                "Shared and per-version index identities differ")
        validated[name] = {**item, "local_view_sha256": frozen_view(sources[name])}
    require(validated["unpruned"]["index_identity"] == validated["optimized"]["index_identity"],
            "Compared logs do not identify the same final index/WAL")
    require(validated["unpruned"]["source_archive"]["sha256"] != validated["optimized"]["source_archive"]["sha256"]
            and all(validated["unpruned"][key] != validated["optimized"][key] for key in
                    ("local_view_sha256", "binary_sha256")), "Distinct source/binary versions required")
    return {"source": provenance(evidence_path), "shared_index": shared, "versions": validated,
            "verification_scope": "Local result/archive/view hashes verified; binary and committed index hashes are linked declarations from separately collected host evidence."}


def profile(rows):
    first = rows[0]
    summary = {key: first[key] for key in
               ("phase", "l", "k", "threads", "queries", "sequence", "prepared_identity", "base_searches", "delta_candidates", "recall")}
    summary["repetitions"] = 3
    summary["warmup_queries_per_repeat"] = first["queries"]
    for metric in METRICS:
        values = [row[metric] for row in rows]
        summary[metric] = statistics.median(values)
        summary[metric + "_range"] = [min(values), max(values)]
    summary["labels"] = []
    for label, (sample, hits) in sorted(label_quality(first).items()):
        items = [next(item for item in row["labels"] if item["label"] == label) for row in rows]
        summary["labels"].append({"label": label, "queries": sample, "strict_id_hits": hits,
                                  "recall": hits / (sample * first["k"]),
                                  **{metric: statistics.median(item[metric] for item in items)
                                     for metric in ("mean_us", "p95_us")}})
    return summary


def summarize(unpruned, optimized, unpruned_source, optimized_source, identity_evidence):
    old, new = read_rows(unpruned), read_rows(optimized)
    require(all(row.get("kind") in ("experiment", "delta_stress", "delta_apply", "query") for row in old),
            "Unexpected unpruned row kind; build/resource times are not query comparisons")
    require(all(row.get("kind") in ("experiment", "query") for row in new), "Optimized log must contain query action only")
    old_exp, new_exp = experiment(old, "delta"), experiment(new, "query")
    common = ("root", "prepared", "n_base", "dimensions", "index_threads", "beam", "storage", "cache_bypass",
              "shared_cache_nodes", "query_scheduling", "warmup")
    require(all(old_exp.get(key) == new_exp.get(key) for key in common), "Compared experiment identities/protocols differ")
    metadata, sequences, applies = stress_metadata(old, old_exp)
    first, groups = query_groups(old, old_exp, {f"delta_{n}": n for n in SIZES}, sequences)
    require(integer(metadata.get("queries"), 1) and integer(metadata.get("k"), 1)
            and all(metadata.get(key) == first.get(key) for key in ("queries", "k", "prepared_identity")),
            "Stress proof/query workload identity differs")
    labels = set(label_quality(first))
    require(type(metadata.get("labels")) is list and len(metadata["labels"]) == len(labels)
            and set(metadata["labels"]) == labels and metadata.get("queried_label_count") == len(labels),
            "Stress labels do not match every queried label")
    final_sequence = sequences["delta_10000"]
    _, new_groups = query_groups(new, new_exp, {"query": 10000}, {"query": final_sequence}, first)
    for breadth in BREADTHS:
        baseline = groups["delta_0", breadth][0]
        for size in SIZES[1:]:
            row = groups[f"delta_{size}", breadth][0]
            require(label_quality(row) == label_quality(baseline) and row["base_searches"] == baseline["base_searches"],
                    "Far inserts changed per-label quality or immutable-base search work")
        candidate = new_groups["query", breadth][0]
        require(label_quality(candidate) == label_quality(baseline)
                and candidate["base_searches"] == baseline["base_searches"],
                "Optimized query changed per-label quality or immutable-base search work")
    identity = identities(identity_evidence, {"unpruned": unpruned, "optimized": optimized},
                          {"unpruned": unpruned_source, "optimized": optimized_source},
                          {"unpruned": old_exp, "optimized": new_exp}, metadata, final_sequence)
    old_profiles = [profile(groups[f"delta_{size}", breadth]) for size in SIZES for breadth in BREADTHS]
    new_profiles = [profile(new_groups["query", breadth]) for breadth in BREADTHS]
    comparisons = []
    for candidate in new_profiles:
        baseline = next(item for item in old_profiles if item["phase"] == "delta_10000" and item["l"] == candidate["l"])
        comparisons.append({"l": candidate["l"], "queries": candidate["queries"], "k": candidate["k"],
                            "threads": candidate["threads"], "synthetic_delta_records": 10000,
                            "qps_ratio_optimized_over_unpruned": candidate["qps"] / baseline["qps"],
                            "qps_change_percent": 100 * (candidate["qps"] / baseline["qps"] - 1),
                            "latency_reduction_percent": {metric: 100 * (1 - candidate[metric] / baseline[metric])
                                                           for metric in ("mean_us", "p50_us", "p95_us", "p99_us")},
                            "all_per_label_strict_id_hit_counts_unchanged": True})
    return {"schema_version": 1, "status": "all assertions passed", "scope": SCOPE,
            "inputs": {"unpruned": provenance(unpruned), "optimized": provenance(optimized)},
            "identity": identity, "proof": metadata,
            "validation": {"unpruned_query_rows": 36, "optimized_query_rows": 9, "breadths": list(BREADTHS),
                           "delta_sizes": list(SIZES), "repeats_per_configuration": 3,
                           "all_validity_and_fallback_counters_zero": True,
                           "all_delta_candidate_counts_equal_records_times_queries": True,
                           "per_label_quality_and_base_work_unchanged_across_all_phases_and_versions": True,
                           "final_sequence": final_sequence, "authoritative_query_timing": True},
            "durable_insert_diagnostics": applies, "unpruned_configurations": old_profiles,
            "optimized_10000_configurations": new_profiles, "optimization_comparison": comparisons,
            "interpretation": [
                "All synthetic inserts are the all-255 vector and carry every queried natural label. Every query matches the complete delta; this is a deliberately adverse matching-delta scan scenario.",
                "The logged exact integer-distance proof keeps published natural top-k truth current. Validation checks strict-ID hit counts per label, not ordered per-query result equivalence.",
                "Optimization retains exact matching-delta distance evaluation while selecting only delta top-k before base-result merges.",
                "Comparisons use medians of three query repeats and their reported latency percentiles; they are not pooled tails, confidence intervals or independent quality samples.",
                "Build, input loading, compilation, durable insert batches, process resources and warmup are excluded from query latency/QPS comparisons. Durable insert diagnostics are reported separately.",
                "Inserts complete before measured query phases. These results do not establish sustained concurrent-write capacity, natural churn behavior or a production latency guarantee."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unpruned", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--unpruned-source", type=Path, required=True)
    parser.add_argument("--optimized-source", type=Path, required=True)
    parser.add_argument("--identity-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.resolve() not in {getattr(args, key).resolve() for key in
                ("unpruned", "optimized", "unpruned_source", "optimized_source", "identity_evidence")},
            "Output must not overwrite an input")
    result = summarize(args.unpruned, args.optimized, args.unpruned_source, args.optimized_source, args.identity_evidence)
    result["analysis_script"] = provenance(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "query_rows": 45, "optimization_comparison": result["optimization_comparison"]}))


if __name__ == "__main__":
    main()
