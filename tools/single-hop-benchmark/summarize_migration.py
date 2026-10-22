#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Validate and summarize final same-artifact search migration measurements.

Defaults refer to the retained local experiment. No remote operations or raw
large index reads are performed. Earlier inline and large-K C++ baselines are
explicitly distinguished from the refreshed final top-10 comparison.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


SHARED_FIELDS = ("index_prefix", "prepared_identity", "queries", "beam", "cache_bypass",
                 "shared_cache_nodes", "latency_contract", "starts", "n_base", "k", "threads")
METRICS = ("recall", "qps", "mean_us", "p50_us", "p95_us", "p99_us")
VALIDITY = ("filter_violations", "duplicate_results", "short_results")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right):
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)


def provenance(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_measurements(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(rows, f"No measurements: {path}")
    groups = defaultdict(list)
    for row in rows:
        require(all(row.get(name) is not None for name in SHARED_FIELDS), f"Missing protocol fields: {path}")
        require(row.get("timing_authoritative") is True, f"Non-authoritative timing: {path}")
        require(row["latency_contract"] == "public_search_call", f"Incorrect latency contract: {path}")
        require(row["cache_bypass"] is True and row["shared_cache_nodes"] == 0,
                f"Expected uncached native graph reads: {path}")
        require(row.get("storage") in ("native_io_uring", "linux_aio_direct"), f"Non-native storage: {path}")
        require(all(type(row.get(name)) is int and row[name] == 0 for name in VALIDITY),
                f"Invalid, duplicate, incomplete or unvalidated results: {path}")
        require(all(type(row.get(name)) in (int, float) and math.isfinite(row[name]) for name in METRICS),
                f"Missing/non-finite metrics: {path}")
        require(0 <= row["recall"] <= 1 and row["qps"] > 0 and row["mean_us"] > 0,
                f"Out-of-range metrics: {path}")
        require(0 < row["p50_us"] <= row["p95_us"] <= row["p99_us"], f"Invalid percentiles: {path}")
        require(type(row.get("repeat")) is int, f"Missing repeat number: {path}")
        if "warmup_queries" in row:
            require(row["warmup_queries"] == row["queries"], f"Incomplete warmup: {path}")
        labels = row.get("labels")
        require(labels and len({item["label"] for item in labels}) == len(labels), f"Incomplete label evidence: {path}")
        require(all(type(item.get("n")) is int and item["n"] > 0 and type(item.get("eligible")) is int
                    and item["eligible"] > 0 and 0 <= item["recall"] <= 1 for item in labels),
                f"Invalid label counts or quality: {path}")
        require(sum(item["n"] for item in labels) == row["queries"], f"Label query counts do not sum: {path}")
        weighted = sum(item["recall"] * item["n"] for item in labels) / row["queries"]
        require(close(weighted, row["recall"]), f"Aggregate/per-label recall differs: {path}")
        groups[(row["mode"], row["l"], row["k"], row["threads"])].append(row)
    require(all(all(row.get(field) == rows[0].get(field) for field in
                    SHARED_FIELDS + ("storage", "implementation")) for row in rows),
            f"Input configurations use different dataset or protocol identities: {path}")
    for key, observations in groups.items():
        require(len(observations) == 3 and sorted(r["repeat"] for r in observations) == [0, 1, 2],
                f"Exactly three unique repeats 0,1,2 required: {path}, {key}")
        first = observations[0]
        fields = SHARED_FIELDS + ("storage", "implementation", "adaptive_sample", "adaptive_scale", "rerank_candidates")
        require(all(all(row.get(field) == first.get(field) for field in fields) for row in observations),
                f"Repeat protocol differs: {path}, {key}")
        first_labels = {item["label"]: item for item in first["labels"]}
        for row in observations:
            labels = {item["label"]: item for item in row["labels"]}
            require(labels.keys() == first_labels.keys(), f"Repeated label set differs: {path}, {key}")
            for label, item in labels.items():
                reference = first_labels[label]
                require((item["n"], item["eligible"]) == (reference["n"], reference["eligible"])
                        and close(item["recall"], reference["recall"]), f"Repeated label quality differs: {path}, {key}, {label}")
    return groups


def select(groups, mode, l, k=10):
    key = (mode, l, k, 8)
    require(key in groups, f"Missing configuration: {key}")
    return groups[key]


def summarize(rows):
    first = rows[0]
    result = {name: first[name] for name in SHARED_FIELDS + ("mode", "l", "implementation", "storage")}
    result.update({"repetitions": 3, "timing_authoritative": True, "validation": {name: 0 for name in VALIDITY}})
    for metric in METRICS:
        values = [row[metric] for row in rows]
        result[metric] = statistics.median(values)
        result[metric + "_range"] = [min(values), max(values)]
    for source, destination in (("reader_requests", "requested_reads_per_query"),
                                ("reader_bytes", "requested_bytes_per_query")):
        if all(source in row for row in rows):
            result[destination] = statistics.median(row[source] / row["queries"] for row in rows)
    if all("mean_cmps" in row for row in rows):
        result["pq_comparisons_per_query"] = statistics.median(row["mean_cmps"] for row in rows)
    result["warmup_queries_recorded"] = all("warmup_queries" in row for row in rows)
    result["labels"] = []
    for item in sorted(first["labels"], key=lambda item: item["label"]):
        repeated = [next(x for x in row["labels"] if x["label"] == item["label"]) for row in rows]
        label = {name: item[name] for name in ("label", "n", "eligible", "recall")}
        for metric in ("mean_us", "p50_us", "p95_us"):
            if all(metric in row for row in repeated):
                label[metric] = statistics.median(row[metric] for row in repeated)
        result["labels"].append(label)
    return result


def compatible(left, right):
    require(all(left[field] == right[field] for field in SHARED_FIELDS),
            "Compared configurations differ in dataset, index, concurrency, starts, cache, query count, K or latency contract")


def pair(left, right, equal_recall=False):
    compatible(left, right)
    require(left["l"] == right["l"], "Same-L comparison requested with different breadths")
    if equal_recall:
        require(close(left["recall"], right["recall"]), "Same-L strict recall differs")
    right_labels = {item["label"]: item for item in right["labels"]}
    require({item["label"] for item in left["labels"]} == right_labels.keys(), "Compared label sets differ")
    labels = []
    for item in left["labels"]:
        reference = right_labels[item["label"]]
        require((item["n"], item["eligible"]) == (reference["n"], reference["eligible"]), "Compared label workloads differ")
        if equal_recall:
            require(close(item["recall"], reference["recall"]), "Per-label strict recall differs")
        labels.append({"label": item["label"], "queries": item["n"], "eligible": item["eligible"],
                       "rust_recall": item["recall"], "cpp_recall": reference["recall"],
                       "rust_p95_us": item.get("p95_us"), "cpp_p95_us": reference.get("p95_us"),
                       "p95_ratio_rust_over_cpp": item["p95_us"] / reference["p95_us"]
                       if "p95_us" in item and "p95_us" in reference else None})
    return {"l": left["l"], "rust": left, "cpp": right,
            "qps_ratio_rust_over_cpp": left["qps"] / right["qps"],
            "p99_ratio_rust_over_cpp": left["p99_us"] / right["p99_us"],
            "p99_reduction_fraction": 1 - left["p99_us"] / right["p99_us"],
            "labels": labels}


def parity_proof(path, pairs, cpp_groups):
    raw = json.loads(path.read_text())
    require(raw["queries"] == 2000 and raw["k"] == 10 and raw["dimensions"] == 192, "Unexpected parity workload")
    groups = [group for group in raw["groups"] if group["mode"] == "single"]
    require(len(groups) == 3 and {g["l"] for g in groups} == {60, 64, 144}, "Incomplete ordered parity comparisons")
    zero_fields = ("ordered_results_differ", "result_sets_differ", "rust_saved_distance_mismatches", "cpp_saved_distance_mismatches")
    for group in groups:
        require(group["queries"] == raw["queries"] and all(group[field] == 0 for field in zero_fields), "Ordered result/distance parity failed")
        require(group["rust_strict_hits"] == group["cpp_strict_hits"]
                and group["rust_cutoff_hits"] == group["cpp_cutoff_hits"], "Exact/strict parity failed")
        performance = next(item for item in pairs if item["l"] == group["l"])
        require(close(group["rust_strict_recall"], performance["rust"]["recall"]), "Dump/performance recall differs")
        labels = {item["label"]: item for item in group["labels"]}
        require(labels.keys() == {item["label"] for item in performance["labels"]}, "Parity/performance labels differ")
        require(sum(item["queries"] for item in labels.values()) == raw["queries"], "Parity label counts differ")
        for item in performance["labels"]:
            proof = labels[item["label"]]
            require(proof["queries"] == item["queries"] and all(proof[field] == 0 for field in zero_fields)
                    and proof["rust_strict_hits"] == proof["cpp_strict_hits"]
                    and close(proof["rust_strict_recall"], item["rust_recall"]), "Per-label ordered parity failed")
        proof_hash = raw["cpp_inputs"][str(group["l"])]["ids_sha256"]
        require(all(row["result_ids_sha256"] == proof_hash for row in select(cpp_groups, "single", group["l"])),
                "Refreshed C++ IDs differ from independently compared C++ IDs")
    return {"status": "exact ordered ID and full-distance parity on every compared query and label",
            "unique_queries": raw["queries"], "labels": len(groups[0]["labels"]),
            "breadths": [60, 64, 144], "query_configuration_comparisons": sum(g["queries"] for g in groups),
            "independent_queries": raw["queries"],
            "interpretation": "6,000 comparisons reuse the same 2,000 queries at three breadths; they are not 6,000 independent queries",
            **{field: 0 for field in zero_fields}, "per_label_zero_difference_verified": True,
            "refreshed_cpp_result_hashes_match_independent_proof": True, "source": provenance(path)}


def cpp_manifests(top10_path, k1000_path):
    manifests = [json.loads(path.read_text()) for path in (top10_path, k1000_path)]
    for manifest in manifests:
        require(manifest["timing_authoritative"] is True and manifest["process_cpu_affinity"] == list(range(8)),
                "C++ manifest does not use authoritative eight-core affinity")
        require(manifest["environment"]["MKL_NUM_THREADS"] == "1"
                and manifest["environment"]["OMP_TOOL"] == "disabled", "C++ native runtime environment differs")
    fields = ("input_points", "dimensions", "metric", "pq_bytes", "dummy_replica_threshold", "process_cpu_affinity")
    require(all(manifests[0][field] == manifests[1][field] for field in fields), "C++ top-10/candidate manifests differ")
    binary = manifests[0]["binaries"]["search_disk_index"]
    require(binary == manifests[1]["binaries"]["search_disk_index"], "Historical C++ candidate binary differs")
    return {"same_search_binary_sha256": binary, "cpu_affinity": manifests[0]["process_cpu_affinity"],
            "MKL_NUM_THREADS": "1", "OMP_TOOL": "disabled", "pq_bytes": manifests[0]["pq_bytes"],
            "manifest_identity_fields_checked": list(fields)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2] / "benchmark-results/single-hop"
    migration = root / "migration/results"
    parser.add_argument("--rust-top10", type=Path, default=migration / "recall-fixed-t8.jsonl")
    parser.add_argument("--cpp-top10", type=Path, default=migration / "cpp-refresh/cpp-refresh.normalized.jsonl")
    parser.add_argument("--parity", type=Path, default=migration / "recall-tie-analysis-final.json")
    parser.add_argument("--rust-k1000", type=Path, default=migration / "recall-fixed-k1000-t8.jsonl")
    parser.add_argument("--cpp-k1000", type=Path, default=root / "results/native-cpp-k1000.jsonl")
    parser.add_argument("--historical-inline", type=Path, default=root / "results/rust-main-t8-inline.jsonl")
    parser.add_argument("--cpp-top10-manifest", type=Path, default=migration / "cpp-refresh/cpp-refresh.query.manifest.json")
    parser.add_argument("--cpp-k1000-manifest", type=Path, default=root / "results/native-cpp-k1000.manifest.json")
    parser.add_argument("--output", type=Path, default=migration / "search-final-summary.json")
    args = parser.parse_args()
    inputs = {name: getattr(args, name) for name in ("rust_top10", "cpp_top10", "parity", "rust_k1000", "cpp_k1000", "historical_inline",
                                                   "cpp_top10_manifest", "cpp_k1000_manifest")}
    manifest_checks = cpp_manifests(args.cpp_top10_manifest, args.cpp_k1000_manifest)
    rust, cpp = read_measurements(args.rust_top10), read_measurements(args.cpp_top10)
    pairs = [pair(summarize(select(rust, "single", l)), summarize(select(cpp, "single", l)), True) for l in (60, 64, 144)]
    proof = parity_proof(args.parity, pairs, cpp)
    strict95 = pairs[0]["rust"]
    multihop64 = summarize(select(rust, "multihop", 64))
    compatible(strict95, multihop64)
    multihop_floor = max((summarize(rows) for key, rows in rust.items()
                         if key[0] == "multihop" and min(row["recall"] for row in rows) >= .95), key=lambda row: row["qps"])
    historical = summarize(select(read_measurements(args.historical_inline), "inline", 320))
    compatible(strict95, historical)
    require(min(historical["recall_range"]) >= .95, "Historical inline baseline misses the stated floor")
    candidates = read_measurements(args.rust_k1000)
    cpp_candidates = read_measurements(args.cpp_k1000)
    k1000_pair = pair(summarize(select(candidates, "single", 4000, 1000)),
                      summarize(select(cpp_candidates, "single", 4000, 1000)), True)
    multi_k1000 = summarize(select(candidates, "multihop", 4000, 1000))
    compatible(k1000_pair["rust"], multi_k1000)
    require(k1000_pair["rust"]["index_prefix"] == strict95["index_prefix"]
            and k1000_pair["rust"]["prepared_identity"].split(":")[-1] == strict95["prepared_identity"].split(":")[-1],
            "Top-10/candidate workloads do not share the index and label metadata")
    result = {"status": "all assertions passed", "scope": "search-only same-artifact native Linux experiment; separate from Rust-built lifecycle evidence",
              "provenance": {name: provenance(path) for name, path in inputs.items()},
              "validation": {"exactly_three_repeats_per_input_configuration": True,
                             "all_input_validity_counters_zero": True, "authoritative_timings_required": True,
                             "matched_dataset_index_concurrency_cache_K_query_count_and_latency_contract": True,
                             "storage_backends": {"rust": "native_io_uring", "cpp": "linux_aio_direct"},
                             "cpp_manifest_checks": manifest_checks,
                             "warmup_note": "full warmup asserted where row fields are present; Rust rows do not serialize the warmup count",
                             "latency_summary": "median of three per-repeat public-call percentiles; not a pooled percentile or confidence interval"},
              "same_recall_same_L_top10": pairs, "ordered_result_parity": proof,
              "current_round_all_modes": [summarize(rows) for _, rows in sorted(rust.items())],
              "multihop_established_fixed_L64": {"measurement": multihop64,
                  "strict_L60_qps_ratio": strict95["qps"] / multihop64["qps"],
                  "qualification": "fixed established baseline; final multihop L60 is faster and also meets the 95% floor"},
              "fastest_final_measured_multihop_at_95_floor": {"measurement": multihop_floor,
                  "strict_L60_qps_ratio": strict95["qps"] / multihop_floor["qps"],
                  "interpretation": "common recall floor, not identical recall or a claim of globally optimal tuning"},
              "historical_inline_L320": {"status": "historical frozen baseline; not refreshed in the final round at L320",
                  "measurement": historical, "strict_L60_qps_ratio_historical": strict95["qps"] / historical["qps"],
                  "qualification": "cross-round descriptive comparison; current final-round inline measurements at L60/64/144 all miss 95%"},
              "k1000_L4000": {**k1000_pair,
                  "cpp_status": "earlier retained baseline; unchanged C++ binary and matched workload/protocol, not a new same-round refresh",
                  "multihop": multi_k1000, "strict_over_multihop_qps_ratio": k1000_pair["rust"]["qps"] / multi_k1000["qps"],
                  "ordered_result_parity_established": False,
                  "qualification": "aggregate and every label's strict recall match C++; 256 unique queries, four per label, so tail and label estimates are coarse; recall is below 99%; one measured L, not a tuning optimum"},
              "limitations": ["read requests/bytes are logical reader requests, not physical SSD IOPS",
                              "same queries are reused for three timing repeats; deterministic repeated recall does not add independent quality samples",
                              "per-label tails use only the recorded small query samples and do not establish statistical significance",
                              "the shared graph was C++-built StitchedVamana; this summary does not measure a Rust-built graph or update lifecycle"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for item in pairs:
        print(json.dumps({"l": item["l"], "recall": item["rust"]["recall"], "rust_qps": item["rust"]["qps"],
                          "cpp_qps": item["cpp"]["qps"], "qps_ratio": item["qps_ratio_rust_over_cpp"],
                          "rust_p99_us": item["rust"]["p99_us"], "cpp_p99_us": item["cpp"]["p99_us"]}))
    print(json.dumps({"output": str(args.output), "status": result["status"], "ordered_parity_comparisons": 6000,
                      "unique_parity_queries": 2000, "k1000_qps_ratio": k1000_pair["qps_ratio_rust_over_cpp"]}))


if __name__ == "__main__":
    main()
