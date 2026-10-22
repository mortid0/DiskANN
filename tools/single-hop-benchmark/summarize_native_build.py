#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Validate final native 10M build/query evidence against a fresh C++ reference.

Build and query logs remain separate. Both searches must use the same 2,000
prepared top-10 queries at L60/64/144, eight workers and three repeats. Their
graphs and entry points differ; this is a complete native artifact comparison,
not an ordered-ID parity or isolated search-language comparison.

Identity sidecar schema_version=1 requires binary_sha256, source_archive_sha256,
codebook_sha256, committed_codebook_sha256, current_sha256, catalog_sha256,
build_log_sha256, query_log_sha256, host and absolute root. Local log/archive
hashes are verified; native binary/index/codebook hashes are separately collected
host declarations. An adjacent C++ .query.manifest.json is checked when present.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path, PurePosixPath
import tarfile
import hashlib

from summarize_delta import (BREADTHS, digest, finite, integer, label_quality,
                             one, provenance, read_rows, require)
from summarize_lifecycle import phase_summary, resources
from summarize_migration import read_measurements, summarize as cpp_summary


COMMON = ("root", "prepared", "n_base", "dimensions", "index_threads", "beam", "shared_cache_nodes",
          "storage", "cache_bypass", "query_scheduling", "warmup")
SHAPE = {"n_base": 10000000, "dimensions": 192, "index_threads": 8}
HASH_FIELDS = ("binary_sha256", "source_archive_sha256", "codebook_sha256", "committed_codebook_sha256",
               "current_sha256", "catalog_sha256", "build_log_sha256", "query_log_sha256")
KEY_SOURCES = ("diskann-disk/src/filtered/native.rs", "diskann-disk/src/filtered/index.rs",
               "diskann-disk/src/filtered/view.rs", "diskann-disk/src/build/filtered.rs")


def matches_absolute(absolute, logged):
    absolute, logged = PurePosixPath(absolute), PurePosixPath(logged)
    return absolute == logged if logged.is_absolute() else absolute.parts[-len(logged.parts):] == logged.parts


def native_experiment(rows, action):
    item = one([row for row in rows if row.get("kind") == "experiment"], f"native {action} experiment")
    require(item.get("action") == action and all(type(item.get(key)) is int and item[key] == value
                                                for key, value in SHAPE.items()), "Unexpected native build/query shape")
    require(item.get("beam") == 4 and item.get("shared_cache_nodes") == 0
            and item.get("cache_bypass") is True and item.get("storage") == "native_io_uring"
            and item.get("query_scheduling") == "dynamic_one"
            and item.get("warmup") == "full_same_process_same_worker_count",
            "Unexpected native storage, cache, warmup or scheduling protocol")
    require(all(type(item.get(key)) is str and item[key] for key in ("root", "prepared")),
            "Missing native index/prepared identity")
    return item


def native_queries(rows):
    require(len(rows) == 10 and all(row.get("kind") in ("experiment", "query") for row in rows),
            "Expected query-only log with one experiment and nine query rows")
    exp = native_experiment(rows, "query")
    groups = defaultdict(list)
    first = one([row for row in rows if row.get("kind") == "query" and row.get("l") == 60
                 and row.get("repeat") == 0], "native L60 repeat zero")
    for row in rows:
        if row.get("kind") != "query":
            continue
        require(row.get("label_l_plan") is None, "Fixed-L native summary rejects per-label breadth plans")
        require(all(item.get("actual_l", row["l"]) == row["l"] for item in row["labels"]),
                "Fixed-L native row contains mixed per-label breadths")
        require(row.get("implementation") == "rust_native_lifecycle" and row.get("phase") == "query"
                and row.get("strict_id_recall") is True and row.get("timing_authoritative") is True
                and row.get("latency_contract") == "public_snapshot_search_call", "Unexpected native search measurement")
        require(all(type(row.get(key)) is int and row[key] == value for key, value in
                    {"queries": 2000, "warmup_queries": 2000, "k": 10, "threads": 8, "sequence": 0,
                     "filter_violations": 0, "duplicate_results": 0, "short_results": 0,
                     "exact_fallback_queries": 0, "delta_candidates": 0, "base_searches": 2000}.items()),
                "Native workload, empty delta, query sequence or validity counters differ")
        require(integer(row.get("l")) and row["l"] in BREADTHS
                and integer(row.get("repeat")) and row["repeat"] < 3, "Invalid native breadth/repeat")
        require(type(row.get("prepared_identity")) is str
                and len(row["prepared_identity"].split(":")) == 4
                and all(digest(value) for value in row["prepared_identity"].split(":"))
                and row["prepared_identity"] == first["prepared_identity"], "Native prepared asset identities differ")
        require(all(finite(row.get(metric), strict=True) for metric in
                    ("qps", "mean_us", "p50_us", "p95_us", "p99_us", "wall_seconds"))
                and row["p50_us"] <= row["p95_us"] <= row["p99_us"]
                and math.isclose(row["qps"], row["queries"] / row["wall_seconds"], rel_tol=1e-12),
                "Invalid native timing or QPS/wall consistency")
        label_quality(row)
        groups[row["l"]].append(row)
    require(set(groups) == set(BREADTHS), "Native breadth matrix is incomplete")
    samples = {label: count for label, (count, _) in label_quality(first).items()}
    for breadth, observations in groups.items():
        require(len(observations) == 3 and sorted(row["repeat"] for row in observations) == [0, 1, 2],
                f"Exactly three native repeats required at L{breadth}")
        quality = label_quality(observations[0])
        require(all(label_quality(row) == quality for row in observations)
                and {label: count for label, (count, _) in quality.items()} == samples,
                "Repeated native quality or cross-breadth label samples differ")
    return exp, groups


def validate_cpp(path, native_groups):
    groups = read_measurements(path)
    require(set(groups) == {("single", breadth, 10, 8) for breadth in BREADTHS},
            "Expected exactly nine fresh C++ strict top-10 rows at L60/64/144 and eight threads")
    converted = {}
    eligible_reference = {item["label"]: item["eligible"] for item in groups["single", 60, 10, 8][0]["labels"]}
    for breadth in BREADTHS:
        observations = groups["single", breadth, 10, 8]
        require(all(row.get("implementation") == "cpp_native_linux" and row.get("starts") == "label"
                    and row.get("strict_id_recall") is True and row.get("n_base") == SHAPE["n_base"]
                    and row.get("queries") == 2000 and row.get("warmup_queries") == 2000
                    and row.get("beam") == 4 and row.get("storage") == "linux_aio_direct"
                    and row.get("prepared_identity") == native_groups[breadth][0]["prepared_identity"]
                    for row in observations), "C++ corpus, workload or strict/native I/O protocol differs")
        for row in observations:
            require({item["label"]: item["eligible"] for item in row["labels"]} == eligible_reference,
                    "C++ label eligibility counts differ across breadths/repeats")
            if "exact_fallback_queries" in row:
                require(type(row["exact_fallback_queries"]) is int and row["exact_fallback_queries"] == 0,
                        "Unexpected C++ fallback counter")
            adapted = {**row, "labels": [{**item, "queries": item["n"]} for item in row["labels"]]}
            quality = label_quality(adapted)
            native = label_quality(native_groups[breadth][0])
            require({label: sample for label, (sample, _) in quality.items()}
                    == {label: sample for label, (sample, _) in native.items()}, "C++/native label samples differ")
        converted[breadth] = observations
    manifest_path = path.with_name(path.name.replace(".normalized.jsonl", ".query.manifest.json"))
    manifest_evidence = {"available": False}
    if manifest_path != path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        require(manifest.get("timing_authoritative") is True and manifest.get("input_points") == SHAPE["n_base"]
                and manifest.get("dimensions") == SHAPE["dimensions"] and manifest.get("metric") == "uint8_l2"
                and manifest.get("pq_bytes") == 32 and manifest.get("process_cpu_affinity") == list(range(8)),
                "C++ manifest corpus/PQ/affinity protocol differs")
        runs = manifest.get("runs", [])
        require(len(runs) == 9 and all(row.get("exit_code") == 0 for row in runs)
                and sorted((run.get("normalized") for run in runs), key=lambda row: (row["l"], row["repeat"]))
                == sorted((row for observations in converted.values() for row in observations),
                          key=lambda row: (row["l"], row["repeat"])), "C++ normalized evidence is not bound to its run manifest")
        manifest_evidence = {"available": True, "source": provenance(manifest_path),
                             "search_binary_sha256": manifest["binaries"]["search_disk_index"],
                             "cpu_affinity": manifest["process_cpu_affinity"]}
    return converted, manifest_evidence


def archive_sources(path):
    result = {}
    with tarfile.open(path, "r:*") as archive:
        for source in KEY_SOURCES:
            member = one([member for member in archive.getmembers() if member.isfile()
                          and (member.name == source or member.name.endswith("/" + source))], source)
            result[source] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    return result


def summarize(build_path, query_path, cpp_path, source_archive, identity_path):
    build_rows, query_rows = read_rows(build_path), read_rows(query_path)
    require(len(build_rows) == 2 and {row.get("kind") for row in build_rows} == {"experiment", "build"},
            "Expected build-only log with exactly experiment and build rows")
    build_exp = native_experiment(build_rows, "build")
    query_exp, native_groups = native_queries(query_rows)
    require(all(build_exp.get(key) == query_exp.get(key) for key in COMMON), "Build/query native identities differ")
    build = one([row for row in build_rows if row.get("kind") == "build"], "native build row")
    require(build.get("implementation") == "rust_native_lifecycle" and build.get("rows") == SHAPE["n_base"]
            and build.get("dimensions") == SHAPE["dimensions"] and build.get("threads") == 8
            and build.get("root") == build_exp["root"] and build.get("skipped_label") == 200386
            and finite(build.get("seconds_including_input_load"), strict=True), "Invalid native build metadata/timing")
    config = build.get("config", {})
    require(all(type(config.get(key)) is int and config[key] == value for key, value in
                {"label_degree": 32, "max_degree": 128, "l_build": 100, "pq_chunks": 32, "seed": 0,
                 "max_delta_records": 10000, "max_delta_bytes": 67108864}.items())
            and finite(config.get("alpha"), strict=True) and math.isclose(config["alpha"], 1.2, abs_tol=1e-6)
            and type(config.get("initial_pq_codebook")) is str and config["initial_pq_codebook"],
            "Unexpected native graph/PQ config or missing frozen external codebook")
    cpp_groups, cpp_manifest = validate_cpp(cpp_path, native_groups)
    reference = cpp_groups[60][0]
    require(matches_absolute(reference["index_prefix"] + "_pq_pivots.bin", config["initial_pq_codebook"]),
            "Native initial codebook does not identify the C++ reference's pivot asset")
    label_counts = {item["label"]: item["eligible"] for item in reference["labels"]}
    require(build.get("labels") == len(label_counts) and build.get("memberships") == sum(label_counts.values()),
            "Built label/membership counts differ from reference postings")
    identity = json.loads(identity_path.read_text())
    require(identity.get("schema_version") == 1 and all(digest(identity.get(key)) for key in HASH_FIELDS),
            "Invalid native identity evidence hashes/schema")
    require(identity["source_archive_sha256"] == provenance(source_archive)["sha256"]
            and identity["build_log_sha256"] == provenance(build_path)["sha256"]
            and identity["query_log_sha256"] == provenance(query_path)["sha256"]
            and identity["codebook_sha256"] == identity["committed_codebook_sha256"],
            "Native source/log hashes or frozen codebook copy identity differ")
    require(type(identity.get("host")) is str and identity["host"]
            and type(identity.get("root")) is str and PurePosixPath(identity["root"]).is_absolute()
            and matches_absolute(identity["root"], build_exp["root"]), "Invalid native host/index identity")
    code_fields = ("cpp_compressed_codes_sha256", "committed_compressed_codes_sha256")
    codes = {"available": False}
    if any(key in identity for key in code_fields):
        require(all(digest(identity.get(key)) for key in code_fields), "Incomplete or invalid compressed-code hash pair")
        equal = identity[code_fields[0]] == identity[code_fields[1]]
        codes = {"available": True, "serialized_compressed_files_byte_identical": equal,
                 "rows": SHAPE["n_base"], "pq_bytes_per_row": config["pq_chunks"],
                 "cpp_sha256": identity[code_fields[0]], "native_sha256": identity[code_fields[1]],
                 "verification_scope": "Separately collected host hashes of the complete serialized compressed PQ files.",
                 "interpretation": ("Rust reencoding with the frozen C++ codebook produced byte-identical full-workload PQ codes."
                                    if equal else "Frozen codebook reuse is established; serialized PQ code bytes differ. No code-byte parity or isolated graph effect is asserted.")}
    native_profiles = [phase_summary(native_groups[breadth]) for breadth in BREADTHS]
    cpp_profiles = [cpp_summary(cpp_groups[breadth]) for breadth in BREADTHS]
    comparisons = []
    for native, cpp in zip(native_profiles, cpp_profiles):
        previous = {item["label"]: item for item in cpp["labels"]}
        labels = [{"label": item["label"], "queries": item["queries"], "eligible": previous[item["label"]]["eligible"],
                   "cpp_strict_id_recall": previous[item["label"]]["recall"], "native_strict_id_recall": item["recall"],
                   "difference_percentage_points": 100 * (item["recall"] - previous[item["label"]]["recall"])}
                  for item in native["labels"]]
        comparisons.append({"l": native["l"], "queries": 2000, "k": 10, "threads": 8,
                            "cpp_strict_id_recall": cpp["recall"], "native_strict_id_recall": native["recall"],
                            "difference_percentage_points": 100 * (native["recall"] - cpp["recall"]),
                            "qps_ratio_native_over_cpp": native["qps"] / cpp["qps"],
                            "latency_ratio_native_over_cpp": {metric: native[metric] / cpp[metric]
                                                              for metric in ("mean_us", "p50_us", "p95_us", "p99_us")},
                            "labels_with_lower_recall": sum(item["difference_percentage_points"] < -1e-10 for item in labels),
                            "labels_with_higher_recall": sum(item["difference_percentage_points"] > 1e-10 for item in labels),
                            "worst_label_difference": min(labels, key=lambda item: item["difference_percentage_points"]),
                            "labels": labels})
    return {"schema_version": 1, "status": "all assertions passed",
            "scope": "Native Rust-built 10M Stitched graph with frozen C++ PQ codebook versus a different C++-built graph; native centroid entry points.",
            "inputs": {name: provenance(path) for name, path in
                       (("build", build_path), ("query", query_path), ("cpp", cpp_path), ("source_archive", source_archive), ("identity", identity_path))},
            "identity": identity,
            "identity_verification_scope": "Local log/archive hashes verified; binary, committed-index and codebook hashes are linked declarations from separately collected native host evidence.",
            "frozen_key_source_sha256": archive_sources(source_archive), "cpp_manifest": cpp_manifest,
            "compressed_code_comparison": codes,
            "native_build": {"experiment": build_exp, "metadata": build,
                             "seconds_including_input_load": build["seconds_including_input_load"],
                             "whole_process_resources": resources(build_path.with_suffix(".resources.txt"))},
            "validation": {"native_build_rows": 2, "native_query_rows": 9, "cpp_query_rows": 9,
                           "breadths": list(BREADTHS), "repeats_per_configuration": 3,
                           "unique_prepared_queries": 2000, "all_invalid_short_duplicate_counters_zero": True,
                           "native_exact_fallback_and_matching_delta_counters_zero": True,
                           "frozen_external_and_committed_codebook_hashes_equal": True,
                           "per_label_query_sample_counts_equal": True, "authoritative_native_query_timing": True},
            "native_configurations": native_profiles, "cpp_configurations": cpp_profiles, "same_l_comparisons": comparisons,
            "interpretation": [
                "The native graph is newly built in Rust and uses its own centroid entries. The reference graph/entries were built in C++; this does not isolate a search-language or graph-builder effect.",
                "The saved C++ PQ codebook is copied into the native generation and used for Rust compression. Equal declared codebook hashes do not imply graph or neighbor-list identity.",
                "Strict-ID recall is compared against the same published exact ground truth per label. Different graphs can return different IDs, including equal-distance cutoff ties; no ordered per-query parity claim follows.",
                "Same-L speed comparisons are descriptive at each observed recall, not matched-quality speedups or exhaustive tuning results.",
                "Native timings cover public_snapshot_search_call; C++ timings cover public_search_call. Each includes its public query API work, with full same-process warmup and native cache-bypassed SSD reads.",
                "Build duration and whole-build-process resources remain separate from query latency/QPS. Query values are medians of three repeats; median percentiles are not pooled tails or confidence intervals.",
                "The same 2,000 queries are reused across breadths and repeats. Per-label differences do not establish statistical significance or production latency guarantees."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("build", "query", "cpp", "source-archive", "identity-evidence", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(args.output.resolve() not in {path.resolve() for path in
                (args.build, args.query, args.cpp, args.source_archive, args.identity_evidence)}, "Output must not overwrite inputs")
    result = summarize(args.build, args.query, args.cpp, args.source_archive, args.identity_evidence)
    result["analysis_script"] = provenance(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "same_l_comparisons": [{key: value for key, value in row.items() if key != "labels"}
                                             for row in result["same_l_comparisons"]]}))


if __name__ == "__main__":
    main()
