#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Freeze, calibrate, then evaluate float32 filtered SSD engines and a global baseline.

The primary comparison shares a C++ graph/PQ pair and label entries. All engines
use the same requested L in fixed_l; target90 uses each engine's minimum measured
calibration L attaining aggregate strict-ID recall >= 90%. No held-out quality
is read until calibration and its policy are complete. Inline filtering may
expand capacity by its frozen multiplier and reports its actual IO/PQ/hops.
"""
import argparse
import datetime
import hashlib
from fractions import Fraction
import itertools
import json
import math
import mmap
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

ENGINES = ("cpp", "rust_single", "rust_inline")
SUFFIXES = ("_disk.index", "_disk.index_labels.txt", "_disk.index_labels_map.txt",
            "_disk.index_labels_to_medoids.txt", "_pq_pivots.bin", "_pq_compressed.bin")
DEFAULT_GRID = (10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 8192)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def matrix_shape(path):
    with Path(path).open("rb") as stream:
        return struct.unpack("<II", stream.read(8))


def check_record(item):
    require(record(item["path"]) == item, f"Frozen artifact changed: {item['path']}")


def verify_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    artifacts = manifest["artifacts"]
    for name in ("base.fbin", "query.fbin", "query_labels.u32bin", "query_labels.txt", "groundtruth.bin", "labels.txt"):
        require(name in artifacts, f"Prepared identity missing {name}")
    for name, item in artifacts.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Artifact names must be safe relative paths")
        path = directory / name
        require(sha(path) == item["sha256"], f"Prepared artifact changed: {path}")
        if "bytes" in item:
            require(path.stat().st_size == item["bytes"], f"Prepared artifact size changed: {path}")
    n, dims = matrix_shape(directory / "query.fbin")
    require(n > 0 and dims >= 32 and (directory / "query.fbin").stat().st_size == 8 + n * dims * 4,
            "Invalid float32 query matrix")
    require(matrix_shape(directory / "query_labels.u32bin") == (n, 1), "Query labels shape mismatch")
    ng, k = matrix_shape(directory / "groundtruth.bin")
    require(ng == n and k > 0 and (directory / "groundtruth.bin").stat().st_size == 8 + n * k * 8,
            "Exact ground truth must have IDs and distances")
    labels = read_u32(directory / "query_labels.u32bin")[2]
    for label in set(labels):
        name = f"label_{label}.u32bin"
        require(name in artifacts, f"Posting list missing from prepared manifest: {name}")
    text_labels = [int(line) for line in (directory / "query_labels.txt").read_text().splitlines()]
    require(text_labels == labels, "Text/binary query labels differ")
    return n, dims, k


def read_u32(path):
    data = Path(path).read_bytes()
    require(len(data) >= 8, f"Short matrix: {path}")
    n, d = struct.unpack_from("<II", data)
    require(len(data) >= 8 + n * d * 4, f"Truncated matrix: {path}")
    return n, d, list(struct.unpack_from(f"<{n*d}I", data, 8))


def percentile(values, fraction):
    values = sorted(values)
    require(values, "Empty latency sample")
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def audit_distances(row, prepared, engine, raw, results, n, k):
    """Direct f64 coordinate oracle; tolerance bounds f32 subtract/square/sum error.

    Use 4*gamma_(d+2) relative error and 1e-7 absolute error, where gamma_m=
    m*2^-24/(1-m*2^-24). The factor allows SIMD reduction order and f32 operation
    rounding. This checks returned distances, not ANN candidate completeness.
    """
    base_n, dimensions = matrix_shape(prepared / "base.fbin")
    require(matrix_shape(prepared / "query.fbin") == (n, dimensions), "Distance audit dimensions differ")
    require((prepared / "base.fbin").stat().st_size == 8 + base_n * dimensions * 4,
            "Distance audit base size differs")
    unit = (dimensions + 2) * 2. ** -24
    require(unit < 1, "Unsupported distance-error bound dimension")
    relative_tolerance = 4 * unit / (1 - unit)
    cpp_distances = None
    if engine == "cpp":
        require("result_distances_record" in row, "C++ returned distance artifact is required")
        payload = Path(row["result_distances_record"]["path"]).read_bytes()
        require(struct.unpack_from("<II", payload) == (n, k) and len(payload) == 8 + n*k*4,
                "C++ returned distance shape differs")
        cpp_distances = struct.unpack_from(f"<{n*k}f", payload, 8)
    checked, maximum_absolute_error = 0, 0.
    with (prepared / "base.fbin").open("rb") as base_file, (prepared / "query.fbin").open("rb") as query_file:
        with mmap.mmap(base_file.fileno(), 0, access=mmap.ACCESS_READ) as base, mmap.mmap(query_file.fileno(), 0, access=mmap.ACCESS_READ) as queries:
            for i, item in enumerate(raw):
                returned = item["returned_results"] if engine == "cpp" else item["returned"]
                ids = results[i*k:i*k+returned] if engine == "cpp" else item["ids"]
                distances = cpp_distances[i*k:i*k+returned] if engine == "cpp" else item["distances"]
                require(len(distances) == returned and all(isinstance(d, (int, float)) and math.isfinite(d) and d >= 0 for d in distances)
                        and all(a <= b for a, b in zip(distances, distances[1:])),
                        "Returned distances must be finite, nonnegative and nondecreasing")
                query = struct.unpack_from(f"<{dimensions}f", queries, 8+i*dimensions*4)
                for result_id, distance in zip(ids, distances):
                    require(0 <= result_id < base_n, "Returned vector ID outside corpus")
                    vector = struct.unpack_from(f"<{dimensions}f", base, 8+result_id*dimensions*4)
                    direct = math.fsum((float(a)-float(b))**2 for a, b in zip(query, vector))
                    error = abs(distance-direct)
                    require(error <= 1e-7 + relative_tolerance * direct,
                            f"Returned distance disagrees with direct coordinates: query={i}, ID={result_id}")
                    maximum_absolute_error = max(maximum_absolute_error, error)
                    checked += 1
    return {"checked_returned_distances": checked, "oracle": "direct_float64_coordinate_squared_l2",
            "relative_tolerance": relative_tolerance, "absolute_tolerance": 1e-7,
            "maximum_absolute_error": maximum_absolute_error}


def audit_output(row, prepared, engine):
    """Recompute strict recall/eligibility and percentiles from retained raw data."""
    n, k, truth = read_u32(prepared / "groundtruth.bin")
    _, _, labels = read_u32(prepared / "query_labels.u32bin")
    raw = [json.loads(line) for line in Path(row["query_details"]["path"]).read_text().splitlines()]
    results = None
    if engine == "cpp":
        raw = sorted((item for item in raw if item["kind"] == "query"), key=lambda item: item["query_index"])
        nr, kr, results = read_u32(row["result_ids_record"]["path"])
        require((nr, kr) == (n, k), "C++ result shape differs")
    require(len(raw) == n, "Incomplete raw query observations")
    postings = {label: read_u32(prepared / f"label_{label}.u32bin")[2] for label in set(labels)}
    posting_sets = {label: set(ids) for label, ids in postings.items()}
    groups = {}
    total_hits, short_results, times = 0, 0, []
    for i, item in enumerate(raw):
        require(item.get("query_index", item.get("query")) == i, "Query IDs must occur exactly once in order")
        if engine == "cpp":
            returned = item["returned_results"]
            ids = results[i*k:i*k+returned]
            latency = item["latency_us"]
        else:
            require(item["label"] == labels[i] and item["l"] == row["l"], "Raw query label/L mismatch")
            returned, ids, latency = item["returned"], item["ids"], item["micros"]
        require(type(returned) is int and 0 <= returned <= k and len(ids) == returned and len(set(ids)) == returned,
                "Duplicate or invalid raw results")
        short_results += int(returned < k)
        require(all(result in posting_sets[labels[i]] for result in ids), "Raw result violates exact filter")
        require(isinstance(latency, (int, float)) and math.isfinite(latency) and latency >= 0,
                "Invalid raw latency")
        hits = len(set(ids).intersection(truth[i*k:(i+1)*k]))
        total_hits += hits
        times.append(latency)
        groups.setdefault(labels[i], []).append((hits, latency))
    require(row["short_results"] == short_results, "Short-result count differs from raw IDs")
    require(math.isclose(row["recall"], total_hits / (n*k), abs_tol=1e-12), "Aggregate recall differs from raw IDs")
    require(math.isclose(row["p99_us"], percentile(times, .99), rel_tol=1e-9, abs_tol=1e-8), "Aggregate p99 differs from raw queries")
    require(len(row["labels"]) == len(groups), "Per-label coverage differs")
    for item in row["labels"]:
        group = groups[item["label"]]
        require(item["n"] == len(group) and item["eligible"] == len(postings[item["label"]]), "Per-label sample/eligible count differs")
        require(math.isclose(item["recall"], sum(pair[0] for pair in group) / (len(group)*k), abs_tol=1e-12), "Per-label recall differs from raw IDs")
        require(math.isclose(item["p99_us"], percentile([pair[1] for pair in group], .99), rel_tol=1e-9, abs_tol=1e-8), "Per-label p99 differs from raw queries")
    require(isinstance(row.get("wall_seconds"), (int, float)) and row["wall_seconds"] > 0
            and math.isclose(row["qps"], n / row["wall_seconds"], rel_tol=2e-9),
            "Mixed QPS differs from measured query count/wall time")
    distance_audit = audit_distances(row, prepared, engine, raw, results, n, k)
    return {"unique_queries": n, "strict_id_hits": total_hits, "total_truth_ids": n*k,
            "raw_result_filter_and_duplicate_checks": "pass", "raw_latency_percentiles": "pass",
            "distance_audit": distance_audit}


def assets(protocol):
    return [*protocol["index_assets"].values(), *protocol["sources"],
            *protocol["binaries"].values(), protocol["split_commitment"],
            protocol["calibration_manifest"], protocol["holdout_manifest"],
            *protocol["tool_sources"], *protocol.get("split_audit", {}).get("closure", []),
            *protocol.get("global_index_assets", {}).values(),
            *protocol.get("construction_evidence", [])]


def check_frozen(protocol):
    for item in assets(protocol):
        check_record(item)


def affinity(protocol):
    if protocol.get("functional_only"):
        return
    require(len(protocol["cpu_affinity"]) > 0 and len(set(protocol["cpu_affinity"])) == len(protocol["cpu_affinity"]), "CPU cores must be nonempty and unique")
    require(hasattr(os, "sched_getaffinity"), "Authoritative runner requires Linux CPU affinity")
    require(sorted(os.sched_getaffinity(0)) == protocol["cpu_affinity"],
            "Run on the protocol's exclusive physical cores")


def audit_split(calibration, validation, split_record):
    """Bind prepared partitions to their commitment and check leakage directly."""
    proof = json.loads(Path(split_record["path"]).read_text())
    require(proof.get("ann_results_used", proof.get("invariants", {}).get("ann_results_used")) is False
            and proof.get("fresh_validation_queries_searched", proof.get("invariants", {}).get("fresh_validation_queries_searched")) is False,
            "Split must be committed before ANN results or fresh validation searches")
    split_values = {}
    for split, directory in (("calibration", Path(calibration)), ("validation", Path(validation))):
        manifest = json.loads((directory / "manifest.json").read_text())
        require(manifest.get("split") == split, "Prepared split phase is incorrect")
        binding = manifest.get("split_commitment", {})
        require(binding.get("sha256") == split_record["sha256"]
                and binding.get("bytes") == split_record["bytes"],
                "Prepared manifest is bound to another split commitment")
        promised = proof["outputs"][split]
        require(manifest["query_count"] == promised["query_count"], "Committed query count differs")
        for name, item in promised["artifacts"].items():
            require(manifest["artifacts"][name]["sha256"] == item["sha256"]
                    and sha(directory / name) == item["sha256"],
                    f"Committed partition artifact differs: {split}/{name}")
        _, _, ids = read_u32(directory / "query_ids.u32bin")
        metadata = [json.loads(line) for line in (directory / "query_metadata.jsonl").read_text().splitlines()]
        require(len(ids) == len(metadata) == manifest["query_count"] and len(set(ids)) == len(ids),
                "Duplicate or missing query IDs/metadata")
        require([item.get("query_id", item.get("qid")) for item in metadata] == ids,
                "Query IDs differ from metadata")
        payload = (directory / "query.fbin").read_bytes()
        n, d = struct.unpack_from("<II", payload)
        require(n == len(ids) and len(payload) == 8+n*d*4, "Committed query vector shape differs")
        embeddings = {hashlib.sha256(payload[8+i*d*4:8+(i+1)*d*4]).hexdigest() for i in range(n)}
        require(len(embeddings) == n, "Duplicate byte-identical query embeddings within partition")
        families = {field: {item[field] for item in metadata if field in item}
                    for field in ("positive_item_id", "user_hash", "text_hash", "group")}
        families["numeric_asin"] = {item["original_filter_parameters"][2] for item in metadata
                                    if "original_filter_parameters" in item}
        require(any(families.values()), "Missing query-family metadata for leakage checks")
        split_values[split] = {"ids": set(ids), "embeddings": embeddings, "families": families}
    a, b = split_values["calibration"], split_values["validation"]
    require(a["ids"].isdisjoint(b["ids"]), "Query IDs leak across partitions")
    require(a["embeddings"].isdisjoint(b["embeddings"]), "Query embeddings leak across partitions")
    for field in a["families"]:
        require(a["families"][field].isdisjoint(b["families"][field]),
                f"Query family leaks across partitions: {field}")
    closure = []
    assignment_status = "no_group_assignment_file"
    if proof.get("group_assignments"):
        committed = proof["group_assignments"]
        # The commitment is immutable across hosts. Resolve its supporting file
        # beside the actual commitment rather than requiring the original Mac path.
        actual_path = Path(split_record["path"]).parent / Path(committed["path"]).name
        actual = record(actual_path)
        require(actual["sha256"] == committed["sha256"] and actual["bytes"] == committed["bytes"],
                "Committed group assignments changed or were not transferred")
        closure.append(actual)
        assignments = json.loads(actual_path.read_text())
        if assignments and "source_query_rows" in assignments[0]:
            retained = {split: set() for split in ("calibration", "validation")}
            all_source_rows = set()
            for item in assignments:
                require(item["split"] in retained and all_source_rows.isdisjoint(item["source_query_rows"]),
                        "Group assignments overlap or have invalid partition names")
                require(set(item["retained_query_rows"]).issubset(item["source_query_rows"]),
                        "Retained group rows are outside their source component")
                all_source_rows.update(item["source_query_rows"])
                retained[item["split"]].update(item["retained_query_rows"])
            for split, directory in (("calibration", Path(calibration)), ("validation", Path(validation))):
                metadata = [json.loads(line) for line in (directory / "query_metadata.jsonl").read_text().splitlines()]
                require({item["id"] for item in metadata} == retained[split],
                        "Prepared queries do not match committed retained source groups")
        elif assignments and "query_id" in assignments[0]:
            by_id, group_phase = {}, {}
            for item in assignments:
                require(item["query_id"] not in by_id, "Duplicate query in group assignments")
                by_id[item["query_id"]] = item
                require(group_phase.setdefault(item["group"], item["split"]) == item["split"],
                        "A committed source group crosses partitions")
            for split, directory in (("calibration", Path(calibration)), ("validation", Path(validation))):
                metadata = [json.loads(line) for line in (directory / "query_metadata.jsonl").read_text().splitlines()]
                require({item["query_id"] for item in metadata} == {qid for qid, item in by_id.items() if item["split"] == split and item["retained"]},
                        "Prepared queries do not match committed retained source groups")
                require(all(item["group"] == by_id[item["query_id"]]["group"] for item in metadata),
                        "Prepared query group differs from source commitment")
        else:
            raise ValueError("Unsupported group-assignment schema")
        assignment_status = "all_selected_queries_bound_to_full_source_groups"
    return {"query_ids_disjoint": True, "query_embeddings_disjoint": True,
            "query_family_fields_disjoint": [field for field, values in a["families"].items() if values],
            "bound_split_output_artifacts": True, "group_assignment_binding": assignment_status, "closure": closure}


def engine_orders(engines):
    if len(engines) == 3:
        return [list(order) for order in itertools.permutations(engines)]
    require(len(engines) == 4 and len(set(engines)) == 4, "Expected three or four distinct engines")
    rotations = [engines[i:]+engines[:i] for i in range(4)]
    return rotations + [list(reversed(order)) for order in rotations]


def freeze(args):
    require(args.out and args.prepared and args.holdout_prepared and args.split_commitment
            and args.cpp_bin_dir and args.rust_binary and args.index_prefix and args.sources,
            "Freeze requires paths to both prepared sets, split proof, binaries, index and source archives")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    require(args.prepared.resolve() != args.holdout_prepared.resolve(), "Calibration and holdout must differ")
    cal_manifest = record(args.prepared / "manifest.json")
    hold_manifest = record(args.holdout_prepared / "manifest.json")
    require(cal_manifest["sha256"] != hold_manifest["sha256"], "Calibration and holdout manifests must differ")
    cal = json.loads(args.prepared.joinpath("manifest.json").read_text())
    hold = json.loads(args.holdout_prepared.joinpath("manifest.json").read_text())
    require(type(cal.get("base_vectors")) is int and cal["base_vectors"] > 0
            and cal["base_vectors"] == hold.get("base_vectors")
            and cal["artifacts"]["labels.txt"]["sha256"] == hold["artifacts"]["labels.txt"]["sha256"]
            and cal["artifacts"]["base.fbin"]["sha256"] == hold["artifacts"]["base.fbin"]["sha256"],
            "Calibration and holdout must share the exact corpus and label membership")
    require(sorted(set(args.grid)) == args.grid and min(args.grid) >= 10,
            "Calibration grid must be sorted, unique and >= K=10")
    verify_prepared(args.prepared)
    verify_prepared(args.holdout_prepared)
    split_record = record(args.split_commitment)
    split_audit = audit_split(args.prepared, args.holdout_prepared, split_record)
    index_build_path = Path(str(args.index_prefix.resolve()) + ".manifest.json")
    index_build = json.loads(index_build_path.read_text())
    require(index_build.get("status") == "complete" and index_build.get("base_sha256") == cal["artifacts"]["base.fbin"]["sha256"],
            "Filtered construction manifest does not bind the prepared corpus")
    filtered_assets = {suffix: record(str(args.index_prefix.resolve()) + suffix) for suffix in SUFFIXES}
    for suffix, item in filtered_assets.items():
        require(index_build["artifacts"][suffix]["sha256"] == item["sha256"],
                "Filtered construction manifest does not bind index artifacts")
    construction_evidence = [record(index_build_path)]
    construction = {"shared_filtered_graph": {key: index_build[key] for key in ("label_degree", "stitched_degree", "build_l")}}
    global_assets = {}
    engines = list(ENGINES)
    orders = engine_orders(engines)
    if args.inline_prefix:
        global_assets = {suffix: record(str(args.inline_prefix.resolve()) + suffix) for suffix in SUFFIXES}
        for suffix in ("_pq_pivots.bin", "_pq_compressed.bin"):
            require(global_assets[suffix]["sha256"] == filtered_assets[suffix]["sha256"],
                    "Global adaptive index must share the exact original PQ pivots and per-ID codes")
        global_build_path = Path(str(args.inline_prefix.resolve()) + ".manifest.json")
        global_build = json.loads(global_build_path.read_text())
        require(global_build.get("status") == "complete" and global_build.get("base_sha256") == index_build["base_sha256"],
                "Global graph construction belongs to another corpus")
        require((global_build.get("label_degree"), global_build.get("stitched_degree"), global_build.get("build_l")) == (64, 64, 100),
                "Global adaptive baseline requires preregistered R64/buildL100 construction")
        for suffix, item in global_assets.items():
            require(global_build["artifacts"][suffix]["sha256"] == item["sha256"],
                    "Global construction manifest does not bind index artifacts")
        selection = global_build["global_entry_selection"]
        proof_record = record(selection["proof"]["path"])
        require(proof_record["sha256"] == selection["proof"]["sha256"], "Global entry selector proof changed")
        entry_proof = json.loads(Path(proof_record["path"]).read_text())
        require(entry_proof["base_sha256"] == index_build["base_sha256"] and entry_proof["query_inputs_used"] is False
                and entry_proof["entry_point"] == global_build["global_entry_point"] == global_build["disk_metadata"]["medoid"],
                "Global entry must come from the frozen base-only selector and match the graph header")
        construction_evidence += [record(global_build_path), proof_record]
        construction["global_graph"] = {key: global_build[key] for key in ("label_degree", "stitched_degree", "build_l", "global_entry_point")}
        engines.append("rust_inline_global")
        orders = engine_orders(engines)
    script = Path(__file__).resolve()
    cpp_runner = script.parent / "cpp-port/run_f32_reference.py"
    proto = {"schema_version": 1, "created": now(), "dataset": args.dataset,
             "scope": "shared_cpp_filtered_graph_runtime_comparison_with_optional_global_adaptive_baseline",
             "engines": engines, "global_index_assets": global_assets,
             "topology_construction": construction, "construction_evidence": construction_evidence,
             "global_index_prefix": str(args.inline_prefix.resolve()) if args.inline_prefix else None,
             "engine_topology": {engine: "global_graph" if engine == "rust_inline_global" else "shared_filtered_graph" for engine in engines},
             "functional_only": args.functional_smoke,
             "timing_authoritative": not args.functional_smoke,
             "calibration_manifest": cal_manifest, "holdout_manifest": hold_manifest,
             "split_commitment": split_record, "split_audit": split_audit,
             "binaries": {"cpp": record(args.cpp_bin_dir / "search_disk_index"),
                          "rust": record(args.rust_binary)},
             "sources": [record(path) for path in args.sources],
             "index_prefix": str(args.index_prefix.resolve()),
             "index_assets": filtered_assets,
             "tool_sources": [record(script), record(cpp_runner)],
             "cpp_bin_dir": str(args.cpp_bin_dir.resolve()), "cpp_runner": str(cpp_runner),
             "cpu_affinity": args.cores, "threads": len(args.cores), "beam": 4, "k": 10,
             "cache_nodes": 0, "fixed_l": 64, "calibration_grid": args.grid,
             "adaptive_sample": 1000, "adaptive_scale": 16,
             "starts": "label", "metric": "squared_l2", "scalar_type": "float32",
             "selection": "minimum_tested_requested_L_with_aggregate_strict_ID_recall_at_least_0.90_and_zero_validity_errors",
             "grid_stop_policy": "each_engine_stops_at_first_complete_90_percent_result_in_ascending_L_order",
             "quality_unit": "unique_query_not_repeated_timings",
             "evaluation_orders": orders,
             "latency_scope": "closed_loop_public_search_call_no_queue_or_network",
             "freshness": "holdout_quality_not_read_by_freeze_or_calibration"}
    save(out / "protocol.json", proto)
    print(json.dumps({"protocol": str(out / "protocol.json"), "sha256": sha(out / "protocol.json")}))


def valid_row(row):
    require(all(type(row.get(key)) is int and row[key] == 0 for key in
                ("filter_violations", "duplicate_results")), "Invalid or duplicate results")
    require(type(row.get("short_results")) is int and 0 <= row["short_results"] <= row["queries"],
            "Invalid short-result count")
    require(row.get("strict_id_recall") is True, "Strict ID recall contract missing")
    require(isinstance(row.get("recall"), (int, float)) and math.isfinite(row["recall"])
            and 0 <= row["recall"] <= 1, "Invalid recall")


def reaches_target(row, target=.9):
    ratio = Fraction(str(target))
    audit = row.get("raw_audit", {})
    total = audit.get("total_truth_ids", row["queries"] * row.get("k", 10))
    hits = audit.get("strict_id_hits", round(row["recall"] * total))
    require(type(hits) is int and type(total) is int and 0 <= hits <= total and total > 0,
            "Invalid integer calibration hit counts")
    require(math.isclose(row["recall"], hits/total, abs_tol=1e-12), "Calibration hit counts differ from recall")
    return hits * ratio.denominator >= total * ratio.numerator


def choose_target(rows, target=.9, engines=ENGINES):
    plan = {}
    for engine in engines:
        candidates = [row for row in rows if row["engine"] == engine]
        require(candidates, f"Missing calibration engine {engine}")
        require(len({row["l"] for row in candidates}) == len(candidates), "Duplicate calibration L")
        for row in candidates:
            valid_row(row)
        passing = [row for row in candidates if reaches_target(row, target) and row["short_results"] == 0]
        if passing:
            selected = min(passing, key=lambda row: row["l"])
            plan[engine] = {"status": "calibrated", "l": selected["l"],
                            "calibration_recall": selected["recall"],
                            "calibration_queries": selected["queries"]}
        else:
            plan[engine] = {"status": "unreached_within_grid", "l": None,
                            "maximum_tested_l": max(row["l"] for row in candidates),
                            "best_calibration_recall": max(row["recall"] for row in candidates)}
    return plan


def check_grid(protocol, rows, plan):
    for engine in protocol["engines"]:
        measured = sorted(row["l"] for row in rows if row["engine"] == engine)
        expected = [l for l in protocol["calibration_grid"] if plan[engine]["l"] is None or l <= plan[engine]["l"]]
        require(measured == expected, f"Calibration skipped or added a grid point for {engine}")


def run_engine(protocol, prepared, output, engine, breadth, authoritative):
    output.mkdir(parents=True, exist_ok=False)
    authoritative = authoritative and not protocol.get("functional_only", False)
    threads = protocol["threads"]
    if engine == "cpp":
        command = [sys.executable, protocol["cpp_runner"], "--bin-dir", protocol["cpp_bin_dir"],
                   "--prepared-data", str(prepared), "--output-dir", str(output / "cpp"),
                   "--name", "cpp", "--index-prefix", protocol["index_prefix"], "--query-only",
                   "--search-threads", str(threads), "--l", str(breadth), "--reps", "1",
                   "--k", str(protocol["k"]), "--beam", str(protocol["beam"]), "--cache-nodes", "0"]
        if authoritative:
            command.append("--timing-authoritative")
        aggregate = output / "cpp/cpp.normalized.jsonl"
    else:
        aggregate = output / "rust.jsonl"
        mode = "single" if engine == "rust_single" else "inline"
        prefix = protocol["global_index_prefix"] if engine == "rust_inline_global" else protocol["index_prefix"]
        starts = "global" if engine == "rust_inline_global" else protocol["starts"]
        command = [protocol["binaries"]["rust"]["path"], "--prefix", prefix,
                   "--prepared", str(prepared), "--out", str(aggregate), "--result-dump", str(output / "queries.jsonl"),
                   "--threads", str(threads), "--ls", str(breadth), "--reps", "1", "--k", str(protocol["k"]),
                   "--beam", str(protocol["beam"]), "--modes", mode, "--starts", starts,
                   "--bypass", "true", "--timing-authoritative", str(authoritative).lower(),
                   "--adaptive-sample", str(protocol["adaptive_sample"]), "--adaptive-scale", str(protocol["adaptive_scale"])]
    environment = os.environ.copy()
    environment.update(OMP_TOOL="disabled", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    start = time.monotonic()
    with (output / "process.log").open("x") as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
    execution = {"engine": engine, "command": command, "completed": now(),
                 "elapsed_seconds": time.monotonic() - start, "exit_code": process.returncode}
    save(output / "command.json", execution)
    require(process.returncode == 0, f"{engine} failed; see {output / 'process.log'}")
    rows = [json.loads(line) for line in aggregate.read_text().splitlines()]
    require(len(rows) == 1, "Expected exactly one measured repeat")
    row = rows[0]
    require(row["l"] == breadth and row["threads"] == threads and row["repeat"] == 0
            and row["beam"] == protocol["beam"] and row["k"] == protocol["k"]
            and row["warmup_queries"] == row["queries"], "Engine execution contract differs")
    valid_row(row)
    row.update(engine=engine, aggregate=record(aggregate), command_record=record(output / "command.json"))
    if engine == "cpp":
        row["query_details"] = record(row["raw_stats"])
        row["result_ids_record"] = record(row["result_ids"])
        if row.get("result_distances"):
            row["result_distances_record"] = record(row["result_distances"])
    else:
        row["query_details"] = record(output / "queries.jsonl")
    row["raw_audit"] = audit_output(row, prepared, engine)
    return row


def calibrate(protocol, args):
    check_frozen(protocol)
    affinity(protocol)
    prepared = Path(protocol["calibration_manifest"]["path"]).parent
    n, dims, k = verify_prepared(prepared)
    require(k == protocol["k"], "Calibration K differs")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    done = set()
    for breadth in protocol["calibration_grid"]:
        for engine in protocol["engines"]:
            if engine in done:
                continue
            row = run_engine(protocol, prepared, out / f"L{breadth}/{engine}", engine, breadth, False)
            require(row["queries"] == n, "Incomplete calibration")
            rows.append(row)
            if reaches_target(row) and row["short_results"] == 0:
                done.add(engine)
            print(json.dumps({"phase": "calibration", "engine": engine, "l": breadth, "recall": row["recall"]}), flush=True)
    save(out / "calibration-grid.json", {"schema_version": 1, "protocol": record(args.protocol), "rows": rows})
    plan = choose_target(rows, engines=protocol["engines"])
    check_grid(protocol, rows, plan)
    check_frozen(protocol)
    verify_prepared(prepared)
    save(out / "calibration.json", {"schema_version": 1, "completed": now(),
         "protocol": record(args.protocol), "dimensions": dims, "queries": n, "rows": rows,
         "target90": plan, "heldout_quality_read": False})


def evaluate(protocol, args):
    require(args.calibration, "Evaluation requires completed calibration")
    calibration = json.loads(args.calibration.read_text())
    require(calibration["protocol"] == record(args.protocol), "Calibration belongs to another frozen protocol")
    require(calibration["target90"] == choose_target(calibration["rows"], engines=protocol["engines"]), "Calibration policy altered")
    check_grid(protocol, calibration["rows"], calibration["target90"])
    for row in calibration["rows"]:
        for key in ("aggregate", "query_details", "command_record"):
            check_record(row[key])
        if "result_ids_record" in row:
            check_record(row["result_ids_record"])
        if "result_distances_record" in row:
            check_record(row["result_distances_record"])
    cal_prepared = Path(protocol["calibration_manifest"]["path"]).parent
    verify_prepared(cal_prepared)
    for row in calibration["rows"]:
        loaded = [json.loads(line) for line in Path(row["aggregate"]["path"]).read_text().splitlines()]
        require(len(loaded) == 1 and all(row.get(key) == value for key, value in loaded[0].items()),
                "Calibration row differs from retained engine aggregate")
        require(row["raw_audit"] == audit_output(row, cal_prepared, row["engine"]),
                "Calibration raw audit differs")
    check_frozen(protocol)
    affinity(protocol)
    prepared = Path(protocol["holdout_manifest"]["path"]).parent
    n, dims, k = verify_prepared(prepared)
    require(k == protocol["k"], "Heldout K differs")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # One evaluation for this committed split. Changing L or candidates afterwards
    # requires new data; retries preserve the failed evidence rather than overwrite.
    claim = Path(protocol["split_commitment"]["path"]).with_suffix(".f32-evaluation-claim.json")
    save(claim, {"created": now(), "protocol": record(args.protocol),
                 "calibration": record(args.calibration), "output": str(out)})
    rows = []
    for repeat, order in enumerate(protocol["evaluation_orders"]):
        for scope in ("fixed_l", "target90"):
            for engine in order:
                breadth = protocol["fixed_l"] if scope == "fixed_l" else calibration["target90"][engine]["l"]
                if breadth is None:
                    continue
                row = run_engine(protocol, prepared, out / f"r{repeat}/{scope}/{engine}", engine, breadth, True)
                require(row["queries"] == n, "Incomplete heldout workload")
                row.update(scope=scope, paired_repeat=repeat, order_in_repeat=order.index(engine))
                rows.append(row)
                print(json.dumps({"phase": "heldout", "scope": scope, "repeat": repeat,
                                  "engine": engine, "l": breadth, "recall": row["recall"],
                                  "qps": row["qps"], "p99_us": row["p99_us"]}), flush=True)
    check_frozen(protocol)
    verify_prepared(prepared)
    result = {"schema_version": 1, "completed": now(), "protocol": record(args.protocol),
              "calibration": record(args.calibration), "claim": record(claim),
              "queries": n, "dimensions": dims, "rows": rows,
              "quality_unit": "unique_query_not_repeated_timings",
              "functional_only": protocol.get("functional_only", False),
              "timing_authoritative": protocol["timing_authoritative"]}
    save(out / "evaluation.json", result)
    summary = []
    for scope in ("fixed_l", "target90"):
        for engine in protocol["engines"]:
            group = [row for row in rows if row["scope"] == scope and row["engine"] == engine]
            if not group:
                require(scope == "target90" and calibration["target90"][engine]["l"] is None,
                        "Unexpected missing engine/scope")
                summary.append({"scope": scope, "engine": engine, "status": "unreached_within_calibration_grid"})
                continue
            require(len(group) == len(protocol["evaluation_orders"]) and len({row["recall"] for row in group}) == 1,
                    "Repeated quality differs or timing repetitions incomplete")
            labels = []
            for label in sorted({item["label"] for item in group[0]["labels"]}):
                slices = [next(item for item in row["labels"] if item["label"] == label) for row in group]
                require(len({item["recall"] for item in slices}) == 1, "Per-label repeated quality differs")
                labels.append({"label": label, "queries": slices[0]["n"], "eligible": slices[0]["eligible"],
                               "recall": slices[0]["recall"],
                               "p99_us_median": statistics.median(item["p99_us"] for item in slices),
                               "p99_us_range": [min(item["p99_us"] for item in slices), max(item["p99_us"] for item in slices)]})
            summary.append({"scope": scope, "engine": engine, "l": group[0]["l"],
                            "recall": group[0]["recall"], "achieves_90_on_holdout": group[0]["recall"] >= .9,
                            "short_results": group[0]["short_results"],
                            "qps_median": statistics.median(row["qps"] for row in group),
                            "qps_range": [min(row["qps"] for row in group), max(row["qps"] for row in group)],
                            "p99_us_median": statistics.median(row["p99_us"] for row in group), "labels": labels})
    save(out / "summary.json", {"schema_version": 1, "evaluation": record(out / "evaluation.json"),
                                "summary": summary, "quality_queries": n,
                                "timing_repeats": len(protocol["evaluation_orders"]), "closed_loop_no_queue_or_network": True,
                                "functional_only": protocol.get("functional_only", False),
                                "timing_authoritative": protocol["timing_authoritative"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "calibrate", "evaluate"), required=True)
    for name in ("protocol", "out", "prepared", "holdout-prepared", "split-commitment", "cpp-bin-dir",
                 "rust-binary", "index-prefix", "inline-prefix", "calibration"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--sources", nargs="+", type=Path)
    parser.add_argument("--dataset", default="unspecified")
    parser.add_argument("--functional-smoke", action="store_true", help="Mac/synthetic functional checks; every timing explicitly non-authoritative")
    parser.add_argument("--grid", nargs="+", type=int, default=list(DEFAULT_GRID))
    parser.add_argument("--cores", nargs="+", type=int, default=list(range(8)))
    args = parser.parse_args()
    require(args.out, "--out is required")
    if args.phase == "freeze":
        freeze(args)
    else:
        require(args.protocol, "--protocol is required")
        protocol = json.loads(args.protocol.read_text())
        (calibrate if args.phase == "calibrate" else evaluate)(protocol, args)


if __name__ == "__main__":
    main()
