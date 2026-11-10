#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Audit float32 SSD results with committed C4/Books families and portable paths.

Strict ground-truth IDs remain primary. Distance-cutoff recall is a separately
named numerical-tie diagnostic. Repeated timings never enlarge quality sample
counts; paired confidence intervals resample original query-family components.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

import numpy as np

ENGINES = ("cpp", "rust_single", "rust_inline")
KNOWN_ENGINES = (*ENGINES, "rust_inline_global")
SCOPES = ("fixed_l", "target90")
SEED = 20261028
PATH_RELOCATIONS = []


def resolve_path(path):
    path = Path(path)
    for old, new in sorted(PATH_RELOCATIONS, key=lambda pair: len(pair[0].parts), reverse=True):
        if path.is_relative_to(old):
            return new / path.relative_to(old)
    return path


def configure_relocations(values):
    mappings = []
    for value in values:
        require("=" in value, "Relocation must have OLD_ABSOLUTE_PREFIX=NEW_ABSOLUTE_PREFIX form")
        old, new = map(Path, value.split("=", 1))
        require(old.is_absolute() and new.is_absolute(), "Relocation prefixes must be absolute paths")
        require(old != new and all(old != previous[0] for previous in mappings), "Duplicate or ineffective relocation prefix")
        mappings.append((old, new))
    PATH_RELOCATIONS[:] = mappings
    return mappings


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def check_record(value):
    path = resolve_path(value["path"])
    require(path.stat().st_size == value["bytes"] and sha256(path) == value["sha256"], f"Artifact changed: {path}")
    return path


def matrix(path, dtype):
    path = Path(path)
    header = np.fromfile(path, dtype="<u4", count=2)
    require(len(header) == 2, f"Truncated matrix: {path}")
    n, d = map(int, header)
    require(n > 0 and d > 0 and path.stat().st_size == 8 + n*d*np.dtype(dtype).itemsize,
            f"Malformed matrix: {path}")
    return np.memmap(path, mode="r", offset=8, shape=(n, d), dtype=dtype)


def percentile(values, fraction):
    values = sorted(values)
    require(values, "Empty percentile sample")
    return float(values[max(0, math.ceil(len(values)*fraction)-1)])


def query_families(metadata, recorded_assignments=None):
    """Use pre-dedup components when available, preserving duplicate bridges."""
    if recorded_assignments is not None:
        require(all("id" in row for row in metadata), "C4 query rows lack original preparation IDs")
        membership = {}
        for family, group in enumerate(recorded_assignments):
            for row in group["retained_query_rows"]:
                require(row not in membership, "Query occurs in multiple recorded families")
                membership[row] = family
        require(all(row["id"] in membership for row in metadata), "Query missing from recorded source family")
        return np.array([membership[row["id"]] for row in metadata], dtype=np.int64)
    n = len(metadata)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    seen = {}
    for i, row in enumerate(metadata):
        if "group" in row:
            keys = [("source_group", row["group"])]
        else:
            require(all(key in row for key in ("positive_item_id", "user_hash", "text_hash")),
                    "Query metadata lacks audited family information")
            keys = [(key, row[key]) for key in ("positive_item_id", "user_hash", "text_hash")]
        for key in keys:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def committed_query_families(metadata, assignments, split):
    """Bind selected rows to either complete preregistered family schema."""
    require(split in ("calibration", "validation") and assignments, "Invalid committed family partition")
    c4_keys = {"source_query_rows", "retained_query_rows", "split"}
    books_keys = {"query_id", "group", "retained", "split"}
    if all(c4_keys.issubset(group) for group in assignments):
        require(all("id" in row for row in metadata), "C4 query rows lack original preparation IDs")
        source_rows, retained, groups = set(), {}, {}
        for family, group in enumerate(assignments):
            require(group["split"] in ("calibration", "validation"), "Invalid C4 source family partition")
            original = group["source_query_rows"]
            kept = group["retained_query_rows"]
            require(len(set(original)) == len(original) and source_rows.isdisjoint(original), "C4 source families overlap")
            require(len(set(kept)) == len(kept) and set(kept).issubset(original), "Invalid C4 retained family rows")
            source_rows.update(original)
            for row in kept:
                require(row not in retained, "Query occurs in multiple retained C4 families")
                retained[row], groups[row] = group["split"], family
        ids = [row["id"] for row in metadata]
        require(len(ids) == len(set(ids)) and set(ids) == {row for row, phase in retained.items() if phase == split},
                "Selected C4 rows differ from committed retained partition")
        return np.array([groups[row] for row in ids], dtype=np.int64), "c4_pre_dedup_connected_components"
    if all(books_keys.issubset(group) for group in assignments):
        by_id, group_phase = {}, {}
        for group in assignments:
            qid = group["query_id"]
            require(type(qid) is int and qid >= 0 and type(group["group"]) is int
                    and type(group["retained"]) is bool and group["split"] in ("calibration", "validation"),
                    "Invalid Books source family record")
            require(qid not in by_id, "Duplicate Books query ID in committed assignments")
            by_id[qid] = group
            require(group_phase.setdefault(group["group"], group["split"]) == group["split"],
                    "A committed Books source family crosses partitions")
        require(all("query_id" in row and "group" in row for row in metadata), "Books query metadata lacks source IDs/groups")
        ids = [row["query_id"] for row in metadata]
        expected = {qid for qid, group in by_id.items() if group["split"] == split and group["retained"]}
        require(len(ids) == len(set(ids)) and set(ids) == expected, "Selected Books queries differ from committed retained partition")
        for row in metadata:
            source = by_id[row["query_id"]]
            require(row["group"] == source["group"], "Selected Books group differs from committed source family")
            if "asin" in source:
                parameters = row.get("original_filter_parameters", [])
                require(len(parameters) == 5 and parameters[2] == source["asin"], "Selected Books ASIN differs from committed source family")
        # Keep the actual preregistered numeric group IDs. No reconstruction from
        # only retained queries can split a family connected by a removed row.
        return np.array([row["group"] for row in metadata], dtype=np.int64), "books_committed_numeric_asin_embedding_groups"
    raise ValueError("Unsupported or mixed committed family schema")


def rows_by_label(query_labels):
    groups = {}
    for row, label in enumerate(query_labels):
        groups.setdefault(int(label), []).append(row)
    return {label: np.asarray(rows, dtype=np.int64) for label, rows in groups.items()}


def bootstrap_families(hits, family_ids, k, samples=4000, seed=SEED):
    """Percentile cluster-bootstrap absolute recall and paired method differences.

    hits has one unique query per row and one engine per column. Resample whole
    families uniformly, retaining all queries in each sampled family. Ratios use
    the sampled number of requests; unequal family sizes remain intact.
    """
    hits = np.asarray(hits, dtype=np.float64)
    require(hits.ndim == 2 and len(hits) == len(family_ids) and k > 0 and samples > 0,
            "Invalid bootstrap dimensions")
    unique, inverse = np.unique(family_ids, return_inverse=True)
    count = len(unique)
    point = hits.sum(axis=0)/(len(hits)*k)
    if count < 2:
        return {"families": count, "samples": samples, "recall": point.tolist(),
                "recall_ci95": [None] * hits.shape[1], "paired_difference_ci95": None,
                "limitation": "A single independent family cannot estimate population uncertainty"}
    totals = np.zeros((count, hits.shape[1]))
    sizes = np.bincount(inverse).astype(np.float64)
    np.add.at(totals, inverse, hits)
    draws = np.empty((samples, hits.shape[1]))
    rng = np.random.default_rng(seed)
    for first in range(0, samples, 128):
        last = min(first + 128, samples)
        selected = rng.integers(count, size=(last-first, count))
        draws[first:last] = totals[selected].sum(axis=1)/(sizes[selected].sum(axis=1)[:, None]*k)
    intervals = np.quantile(draws, [.025, .975], axis=0).T
    paired = {}
    for a in range(hits.shape[1]):
        for b in range(a):
            paired[f"{a}-{b}"] = np.quantile(draws[:, a]-draws[:, b], [.025, .975]).tolist()
    return {"families": count, "samples": samples, "recall": point.tolist(),
            "recall_ci95": intervals.tolist(), "paired_difference_ci95": paired}


def direct_distances(base, queries, ids):
    ids = np.asarray(ids)
    require(ids.ndim == 2 and len(ids) == len(queries), "Result/query shape differs")
    out = np.full(ids.shape, np.nan, dtype=np.float64)
    for first in range(0, len(ids), 256):
        last = min(first+256, len(ids))
        block = ids[first:last]
        valid = block >= 0
        require(np.all(block[valid] < len(base)), "Returned ID outside base")
        safe = np.where(valid, block, 0)
        difference = np.asarray(base[safe], dtype=np.float64) - np.asarray(queries[first:last], dtype=np.float64)[:, None, :]
        values = np.einsum("ijk,ijk->ij", difference, difference)
        values[~valid] = np.nan
        out[first:last] = values
    return out


def recall_counts(ids, returned, truth_ids, direct, cutoff, dimensions):
    """Strict IDs primary, cutoff with f64 numerical error band diagnostic only."""
    n, k = truth_ids.shape
    strict = np.zeros(n, dtype=np.int64)
    tolerant = np.zeros(n, dtype=np.int64)
    tolerance = np.finfo(np.float64).eps * 64 * dimensions * np.maximum(1, np.abs(cutoff))
    for i in range(n):
        row = ids[i, :returned[i]]
        require(len(set(map(int, row))) == len(row), "Duplicate returned ID")
        strict[i] = len(set(map(int, row)).intersection(map(int, truth_ids[i])))
        within = direct[i, :returned[i]] <= cutoff[i] + tolerance[i]
        truth_set = set(map(int, truth_ids[i]))
        for result, distance, credited in zip(row, direct[i, :returned[i]], within):
            require(not credited or int(result) in truth_set or abs(distance-cutoff[i]) <= tolerance[i],
                    "A returned non-truth ID is strictly closer than the claimed exact cutoff")
        tolerant[i] = int(np.sum(within))
    require(np.all(tolerant >= strict), "Exact-truth IDs exceed independently recomputed cutoff")
    return strict, tolerant, tolerance


def read_observations(row, prepared, truth_ids, base, queries, query_labels, postings, cutoff, label_rows=None):
    engine = row["engine"]
    n, k = truth_ids.shape
    raw_path = check_record(row["query_details"])
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    ids = np.full((n, k), -1, dtype="<i8")
    distances = np.full((n, k), np.nan, dtype="<f8")
    returned = np.zeros(n, dtype=np.int64)
    latency = np.zeros(n)
    if engine == "cpp":
        raw = sorted((item for item in raw if item["kind"] == "query"), key=lambda item: item["query_index"])
        cpp_ids = matrix(check_record(row["result_ids_record"]), "<u4")
        cpp_distances = matrix(check_record(row["result_distances_record"]), "<f4")
        require(cpp_ids.shape == cpp_distances.shape == (n, k), "C++ result shapes differ")
    require(len(raw) == n, "Incomplete raw observations")
    for i, item in enumerate(raw):
        require(item.get("query_index", item.get("query")) == i, "Repeated/unordered raw query ID")
        count = item["returned_results"] if engine == "cpp" else item["returned"]
        require(type(count) is int and 0 <= count <= k, "Invalid returned count")
        if engine == "cpp":
            row_ids, row_distances = cpp_ids[i, :count], cpp_distances[i, :count]
            latency[i] = item["latency_us"]
        else:
            require(item["label"] == query_labels[i] and item["l"] == row["l"], "Rust raw label/L differs")
            row_ids, row_distances = item["ids"], item["distances"]
            latency[i] = item["micros"]
        require(len(row_ids) == len(row_distances) == count, "Invalid raw result lengths")
        ids[i, :count], distances[i, :count], returned[i] = row_ids, row_distances, count
        members = postings[int(query_labels[i])]
        positions = np.searchsorted(members, row_ids)
        require(np.all(positions < len(members)) and np.array_equal(members[positions], row_ids), "Returned result violates filter")
        require(np.all(np.isfinite(row_distances)) and np.all(np.asarray(row_distances) >= 0)
                and np.all(np.diff(row_distances) >= 0), "Invalid/unsorted returned distances")
    require(np.all(np.isfinite(latency)) and np.all(latency >= 0), "Invalid raw latency")
    direct = direct_distances(base, queries, ids)
    relative = 4*((queries.shape[1]+2)*2.**-24)/(1-(queries.shape[1]+2)*2.**-24)
    valid = ids >= 0
    error = np.abs(distances-direct)
    require(np.all(error[valid] <= 1e-7 + relative*direct[valid]), "Returned distances disagree with coordinates")
    strict, tolerant, tolerance = recall_counts(ids, returned, truth_ids, direct, cutoff, queries.shape[1])
    require(row["queries"] == n and row["short_results"] == int(np.sum(returned < k)), "Query/short-result totals differ")
    require(math.isclose(row["recall"], float(strict.sum()/(n*k)), abs_tol=1e-12), "Strict recall differs from raw IDs")
    require(math.isclose(row["p99_us"], percentile(latency, .99), rel_tol=1e-9, abs_tol=1e-8), "Mixed p99 differs")
    require(math.isclose(row["qps"], n/row["wall_seconds"], rel_tol=2e-9), "Mixed QPS must derive from measured wall time")
    require(row["filter_violations"] == row["duplicate_results"] == 0, "Validity errors must remain zero")
    label_rows = rows_by_label(query_labels) if label_rows is None else label_rows
    require(len(row["labels"]) == len(label_rows) and {entry["label"] for entry in row["labels"]} == set(label_rows), "Per-label observation coverage differs")
    for entry in row["labels"]:
        selected = label_rows[entry["label"]]
        require(entry["n"] == len(selected) and entry["eligible"] == len(postings[entry["label"]]), "Per-label counts differ")
        require(math.isclose(entry["recall"], float(strict[selected].sum()/(len(selected)*k)), abs_tol=1e-12), "Per-label strict recall differs")
        require(math.isclose(entry["p99_us"], percentile(latency[selected], .99), rel_tol=1e-9, abs_tol=1e-8), "Per-label p99 differs")
    return {"strict": strict, "cutoff": tolerant, "latency": latency,
            "returned": returned, "distance_error_max": float(error[valid].max(initial=0)),
            "cutoff_tolerance_max": float(tolerance.max())}


def ci_pp(interval):
    return None if interval is None else [float(x*100) for x in interval]


def timing_summary(rows):
    return {"repeats": len(rows), "requested_l": rows[0]["l"],
            "qps_median": statistics.median(r["qps"] for r in rows),
            "qps_range": [min(r["qps"] for r in rows), max(r["qps"] for r in rows)],
            "mixed_p99_us_median": statistics.median(r["p99_us"] for r in rows),
            "mixed_p99_us_range": [min(r["p99_us"] for r in rows), max(r["p99_us"] for r in rows)],
            "short_results": rows[0]["short_results"]}


def scope_summary(scope, rows, observations, query_labels, families, postings, metadata_labels, k, bootstrap_samples,
                  engine_order=ENGINES, expected_repeats=6):
    by_engine = {engine: sorted([r for r in rows if r["engine"] == engine], key=lambda r: r["paired_repeat"]) for engine in engine_order}
    engines = [engine for engine in engine_order if by_engine[engine]]
    require(engines, "Empty evaluation scope")
    for engine in engines:
        require(len(by_engine[engine]) == expected_repeats and [r["paired_repeat"] for r in by_engine[engine]] == list(range(expected_repeats)), "Timing repetitions incomplete")
        first = observations[id(by_engine[engine][0])]
        require(all(np.array_equal(observations[id(row)][name], first[name]) for row in by_engine[engine] for name in ("strict", "cutoff", "returned")),
                "Repeated quality differs; repeated timings cannot be pooled as new quality samples")
    strict = np.stack([observations[id(by_engine[e][0])]["strict"] for e in engines], axis=1)
    cutoff = np.stack([observations[id(by_engine[e][0])]["cutoff"] for e in engines], axis=1)
    global_ci = bootstrap_families(strict, families, k, bootstrap_samples)
    result = {"scope": scope, "quality_queries": len(query_labels), "families": global_ci["families"],
              "engines": {}, "comparisons_to_cpp": {}, "labels": []}
    cpp_col = engines.index("cpp") if "cpp" in engines else None
    label_rows = rows_by_label(query_labels)
    label_timings = {}
    for engine in engines:
        for row in by_engine[engine]:
            indexed = {entry["label"]: entry for entry in row["labels"]}
            require(len(indexed) == len(row["labels"]) and set(indexed) == set(label_rows), "Timing label coverage differs")
            label_timings[id(row)] = indexed
    for column, engine in enumerate(engines):
        result["engines"][engine] = {**timing_summary(by_engine[engine]),
            "strict_id_recall": float(strict[:, column].sum()/(len(strict)*k)),
            "strict_id_recall_ci95": global_ci["recall_ci95"][column],
            "distance_cutoff_recall_diagnostic": float(cutoff[:, column].sum()/(len(cutoff)*k)),
            "cutoff_additional_hits": int((cutoff[:, column]-strict[:, column]).sum()),
            "queries_with_cutoff_tie_benefit": int(np.sum(cutoff[:, column] > strict[:, column])),
            "achieves_90_on_holdout": bool(strict[:, column].sum()/(len(strict)*k) >= .9)}
        if cpp_col is not None and engine != "cpp":
            base_rows, other_rows = by_engine["cpp"], by_engine[engine]
            ratios = [b["qps"]/a["qps"] for a, b in zip(base_rows, other_rows)]
            tails = [b["p99_us"]/a["p99_us"] for a, b in zip(base_rows, other_rows)]
            pair_ci = global_ci["paired_difference_ci95"]
            result["comparisons_to_cpp"][engine] = {
                "strict_id_recall_difference_pp": float((strict[:, column]-strict[:, cpp_col]).sum()/(len(strict)*k)*100),
                "paired_family_ci95_pp": None if pair_ci is None else ci_pp(pair_ci[f"{column}-{cpp_col}"]),
                "qps_ratio_paired_median": statistics.median(ratios), "qps_ratio_range": [min(ratios), max(ratios)],
                "p99_ratio_paired_median": statistics.median(tails), "p99_ratio_range": [min(tails), max(tails)]}
    for label in sorted(postings):
        selected = label_rows.get(label, np.empty(0, dtype=np.int64))
        n = len(selected)
        if not n:
            continue
        ci = bootstrap_families(strict[selected], families[selected], k, bootstrap_samples, SEED+int(label))
        flags = []
        if n == 1:
            flags.append("single_query_p99_is_one_observed_latency")
        if n < 1000:
            flags.append("fewer_than10_expected_observations_in_the_top1_percent_tail")
        if ci["families"] < 30:
            flags.append("fewer_than30_independent_query_families")
        entry = {"label": int(label), "category": metadata_labels.get(label), "queries": n,
                 "families": ci["families"], "eligible_vectors": len(postings[label]),
                 "limitations": flags, "engines": {}, "comparisons_to_cpp": {}}
        for column, engine in enumerate(engines):
            p99 = [label_timings[id(row)][label]["p99_us"] for row in by_engine[engine]]
            entry["engines"][engine] = {"requested_l": by_engine[engine][0]["l"],
                "strict_id_hits": int(strict[selected, column].sum()), "truth_ids": n*k,
                "strict_id_recall": float(strict[selected, column].sum()/(n*k)),
                "strict_id_recall_ci95": ci["recall_ci95"][column],
                "distance_cutoff_recall_diagnostic": float(cutoff[selected, column].sum()/(n*k)),
                "cutoff_additional_hits": int((cutoff[selected, column]-strict[selected, column]).sum()),
                "queries_with_cutoff_tie_benefit": int(np.sum(cutoff[selected, column] > strict[selected, column])),
                "p99_us_median": statistics.median(p99), "p99_us_range": [min(p99), max(p99)]}
            if cpp_col is not None and engine != "cpp":
                pair_ci = ci["paired_difference_ci95"]
                entry["comparisons_to_cpp"][engine] = {
                    "strict_id_recall_difference_pp": float((strict[selected, column]-strict[selected, cpp_col]).sum()/(n*k)*100),
                    "paired_family_ci95_pp": None if pair_ci is None else ci_pp(pair_ci[f"{column}-{cpp_col}"])}
        result["labels"].append(entry)
    return result


def analyze(args):
    evaluation_path = args.evaluation.resolve()
    evaluation = json.loads(evaluation_path.read_text())
    protocol_path = check_record(evaluation["protocol"])
    protocol = json.loads(protocol_path.read_text())
    engine_order = tuple(engine for engine in KNOWN_ENGINES if engine in protocol.get("engines", ENGINES))
    require(engine_order and set(engine_order) == set(protocol.get("engines", ENGINES)), "Unknown frozen engine profile")
    expected_repeats = len(protocol["evaluation_orders"])
    require(expected_repeats > 0, "Empty timing schedule")
    for item in [*protocol.get("index_assets", {}).values(), *protocol.get("global_index_assets", {}).values(), *protocol.get("sources", []),
                 *protocol.get("binaries", {}).values(), *protocol.get("tool_sources", []),
                 *[protocol[key] for key in ("split_commitment", "calibration_manifest", "holdout_manifest") if key in protocol]]:
        check_record(item)
    calibration_path = check_record(evaluation["calibration"])
    calibration = json.loads(calibration_path.read_text())
    require(calibration["protocol"]["sha256"] == evaluation["protocol"]["sha256"], "Calibration belongs to another protocol")
    check_record(evaluation["claim"])
    prepared = check_record(protocol["holdout_manifest"]).parent
    manifest = json.loads((prepared / "manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        require(sha256(prepared / name) == expected["sha256"], f"Prepared data changed: {name}")
    for row in evaluation["rows"]:
        for key in ("aggregate", "command_record", "query_details", "result_ids_record", "result_distances_record"):
            if key in row:
                check_record(row[key])
        aggregate = [json.loads(line) for line in resolve_path(row["aggregate"]["path"]).read_text().splitlines()]
        require(len(aggregate) == 1 and all(row.get(key) == value for key, value in aggregate[0].items()),
                "Evaluation row differs from retained engine aggregate")
    base, queries = matrix(prepared / "base.fbin", "<f4"), matrix(prepared / "query.fbin", "<f4")
    labels = matrix(prepared / "query_labels.u32bin", "<u4").ravel()
    n, dims = queries.shape
    require(base.shape[1] == dims and len(labels) == n == evaluation["queries"], "Corpus/query/evaluation shapes differ")
    truth_path = prepared / "groundtruth.bin"
    ng, k = map(int, np.fromfile(truth_path, dtype="<u4", count=2))
    require(ng == n and k == 10 and truth_path.stat().st_size == 8+n*k*8, "Truth shape differs")
    truth_ids = np.memmap(truth_path, mode="r", offset=8, shape=(n, k), dtype="<i4")
    truth_d64 = direct_distances(base, queries, truth_ids)
    require(np.all(np.isfinite(truth_d64)), "Nonfinite exact truth coordinates")
    cutoff = truth_d64.max(axis=1)
    postings = {int(label): matrix(prepared / f"label_{label}.u32bin", "<u4").ravel() for label in np.unique(labels)}
    for i, label in enumerate(labels):
        members = postings[int(label)]
        positions = np.searchsorted(members, truth_ids[i])
        require(np.all(positions < len(members)) and np.array_equal(members[positions], truth_ids[i]), "Exact truth violates label")
        require(len(set(map(int, truth_ids[i]))) == k, "Duplicate exact truth ID")
    metadata = [json.loads(line) for line in (prepared / "query_metadata.jsonl").read_text().splitlines()]
    require(len(metadata) == n and all(row.get("label", int(labels[i])) == labels[i] for i, row in enumerate(metadata)), "Query metadata differs")
    qids = matrix(prepared / "query_ids.u32bin", "<u4").ravel()
    require([row.get("qid", row.get("query_id")) for row in metadata] == list(qids), "Original query IDs differ")
    family_binding = "metadata_only_synthetic_fixture"
    commitment_record = manifest.get("split_commitment")
    commitment = json.loads(check_record(commitment_record).read_text()) if commitment_record is not None else None
    if commitment is not None and commitment.get("group_assignments"):
        group_path = check_record(commitment["group_assignments"])
        families, family_binding = committed_query_families(metadata, json.loads(group_path.read_text()), manifest.get("split"))
    else:
        require(protocol.get("functional_only", False) or protocol.get("timing_authoritative") is False,
                "Authoritative analysis requires committed source-family assignments")
        families = query_families(metadata)
    metadata_labels = {int(x.get("original_label", x.get("label"))): x.get("category", x.get("name")) for x in manifest["labels"]}
    observations = {}
    label_rows = rows_by_label(labels)
    for row in evaluation["rows"]:
        require(row["engine"] in engine_order and row["scope"] in SCOPES, "Unknown engine/scope")
        expected_l = protocol["fixed_l"] if row["scope"] == "fixed_l" else calibration["target90"][row["engine"]]["l"]
        require(row["l"] == expected_l, "Held-out L differs from frozen fixed/calibrated policy")
        require(row["order_in_repeat"] == protocol["evaluation_orders"][row["paired_repeat"]].index(row["engine"]),
                "Measured engine order differs from balanced protocol")
        observations[id(row)] = read_observations(row, prepared, truth_ids, base, queries, labels, postings, cutoff, label_rows)
    scopes = []
    for scope in SCOPES:
        rows = [row for row in evaluation["rows"] if row["scope"] == scope]
        expected_engines = set(engine_order) if scope == "fixed_l" else {engine for engine in engine_order if calibration["target90"][engine]["l"] is not None}
        require({row["engine"] for row in rows} == expected_engines, "Required evaluation engine/scope is missing")
        if rows:
            scopes.append(scope_summary(scope, rows, observations, labels, families, postings, metadata_labels, k, args.bootstrap_samples,
                                        engine_order, expected_repeats))
    result = {"schema_version": 1, "analyzer": record(Path(__file__)), "evaluation": record(evaluation_path),
              "protocol": record(protocol_path), "manifest": record(prepared / "manifest.json"),
              "path_relocations": [{"original_prefix": str(old), "local_prefix": str(new)} for old, new in PATH_RELOCATIONS],
              "relocated_inputs_require_original_sha256_and_bytes": True,
              "dataset": protocol["dataset"], "engine_profiles": list(engine_order),
              "balanced_timing_orders": expected_repeats, "family_assignment_binding": family_binding,
              "quality_unit": "unique_request_with_family_cluster_bootstrap",
              "timing_repetitions_are_not_quality_samples": True, "bootstrap": {"seed": SEED,
                  "samples": args.bootstrap_samples, "method": "paired percentile whole-family resampling; unequal family sizes retained"},
              "strict_id_recall_primary": True,
              "calibrated_90_percent_policies": calibration["target90"],
              "timing_authoritative": protocol.get("timing_authoritative", True),
              "distance_cutoff_diagnostic": {"definition": "Returned eligible IDs whose directly recomputed float64 squared L2 is <= exact kth-neighbor distance plus conservative float64 numerical band; no float32 distance tolerance used for hit credit",
                  "maximum_cutoff_tolerance": max(x["cutoff_tolerance_max"] for x in observations.values()),
                  "changes_90_percent_selection": False},
              "limitations": ["Confidence intervals including zero do not establish equivalence or population parity",
                  "Repeated closed-loop timing ranges do not estimate distributed/open-loop queueing tails",
                  "A different nearest-neighbor ID with a tied distance remains a miss in the primary strict-ID metric",
                  "Books original-range and atomic-bank workloads share query embeddings and must not be pooled as independent quality samples"],
              "scopes": scopes}
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.json").open("x") as target:
        json.dump(result, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")
    with (output / "per-label.csv").open("x", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["scope", "engine", "label", "category", "queries", "families", "eligible", "requested_l", "strict_recall", "ci95_low", "ci95_high", "cutoff_recall_diagnostic", "cutoff_additional_hits", "p99_us_median", "p99_us_min", "p99_us_max", "delta_cpp_pp", "paired_ci95_low_pp", "paired_ci95_high_pp", "limitations"])
        for scope in scopes:
            for label in scope["labels"]:
                for engine, method in label["engines"].items():
                    ci = method["strict_id_recall_ci95"] or [None, None]
                    comparison = label["comparisons_to_cpp"].get(engine, {})
                    pair_ci = comparison.get("paired_family_ci95_pp") or [None, None]
                    writer.writerow([scope["scope"], engine, label["label"], label["category"], label["queries"], label["families"], label["eligible_vectors"], method["requested_l"], method["strict_id_recall"], *ci, method["distance_cutoff_recall_diagnostic"], method["cutoff_additional_hits"], method["p99_us_median"], *method["p99_us_range"], comparison.get("strict_id_recall_difference_pp"), *pair_ci, ";".join(label["limitations"])])
    print(json.dumps({"summary": str(output / "summary.json"), "queries": n, "families": len(set(families)), "scopes": len(scopes)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--relocate-prefix", action="append", default=[], metavar="OLD=NEW",
                        help="Resolve immutable artifact paths after copying data; original bytes and SHA-256 must still match")
    args = parser.parse_args()
    require(args.bootstrap_samples > 0, "Bootstrap sample count must be positive")
    configure_relocations(args.relocate_prefix)
    analyze(args)


if __name__ == "__main__":
    main()
