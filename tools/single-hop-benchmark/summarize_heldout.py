#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Audit a frozen per-label L policy on disjoint natural public queries.

This tool never selects or changes breadths and does not fail on recall
regressions. It checks retained per-query results against published strict-ID
truth, reports paired query bootstrap uncertainty, and summarizes independently
timed repetitions. Per-label latency is service latency in the mixed workload;
no isolated per-label QPS or production arrival-rate SLO is inferred.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

import numpy as np

from plan_label_l import archived_files, load_plan, planning
from prepare_yfcc import EXPECTED_SHA256
from summarize_delta import digest, finite, integer, one, read_rows, require


METRICS = ("qps", "mean_us", "p50_us", "p95_us", "p99_us")
FROZEN_SHA = "df42e65440af0a891185913e91f5f22cb6808c9251b2c1d8f6ca968d5bda2e59"


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def asset(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def matrix(path, dtype, truth=False):
    raw = path.read_bytes()
    require(len(raw) >= 8, f"Truncated matrix header: {path}")
    rows, cols = struct.unpack_from("<II", raw)
    size = rows * cols * np.dtype(dtype).itemsize
    require(rows > 0 and cols > 0 and len(raw) == 8 + size * (2 if truth else 1),
            f"Matrix dimensions/length differ: {path}")
    values = np.frombuffer(raw, dtype=dtype, count=rows * cols, offset=8).reshape(rows, cols)
    distances = (np.frombuffer(raw, dtype="<f4", count=rows * cols, offset=8 + size).reshape(rows, cols)
                 if truth else None)
    return values, distances


def prepared(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    require(manifest.get("base_vectors") == 10_000_000 and manifest.get("dimension") == 192
            and manifest.get("k") == 10 and manifest.get("dtype") == "uint8"
            and manifest.get("metric") == "L2", "Unexpected prepared corpus/metric")
    names = ("query.u8bin", "query_labels.u32bin", "query_ids.u32bin", "groundtruth.bin", "labels.txt", "query_labels.txt")
    for name in names:
        expected = manifest.get("artifacts", {}).get(name, {})
        require(digest(expected.get("sha256")) and expected["sha256"] == sha(directory / name)
                and expected.get("bytes") == (directory / name).stat().st_size,
                f"Prepared asset hash/size differs: {name}")
    queries, _ = matrix(directory / "query.u8bin", "u1")
    labels, _ = matrix(directory / "query_labels.u32bin", "<u4")
    ids, _ = matrix(directory / "query_ids.u32bin", "<u4")
    truth, distances = matrix(directory / "groundtruth.bin", "<u4", truth=True)
    count = len(queries)
    require(queries.shape == (count, 192) and labels.shape == (count, 1) and ids.shape == (count, 1)
            and truth.shape == (count, 10) and manifest.get("query_count") == count,
            "Prepared vector/ID/label/truth sample shapes differ")
    require(len(set(map(int, ids[:, 0]))) == count and np.all(truth < 10_000_000)
            and np.all(np.isfinite(distances)) and np.all(distances >= 0), "Invalid source query IDs/truth")
    require([int(value) for value in (directory / "query_labels.txt").read_text().splitlines()]
            == labels[:, 0].tolist(), "C++ text query filters differ from Rust binary query filters")
    return {"manifest": manifest, "queries": queries, "labels": labels[:, 0], "ids": ids[:, 0],
            "truth": truth, "distances": distances,
            "identity": ":".join(manifest["artifacts"][name]["sha256"] for name in
                                  ("query.u8bin", "query_labels.u32bin", "groundtruth.bin", "labels.txt"))}


def sparse_labels(path):
    raw = path.read_bytes()
    require(len(raw) >= 24, "Truncated public CSR metadata")
    rows, columns, entries = struct.unpack_from("<qqq", raw)
    require(rows > 0 and columns > 0 and entries > 0
            and len(raw) == 24 + 8 * (rows + 1) + 8 * entries, "Public CSR metadata dimensions differ")
    pointers = np.frombuffer(raw, dtype="<i8", count=rows + 1, offset=24)
    indices = np.frombuffer(raw, dtype="<i4", count=entries, offset=24 + 8 * (rows + 1))
    require(pointers[0] == 0 and pointers[-1] == entries and np.all(np.diff(pointers) >= 0)
            and np.all(indices >= 0) and np.all(indices < columns), "Invalid public CSR pointers/labels")
    return pointers, indices


def holdout_proof(tuning, heldout, plan, public_queries, public_metadata, public_truth):
    require(len(tuning["queries"]) == 2000 and len(heldout["queries"]) > len(tuning["queries"]),
            "Holdout must be larger than the original 2,000 tuning queries")
    tuning_ids, heldout_ids = set(map(int, tuning["ids"])), set(map(int, heldout["ids"]))
    require(not tuning_ids & heldout_ids, "Tuning and holdout source query IDs overlap")
    tuning_vectors = {row.tobytes() for row in tuning["queries"]}
    heldout_vectors = [row.tobytes() for row in heldout["queries"]]
    require(not tuning_vectors & set(heldout_vectors), "Tuning and holdout query embeddings overlap")
    require(len(set(heldout_vectors)) == len(heldout_vectors), "Holdout contains duplicate query embeddings")
    public, _ = matrix(public_queries, "u1")
    truth, distances = matrix(public_truth, "<u4", truth=True)
    ptr, labels = sparse_labels(public_metadata)
    require(public.shape == (len(ptr) - 1, 192) and truth.shape == (len(public), 10),
            "Public query/truth/metadata shapes differ")
    single_rows = np.flatnonzero(np.diff(ptr) == 1)
    single_labels = labels[ptr[single_rows]]
    candidates = [(int(row), int(label)) for row, label in zip(single_rows, single_labels)
                  if int(label) in plan and int(row) not in tuning_ids
                  and public[row].tobytes() not in tuning_vectors]
    require({row for row, _ in candidates} == heldout_ids,
            "Holdout does not contain exactly all remaining eligible public one-tag queries")
    for workload in (tuning, heldout):
        ids = workload["ids"]
        require(np.all(ids < len(public)) and np.all(np.diff(ptr)[ids] == 1)
                and np.array_equal(public[ids], workload["queries"])
                and np.array_equal(labels[ptr[ids]], workload["labels"])
                and np.array_equal(truth[ids], workload["truth"])
                and np.array_equal(distances[ids], workload["distances"]),
                "Prepared vectors, filters or exact truth differ from original source query rows")
    require(set(map(int, tuning["labels"])) == set(plan)
            and set(map(int, heldout["labels"])) <= set(plan), "Frozen plan/query label coverage differs")
    require(tuning["manifest"]["artifacts"]["labels.txt"]["sha256"]
            == heldout["manifest"]["artifacts"]["labels.txt"]["sha256"], "Base label corpus changed")
    counts = Counter(map(int, heldout["labels"]))
    return {"unique_tuning_queries": len(tuning_ids), "unique_heldout_queries": len(heldout_ids),
            "source_query_id_overlap": 0, "query_embedding_overlap_across_splits": 0,
            "duplicate_heldout_embeddings": 0, "all_remaining_eligible_public_queries_included": True,
            "frozen_plan_labels": len(plan), "tested_labels": len(counts),
            "untested_labels": sorted(set(plan) - counts.keys()),
            "tested_raised_l_labels": sorted(label for label in counts if plan[label] > 64),
            "untested_raised_l_labels": sorted(label for label in plan if label not in counts and plan[label] > 64),
            "largest_tested_l": max(plan[label] for label in counts), "largest_frozen_plan_l": max(plan.values()),
            "labels_with_at_least_100_queries": sum(count >= 100 for count in counts.values()),
            "query_weighted_mean_l": sum(count * plan[label] for label, count in counts.items()) / len(heldout_ids),
            "source_assets": {"queries": asset(public_queries), "metadata": asset(public_metadata),
                              "exact_truth": asset(public_truth)}}


def percentile(values, p):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * p) - 1)]


def close(actual, expected, message):
    require(finite(actual) and math.isclose(actual, expected, rel_tol=2e-8, abs_tol=2e-7), message)


def valid_result_ids(ids, workload, postings):
    require(ids.shape == workload["truth"].shape and np.all(ids < 10_000_000), "Returned ID shape/range differs")
    require(np.all(np.diff(np.sort(ids, axis=1), axis=1) != 0), "Duplicate result IDs")
    for label in sorted(set(map(int, workload["labels"]))):
        selected = ids[workload["labels"] == label]
        members = postings[label]
        positions = np.searchsorted(members, selected)
        require(np.all(positions < len(members)) and np.array_equal(members[positions], selected),
                f"Returned ID violates label {label}")
    return np.asarray([len(set(map(int, row)) & set(map(int, exact))) for row, exact in
                       zip(ids, workload["truth"])], dtype=np.int64)


def checked_query(row, workload, plan, native, policy=True, authoritative=True):
    count = len(workload["queries"])
    require(row.get("queries") == count and row.get("warmup_queries") == count and row.get("k") == 10
            and row.get("threads") == 8 and row.get("l") == 64 and integer(row.get("repeat"))
            and row.get("timing_authoritative") is authoritative and row.get("strict_id_recall") is True
            and row.get("prepared_identity") == workload["identity"], "Query workload/protocol differs")
    require(all(type(row.get(key)) is int and row[key] == 0 for key in
                ("filter_violations", "duplicate_results", "short_results")), "Invalid filtered result counters")
    require(all(finite(row.get(metric), strict=True) for metric in METRICS), "Invalid aggregate timing")
    if native:
        require(row.get("implementation") == "rust_native_lifecycle" and row.get("phase") == "query"
                and row.get("latency_contract") == "public_snapshot_search_call"
                and row.get("sequence") == 0 and row.get("delta_candidates") == 0
                and row.get("base_searches") == count and row.get("exact_fallback_queries") == 0,
                "Native snapshot/delta/fallback protocol differs")
        if policy:
            meta = row.get("label_l_plan", {})
            require(meta.get("content_sha256") == FROZEN_SHA
                    and meta.get("entries") == [{"label": label, "l": plan[label]} for label in sorted(plan)],
                    "Frozen policy hash/entries differ")
            unused = sorted(set(plan) - set(map(int, workload["labels"])))
            require(row.get("label_l_tuning_scope") == "frozen_policy_heldout"
                    and row.get("unused_plan_labels") == unused and meta.get("unused_plan_labels") == unused,
                    "Heldout policy scope/untested-label disclosure differs")
        else:
            require(row.get("label_l_plan") is None, "Fixed-L control must not use a breadth policy")
        require(all((plan.get(item.get("label")) if policy else 64) == item.get("actual_l")
                    for item in row.get("labels", [])), "Actual per-label breadth differs from frozen policy")
        require(finite(row.get("wall_seconds"), strict=True), "Missing native wall clock")
        close(row["qps"], count / row["wall_seconds"], "Native QPS/wall clock differs")
    else:
        require(row.get("implementation") == "cpp_native_linux" and row.get("mode") == "single"
                and row.get("starts") == "label" and row.get("n_base") == 10_000_000
                and row.get("beam") == 4 and row.get("cache_bypass") is True
                and row.get("shared_cache_nodes") == 0 and row.get("storage") == "linux_aio_direct"
                and row.get("latency_contract") == "public_search_call", "C++ query/I/O protocol differs")
        require(row.get("label_l_plan") is None, "C++ reference must use fixed L64")


def validate_observations(row, ids, latencies, workload, postings, native):
    require(len(latencies) == len(workload["queries"]) and all(finite(x, strict=True) for x in latencies),
            "Invalid or incomplete raw query latencies")
    hits = valid_result_ids(ids, workload, postings)
    close(row.get("recall"), float(hits.sum()) / (len(hits) * 10), "Raw and aggregate strict-ID recall differ")
    close(row.get("mean_us"), sum(latencies) / len(latencies), "Raw and aggregate mean latency differ")
    for name, p in (("p50_us", .5), ("p95_us", .95), ("p99_us", .99)):
        close(row.get(name), percentile(latencies, p), f"Raw and aggregate {name} differ")
    by_label = {item["label"]: item for item in row.get("labels", [])}
    require(len(by_label) == len(row.get("labels", [])) and set(by_label) == set(map(int, workload["labels"])),
            "Per-label observations incomplete/repeated")
    summary = {}
    for label, item in sorted(by_label.items()):
        mask = workload["labels"] == label
        count = int(mask.sum())
        times = [time for time, selected in zip(latencies, mask) if selected]
        require(item.get("queries" if native else "n") == count, "Per-label sample count differs")
        if not native:
            require(item.get("eligible") == len(postings[label]), "C++ eligible label count differs")
        close(item.get("recall"), float(hits[mask].sum()) / (count * 10), "Raw and per-label recall differ")
        close(item.get("mean_us"), sum(times) / count, "Raw and per-label mean latency differ")
        for name, p in (("p95_us", .95), ("p99_us", .99)):
            close(item.get(name), percentile(times, p), f"Raw and per-label {name} differ")
        summary[label] = {"mean_us": sum(times) / count, "p99_us": percentile(times, .99)}
    return {"repeat": row["repeat"], "hits": hits, "ids": ids, "latencies": np.asarray(latencies), "labels": summary,
            **{metric: row[metric] for metric in METRICS}}


def local_path(path, root, remote_root):
    path = Path(path)
    if path.is_file():
        return path
    if path.is_absolute():
        try:
            path = path.relative_to(remote_root)
        except ValueError as error:
            raise ValueError(f"Remote artifact is outside retained output root: {path}") from error
    require(".." not in path.parts, "Retained artifact path traverses output root")
    result = root / path
    require(result.is_file(), f"Missing retained output artifact: {result}")
    return result


def native_observations(path, workload, plan, postings, root, remote_root, policy=True):
    rows = [json.loads(path.read_text())] if path.suffix == ".json" else read_rows(path)
    aggregate = [row for row in rows if row.get("kind") == "query"]
    experiments = [row for row in rows if row.get("kind") == "experiment"]
    if not experiments:
        original = local_path(aggregate[0]["original_aggregate"], root, remote_root)
        experiments = [row for row in read_rows(original) if row.get("kind") == "experiment"]
    experiment = one(experiments, "native experiment")
    require(experiment.get("action") == "query" and experiment.get("n_base") == 10_000_000
            and experiment.get("dimensions") == 192 and experiment.get("index_threads") == 8
            and experiment.get("beam") == 4 and experiment.get("shared_cache_nodes") == 0
            and experiment.get("cache_bypass") is True and experiment.get("storage") == "native_io_uring"
            and experiment.get("warmup") == "full_same_process_same_worker_count"
            and experiment.get("query_scheduling") == "dynamic_one", "Native experiment protocol differs")
    require(len(rows) in (len(aggregate), len(aggregate) + 1), "Unexpected native query log rows")
    result, inputs = [], []
    for row in aggregate:
        checked_query(row, workload, plan, True, policy=policy, authoritative=policy)
        meta = row.get("raw_queries", {})
        raw_path = local_path(row.get("raw_queries_relative_path", row.get("local_raw_queries", meta.get("path", ""))),
                              root, remote_root)
        require(meta.get("queries") == row["queries"] and meta.get("sha256") == sha(raw_path),
                "Native retained query details hash/count differs")
        selected = sorted(read_rows(raw_path), key=lambda item: item.get("query_index", -1))
        require([item.get("query_index") for item in selected] == list(range(row["queries"])),
                "Native raw query indices incomplete/repeated")
        for index, item in enumerate(selected):
            require(item.get("label") == int(workload["labels"][index]) and item.get("l") == 64
                    and item.get("actual_l") == (plan[item["label"]] if policy else 64) and item.get("k") == 10
                    and item.get("repeat") == row.get("local_repeat", row["repeat"])
                    and item.get("phase") == "query" and item.get("sequence") == 0
                    and item.get("exact_fallback") is False and item.get("base_searches") == 1
                    and item.get("delta_candidates") == 0, "Native raw query policy/snapshot differs")
        ids = np.asarray([item.get("ids") for item in selected], dtype=np.int64)
        observed = validate_observations(row, ids, [item.get("latency_us") for item in selected],
                                         workload, postings, True)
        require(all(item.get("strict_id_hits") == int(hits) for item, hits in zip(selected, observed["hits"])),
                "Native declared per-query hit counts differ from retained IDs")
        result.append(observed)
        inputs.append({"repeat": row["repeat"], "raw_queries": asset(raw_path)})
    return experiment, sorted(result, key=lambda row: row["repeat"]), inputs


def local_asset(path, directory):
    path = Path(path)
    candidate = path if path.is_file() else directory / path.name if directory is not None else path
    require(candidate.is_file(), f"Missing retained local C++ asset: {candidate}")
    return candidate


def cpp_observations(path, directory, workload, plan, postings, root, remote_root):
    rows = read_rows(path)
    result, inputs = [], []
    for row in rows:
        checked_query(row, workload, plan, False)
        raw_path = (local_path(row["raw_stats_relative_path"], root, remote_root)
                    if "raw_stats_relative_path" in row else
                    local_asset(row.get("local_raw_stats", row.get("raw_stats", "")), directory))
        ids_path = (local_path(row["result_ids_relative_path"], root, remote_root)
                    if "result_ids_relative_path" in row else
                    local_asset(row.get("local_result_ids", row.get("result_ids", "")), directory))
        require(sha(ids_path) == row.get("result_ids_sha256"), "C++ retained ID file hash differs")
        raw = read_rows(raw_path)
        summary = one([item for item in raw if item.get("kind") == "summary"], "C++ raw summary")
        close(summary.get("qps"), row["qps"], "C++ raw/normalized throughput differs")
        require(summary.get("query_count") == row["queries"] and summary.get("warmup_queries") == row["queries"]
                and summary.get("threads") == 8 and summary.get("k") == 10 and summary.get("l") == 64
                and summary.get("beam_width") == 4 and summary.get("latency_contract") == "public_search_call",
                "C++ raw summary protocol differs")
        selected = sorted([item for item in raw if item.get("kind") == "query"],
                          key=lambda item: item.get("query_index", -1))
        require(len(raw) == len(selected) + 1
                and [item.get("query_index") for item in selected] == list(range(row["queries"])),
                "C++ raw query indices incomplete/repeated")
        require(all(item.get("returned_results") == 10 and item.get("threads") == 8
                    and item.get("k") == 10 and item.get("l") == 64 and item.get("beam_width") == 4
                    for item in selected), "C++ raw query/result-count protocol differs")
        ids, _ = matrix(ids_path, "<u4")
        result.append(validate_observations(row, ids, [item.get("latency_us") for item in selected],
                                            workload, postings, False))
        inputs.append({"repeat": row["repeat"], "raw_queries": asset(raw_path), "result_ids": asset(ids_path)})
    return sorted(result, key=lambda row: row["repeat"]), inputs


def repeat_summary(rows):
    repeats = [row["repeat"] for row in rows]
    require(len(rows) >= 3 and repeats == list(range(len(rows))), "At least three consecutive unique repeats required")
    require(all(np.array_equal(row["hits"], rows[0]["hits"]) for row in rows),
            "Per-query strict-ID hit counts differ across timing repetitions")
    return {"repetitions": len(rows), "strict_id_recall": float(rows[0]["hits"].mean()) / 10,
            **{metric: statistics.median(row[metric] for row in rows) for metric in METRICS},
            "ranges": {metric: [min(row[metric] for row in rows), max(row[metric] for row in rows)]
                       for metric in METRICS},
            "repeat_metrics": [{"repeat": row["repeat"], **{key: row[key] for key in METRICS}} for row in rows]}


def paired_bootstrap(differences, k, repetitions, seed, families=1):
    """Resample paired queries using their integer hit-difference histogram."""
    values, counts = np.unique(differences, return_counts=True)
    rng = np.random.default_rng(seed)
    samples = rng.multinomial(len(differences), counts / counts.sum(), size=repetitions) @ values
    samples = samples / (len(differences) * k) * 100
    alpha = .025 / families
    return {"difference_percentage_points": float(np.mean(differences)) / k * 100,
            "paired_query_bootstrap_95_percent_interval_pp": np.quantile(samples, [.025, .975]).tolist(),
            "paired_query_bootstrap_simultaneous_95_percent_interval_pp":
                np.quantile(samples, [alpha, 1 - alpha]).tolist(),
            "bootstrap_empirical_difference_variance_is_zero": len(values) == 1,
            "rust_wins_queries": int(np.sum(differences > 0)), "cpp_wins_queries": int(np.sum(differences < 0)),
            "equal_hits_queries": int(np.sum(differences == 0))}


def integer_distances(base, queries, ids):
    distances = np.empty(ids.shape, dtype=np.int64)
    for start in range(0, len(ids), 512):
        stop = min(start + 512, len(ids))
        vectors = base[ids[start:stop]].astype(np.int32)
        vectors -= queries[start:stop, None, :].astype(np.int32)
        vectors *= vectors
        distances[start:stop] = vectors.sum(axis=2, dtype=np.int64)
    return distances


def cutoff_diagnostic(path, workload, observations):
    """Credit distinct alternative IDs tied at the exact Kth distance."""
    if path is None:
        return None
    base_sha = sha(path)
    with path.open("rb") as stream:
        shape = struct.unpack("<II", stream.read(8))
    require(shape == (10_000_000, 192) and path.stat().st_size == 8 + shape[0] * shape[1]
            and base_sha == EXPECTED_SHA256["base.10M.u8bin"], "Base artifact differs from pinned 10M vectors")
    base = np.memmap(path, mode="r", dtype="u1", offset=8, shape=shape)
    truth_distances = integer_distances(base, workload["queries"], workload["truth"])
    require(np.array_equal(truth_distances, workload["distances"])
            and np.all(np.diff(truth_distances, axis=1) >= 0),
            "Published truth distances differ from exact integer distances or are not sorted")
    cutoff = truth_distances[:, -1, None]
    result = {"base": {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": base_sha},
              "metric": "distinct returned IDs with exact integer squared L2 <= exact Kth truth distance, divided by K",
              "strict_id_recall_remains_primary": True, "policy_unchanged": True,
              "all_saved_truth_distances_recomputed_exactly": True, "engines": {}}
    for name, observation in observations.items():
        distances = integer_distances(base, workload["queries"], observation["ids"])
        truth_member = (observation["ids"][:, :, None] == workload["truth"][:, None, :]).any(axis=2)
        require(not np.any((distances < cutoff) & ~truth_member),
                "A non-truth returned ID is strictly closer than the published Kth exact truth distance")
        hits = (distances <= cutoff).sum(axis=1, dtype=np.int64)
        require(np.all(hits >= observation["hits"]), "Exact distance qualification loses a strict-ID truth hit")
        result["engines"][name] = {"hits": hits,
            "strict_id_recall": float(observation["hits"].mean()) / 10,
            "distance_cutoff_recall": float(hits.mean()) / 10,
            "alternative_equal_cutoff_tie_hits": int((hits - observation["hits"]).sum())}
    return result


def identity_proof(path, tuning_identity_path, native_path, cpp_path, native_exp, plan_path, archive_override=None):
    evidence = json.loads(path.read_text())
    tuning = json.loads(tuning_identity_path.read_text())
    require(evidence.get("schema_version") == 1 and evidence.get("cpu_affinity") == list(range(8))
            and evidence.get("host") == tuning.get("shared_index", {}).get("host"),
            "Heldout host/affinity identity differs")
    require(evidence.get("plan_before_sha256") == FROZEN_SHA == evidence.get("plan_after_sha256") == sha(plan_path),
            "Frozen policy changed before/during heldout evaluation")
    native, cpp = evidence.get("native", {}), evidence.get("cpp", {})
    for name, item, log in (("native", native, native_path), ("cpp", cpp, cpp_path)):
        require(digest(item.get("binary_sha256")) and item.get("log_sha256") == sha(log),
                f"{name} binary/log identity is missing or differs")
        require(item.get("binary_before_sha256") == item["binary_sha256"],
                f"{name} binary changed during evaluation")
        before, after = item.get("index_before", {}), item.get("index_after", {})
        require(before and before == after and all(digest(value) for value in before.values()),
                f"{name} index identity changed during evaluation")
    require(native["index_before"] == tuning["mixed"]["index_identity"], "Heldout native graph differs from tuned graph")
    require(native_exp.get("root") == tuning.get("shared_index", {}).get("root")
            or str(tuning["shared_index"]["root"]).endswith("/" + native_exp.get("root", "")),
            "Heldout native index root differs from tuned root")
    archive = native.get("source_archive", {})
    archive_path = archive_override or Path(archive.get("path", ""))
    require(digest(archive.get("sha256")) and archive.get("before_sha256") == archive["sha256"]
            and sha(archive_path) == archive["sha256"],
            "Heldout source archive hash differs")
    old_archive = tuning["mixed"]["source_archive"]
    require(sha(Path(old_archive["path"])) == old_archive["sha256"], "Tuning source archive hash differs")
    old, new = archived_files(Path(old_archive["path"])), archived_files(archive_path)
    require(set(old) == set(new), "Heldout/tuning archived source sets differ")
    changed = [name for name in old if old[name] != new[name]]
    require(set(changed) <= {"diskann-disk/examples/filtered_lifecycle.rs"},
            "Search/build/lifecycle algorithm changed after breadth tuning")
    return {"source": asset(path), "tuning_identity": asset(tuning_identity_path),
            "declarations": evidence, "heldout_local_source_archive": asset(archive_path),
            "changed_archived_source_files": sorted(changed),
            "verification_scope": "Local logs/archives are hashed. Binary, immutable native generation and C++ artifact identities are separately collected host declarations, bound consistently before and after timing."}


def jobs_proof(path, native_path, cpp_path, artifact_root=None):
    """Bind global repeat renumbering to byte-exact independently timed jobs."""
    manifest = json.loads(path.read_text())
    jobs = manifest.get("runs", manifest.get("jobs", []))
    require(manifest.get("schema_version") == 1 and len(jobs) >= 6 and len(jobs) % 2 == 0,
            "Missing complete paired-job manifest")
    require(manifest.get("status", "complete") == "complete", "Paired-job run did not complete")
    merged = {"rust": [row for row in read_rows(native_path) if row.get("kind") == "query"],
              "cpp": read_rows(cpp_path)}
    allowed = {"repeat", "paired_repeat", "local_repeat", "local_raw_queries", "local_raw_stats", "local_result_ids",
               "raw_queries_relative_path", "raw_stats_relative_path", "result_ids_relative_path",
               "original_aggregate", "original_aggregate_sha256"}
    seen = set()
    native_experiments = []
    remote_root = Path(manifest.get("output_root", path.parent))
    artifact_root = artifact_root or path.parent
    for index, job in enumerate(jobs):
        name, repeat = job.get("implementation"), job.get("paired_repeat")
        require(name in merged and integer(repeat) and job.get("local_repeat") == 0 and job.get("exit_code") == 0,
                "Invalid paired-job identity/status")
        key = name, repeat
        require(key not in seen, "Repeated paired-job identity")
        seen.add(key)
        original = job.get("aggregate", {})
        source = local_path(job.get("aggregate_path", original.get("path", "")), artifact_root, remote_root)
        require(job.get("aggregate_sha256", original.get("sha256")) == sha(source),
                "Original single-job aggregate hash differs")
        rows = read_rows(source)
        if name == "rust":
            native_experiments.extend(item for item in rows if item.get("kind") == "experiment")
        raw = one([row for row in rows if row.get("kind") == "query"] if name == "rust" else rows,
                  "original single-job query aggregate")
        row = one([row for row in merged[name] if row.get("repeat") == repeat], "merged paired-repeat observation")
        require(raw.get("repeat") == 0 and row.get("local_repeat") == 0 and row.get("paired_repeat") == repeat
                and {key: value for key, value in row.items() if key not in allowed}
                == {key: value for key, value in raw.items() if key not in allowed},
                "Merged observation differs from byte-exact original beyond repeat/local-path annotations")
        start, end = job.get("start", job.get("start_utc")), job.get("end", job.get("end_utc"))
        require(type(start) is str and type(end) is str and start < end, "Invalid single-job time interval")
        if index:
            previous_end = jobs[index - 1].get("end", jobs[index - 1].get("end_utc"))
            require(previous_end <= start, "Timing jobs overlap")
    for index in range(0, len(jobs), 2):
        require({job["implementation"] for job in jobs[index:index + 2]} == {"cpp", "rust"}
                and {job["paired_repeat"] for job in jobs[index:index + 2]} == {index // 2},
                "Each adjacent timing pair must contain one C++ and one Rust job")
    require(seen == {(name, repeat) for name in merged for repeat in range(len(jobs) // 2)},
            "Paired job repeat matrix incomplete")
    require(not native_experiments or (len(native_experiments) == len(jobs) // 2
            and all(item == native_experiments[0] for item in native_experiments)),
            "Native query experiments differ across paired jobs")
    return {"source": asset(path), "pairs": len(jobs) // 2, "jobs": jobs,
            "merged_observations_match_original_aggregates": True, "declared_job_intervals_do_not_overlap": True}


def control_observations(path, manifest, workload, plan, postings, root, remote_root, native_exp):
    record = manifest.get("fixed_control")
    if record is None:
        require(path is None, "Control supplied without a declared diagnostic job")
        return None
    path = path or root / "fixed-control.normalized.json"
    aggregate = local_path(record["aggregate_path"], root, remote_root)
    require(record.get("exit_code") == 0 and record.get("implementation") == "rust"
            and record.get("aggregate_sha256") == sha(aggregate), "Invalid fixed-L diagnostic identity")
    row = json.loads(path.read_text())
    original = one([row for row in read_rows(aggregate) if row.get("kind") == "query"],
                   "fixed-L diagnostic query row")
    added = {"repeat", "paired_repeat", "local_repeat", "original_aggregate", "original_aggregate_sha256",
             "local_raw_queries", "raw_queries_relative_path"}
    require(row.get("local_repeat") == 0 and row.get("original_aggregate_sha256") == sha(aggregate)
            and {key: value for key, value in row.items() if key not in added}
            == {key: value for key, value in original.items() if key not in added},
            "Fixed-L diagnostic normalized row differs from original")
    exp, rows, inputs = native_observations(path, workload, plan, postings, root, remote_root, policy=False)
    require(exp == native_exp and len(rows) == 1, "Fixed-L diagnostic graph/workload differs")
    return {"row": rows[0], "source": asset(path), "retained_inputs": inputs,
            "timing_excluded": True, "policy_retuned": False}


def summarize(args):
    require(sha(args.plan) == FROZEN_SHA, "Policy bytes differ from the previously frozen plan")
    plan = load_plan(args.plan)
    evidence = json.loads(args.plan_evidence.read_text())
    data, recomputed = planning(Path(evidence["inputs"]["grid"]["path"]),
                                Path(evidence["inputs"]["cpp"]["path"]), args.plan.resolve())
    require(data == args.plan.read_bytes() and recomputed == evidence,
            "Frozen plan/evidence differs from independently recomputed original tuning selection")
    tuning, heldout = prepared(args.tuning_prepared), prepared(args.prepared)
    require(tuning["identity"] == evidence["prepared_identity"], "Tuning prepared assets differ from plan selection")
    for path, name in ((args.public_queries, "query.public.100K.u8bin"),
                       (args.public_metadata, "query.metadata.public.100K.spmat"),
                       (args.public_truth, "GT.public.ibin")):
        require(sha(path) == EXPECTED_SHA256[name], "Public source asset differs from the previously pinned dataset")
    coverage = holdout_proof(tuning, heldout, plan, args.public_queries, args.public_metadata, args.public_truth)
    postings = {}
    for label in sorted(set(map(int, heldout["labels"]))):
        values, _ = matrix(args.prepared / f"label_{label}.u32bin", "<u4")
        require(values.shape[1] == 1 and np.all(np.diff(values[:, 0].astype(np.int64)) > 0), "Invalid posting list")
        postings[label] = values[:, 0]
    valid_result_ids(heldout["truth"], heldout, postings)
    manifest = json.loads(args.jobs_manifest.read_text())
    artifact_root = args.artifact_root or args.jobs_manifest.parent
    remote_root = Path(manifest["output_root"])
    native_exp, native_rows, native_inputs = native_observations(args.native, heldout, plan, postings,
                                                               artifact_root, remote_root)
    cpp_rows, cpp_inputs = cpp_observations(args.cpp, args.cpp_raw_dir, heldout, plan, postings,
                                          artifact_root, remote_root)
    native, cpp = repeat_summary(native_rows), repeat_summary(cpp_rows)
    require(len(native_rows) == len(cpp_rows), "C++/Rust repetition counts differ")
    identity = identity_proof(args.identity, args.tuning_identity, args.native, args.cpp, native_exp,
                              args.plan, args.source_archive)
    require(identity["declarations"].get("output_root") == str(remote_root)
            and manifest.get("identity") == identity["declarations"]
            and manifest.get("prepared_manifest_sha256") == sha(args.prepared / "manifest.json")
            and identity["declarations"].get("prepared_manifest_sha256") == sha(args.prepared / "manifest.json"),
            "Paired-run manifest, host identity or prepared-asset binding differs")
    runner_sources = {"runner_sha256": Path(__file__).with_name("run_heldout.py"),
                      "cpp_runner_sha256": Path(__file__).parent / "cpp-port" / "run_reference.py"}
    require(all(identity["declarations"].get(key) == sha(path) for key, path in runner_sources.items()),
            "Retained local runner/normalizer sources differ from host declarations")
    identity["local_runner_sources"] = {key: asset(path) for key, path in runner_sources.items()}
    jobs = jobs_proof(args.jobs_manifest, args.native, args.cpp, artifact_root)
    control = control_observations(args.fixed_control, manifest, heldout, plan, postings,
                                   artifact_root, remote_root, native_exp)
    diagnostic_observations = {"rust_policy": native_rows[0], "cpp_L64": cpp_rows[0]}
    if control is not None:
        diagnostic_observations["rust_fixed64"] = control["row"]
    cutoff = cutoff_diagnostic(args.base, heldout, diagnostic_observations)
    original_cpp_path = Path(evidence["inputs"]["cpp"]["path"])
    original_cpp = read_rows(original_cpp_path)
    require(all(row.get("index_prefix") == original_cpp[0].get("index_prefix") for row in read_rows(args.cpp)),
            "Heldout C++ index prefix differs from original frozen reference")
    original_manifest_path = original_cpp_path.with_name(original_cpp_path.name.replace(".normalized.jsonl", ".query.manifest.json"))
    original_manifest = json.loads(original_manifest_path.read_text())
    require(identity["declarations"]["cpp"]["binary_sha256"] == original_manifest["binaries"]["search_disk_index"],
            "Heldout C++ binary differs from original frozen reference")
    original_index = json.loads(args.cpp_index_manifest.read_text())
    prefix = original_index.get("index_prefix", "")
    require(prefix and Path(original_cpp[0]["index_prefix"]).name == prefix, "Original C++ index names differ")
    expected_cpp_assets = {name.removeprefix(prefix): item["sha256"]
                           for name, item in original_index.get("artifacts", {}).items()}
    require(len(expected_cpp_assets) == 6
            and identity["declarations"]["cpp"]["index_before"] == expected_cpp_assets,
            "Heldout C++ graph/PQ/labels differ from the original immutable reference artifacts")
    identity["original_cpp_index_manifest"] = asset(args.cpp_index_manifest)
    identity["cpp_artifacts_match_original_reference"] = True
    differences = native_rows[0]["hits"] - cpp_rows[0]["hits"]
    labels = []
    families = len(postings)
    for label in sorted(postings):
        mask = heldout["labels"] == label
        count = int(mask.sum())
        latency = {name: {**{key: statistics.median(row["labels"][label][key] for row in rows)
                          for key in ("mean_us", "p99_us")},
                         "ranges": {key: [min(row["labels"][label][key] for row in rows),
                                          max(row["labels"][label][key] for row in rows)]
                                    for key in ("mean_us", "p99_us")}}
                   for name, rows in (("rust", native_rows), ("cpp", cpp_rows))}
        labels.append({"label": label, "queries": count, "k": 10, "rust_l": plan[label], "cpp_l": 64,
                       "eligible": len(postings[label]),
                       "rust_strict_id_recall": float(native_rows[0]["hits"][mask].mean()) / 10,
                       "cpp_strict_id_recall": float(cpp_rows[0]["hits"][mask].mean()) / 10,
                       **paired_bootstrap(differences[mask], 10, args.bootstrap_repetitions,
                                          args.bootstrap_seed + label, families),
                       "latency_median_over_repeats": latency,
                       "latency_ratio_rust_over_cpp": {key: latency["rust"][key] / latency["cpp"][key]
                                                       for key in ("mean_us", "p99_us")},
                       "p99_tail_query_order_statistics": count - math.ceil(count * .99) + 1,
                       "p99_fewer_than_100_queries": count < 100})
        if control is not None:
            fixed_hits = control["row"]["hits"][mask]
            labels[-1]["fixed64_quality_only_control"] = {
                "strict_id_recall": float(fixed_hits.mean()) / 10,
                "policy_minus_fixed64": paired_bootstrap(native_rows[0]["hits"][mask] - fixed_hits, 10,
                    args.bootstrap_repetitions, args.bootstrap_seed + label + 1_000_000, families),
                "fixed64_minus_cpp_difference_percentage_points":
                    float((fixed_hits - cpp_rows[0]["hits"][mask]).mean()) / 10 * 100}
        if cutoff is not None:
            values = cutoff["engines"]
            labels[-1]["distance_cutoff_diagnostic"] = {name: {
                "recall": float(item["hits"][mask].mean()) / 10,
                "alternative_equal_cutoff_tie_hits": int((item["hits"][mask]
                    - diagnostic_observations[name]["hits"][mask]).sum())}
                for name, item in values.items()}
            labels[-1]["distance_cutoff_diagnostic"]["policy_minus_cpp_difference_percentage_points"] = float(
                (values["rust_policy"]["hits"][mask] - values["cpp_L64"]["hits"][mask]).mean()) / 10 * 100
    quality = paired_bootstrap(differences, 10, args.bootstrap_repetitions, args.bootstrap_seed, 1)
    tested = set(postings)
    tuning_counts = Counter(map(int, tuning["labels"]))
    common_tuning_queries = sum(tuning_counts[label] for label in tested)
    standardized = {"scope": f"Original tuning label weights, renormalized to the {len(postings)} covered labels; no throughput inference",
                    "covered_original_tuning_queries": common_tuning_queries,
                    "excluded_original_tuning_queries": len(tuning["queries"]) - common_tuning_queries,
                    **{name + "_strict_id_recall": sum(tuning_counts[row["label"]] * row[name + "_strict_id_recall"]
                                                       for row in labels) / common_tuning_queries
                       for name in ("rust", "cpp")}}
    standardized["difference_percentage_points"] = 100 * (standardized["rust_strict_id_recall"]
                                                            - standardized["cpp_strict_id_recall"])
    paired_performance = [{"paired_repeat": rust["repeat"], "qps_ratio_rust_over_cpp": rust["qps"] / cpp["qps"],
                           "p99_ratio_rust_over_cpp": rust["p99_us"] / cpp["p99_us"]}
                          for rust, cpp in zip(native_rows, cpp_rows)]
    observed_negative = [row["label"] for row in labels if row["difference_percentage_points"] < 0]
    simultaneous_negative = [row["label"] for row in labels if
                             row["paired_query_bootstrap_simultaneous_95_percent_interval_pp"][1] < 0]
    control_summary = {"available": False}
    if control is not None:
        fixed_hits = control["row"]["hits"]
        control_summary = {"available": True, "source": control["source"],
            "retained_inputs": control["retained_inputs"], "timing_excluded": True, "policy_retuned": False,
            "strict_id_recall": float(fixed_hits.mean()) / 10,
            "policy_minus_fixed64": paired_bootstrap(native_rows[0]["hits"] - fixed_hits, 10,
                args.bootstrap_repetitions, args.bootstrap_seed + 1_000_000),
            "fixed64_minus_cpp_difference_percentage_points": float((fixed_hits - cpp_rows[0]["hits"]).mean()) / 10 * 100}
    cutoff_summary = {"available": False}
    if cutoff is not None:
        values = cutoff["engines"]
        cutoff_summary = {**{key: value for key, value in cutoff.items() if key != "engines"}, "available": True,
            "engines": {name: {key: value for key, value in item.items() if key != "hits"}
                        for name, item in values.items()},
            "policy_minus_cpp": paired_bootstrap(values["rust_policy"]["hits"] - values["cpp_L64"]["hits"],
                10, args.bootstrap_repetitions, args.bootstrap_seed + 2_000_000),
            "observed_negative_labels": [row["label"] for row in labels
                if row["distance_cutoff_diagnostic"]["policy_minus_cpp_difference_percentage_points"] < 0]}
    return {"schema_version": 1, "status": "all provenance, frozen-policy and query-validation checks passed",
            "scope": "held_out_all_remaining_natural_one_tag_public_queries_frozen_per_label_policy_no_retuning",
            "inputs": {name: asset(path) for name, path in (("native", args.native),
                       ("cpp", args.cpp), ("plan", args.plan), ("plan_evidence", args.plan_evidence),
                       ("heldout_manifest", args.prepared / "manifest.json"),
                       ("tuning_manifest", args.tuning_prepared / "manifest.json"))},
            "cpp_retained_inputs": cpp_inputs, "native_retained_inputs": native_inputs,
            "identity": identity, "timing_jobs": jobs, "coverage": coverage,
            "frozen_plan": {"sha256": FROZEN_SHA, "retuned": False,
                            "entries": [{"label": label, "l": plan[label]} for label in sorted(plan)]},
            "rust": native, "cpp": cpp, "qps_ratio_rust_over_cpp": native["qps"] / cpp["qps"],
            "p99_ratio_rust_over_cpp": native["p99_us"] / cpp["p99_us"],
            "paired_performance": paired_performance,
            "paired_qps_ratio_median": statistics.median(row["qps_ratio_rust_over_cpp"] for row in paired_performance),
            "pairs_with_rust_higher_qps": sum(row["qps_ratio_rust_over_cpp"] > 1 for row in paired_performance),
            "query_weighted_quality": quality,
            "fixed64_quality_only_control": control_summary,
            "distance_cutoff_diagnostic": cutoff_summary,
            "original_tuning_weights_restricted_to_tested_labels": standardized,
            "macro_label_difference_percentage_points": statistics.mean(row["difference_percentage_points"] for row in labels),
            "observed_negative_labels": observed_negative, "simultaneous_bootstrap_negative_labels": simultaneous_negative,
            "tested_labels_observed_nonnegative": len(tested) - len(observed_negative), "labels": labels,
            "validation": {"all_result_and_fallback_counters_zero": True,
                "retained_ids_filter_memberships_and_hit_counts_recomputed": True,
                "raw_latencies_match_aggregate_and_label_mean_p99": True,
                "quality_repetitions_not_counted_as_independent_queries": True,
                "unique_quality_queries": len(heldout["queries"]),
                "timed_queries_per_engine": len(native_rows) * len(heldout["queries"]),
                "every_tested_label_observed_recall_at_least_cpp": not observed_negative,
                "median_qps_higher_than_cpp": native["qps"] > cpp["qps"],
                "qps_higher_than_cpp_in_every_pair": all(row["qps_ratio_rust_over_cpp"] > 1 for row in paired_performance),
                "median_p99_no_higher_than_cpp": native["p99_us"] <= cpp["p99_us"]},
            "uncertainty": {"resampling_unit": "paired query hit-count difference; neighbors and timing repeats are not independent quality samples",
                "bootstrap_repetitions": args.bootstrap_repetitions, "seed": args.bootstrap_seed,
                "numpy_version": np.__version__,
                "marginal_confidence": .95, "simultaneous_family_labels": families,
                "simultaneous_method": "Bonferroni-adjusted percentile bootstrap, family of tested labels only",
                "interval_coverage": "Nominal approximate bootstrap coverage, not a distribution-free bound",
                "noninferiority_margin_prespecified": None,
                "limitations": "Intervals describe empirical query resampling, not a guarantee for future ads traffic. A constant observed difference gives a degenerate interval; rare/unseen queries may behave differently. No noninferiority margin was selected after seeing the holdout."},
            "interpretation": [
                "Recall shortfalls are reported, never rejected or used to alter the frozen plan.",
                "The holdout contains every eligible remaining public one-tag query. Labels exhausted by tuning are untested, rather than passing.",
                "Query-weighted throughput reflects this natural holdout mixture; it differs from the label-balanced tuning mixture.",
                "Per-label mean/p99 describe service calls inside a concurrent mixed workload. No per-label QPS is inferred.",
                "Performance summaries are medians of complete warmed repetitions, with raw repeat ranges; per-label tails with few samples are explicitly flagged.",
                "Graphs, entry points and breadths differ; this compares complete frozen configurations, not isolated language speed.",
                "Closed-loop eight-worker QPS and latency do not establish a production arrival-rate SLO."],
            "analysis_script": asset(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "tuning-prepared", "public-queries", "public-metadata", "public-truth", "plan",
                 "plan-evidence", "native", "cpp", "cpp-index-manifest", "identity", "tuning-identity", "jobs-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpp-raw-dir", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--fixed-control", type=Path)
    parser.add_argument("--base", type=Path, help="Optional pinned base vectors for exact cutoff-tie diagnostics")
    parser.add_argument("--bootstrap-repetitions", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261023)
    args = parser.parse_args()
    require(args.bootstrap_repetitions >= 10000 and args.bootstrap_seed >= 0, "Invalid bootstrap settings")
    inputs = [value.resolve() for key, value in vars(args).items() if isinstance(value, Path) and key != "output"]
    require(args.output.resolve() not in inputs, "Output must not overwrite an input")
    result = summarize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "heldout_queries": result["coverage"]["unique_heldout_queries"],
                      "tested_labels": result["coverage"]["tested_labels"],
                      "qps_ratio_rust_over_cpp": result["qps_ratio_rust_over_cpp"],
                      "observed_negative_labels": result["observed_negative_labels"]}))


if __name__ == "__main__":
    main()
