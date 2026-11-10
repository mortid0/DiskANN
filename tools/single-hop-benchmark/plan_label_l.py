#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Choose and validate an in-sample per-label breadth plan against fixed C++ L64.

``plan`` uses a quality-only native grid and writes a direct label-to-L JSON
object plus a separate evidence file. ``summarize`` validates three independent
timing repetitions of that exact plan. The two graphs differ, and the workload
used to select L is reused for evaluation: neither mode establishes held-out
quality, general segment floors, or a production arrival-rate latency guarantee.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

from summarize_delta import digest, finite, integer, label_quality, provenance, read_rows, require
from summarize_migration import read_measurements, summarize as cpp_summary
from summarize_native_build import native_experiment


GRID = (64, 68, 72, 80, 96, 112, 128, 144)
BASE_L = 64
LABELS = 64
VALIDITY = ("filter_violations", "duplicate_results", "short_results", "exact_fallback_queries")
METRICS = ("qps", "mean_us", "p50_us", "p95_us", "p99_us", "wall_seconds")
PROTOCOL = ("implementation", "queries", "k", "threads", "prepared_identity", "sequence",
            "latency_contract", "strict_id_recall")


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_l(row):
    require(row.get("label_l_plan") is None, "A diagnostic grid must use fixed-L observations")
    require(all(item.get("actual_l", row["l"]) == row["l"] for item in row["labels"]),
            "A fixed-L row contains a different per-label breadth")


def native_rows(path, mixed=False):
    rows = read_rows(path)
    require(all(row.get("kind") in ("experiment", "query") for row in rows),
            "Expected query-only native log")
    exp = native_experiment(rows, "query")
    queries = [row for row in rows if row.get("kind") == "query"]
    require(queries, "No native query observations")
    first = queries[0]
    for row in queries:
        require(all(row.get(key) == first.get(key) for key in PROTOCOL), "Native query identities differ")
        require(row.get("implementation") == "rust_native_lifecycle" and row.get("phase") == "query"
                and row.get("strict_id_recall") is True
                and row.get("latency_contract") == "public_snapshot_search_call",
                "Unexpected native strict search contract")
        require(all(type(row.get(key)) is int and row[key] == expected for key, expected in
                    {"queries": 2000, "warmup_queries": 2000, "k": 10, "threads": 8,
                     "sequence": 0, "delta_candidates": 0, "base_searches": 2000}.items()),
                "Unexpected query count, sequence, delta, warmup or concurrency")
        require(all(type(row.get(key)) is int and row[key] == 0 for key in VALIDITY),
                "Invalid, short, duplicate, fallback or unvalidated native result")
        identity = row.get("prepared_identity")
        require(type(identity) is str and len(identity.split(":")) == 4
                and all(digest(part) for part in identity.split(":")), "Invalid prepared asset identity")
        require(integer(row.get("l"), BASE_L) and integer(row.get("repeat")), "Invalid breadth/repeat")
        require(row.get("timing_authoritative") is mixed,
                "Plan evaluation must be authoritative; grid observations must be diagnostic")
        require(row.get("label_l_tuning_scope") == "empirical_on_existing_prepared_workload_not_held_out"
                if mixed else row.get("label_l_tuning_scope") is None,
                "Missing mixed-plan tuning disclosure or unexpected fixed-grid tuning policy")
        require(all(finite(row.get(key), strict=True) for key in METRICS)
                and row["p50_us"] <= row["p95_us"] <= row["p99_us"]
                and math.isclose(row["qps"], row["queries"] / row["wall_seconds"], rel_tol=1e-12),
                "Invalid timing metadata (grid timing is never summarized)")
        quality = label_quality(row)
        require(len(quality) == LABELS, "Expected all 64 query labels")
        if not mixed:
            fixed_l(row)
    return exp, queries


def cpp_target(path, native):
    groups = read_measurements(path)
    key = ("single", BASE_L, 10, 8)
    require(key in groups, "Missing fresh C++ fixed L64 reference")
    rows = groups[key]
    qualities = []
    for row in rows:
        require(row.get("implementation") == "cpp_native_linux" and row.get("starts") == "label"
                and row.get("strict_id_recall") is True and row.get("queries") == 2000
                and row.get("n_base") == 10000000 and row.get("warmup_queries") == 2000
                and row.get("beam") == 4 and row.get("storage") == "linux_aio_direct"
                and row["prepared_identity"] == native["prepared_identity"],
                "C++/native corpus, workload, strict-query or I/O protocol differs")
        require(row.get("label_l_plan") is None, "C++ target must use fixed L64")
        quality = label_quality({**row, "labels": [{**item, "queries": item["n"]} for item in row["labels"]]})
        require(len(quality) == LABELS and {label: n for label, (n, _) in quality.items()}
                == {label: n for label, (n, _) in label_quality(native).items()},
                "C++/native per-label samples differ")
        qualities.append(quality)
    require(all(quality == qualities[0] for quality in qualities), "Repeated C++ target quality differs")
    return rows, qualities[0]


def select_plan(grid, target):
    """Choose by integer hit counts; quality need not be monotone in L."""
    selected = {}
    evidence = []
    for label in sorted(target):
        sample, hits = target[label]
        observations = []
        for breadth in GRID:
            require(label in grid[breadth] and grid[breadth][label][0] == sample,
                    "Grid label coverage or samples differ from C++ targets")
            observed = grid[breadth][label][1]
            observations.append({"l": breadth, "strict_id_hits": observed,
                                 "strict_id_recall": observed / (sample * 10),
                                 "difference_hits_from_CPP_L64": observed - hits})
        choices = [row for row in observations if row["strict_id_hits"] >= hits]
        require(choices, f"Label {label} does not reach C++ L64 quality at any measured L through {max(GRID)}")
        choice = choices[0]
        selected[label] = choice["l"]
        evidence.append({"label": label, "queries": sample, "k": 10,
                         "target_CPP_l": BASE_L, "target_CPP_strict_id_hits": hits,
                         "target_CPP_strict_id_recall": hits / (sample * 10),
                         "selected_l": choice["l"], "selected_strict_id_hits": choice["strict_id_hits"],
                         "selected_strict_id_recall": choice["strict_id_recall"],
                         "selected_difference_hits": choice["strict_id_hits"] - hits,
                         "measured_grid": observations})
    return selected, evidence


def planning(grid_path, cpp_path, plan_path):
    grid_path, cpp_path, plan_path = grid_path.resolve(), cpp_path.resolve(), plan_path.resolve()
    exp, rows = native_rows(grid_path)
    groups = defaultdict(list)
    for row in rows:
        groups[row["l"]].append(row)
    require(set(groups) == set(GRID) and all(len(items) == 1 and items[0]["repeat"] == 0
                                            for items in groups.values()),
            "Expected one repeat-zero diagnostic observation at every fixed grid L")
    cpp_rows, target = cpp_target(cpp_path, rows[0])
    plan, labels = select_plan({l: label_quality(items[0]) for l, items in groups.items()}, target)
    data = (json.dumps({str(label): l for label, l in plan.items()}, indent=2) + "\n").encode()
    histogram = {str(l): list(plan.values()).count(l) for l in GRID if l in plan.values()}
    result = {"schema_version": 1, "status": "all quality assertions passed",
              "scope": "in_sample_per_label_breadth_tuning_against_fresh_CPP_fixed_L64",
              "inputs": {"grid": provenance(grid_path), "cpp": provenance(cpp_path)},
              "native_experiment": exp, "prepared_identity": rows[0]["prepared_identity"],
              "plan": {"path": str(plan_path), "content_sha256": hashlib.sha256(data).hexdigest(),
                       "entries": [{"label": label, "l": l} for label, l in plan.items()],
                       "label_count": len(plan), "nominal_l": BASE_L, "breadth_histogram": histogram,
                       "query_weighted_mean_l": sum(row["queries"] * row["selected_l"] for row in labels) / 2000,
                       "query_weighted_l_increase_percent": 100 *
                           (sum(row["queries"] * row["selected_l"] for row in labels) / (2000 * BASE_L) - 1)},
              "validation": {"grid_breadths": list(GRID), "observations_per_L": 1,
                             "all_64_labels_present": True, "all_validity_and_fallback_counters_zero": True,
                             "CPP_target_repetitions": len(cpp_rows), "every_label_meets_CPP_L64_hits": True},
              "labels": labels,
              "interpretation": ["Each label uses the smallest qualifying measured L; no interpolation or monotone-recall assumption is used.",
                  "Plan selection reuses the same 2,000 queries that will evaluate it. This is in-sample tuning, not held-out validation or a general per-label recall guarantee.",
                  "Grid observations are single quality diagnostics. Their timings are excluded; a separate three-repeat mixed-plan run is required for performance.",
                  "The native and C++ graphs/entry points differ. Matching strict-ID hit counts is a finite-workload quality floor, not ordered-result parity."]}
    return data, result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Repeated JSON object key")
        result[key] = value
    return result


def load_plan(path):
    item = json.loads(path.read_text(), object_pairs_hook=unique_object)
    require(type(item) is dict and len(item) == LABELS, "Plan must cover exactly 64 labels")
    plan = {}
    for key, value in item.items():
        require(type(key) is str and key.isdecimal() and str(int(key)) == key
                and 0 <= int(key) < 2**32 and type(value) is int and value in GRID,
                "Invalid canonical label or unmeasured plan breadth")
        plan[int(key)] = value
    return plan


def archived_files(path):
    files = {}
    with tarfile.open(path, "r:*") as archive:
        for item in archive:
            if not item.isfile():
                continue
            name = item.name.removeprefix("./")
            require(name not in files, "Duplicate archived source path")
            files[name] = hashlib.sha256(archive.extractfile(item).read()).hexdigest()
    required = {"diskann-disk/src/filtered/" + name + ".rs" for name in
                ("native", "index", "view", "updates", "persistence", "metadata")}
    required.update({"diskann-disk/src/build/filtered.rs", "diskann/src/neighbor/queue.rs",
                     "diskann/src/graph/search/single_hop_filter_search.rs",
                     "diskann-disk/examples/filtered_lifecycle.rs"})
    require(required <= files.keys(), "Source archive is missing build/lifecycle/search files")
    return files


def identity_evidence(path, grid_path, query_path, native_exp):
    if path is None:
        return {"available": False,
                "scope": "Log root/protocol and dataset hashes are bound; no binary or CURRENT/catalog/WAL identity is asserted."}
    evidence = json.loads(path.read_text())
    require(evidence.get("schema_version") == 1, "Invalid grid/mixed identity evidence schema")
    shared = evidence.get("shared_index", {})
    require(type(shared.get("host")) is str and shared["host"]
            and type(shared.get("root")) is str and Path(shared["root"]).is_absolute(),
            "Invalid shared native host/root identity")
    logged_root = Path(native_exp["root"])
    root = Path(shared["root"])
    require(root == logged_root if logged_root.is_absolute()
            else root.parts[-len(logged_root.parts):] == logged_root.parts, "Identity root differs from native query root")
    versions = {}
    source_files = {}
    for name, log in (("grid", grid_path), ("mixed", query_path)):
        item = evidence.get(name, {})
        require(digest(item.get("log_sha256")) and item["log_sha256"] == file_sha(log)
                and digest(item.get("binary_sha256")), "Grid/mixed log hash or binary declaration differs")
        archive = item.get("source_archive", {})
        require(type(archive.get("path")) is str and digest(archive.get("sha256")), "Missing source archive identity")
        archive_path = Path(archive["path"])
        require(file_sha(archive_path) == archive["sha256"], "Source archive hash differs")
        index = item.get("index_identity", {})
        require(set(index) == {"current_sha256", "catalog_sha256", "journal_sha256"}
                and all(digest(value) for value in index.values()), "Invalid immutable index declarations")
        versions[name] = {**item, "local_source_archive": provenance(archive_path)}
        source_files[name] = archived_files(archive_path)
    require(versions["grid"]["index_identity"] == versions["mixed"]["index_identity"],
            "Grid/mixed committed index or journal identity changed")
    require(source_files["grid"].keys() == source_files["mixed"].keys(), "Archived source sets differ")
    changed = [name for name in sorted(source_files["grid"])
               if source_files["grid"][name] != source_files["mixed"][name]]
    require(changed == ["diskann-disk/examples/filtered_lifecycle.rs"],
            "Grid/mixed archived sources differ beyond the benchmark example")
    return {"available": True, "source": provenance(path), "shared_index": shared,
            "versions": versions, "changed_archived_files": changed,
            "unchanged_archived_files": len(source_files["grid"]) - len(changed),
            "verification_scope": "Local grid/mixed logs and source archives are hashed; all archived files except the benchmark example match. Binary and committed index/journal hashes are linked declarations from separately collected host evidence."}


def evaluate(query_path, plan_path, evidence_path, cpp_path, identity_path=None):
    query_path, plan_path, evidence_path, cpp_path = (path.resolve() for path in
                                                    (query_path, plan_path, evidence_path, cpp_path))
    evidence = json.loads(evidence_path.read_text())
    require(evidence.get("schema_version") == 1
            and evidence.get("scope") == "in_sample_per_label_breadth_tuning_against_fresh_CPP_fixed_L64"
            and evidence.get("status") == "all quality assertions passed", "Invalid plan evidence")
    grid_path = Path(evidence["inputs"]["grid"]["path"])
    # Recompute selection rather than trusting an editable sidecar's quality claims.
    planned_data, recomputed = planning(grid_path, cpp_path, plan_path)
    require(planned_data == plan_path.read_bytes() and recomputed == evidence,
            "Plan/evidence do not match independently recomputed grid selection")
    plan = load_plan(plan_path)
    exp, rows = native_rows(query_path, mixed=True)
    require(len(rows) == 3 and sorted(row["repeat"] for row in rows) == [0, 1, 2]
            and all(row["l"] == BASE_L for row in rows), "Expected three unique mixed-plan L64 repeats")
    require(exp == evidence["native_experiment"] and rows[0]["prepared_identity"] == evidence["prepared_identity"],
            "Mixed query index or workload differs from the tuning grid")
    entries = [{"label": label, "l": plan[label]} for label in sorted(plan)]
    expected_hash = file_sha(plan_path)
    for row in rows:
        meta = row.get("label_l_plan")
        require(type(meta) is dict and type(meta.get("path")) is str and meta["path"]
                and meta.get("content_sha256") == expected_hash and meta.get("entries") == entries,
                "Mixed query plan hash/entries are missing or different")
        require({item["label"]: item.get("actual_l") for item in row["labels"]} == plan,
                "Reported actual per-label breadths differ from plan")
    cpp_rows, target = cpp_target(cpp_path, rows[0])
    first_quality = label_quality(rows[0])
    require(all(label_quality(row) == first_quality for row in rows), "Mixed-plan repeat quality differs")
    label_rows = []
    for label, (sample, hits) in sorted(target.items()):
        actual = first_quality[label][1]
        require(actual >= hits, f"Mixed-plan label {label} falls below C++ L64 strict-ID recall")
        label_rows.append({"label": label, "queries": sample, "l": plan[label], "k": 10,
                           "CPP_L64_strict_id_hits": hits, "native_strict_id_hits": actual,
                           "CPP_L64_strict_id_recall": hits / (sample * 10),
                           "native_strict_id_recall": actual / (sample * 10),
                           "difference_hits": actual - hits,
                           "difference_percentage_points": 100 * (actual - hits) / (sample * 10)})
    native = {key: statistics.median(row[key] for row in rows) for key in METRICS}
    native["recall"] = sum(hits for _, hits in first_quality.values()) / 20000
    native["repetitions"] = 3
    native["ranges"] = {key: [min(row[key] for row in rows), max(row[key] for row in rows)] for key in METRICS}
    reference = cpp_summary(cpp_rows)
    return {"schema_version": 1, "status": "all mixed-plan assertions passed",
            "scope": "authoritative_native_mixed_label_L_vs_different_CPP_graph_fixed_L64_on_in_sample_tuned_queries",
            "inputs": {name: provenance(path) for name, path in
                       (("query", query_path), ("plan", plan_path), ("plan_evidence", evidence_path), ("cpp", cpp_path))},
            "native_experiment": exp, "plan": evidence["plan"],
            "identity": identity_evidence(identity_path, grid_path, query_path, exp),
            "validation": {"repetitions": 3, "unique_queries": 2000, "all_64_labels_meet_CPP_L64_hits_each_repeat": True,
                           "all_validity_and_fallback_counters_zero": True, "matching_delta_candidates": 0,
                           "full_same_process_warmup_each_repeat": True, "plan_hash_and_actual_breadths_verified": True,
                           "selection_recomputed_from_diagnostic_grid": True},
            "native": native, "CPP_fixed_L64": reference,
            "qps_ratio_native_over_CPP": native["qps"] / reference["qps"],
            "latency_ratio_native_over_CPP": {key: native[key] / reference[key]
                                              for key in ("mean_us", "p50_us", "p95_us", "p99_us")},
            "labels": label_rows,
            "interpretation": ["The native plan meets every label's fixed-C++-L64 strict-ID hit count on this sample in every repeat; result IDs need not be identical.",
                "The same 2,000 queries selected and evaluated the plan. These are in-sample matched-quality observations, not held-out per-label guarantees or additional independent quality samples.",
                "Graphs, entry points and per-label breadths differ. The throughput ratio compares complete measured configurations and does not isolate search-language or graph-builder speed.",
                "Each latency uses its public query API contract (native snapshot search, C++ public search), full same-process warmup and native cache-bypassed SSD reads.",
                "Throughput and percentiles are medians of three repeats, not pooled tails, confidence intervals or a production arrival-rate SLO."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="mode", required=True)
    plan = commands.add_parser("plan")
    for name in ("grid", "cpp", "output", "evidence"):
        plan.add_argument("--" + name, type=Path, required=True)
    summary = commands.add_parser("summarize")
    for name in ("query", "plan", "evidence", "cpp", "output"):
        summary.add_argument("--" + name, type=Path, required=True)
    summary.add_argument("--identity-evidence", type=Path)
    args = parser.parse_args()
    if args.mode == "plan":
        require(len({path.resolve() for path in (args.grid, args.cpp, args.output, args.evidence)}) == 4,
                "Input/output paths must be distinct")
        data, result = planning(args.grid, args.cpp, args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output": str(args.output), "evidence": str(args.evidence),
                          **{key: value for key, value in result["plan"].items() if key != "entries"}}))
    else:
        require(args.output.resolve() not in {path.resolve() for path in
                (args.query, args.plan, args.evidence, args.cpp)}, "Output must not overwrite inputs")
        require(args.identity_evidence is None or args.output.resolve() != args.identity_evidence.resolve(),
                "Output must not overwrite identity evidence")
        result = evaluate(args.query, args.plan, args.evidence, args.cpp, args.identity_evidence)
        result["analysis_script"] = provenance(Path(__file__))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output": str(args.output), "status": result["status"],
                          "qps_ratio_native_over_CPP": result["qps_ratio_native_over_CPP"]}))


if __name__ == "__main__":
    main()
