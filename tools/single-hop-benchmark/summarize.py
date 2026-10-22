#!/usr/bin/env python3
"""Compare measured configurations at a recall floor, without interpolation."""

import argparse
import collections
import json
import statistics
from pathlib import Path


REQUIRED_IDENTITY_FIELDS = (
    "implementation", "index_prefix", "prepared_identity", "queries", "beam",
    "cache_bypass", "shared_cache_nodes", "storage", "latency_contract", "timing_authoritative",
    "mode", "starts", "n_base", "k", "threads", "l",
)
TUNING_FIELDS = ("l", "adaptive_sample", "adaptive_scale")
IDENTITY_FIELDS = REQUIRED_IDENTITY_FIELDS + ("adaptive_sample", "adaptive_scale")
METRICS = ("recall", "qps", "mean_us", "p50_us", "p95_us", "p99_us")
OPTIONAL_METRICS = ("mean_cmps", "mean_full_cmps", "mean_hops", "reader_time_us")
MIN_REPETITIONS = 3


def identity(row, fields):
    # Accept either a fingerprint string or a structured dataset identity.
    return tuple(json.dumps(row.get(key), sort_keys=True, separators=(",", ":"))
                 for key in fields)


def summarize_labels(rows):
    """Keep label diagnostics separate from aggregate recall eligibility."""
    if any(not r.get("labels") for r in rows):
        return [], False
    label_sets = [{item["label"] for item in r["labels"]} for r in rows]
    if any(labels != label_sets[0] for labels in label_sets):
        return [], False
    grouped = collections.defaultdict(list)
    for row in rows:
        for item in row["labels"]:
            grouped[item["label"]].append(item)
    labels = []
    for label, observations in sorted(grouped.items()):
        if len(observations) != len(rows) or any(
            (item.get("n"), item.get("eligible")) !=
            (observations[0].get("n"), observations[0].get("eligible"))
            for item in observations
        ):
            return [], False
        recalls = [item["recall"] for item in observations]
        result = {"label": label, "n": observations[0].get("n"),
                  "eligible": observations[0].get("eligible"),
                  "recall": statistics.median(recalls),
                  "recall_min": min(recalls), "recall_max": max(recalls)}
        for metric in ("mean_us", "p50_us", "p95_us"):
            values = [item[metric] for item in observations if item.get(metric) is not None]
            complete = len(values) == len(observations)
            result[metric] = statistics.median(values) if complete else None
            result[f"{metric}_min"] = min(values) if complete else None
            result[f"{metric}_max"] = max(values) if complete else None
        labels.append(result)
    return labels, True


def compare_label_recalls(selected, max_drop=0.01):
    """Flag observed label recall drops against a compatible selected multi-hop run.

    The comparison uses the same global recall requirement, dataset, index, k,
    concurrency, starts and I/O settings. Warnings do not change eligibility.
    """
    if not 0 <= max_drop <= 1:
        raise ValueError("label recall drop threshold must be between zero and one")
    comparison_fields = tuple(key for key in REQUIRED_IDENTITY_FIELDS
                              if key not in ("l", "mode"))
    baselines = {identity(row, comparison_fields): row for row in selected
                 if row.get("mode") == "multihop" and row.get("eligible")}
    comparisons = []
    for row in selected:
        if row.get("mode") == "multihop":
            continue
        result = {key: row.get(key) for key in comparison_fields + ("mode",) + TUNING_FIELDS}
        result["drop_threshold_percentage_points"] = 100 * max_drop
        if not row.get("eligible"):
            result["status"] = "candidate does not meet the global recall and validation requirements"
            comparisons.append(result)
            continue
        baseline = baselines.get(identity(row, comparison_fields))
        if baseline is None:
            result["status"] = "no eligible multi-hop baseline with matching dataset, index, k, threads, starts and I/O"
        elif not row.get("labels_complete") or not baseline.get("labels_complete"):
            result["status"] = "per-label measurements unavailable or inconsistent across repeats"
        else:
            candidate_labels = {item["label"]: item for item in row["labels"]}
            baseline_labels = {item["label"]: item for item in baseline["labels"]}
            if candidate_labels.keys() != baseline_labels.keys() or any(
                (candidate_labels[label]["n"], candidate_labels[label]["eligible"]) !=
                (baseline_labels[label]["n"], baseline_labels[label]["eligible"])
                for label in candidate_labels
            ):
                result["status"] = "per-label query or eligible-set counts do not match"
            else:
                warnings = []
                for label, candidate in candidate_labels.items():
                    reference = baseline_labels[label]
                    drop = reference["recall"] - candidate["recall"]
                    if drop > max_drop + 1e-12:
                        warnings.append({"label": label, "queries": candidate["n"],
                                         "eligible": candidate["eligible"],
                                         "candidate_recall": candidate["recall"],
                                         "multihop_recall": reference["recall"],
                                         "drop_percentage_points": 100 * drop})
                result.update({"status": "observed label recall drops" if warnings else "no label recall drops above threshold",
                               "candidate_global_recall": row["recall"],
                               "multihop_global_recall": baseline["recall"],
                               "multihop_l": baseline["l"], "warnings": warnings,
                               "interpretation": "diagnostic on this query sample; does not change aggregate eligibility or establish statistical significance"})
        comparisons.append(result)
    return comparisons


def summarize(paths, recall_floor, label_drop_threshold=None):
    if not 0 <= recall_floor <= 1:
        raise ValueError("recall floor must be between zero and one")
    groups = collections.defaultdict(list)
    seen_paths = set()
    for path in paths:
        path = Path(path).resolve()
        if path in seen_paths:
            raise ValueError(f"input file passed more than once: {path}")
        seen_paths.add(path)
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                missing = [key for key in METRICS if row.get(key) is None]
                if missing:
                    raise ValueError(
                        f"{path}:{line_number}: normalized measurement required; missing {missing}"
                    )
                key = identity(row, IDENTITY_FIELDS)
                groups[key].append(row)
    configurations = []
    for rows in groups.values():
        result = {key: rows[0].get(key) for key in IDENTITY_FIELDS}
        for metric in METRICS + OPTIONAL_METRICS:
            values = [r[metric] for r in rows if r.get(metric) is not None]
            result[metric] = statistics.median(values) if values else None
            result[f"{metric}_min"] = min(values) if values else None
            result[f"{metric}_max"] = max(values) if values else None
        result["repetitions"] = len(rows)
        missing_identity = [key for key in REQUIRED_IDENTITY_FIELDS if result[key] is None]
        result["missing_identity_fields"] = missing_identity
        validation_complete = all(
            isinstance(r.get(key), int) and not isinstance(r.get(key), bool) and r[key] >= 0
            for r in rows for key in ("filter_violations", "short_results", "duplicate_results")
        )
        result["validation_complete"] = validation_complete
        for metric in ("filter_violations", "short_results", "duplicate_results"):
            result[metric] = sum(r[metric] for r in rows) if validation_complete else None
        for metric, source in (("requests_per_query", "reader_requests"),
                               ("bytes_per_query", "reader_bytes")):
            values = [r[source] / r["queries"] for r in rows
                      if r.get(source) is not None and r.get("queries", 0) > 0]
            result[metric] = statistics.median(values) if values else None
            result[f"{metric}_min"] = min(values) if values else None
            result[f"{metric}_max"] = max(values) if values else None
        result["eligible"] = (
            len(rows) >= MIN_REPETITIONS and not missing_identity and validation_complete
            and all(r.get("timing_authoritative") is True for r in rows)
            and result["latency_contract"] == "public_search_call"
            and result["recall_min"] >= recall_floor
            and result["filter_violations"] == 0 and result["short_results"] == 0
            and result["duplicate_results"] == 0
        )
        result["labels"], result["labels_complete"] = summarize_labels(rows)
        configurations.append(result)
    candidates = collections.defaultdict(list)
    selection_fields = tuple(key for key in IDENTITY_FIELDS if key not in TUNING_FIELDS)
    for row in configurations:
        key = identity(row, selection_fields)
        candidates[key].append(row)
    selected = []
    for rows in candidates.values():
        valid = [r for r in rows if r["eligible"]]
        if valid:
            selected.append(max(valid, key=lambda r: r["qps"]))
        else:
            selected.append({
                **{key: rows[0].get(key) for key in selection_fields},
                "status": "no measured configuration meets three-repeat, recall, identity, authoritative-timing and explicit result-validation requirements",
                "best_measured_recall": max(r["recall"] for r in rows),
            })
    result = {"recall_floor": recall_floor,
            "selection": "fastest median-QPS measured configuration with at least three repeats; every repeat must have authoritative timing, meet recall floor and have explicit result validation",
            "selected": selected, "configurations": configurations}
    if label_drop_threshold is not None:
        result["per_label_comparisons"] = compare_label_recalls(selected, label_drop_threshold)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--recall", type=float, default=0.95)
    parser.add_argument("--per-label-regressions", action="store_true",
                        help="Report label recall drops against compatible selected multi-hop runs")
    parser.add_argument("--label-drop-threshold", type=float, default=0.01,
                        help="Recall-fraction drop to flag; .01 means one percentage point")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.inputs, args.recall,
                       args.label_drop_threshold if args.per_label_regressions else None)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    for row in result["selected"]:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
