#!/usr/bin/env python3
"""Plot validated final native Linux normalized measurements, never diagnostics.

Requires matplotlib only when rendering. Default profile: full YFCC10M,
k=10, 2000 queries, eight threads, label starts, beam four and no node cache.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile

from summarize import IDENTITY_FIELDS, METRICS, MIN_REPETITIONS, identity, summarize


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rejection_reasons(row, args):
    reasons = []
    native = ((row.get("implementation") == "cpp_native_linux" and
               row.get("storage") == "linux_aio_direct") or
              (row.get("implementation") == "rust" and row.get("storage") == "native_io_uring"))
    if not native:
        reasons.append("not_native_linux_direct_io")
    if row.get("timing_authoritative") is not True:
        reasons.append("timing_not_authoritative")
    if row.get("latency_contract") != "public_search_call":
        reasons.append("not_public_call_latency")
    if row.get("cache_bypass") is not True or row.get("shared_cache_nodes") != 0:
        reasons.append("different_cache_contract")
    for metric in ("filter_violations", "short_results", "duplicate_results"):
        value = row.get(metric)
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            reasons.append(f"unknown_or_nonzero_{metric}")
    for field in ("n_base", "queries", "k", "threads", "beam", "starts"):
        if row.get(field) != getattr(args, field):
            reasons.append(f"different_{field}")
    if args.index_prefix is not None and row.get("index_prefix") != args.index_prefix:
        reasons.append("different_index_prefix")
    if args.prepared_identity is not None and row.get("prepared_identity") != args.prepared_identity:
        reasons.append("different_prepared_identity")
    for metric in METRICS:
        value = row.get(metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            reasons.append(f"invalid_{metric}")
    if not reasons:
        if not 0 <= row["recall"] <= 1 or row["qps"] <= 0 or any(row[key] < 0 for key in METRICS if key not in ("recall", "qps")):
            reasons.append("invalid_metric_range")
        if not row["p50_us"] <= row["p95_us"] <= row["p99_us"]:
            reasons.append("unordered_latency_percentiles")
    return reasons


def aggregate_inputs(paths, args):
    source_files = []
    file_hashes = set()
    groups = defaultdict(list)
    exclusions = []
    excluded_reasons = Counter()
    for supplied in paths:
        path = Path(supplied).resolve()
        digest = sha256(path)
        if digest in file_hashes:
            raise ValueError(f"Duplicate input contents would double-count measurements: {path}")
        file_hashes.add(digest)
        source_files.append({"path": str(path), "sha256": digest})
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            reasons = rejection_reasons(row, args)
            if reasons:
                excluded_reasons.update(reasons)
                exclusions.append({"path": str(path), "line": line_number, "reasons": reasons})
            else:
                row["plot_source"] = {"path": str(path), "line": line_number}
                groups[identity(row, IDENTITY_FIELDS)].append(row)
    short_repeats = []
    retained = []
    for rows in groups.values():
        if len(rows) < args.min_repeats:
            short_repeats.append({**{key: rows[0].get(key) for key in IDENTITY_FIELDS},
                                  "repetitions": len(rows)})
        else:
            retained.extend(rows)
    if not retained:
        raise ValueError("No validated authoritative native Linux measurements match the profile/repeat requirement; no plot generated")
    for field in ("prepared_identity", "index_prefix"):
        values = {json.dumps(row.get(field), sort_keys=True) for row in retained}
        if len(values) != 1 or "null" in values:
            raise ValueError(f"Multiple or missing {field} values: split the experiment with --{field.replace('_', '-')}")
    # Reuse the exact configuration grouping, floor selection and label diagnostics.
    with tempfile.TemporaryDirectory(prefix="diskann-final-plot-") as folder:
        selected_input = Path(folder) / "validated.jsonl"
        selected_input.write_text("\n".join(json.dumps(row, allow_nan=False) for row in retained) + "\n")
        summary = summarize([selected_input], args.recall_floor, 0.01)
    return summary, {"inputs": source_files, "retained_rows": len(retained),
                     "exclusion_reason_counts": dict(excluded_reasons), "excluded_rows": exclusions,
                     "configurations_below_min_repeats": short_repeats,
                     "retained_sources": [row["plot_source"] for row in retained]}


def method_name(row, include_tuning=False):
    if row["implementation"] == "cpp_native_linux":
        name = "C++ single-hop"
    else:
        name = {"single": "Rust single-hop", "multihop": "Rust multi-hop",
                "inline": "Rust adaptive inline"}.get(row["mode"], f"Rust {row['mode']}")
    if include_tuning and row["mode"] == "inline":
        name += f" (sample={row.get('adaptive_sample')}, scale={row.get('adaptive_scale')})"
    return name


def color(row):
    if row["implementation"] == "cpp_native_linux":
        return "#666666"
    return {"single": "#0072B2", "multihop": "#D55E00", "inline": "#009E73"}.get(row["mode"], "#CC79A7")


def method_order(row):
    return (0 if row["implementation"] == "cpp_native_linux" else
            {"single": 1, "multihop": 2, "inline": 3}.get(row["mode"], 4),
            str(row.get("adaptive_sample")), str(row.get("adaptive_scale")))


def save_figure(figure, prefix, suffix):
    files = []
    for extension in ("png", "svg"):
        path = Path(f"{prefix}-{suffix}.{extension}")
        figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        files.append({"path": str(path.resolve()), "sha256": sha256(path)})
    return files


def render_curves(plt, summary, args):
    figure, axis = plt.subplots(figsize=(10.8, 6.2))
    series = defaultdict(list)
    fields = ("implementation", "mode", "adaptive_sample", "adaptive_scale")
    for row in summary["configurations"]:
        series[identity(row, fields)].append(row)
    for index, rows in enumerate(sorted(series.values(), key=lambda rows: method_order(rows[0]))):
        rows.sort(key=lambda row: row["l"])
        x = [100 * row["recall"] for row in rows]
        y = [row["qps"] for row in rows]
        xerr = [[100 * (row["recall"] - row["recall_min"]) for row in rows],
                [100 * (row["recall_max"] - row["recall"]) for row in rows]]
        yerr = [[row["qps"] - row["qps_min"] for row in rows],
                [row["qps_max"] - row["qps"] for row in rows]]
        axis.errorbar(x, y, xerr=xerr, yerr=yerr, color=color(rows[0]),
                      marker=("o", "s", "^", "D", "v", "P")[index % 6],
                      linestyle=("-", "--", "-.", ":")[index % 4], linewidth=1.5,
                      markersize=5, capsize=3, label=method_name(rows[0], True))
    axis.axvline(100 * args.recall_floor, color="black", linewidth=1, linestyle="--",
                 label=f"Required recall: {100 * args.recall_floor:g}%")
    axis.set_xlabel(f"Strict filtered ID recall@{args.k} (%)")
    axis.set_ylabel("Queries per second (log scale)")
    axis.set_yscale("log")
    axis.grid(True, alpha=.25, which="both")
    axis.set_title(f"Filtered SSD search on {args.n_base / 1_000_000:g}M vectors\n"
                   f"k={args.k}, {args.queries:,} queries, {args.threads} threads, same graph/PQ and label starts")
    axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    figure.text(.01, .01, "Native Linux; direct I/O, no shared node cache. Markers are measured configurations; lines guide the eye.\n"
                         "Medians across repeats; whiskers show repeat min–max. C++ uses native AIO; Rust uses io_uring.", fontsize=8)
    figure.tight_layout(rect=(0, .065, 1, 1))
    files = save_figure(figure, args.out_prefix, "recall-qps")
    plt.close(figure)
    return files


def render_matched(plt, summary, args):
    rows = sorted(summary["selected"], key=method_order)
    figure, axes = plt.subplots(1, 3, figsize=(14, 6.2))
    panels = (("qps", 1, "Queries per second", ",.0f"),
              ("p99_us", .001, "p99 public-call latency (ms)", ".2f"),
              ("requests_per_query", 1, "Logical read requests / query", ".1f"))
    labels = []
    for row in rows:
        label = method_name(row)
        if row.get("eligible"):
            label += f"\nL={row['l']}"
            if row["mode"] == "inline":
                label += f", sample={row.get('adaptive_sample')}, scale={row.get('adaptive_scale')}"
        labels.append(label)
    for axis, (metric, scale, title, number_format) in zip(axes, panels):
        for position, row in enumerate(rows):
            value = row.get(metric) if row.get("eligible") else None
            if value is None:
                axis.text(position, .5, "No qualifying\nmeasurement" if not row.get("eligible") else "Metric unavailable",
                          ha="center", va="center", fontsize=8, rotation=90,
                          transform=axis.get_xaxis_transform())
                continue
            lo, hi = row.get(f"{metric}_min", value), row.get(f"{metric}_max", value)
            axis.bar(position, value * scale, width=.65, color=color(row),
                     yerr=[[max(0, value - lo) * scale], [max(0, hi - value) * scale]], capsize=3)
            axis.annotate(format(value * scale, number_format), (position, value * scale),
                          xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
        axis.set_xticks(range(len(rows)), labels, rotation=25, ha="right", fontsize=8)
        axis.set_title(title, fontsize=10)
        axis.grid(True, axis="y", alpha=.25)
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0)
        axis.margins(y=.18)
    figure.suptitle(f"Measured configurations meeting {100 * args.recall_floor:g}% recall in every repeat\n"
                    f"{args.n_base / 1_000_000:g}M vectors · k={args.k} · {args.queries:,} queries · {args.threads} threads", fontsize=12)
    figure.text(.01, .01, "Bars select the highest median QPS among measured configurations meeting the common recall/result-validity floor; other panels use that same configuration.\n"
                         "Bars are repeat medians; whiskers are repeat min–max, not confidence intervals. Per-label regressions remain in the provenance JSON.", fontsize=8)
    figure.tight_layout(rect=(0, .075, 1, .91))
    files = save_figure(figure, args.out_prefix, "matched-floor")
    plt.close(figure)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-prefix", type=Path,
                        default=Path("benchmark-results/single-hop/results/native-10m-t8"))
    parser.add_argument("--recall-floor", type=float, default=.95)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--n-base", type=int, default=10_000_000)
    parser.add_argument("--queries", type=int, default=2000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--starts", default="label", choices=("label",))
    parser.add_argument("--index-prefix")
    parser.add_argument("--prepared-identity")
    args = parser.parse_args()
    if args.min_repeats < MIN_REPETITIONS or not 0 < args.recall_floor <= 1:
        parser.error("at least three repeats and recall floor in (0,1] are required")
    summary, provenance = aggregate_inputs(args.inputs, args)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise SystemExit("Rendering requires matplotlib in the selected Python environment; no diagnostic/fake data fallback is used") from error
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.dpi": 200})
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    figures = render_curves(plt, summary, args) + render_matched(plt, summary, args)
    provenance.update({"created_utc": datetime.now(timezone.utc).isoformat(),
                       "profile": {key: getattr(args, key) for key in ("n_base", "queries", "k", "threads", "beam", "starts")},
                       "recall_floor": args.recall_floor, "minimum_repeats": args.min_repeats,
                       "cache_bypass": True, "shared_cache_nodes": 0,
                       "latency_contract": "public_search_call", "matplotlib_version": matplotlib.__version__,
                       "python_version": platform.python_version(),
                       "plot_script_sha256": sha256(Path(__file__)),
                       "summarizer_sha256": sha256(Path(__file__).with_name("summarize.py")),
                       "summary": summary, "figures": figures,
                       "interpretation": "Only validated authoritative native Linux measurements. No interpolated recall-floor result; error bars are repeat ranges, not confidence intervals. Closed-loop per-process measurements are not distributed service SLOs."})
    output = Path(f"{args.out_prefix}-provenance.json")
    output.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(figures)} figures and {output}")


if __name__ == "__main__":
    main()
