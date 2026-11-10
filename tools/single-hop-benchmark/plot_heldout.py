#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Plot validated held-out per-label recall differences and simultaneous intervals.

Example: python plot_heldout.py --summary heldout-summary.json --out recall-by-label
Writes recall-by-label.png and recall-by-label.svg from actual summary values.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


INTERVAL = "paired_query_bootstrap_simultaneous_95_percent_interval_pp"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_summary(path):
    summary = json.loads(path.read_text())
    require(summary.get("schema_version") == 1
            and summary.get("status") == "all provenance, frozen-policy and query-validation checks passed",
            "Expected a fully validated held-out summary")
    require(summary.get("frozen_plan", {}).get("retuned") is False,
            "Figure requires the unchanged frozen policy")
    labels = summary.get("labels")
    require(type(labels) is list and labels, "Summary contains no tested labels")
    seen = set()
    for row in labels:
        require(type(row.get("label")) is int and row["label"] not in seen,
                "Invalid or duplicate tested label")
        seen.add(row["label"])
        require(type(row.get("queries")) is int and row["queries"] > 0
                and type(row.get("rust_l")) is int and row["rust_l"] >= 64,
                "Invalid sample count or actual Rust L")
        interval = row.get(INTERVAL)
        require(type(interval) is list and len(interval) == 2
                and all(type(value) in (int, float) and math.isfinite(value) for value in interval)
                and interval[0] <= interval[1]
                and type(row.get("difference_percentage_points")) in (int, float)
                and math.isfinite(row["difference_percentage_points"]),
                "Invalid recall difference or simultaneous interval")
    coverage = summary.get("coverage", {})
    require(coverage.get("tested_labels") == len(labels)
            and coverage.get("unique_heldout_queries") == sum(row["queries"] for row in labels)
            and type(coverage.get("frozen_plan_labels")) is int
            and coverage["frozen_plan_labels"] >= len(labels), "Summary coverage differs from plotted rows")
    uncertainty = summary.get("uncertainty", {})
    require(uncertainty.get("simultaneous_family_labels") == len(labels)
            and uncertainty.get("simultaneous_method")
            == "Bonferroni-adjusted percentile bootstrap, family of tested labels only",
            "Unexpected simultaneous interval method or tested-label family")
    return summary


def plot(summary_path, output):
    summary = load_summary(summary_path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    groups = [[row for row in summary["labels"] if row["queries"] >= 100],
              [row for row in summary["labels"] if row["queries"] < 100]]
    groups = [sorted(rows, key=lambda row: (row["difference_percentage_points"], row["label"]))
              for rows in groups]
    require(all(groups), "Expected both at-least-100 and sparse-label panels")
    coverage = summary["coverage"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.titleweight": "bold",
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    height = max(10.0, .29 * len(summary["labels"]) + 3.2)
    figure, axes = plt.subplots(2, 1, figsize=(12.4, height),
                                gridspec_kw={"height_ratios": [len(rows) + 1.5 for rows in groups]})
    figure.subplots_adjust(left=.235, right=.88, top=.895, bottom=.105, hspace=.29)
    colors = {"higher": "#126782", "lower": "#C64A3A", "equal": "#59636A"}
    titles = ("At least 100 held-out queries per label", "Sparse labels: fewer than 100 held-out queries")
    for axis, rows, title in zip(axes, groups, titles):
        extent = max(abs(value) for row in rows
                     for value in (*row[INTERVAL], row["difference_percentage_points"]))
        extent = max(extent * 1.17, .1)
        axis.set_xlim(-extent, extent)
        axis.set_ylim(len(rows) - .35, -.8)
        axis.set_facecolor("#FAFBFC")
        axis.axvspan(-extent, 0, color="#C64A3A", alpha=.035, zorder=0)
        axis.axvline(0, color="#68737D", linewidth=.9, linestyle=(0, (3, 3)), zorder=1)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=7))
        axis.grid(axis="x", color="#DDE3E8", linewidth=.65)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for position, row in enumerate(rows):
            difference = row["difference_percentage_points"]
            low, high = row[INTERVAL]
            color = colors["higher" if difference > 0 else "lower" if difference < 0 else "equal"]
            axis.hlines(position, low, high, color=color, linewidth=1.45, zorder=2)
            axis.plot([low, high], [position, position], linestyle="none", marker="|",
                      markersize=6, markeredgewidth=1.2, color=color, zorder=2)
            axis.scatter([difference], [position], color=color, s=27, edgecolors="white",
                         linewidths=.6, zorder=3)
            axis.text(1.025, position, f"{difference:+.3f}", va="center", ha="left",
                      transform=axis.get_yaxis_transform(), fontsize=9, color=color)
        axis.set_yticks(range(len(rows)))
        axis.set_yticklabels([f"{row['label']:>5}   n={row['queries']:>5,}   L={row['rust_l']}"
                              for row in rows], fontfamily="DejaVu Sans Mono", fontsize=9)
        axis.tick_params(axis="y", length=0, pad=8)
        axis.tick_params(axis="x", length=0, labelsize=9, pad=5)
        axis.set_title(f"{title} ({len(rows)} labels)", loc="left", pad=14)
        axis.set_xlabel("Rust − C++ recall@10 difference (percentage points)", labelpad=7)
        axis.text(1.025, 1.025, "Observed Δ", transform=axis.transAxes,
                  fontsize=9, color="#59636A", ha="left")
    figure.suptitle("Frozen per-label L policy on held-out queries", x=.055, y=.984,
                   ha="left", fontsize=19, fontweight="bold", color="#23313C")
    figure.text(.055, .955,
                f"{coverage['unique_heldout_queries']:,} untouched natural queries · "
                f"{coverage['tested_labels']}/{coverage['frozen_plan_labels']} labels · "
                "Rust-built index vs C++ L64 · unchanged policy",
                fontsize=11, color="#59636A")
    handles = [Line2D([0], [0], marker="o", linestyle="none", color=colors[key],
                      markersize=5, label=label)
               for key, label in (("higher", "Rust higher"), ("lower", "Rust lower"),
                                  ("equal", "Equal observed recall"))]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(.05, .943),
                  frameon=False, ncol=3, handletextpad=.4, columnspacing=1.8, fontsize=9)
    missing = coverage["frozen_plan_labels"] - coverage["tested_labels"]
    figure.text(.055, .066,
                f"Bars: nominal simultaneous 95% Bonferroni percentile bootstrap intervals over {len(summary['labels'])} tested labels.\n"
                "A constant observed difference can produce a zero-width interval; this does not guarantee future-query equality.\n"
                f"{missing} frozen labels have no untouched public queries. Panel scales differ; sparse-label estimates are uncertain.",
                fontsize=9, color="#59636A", linespacing=1.5, va="top")
    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    figure.text(.055, .009, f"Source: {summary_path.name} · SHA-256 {digest[:16]}…",
                fontsize=8, color="#87939D")
    prefix = output.with_suffix("") if output.suffix.lower() in (".png", ".svg") else output
    targets = [prefix.with_suffix(".png"), prefix.with_suffix(".svg")]
    require(all(path.resolve() != summary_path.resolve() for path in targets),
            "Figure output must not overwrite the source summary")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for path in targets:
        figure.savefig(path, dpi=190,
                       metadata={"Title": "Frozen per-label L held-out recall validation",
                                 "Description": f"Source summary SHA-256: {digest}",
                                 "Creator": "DiskANN plot_heldout.py"})
    plt.close(figure)
    print(json.dumps({"summary": str(summary_path), "summary_sha256": digest,
                      "tested_labels": len(summary["labels"]),
                      "panels": {"at_least_100_queries": len(groups[0]), "fewer_than_100_queries": len(groups[1])},
                      "outputs": [str(path) for path in targets]}, indent=2))
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output filename prefix, or PNG/SVG path")
    args = parser.parse_args()
    plot(args.summary, args.out)


if __name__ == "__main__":
    main()
