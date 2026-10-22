#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Summarize only quality from graph/PQ pairing diagnostics, with no timing fields."""

import argparse
import hashlib
import json
import math
from pathlib import Path


COMMON = ("implementation", "prepared_identity", "queries", "n_base", "dimensions", "k", "threads",
          "mode", "rerank_candidates", "starts", "beam", "storage", "cache_bypass", "shared_cache_nodes")
VALIDITY = ("filter_violations", "duplicate_results", "short_results")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def provenance(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def count_hits(recall, denominator):
    require(type(recall) in (int, float) and math.isfinite(recall) and 0 <= recall <= 1, "Invalid strict recall")
    hits = round(recall * denominator)
    require(abs(recall * denominator - hits) < 1e-7, "Recall is not an integer strict-ID hit count")
    return hits


def load(path, graph, pq):
    raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(len(raw) == 3 and {row["l"] for row in raw} == {60, 64, 144}, "Expected one observation at each L60/64/144")
    first = raw[0]
    require(first["implementation"] == "rust" and first["mode"] == "single"
            and first["rerank_candidates"] == "expanded" and first["starts"] == "label", "Expected strict expanded Rust search")
    require(first["header"] == ("cpp" if graph == "cpp" else "native"), "Unexpected graph header")
    require(first["cache_bypass"] is True and first["shared_cache_nodes"] == 0, "Unexpected cache protocol")
    rows = {}
    for row in raw:
        require(all(row.get(key) == first.get(key) for key in COMMON), "Within-cell workload/protocol differs")
        require(all(row.get(key) == first.get(key) for key in
                    ("index_prefix", "pq_prefix", "header", "label_metadata", "label_metadata_identity", "entry_prefix", "entry_source")),
                "Within-cell graph/PQ/entry identity differs")
        require(all(type(row.get(key)) is int and row[key] == 0 for key in VALIDITY), "Invalid, duplicate or short results")
        require(type(row.get("repeat")) is int and row["repeat"] == 0, "Unexpected repeated diagnostic observation")
        require(type(row["queries"]) is int and row["queries"] > 0 and type(row["k"]) is int and row["k"] > 0, "Invalid query shape")
        hits = count_hits(row["recall"], row["queries"] * row["k"])
        labels = []
        for item in row["labels"]:
            require(type(item["n"]) is int and item["n"] > 0 and type(item["eligible"]) is int and item["eligible"] > 0,
                    "Invalid per-label query/membership count")
            correct = count_hits(item["recall"], item["n"] * row["k"])
            labels.append({"label": item["label"], "queries": item["n"], "eligible": item["eligible"],
                           "strict_id_hits": correct, "strict_id_recall": correct / (item["n"] * row["k"])})
        require(len({item["label"] for item in labels}) == len(labels)
                and sum(item["queries"] for item in labels) == row["queries"]
                and sum(item["strict_id_hits"] for item in labels) == hits, "Per-label quality evidence is incomplete")
        rows[row["l"]] = {"graph": graph, "pq": pq, "l": row["l"], "queries": row["queries"], "k": row["k"],
                          "strict_id_hits": hits, "strict_id_recall": hits / (row["queries"] * row["k"]),
                          "labels": sorted(labels, key=lambda item: item["label"])}
    identity = {key: first.get(key) for key in
                ("index_prefix", "pq_prefix", "header", "label_metadata", "label_metadata_identity", "entry_prefix", "entry_source")}
    return {"source": provenance(path), "identity": identity, "protocol": {key: first[key] for key in COMMON}, "rows": rows}


def difference(before, after, name):
    require((before["l"], before["k"], before["queries"]) == (after["l"], after["k"], after["queries"]), "Compared quality shapes differ")
    previous = {item["label"]: item for item in before["labels"]}
    require(previous.keys() == {item["label"] for item in after["labels"]}, "Compared label sets differ")
    labels = []
    for item in after["labels"]:
        old = previous[item["label"]]
        require((old["queries"], old["eligible"]) == (item["queries"], item["eligible"]), "Compared label workloads differ")
        labels.append({"label": item["label"], "queries": item["queries"], "eligible": item["eligible"],
                       "before_strict_id_recall": old["strict_id_recall"], "after_strict_id_recall": item["strict_id_recall"],
                       "hit_difference": item["strict_id_hits"] - old["strict_id_hits"],
                       "difference_percentage_points": 100 * (item["strict_id_recall"] - old["strict_id_recall"])})
    return {"contrast": name, "l": before["l"], "before": {"graph": before["graph"], "pq": before["pq"]},
            "after": {"graph": after["graph"], "pq": after["pq"]},
            "before_strict_id_recall": before["strict_id_recall"], "after_strict_id_recall": after["strict_id_recall"],
            "hit_difference": after["strict_id_hits"] - before["strict_id_hits"],
            "difference_percentage_points": 100 * (after["strict_id_recall"] - before["strict_id_recall"]),
            "labels_with_lower_recall": sum(item["hit_difference"] < 0 for item in labels),
            "labels_with_higher_recall": sum(item["hit_difference"] > 0 for item in labels),
            "minimum_label_difference": min(labels, key=lambda item: item["hit_difference"] / item["queries"]),
            "maximum_label_difference": max(labels, key=lambda item: item["hit_difference"] / item["queries"]), "labels": labels}


def summarize(directory, entry_cell=None):
    cells = {(graph, pq): load(directory / f"pairing-{graph}-{pq}.jsonl", graph, pq)
             for graph in ("cpp", "rust") for pq in ("cpp", "rust")}
    reference = cells["cpp", "cpp"]
    require(all(cell["protocol"] == reference["protocol"] for cell in cells.values()), "2x2 workload/protocol differs")
    cpp_prefix = reference["identity"]["index_prefix"]
    native_prefix = cells["rust", "rust"]["identity"]["index_prefix"]
    require(cpp_prefix != native_prefix, "Graph identities are not distinct")
    for (graph, pq), cell in cells.items():
        require(cell["identity"]["index_prefix"] == (cpp_prefix if graph == "cpp" else native_prefix)
                and cell["identity"]["pq_prefix"] == (cpp_prefix if pq == "cpp" else native_prefix), "2x2 graph/PQ identity mismatch")
        require(cell["identity"]["entry_prefix"] is None, "Base2x2 already contains entry overrides")
    require(cells["rust", "cpp"]["identity"]["label_metadata_identity"] == cells["rust", "rust"]["identity"]["label_metadata_identity"]
            and cells["rust", "cpp"]["identity"]["label_metadata_identity"] is not None, "Native label metadata differs between PQ cells")
    contrasts = []
    interactions = []
    for l in (60, 64, 144):
        cc, cr, rc, rr = (cells[key]["rows"][l] for key in (("cpp", "cpp"), ("cpp", "rust"), ("rust", "cpp"), ("rust", "rust")))
        specifications = [(cc, cr, "native_PQ_minus_CPP_PQ_on_CPP_graph_and_entries"),
                          (rc, rr, "native_PQ_minus_CPP_PQ_on_native_graph_and_entries"),
                          (cc, rc, "native_graph_and_entries_minus_CPP_graph_and_entries_with_CPP_PQ"),
                          (cr, rr, "native_graph_and_entries_minus_CPP_graph_and_entries_with_native_PQ"),
                          (cc, rr, "complete_native_artifacts_minus_reference_artifacts")]
        contrasts.extend(difference(before, after, name) for before, after, name in specifications)
        interactions.append({"l": l, "difference_of_differences_percentage_points": 100 *
            ((rr["strict_id_recall"] - rc["strict_id_recall"]) - (cr["strict_id_recall"] - cc["strict_id_recall"])),
            "meaning": "PQ effect depends on the graph/entry pair; effects are not independently additive"})
    output = {"status": "all quality assertions passed", "scope": "quality-only graph/PQ artifact diagnostic; every timing and work-count metric is deliberately excluded",
              "protocol": reference["protocol"], "unique_queries": reference["protocol"]["queries"],
              "quality_observations_per_cell_and_L": 1, "note_on_samples": "the same query sample is reused in every cell and breadth; cells do not add independent queries",
              "cells": [{"graph": graph, "pq": pq, "source": cell["source"], "identity": cell["identity"],
                         "quality": [cell["rows"][l] for l in (60, 64, 144)]} for (graph, pq), cell in cells.items()],
              "contrasts": contrasts, "interaction": interactions,
              "interpretation": ["Changing PQ while holding graph and entry points fixed isolates the effect of these trained PQ artifacts on this query sample.",
                  "Changing graph families in the base2x2 also changes entry points; call this a graph-and-entry effect, not a pure graph-builder effect.",
                  "The codebooks differ in training data/sampling and construction; this does not establish that a language or quantizer implementation is inherently worse.",
                  "Strict-ID recall can include equal-distance ID choices. No independent result-distance tie analysis was performed for these cells.",
                  "Per-label differences use small query samples and do not establish statistical significance or a production segment SLA.",
                  "No throughput or latency claim follows from these single-observation diagnostic runs."]}
    if entry_cell is not None:
        extra = load(entry_cell, "rust", "cpp")
        require(extra["protocol"] == reference["protocol"] and extra["identity"]["index_prefix"] == native_prefix
                and extra["identity"]["pq_prefix"] == cpp_prefix and extra["identity"]["entry_prefix"] == cpp_prefix
                and extra["identity"]["entry_source"] == "cpp_sidecar_override"
                and extra["identity"]["label_metadata_identity"] == cells["rust", "cpp"]["identity"]["label_metadata_identity"],
                "Entry-only intervention changes other artifacts")
        output["CPP_entry_override_on_native_graph_with_CPP_PQ"] = {"source": extra["source"], "identity": extra["identity"],
            "quality": [extra["rows"][l] for l in (60, 64, 144)],
            "entry_only_effect": [difference(cells["rust", "cpp"]["rows"][l], extra["rows"][l], "CPP_entries_minus_native_initial_entries") for l in (60, 64, 144)],
            "graph_only_effect_with_CPP_PQ_and_CPP_entries": [difference(cells["cpp", "cpp"]["rows"][l], extra["rows"][l], "native_graph_minus_CPP_graph_with_CPP_PQ_and_entries") for l in (60, 64, 144)]}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    directory = Path(__file__).resolve().parents[2] / "benchmark-results/single-hop/migration/results"
    parser.add_argument("--directory", type=Path, default=directory)
    parser.add_argument("--entry-cell", type=Path)
    parser.add_argument("--output", type=Path, default=directory / "pairing-summary.json")
    args = parser.parse_args()
    result = summarize(args.directory, args.entry_cell)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for cell in result["cells"]:
        print(json.dumps({"graph": cell["graph"], "pq": cell["pq"],
                          "strict_recall_by_L": {row["l"]: row["strict_id_recall"] for row in cell["quality"]}}))
    print(json.dumps({"output": str(args.output), "status": result["status"], "timing_fields_included": False}))


if __name__ == "__main__":
    main()
