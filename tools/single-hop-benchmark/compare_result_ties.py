#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Compare retained strict-filter results using exact integer distances.

Strict ID recall remains the primary metric. The additional distance-cutoff
metric credits any distinct result at or below the exact Kth truth distance,
so alternative equal-distance IDs do not look like traversal misses.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import mmap
from pathlib import Path
import struct


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(path, kind):
    raw = path.read_bytes()
    rows, columns = struct.unpack_from("<II", raw)
    size = rows * columns
    if len(raw) < 8 + 4 * size:
        raise ValueError(f"Truncated matrix: {path}")
    values = struct.unpack_from(f"<{size}{kind}", raw, 8)
    return rows, columns, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cpp-results", nargs="+", type=Path, required=True)
    parser.add_argument("--rust-dump", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", type=Path, help="Optional per-query differences, normally in ignored artifacts")
    args = parser.parse_args()

    truth_path = args.prepared / "groundtruth.bin"
    n, k, flat_truth = matrix(truth_path, "I")
    raw_truth = truth_path.read_bytes()
    truth_distances = struct.unpack_from(f"<{n * k}f", raw_truth, 8 + n * k * 4)
    query_raw = (args.prepared / "query.u8bin").read_bytes()
    query_n, dimensions = struct.unpack_from("<II", query_raw)
    if query_n != n or len(query_raw) != 8 + n * dimensions:
        raise ValueError("Query dimensions do not match the truth matrix")
    labels = [int(line) for line in (args.prepared / "query_labels.txt").read_text().splitlines()]
    if len(labels) != n:
        raise ValueError("Query label count does not match the truth matrix")
    truth = [set(flat_truth[i * k:(i + 1) * k]) for i in range(n)]
    cutoffs = [truth_distances[(i + 1) * k - 1] for i in range(n)]
    cpp = {}
    cpp_paths = {}
    for path in args.cpp_results:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["threads"] != 8 or row["repeat"] != 0:
                continue
            result_path = Path(row.get("local_result_ids", row["result_ids"]))
            result_n, result_k, ids = matrix(result_path, "I")
            if (result_n, result_k) != (n, k):
                raise ValueError(f"C++ dimensions differ: {result_path}")
            distance_path = Path(str(result_path).replace("_idx_uint32.bin", "_dists_float.bin"))
            distance_n, distance_k, distances = matrix(distance_path, "f")
            if (distance_n, distance_k) != (n, k):
                raise ValueError(f"C++ distance dimensions differ: {distance_path}")
            cpp[row["l"]] = [(ids[i * k:(i + 1) * k], distances[i * k:(i + 1) * k]) for i in range(n)]
            cpp_paths[row["l"]] = {"ids": str(result_path), "ids_sha256": sha256(result_path),
                                    "distances": str(distance_path), "distances_sha256": sha256(distance_path)}

    rust = defaultdict(dict)
    for path in args.rust_dump:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row["mode"], row["l"], row.get("repeat", 0))
            query = row["query"]
            if query in rust[key] or not 0 <= query < n or row["label"] != labels[query]:
                raise ValueError("Repeated or mismatched Rust query")
            if len(row["ids"]) != row["returned"] or len(row["distances"]) != row["returned"]:
                raise ValueError("Rust returned count differs from result arrays")
            rust[key][query] = row

    with (args.prepared / "base.u8bin").open("rb") as source:
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as vectors:
            base_n, base_dimensions = struct.unpack_from("<II", vectors)
            if dimensions != base_dimensions or len(vectors) != 8 + base_n * dimensions:
                raise ValueError("Base/query dimensions differ")
            distance_cache = {}

            def exact(query, vertex):
                key = (query, vertex)
                if key not in distance_cache:
                    if not 0 <= vertex < base_n:
                        raise ValueError(f"Invalid result ID: {vertex}")
                    qoffset = 8 + query * dimensions
                    boffset = 8 + vertex * dimensions
                    distance_cache[key] = sum((a - b) ** 2 for a, b in zip(
                        query_raw[qoffset:qoffset + dimensions], vectors[boffset:boffset + dimensions]))
                return distance_cache[key]

            for i in range(n):
                for j in range(k):
                    if exact(i, flat_truth[i * k + j]) != truth_distances[i * k + j]:
                        raise ValueError("Saved ground-truth distance does not equal exact integer squared L2")

            summaries = []
            differences = []
            for (mode, l, repeat), queries in sorted(rust.items()):
                if len(queries) != n or l not in cpp:
                    raise ValueError(f"Incomplete Rust/C++ query comparison: {mode}, L={l}, repeat={repeat}")
                counts = defaultdict(int)
                by_label = defaultdict(lambda: defaultdict(int))
                for i in range(n):
                    row = queries[i]
                    rust_ids, cpp_ids = row["ids"], cpp[l][i][0]
                    if len(set(rust_ids)) != len(rust_ids) or len(set(cpp_ids)) != len(cpp_ids):
                        raise ValueError("Duplicate result ID")
                    rdist = [exact(i, vertex) for vertex in rust_ids]
                    cdist = [exact(i, vertex) for vertex in cpp_ids]
                    if any(vertex not in truth[i] and distance < cutoffs[i] for vertex, distance in
                           zip(list(rust_ids) + list(cpp_ids), rdist + cdist)):
                        raise ValueError("A non-truth result is strictly closer than the exact Kth truth distance")
                    rstrict = len(set(rust_ids) & truth[i])
                    cstrict = len(set(cpp_ids) & truth[i])
                    rtie = sum(distance <= cutoffs[i] for distance in rdist)
                    ctie = sum(distance <= cutoffs[i] for distance in cdist)
                    delta = {"queries": 1, "rust_strict_hits": rstrict, "cpp_strict_hits": cstrict,
                             "rust_cutoff_hits": rtie, "cpp_cutoff_hits": ctie,
                             "rust_alternative_cutoff_ties": rtie - rstrict,
                             "cpp_alternative_cutoff_ties": ctie - cstrict,
                             "ordered_results_differ": int(tuple(rust_ids) != tuple(cpp_ids)),
                             "result_sets_differ": int(set(rust_ids) != set(cpp_ids)),
                             "rust_strict_wins": int(rstrict > cstrict), "cpp_strict_wins": int(cstrict > rstrict),
                             "rust_cutoff_wins": int(rtie > ctie), "cpp_cutoff_wins": int(ctie > rtie),
                             "rust_saved_distance_mismatches": sum(a != b for a, b in zip(rdist, row["distances"])),
                             "cpp_saved_distance_mismatches": sum(a != b for a, b in zip(cdist, cpp[l][i][1]))}
                    for name, value in delta.items():
                        counts[name] += value
                        by_label[labels[i]][name] += value
                    if delta["result_sets_differ"]:
                        differences.append({"mode": mode, "l": l, "repeat": repeat, "query": i,
                                            "label": labels[i], "cutoff": cutoffs[i], **delta,
                                            "rust_ids": rust_ids, "cpp_ids": cpp_ids,
                                            "rust_distances": rdist, "cpp_distances": cdist,
                                            "missing_truth_rust": sorted(truth[i] - set(rust_ids)),
                                            "missing_truth_cpp": sorted(truth[i] - set(cpp_ids))})
                def finish(values):
                    denominator = values["queries"] * k
                    return {**values, "rust_strict_recall": values["rust_strict_hits"] / denominator,
                            "cpp_strict_recall": values["cpp_strict_hits"] / denominator,
                            "rust_cutoff_recall": values["rust_cutoff_hits"] / denominator,
                            "cpp_cutoff_recall": values["cpp_cutoff_hits"] / denominator}
                summaries.append({"mode": mode, "l": l, "repeat": repeat, **finish(counts),
                                  "labels": [{"label": label, **finish(values)} for label, values in sorted(by_label.items())]})

    output = {"metric": "strict_id_recall_primary_with_exact_distance_cutoff_diagnostic",
              "cutoff_metric": "distinct returned IDs with exact squared L2 <= saved exact Kth truth distance, divided by K",
              "exact_distance": "Python integer squared L2 recomputed from uint8 base/query vectors",
              "queries": n, "k": k, "dimensions": dimensions,
              "cpp_inputs": cpp_paths,
              "rust_inputs": [{"path": str(path), "sha256": sha256(path)} for path in args.rust_dump],
              "truth": {"path": str(truth_path), "sha256": sha256(truth_path)},
              "groups": summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if args.details:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        args.details.write_text("".join(json.dumps(row) + "\n" for row in differences))
    for group in summaries:
        print(json.dumps({key: value for key, value in group.items() if key != "labels"}))


if __name__ == "__main__":
    main()
