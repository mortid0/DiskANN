#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Prepare untouched natural one-tag queries for a frozen label-to-L policy.

Uses only the Python standard library. The complete remaining eligible query pool
is retained after excluding tuning IDs and every embedding seen during tuning.
Published full-base truth is copied by original row, then every truth membership
and integer squared distance is checked independently. Existing data is untouched.
"""

import argparse
from array import array
from bisect import bisect_left
from collections import Counter
import copy
import hashlib
import json
import mmap
import os
from pathlib import Path
import random
import struct
import sys
import time


SEED = 20261022


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def read_matrix(path, width):
    blob = path.read_bytes()
    require(len(blob) >= 8, f"Truncated matrix: {path}")
    rows, columns = struct.unpack_from("<II", blob)
    require(len(blob) == 8 + rows * columns * width, f"Malformed matrix: {path}")
    return rows, columns, blob[8:]


def u32_values(path):
    rows, columns, blob = read_matrix(path, 4)
    require(columns == 1, f"Expected one-column ID matrix: {path}")
    values = array("I", blob)
    require(values.itemsize == 4, "Python unsigned-int arrays must use four bytes")
    if sys.byteorder != "little":
        values.byteswap()
    return rows, values


def sparse_labels(path):
    blob = path.read_bytes()
    require(len(blob) >= 24, f"Truncated metadata: {path}")
    rows, columns, nnz = struct.unpack_from("<qqq", blob)
    require(rows > 0 and columns > 0 and nnz >= 0
            and len(blob) == 24 + 8 * (rows + 1) + 8 * nnz,
            f"Malformed sparse metadata: {path}")
    pointers = struct.unpack_from(f"<{rows + 1}q", blob, 24)
    indices = struct.unpack_from(f"<{nnz}i", blob, 24 + 8 * (rows + 1))
    require(pointers[0] == 0 and pointers[-1] == nnz
            and all(0 <= a <= b <= nnz for a, b in zip(pointers, pointers[1:]))
            and all(0 <= label < columns for label in indices),
            f"Invalid sparse offsets/labels: {path}")
    return rows, columns, pointers, indices


def write_matrix(path, rows, columns, content):
    with path.open("xb") as output:
        output.write(struct.pack("<II", rows, columns))
        output.write(content)


def canonical_plan(path):
    def unique(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, "Duplicate label key in frozen policy")
            output[key] = value
        return output

    raw = json.loads(path.read_text(), object_pairs_hook=unique)
    require(type(raw) is dict and len(raw) == 64, "Frozen policy must contain exactly 64 labels")
    require(all(type(key) is str and key.isdecimal() and str(int(key)) == key
                and 0 <= int(key) < 2**32 and type(value) is int and value >= 64
                for key, value in raw.items()), "Invalid frozen policy entry")
    return {int(label): value for label, value in raw.items()}


def prepare(data, tuning, policy, output):
    started = time.monotonic()
    require(not output.exists(), f"Refusing to overwrite existing output: {output}")
    plan = canonical_plan(policy)
    tuning_manifest_path = tuning / "manifest.json"
    original = json.loads(tuning_manifest_path.read_text())
    records = {item["original_label"]: item for item in original["labels"]}
    require(set(records) == set(plan), "Frozen policy and existing prepared label sets differ")
    require(original["base_vectors"] == 10_000_000 and original["dimension"] == 192
            and original["k"] == 10 and original["metric"] == "L2"
            and original["dtype"] == "uint8", "Expected the frozen full YFCC10M workload")
    sources = {item["file"]: item for item in original["sources"]}
    verified_sources = []
    names = ("query.public.100K.u8bin", "query.metadata.public.100K.spmat", "GT.public.ibin")
    for name in names:
        actual = provenance(data / name)
        require(actual["bytes"] == sources[name]["bytes"]
                and actual["sha256"] == sources[name]["sha256"], f"Source identity mismatch: {name}")
        verified_sources.append(actual)
    for name in ("query.u8bin", "query_labels.u32bin", "query_ids.u32bin", "groundtruth.bin"):
        actual = provenance(tuning / name)
        require(actual["bytes"] == original["artifacts"][name]["bytes"]
                and actual["sha256"] == original["artifacts"][name]["sha256"],
                f"Tuning artifact identity mismatch: {name}")

    query_count, dimensions, vectors = read_matrix(data / names[0], 1)
    metadata_count, _, pointers, labels = sparse_labels(data / names[1])
    require(query_count == metadata_count == 100_000 and dimensions == 192,
            "Source query dimensions or row counts differ")
    tuning_count, tuning_ids = u32_values(tuning / "query_ids.u32bin")
    tuning_label_count, tuning_labels = u32_values(tuning / "query_labels.u32bin")
    tuning_vector_count, tuning_dimensions, tuning_vectors = read_matrix(tuning / "query.u8bin", 1)
    require(tuning_count == tuning_label_count == tuning_vector_count == original["query_count"] == 2000
            and tuning_dimensions == dimensions and len(set(tuning_ids)) == tuning_count,
            "Invalid tuning query count, shape or duplicate IDs")
    tuning_id_set = set(tuning_ids)
    tuning_embeddings = set()
    for position, (query_id, label) in enumerate(zip(tuning_ids, tuning_labels)):
        require(query_id < query_count and pointers[query_id + 1] - pointers[query_id] == 1
                and labels[pointers[query_id]] == label and label in plan,
                "Tuning IDs disagree with original natural predicates")
        embedding = vectors[query_id * dimensions:(query_id + 1) * dimensions]
        require(embedding == tuning_vectors[position * dimensions:(position + 1) * dimensions],
                "Tuning IDs disagree with original embeddings")
        tuning_embeddings.add(embedding)

    selected = []
    excluded_ids = []
    excluded_embeddings = []
    observed_embeddings = set()
    available = Counter()
    single_count = 0
    for query_id in range(query_count):
        if pointers[query_id + 1] - pointers[query_id] != 1:
            continue
        single_count += 1
        label = labels[pointers[query_id]]
        if label not in plan:
            continue
        available[label] += 1
        if query_id in tuning_id_set:
            excluded_ids.append(query_id)
            continue
        embedding = vectors[query_id * dimensions:(query_id + 1) * dimensions]
        if embedding in tuning_embeddings:
            excluded_embeddings.append(query_id)
            continue
        require(embedding not in observed_embeddings,
                f"Duplicate held-out embedding (including across labels): original query {query_id}")
        observed_embeddings.add(embedding)
        selected.append((query_id, label))
    require(len(excluded_ids) == tuning_count, "Not every tuning ID was excluded from the eligible pool")
    require(selected, "No untouched eligible queries remain")
    random.Random(SEED).shuffle(selected)
    counts = Counter(label for _, label in selected)
    require(sum(available.values()) == len(selected) + len(excluded_ids) + len(excluded_embeddings),
            "Eligible query accounting differs")

    truth_blob = (data / names[2]).read_bytes()
    truth_queries, k = struct.unpack_from("<II", truth_blob)
    require((truth_queries, k) == (query_count, 10)
            and len(truth_blob) == 8 + query_count * k * 8, "Malformed full-base ground truth")
    selected_vectors = bytearray()
    selected_ids = bytearray()
    selected_labels = bytearray()
    truth_ids = bytearray()
    truth_distances = bytearray()
    postings = {}
    posting_artifacts = {}
    for label in sorted(counts):
        path = tuning / f"label_{label}.u32bin"
        count, values = u32_values(path)
        require(count == records[label]["base_count"]
                and all(a < b for a, b in zip(values, values[1:]))
                and values[0] < 10_000_000 and values[-1] < 10_000_000,
                f"Posting list differs from existing index: {label}")
        postings[label] = values
        posting_artifacts[str(label)] = provenance(path)
    base_path = tuning / "base.u8bin"
    require(base_path.stat().st_size == 8 + 10_000_000 * dimensions, "Unexpected full-base size")
    with base_path.open("rb") as base_file, mmap.mmap(base_file.fileno(), 0, access=mmap.ACCESS_READ) as base:
        require(struct.unpack_from("<II", base) == (10_000_000, dimensions), "Unexpected full-base shape")
        for query_id, label in selected:
            query = vectors[query_id * dimensions:(query_id + 1) * dimensions]
            selected_vectors.extend(query)
            selected_ids.extend(struct.pack("<I", query_id))
            selected_labels.extend(struct.pack("<I", label))
            id_start = 8 + query_id * k * 4
            distance_start = 8 + query_count * k * 4 + query_id * k * 4
            id_blob = truth_blob[id_start:id_start + k * 4]
            distance_blob = truth_blob[distance_start:distance_start + k * 4]
            ids = struct.unpack("<10i", id_blob)
            distances = struct.unpack("<10f", distance_blob)
            require(len(set(ids)) == k and all(a <= b for a, b in zip(distances, distances[1:])),
                    f"Duplicate truth IDs or unsorted truth distances: query {query_id}")
            for point_id, expected in zip(ids, distances):
                position = bisect_left(postings[label], point_id)
                require(position < len(postings[label]) and postings[label][position] == point_id,
                        f"Truth violates predicate: query {query_id}, label {label}, point {point_id}")
                point = base[8 + point_id * dimensions:8 + (point_id + 1) * dimensions]
                distance = sum((a - b) * (a - b) for a, b in zip(query, point))
                require(distance == expected,
                        f"Truth distance mismatch: query {query_id}, point {point_id}: {distance} != {expected}")
            truth_ids.extend(id_blob)
            truth_distances.extend(distance_blob)

    output.mkdir(parents=True, exist_ok=False)
    # Relative targets survive transfer when both prepared directories are siblings.
    for name in ("base.u8bin", "labels.txt", *(f"label_{label}.u32bin" for label in sorted(plan))):
        (output / name).symlink_to(os.path.relpath(tuning / name, output))
    count = len(selected)
    write_matrix(output / "query.u8bin", count, dimensions, selected_vectors)
    write_matrix(output / "query_ids.u32bin", count, 1, selected_ids)
    write_matrix(output / "query_labels.u32bin", count, 1, selected_labels)
    write_matrix(output / "groundtruth.bin", count, k, truth_ids + truth_distances)
    (output / "query_labels.txt").write_text("".join(f"{label}\n" for _, label in selected))

    missing = sorted(set(plan) - set(counts))
    holdout = {"tuning_query_count": tuning_count, "tuning_unique_embeddings": len(tuning_embeddings),
               "eligible_natural_one_tag_queries_before_exclusions": sum(available.values()),
               "excluded_tuning_ids": len(excluded_ids),
               "additional_tuning_embedding_exclusions_cross_label": len(excluded_embeddings),
               "heldout_query_count": count, "heldout_unique_embeddings": len(observed_embeddings),
               "duplicate_heldout_embeddings_including_cross_label": 0,
               "covered_labels": sorted(counts), "missing_labels": missing,
               "frozen_label_count": len(plan), "covered_label_count": len(counts),
               "query_weighted_mean_l": sum(counts[label] * plan[label] for label in counts) / count,
               "covered_label_macro_mean_l": sum(plan[label] for label in counts) / len(counts)}
    evidence = {"schema_version": 1, "scope": "complete_untouched_natural_one_tag_public_query_pool",
                "seed": SEED, "tuning_prepared_manifest": provenance(tuning_manifest_path),
                "tuning_query_ids": provenance(tuning / "query_ids.u32bin"),
                "policy": provenance(policy), "verified_raw_sources": verified_sources,
                "posting_artifacts": posting_artifacts, "holdout": holdout,
                "excluded_original_query_ids": sorted(excluded_ids),
                "additional_original_query_ids_excluded_for_tuning_embeddings": excluded_embeddings,
                "checks": {"tuning_ids_vectors_and_natural_predicates_reconstructed_from_raw_sources": True,
                           "zero_tuning_id_intersection": True,
                           "zero_tuning_embedding_intersection_including_cross_label": True,
                           "zero_heldout_duplicate_embeddings_including_cross_label": True,
                           "all_published_truth_ids_eligible": True,
                           "all_published_truth_distances_equal_independent_integer_squared_l2": True,
                           "individually_checked_truth_pairs": count * k,
                           "nearest_neighbor_exhaustive_search_recomputed": False}}
    evidence_path = output / "exclusion-evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    manifest = copy.deepcopy(original)
    manifest.update({"seed": SEED, "query_count": count,
                     "source_prepared_dataset": tuning.name,
                     "groundtruth": "Published full-10M exact filtered top-10, untouched selected query rows; every ID membership and integer squared-L2 distance independently verified",
                     "query_selection": "All original one-tag queries in the frozen label set after excluding tuning IDs and any tuning embedding across labels; Python Random(20261022) shuffle",
                     "validation_scope": "held_out_natural_single_tag_public_queries_frozen_label_policy",
                     "frozen_label_l_policy": {**provenance(policy), "entries": [{"label": label, "l": plan[label]} for label in sorted(plan)]},
                     "holdout": holdout, "original_one_tag_queries": single_count,
                     "exclusion_evidence": {"file": evidence_path.name, **provenance(evidence_path)}})
    for record in manifest["labels"]:
        label = record["original_label"]
        require(record["available_one_tag_queries"] == available[label], "Natural label availability differs")
        record.update({"selected_queries": counts[label], "frozen_l": plan[label],
                       "heldout_coverage": "observed" if counts[label] else "untestable_public_queries_exhausted_by_tuning"})
    manifest["artifacts"] = {name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
                             for name in ("query.u8bin", "query_labels.u32bin", "query_ids.u32bin",
                                          "groundtruth.bin", "labels.txt", "query_labels.txt")}
    (output / "label_counts.json").write_text(json.dumps(manifest["labels"], indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"prepared": str(output), "query_count": count, "covered_labels": len(counts),
                      "missing_labels": missing, "prepared_identity": ":".join(manifest["artifacts"][name]["sha256"] for name in ("query.u8bin", "query_labels.u32bin", "groundtruth.bin", "labels.txt")),
                      "manifest_sha256": sha256(output / "manifest.json"),
                      "seconds": time.monotonic() - started}, indent=2), flush=True)
    return manifest, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark-results/single-hop/data"))
    parser.add_argument("--tuning-prepared", type=Path)
    parser.add_argument("--policy", type=Path,
                        default=Path("benchmark-results/single-hop/migration/results/native-label-l-plan.json"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    data = args.data_dir.resolve()
    prepare(data, (args.tuning_prepared or data / "prepared-10m").resolve(), args.policy.resolve(),
            (args.out or data / "prepared-10m-heldout").resolve())


if __name__ == "__main__":
    main()
