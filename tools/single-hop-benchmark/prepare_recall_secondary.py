#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Seal supplemental atomic-label validation from untouched two-tag query vectors.

This workload intentionally changes each original two-tag AND predicate to one
constituent label. Full10M top10 truth is recomputed exactly for that new atomic
predicate; the published two-tag truth is never reused.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import numpy as np

from prepare_recall_validation import (DOWNLOAD_ROOT, POLICY_SHA256, TARGETS, matrix,
                                       provenance, require, sha256, sparse, write_matrix)

SEED = 20261024
PRIMARY_COMMITMENT = "3633c03931652bfb1f2cba352b2475732fac82c0c358899c510558985aae52c2"


def rank(scope, *parts):
    return hashlib.sha256(":".join(map(str, (SEED, scope, *parts))).encode()).digest()


def select_candidates(sources, excluded_embeddings, targets):
    by_embedding = {}
    excluded = []
    duplicates = []
    available = Counter()
    for namespace, queries, pointers, labels in sources:
        for query_id, vector in enumerate(queries):
            original_tags = tuple(map(int, labels[pointers[query_id]:pointers[query_id + 1]]))
            eligible = set(original_tags) & targets
            if len(original_tags) != 2 or not eligible:
                continue
            available[namespace] += 1
            embedding = vector.tobytes()
            embedding_sha = hashlib.sha256(embedding).hexdigest()
            row = {"source_query_namespace": namespace, "source_query_id": query_id,
                   "original_and_tags": list(original_tags), "embedding_sha256": embedding_sha}
            if embedding in excluded_embeddings:
                excluded.append(row)
                continue
            label = min(eligible, key=lambda value: rank("owner", embedding_sha, value))
            row["atomic_label"] = label
            previous = by_embedding.get(embedding)
            if previous is None:
                by_embedding[embedding] = row
            elif rank("representative", namespace, query_id) < rank(
                    "representative", previous["source_query_namespace"], previous["source_query_id"]):
                duplicates.append(previous)
                by_embedding[embedding] = row
            else:
                duplicates.append(row)
    rows = sorted(by_embedding.values(), key=lambda row: rank("order", row["embedding_sha256"]))
    return rows, excluded, duplicates, available


def exact_truth(base, queries, labels, postings, k=10, point_block=8192, query_block=256, progress=None):
    require(k > 0 and point_block >= k and query_block > 0, "Invalid exact truth block sizes")
    require(base.shape[1] == queries.shape[1], "Query/base dimensions differ")
    # Every uint8 192D squared distance is at most12,484,800. Integer dot products,
    # norms, additions and subtractions are exactly representable in float64.
    out_ids = np.full((len(queries), k), -1, dtype="<i4")
    out_distances = np.full((len(queries), k), np.inf, dtype="<f4")
    comparisons = {}
    for label in sorted(set(map(int, labels))):
        members = postings[label]
        require(len(members) >= k, f"Too few matches for label{label}")
        query_rows = np.flatnonzero(labels == label)
        comparisons[str(label)] = len(query_rows) * len(members)
        started = time.monotonic()
        for qbegin in range(0, len(query_rows), query_block):
            selected_rows = query_rows[qbegin:qbegin + query_block]
            q = np.asarray(queries[selected_rows], dtype=np.float64)
            qnorm = np.einsum("ij,ij->i", q, q)
            best_distances = np.full((len(q), k), np.inf)
            best_ids = np.full((len(q), k), -1, dtype=np.int32)
            for begin in range(0, len(members), point_block):
                point_ids = members[begin:begin + point_block]
                x = np.asarray(base[point_ids], dtype=np.float64)
                distances = qnorm[:, None] + np.einsum("ij,ij->i", x, x)[None, :] - 2 * (q @ x.T)
                local_k = min(k, len(point_ids))
                cutoffs = np.partition(distances, local_k - 1, axis=1)[:, local_k - 1].copy()
                for position in range(len(q)):
                    eligible = np.flatnonzero(distances[position] <= cutoffs[position])
                    candidate_ids = np.concatenate((best_ids[position], point_ids[eligible]))
                    candidate_distances = np.concatenate((best_distances[position], distances[position, eligible]))
                    chosen = np.lexsort((candidate_ids, candidate_distances))[:k]
                    best_ids[position] = candidate_ids[chosen]
                    best_distances[position] = candidate_distances[chosen]
            require(np.all(best_ids >= 0) and np.all(best_distances == np.floor(best_distances)), "Noninteger or incomplete exact truth")
            out_ids[selected_rows] = best_ids
            out_distances[selected_rows] = best_distances
        if progress:
            progress({"label": label, "queries": len(query_rows), "matching_base_vectors": len(members),
                      "exhaustive_distances": comparisons[str(label)], "seconds": time.monotonic() - started})
    return out_ids, out_distances, comparisons


def scalar_verify(base, queries, labels, ids, distances, postings, sample_count=100):
    rows = sorted(range(len(queries)), key=lambda row: rank("scalar-check", row))[:sample_count]
    count = 0
    for row in rows:
        members = postings[int(labels[row])]
        for point_id, expected in zip(ids[row], distances[row]):
            index = np.searchsorted(members, point_id)
            require(index < len(members) and members[index] == point_id, "Exact truth violates atomic predicate")
            actual = sum((int(a) - int(b)) ** 2 for a, b in zip(queries[row], base[point_id]))
            require(actual == expected, "Exact truth differs from independent scalar squared L2")
            count += 1
    return count


def prepare(data, raw, primary, policy, output, query_block=256, point_block=8192):
    require(not output.exists(), f"Refusing to overwrite sealed secondary data:{output}")
    require(sha256(primary / "split-commitment.json") == PRIMARY_COMMITMENT, "Primary commitment changed")
    require(sha256(policy) == POLICY_SHA256, "Original L policy changed")
    plan = {int(label): value for label, value in json.loads(policy.read_text()).items()}
    tuning = data / "prepared-10m"
    original = json.loads((tuning / "manifest.json").read_text())
    exclusions = set()
    excluded_artifacts = []
    for directory in (data / "prepared-10m", data / "prepared-10m-heldout", data / "prepared-1m", data / "prepared-smoke",
                      primary / "prepared-private-natural-calibration", primary / "prepared-private-natural-validation"):
        path = directory / "query.u8bin"
        require(path.exists(), f"Missing previously exposed or reserved query corpus:{path}")
        exclusions.update(row.tobytes() for row in matrix(path, "u1"))
        excluded_artifacts.append(provenance(path))
    expected_sources = {item["file"]: item for item in original["sources"]}
    expected_sources.update({item["file"]: item for item in json.loads((primary / "split-commitment.json").read_text())["sources"]})
    sources = []
    source_records = []
    source_vectors = {}
    for namespace, qpath, metadata in (
            ("public_100K", data / "query.public.100K.u8bin", data / "query.metadata.public.100K.spmat"),
            ("official_private_2727415019_100K", raw / "query.private.2727415019.100K.u8bin", raw / "query.metadata.private.2727415019.100K.spmat")):
        for path in (qpath, metadata):
            expected = expected_sources[path.name]
            require(path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"], f"Source identity changed:{path}")
        queries = matrix(qpath, "u1")
        count, pointers, labels = sparse(metadata)
        require(queries.shape == (count, 192) == (100000, 192), "Unexpected source query dimensions")
        sources.append((namespace, queries, pointers, labels))
        source_vectors[namespace] = queries
        source_records += [{"file": path.name, "url": DOWNLOAD_ROOT + path.name, **provenance(path)} for path in (qpath, metadata)]
    rows, excluded, duplicates, available = select_candidates(sources, exclusions, TARGETS)
    require(set(row["atomic_label"] for row in rows) == TARGETS, "Missing target label")
    queries = np.array([source_vectors[row["source_query_namespace"]][row["source_query_id"]] for row in rows], dtype="u1")
    labels = np.array([row["atomic_label"] for row in rows], dtype="<u4")
    output.mkdir(parents=True, exist_ok=False)
    for index, row in enumerate(rows):
        row["query_index"] = index
    rows_path = output / "source-query-rows.jsonl"
    rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    write_matrix(output / "query.u8bin", queries, "u1")
    write_matrix(output / "query_labels.u32bin", labels, "<u4")
    write_matrix(output / "query_ids.u32bin", [row["source_query_id"] for row in rows], "<u4")
    write_matrix(output / "source_namespace.u8bin", [int(row["source_query_namespace"] != "public_100K") for row in rows], "u1")
    (output / "query_labels.txt").write_text("".join(f"{label}\n" for label in labels))
    for filename in ("base.u8bin", "labels.txt", *(f"label_{label}.u32bin" for label in sorted(plan))):
        (output / filename).symlink_to(os.path.relpath(tuning / filename, output))
    proof = {
        "schema_version": 1, "scope": "supplemental_shifted_atomic_label_validation_only",
        "seed": SEED, "source_records": source_records, "primary_commitment_sha256": PRIMARY_COMMITMENT,
        "policy": provenance(policy), "excluded_query_artifacts": excluded_artifacts,
        "excluded_embeddings": len(exclusions), "excluded_rows": excluded, "duplicate_rows_removed": duplicates,
        "eligible_two_tag_source_rows_before_exclusions": dict(available),
        "query_count": len(rows), "per_label_queries": dict(sorted(Counter(map(int, labels)).items())),
        "source_namespace_counts": dict(Counter(row["source_query_namespace"] for row in rows)),
        "source_query_rows": provenance(rows_path),
        "query_artifacts_before_truth": {name: provenance(output / name) for name in
                                         ("query.u8bin", "query_labels.u32bin", "query_ids.u32bin", "source_namespace.u8bin")},
        "original_predicate": "AND of both original source tags",
        "evaluated_predicate": "Exactly one constituent tag, chosen by fixed SHA256 rank; no conjunction",
        "distribution_shift": "Derived atomic-label predicates from untouched two-tag query vectors; analyze separately from natural one-tag workload",
        "selection_protocol": "Exclude all previously exposed public embeddings and both reserved private natural partitions globally; group duplicate embeddings across source namespaces; choose representative by SHA256 rank; choose atomic-label owner by SHA256('20261024:owner:embedding_sha256:label'); order by fixed independent SHA256 rank",
        "validation_only": True, "calibration_queries": 0, "search_quality_not_inspected": True,
        "selection_frozen_before_exact_truth_and_search": True,
        "preparer": provenance(Path(__file__).resolve()),
    }
    selection = output / "selection-commitment.json"
    selection.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"selection_commitment_sha256": sha256(selection), "queries": len(rows),
                      "per_label_queries": proof["per_label_queries"], "sources": proof["source_namespace_counts"]}), flush=True)
    base = matrix(data / "base.10M.u8bin", "u1")
    expected_base = next(item for item in original["sources"] if item["file"] == "base.10M.u8bin")
    require(sha256(data / "base.10M.u8bin") == expected_base["sha256"] and base.shape == (10000000, 192), "Base identity changed")
    postings = {label: matrix(tuning / f"label_{label}.u32bin", "<u4")[:, 0] for label in TARGETS}
    original_postings = json.loads((primary / "exclusion-evidence.json").read_text())["posting_artifacts"]
    for label in TARGETS:
        require(sha256(tuning / f"label_{label}.u32bin") == original_postings[str(label)]["sha256"], "Posting identity changed")
    started = time.monotonic()
    ids, distances, comparisons = exact_truth(base, queries, labels, postings, query_block=query_block,
                                             point_block=point_block, progress=lambda record: print(json.dumps(record), flush=True))
    scalar_checks = scalar_verify(base, queries, labels, ids, distances, postings)
    # Check every result membership, sorted distance/id order, and integer distance vectorially too.
    for label in TARGETS:
        members = postings[label]
        filtered = ids[labels == label]
        require(np.array_equal(members[np.searchsorted(members, filtered)], filtered), "Truth membership failed")
    for begin in range(0, len(queries), 1024):
        difference = np.asarray(base[ids[begin:begin + 1024]], dtype=np.int32) - queries[begin:begin + 1024, None, :].astype(np.int32)
        actual = np.einsum("ijk,ijk->ij", difference, difference, dtype=np.int64)
        require(np.array_equal(actual, distances[begin:begin + 1024]), "Integer truth verification failed")
    with (output / "groundtruth.bin").open("xb") as target:
        target.write(struct.pack("<II", len(rows), 10))
        ids.tofile(target)
        distances.tofile(target)
    manifest = copy.deepcopy(original)
    counts = Counter(map(int, labels))
    manifest.update({"query_count": len(rows), "seed": SEED, "sources": source_records,
                     "split": "secondary_validation", "source_query_namespace": "mixed_public_and_official_private_2727415019_100K",
                     "validation_scope": "recall_fix_supplemental_shifted_atomic_label_validation_only",
                     "source_query_rows": provenance(rows_path), "selection_commitment": provenance(selection),
                     "distribution_shift": proof["distribution_shift"],
                     "query_selection": proof["selection_protocol"],
                     "groundtruth": "Full10M exhaustive exact squared L2 against all points carrying the derived atomic label; float64 exact integer math; ties ordered by ascending original point ID; original AND truth never used",
                     "verification": {"queries": len(rows), "base_vectors": len(base), "k": 10,
                                      "exhaustive_distance_evaluations": comparisons,
                                      "all_truth_ids_label_eligible": True, "all_truth_integer_distances_independently_recomputed": True,
                                      "independent_scalar_checks": scalar_checks,
                                      "exact_truth_seconds": time.monotonic() - started},
                     "frozen_label_l_policy": {**provenance(policy), "entries": [{"label": label, "l": plan[label]} for label in sorted(plan)]}})
    manifest.pop("original_one_tag_queries", None)
    manifest.pop("held_out_two_tag_and_queries", None)
    for record in manifest["labels"]:
        record["selected_queries"] = counts[record["original_label"]]
        record["frozen_l"] = plan[record["original_label"]]
    manifest["artifacts"] = {name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)} for name in
                             ("query.u8bin", "query_ids.u32bin", "query_labels.u32bin", "groundtruth.bin", "labels.txt", "query_labels.txt", "source_namespace.u8bin", "source-query-rows.jsonl")}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "label_counts.json").write_text(json.dumps(manifest["labels"], indent=2) + "\n")
    print(json.dumps({"complete": True, "prepared": str(output), "queries": len(rows),
                      "manifest_sha256": sha256(output / "manifest.json"), "selection_sha256": sha256(selection),
                      "truth_sha256": sha256(output / "groundtruth.bin"), "seconds": time.monotonic() - started}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("benchmark-results/single-hop/data"))
    parser.add_argument("--raw", type=Path, default=Path("benchmark-results/single-hop/recall-fix/data/raw"))
    parser.add_argument("--primary", type=Path, default=Path("benchmark-results/single-hop/recall-fix/data/natural-private"))
    parser.add_argument("--policy", type=Path, default=Path("benchmark-results/single-hop/migration/results/native-label-l-plan.json"))
    parser.add_argument("--out", type=Path, default=Path("benchmark-results/single-hop/recall-fix/data/secondary-two-tag/prepared-secondary-atomic-validation"))
    parser.add_argument("--query-block", type=int, default=256)
    parser.add_argument("--point-block", type=int, default=8192)
    args = parser.parse_args()
    prepare(args.data.resolve(), args.raw.resolve(), args.primary.resolve(), args.policy.resolve(), args.out.resolve(),
            query_block=args.query_block, point_block=args.point_block)


if __name__ == "__main__":
    main()
