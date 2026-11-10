#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Seal natural, unused YFCC queries before evaluating a recall-fix candidate.

The official benchmark's released formerly-private query set provides fresh
natural one-tag predicates. All public query embeddings, including unused ones,
are excluded globally. Selection and the 25% calibration / 75% validation split
use only label, original query ID, and fixed SHA-256 ranks, never search results.
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

SEED = 20261023
OFFICIAL_SOURCE = "https://raw.githubusercontent.com/harsha-simhadri/big-ann-benchmarks/main/benchmark/datasets.py"
DOWNLOAD_ROOT = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/"
SOURCES = {
    "query.private.2727415019.100K.u8bin": (19200008, "0e06b358d41c2b9f4c7f5c2bc81f80061bc4d4b38d67b3c0b23d1700a3d8734d"),
    "query.metadata.private.2727415019.100K.spmat": (1905808, "174f0e04871223f8273b9516aff7a422c91ffb01c98d573c6746cf9718c70884"),
    "GT.private.2727415019.ibin": (8000008, "8d5628ce7a706d1ee346518515552ff8ee17de52daae3206be07e0273b932eae"),
}
TARGETS = {9, 24, 89, 98, 2495}
POLICY_SHA256 = "df42e65440af0a891185913e91f5f22cb6808c9251b2c1d8f6ca968d5bda2e59"


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


def matrix(path, dtype):
    shape = np.fromfile(path, dtype="<u4", count=2)
    require(len(shape) == 2, f"Truncated matrix: {path}")
    n, d = map(int, shape)
    require(path.stat().st_size == 8 + n * d * np.dtype(dtype).itemsize, f"Malformed matrix: {path}")
    return np.memmap(path, mode="r", dtype=dtype, offset=8, shape=(n, d))


def sparse(path):
    shape = np.fromfile(path, dtype="<i8", count=3)
    require(len(shape) == 3, f"Truncated sparse matrix: {path}")
    n, columns, nnz = map(int, shape)
    require(n > 0 and columns > 0 and nnz >= 0 and path.stat().st_size == 24 + 8 * (n + 1) + 8 * nnz,
            f"Malformed sparse matrix: {path}")
    ptr = np.memmap(path, mode="r", dtype="<i8", offset=24, shape=n + 1)
    idx = np.memmap(path, mode="r", dtype="<i4", offset=24 + 8 * (n + 1), shape=nnz)
    require(ptr[0] == 0 and ptr[-1] == nnz and np.all(ptr[1:] >= ptr[:-1])
            and np.all((idx >= 0) & (idx < columns)), f"Invalid sparse offsets or labels: {path}")
    return n, ptr, idx


def write_matrix(path, values, dtype):
    values = np.asarray(values, dtype=dtype)
    if values.ndim == 1:
        values = values[:, None]
    with path.open("xb") as output:
        output.write(struct.pack("<II", *values.shape))
        values.tofile(output)


def rank(scope, query_id, label=0):
    return hashlib.sha256(f"{SEED}:{scope}:{query_id}:{label}".encode()).digest()


def select_fresh(queries, pointers, labels, eligible_labels, excluded_embeddings):
    selected = []
    excluded = []
    duplicates = []
    seen = set()
    available = Counter()
    for query_id, vector in enumerate(queries):
        if pointers[query_id + 1] - pointers[query_id] != 1:
            continue
        label = int(labels[pointers[query_id]])
        if label not in eligible_labels:
            continue
        available[label] += 1
        embedding = vector.tobytes()
        if embedding in excluded_embeddings:
            excluded.append((query_id, label))
            continue
        if embedding in seen:
            duplicates.append((query_id, label))
            continue
        seen.add(embedding)
        selected.append((query_id, label))
    return selected, excluded, duplicates, available


def split_queries(selected):
    calibration = []
    validation = []
    for label in sorted({label for _, label in selected}):
        group = sorted((row for row in selected if row[1] == label),
                       key=lambda row: rank("split", *row))
        # Exact floor n/4; sparse labels may have zero calibration queries.
        cut = len(group) // 4
        calibration.extend(group[:cut])
        validation.extend(group[cut:])
    calibration.sort(key=lambda row: rank("order-calibration", *row))
    validation.sort(key=lambda row: rank("order-validation", *row))
    return calibration, validation


def verify_truth(base, queries, selected, truth_ids, truth_distances, postings, k):
    checks = 0
    for begin in range(0, len(selected), 1024):
        group = selected[begin:begin + 1024]
        original_ids = np.array([row for row, _ in group], dtype=np.int64)
        ids = np.asarray(truth_ids[original_ids])
        distances = np.asarray(truth_distances[original_ids])
        require(np.all((ids >= 0) & (ids < len(base))) and np.all(distances[:, 1:] >= distances[:, :-1]),
                "Invalid IDs or unsorted published truth")
        require(all(len(set(map(int, row))) == k for row in ids), "Duplicate published truth IDs")
        vectors = np.asarray(queries[original_ids], dtype=np.int32)
        points = np.asarray(base[ids], dtype=np.int32)
        difference = points - vectors[:, None, :]
        actual = np.einsum("ijk,ijk->ij", difference, difference, dtype=np.int64)
        require(np.array_equal(actual, distances), "Published truth differs from exact integer squared L2")
        for (_, label), row_ids in zip(group, ids):
            members = postings[label]
            positions = np.searchsorted(members, row_ids)
            require(np.all(positions < len(members)) and np.array_equal(members[positions], row_ids),
                    f"Published truth violates atomic predicate: label {label}")
        checks += len(group) * k
    return checks


def write_split(output, name, selected, queries, truth_ids, truth_distances, original, tuning,
                plan, policy, source_records, available, exclusion_sha, source_counts):
    output.mkdir(exist_ok=False)
    for filename in ("base.u8bin", "labels.txt", *(f"label_{label}.u32bin" for label in sorted(plan))):
        (output / filename).symlink_to(os.path.relpath(tuning / filename, output))
    ids = np.array([row for row, _ in selected], dtype="<u4")
    labels = np.array([label for _, label in selected], dtype="<u4")
    write_matrix(output / "query.u8bin", queries[ids], "u1")
    write_matrix(output / "query_ids.u32bin", ids, "<u4")
    write_matrix(output / "query_labels.u32bin", labels, "<u4")
    with (output / "groundtruth.bin").open("xb") as target:
        target.write(struct.pack("<II", len(selected), 10))
        np.asarray(truth_ids[ids], dtype="<i4").tofile(target)
        np.asarray(truth_distances[ids], dtype="<f4").tofile(target)
    (output / "query_labels.txt").write_text("".join(f"{label}\n" for label in labels))
    counts = Counter(map(int, labels))
    manifest = copy.deepcopy(original)
    manifest.update({
        "seed": SEED, "query_count": len(selected), "sources": source_records,
        "original_one_tag_queries": source_counts[1], "held_out_two_tag_and_queries": source_counts[2],
        "label_selection": "The original frozen 64-label policy set; no labels selected using these fresh queries",
        "verification": {"base_vectors": original["base_vectors"],
                         "queries": len(selected), "k": 10, "query_predicates_verified": True,
                         "all_truth_pairs_distance_checked": len(selected) * 10,
                         "all_truth_ids_label_membership_checked": True,
                         "truth_source": "official published full10M exact truth; exhaustive recomputation not performed"},
        "validation_scope": f"recall_fix_fresh_natural_private_{name}",
        "query_selection": f"Natural one-tag rows from the official formerly-private 100K query set, globally disjoint from all public query embeddings; deterministic per-label floor(25%) calibration, remainder validation; this is the {name} partition",
        "groundtruth": "Official published full-10M exact one-tag top10, every selected truth ID membership and integer squared-L2 distance independently verified; not exhaustive recomputation",
        "official_dataset_definition": OFFICIAL_SOURCE,
        "source_query_namespace": "official_private_2727415019_100K",
        "original_public_query_namespace": "public_100K (numeric IDs may coincide; embeddings cannot)",
        "distribution_shift": "None in predicate semantics: original natural one-tag queries, original labels and full10M base; fresh source is the official formerly-private split",
        "split": name, "split_exclusion_evidence_sha256": exclusion_sha,
        "frozen_label_l_policy": {**provenance(policy), "entries": [{"label": label, "l": plan[label]} for label in sorted(plan)]},
        "covered_label_count": len(counts),
    })
    for record in manifest["labels"]:
        label = record["original_label"]
        record.update({"available_one_tag_queries": available[label], "selected_queries": counts[label],
                       "frozen_l": plan[label], "fresh_coverage": "observed" if counts[label] else "none"})
    manifest["artifacts"] = {filename: {"bytes": (output / filename).stat().st_size,
                                        "sha256": sha256(output / filename)} for filename in
                             ("query.u8bin", "query_ids.u32bin", "query_labels.u32bin", "groundtruth.bin", "labels.txt", "query_labels.txt")}
    (output / "label_counts.json").write_text(json.dumps(manifest["labels"], indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"path": str(output), "queries": len(selected), "labels": len(counts),
            "target_counts": {str(label): counts[label] for label in sorted(TARGETS)},
            "manifest_sha256": sha256(output / "manifest.json"),
            "artifacts": manifest["artifacts"]}


def prepare(data, raw, tuning, policy, output):
    started = time.monotonic()
    require(not output.exists(), f"Refusing to overwrite existing prepared data: {output}")
    require(sha256(policy) == POLICY_SHA256, "Frozen label-to-L policy changed")
    plan = {int(label): value for label, value in json.loads(policy.read_text()).items()}
    require(len(plan) == 64 and all(type(value) is int and value >= 64 for value in plan.values()), "Invalid original policy")
    original = json.loads((tuning / "manifest.json").read_text())
    require(set(plan) == {row["original_label"] for row in original["labels"]}, "Label set changed")
    source_records = []
    for name, (size, expected) in SOURCES.items():
        item = provenance(raw / name)
        require(item["bytes"] == size and item["sha256"] == expected, f"Official source changed: {name}")
        source_records.append({"file": name, "url": DOWNLOAD_ROOT + name, **item})
    for name in ("base.10M.u8bin", "query.public.100K.u8bin"):
        expected = next(item for item in original["sources"] if item["file"] == name)
        item = provenance(data / name)
        require(item["bytes"] == expected["bytes"] and item["sha256"] == expected["sha256"], f"Base or original public source changed: {name}")
        source_records.append({"file": name, "url": DOWNLOAD_ROOT + name, **item})
    queries = matrix(raw / "query.private.2727415019.100K.u8bin", "u1")
    public = matrix(data / "query.public.100K.u8bin", "u1")
    base = matrix(data / "base.10M.u8bin", "u1")
    rows, pointers, labels = sparse(raw / "query.metadata.private.2727415019.100K.spmat")
    require(queries.shape == public.shape == (100000, 192) and rows == 100000
            and base.shape == (10000000, 192), "Unexpected source dimensions")
    public_embeddings = {row.tobytes() for row in public}
    selected, excluded, duplicate_rows, available = select_fresh(queries, pointers, labels, set(plan), public_embeddings)
    require(not duplicate_rows, "Fresh source contains duplicate eligible embeddings; review grouping before splitting")
    calibration, validation = split_queries(selected)
    require(set(calibration).isdisjoint(validation) and len(calibration) + len(validation) == len(selected), "Invalid split")
    require({label for _, label in validation} == set(plan), "Validation does not cover all frozen labels")
    truth_path = raw / "GT.private.2727415019.ibin"
    truth_shape = tuple(map(int, np.fromfile(truth_path, dtype="<u4", count=2)))
    require(truth_shape == (100000, 10), "Unexpected official truth shape")
    truth_ids = np.memmap(truth_path, mode="r", dtype="<i4", offset=8, shape=truth_shape)
    truth_distances = np.memmap(truth_path, mode="r", dtype="<f4", offset=8 + 4 * 100000 * 10, shape=truth_shape)
    postings = {}
    posting_artifacts = {}
    for record in original["labels"]:
        label = record["original_label"]
        posting_path = tuning / f"label_{label}.u32bin"
        values = matrix(posting_path, "<u4")
        require(values.shape == (record["base_count"], 1), f"Posting count changed: {label}")
        postings[label] = values[:, 0]
        require(np.all(postings[label][1:] > postings[label][:-1]), f"Postings not sorted unique: {label}")
        posting_artifacts[str(label)] = provenance(posting_path)
    checked = verify_truth(base, queries, selected, truth_ids, truth_distances, postings, 10)
    output.mkdir(parents=True, exist_ok=False)
    evidence = {
        "schema_version": 1, "seed": SEED, "query_source": "official formerly-private natural one-tag queries",
        "official_dataset_definition": OFFICIAL_SOURCE, "policy": provenance(policy),
        "public_embeddings_excluded": len(public_embeddings), "fresh_unique_embeddings": len(selected),
        "excluded_query_rows": [{"query_id": row, "label": label} for row, label in excluded],
        "excluded_duplicates": duplicate_rows,
        "eligible_before_exclusions": sum(available.values()), "fresh_natural_queries": len(selected),
        "posting_artifacts": posting_artifacts,
        "checks": {"all_public_query_embeddings_excluded_globally": True,
                   "all_fresh_embeddings_unique_globally": True,
                   "calibration_validation_query_and_embedding_disjoint": True,
                   "original_predicates_preserved": True,
                   "published_truth_ids_eligible": True,
                   "published_truth_distances_equal_independent_integer_squared_l2": True,
                   "truth_pairs_checked": checked, "exhaustive_truth_recomputed": False},
        "split_protocol": "For each label sort by SHA256('20261023:split:original_private_query_id:label'), first floor(n/4) calibration, remainder sealed validation. Independent SHA ranks shuffle the final partition order. No search results consumed.",
        "split_before_candidate_quality_evaluation": True,
    }
    evidence_path = output / "exclusion-evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    evidence_sha = sha256(evidence_path)
    source_counts = Counter(map(int, np.diff(pointers)))
    descriptions = {}
    for name, rows_selected in (("calibration", calibration), ("validation", validation)):
        descriptions[name] = write_split(output / f"prepared-private-natural-{name}", name, rows_selected,
                                         queries, truth_ids, truth_distances, original, tuning,
                                         plan, policy, source_records, available, evidence_sha, source_counts)
    commitment = {"schema_version": 1, "seed": SEED, "scope": "fresh_private_natural_full64_labels",
                  "policy_sha256": POLICY_SHA256, "exclusion_evidence_sha256": evidence_sha,
                  "sources": source_records, "partitions": descriptions,
                  "all_source_rows_and_embeddings_disjoint_between_partitions": True,
                  "preparer": provenance(Path(__file__).resolve()),
                  "validation_quality_not_examined_before_candidate_frozen": True}
    commitment_path = output / "split-commitment.json"
    commitment_path.write_text(json.dumps(commitment, indent=2) + "\n")
    print(json.dumps({"commitment_path": str(commitment_path), "commitment_sha256": sha256(commitment_path),
                      "calibration": {k: v for k, v in descriptions["calibration"].items() if k != "artifacts"},
                      "validation": {k: v for k, v in descriptions["validation"].items() if k != "artifacts"},
                      "seconds": time.monotonic() - started}, indent=2), flush=True)
    return commitment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("benchmark-results/single-hop/data"))
    parser.add_argument("--raw", type=Path, default=Path("benchmark-results/single-hop/recall-fix/data/raw"))
    parser.add_argument("--tuning", type=Path)
    parser.add_argument("--policy", type=Path, default=Path("benchmark-results/single-hop/migration/results/native-label-l-plan.json"))
    parser.add_argument("--out", type=Path, default=Path("benchmark-results/single-hop/recall-fix/data/natural-private"))
    args = parser.parse_args()
    prepare(args.data.resolve(), args.raw.resolve(), (args.tuning or args.data / "prepared-10m").resolve(),
            args.policy.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
