#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Complete immutable Amazon-C4 manifest closure before the first ANN search.

Kept separate from the frozen embedding producer. Verifies both sealed partitions,
all category memberships, query families, and every selected truth pair, then adds
postings hashes to the runtime manifests. Existing pre-audit manifests are retained.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def matrix(path, dtype):
    path = Path(path)
    n, d = np.fromfile(path, dtype="<u4", count=2)
    n, d = int(n), int(d)
    require(n > 0 and d > 0 and path.stat().st_size == 8 + n * d * np.dtype(dtype).itemsize,
            f"Malformed matrix: {path}")
    return np.memmap(path, mode="r", offset=8, shape=(n, d), dtype=dtype)


def verify_postings(point_labels, label_records, postings):
    seen = np.zeros(len(point_labels), dtype=np.uint8)
    for entry in label_records:
        label = int(entry["label"])
        expected = np.flatnonzero(point_labels == label)
        actual = np.asarray(postings[label])
        require(len(actual) == entry["base_count"] and np.array_equal(actual, expected),
                f"Category posting differs from full corpus metadata: label{label}")
        require(len(actual) >= 10, f"Undersized category: label{label}")
        seen[actual] += 1
    require(np.all(seen == 1), "Categories must partition the retained corpus exactly")


def verify_truth(base, queries, labels, postings, ids, distances32, distances64):
    n, k = ids.shape
    require(n == len(queries) == len(labels) and k == 10, "Unexpected truth/query dimensions")
    require(np.all((ids >= 0) & (ids < len(base))) and np.all(np.isfinite(distances64)), "Invalid truth IDs/distances")
    require(np.array_equal(distances32, distances64.astype("<f4")), "Rounded ground-truth distances differ")
    for start in range(0, n, 512):
        stop = min(start + 512, n)
        selected = ids[start:stop]
        difference = np.asarray(base[selected], dtype=np.float64) - np.asarray(queries[start:stop], dtype=np.float64)[:, None, :]
        actual = np.einsum("ijk,ijk->ij", difference, difference)
        require(np.allclose(actual, distances64[start:stop], rtol=0, atol=1e-14), "Ground-truth distance coordinate check failed")
        for row in range(start, stop):
            members = postings[int(labels[row])]
            positions = np.searchsorted(members, ids[row])
            require(np.all(positions < len(members)) and np.array_equal(members[positions], ids[row]), "Ground truth violates atomic category")
            require(len(set(map(int, ids[row]))) == k, "Duplicate truth ID")
            order = np.lexsort((ids[row], distances64[row]))
            require(np.array_equal(order, np.arange(k)), "Truth is not sorted by distance then ID")
    return n * k


def json_rows(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source]


def audit(root):
    sealed = root / "sealed"
    target = sealed / "prepared-audit.json"
    require(not target.exists(), "Prepared audit already sealed; do not replace it")
    commitment_path = sealed / "split-commitment.json"
    commitment = json.loads(commitment_path.read_text())
    embedding_path = root / "embedding" / "embedding-identity.json"
    require(sha256(embedding_path) == commitment["embedding_identity"]["sha256"], "Embedding identity changed")
    embedding = json.loads(embedding_path.read_text())
    configuration_path = root / "embedding" / "configuration.json"
    require(sha256(configuration_path) == embedding["configuration"]["sha256"], "Embedding configuration changed")
    configuration = json.loads(configuration_path.read_text())
    metadata_path = root / "metadata" / "metadata.json"
    require(sha256(metadata_path) == configuration["metadata"]["sha256"], "Embedding metadata changed")
    metadata = json.loads(metadata_path.read_text())
    for name, expected in metadata["artifacts"].items():
        require(sha256(root / "metadata" / name) == expected["sha256"], f"Corpus metadata changed: {name}")
    for name, expected in configuration["model_files"].items():
        require(sha256(root / "model" / name) == expected["sha256"], f"Encoder model changed: {name}")
    require(sha256(configuration["source"]["path"]) == configuration["source"]["sha256"], "Frozen embedding producer source changed")
    require(sha256(sealed / "labels.txt") == metadata["artifacts"]["labels.txt"]["sha256"], "Prepared category rows differ from encoder corpus")
    base = matrix(sealed / "base.fbin", "<f4")
    require(base.shape == (commitment["base_vectors"], commitment["dimension"]), "Base dimensions differ")
    require(sha256(sealed / "base.fbin") == embedding["outputs"]["base"]["sha256"], "Base embedding changed")
    point_labels = np.loadtxt(sealed / "labels.txt", dtype="<u4")
    require(len(point_labels) == len(base), "Point-label count differs")
    label_records = commitment["labels"]
    postings = {int(entry["label"]): matrix(sealed / f"label_{entry['label']}.u32bin", "<u4").ravel() for entry in label_records}
    verify_postings(point_labels, label_records, postings)
    assignment_path = Path(commitment["group_assignments"]["path"])
    require(sha256(assignment_path) == commitment["group_assignments"]["sha256"], "Recorded source-family assignments changed")
    assignments = json.loads(assignment_path.read_text())
    assigned_source_rows = [row for group in assignments for row in group["source_query_rows"]]
    require(sorted(assigned_source_rows) == list(range(metadata["retained_source_query_count"])), "Recorded families do not partition source queries")
    assigned_retained = {}
    for group in assignments:
        require(group["split"] in ("calibration", "validation"), "Invalid recorded partition")
        require(set(group["retained_query_rows"]).issubset(group["source_query_rows"]), "Retained query outside its source family")
        for row in group["retained_query_rows"]:
            require(row not in assigned_retained, "Query retained in multiple families")
            assigned_retained[row] = group["split"]
    manifests, query_groups, retained_rows = {}, {}, {}
    for split in ("calibration", "validation"):
        prepared = sealed / ("prepared-" + split)
        path = prepared / "manifest.json"
        manifest = json.loads(path.read_text())
        for name, expected in manifest["artifacts"].items():
            require(sha256(prepared / name) == expected["sha256"], f"Prepared artifact changed: {split}/{name}")
        for name, expected in commitment["outputs"][split]["artifacts"].items():
            require(sha256(prepared / name) == expected["sha256"], f"Sealed query changed: {split}/{name}")
        require(sha256(commitment_path) == manifest["split_commitment"]["sha256"], "Manifest commitment mismatch")
        queries = matrix(prepared / "query.fbin", "<f4")
        labels = matrix(prepared / "query_labels.u32bin", "<u4").ravel()
        qids = matrix(prepared / "query_ids.u32bin", "<u4").ravel()
        truth_path = prepared / "groundtruth.bin"
        n, k = map(int, np.fromfile(truth_path, dtype="<u4", count=2))
        require(truth_path.stat().st_size == 8 + n * k * 8, "Malformed ground truth")
        ids = np.memmap(truth_path, mode="r", dtype="<i4", offset=8, shape=(n, k))
        distances32 = np.memmap(truth_path, mode="r", dtype="<f4", offset=8 + n * k * 4, shape=(n, k))
        distances64 = matrix(prepared / "groundtruth_distances.f64bin", "<f8")
        checked = verify_truth(base, queries, labels, postings, ids, distances32, distances64)
        query_meta = json_rows(prepared / "query_metadata.jsonl")
        require(len(query_meta) == n and [q["qid"] for q in query_meta] == list(qids), "Query metadata/qids differ")
        require(all("id" in q for q in query_meta), "Prepared query lacks original preparation row")
        retained_rows[split] = {q["id"] for q in query_meta}
        require(retained_rows[split] == {row for row, partition in assigned_retained.items() if partition == split},
                "Prepared queries differ from recorded pre-dedup source families")
        query_groups[split] = {kind: {q[kind] for q in query_meta} for kind in ("qid", "positive_item_id", "user_hash", "text_hash")}
        query_groups[split]["embedding"] = {hashlib.sha256(v.tobytes()).hexdigest() for v in queries}
        require(len(query_groups[split]["embedding"]) == n, "Duplicate query embedding inside split")
        additional = {name: record(prepared / name) for name in ("groundtruth_distances.f64bin", "truth-audit.json",
                      *(f"label_{label}.u32bin" for label in sorted(postings)))}
        manifest["artifacts"].update(additional)
        counts = Counter(map(int, labels))
        require(all(entry["selected_queries"] == counts[entry["original_label"]] for entry in manifest["labels"]), "Per-label query counts differ")
        manifests[split] = {"prepared": prepared, "manifest": manifest,
                            "original_manifest": record(path), "truth_pairs_checked": checked}
    for kind in query_groups["calibration"]:
        require(query_groups["calibration"][kind].isdisjoint(query_groups["validation"][kind]),
                f"Calibration/holdout overlap: {kind}")
    # Author manifest closure only after every cross-partition check succeeds.
    manifest_outputs = {}
    for split, result in manifests.items():
        prepared, manifest = result["prepared"], result["manifest"]
        original = prepared / "manifest.before-postings-audit.json"
        require(not original.exists(), "Pre-audit manifest archive already exists")
        (prepared / "manifest.json").rename(original)
        manifest["postings_audit"] = {"tool": record(Path(__file__)), "prior_manifest": record(original),
                                      "all_memberships_match_labels": True,
                                      "all_truth_pairs_checked": result["truth_pairs_checked"],
                                      "cross_split_family_and_embedding_overlap": False,
                                      "ann_results_used": False}
        staged = prepared / "manifest.audit-staging.json"
        with staged.open("x") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        staged.rename(prepared / "manifest.json")
        manifest_outputs[split] = record(prepared / "manifest.json")
    result = {"schema_version": 1, "tool": record(Path(__file__)), "split_commitment": record(commitment_path),
              "all_category_postings_verified": len(postings), "base_vectors": len(base),
              "truth_pairs_checked": sum(x["truth_pairs_checked"] for x in manifests.values()),
              "cross_split_overlap_checked": list(query_groups["calibration"]),
              "cross_split_overlap": False, "recorded_pre_dedup_families_verified": len(assignments),
              "ann_results_used": False,
              "manifests": manifest_outputs}
    with target.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    audit(args.root.resolve())


if __name__ == "__main__":
    main()
