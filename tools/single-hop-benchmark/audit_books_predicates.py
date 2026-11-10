#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Independently audit Books predicate semantics and grouped query identities.

Uses full-column Boolean scans, not the producer's time-index lookup. It reads no
ANN result files and no source neighbor/distance datasets. Output is a new proof;
all sealed datasets and producer commitments remain untouched.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np

PUBLISHER_SHA256 = "a1c5b2c1294c05d92046345aa2558d1cc51d71883613ec46b3644b69e54cad6b"
PUBLISHER_BYTES = 24810848512


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def check_record(path, expected):
    actual = record(path)
    require(actual["bytes"] == expected["bytes"] and actual["sha256"] == expected["sha256"], f"Checksum mismatch: {path}")
    return actual


def matrix(path, dtype):
    shape = np.fromfile(path, dtype="<u4", count=2)
    require(len(shape) == 2, f"Truncated matrix {path}")
    rows, columns = map(int, shape)
    require(Path(path).stat().st_size == 8 + rows * columns * np.dtype(dtype).itemsize, f"Malformed matrix {path}")
    return np.memmap(path, mode="r", offset=8, shape=(rows, columns), dtype=dtype)


def direct_members(attributes, parameters):
    """Review's five predicates, independently expressed without producer bounds."""
    timestamp, length, product, rating, votes = [int(value) for value in parameters]
    rating_ranges = {1: (1, 2), 2: (1, 2), 3: (3, 4), 4: (3, 5), 5: (4, 5)}
    require(rating in rating_ranges, "Invalid source review rating")
    first_rating, last_rating = rating_ranges[rating]
    product_radius = (2147483647 - 1048767) * 0.05
    first_product, last_product = int(product - product_radius), int(product + product_radius)
    # Every comparison scans the original full base column. No derived posting,
    # producer time index, or saved low/high metadata influences this mask.
    eligible = attributes[:, 0] >= timestamp - 7 * 86400
    eligible &= attributes[:, 0] <= timestamp
    eligible &= attributes[:, 1] >= length - 30
    eligible &= attributes[:, 1] <= length + 30
    eligible &= attributes[:, 2] >= first_product
    eligible &= attributes[:, 2] <= last_product
    eligible &= attributes[:, 3] >= first_rating
    eligible &= attributes[:, 3] <= last_rating
    eligible &= attributes[:, 4] >= votes // 10
    return np.flatnonzero(eligible).astype("<u4")


def stratified_original(entries, count):
    require(0 < count <= len(entries), "Invalid original predicate audit count")
    ordered = sorted(entries, key=lambda entry: (entry["base_count"], hashlib.sha256(f"books-membership-audit-v1:{entry['label']}".encode()).digest()))
    return [ordered[((2 * i + 1) * len(ordered)) // (2 * count)] for i in range(count)]


def compare_posting(path, identity, expected, base_count):
    actual = check_record(path, identity)
    values = matrix(path, "<u4")
    require(values.shape == (base_count, 1), f"Posting shape/cardinality differs: {path}")
    ids = values.ravel()
    require(np.all(ids[1:] > ids[:-1]), f"Posting IDs are not strictly sorted: {path}")
    require(np.array_equal(ids, expected), f"Full predicate membership mismatch: {path}")
    return actual


def source_metadata(root):
    import h5py
    raw = root / "raw" / "amazon-384-euclidean-5filter.hdf5"
    require(raw.stat().st_size == PUBLISHER_BYTES, "Raw source size differs from publisher")
    raw_identity = record(raw)
    require(raw_identity["sha256"] == PUBLISHER_SHA256, "Raw source differs from publisher LFS SHA256")
    exported = root / "export"
    export_identity = json.loads((exported / "export-identity.json").read_text())
    require(export_identity["source"]["sha256"] == PUBLISHER_SHA256, "Export provenance source differs")
    records = {}
    with h5py.File(raw, "r") as source:
        for dataset_name, filename, dtype in (("train_label", "base_attributes.i32bin", "<i4"),
                                               ("test_label", "query_attributes.i32bin", "<i4"),
                                               ("test_vec", "query.fbin", "<f4")):
            data = source[dataset_name]
            digest = hashlib.sha256(struct.pack("<II", *data.shape))
            for first in range(0, len(data), 10000):
                digest.update(np.asarray(data[first:first + 10000], dtype=dtype).tobytes())
            actual = check_record(exported / filename, export_identity["artifacts"][filename])
            require(digest.hexdigest() == actual["sha256"], f"Raw HDF5/extracted metadata differs: {filename}")
            records[filename] = actual
    return {"raw_hdf5": raw_identity, "export_identity": record(exported / "export-identity.json"),
            "raw_hdf5_metadata_equals_export": True, "files": records}


def verify_group_separation(partitions):
    for field in ("query_id", "embedding", "group", "asin"):
        left = {row[field] for row in partitions["calibration"]}
        right = {row[field] for row in partitions["validation"]}
        require(left.isdisjoint(right), f"Cross-split {field} leakage")


def query_identity(root, workload, commitment, source_queries, source_attributes):
    sealed = root / ("sealed-" + workload)
    group_record = check_record(sealed / "group-assignments.json", commitment["group_assignments"])
    group_rows = json.loads((sealed / "group-assignments.json").read_text())
    require(len(group_rows) == len(source_queries), "Incomplete original-query family assignment")
    groups = {row["query_id"]: row for row in group_rows}
    require(set(groups) == set(range(len(source_queries))), "Original family assignment IDs are incomplete")
    asin_groups, embedding_groups = {}, {}
    for row, (vector, metadata) in enumerate(zip(source_queries, source_attributes)):
        item = groups[row]
        require(item["asin"] == int(metadata[2]), "Original-query family ASIN differs from source")
        require(item["split"] in ("calibration", "validation"), "Invalid original-query split")
        signature = (item["group"], item["split"])
        embedding = hashlib.sha256(vector.tobytes()).hexdigest()
        for table, key in ((asin_groups, item["asin"]), (embedding_groups, embedding)):
            require(key not in table or table[key] == signature, "Source family/embedding component split is inconsistent")
            table[key] = signature
    partitions, artifact_records = {}, {}
    for split in ("calibration", "validation"):
        prepared = sealed / ("prepared-" + split)
        frozen = commitment["outputs"][split]
        for filename, expected in frozen["artifacts"].items():
            artifact_records[f"{split}/{filename}"] = check_record(prepared / filename, expected)
        qids = matrix(prepared / "query_ids.u32bin", "<u4").ravel()
        qlabels = matrix(prepared / "query_labels.u32bin", "<u4").ravel()
        vectors = matrix(prepared / "query.fbin", "<f4")
        metadata = [json.loads(line) for line in (prepared / "query_metadata.jsonl").read_text().splitlines()]
        require(len(qids) == len(qlabels) == len(vectors) == len(metadata) == frozen["query_count"], "Prepared query cardinality mismatch")
        require(len(set(map(int, qids))) == len(qids), "Repeated source query ID in prepared partition")
        require(np.array_equal(vectors, source_queries[qids]), "Prepared embeddings differ from source query IDs")
        rows = []
        for qid, label, vector, meta in zip(qids, qlabels, vectors, metadata):
            qid, label = int(qid), int(label)
            require(meta["query_id"] == qid and meta["label"] == label and meta["group"] == groups[qid]["group"], "Prepared query identity/group mismatch")
            require(groups[qid]["retained"] and groups[qid]["split"] == split, "Prepared query violates original group assignment")
            require(meta["original_filter_parameters"] == source_attributes[qid].tolist(), "Prepared original predicate parameters differ")
            if workload == "original":
                require(label == qid + 1, "Original query-to-filter label association changed")
            rows.append({"query_id": qid, "embedding": hashlib.sha256(vector.tobytes()).hexdigest(),
                         "group": meta["group"], "asin": int(source_attributes[qid, 2])})
        partitions[split] = rows
    verify_group_separation(partitions)
    all_rows = partitions["calibration"] + partitions["validation"]
    require(len({row["embedding"] for row in all_rows}) == len(all_rows), "Exact query duplicates survived global deduplication")
    require({row["query_id"] for row in all_rows} == {row["query_id"] for row in group_rows if row["retained"]}, "Prepared retained-query coverage mismatch")
    return {"group_assignments": group_record, "partition_counts": {key: len(value) for key, value in partitions.items()},
            "partition_unique_groups": {key: len({row["group"] for row in value}) for key, value in partitions.items()},
            "disjoint_source_query_ids_embeddings_groups_and_numeric_asins": True,
            "all_prepared_queries_match_original_source_rows": True, "frozen_query_artifacts": artifact_records}


def audit(root, original_samples=64):
    started = time.monotonic()
    metadata_proof = source_metadata(root)
    attributes = matrix(root / "export" / "base_attributes.i32bin", "<i4")
    query_attributes = matrix(root / "export" / "query_attributes.i32bin", "<i4")
    source_queries = matrix(root / "export" / "query.fbin", "<f4")
    results = {}
    for workload in ("bank", "original"):
        sealed = root / ("sealed-" + workload)
        commitment_path = sealed / "split-commitment.json"
        commitment = json.loads(commitment_path.read_text())
        require(commitment["ann_results_used"] is False and commitment["fresh_validation_queries_searched"] is False,
                "Expected pre-search preparation commitment")
        labels = commitment["labels"]
        selected = labels if workload == "bank" else stratified_original(labels, original_samples)
        if workload == "bank":
            require(len(selected) == 64, "Expected complete 64-predicate bank audit")
        query_proof = query_identity(root, workload, commitment, source_queries, query_attributes)
        final_manifests = {}
        for split in ("calibration", "validation"):
            path = sealed / ("prepared-" + split) / "manifest.json"
            if path.exists():
                manifest = json.loads(path.read_text())
                require(manifest["split_commitment"]["sha256"] == sha(commitment_path), "Prepared manifest refers to another split commitment")
                final_manifests[split] = (manifest, record(path))
        checked = []
        for position, entry in enumerate(selected):
            label = entry["label"]
            parameters = entry["predicate"]["parameters"]
            source_row = entry["source_base_row"] if workload == "bank" else entry["source_query_row"]
            source_values = attributes[source_row] if workload == "bank" else query_attributes[source_row]
            require(parameters == source_values.tolist(), "Predicate parameter provenance differs from source row")
            direct = direct_members(attributes, parameters)
            filename = f"label_{label}.u32bin"
            posting = compare_posting(sealed / filename, entry["posting"], direct, entry["base_count"])
            for split, (manifest, _) in final_manifests.items():
                expected = manifest["artifacts"][filename]
                require(expected["sha256"] == posting["sha256"] and expected["bytes"] == posting["bytes"], "Final manifest posting identity differs")
            checked.append({"label": label, "source_row": source_row, "members": len(direct),
                            "posting_sha256": posting["sha256"], "all_sorted_member_ids_equal": True})
            if (position + 1) % 16 == 0:
                print(f"AUDIT {workload}: {position + 1}/{len(selected)} full-column predicate scans", flush=True)
        results[workload] = {"split_commitment": record(commitment_path), "query_identity": query_proof,
                             "selection": "all 64 frozen base-only predicates" if workload == "bank" else "64 fixed metadata-cardinality mid-quantiles; SHA256 label tie-break; no ANN outcomes used",
                             "total_published_labels": len(labels), "predicates_checked": len(selected),
                             "full_base_rows_per_predicate": len(attributes), "full_column_boolean_rows_evaluated": len(attributes) * len(selected),
                             "checked_predicates": checked, "prepared_manifests_available": {key: value[1] for key, value in final_manifests.items()}}
    return {"schema_version": 1, "status": "pass", "auditor_source": record(Path(__file__)),
            "elapsed_seconds": time.monotonic() - started, "method": "Independent complete base-column Boolean scans over the reviewed five-filter expression; no producer imports or precomputed candidate subsets",
            "ann_results_read": False, "published_source_neighbors_or_distances_read": False,
            "maximum_cpu_parallelism": 1, "source_metadata": metadata_proof, "workloads": results,
            "limit": "All 64 primary bank predicates are exhaustively audited. Secondary original workload samples 64 of its predicates; no claim of auditing every original predicate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--original-samples", type=int, default=64)
    args = parser.parse_args()
    require(not args.out.exists(), "Audit output already exists")
    result = audit(args.root.resolve(), args.original_samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "pass", "proof": str(args.out), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
