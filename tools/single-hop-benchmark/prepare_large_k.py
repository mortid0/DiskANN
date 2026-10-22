#!/usr/bin/env python3
"""Derive balanced exact k=1000/k=100 YFCC10M query sets without changing labels.

The source must be the completed prepared-10m dataset. Existing destinations are
never overwritten. Base vectors, C++ label rows and membership postings are shared
through relative symlinks; only queries, truth and provenance are newly written.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import resource
import struct
import tempfile
import time

import numpy as np

from prepare_yfcc import exact_truth, log, matrix, sha256, sparse, write_matrix


def peak_rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if platform.system() == "Darwin" else rss * 1024)


def shared_link(source, destination):
    destination.symlink_to(os.path.relpath(source, destination.parent))


def integer_check(base, query, members, k):
    """Independent integer-distance verification, with bounded temporary memory."""
    scores = np.empty(len(members), dtype=np.int32)
    for start in range(0, len(members), 16384):
        stop = min(start + 16384, len(members))
        delta = base[members[start:stop]].astype(np.int16) - query.astype(np.int16)
        scores[start:stop] = np.einsum("ij,ij->i", delta, delta, dtype=np.int32)
    partition = np.argpartition(scores, k - 1)[:k]
    candidates = np.flatnonzero(scores <= scores[partition].max())
    order = np.lexsort((members[candidates], scores[candidates]))[:k]
    selected = candidates[order]
    return members[selected], scores[selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("benchmark-results/single-hop/data/prepared-10m"))
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--queries-per-label", type=int, default=4)
    args = parser.parse_args()
    if args.queries_per_label < 1:
        parser.error("--queries-per-label must be positive")
    began = time.monotonic()
    cpu_start = resource.getrusage(resource.RUSAGE_SELF)
    source = args.source.resolve()
    source_manifest = json.loads((source / "manifest.json").read_text())
    if (source_manifest["base_vectors"], source_manifest["dimension"],
            source_manifest["dtype"]) != (10_000_000, 192, "uint8"):
        raise ValueError("This workflow requires the full uint8/192-dimensional YFCC10M base")
    destinations = {k: source.parent / f"prepared-10m-k{k}" for k in (1000, 100)}
    for destination in destinations.values():
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {destination}")

    for name in ("query.u8bin", "query_labels.u32bin", "labels.txt"):
        if sha256(source / name) != source_manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"Prepared input checksum mismatch for {name}")
    base_hash = next(item["sha256"] for item in source_manifest["sources"]
                     if item["file"] == "base.10M.u8bin")
    if sha256(source / "base.u8bin") != base_hash:
        raise ValueError("Full base checksum mismatch")

    base = matrix(source / "base.u8bin", "u1")
    all_queries = matrix(source / "query.u8bin", "u1")
    all_labels = matrix(source / "query_labels.u32bin", "<u4")[:, 0]
    all_query_ids = matrix(source / "query_ids.u32bin", "<u4")[:, 0]
    labels = sorted(int(item["original_label"]) for item in source_manifest["labels"])
    if len(labels) != 64 or base.shape != (10_000_000, 192):
        raise ValueError("Expected the completed 64-label full-10M preparation")
    rng = np.random.default_rng(args.seed)
    rows = []
    for label in labels:
        available = np.flatnonzero(all_labels == label)
        if len(available) < args.queries_per_label:
            raise ValueError(f"Too few existing queries for label {label}")
        rows.extend(map(int, rng.choice(available, size=args.queries_per_label, replace=False)))
    rng.shuffle(rows)
    query_rows = np.asarray(rows, dtype="<u4")
    queries = np.asarray(all_queries[query_rows])
    query_labels = np.asarray(all_labels[query_rows])
    query_ids = np.asarray(all_query_ids[query_rows])
    original_queries = matrix(source.parent / "query.public.100K.u8bin", "u1")
    _, _, qptr, qidx = sparse(source.parent / "query.metadata.public.100K.spmat")
    if not np.array_equal(queries, original_queries[query_ids]) or not np.all(
            np.diff(qptr)[query_ids] == 1) or not np.array_equal(
                qidx[qptr[query_ids]], query_labels):
        raise ValueError("Selection does not preserve original one-tag query semantics")
    postings = {label: matrix(source / f"label_{label}.u32bin", "<u4")[:, 0]
                for label in labels}
    for label, members in postings.items():
        if len(members) < 1000 or np.any(members[1:] <= members[:-1]) or members[-1] >= len(base):
            raise ValueError(f"Invalid or undersized full-base posting list for label {label}")

    log(f"Computing exact top-1000 for {len(rows)} queries, {len(labels)} natural labels, full 10M base")
    truth_ids, truth_distances = exact_truth(base, queries, query_labels, postings, 1000)
    for row, label in enumerate(query_labels):
        ids = truth_ids[row]
        members = postings[int(label)]
        positions = np.searchsorted(members, ids)
        if np.any(positions >= len(members)) or not np.array_equal(members[positions], ids):
            raise ValueError(f"Ground truth violates query {row}'s natural label")
        delta = base[ids].astype(np.int16) - queries[row].astype(np.int16)
        distances = np.einsum("ij,ij->i", delta, delta, dtype=np.int32)
        if not np.array_equal(distances, truth_distances[row]) or np.any(distances[1:] < distances[:-1]):
            raise ValueError(f"Ground-truth distance mismatch for query {row}")
    check_labels = sorted(labels, key=lambda label: len(postings[label]))[:2]
    check_labels += sorted(labels, key=lambda label: len(postings[label]))[-2:]
    for label in check_labels:
        row = int(np.flatnonzero(query_labels == label)[0])
        check_ids, check_distances = integer_check(base, queries[row], postings[label], 1000)
        if not np.array_equal(check_ids, truth_ids[row]) or not np.array_equal(check_distances, truth_distances[row]):
            raise ValueError(f"Independent integer brute-force mismatch for label {label}")
        log(f"Independent integer exact top-1000 verified for label {label}: {len(postings[label])} matches")

    cpu_end = resource.getrusage(resource.RUSAGE_SELF)
    preparation = {"wall_seconds": time.monotonic() - began,
                   "user_cpu_seconds": cpu_end.ru_utime - cpu_start.ru_utime,
                   "system_cpu_seconds": cpu_end.ru_stime - cpu_start.ru_stime,
                   "peak_rss_bytes": peak_rss_bytes(),
                   "queries_per_label": args.queries_per_label,
                   "independent_integer_check_labels": check_labels,
                   "all_returned_distances_and_memberships_verified": True}
    for k, destination in destinations.items():
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        for name in ["base.u8bin", "labels.txt"] + [f"label_{label}.u32bin" for label in labels]:
            shared_link(source / name, staging / name)
        write_matrix(staging / "query.u8bin", queries, "u1")
        write_matrix(staging / "query_labels.u32bin", query_labels, "<u4")
        write_matrix(staging / "query_ids.u32bin", query_ids, "<u4")
        write_matrix(staging / "source_prepared_query_rows.u32bin", query_rows, "<u4")
        np.savetxt(staging / "query_labels.txt", query_labels, fmt="%d")
        with (staging / "groundtruth.bin").open("wb") as output:
            output.write(struct.pack("<II", len(rows), k))
            np.asarray(truth_ids[:, :k], dtype="<i4").tofile(output)
            np.asarray(truth_distances[:, :k], dtype="<f4").tofile(output)
        manifest = copy.deepcopy(source_manifest)
        manifest.pop("verification", None)
        manifest.update({"seed": args.seed, "k": k, "query_count": len(rows),
                         "source_prepared_dataset": str(source),
                         "source_prepared_manifest_sha256": sha256(source / "manifest.json"),
                         "query_selection": "Seeded selection of equal query counts per natural label from prepared-10m, followed by seeded shuffle",
                         "groundtruth": "Exact full-base squared L2, float64 chunked exhaustive candidate scan, ties broken by original id; k=100 is the prefix of the same k=1000 truth",
                         "preparation": preparation})
        for record in manifest["labels"]:
            record["selected_queries"] = int(np.sum(query_labels == record["original_label"]))
        artifact_names = ("query.u8bin", "query_labels.u32bin", "query_ids.u32bin", "groundtruth.bin",
                          "labels.txt", "query_labels.txt", "source_prepared_query_rows.u32bin")
        manifest["artifacts"] = {name: {"bytes": (staging / name).stat().st_size,
                                        "sha256": sha256(staging / name)} for name in artifact_names}
        manifest["prepared_identity"] = ":".join(manifest["artifacts"][name]["sha256"]
                                                 for name in ("query.u8bin", "query_labels.u32bin", "groundtruth.bin", "labels.txt"))
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "label_counts.json").write_text(json.dumps(manifest["labels"], indent=2) + "\n")
        staging.rename(destination)
        results = source.parent.parent / "results"
        results.mkdir(parents=True, exist_ok=True)
        (results / f"dataset-10m-k{k}-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        log(f"READY {destination}: {len(rows)} balanced queries, exact k={k}")
    log(f"Finished in {time.monotonic() - began:.1f}s; peak RSS {peak_rss_bytes() / 1024**3:.3f} GiB")


if __name__ == "__main__":
    main()
