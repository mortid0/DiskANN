#!/usr/bin/env python3
"""Prepare reproducible single-label YFCC-10M pilots and full-size comparisons.

Only original one-tag queries are selected. Two-tag AND queries are held out.
The full-size truth is copied by row; prefix truth is recomputed exactly.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np


SOURCE = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/"
FILES = {
    "base.10M.u8bin": 1920000008,
    "base.metadata.10M.spmat": 945683840,
    "query.public.100K.u8bin": 19200008,
    "query.metadata.public.100K.spmat": 1907024,
    "GT.public.ibin": 8000008,
}
EXPECTED_SHA256 = {
    "base.10M.u8bin": "589030afcbcc44a50ff798cec935f0980815a02eb7075b224d4f7c12febe96bf",
    "base.metadata.10M.spmat": "2f9c9a533fde31cb062edfdf5c9410f088bb5d7bf4a721ab98c47e3d2954287d",
    "query.public.100K.u8bin": "77a937d67baf7523a7d9089a6e49c0ac8c6a3e7eb5499d759afeacf0a843df82",
    "query.metadata.public.100K.spmat": "93eae99029c34d167f7d030c60c6b4075c039247b3204cc6c34f8a44066983bd",
    "GT.public.ibin": "db8a37bbbf1d32681fb01e0c4694044cf731d91c8e34ccf4e407d7ee9a14e251",
}
SENTINEL = 200386
SEED = 20261021


def log(message):
    print(message, flush=True)


def matrix(path, dtype):
    n, d = np.fromfile(path, dtype="<u4", count=2)
    n, d = int(n), int(d)
    expected = 8 + n * d * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"Unexpected size for {path}: expected {expected}")
    return np.memmap(path, mode="r", dtype=dtype, offset=8, shape=(n, d))


def sparse(path):
    rows, cols, nnz = map(int, np.fromfile(path, dtype="<i8", count=3))
    if path.stat().st_size != 24 + 8 * (rows + 1) + 8 * nnz:
        raise ValueError(f"Malformed sparse matrix: {path}")
    ptr = np.memmap(path, mode="r", dtype="<i8", offset=24, shape=rows + 1)
    idx = np.memmap(path, mode="r", dtype="<i4", offset=24 + 8 * (rows + 1), shape=nnz)
    if ptr[0] != 0 or ptr[-1] != nnz or cols != SENTINEL:
        raise ValueError(f"Unexpected sparse metadata header in {path}")
    return rows, cols, ptr, idx


def write_matrix(path, values, dtype):
    values = np.asarray(values, dtype=dtype)
    if values.ndim == 1:
        values = values[:, None]
    with path.open("wb") as out:
        out.write(struct.pack("<II", *values.shape))
        values.tofile(out)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select_labels(bidx, qptr, qidx, max_labels):
    frequencies = np.bincount(bidx, minlength=SENTINEL)
    single_rows = np.flatnonzero(np.diff(qptr) == 1)
    single_labels = qidx[qptr[single_rows]]
    query_counts = np.bincount(single_labels, minlength=SENTINEL)
    # Four logarithmic/selectivity regimes; no synthetic labels enter the query set.
    edges = [0.0001, 0.001, 0.01, 0.1, 0.2000001]
    rng = np.random.default_rng(SEED)
    selected = []
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        candidates = np.flatnonzero((frequencies >= 10_000_000 * lo) &
                                   (frequencies < 10_000_000 * hi) & (query_counts >= 10))
        candidates = rng.permutation(candidates)
        bins.append(list(map(int, candidates)))
    # Round-robin bins to avoid silently dropping rare labels when a bin is small.
    while len(selected) < max_labels and any(bins):
        for group in bins:
            if group and len(selected) < max_labels:
                selected.append(group.pop())
    selected.sort()
    if not selected:
        raise ValueError("No labels fit the requested natural selectivity regimes")
    return selected, frequencies, single_rows, single_labels, query_counts


def select_queries(selected, single_rows, single_labels, limit):
    rng = np.random.default_rng(SEED)
    queues = {label: list(map(int, rng.permutation(single_rows[single_labels == label])))
              for label in selected}
    chosen = []
    while len(chosen) < limit and any(queues.values()):
        for label in selected:
            if queues[label] and len(chosen) < limit:
                chosen.append((queues[label].pop(), label))
    chosen.sort()
    rng.shuffle(chosen)
    return np.array([x[0] for x in chosen], dtype="<u4"), np.array([x[1] for x in chosen], dtype="<u4")


def copy_prefix(source, target, count, dimension):
    with source.open("rb") as src, target.open("wb") as dst:
        src.seek(8)
        dst.write(struct.pack("<II", count, dimension))
        remaining = count * dimension
        while remaining:
            block = src.read(min(remaining, 16 << 20))
            if not block:
                raise ValueError("Truncated vector input")
            dst.write(block)
            remaining -= len(block)


def write_labels(path, ids, labels, count):
    sentinel = f"{SENTINEL}\n"
    # ids are ordered by original metadata position, hence by base-vector id.
    boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    previous = -1
    with path.open("w", buffering=1 << 20) as out:
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            row = int(ids[start])
            out.write(sentinel * (row - previous - 1))
            out.write(",".join(map(str, labels[start:stop])) + "\n")
            previous = row
        out.write(sentinel * (count - previous - 1))


def exact_truth(base, queries, query_labels, postings, k):
    ids = np.full((len(queries), k), -1, dtype="<i4")
    distances = np.full((len(queries), k), np.inf, dtype="<f4")
    for label in sorted(postings):
        rows = np.flatnonzero(query_labels == label)
        members = postings[label]
        if len(members) < k:
            raise ValueError(f"Label {label} has fewer than {k} prefix matches")
        # float64 makes these integer-valued squared L2 computations exact.
        q = np.asarray(queries[rows], dtype=np.float64)
        qnorm = np.einsum("ij,ij->i", q, q)
        best_d = np.full((len(rows), k), np.inf)
        best_i = np.full((len(rows), k), -1, dtype=np.int32)
        for begin in range(0, len(members), 16384):
            block_ids = members[begin:begin + 16384]
            x = np.asarray(base[block_ids], dtype=np.float64)
            d = qnorm[:, None] + np.einsum("ij,ij->i", x, x)[None, :] - 2 * (q @ x.T)
            candidate_d = np.concatenate((best_d, d), axis=1)
            candidate_i = np.concatenate((best_i, np.broadcast_to(block_ids, d.shape)), axis=1)
            # Lexicographic tie breaking by id gives deterministic exact neighbors.
            part = np.argpartition(candidate_d, k - 1, axis=1)[:, :k]
            cutoffs = np.take_along_axis(candidate_d, part, axis=1).max(axis=1)
            for row in range(len(rows)):
                tied = np.flatnonzero(candidate_d[row] <= cutoffs[row])
                order = np.lexsort((candidate_i[row, tied], candidate_d[row, tied]))[:k]
                take = tied[order]
                best_d[row], best_i[row] = candidate_d[row, take], candidate_i[row, take]
        ids[rows], distances[rows] = best_i, best_d
        log(f"exact truth label {label}: {len(rows)} queries, {len(members)} matches")
    return ids, distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark-results/single-hop/data"))
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000000, 10000000])
    parser.add_argument("--max-labels", type=int, default=64)
    parser.add_argument("--pilot-queries", type=int, default=1000)
    parser.add_argument("--full-queries", type=int, default=2000)
    parser.add_argument("--skip-hashes", action="store_true", help="Only for quick local debugging")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    started = time.time()
    sources = []
    for name, size in FILES.items():
        path = data / name
        if path.stat().st_size != size:
            raise ValueError(f"Input size mismatch for {path}")
        digest = None if args.skip_hashes else sha256(path)
        if digest is not None and digest != EXPECTED_SHA256[name]:
            raise ValueError(f"Source SHA-256 mismatch for {name}: {digest}")
        sources.append({"file": name, "url": SOURCE + name, "bytes": size, "sha256": digest})
    log("Source sizes and hashes verified")
    base = matrix(data / "base.10M.u8bin", "u1")
    queries = matrix(data / "query.public.100K.u8bin", "u1")
    bn, _, bptr, bidx = sparse(data / "base.metadata.10M.spmat")
    qn, _, qptr, qidx = sparse(data / "query.metadata.public.100K.spmat")
    if base.shape != (bn, 192) or queries.shape != (qn, 192):
        raise ValueError("Vector/metadata dimensions differ")
    selected, frequencies, single_rows, single_labels, query_counts = select_labels(bidx, qptr, qidx, args.max_labels)
    log(f"Selected {len(selected)} natural labels; {len(single_rows)} original one-tag queries")
    chosen_positions = np.flatnonzero(np.isin(bidx, selected, kind="table"))
    member_labels = np.asarray(bidx[chosen_positions])
    member_ids = (np.searchsorted(bptr, chosen_positions, side="right") - 1).astype("<u4")
    del chosen_positions
    full_gt_path = data / "GT.public.ibin"
    gt_n, gt_k = map(int, np.fromfile(full_gt_path, dtype="<u4", count=2))
    if (gt_n, gt_k) != (100000, 10):
        raise ValueError("Unexpected published ground truth header")
    full_i = np.memmap(full_gt_path, mode="r", dtype="<i4", offset=8, shape=(gt_n, gt_k))
    full_d = np.memmap(full_gt_path, mode="r", dtype="<f4", offset=8 + 4 * gt_n * gt_k, shape=(gt_n, gt_k))
    for count in args.sizes:
        if not 0 < count <= len(base) or count % 1000000:
            raise ValueError("Sizes must be whole millions in [1M,10M]")
        out = data / f"prepared-{count // 1000000}m"
        out.mkdir(parents=True, exist_ok=True)
        limit = args.full_queries if count == len(base) else args.pilot_queries
        query_ids, query_labels = select_queries(selected, single_rows, single_labels, limit)
        write_matrix(out / "query.u8bin", queries[query_ids], "u1")
        write_matrix(out / "query_labels.u32bin", query_labels, "<u4")
        write_matrix(out / "query_ids.u32bin", query_ids, "<u4")
        write_matrix(out / "all_original_single_tag_query_ids.u32bin", single_rows, "<u4")
        write_matrix(out / "held_out_and_query_ids.u32bin", np.flatnonzero(np.diff(qptr) == 2), "<u4")
        np.savetxt(out / "query_labels.txt", query_labels, fmt="%d")
        subset_end = int(np.searchsorted(member_ids, count))
        ids, labels = member_ids[:subset_end], member_labels[:subset_end]
        postings = {label: ids[labels == label] for label in selected}
        for label, members in postings.items():
            if len(members) < 10 or np.any(members[1:] <= members[:-1]):
                raise ValueError(f"Invalid or undersized label {label} postings")
            write_matrix(out / f"label_{label}.u32bin", members, "<u4")
        log(f"Writing {count} C++ label rows")
        write_labels(out / "labels.txt", ids, labels, count)
        target_base = out / "base.u8bin"
        if count == len(base):
            if not target_base.exists():
                target_base.symlink_to(Path("../base.10M.u8bin"))
        else:
            copy_prefix(data / "base.10M.u8bin", target_base, count, 192)
        if count == len(base):
            truth_i, truth_d = np.array(full_i[query_ids]), np.array(full_d[query_ids])
            truth_source = "Published full-10M exact filtered top-10, selected query rows"
        else:
            truth_i, truth_d = exact_truth(base[:count], queries[query_ids], query_labels, postings, gt_k)
            truth_source = "Recomputed exact squared L2 on the prefix; float64 arithmetic, id tie-break"
        for row, label in enumerate(query_labels):
            members = postings[int(label)]
            positions = np.searchsorted(members, truth_i[row])
            if np.any(positions >= len(members)) or not np.array_equal(members[positions], truth_i[row]):
                raise ValueError(f"Ground truth violates label for query row {row}")
        with (out / "groundtruth.bin").open("wb") as dst:
            dst.write(struct.pack("<II", len(query_ids), gt_k))
            np.asarray(truth_i, dtype="<i4").tofile(dst)
            np.asarray(truth_d, dtype="<f4").tofile(dst)
        label_records = [{"original_label": label, "cpp_input_label": label,
                          "base_count": len(postings[label]), "full_10m_count": int(frequencies[label]),
                          "match_fraction": len(postings[label]) / count,
                          "available_one_tag_queries": int(query_counts[label]),
                          "selected_queries": int(np.sum(query_labels == label))} for label in selected]
        manifest = {"dataset": "BigANN NeurIPS23 YFCC10M filtered", "license": "CC BY 4.0",
                    "attribution": "https://big-ann-benchmarks.com/neurips23.html", "sources": sources,
                    "seed": SEED, "base_vectors": count, "dimension": 192, "dtype": "uint8",
                    "metric": "L2", "k": gt_k, "query_count": len(query_ids), "groundtruth": truth_source,
                    "original_one_tag_queries": len(single_rows), "held_out_two_tag_and_queries": int(np.sum(np.diff(qptr) == 2)),
                    "label_selection": f"Up to {args.max_labels} labels balanced across match fractions[.0001,.001,.01,.1,.2], >=10 original one-tag queries each",
                    "query_selection": "Seeded round-robin labels, then seeded shuffled query order",
                    "sentinel": {"label": SENTINEL, "meaning": "No selected natural labels; never queried; not universal"},
                    "labels": label_records,
                    "artifacts": {name: {"bytes": (out / name).stat().st_size,
                                         "sha256": sha256(out / name)}
                                  for name in ("query.u8bin", "query_labels.u32bin", "query_ids.u32bin",
                                               "groundtruth.bin", "labels.txt", "query_labels.txt")}}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (out / "label_counts.json").write_text(json.dumps(label_records, indent=2) + "\n")
        results = data.parent / "results"
        results.mkdir(parents=True, exist_ok=True)
        (results / f"dataset-{count // 1000000}m-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        log(f"READY {out}: {count} vectors, {len(query_ids)} queries, {len(selected)} labels")
    log(f"Finished in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
