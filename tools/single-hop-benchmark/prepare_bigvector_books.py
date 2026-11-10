#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Prepare faithful AmazonBooks range predicates and an explicit reusable bank.

The original workload materializes each published five-field predicate as an
atomic label. The optional reusable bank takes parameters only from base rows,
then independently assigns query vectors to those known predicates. Neither is
an implementation of arbitrary range search. No ANN results enter preparation.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
import urllib.request

DATASET = "Patrickcode/BigVectorBench"
REVISION = "71b4af4d473cdcd0e149f349734fb68a312a487a"
FILENAME = "amazon-384-euclidean-5filter.hdf5"
SOURCE_BYTES = 24810848512
SOURCE_SHA256 = "a1c5b2c1294c05d92046345aa2558d1cc51d71883613ec46b3644b69e54cad6b"
URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{FILENAME}"
SEED = 20261026
SENTINEL = 200386
NAMES = ["unixtime", "str_len", "asin", "overall", "vote"]
OFFICIAL_EXPRESSION = '''
def filter_expr(unixtime, str_len, asin, overall, vote):
    head_over = [0, 1, 1, 3, 3, 4]
    tail_over = [0, 2, 2, 4, 5, 5]
    asin_min = 1048767
    asin_max = 2147483647
    asin_scp = asin_max - asin_min
    unixtime_head = unixtime - 7 * 24 * 60 * 60
    unixtime_tail = unixtime
    str_len_head = str_len - 30
    str_len_tail = str_len + 30
    asin_head = int(asin - 0.05 * asin_scp)
    asin_tail = int(asin + 0.05 * asin_scp)
    overall_head = head_over[overall]
    overall_tail = tail_over[overall]
    vote_head = int(vote // 10)
    return f"unixtime >= {unixtime_head} and unixtime <= {unixtime_tail} and str_len >= {str_len_head} and str_len <= {str_len_tail} and asin >= {asin_head} and asin <= {asin_tail} and overall >= {overall_head} and overall <= {overall_tail} and vote >= {vote_head}"
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path, value):
    with Path(path).open("x") as target:
        json.dump(value, target, indent=2, sort_keys=True)
        target.write("\n")


def write_matrix(path, values, dtype):
    import numpy as np
    values = np.asarray(values, dtype=dtype)
    if values.ndim == 1:
        values = values[:, None]
    require(values.ndim == 2, "Only two-dimensional matrices are supported")
    with Path(path).open("xb") as target:
        target.write(struct.pack("<II", *values.shape))
        values.tofile(target)


def matrix(path, dtype="<f4"):
    import numpy as np
    shape = np.fromfile(path, dtype="<u4", count=2)
    require(len(shape) == 2, f"Truncated matrix {path}")
    n, d = map(int, shape)
    require(Path(path).stat().st_size == 8 + n * d * np.dtype(dtype).itemsize, f"Malformed matrix {path}")
    return np.memmap(path, mode="r", dtype=dtype, offset=8, shape=(n, d))


def download(root):
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / FILENAME
    if not path.exists():
        partial = raw / (FILENAME + ".part")
        with urllib.request.urlopen(URL, timeout=180) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, 16 << 20)
        require(partial.stat().st_size == SOURCE_BYTES, "Downloaded source size mismatch")
        partial.rename(path)
    require(path.stat().st_size == SOURCE_BYTES, "Source size mismatch")
    actual = record(path)
    require(actual["sha256"] == SOURCE_SHA256, "Source SHA-256 differs from publisher LFS identity")
    proof = raw / "source-identity.json"
    if not proof.exists():
        write_json(proof, {**actual, "url": URL, "revision": REVISION, "dataset": DATASET})
    print(f"VERIFIED {path}: {SOURCE_BYTES} bytes", flush=True)


def validate_schema(source, strict=True):
    """Validate metadata without executing the embedded Python function."""
    require(source.attrs.get("type") == "filter-ann", "Expected filter-ann HDF5")
    require(source.attrs.get("distance") == "euclidean", "Expected Euclidean source")
    require(list(source.attrs["label_names"]) == NAMES, "Unexpected filter fields/order")
    require(list(source.attrs["label_types"]) == ["int32"] * 5, "Unexpected filter field types")
    supplied = ast.dump(ast.parse(source.attrs["filter_expr_func"]), include_attributes=False)
    expected = ast.dump(ast.parse(OFFICIAL_EXPRESSION), include_attributes=False)
    require(supplied == expected, "Published five-filter expression changed")
    n, d = source["train_vec"].shape
    q = len(source["test_vec"])
    for key, shape, dtype in (("train_vec", (n, d), "float32"), ("test_vec", (q, d), "float32"),
                              ("train_label", (n, 5), "int32"), ("test_label", (q, 5), "int32"),
                              ("neighbors", (q, 100), "int32"), ("distances", (q, 100), "float32")):
        require(source[key].shape == shape and str(source[key].dtype) == dtype, f"Malformed source dataset {key}")
    if strict:
        require((n, d, q) == (15928208, 384, 10000), "Unexpected full AmazonBooks release shape")


def predicate(parameters):
    """Return inclusive bounds with exactly the publisher's integer rounding."""
    u, length, asin, rating, vote = map(int, parameters)
    require(1 <= rating <= 5, "Review rating must lie in [1,5]")
    span = 0.05 * (2147483647 - 1048767)
    low = [u - 604800, length - 30, int(asin - span), [0, 1, 1, 3, 3, 4][rating], vote // 10]
    high = [u, length + 30, int(asin + span), [0, 2, 2, 4, 5, 5][rating], 2147483647]
    return {"parameters": [u, length, asin, rating, vote], "low_inclusive": low,
            "high_inclusive": high, "field_order": NAMES}


class PredicateIndex:
    """Use the narrow time range to avoid scanning 15.9M rows per predicate."""

    def __init__(self, labels):
        import numpy as np
        require(labels.ndim == 2 and labels.shape[1] == 5, "Expected five metadata fields")
        self.labels = labels
        self.order = np.argsort(labels[:, 0], kind="stable")
        self.times = labels[self.order, 0]

    def members(self, spec):
        import numpy as np
        low, high = spec["low_inclusive"], spec["high_inclusive"]
        first = np.searchsorted(self.times, low[0])
        last = np.searchsorted(self.times, high[0], side="right")
        ids = self.order[first:last]
        values = self.labels[ids]
        keep = np.ones(len(ids), dtype=bool)
        for column in range(1, 5):
            keep &= (values[:, column] >= low[column]) & (values[:, column] <= high[column])
        return np.sort(ids[keep]).astype("<u4")


def fixed_bank(labels, count=64, candidates=2048, minimum=100):
    """Choose reusable predicates from base metadata only, spanning cardinalities."""
    import numpy as np
    require(0 < count <= candidates <= len(labels), "Invalid bank sample sizes")
    index = PredicateIndex(labels)
    generator = np.random.Generator(np.random.PCG64(SEED))
    sampled = generator.choice(len(labels), size=candidates, replace=False)
    choices, seen = [], set()
    for row in sampled:
        spec = predicate(labels[row])
        signature = tuple(spec["low_inclusive"] + spec["high_inclusive"])
        if signature in seen:
            continue
        seen.add(signature)
        members = index.members(spec)
        if len(members) >= minimum:
            tie = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
            choices.append((len(members), tie, int(row), spec, members))
    require(len(choices) >= count, "Insufficient eligible base-only bank candidates")
    choices.sort(key=lambda entry: entry[:2])
    positions = [((2 * i + 1) * len(choices)) // (2 * count) for i in range(count)]
    bank = []
    for label, position in enumerate(positions, 1):
        _, _, row, spec, members = choices[position]
        bank.append({"label": label, "source_base_row": row, "predicate": spec, "members": members})
    return bank


def rank(scope, value):
    return hashlib.sha256(f"{SEED}:{scope}:{value}".encode()).digest()


def grouped_split(queries, attributes):
    """Keep exact embedding duplicates and numeric-ASIN families in one split."""
    parent = list(range(len(queries)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    asins, embeddings, duplicate = {}, {}, []
    for row, (vector, metadata) in enumerate(zip(queries, attributes)):
        asin = int(metadata[2])
        if asin in asins:
            union(row, asins[asin])
        asins.setdefault(asin, row)
        digest = hashlib.sha256(vector.tobytes()).hexdigest()
        if digest in embeddings:
            original = embeddings[digest]
            union(row, original)
            duplicate.append({"query_id": row, "retained_query_id": original, "embedding_sha256": digest})
        embeddings.setdefault(digest, row)
    groups = [find(i) for i in range(len(queries))]
    split = ["calibration" if int.from_bytes(rank("group-split", g)[:8], "little") % 4 == 0 else "validation" for g in groups]
    return groups, split, duplicate


def export(root):
    import h5py
    import numpy as np
    raw = root / "raw" / FILENAME
    require(raw.stat().st_size == SOURCE_BYTES and sha256(raw) == SOURCE_SHA256, "Unauthenticated AmazonBooks release")
    out = root / "export"
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    with h5py.File(raw, "r") as source:
        validate_schema(source)
        for key, filename in (("train_vec", "base.fbin"), ("test_vec", "query.fbin"),
                              ("train_label", "base_attributes.i32bin"), ("test_label", "query_attributes.i32bin")):
            dataset = source[key]
            with (out / filename).open("xb") as target:
                target.write(struct.pack("<II", *dataset.shape))
                for first in range(0, len(dataset), 10000):
                    block = np.asarray(dataset[first:first + 10000], dtype="<f4" if "vec" in key else "<i4")
                    require(np.all(np.isfinite(block)), f"Non-finite values in {key}")
                    block.tofile(target)
        queries = matrix(out / "query.fbin")
        norms = np.linalg.norm(np.asarray(queries, dtype=np.float64), axis=1)
        write_json(out / "export-identity.json", {
            "source": {**record(raw), "url": URL, "revision": REVISION}, "elapsed_seconds": time.monotonic() - started,
            "preexisting_embeddings": True, "publisher_model_family": "sentence-transformers/all-MiniLM-L12-v2",
            "historical_encoder_revision_and_max_length": "not published; do not claim exact encoder parity with newly generated C4",
            "embedding_transformation": "none; exact stored float32 coordinates retained",
            "query_norm_min_max": [float(norms.min()), float(norms.max())],
            "schema_expression_ast_sha256": hashlib.sha256(ast.dump(ast.parse(OFFICIAL_EXPRESSION)).encode()).hexdigest(),
            "artifacts": {name: record(out / name) for name in ("base.fbin", "query.fbin", "base_attributes.i32bin", "query_attributes.i32bin")},
            "preparation_source": record(Path(__file__))})
    print("EXPORT COMPLETE", flush=True)


def write_point_labels(path, postings, rows, chunk=1000000):
    import numpy as np
    overlap = np.zeros(rows, dtype="<u4")
    for members in postings.values():
        overlap[members] += 1
    with path.open("x", buffering=1 << 20) as target:
        for first in range(0, rows, chunk):
            last = min(first + chunk, rows)
            point_parts, label_parts = [], []
            for label, members in postings.items():
                lo, hi = np.searchsorted(members, [first, last])
                ids = members[lo:hi]
                if len(ids):
                    point_parts.append(ids)
                    label_parts.append(np.full(len(ids), label, dtype="<u4"))
            if not point_parts:
                target.write(f"{SENTINEL}\n" * (last - first))
                continue
            points, labels = np.concatenate(point_parts), np.concatenate(label_parts)
            order = np.lexsort((labels, points))
            points, labels = points[order], labels[order]
            edges = np.r_[0, np.flatnonzero(points[1:] != points[:-1]) + 1, len(points)]
            previous = first - 1
            for a, b in zip(edges[:-1], edges[1:]):
                row = int(points[a])
                target.write(f"{SENTINEL}\n" * (row - previous - 1))
                target.write(",".join(map(str, labels[a:b])) + "\n")
                previous = row
            target.write(f"{SENTINEL}\n" * (last - previous - 1))
    return {"memberships": int(sum(map(len, postings.values()))), "covered_rows": int(np.count_nonzero(overlap)),
            "unqueried_sentinel_rows": int(np.count_nonzero(overlap == 0)),
            "max_row_memberships": int(overlap.max()), "sentinel_label": SENTINEL,
            "unlabeled_rust_rows": "sentinel skipped when constructing label graphs; C++ may index its unqueried label"}


def seal(root, workload):
    import numpy as np
    exported = root / "export"
    identity = json.loads((exported / "export-identity.json").read_text())
    for name, spec in identity["artifacts"].items():
        require(sha256(exported / name) == spec["sha256"], f"Exported source changed: {name}")
    base = matrix(exported / "base.fbin")
    queries = matrix(exported / "query.fbin")
    attributes = matrix(exported / "base_attributes.i32bin", "<i4")
    query_attributes = matrix(exported / "query_attributes.i32bin", "<i4")
    groups, splits, duplicates = grouped_split(queries, query_attributes)
    duplicate_ids = {entry["query_id"] for entry in duplicates}
    out = root / ("sealed-" + workload)
    out.mkdir(exist_ok=False)
    (out / "base.fbin").symlink_to(os.path.relpath(exported / "base.fbin", out))
    excluded = []
    if workload == "original":
        index = PredicateIndex(attributes)
        bank = []
        for row in range(len(queries)):
            if row in duplicate_ids:
                continue
            spec = predicate(query_attributes[row])
            members = index.members(spec)
            if len(members) < 10:
                excluded.append({"query_id": row, "reason": "fewer_than_k_eligible_points", "eligible_points": len(members)})
            else:
                bank.append({"label": row + 1, "source_query_row": row, "predicate": spec, "members": members})
        assignments = {entry["source_query_row"]: entry["label"] for entry in bank}
        semantics = "Original query-specific five-field range conjunctions, exactly materialized as atomic labels. All predicate definitions are known at index construction, including future query predicates. This measures a specialized known-predicate index, not arbitrary range filtering. Each predicate has one distinct query vector; no population per-label p99 can be inferred."
    else:
        bank = fixed_bank(attributes)
        available = sorted((row for row in range(len(queries)) if row not in duplicate_ids), key=lambda row: rank("bank-assignment", row))
        assignments = {row: bank[position % len(bank)]["label"] for position, row in enumerate(available)}
        semantics = "Constructed reusable 64-predicate bank: every five-field conjunction uses the unchanged publisher formula, with parameters selected only from base metadata; query vectors are independently assigned round-robin in fixed SHA256 order. Query-predicate coupling differs from the original benchmark. No queries or ANN outcomes selected the predicate definitions."
    postings = {entry["label"]: entry["members"] for entry in bank}
    for label, members in postings.items():
        require(len(members) >= 10 and np.all(members[1:] > members[:-1]), f"Invalid posting list {label}")
        write_matrix(out / f"label_{label}.u32bin", members, "<u4")
    labels = []
    for entry in bank:
        labels.append({**{key: value for key, value in entry.items() if key != "members"},
                       "original_label": entry["label"], "cpp_input_label": entry["label"],
                       "base_count": len(entry["members"]), "selectivity": len(entry["members"]) / len(base),
                       "posting": record(out / f"label_{entry['label']}.u32bin")})
    membership = write_point_labels(out / "labels.txt", postings, len(base))
    require(membership["max_row_memberships"] <= 128, "Protected label edges may exceed R128; protocol needs revision before building")
    write_json(out / "duplicate-queries.json", duplicates)
    write_json(out / "metadata-exclusions.json", excluded)
    write_json(out / "group-assignments.json", [{"query_id": row, "group": groups[row], "split": splits[row], "asin": int(query_attributes[row, 2]), "retained": row in assignments} for row in range(len(queries))])
    outputs = {}
    for split in ("calibration", "validation"):
        selected = sorted((row for row in assignments if splits[row] == split), key=lambda row: rank("query-order-" + split, row))
        prepared = out / ("prepared-" + split)
        prepared.mkdir()
        for name in ("base.fbin", "labels.txt", *(f"label_{label}.u32bin" for label in postings)):
            (prepared / name).symlink_to(os.path.relpath(out / name, prepared))
        write_matrix(prepared / "query.fbin", queries[selected], "<f4")
        write_matrix(prepared / "query_ids.u32bin", selected, "<u4")
        qlabels = [assignments[row] for row in selected]
        write_matrix(prepared / "query_labels.u32bin", qlabels, "<u4")
        (prepared / "query_labels.txt").write_text("".join(f"{label}\n" for label in qlabels))
        with (prepared / "query_metadata.jsonl").open("x") as target:
            for row in selected:
                target.write(json.dumps({"query_id": row, "group": groups[row], "label": assignments[row], "original_filter_parameters": query_attributes[row].tolist()}) + "\n")
        outputs[split] = {"path": str(prepared), "query_count": len(selected), "per_label_counts": dict(Counter(qlabels)),
                          "artifacts": {name: record(prepared / name) for name in ("query.fbin", "query_ids.u32bin", "query_labels.u32bin", "query_labels.txt", "query_metadata.jsonl")}}
    proof = {"schema_version": 1, "dataset": DATASET, "revision": REVISION, "workload": workload,
             "seed": SEED, "filter_semantics": semantics, "labels": labels, "membership": membership,
             "base_vectors": len(base), "dimension": base.shape[1], "dtype": "float32", "metric": "L2", "k": 10,
             "partition_rule": "Union query rows sharing numeric-ASIN family or exact float32 embedding; fixed-seed SHA256 component-min-ID modulo4=0 calibration, otherwise validation; first exact duplicate retained globally",
             "fresh_validation_queries_searched": False, "ann_results_used": False, "source_truth_used_to_select_queries": False,
             "duplicates": record(out / "duplicate-queries.json"), "metadata_exclusions": record(out / "metadata-exclusions.json"),
             "group_assignments": record(out / "group-assignments.json"), "source_export": record(exported / "export-identity.json"),
             "artifacts": {name: record(out / name) for name in ("base.fbin", "labels.txt")}, "outputs": outputs,
             "bank_sampling": None if workload == "original" else {"source": "base metadata only", "candidate_rows": 2048, "selected": 64, "minimum_members": 100, "generator": "NumPy PCG64 fixed seed; uniform without replacement; choose 64 evenly spaced cardinality-ranked candidates"},
             "fixed_L": 64,
             "calibration_L_grid": [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 8192],
             "ninety_percent_recall": "Choose method-specific smallest measured L reaching90% mixed calibration recall; freeze before validation; report achieved validation recall without retuning",
             "preparation_source": record(Path(__file__))}
    write_json(out / "split-commitment.json", proof)
    print("FRESH SPLIT SEALED " + json.dumps({key: value["query_count"] for key, value in outputs.items()}), flush=True)


def truth(root, workload, args):
    import numpy as np
    # This same tested exhaustive float32-vector oracle is used for C4.
    from prepare_amazon_c4 import audit_truth_direct, exact_truth
    sealed = root / ("sealed-" + workload)
    commitment = json.loads((sealed / "split-commitment.json").read_text())
    require(sha256(sealed / "base.fbin") == commitment["artifacts"]["base.fbin"]["sha256"], "Sealed base changed")
    base = matrix(sealed / "base.fbin")
    postings = {}
    for entry in commitment["labels"]:
        path = sealed / f"label_{entry['label']}.u32bin"
        require(sha256(path) == entry["posting"]["sha256"], "Sealed predicate membership changed")
        postings[entry["label"]] = matrix(path, "<u4").ravel()
    for split in ("calibration", "validation"):
        prepared = sealed / ("prepared-" + split)
        for name, spec in commitment["outputs"][split]["artifacts"].items():
            require(sha256(prepared / name) == spec["sha256"], "Sealed queries changed")
        queries = matrix(prepared / "query.fbin")
        qlabels = matrix(prepared / "query_labels.u32bin", "<u4").ravel()
        started = time.monotonic()
        ids, distances, pairs = exact_truth(base, queries, qlabels, postings, 10, args.query_block, args.base_block)
        for row, label in enumerate(qlabels):
            members = postings[int(label)]
            positions = np.searchsorted(members, ids[row])
            require(np.all(positions < len(members)) and np.array_equal(members[positions], ids[row]),
                    "Computed truth violates predicate membership")
        audit = audit_truth_direct(base, queries, qlabels, postings, ids, distances, args.audit_per_label)
        with (prepared / "groundtruth.bin").open("xb") as target:
            target.write(struct.pack("<II", len(queries), 10))
            np.asarray(ids, dtype="<i4").tofile(target)
            np.asarray(distances, dtype="<f4").tofile(target)
        write_matrix(prepared / "groundtruth_distances.f64bin", distances, "<f8")
        labels = [{**entry, "selected_queries": int(np.count_nonzero(qlabels == entry["label"]))} for entry in commitment["labels"]]
        manifest = {"dataset": "BigVectorBench AmazonBooks", "workload": workload, "base_vectors": len(base),
                    "dimension": base.shape[1], "dtype": "float32", "metric": "L2", "k": 10, "query_count": len(queries),
                    "source_query_namespace": f"{DATASET}@{REVISION}", "labels": labels, "split": split,
                    "filter_semantics": commitment["filter_semantics"], "query_selection": commitment["partition_rule"],
                    "split_commitment": record(sealed / "split-commitment.json"),
                    "groundtruth": "Exhaustive full-posting squared L2 over original stored float32 coordinates; float64 accumulation; distance then ID ordering; float32 distance export; source published top100 unused",
                    "truth_evidence": {"evaluated_pairs": pairs, "elapsed_seconds": time.monotonic() - started,
                                       "float64_distances": record(prepared / "groundtruth_distances.f64bin"),
                                       "oracle_source": record(Path(__file__).with_name("prepare_amazon_c4.py")),
                                       "independent_direct_coordinate_audit": audit,
                                       "all_truth_ids_membership_checked": True},
                    "preparation_source": record(Path(__file__)),
                    "artifacts": {**{name: record(prepared / name) for name in ("base.fbin", "query.fbin", "query_ids.u32bin", "query_labels.u32bin", "groundtruth.bin", "labels.txt", "query_labels.txt", "query_metadata.jsonl")},
                                  **{f"label_{entry['label']}.u32bin": entry["posting"] for entry in labels}}}
        write_json(prepared / "manifest.json", manifest)
        write_json(prepared / "label_counts.json", labels)
        print(f"READY {prepared}: {len(queries)} queries, {pairs} exhaustive distances", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=["download", "export", "seal", "truth"], required=True)
    parser.add_argument("--workload", choices=["original", "bank"], default="original")
    parser.add_argument("--query-block", type=int, default=64)
    parser.add_argument("--base-block", type=int, default=16384)
    parser.add_argument("--audit-per-label", type=int, default=2)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    require(args.query_block > 0 and args.base_block > 0 and args.audit_per_label > 0, "Positive truth block/audit sizes required")
    if args.stage == "download":
        download(args.root)
    elif args.stage == "export":
        export(args.root)
    elif args.stage == "seal":
        seal(args.root, args.workload)
    else:
        truth(args.root, args.workload, args)


if __name__ == "__main__":
    main()
