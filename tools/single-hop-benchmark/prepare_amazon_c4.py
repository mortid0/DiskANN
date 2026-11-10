#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Pinned Amazon-C4 acquisition, fp32 embedding, grouped holdout, and exact truth.

Stages are separate so embedding/build work can be scheduled without overlapping
latency measurements. No ANN results are read by any preparation stage.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import time
import unicodedata
import urllib.request

DATASET = "McAuley-Lab/Amazon-C4"
DATA_REVISION = "39322697749a88d179f88d322a2fe4765b655c98"
MODEL = "sentence-transformers/all-MiniLM-L12-v2"
MODEL_REVISION = "a50ef00143b4d5391434df20ae11632588ac25be"
SEED = 20261026
DIMENSION = 384
MAX_LENGTH = 256
SOURCE_BASE_COUNT = 1058417
SOURCE_QUERY_COUNT = 21223
RAW_FILES = {
    "sampled_item_metadata_1M.jsonl": (643452148, "c307b1017e0ff15a799e479a91ad59b55df91fd1b52bf1ea4dee01aed944cc74", None),
    "test.csv": (12525153, "1372e8659ca475c3e08561ed09042837891ae0f7151650d38e5c6e93858e4186", None),
}
# Git blob IDs bind the ordinary files; LFS SHA-256 binds the fp32 weights.
MODEL_FILES = {
    "model.safetensors": (133466304, "d2d541e5f101695ae495eacd867a8d025ecfe8f9674fb23aa6cf93cdb60a5542", None),
    "config.json": (615, None, "997449989924d15c7b98619a97c899740c99854d"),
    "tokenizer.json": (466247, None, "cb202bfe2e3c98645018a6d12f182a434c9d3e02"),
    "tokenizer_config.json": (352, None, "160e50cfb71e8ec988666144435831d7257cd3b8"),
    "special_tokens_map.json": (112, None, "e7b0375001f109a6b8873d756ad4f7bbb15fbaa5"),
    "vocab.txt": (231508, None, "fb140275c155a9c7c5a3b3e0e77a9e839594a938"),
    "sentence_bert_config.json": (53, None, "5fd10429389515d3e5cccdeda08cae5fea1ae82e"),
    "1_Pooling/config.json": (190, None, "d1514c3162bbe87b343f565fadc62e6c06f04f03"),
    "modules.json": (349, None, "952a9b81c0bfd99800fabf352f69c7ccd46c5e43"),
}


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


def write_json(path, value):
    with Path(path).open("x") as target:
        json.dump(value, target, indent=2, sort_keys=True)
        target.write("\n")


def checked_download(url, path, spec):
    size, digest, blob = spec
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=180) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, 8 << 20)
        require(temporary.stat().st_size == size, f"Download size mismatch: {path}")
        temporary.rename(path)
    require(path.stat().st_size == size, f"Source size mismatch: {path}")
    actual = record(path)
    if digest:
        require(actual["sha256"] == digest, f"Source SHA-256 mismatch: {path}")
    if blob:
        value = path.read_bytes()
        actual_blob = hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()
        require(actual_blob == blob, f"Source git-blob mismatch: {path}")
    print(f"verified {path.name}: {actual['bytes']} bytes", flush=True)
    return {**actual, "url": url, "git_blob": blob}


def download(root):
    sources = {}
    for name, spec in RAW_FILES.items():
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REVISION}/{name}"
        sources[name] = checked_download(url, root / "raw" / name, spec)
    model = {}
    for name, spec in MODEL_FILES.items():
        url = f"https://huggingface.co/{MODEL}/resolve/{MODEL_REVISION}/{name}"
        model[name] = checked_download(url, root / "model" / name, spec)
    identity = {"dataset": DATASET, "dataset_revision": DATA_REVISION,
                "model": MODEL, "model_revision": MODEL_REVISION,
                "raw": sources, "model_files": model, "preparation_source": record(Path(__file__))}
    target = root / "download-identity.json"
    if target.exists():
        prior = json.loads(target.read_text())
        require(prior["raw"] == sources and prior["model_files"] == model, "Existing download identity differs")
    else:
        write_json(target, identity)


def canonical_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def rank(scope, value):
    return hashlib.sha256(f"{SEED}:{scope}:{value}".encode()).digest()


def metadata(root):
    identity = json.loads((root / "download-identity.json").read_text())
    for name in RAW_FILES:
        require(sha256(root / "raw" / name) == identity["raw"][name]["sha256"], "Raw source modified")
    raw = root / "raw" / "sampled_item_metadata_1M.jsonl"
    categories, all_products, excluded_products = set(), set(), []
    valid_products = []
    with raw.open() as source:
        for source_id, line in enumerate(source):
            row = json.loads(line)
            require(all(isinstance(row.get(key), str) and row[key].strip() for key in ("item_id", "category")), "Invalid product identity/category")
            require(isinstance(row.get("metadata"), str), "Non-string product metadata")
            require(row["item_id"] not in all_products, "Duplicate product parent ASIN")
            all_products.add(row["item_id"])
            if not row["metadata"].strip():
                excluded_products.append({"source_row": source_id, "item_id": row["item_id"], "category": row["category"], "reason": "whitespace-only product text"})
                continue
            categories.add(row["category"])
            valid_products.append((row["item_id"], row["category"], source_id))
    require(len(all_products) == SOURCE_BASE_COUNT, f"Unexpected source product count: {len(all_products)}")
    labels = {category: i + 1 for i, category in enumerate(sorted(categories))}
    products = {item: (i, labels[category]) for i, (item, category, _) in enumerate(valid_products)}
    counts = {label: 0 for label in labels.values()}
    for _, label in products.values():
        counts[label] += 1
    queries, duplicate_text, excluded_queries, seen_qids, seen_text = [], [], [], set(), {}
    with (root / "raw" / "test.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            qid = int(row["qid"])
            require(qid not in seen_qids and row["item_id"] in all_products, "Duplicate qid or missing positive product in raw corpus")
            seen_qids.add(qid)
            if row["item_id"] not in products:
                excluded_queries.append({"qid": qid, "positive_item_id": row["item_id"], "reason": "positive product excluded for whitespace-only text"})
                continue
            text = row["query"]
            canonical = canonical_text(text)
            require(canonical, "Empty query text")
            if canonical in seen_text:
                duplicate_text.append({"qid": qid, "retained_qid": seen_text[canonical]})
            else:
                seen_text[canonical] = qid
            product_id, label = products[row["item_id"]]
            queries.append({"id": len(queries), "qid": qid, "label": label, "text": text,
                            "positive_item_id": row["item_id"], "positive_base_id": product_id,
                            "user_hash": hashlib.sha256(row["user_id"].encode()).hexdigest(),
                            "text_hash": hashlib.sha256(canonical.encode()).hexdigest()})
    require(len(seen_qids) == SOURCE_QUERY_COUNT and seen_qids == set(range(SOURCE_QUERY_COUNT)), "Unexpected official query IDs")
    # All content validation precedes output creation. An interrupted write leaves
    # a separate staging directory; completed metadata is atomically installed.
    out = root / "metadata"
    if out.exists() and not any(out.iterdir()):
        out.rmdir()
    require(not out.exists(), "Completed or nonempty metadata output already exists")
    stage = root / f"metadata.staging-{os.getpid()}"
    stage.mkdir(exist_ok=False)
    with raw.open() as source, (stage / "products.jsonl").open("x") as corpus, (stage / "labels.txt").open("x") as label_file:
        retained = 0
        for source_id, line in enumerate(source):
            row = json.loads(line)
            if row["item_id"] not in products:
                continue
            i, label = products[row["item_id"]]
            require(i == retained, "Stable product remap is inconsistent")
            retained += 1
            corpus.write(json.dumps({"id": i, "source_id": source_id, "item_id": row["item_id"], "label": label, "text": row["metadata"]}, ensure_ascii=False) + "\n")
            label_file.write(f"{label}\n")
    with (stage / "queries.jsonl").open("x") as target:
        target.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in queries)
    write_json(stage / "excluded-products.json", excluded_products)
    write_json(stage / "excluded-queries.json", excluded_queries)
    artifacts = {}
    for name in ("products.jsonl", "queries.jsonl", "labels.txt", "excluded-products.json", "excluded-queries.json"):
        artifacts[name] = {**record(stage / name), "path": str(out / name)}
    write_json(stage / "metadata.json", {"dataset": DATASET, "dataset_revision": DATA_REVISION,
               "source_base_count": len(all_products), "base_count": len(products), "source_query_count": len(seen_qids),
               "retained_source_query_count": len(queries), "unique_text_query_count": len(seen_text),
               "labels": [{"label": label, "category": cat, "base_count": counts[label], "match_fraction": counts[label] / len(products)} for cat, label in labels.items()],
               "duplicate_query_texts": duplicate_text,
               "empty_text_policy": "Exclude whitespace-only metadata products; preserve raw source row and parentASIN mapping; exclude queries whose positive product was removed; never substitute IDs for text",
               "excluded_base_count": len(excluded_products), "excluded_query_count": len(excluded_queries),
               "filter_semantics": "One atomic category request, inferred from the query's known positive product; constructed predicate, not an observed user-supplied category",
               "artifacts": artifacts, "download_identity": record(root / "download-identity.json")})
    stage.rename(out)
    print(f"METADATA READY: {len(products)} products, {len(queries)} queries, {len(labels)} categories; excluded {len(excluded_products)} empty-text products and {len(excluded_queries)} queries", flush=True)


def matrix(path, dtype="<f4"):
    import numpy as np
    path = Path(path)
    shape = np.fromfile(path, dtype="<u4", count=2)
    require(len(shape) == 2, f"Truncated matrix: {path}")
    n, d = map(int, shape)
    require(n > 0 and d > 0 and path.stat().st_size == 8 + n * d * np.dtype(dtype).itemsize, f"Malformed matrix: {path}")
    return np.memmap(path, mode="r", dtype=dtype, offset=8, shape=(n, d))


def write_matrix(path, values, dtype):
    import numpy as np
    values = np.asarray(values, dtype=dtype)
    if values.ndim == 1:
        values = values[:, None]
    require(values.ndim == 2, "Expected matrix")
    with Path(path).open("xb") as target:
        target.write(struct.pack("<II", *values.shape))
        values.tofile(target)


def json_rows(path):
    with Path(path).open() as source:
        for line in source:
            yield json.loads(line)


def embedding_config(root, args):
    import numpy as np
    import torch
    import transformers
    model_identity = json.loads((root / "download-identity.json").read_text())["model_files"]
    for name, spec in model_identity.items():
        require(sha256(root / "model" / name) == spec["sha256"], f"Model changed: {name}")
    return {"model": MODEL, "revision": MODEL_REVISION, "model_files": model_identity,
            "model_dtype": "float32", "output_dtype": "float32", "dimension": DIMENSION,
            "pooling": "attention-mask weighted mean of final hidden states, then float32 L2 normalization",
            "maximum_tokens": MAX_LENGTH, "prefix": "none", "truncation": True,
            "threads": args.threads, "batch_size": args.batch_size, "checkpoint_rows": 4096,
            "device": args.device, "length_bucketing": "stable ascending raw-text length within each4096rowcheckpoint",
            "metadata": record(root / "metadata" / "metadata.json"),
            "source": record(Path(__file__)),
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                         "transformers": transformers.__version__},
            "torch_deterministic_algorithms": True, "allow_tf32": False}


def embed(root, args):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(SEED)
    config = embedding_config(root, args)
    out = root / "embedding"
    out.mkdir(exist_ok=True)
    config_path = out / "configuration.json"
    if config_path.exists():
        require(json.loads(config_path.read_text()) == config, "Embedding resume configuration/source mismatch")
    else:
        write_json(config_path, config)
    tokenizer = AutoTokenizer.from_pretrained(root / "model", local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(root / "model", local_files_only=True, trust_remote_code=False,
                                     torch_dtype=torch.float32).eval().to(args.device)
    require(model.config.hidden_size == DIMENSION, "Unexpected embedding dimension")
    meta = json.loads((root / "metadata" / "metadata.json").read_text())
    results = {}
    for kind, count, filename in (("base", meta["base_count"], "products.jsonl"),
                                   ("query", meta["retained_source_query_count"], "queries.jsonl")):
        target = out / f"{kind}.fbin"
        checkpoints = out / f"{kind}.checkpoints.jsonl"
        if not target.exists():
            with target.open("xb") as handle:
                handle.write(struct.pack("<II", count, DIMENSION))
                handle.truncate(8 + count * DIMENSION * 4)
        current = matrix(target)
        require(current.shape == (count, DIMENSION), "Embedding output shape mismatch")
        del current
        vectors = np.memmap(target, mode="r+", dtype="<f4", offset=8, shape=(count, DIMENSION))
        completed = 0
        if checkpoints.exists():
            for prior in json_rows(checkpoints):
                require(prior["start"] == completed and prior["stop"] <= count,
                        "Noncontiguous embedding checkpoints")
                actual = hashlib.sha256(vectors[prior["start"]:prior["stop"]].tobytes()).hexdigest()
                require(actual == prior["sha256"], "Embedding checkpoint modified")
                completed = prior["stop"]
        stream = iter(json_rows(root / "metadata" / filename))
        for _ in range(completed):
            next(stream)
        for start in range(completed, count, 4096):
            began = time.monotonic()
            stop = min(start + 4096, count)
            rows = [next(stream) for _ in range(stop - start)]
            require([r["id"] for r in rows] == list(range(start, stop)), "Metadata row IDs out of order")
            order = sorted(range(len(rows)), key=lambda i: (len(rows[i]["text"]), i))
            with torch.inference_mode():
                for offset in range(0, len(order), args.batch_size):
                    batch = order[offset:offset + args.batch_size]
                    inputs = tokenizer([rows[i]["text"] for i in batch], max_length=MAX_LENGTH,
                                       padding=True, truncation=True, return_tensors="pt")
                    inputs = {key: value.to(args.device) for key, value in inputs.items()}
                    hidden = model(**inputs).last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                    result = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy().astype("<f4")
                    require(np.all(np.isfinite(result)) and np.allclose(np.linalg.norm(result, axis=1), 1, atol=2e-6),
                            "Invalid normalized embedding")
                    vectors[np.asarray(batch) + start] = result
            vectors.flush()
            digest = hashlib.sha256(vectors[start:stop].tobytes()).hexdigest()
            with checkpoints.open("a") as log:
                log.write(json.dumps({"start": start, "stop": stop, "sha256": digest,
                                      "elapsed_seconds": time.monotonic() - began}) + "\n")
                log.flush()
                os.fsync(log.fileno())
            print(f"EMBED {kind}: {stop}/{count}, {(stop-start)/(time.monotonic()-began):.1f} rows/s", flush=True)
        results[kind] = {**record(target), "checkpoints": record(checkpoints)}
    final = out / "embedding-identity.json"
    if not final.exists():
        write_json(final, {"configuration": record(config_path), "outputs": results,
                           "no_ann_search_performed": True})
    else:
        require(json.loads(final.read_text())["outputs"] == results, "Completed embedding identity mismatch")
    print("EMBEDDINGS COMPLETE", flush=True)


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, row):
        while self.parent[row] != row:
            self.parent[row] = self.parent[self.parent[row]]
            row = self.parent[row]
        return row

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def grouped_split(queries, vectors):
    """Disjoint positive-product, user, normalized-text, and exact-vector groups."""
    require(len(queries) == len(vectors), "Query rows and vectors differ")
    groups = UnionFind(len(queries))
    seen = {kind: {} for kind in ("positive_item_id", "user_hash", "text_hash", "embedding")}
    duplicate_rows = set()
    duplicate_reasons = []
    for i, query in enumerate(queries):
        for kind in seen:
            key = hashlib.sha256(vectors[i].tobytes()).hexdigest() if kind == "embedding" else query[kind]
            if key in seen[kind]:
                previous = seen[kind][key]
                groups.union(i, previous)
                if kind in ("embedding", "text_hash"):
                    duplicate_rows.add(i)
                    duplicate_reasons.append({"row": i, "retained_prior_row": previous, "reason": kind})
            else:
                seen[kind][key] = i
    components = {}
    for i in range(len(queries)):
        components.setdefault(groups.find(i), []).append(i)
    partitions = {"calibration": [], "validation": []}
    assignments = []
    for component in components.values():
        # Rank by complete qid set; group membership cannot depend on ANN output.
        key = ",".join(map(str, sorted(queries[i]["qid"] for i in component)))
        split = "calibration" if int.from_bytes(rank("group-split", key), "big") % 4 == 0 else "validation"
        retained = [i for i in component if i not in duplicate_rows]
        partitions[split].extend(retained)
        assignments.append({"split": split, "source_query_rows": component, "retained_query_rows": retained})
    for split in partitions:
        partitions[split].sort(key=lambda i: rank("order-" + split, queries[i]["qid"]))
    require(partitions["calibration"] and partitions["validation"], "Empty calibration or validation partition")
    return partitions, assignments, duplicate_reasons


def seal(root):
    import numpy as np
    embedding = json.loads((root / "embedding" / "embedding-identity.json").read_text())
    for kind in ("base", "query"):
        require(sha256(root / "embedding" / f"{kind}.fbin") == embedding["outputs"][kind]["sha256"], "Embedding changed")
    out = root / "sealed"
    out.mkdir(exist_ok=False)
    queries = list(json_rows(root / "metadata" / "queries.jsonl"))
    vectors = matrix(root / "embedding" / "query.fbin")
    partitions, assignments, duplicates = grouped_split(queries, vectors)
    write_json(out / "group-assignments.json", assignments)
    write_json(out / "duplicate-queries.json", duplicates)
    metadata_record = json.loads((root / "metadata" / "metadata.json").read_text())
    labels = {entry["label"]: entry for entry in metadata_record["labels"]}
    product_labels = np.loadtxt(root / "metadata" / "labels.txt", dtype="<u4")
    for label in labels:
        write_matrix(out / f"label_{label}.u32bin", np.flatnonzero(product_labels == label), "<u4")
    (out / "base.fbin").symlink_to(os.path.relpath(root / "embedding" / "base.fbin", out))
    (out / "labels.txt").symlink_to(os.path.relpath(root / "metadata" / "labels.txt", out))
    outputs = {}
    for split, rows in partitions.items():
        prepared = out / ("prepared-" + split)
        prepared.mkdir()
        for name in ("base.fbin", "labels.txt", *(f"label_{label}.u32bin" for label in sorted(labels))):
            (prepared / name).symlink_to("../" + name)
        write_matrix(prepared / "query.fbin", vectors[rows], "<f4")
        write_matrix(prepared / "query_ids.u32bin", [queries[i]["qid"] for i in rows], "<u4")
        write_matrix(prepared / "query_labels.u32bin", [queries[i]["label"] for i in rows], "<u4")
        (prepared / "query_labels.txt").write_text("".join(f"{queries[i]['label']}\n" for i in rows))
        with (prepared / "query_metadata.jsonl").open("x") as target:
            target.writelines(json.dumps({key: value for key, value in queries[i].items() if key != "text"}) + "\n" for i in rows)
        counts = {label: sum(queries[i]["label"] == label for i in rows) for label in labels}
        outputs[split] = {"path": str(prepared), "query_count": len(rows), "per_label_counts": counts,
                           "artifacts": {name: record(prepared / name) for name in ("query.fbin", "query_ids.u32bin", "query_labels.u32bin", "query_labels.txt", "query_metadata.jsonl")}}
    protocol = {"schema_version": 1, "dataset": DATASET, "dataset_revision": DATA_REVISION,
                "seed": SEED, "partition_rule": "SHA256 fixed-seed group rank modulo4:0calibration;1,2,3validation",
                "groups": "Connected components sharing positive parentASIN, hashed userID, NFKC/casefold/whitespace-normalized query text, or byte-identical float32 query embedding",
                "duplicates": "Retain first original qid per normalized text or exact embedding globally; all duplicate relationships still join groups before partition",
                "invariants": {"ann_results_used": False, "fresh_validation_queries_searched": False,
                               "base_corpus_shared": True, "query_family_cross_split": False},
                "group_assignments": record(out / "group-assignments.json"),
                "duplicate_queries": record(out / "duplicate-queries.json"),
                "embedding_identity": record(root / "embedding" / "embedding-identity.json"),
                "preparation_source": record(Path(__file__)), "outputs": outputs,
                "labels": list(labels.values()), "metric": "L2", "dtype": "float32", "dimension": DIMENSION,
                "k": 10, "base_vectors": metadata_record["base_count"],
                "filter_semantics": metadata_record["filter_semantics"],
                "ninety_percent_recall": "Choose method-specific smallest measured L reaching90% mixed calibration recall; freeze before validation; report achieved validation recall without retuning",
                "fixed_L": 64,
                "calibration_L_grid": [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 8192],
                "measurement": {"k": 10, "beam": 4, "threads": 8,
                                "holdout_orders": 6, "adaptive_sample": 1000,
                                "adaptive_scale": 16, "adaptive_max_L_multiplier": 16}}
    write_json(out / "split-commitment.json", protocol)
    print("FRESH SPLIT SEALED: " + json.dumps({s: x["query_count"] for s, x in outputs.items()}), flush=True)


def topk_deterministic(distances, ids, k):
    import numpy as np
    part = np.argpartition(distances, k - 1)[:k]
    cutoff = distances[part].max()
    tied = np.flatnonzero(distances <= cutoff)
    order = np.lexsort((ids[tied], distances[tied]))[:k]
    return tied[order]


def exact_truth(base, queries, query_labels, postings, k, query_block=64, base_block=16384):
    """Exhaustive blocked float64 squared-L2 over all eligible float32 vectors."""
    import numpy as np
    ids = np.full((len(queries), k), -1, dtype="<i4")
    distances = np.full((len(queries), k), np.inf, dtype="<f8")
    pairs = 0
    for label in sorted(postings):
        rows = np.flatnonzero(query_labels == label)
        members = postings[label]
        require(len(members) >= k, f"Label {label} has fewer than k matches")
        for first in range(0, len(rows), query_block):
            qrows = rows[first:first + query_block]
            q = np.asarray(queries[qrows], dtype=np.float64)
            qnorm = np.einsum("ij,ij->i", q, q)
            best_d = np.full((len(qrows), k), np.inf)
            best_i = np.full((len(qrows), k), -1, dtype=np.int32)
            for start in range(0, len(members), base_block):
                xids = members[start:start + base_block]
                x = np.asarray(base[xids], dtype=np.float64)
                d = qnorm[:, None] + np.einsum("ij,ij->i", x, x)[None, :] - 2 * (q @ x.T)
                np.maximum(d, 0, out=d)
                candidate_d = np.concatenate((best_d, d), axis=1)
                candidate_i = np.concatenate((best_i, np.broadcast_to(xids, d.shape)), axis=1)
                # The norm/dot identity is only a candidate screen. Include an
                # intentionally conservative error band, then rank candidates by
                # directly accumulated float64 coordinate differences. This also
                # preserves ID ordering for exact duplicates at the cutoff.
                xnorm_max = float(np.einsum("ij,ij->i", x, x).max())
                for row in range(len(qrows)):
                    screen = topk_deterministic(candidate_d[row], candidate_i[row], k)
                    cutoff = candidate_d[row, screen].max()
                    error_band = np.finfo(np.float64).eps * 128 * base.shape[1] * (qnorm[row] + xnorm_max + 1)
                    eligible = np.flatnonzero((candidate_d[row] <= cutoff + error_band) & (candidate_i[row] >= 0))
                    refined_ids = candidate_i[row, eligible]
                    difference = np.asarray(base[refined_ids], dtype=np.float64) - q[row]
                    refined_d = np.einsum("ij,ij->i", difference, difference)
                    take = topk_deterministic(refined_d, refined_ids, min(k, len(refined_ids)))
                    best_d[row].fill(np.inf)
                    best_i[row].fill(-1)
                    best_d[row, :len(take)], best_i[row, :len(take)] = refined_d[take], refined_ids[take]
                pairs += len(qrows) * len(xids)
            ids[qrows], distances[qrows] = best_i, best_d
        print(f"TRUTH label{label}: {len(rows)} queries x {len(members)} members", flush=True)
    require(np.all(ids >= 0) and np.all(np.isfinite(distances)), "Incomplete exhaustive truth")
    return ids, distances, pairs


def audit_truth_direct(base, queries, query_labels, postings, ids, distances, per_label=2):
    """Independent coordinate-difference scan; no norm/dot screening."""
    import numpy as np
    checks, pairs = [], 0
    k = ids.shape[1]
    for label in sorted(postings):
        rows = sorted(np.flatnonzero(query_labels == label), key=lambda i: rank("truth-audit", int(i)))[:per_label]
        members = postings[label]
        for row in rows:
            best_d = np.full(k, np.inf)
            best_i = np.full(k, -1, dtype="<i4")
            for start in range(0, len(members), 16384):
                xids = members[start:start + 16384]
                difference = np.asarray(base[xids], dtype=np.float64) - np.asarray(queries[row], dtype=np.float64)
                d = np.einsum("ij,ij->i", difference, difference)
                candidate_d = np.concatenate((best_d, d))
                candidate_i = np.concatenate((best_i, xids))
                order = np.lexsort((candidate_i, candidate_d))[:k]
                best_d, best_i = candidate_d[order], candidate_i[order]
                pairs += len(xids)
            require(np.array_equal(best_i, ids[row]), f"Independent exact truth ID disagreement at label{label} row{row}")
            require(np.array_equal(best_d, distances[row]), f"Independent exact truth distance disagreement at label{label} row{row}")
            checks.append({"label": int(label), "prepared_query_row": int(row), "eligible_vectors": len(members)})
    return {"method": "independent exhaustive direct float64 coordinate differences; full lexicographic distance/ID sorting per block; no norm/dot screening",
            "queries": checks, "evaluated_pairs": pairs, "all_checked_ids_and_distances_identical": True}


def truth(root, args):
    import numpy as np
    sealed = root / "sealed"
    commitment = json.loads((sealed / "split-commitment.json").read_text())
    base = matrix(sealed / "base.fbin")
    postings = {entry["label"]: matrix(sealed / f"label_{entry['label']}.u32bin", "<u4").ravel() for entry in commitment["labels"]}
    for split in ("calibration", "validation"):
        prepared = sealed / ("prepared-" + split)
        for name, spec in commitment["outputs"][split]["artifacts"].items():
            require(sha256(prepared / name) == spec["sha256"], "Sealed queries modified")
        require(not (prepared / "groundtruth.bin").exists(), "Truth output exists; do not overwrite sealed artifacts")
        queries = matrix(prepared / "query.fbin")
        labels = matrix(prepared / "query_labels.u32bin", "<u4").ravel()
        started = time.monotonic()
        ids, distances, pairs = exact_truth(base, queries, labels, postings, 10, args.query_block, args.base_block)
        for row, label in enumerate(labels):
            members = postings[int(label)]
            positions = np.searchsorted(members, ids[row])
            require(np.all(positions < len(members)) and np.array_equal(members[positions], ids[row]), "Truth predicate violation")
        audit = audit_truth_direct(base, queries, labels, postings, ids, distances, args.audit_per_label)
        write_json(prepared / "truth-audit.json", audit)
        with (prepared / "groundtruth.bin").open("xb") as target:
            target.write(struct.pack("<II", len(queries), 10))
            ids.astype("<i4").tofile(target)
            distances.astype("<f4").tofile(target)
        write_matrix(prepared / "groundtruth_distances.f64bin", distances, "<f8")
        records = []
        for entry in commitment["labels"]:
            label = entry["label"]
            records.append({**entry, "original_label": label, "cpp_input_label": label,
                            "selected_queries": int(np.sum(labels == label)),
                            "available_one_tag_queries": sum(x["per_label_counts"].get(str(label), x["per_label_counts"].get(label, 0)) for x in commitment["outputs"].values())})
        manifest = {"dataset": "Amazon-C4 category filtered product search", "base_vectors": len(base),
                    "dimension": DIMENSION, "dtype": "float32", "metric": "L2", "k": 10,
                    "query_count": len(queries), "labels": records, "split": split,
                    "source_query_namespace": f"Amazon-C4@{DATA_REVISION}",
                    "groundtruth": "Exhaustive full-posting squared L2; float64 arithmetic over stored float32 coordinates; distance then ID ordering; conservative floating-point screening band followed by direct coordinate-distance ranking; binary distances rounded float32",
                    "filter_semantics": commitment["filter_semantics"],
                    "query_selection": commitment["groups"],
                    "split_commitment": record(sealed / "split-commitment.json"),
                    "truth_evidence": {"evaluated_pairs": pairs, "elapsed_seconds": time.monotonic() - started,
                                       "all_truth_ids_membership_checked": True,
                                       "query_block": args.query_block, "base_block": args.base_block,
                                       "float64_distances": record(prepared / "groundtruth_distances.f64bin"),
                                       "independent_direct_audit": record(prepared / "truth-audit.json")},
                    "preparation_source": record(Path(__file__)),
                    "artifacts": {name: record(prepared / name) for name in ("base.fbin", "query.fbin", "query_ids.u32bin", "query_labels.u32bin", "groundtruth.bin", "labels.txt", "query_labels.txt", "query_metadata.jsonl")}}
        write_json(prepared / "manifest.json", manifest)
        write_json(prepared / "label_counts.json", records)
        print(f"READY {prepared}: {len(queries)} queries, {pairs} exhaustive distances", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=["download", "metadata", "embed", "seal", "truth"], required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--query-block", type=int, default=64)
    parser.add_argument("--base-block", type=int, default=16384)
    parser.add_argument("--audit-per-label", type=int, default=2)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    require(args.threads > 0 and args.batch_size > 0 and args.audit_per_label > 0
            and args.query_block > 0 and args.base_block > 0, "Invalid thread, batch, or block count")
    if args.stage == "download":
        download(args.root)
    elif args.stage == "metadata":
        metadata(args.root)
    elif args.stage == "embed":
        embed(args.root, args)
    elif args.stage == "seal":
        seal(args.root)
    else:
        truth(args.root, args)


if __name__ == "__main__":
    main()
