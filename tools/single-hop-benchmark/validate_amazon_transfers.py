#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Read-only hash/split validation of relocated Amazon benchmark preparations.

Imports the frozen runner's preparation and family-leakage auditors. It performs
no ANN search or index build. Hashes of shared physical files are reused only
while inode, size, modification time, and change time remain identical.
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


class HashCache:
    def __init__(self):
        self.entries = {}
        self.reused = 0

    @staticmethod
    def key(path):
        value = path.stat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def sha(self, path):
        path = Path(path).resolve(strict=True)
        before = self.key(path)
        previous = self.entries.get(str(path))
        if previous is not None:
            require(previous["stat"] == before, f"Transferred file changed during validation: {path}")
            self.reused += 1
            return previous["sha256"]
        h = hashlib.sha256()
        if before[2] >= (1 << 30):
            print(json.dumps({"phase": "hash_begin", "path": str(path), "bytes": before[2]}), flush=True)
        with path.open("rb") as source:
            for block in iter(lambda: source.read(16 << 20), b""):
                h.update(block)
        require(self.key(path) == before, f"Transferred file changed while hashing: {path}")
        self.entries[str(path)] = {"path": str(path), "bytes": before[2], "sha256": h.hexdigest(), "stat": before}
        return h.hexdigest()

    def record(self, path):
        path = Path(path).resolve(strict=True)
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": self.sha(path)}

    def check(self, path, expected):
        actual = self.record(path)
        require(actual["sha256"] == expected["sha256"] and actual["bytes"] == expected["bytes"], f"Transferred identity differs: {path}")
        return actual

    def unchanged(self):
        for path, entry in self.entries.items():
            require(self.key(Path(path)) == entry["stat"], f"Transferred file changed after validation: {path}")


def source_match(expected, candidates, cache):
    for path in candidates:
        if path.is_file() and path.stat().st_size == expected["bytes"] and cache.sha(path) == expected["sha256"]:
            return cache.record(path)
    raise ValueError(f"Frozen preparation source snapshot unavailable: {expected['sha256']}")


def source_chain(root, name, runner_path, cache):
    code = runner_path.parent
    if name == "c4":
        data = root/"amazon-c4/data"
        identity_path = data/"embedding/embedding-identity.json"
        identity = json.loads(identity_path.read_text())
        configuration = data/"embedding/configuration.json"
        cache.check(configuration, identity["configuration"])
        config = json.loads(configuration.read_text())
        meta_path = data/"metadata/metadata.json"
        cache.check(meta_path, config["metadata"])
        meta = json.loads(meta_path.read_text())
        for file, expected in meta["artifacts"].items():
            cache.check(data/"metadata"/file, expected)
        for file, expected in config["model_files"].items():
            cache.check(data/"model"/file, expected)
        for kind in ("base", "query"):
            cache.check(data/"embedding"/(kind+".fbin"), identity["outputs"][kind])
        download_path = data/"download-identity.json"
        cache.check(download_path, meta["download_identity"])
        download = json.loads(download_path.read_text())
        for file, expected in download["raw"].items():
            cache.check(data/"raw"/file, expected)
        producer = source_match(config["source"], [code/"prepare_amazon_c4.py"], cache)
        return {"embedding_identity": cache.record(identity_path), "configuration": cache.record(configuration),
                "metadata": cache.record(meta_path), "download_identity": cache.record(download_path),
                "embedding_producer": producer, "raw_and_model_objects_verified": True,
                "historical_downloader_source": "Identity retained; downloaded objects checked against their exact pinned hashes"}
    books = root/"amazon-books"
    path = books/"export/export-identity.json"
    exported = json.loads(path.read_text())
    for file, expected in exported["artifacts"].items():
        cache.check(books/"export"/file, expected)
    candidates = [code/"prepare_bigvector_books.py", *(books/"sources").glob("prepare_bigvector_books*.py")]
    producer = source_match(exported["preparation_source"], candidates, cache)
    return {"export_identity": cache.record(path), "export_producer": producer,
            "publisher_source": exported["source"],
            "publisher_hdf5_transferred": False,
            "publisher_source_scope": "Original publisher HDF5 retained locally; remote exports independently match its frozen exporter identities"}


def validate(args):
    root, runner_path, output = args.benchmark_root.resolve(), args.runner.resolve(), args.out.resolve()
    require(not output.exists(), "Validation output exists; choose a new evidence directory")
    cache = HashCache()
    spec = importlib.util.spec_from_file_location("frozen_f32_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    original_sha = runner.sha
    runner.sha = cache.sha
    started = time.monotonic()
    locations = {"c4": root/"amazon-c4/data/sealed", "books-bank": root/"amazon-books/sealed-bank",
                 "books-original": root/"amazon-books/sealed-original"}
    results, chains = {}, {}
    for name in args.workloads:
        sealed = locations[name]
        print(json.dumps({"phase": "verify_prepared", "workload": name}), flush=True)
        symlinks = [path for path in sealed.rglob("*") if path.is_symlink()]
        for path in symlinks:
            require(not path.readlink().is_absolute() and path.resolve(strict=True).is_relative_to(root),
                    f"Transferred symlink escaped its benchmark tree: {path}")
        proof_path = sealed/"split-commitment.json"
        proof = json.loads(proof_path.read_text())
        partitions = {}
        for split in ("calibration", "validation"):
            prepared = sealed/("prepared-"+split)
            n, dimension, k = runner.verify_prepared(prepared)
            manifest = json.loads((prepared/"manifest.json").read_text())
            expected_source = manifest["preparation_source"]
            candidates = [runner_path.parent/Path(expected_source["path"]).name,
                          *(root/"amazon-books/sources").glob("prepare_bigvector_books*.py")]
            source = source_match(expected_source, candidates, cache)
            partitions[split] = {"manifest": cache.record(prepared/"manifest.json"), "queries": n,
                                 "dimension": dimension, "k": k, "declared_labels": len(manifest["labels"]),
                                 "verified_prepared_artifacts": len(manifest["artifacts"]),
                                 "all_artifact_sizes_and_sha256_verified": True, "preparation_source": source}
        split_audit = runner.audit_split(sealed/"prepared-calibration", sealed/"prepared-validation", runner.record(proof_path))
        source = source_match(proof["preparation_source"], [runner_path.parent/Path(proof["preparation_source"]["path"]).name,
                              *(root/"amazon-books/sources").glob("prepare_bigvector_books*.py")], cache)
        if name == "c4":
            chain = source_chain(root, "c4", runner_path, cache)
            cache.check(root/"amazon-c4/data/embedding/embedding-identity.json", proof["embedding_identity"])
        elif "books" not in chains:
            chains["books"] = source_chain(root, "books", runner_path, cache)
            chain = chains["books"]
        else:
            chain = chains["books"]
        results[name] = {"partitions": partitions, "split_commitment": cache.record(proof_path),
                         "split_audit": split_audit, "commitment_source": source,
                         "relative_symlinks_verified": len(symlinks), "source_identity_chain": chain}
        print(json.dumps({"phase": "workload_verified", "workload": name,
                          "calibration_queries": partitions["calibration"]["queries"],
                          "validation_queries": partitions["validation"]["queries"]}), flush=True)
    cache.unchanged()
    runner.sha = original_sha
    output.mkdir(parents=True)
    inventory = [{key: value for key, value in entry.items() if key != "stat"} for entry in cache.entries.values()]
    inventory_path = output/"verified-file-inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2)+"\n")
    result = {"schema_version": 1, "completed": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "read_only_dataset_validation": True, "ann_search_performed": False, "index_build_performed": False,
              "manifests_or_proofs_rewritten": False, "validator": cache.record(Path(__file__)),
              "frozen_runner": cache.record(runner_path), "workloads": results,
              "unique_files_hashed": len(inventory), "unique_bytes_hashed": sum(item["bytes"] for item in inventory),
              "shared_unchanged_file_hashes_reused": cache.reused,
              "elapsed_seconds": time.monotonic()-started, "inventory": cache.record(inventory_path)}
    (output/"validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"validation": str(output/"validation.json"), "workloads": args.workloads,
                      "unique_files_hashed": len(inventory), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workloads", nargs="+", choices=("c4", "books-bank", "books-original"), default=["c4", "books-bank", "books-original"])
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
