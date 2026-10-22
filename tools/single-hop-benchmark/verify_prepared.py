#!/usr/bin/env python3
"""Verify transferred prepared YFCC datasets using only the Python standard library.

Checks recorded hashes, binary shapes, all predicate memberships and every truth
ID. This verifies transfer/integrity; it does not recompute nearest neighbors.
"""

import argparse
from array import array
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import sys
import time


class VerificationError(ValueError):
    pass


class Verifier:
    def __init__(self, directories):
        self.directories = directories
        self.hash_cache = {}
        self.matrix_cache = {}
        self.posting_checks = set()
        self.membership_checks = set()

    @staticmethod
    def file_key(path):
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def sha256(self, path):
        key = self.file_key(path)
        if key not in self.hash_cache:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(16 << 20), b""):
                    digest.update(block)
            self.hash_cache[key] = digest.hexdigest()
        return self.hash_cache[key]

    @staticmethod
    def header(path, width):
        with path.open("rb") as source:
            header = source.read(8)
        if len(header) != 8:
            raise VerificationError(f"{path}: truncated matrix header")
        rows, columns = struct.unpack("<II", header)
        expected = 8 + rows * columns * width
        if path.stat().st_size != expected:
            raise VerificationError(f"{path}: expected {expected} bytes for {rows}x{columns}, "
                                    f"found {path.stat().st_size}")
        return rows, columns

    def u32_matrix(self, path):
        key = self.file_key(path)
        if key not in self.matrix_cache:
            rows, columns = self.header(path, 4)
            values = array("I")
            if values.itemsize != 4:
                raise VerificationError("This Python array implementation does not have 32-bit unsigned integers")
            with path.open("rb") as source:
                source.seek(8)
                values.fromfile(source, rows * columns)
            if sys.byteorder != "little":
                values.byteswap()
            self.matrix_cache[key] = (rows, columns, values)
        return self.matrix_cache[key]

    def full_base_source(self, directory, manifest, expected_hash):
        candidates = [directory.parent / "base.10M.u8bin",
                      directory.parent / "prepared-10m" / "base.u8bin"]
        candidates += [item / "base.u8bin" for item in self.directories]
        if manifest.get("source_prepared_dataset"):
            candidates.append(directory.parent / Path(manifest["source_prepared_dataset"]).name / "base.u8bin")
        for candidate in candidates:
            if candidate.exists():
                try:
                    if self.header(candidate, 1) == (10_000_000, manifest["dimension"]) and self.sha256(candidate) == expected_hash:
                        return candidate
                except (OSError, VerificationError):
                    continue
        raise VerificationError(f"{directory}: prefix base has no recorded standalone hash; "
                                "provide the authenticated full base as sibling base.10M.u8bin, "
                                "prepared-10m/base.u8bin, or another directory argument")

    def verify_base(self, directory, manifest):
        base = directory / "base.u8bin"
        shape = self.header(base, 1)
        expected_shape = (manifest["base_vectors"], manifest["dimension"])
        if shape != expected_shape or manifest.get("dtype") != "uint8":
            raise VerificationError(f"{base}: shape/type {shape}/uint8 differs from manifest {expected_shape}/{manifest.get('dtype')}")
        sources = [item for item in manifest.get("sources", []) if item.get("file") == "base.10M.u8bin"]
        if len(sources) != 1 or not sources[0].get("sha256"):
            raise VerificationError(f"{directory}: missing unique full-base source SHA-256")
        actual = self.sha256(base)
        expected = sources[0]["sha256"]
        if shape[0] == 10_000_000:
            if actual != expected:
                raise VerificationError(f"{base}: full-base SHA-256 differs from recorded source")
            method = "full-base SHA-256 equals recorded source"
        elif "base.u8bin" in manifest["artifacts"]:
            if actual != manifest["artifacts"]["base.u8bin"]["sha256"]:
                raise VerificationError(f"{base}: prefix SHA-256 differs from recorded artifact")
            method = "prefix-base SHA-256 equals recorded artifact"
        else:
            original = self.full_base_source(directory, manifest, expected)
            digest = hashlib.sha256(struct.pack("<II", *shape))
            remaining = shape[0] * shape[1]
            with original.open("rb") as source:
                source.seek(8)
                while remaining:
                    block = source.read(min(remaining, 16 << 20))
                    if not block:
                        raise VerificationError(f"{original}: truncated while checking prefix")
                    digest.update(block)
                    remaining -= len(block)
            if actual != digest.hexdigest():
                raise VerificationError(f"{base}: bytes differ from authenticated full-base prefix")
            method = "prefix reconstructed from authenticated full-base source"
        return {"sha256": actual, "bytes": base.stat().st_size, "method": method}

    def verify_postings(self, directory, manifest, points):
        postings = {}
        records = {}
        for record in manifest["labels"]:
            label = int(record["original_label"])
            if label in postings:
                raise VerificationError(f"{directory}: duplicate label {label} in manifest")
            path = directory / f"label_{label}.u32bin"
            count, width, values = self.u32_matrix(path)
            if width != 1 or count != record["base_count"]:
                raise VerificationError(f"{path}: count/width {count}/{width} differs from label manifest")
            check_key = (self.file_key(path), points)
            if check_key not in self.posting_checks:
                previous = -1
                for offset, value in enumerate(values):
                    if value <= previous or value >= points:
                        raise VerificationError(f"{path}: unsorted, duplicate or out-of-range ID {value} at posting {offset}")
                    previous = value
                self.posting_checks.add(check_key)
            if count == 0 or count < manifest["k"]:
                raise VerificationError(f"{path}: only {count} eligible points for k={manifest['k']}")
            fraction = record.get("match_fraction")
            if fraction is not None and not math.isclose(fraction, count / points, rel_tol=1e-12):
                raise VerificationError(f"{path}: match_fraction differs from count/base_vectors")
            postings[label] = values
            records[label] = record
        if not postings:
            raise VerificationError(f"{directory}: no selected labels")
        return postings, records

    def verify_cpp_labels(self, directory, manifest, postings, points):
        path = directory / "labels.txt"
        key = (self.file_key(path), tuple((label, self.file_key(directory / f"label_{label}.u32bin"))
                                         for label in sorted(postings)))
        if key in self.membership_checks:
            return "same authenticated label/posting files already checked"
        positions = {label: 0 for label in postings}
        sentinel = int(manifest["sentinel"]["label"])
        row_count = 0
        with path.open("r", encoding="ascii") as source:
            for row, line in enumerate(source):
                row_count += 1
                try:
                    labels = [int(value) for value in line.strip().split(",")]
                except ValueError as error:
                    raise VerificationError(f"{path}: invalid numeric labels at base row {row}") from error
                if not labels or len(set(labels)) != len(labels):
                    raise VerificationError(f"{path}: empty or duplicate labels at base row {row}")
                if sentinel in labels:
                    if labels != [sentinel]:
                        raise VerificationError(f"{path}: sentinel mixed with selected labels at base row {row}")
                    continue
                for label in labels:
                    if label not in postings:
                        raise VerificationError(f"{path}: unexpected label {label} at base row {row}")
                    position = positions[label]
                    if position >= len(postings[label]) or postings[label][position] != row:
                        raise VerificationError(f"{path}: label {label} disagrees with posting list at base row {row}")
                    positions[label] += 1
        if row_count != points or any(positions[label] != len(postings[label]) for label in postings):
            raise VerificationError(f"{path}: base-row count or complete posting membership differs from manifest")
        self.membership_checks.add(key)
        return "all base label rows and all posting memberships cross-checked"

    def verify(self, directory):
        started = time.monotonic()
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        artifacts = manifest.get("artifacts", {})
        required = ("query.u8bin", "query_labels.u32bin", "groundtruth.bin", "labels.txt", "query_labels.txt")
        if any(name not in artifacts for name in required):
            raise VerificationError(f"{manifest_path}: required artifact size/hash records are missing")
        for name, record in artifacts.items():
            path = directory / name
            if path.stat().st_size != record["bytes"]:
                raise VerificationError(f"{path}: size differs from manifest")
            if self.sha256(path) != record["sha256"]:
                raise VerificationError(f"{path}: SHA-256 differs from manifest")
        expected_identity = ":".join(artifacts[name]["sha256"] for name in
                                     ("query.u8bin", "query_labels.u32bin", "groundtruth.bin", "labels.txt"))
        if manifest.get("prepared_identity", expected_identity) != expected_identity:
            raise VerificationError(f"{manifest_path}: prepared_identity differs from artifact hashes")
        base = self.verify_base(directory, manifest)
        points, dimension = manifest["base_vectors"], manifest["dimension"]
        count, query_dim = self.header(directory / "query.u8bin", 1)
        if (count, query_dim) != (manifest["query_count"], dimension):
            raise VerificationError(f"{directory}: query matrix shape differs from manifest")
        label_count, label_dim, query_labels = self.u32_matrix(directory / "query_labels.u32bin")
        if (label_count, label_dim) != (count, 1):
            raise VerificationError(f"{directory}: query label dimensions differ from query matrix")
        text_labels = [int(line) for line in (directory / "query_labels.txt").read_text().splitlines()]
        if text_labels != list(query_labels):
            raise VerificationError(f"{directory}: C++ text query labels differ from binary query labels")
        if "query_ids.u32bin" in artifacts:
            id_count, id_dim, query_ids = self.u32_matrix(directory / "query_ids.u32bin")
            if (id_count, id_dim) != (count, 1) or len(set(query_ids)) != count or any(value >= 100_000 for value in query_ids):
                raise VerificationError(f"{directory}: invalid original public-query ID matrix")
        postings, records = self.verify_postings(directory, manifest, points)
        histogram = Counter(query_labels)
        if set(histogram) != set(postings):
            raise VerificationError(f"{directory}: query labels and selected manifest labels differ")
        for label, n in histogram.items():
            if n != records[label]["selected_queries"]:
                raise VerificationError(f"{directory}: label {label} query count {n} differs from manifest")
        cpp_labels = self.verify_cpp_labels(directory, manifest, postings, points)
        truth_path = directory / "groundtruth.bin"
        n, k = self.header(truth_path, 8)
        if (n, k) != (count, manifest["k"]):
            raise VerificationError(f"{truth_path}: ground-truth shape differs from query contract")
        ids, distances = array("I"), array("f")
        with truth_path.open("rb") as source:
            source.seek(8)
            ids.fromfile(source, n * k)
            distances.fromfile(source, n * k)
        if sys.byteorder != "little":
            ids.byteswap()
            distances.byteswap()
        for row, label in enumerate(query_labels):
            members = postings[label]
            seen = set()
            previous = -math.inf
            for column in range(k):
                offset = row * k + column
                value, distance = ids[offset], distances[offset]
                position = bisect_left(members, value)
                if value >= points or value in seen or position == len(members) or members[position] != value:
                    raise VerificationError(f"{truth_path}: ineligible, duplicate or out-of-range ID {value} at query {row}, rank {column}")
                if not math.isfinite(distance) or distance < 0 or distance < previous:
                    raise VerificationError(f"{truth_path}: nonfinite, negative or unsorted distance at query {row}, rank {column}")
                seen.add(value)
                previous = distance
        return {"directory": str(directory), "verified": True, "base_vectors": points,
                "dimensions": dimension, "queries": count, "k": k, "labels": len(postings),
                "verified_truth_memberships": count * k, "verified_posting_memberships": sum(map(len, postings.values())),
                "manifest_sha256": self.sha256(manifest_path), "prepared_identity": expected_identity,
                "verified_artifact_files": sorted(artifacts), "base": base,
                "cpp_label_crosscheck": cpp_labels, "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    directories = [path.resolve() for path in args.directories]
    verifier = Verifier(directories)
    results = []
    for directory in directories:
        try:
            result = verifier.verify(directory)
            print(f"VERIFIED {directory.name}: base={result['base_vectors']} queries={result['queries']} "
                  f"k={result['k']} labels={result['labels']} ({result['seconds']:.1f}s)", flush=True)
        except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
            result = {"directory": str(directory), "verified": False, "error": str(error)}
            print(f"FAILED {directory}: {error}", file=sys.stderr, flush=True)
        results.append(result)
    report = {"verified": all(item["verified"] for item in results),
              "verification_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "platform": platform.platform(),
              "verifier_sha256": verifier.sha256(Path(__file__)),
              "scope": "transfer/integrity, dimensions and all recorded memberships; no new exhaustive nearest-neighbor calculation",
              "datasets": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
