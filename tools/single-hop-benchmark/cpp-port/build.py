#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Build the pinned, explicitly adapted C++ disk reference on Apple arm64."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tarfile
import tempfile

CPP_COMMIT = "78256bbab4685e1774e78d331e081a153be26823"
BOOST_VERSION = "1.85.0"
BOOST_URL = "https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.bz2"
BOOST_SHA256 = "7009fe1faa1697476bdc7027703a2badb84e849b7b0baad5086b087b971f8617"


def run(arguments: list[str], *, cwd: Path | None = None) -> None:
    print("Running:", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="New output directory for the exported C++ source")
    parser.add_argument("--deps-dir", type=Path, required=True,
                        help="Local dependency cache; no system packages are installed")
    parser.add_argument("--boost-archive", type=Path,
                        help="Existing Boost 1.85 source archive; verified before use")
    parser.add_argument("--omp-root", type=Path, default=Path("/opt/homebrew/opt/libomp"))
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
        parser.error("the adapter is for native Apple arm64 only")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not (args.omp_root / "lib/libomp.dylib").is_file():
        parser.error("libomp must already exist; pass its installation as --omp-root")

    source = args.source_dir.resolve()
    dependencies = args.deps_dir.resolve()
    if source.exists():
        parser.error("--source-dir must not already exist; existing files are never overwritten")
    dependencies.mkdir(parents=True, exist_ok=True)
    boost_archive = args.boost_archive or dependencies / "boost_1_85_0.tar.bz2"
    boost_archive = boost_archive.resolve()
    if not boost_archive.exists():
        run(["curl", "-L", "--max-time", "180", "--fail", "--output", str(boost_archive), BOOST_URL])
    if sha256(boost_archive) != BOOST_SHA256:
        parser.error("Boost archive checksum mismatch")
    boost = dependencies / "boost_1_85_0"
    if not (boost / "boost/version.hpp").is_file():
        with tarfile.open(boost_archive) as archive:
            members = [entry for entry in archive.getmembers()
                       if entry.name.startswith("boost_1_85_0/boost/")
                       or entry.name.startswith("boost_1_85_0/libs/program_options/src/")
                       or entry.name == "boost_1_85_0/LICENSE_1_0.txt"]
            archive.extractall(dependencies, members=members, filter="data")

    own_folder = Path(__file__).resolve().parent
    repository = own_folder.parents[2]
    source.mkdir(parents=True)
    with tempfile.TemporaryFile() as exported:
        subprocess.run(["git", "archive", "--format=tar", CPP_COMMIT], cwd=repository,
                       stdout=exported, check=True)
        exported.seek(0)
        with tarfile.open(fileobj=exported, mode="r:") as archive:
            archive.extractall(source, filter="data")
    run(["patch", "--batch", "--forward", "-p1", "--input", str(own_folder / "macos-reference.patch")], cwd=source)
    build = source / "build-macos"
    run(["cmake", "-S", str(source / "portability"), "-B", str(build),
         "-DCMAKE_BUILD_TYPE=Release", f"-DBOOST_ROOT={boost}", f"-DOMP_ROOT={args.omp_root}"])
    run(["cmake", "--build", str(build), "--target", "build_disk_index", "search_disk_index", "build_stitched_index", "create_disk_layout_reference", "encode_pq_reference", "-j", str(args.jobs)])
    manifest = {
        "source_commit": CPP_COMMIT,
        "boost_version": BOOST_VERSION,
        "boost_source_sha256": BOOST_SHA256,
        "adapter_sha256": sha256(own_folder / "macos-reference.patch"),
        "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "source_directory": str(source),
        "build_directory": str(build),
        "compiler": subprocess.check_output(["clang++", "--version"], text=True),
        "cmake": subprocess.check_output(["cmake", "--version"], text=True),
        "compiler_flags": "-std=c++17 -O3 -DNDEBUG -Xpreprocessor -fopenmp; no AVX intrinsics",
        "math_backend": "Apple Accelerate CBLAS, LP64; OPQ unsupported",
        "disk_reader": "blocking positional pread with F_NOCACHE; sequential requests within each beam",
        "query_stats_environment": "DISKANN_REFERENCE_STATS_JSONL",
        "production_cpp_timing_parity": False,
    }
    (source / "reference-build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"build_disk_index": str(build / "build_disk_index"),
                      "search_disk_index": str(build / "search_disk_index"),
                      "manifest": str(source / "reference-build-manifest.json")}))


if __name__ == "__main__":
    main()
