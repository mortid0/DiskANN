#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Read-only Linux host/dependency probe; never installs or changes packages."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess


def command(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=15)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    libraries = ["libaio-dev", "libboost-program-options-dev", "libgoogle-perftools-dev", "libmkl-full-dev",
                 "intel-oneapi-mkl-devel", "intel-oneapi-openmp-devel"]
    mkl_candidates = [Path("/opt/intel/oneapi/mkl/latest"), Path("/opt/intel/mkl"), Path("/usr/include/mkl")]
    result = {
        "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "tools": {name: shutil.which(name) for name in ["g++", "cmake", "make", "git", "curl", "python3"]},
        "compiler": command(["g++", "--version"]),
        "cpu": command(["lscpu"]),
        "memory": command(["free", "-b"]),
        "disk": command(["df", "-B1", str(args.workdir.resolve())]),
        "package_status": command(["dpkg-query", "-W", "-f=${binary:Package} ${Version} ${db:Status-Abbrev}\n", *libraries]),
        "mkl_candidates": {str(path): path.exists() for path in mkl_candidates},
        "dynamic_libraries": command(["ldconfig", "-p"]),
        "aio_max_requests": Path("/proc/sys/fs/aio-max-nr").read_text().strip()
                            if Path("/proc/sys/fs/aio-max-nr").is_file() else None,
        "actions": "read_only_no_packages_installed",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
