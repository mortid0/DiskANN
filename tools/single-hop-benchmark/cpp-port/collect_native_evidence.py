#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Read native build identity, dependencies and runtime configuration; run outside timing windows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def capture(command):
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / 'cpp'
    binary_folder = source / 'build-native/apps'
    dependencies = root / 'cpp-deps'
    packages = []
    for package in sorted((dependencies / 'downloads').glob('*.deb')):
        packages.append({'file': package.name, 'sha256': sha(package),
                         'package': capture(['dpkg-deb', '-f', str(package), 'Package']),
                         'version': capture(['dpkg-deb', '-f', str(package), 'Version'])})
    flag_prefixes = ('CMAKE_CXX_FLAGS:', 'CMAKE_CXX_FLAGS_RELEASE:', 'CMAKE_EXE_LINKER_FLAGS:',
                     'MKL_PATH:', 'MKL_INCLUDE_PATH:', 'OMP_PATH:', 'BOOST_ROOT:',
                     'Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE:')
    flags = [line for line in (source / 'build-native/CMakeCache.txt').read_text().splitlines()
             if line.startswith(flag_prefixes)]
    evidence = {
        'source_commit': '78256bbab4685e1774e78d331e081a153be26823',
        'source_archive_sha256': sha(root / 'diskann-cpp-native-source.tar'),
        'native_metrics_patch_sha256': sha(root / 'native-query-metrics.patch'),
        'patched_source_sha256': {line[6:]: sha(source / line[6:])
            for line in (root / 'native-query-metrics.patch').read_text().splitlines()
            if line.startswith('+++ b/')},
        'native_compiled': True,
        'platform': platform.platform(), 'machine': platform.machine(),
        'cpu': capture(['lscpu']), 'compiler': capture(['g++', '--version']),
        'cmake': capture(['cmake', '--version']), 'cmake_cache_flags': flags,
        'binaries': {name: {'sha256': sha(binary_folder / name),
                            'linked_libraries': capture(['ldd', str(binary_folder / name)])}
                     for name in ['search_disk_index', 'build_disk_index']},
        'packages': packages,
        'system_openmp_sha256': sha(Path('/usr/lib/llvm-18/lib/libomp.so.5')),
        'libaio_runtime': {
            'symlink': os.readlink(dependencies / 'root/usr/lib/x86_64-linux-gnu/libaio.so.1t64')
                if (dependencies / 'root/usr/lib/x86_64-linux-gnu/libaio.so.1t64').is_symlink() else None,
            'sha256': sha(dependencies / 'root/usr/lib/x86_64-linux-gnu/libaio.so.1t64')},
        'runtime_environment': {'OMP_TOOL': 'disabled', 'MKL_NUM_THREADS': '1',
            'LD_LIBRARY_PATH': str(dependencies / 'root/usr/lib/x86_64-linux-gnu') + ':/usr/lib/llvm-18/lib'},
        'benchmark_cpu_affinity': list(range(8)),
        'runtime_workaround': {
            'observed': 'Initial query crashed with Archer/TSAN warning; OMP_TOOL=disabled completed native smoke without rebuilding',
            'scope': 'Optional OpenMP tooling disabled; native SIMD, MKL and Linux AIO retained',
            'consistent_with_upstream_report': 'https://github.com/llvm/llvm-project/issues/122319'},
        'system_packages_installed': False,
    }
    args.output.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'evidence': str(args.output), 'packages': len(packages), 'native_compiled': True}))


if __name__ == '__main__':
    main()
