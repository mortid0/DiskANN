#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Build official C++ StitchedVamana, place it on SSD, and reuse a pinned PQ artifact pair."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def execute(command, log, environment):
    started = time.monotonic()
    with log.open('w') as stream:
        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return {'command': command, 'log': str(log), 'elapsed_seconds': time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--prepared-data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pq-prefix', type=Path, required=True)
    parser.add_argument('--name', default='yfcc1m_stitched_R32_R128_L100')
    parser.add_argument('--encode-pq', action='store_true', help='Encode this base with the supplied pivot table before building; no retraining')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--label-degree', type=int, default=32)
    parser.add_argument('--stitched-degree', type=int, default=128)
    parser.add_argument('--build-l', type=int, default=100)
    args = parser.parse_args()
    data = args.prepared_data.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / args.name
    work = output / (args.name + '.work')
    if work.exists() or Path(str(prefix) + '_disk.index').exists():
        parser.error('output already exists; choose a new --name')
    work.mkdir()
    base = work / 'base.u8bin'
    base.symlink_to(data / 'base.u8bin')
    memory_graph = work / 'mem.index'
    environment = os.environ.copy()
    environment.update({'VECLIB_MAXIMUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': str(args.threads)})
    pq_prefix = args.pq_prefix.resolve()
    encoding = None
    if args.encode_pq:
        source_pivots = Path(str(pq_prefix) + '_pq_pivots.bin')
        target_pivots = Path(str(prefix) + '_pq_pivots.bin')
        try:
            target_pivots.hardlink_to(source_pivots)
        except OSError:
            shutil.copy2(source_pivots, target_pivots)
        encoding = execute([str(args.bin_dir.resolve() / 'encode_pq_reference'), str(base),
                            str(target_pivots), str(prefix) + '_pq_compressed.bin', '32', str(args.threads)],
                           output / (args.name + '.encode-pq.log'), environment)
        encoding['training_pivots_source'] = str(source_pivots)
        encoding['training_pivots_sha256'] = sha(source_pivots)
        pq_prefix = prefix
    with Path(str(pq_prefix) + '_pq_compressed.bin').open('rb') as stream:
        pq_rows, pq_bytes = struct.unpack('<II', stream.read(8))
    with base.open('rb') as stream:
        base_rows, _ = struct.unpack('<II', stream.read(8))
    if pq_rows != base_rows or pq_bytes != 32:
        raise RuntimeError('shared PQ rows/bytes differ from this base; use --encode-pq')
    command = [str(args.bin_dir.resolve() / 'build_stitched_index'), '--data_type', 'uint8',
               '--data_path', str(base), '--label_file', str(data / 'labels.txt'),
               '--index_path_prefix', str(memory_graph), '-T', str(args.threads),
               '-R', str(args.label_degree), '-L', str(args.build_l),
               '--stitched_R', str(args.stitched_degree), '--alpha', '1.2']
    manifest = {'method': 'official_cpp_stitched_vamana', 'label_degree': args.label_degree,
                'stitched_degree': args.stitched_degree, 'build_l': args.build_l, 'threads': args.threads,
                'build': execute(command, output / (args.name + '.build.log'), environment),
                'pq_encoding': encoding,
                'build_binaries_sha256': {name: sha(args.bin_dir.resolve() / name) for name in
                                         ['build_stitched_index', 'create_disk_layout_reference']}}
    command = [str(args.bin_dir.resolve() / 'create_disk_layout_reference'), str(base), str(memory_graph),
               str(prefix) + '_disk.index']
    manifest['disk_layout'] = execute(command, output / (args.name + '.layout.log'), environment)
    for suffix in ['_labels.txt', '_labels_map.txt', '_labels_to_medoids.txt']:
        shutil.copy2(str(memory_graph) + suffix, str(prefix) + '_disk.index' + suffix)
    manifest['shared_pq'] = {}
    for suffix in ['_pq_pivots.bin', '_pq_compressed.bin']:
        source = Path(str(pq_prefix) + suffix)
        target = Path(str(prefix) + suffix)
        if source != target:
            try:
                target.hardlink_to(source)
            except OSError:
                shutil.copy2(source, target)
        manifest['shared_pq'][suffix] = {'source': str(source), 'target': str(target), 'sha256': sha(target)}
    with Path(str(prefix) + '_pq_compressed.bin').open('rb') as stream:
        manifest['pq_rows'], manifest['pq_bytes'] = struct.unpack('<II', stream.read(8))
    with base.open('rb') as stream:
        base_rows, _ = struct.unpack('<II', stream.read(8))
    if manifest['pq_rows'] != base_rows:
        raise RuntimeError('PQ artifact row count does not match base data')
    (output / (args.name + '.manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'index_prefix': str(prefix), 'manifest': str(output / (args.name + '.manifest.json'))}))


if __name__ == '__main__':
    main()
