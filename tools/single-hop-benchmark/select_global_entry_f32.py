#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Select a deterministic global entry from corpus vectors, without query inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def record(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def select(base_path, block_rows=8192):
    base_path = Path(base_path)
    before = record(base_path)
    with base_path.open('rb') as stream:
        rows, dimensions = struct.unpack('<II', stream.read(8))
    if not rows or not dimensions or block_rows <= 0 or base_path.stat().st_size != 8 + rows*dimensions*4:
        raise ValueError('Invalid float32 base matrix or block size')
    base = np.memmap(base_path, dtype='<f4', mode='r', offset=8, shape=(rows, dimensions))
    total = np.zeros(dimensions, dtype=np.float64)
    for start in range(0, rows, block_rows):
        block = base[start:start+block_rows]
        if not np.all(np.isfinite(block)):
            raise ValueError('Nonfinite corpus coordinates')
        total += np.sum(block, axis=0, dtype=np.float64)
    centroid = total / rows
    best = (float('inf'), rows)
    for start in range(0, rows, block_rows):
        delta = np.asarray(base[start:start+block_rows], dtype=np.float64) - centroid
        distances = np.sum(delta*delta, axis=1, dtype=np.float64)
        if not np.all(np.isfinite(distances)):
            raise ValueError('Nonfinite centroid distances')
        local = int(np.argmin(distances))
        best = min(best, (float(distances[local]), start+local))
    after = record(base_path)
    if before != after:
        raise ValueError('Base corpus changed while selecting global entry')
    return {'selector': 'nearest_to_full_base_centroid_float64_distance_then_ID',
            'base_sha256': after['sha256'], 'base_before': before, 'base_after': after, 'dimensions': dimensions,
            'base_vectors': rows, 'entry_point': best[1], 'squared_distance_to_centroid': best[0],
            'block_rows': block_rows, 'numpy_version': np.__version__,
            'selector_source': record(Path(__file__)), 'query_inputs_used': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--block-rows', type=int, default=8192)
    args = parser.parse_args()
    result = select(args.base, args.block_rows)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'entry_point': result['entry_point'], 'proof': str(args.out)}))


if __name__ == '__main__':
    main()
