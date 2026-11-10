#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Prepare a base-only ordinary-label global graph input and centroid proof.

This helper reads no queries or filters from the benchmark. A single normal
label covers the corpus; it is not the C++ universal-label feature. Construction
and ANN evaluation are separate, explicitly scheduled operations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def prepare(base, expected_sha, out, selector, expected_selector_sha, python=sys.executable):
    base, out, selector = Path(base).resolve(), Path(out).resolve(), Path(selector).resolve()
    before, selector_before = record(base), record(selector)
    require(before['sha256'] == expected_sha, 'Transferred base SHA256 differs from dataset commitment')
    require(selector_before['sha256'] == expected_selector_sha, 'Selector differs from frozen source')
    with base.open('rb') as stream:
        header = stream.read(8)
    require(len(header) == 8, 'Truncated base matrix header')
    rows, dimensions = struct.unpack('<II', header)
    require(rows > 0 and dimensions > 0 and before['bytes'] == 8 + rows * dimensions * 4, 'Malformed float32 base matrix')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'base.fbin').symlink_to(os.path.relpath(base, out))
    with (out / 'labels.txt').open('x') as stream:
        for first in range(0, rows, 65536):
            stream.write('1\n' * min(65536, rows - first))
    command = [str(python), str(selector), '--base', str(out / 'base.fbin'), '--out', str(out / 'global-entry-proof.json')]
    environment = os.environ.copy()
    environment.update({key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})
    started = time.monotonic()
    with (out / 'selector.log').open('x') as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    proof = json.loads((out / 'global-entry-proof.json').read_text())
    require(proof['base_sha256'] == expected_sha and proof['base_vectors'] == rows and proof['dimensions'] == dimensions,
            'Global entry proof does not bind the expected corpus')
    require(proof['query_inputs_used'] is False and 0 <= proof['entry_point'] < rows
            and proof['selector_source']['sha256'] == expected_selector_sha, 'Invalid or unexpected global entry selector proof')
    after, selector_after = record(base), record(selector)
    require(before == after and selector_before == selector_after, 'Base or frozen selector changed while preparing inputs')
    result = {'schema_version': 1, 'status': 'complete', 'scope': 'Global graph input only; no graph build or ANN query performed',
              'ordinary_label': 1, 'universal_label_semantics_used': False, 'label_rows': rows, 'base_vectors': rows,
              'dimensions': dimensions, 'dtype': 'float32', 'metric': 'L2', 'query_inputs_used': False,
              'global_entry_point': proof['entry_point'], 'selector_command': command,
              'selector_seconds': time.monotonic() - started, 'base_before': before, 'base_after': after,
              'selector_before': selector_before, 'selector_after': selector_after,
              'build_settings_to_use': {'label_degree': 64, 'stitched_degree': 64, 'build_l': 100, 'alpha': 1.2,
                                        'pq': 'Reuse the frozen filtered corpus PQ pair and its corpus-bound manifest; no retraining'},
              'artifacts': {name: record(out / name) for name in ('base.fbin', 'labels.txt', 'global-entry-proof.json', 'selector.log')},
              'preparation_source': record(Path(__file__))}
    with (out / 'global-inputs.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--base-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selector', type=Path, required=True)
    parser.add_argument('--selector-sha256', required=True)
    args = parser.parse_args()
    result = prepare(args.base, args.base_sha256, args.output, args.selector, args.selector_sha256)
    print(json.dumps({'output': str(args.output), 'base_vectors': result['base_vectors'], 'entry_point': result['global_entry_point']}))


if __name__ == '__main__':
    main()
