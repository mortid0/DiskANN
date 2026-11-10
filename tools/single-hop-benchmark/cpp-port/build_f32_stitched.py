#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Build a checksum-bound float32 stitched index without overwriting artifacts.

The float32 PQ helper corrects upstream generate_pq's unconditional OPQ encoder
argument. The copied build_stitched_f32 helper fixes 32-bit extraction offsets
for corpora above 4 GiB and optionally omits an unqueried label graph. All three search
implementations consume the same published graph, labels and plain-PQ pair.
"""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time


SUFFIXES = ('_disk.index', '_disk.index_labels.txt', '_disk.index_labels_map.txt',
            '_disk.index_labels_to_medoids.txt', '_pq_pivots.bin', '_pq_compressed.bin')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            result.update(block)
    return result.hexdigest()


def record(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'access_path': str(path.absolute()),
            'bytes': path.stat().st_size, 'sha256': sha(path)}


def execute(command, log, environment):
    started = time.monotonic()
    with log.open('x') as stream:
        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return {'command': command, 'log': str(log), 'elapsed_seconds': time.monotonic() - started,
            'log_identity': record(log)}


def matrix_header(path, width, offset=0, expected_end=None):
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        raw = stream.read(8)
    require(len(raw) == 8, f'Truncated matrix header: {path}')
    rows, columns = struct.unpack('<II', raw)
    end = offset + 8 + rows * columns * width
    require(rows > 0 and columns > 0 and end <= Path(path).stat().st_size,
            f'Malformed matrix shape/payload: {path}')
    if expected_end is not None:
        require(end == expected_end, f'Matrix file/block size mismatch: {path}')
    return rows, columns


def validate_labels(path, rows):
    memberships, maximum, seen_rows = 0, 0, 0
    with Path(path).open('rb') as stream:
        for line in stream:
            values = line.strip().split(b',')
            require(all(re.fullmatch(rb'[0-9]+', value) for value in values), 'Invalid or blank point label row')
            labels = list(map(int, values))
            require(all(label < 2 ** 32 for label in labels) and len(set(labels)) == len(labels),
                    'Invalid or duplicate point label')
            memberships += len(labels)
            maximum = max(maximum, len(labels))
            seen_rows += 1
    require(seen_rows == rows, 'Base and point-label row counts differ')
    return {'rows': rows, 'memberships_including_unqueried_labels': memberships, 'maximum_labels_per_row': maximum}


def validate_pq(prefix, base_rows, dimensions, chunks=32):
    pivots, codes = (Path(str(prefix) + suffix) for suffix in ('_pq_pivots.bin', '_pq_compressed.bin'))
    require(not Path(str(pivots) + '_rotation_matrix.bin').exists()
            and not Path(str(prefix) + '_pq_rotation_matrix.bin').exists(), 'OPQ rotation artifacts are unsupported')
    require(matrix_header(codes, 1, expected_end=codes.stat().st_size) == (base_rows, chunks),
            'PQ code rows/chunks differ from this base')
    with pivots.open('rb') as stream:
        require(matrix_header(pivots, 8, expected_end=40) == (4, 1), 'Expected four-offset plain-PQ metadata')
        stream.seek(8)
        offsets = struct.unpack('<4Q', stream.read(32))
        require(offsets[0] == 4096 and all(a < b for a, b in zip(offsets, offsets[1:]))
                and offsets[3] == pivots.stat().st_size, 'Invalid or trailing PQ pivot blocks')
        require(matrix_header(pivots, 4, offsets[0], offsets[1]) == (256, dimensions), 'PQ pivot dimensions differ from base')
        require(matrix_header(pivots, 4, offsets[1], offsets[2]) == (dimensions, 1), 'PQ centroid dimensions differ from base')
        require(matrix_header(pivots, 4, offsets[2], offsets[3]) == (chunks + 1, 1), 'PQ chunk-boundary shape differs')
        for first, count in ((offsets[0] + 8, 256 * dimensions), (offsets[1] + 8, dimensions)):
            stream.seek(first)
            values = array('f')
            values.fromfile(stream, count)
            if sys.byteorder != 'little':
                values.byteswap()
            require(all(map(math.isfinite, values)), 'Non-finite PQ pivots or centroid')
        stream.seek(offsets[2] + 8)
        boundaries = struct.unpack(f'<{chunks + 1}I', stream.read(4 * (chunks + 1)))
        require(boundaries[0] == 0 and boundaries[-1] == dimensions
                and all(a < b for a, b in zip(boundaries, boundaries[1:])), 'Invalid PQ chunk boundaries')
    return {'rows': base_rows, 'bytes_per_point': chunks, 'dimensions': dimensions, 'centers': 256,
            'chunk_boundaries': list(boundaries), 'plain_pq': True,
            'files': {'_pq_pivots.bin': record(pivots), '_pq_compressed.bin': record(codes)}}


def validate_pq_provenance(prefix, proof, base_sha256, pair):
    data = json.loads(Path(proof).read_text())
    require(data.get('base_sha256') == base_sha256, 'Reused PQ provenance belongs to another base corpus')
    for suffix, identity in pair['files'].items():
        recorded = data.get('shared_pq', {}).get(suffix, {})
        require(recorded.get('sha256') == identity['sha256'], f'Reused PQ provenance does not bind {suffix}')
    return record(proof)


def validate_entry_provenance(path, entry, rows, dimensions, base_sha256):
    proof = json.loads(Path(path).read_text())
    require(proof.get('selector') == 'nearest_to_full_base_centroid_float64_distance_then_ID'
            and proof.get('query_inputs_used') is False, 'Global entry selector must be the audited base-only centroid policy')
    require(proof.get('entry_point') == entry and proof.get('base_vectors') == rows
            and proof.get('dimensions') == dimensions and proof.get('base_sha256') == base_sha256,
            'Global entry provenance does not match this corpus or entry ID')
    source = proof.get('selector_source', {})
    original = Path(source.get('path', ''))
    candidates = [original, Path(__file__).parent / original.name, Path(__file__).parent.parent / original.name]
    matching = [candidate for candidate in candidates if candidate.is_file()
                and candidate.stat().st_size == source.get('bytes') and sha(candidate) == source.get('sha256')]
    require(bool(matching), 'Global entry selector source is unavailable or changed')
    return {'proof': record(path), 'selector_source': record(matching[0]), 'selector': proof['selector']}


def unchanged(before):
    after = {name: record(Path(identity.get('access_path', identity['path']))) for name, identity in before.items()}
    for name, identity in before.items():
        require(after[name]['path'] == identity['path'] and after[name]['bytes'] == identity['bytes']
                and after[name]['sha256'] == identity['sha256'],
                f'Input changed during construction: {name}')
    return after


def copy_exclusive(source, target):
    with Path(source).open('rb') as src, Path(target).open('xb') as dst:
        shutil.copyfileobj(src, dst, 16 << 20)


def validate_disk(path, rows, dimensions):
    with path.open('rb') as source:
        header = source.read(80)
    require(len(header) == 80 and struct.unpack_from('<II', header) == (9, 1), 'Unsupported SSD metadata shape')
    n, d, medoid, length, per_sector, frozen, frozen_id, reorder, size = struct.unpack_from('<9Q', header, 8)
    require((n, d) == (rows, dimensions) and medoid < rows and frozen in (0, 1)
            and (not frozen or frozen_id < rows) and reorder == 0, 'SSD metadata does not match this corpus')
    require(length >= 4 * dimensions + 4 and (length - 4 * dimensions - 4) % 4 == 0,
            'Invalid float32 SSD node length')
    require(per_sector == 4096 // length, 'SSD nodes-per-sector mismatch')
    sectors = (rows + per_sector - 1) // per_sector if per_sector else rows * ((length + 4095) // 4096)
    require(size == path.stat().st_size == (sectors + 1) * 4096, 'SSD file size does not match layout metadata')
    return {'rows': n, 'dimensions': d, 'medoid': medoid, 'max_node_bytes': length, 'nodes_per_sector': per_sector,
            'frozen_points': frozen, 'reorder': reorder, 'bytes': size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--prepared-data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pq-prefix', type=Path, help='Existing plain-PQ pair bound to this base by companion manifest')
    parser.add_argument('--pq-provenance', type=Path, help='Override <pq-prefix>.manifest.json for the base/PQ checksum binding')
    parser.add_argument('--build-evidence', type=Path, help='Compiler/source/library evidence for the supplied executables')
    parser.add_argument('--training-points', type=int, default=100000)
    parser.add_argument('--name', default='f32_stitched_R32_R128_L100')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--label-degree', type=int, default=32)
    parser.add_argument('--stitched-degree', type=int, default=128)
    parser.add_argument('--build-l', type=int, default=100)
    parser.add_argument('--skip-label-build', type=int, help='Omit only this explicitly unqueried label graph; requires build_stitched_f32')
    parser.add_argument('--global-entry-point', type=int, help='Original base ID for an optional global-graph baseline; omission preserves upstream ID0')
    parser.add_argument('--global-entry-provenance', type=Path, help='Audited base-only centroid selector proof; required with --global-entry-point')
    args = parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.name) is not None, 'Name must be one safe filename component')
    require(args.threads > 0 and 0 < args.label_degree <= args.stitched_degree and args.build_l >= args.label_degree,
            'Invalid construction concurrency/degree/L')
    require(args.training_points >= 256, 'PQ training sample must target at least256 rows')
    require(args.pq_prefix is not None or args.pq_provenance is None, '--pq-provenance requires --pq-prefix')
    require((args.global_entry_point is None) == (args.global_entry_provenance is None),
            '--global-entry-point and --global-entry-provenance must be supplied together')
    data, output, bins = args.prepared_data.resolve(), args.output_dir.resolve(), args.bin_dir.resolve()
    base_path, label_path = data / 'base.fbin', data / 'labels.txt'
    base_rows, dimensions = matrix_header(base_path, 4, expected_end=base_path.stat().st_size)
    require(base_rows < 2 ** 32 and dimensions >= 32, 'Base IDs must fit uint32 and dimensions must support 32 PQ chunks')
    entry_point = 0 if args.global_entry_point is None else args.global_entry_point
    require(0 <= entry_point < base_rows, 'Global entry point is outside the base corpus')
    label_stats = validate_labels(label_path, base_rows)
    if args.skip_label_build is not None:
        require(0 <= args.skip_label_build < 2 ** 32, 'Skipped label must fit uint32')
        with (data / 'query_labels.txt').open() as query_labels:
            require(all(int(line) != args.skip_label_build for line in query_labels), 'Cannot skip a queried label graph')
        manifest = json.loads((data / 'manifest.json').read_text())
        require(all(int(entry.get('label', entry.get('original_label'))) != args.skip_label_build
                    for entry in manifest['labels']), 'Skipped label must be absent from the complete workload label bank')
    builder = 'build_stitched_f32'
    binary_names = [builder, 'utils/create_disk_layout'] + ([] if args.pq_prefix else ['utils/generate_pq_f32'])
    source_paths = {'runner': Path(__file__), 'pq_helper_source': Path(__file__).with_name('generate_pq_f32.cpp'),
                    'stitched_helper_source': Path(__file__).with_name('build_stitched_f32.cpp')}
    if args.build_evidence:
        source_paths['build_evidence'] = args.build_evidence.resolve()
    inputs = {'base': record(base_path), 'labels': record(label_path),
              **{f'binary:{name}': record(bins / name) for name in binary_names},
              **{f'source:{name}': record(path) for name, path in source_paths.items()}}
    entry_selection = None
    if args.global_entry_point is not None:
        entry_selection = validate_entry_provenance(args.global_entry_provenance, entry_point, base_rows, dimensions,
                                                    inputs['base']['sha256'])
        inputs['global_entry_provenance'] = entry_selection['proof']
        inputs['global_entry_selector_source'] = entry_selection['selector_source']
    pq_source = args.pq_prefix.resolve() if args.pq_prefix else None
    if pq_source:
        pair = validate_pq(pq_source, base_rows, dimensions)
        proof = args.pq_provenance.resolve() if args.pq_provenance else Path(str(pq_source) + '.manifest.json')
        inputs['pq_provenance'] = validate_pq_provenance(pq_source, proof, inputs['base']['sha256'], pair)
        inputs.update({f'pq:{suffix}': identity for suffix, identity in pair['files'].items()})
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.glob(args.name + '*')), 'Output prefix or partial artifacts already exist; choose a new name')
    work = output / (args.name + '.work')
    work.mkdir()  # Atomic exclusive claim; retained on failure for diagnosis.
    prefix, staged = output / args.name, work / args.name
    base = work / 'base.fbin'
    base.symlink_to(base_path)
    memory_graph = work / 'mem.index'
    environment = os.environ.copy()
    environment.update({'OMP_TOOL': 'disabled', 'VECLIB_MAXIMUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': str(args.threads)})
    encoding = None
    if pq_source is None:
        probability = min(1.0, args.training_points / base_rows)
        encoding = execute([str(bins / 'utils/generate_pq_f32'), 'float', str(base), str(staged), '32', str(probability), '0'],
                           work / 'pq.log', environment)
        encoding.update(training_sample_probability=probability, requested_training_points=args.training_points,
                        rng_policy='Pinned upstream Bernoulli sampler; actual artifacts frozen by hash',
                        upstream_correction='generate_pq_f32 encodes with use_opq=false; stock generate_pq unconditionally passes true')
    else:
        for suffix in ('_pq_pivots.bin', '_pq_compressed.bin'):
            copy_exclusive(Path(str(pq_source) + suffix), Path(str(staged) + suffix))
    pair = validate_pq(staged, base_rows, dimensions)  # Must pass before expensive graph construction.
    inputs_after_encoding = unchanged(inputs)
    command = [str(bins / builder), '--data_type', 'float', '--data_path', str(base),
               '--label_file', str(label_path), '--index_path_prefix', str(memory_graph), '-T', str(args.threads),
               '-R', str(args.label_degree), '-L', str(args.build_l), '--stitched_R', str(args.stitched_degree), '--alpha', '1.2']
    if args.skip_label_build is not None:
        command.extend(['--skip_label_build', str(args.skip_label_build)])
    if args.global_entry_point is not None:
        command.extend(['--global_entry_point', str(entry_point)])
    build = execute(command, work / 'build.log', environment)
    layout = execute([str(bins / 'utils/create_disk_layout'), 'float', str(base), str(memory_graph), str(staged) + '_disk.index'],
                     work / 'layout.log', environment)
    disk = validate_disk(Path(str(staged) + '_disk.index'), base_rows, dimensions)
    require(disk['medoid'] == entry_point, 'Serialized global entry differs from the declared base ID')
    for suffix in ('_labels.txt', '_labels_map.txt', '_labels_to_medoids.txt'):
        copy_exclusive(Path(str(memory_graph) + suffix), Path(str(staged) + '_disk.index' + suffix))
    pair_after = validate_pq(staged, base_rows, dimensions)
    require(pair_after['files'] == pair['files'], 'Frozen PQ pair changed during graph construction')
    inputs_after = unchanged(inputs)
    # Publication uses exclusive hard links from our own staging files. The final
    # manifest is written last; incomplete runs never acquire a success manifest.
    for suffix in SUFFIXES:
        os.link(Path(str(staged) + suffix), Path(str(prefix) + suffix))
    artifacts = {suffix: record(Path(str(prefix) + suffix)) for suffix in SUFFIXES}
    manifest = {'schema_version': 2, 'method': 'official_cpp_float32_stitched_vamana', 'dtype': 'float32',
                'status': 'complete', 'index_prefix': str(prefix), 'base_sha256': inputs['base']['sha256'],
                'labels_sha256': inputs['labels']['sha256'], 'runner_sha256': inputs['source:runner']['sha256'],
                'label_degree': args.label_degree, 'stitched_degree': args.stitched_degree, 'build_l': args.build_l,
                'threads': args.threads, 'labels': label_stats, 'build': build, 'disk_layout': layout,
                'pq_encoding': encoding, 'pq_rows': base_rows, 'pq_bytes': 32, 'pq_validation': pair_after,
                'shared_pq': {suffix: {**artifacts[suffix], 'source': str(pq_source) + suffix if pq_source else str(staged) + suffix,
                                      'target': str(prefix) + suffix} for suffix in ('_pq_pivots.bin', '_pq_compressed.bin')},
                'unqueried_label_graph_omitted': args.skip_label_build,
                'global_entry_point': entry_point, 'global_entry_selection': entry_selection,
                'stitched_source_correction': 'Copied pinned StitchedVamana uses size_t vector-byte strides before multiplying by global point IDs, avoiding upstream uint32 overflow above 4 GiB; queried graph construction otherwise unchanged',
                'omission_scope': None if args.skip_label_build is None else 'Only the declared unqueried sentinel label graph is omitted; per-point sentinel metadata and all base vectors remain; queried predicate graph construction is unchanged',
                'disk_metadata': disk, 'inputs_before': inputs, 'inputs_after_encoding': inputs_after_encoding,
                'inputs_after': inputs_after, 'artifacts': artifacts,
                'build_binaries_sha256': {name: inputs[f'binary:{name}']['sha256'] for name in binary_names}}
    target = Path(str(prefix) + '.manifest.json')
    completed = work / 'complete.manifest.json'
    with completed.open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.link(completed, target)
    print(json.dumps({'index_prefix': str(prefix), 'manifest': str(target)}))


if __name__ == '__main__':
    main()
