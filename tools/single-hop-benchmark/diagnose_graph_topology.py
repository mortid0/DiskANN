#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Read-only global degrees and reproducible label-edge samples for SSD graphs.

Global degree scans hash complete graph files without retaining adjacency lists.
Per-label eligible edge counts and graph overlaps are explicitly sampled. This
does not isolate final pruning from nondeterministic concurrent construction.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import struct

from diagnose_graph_edges import DiskGraph, matrix_u32


def node_offset(graph, vertex):
    if graph.per_block:
        return ((1 + vertex // graph.per_block) * graph.block
                + (vertex % graph.per_block) * graph.width)
    blocks = (graph.width + graph.block - 1) // graph.block
    return (1 + vertex * blocks) * graph.block


def validate_layout(graph):
    if graph.rows <= 0 or graph.dimensions <= 0:
        raise ValueError('graph requires positive rows and dimensions')
    if graph.per_block != graph.block // graph.width:
        raise ValueError('inconsistent node packing metadata')
    expected = node_offset(graph, graph.rows - 1) + graph.width
    if expected > len(graph.data):
        raise ValueError('graph is shorter than its declared node layout')


def degree_summary(histogram):
    count = sum(histogram.values())
    if count == 0:
        raise ValueError('empty degree population')

    def quantile(numerator, denominator):
        rank = (numerator * count + denominator - 1) // denominator
        running = 0
        for degree, frequency in sorted(histogram.items()):
            running += frequency
            if running >= rank:
                return degree
        raise AssertionError('unreachable quantile')

    return {'rows': count, 'edges': sum(degree * n for degree, n in histogram.items()),
            'mean': sum(degree * n for degree, n in histogram.items()) / count,
            'min': min(histogram), 'max': max(histogram), 'p50': quantile(1, 2),
            'p99': quantile(99, 100),
            'histogram': {str(degree): n for degree, n in sorted(histogram.items())}}


def global_degrees(graph):
    histogram = Counter()
    for vertex in range(graph.rows):
        offset = node_offset(graph, vertex) + graph.dimensions
        degree = struct.unpack_from('<I', graph.data, offset)[0]
        if degree > graph.degree_bound:
            raise ValueError(f'invalid degree {degree} at row {vertex} in {graph.path}')
        histogram[degree] += 1
    digest = hashlib.sha256()
    for start in range(0, len(graph.data), 8 * 1024 * 1024):
        digest.update(graph.data[start:start + 8 * 1024 * 1024])
    return {'file_sha256': digest.hexdigest(), **degree_summary(histogram),
            'zero_degree_rows': histogram[0],
            'rows_below_degree_bound': sum(n for degree, n in histogram.items()
                                           if degree < graph.degree_bound),
            'rows_at_degree_bound': histogram[graph.degree_bound]}


def sample_label(graphs, prepared, label, sample_rows, seed):
    rows, columns, posting = matrix_u32(prepared / f'label_{label}.u32bin')
    total_rows = next(iter(graphs.values())).rows
    if (columns != 1 or rows == 0 or posting[-1] >= total_rows
            or any(left >= right for left, right in zip(posting, posting[1:]))):
        raise ValueError(f'label {label} requires sorted unique nonempty in-bounds postings')
    members = frozenset(posting)
    # A label's sample remains identical when the requested label list changes.
    sample = sorted(random.Random((seed << 32) ^ label).sample(posting, min(sample_rows, rows)))
    sample_bytes = struct.pack(f'<{len(sample)}I', *sample)
    degrees = {name: Counter() for name in graphs}
    eligible_degrees = {name: Counter() for name in graphs}
    validity = {name: Counter() for name in graphs}
    pair_counts = {f'{first}_vs_{second}': Counter()
                   for first, second in itertools.combinations(graphs, 2)}
    for source in sample:
        edges = {}
        for name, graph in graphs.items():
            raw = graph.neighbors(source)
            unique = set(raw)
            validity[name]['raw_edges'] += len(raw)
            validity[name]['duplicate_edges'] += len(raw) - len(unique)
            validity[name]['self_edges'] += sum(neighbor == source for neighbor in raw)
            validity[name]['out_of_range_edges'] += sum(neighbor >= graph.rows for neighbor in raw)
            valid = {neighbor for neighbor in unique if neighbor < graph.rows and neighbor != source}
            validity[name]['valid_unique_nonself_edges'] += len(valid)
            validity[name]['valid_unique_nonself_eligible_edges'] += len(valid & members)
            degrees[name][len(raw)] += 1
            eligible_degrees[name][len(valid & members)] += 1
            edges[name] = valid
        for first, second in itertools.combinations(graphs, 2):
            before, after = edges[first], edges[second]
            counts = pair_counts[f'{first}_vs_{second}']
            counts['shared_edges'] += len(before & after)
            counts['first_only_edges'] += len(before - after)
            counts['second_only_edges'] += len(after - before)
            counts['shared_eligible_edges'] += len(before & after & members)
            counts['first_only_eligible_edges'] += len((before - after) & members)
            counts['second_only_eligible_edges'] += len((after - before) & members)
        # DiskGraph's cache is useful for short routing traces, not a full corpus.
        # Drop every sampled row after all graph comparisons have consumed it.
        for graph in graphs.values():
            graph.cache.clear()
    posting_digest = hashlib.sha256((prepared / f'label_{label}.u32bin').read_bytes()).hexdigest()
    return {'label': label, 'posting_rows': rows, 'posting_sha256': posting_digest,
            'sample_rows': len(sample), 'sample_sha256': hashlib.sha256(sample_bytes).hexdigest(),
            'sample_policy': 'uniform without replacement, per-label independent deterministic seed',
            'sample_graphs': {name: {'degree': degree_summary(degrees[name]),
                                     'eligible_degree': degree_summary(eligible_degrees[name]),
                                     'edge_validation': dict(validity[name])}
                              for name in graphs},
            'sample_pairwise_edge_overlap': {key: dict(value) for key, value in pair_counts.items()}}


def analyze(args):
    graphs = {}
    try:
        for item in args.graph:
            name, separator, path = item.partition('=')
            if not separator or not name or name in graphs:
                raise ValueError('--graph requires a unique NAME=PATH')
            graph = DiskGraph(path)
            graphs[name] = graph
            validate_layout(graph)
        if len(graphs) < 2 or len({(g.rows, g.dimensions) for g in graphs.values()}) != 1:
            raise ValueError('at least two graphs with matching shapes are required')
        labels = (sorted(int(path.stem.removeprefix('label_'))
                         for path in args.prepared.glob('label_*.u32bin'))
                  if args.all_labels else sorted(set(args.labels)))
        result = {'schema_version': 1, 'scope': 'read-only graph topology; no query or performance claims',
                  'graphs': {name: graph.identity() for name, graph in graphs.items()},
                  'seed': args.seed, 'requested_sample_rows_per_label': args.sample_rows,
                  'full_degree_scan': args.full_degrees, 'labels': [],
                  'limitations': [
                      'Degree scans inspect every degree field, not every edge ID. Edge validation and label membership counts use the stated samples.',
                      'Global degrees include every stored row. Graphs can differ in coverage of unqueried labels or sentinel-only rows; per-target eligible-edge samples are the matched-label comparison.',
                      'Concurrent independent Vamana builds can differ beyond final stitching; graph differences alone are not a causal ablation.',
                      'Higher eligible degree does not guarantee higher fixed-L recall or unchanged throughput.',
                  ]}
        if args.full_degrees:
            for name, graph in graphs.items():
                result['graphs'][name]['global_degree'] = global_degrees(graph)
        for label in labels:
            result['labels'].append(sample_label(graphs, args.prepared, label,
                                                  args.sample_rows, args.seed))
        return result
    finally:
        for graph in graphs.values():
            graph.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--labels', type=int, nargs='+', default=[9, 24, 89, 98, 2495])
    parser.add_argument('--all-labels', action='store_true')
    parser.add_argument('--sample-rows', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=293748)
    parser.add_argument('--full-degrees', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.sample_rows <= 0 or args.seed < 0 or args.out.exists():
        parser.error('positive sample-rows, nonnegative seed and a fresh output are required')
    args.out.write_text(json.dumps(analyze(args), indent=2) + '\n')


if __name__ == '__main__':
    main()
