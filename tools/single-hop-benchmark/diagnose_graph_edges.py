#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Inspect frozen/new SSD routing edges without modifying index or search results.

Only returned Rust results are known expanded vertices. An edge from one of them
can establish discovery of a missed ground-truth point, but is not a full trace.
Graph samples describe geometry and do not imply recall or timing improvements.
"""
import argparse
from array import array
from collections import Counter
import json
import mmap
from pathlib import Path
import random
import statistics
import struct
import sys


def matrix_u32(path):
    with Path(path).open('rb') as stream:
        rows, columns = struct.unpack('<II', stream.read(8))
        values = array('I')
        values.frombytes(stream.read(rows * columns * 4))
        if sys.byteorder != 'little':
            values.byteswap()
        if len(values) != rows * columns:
            raise ValueError(f'truncated u32 matrix: {path}')
    return rows, columns, values


class DiskGraph:
    """Native v1 or C++ packed uint8/no-associated-data disk graph."""
    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open('rb')
        self.data = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        header_bytes, columns = struct.unpack_from('<II', self.data)
        if columns != 1 or header_bytes not in (9, 72, 80, 96):
            raise ValueError(f'unrecognized graph header: {header_bytes}x{columns}')
        self.rows, self.dimensions, _, self.width, self.per_block = struct.unpack_from('<5Q', self.data, 8)
        self.block = struct.unpack_from('<Q', self.data, 88)[0] if header_bytes == 96 else 4096
        if not self.block or not self.width:
            raise ValueError('zero disk block or node width')
        self.degree_bound = (self.width - self.dimensions) // 4 - 1
        if self.degree_bound < 0:
            raise ValueError('this tool requires uint8 vector and no associated data')
        if header_bytes == 96 and struct.unpack_from('<Q', self.data, 80)[0] != 0:
            raise ValueError('associated data unsupported')
        self.cache = {}

    def neighbors(self, vertex):
        if vertex not in self.cache:
            if not 0 <= vertex < self.rows:
                raise ValueError(f'out-of-range vertex {vertex}')
            if self.per_block:
                offset = (1 + vertex // self.per_block) * self.block + (vertex % self.per_block) * self.width
            else:
                offset = (1 + vertex * ((self.width + self.block - 1) // self.block)) * self.block
            offset += self.dimensions
            count = struct.unpack_from('<I', self.data, offset)[0]
            if count > self.degree_bound:
                raise ValueError(f'invalid degree {count} at {vertex}')
            self.cache[vertex] = struct.unpack_from(f'<{count}I', self.data, offset + 4)
        return self.cache[vertex]

    def identity(self):
        return {'path': str(self.path), 'bytes': len(self.data), 'rows': self.rows,
                'dimensions': self.dimensions, 'node_bytes': self.width,
                'nodes_per_block': self.per_block, 'block_bytes': self.block,
                'maximum_degree': self.degree_bound}

    def close(self):
        self.data.close()
        self.stream.close()


def summarize(values):
    ordered = sorted(values)
    return {'mean': statistics.mean(values), 'min': ordered[0], 'max': ordered[-1],
            'median': statistics.median(values), 'p99': ordered[(99 * len(values) + 99) // 100 - 1]}


def analyze(args):
    graphs = {'old': DiskGraph(args.old_graph), 'new': DiskGraph(args.new_graph)}
    try:
        if [(g.rows, g.dimensions) for g in graphs.values()][0] != [(g.rows, g.dimensions) for g in graphs.values()][1]:
            raise ValueError('graph dataset shapes differ')
        qrows, qcols, query_labels = matrix_u32(args.prepared / 'query_labels.u32bin')
        grows, k, gold = matrix_u32(args.prepared / 'groundtruth.bin')
        crows, ck, cpp = matrix_u32(args.cpp_result_ids)
        if qcols != 1 or grows != qrows or crows != qrows or ck != k:
            raise ValueError('query, truth and result shapes differ')
        rust = [None] * qrows
        with args.rust_query_details.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('kind') != 'query_detail':
                    continue
                i = row['query_index']
                if rust[i] is not None or row['label'] != query_labels[i]:
                    raise ValueError('duplicate query index or wrong label in raw Rust details')
                rust[i] = row['ids']
        if any(row is None for row in rust):
            raise ValueError('incomplete Rust query details')
        result = {'schema_version': 1, 'scope': 'quality-only edge inspection; no performance claims',
                  'graphs': {name: graph.identity() for name, graph in graphs.items()}, 'labels': []}
        rng = random.Random(args.seed)
        for label in args.labels:
            prows, pcols, posting = matrix_u32(args.prepared / f'label_{label}.u32bin')
            if pcols != 1:
                raise ValueError('postings must have one column')
            members = set(posting)
            sample = sorted(rng.sample(posting, min(args.sample_rows, prows)))
            degrees = {name: [] for name in graphs}
            label_degrees = {name: [] for name in graphs}
            edge_changes = Counter()
            for source in sample:
                edges = {name: set(graph.neighbors(source)) for name, graph in graphs.items()}
                for name in graphs:
                    degrees[name].append(len(edges[name]))
                    label_degrees[name].append(len(edges[name] & members))
                edge_changes['new_edges'] += len(edges['new'] - edges['old'])
                edge_changes['removed_edges'] += len(edges['old'] - edges['new'])
                edge_changes['new_label_edges'] += len((edges['new'] - edges['old']) & members)
                edge_changes['removed_label_edges'] += len((edges['old'] - edges['new']) & members)
            diagnostics = Counter()
            examples = []
            for i, qlabel in enumerate(query_labels):
                if qlabel != label:
                    continue
                diagnostics['queries'] += 1
                truth = set(gold[i * k:(i + 1) * k])
                lost = (set(cpp[i * k:(i + 1) * k]) & truth) - set(rust[i])
                if not lost:
                    continue
                diagnostics['queries_with_cpp_only_true_hits'] += 1
                diagnostics['cpp_only_true_hits'] += len(lost)
                discovered = set()
                added_routes = set()
                for source in rust[i]:
                    before = set(graphs['old'].neighbors(source))
                    after = set(graphs['new'].neighbors(source))
                    discovered.update(before & lost)
                    added_routes.update((after - before) & lost)
                diagnostics['old_hit_discovered_from_returned_expanded_vertex'] += len(discovered)
                diagnostics['new_route_from_old_returned_expanded_vertex'] += len(added_routes)
                diagnostics['new_route_for_not_previously_discovered_hit'] += len(added_routes - discovered)
                if len(examples) < 20:
                    examples.append({'query_index': i, 'cpp_only_true_hits': sorted(lost),
                                     'old_discovered_from_returned_vertices': sorted(discovered),
                                     'new_route_from_old_returned_vertices': sorted(added_routes)})
            result['labels'].append({'label': label, 'posting_rows': prows, 'sample_rows': len(sample),
                                     'sample_seed': args.seed,
                                     'degree': {name: summarize(degrees[name]) for name in graphs},
                                     'in_label_degree': {name: summarize(label_degrees[name]) for name in graphs},
                                     'sample_edge_changes': dict(edge_changes),
                                     'cpp_only_hit_diagnostics': dict(diagnostics), 'examples': examples})
        result['limitations'] = [
            'Independent concurrent Vamana builds can add/remove edges beyond the final-prune change; graph diff alone is not a causal ablation.',
            'Only old returned IDs are known expanded. A missing edge from these IDs does not establish that search failed to discover the target elsewhere.',
            'An old edge from a returned expanded ID proves a target was discovered, but PQ priority can keep it from being expanded and exact-reranked.',
            'New graph edges can change fixed-L traversal non-monotonically; added routes do not guarantee recall improvement.',
            'Per-label graph degrees use a reproducible uniform posting sample, not all graph rows.'
        ]
        return result
    finally:
        for graph in graphs.values():
            graph.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-graph', type=Path, required=True)
    parser.add_argument('--new-graph', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--cpp-result-ids', type=Path, required=True)
    parser.add_argument('--rust-query-details', type=Path, required=True)
    parser.add_argument('--labels', type=int, nargs='+', default=[9, 24, 89, 98, 2495])
    parser.add_argument('--sample-rows', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=293748)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.sample_rows <= 0 or args.out.exists():
        parser.error('sample-rows must be positive and output must not exist')
    args.out.write_text(json.dumps(analyze(args), indent=2) + '\n')


if __name__ == '__main__':
    main()
