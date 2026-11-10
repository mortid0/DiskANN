#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Topology diagnostics distinguish global counts from sampled valid routing edges."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest


DIRECTORY = Path(__file__).parent
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location('diagnose_graph_topology', DIRECTORY / 'diagnose_graph_topology.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_graph(path, adjacency, cpp=False):
    block, dimensions, degree = 4096, 2, 3
    width = dimensions + 4 + 4 * degree
    data = bytearray(block * 2)
    if cpp:
        struct.pack_into('<II9Q', data, 0, 9, 1, len(adjacency), dimensions, 0,
                         width, block // width, 0, 0, 0, len(data))
    else:
        struct.pack_into('<II10Q2Q', data, 0, 96, 1, len(adjacency), dimensions, 0,
                         width, block // width, 0, 0, 0, len(data), 0, block, 1)
    for vertex, neighbors in enumerate(adjacency):
        offset = block + vertex * width + dimensions
        struct.pack_into('<I', data, offset, len(neighbors))
        struct.pack_into(f'<{len(neighbors)}I', data, offset + 4, *neighbors)
    path.write_bytes(data)


def write_posting(path, posting):
    path.write_bytes(struct.pack('<II', len(posting), 1)
                     + struct.pack(f'<{len(posting)}I', *posting))


class TopologyDiagnostics(unittest.TestCase):
    def test_three_graphs_global_degrees_validity_and_frozen_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new, cpp = (root / name for name in ['old.index', 'new.index', 'cpp.index'])
            write_graph(old, [[1, 2], [0], [2, 3, 3], [99]])
            write_graph(new, [[1, 2, 3], [0], [1, 3], [2]])
            write_graph(cpp, [[1], [0, 2], [0, 1], [1]], cpp=True)
            write_posting(root / 'label_9.u32bin', [0, 1, 2, 3])
            write_posting(root / 'label_24.u32bin', [1, 2])
            args = argparse.Namespace(graph=[f'old={old}', f'new={new}', f'cpp={cpp}'],
                                      prepared=root, all_labels=False, labels=[9],
                                      full_degrees=True, sample_rows=4, seed=42)
            result = module.analyze(args)
            global_old = result['graphs']['old']['global_degree']
            self.assertEqual(global_old['histogram'], {'1': 2, '2': 1, '3': 1})
            self.assertEqual(global_old['edges'], 7)
            self.assertEqual(global_old['file_sha256'], hashlib.sha256(old.read_bytes()).hexdigest())
            self.assertEqual(result['graphs']['cpp']['global_degree']['edges'], 6)
            label = result['labels'][0]
            validation = label['sample_graphs']['old']['edge_validation']
            self.assertEqual(validation['self_edges'], 1)
            self.assertEqual(validation['duplicate_edges'], 1)
            self.assertEqual(validation['out_of_range_edges'], 1)
            self.assertEqual(validation['valid_unique_nonself_eligible_edges'], 4)
            self.assertEqual(label['sample_pairwise_edge_overlap']['old_vs_new'], {
                'shared_edges': 4, 'first_only_edges': 0, 'second_only_edges': 3,
                'shared_eligible_edges': 4, 'first_only_eligible_edges': 0,
                'second_only_eligible_edges': 3})
            args.all_labels = True
            args.full_degrees = False
            repeated = module.analyze(args)
            self.assertEqual(repeated['labels'][0]['sample_sha256'], label['sample_sha256'])
            self.assertEqual([row['label'] for row in repeated['labels']], [9, 24])

    def test_truncated_declared_layout_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'truncated.index'
            write_graph(path, [[], []])
            path.write_bytes(path.read_bytes()[:4096])
            graph = module.DiskGraph(path)
            try:
                with self.assertRaisesRegex(ValueError, 'shorter'):
                    module.validate_layout(graph)
            finally:
                graph.close()


if __name__ == '__main__':
    unittest.main()
