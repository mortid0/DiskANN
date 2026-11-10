#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Synthetic routing-evidence check for the read-only graph inspector."""
import argparse
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('diagnose_graph_edges', Path(__file__).with_name('diagnose_graph_edges.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_matrix(path, rows, columns, values):
    path.write_bytes(struct.pack('<II', rows, columns) + struct.pack(f'<{len(values)}I', *values))


def write_graph(path, adjacency):
    block, dimensions, degree = 4096, 2, 3
    width = dimensions + 4 + 4 * degree
    data = bytearray(block * 2)
    struct.pack_into('<II10Q3Q', data, 0, 96, 1, len(adjacency), dimensions, 0, width,
                     block // width, 0, 0, 0, len(data), 0, block, 1, 0)
    for vertex, neighbors in enumerate(adjacency):
        offset = block + vertex * width + dimensions
        struct.pack_into('<I', data, offset, len(neighbors))
        struct.pack_into(f'<{len(neighbors)}I', data, offset + 4, *neighbors)
    path.write_bytes(data)


class RoutingEvidence(unittest.TestCase):
    def test_distinguishes_already_discovered_hit_from_added_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = root / 'old.index', root / 'new.index'
            write_graph(old, [[], [2], [], [], []])
            write_graph(new, [[], [2, 4], [], [4], []])
            write_matrix(root / 'query_labels.u32bin', 1, 1, [9])
            write_matrix(root / 'groundtruth.bin', 1, 2, [2, 4])
            write_matrix(root / 'cpp.bin', 1, 2, [2, 4])
            write_matrix(root / 'label_9.u32bin', 5, 1, [0, 1, 2, 3, 4])
            raw = root / 'rust.jsonl'
            raw.write_text(json.dumps({'kind': 'query_detail', 'query_index': 0, 'label': 9, 'ids': [1, 3]}) + '\n')
            result = module.analyze(argparse.Namespace(old_graph=old, new_graph=new, prepared=root,
                cpp_result_ids=root / 'cpp.bin', rust_query_details=raw, labels=[9], sample_rows=5, seed=1))
            label = result['labels'][0]
            self.assertEqual(label['sample_edge_changes']['new_edges'], 2)
            self.assertEqual(label['sample_edge_changes']['removed_edges'], 0)
            diagnostic = label['cpp_only_hit_diagnostics']
            self.assertEqual(diagnostic['cpp_only_true_hits'], 2)
            self.assertEqual(diagnostic['old_hit_discovered_from_returned_expanded_vertex'], 1)
            self.assertEqual(diagnostic['new_route_from_old_returned_expanded_vertex'], 1)
            self.assertEqual(diagnostic['new_route_for_not_previously_discovered_hit'], 1)
            self.assertEqual(label['examples'][0]['old_discovered_from_returned_vertices'], [2])
            self.assertEqual(label['examples'][0]['new_route_from_old_returned_vertices'], [4])


if __name__ == '__main__':
    unittest.main()
