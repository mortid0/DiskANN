#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Global corpus inputs use one ordinary label and never need query artifacts."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('prepare_global_graph_f32', Path(__file__).with_name('prepare_global_graph_f32.py'))
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class GlobalInputTests(unittest.TestCase):
    def fixture(self, root):
        base = root / 'base.fbin'
        base.write_bytes(struct.pack('<II6f', 3, 2, 0, 0, 1, 0, 2, 0))
        selector = Path(__file__).with_name('select_global_entry_f32.py')
        return base, selector, PREPARE.record(base)['sha256'], PREPARE.record(selector)['sha256']

    def test_one_ordinary_label_and_exact_base_only_centroid(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, selector, digest, selector_digest = self.fixture(root)
            out = root / 'global-input'
            proof = PREPARE.prepare(base, digest, out, selector, selector_digest)
            self.assertEqual((out / 'labels.txt').read_text(), '1\n1\n1\n')
            self.assertTrue((out / 'base.fbin').is_symlink())
            self.assertEqual(proof['global_entry_point'], 1)
            self.assertFalse(proof['universal_label_semantics_used'])
            self.assertFalse(proof['query_inputs_used'])
            self.assertEqual(proof['build_settings_to_use']['label_degree'], 64)
            self.assertEqual(proof, json.loads((out / 'global-inputs.json').read_text()))
            self.assertFalse((out / 'query.fbin').exists())
            with self.assertRaises(FileExistsError):
                PREPARE.prepare(base, digest, out, selector, selector_digest)

    def test_unverified_base_or_selector_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, selector, digest, selector_digest = self.fixture(root)
            out = root / 'global-input'
            with self.assertRaisesRegex(ValueError, 'Transferred base'):
                PREPARE.prepare(base, 'wrong', out, selector, selector_digest)
            self.assertFalse(out.exists())
            with self.assertRaisesRegex(ValueError, 'frozen source'):
                PREPARE.prepare(base, digest, out, selector, 'wrong')
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
