# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import struct
from pathlib import Path
import tempfile
import unittest
import select_global_entry_f32 as entry


class GlobalEntryTests(unittest.TestCase):
    def fixture(self, values):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'base.fbin'
        path.write_bytes(struct.pack('<II', len(values), 1) + struct.pack(f'<{len(values)}f', *values))
        return path

    def test_nearest_centroid_member_and_distance_tie_id(self):
        result = entry.select(self.fixture([0., 2., 3., 9.]), block_rows=2)
        self.assertEqual(result['entry_point'], 2)
        self.assertFalse(result['query_inputs_used'])
        self.assertEqual(entry.select(self.fixture([0., 2.]))['entry_point'], 0)

    def test_nonfinite_base_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            entry.select(self.fixture([0., float('nan')]))


if __name__ == '__main__':
    unittest.main()
