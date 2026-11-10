#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Read-only hash reuse must detect mutations and bind historical source snapshots."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("validate_transfer", Path(__file__).with_name("validate_amazon_transfers.py"))
validate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validate)


class ValidationTests(unittest.TestCase):
    def test_symlink_hash_reuse_does_not_skip_mutation_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root/'base'
            real.write_bytes(b'frozen vectors')
            alias = root/'alias'
            alias.symlink_to('base')
            cache = validate.HashCache()
            self.assertEqual(cache.sha(real), cache.sha(alias))
            self.assertEqual(len(cache.entries), 1)
            self.assertEqual(cache.reused, 1)
            real.write_bytes(b'changed vector')
            with self.assertRaisesRegex(ValueError, 'changed'):
                cache.sha(alias)

    def test_matching_source_snapshot_is_selected_by_hash_and_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, current = root/'old.py', root/'current.py'
            old.write_text('historical frozen source')
            current.write_text('a different current version')
            cache = validate.HashCache()
            expected = cache.record(old)
            self.assertEqual(validate.source_match(expected, [current, old], cache)['sha256'], expected['sha256'])
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                validate.source_match(expected, [current], cache)

    def test_size_or_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data'
            path.write_bytes(b'abc')
            cache = validate.HashCache()
            expected = cache.record(path)
            expected['sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'differs'):
                cache.check(path, expected)


if __name__ == '__main__':
    unittest.main()
