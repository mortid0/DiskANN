#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Ensure spatial routing cannot relabel exposed calibration as fresh validation."""
import json
import unittest
from unittest.mock import patch

import run_recall_fix_v3 as runner
import test_run_recall_fix_v2 as fixture


class SpatialEntryProtocolTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.SecondCandidateRunnerTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.amendment = self.case.case.root / 'amendment.json'
        self.amendment.write_text(json.dumps({'secondary_validation_unsearched': True}))
        original = self.case.case.arguments
        self.case.case.arguments = lambda **kwargs: original(**kwargs) + ['--development-amendment', str(self.amendment)]

    def invoke(self, phase='secondary-validation', quality_only=False):
        with patch.object(fixture, 'runner', runner):
            self.case.invoke(phase=phase, quality_only=quality_only)

    def test_exposed_secondary_calibration_is_development_only(self):
        self.case.write_split('secondary-calibration')
        self.invoke(phase='secondary-development', quality_only=True)
        c = self.case.case
        freeze = json.loads((c.out / 'candidate-freeze.json').read_text())
        identity = json.loads((c.out / 'identity.json').read_text())
        self.assertEqual(len(c.commands), 3)
        self.assertEqual(identity['query_freshness'], 'previously_exposed_development')
        self.assertEqual(identity['development_amendment_sha256'], runner.sha(self.amendment))
        self.assertIn('candidate 3 cannot claim', freeze['statement'])

    def test_original_final_partition_remains_fresh_with_six_orders(self):
        self.invoke()
        c = self.case.case
        identity = json.loads((c.out / 'identity.json').read_text())
        self.assertEqual(len(c.commands), 18)
        self.assertEqual(identity['query_freshness'], 'previously_unsearched_partition')
        self.assertEqual(identity['development_amendment_sha256'], runner.sha(self.amendment))

    def test_calibration_cannot_be_used_as_final_validation(self):
        self.case.write_split('secondary-calibration')
        with self.assertRaises(ValueError):
            self.invoke()
        self.assertEqual(self.case.case.commands, [])

    def test_unreserved_validation_rejected_before_search(self):
        self.amendment.write_text(json.dumps({'secondary_validation_unsearched': False}))
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.case.case.commands, [])


if __name__ == '__main__':
    unittest.main()
