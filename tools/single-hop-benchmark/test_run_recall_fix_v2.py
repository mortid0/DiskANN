#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Protect fresh-vs-development scope and held-out partition bindings."""
from contextlib import ExitStack
import json
import sys
import unittest
from unittest.mock import patch

import run_recall_fix_v2 as runner
import run_recall_fix as primary_runner
import test_run_recall_fix as fixture


class SecondCandidateRunnerTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.RecallFixRunnerTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.write_split('secondary-validation')

    def write_split(self, phase):
        c = self.case
        suffix = phase.split('-')[1]
        c.prepared_manifest.update(
            split=phase.replace('-', '_'),
            source_query_namespace='mixed_public_and_official_private_2727415019_100K',
            validation_scope=f'recall_fix_supplemental_shifted_atomic_label_{suffix}_v2',
            split_protocol_sha256='protocol', source_pool_manifest={'sha256': 'pool'})
        path = c.prepared / 'manifest.json'
        path.write_text(json.dumps(c.prepared_manifest))
        c.proof.write_text(json.dumps({
            'scope': 'supplemental_shifted_atomic_label_v2_split',
            'policy_sha256': runner.sha(c.plan), 'split_protocol_sha256': 'protocol',
            'original_pool_manifest_sha256': 'pool',
            'partitions': {phase: {'queries': 2001, 'manifest_sha256': runner.sha(path),
                                   'artifacts': c.prepared_manifest['artifacts']}}}))

    def invoke(self, phase='secondary-validation', quality_only=False):
        c = self.case
        arguments = c.arguments(phase=phase, quality_only=quality_only)
        arguments += ['--candidate-binary', str(c.binaries['candidate'])]
        with ExitStack() as stack:
            stack.enter_context(patch.object(fixture, 'runner', runner))
            stack.enter_context(patch.object(c, 'arguments', return_value=arguments))
            stack.enter_context(patch.object(primary_runner, 'FROZEN_PLAN', runner.sha(c.plan)))
            c.invoke()

    def test_fresh_validation_keeps_six_orders_and_original_search_breadth(self):
        self.invoke()
        c = self.case
        manifest = json.loads((c.out / 'triple-run-manifest.json').read_text())
        self.assertEqual(len(c.commands), 18)
        self.assertTrue(manifest['timing_authoritative'])
        self.assertEqual(manifest['query_freshness'], 'previously_unsearched_partition')
        self.assertEqual(manifest['phase'], 'secondary-validation')
        for command in c.commands:
            if command[0] != sys.executable:
                self.assertEqual(command[command.index('--l') + 1], '64')
                self.assertEqual(command[command.index('--label-l') + 1], str(c.plan.resolve()))

    def test_calibration_is_quality_only_and_cannot_open_validation_partition(self):
        with self.assertRaises(ValueError):
            self.invoke(phase='secondary-calibration', quality_only=True)
        self.assertEqual(self.case.commands, [])
        self.write_split('secondary-calibration')
        self.invoke(phase='secondary-calibration', quality_only=True)
        self.assertEqual(len(self.case.commands), 3)
        identity = json.loads((self.case.out / 'identity.json').read_text())
        self.assertEqual(identity['phase'], 'secondary-calibration')
        self.assertEqual(identity['query_freshness'], 'previously_unsearched_partition')

    def test_previously_exposed_primary_queries_are_explicit_development_only(self):
        c = self.case
        c.prepared_manifest.update(split='validation',
            source_query_namespace='official_private_2727415019_100K')
        c.write_manifest_and_proof()
        self.invoke(phase='development-recheck', quality_only=True)
        freeze = json.loads((c.out / 'candidate-freeze.json').read_text())
        manifest = json.loads((c.out / 'triple-run-manifest.json').read_text())
        self.assertIn('cannot claim these queries as fresh', freeze['statement'])
        self.assertEqual(manifest['query_freshness'], 'previously_exposed_development')
        self.assertFalse(manifest['timing_authoritative'])
        self.assertEqual(len(c.commands), 3)

    def test_exposed_queries_can_measure_performance_without_a_fresh_quality_claim(self):
        c = self.case
        c.prepared_manifest.update(split='validation',
            source_query_namespace='official_private_2727415019_100K')
        c.write_manifest_and_proof()
        self.invoke(phase='development-recheck', quality_only=False)
        manifest = json.loads((c.out / 'triple-run-manifest.json').read_text())
        self.assertEqual(len(c.commands), 18)
        self.assertTrue(manifest['timing_authoritative'])
        self.assertEqual(manifest['query_freshness'], 'previously_exposed_development')

    def test_mutation_after_commitment_prevents_any_search(self):
        c = self.case
        (c.prepared / 'query_ids.u32bin').write_bytes(b'mutated')
        with self.assertRaisesRegex(ValueError, 'artifact'):
            self.invoke()
        self.assertEqual(c.commands, [])
        self.assertFalse(c.out.exists())


if __name__ == '__main__':
    unittest.main()
