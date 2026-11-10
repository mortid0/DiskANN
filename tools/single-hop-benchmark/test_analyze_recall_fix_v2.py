#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Check the v2 development/freshness separation and calibration asset lock."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import analyze_recall_fix_v2 as analysis
import test_analyze_recall_fix as v1_tests


class IndependentSplit(unittest.TestCase):
    def test_per_label_floor_quarter_partition_is_disjoint_complete_and_reproducible(self):
        labels = np.asarray([9] * 7 + [24] * 9 + [98] * 3)
        a = analysis.reconstruct_split(labels)
        b = analysis.reconstruct_split(labels)
        self.assertEqual(a, b)
        self.assertFalse(set(a['calibration']) & set(a['validation']))
        self.assertEqual(set(a['calibration']) | set(a['validation']), set(range(len(labels))))
        self.assertEqual(list(labels[a['calibration']]).count(9), 1)
        self.assertEqual(list(labels[a['calibration']]).count(24), 2)
        self.assertEqual(list(labels[a['calibration']]).count(98), 0)
        expected = sorted(range(7), key=lambda index: hashlib.sha256(f'20261025:split:{index}:9'.encode()).digest())[0]
        self.assertIn(expected, a['calibration'])


class V2Identity(unittest.TestCase):
    def fixture(self, root, phase='development-recheck', quality_only=False):
        args, logs, manifest, identity = v1_tests.FrozenIdentity().fixture(root)
        args.phase, args.quality_only = phase, quality_only
        args.primary_commitment = args.commitment
        args.split_commitment = args.commitment
        args.calibration_identity = None
        freshness = 'previously_exposed_development' if phase == 'development-recheck' else 'previously_unsearched_partition'
        identity.update(phase=phase, query_freshness=freshness,
                        runner_sha256=analysis.v1.sha(Path(analysis.__file__).with_name('run_recall_fix_v2.py')))
        for name in ('baseline', 'candidate'):
            before = copy.deepcopy(identity['engines'][name]['index_before'])
            before['files_sha256'].update({'vectors.bin': 'a' * 64, 'index_pq_pivots.bin': 'b' * 64,
                                          'index_pq_compressed.bin': 'c' * 64})
            identity['engines'][name]['index_before'] = before
            identity['engines'][name]['index_after'] = copy.deepcopy(before)
        freeze = json.loads(args.candidate_freeze.read_text())
        freeze.update(phase=phase, query_freshness=freshness,
                      index_identity={name: identity['engines'][name]['index_before'] for name in analysis.v1.ENGINES})
        args.candidate_freeze.write_text(json.dumps(freeze))
        identity['candidate_freeze_sha256'] = analysis.v1.sha(args.candidate_freeze)
        args.identity.write_text(json.dumps(identity))
        manifest.update(phase=phase, query_freshness=freshness, timing_authoritative=not quality_only, cpu_affinity=list(range(8)),
                        identity=identity, candidate_freeze_sha256=analysis.v1.sha(args.candidate_freeze))
        return args, logs, manifest, identity

    def test_timed_development_stays_exposed_and_does_not_require_fresh_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, logs, manifest, _ = self.fixture(Path(temporary))
            result = analysis.identity_proof(args, logs, manifest)
            self.assertEqual(result['query_freshness'], 'previously_exposed_development')
            self.assertIsNone(result['calibration_identity'])

    def test_rejects_development_relabeled_as_fresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, logs, manifest, identity = self.fixture(Path(temporary))
            identity['query_freshness'] = 'previously_unsearched_partition'
            args.identity.write_text(json.dumps(identity))
            with self.assertRaisesRegex(ValueError, 'phase/freshness'):
                analysis.identity_proof(args, logs, manifest)

    def test_calibration_accepts_one_quality_only_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, logs, manifest, _ = self.fixture(Path(temporary), phase='secondary-calibration', quality_only=True)
            result = analysis.identity_proof(args, logs, manifest)
            self.assertEqual(result['query_freshness'], 'previously_unsearched_partition')

    def test_fresh_validation_requires_calibration_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, logs, manifest, _ = self.fixture(Path(temporary), phase='secondary-validation')
            with self.assertRaisesRegex(ValueError, 'must bind frozen calibration assets'):
                analysis.identity_proof(args, logs, manifest)

    def test_fresh_validation_accepts_same_calibrated_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, logs, manifest, identity = self.fixture(Path(temporary), phase='secondary-validation')
            calibrated = copy.deepcopy(identity)
            calibrated['phase'] = 'secondary-calibration'
            args.calibration_identity = Path(temporary) / 'calibration-identity.json'
            args.calibration_identity.write_text(json.dumps(calibrated))
            result = analysis.identity_proof(args, logs, manifest)
            self.assertIsNotNone(result['calibration_identity'])

    def test_rejects_candidate_changed_after_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, logs, manifest, identity = self.fixture(Path(temporary), phase='secondary-validation')
            calibrated = copy.deepcopy(identity)
            calibrated['phase'] = 'secondary-calibration'
            calibrated['engines']['candidate']['binary_sha256'] = 'f' * 64
            args.calibration_identity = Path(temporary) / 'calibration-identity.json'
            args.calibration_identity.write_text(json.dumps(calibrated))
            with self.assertRaisesRegex(ValueError, 'assets changed between calibration'):
                analysis.identity_proof(args, logs, manifest)

    def test_rejects_missing_pq_files_in_both_inventories(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, logs, manifest, identity = self.fixture(Path(temporary))
            for name in ('baseline', 'candidate'):
                del identity['engines'][name]['index_before']['files_sha256']['index_pq_pivots.bin']
                del identity['engines'][name]['index_after']['files_sha256']['index_pq_pivots.bin']
            freeze = json.loads(args.candidate_freeze.read_text())
            freeze['index_identity'] = {name: identity['engines'][name]['index_before'] for name in analysis.v1.ENGINES}
            args.candidate_freeze.write_text(json.dumps(freeze))
            identity['candidate_freeze_sha256'] = analysis.v1.sha(args.candidate_freeze)
            args.identity.write_text(json.dumps(identity))
            manifest.update(identity=identity, candidate_freeze_sha256=analysis.v1.sha(args.candidate_freeze))
            with self.assertRaisesRegex(ValueError, 'frozen vector order or PQ artifacts'):
                analysis.identity_proof(args, logs, manifest)


if __name__ == '__main__':
    unittest.main()
