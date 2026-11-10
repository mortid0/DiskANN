#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Guard fixed-L dual-entry policy, immutable pool and one final fresh launch."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import run_recall_fix_v4 as runner
import test_run_recall_fix_v2 as previous


class DualEntryRunnerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = previous.SecondCandidateRunnerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.case = self.fixture.case
        c = self.case
        self.candidate = c.matching_explicit_candidate()
        self.amendment = c.root / 'v4-amendment.json'
        self.protocol_path = c.root / 'v4-protocol.json'
        self.protocol_path.write_text('{"frozen": true}\n')
        self.calibration_path = c.root / 'v9-development-identity.json'
        self.publication = c.root / 'pool-publication.json'
        c.build_log = self.publication
        generation = self.candidate / 'generation-00000000000000000001'
        published = {str(path): runner.sha(path) for path in generation.iterdir() if path.is_file()}
        published.update({str(self.candidate / name): runner.sha(self.candidate / name) for name in ('CURRENT', 'updates.wal')})
        self.publication.write_text(json.dumps({'output_root': str(self.candidate), 'output_files_sha256': published}))
        original = c.arguments
        def arguments(**kwargs):
            result = original(**kwargs) + ['--candidate-root', str(self.candidate), '--development-amendment', str(self.amendment),
                                           '--protocol-freeze', str(self.protocol_path)]
            if self.calibration_path.exists():
                result += ['--calibration-identity', str(self.calibration_path)]
            return result
        c.arguments = arguments
        self.prepare_bindings()

    def prepare_bindings(self):
        c = self.case
        self.split_sha = runner.sha(c.proof)
        self.amendment.write_text(json.dumps({'secondary_validation_unsearched': True,
            'secondary_split_sha256': self.split_sha, 'entry_policy': runner.ENTRY_POLICY,
            'final_variant': True, 'candidate_root': str(self.candidate),
            'entry_publication_sha256': runner.sha(self.publication)}))
        self.protocol = {'archive': {'sha256': 'a' * 64},
                         'runtime_candidate_binary_sha256': runner.sha(c.binaries['candidate']),
                         'development_amendment_sha256': runner.sha(self.amendment)}
        native = {'baseline': c.catalogs['baseline'].parent.parent, 'candidate': self.candidate}
        indexes = {name: runner.native_identity(root) for name, root in native.items()}
        indexes['cpp'] = {suffix: runner.sha(c.root / ('indexes/yfcc10m_stitched_R32_R128_L100' + suffix)) for suffix in runner.CPP_SUFFIXES}
        sources = {'baseline': c.root / 'artifacts/heldout/lifecycle-v5-heldout-complete-sources.tar.gz', 'candidate': c.archive}
        self.calibration = {'protocol_version': 4, 'phase': 'secondary-development',
            'query_freshness': 'previously_exposed_development', 'entry_policy': copy.deepcopy(runner.ENTRY_POLICY),
            'protocol_source_freeze_sha256': runner.sha(self.protocol_path),
            'development_amendment_sha256': runner.sha(self.amendment),
            'engines': {name: {'index_after': indexes[name], 'binary_sha256': runner.sha(c.binaries[name]),
                              **({'source_archive': {'sha256': runner.sha(sources[name])}} if name in sources else {})}
                        for name in runner.ENGINES}}
        self.calibration_path.write_text(json.dumps(self.calibration))

    def invoke(self, phase='secondary-validation', quality_only=False):
        with ExitStack() as stack:
            stack.enter_context(patch.object(previous, 'runner', runner))
            stack.enter_context(patch.object(runner, 'SPLIT_SHA', self.split_sha))
            stack.enter_context(patch.object(runner, 'validate_protocol_freeze', return_value=self.protocol))
            stack.enter_context(patch.object(runner, 'validate_runtime_archive'))
            self.fixture.invoke(phase=phase, quality_only=quality_only)

    def test_final_launch_has_six_orders_same_v9_and_one_retained_claim(self):
        self.invoke()
        c = self.case
        self.assertEqual(len(c.commands), 18)
        identity = json.loads((c.out / 'identity.json').read_text())
        self.assertEqual(identity['entry_policy'], runner.ENTRY_POLICY)
        self.assertEqual(identity['query_freshness'], 'previously_unsearched_partition')
        self.assertEqual(identity['calibration_identity_sha256'], runner.sha(self.calibration_path))
        claim = identity['final_validation_claim']
        self.assertEqual(runner.sha(Path(claim['path'])), runner.sha(c.out / claim['retained_copy']))
        self.assertEqual(runner.sha(Path(claim['path'])), claim['sha256'])

    def test_second_final_launch_is_rejected_before_any_search(self):
        self.invoke()
        c = self.case
        c.commands.clear()
        c.out = c.root / 'results/second-final-attempt'
        with self.assertRaisesRegex(ValueError, 'already been claimed'):
            self.invoke()
        self.assertEqual(c.commands, [])
        self.assertFalse(c.out.exists())

    def test_wrong_v9_calibration_binary_is_rejected_before_claim_or_search(self):
        self.calibration['engines']['candidate']['binary_sha256'] = 'f' * 64
        self.calibration_path.write_text(json.dumps(self.calibration))
        with self.assertRaisesRegex(ValueError, 'assets differ'):
            self.invoke()
        self.assertEqual(self.case.commands, [])
        self.assertFalse((self.case.root / 'artifacts/recall-fix/v4-final-validation-claim.json').exists())

    def test_wrong_calibration_protocol_cannot_be_reused(self):
        self.calibration['entry_policy']['maximum_initial_seeds'] = 3
        self.calibration_path.write_text(json.dumps(self.calibration))
        with self.assertRaisesRegex(ValueError, 'same V9'):
            self.invoke()
        self.assertEqual(self.case.commands, [])

    def test_exposed_atomic_development_has_no_fresh_launch_claim(self):
        self.fixture.write_split('secondary-calibration')
        self.prepare_bindings()
        self.invoke(phase='secondary-development', quality_only=True)
        identity = json.loads((self.case.out / 'identity.json').read_text())
        self.assertEqual(identity['query_freshness'], 'previously_exposed_development')
        self.assertIsNone(identity['final_validation_claim'])
        self.assertEqual(len(self.case.commands), 3)

    def test_mutated_published_pool_is_rejected_before_search(self):
        (self.candidate / 'generation-00000000000000000001/index_disk.index').write_bytes(b'mutated graph')
        with self.assertRaisesRegex(ValueError, 'Published spatial-entry artifact changed'):
            self.invoke()
        self.assertEqual(self.case.commands, [])

    def test_invalid_policy_and_unreserved_validation_are_rejected(self):
        amendment = json.loads(self.amendment.read_text())
        amendment['secondary_validation_unsearched'] = False
        self.amendment.write_text(json.dumps(amendment))
        with self.assertRaisesRegex(ValueError, 'recorded dual-entry'):
            self.invoke()
        self.assertEqual(self.case.commands, [])

    def test_pinned_original_secondary_split_is_required(self):
        with patch.object(runner, 'FROZEN_PLAN', runner.sha(self.case.plan)):
            with self.assertRaisesRegex(ValueError, 'original sealed secondary split'):
                runner.validate_fresh(self.case.proof, self.case.prepared, 'secondary-validation', 2001)


if __name__ == '__main__':
    unittest.main()
