#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Verify serial comparison ordering and immutable fresh-validation bindings."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import itertools
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_recall_fix as runner


class RecallFixRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.prepared = self.root / 'recall-fix/data/natural-private/prepared-private-natural-validation'
        self.out = self.root / 'results/recall-fix'
        self.plan = self.root / 'plan.json'
        self.proof = self.root / 'fresh-proof.json'
        self.archive = self.root / 'candidate.tar.gz'
        self.build_log = self.root / 'candidate-build.json'
        self.catalogs = {engine: self.root / f'indexes/{name}/generation-00000000000000000001/CATALOG'
                         for engine, name in [('baseline', 'rust-native-10m-frozen'),
                                              ('candidate', 'rust-native-10m-recall-preserve')]}
        self.binaries = {
            'cpp': self.root / 'cpp/build-native/apps/search_disk_index',
            'baseline': self.root / 'artifacts/heldout/filtered_lifecycle.v5-heldout',
            'candidate': self.root / 'artifacts/recall-fix/filtered_lifecycle.v6-preserve-union'}
        assets = [self.plan, self.archive, self.build_log, *self.catalogs.values(), *self.binaries.values(),
                  self.root / 'artifacts/heldout/lifecycle-v5-heldout-complete-sources.tar.gz',
                  self.root / 'rust/tools/single-hop-benchmark/cpp-port/run_reference.py']
        for name in ('rust-native-10m-frozen', 'rust-native-10m-recall-preserve'):
            assets += [self.root / f'indexes/{name}/CURRENT', self.root / f'indexes/{name}/updates.wal']
        assets += [self.root / ('indexes/yfcc10m_stitched_R32_R128_L100' + suffix)
                   for suffix in runner.CPP_SUFFIXES]
        for asset in assets:
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_bytes(asset.name.encode())
        self.plan.write_text('{"1": 64}\n')
        self.prepared.mkdir(parents=True)
        (self.prepared / 'query.u8bin').write_bytes(struct.pack('<II', 2001, 192) + bytes(2001 * 192))
        (self.prepared / 'query_ids.u32bin').write_bytes(struct.pack('<II', 2001, 1) + bytes(2001 * 4))
        (self.prepared / 'query_labels.u32bin').write_bytes(struct.pack('<II', 2001, 1) + bytes(2001 * 4))
        (self.prepared / 'groundtruth.bin').write_bytes(struct.pack('<II', 2001, 10) + bytes(2001 * 80))
        self.prepared_manifest = {
            'query_count': 2001, 'split': 'validation',
            'source_query_namespace': 'official_private_2727415019_100K',
            'validation_scope': 'recall_fix_fresh_natural_private_validation',
            'artifacts': {name: {'bytes': (self.prepared / name).stat().st_size,
                                 'sha256': runner.sha(self.prepared / name)} for name in
                          ('query.u8bin', 'query_ids.u32bin', 'query_labels.u32bin', 'groundtruth.bin')}}
        self.write_manifest_and_proof()
        self.commands = []
        self.mutate = None

    def write_manifest_and_proof(self):
        manifest_path = self.prepared / 'manifest.json'
        manifest_path.write_text(json.dumps(self.prepared_manifest))
        self.proof.write_text(json.dumps({
            'schema_version': 1, 'scope': 'fresh_private_natural_full64_labels',
            'policy_sha256': runner.sha(self.plan),
            'partitions': {self.prepared_manifest['split']: {
                'queries': 2001, 'manifest_sha256': runner.sha(manifest_path),
                'artifacts': self.prepared_manifest['artifacts']}},
            'all_source_rows_and_embeddings_disjoint_between_partitions': True,
            'validation_quality_not_examined_before_candidate_frozen': True}))

    def arguments(self, phase='validation', quality_only=False, candidate_root=None):
        arguments = ['run_recall_fix.py', '--root', str(self.root), '--prepared', str(self.prepared),
                     '--out', str(self.out), '--plan', str(self.plan), '--fresh-proof', str(self.proof),
                     '--candidate-source-archive', str(self.archive), '--candidate-build-log', str(self.build_log),
                     '--phase', phase]
        if quality_only:
            arguments.append('--quality-only')
        if candidate_root is not None:
            arguments.extend(('--candidate-root', str(candidate_root)))
        return arguments

    def fake_run(self, command, **kwargs):
        # Freeze and dataset proof must be materialized before the first engine executes.
        freeze_path = self.out / 'candidate-freeze.json'
        self.assertTrue(freeze_path.exists())
        freeze = json.loads(freeze_path.read_text())
        self.assertEqual(freeze['fresh_proof_sha256'], runner.sha(self.proof))
        self.assertEqual(freeze['prepared_manifest_sha256'], runner.sha(self.prepared / 'manifest.json'))
        self.commands.append(command)

        def value(flag):
            return command[command.index(flag) + 1]

        cpp = command[0] == sys.executable
        if cpp:
            engine = 'cpp'
            folder = Path(value('--output-dir'))
            folder.mkdir(parents=True)
            aggregate = folder / (value('--name') + '.normalized.jsonl')
            row = {'queries': 2001, 'repeat': 0, 'recall': .95, 'qps': 1000, 'p99_us': 5000,
                   'raw_stats': str(folder / 'raw-stats.jsonl'), 'result_ids': str(folder / 'ids.bin')}
            rows = [row]
        else:
            engine = next(name for name, path in self.binaries.items() if str(path) == command[0])
            aggregate = Path(value('--out'))
            row = {'kind': 'query', 'queries': 2001, 'repeat': 0, 'recall': .96, 'qps': 1100,
                   'p99_us': 4500, 'raw_queries': {'path': str(Path(value('--query-details-dir')) / 'queries.jsonl')}}
            rows = [{'kind': 'experiment', 'engine': engine}, row]
        aggregate.write_text(''.join(json.dumps(item) + '\n' for item in rows))
        if self.mutate is not None and len(self.commands) == 1:
            self.mutate()
        return subprocess.CompletedProcess(command, 0)

    def invoke(self, *, affinity=None, frozen=True, **arguments):
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', self.arguments(**arguments)))
            stack.enter_context(patch.object(runner.os, 'sched_getaffinity', create=True,
                                            return_value=set(range(8)) if affinity is None else affinity))
            stack.enter_context(patch.object(runner.subprocess, 'run', side_effect=self.fake_run))
            if frozen:
                stack.enter_context(patch.object(runner, 'FROZEN_PLAN', runner.sha(self.plan)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            runner.main()

    def test_all_six_permutations_serialized_and_raw_observations_bound(self):
        self.invoke()
        engines = ['cpp' if command[0] == sys.executable else
                   next(name for name, path in self.binaries.items() if str(path) == command[0])
                   for command in self.commands]
        self.assertEqual(engines, [engine for order in itertools.permutations(runner.ENGINES) for engine in order])
        manifest = json.loads((self.out / 'triple-run-manifest.json').read_text())
        self.assertEqual(manifest['status'], 'complete')
        self.assertTrue(manifest['timing_authoritative'])
        self.assertEqual(len(manifest['runs']), 18)
        for command in self.commands:
            if command[0] != sys.executable:
                self.assertEqual(command[command.index('--label-l') + 1], str(self.plan.resolve()))
                self.assertEqual(command[command.index('--l') + 1], '64')
                self.assertEqual(command[command.index('--limit') + 1], '2001')
                self.assertEqual(command[command.index('--timing-authoritative') + 1], 'true')
            else:
                self.assertIn('--timing-authoritative', command)
        for engine in runner.ENGINES:
            rows = [json.loads(line) for line in (self.out / (engine + '.normalized.jsonl')).read_text().splitlines()]
            rows = [row for row in rows if row.get('kind') != 'experiment']
            self.assertEqual([row['repeat'] for row in rows], list(range(6)))
            for row in rows:
                self.assertEqual(row['local_repeat'], 0)
                self.assertEqual(runner.sha(row['original_aggregate']), row['original_aggregate_sha256'])
                raw_rows = [json.loads(line) for line in Path(row['original_aggregate']).read_text().splitlines()]
                self.assertEqual([raw['repeat'] for raw in raw_rows if raw.get('kind') != 'experiment'], [0])
        identity = json.loads((self.out / 'identity.json').read_text())
        for engine in runner.ENGINES:
            self.assertEqual(identity['engines'][engine]['index_before'], identity['engines'][engine]['index_after'])

    def test_calibration_is_one_quality_only_triple(self):
        self.prepared_manifest.update(split='calibration', validation_scope='recall_fix_fresh_natural_private_calibration')
        self.write_manifest_and_proof()
        self.invoke(phase='calibration', quality_only=True)
        self.assertEqual(len(self.commands), 3)
        for command in self.commands:
            if command[0] == sys.executable:
                self.assertNotIn('--timing-authoritative', command)
            else:
                self.assertEqual(command[command.index('--timing-authoritative') + 1], 'false')
        manifest = json.loads((self.out / 'triple-run-manifest.json').read_text())
        self.assertFalse(manifest['timing_authoritative'])

    def test_rejects_phase_timing_mismatch(self):
        for arguments in ({'phase': 'calibration'}, {'phase': 'validation', 'quality_only': True}):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self.invoke(**arguments)
        self.assertEqual(self.commands, [])
        self.assertFalse(self.out.exists())

    def test_rejects_wrong_core_affinity_and_policy_before_launch(self):
        for arguments in ({'affinity': {0, 1}}, {'frozen': False}):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self.invoke(**arguments)
        self.assertEqual(self.commands, [])
        self.assertFalse(self.out.exists())

    def test_rejects_calibration_data_claimed_as_validation(self):
        self.prepared_manifest.update(split='calibration')
        self.write_manifest_and_proof()
        with self.assertRaises((SystemExit, ValueError)):
            self.invoke()
        self.assertEqual(self.commands, [])

    def test_rejects_wrong_query_namespace(self):
        self.prepared_manifest['source_query_namespace'] = 'public_100K'
        self.write_manifest_and_proof()
        with self.assertRaises((SystemExit, ValueError)):
            self.invoke()
        self.assertEqual(self.commands, [])

    def test_rejects_data_artifact_changed_after_split_commitment(self):
        (self.prepared / 'query.u8bin').write_bytes(struct.pack('<II', 2001, 192) + bytes([1]) * (2001 * 192))
        with self.assertRaises((SystemExit, ValueError)):
            self.invoke()
        self.assertEqual(self.commands, [])

    def test_index_mutation_records_failed_identity_check(self):
        self.mutate = lambda: self.catalogs['candidate'].write_text('mutated catalog')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.invoke()
        manifest = json.loads((self.out / 'triple-run-manifest.json').read_text())
        self.assertEqual(manifest['status'], 'failed_identity_check')
        self.assertFalse((self.out / 'identity.json').exists())

    def test_query_artifact_mutation_during_runs_is_rejected(self):
        query = self.prepared / 'query.u8bin'
        self.mutate = lambda: query.write_bytes(struct.pack('<II', 2001, 192) + bytes([2]) * (2001 * 192))
        with self.assertRaisesRegex(ValueError, 'artifact|changed|commitment'):
            self.invoke()
        manifest = json.loads((self.out / 'triple-run-manifest.json').read_text())
        self.assertEqual(manifest['status'], 'failed_identity_check')
        self.assertFalse((self.out / 'identity.json').exists())

    def test_native_data_file_mutation_is_rejected_without_catalog_change(self):
        data = self.catalogs['baseline'].parent / 'index_disk.index'
        data.write_bytes(b'original graph')
        self.mutate = lambda: data.write_bytes(b'changed graph')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.invoke()
        manifest = json.loads((self.out / 'triple-run-manifest.json').read_text())
        self.assertEqual(manifest['status'], 'failed_identity_check')

    def matching_explicit_candidate(self):
        root = self.root / 'indexes/rust-native-10m-recall-preserve-codebook'
        generation = root / 'generation-00000000000000000001'
        generation.mkdir(parents=True, exist_ok=True)
        (root / 'CURRENT').write_bytes(b'corrected current')
        (root / 'updates.wal').write_bytes(b'empty journal')
        (generation / 'CATALOG').write_bytes(b'corrected catalog')
        for name in ('index_pq_pivots.bin', 'index_pq_compressed.bin', 'vectors.bin'):
            content = ('frozen original ' + name).encode()
            (self.catalogs['baseline'].parent / name).write_bytes(content)
            (generation / name).write_bytes(content)
        (self.catalogs['baseline'].parent / 'index_disk.index').write_bytes(b'old graph')
        (generation / 'index_disk.index').write_bytes(b'new graph preserving union')
        return root

    def test_explicit_candidate_rejects_changed_pq_before_any_launch(self):
        candidate = self.matching_explicit_candidate()
        generation = candidate / 'generation-00000000000000000001'
        for name in ('index_pq_pivots.bin', 'index_pq_compressed.bin'):
            path = generation / name
            original = path.read_bytes()
            path.write_bytes(b'retrained or changed quantization')
            with self.subTest(artifact=name), self.assertRaises(SystemExit):
                self.invoke(candidate_root=candidate)
            self.assertEqual(self.commands, [])
            self.assertFalse(self.out.exists())
            path.write_bytes(original)

    def test_explicit_candidate_rejects_changed_vectors_before_any_launch(self):
        candidate = self.matching_explicit_candidate()
        (candidate / 'generation-00000000000000000001/vectors.bin').write_bytes(b'changed vectors or ordinal order')
        with self.assertRaises(SystemExit):
            self.invoke(candidate_root=candidate)
        self.assertEqual(self.commands, [])
        self.assertFalse(self.out.exists())

    def test_explicit_candidate_accepts_frozen_pq_and_vectors_with_different_graph(self):
        candidate = self.matching_explicit_candidate()
        self.invoke(candidate_root=candidate)
        self.assertEqual(len(self.commands), 18)
        for command in self.commands:
            if command[0] == str(self.binaries['candidate']):
                self.assertEqual(command[command.index('--root') + 1], str(candidate))
            elif command[0] == str(self.binaries['baseline']):
                self.assertEqual(command[command.index('--root') + 1], str(self.catalogs['baseline'].parent.parent))
        freeze = json.loads((self.out / 'candidate-freeze.json').read_text())
        self.assertEqual(freeze['candidate_root'], str(candidate))
        identities = freeze['index_identity']
        for name in ('index_pq_pivots.bin', 'index_pq_compressed.bin', 'vectors.bin'):
            self.assertEqual(identities['candidate']['files_sha256'][name], identities['baseline']['files_sha256'][name])
        self.assertNotEqual(identities['candidate']['files_sha256']['index_disk.index'], identities['baseline']['files_sha256']['index_disk.index'])

    def test_failed_subprocess_stops_following_engines(self):
        def failure(command, **kwargs):
            self.commands.append(command)
            return subprocess.CompletedProcess(command, 7)
        with patch.object(self, 'fake_run', side_effect=failure), self.assertRaisesRegex(RuntimeError, 'failed'):
            self.invoke()
        self.assertEqual(len(self.commands), 1)
        manifest = json.loads((self.out / 'triple-run-manifest.json').read_text())
        self.assertEqual(manifest['status'], 'failed')
        self.assertEqual(manifest['failed_run']['exit_code'], 7)


if __name__ == '__main__':
    unittest.main()
