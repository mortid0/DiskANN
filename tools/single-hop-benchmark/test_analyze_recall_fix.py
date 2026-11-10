#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Check three-engine reporting and byte-exact serial-job provenance."""
import itertools
import argparse
import io
import tarfile
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import analyze_recall_fix as analysis


class ThreeEngineComparisons(unittest.TestCase):
    def observations(self, hits, qps):
        return [{'repeat': repeat, 'hits': np.asarray(hits), 'ids': np.arange(40).reshape(4, 10),
                 'latencies': np.ones(4),
                 'labels': {9: {'mean_us': 2, 'p99_us': 3}, 24: {'mean_us': 4, 'p99_us': 5}},
                 'qps': qps + repeat, 'mean_us': 3, 'p50_us': 3, 'p95_us': 5, 'p99_us': 5}
                for repeat in range(6)]

    def test_reports_target_regression_and_candidate_baseline_gain_without_hiding(self):
        observations = {'cpp': self.observations([9, 9, 9, 10], 100),
                        'baseline': self.observations([8, 9, 9, 10], 120),
                        'candidate': self.observations([9, 9, 8, 10], 125)}
        workload = {'labels': np.asarray([9, 9, 24, 24])}
        result = analysis.compare_engines(observations, workload, {9: 64, 24: 64},
                                         {9: np.arange(50), 24: np.arange(70)}, 10000, 1)
        self.assertEqual(result['label_comparison']['candidate_minus_cpp']['lower_labels'], [24])
        self.assertEqual(result['label_comparison']['candidate_minus_baseline']['higher_labels'], [9])
        self.assertEqual(result['label_comparison']['candidate_minus_baseline']['lower_labels'], [24])
        self.assertEqual(result['per_label'][0]['rust_l'], 64)
        self.assertEqual(result['per_label'][1]['candidate_minus_cpp_pp'], -5)
        self.assertEqual(result['contrasts']['candidate_minus_baseline']['positive_wins_queries'], 1)
        self.assertEqual(result['contrasts']['candidate_minus_baseline']['reference_wins_queries'], 1)
        self.assertGreater(result['contrasts']['candidate_minus_cpp']['timing_ratios']['qps']['median_paired_ratio'], 1)

    def test_calibration_quality_mode_excludes_all_performance_claims(self):
        observations = {name: self.observations([9, 9, 9, 10], 100)[:1] for name in analysis.ENGINES}
        result = analysis.compare_engines(observations, {'labels': np.asarray([9, 9, 24, 24])},
                                         {9: 64, 24: 64}, {9: np.arange(50), 24: np.arange(70)},
                                         10000, 1, quality_only=True)
        self.assertNotIn('qps', result['engines']['candidate'])
        self.assertNotIn('candidate_p99_ms', result['per_label'][0])
        self.assertEqual(result['contrasts']['candidate_minus_cpp']['timing_ratios'], {})

    def test_rejects_changed_ordered_result_ids_across_timing_repeats(self):
        observations = {name: self.observations([9, 9, 9, 10], 100) for name in analysis.ENGINES}
        observations['candidate'][1]['ids'][0, 0] = 100
        with self.assertRaisesRegex(ValueError, 'ordered result IDs differ'):
            analysis.compare_engines(observations, {'labels': np.asarray([9, 9, 24, 24])},
                                     {9: 64, 24: 64}, {9: np.arange(50), 24: np.arange(70)}, 10000, 1)


class TimingJobProof(unittest.TestCase):
    def fixture(self, root):
        jobs = []
        merged = {name: [] for name in analysis.ENGINES}
        for repeat, order in enumerate(itertools.permutations(analysis.ENGINES)):
            for position, name in enumerate(order):
                source = root / f'{repeat}-{name}.jsonl'
                raw = {'repeat': 0, 'qps': 100, 'queries': 2}
                if name != 'cpp':
                    raw['kind'] = 'query'
                rows = [raw] if name == 'cpp' else [{'kind': 'experiment', 'root': f'index-{name}'}, raw]
                source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                index = repeat * 3 + position
                start, end = f'2026-10-23T00:{index:02d}:00Z', f'2026-10-23T00:{index:02d}:59Z'
                jobs.append({'implementation': name, 'paired_repeat': repeat, 'local_repeat': 0,
                             'order_in_pair': position, 'exit_code': 0, 'aggregate_path': str(source),
                             'aggregate_sha256': analysis.sha(source), 'start': start, 'end': end})
                merged[name].append(dict(raw, repeat=repeat, paired_repeat=repeat, local_repeat=0,
                    original_aggregate=str(source), original_aggregate_sha256=analysis.sha(source)))
        logs = {}
        for name, rows in merged.items():
            path = root / f'{name}.normalized.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            logs[name] = path
        manifest = {'schema_version': 1, 'status': 'complete', 'runs': jobs, 'output_root': str(root)}
        path = root / 'triple-run-manifest.json'
        path.write_text(json.dumps(manifest))
        return path, logs, manifest

    def test_all_six_serial_orders_bind_byte_exact_original_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, logs, _ = self.fixture(root)
            result = analysis.triple_jobs_proof(manifest, logs, root)
            self.assertEqual(result['blocks'], 6)
            self.assertEqual(len({tuple(order) for order in result['orders']}), 6)

    def test_calibration_accepts_only_one_complete_serial_triple(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, logs, manifest = self.fixture(root)
            manifest['runs'] = manifest['runs'][:3]
            path.write_text(json.dumps(manifest))
            for name, log in logs.items():
                rows = [json.loads(line) for line in log.read_text().splitlines()]
                log.write_text(json.dumps(rows[0]) + '\n')
            result = analysis.triple_jobs_proof(path, logs, root, quality_only=True)
            self.assertEqual(result['blocks'], 1)
            with self.assertRaisesRegex(ValueError, 'Incomplete serial'):
                analysis.triple_jobs_proof(path, logs, root)

    def test_rejects_metric_rewrite_during_repeat_renumbering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, logs, _ = self.fixture(root)
            rows = [json.loads(line) for line in logs['candidate'].read_text().splitlines()]
            rows[0]['qps'] += 1
            logs['candidate'].write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'Merged timing row changed'):
                analysis.triple_jobs_proof(manifest, logs, root)

    def test_rejects_overlapping_timing_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, logs, manifest = self.fixture(root)
            manifest['runs'][1]['start'] = manifest['runs'][0]['start']
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Timing jobs overlap'):
                analysis.triple_jobs_proof(path, logs, root)


class CorrectedPQClone(unittest.TestCase):
    def fixture(self, root):
        host = Path('/remote')
        candidate = host / 'indexes/rust-native-10m-recall-preserve-codebook'
        source = host / 'indexes/rust-native-10m-recall-preserve'
        donor = host / 'indexes/rust-native-10m-frozen'
        generation = 'generation-00000000000000000001'
        source_files = {'index_disk.index': '4' * 64, 'vectors.bin': '5' * 64, 'labels.bin': '6' * 64,
                        'index_pq_pivots.bin': '7' * 64, 'index_pq_compressed.bin': '8' * 64, 'CATALOG': '9' * 64}
        donor_files = dict(source_files, **{'index_pq_pivots.bin': 'a' * 64, 'index_pq_compressed.bin': 'b' * 64, 'CATALOG': '2' * 64})
        incoming = {str(source / generation / name): digest for name, digest in source_files.items()}
        incoming.update({str(donor / generation / name): digest for name, digest in donor_files.items()})
        incoming.update({'indexes/rust-native-10m-frozen/CURRENT': '1' * 64,
                         'indexes/rust-native-10m-frozen/updates.wal': '3' * 64})
        bindings = {'rows': 10000000, 'dimensions': 192, 'postings_equal': True,
                    'stable_ids': 'identity 0..N in both generations', 'source_entries_preserved': True}
        provenance = {'schema_version': 1, 'source_root': 'indexes/rust-native-10m-recall-preserve',
                      'pq_donor_root': 'indexes/rust-native-10m-frozen', 'source_manifest_preserved_verbatim': True,
                      'replaced_files': ['index_pq_pivots.bin', 'index_pq_compressed.bin'],
                      'input_sha256': incoming, 'bindings': bindings,
                      'original_source_manifest': {'encoding': 'u8', 'base_sequence': 0, 'pq_reused': False}}
        provenance_path = root / 'clone-provenance.json'
        provenance_path.write_text(json.dumps(provenance))
        installed = dict(donor_files, **{'index_disk.index': source_files['index_disk.index'], 'CATALOG': 'd' * 64,
                                         'clone-provenance.json': analysis.sha(provenance_path)})
        outgoing = {str(candidate / generation / name): digest for name, digest in installed.items()}
        outgoing.update({'indexes/rust-native-10m-recall-preserve-codebook/CURRENT': 'c' * 64,
                         'indexes/rust-native-10m-recall-preserve-codebook/updates.wal': 'e' * 64})
        result = {'schema_version': 1, 'status': 'published and reopened successfully',
                  'source_root': 'indexes/rust-native-10m-recall-preserve',
                  'pq_donor_root': 'indexes/rust-native-10m-frozen',
                  'output_root': 'indexes/rust-native-10m-recall-preserve-codebook',
                  'generation_directory': str(candidate / generation), 'generation': 1, 'validated_sequence': 0,
                  'source_manifest_preserved_verbatim': True, 'bindings': bindings,
                  'input_sha256': incoming, 'output_sha256': outgoing}
        result_path = root / 'clone-result.json'
        result_path.write_text(json.dumps(result))
        build_path = root / 'original-build.jsonl'
        build_path.write_text(json.dumps({'kind': 'build', 'implementation': 'rust_native_lifecycle',
            'root': 'indexes/rust-native-10m-recall-preserve', 'rows': 10000000, 'dimensions': 192,
            'config': {'initial_pq_codebook': None}}) + '\n')
        engines = {'candidate': {'index_before': {'files_sha256': installed, 'catalog_sha256': 'd' * 64,
                                                  'current_sha256': 'c' * 64, 'journal_sha256': 'e' * 64}},
                   'baseline': {'index_before': {'files_sha256': donor_files, 'catalog_sha256': '2' * 64,
                                                 'current_sha256': '1' * 64, 'journal_sha256': '3' * 64}}}
        args = argparse.Namespace(clone_result=result_path, clone_provenance=provenance_path,
                                  original_graph_build_log=build_path, candidate_build_log=result_path)
        return args, engines, candidate, result

    def test_mixed_relative_root_envelopes_and_absolute_generation_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, engines, candidate, _ = self.fixture(Path(temporary))
            result = analysis.clone_proof(args, engines, candidate)
            self.assertTrue(result['installed_pq_files_equal_frozen_baseline'])
            self.assertTrue(result['candidate_graph_vectors_labels_preserved'])
            self.assertIsNone(result['original_build_initial_pq_codebook'])

    def test_rejects_installed_pq_pair_changed_from_frozen_donor(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, engines, candidate, result = self.fixture(Path(temporary))
            engines['candidate']['index_before']['files_sha256']['index_pq_compressed.bin'] = 'f' * 64
            result['output_sha256'][str(candidate / 'generation-00000000000000000001/index_pq_compressed.bin')] = 'f' * 64
            args.clone_result.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'PQ file differs from frozen baseline'):
                analysis.clone_proof(args, engines, candidate)

    def test_rejects_graph_change_during_pq_republication(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, engines, candidate, result = self.fixture(Path(temporary))
            engines['candidate']['index_before']['files_sha256']['index_disk.index'] = 'f' * 64
            result['output_sha256'][str(candidate / 'generation-00000000000000000001/index_disk.index')] = 'f' * 64
            args.clone_result.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'changed source geometry'):
                analysis.clone_proof(args, engines, candidate)


class SupplementalNamespaces(unittest.TestCase):
    def fixture(self, root):
        def matrix(path, values, dtype):
            values = np.asarray(values, dtype=dtype)
            path.write_bytes(np.asarray(values.shape, dtype='<u4').tobytes() + values.tobytes())
        matrix(root / 'query.u8bin', np.zeros((2, 192)), 'u1')
        matrix(root / 'query_labels.u32bin', [[9], [24]], '<u4')
        matrix(root / 'query_ids.u32bin', [[14], [14]], '<u4')
        matrix(root / 'source_namespace.u8bin', [[0], [1]], 'u1')
        truth = np.arange(20, dtype='<u4').reshape(2, 10)
        matrix(root / 'groundtruth.bin', truth, '<u4')
        with (root / 'groundtruth.bin').open('ab') as stream:
            stream.write(truth.astype('<f4').tobytes())
        (root / 'labels.txt').write_text('9,24\n')
        (root / 'query_labels.txt').write_text('9\n24\n')
        (root / 'source-query-rows.jsonl').write_text('{}\n{}\n')
        names = ('query.u8bin', 'query_labels.u32bin', 'query_ids.u32bin', 'groundtruth.bin',
                 'labels.txt', 'query_labels.txt', 'source_namespace.u8bin', 'source-query-rows.jsonl')
        artifacts = {name: {'sha256': analysis.sha(root / name), 'bytes': (root / name).stat().st_size} for name in names}
        manifest = {'base_vectors': 10000000, 'dimension': 192, 'k': 10, 'dtype': 'uint8', 'metric': 'L2',
                    'query_count': 2, 'artifacts': artifacts, 'source_query_rows': artifacts['source-query-rows.jsonl']}
        (root / 'manifest.json').write_text(json.dumps(manifest))
        return manifest

    def test_same_numeric_id_in_different_namespaces_is_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            result = analysis.supplemental_prepared(root)
            self.assertEqual(result['ids'].tolist(), [14, 14])
            self.assertEqual(result['namespace'].tolist(), [0, 1])

    def test_duplicate_namespace_id_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.fixture(root)
            path = root / 'source_namespace.u8bin'
            raw = bytearray(path.read_bytes())
            raw[-1] = 0
            path.write_bytes(raw)
            manifest['artifacts'][path.name]['sha256'] = analysis.sha(path)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'namespace/ID pairs repeat'):
                analysis.supplemental_prepared(root)


class FrozenIdentity(unittest.TestCase):
    def fixture(self, root):
        prepared, calibration = root / 'validation', root / 'calibration'
        prepared.mkdir()
        calibration.mkdir()
        for directory in (prepared, calibration):
            (directory / 'manifest.json').write_text('{}')
        commitment, build = root / 'commitment.json', root / 'build.jsonl'
        commitment.write_text('{}')
        build.write_text('{}')
        archives = {}
        for name in ('baseline', 'candidate'):
            archive = root / f'{name}.tar.gz'
            with tarfile.open(archive, 'w:gz') as stream:
                files = [('diskann-disk/src/build/filtered.rs', name.encode())]
                files.extend((filename, b'unchanged') for filename in (
                    'diskann/src/graph/search/single_hop_filter_search.rs', 'diskann/src/neighbor/queue.rs',
                    'diskann-disk/examples/filtered_lifecycle.rs',
                    *(f'diskann-disk/src/filtered/{part}.rs' for part in
                      ('native', 'index', 'view', 'updates', 'persistence', 'metadata'))))
                for filename, contents in files:
                    info = tarfile.TarInfo(filename)
                    info.size = len(contents)
                    stream.addfile(info, io.BytesIO(contents))
            archives[name] = archive
        candidate_root = '/remote/indexes/rust-native-10m-recall-preserve'
        roots = {'candidate': candidate_root, 'baseline': '/remote/indexes/rust-native-10m-frozen'}
        logs = {}
        for name in analysis.ENGINES:
            path = root / f'{name}.jsonl'
            if name == 'cpp':
                rows = [{'prepared': '/remote/validation',
                         'index_prefix': '/remote/indexes/yfcc10m_stitched_R32_R128_L100'}]
            else:
                rows = [{'kind': 'experiment', 'root': roots[name], 'prepared': '/remote/validation',
                         'action': 'query', 'n_base': 10000000, 'dimensions': 192, 'index_threads': 8,
                         'beam': 4, 'shared_cache_nodes': 0, 'cache_bypass': True, 'storage': 'native_io_uring',
                         'warmup': 'full_same_process_same_worker_count', 'query_scheduling': 'dynamic_one'}]
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            logs[name] = path
        indexes = {name: {'current_sha256': '1' * 64, 'catalog_sha256': '2' * 64,
                          'journal_sha256': '3' * 64, 'files_sha256': {'index_disk.index': '4' * 64}}
                   for name in ('baseline', 'candidate')}
        indexes['cpp'] = {'_disk.index': '5' * 64}
        engines = {name: {'binary_sha256': '6' * 64, 'binary_before_sha256': '6' * 64,
                         'log_sha256': analysis.sha(logs[name]), 'index_before': indexes[name],
                         'index_after': indexes[name]} for name in analysis.ENGINES}
        for name in ('baseline', 'candidate'):
            engines[name]['source_archive'] = {'sha256': analysis.sha(archives[name]),
                                               'before_sha256': analysis.sha(archives[name])}
        freeze = {'schema_version': 1, 'phase': 'validation', 'timestamp_utc': '2026-10-23T00:00:00Z',
                  'plan_sha256': analysis.FROZEN_SHA, 'fresh_proof_sha256': analysis.sha(commitment),
                  'prepared_manifest_sha256': analysis.sha(prepared / 'manifest.json'),
                  'candidate_build_log_sha256': analysis.sha(build),
                  'binary_sha256': {name: engines[name]['binary_sha256'] for name in analysis.ENGINES},
                  'source_archive_sha256': {name: analysis.sha(archives[name]) for name in archives},
                  'index_identity': indexes, 'candidate_root': candidate_root}
        freeze_path = root / 'freeze.json'
        freeze_path.write_text(json.dumps(freeze))
        identity = {'schema_version': 1, 'cpu_affinity': list(range(8)), 'host': 'host',
                    'output_root': '/remote/output', 'plan_before_sha256': analysis.FROZEN_SHA,
                    'plan_after_sha256': analysis.FROZEN_SHA, 'engines': engines,
                    'prepared_manifest_sha256': analysis.sha(prepared / 'manifest.json'),
                    'candidate_freeze_sha256': analysis.sha(freeze_path),
                    'fresh_proof_sha256': analysis.sha(commitment),
                    'runner_sha256': analysis.sha(Path(analysis.__file__).with_name('run_recall_fix.py')),
                    'cpp_runner_sha256': analysis.sha(Path(analysis.__file__).parent / 'cpp-port/run_reference.py')}
        identity_path, prior_path = root / 'identity.json', root / 'prior.json'
        identity_path.write_text(json.dumps(identity))
        prior = {'host': 'host', 'native': dict(engines['baseline']), 'cpp': engines['cpp']}
        prior['native']['index_before'] = {key: value for key, value in indexes['baseline'].items() if key != 'files_sha256'}
        prior_path.write_text(json.dumps(prior))
        manifest = {'prepared_manifest_sha256': identity['prepared_manifest_sha256'], 'prepared': '/remote/validation',
                    'output_root': '/remote/output', 'identity': identity, 'phase': 'validation',
                    'timing_authoritative': True, 'candidate_freeze_sha256': analysis.sha(freeze_path),
                    'fresh_proof_sha256': analysis.sha(commitment),
                    'runs': [{'implementation': name, 'start': '2026-10-23T00:00:01Z',
                              'command': ['search', '--index-prefix', '/remote/indexes/yfcc10m_stitched_R32_R128_L100']
                              if name == 'cpp' else ['search', '--root', roots[name]]} for name in analysis.ENGINES]}
        args = argparse.Namespace(identity=identity_path, prior_identity=prior_path, prepared=prepared,
                                  calibration_prepared=calibration, quality_only=False,
                                  baseline_source_archive=archives['baseline'], candidate_source_archive=archives['candidate'],
                                  candidate_freeze=freeze_path, commitment=commitment, candidate_build_log=build)
        return args, logs, manifest, identity

    def test_full_generation_hashes_extend_prior_identity_without_changing_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, logs, manifest, _ = self.fixture(Path(temporary))
            result = analysis.identity_proof(args, logs, manifest)
            self.assertEqual(result['changed_archived_source_files'], ['diskann-disk/src/build/filtered.rs'])

    def test_rejects_unbound_generation_file_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, logs, manifest, identity = self.fixture(Path(temporary))
            identity['engines']['candidate']['index_after'] = dict(identity['engines']['candidate']['index_after'])
            identity['engines']['candidate']['index_after']['files_sha256'] = {'index_disk.index': 'f' * 64}
            args.identity.write_text(json.dumps(identity))
            with self.assertRaisesRegex(ValueError, 'index identity changed'):
                analysis.identity_proof(args, logs, manifest)


if __name__ == '__main__':
    unittest.main()
