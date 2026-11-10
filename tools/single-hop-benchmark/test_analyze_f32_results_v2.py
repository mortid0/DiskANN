#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Family bootstrap, strict-ID tie diagnostics, raw validity, and repetition guards."""

import importlib.util
import itertools
import struct
import types
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("analyze_f32", Path(__file__).with_name("analyze_f32_results_v2.py"))
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)


class BootstrapTests(unittest.TestCase):
    def test_c4_family_transitive_connections_and_books_presplit_group(self):
        c4 = [{"positive_item_id": "a", "user_hash": "u1", "text_hash": "t1"},
              {"positive_item_id": "a", "user_hash": "u2", "text_hash": "t2"},
              {"positive_item_id": "b", "user_hash": "u2", "text_hash": "t3"},
              {"positive_item_id": "c", "user_hash": "u3", "text_hash": "t4"}]
        self.assertEqual(analyze.query_families(c4).tolist(), [0, 0, 0, 3])
        self.assertEqual(analyze.query_families([{"group": 7}, {"group": 2}, {"group": 7}]).tolist(), [0, 1, 0])

    def test_recorded_pre_dedup_component_preserves_removed_bridge(self):
        metadata = [{"id": 0, "positive_item_id": "a", "user_hash": "u1", "text_hash": "t1"},
                    {"id": 2, "positive_item_id": "b", "user_hash": "u2", "text_hash": "t2"}]
        assignments = [{"source_query_rows": [0, 1, 2], "retained_query_rows": [0, 2]}]
        self.assertEqual(analyze.query_families(metadata).tolist(), [0, 1])
        self.assertEqual(analyze.query_families(metadata, assignments).tolist(), [0, 0])

    def test_missing_family_metadata_fails(self):
        with self.assertRaisesRegex(ValueError, "family"):
            analyze.query_families([{"qid": 1}])

    def test_bootstrap_resamples_entire_families_and_is_reproducible(self):
        hits = np.array([[0], [0], [10], [10]])
        family = np.array([0, 0, 1, 1])
        result = analyze.bootstrap_families(hits, family, 10, 1000)
        self.assertEqual(result, analyze.bootstrap_families(hits, family, 10, 1000))
        self.assertEqual(result["families"], 2)
        self.assertEqual(result["recall"], [.5])
        self.assertEqual(result["recall_ci95"], [[0, 1]])

    def test_paired_bootstrap_preserves_constant_method_difference(self):
        hits = np.array([[2, 3], [4, 5], [6, 7], [8, 9]])
        result = analyze.bootstrap_families(hits, np.arange(4), 10, 1000)
        np.testing.assert_allclose(result["paired_difference_ci95"]["1-0"], [.1, .1], atol=1e-15)

    def test_single_family_cannot_estimate_population_uncertainty(self):
        result = analyze.bootstrap_families([[10], [9]], [1, 1], 10, 20)
        self.assertEqual(result["recall_ci95"], [None])
        self.assertIsNone(result["paired_difference_ci95"])


class DistanceTests(unittest.TestCase):
    def test_padding_never_looks_like_last_base_vector(self):
        base = np.array([[1., 2.], [3., 4.]], dtype="<f4")
        q = np.array([[1., 2.]], dtype="<f4")
        distances = analyze.direct_distances(base, q, np.array([[0, -1]]))
        self.assertEqual(distances[0, 0], 0)
        self.assertTrue(np.isnan(distances[0, 1]))

    def test_strict_ids_stay_primary_when_cutoff_neighbors_tie(self):
        ids = np.array([[0, 2], [3, -1]])
        truth = np.array([[0, 1], [3, 4]])
        direct = np.array([[.5, 1.], [.25, np.nan]])
        strict, tolerant, _ = analyze.recall_counts(ids, [2, 1], truth, direct, np.array([1., .25]), 384)
        self.assertEqual(strict.tolist(), [1, 1])
        self.assertEqual(tolerant.tolist(), [2, 1])

    def test_nontruth_strictly_closer_than_cutoff_rejects_bad_truth(self):
        with self.assertRaisesRegex(ValueError, "strictly closer"):
            analyze.recall_counts(np.array([[0, 2]]), [2], np.array([[0, 1]]), np.array([[.5, .75]]), np.array([1.]), 384)

    def test_duplicate_returned_id_is_not_a_tie_benefit(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analyze.recall_counts(np.array([[0, 0]]), [2], np.array([[0, 1]]), np.array([[0., 0.]]), np.array([1.]), 384)


class RepetitionTests(unittest.TestCase):
    def test_six_timings_do_not_create_six_quality_samples(self):
        rows, obs = [], {}
        for engine in analyze.ENGINES:
            for repeat in range(6):
                row = {"engine": engine, "paired_repeat": repeat, "l": 64, "qps": 1000,
                       "p99_us": 100, "short_results": 0,
                       "labels": [{"label": 1, "p99_us": 100}]}
                rows.append(row)
                obs[id(row)] = {"strict": np.array([9, 8]), "cutoff": np.array([9, 8]), "returned": np.array([10, 10])}
        summary = analyze.scope_summary("fixed_l", rows, obs, np.array([1, 1]), np.array([0, 1]), {1: np.arange(100)}, {1: "Books"}, 10, 100)
        self.assertEqual(summary["quality_queries"], 2)
        self.assertEqual(summary["families"], 2)
        self.assertEqual(summary["engines"]["cpp"]["repeats"], 6)
        self.assertEqual(summary["labels"][0]["queries"], 2)
        self.assertIn("fewer_than10_expected_observations_in_the_top1_percent_tail", summary["labels"][0]["limitations"])
        obs[id(rows[-1])]["strict"] = np.array([10, 8])
        with self.assertRaisesRegex(ValueError, "Repeated quality"):
            analyze.scope_summary("fixed_l", rows, obs, np.array([1, 1]), np.array([0, 1]), {1: np.arange(100)}, {}, 10, 100)

    def test_incomplete_timing_schedule_rejected(self):
        rows = [{"engine": "cpp", "paired_repeat": i} for i in range(5)]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            analyze.scope_summary("fixed_l", rows, {}, np.array([1]), np.array([0]), {1: np.arange(10)}, {}, 10, 10)


class AnalysisIntegrationTests(unittest.TestCase):
    def fixture(self, root, engine_order=analyze.ENGINES, timing_orders=None):
        def write_json(path, value):
            path.write_text(json.dumps(value))
        def write_matrix(path, values, dtype):
            values = np.asarray(values, dtype=dtype)
            if values.ndim == 1:
                values = values[:, None]
            path.write_bytes(struct.pack("<II", *values.shape) + values.tobytes())
        prepared = root / "prepared"
        prepared.mkdir()
        base = np.zeros((21, 32), dtype="<f4")
        base[:, 0] = np.r_[np.arange(20), 9]
        query = base[[0, 10]]
        write_matrix(prepared / "base.fbin", base, "<f4")
        write_matrix(prepared / "query.fbin", query, "<f4")
        write_matrix(prepared / "query_labels.u32bin", [1, 1], "<u4")
        write_matrix(prepared / "query_ids.u32bin", [11, 12], "<u4")
        write_matrix(prepared / "label_1.u32bin", np.arange(21), "<u4")
        (prepared / "labels.txt").write_text("1\n" * 21)
        (prepared / "query_labels.txt").write_text("1\n1\n")
        metadata = [{"qid": i+11, "label": 1, "positive_item_id": str(i), "user_hash": str(i), "text_hash": str(i)} for i in range(2)]
        (prepared / "query_metadata.jsonl").write_text("".join(json.dumps(q) + "\n" for q in metadata))
        truth_ids = []
        for q in query:
            d = np.square(base.astype("<f8")-q).sum(axis=1)
            truth_ids.append(np.lexsort((np.arange(21), d))[:10])
        truth_ids = np.array(truth_ids, dtype="<i4")
        truth_distances = analyze.direct_distances(base, query, truth_ids)
        (prepared / "groundtruth.bin").write_bytes(struct.pack("<II", 2, 10) + truth_ids.tobytes() + truth_distances.astype("<f4").tobytes())
        names = ["base.fbin", "query.fbin", "query_labels.u32bin", "query_ids.u32bin", "label_1.u32bin", "labels.txt", "query_labels.txt", "query_metadata.jsonl", "groundtruth.bin"]
        manifest = {"labels": [{"label": 1, "original_label": 1, "category": "Books"}], "artifacts": {name: analyze.record(prepared / name) for name in names}}
        write_json(prepared / "manifest.json", manifest)
        orders = timing_orders or [list(order) for order in itertools.permutations(engine_order)]
        protocol_path = root / "protocol.json"
        write_json(protocol_path, {"dataset": "synthetic_tie_fixture", "holdout_manifest": analyze.record(prepared / "manifest.json"), "fixed_l": 64, "engines": list(engine_order),
                                  "evaluation_orders": orders, "timing_authoritative": False})
        calibration_path = root / "calibration.json"
        write_json(calibration_path, {"protocol": analyze.record(protocol_path), "target90": {engine: {"status": "calibrated", "l": 64} for engine in engine_order}})
        claim = root / "claim.json"
        write_json(claim, {"fixture": True})
        rows = []
        for scope in analyze.SCOPES:
            for engine in engine_order:
                results = truth_ids.copy()
                if engine == "rust_single":
                    self.assertEqual(results[0, -1], 9)
                    results[0, -1] = 20
                ds = analyze.direct_distances(base, query, results).astype("<f4")
                raw_path = root / f"{scope}.{engine}.raw.jsonl"
                micros = 80 if engine == "rust_single" else 100
                if engine == "cpp":
                    raw = [{"kind": "query", "query_index": i, "returned_results": 10, "latency_us": micros} for i in range(2)]
                    ids_path, ds_path = root / f"{scope}.{engine}.ids", root / f"{scope}.{engine}.distances"
                    write_matrix(ids_path, results, "<u4")
                    write_matrix(ds_path, ds, "<f4")
                else:
                    raw = [{"query": i, "label": 1, "l": 64, "returned": 10, "ids": results[i].tolist(), "distances": ds[i].tolist(), "micros": micros} for i in range(2)]
                raw_path.write_text("".join(json.dumps(x) + "\n" for x in raw))
                recall = .95 if engine == "rust_single" else 1.
                qps = 1200 if engine == "rust_single" else 1000
                aggregate = {"queries": 2, "l": 64, "qps": qps, "wall_seconds": 2/qps, "recall": recall, "p99_us": micros,
                             "short_results": 0, "filter_violations": 0, "duplicate_results": 0,
                             "labels": [{"label": 1, "n": 2, "eligible": 21, "recall": recall, "p99_us": micros}]}
                aggregate_path = root / f"{scope}.{engine}.aggregate.jsonl"
                aggregate_path.write_text(json.dumps(aggregate) + "\n")
                command = root / f"{scope}.{engine}.command.json"
                write_json(command, {"fixture": True})
                for repeat in range(len(orders)):
                    row = {**aggregate, "engine": engine, "scope": scope, "paired_repeat": repeat, "order_in_repeat": orders[repeat].index(engine),
                           "aggregate": analyze.record(aggregate_path), "command_record": analyze.record(command), "query_details": analyze.record(raw_path)}
                    if engine == "cpp":
                        row.update(result_ids_record=analyze.record(ids_path), result_distances_record=analyze.record(ds_path))
                    rows.append(row)
        evaluation_path = root / "evaluation.json"
        write_json(evaluation_path, {"protocol": analyze.record(protocol_path), "calibration": analyze.record(calibration_path),
                                     "claim": analyze.record(claim), "queries": 2, "rows": rows})
        return evaluation_path

    def test_end_to_end_separates_alias_diagnostic_from_primary_recall_and_timings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = self.fixture(root)
            output = root / "analysis"
            args = types.SimpleNamespace(evaluation=evaluation, out=output, bootstrap_samples=100)
            analyze.analyze(args)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["quality_unit"], "unique_request_with_family_cluster_bootstrap")
            self.assertFalse(summary["timing_authoritative"])
            self.assertEqual(len(summary["scopes"]), 2)
            fixed = summary["scopes"][0]
            self.assertEqual(fixed["quality_queries"], 2)
            self.assertEqual(fixed["engines"]["rust_single"]["strict_id_recall"], .95)
            self.assertEqual(fixed["engines"]["rust_single"]["distance_cutoff_recall_diagnostic"], 1.)
            self.assertEqual(fixed["engines"]["rust_single"]["cutoff_additional_hits"], 1)
            self.assertEqual(fixed["comparisons_to_cpp"]["rust_single"]["qps_ratio_paired_median"], 1.2)
            self.assertEqual(fixed["comparisons_to_cpp"]["rust_single"]["strict_id_recall_difference_pp"], -5.)
            self.assertEqual(len((output / "per-label.csv").read_text().splitlines()), 7)

    def test_optional_fourth_profile_and_eight_balanced_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engines = analyze.KNOWN_ENGINES
            rotations = [list(engines[i:] + engines[:i]) for i in range(4)]
            orders = rotations + [list(reversed(order)) for order in rotations]
            evaluation = self.fixture(root, engines, orders)
            output = root / "analysis"
            analyze.analyze(types.SimpleNamespace(evaluation=evaluation, out=output, bootstrap_samples=50))
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["engine_profiles"], list(engines))
            self.assertEqual(summary["balanced_timing_orders"], 8)
            self.assertEqual(summary["scopes"][0]["engines"]["rust_inline_global"]["repeats"], 8)
            self.assertEqual(len((output / "per-label.csv").read_text().splitlines()), 9)

    def test_unreached_90_profile_remains_in_fixed_l_and_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation_path = self.fixture(root)
            evaluation = json.loads(evaluation_path.read_text())
            calibration_path = Path(evaluation["calibration"]["path"])
            calibration = json.loads(calibration_path.read_text())
            calibration["target90"]["rust_inline"] = {"status": "unreached_within_grid", "l": None, "best_calibration_recall": .89}
            calibration_path.write_text(json.dumps(calibration))
            evaluation["calibration"] = analyze.record(calibration_path)
            evaluation["rows"] = [row for row in evaluation["rows"] if not (row["scope"] == "target90" and row["engine"] == "rust_inline")]
            evaluation_path.write_text(json.dumps(evaluation))
            output = root / "analysis"
            analyze.analyze(types.SimpleNamespace(evaluation=evaluation_path, out=output, bootstrap_samples=50))
            summary = json.loads((output / "summary.json").read_text())
            self.assertIn("rust_inline", summary["scopes"][0]["engines"])
            self.assertNotIn("rust_inline", summary["scopes"][1]["engines"])
            self.assertEqual(summary["calibrated_90_percent_policies"]["rust_inline"]["status"], "unreached_within_grid")

    def test_mutated_aggregate_is_rejected_before_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = self.fixture(root)
            rows = json.loads(evaluation.read_text())
            rows["rows"][0]["qps"] = 9999
            evaluation.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "retained engine aggregate"):
                analyze.analyze(types.SimpleNamespace(evaluation=evaluation, out=root / "analysis", bootstrap_samples=10))


class RelocationTests(unittest.TestCase):
    def tearDown(self):
        analyze.configure_relocations([])

    def test_relocated_record_requires_original_content_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            actual = Path(directory) / "file.json"
            actual.write_text("frozen content")
            expected = analyze.record(actual)
            expected["path"] = "/old/benchmark/file.json"
            analyze.configure_relocations(["/old/benchmark=" + str(Path(directory))])
            self.assertEqual(analyze.check_record(expected), actual)
            actual.write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                analyze.check_record(expected)

    def test_longest_path_prefix_wins_without_textual_partial_match(self):
        analyze.configure_relocations(["/old=/new", "/old/nested=/specific"])
        self.assertEqual(analyze.resolve_path("/old/nested/a"), Path("/specific/a"))
        self.assertEqual(analyze.resolve_path("/old/a"), Path("/new/a"))
        self.assertEqual(analyze.resolve_path("/oldish/a"), Path("/oldish/a"))

    def test_relative_duplicate_or_ineffective_mappings_rejected(self):
        for values in (["old=/new"], ["/old=relative"], ["/old=/old"], ["/old=/a", "/old=/b"]):
            with self.assertRaises(ValueError):
                analyze.configure_relocations(values)

    def test_full_analysis_resolves_protocol_calibration_and_raw_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            evaluation_path = AnalysisIntegrationTests().fixture(root)
            old = Path("/unavailable/benchmark")
            def moved_record(path):
                value = analyze.record(path)
                value["path"] = str(old / path.relative_to(root))
                return value
            def move_paths(value):
                if isinstance(value, list):
                    return [move_paths(x) for x in value]
                if isinstance(value, dict):
                    result = {key: move_paths(x) for key, x in value.items()}
                    if "path" in result and isinstance(result["path"], str) and Path(result["path"]).is_relative_to(root):
                        result["path"] = str(old / Path(result["path"]).relative_to(root))
                    return result
                return value
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(move_paths(json.loads(protocol_path.read_text()))))
            calibration_path = root / "calibration.json"
            calibration = json.loads(calibration_path.read_text())
            calibration["protocol"] = moved_record(protocol_path)
            calibration_path.write_text(json.dumps(calibration))
            evaluation = move_paths(json.loads(evaluation_path.read_text()))
            evaluation["protocol"] = moved_record(protocol_path)
            evaluation["calibration"] = moved_record(calibration_path)
            evaluation_path.write_text(json.dumps(evaluation))
            analyze.configure_relocations([str(old) + "=" + str(root)])
            output = root / "relocated-analysis"
            analyze.analyze(types.SimpleNamespace(evaluation=evaluation_path, out=output, bootstrap_samples=20))
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["path_relocations"], [{"original_prefix": str(old), "local_prefix": str(root)}])
            self.assertEqual(summary["scopes"][0]["engines"]["rust_single"]["strict_id_recall"], .95)

if __name__ == "__main__":
    unittest.main()
