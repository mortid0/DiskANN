#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Regression tests for frozen-policy holdout integrity and paired analysis."""

import copy
import io
import json
from pathlib import Path
import struct
import tempfile
import tarfile
import unittest

import numpy as np

import summarize_heldout as summary


def write_matrix(path, values, distances=None):
    with path.open("wb") as stream:
        stream.write(struct.pack("<II", *values.shape))
        stream.write(values.tobytes())
        if distances is not None:
            stream.write(distances.tobytes())


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class HeldoutProofTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        count = 4001
        queries = np.zeros((count, 192), dtype="u1")
        queries[:, 0] = np.arange(count) % 256
        queries[:, 1] = np.arange(count) // 256
        truth = np.broadcast_to(np.arange(10, dtype="<u4"), (count, 10)).copy()
        distances = np.broadcast_to(np.arange(10, dtype="<f4"), (count, 10)).copy()
        labels = np.ones(count, dtype="<u4")
        self.queries_path = self.root / "public.u8bin"
        self.truth_path = self.root / "truth.bin"
        self.metadata_path = self.root / "metadata.spmat"
        write_matrix(self.queries_path, queries)
        write_matrix(self.truth_path, truth, distances)
        with self.metadata_path.open("wb") as stream:
            stream.write(struct.pack("<qqq", count, 2, count))
            stream.write(np.arange(count + 1, dtype="<i8").tobytes())
            stream.write(labels.astype("<i4").tobytes())
            stream.write(np.ones(count, dtype="<f4").tobytes())

        def workload(start, stop):
            return {"queries": queries[start:stop].copy(), "labels": labels[start:stop].copy(),
                    "ids": np.arange(start, stop, dtype="<u4"), "truth": truth[start:stop].copy(),
                    "distances": distances[start:stop].copy(),
                    "manifest": {"artifacts": {"labels.txt": {"sha256": "a" * 64}}}}

        self.tuning = workload(0, 2000)
        self.heldout = workload(2000, count)

    def proof(self):
        return summary.holdout_proof(self.tuning, self.heldout, {1: 64},
                                     self.queries_path, self.metadata_path, self.truth_path)

    def test_proves_all_remaining_rows_and_source_truth(self):
        result = self.proof()
        self.assertTrue(result["all_remaining_eligible_public_queries_included"])
        self.assertEqual(result["unique_heldout_queries"], 2001)
        self.assertEqual(result["query_weighted_mean_l"], 64)

    def test_rejects_tuning_query_id_reuse(self):
        self.heldout["ids"][0] = 0
        with self.assertRaisesRegex(ValueError, "source query IDs overlap"):
            self.proof()

    def test_rejects_tuning_embedding_reuse_even_with_different_source_id(self):
        self.heldout["queries"][0] = self.tuning["queries"][0]
        with self.assertRaisesRegex(ValueError, "query embeddings overlap"):
            self.proof()

    def test_rejects_heldout_duplicate_embeddings(self):
        self.heldout["queries"][0] = self.heldout["queries"][1]
        with self.assertRaisesRegex(ValueError, "duplicate query embeddings"):
            self.proof()

    def test_rejects_query_cherry_picking(self):
        for key in ("queries", "labels", "ids", "truth", "distances"):
            self.heldout[key] = self.heldout[key][1:]
        with self.assertRaisesRegex(ValueError, "larger than"):
            self.proof()
        # Preserve a larger-than-tuning sample while omitting an eligible row.
        self.heldout["ids"] = np.append(self.heldout["ids"], 9000)
        self.heldout["queries"] = np.vstack((self.heldout["queries"], np.full((1, 192), 255, dtype="u1")))
        with self.assertRaisesRegex(ValueError, "exactly all remaining"):
            self.proof()

    def test_rejects_truth_changed_after_split(self):
        self.heldout["truth"][0, 0] = 123
        with self.assertRaisesRegex(ValueError, "exact truth differ"):
            self.proof()

    def test_rejects_base_labels_changed_between_splits(self):
        self.heldout["manifest"]["artifacts"]["labels.txt"]["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "Base label corpus changed"):
            self.proof()


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.broadcast_to(np.arange(10, dtype=np.int64), (4, 10)).copy()
        self.times = [100., 200., 300., 400.]
        self.workload = {"queries": np.zeros((4, 192), dtype="u1"),
                         "truth": self.ids.copy(), "labels": np.asarray([1, 1, 2, 2])}
        self.postings = {1: np.arange(30), 2: np.arange(30)}
        self.row = {"repeat": 0, "recall": 1, "qps": 1000, "mean_us": 250,
                    "p50_us": 200, "p95_us": 400, "p99_us": 400,
                    "labels": [{"label": 1, "queries": 2, "recall": 1, "mean_us": 150,
                                "p95_us": 200, "p99_us": 200},
                               {"label": 2, "queries": 2, "recall": 1, "mean_us": 350,
                                "p95_us": 400, "p99_us": 400}]}

    def observe(self):
        return summary.validate_observations(self.row, self.ids, self.times,
                                             self.workload, self.postings, True)

    def test_independently_recomputes_quality_and_label_p99(self):
        result = self.observe()
        self.assertEqual(result["hits"].tolist(), [10, 10, 10, 10])
        self.assertEqual(result["labels"][1]["p99_us"], 200)
        self.assertNotIn("qps", result["labels"][1])

    def test_rejects_short_duplicate_and_wrong_filter_results(self):
        original = self.ids.copy()
        self.ids = self.ids[:, :9]
        with self.assertRaisesRegex(ValueError, "shape/range"):
            self.observe()
        self.ids = original.copy()
        self.ids[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.observe()
        self.ids = original.copy()
        self.ids[0, 0] = 900
        with self.assertRaisesRegex(ValueError, "violates label"):
            self.observe()

    def test_accepts_real_recall_regression_and_rejects_hidden_aggregate_change(self):
        self.ids[0, 0] = 15
        self.row["recall"] = .975
        self.row["labels"][0]["recall"] = .95
        self.assertEqual(self.observe()["hits"].tolist(), [9, 10, 10, 10])
        self.row["recall"] = 1
        with self.assertRaisesRegex(ValueError, "aggregate strict-ID recall"):
            self.observe()

    def test_rejects_raw_latency_or_label_p99_tampering(self):
        self.row["p99_us"] = 399
        with self.assertRaisesRegex(ValueError, "aggregate p99"):
            self.observe()
        self.row["p99_us"] = 400
        self.row["labels"][0]["p99_us"] = 100
        with self.assertRaisesRegex(ValueError, "per-label p99"):
            self.observe()

    def test_timing_repeats_do_not_multiply_quality_samples(self):
        first = self.observe()
        rows = [{**first, "repeat": repeat} for repeat in range(6)]
        result = summary.repeat_summary(rows)
        self.assertEqual(result["repetitions"], 6)
        self.assertEqual(result["strict_id_recall"], 1)
        rows[-1] = {**rows[-1], "hits": np.asarray([9, 10, 10, 10])}
        with self.assertRaisesRegex(ValueError, "hit counts differ"):
            summary.repeat_summary(rows)


class StatisticalTests(unittest.TestCase):
    def test_exact_integer_distances_avoid_uint8_overflow_across_chunks(self):
        base = np.asarray([[0] * 192, [255] * 192], dtype="u1")
        queries = np.zeros((513, 192), dtype="u1")
        queries[512] = 255
        ids = np.tile(np.asarray([[0, 1]], dtype=np.int64), (513, 1))
        result = summary.integer_distances(base, queries, ids)
        self.assertEqual(result[0].tolist(), [0, 192 * 255**2])
        self.assertEqual(result[512].tolist(), [192 * 255**2, 0])

    def test_resamples_pairs_not_individual_neighbors(self):
        delta = np.asarray([-10, 0, 10] * 20)
        result = summary.paired_bootstrap(delta, 10, 10000, 7, 34)
        self.assertEqual(result["difference_percentage_points"], 0)
        self.assertEqual(result["rust_wins_queries"], 20)
        self.assertEqual(result["cpp_wins_queries"], 20)
        low, high = result["paired_query_bootstrap_95_percent_interval_pp"]
        lower, higher = result["paired_query_bootstrap_simultaneous_95_percent_interval_pp"]
        self.assertLessEqual(lower, low)
        self.assertGreaterEqual(higher, high)
        self.assertFalse(result["bootstrap_empirical_difference_variance_is_zero"])
        self.assertEqual(result, summary.paired_bootstrap(delta, 10, 10000, 7, 34))

    def test_constant_observed_difference_explicitly_flags_degenerate_interval(self):
        result = summary.paired_bootstrap(np.zeros(20, dtype=np.int64), 10, 10000, 7, 34)
        self.assertTrue(result["bootstrap_empirical_difference_variance_is_zero"])
        self.assertEqual(result["paired_query_bootstrap_95_percent_interval_pp"], [0, 0])


class PairedJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.native_path, self.cpp_path = self.root / "native.jsonl", self.root / "cpp.jsonl"
        self.manifest_path = self.root / "jobs.json"
        merged, jobs = {"rust": [{"kind": "experiment"}], "cpp": []}, []
        for repeat in range(3):
            for position, name in enumerate(("cpp", "rust")):
                raw = {"repeat": 0, "qps": 1000, "implementation": name}
                if name == "rust":
                    raw["kind"] = "query"
                path = self.root / f"{name}-{repeat}.jsonl"
                write_jsonl(path, [raw])
                merged[name].append({**raw, "repeat": repeat, "paired_repeat": repeat,
                                     "local_repeat": 0, "local_raw_queries": "local/path"})
                minute = repeat * 2 + position
                jobs.append({"implementation": name, "paired_repeat": repeat, "local_repeat": 0,
                             "aggregate": {"path": str(path), "sha256": summary.sha(path)}, "exit_code": 0,
                             "start_utc": f"2026-10-22T01:{minute:02}:00Z",
                             "end_utc": f"2026-10-22T01:{minute:02}:30Z"})
        write_jsonl(self.native_path, merged["rust"])
        write_jsonl(self.cpp_path, merged["cpp"])
        self.manifest = {"schema_version": 1, "jobs": jobs}
        self.manifest_path.write_text(json.dumps(self.manifest))

    def check(self):
        return summary.jobs_proof(self.manifest_path, self.native_path, self.cpp_path)

    def test_binds_renumbered_repeats_to_original_logs(self):
        self.assertEqual(self.check()["pairs"], 3)

    def test_rejects_overlapping_timing_jobs(self):
        self.manifest["jobs"][1]["start_utc"] = self.manifest["jobs"][0]["start_utc"]
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.check()

    def test_rejects_merged_metric_rewrite(self):
        rows = summary.read_rows(self.cpp_path)
        rows[0]["qps"] = 2000
        write_jsonl(self.cpp_path, rows)
        with self.assertRaisesRegex(ValueError, "Merged observation differs"):
            self.check()

    def test_rejects_missing_job_and_modified_source(self):
        self.manifest["jobs"] = self.manifest["jobs"][:-1]
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "paired-job manifest"):
            self.check()


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.native, self.cpp, self.plan = (self.root / name for name in ("native", "cpp", "plan"))
        for path in (self.native, self.cpp, self.plan):
            path.write_text("immutable")
        self.old_archive, self.new_archive = self.root / "old.tar.gz", self.root / "new.tar.gz"
        required = {"diskann-disk/src/filtered/" + name + ".rs" for name in
                    ("native", "index", "view", "updates", "persistence", "metadata")}
        required.update({"diskann-disk/src/build/filtered.rs", "diskann/src/neighbor/queue.rs",
                         "diskann/src/graph/search/single_hop_filter_search.rs",
                         "diskann-disk/examples/filtered_lifecycle.rs"})
        self.files = {name: b"same algorithms" for name in required}
        self.archive(self.old_archive)
        self.files["diskann-disk/examples/filtered_lifecycle.rs"] = b"new benchmark details only"
        self.archive(self.new_archive)
        self.index = {key: "a" * 64 for key in ("current_sha256", "catalog_sha256", "journal_sha256")}
        self.tuning_path, self.identity_path = self.root / "tuning.json", self.root / "identity.json"
        tuning = {"shared_index": {"host": "test", "root": "/ssd/native"},
                  "mixed": {"index_identity": self.index,
                            "source_archive": {"path": str(self.old_archive), "sha256": summary.sha(self.old_archive)}}}
        self.tuning_path.write_text(json.dumps(tuning))
        self.evidence = {"schema_version": 1, "host": "test", "cpu_affinity": list(range(8)),
                         "plan_before_sha256": summary.FROZEN_SHA, "plan_after_sha256": summary.FROZEN_SHA}
        for name, path in (("native", self.native), ("cpp", self.cpp)):
            self.evidence[name] = {"binary_sha256": "b" * 64, "binary_before_sha256": "b" * 64,
                                   "log_sha256": summary.sha(path), "index_before": self.index,
                                   "index_after": self.index}
        self.evidence["native"]["source_archive"] = {
            "path": str(self.new_archive), "sha256": summary.sha(self.new_archive),
            "before_sha256": summary.sha(self.new_archive)}

    def archive(self, path):
        with tarfile.open(path, "w:gz") as archive:
            for name, data in sorted(self.files.items()):
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))

    def check(self):
        # identity_proof binds the task's exact policy digest; use its real bytes.
        self.identity_path.write_text(json.dumps(self.evidence))
        from unittest.mock import patch
        actual = summary.sha
        with patch.object(summary, "sha", side_effect=lambda path: summary.FROZEN_SHA
                          if path == self.plan else actual(path)):
            return summary.identity_proof(self.identity_path, self.tuning_path, self.native,
                                          self.cpp, {"root": "/ssd/native"}, self.plan)

    def test_accepts_instrumentation_change_with_same_frozen_search_index(self):
        result = self.check()
        self.assertEqual(result["changed_archived_source_files"], ["diskann-disk/examples/filtered_lifecycle.rs"])

    def test_rejects_binary_or_index_change_during_timing(self):
        self.evidence["native"]["binary_before_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "binary changed"):
            self.check()
        self.evidence["native"]["binary_before_sha256"] = "b" * 64
        self.evidence["native"]["index_after"] = {**self.index, "catalog_sha256": "c" * 64}
        with self.assertRaisesRegex(ValueError, "index identity changed"):
            self.check()

    def test_rejects_different_index_even_when_stable_within_holdout(self):
        different = {**self.index, "catalog_sha256": "c" * 64}
        self.evidence["native"]["index_before"] = different
        self.evidence["native"]["index_after"] = different
        with self.assertRaisesRegex(ValueError, "graph differs from tuned graph"):
            self.check()

    def test_rejects_search_algorithm_change_after_tuning(self):
        self.files["diskann/src/graph/search/single_hop_filter_search.rs"] = b"changed traversal"
        self.archive(self.new_archive)
        self.evidence["native"]["source_archive"].update(
            sha256=summary.sha(self.new_archive), before_sha256=summary.sha(self.new_archive))
        with self.assertRaisesRegex(ValueError, "algorithm changed after breadth tuning"):
            self.check()


if __name__ == "__main__":
    unittest.main()
