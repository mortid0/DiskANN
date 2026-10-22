#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Exercise delta evidence validation, including malformed provenance/quality."""

import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import summarize_delta as summary


class DeltaEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.old_path = self.root / "unpruned.jsonl"
        self.new_path = self.root / "optimized.jsonl"
        self.old_source = self.root / "unpruned.tar.gz"
        self.new_source = self.root / "optimized.tar.gz"
        self.evidence_path = self.root / "identity.json"
        for path, contents in ((self.old_source, b"old full sorting"),
                               (self.new_source, b"new partial selection")):
            with tarfile.open(path, "w:gz") as archive:
                member = tarfile.TarInfo(summary.VIEW)
                member.size = len(contents)
                archive.addfile(member, io.BytesIO(contents))
        experiment = {"kind": "experiment", "action": "delta", "root": "indexes/base",
                      "prepared": "data/prepared", "n_base": 300, "dimensions": 4,
                      "index_threads": 8, "beam": 4, "shared_cache_nodes": 0,
                      "storage": "native_io_uring", "cache_bypass": True,
                      "query_scheduling": "dynamic_one", "warmup": "full_same_process_same_worker_count",
                      "update_scope": "synthetic_far_inserts_matching_every_queried_natural_label; exact_distance_proof_required"}
        identity = ":".join(["a" * 64] * 4)
        proof = {"kind": "delta_stress", "action": "delta", "sizes": [100, 1000, 10000],
                 "scope": summary.SCOPE, "far_vector_value": 255, "dimensions": 4,
                 "labels": [7, 9], "queried_label_count": 2, "queries": 2, "k": 1,
                 "prepared_identity": identity, "generation": 2, "initial_sequence": 1,
                 "base_sequence": 1, "minimum_exact_integer_far_distance": 1000,
                 "minimum_far_minus_published_kth": 500.0,
                 "groundtruth_unchanged_by_inserts": True,
                 "proof": "every exact integer squared-L2 far distance strictly exceeds that query's published kth distance",
                 "max_delta_records": 10000, "max_delta_bytes": 67108864,
                 "largest_phase_accounted_delta_bytes": 1000000,
                 "max_frame_bytes": 16777216, "effective_frame_cap": 16777216,
                 "max_batch_records": 1000, "max_journal_bytes": 67108864,
                 "conservative_final_journal_bytes": 10000000, "first_stable_id": 300,
                 "compaction_performed": False, "graph_unchanged": True}
        self.old = [experiment, proof]
        self.new = [{**experiment, "action": "query"}]
        sequences = {0: 1, 100: 3, 1000: 4, 10000: 13}
        previous = 0
        for size in summary.SIZES:
            if size:
                batches = {100: 2, 1000: 1, 10000: 9}[size]
                self.old.append({"kind": "delta_apply", "phase": f"delta_{size}",
                                 "scope": summary.SCOPE, "inserted_this_phase": size - previous,
                                 "cumulative_delta_records": size, "durable_batches": batches,
                                 "seconds": 1.0, "mutations_per_second": float(size - previous),
                                 "sequence": sequences[size], "base_sequence": 1,
                                 "journal_bytes": size * 100, "compaction_performed": False})
                previous = size
            for breadth in summary.BREADTHS:
                for repeat in range(3):
                    labels = [{"label": label, "queries": 1, "recall": recall,
                               "mean_us": 1000.0, "p95_us": 1000.0}
                              for label, recall in ((7, 1.0), (9, 0.0))]
                    row = {"kind": "query", "implementation": "rust_native_lifecycle",
                           "phase": f"delta_{size}", "sequence": sequences[size],
                           "queries": 2, "warmup_queries": 2, "threads": 8, "l": breadth,
                           "k": 1, "repeat": repeat, "prepared_identity": identity,
                           "timing_authoritative": True, "latency_contract": "public_snapshot_search_call",
                           "qps": 1000.0, "wall_seconds": 0.002, "mean_us": 1000.0,
                           "p50_us": 900.0, "p95_us": 1100.0, "p99_us": 1200.0,
                           "recall": 0.5, "strict_id_recall": True,
                           "filter_violations": 0, "duplicate_results": 0, "short_results": 0,
                           "exact_fallback_queries": 0, "base_searches": 2,
                           "delta_candidates": size * 2, "labels": labels,
                           "synthetic_delta_records": size,
                           "stress_scope": "new_far_vectors_matching_every_queried_natural_label",
                           "groundtruth_unchanged_by_inserts": True}
                    self.old.append(row)
                    if size == 10000:
                        self.new.append({**copy.deepcopy(row), "phase": "query", "qps": 2000.0,
                                         "wall_seconds": 0.001, "mean_us": 500.0, "p50_us": 450.0,
                                         "p95_us": 550.0, "p99_us": 600.0})
        index = {"generation": 2, "base_sequence": 1, "sequence": 13,
                 **{name: "b" * 64 for name in ("current_sha256", "catalog_sha256", "journal_sha256")}}
        # Keep each version distinct; only the declared unchanged index is shared.
        self.evidence = {"schema_version": 1, "shared_index": {"host": "test-host", "root": "/benchmark/indexes/base"}}
        for name, binary, source in (("unpruned", "c", self.old_source), ("optimized", "d", self.new_source)):
            self.evidence[name] = {"binary_sha256": binary * 64,
                                   "source_archive": {"path": str(source), "sha256": summary.provenance(source)["sha256"]},
                                   "index_identity": copy.deepcopy(index)}

    def write(self):
        for path, rows in ((self.old_path, self.old), (self.new_path, self.new)):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.evidence["unpruned"]["log_sha256"] = summary.provenance(self.old_path)["sha256"]
        self.evidence["optimized"]["log_sha256"] = summary.provenance(self.new_path)["sha256"]
        self.evidence_path.write_text(json.dumps(self.evidence))

    def run_summary(self):
        self.write()
        return summary.summarize(self.old_path, self.new_path, self.old_source,
                                 self.new_source, self.evidence_path)

    def test_complete_grid_preserves_quality_and_separates_query_timings(self):
        result = self.run_summary()
        self.assertEqual(result["validation"]["unpruned_query_rows"], 36)
        self.assertEqual(result["validation"]["optimized_query_rows"], 9)
        self.assertEqual(result["validation"]["final_sequence"], 13)
        for comparison in result["optimization_comparison"]:
            self.assertEqual(comparison["qps_ratio_optimized_over_unpruned"], 2.0)
            self.assertEqual(comparison["latency_reduction_percent"]["p95_us"], 50.0)

    def test_rejects_incomplete_or_duplicate_repeats(self):
        self.new[-1]["repeat"] = self.new[-2]["repeat"]
        with self.assertRaisesRegex(ValueError, "unique repeats"):
            self.run_summary()

    def test_rejects_per_label_recall_regression_hidden_by_equal_aggregate(self):
        for row in self.new[1:]:
            row["labels"][0]["recall"], row["labels"][1]["recall"] = 0.0, 1.0
        with self.assertRaisesRegex(ValueError, "per-label quality"):
            self.run_summary()

    def test_rejects_fallbacks_or_wrong_delta_accounting(self):
        original = copy.deepcopy(self.new)
        for field, value in (("exact_fallback_queries", 1), ("delta_candidates", 19999), ("sequence", 12)):
            with self.subTest(field=field):
                self.new = copy.deepcopy(original)
                self.new[1][field] = value
                with self.assertRaises(ValueError):
                    self.run_summary()

    def test_rejects_changed_final_index_and_unbound_source_archive(self):
        original = copy.deepcopy(self.evidence)
        for field in ("catalog_sha256", "journal_sha256"):
            with self.subTest(field=field):
                self.evidence = copy.deepcopy(original)
                self.evidence["optimized"]["index_identity"][field] = "e" * 64
                with self.assertRaisesRegex(ValueError, "same final index"):
                    self.run_summary()
        self.evidence = copy.deepcopy(original)
        self.evidence["optimized"]["source_archive"]["sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "local assets"):
            self.run_summary()

    def test_rejects_nonfinite_json_and_nonpositive_distance_proof(self):
        self.old[1]["minimum_far_minus_published_kth"] = 0.0
        with self.assertRaisesRegex(ValueError, "proof margin"):
            self.run_summary()
        self.old[1]["minimum_far_minus_published_kth"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite JSON"):
            self.run_summary()


if __name__ == "__main__":
    unittest.main()
