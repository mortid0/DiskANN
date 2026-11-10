#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Reject incompatible PQ and changed inputs before publishing a built index."""

import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("build_f32_stitched", Path(__file__).with_name("build_f32_stitched.py"))
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def pair(prefix, rows=4, dimensions=32, chunks=32):
    offsets = [4096, 4096 + 8 + 256 * dimensions * 4]
    offsets.extend([offsets[-1] + 8 + dimensions * 4, offsets[-1] + 8 + dimensions * 4 + 8 + (chunks + 1) * 4])
    with Path(str(prefix) + "_pq_pivots.bin").open("wb") as target:
        target.write(struct.pack("<II4Q", 4, 1, *offsets))
        target.seek(offsets[0])
        target.write(struct.pack("<II", 256, dimensions) + b"\0" * (256 * dimensions * 4))
        target.write(struct.pack("<II", dimensions, 1) + b"\0" * (dimensions * 4))
        boundaries = [i * dimensions // chunks for i in range(chunks + 1)]
        target.write(struct.pack(f"<II{chunks + 1}I", chunks + 1, 1, *boundaries))
    Path(str(prefix) + "_pq_compressed.bin").write_bytes(struct.pack("<II", rows, chunks) + b"\0" * (rows * chunks))
    return offsets


class BuildGuards(unittest.TestCase):
    def test_valid_pair_and_code_shape_size_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder) / "index"
            pair(prefix)
            self.assertEqual(BUILD.validate_pq(prefix, 4, 32)["chunk_boundaries"], list(range(33)))
            path = Path(str(prefix) + "_pq_compressed.bin")
            path.write_bytes(path.read_bytes() + b"\0")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                BUILD.validate_pq(prefix, 4, 32)
            pair(prefix)
            with self.assertRaisesRegex(ValueError, "code rows/chunks"):
                BUILD.validate_pq(prefix, 5, 32)
            with self.assertRaisesRegex(ValueError, "pivot dimensions"):
                BUILD.validate_pq(prefix, 4, 64)

    def test_pivot_finiteness_offsets_chunk_boundaries_and_opq(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder) / "index"
            path = Path(str(prefix) + "_pq_pivots.bin")
            offsets = pair(prefix)
            with path.open("r+b") as target:
                target.seek(offsets[0] + 8)
                target.write(struct.pack("<f", float("nan")))
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                BUILD.validate_pq(prefix, 4, 32)
            offsets = pair(prefix)
            with path.open("r+b") as target:
                target.seek(offsets[2] + 12)
                target.write(struct.pack("<I", 0))
            with self.assertRaisesRegex(ValueError, "chunk boundaries"):
                BUILD.validate_pq(prefix, 4, 32)
            pair(prefix)
            path.write_bytes(path.read_bytes() + b"\0")
            with self.assertRaisesRegex(ValueError, "trailing PQ"):
                BUILD.validate_pq(prefix, 4, 32)
            pair(prefix)
            Path(str(path) + "_rotation_matrix.bin").touch()
            with self.assertRaisesRegex(ValueError, "OPQ"):
                BUILD.validate_pq(prefix, 4, 32)

    def test_reuse_provenance_binds_corpus_and_both_pq_files(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder) / "index"
            pair(prefix)
            proof = Path(str(prefix) + ".manifest.json")
            pq = BUILD.validate_pq(prefix, 4, 32)
            manifest = {"base_sha256": "base", "shared_pq": {suffix: {"sha256": value["sha256"]} for suffix, value in pq["files"].items()}}
            proof.write_text(json.dumps(manifest))
            BUILD.validate_pq_provenance(prefix, proof, "base", pq)
            with self.assertRaisesRegex(ValueError, "another base"):
                BUILD.validate_pq_provenance(prefix, proof, "different", pq)
            manifest["shared_pq"]["_pq_compressed.bin"]["sha256"] = "corrupt"
            proof.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "does not bind"):
                BUILD.validate_pq_provenance(prefix, proof, "base", pq)

    def test_labels_row_count_duplicates_and_blank_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.txt"
            path.write_text("1,2\n200386\n")
            self.assertEqual(BUILD.validate_labels(path, 2)["maximum_labels_per_row"], 2)
            for text, message in (("1\n", "row counts"), ("1,1\n2\n", "duplicate"), ("1\n\n", "blank"), ("1\n4294967296\n", "Invalid")):
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, message):
                    BUILD.validate_labels(path, 2)

    def test_changed_input_and_exclusive_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            path, target = Path(folder) / "input", Path(folder) / "target"
            path.write_bytes(b"first")
            before = {"input": BUILD.record(path)}
            self.assertEqual(BUILD.unchanged(before), before)
            BUILD.copy_exclusive(path, target)
            with self.assertRaises(FileExistsError):
                BUILD.copy_exclusive(path, target)
            path.write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "Input changed"):
                BUILD.unchanged(before)
            self.assertEqual(target.read_bytes(), b"first")
            called = Path(folder) / "executable-link"
            called.symlink_to(target)
            binding = {"binary": BUILD.record(called)}
            called.unlink()
            called.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "Input changed"):
                BUILD.unchanged(binding)

    def fixture(self, root):
        data, bins, output = root / "data", root / "bin", root / "output"
        data.mkdir()
        (bins / "utils").mkdir(parents=True)
        (data / "base.fbin").write_bytes(struct.pack("<II", 4, 32) + b"\0" * 4 * 32 * 4)
        (data / "labels.txt").write_text("1\n" * 4)
        (data / "query_labels.txt").write_text("1\n")
        (data / "manifest.json").write_text(json.dumps({"labels": [{"label": 1}]}))
        for name in ("build_stitched_f32", "utils/create_disk_layout", "utils/generate_pq_f32"):
            (bins / name).write_bytes(b"fake executable for orchestration guard fixture")
        argv = ["build_f32_stitched.py", "--bin-dir", str(bins), "--prepared-data", str(data), "--output-dir", str(output), "--name", "fixture", "--training-points", "256"]
        return data, bins, output, argv

    def fake_execute(self, command, log, environment):
        log.write_text("mock stage output\n")
        name = Path(command[0]).name
        if name == "generate_pq_f32":
            self.assertEqual(command[-1], "0")
            pair(Path(command[3]))
        elif name == "build_stitched_f32":
            self.active_entry = int(command[command.index("--global_entry_point") + 1]) if "--global_entry_point" in command else 0
            prefix = Path(command[command.index("--index_path_prefix") + 1])
            prefix.write_bytes(b"mock memory graph")
            for suffix in ("_labels.txt", "_labels_map.txt", "_labels_to_medoids.txt"):
                Path(str(prefix) + suffix).write_text("fixture metadata\n")
        elif name == "create_disk_layout":
            path = Path(command[-1])
            length = 4 * 32 + (32 + 1) * 4
            size = 8192
            path.write_bytes(struct.pack("<II9Q", 9, 1, 4, 32, self.active_entry, length, 4096 // length, 0, 0, 0, size) + b"\0" * (size - 80))
        else:
            self.fail(command)
        return {"command": command, "log": str(log), "elapsed_seconds": 0, "log_identity": BUILD.record(log)}

    def test_complete_publication_is_checksum_bound_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, output, argv = self.fixture(Path(folder))
            with patch.object(sys, "argv", argv), patch.object(BUILD, "execute", side_effect=self.fake_execute):
                BUILD.main()
            path = output / "fixture.manifest.json"
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(set(manifest["artifacts"]), set(BUILD.SUFFIXES))
            self.assertEqual(manifest["inputs_before"], manifest["inputs_after"])
            self.assertIn("use_opq=false", manifest["pq_encoding"]["upstream_correction"])
            self.assertEqual(manifest["global_entry_point"], 0)
            self.assertIsNone(manifest["global_entry_selection"])
            for identity in manifest["artifacts"].values():
                self.assertEqual(BUILD.sha(Path(identity["path"])), identity["sha256"])
            with patch.object(sys, "argv", argv), patch.object(BUILD, "execute", side_effect=self.fake_execute):
                with self.assertRaisesRegex(ValueError, "partial artifacts already exist"):
                    BUILD.main()
            self.assertEqual(json.loads(path.read_text()), manifest)

    def test_bad_training_output_aborts_before_graph_and_never_publishes(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, output, argv = self.fixture(Path(folder))
            called = []

            def bad(command, log, environment):
                called.append(Path(command[0]).name)
                result = self.fake_execute(command, log, environment)
                codes = Path(command[3] + "_pq_compressed.bin")
                codes.write_bytes(codes.read_bytes()[:-1])
                return result

            with patch.object(sys, "argv", argv), patch.object(BUILD, "execute", side_effect=bad):
                with self.assertRaisesRegex(ValueError, "Malformed matrix"):
                    BUILD.main()
            self.assertEqual(called, ["generate_pq_f32"])
            self.assertFalse((output / "fixture.manifest.json").exists())
            self.assertEqual(list(output.glob("fixture_*")), [])

    def test_skipping_any_declared_workload_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, _, argv = self.fixture(Path(folder))
            with patch.object(sys, "argv", argv + ["--skip-label-build", "1"]):
                with self.assertRaisesRegex(ValueError, "Cannot skip a queried"):
                    BUILD.main()

    def test_global_entry_requires_base_only_matching_proof_and_is_forwarded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data, _, output, argv = self.fixture(root)
            selector = root / "selector.py"
            selector.write_text("# fake selector for orchestration fixture only\n")
            proof = root / "entry-proof.json"
            evidence = {"selector": "nearest_to_full_base_centroid_float64_distance_then_ID",
                        "base_sha256": BUILD.sha(data / "base.fbin"), "base_vectors": 4,
                        "dimensions": 32, "entry_point": 3, "query_inputs_used": False,
                        "selector_source": BUILD.record(selector)}
            proof.write_text(json.dumps(evidence))
            options = ["--global-entry-point", "3", "--global-entry-provenance", str(proof)]
            with patch.object(sys, "argv", argv + options), patch.object(BUILD, "execute", side_effect=self.fake_execute):
                BUILD.main()
            manifest = json.loads((output / "fixture.manifest.json").read_text())
            command = manifest["build"]["command"]
            self.assertEqual(command[command.index("--global_entry_point") + 1], "3")
            self.assertEqual(manifest["global_entry_point"], 3)
            self.assertEqual(manifest["disk_metadata"]["medoid"], 3)
            self.assertEqual(manifest["global_entry_selection"]["proof"]["sha256"], BUILD.sha(proof))
            self.assertIn("global_entry_selector_source", manifest["inputs_after"])
            evidence["query_inputs_used"] = True
            proof.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError, "base-only centroid"):
                BUILD.validate_entry_provenance(proof, 3, 4, 32, BUILD.sha(data / "base.fbin"))

    def test_global_entry_missing_proof_bounds_and_corpus_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, _, _, argv = self.fixture(root)
            with patch.object(sys, "argv", argv + ["--global-entry-point", "1"]):
                with self.assertRaisesRegex(ValueError, "supplied together"):
                    BUILD.main()
            proof = root / "entry-proof.json"
            proof.write_text("{}")
            with patch.object(sys, "argv", argv + ["--global-entry-point", "4", "--global-entry-provenance", str(proof)]):
                with self.assertRaisesRegex(ValueError, "outside the base"):
                    BUILD.main()
            source = BUILD.record(Path(__file__))
            proof.write_text(json.dumps({"selector": "nearest_to_full_base_centroid_float64_distance_then_ID",
                                         "query_inputs_used": False, "entry_point": 3, "base_vectors": 4,
                                         "dimensions": 32, "base_sha256": "different", "selector_source": source}))
            with self.assertRaisesRegex(ValueError, "does not match"):
                BUILD.validate_entry_provenance(proof, 3, 4, 32, "expected")


if __name__ == "__main__":
    unittest.main()
