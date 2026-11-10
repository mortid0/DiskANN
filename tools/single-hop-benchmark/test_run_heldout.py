#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Exercise frozen-protocol safeguards without launching benchmark processes."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_heldout as runner


class HeldoutRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "data/prepared-heldout"
        self.out = self.root / "results/heldout"
        self.native = self.root / "native.bin"
        self.archive = self.root / "sources.tar.gz"
        self.plan = self.root / "plan.json"
        self.catalog = self.root / "indexes/rust-native-10m-frozen/generation-00000000000000000001/CATALOG"
        self.cpp_binary = self.root / "cpp/build-native/apps/search_disk_index"
        assets = [self.native, self.archive, self.plan, self.catalog, self.cpp_binary,
                  self.root / "indexes/rust-native-10m-frozen/CURRENT",
                  self.root / "indexes/rust-native-10m-frozen/updates.wal",
                  self.root / "rust/tools/single-hop-benchmark/cpp-port/run_reference.py"]
        assets += [self.root / ("indexes/yfcc10m_stitched_R32_R128_L100" + suffix)
                   for suffix in ("_disk.index", "_disk.index_labels.txt", "_disk.index_labels_map.txt",
                                  "_disk.index_labels_to_medoids.txt", "_pq_pivots.bin", "_pq_compressed.bin")]
        for asset in assets:
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_bytes(asset.name.encode())
        self.plan.write_text('{"1": 64}\n')
        self.prepared.mkdir(parents=True)
        (self.prepared / "query.u8bin").write_bytes(struct.pack("<II", 2001, 192))
        (self.prepared / "manifest.json").write_text('{"query_count": 2001}\n')
        self.commands = []
        self.mutate = None

    def command(self, repeats=3, authoritative=True, control=False):
        args = ["run_heldout.py", "--root", str(self.root), "--prepared", str(self.prepared),
                "--out", str(self.out), "--native-binary", str(self.native),
                "--source-archive", str(self.archive), "--plan", str(self.plan),
                "--repeats", str(repeats)]
        if authoritative:
            args.append("--timing-authoritative")
        if control:
            args.append("--fixed-control")
        return args

    def fake_run(self, command, **kwargs):
        self.commands.append(command)
        cpp = command[0] == sys.executable

        def value(flag):
            return command[command.index(flag) + 1]

        if cpp:
            folder = Path(value("--output-dir"))
            folder.mkdir(parents=True)
            aggregate = folder / (value("--name") + ".normalized.jsonl")
            row = {"queries": 2001, "repeat": 0, "recall": .95, "qps": 1000,
                   "p99_us": 5000, "raw_stats": str(folder / "query-stats.tsv"),
                   "result_ids": str(folder / "ids.bin")}
            rows = [row]
        else:
            aggregate = Path(value("--out"))
            row = {"kind": "query", "queries": 2001, "repeat": 0, "recall": .95,
                   "qps": 1200, "p99_us": 4000,
                   "raw_queries": {"path": str(Path(value("--query-details-dir")) / "queries.jsonl")}}
            rows = [{"kind": "experiment", "source": "mock"}, row]
        aggregate.write_text("".join(json.dumps(row) + "\n" for row in rows))
        if self.mutate is not None and len(self.commands) == 1:
            self.mutate()
        return subprocess.CompletedProcess(command, 0)

    def invoke(self, *, affinity=None, frozen=True, **kwargs):
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", self.command(**kwargs)))
            stack.enter_context(patch.object(runner.os, "sched_getaffinity", create=True,
                                            return_value=set(range(8)) if affinity is None else affinity))
            stack.enter_context(patch.object(runner.subprocess, "run", side_effect=self.fake_run))
            if frozen:
                stack.enter_context(patch.object(runner, "FROZEN_PLAN", runner.sha(self.plan)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            runner.main()

    def test_rejects_changed_original_policy_before_launching_or_creating_output(self):
        with self.assertRaises(SystemExit):
            self.invoke(frozen=False)
        self.assertEqual(self.commands, [])
        self.assertFalse(self.out.exists())

    def test_requires_exact_physical_core_affinity(self):
        for affinity in ({0, 1}, set(range(1, 9)), set(range(16))):
            with self.subTest(affinity=affinity), self.assertRaises(SystemExit):
                self.invoke(affinity=affinity)
        self.assertEqual(self.commands, [])
        self.assertFalse(self.out.exists())

    def test_requires_three_pairs_and_explicit_authoritative_timing(self):
        for arguments in ({"repeats": 2}, {"authoritative": False}):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self.invoke(**arguments)
        self.assertEqual(self.commands, [])
        self.assertFalse(self.out.exists())

    def test_serial_alternating_pairs_preserve_raw_repeat_and_frozen_policy(self):
        self.invoke()
        engines = ["cpp" if command[0] == sys.executable else "rust" for command in self.commands]
        self.assertEqual(engines, ["cpp", "rust", "rust", "cpp", "cpp", "rust"])
        manifest = json.loads((self.out / "paired-run-manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual([(row["paired_repeat"], row["order_in_pair"]) for row in manifest["runs"]],
                         [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])
        for engine in ("rust", "cpp"):
            rows = [json.loads(line) for line in (self.out / (engine + ".normalized.jsonl")).read_text().splitlines()]
            rows = [row for row in rows if row.get("kind") != "experiment"]
            self.assertEqual([row["repeat"] for row in rows], [0, 1, 2])
            for row in rows:
                self.assertEqual(row["local_repeat"], 0)
                raw = [json.loads(line) for line in Path(row["original_aggregate"]).read_text().splitlines()]
                raw = [item for item in raw if item.get("kind") != "experiment"]
                self.assertEqual(raw[0]["repeat"], 0)
                self.assertEqual(runner.sha(row["original_aggregate"]), row["original_aggregate_sha256"])
        for command in self.commands:
            if command[0] != sys.executable:
                self.assertEqual(command[command.index("--label-l") + 1], str(self.plan.resolve()))
                self.assertIn("--allow-unused-plan-labels", command)
        identity = json.loads((self.out / "identity.json").read_text())
        self.assertEqual(identity["plan_before_sha256"], identity["plan_after_sha256"])
        self.assertEqual(identity["native"]["index_before"], identity["native"]["index_after"])

    def test_fixed_control_is_separate_untuned_and_diagnostic(self):
        self.invoke(control=True)
        self.assertEqual(len(self.commands), 7)
        command = self.commands[-1]
        self.assertEqual(command[command.index("--timing-authoritative") + 1], "false")
        self.assertNotIn("--label-l", command)
        manifest = json.loads((self.out / "paired-run-manifest.json").read_text())
        self.assertEqual(len(manifest["runs"]), 6)
        self.assertIsNotNone(manifest["fixed_control"])

    def test_index_change_is_rejected_after_measurement(self):
        self.mutate = lambda: self.catalog.write_text("changed catalog")
        with self.assertRaisesRegex(ValueError, "Index or policy changed"):
            self.invoke()
        manifest = json.loads((self.out / "paired-run-manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed_identity_check")
        self.assertFalse((self.out / "identity.json").exists())

    def test_binary_change_is_rejected_after_measurement(self):
        self.mutate = lambda: self.native.write_text("changed binary")
        with self.assertRaisesRegex(ValueError, "Index or policy changed"):
            self.invoke()
        manifest = json.loads((self.out / "paired-run-manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed_identity_check")

    def test_failed_subprocess_stops_next_run_and_records_failure(self):
        def failure(command, **kwargs):
            self.commands.append(command)
            return subprocess.CompletedProcess(command, 7)

        with patch.object(self, "fake_run", side_effect=failure), self.assertRaisesRegex(RuntimeError, "pair0 failed"):
            self.invoke()
        self.assertEqual(len(self.commands), 1)
        manifest = json.loads((self.out / "paired-run-manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failed_run"]["exit_code"], 7)


if __name__ == "__main__":
    unittest.main()
