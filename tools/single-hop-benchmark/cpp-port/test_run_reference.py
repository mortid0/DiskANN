#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Check arbitrary-size held-out validation and independently derived label tails."""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("run_reference", Path(__file__).with_name("run_reference.py"))
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


def matrix(path, values, width=1):
    path.write_bytes(struct.pack("<II", len(values) // width, width)
                     + struct.pack(f"<{len(values)}I", *values))


class HeldOutValidatorTest(unittest.TestCase):
    def test_larger_query_set_and_label_p99(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            # More than the old 2,000-query sample, with interleaved label slices.
            n = 2003
            labels = [7 if i % 2 else 11 for i in range(n)]
            truth = [i % 4 for i in range(n)]
            matrix(data / "groundtruth.bin", truth)
            (data / "query_labels.txt").write_text("".join(f"{label}\n" for label in labels))
            for label in set(labels):
                matrix(data / f"label_{label}.u32bin", list(range(4)))
            (data / "manifest.json").write_text(json.dumps({"artifacts": {
                name: {"sha256": name} for name in
                ("query.u8bin", "query_labels.u32bin", "groundtruth.bin", "labels.txt")}}))
            matrix(data / "results.bin", truth)
            metrics = [{"returned_results": 1, "latency_us": i + 1} for i in range(n)]
            result = REFERENCE.Validator(data, points=4, k=1).validate(data / "results.bin", metrics)
            self.assertEqual(result["recall"], 1)
            self.assertEqual(result["filter_violations"], 0)
            self.assertEqual(result["short_results"], 0)
            self.assertEqual(result["duplicate_results"], 0)
            slices = {entry["label"]: entry for entry in result["labels"]}
            self.assertEqual(slices[7]["n"], 1001)
            self.assertEqual(slices[11]["n"], 1002)
            self.assertEqual(slices[7]["p99_us"], 1982)
            self.assertEqual(slices[11]["p99_us"], 1983)
            self.assertLess(slices[7]["p95_us"], slices[7]["p99_us"])


if __name__ == "__main__":
    unittest.main()
