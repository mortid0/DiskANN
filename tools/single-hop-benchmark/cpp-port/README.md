# C++ reference for the single-hop experiment

Pinned upstream C++ source: `78256bbab4685e1774e78d331e081a153be26823` (`cpp_main`).

The native Apple arm64 adapter exists to verify the complete pipeline locally. It preserves the filtered graph construction, label-aware pruning, entry points, PQ training algorithm and strict filtered search. Its timing is a **portable C++ reference**, not a substitute for native x86 Linux production C++ performance.

## Reproduce the Apple arm64 reference

Requirements: Python 3.12+, Apple command-line developer tools, CMake, and an existing `libomp` installation. The script downloads a checksum-pinned Boost 1.85 source archive into the requested dependency cache. It does not install system packages or modify the current Rust checkout.

```sh
python3 tools/single-hop-benchmark/cpp-port/build.py \
  --source-dir /private/tmp/diskann-cpp-reference \
  --deps-dir /private/tmp/diskann-cpp-deps --jobs 2
```

`macos-reference.patch` is the complete patch against the pinned C++ source. It selects existing scalar/auto-vectorized distance paths, substitutes Apple Accelerate CBLAS for MKL, and uses blocking positional reads with `F_NOCACHE`. Requests within a beam execute sequentially; concurrent query threads still run in parallel. OPQ is explicitly unsupported and throws. Benchmark mode uses plain PQ with full uint8 vectors on disk.

Other changes repair platform assumptions: unconditional x86 headers/prefetch, POSIX `strerror_r`, and Darwin's distinct `uint64_t`/`size_t` types. The stitched application also corrects an inclusive random upper bound from `size()` to `size()-1`. Search output is bounded by the actual number of candidates; missing results are padded with the maximum ID and infinity. This repairs an upstream out-of-bounds read on disconnected filters and leaves the search decisions for populated results unchanged. `macos-build-evidence.json` records the actual local compiler, flags, source and dependency hashes, binary hashes, and smoke-test outcome.

## Build/query prepared data

The prepared folder must contain `base.u8bin`, `query.u8bin`, `labels.txt`, `query_labels.txt` and `groundtruth.bin`. Vector files contain a little-endian `(u32 rows, u32 dimensions)` header followed by row-major uint8 values. Labels are comma-separated original numeric labels per base row; query files contain one label per row.

```sh
python3 tools/single-hop-benchmark/cpp-port/run_reference.py \
  --bin-dir /private/tmp/diskann-cpp-reference/build-macos \
  --prepared-data benchmark-results/single-hop/data/prepared-1m \
  --output-dir benchmark-results/single-hop/indexes/cpp \
  --name yfcc1m_R64_L100 --build-threads 8 --build-dram-gib 16
```

The driver fixes `R=64`, `Lbuild=100`, `FilteredLbuild=100`, `QD=32`, no dummy replicas, full vectors on disk, beam 4, K=10 and query threads 1/8. It verifies that produced PQ codes have exactly 32 bytes per vector and retains commands, logs and output IDs. Supply `--build-only` or `--query-only` to separate stages. Existing index or measurement outputs are never silently overwritten. Query-only runs write a separate manifest.

For the supplied YFCC natural multilabel fixture, this default FilteredVamana configuration disconnected several rare-label entry points and is an invalid quality baseline. Its retained results document that failure. Build the official StitchedVamana graph to obtain a usable baseline, reusing an existing 32-byte PQ codebook:

```sh
python3 tools/single-hop-benchmark/cpp-port/build_stitched.py \
  --bin-dir /private/tmp/diskann-cpp-reference/build-macos \
  --prepared-data benchmark-results/single-hop/data/prepared-1m \
  --output-dir benchmark-results/single-hop/indexes/cpp \
  --pq-prefix benchmark-results/single-hop/indexes/cpp/yfcc1m_R64_L100 \
  --name yfcc1m_stitched_R32_R128_L100 --threads 8
```

The default stitched parameters are per-label `R=32`, final `R=128`, `Lbuild=100`, and alpha 1.2. The thin disk-layout helper calls the original C++ disk serializer. For a different base size, add `--encode-pq` to encode that base with the supplied pivot table; the thin encoder calls the original C++ PQ encoder without retraining. The source codebook, code hashes, build commands and actual stage durations are recorded. Temporary per-label vector and graph files live in the output work directory.

Use `run_reference.py --query-only --name yfcc1m_stitched_R32_R128_L100` with the remaining path arguments from the first example. To store measurement outputs separately, supply `--index-prefix /absolute/path/to/the/index_prefix`, choose a new output `--name` and set `--output-dir` to the experiment folder. Each thread-count/L configuration first executes a storage warmup process. Every measured process then executes all queries with the same worker count, L/K/filter and search API before its timed loop; this also initializes worker scratch in the measured process. Three measured repeats run by default. The runner enables this with `DISKANN_REFERENCE_WARMUP_QUERIES=1`; direct CLI use can set the same variable. Warmup is necessary because `F_NOCACHE` does not guarantee that freshly written file pages have already left the cache. Loading the index, warmup, result serialization and independent correctness checks occur outside query-loop wall timing. No CPU-heavy builds should run concurrently with performance measurements. Only add `--timing-authoritative` after deliberately establishing this quiet measurement window; the default marks timings as functional evidence only.

Instrumented binaries write summary and per-query JSONL when `DISKANN_REFERENCE_STATS_JSONL` is set. Per-query latency uses a monotonic outer timer around the complete public search call, including PQ preprocessing and scratch initialization. Label conversion and correctness validation are outside that latency; the original inner search timer is retained separately. QPS uses the driver query-loop wall time. Query rows include separate PQ/full distance comparisons, predicate checks, requested read count/bytes, hops, cache hits and the actual result count. These read counts are logical requests, not physical SSD IOPS. JSON serialization runs outside the timed query loop.

`<name>.normalized.jsonl` reports independently validated strict ID recall@10, per-label recall, filter violations, short queries, duplicate IDs and latency percentiles from the outer query timers. K must equal the exact ground-truth width. PQ comparisons are `mean_cmps`; full-vector comparisons are separate. Older binaries lacking result-count instrumentation leave `short_results` unknown and cannot establish a valid comparison. The prepared identity joins the query, query-label, ground-truth and base-label SHA-256 values, in that order.

C++ remaps original labels. Import `<prefix>_disk.index_labels_map.txt` (original label TAB internal label) alongside `_disk.index_labels.txt` and `_disk.index_labels_to_medoids.txt`. The disk metadata sector begins with two u32 values, then nine u64 values when reorder is disabled; nodes start at byte 4096. The driver retains `_disk.index`, `_pq_pivots.bin`, `_pq_compressed.bin`, and label metadata. No intermediate in-memory graph is required for Rust's SSD importer.

## Native x86 Linux run

First run the read-only probe; it never installs packages:

```sh
python3 tools/single-hop-benchmark/cpp-port/probe_linux.py --workdir /chosen/ssd/workdir
```

The upstream C++ build needs GCC/OpenMP, Boost program_options, Linux libaio, tcmalloc and Intel MKL/iomp5. The pinned CMake file expects particular MKL library names. Use its `OMP_PATH`, `MKL_PATH` and `MKL_INCLUDE_PATH` configuration when required by the installed packaging. Do not install packages until the host's existing environment and administration requirements have been checked.

Export the pinned source and compile its ordinary CMake targets on that host. `run_reference.py` also accepts an unmodified native C++ build directory; native upstream binaries retain their ordinary mean/p99.9 output in logs. Apply `native-query-metrics.patch` before compiling to obtain validated result counts, complete public-call latency, counters and JSONL output. This second patch preserves native AVX, MKL and Linux AIO behavior and adds the same short-output bounds repair described above. Record its hash alongside results. It applies cleanly to the pinned source and compiled successfully with GCC 13.3 on the supplied Linux host. Compare C++ and Rust on the same native CPU, SSD, filesystem, memory/cache budget and query concurrency. The Mac reference cannot establish native Linux latency parity.

On the supplied Ryzen host, launch the driver under `taskset -c 0-7` to match Rust's eight physical-core affinity. The runner records the inherited affinity and sets MKL threads to one. Ubuntu dependency packages can be downloaded and extracted into a private directory without a system installation; retain their versions and SHA-256 values with the build evidence.

`native-build.sh` reproduces that private Ubuntu setup. Place a `git archive` tar of the pinned commit at `<folder>/diskann-cpp-native-source.tar`, copy `native-query-metrics.patch` there, then run `bash native-build.sh <folder>` with a fresh absolute folder. It builds the native query and disk-builder targets with the upstream AVX2/FMA, native CPU tuning, MKL and Linux AIO configuration. Existing LLVM 18 OpenMP is used through its `libiomp5.so` compatibility alias.

The host's LLVM 18 OpenMP tooling initially printed a TSan warning and crashed in combination with tcmalloc. Setting `OMP_TOOL=disabled` fixed the native smoke without a rebuild and disables optional instrumentation. This matches an [upstream Archer/tcmalloc report](https://github.com/llvm/llvm-project/issues/122319). The driver sets and records this environment value. Native SIMD, MKL and Linux AIO remain enabled. `native-smoke-evidence.json` records independent validation: zero missing, duplicate or filter-invalid results at L20/40/80/160 and 100% strict recall@10 at L80/160.

## Verification and interpretation

The local smoke test builds a real 4,000-vector uint8 filtered disk index with overlapping labels, then independently checks returned IDs against exact filtered ground truth. It passed at threads 1 and 8, with zero invalid/duplicate results and 100% recall@10 at L=80/160. Re-encoding that base with the existing PQ pivot table reproduced its code file byte-for-byte. The complete adapter patch was applied to a fresh export and all 21 files matched the compiled source byte-for-byte; `patch-verification.json` records both patch hashes.

On the 1M-vector YFCC pilot, the official stitched graph returned strict ID recall@10 of 88.68%, 96.56%, 99.04% and 99.75% for L=20/40/80/160 across 1,000 queries, with zero short queries, filter violations or duplicates. Those initial runs establish functional quality only; they overlapped other work and were performed shortly after writes, so their timings are not authoritative.

The final native Linux experiment used the shared 10M-vector stitched graph, 2,000 queries, K=10, beam 4, zero graph-cache nodes, eight physical cores and three repeats per configuration. All 36 measured repeats passed strict ID, filter, duplicate and result-count checks. The fastest measured eight-thread points meeting the overall recall floors were:

| Recall floor | L | Actual recall | Median QPS | Median p99 |
| --- | ---: | ---: | ---: | ---: |
| 95% | 60 | 95.570% | 3,476 | 3.388 ms |
| 99% | 144 | 99.065% | 1,743 | 5.317 ms |

The fixed one-thread L64 check reached 96.025% recall, 407 QPS and 3.043 ms median p99. `native-10m-summary.json` retains all configurations; `native-linux-build-evidence.json` records native source, patch, binary, dependency, affinity and environment identity. These are local retrieval measurements on this public image dataset; production arrival queueing and an advertising workload remain separate validation tasks. Per-label recall and latency are retained in normalized rows.

The supplementary candidate-volume profiles use the shared 256-query sample, three repeats, eight threads and the same protocol. All six repeats passed independent ID, label, duplicate and complete-result validation:

| k | L | Recall | Median QPS | Median p99 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 400 | 97.031% | 673 | 13.109 ms |
| 1,000 | 4,000 | 98.801% | 66.2 | 147.648 ms |

`native-candidate-summary.json` retains these observed configurations and their ranges. They do not establish the fastest possible tuning for either k, and 256 queries provide coarse tail estimates. Their GNU-time maximum RSS values were 571,776 KiB and 698,340 KiB for the complete driver jobs, including sequential query processes, initialization, warmup, validation and serialization. These values do not isolate search scratch or measure the combined simultaneous memory of parent and child processes. Top-10 C++ peak RSS was not instrumented.

Use matched recall and per-label slices for comparisons. Keep build time and memory separate from query performance. Record query warmup and cache policy. The upstream build has no unified public seed flag: sampling uses `std::random_device` and parallel updates can change graph artifacts. Preserve artifact hashes and use the same graph/PQ files for Rust ablations. End-to-end service queueing, fanout, shard merging and network time are outside these per-process query timings.
