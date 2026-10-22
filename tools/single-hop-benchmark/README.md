# Filtered SSD prototype experiment

The original experiment imports the **same C++ filtered graph and PQ artifacts**
into Rust, then compares strict single-hop, multi-hop, and adaptive-inline traversal.
The migration phase adds corrected C++-equivalent strict search and a native Rust
filtered build/update/persistence API. Existing search defaults remain unchanged.
See `benchmark-results/single-hop/migration/REPORT.md` for current evidence and
`diskann-disk/src/filtered/README.md` for the serving contract.

## Dataset and scope

Use the public BigANN 2023 YFCC10M filtered-search dataset: 10 million 192-dimensional
uint8 CLIP embeddings, natural metadata tags, 100,000 public image queries, and
exact filtered top-10 ground truth. Release terms: CC BY 4.0. Attribution and source:
<https://big-ann-benchmarks.com/neurips23.html>.

The initial experiment selects natural single-tag queries and a recorded subset of
labels across selectivities. The original two-tag queries use AND semantics and
are excluded from the single-label experiment. Exact top-100/top-1,000 checks use
256 original one-tag queries on the full 10M base. These tests do not establish
support for arbitrary conjunctions or production ad-service latency SLOs.

Full-10M published ground truth can be selected by query row. A cropped base needs
new exact ground truth; discarding out-of-crop IDs from the published top-10 is
incorrect. Dataset preparation records selections, counts, file sizes and hashes.

## Prototype API

The disk search API exposes opt-in modes; callers supply eligible dataset IDs as
entry points. Existing search defaults are unchanged.

```rust
let mode = SearchMode::single_hop_filter(|id| membership.contains(id))
    .with_starting_points(label_entry_points);
let result = searcher.search(query, k, l, Some(4), mode)?;
```

Use `SearchMode::multihop_filter(predicate)` for the two-hop expansion comparison,
or the existing `inline_filter` mode. All entry IDs must be inside the graph;
strict mode also requires each to satisfy the predicate. Empty entry lists search
no graph nodes. The public result buffer retains its existing padding contract;
use `stats.result_count` to distinguish valid results from padding.

The native `FilteredIndex` path supplies label-aware construction/pruning, persisted
memberships, centroid entry-point selection and bounded online updates. An arbitrary
unfiltered graph does not become suitable merely by enabling this search mode.

```sh
cargo build --release -p diskann-disk --example filtered_lifecycle
target/release/examples/filtered_lifecycle \
  --action all --root NEW_NATIVE_INDEX --prepared PREPARED_DATA --out LIFECYCLE.jsonl \
  --threads 8 --query-threads 8 --l 60,64,144 --reps 3 --updates 100 \
  --timing-authoritative true
```

An optional `--pq-codebook PATH` copies a compatible trained L2 codebook into the
initial generation. Without it, the first build trains in Rust. Compactions retain
the committed codebook while encoding changed vectors and rebuilding the graph.
Use `compact_with_validation` to reject rebuilt per-label quality regressions.
The example's `all` scenario mutates unqueried natural rows and a fresh probe label;
it separately validates exact changed-record truth and preserves the supplied
natural-query ground truth. For matching-update overhead, use `--action delta`
on a freshly compacted index and `--delta-sizes 100,1000,10000`. This proves that
synthetic far inserts cannot change the supplied top-k before timing them.

The query-only `--label-l PLAN.json` option measures a label-to-L policy through
the same public API. It requires one nominal `--l`, complete queried-label coverage
and actual L values at least that nominal value. Both warmup and measurement use
the same policy. `plan_label_l.py plan` selects the smallest observed breadth per
label meeting C++ L64 hit counts from a diagnostic grid; its `summarize` mode
validates the policy's authoritative repeats and source/index identity. Selection
and evaluation on the same queries are explicitly in-sample. Fixed-L summaries
reject policy rows so they cannot be pooled as fixed-L measurements.

## Artifact compatibility

The Rust example reads the legacy C++ nine-field disk header and translates it in
memory. Graph records and PQ codes are unchanged. The supported prototype format
is uint8/L2, uncompressed disk vectors, no reorder vectors, one block per node,
4096-byte blocks, one entry point per label, and no synthetic dummy nodes. It checks entry-point membership,
PQ/graph dimensions and IDs. Unsupported layouts fail rather than silently change
the experiment. Label maps translate C++ internal labels back to original labels.

Legacy filenames are `<prefix>_disk.index`, `<prefix>_pq_pivots.bin`,
`<prefix>_pq_compressed.bin`, `<prefix>_disk.index_labels_map.txt`, and
`<prefix>_disk.index_labels_to_medoids.txt`.

## Measurement

Run an optimized build with the repository toolchain:

```sh
cargo build --release -p diskann-disk --example filtered_ssd_bench
target/release/examples/filtered_ssd_bench \
  --prefix INDEX_PREFIX --prepared PREPARED_DATA --out RESULTS.jsonl \
  --modes inline,multihop,single --starts label \
  --ls 20,40,80,160,320 --threads 1,8 --reps 3 --limit 2000 --k 10 --bypass true \
  --timing-authoritative true
```

Strict search requires matching label entries. Optional `--starts global` diagnostics
need a topology check first: a stitched union can contain disconnected components,
so failure from its global medoid is not a conventional unfiltered-graph baseline.
Prepared files are `query.u8bin`, `query_labels.u32bin`, `groundtruth.bin`, and
`label_<original-id>.u32bin` sorted postings. Matrices have little-endian u32
row/column headers; ground truth includes IDs followed by distances.

The example uses positional file reads and macOS F_NOCACHE when bypass is enabled.
On Linux it uses the existing native O_DIRECT + io_uring reader, with counters
wrapped around its submitted batches. Both paths use aligned buffers. Full warmup
uses the same worker count as measurement to initialize pooled scratch/readers.
Use `--timing-authoritative true` only for uncontended measured runs; the default
false marks diagnostic runs that the summarizer cannot select as performance wins.
The default Rust macOS storage reader loads the whole graph into memory and must
not be used to claim an SSD benchmark. No shared graph-node cache is used here;
PQ codes and label postings remain in RAM. Reader requests/bytes are actual file
requests, **not physical-device IOPS**, and OS/device caches can still affect
results. Fixed concurrency measures throughput and service time, not open-loop
queueing latency or distributed service p99. Allocation is included in query time.

The legacy C++ reference needs explicit macOS portability adapters on Apple Silicon.
Preserve the pinned upstream commit, complete patch, compiler options and adapter
description. Native ARM Rust versus scalar/ported C++ is a local system comparison;
it is not a claim about upstream x86/Linux performance or language efficiency.
The same-graph Rust comparison isolates traversal more directly.

```sh
python3 tools/single-hop-benchmark/summarize.py RESULTS.jsonl \
  --recall 0.95 --output SUMMARY.json
```

Select the fastest **measured** configuration satisfying the same recall floor,
with at least three authoritative repeats and explicit zero result-error counters.
Report missing/short results, filter violations, per-label recall, p50/p95/p99,
reader bytes/requests, search/PQ distance comparisons and repeat variability.
The Rust comparison counter excludes full-precision reranking work; compare C++
PQ and full-distance counters separately. Optional `--adaptive-sample` and
`--adaptive-scale` parameters are recorded as distinct tuning configurations. A faster
configuration with lower recall is not a win. Per-label recall and mean/p50/p95 latency diagnostics remain visible; their small
samples do not establish reliable per-label tails. Regressions on particular labels
must remain visible even when average recall improves. Pass
`--per-label-regressions` to the summarizer for observed label drops against
a compatible multi-hop configuration. This diagnostic is not a statistical
guarantee or an automatic production rollout decision.

For candidate-volume tests, `prepare_large_k.py` produces exact top-100 and
top-1000 truth for the same 256 balanced queries on all 10M vectors. It preserves
original single-tag predicates. These smaller sets test scaling and correctness;
their tail estimates are less stable than the 2000-query top-10 set. The benchmark
requires k to equal the supplied ground-truth width.

## Native Linux replay

The native runs use eight physical cores (`taskset -c 0-7`), the native
`O_DIRECT`/`io_uring` reader, and the frozen shared index. After placing the
optimized Rust checkout, data and index under the same benchmark folder, replay
one profile with:

```sh
bash tools/single-hop-benchmark/native-rust-runs.sh \
  /absolute/benchmark/folder main-t8-single-refine
```

The script lists the measured grids, refuses to overwrite results, records its
command and binary hash, and retains whole-process resource usage. Main top-10
profiles use all 2,000 prepared queries; candidate profiles use all 256 queries.
The `k1000-inline-quality` profile is explicitly diagnostic. Its native run used a
ten-minute external cap, including warmup:

```sh
timeout 600 bash tools/single-hop-benchmark/native-rust-runs.sh \
  /absolute/benchmark/folder k1000-inline-quality
```

Run profiles sequentially in a quiet host window. For C++ use the pinned native build and
reference driver in [cpp-port](cpp-port/README.md); `OMP_TOOL=disabled` avoids the
LLVM Archer/tcmalloc startup conflict without replacing its native kernels.

The completed results, selected recall floors, per-label checks and platform
limitations are in the [experiment report](../../benchmark-results/single-hop/REPORT.md).
All authoritative work is complete: 17 normalized input files contain 135 measured
repeats across 45 configurations, each repeated three times with zero short results,
filter violations or duplicates. This includes eight-thread top-10 sweeps at the
95%/99% floors, the primary 95%-floor one-thread profile, and fixed top-100/top-1,000
candidate profiles. Profile identity and tuning scope remain separate.

The final native `k1000-inline-quality` diagnostic completed all 256 queries at
`L=64000`, sample 1000 and scale 16: 95.715625% strict ID recall, **nine short
queries**, zero filter violations and zero duplicates. Its single row has
`timing_authoritative: false`. It fails the complete-result contract and is
excluded from performance summaries, figures and ratios; no qualified native
top-1,000 inline comparison is claimed. See the
[diagnostic output](../../benchmark-results/single-hop/results/rust-k1000-inline-quality.jsonl).

Final checks and identities are retained in
[final-checks.json](../../benchmark-results/single-hop/results/final-checks.json),
[final-native-inventory.json](../../benchmark-results/single-hop/results/final-native-inventory.json),
and [rust-final-review.json](../../benchmark-results/single-hop/results/rust-final-review.json).

