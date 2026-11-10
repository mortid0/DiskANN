# Cross-corpus float32 filtered SSD benchmark

The current Linux pipeline uses `filtered_ssd_f32_v3`,
`run_f32_benchmark_v3.py`, and `cpp-port/run_f32_reference.py`. It compares C++ strict filtered
SSD search, Rust strict single-hop search, and Rust adaptive inline filtering on
one C++ graph, PQ pair, and label entries. This isolates traversal/runtime costs.
It does not measure the quality of independently built Rust graphs.

Prepared input consists of:

- `base.fbin`, `query.fbin`: little-endian uint32 rows/dimensions and float32 values.
- `query_labels.u32bin`, `query_labels.txt`: one exact atomic label per query.
- `labels.txt`: C++ comma-separated numeric labels, one row per indexed vector.
- `label_<id>.u32bin`: sorted unique eligible IDs, uint32 rows/width=1 header.
- `groundtruth.bin`: uint32 query count/K, then query-major uint32 IDs and float32
  squared-L2 distances. Ground truth must be exhaustive for each atomic predicate.
- `manifest.json`: `base_vectors` and an `artifacts` object binding every query
  matrix, label file, ground-truth file, and posting list by SHA-256.

Calibration and holdout must be reserved before ANN tuning; repeated or related
query groups must stay in one partition. The preparation commitment records that
split. Freeze the complete source/binary/index identities before calibration:

```sh
cargo +1.97.1 build --release --offline -p diskann-disk --example filtered_ssd_f32_v3
python3 tools/single-hop-benchmark/run_f32_benchmark_v3.py --phase freeze \
  --dataset amazon-c4 --out /path/protocol \
  --prepared /path/prepared-calibration --holdout-prepared /path/prepared-validation \
  --split-commitment /path/split-commitment.json \
  --cpp-bin-dir /path/cpp/apps --rust-binary /path/filtered_ssd_f32_v3 \
  --index-prefix /path/index --inline-prefix /path/global_R64_index --sources /path/pinned-sources.tar.gz

taskset -c 0-7 python3 tools/single-hop-benchmark/run_f32_benchmark_v3.py \
  --phase calibrate --protocol /path/protocol/protocol.json --out /path/calibration

taskset -c 0-7 python3 tools/single-hop-benchmark/run_f32_benchmark_v3.py \
  --phase evaluate --protocol /path/protocol/protocol.json \
  --calibration /path/calibration/calibration.json --out /path/validation
```

Set the C++ shared-library environment before launching. The default protocol uses
eight exclusive physical cores, beam 4, K=10, no graph cache, direct SSD reads,
label starts, and a complete same-process warmup before each measured run.

The fixed comparison requests L=64 for all engines. The separate target-90 point
chooses each engine's lowest calibration L reaching 90% aggregate strict-ID recall
with complete K results. The ascending preregistered grid is
10/16/24/32/48/64/96/128/192/256/512/1024/2048/4096/8192; each engine stops at its
first passing point. Held-out recall is then reported as observed, even if it
falls below 90%. This policy does not promise 90% for each label.

Adaptive inline filtering uses sample count 1000 and maximum multiplier 16. Its
requested L is therefore not an equal-work budget: capacity may grow to 16×L.
Exact final capacity is not exposed by the library. Raw observations retain the
configured bound and actual PQ comparisons, graph hops, and IO operations.

Evaluation runs all six permutations of engine order for each comparison scope.
It writes measured mixed QPS and per-label strict recall/p99, with raw query IDs,
returned neighbors, latencies, and independent eligibility/recall/percentile
checks. Repeated timings do not increase the quality sample size. Closed-loop
search-call p99 excludes network, admission queue, and service orchestration.

Evaluation creates one atomic claim beside the split commitment. Do not delete
it to tune against the same holdout. A failed run keeps its raw evidence.

## Version 2 and the global adaptive baseline

The version 1 and 2 examples/runners remain frozen with their synthetic evidence.
The current version 3 keeps the version 2 global baseline design and accepts optional `--inline-prefix GLOBAL_PREFIX` at protocol creation.
This adds `rust_inline_global`, keeping the three shared-filtered-graph profiles
as controls. The global profile starts from the global graph header medoid and
applies the original query posting lists as its exact eligibility predicate.

Both graph prefixes need the completed companion `.manifest.json` produced by
`cpp-port/build_f32_stitched.py`. The global graph must share the exact same base,
PQ pivots and per-vector PQ codes. Its preregistered construction uses one
corpus-wide construction label, R64, build L100 and alpha 1.2. The filtered graph
uses per-label R32 with stitched maximum R128. These are different graph-building
budgets, explicitly recorded in the protocol; shared-graph inline remains the
control for separating graph topology from traversal differences.

The global entry is the real base member closest to the full-corpus centroid,
computed without queries:

```sh
python3 tools/single-hop-benchmark/select_global_entry_f32.py \
  --base /path/base.fbin --out /path/global-entry-proof.json
```

Pass its ID and proof to the builder's `--global-entry-point` and
`--global-entry-provenance` options. The selector records unchanged base hashes
before/after, its source, and the distance/ID ordering rule. The builder verifies
that the serialized disk header preserves the chosen ID.

With four profiles, evaluation uses eight balanced engine orders: four cyclic
rotations and their reverses. Every engine appears twice in every position, and
every pair runs in each precedence order four times. The target-90 decision uses
integer ground-truth hit counts. If an engine never reaches 90% with complete K
within the frozen calibration grid, its fixed-L results still run; its target-90
point is reported as unavailable.

`--functional-smoke` permits Mac synthetic checks and marks every measurement as
non-authoritative. Such runs verify formats, eligibility, distances, calibration
and scheduling; their QPS/p99 are not Linux or product-dataset measurements.

## Version 3 production predicate representation

All Rust profiles use the existing production `LabelMetadata` CSR representation:
contiguous uint64 offsets, uint32 per-point labels, and retained sorted posting
lists. Every declared workload label is loaded, including labels without queries
in the current partition, so calibration and holdout use the same predicate bank.
C++ filtered entries validate this metadata for every Rust profile; the global
profile still starts from its independently selected graph-header medoid.

This replaces the experimental per-label dense bitmaps, which would require
roughly 19 GB for the original Books workload. Each output records the predicate
representation, complete label-bank size, membership count and logical CSR/posting
array bytes. Those byte counts exclude container/allocator overhead and graph/PQ
storage; they are not process RSS measurements.

Version 3 was checked against the frozen bitmap runtime on shared and global
graphs and graphs containing zero-degree orphan rows, at L64 and L512. All 896
paired query executions across ten cases preserved ordered result IDs, float32
distance bits, PQ comparisons, IO operations and graph hops. These are functional
checks using 192 unique synthetic vectors, with no performance interpretation.

Use `analyze_f32_results_v2.py` for independent result auditing and per-label
summaries. It supports explicit path relocation when immutable preparation proofs
were copied between Mac and Linux. The source freeze and raw evidence are in
`benchmark-results/single-hop/amazon-f32-functional-smoke/`.
