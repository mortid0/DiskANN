# Rust single-hop recall correction and native lifecycle

This phase preserves the earlier benchmark report/inventory as historical evidence.
The search comparison uses the same immutable C++-built YFCC10M graph/PQ bytes,
2,000 published single-label queries, exact top-10 truth, no shared graph-node cache,
native SSD readers, beam 4 and eight pinned physical cores on the authorized Linux
host. Each timing configuration has three measured repetitions after warmup.
The 64 queried natural labels contain 5,936–1,915,985 members each, or about
0.059%–19.16% of the base. Each label has 10–43 queries in this fixed evaluation;
the workload is not weighted to a production ad-traffic distribution.

## Recall cause and correction

The strict Rust prototype had three semantic differences from C++:

1. It reranked only the surviving bounded PQ frontier. C++ reranks every expanded
   vertex (`full_retset`). An expanded vertex can have a good exact distance and
   later be evicted by approximate PQ distance. Rust now retains expanded IDs and
   uses the exact distances already read during traversal; no extra rerank SSD
   reads are needed. Finite I/O budgets exclude unexpanded beam candidates.
2. It counted a real dataset entry point as a frozen synthetic point, allocating
   L+1 slots. Strict search now uses exactly L slots. This removes an unfair extra
   capacity advantage and makes the traversal comparison meaningful.
3. Rust's PQ queue and unstable exact reranking resolved equal distances differently.
   Strict search now orders both by (distance, ID), matching C++. This matters on
   integer YFCC vectors with exact-distance ties. Before this fix, remaining strict
   ID-recall differences could occur even with equivalent distances.

The new queue insertion policy is stateless and used only by strict search.
Established multihop, graph and adaptive-inline queue/sort policies are unchanged.
`single_hop_filter_frontier` remains an explicit ablation.

The final independent comparison recomputes integer squared L2 from original vectors.
Corrected strict Rust and C++ have **zero ordered-ID differences, zero set differences,
and zero saved-distance differences** on all 2,000 queries at L60, L64 and L144.
That is 6,000 comparisons of the same 2,000 queries at three breadths, and exact parity
for every label represented in this workload. No increase in L was required.

The final same-queue L64 ablation reaches 95.800% when reranking only the frontier,
versus 96.025% when reranking expanded vertices. Frontier-only recall is lower on
27 labels, with a worst decline of 1.190 percentage points. Both modes measure
about 4,268 QPS; retaining expanded candidates recovers this quality without
widening L or adding SSD rerank reads.

| L | Recall@10, both | C++ QPS | Rust QPS | Rust / C++ | C++ p99 ms | Rust p99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 95.570% | 3,476 | 4,362 | 1.255× | 3.328 | 2.599 |
| 64 | 96.025% | 3,312 | 4,269 | 1.289× | 3.313 | 2.382 |
| 144 | 99.065% | 1,743 | 2,229 | 1.279× | 5.252 | 4.064 |

Values are medians of three repetitions, using the refreshed C++ reference.
At L64 Rust multihop achieves 95.855% recall and 587 QPS: corrected single-hop is
7.27× faster at slightly higher recall. The source still avoids PQ work for rejected
or already visited neighbors; this is an algorithm/work difference, not an isolated
language-speed claim. Tail measurements cover this small fixed workload, not service
p99 under a production arrival process.

For ad candidate generation, the final k1000/L4000 run yields 98.80078125% strict
recall and 89.97 QPS, with p99 105.13 ms and no invalid/short/duplicate outputs.
The earlier unchanged C++ reference reaches the same aggregate recall at 66.16 QPS;
Rust multihop reaches 98.793359375% at 11.63 QPS, p99 730.22 ms. The k1000 C++ timing
is historical; the top-10 table uses a fresh reference. These are single-shard
measurements, not estimates of Microsoft Ads' internal production traffic.

## Rust lifecycle implementation

`diskann-disk::filtered::FilteredIndex` now provides:

- Rust per-label Vamana construction, label-aware stitching/pruning, protected
  per-label connectivity and eligible centroid entry points; native Rust PQ/layout.
- Optional compatible initial codebooks and committed-codebook reuse on compaction,
  so retraining does not change approximate distances after every rebuild. A legacy
  C++ pivot/code fixture is re-encoded to exactly the original bytes by Rust.
- Sparse validated label metadata, persisted entry points and stable application IDs.
- Atomic checksummed generation publication, checksummed framed WAL batches, fsync
  acknowledgments, exclusive writer ownership, reload and incomplete-tail recovery.
- Inserts, deletes, vector replacements and label replacements in a bounded delta.
  Base vertices keep routing eligibility while shadowed/deleted records are removed
  from outputs; exact matching delta candidates merge by stable ID.
- Background compaction with concurrent writes, sequence catchup, atomic publication,
  WAL checkpointing and reader-leased generation garbage collection.
- Optional `compact_with_validation` to reject a rebuilt generation using per-label
  recall/latency floors before publication; retained rejected candidates remain leased.
- Snapshot-consistent synchronous and async search, including safe nested-runtime
  handling and a blocking-worker path that avoids an extra thread hop per async query.

A 512-row local native-PQ/SSD smoke completed build, 100 durable mutations, 75 exact
probe queries, compaction, reopen and garbage collection. All probe queries match
independently computed exact truth. Eight lifecycle integration tests additionally cover a
300-row native SSD graph, actual entry-point deletion/relabeling, atomic rejection,
concurrent compaction/catchup, old snapshots, publication rejection, corruption,
missing journals, admission budgets, scalar mismatch, unsafe f32 admission/reload,
codebook retention through empty compaction, and abandoned staging reclamation.
A separate integration test checks 45 tied-delta merge cases against independent
integer truth. All nine integration tests pass on macOS and native Linux/io_uring.
The final local runs pass 639 Rust tests across core, disk, integrations and the
lifecycle example. Native Linux additionally passes 39 focused library tests and
six example tests. Formatting, whitespace and affected-crate all-target clippy
with warnings denied pass. Seventeen Python validation/planner tests pass separately.

The complete million-vector native Linux lifecycle completed in 133.77 seconds,
with 1.066 GiB peak process RSS. Initial build took 55.91 seconds, compaction 54.69
seconds and checksum-verified reopen 1.79 seconds. Its 100-mutation batch took
0.558 ms to acknowledge. This batch changes rows outside the queried natural labels;
the separate fresh-label probes validate the changed records, rather than treating
stale truth as current. It is not a sustained update-rate measurement.

At L60, million-vector aggregate natural recall moves from 98.56% before updates
to 98.65% after compaction; label 11 falls from 98.824% to 94.706% in a small
17-query cohort. The new compaction acceptance hook can reject such a candidate.
The full 10M lifecycle also completed: initial build 830.21 seconds, compaction
836.85 seconds, checksum-verified reopen 18.16 seconds, and 100 durable mutations
0.637 ms. Whole-process peak RSS was 9.409 GiB, including retained snapshots during
compaction. Compaction overlapped quality-only diagnostics on separate physical
cores, so its elapsed time is an observation under that load. Natural query windows
were isolated. These first lifecycle runs used the archived v1 source, before the
later codebook reuse, centroid selection and delta-merge improvements.

On 10M, L64 aggregate recall moves from 95.775% to 96.110% after compaction;
at L60 label 1444 falls from 98% to 94% in a ten-query cohort. All changed-record
probes match exact truth, and post-compaction/reopen natural quality is identical.
The C++-graph parity above does not assert identical recall for a new graph:
parallel Vamana rebuilds can choose different topology. Validate each rebuilt
generation before activation when a per-label floor is required.

The archived v2 million-vector run uses centroid entries and the copied validated C++
codebook, with no C++ executable involved. Build takes 35.40 seconds, compaction
36.20 seconds and verified reopen 1.79 seconds; the complete lifecycle takes
93.96 seconds with 1.044 GiB peak RSS. The committed pivot bytes remain identical
to the supplied file. L60/L64/L144 aggregate recall is 98.710%/98.850%/99.900%
before updates, after updates, after compaction and after reopen. The worst label
compaction decline is 0.588 percentage points on a 17-query L60 cohort. These
different builds are an end-to-end validation, not an isolated estimate of the
centroid selector's effect.

## Separating graph, entries and PQ

A quality-only 2×2 intervention uses the initial native 10M graph, the original C++
graph, and both frozen PQ artifact pairs. Each cell runs the same 2,000 queries;
timing from these concurrent diagnostic runs is excluded.

| Graph and original entry points | PQ artifacts | Recall L60 | Recall L64 | Recall L144 |
|:--|:--|--:|--:|--:|
| C++ | C++ | 95.570% | 96.025% | 99.065% |
| C++ | Fresh Rust-trained | 95.010% | 95.560% | 98.725% |
| Native Rust | C++ | 95.500% | 96.000% | 99.275% |
| Native Rust | Fresh Rust-trained | 95.275% | 95.775% | 98.930% |

Changing only entry points on the native graph/C++ PQ pair to the C++ label entries
gives 95.625%, 96.110% and 99.265%. Thus the native topology can exceed aggregate
C++ recall at the same L. It still loses on 22/19/8 labels respectively; the worst
L64 decline is 1.190 percentage points on a 42-query label cohort. This entry-only
diagnostic uses C++ entries and does not itself measure the new centroid selector.

These artifact interventions isolate this codebook's quality effect; they do not
prove an inherent language or quantizer implementation deficit. The graph and PQ
effects interact. Preserving a validated codebook and using a central entry reduce
avoidable drift, while per-label publication checks cover the remaining topology
differences. Exact same-graph C++ parity remains the stronger search correctness
result.

## Final native 10M build

The finalized v3 runtime builds the complete 10M Rust graph in 630.85 seconds with
8.203 GiB peak build-process RSS, retaining a validated C++ codebook as an input
asset. No C++ executable performs construction or encoding. Persistent files occupy
10.701 GB (including a separate vector file for compaction and exact fallback).
The subsequent query process peaks at 0.646 GiB RSS, including PQ codes, memberships
and the prepared workload; graph reads use native cache-bypassed SSD I/O.

| L | Native recall@10 | C++ recall@10 | Native QPS | C++ QPS | Native p99 ms | Labels below C++ |
|--:|--:|--:|--:|--:|--:|--:|
| 60 | 95.730% | 95.570% | 4,157 | 3,476 | 2.326 | 20 |
| 64 | 96.180% | 96.025% | 3,957 | 3,312 | 2.450 | 15 |
| 144 | 99.255% | 99.065% | 2,002 | 1,743 | 4.565 | 10 |

These are different graphs and public query APIs. At L64 the native API retains
a 19.49% throughput gain, but its worst label trails C++ by 1.190 percentage points
(label 29, 42 queries). The positive aggregate result does not erase those label
differences.

The complete copied pivot file is byte identical. Re-encoding with Rust changes
6,175 of 320 million PQ code bytes (0.001930%), affecting 0.0617% of rows. The
implementations center/accumulate/reduce floating-point scores differently. All
64 sampled Rust choices are reproduced by a scalar model of its float32 kernel;
the maximum sampled reconstruction-error gap is 0.01921 squared L2. This sample
diagnoses near-boundary rounding and does not bound arbitrary-query ADC or recall.
A code-only quality intervention on the final graph changes no per-label hit counts
at L64 or L144; at L60 the Rust codes lose two hits out of 20,000. Reusing a codebook
therefore does not imply byte-identical C++ codes, unlike the unchanged-artifact
search-parity experiment.

## Matching the new graph's per-label floors

An additional prototype chooses a small label-specific L against the C++ L64
reference, using the same 2,000-query sample. It keeps L64 for 49 labels and raises
the remaining 15: five use L68, five L72, three L80 and two L96. The query-weighted
mean is **L66.674**, just 4.178% above L64. Selection uses each measured label's
integer hit counts; it does not assume recall is monotone with L.

Three isolated measurements of that exact policy give:

| Configuration | Recall@10 | Labels below C++ L64 | QPS | p99 ms |
|:--|--:|--:|--:|--:|
| C++ graph, fixed L64 | 96.025% | 0 (reference) | 3,312 | 3.313 |
| Native Rust graph, fixed L64 | 96.180% | 15 | 3,957 | 2.450 |
| Native Rust graph, per-label L64–96 | 96.420% | **0** | **3,769** | **2.965** |

Every one of the 64 labels meets or exceeds its C++ L64 strict-ID hit count in
every repeat. The policy retains **13.82% higher throughput** and **10.51% lower
p99** than C++, with no invalid, duplicate, short or fallback results. The native
generation, catalog and WAL remain unchanged between grid selection and evaluation.
Only the benchmark driver changes to pass the selected L into the existing public
search API; it applies the same policy during warmup and measurement.

**This is in-sample tuning, not held-out validation.** The grid and evaluation use
the same queries, the two implementations use different graphs and public APIs,
and repeats do not add independent quality samples. The result demonstrates a
measured way to close the observed new-builder gaps without a large global L
increase. Bind a serving policy to its index generation, validate on fresh held-out
queries, and revalidate it after rebuilding. `compact_with_validation` can reject
a candidate when its per-label quality or query budget is unacceptable.

## Bounded matching updates

A separate million-vector stress test adds 100, 1,000 and 10,000 records, each
matching every one of the 64 queried labels. Each synthetic vector's exact integer
distance is proved strictly worse than every query's supplied kth neighbor; the
natural ground truth therefore stays current. This is deliberately adverse matching
delta work, rather than a measured ad-churn distribution.

The first implementation sorted and cloned all matching delta hits during each
base merge. The final implementation selects exact delta top-k first, preserving
the total candidate count, then merges at most k delta hits. A point below the
delta's own top-k cannot enter the combined top-k.

At L64, median throughput is 4,812 QPS with no delta, 4,692 with 100 matches,
4,709 with 1,000 and 3,829 with 10,000 before this optimization. The optimized
10,000-match path reaches 4,116 QPS, a 7.49% gain over the unpruned merge; p99
falls from 2.430 to 2.288 ms. L60/L144 gains are 17.70%/4.10%. All 64 labels'
strict hit counts remain identical, with no filter violations, short/duplicate
results or exact fallbacks. CURRENT, generation catalog and WAL hashes are identical
between the old and optimized 10,000-record runs.

Even optimized, 10,000 matching updates lower L64 throughput by about 14.5% from
the no-delta reference. Compact before delta growth consumes the service's query
budget; the API also enforces record, byte and journal admission limits. Inserts
complete before these measured queries. This does not measure sustained concurrent
writes or certify a production arrival-rate SLO.

## Scope and operation

Atomic label queries, squared L2, u8/f32 and Unix durable publication are implemented.
The mutable delta has record/byte budgets and a separate WAL cap; compact before
admission fails. Matching delta work grows with changed records. Compaction is a
full rebuild and requires working memory/temporary SSD space for another generation.
For fewer than 256 base rows, the fixed-256-centroid PQ format uses an explicit exact
fallback. Shadow filtering can trigger bounded overfetch and an exact posting scan;
query results expose that fallback so it can be monitored separately.

These APIs and benchmarks are a migration implementation, not a production rollout
or a validated arbitrary AND/OR expression engine. The tiny smoke's rebuilt natural
recall changed from 99.375% to 98.75%, illustrating why topology validation matters
independently of the now-correct same-graph search semantics.

See `diskann-disk/src/filtered/README.md` for the API/serving contract and the
`filtered_lifecycle` example for a reproducible build/query/update/compact/reopen run.

## Evidence

- `results/recall-tie-analysis-final.json`: independent per-query/per-label parity.
- `results/recall-fixed-t8.jsonl` and `results/cpp-refresh/cpp-refresh.normalized.jsonl`:
  final same-L timing repetitions.
- `results/recall-tie-analysis-exact-l-before-tie-fix.json`: tie-aware diagnosis before
  matching ID order; preserved as intermediate evidence.
- `results/recall-fixed-k1000-t8.jsonl`: candidate-generation regression run.
- `results/lifecycle-smoke.evidence.json` and `results/lifecycle-smoke.jsonl`:
  exact local lifecycle exercise and fixture provenance.
- `results/search-final-summary.json`: validated timing/parity summary and provenance.
- `results/pairing-summary.json`: graph/PQ/entry quality interventions, without timing.
- `results/lifecycle-10m-summary.json` and `results/lifecycle-1m-frozen-summary.json`:
  complete build/update/compact/reopen phases, probe validation and per-label changes.
- `results/delta-final-summary.json`: bounded matching-delta measurements, exact-truth
  proof, snapshot identity and before/after merge optimization.
- `results/final-test-summary.json`: final local and native Linux test counts.
- `results/lifecycle-10m-frozen-summary.json`: finalized native 10M build/query
  measurements, per-label C++ comparison and complete codebook/code identities.
- `results/native-label-l-summary.json` and `results/native-label-l-plan.json`:
  observed per-label floor matching, exact policy, timing repeats and scope limits.
- `results/pq-code-differences.json`, `results/pq-code-boundary-sample-analysis.json`
  and `results/pq-reencoding-quality-effect.json`: numerical assignment diagnosis
  and code-only quality intervention.
- `results/lifecycle-v1-sources.tar.gz`,
  `results/lifecycle-v2-unpruned-complete-sources.tar.gz` and
  `results/lifecycle-v3-optimized-complete-sources.tar.gz`: executable source snapshots
  to extract over the pinned Rust base. Complete-source manifests bind the later
  bundles to the native checkout, including module exports and the SSD layout writer.
- `results/lifecycle-v4-label-l-complete-sources.tar.gz`: benchmark-only policy
  extension; all runtime/library sources match v3.
- `results/lifecycle-review-regressions.json`: reproduced edge cases and their fixes.
- `results/source-provenance-final.json`: final source/binary epochs, complete host
  verification, validated summaries and retained historical inventory identity.
