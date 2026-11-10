# Experimental immutable filtered disk bundles

Status: draft for review; this proposal is not an accepted general index format.
Created: 2026-11-10.

## Summary

Add an opt-in, immutable export for experimental StitchedVamana search. Reuse the
ordinary disk graph and PQ encodings, and store label membership and entry points
as a versioned `diskann-record` object. A completion manifest binds those artifacts
by length and SHA-256. Reopening needs compressed membership and PQ codes in memory,
without retaining construction adjacency or reparsing the source label documents.

## Motivation

A constructed graph alone cannot support a later search process: the process also
needs the label dictionary, memberships, and construction-selected entry points.
Text sidecars lack an unambiguous encoding for arbitrary names and do not identify
which graph they describe. Keeping a `StitchedGraph` alive retains unnecessary
adjacency and makes the disk search API depend on construction.

The goals are independent-process reopen, explicit format evolution, detection of
corruption and mixed artifacts before querying, and a bounded, validated decoder.
This experiment does not introduce a database lifecycle.

## Proposal

All implementation and tooling require `experimental_single_hop_search`. The
construction path accepts `f32` L2 vectors, using the existing binary matrix reader,
and JSON-lines documents from the existing label parser and inverted index. A
selected JSON-pointer field holds a string or an array of strings. Empty arrays or absent
fields mean no labels; malformed values are rejected. Names retain their UTF-8
bytes, including escaped quotes, slashes, and newlines.

The builder constructs per-label Vamana graphs and prunes their union. Only rows
with no retained neighbor receive a fallback edge to the next row (wrapping at
the end). Single-hop traversal still requires membership at every step, so these
fallback edges do not admit unmatching rows. Export rejects datasets with fewer
than two rows. The existing disk writer's nonzero-degree invariant is unchanged.
PQ training in the CLI requires at least `NUM_PQ_CENTROIDS` rows.

### Bundle layout and validation

`build_filtered_index` creates a fresh output directory and uses the fixed prefix
`index`. The standard path utilities select the graph, PQ pivot, and compressed
PQ filenames. The temporary memory graph is removed after the disk writer closes.
`labels.json` is a `diskann-record` manifest, referring to one binary artifact.
`bundle.json` is written last and is required to open an index.

The completion manifest is strict JSON with no defaults or aliases:

```json
{
  "format": "diskann-experimental-single-hop-f32-l2",
  "version": 1,
  "graph": {"bytes": 1234, "sha256": "64 lowercase hex digits"},
  "pq_pivots": {"bytes": 1234, "sha256": "64 lowercase hex digits"},
  "pq_vectors": {"bytes": 1234, "sha256": "64 lowercase hex digits"},
  "labels": {"bytes": 1234, "sha256": "64 lowercase hex digits"}
}
```

The lengths above illustrate the schema, not valid artifact sizes. The final
fingerprint covers `labels.json`, whose record in turn binds the binary labels
artifact by length and digest. Paths are fixed by the reader; this manifest cannot
redirect it to arbitrary filenames. An open streams and checks every fingerprint
before constructing a searcher. This costs one sequential read of the graph at
open; query-time validation would not catch wrong metadata before use.

The label record version is `0.0`, with encoding tag
`diskann-single-hop-labels/bincode-fixed-le-roaring`. Its artifact contains, in order:

1. A `u32` row count.
2. A sequence of labels, sorted by `u32` label ID. Each contains an ID, a byte
   sequence holding a portable Roaring bitmap, and a sequence of `u32` entry points.
3. A sequence of names, sorted lexicographically by UTF-8 string, each with its ID.

The outer encoding is bincode 1 with fixed-width integers and little-endian byte
order; sequence and string lengths are `u64`. Decoding rejects trailing bytes,
including trailing bytes inside a bitmap. The record stores the artifact handle,
encoding tag, exact byte length, and 32-byte SHA-256 digest. Artifacts exceeding
1 GiB are rejected before allocation. The digest and exact length are checked
before decoding. Membership must be nonempty and in range; entry points must be
sorted, distinct members; names must reference known labels. Duplicate or
out-of-order labels and names are rejected rather than silently overwritten.

Unknown bundle versions, record versions, or encodings are errors. Future changes
must explicitly define a migration or retain a reader for prior versions; no
automatic rebuild or silent reinterpretation is implied. SHA-256 detects accidental
corruption and mixed files; it does not authenticate an index against an attacker
who can rewrite both data and manifests.

### Publication and lifetime

Files must remain immutable after export. A missing or partial completion manifest
makes a failed build unopenable. There are no updates, snapshots, compaction,
journals, locks, garbage collection, or crash-durability guarantees. In particular,
writing the manifest last does not promise persistence across power loss.
The encoding and tests do not depend on Unix APIs; actual I/O uses the existing
platform-specific disk provider. Platform execution belongs in CI.

`LabelMetadata` is independent of `StitchedGraph`; disk search borrows a selected
Roaring bitmap once per query and tests membership without binary-searching a
posting list. The supported bundle loader verifies graph/PQ/label identity.
Low-level callers supplying metadata directly must use the same row ordering.

## Trade-offs

Reusing `diskann-record` provides artifact handles and version dispatch. A small
outer manifest binds the existing graph and PQ files without copying their data
into a second serialization layer. This deliberately leaves the general index
serialization design open. Integrating that identity binding into a future common
record may remove the outer manifest.

Roaring bitmaps trade format/library coupling for compact membership and fast
lookup. The encoding tag fixes that choice for this version. Neither dual postings
plus CSR nor text parsing is needed at open. Full graph hashing adds open latency;
authenticated chunk checksums or trusted prevalidation could be studied separately.

The in-memory provider uses provider-owned entry points and falls back to the first
live member in posting order when every selected entry point has been deleted.
This preserves access to live members, but cannot guarantee connectivity or recall
after arbitrary updates; searches do not have snapshot isolation.

## Evaluation

Regression tests cover exported isolated rows, process-separated build/reopen,
cached and uncached stitched searches, exact matching memberships, escaped names,
truncation at every label-artifact prefix, same-size corruption, zero tails,
unknown versions, and invalid metadata. The baseline coordinates use exactly
representable distances for architecture-independent equality checks.

The supported benchmark compares single-hop on a stitched graph with inline and
multihop on a separate ordinary Vamana graph with its normal global entry point.
It reports recall, distance comparisons, latency, and construction cost for each
setup. The comparison is experimental evidence, not a claim of general superiority.
Publish dataset-specific measurements and hardware details in the project wiki.

## Future work

FilteredVamana, universal-label semantics, dense-point splitting, non-L2 construction,
out-of-core construction, and a production format/migration policy are separate
proposals. This draft does not claim to settle the general serialization RFC.

## References

- [Label input format](00105-label-data-format-rfc.md)
- [Filtered algorithms](01128-filtered-algorithms.md)
- [Benchmark input and comparison](../diskann-benchmark/README.md#experimental-single-hop-construction-and-search)
