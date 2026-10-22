# Native filtered SSD lifecycle

`FilteredIndex<T>` supplies Rust-only filtered construction, persistence and online
updates for squared-L2 `u8`/`f32` vectors. The graph and PQ/layout are built by Rust;
no C++ executable or prebuilt C++ graph is required. Queries accept one atomic
stable label. Label IDs may be sparse and records may have multiple labels.

```rust,ignore
use std::collections::BTreeMap;
use diskann_disk::filtered::{FilteredIndex, FilteredIndexConfig, updates::Mutation};
use diskann_utils::views::rowmajor;

let config = FilteredIndexConfig { pq_chunks: 16, ..Default::default() };
let index = FilteredIndex::build("/ssd/ads", vectors, postings, config).await?;
let before = index.snapshot();
index.apply(&[Mutation::Upsert { id: 42, vector, labels: vec![7, 19] }])?;
let results = index.snapshot().search_async(query, 7, 100, 400).await?;
index.compact().await?;
// `before` remains valid until dropped, even across compaction.
drop(before);
index.collect_garbage()?;
```

`vectors` is a row-major matrix and `postings` maps label IDs to sorted unique row
ordinals. Initial stable IDs equal row ordinals. `insert` allocates monotonically
increasing IDs; compaction drops deleted rows and persists an ordinal-to-stable-ID
mapping. `replace_labels` and `replace_vector` preserve the remaining record fields.
`open` validates artifact checksums, metadata and scalar encoding, then replays the
journal. The implementation rejects missing committed journals and complete corrupt
frames. It recovers only an incomplete final journal append.

Readers hold immutable snapshots and never take the mutation lock. Each nonempty
batch stages its entire delta before journaling, then appends/fsyncs one framed
transaction before publishing its sequence. Original base memberships remain usable
for navigation: deleted, replaced and relabeled base records are removed from outputs,
while their live replacements are searched exactly in the delta and merged by stable
ID. Query results include sequence, retry count, delta candidates and exact-fallback
status. An absent label returns no hits; results are never padded with invalid IDs.
Shadowed labels fetch enough exact-sorted base candidates before merging: a full
set of far delta hits cannot prematurely hide better surviving base neighbors.
Original-label shadow sets are maintained per snapshot, so untouched labels retain
their small top-k output path even when unrelated records change.

Strict graph search reranks every expanded vertex using exact distances already
loaded during traversal. Its bounded PQ frontier has exactly `L` slots and breaks
PQ/exact-distance ties by ID. Established graph, multihop and adaptive-inline modes
keep their existing queue/sort policies.

Compaction captures a sequence, materializes live records, builds and fsyncs a new
Rust graph generation and re-encodes vectors with the committed PQ codebook, replays
newer transactions under the writer lock, atomically
publishes `CURRENT`, then checkpoints the journal. Mutations and queries continue
while the new graph is built. Caller-held snapshots lease old generation files;
garbage collection removes them only after their final lease ends. A snapshot also
retains the exclusive store lease if it outlives the index owner. A failed durable
write with uncertain outcome blocks further writes/collection until reopen.

Keeping the codebook fixed avoids unnecessary approximate-distance changes after
compaction. The first build trains in Rust by default. Set `initial_pq_codebook`
to reuse a compatible trained codebook, including a legacy C++ L2 pivot file; the
generation validates and owns its copy. Reopening and compacting do not need the
external file. Even an empty compacted generation retains its codebook. Graph
rebuilding can still change recall, so validate candidates against per-label floors.

The delta keeps its exact top-k before merging, while reporting the full matching
candidate count. Admission bounds the number of records and journal bytes, rather
than making arbitrarily large matching deltas free. Relative store roots become
absolute when opened. Failed preparations remove partial staging files, and garbage
collection also reclaims recognized crash-orphan staging directories.

Use `compact_with_validation` to evaluate the captured candidate snapshot against
held-out per-label recall and latency floors before publication. Rejection leaves
the committed generation and WAL unchanged. The callback can retain a snapshot;
its files remain leased even if publication is rejected. The default `compact`
checks structural correctness but does not assert an application-specific recall floor.

Run construction and compaction on a background runtime/thread in a service. They
perform blocking CPU/SSD work and require working memory for the live base, per-label
Vamana construction and stitched edges. `search_async` uses Tokio's blocking pool;
synchronous search is suited to dedicated search workers. Index sizes below 256
rows use an explicit exact-base fallback because the native PQ format has 256
centroids. Short approximate results after shadow filtering trigger bounded retries
and, if needed, an exact posting scan; monitor `exact_fallback` separately.

This is an opt-in migration implementation, not a claim that arbitrary metadata
expressions or unrestricted update rates are solved. Only atomic label queries and
L2 are implemented here. Deltas have record/byte limits, and WAL growth has its own
limit; compact before admission fails. Exact delta work grows with matching changed
records. Compaction is a complete rebuild, rather than in-place SSD graph surgery.
The builder protects per-label connectivity and reports insufficient degree instead
of silently dropping it. Rust-built topology/PQ may require separate recall tuning
from the imported C++ graph. Durable publication currently requires Unix directory
fsync semantics; Windows durable stores are explicitly unsupported.
Float vectors and queries require finite components and a per-vector squared norm
at most `f32::MAX / 8`, accumulated in f64, to keep native exact distances finite.
Updates are validated before WAL append, and stored f32 values are checked on reopen.

The executable `filtered_lifecycle` builds/queries/exercises a prepared dataset.
See `benchmark-results/single-hop/migration/REPORT.md` for measured coverage and limits.
