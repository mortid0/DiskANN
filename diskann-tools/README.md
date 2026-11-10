# DiskANN tools

## Experimental filtered disk construction and search

The `experimental_single_hop_search` feature enables immutable stitched disk
indexes for `f32` vectors with squared L2 distance. Construction runs in RAM.
Use the existing binary matrix format and JSON-lines label documents with one
`doc_id` per vector row. `/label` can be a string or an array of strings; missing
or empty labels are supported. Entry points are derived during construction.

```sh
cargo run --release -p diskann-tools --features experimental_single_hop_search \
  --bin build_filtered_index -- \
  --data-file vectors.bin --labels labels.jsonl --label-field /label \
  --output-dir filtered-index --label-degree 16 --max-degree 32 \
  --l-build 64 --pq-bytes 8 --threads 4

cargo run --release -p diskann-tools --features experimental_single_hop_search \
  --bin search_filtered_index -- \
  --index-dir filtered-index --query-file queries.bin --label blue --k 10 --l 64
```

The output directory must be new. PQ training requires at least 256 rows, and
`pq-bytes` must be between one and the vector dimension. Search verifies the
graph, PQ data, and compact label metadata before opening. It emits one JSON row
per query containing IDs, squared distances, and distance-comparison counts.
Unknown labels return empty results. `--cache-nodes` enables the ordinary node
cache; `--io-limit` bounds disk loads and may produce fewer than `k` results.

Keep exported files immutable while using them. An interrupted build without a
valid completion manifest cannot be opened. The experimental export does not
provide update, recovery, or crash-durability protocols. Its proposed wire format
and validation are described in the [draft RFC](../rfcs/00000-experimental-filtered-disk-format.md).

For comparisons against inline and multihop in their ordinary Vamana setup, use
the [supported benchmark job](../diskann-benchmark/README.md#experimental-single-hop-construction-and-search),
which accepts upstream query-predicate files and multi-label documents.
