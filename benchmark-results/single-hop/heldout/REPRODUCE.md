# Frozen-policy held-out validation

Run from the repository root. Existing original indexes and policy are prerequisites;
these commands do not rebuild or retune them. Keep the historical migration results.

## Prepare the independent query pool

See `python3 tools/single-hop-benchmark/prepare_heldout.py --help` for explicit
source paths. Preparation includes every remaining eligible public one-tag query
for the frozen 64 labels after excluding tuning IDs and all tuning embeddings.
Transfer `data/prepared-10m-heldout/` preserving its relative symlinks to the original
`data/prepared-10m/` directory. Do not dereference and duplicate the base vectors.
The retained `manifest.json` and `exclusion-evidence.json` identify exact sources.

## Build the instrumented runner

Use the source bundle `results/lifecycle-v5-heldout-complete-sources.tar.gz` over the
same pinned Rust checkout as the migration. It changes only the example driver from
the archived v4 sources. On the Linux host, from its `rust/` directory:

```sh
export PATH=/home/alex/.cargo/bin:$PATH
cargo +1.97.1 test --offline -p diskann-disk --example filtered_lifecycle
cargo +1.97.1 build --offline --release -p diskann-disk --example filtered_lifecycle
```

Preserve this binary separately from the prior v4 binary. Deploy the current
`cpp-port/run_reference.py` and `run_heldout.py` helpers. The existing C++ search
binary remains unchanged.

## Run six serial pairs

The machine must be otherwise idle. From `/home/alex/diskann-single-hop-benchmark`,
choose a fresh output directory; the runner refuses to overwrite an existing one.

```sh
taskset -c 0-7 python3 rust/tools/single-hop-benchmark/run_heldout.py \
  --root /home/alex/diskann-single-hop-benchmark \
  --prepared /home/alex/diskann-single-hop-benchmark/data/prepared-10m-heldout \
  --out /home/alex/diskann-single-hop-benchmark/results/heldout-NEW \
  --native-binary /home/alex/diskann-single-hop-benchmark/artifacts/heldout/filtered_lifecycle.v5-heldout \
  --source-archive /home/alex/diskann-single-hop-benchmark/artifacts/heldout/lifecycle-v5-heldout-complete-sources.tar.gz \
  --plan /home/alex/diskann-single-hop-benchmark/results/migration/native-label-l-plan.json \
  --repeats 6 --fixed-control --timing-authoritative
```

Each engine performs a complete same-process warmup with matching eight workers.
Pairs alternate C++/Rust and Rust/C++. Query details are written after measurement.
The fixed-L64 control runs after all pairs and supplies quality diagnostics only.
Policy, binaries and index identities are checked before and after the run.

## Audit and summarize

Retain the entire output directory, including original per-job logs, query details,
C++ result matrices, manifests and normalized results. The analyzer maps original
remote paths through the retained directory; do not rewrite the raw result files.
Use `python3 tools/single-hop-benchmark/summarize_heldout.py --help` and the retained
`analysis-command.json` for the full original invocation. NumPy is required.
The analyzer independently verifies source split, exact truth, frozen policy
selection, original-job bindings, all result IDs, latency statistics and identities.

Recall uncertainty resamples paired queries, once per unique query. Six timing
repeats do not create six times as many independent quality samples. The report
includes marginal and simultaneous approximate bootstrap intervals. A lack of a
significant shortfall does not establish a per-label noninferiority guarantee.
