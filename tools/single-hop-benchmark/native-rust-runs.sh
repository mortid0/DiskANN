#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Replay one measured profile in a coordinated quiet native Linux host window.
set -euo pipefail
benchmark_root="${1:?Usage: native-rust-runs.sh /absolute/benchmark/folder PROFILE}"
profile="${2:?A profile is required}"
dataset=prepared-10m
result_k=10
workers=8
repeats=3
authoritative=true
case "$profile" in
    main-t8-single) modes=single; search_ls=40,80,160,320,640 ;;
    main-t8-single-refine) modes=single; search_ls=56,60,64,176,192 ;;
    main-t8-multihop) modes=multihop; search_ls=48,56,64,80,128,144,160 ;;
    main-t8-inline) modes=inline; search_ls=256,320,640,1024,1280 ;;
    main-t8-inline-high) modes=inline; search_ls=2560 ;;
    main-t1-filtered) modes=single,multihop; search_ls=64; workers=1 ;;
    main-t1-inline) modes=inline; search_ls=320; workers=1 ;;
    k100-filtered) modes=single,multihop; search_ls=400; result_k=100 ;;
    k100-inline) modes=inline; search_ls=6400; result_k=100 ;;
    k1000-filtered) modes=single,multihop; search_ls=4000; result_k=1000 ;;
    k1000-inline-quality)
        modes=inline; search_ls=64000; result_k=1000; repeats=1; authoritative=false ;;
    *) printf 'Unknown profile: %s\n' "$profile" >&2; exit 2 ;;
esac
query_limit=2000
if test "$result_k" -ne 10; then
    dataset="prepared-10m-k$result_k"
    query_limit=256
fi
cd "$benchmark_root"
mkdir -p results
result_path="results/rust-$profile.jsonl"
test ! -e "$result_path"
command=(taskset -c 0-7 rust/target/release/examples/filtered_ssd_bench
    --prefix "$benchmark_root/indexes/yfcc10m_stitched_R32_R128_L100"
    --prepared "data/$dataset" --out "$result_path"
    --modes "$modes" --starts label --ls "$search_ls" --threads "$workers"
    --reps "$repeats" --limit "$query_limit" --k "$result_k" --bypass true
    --adaptive-sample 1000 --adaptive-scale 16 --timing-authoritative "$authoritative")
{
    date --utc --iso-8601=seconds
    printf '%q ' "${command[@]}"
    printf '\n'
    sha256sum rust/target/release/examples/filtered_ssd_bench
} > "results/rust-$profile.run.txt"
/usr/bin/time -v -o "results/rust-$profile.resources.txt" "${command[@]}"
