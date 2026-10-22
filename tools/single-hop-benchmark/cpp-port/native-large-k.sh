#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Run only in a coordinated quiet host window after both prepared sets are copied.
set -euo pipefail
benchmark_root="${1:?Usage: native-large-k.sh /absolute/benchmark/folder}"
mkdir -p "$benchmark_root/results"
for result_k in 100 1000; do
    if test "$result_k" -eq 100; then search_l=400; else search_l=4000; fi
    /usr/bin/time -v -o "$benchmark_root/results/native-cpp-k$result_k.resources.txt" \
        env LD_LIBRARY_PATH="$benchmark_root/cpp-deps/root/usr/lib/x86_64-linux-gnu:/usr/lib/llvm-18/lib" \
        OMP_TOOL=disabled taskset -c 0-7 python3 "$benchmark_root/run_reference.py" \
        --bin-dir "$benchmark_root/cpp/build-native/apps" \
        --prepared-data "$benchmark_root/data/prepared-10m-k$result_k" \
        --output-dir "$benchmark_root/results/native-cpp-k$result_k" \
        --index-prefix "$benchmark_root/indexes/yfcc10m_stitched_R32_R128_L100" \
        --name "cpp-k$result_k" --query-only --search-threads 8 \
        --k "$result_k" --l "$search_l" --reps 3 --timing-authoritative
done
