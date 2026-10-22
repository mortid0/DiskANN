#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Run on Ubuntu 24.04 x86-64. Extracts private dependencies; no sudo/system install.
set -euo pipefail
benchmark_root="${1:?Usage: native-build.sh /absolute/isolated/benchmark/folder}"
source_dir="$benchmark_root/cpp"
dependency_dir="$benchmark_root/cpp-deps"
package_dir="$dependency_dir/downloads"
dependency_root="$dependency_dir/root"
library_dir="$dependency_root/usr/lib/x86_64-linux-gnu"
if test -e "$source_dir/CMakeLists.txt"; then
    echo "Source directory already populated; choose a fresh benchmark folder." >&2
    exit 1
fi
mkdir -p "$source_dir" "$package_dir"
cd "$package_dir"
apt-get download libboost1.83-dev libboost-program-options1.83-dev \
    libboost-program-options1.83.0 libaio-dev libaio1t64 libgoogle-perftools-dev \
    libgoogle-perftools4t64 libtcmalloc-minimal4t64 libmkl-dev libmkl-core \
    libmkl-def libmkl-vml-def libmkl-intel-ilp64 libmkl-intel-thread \
    libmkl-avx2 libmkl-vml-avx2 libmkl-avx512 libmkl-vml-avx512 libmkl-locale
for package in "$package_dir"/*.deb; do
    dpkg-deb -x "$package" "$dependency_root"
done
sha256sum "$package_dir"/*.deb > "$dependency_dir/package-sha256.txt"
tar -xf "$benchmark_root/diskann-cpp-native-source.tar" -C "$source_dir"
cd "$source_dir"
patch -p1 -i "$benchmark_root/native-query-metrics.patch"
cmake -S "$source_dir" -B "$source_dir/build-native" -DCMAKE_BUILD_TYPE=Release \
    -DBOOST_ROOT="$dependency_root/usr" -DBoost_NO_BOOST_CMAKE=ON \
    -DBOOST_LIBRARYDIR="$library_dir" \
    -DBoost_PROGRAM_OPTIONS_LIBRARY_RELEASE="$library_dir/libboost_program_options.so.1.83.0" \
    -DOMP_PATH=/usr/lib/llvm-18/lib -DMKL_PATH="$library_dir" \
    -DMKL_INCLUDE_PATH="$dependency_root/usr/include/mkl" \
    "-DCMAKE_CXX_FLAGS=-I$dependency_root/usr/include -L$library_dir" \
    "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,$library_dir,-rpath,/usr/lib/llvm-18/lib"
cmake --build "$source_dir/build-native" --target search_disk_index build_disk_index --parallel 8
printf '%s\n' "Runtime LD_LIBRARY_PATH=$library_dir:/usr/lib/llvm-18/lib"
