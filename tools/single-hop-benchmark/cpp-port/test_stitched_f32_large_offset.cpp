// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// A sparse source crosses the uint32 byte-offset boundary without allocating
// four GiB of physical disk. Exercise the real per-label extraction routine.
#define main benchmark_stitched_application_main
#include "build_stitched_f32.cpp"
#undef main

#include <filesystem>
#include <limits>

int main(int argc, char **argv)
{
    if (argc != 2)
        throw std::invalid_argument("Usage: test_stitched_f32_large_offset EMPTY_WORK_DIRECTORY");
    const std::filesystem::path directory(argv[1]);
    if (!std::filesystem::is_directory(directory) || !std::filesystem::is_empty(directory))
        throw std::invalid_argument("Test requires an existing empty directory");
    const path base = (directory / "sparse.fbin").string();
    constexpr uint32_t dimension = 384;
    constexpr size_t row_bytes = sizeof(float) * dimension;
    constexpr uint32_t boundary = std::numeric_limits<uint32_t>::max() / row_bytes;
    constexpr uint32_t rows = boundary + 6;
    const std::vector<uint32_t> selected = {boundary - 1, boundary + 1, boundary + 5};
    auto expected = [](size_t selected_index, size_t column) {
        return static_cast<float>((selected_index + 1) * 1024 + column) / 512.0f;
    };
    {
        std::ofstream output(base, std::ios::binary);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
        output.write(reinterpret_cast<const char *>(&dimension), sizeof(dimension));
        output.seekp(8 + static_cast<size_t>(rows) * row_bytes - 1);
        output.put(0);
        for (size_t i = 0; i < selected.size(); ++i)
        {
            output.seekp(8 + static_cast<size_t>(selected[i]) * row_bytes);
            for (size_t column = 0; column < dimension; ++column)
            {
                const float value = expected(i, column);
                output.write(reinterpret_cast<const char *>(&value), sizeof(value));
            }
        }
    }
    const std::string label = "7";
    std::vector<label_set> point_labels(rows);
    for (auto id : selected)
        point_labels[id].insert(label);
    tsl::robin_map<std::string, uint32_t> counts;
    counts[label] = static_cast<uint32_t>(selected.size());
    label_set all_labels;
    all_labels.insert(label);
    const auto mappings = diskann::generate_label_specific_vector_files_f32<float>(
        base, counts, std::move(point_labels), all_labels);
    if (mappings.at(label) != selected)
        throw std::runtime_error("Extraction changed global vector IDs");
    const auto extracted = base + "_" + label;
    if (std::filesystem::file_size(extracted) != 8 + selected.size() * row_bytes)
        throw std::runtime_error("Extraction output shape/size differs");
    std::ifstream input(extracted, std::ios::binary);
    input.exceptions(std::ios::badbit | std::ios::failbit);
    uint32_t actual_rows, actual_dimension;
    input.read(reinterpret_cast<char *>(&actual_rows), sizeof(actual_rows));
    input.read(reinterpret_cast<char *>(&actual_dimension), sizeof(actual_dimension));
    if (actual_rows != selected.size() || actual_dimension != dimension)
        throw std::runtime_error("Extraction header differs");
    for (size_t i = 0; i < selected.size(); ++i)
        for (size_t column = 0; column < dimension; ++column)
        {
            float actual;
            input.read(reinterpret_cast<char *>(&actual), sizeof(actual));
            if (actual != expected(i, column))
                throw std::runtime_error("Vector corruption at the 4 GiB boundary");
        }
    std::cout << "PASS: exact extraction before and after 4 GiB; global IDs preserved" << std::endl;
}
