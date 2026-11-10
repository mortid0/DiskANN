// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
// Functional float32 PQ helper: preserve the requested non-OPQ encoding.
#include "math_utils.h"
#include "pq.h"
#include "partition.h"
#include <cmath>
#include <memory>
#include <stdexcept>

int main(int argc, char **argv) {
    try {
        if (argc != 7 || std::string(argv[1]) != "float" || std::string(argv[6]) != "0")
            throw std::invalid_argument("Usage: generate_pq_f32 float BASE.fbin PREFIX PQ_BYTES SAMPLE_PROBABILITY 0; OPQ unsupported");
        const size_t chunks = std::stoul(argv[4]);
        const float probability = std::stof(argv[5]);
        if (chunks == 0 || !std::isfinite(probability) || probability <= 0 || probability > 1)
            throw std::invalid_argument("Invalid PQ bytes or sampling probability");
        size_t rows = 0, dimensions = 0;
        float *raw = nullptr;
        gen_random_slice<float>(argv[2], probability, raw, rows, dimensions);
        std::unique_ptr<float[]> sample(raw);
        if (rows < 256 || chunks > dimensions)
            throw std::invalid_argument("PQ requires at least 256 sampled rows and chunks <= dimensions");
        const std::string pivots = std::string(argv[3]) + "_pq_pivots.bin";
        const std::string codes = std::string(argv[3]) + "_pq_compressed.bin";
        diskann::generate_pq_pivots(sample.get(), rows, static_cast<uint32_t>(dimensions), 256,
                                    static_cast<uint32_t>(chunks), 15, pivots);
        diskann::generate_pq_data_from_pivots<float>(argv[2], 256, static_cast<uint32_t>(chunks),
                                                     pivots, codes, false);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
