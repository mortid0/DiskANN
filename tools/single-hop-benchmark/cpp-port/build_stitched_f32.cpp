// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// Derived from DiskANN cpp_main 78256bbab4685e1774e78d331e081a153be26823.
// Benchmark-only adapter: omit an explicitly unqueried label graph while retaining
// all points and their persisted labels. Search algorithms are unchanged.
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include "filter_utils.h"
#include <omp.h>
#ifndef _WINDOWS
#include <sys/uio.h>
#endif

#include "index.h"
#include "memory_mapper.h"
#include "parameters.h"
#include "utils.h"
#include "program_options_utils.hpp"

namespace po = boost::program_options;
typedef std::tuple<std::vector<std::vector<uint32_t>>, uint64_t> stitch_indices_return_values;

/*
 * Inline function to display progress bar.
 */
inline void print_progress(double percentage)
{
    int val = (int)(percentage * 100);
    int lpad = (int)(percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}

/*
 * Inline function to generate a random integer in a range.
 */
inline size_t random(size_t range_from, size_t range_to)
{
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<size_t> distr(range_from, range_to);
    return distr(generator);
}

// Keep extraction local to this benchmark adapter. The upstream helper uses
// uint32_t VECTOR_SIZE * uint32_t point_id, which wraps beyond 4 GiB of data.
// Cast before multiplication so large float32 corpora preserve vector identity.
namespace diskann
{
template <typename T>
inline tsl::robin_map<std::string, std::vector<uint32_t>> generate_label_specific_vector_files_f32(
    path input_data_path, tsl::robin_map<std::string, uint32_t> labels_to_number_of_points,
    std::vector<label_set> point_ids_to_labels, label_set all_labels)
{
#ifndef _WINDOWS
    auto file_writing_timer = std::chrono::high_resolution_clock::now();
    diskann::MemoryMapper input_data(input_data_path);
    char *input_start = input_data.getBuf();

    uint32_t number_of_points, dimension;
    std::memcpy(&number_of_points, input_start, sizeof(uint32_t));
    std::memcpy(&dimension, input_start + sizeof(uint32_t), sizeof(uint32_t));
    const size_t VECTOR_SIZE = static_cast<size_t>(dimension) * sizeof(T);
    const size_t METADATA = 2 * sizeof(uint32_t);
    if (number_of_points != point_ids_to_labels.size())
    {
        std::cerr << "Error: number of points in labels file and data file differ." << std::endl;
        throw;
    }

    tsl::robin_map<std::string, iovec *> label_to_iovec_map;
    tsl::robin_map<std::string, uint32_t> label_to_curr_iovec;
    tsl::robin_map<std::string, std::vector<uint32_t>> label_id_to_orig_id;

    // setup iovec list for each label
    for (const auto &lbl : all_labels)
    {
        iovec *label_iovecs = (iovec *)malloc(labels_to_number_of_points[lbl] * sizeof(iovec));
        if (label_iovecs == nullptr)
        {
            throw;
        }
        label_to_iovec_map[lbl] = label_iovecs;
        label_to_curr_iovec[lbl] = 0;
        label_id_to_orig_id[lbl].reserve(labels_to_number_of_points[lbl]);
    }

    // each point added to corresponding per-label iovec list
    for (uint32_t point_id = 0; point_id < number_of_points; point_id++)
    {
        char *curr_point = input_start + METADATA + (VECTOR_SIZE * point_id);
        iovec curr_iovec;

        curr_iovec.iov_base = curr_point;
        curr_iovec.iov_len = VECTOR_SIZE;
        for (const auto &lbl : point_ids_to_labels[point_id])
        {
            *(label_to_iovec_map[lbl] + label_to_curr_iovec[lbl]) = curr_iovec;
            label_to_curr_iovec[lbl]++;
            label_id_to_orig_id[lbl].push_back(point_id);
        }
    }

    // write each label iovec to resp. file
    for (const auto &lbl : all_labels)
    {
        int label_input_data_fd;
        path curr_label_input_data_path(input_data_path + "_" + lbl);
        uint32_t curr_num_pts = labels_to_number_of_points[lbl];

        label_input_data_fd =
            open(curr_label_input_data_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_APPEND, (mode_t)0644);
        if (label_input_data_fd == -1)
            throw;

        // write metadata
        uint32_t metadata[2] = {curr_num_pts, dimension};
        int return_value = write(label_input_data_fd, metadata, sizeof(uint32_t) * 2);
        if (return_value == -1)
        {
            throw;
        }

        // limits on number of iovec structs per writev means we need to perform
        // multiple writevs
        size_t i = 0;
        while (curr_num_pts > IOV_MAX)
        {
            return_value = writev(label_input_data_fd, (label_to_iovec_map[lbl] + (IOV_MAX * i)), IOV_MAX);
            if (return_value == -1)
            {
                close(label_input_data_fd);
                throw;
            }
            curr_num_pts -= IOV_MAX;
            i += 1;
        }
        return_value = writev(label_input_data_fd, (label_to_iovec_map[lbl] + (IOV_MAX * i)), curr_num_pts);
        if (return_value == -1)
        {
            close(label_input_data_fd);
            throw;
        }

        free(label_to_iovec_map[lbl]);
        close(label_input_data_fd);
    }

    std::chrono::duration<double> file_writing_time = std::chrono::high_resolution_clock::now() - file_writing_timer;
    std::cout << "generated " << all_labels.size() << " label-specific vector files for index building in time "
              << file_writing_time.count() << "\n"
              << std::endl;

    return label_id_to_orig_id;
#endif
}
} // namespace diskann

/*
 * function to handle command line parsing.
 *
 * Arguments are merely the inputs from the command line.
 */
void handle_args(int argc, char **argv, std::string &data_type, path &input_data_path, path &final_index_path_prefix,
                 path &label_data_path, std::string &universal_label, uint32_t &num_threads, uint32_t &R, uint32_t &L,
                 uint32_t &stitched_R, float &alpha, std::string &skip_label_build, uint32_t &global_entry_point)
{
    po::options_description desc{
        program_options_utils::make_program_description("build_stitched_f32", "Build a stitched DiskANN index with an optional unqueried label omission.")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("index_path_prefix",
                                       po::value<std::string>(&final_index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("data_path", po::value<std::string>(&input_data_path)->required(),
                                       program_options_utils::INPUT_DATA_PATH);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64),
                                       program_options_utils::MAX_BUILD_DEGREE);
        optional_configs.add_options()("Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
                                       program_options_utils::GRAPH_BUILD_COMPLEXITY);
        optional_configs.add_options()("alpha", po::value<float>(&alpha)->default_value(1.2f),
                                       program_options_utils::GRAPH_BUILD_ALPHA);
        optional_configs.add_options()("label_file", po::value<std::string>(&label_data_path)->default_value(""),
                                       program_options_utils::LABEL_FILE);
        optional_configs.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
                                       program_options_utils::UNIVERSAL_LABEL);
        optional_configs.add_options()("stitched_R", po::value<uint32_t>(&stitched_R)->default_value(100),
                                       "Degree to prune final graph down to");

        optional_configs.add_options()("skip_label_build", po::value<std::string>(&skip_label_build)->default_value(""),
                                       "Original label whose graph is omitted; it must never be queried");

        optional_configs.add_options()("global_entry_point", po::value<uint32_t>(&global_entry_point)->default_value(0),
                                       "Base-only selected global starting point; default preserves upstream zero");

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            exit(0);
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        throw;
    }
}

/*
 * Custom index save to write the in-memory index to disk.
 * Also writes required files for diskANN API -
 *  1. labels_to_medoids
 *  2. universal_label
 *  3. data (redundant for static indices)
 *  4. labels (redundant for static indices)
 */
void save_full_index(path final_index_path_prefix, path input_data_path, uint64_t final_index_size,
                     const std::vector<std::vector<uint32_t>> &stitched_graph,
                     tsl::robin_map<std::string, uint32_t> entry_points, std::string universal_label,
                     path label_data_path, uint32_t global_entry_point)
{
    // aux. file 1
    auto saving_index_timer = std::chrono::high_resolution_clock::now();
    std::ifstream original_label_data_stream;
    original_label_data_stream.exceptions(std::ios::badbit | std::ios::failbit);
    original_label_data_stream.open(label_data_path, std::ios::binary);
    std::ofstream new_label_data_stream;
    new_label_data_stream.exceptions(std::ios::badbit | std::ios::failbit);
    new_label_data_stream.open(final_index_path_prefix + "_labels.txt", std::ios::binary);
    new_label_data_stream << original_label_data_stream.rdbuf();
    original_label_data_stream.close();
    new_label_data_stream.close();

    // aux. file 2
    std::ifstream original_input_data_stream;
    original_input_data_stream.exceptions(std::ios::badbit | std::ios::failbit);
    original_input_data_stream.open(input_data_path, std::ios::binary);
    std::ofstream new_input_data_stream;
    new_input_data_stream.exceptions(std::ios::badbit | std::ios::failbit);
    new_input_data_stream.open(final_index_path_prefix + ".data", std::ios::binary);
    new_input_data_stream << original_input_data_stream.rdbuf();
    original_input_data_stream.close();
    new_input_data_stream.close();

    // aux. file 3
    std::ofstream labels_to_medoids_writer;
    labels_to_medoids_writer.exceptions(std::ios::badbit | std::ios::failbit);
    labels_to_medoids_writer.open(final_index_path_prefix + "_labels_to_medoids.txt");
    for (auto iter : entry_points)
        labels_to_medoids_writer << iter.first << ", " << iter.second << std::endl;
    labels_to_medoids_writer.close();

    // aux. file 4 (only if we're using a universal label)
    if (universal_label != "")
    {
        std::ofstream universal_label_writer;
        universal_label_writer.exceptions(std::ios::badbit | std::ios::failbit);
        universal_label_writer.open(final_index_path_prefix + "_universal_label.txt");
        universal_label_writer << universal_label << std::endl;
        universal_label_writer.close();
    }

    // main index
    uint64_t index_num_frozen_points = 0, index_num_edges = 0;
    uint32_t index_max_observed_degree = 0, index_entry_point = global_entry_point;
    const size_t METADATA = 2 * sizeof(uint64_t) + 2 * sizeof(uint32_t);
    for (auto &point_neighbors : stitched_graph)
    {
        index_max_observed_degree = std::max(index_max_observed_degree, (uint32_t)point_neighbors.size());
    }

    std::ofstream stitched_graph_writer;
    stitched_graph_writer.exceptions(std::ios::badbit | std::ios::failbit);
    stitched_graph_writer.open(final_index_path_prefix, std::ios_base::binary);

    stitched_graph_writer.write((char *)&final_index_size, sizeof(uint64_t));
    stitched_graph_writer.write((char *)&index_max_observed_degree, sizeof(uint32_t));
    stitched_graph_writer.write((char *)&index_entry_point, sizeof(uint32_t));
    stitched_graph_writer.write((char *)&index_num_frozen_points, sizeof(uint64_t));

    size_t bytes_written = METADATA;
    for (uint32_t node_point = 0; node_point < stitched_graph.size(); node_point++)
    {
        uint32_t current_node_num_neighbors = (uint32_t)stitched_graph[node_point].size();
        std::vector<uint32_t> current_node_neighbors = stitched_graph[node_point];
        stitched_graph_writer.write((char *)&current_node_num_neighbors, sizeof(uint32_t));
        bytes_written += sizeof(uint32_t);
        for (const auto &current_node_neighbor : current_node_neighbors)
        {
            stitched_graph_writer.write((char *)&current_node_neighbor, sizeof(uint32_t));
            bytes_written += sizeof(uint32_t);
        }
        index_num_edges += current_node_num_neighbors;
    }

    if (bytes_written != final_index_size)
    {
        std::cerr << "Error: written bytes does not match allocated space" << std::endl;
        throw;
    }

    stitched_graph_writer.close();

    std::chrono::duration<double> saving_index_time = std::chrono::high_resolution_clock::now() - saving_index_timer;
    std::cout << "Stitched graph written in " << saving_index_time.count() << " seconds" << std::endl;
    std::cout << "Stitched graph average degree: " << ((float)index_num_edges) / ((float)(stitched_graph.size()))
              << std::endl;
    std::cout << "Stitched graph max degree: " << index_max_observed_degree << std::endl << std::endl;
}

/*
 * Unions the per-label graph indices together via the following policy:
 *  - any two nodes can only have at most one edge between them -
 *
 * Returns the "stitched" graph and its expected file size.
 */
template <typename T>
stitch_indices_return_values stitch_label_indices(
    path final_index_path_prefix, uint32_t total_number_of_points, label_set all_labels,
    tsl::robin_map<std::string, uint32_t> labels_to_number_of_points,
    tsl::robin_map<std::string, uint32_t> &label_entry_points,
    tsl::robin_map<std::string, std::vector<uint32_t>> label_id_to_orig_id_map)
{
    size_t final_index_size = 0;
    std::vector<std::vector<uint32_t>> stitched_graph(total_number_of_points);

    auto stitching_index_timer = std::chrono::high_resolution_clock::now();
    for (const auto &lbl : all_labels)
    {
        path curr_label_index_path(final_index_path_prefix + "_" + lbl);
        std::vector<std::vector<uint32_t>> curr_label_index;
        size_t curr_label_index_size;
        uint32_t curr_label_entry_point;

        std::tie(curr_label_index, curr_label_index_size) =
            diskann::load_label_index(curr_label_index_path, labels_to_number_of_points[lbl]);
        curr_label_entry_point = (uint32_t)random(0, curr_label_index.size() - 1);
        label_entry_points[lbl] = label_id_to_orig_id_map[lbl][curr_label_entry_point];

        for (uint32_t node_point = 0; node_point < curr_label_index.size(); node_point++)
        {
            uint32_t original_point_id = label_id_to_orig_id_map[lbl][node_point];
            for (auto &node_neighbor : curr_label_index[node_point])
            {
                uint32_t original_neighbor_id = label_id_to_orig_id_map[lbl][node_neighbor];
                std::vector<uint32_t> curr_point_neighbors = stitched_graph[original_point_id];
                if (std::find(curr_point_neighbors.begin(), curr_point_neighbors.end(), original_neighbor_id) ==
                    curr_point_neighbors.end())
                {
                    stitched_graph[original_point_id].push_back(original_neighbor_id);
                    final_index_size += sizeof(uint32_t);
                }
            }
        }
    }

    const size_t METADATA = 2 * sizeof(uint64_t) + 2 * sizeof(uint32_t);
    final_index_size += (total_number_of_points * sizeof(uint32_t) + METADATA);

    std::chrono::duration<double> stitching_index_time =
        std::chrono::high_resolution_clock::now() - stitching_index_timer;
    std::cout << "stitched graph generated in memory in " << stitching_index_time.count() << " seconds" << std::endl;

    return std::make_tuple(std::move(stitched_graph), final_index_size);
}

/*
 * Applies the prune_neighbors function from src/index.cpp to
 * every node in the stitched graph.
 *
 * This is an optional step, hence the saving of both the full
 * and pruned graph.
 */
template <typename T>
void prune_and_save(path final_index_path_prefix, path full_index_path_prefix, path input_data_path,
                    uint32_t stitched_R,
                    tsl::robin_map<std::string, uint32_t> label_entry_points, std::string universal_label,
                    path label_data_path, uint32_t num_threads)
{
    size_t dimension, number_of_label_points;
    auto diskann_cout_buffer = diskann::cout.rdbuf(nullptr);
    auto std_cout_buffer = std::cout.rdbuf(nullptr);
    auto pruning_index_timer = std::chrono::high_resolution_clock::now();

    diskann::get_bin_metadata(input_data_path, number_of_label_points, dimension);

    diskann::Index<T> index(diskann::Metric::L2, dimension, number_of_label_points, nullptr, nullptr, 0, false, false,
                            false, false, 0, false);

    // not searching this index, set search_l to 0
    index.load(full_index_path_prefix.c_str(), num_threads, 1);

    std::cout << "parsing labels" << std::endl;

    index.prune_all_neighbors(stitched_R, 750, 1.2);
    index.save((final_index_path_prefix).c_str());

    diskann::cout.rdbuf(diskann_cout_buffer);
    std::cout.rdbuf(std_cout_buffer);
    std::chrono::duration<double> pruning_index_time = std::chrono::high_resolution_clock::now() - pruning_index_timer;
    std::cout << "pruning performed in " << pruning_index_time.count() << " seconds\n" << std::endl;
}

/*
 * Delete all temporary artifacts.
 * In the process of creating the stitched index, some temporary artifacts are
 * created:
 * 1. the separate bin files for each labels' points
 * 2. the separate diskANN indices built for each label
 * 3. the '.data' file created while generating the indices
 */
void clean_up_artifacts(path input_data_path, path final_index_path_prefix, label_set all_labels)
{
    for (const auto &lbl : all_labels)
    {
        path curr_label_input_data_path(input_data_path + "_" + lbl);
        path curr_label_index_path(final_index_path_prefix + "_" + lbl);
        path curr_label_index_path_data(curr_label_index_path + ".data");

        if (std::remove(curr_label_index_path.c_str()) != 0)
            throw;
        if (std::remove(curr_label_input_data_path.c_str()) != 0)
            throw;
        if (std::remove(curr_label_index_path_data.c_str()) != 0)
            throw;
    }
}

int main(int argc, char **argv)
{
    // 1. handle cmdline inputs
    std::string data_type;
    path input_data_path, final_index_path_prefix, label_data_path;
    std::string universal_label, skip_label_build;
    uint32_t num_threads, R, L, stitched_R, global_entry_point;
    float alpha;

    auto index_timer = std::chrono::high_resolution_clock::now();
    handle_args(argc, argv, data_type, input_data_path, final_index_path_prefix, label_data_path, universal_label,
                num_threads, R, L, stitched_R, alpha, skip_label_build, global_entry_point);

    path labels_file_to_use = final_index_path_prefix + "_label_formatted.txt";
    path labels_map_file = final_index_path_prefix + "_labels_map.txt";

    convert_labels_string_to_int(label_data_path, labels_file_to_use, labels_map_file, universal_label);

    // 2. parse label file and create necessary data structures
    std::vector<label_set> point_ids_to_labels;
    tsl::robin_map<std::string, uint32_t> labels_to_number_of_points;
    label_set all_labels;

    std::tie(point_ids_to_labels, labels_to_number_of_points, all_labels) =
        diskann::parse_label_file(labels_file_to_use, universal_label);

    // The skip argument names an ORIGINAL label. Conversion assigns internal
    // IDs by first occurrence, so looking up the mapping is essential. Keep the
    // original formatted label file for persistence and label-aware pruning;
    // only the transient per-label construction inputs exclude this label.
    if (!skip_label_build.empty())
    {
        if (!universal_label.empty())
            throw std::invalid_argument("Skipped-label construction does not support a universal label");
        std::ifstream mapping(labels_map_file);
        if (!mapping)
            throw std::runtime_error("Cannot open original-to-internal label mapping");
        std::string line, internal_label;
        while (std::getline(mapping, line))
        {
            auto separator = line.find('\t');
            if (separator != std::string::npos && line.substr(0, separator) == skip_label_build)
            {
                internal_label = line.substr(separator + 1);
                break;
            }
        }
        if (internal_label.empty() || all_labels.erase(internal_label) != 1)
            throw std::invalid_argument("Skipped original label does not exist");
        if (all_labels.empty())
            throw std::invalid_argument("Cannot omit every label graph");
        for (auto &labels : point_ids_to_labels)
            labels.erase(internal_label);
        labels_to_number_of_points.erase(internal_label);
        std::cout << "Omitting unqueried original label " << skip_label_build
                  << " (internal " << internal_label << ") from graph construction only" << std::endl;
    }

    // 3. for each label, make a separate data file
    tsl::robin_map<std::string, std::vector<uint32_t>> label_id_to_orig_id_map;
    uint32_t total_number_of_points = (uint32_t)point_ids_to_labels.size();
    if (global_entry_point >= total_number_of_points)
        throw std::invalid_argument("Global starting point is outside the base corpus");

#ifndef _WINDOWS
    if (data_type == "uint8")
        label_id_to_orig_id_map = diskann::generate_label_specific_vector_files_f32<uint8_t>(
            input_data_path, labels_to_number_of_points, point_ids_to_labels, all_labels);
    else if (data_type == "int8")
        label_id_to_orig_id_map = diskann::generate_label_specific_vector_files_f32<int8_t>(
            input_data_path, labels_to_number_of_points, point_ids_to_labels, all_labels);
    else if (data_type == "float")
        label_id_to_orig_id_map = diskann::generate_label_specific_vector_files_f32<float>(
            input_data_path, labels_to_number_of_points, point_ids_to_labels, all_labels);
    else
        throw;
#else
    if (data_type == "uint8")
        label_id_to_orig_id_map = diskann::generate_label_specific_vector_files_compat<uint8_t>(
            input_data_path, labels_to_number_of_points, point_ids_to_labels, all_labels);
    else if (data_type == "int8")
        label_id_to_orig_id_map = diskann::generate_label_specific_vector_files_compat<int8_t>(
            input_data_path, labels_to_number_of_points, point_ids_to_labels, all_labels);
    else if (data_type == "float")
        label_id_to_orig_id_map = diskann::generate_label_specific_vector_files_compat<float>(
            input_data_path, labels_to_number_of_points, point_ids_to_labels, all_labels);
    else
        throw;
#endif

    // 4. for each created data file, create a vanilla diskANN index
    if (data_type == "uint8")
        diskann::generate_label_indices<uint8_t>(input_data_path, final_index_path_prefix, all_labels, R, L, alpha,
                                                 num_threads);
    else if (data_type == "int8")
        diskann::generate_label_indices<int8_t>(input_data_path, final_index_path_prefix, all_labels, R, L, alpha,
                                                num_threads);
    else if (data_type == "float")
        diskann::generate_label_indices<float>(input_data_path, final_index_path_prefix, all_labels, R, L, alpha,
                                               num_threads);
    else
        throw;

    // 5. "stitch" the indices together
    std::vector<std::vector<uint32_t>> stitched_graph;
    tsl::robin_map<std::string, uint32_t> label_entry_points;
    uint64_t stitched_graph_size;

    if (data_type == "uint8")
        std::tie(stitched_graph, stitched_graph_size) =
            stitch_label_indices<uint8_t>(final_index_path_prefix, total_number_of_points, all_labels,
                                          labels_to_number_of_points, label_entry_points, label_id_to_orig_id_map);
    else if (data_type == "int8")
        std::tie(stitched_graph, stitched_graph_size) =
            stitch_label_indices<int8_t>(final_index_path_prefix, total_number_of_points, all_labels,
                                         labels_to_number_of_points, label_entry_points, label_id_to_orig_id_map);
    else if (data_type == "float")
        std::tie(stitched_graph, stitched_graph_size) =
            stitch_label_indices<float>(final_index_path_prefix, total_number_of_points, all_labels,
                                        labels_to_number_of_points, label_entry_points, label_id_to_orig_id_map);
    else
        throw;
    path full_index_path_prefix = final_index_path_prefix + "_full";
    // 5a. save the stitched graph to disk
    save_full_index(full_index_path_prefix, input_data_path, stitched_graph_size, stitched_graph, label_entry_points,
                    universal_label, labels_file_to_use, global_entry_point);

    // The pruner reloads the serialized graph. Release construction-only data
    // first; retaining multiple 16M-node graph copies can exhaust host RAM.
    stitched_graph.clear();
    stitched_graph.shrink_to_fit();
    point_ids_to_labels.clear();
    point_ids_to_labels.shrink_to_fit();
    label_id_to_orig_id_map.clear();

    // 6. run a prune on the stitched index, and save to disk
    if (data_type == "uint8")
        prune_and_save<uint8_t>(final_index_path_prefix, full_index_path_prefix, input_data_path,
                                stitched_R, label_entry_points, universal_label, labels_file_to_use, num_threads);
    else if (data_type == "int8")
        prune_and_save<int8_t>(final_index_path_prefix, full_index_path_prefix, input_data_path,
                               stitched_R, label_entry_points, universal_label, labels_file_to_use, num_threads);
    else if (data_type == "float")
        prune_and_save<float>(final_index_path_prefix, full_index_path_prefix, input_data_path,
                              stitched_R, label_entry_points, universal_label, labels_file_to_use, num_threads);
    else
        throw;

    std::chrono::duration<double> index_time = std::chrono::high_resolution_clock::now() - index_timer;
    std::cout << "pruned/stitched graph generated in " << index_time.count() << " seconds" << std::endl;

    clean_up_artifacts(input_data_path, final_index_path_prefix, all_labels);
}
