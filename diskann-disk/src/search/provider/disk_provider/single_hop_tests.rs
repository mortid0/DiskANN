/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// Singleton labels give the same induced graph in the fixture and the constructed
// metadata, independently of the fixture's ordinary unfiltered adjacency lists.
#[rstest]
#[case(CachingStrategy::None)]
#[case(CachingStrategy::StaticCacheWithBfsNodes(32))]
fn single_hop_uses_constructed_label_entries_and_preserves_other_searches(
    #[case] caching_strategy: CachingStrategy,
) {
    use crate::build::filtered::build_stitched;
    let storage = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));
    let mut searcher = create_disk_index_searcher::<GraphDataF32VectorUnitData>(
        CreateDiskIndexSearcherParams {
            pq_pivot_file_path: TEST_PQ_PIVOT_128DIM,
            pq_compressed_file_path: TEST_PQ_COMPRESSED_128DIM,
            index_path: TEST_INDEX_128DIM,
            index_path_prefix: TEST_INDEX_PREFIX_128DIM,
            caching_strategy,
            ..Default::default()
        },
        &storage,
    );
    let vectors = read_bin::<f32>(&mut storage.open_reader(TEST_DATA_FILE).unwrap()).unwrap();
    let graph = searcher
        .runtime
        .block_on(build_stitched(
            vectors.as_view(),
            [(7, &[0][..]), (9, &[1][..])],
            &Default::default(),
        ))
        .unwrap();
    let metadata = crate::label_metadata::LabelMetadata::from_graph(&graph, &Default::default()).unwrap();
    let query = [0.1; 128];
    let baseline = searcher
        .search(&query, 10, 10, None, SearchMode::graph())
        .unwrap();
    for (label, id) in [(7, 0), (9, 1)] {
        let result = searcher
            .search_single_hop(&query, 10, 10, None, &metadata, label)
            .unwrap();
        assert_eq!(result.stats.result_count, 1);
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0].vertex_id, id);
        assert_eq!(result.stats.cmps, 1);
        assert_eq!(result.stats.query_statistics.total_vertices_loaded, 1);
    }
    let missing = searcher
        .search_single_hop(&query, 10, 10, None, &metadata, 42)
        .unwrap();
    assert!(missing.results.is_empty());
    assert_eq!(missing.stats.cmps, 0);
    assert_eq!(missing.stats.query_statistics.total_vertices_loaded, 0);
    let after = searcher
        .search(&query, 10, 10, None, SearchMode::graph())
        .unwrap();
    assert_eq!(baseline.stats.cmps, after.stats.cmps);
    assert_eq!(
        baseline
            .results
            .iter()
            .map(|r| (r.vertex_id, r.distance))
            .collect::<Vec<_>>(),
        after
            .results
            .iter()
            .map(|r| (r.vertex_id, r.distance))
            .collect::<Vec<_>>()
    );
    assert!(
        searcher
            .search_single_hop(&query[..1], 1, 1, None, &metadata, 7)
            .is_err()
    );
    assert!(
        searcher
            .search_single_hop(&query, 2, 1, None, &metadata, 7)
            .is_err()
    );
    let smaller = searcher
        .runtime
        .block_on(build_stitched(
            rowmajor::Ref::row_vector(vectors.row(0)),
            [(7, &[0][..])],
            &Default::default(),
        ))
        .unwrap();
    let smaller_metadata = crate::label_metadata::LabelMetadata::from_graph(&smaller, &Default::default()).unwrap();
    assert!(
        searcher
            .search_single_hop(&query, 1, 1, None, &smaller_metadata, 7)
            .is_err()
    );
    searcher.index.data_provider.metric = Metric::InnerProduct;
    assert!(
        searcher
            .search_single_hop(&query, 1, 1, None, &metadata, 7)
            .is_err()
    );
}

#[test]
fn single_hop_rerank_respects_io_limit_and_ignores_nan_start_distance() {
    use crate::build::filtered::build_stitched;
    let storage = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));
    let searcher = create_disk_index_searcher::<GraphDataF32VectorUnitData>(
        CreateDiskIndexSearcherParams {
            pq_pivot_file_path: TEST_PQ_PIVOT_128DIM,
            pq_compressed_file_path: TEST_PQ_COMPRESSED_128DIM,
            index_path: TEST_INDEX_128DIM,
            index_path_prefix: TEST_INDEX_PREFIX_128DIM,
            io_limit: 1,
            ..Default::default()
        },
        &storage,
    );
    let vectors = read_bin::<f32>(&mut storage.open_reader(TEST_DATA_FILE).unwrap()).unwrap();
    let graph = searcher
        .runtime
        .block_on(build_stitched(
            vectors.as_view(),
            [(7, &[0][..])],
            &Default::default(),
        ))
        .unwrap();
    let metadata = crate::label_metadata::LabelMetadata::from_graph(&graph, &Default::default()).unwrap();
    drop(graph);
    let result = searcher
        .search_single_hop(&[0.1; 128], 10, 10, Some(4), &metadata, 7)
        .unwrap();
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.stats.query_statistics.total_vertices_loaded, 1);
    let result = searcher
        .search_single_hop(&[f32::NAN; 128], 10, 10, Some(4), &metadata, 7)
        .unwrap();
    assert!(result.results.is_empty());
    assert_eq!(result.stats.query_statistics.total_vertices_loaded, 0);
}

#[test]
fn single_hop_constructed_disk_graph_matches_exact_with_and_without_node_cache() {
    use crate::{
        build::filtered::{StitchedBuildConfig, build_stitched},
        storage::DiskIndexWriter,
    };
    use diskann_providers::{
        model::graph::provider::async_::SimpleNeighborProviderAsync, storage::get_mem_index_file,
    };

    let storage = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));
    let vectors = read_bin::<f32>(&mut storage.open_reader(TEST_DATA_FILE).unwrap()).unwrap();
    let all: Vec<u32> = (0..vectors.nrows() as u32).collect();
    let even: Vec<u32> = all.iter().copied().filter(|id| id % 2 == 0).collect();
    let thirds: Vec<u32> = all.iter().copied().filter(|id| id % 3 == 0).collect();
    let settings = StitchedBuildConfig::new(8, 32, 32, 1.2, 2, 71).unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap();
    let graph = runtime
        .block_on(build_stitched(
            vectors.as_view(),
            [
                (1, all.as_slice()),
                (7, even.as_slice()),
                (13, thirds.as_slice()),
            ],
            &settings,
        ))
        .unwrap();

    let metadata = crate::label_metadata::LabelMetadata::from_graph(&graph, &Default::default()).unwrap();

    // Reuse the canonical graph serializer and disk writer, including the unchanged
    // nonzero-degree invariant. The all-rows label connects every constructed row.
    let prefix = "/stitched-cache-regression";
    let neighbors = SimpleNeighborProviderAsync::new(vectors.nrows(), 0, graph.max_degree(), 1.0);
    for (id, row) in graph.neighbors().iter().enumerate() {
        neighbors.set_neighbors_sync(id, row).unwrap();
    }
    neighbors
        .save_direct(
            storage.as_ref(),
            graph.entry_points()[&1][0],
            &get_mem_index_file(prefix),
        )
        .unwrap();
    DiskIndexWriter::new(TEST_DATA_FILE.into(), prefix.into(), None, 4096)
        .unwrap()
        .create_disk_layout::<GraphDataF32VectorUnitData, _>(storage.as_ref())
        .unwrap();
    let index_path = get_disk_index_file(prefix);
    let query = vectors.row(17);
    let distance = f32::distance(Metric::L2, Some(vectors.ncols()));
    for cache in [
        CachingStrategy::None,
        CachingStrategy::StaticCacheWithBfsNodes(64),
    ] {
        let searcher = create_disk_index_searcher::<GraphDataF32VectorUnitData>(
            CreateDiskIndexSearcherParams {
                pq_pivot_file_path: TEST_PQ_PIVOT_128DIM,
                pq_compressed_file_path: TEST_PQ_COMPRESSED_128DIM,
                index_path: &index_path,
                index_path_prefix: prefix,
                caching_strategy: cache,
                ..Default::default()
            },
            &storage,
        );
        for label in [1, 7, 13] {
            let result = searcher
                .search_single_hop(query, 10, 256, Some(4), &metadata, label)
                .unwrap();
            let mut expected: Vec<_> = graph
                .posting(label)
                .iter()
                .map(|&id| {
                    (
                        id,
                        distance.evaluate_similarity(query, vectors.row(id as usize)),
                    )
                })
                .collect();
            expected.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            expected.truncate(10);
            let actual: Vec<_> = result
                .results
                .iter()
                .map(|r| (r.vertex_id, r.distance))
                .collect();
            assert_eq!(actual, expected);
            assert_eq!(result.stats.cmps as usize, graph.posting(label).len());
        }
    }
    let limited = create_disk_index_searcher::<GraphDataF32VectorUnitData>(
        CreateDiskIndexSearcherParams {
            pq_pivot_file_path: TEST_PQ_PIVOT_128DIM,
            pq_compressed_file_path: TEST_PQ_COMPRESSED_128DIM,
            index_path: &index_path,
            index_path_prefix: prefix,
            io_limit: 2,
            ..Default::default()
        },
        &storage,
    );
    let result = limited
        .search_single_hop(query, 10, 32, Some(4), &metadata, 7)
        .unwrap();
    assert_eq!(
        result.results.len(),
        2,
        "reranking omits unexpanded PQ candidates"
    );
    assert_eq!(result.stats.query_statistics.total_vertices_loaded, 2);
    assert!(result.stats.cmps > 2);
}
