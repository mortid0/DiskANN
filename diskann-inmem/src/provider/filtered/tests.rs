/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;
use diskann::{
    graph::{
        DiskANNIndex,
        glue::{FilteredAccessor, PruneStrategy},
        search::{Knn, SingleHopFilterSearch},
        test::synthetic::Grid,
    },
    provider::{DataProvider, Delete, NeighborAccessorMut, SetElement},
};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::distance::Metric;

use crate::num::{Capacity, MaxDegree};

const SIDE: usize = 10;
type Index = DiskANNIndex<Provider<repr::Full<f32>, u64>>;

fn external(row: usize) -> u64 {
    10 * row as u64 + 1
}

async fn fixture(metric: Metric) -> (Index, LabelIndex<u32, u64>, rowmajor::Owned<f32>) {
    let grid = Grid::Three;
    let data = grid.data(SIDE);
    let provider = Provider::new(
        repr::full::Config::new(
            Capacity::new(data.nrows() + 64),
            MaxDegree::new(6),
            metric,
            rowmajor::Owned::row_vector(vec![0.0; 3].into()),
        )
        .unwrap(),
    )
    .unwrap();
    for (row, vector) in data.rows().enumerate() {
        provider
            .set_element(&Context, &external(row), vector)
            .await
            .unwrap();
    }
    let mut accessor = super::super::Strategy
        .prune_accessor(&provider, &Context, 6)
        .unwrap();
    for (row, neighbors) in grid.neighbors(SIDE).iter().enumerate() {
        let id = provider.to_internal_id(&Context, &external(row)).unwrap();
        let neighbors = neighbors
            .iter()
            .map(|neighbor| {
                provider
                    .to_internal_id(&Context, &external(*neighbor as usize))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        accessor.set_neighbors(id, &neighbors).await.unwrap();
    }
    drop(accessor);
    let postings = BTreeMap::from([
        (0, (0..data.nrows() / 2).map(external).collect()),
        (1, (data.nrows() / 2..data.nrows()).map(external).collect()),
    ]);
    let entry_points = BTreeMap::from([
        (0, vec![external(0), external(1)]),
        (1, vec![external(data.nrows() / 2)]),
    ]);
    let labels = LabelIndex::new(&provider, &postings, &entry_points).unwrap();
    let config = diskann::graph::config::Builder::new(
        6,
        diskann::graph::config::MaxDegree::new(6),
        32,
        metric.into(),
    )
    .build()
    .unwrap();
    (DiskANNIndex::new(config, provider, None), labels, data)
}

async fn assert_entry_points(index: &Index, strategy: &Strategy<u64>, expected: &[u64]) {
    let mut accessor = glue::SearchStrategy::search_accessor(
        strategy,
        index.provider(),
        &Context,
        [0.0; 3].as_slice(),
    )
    .unwrap();
    assert_eq!(
        accessor.num_starting_points().await.unwrap(),
        expected.len()
    );
    let mut starts = Vec::new();
    accessor
        .start_point_distances(|decision, _| {
            assert!(decision.is_accept());
            starts.push(decision.into_inner());
        })
        .await
        .unwrap();
    assert_eq!(
        starts,
        expected
            .iter()
            .map(|id| index.provider().to_internal_id(&Context, id).unwrap())
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn filtered_grid_matches_exhaustive_l2_and_inner_product() {
    for metric in [Metric::L2, Metric::InnerProduct] {
        let (index, labels, data) = fixture(metric).await;
        for label in [0, 1] {
            for query in [[0.3, 4.2, 3.7], [9.1, 1.4, 6.3]] {
                let mut found = Vec::new();
                index
                    .search(
                        SingleHopFilterSearch::new(Knn::new_default(data.nrows()).unwrap()),
                        &labels.strategy(&label),
                        &Context,
                        query.as_slice(),
                        &mut found,
                    )
                    .await
                    .unwrap();
                let mut expected = data
                    .rows()
                    .enumerate()
                    .filter(|(i, _)| (*i < data.nrows() / 2) == (label == 0))
                    .map(|(i, vector)| {
                        let distance = match metric {
                            Metric::L2 => {
                                vector.iter().zip(query).map(|(x, y)| (x - y).powi(2)).sum()
                            }
                            Metric::InnerProduct => {
                                -vector.iter().zip(query).map(|(x, y)| x * y).sum::<f32>()
                            }
                            _ => unreachable!(),
                        };
                        Neighbor::new(external(i), distance)
                    })
                    .collect::<Vec<_>>();
                expected.sort_by(|a, b| a.distance().total_cmp(b.distance()));
                assert_eq!(found.len(), expected.len());
                // Tied distances may be ordered differently by the SIMD and scalar kernels.
                for (actual, expected) in found.iter().zip(&expected) {
                    assert!((actual.distance() - expected.distance()).abs() < 0.0001);
                }
                assert!(
                    found.iter().all(|n| {
                        ((*n.id() - 1) / 10 < (data.nrows() / 2) as u64) == (label == 0)
                    })
                );
                assert!(
                    found
                        .iter()
                        .any(|n| *n.id() == external(if label == 0 { 0 } else { 500 }))
                );
            }
        }
        let mut found = Vec::<Neighbor<u64>>::new();
        index
            .search(
                SingleHopFilterSearch::new(Knn::new_default(10).unwrap()),
                &labels.strategy(&999),
                &Context,
                [0.0; 3].as_slice(),
                &mut found,
            )
            .await
            .unwrap();
        assert!(found.is_empty());
    }
}

#[tokio::test]
async fn deleted_entry_points_fall_back_to_remaining_members() {
    let (index, labels, _) = fixture(Metric::L2).await;
    index
        .provider()
        .delete(&Context, &external(0))
        .await
        .unwrap();
    let mut found = Vec::new();
    let strategy = labels.strategy(&0);
    let stats = index
        .search(
            SingleHopFilterSearch::new(Knn::new_default(32).unwrap()),
            &strategy,
            &Context,
            [0.0; 3].as_slice(),
            &mut found,
        )
        .await
        .unwrap();
    assert_eq!(stats.result_count as usize, found.len());
    assert!(!found.is_empty());
    assert!(found.iter().all(|n| *n.id() != external(0)));
    index
        .provider()
        .delete(&Context, &external(1))
        .await
        .unwrap();
    found.clear();
    index
        .search(
            SingleHopFilterSearch::new(Knn::new_default(32).unwrap()),
            &strategy,
            &Context,
            [0.0; 3].as_slice(),
            &mut found,
        )
        .await
        .unwrap();
    assert!(!found.is_empty());
    assert!(
        found
            .iter()
            .all(|n| *n.id() != external(0) && *n.id() != external(1))
    );

    assert_entry_points(&index, &strategy, &[external(2)]).await;

    for row in 2..SIDE.pow(3) / 2 {
        index
            .provider()
            .delete(&Context, &external(row))
            .await
            .unwrap();
    }
    assert_entry_points(&index, &strategy, &[]).await;
    found.clear();
    index
        .search(
            SingleHopFilterSearch::new(Knn::new_default(32).unwrap()),
            &strategy,
            &Context,
            [0.0; 3].as_slice(),
            &mut found,
        )
        .await
        .unwrap();
    assert!(found.is_empty());
}

#[tokio::test]
async fn label_index_rejects_invalid_metadata() {
    let (index, _, _) = fixture(Metric::L2).await;
    let provider = index.provider();
    assert!(matches!(
        LabelIndex::new(
            provider,
            &BTreeMap::new(),
            &BTreeMap::from([(0, vec![external(0)])])
        ),
        Err(LabelIndexError::MissingLabel),
    ));
    assert!(matches!(
        LabelIndex::new(
            provider,
            &BTreeMap::from([(0, vec![u64::MAX])]),
            &BTreeMap::new()
        ),
        Err(LabelIndexError::MissingPoint),
    ));
    assert!(matches!(
        LabelIndex::new(
            provider,
            &BTreeMap::from([(0, vec![external(0)])]),
            &BTreeMap::from([(0, vec![external(1)])]),
        ),
        Err(LabelIndexError::RejectedEntryPoint),
    ));
}

#[tokio::test]
async fn fallback_preserves_first_posting_order_and_skips_missing_members() {
    let (index, _, _) = fixture(Metric::L2).await;
    let members = [external(3), external(2), external(3), external(4)];
    let labels = LabelIndex::new(
        index.provider(),
        &BTreeMap::from([(0, members.to_vec())]),
        &BTreeMap::new(),
    )
    .unwrap();
    index
        .provider()
        .delete(&Context, &external(3))
        .await
        .unwrap();
    let strategy = labels.strategy(&0);
    assert_entry_points(&index, &strategy, &[external(2)]).await;

    // The fallback must also skip a live member whose query distance is NaN.
    index
        .provider()
        .delete(&Context, &external(2))
        .await
        .unwrap();
    index
        .provider()
        .set_element(&Context, &external(2), [f32::NAN, 0.0, 0.0].as_slice())
        .await
        .unwrap();
    assert_entry_points(&index, &strategy, &[external(4)]).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_deletes_and_inserts_preserve_external_membership() {
    let (index, labels, _) = fixture(Metric::L2).await;
    let index = Arc::new(index);
    let writer = index.clone();
    let updates = tokio::spawn(async move {
        for row in 2..32 {
            writer
                .provider()
                .delete(&Context, &external(row))
                .await
                .unwrap();
            writer
                .provider()
                .set_element(&Context, &(1_000_000 + row as u64), [0.0; 3].as_slice())
                .await
                .unwrap();
            tokio::task::yield_now().await;
        }
    });
    let strategy = labels.strategy(&0);
    for _ in 0..32 {
        let mut found = Vec::new();
        let stats = index
            .search(
                SingleHopFilterSearch::new(Knn::new_default(64).unwrap()),
                &strategy,
                &Context,
                [0.0; 3].as_slice(),
                &mut found,
            )
            .await
            .unwrap();
        assert_eq!(stats.result_count as usize, found.len());
        assert!(found.iter().all(|n| *n.id() < external(500)));
        tokio::task::yield_now().await;
    }
    updates.await.unwrap();
}
