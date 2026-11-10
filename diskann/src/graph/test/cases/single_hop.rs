/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Single-hop traversal invariants and comparison with the ordinary search loop.

use diskann_vector::distance::Metric;

use super::multihop::{EvenFilter, build_1d_index, setup_grid_index};
use crate::{
    ANNResult,
    graph::{
        self, AdjacencyList, DiskANNIndex,
        ext::labeled,
        glue::{self, FilteredAccessor},
        index::SearchStats,
        search::{Knn, SingleHopFilterSearch, record::VisitedSearchRecord},
        test::provider,
    },
    neighbor::{BackInserter, Neighbor},
    provider::HasId,
    test::{
        TestRoot,
        cmp::{assert_eq_verbose, verbose_eq},
        get_or_save_test_results,
        tokio::current_thread_runtime,
    },
};

#[derive(Debug)]
struct AcceptAll;

impl labeled::QueryLabelProvider<u32> for AcceptAll {
    fn is_match(&self, _: u32) -> bool {
        true
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Baseline {
    results: Vec<(u32, f32)>,
    comparisons: u32,
    hops: u32,
}

verbose_eq!(Baseline {
    results,
    comparisons,
    hops,
});

fn baseline(name: &str, stats: SearchStats, neighbors: &[Neighbor<u32>]) {
    let mut root = TestRoot::new("graph/test/cases/single_hop");
    let mut path = root.path();
    let name = path.push(name);
    let actual = Baseline {
        results: neighbors.iter().map(|n| (*n.id(), *n.distance())).collect(),
        comparisons: stats.cmps,
        hops: stats.hops,
    };
    assert_eq!(actual.results.len(), stats.result_count as usize);
    assert!(!stats.range_search_second_round);
    assert_eq_verbose!(get_or_save_test_results(&name, &actual), actual);
}

fn disconnected_index() -> DiskANNIndex<provider::Provider> {
    build_1d_index(
        10,
        5.0,
        AdjacencyList::from_iter_untrusted([0, 1]),
        vec![
            (0, vec![0.0], AdjacencyList::from_iter_untrusted([2, 10])),
            (1, vec![1.0], AdjacencyList::from_iter_untrusted([6])),
            (2, vec![2.0], AdjacencyList::from_iter_untrusted([4])),
            (4, vec![4.0], AdjacencyList::from_iter_untrusted([0])),
            (6, vec![6.0], AdjacencyList::new()),
        ],
        2,
    )
}

#[test]
fn only_matching_neighbors_are_expanded() {
    let index = disconnected_index();
    let rt = current_thread_runtime();
    let mut neighbors = Vec::new();
    let mut recorder = VisitedSearchRecord::new(10);
    let stats = rt
        .block_on(index.search(
            SingleHopFilterSearch::new(Knn::new(10, Some(2)).unwrap()).with_recorder(&mut recorder),
            &labeled::Filtered::new(provider::Strategy::new(), &EvenFilter),
            &provider::Context::new(),
            &[2.25][..],
            &mut neighbors,
        ))
        .unwrap();
    baseline("only_matching_neighbors_are_expanded", stats, &neighbors);
    assert_eq!(
        neighbors.iter().map(|n| *n.id()).collect::<Vec<_>>(),
        [2, 4, 0]
    );
    assert!(recorder.ids().all(|id| id.is_multiple_of(2)));
    assert_eq!(recorder.visited.len(), stats.hops as usize);
    // Node 6 is matching, but its only path goes through rejected node 1.
    assert!(!recorder.ids().any(|id| id == 6));
}

#[test]
fn all_matching_grid_agrees_with_knn() {
    let index = setup_grid_index(7);
    let rt = current_thread_runtime();
    // Binary fractions keep exact baseline distances independent of SIMD reduction order.
    let query = [3.25, 4.0, 2.75];
    let mut ordinary = [Neighbor::default(); 20];
    let mut filtered = [Neighbor::default(); 20];
    let knn = Knn::new(40, Some(4)).unwrap();
    let ordinary_stats = rt
        .block_on(index.search(
            knn,
            &provider::Strategy::new(),
            &provider::Context::new(),
            &query[..],
            &mut BackInserter::new(&mut ordinary),
        ))
        .unwrap();
    let stats = rt
        .block_on(index.search(
            // Knn reserves one extra slot for this provider's frozen start point.
            // Compare equal queue capacities; single-hop starts share the L slots.
            SingleHopFilterSearch::new(Knn::new(41, Some(4)).unwrap()),
            &labeled::Filtered::new(provider::Strategy::new(), &AcceptAll),
            &provider::Context::new(),
            &query[..],
            &mut BackInserter::new(&mut filtered),
        ))
        .unwrap();
    assert_eq_verbose!(ordinary.as_slice(), filtered.as_slice());
    assert_eq!(ordinary_stats.cmps, stats.cmps);
    assert_eq!(ordinary_stats.hops, stats.hops);
    assert_eq!(stats.result_count, 20);
    baseline("all_matching_grid_agrees_with_knn", stats, &filtered);
}

#[test]
fn non_l2_metrics_agree_with_knn() {
    let rt = current_thread_runtime();
    for metric in [Metric::InnerProduct, Metric::Cosine] {
        let config =
            provider::Config::new(metric, 3, provider::StartPoint::new(10, vec![3.0, 4.0]))
                .unwrap();
        let provider = provider::Provider::new_from(
            config,
            [(10, AdjacencyList::from_iter_untrusted([0, 1, 2]))],
            [
                (0, vec![1.0, 0.0], AdjacencyList::new()),
                (1, vec![0.0, 1.0], AdjacencyList::new()),
                (2, vec![3.0, 4.0], AdjacencyList::new()),
            ],
        )
        .unwrap();
        let config =
            graph::config::Builder::new(3, graph::config::MaxDegree::Same, 10, metric.into())
                .build()
                .unwrap();
        let index = DiskANNIndex::new(config, provider, None);
        let mut ordinary = Vec::new();
        let mut filtered = Vec::new();
        let knn = Knn::new_default(10).unwrap();
        rt.block_on(index.search(
            knn,
            &provider::Strategy::new(),
            &provider::Context::new(),
            &[1.0, 0.0][..],
            &mut ordinary,
        ))
        .unwrap();
        let stats = rt
            .block_on(index.search(
                SingleHopFilterSearch::new(knn),
                &labeled::Filtered::new(provider::Strategy::new(), &AcceptAll),
                &provider::Context::new(),
                &[1.0, 0.0][..],
                &mut filtered,
            ))
            .unwrap();
        assert_eq_verbose!(ordinary.as_slice(), filtered.as_slice());
        assert_eq!(stats.result_count, 3);
        baseline(&format!("metric_{metric:?}"), stats, &filtered);
    }
}

// Override only the provider's start-point answers and early-stop signal. Distance
// and adjacency access still use the common test provider and labeled adapter.
struct StartPoints {
    values: Vec<(glue::Decision<u32>, f32)>,
    stop: bool,
}

struct Accessor<'a> {
    inner: labeled::FilteredAccessor<'a, provider::Accessor<'a>>,
    settings: &'a StartPoints,
}

impl<'a> glue::SearchStrategy<'a, provider::Provider, &'a [f32]> for StartPoints {
    type SearchAccessor = Accessor<'a>;
    type SearchAccessorError = provider::DimMismatch;

    fn search_accessor(
        &'a self,
        provider: &'a provider::Provider,
        _: &'a provider::Context,
        query: &'a [f32],
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        Ok(Accessor {
            inner: labeled::FilteredAccessor::new(
                provider::Accessor::new(provider, query)?,
                &EvenFilter,
            ),
            settings: self,
        })
    }
}

impl HasId for Accessor<'_> {
    type Id = u32;
}

impl FilteredAccessor for Accessor<'_> {
    fn start_point_distances<F>(&mut self, mut f: F) -> impl Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(glue::Decision<u32>, f32) + Send,
    {
        for &(id, distance) in &self.settings.values {
            f(id, distance);
        }
        std::future::ready(Ok(()))
    }

    fn num_starting_points(&self) -> impl Future<Output = ANNResult<usize>> + Send {
        std::future::ready(Ok(self.settings.values.len()))
    }

    fn expand_beam_filtered<Itr, P, F>(
        &mut self,
        ids: Itr,
        pred: P,
        f: F,
    ) -> impl Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = u32> + Send,
        P: glue::HybridPredicate<u32> + Send + Sync,
        F: FnMut(glue::Decision<u32>, f32) + Send,
    {
        self.inner.expand_beam_filtered(ids, pred, f)
    }

    fn expand_beam_accept_only<Itr, P, F>(
        &mut self,
        ids: Itr,
        pred: P,
        f: F,
    ) -> impl Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = u32> + Send,
        P: glue::Predicate<u32> + glue::PredicateMut<glue::Accept<u32>> + Send + Sync,
        F: FnMut(glue::Accept<u32>, f32) + Send,
    {
        self.inner.expand_beam_accept_only(ids, pred, f)
    }

    fn terminate_early(&mut self) -> bool {
        self.settings.stop
    }
}

#[test]
fn unavailable_start_points_are_empty_results() {
    let index = disconnected_index();
    let rt = current_thread_runtime();
    for values in [
        vec![],
        vec![(glue::Decision::reject(10), 25.0)],
        vec![(glue::Decision::accept(10), f32::NAN)],
    ] {
        let mut results = Vec::<Neighbor<u32>>::new();
        let stats = rt
            .block_on(index.search_with(
                SingleHopFilterSearch::new(Knn::new_default(10).unwrap()),
                &StartPoints {
                    values,
                    stop: false,
                },
                glue::CopyIds,
                &provider::Context::new(),
                &[0.0][..],
                &mut results,
            ))
            .unwrap();
        assert!(results.is_empty());
        assert_eq!(stats.cmps, 0);
        assert_eq!(stats.hops, 0);
    }
}

#[test]
fn duplicate_start_points_and_early_stop() {
    let index = disconnected_index();
    let rt = current_thread_runtime();
    for stop in [false, true] {
        let mut results = Vec::<Neighbor<u32>>::new();
        let mut recorder = VisitedSearchRecord::new(10);
        let stats = rt
            .block_on(
                index.search_with(
                    SingleHopFilterSearch::new(Knn::new_default(10).unwrap())
                        .with_recorder(&mut recorder),
                    &StartPoints {
                        values: vec![(glue::Decision::accept(10), 25.0); 3],
                        stop,
                    },
                    glue::CopyIds,
                    &provider::Context::new(),
                    &[0.0][..],
                    &mut results,
                ),
            )
            .unwrap();
        if stop {
            assert_eq_verbose!(results.as_slice(), [Neighbor::new(10, 25.0)].as_slice());
            assert_eq!(stats.cmps, 1);
            assert_eq!(stats.hops, 0);
        } else {
            assert_eq!(
                results.iter().map(|n| *n.id()).collect::<Vec<_>>(),
                [0, 2, 4, 10]
            );
            assert_eq!(stats.cmps, 4);
            assert_eq!(stats.hops, 4);
        }
        assert_eq!(recorder.visited.len(), stats.hops as usize);
    }
}

#[test]
fn start_points_share_the_bounded_frontier() {
    let index = disconnected_index();
    let rt = current_thread_runtime();
    let mut results = Vec::<Neighbor<u32>>::new();
    let stats = rt
        .block_on(index.search_with(
            SingleHopFilterSearch::new(Knn::new_default(2).unwrap()),
            &StartPoints {
                values: vec![
                    (glue::Decision::accept(0), 0.0),
                    (glue::Decision::accept(2), 4.0),
                    (glue::Decision::accept(4), 16.0),
                    (glue::Decision::accept(6), 36.0),
                ],
                stop: false,
            },
            glue::CopyIds,
            &provider::Context::new(),
            &[0.0][..],
            &mut results,
        ))
        .unwrap();
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [0, 2]);
    assert_eq!(stats.result_count, 2);
    assert_eq!(stats.hops, 2);
}
