/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Strict filtered traversal for graphs built to preserve label connectivity.

use diskann_utils::future::SendFuture;
use thiserror::Error;

use super::{Knn, Search, record::SearchRecord};
use crate::{
    ANNError, ANNResult,
    error::IntoANNResult,
    graph::{
        glue::{self, FilteredAccessor, SearchPostProcess, SearchStrategy},
        index::{DiskANNIndex, SearchStats},
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::Neighbor,
    provider::DataProvider,
};

/// Experimental strict single-hop search.
///
/// Only matching neighbors enter the frontier or have their distance computed. The
/// graph must preserve connectivity within the requested label subset, and every
/// starting point must match. An ordinary unfiltered graph may give poor recall.
/// Expanded candidates and the remaining frontier are passed to the post-processor
/// in `(distance, ID)` order, including expanded candidates evicted from the
/// bounded frontier. Quantized providers must rerank using exact distances.
/// A post-processor may omit candidates it cannot load within its I/O budget.
pub struct SingleHopFilterSearch<'s, I: Eq> {
    /// Base graph search parameters.
    pub inner: Knn,
    starting_points: &'s [I],
    start_selection: StartSelection<I>,
    rerank_frontier: bool,
    recorder: Option<&'s mut dyn SearchRecord<I>>,
}

impl<I: Eq + std::fmt::Debug> std::fmt::Debug for SingleHopFilterSearch<'_, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SingleHopFilterSearch")
            .field("inner", &self.inner)
            .field("starting_points", &self.starting_points)
            .field("start_selection", &self.start_selection)
            .field("rerank_frontier", &self.rerank_frontier)
            .finish_non_exhaustive()
    }
}

impl<'s, I: Eq> SingleHopFilterSearch<'s, I> {
    /// Use caller-owned matching dataset IDs as entry points. An empty slice
    /// returns no results; duplicates share one frontier slot. IDs must name
    /// initialized vectors and remain valid for the duration of the query.
    pub fn new(inner: Knn, starting_points: &'s [I]) -> Self {
        Self {
            inner,
            starting_points,
            start_selection: StartSelection::All,
            rerank_frontier: false,
            recorder: None,
        }
    }

    /// Choose which of the supplied starting points seed the frontier.
    pub fn with_start_selection(mut self, start_selection: StartSelection<I>) -> Self {
        self.start_selection = start_selection;
        self
    }

    /// Record nodes selected for expansion, with the current hop and comparison counts.
    /// A provider may truncate the selected beam when an I/O budget is exhausted.
    pub fn with_recorder(mut self, recorder: &'s mut dyn SearchRecord<I>) -> Self {
        self.recorder = Some(recorder);
        self
    }

    /// Rerank only the final best `L` approximate candidates.
    ///
    /// This ablation preserves the original prototype's candidate selection. It
    /// can lose an exact nearest neighbor that was expanded and later evicted.
    pub fn with_frontier_reranking(mut self) -> Self {
        self.rerank_frontier = true;
        self
    }
}

/// Which supplied starting points seed a [`SingleHopFilterSearch`] frontier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StartSelection<I> {
    /// Seed every supplied starting point.
    #[default]
    All,
    /// Score the whole pool with the search accessor's (possibly approximate)
    /// distance and seed only the nearest point, breaking ties by lower ID.
    ///
    /// `fallback`, which must belong to the pool, is seeded as well. Every pool
    /// member must still match the filter, and each distinct member counts as one
    /// comparison. Distances are computed with the accessor's start-point routine,
    /// so a disk accessor scores the pool from in-memory PQ codes without I/O.
    Nearest {
        /// An additional pool member to seed alongside the nearest one.
        fallback: Option<I>,
    },
}

#[derive(Debug, Error)]
#[error("single-hop filtered search requires every starting point to match the filter")]
struct RejectedStartingPoint;

#[derive(Debug, Error)]
#[error("fallback starting point must belong to the pool")]
struct FallbackNotInPool;

impl<'a, DP, S, T> Search<'a, DP, S, T> for SingleHopFilterSearch<'_, DP::InternalId>
where
    DP: DataProvider,
    S: SearchStrategy<'a, DP, T, SearchAccessor: FilteredAccessor>,
    T: Copy + Send + Sync,
{
    type Output = SearchStats;

    fn search<O, PP, OB>(
        mut self,
        index: &'a DiskANNIndex<DP>,
        strategy: &'a S,
        processor: PP,
        context: &'a DP::Context,
        query: T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        O: Send,
        PP: SearchPostProcess<S::SearchAccessor, T, O> + Send + Sync,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context, query)
                .into_ann_result()?;
            // Strict entry points are ordinary matching dataset IDs, not frozen
            // build points. They share the L frontier slots with other candidates.
            let mut scratch = index.search_scratch(self.inner.l_value().get(), 0);
            let scratch = &mut *scratch;
            let mut rejected_start = false;
            match self.start_selection {
                StartSelection::All => {
                    accessor
                        .start_point_distances_from(self.starting_points, |decision, distance| {
                            if let glue::Decision::Accept(id) = decision {
                                let id = id.into_inner();
                                if scratch.visited.insert(id) {
                                    scratch
                                        .best
                                        .insert_with_id_ties(Neighbor::new(id, distance));
                                    scratch.cmps += 1;
                                }
                            } else {
                                rejected_start = true;
                            }
                        })
                        .await?;
                }
                StartSelection::Nearest { fallback } => {
                    // Validate before scoring: an invalid pool must not be evaluated.
                    if fallback.is_some_and(|id| !self.starting_points.contains(&id)) {
                        return Err(ANNError::new(FallbackNotInPool));
                    }
                    let mut pool = Vec::with_capacity(self.starting_points.len());
                    accessor
                        .start_point_distances_from(self.starting_points, |decision, distance| {
                            if let glue::Decision::Accept(id) = decision {
                                pool.push(Neighbor::new(id.into_inner(), distance));
                            } else {
                                rejected_start = true;
                            }
                        })
                        .await?;
                    if !rejected_start {
                        pool.sort_unstable_by_key(|n| *n.id());
                        pool.dedup_by_key(|n| *n.id());
                        scratch.cmps += pool.len() as u32;
                        let nearest = pool
                            .iter()
                            .min_by(|a, b| crate::neighbor::ord::fast_distance_total(a, b));
                        let fallback = fallback.and_then(|id| pool.iter().find(|n| *n.id() == id));
                        for start in nearest.into_iter().chain(fallback) {
                            if scratch.visited.insert(*start.id()) {
                                scratch.best.insert_with_id_ties(*start);
                            }
                        }
                    }
                }
            }
            if rejected_start {
                return Err(ANNError::new(RejectedStartingPoint));
            }

            let mut neighbors = Vec::with_capacity(index.max_degree_with_slack());
            // This does not widen the traversal queue. Each expanded ID is retained
            // once, so the additional storage is O(hops), like a disk accessor's
            // full-precision distance cache.
            let mut expanded =
                (!self.rerank_frontier).then(|| Vec::with_capacity(self.inner.l_value().get()));
            while scratch.best.has_notvisited_node() && !accessor.terminate_early() {
                scratch.beam_nodes.clear();
                while scratch.beam_nodes.len() < self.inner.beam_width().get()
                    && let Some(node) = scratch.best.closest_notvisited()
                {
                    scratch.beam_nodes.push(*node.id());
                    if let Some(recorder) = &mut self.recorder {
                        recorder.record(node, scratch.hops, scratch.cmps);
                    }
                    if let Some(expanded) = &mut expanded {
                        expanded.push(node);
                    }
                }
                neighbors.clear();
                accessor
                    .expand_beam_accept_only(
                        scratch.beam_nodes.iter().copied(),
                        glue::NotInMut::new(&mut scratch.visited),
                        |id, distance| neighbors.push(Neighbor::new(id.into_inner(), distance)),
                    )
                    .await?;
                for neighbor in &neighbors {
                    scratch.best.insert_with_id_ties(*neighbor);
                }
                scratch.cmps += neighbors.len() as u32;
                scratch.hops += scratch.beam_nodes.len() as u32;
            }
            let result_count = if let Some(mut expanded) = expanded {
                // Keep discovered candidates on early termination as well. Sorting by
                // ID first removes overlap without allocating a second visited set.
                Extend::extend(&mut expanded, scratch.best.iter());
                expanded.sort_unstable_by_key(|n| *n.id());
                expanded.dedup_by_key(|n| *n.id());
                expanded.sort_unstable_by(crate::neighbor::ord::fast_distance_total);
                processor
                    .post_process(&mut accessor, query, expanded.into_iter(), output)
                    .await
                    .into_ann_result()?
            } else {
                processor
                    .post_process(
                        &mut accessor,
                        query,
                        scratch.best.iter().take(self.inner.l_value().get()),
                        output,
                    )
                    .await
                    .into_ann_result()?
            };
            Ok(SearchStats {
                cmps: scratch.cmps,
                hops: scratch.hops,
                range_search_second_round: false,
                result_count: result_count as u32,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::{
        graph::{
            AdjacencyList,
            ext::labeled::{Filtered, QueryLabelProvider},
            test::provider,
        },
        test::tokio::current_thread_runtime,
    };
    use diskann_vector::distance::Metric;

    #[derive(Debug, Default)]
    struct Trace {
        expanded: Vec<u32>,
        candidates: Vec<u32>,
        stop_after: Option<usize>,
    }

    struct ApproximateStrategy<'a>(&'a Mutex<Trace>, fn(u32) -> f32);

    struct ApproximateAccessor<'a> {
        inner: provider::Accessor<'a>,
        trace: &'a Mutex<Trace>,
        distance: fn(u32) -> f32,
    }

    impl crate::provider::HasId for ApproximateAccessor<'_> {
        type Id = u32;
    }

    fn approximate_distance(id: u32) -> f32 {
        match id {
            10 => 100.0,
            0 => 30.0,
            2 => 20.0,
            4 => 10.0,
            6 => 5.0,
            _ => panic!("a rejected vector must not reach distance evaluation"),
        }
    }

    impl glue::SearchAccessor for ApproximateAccessor<'_> {
        fn terminate_early(&mut self) -> bool {
            let trace = self.trace.lock().unwrap();
            trace
                .stop_after
                .is_some_and(|limit| trace.expanded.len() >= limit)
        }

        async fn start_point_distances_from<F>(&mut self, ids: &[u32], mut f: F) -> ANNResult<()>
        where
            F: FnMut(u32, f32) + Send,
        {
            let distance = self.distance;
            glue::SearchAccessor::start_point_distances_from(&mut self.inner, ids, |id, _| {
                f(id, distance(id));
            })
            .await
        }

        async fn starting_points(&self) -> ANNResult<Vec<u32>> {
            glue::SearchAccessor::starting_points(&self.inner).await
        }

        async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
        where
            F: FnMut(u32, f32) + Send,
        {
            let distance = self.distance;
            glue::SearchAccessor::start_point_distances(&mut self.inner, |id, _| {
                f(id, distance(id));
            })
            .await
        }

        async fn expand_beam<I, P, F>(&mut self, ids: I, pred: P, mut f: F) -> ANNResult<()>
        where
            I: Iterator<Item = u32> + Send,
            P: glue::HybridPredicate<u32> + Send + Sync,
            F: FnMut(u32, f32) + Send,
        {
            let ids: Vec<_> = ids.collect();
            self.trace.lock().unwrap().expanded.extend(&ids);
            let distance = self.distance;
            glue::SearchAccessor::expand_beam(&mut self.inner, ids.into_iter(), pred, |id, _| {
                f(id, distance(id));
            })
            .await
        }
    }

    impl<'a> SearchStrategy<'a, provider::Provider, &'a [f32]> for ApproximateStrategy<'_> {
        type SearchAccessorError = provider::DimMismatch;
        type SearchAccessor = ApproximateAccessor<'a>;

        fn search_accessor(
            &'a self,
            provider: &'a provider::Provider,
            _context: &'a provider::Context,
            query: &'a [f32],
        ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
            Ok(ApproximateAccessor {
                inner: provider::Accessor::new(provider, query)?,
                trace: self.0,
                distance: self.1,
            })
        }
    }

    struct ExactRerank<'a>(&'a Mutex<Trace>);

    impl<A> SearchPostProcess<A, &[f32], u32> for ExactRerank<'_>
    where
        A: crate::provider::HasId<Id = u32>,
    {
        type Error = std::convert::Infallible;

        fn post_process<I, B>(
            &self,
            _accessor: &mut A,
            _query: &[f32],
            candidates: I,
            output: &mut B,
        ) -> impl std::future::Future<Output = Result<usize, Self::Error>> + Send
        where
            I: Iterator<Item = Neighbor<u32>> + Send,
            B: SearchOutputBuffer<u32> + Send + ?Sized,
        {
            let mut candidates: Vec<_> = candidates.collect();
            self.0.lock().unwrap().candidates = candidates.iter().map(|n| *n.id()).collect();
            for candidate in &mut candidates {
                let id = *candidate.id();
                *candidate = Neighbor::new(id, (id as f32).powi(2));
            }
            candidates.sort_unstable_by(crate::neighbor::ord::fast_distance);
            std::future::ready(Ok(output.extend(candidates)))
        }
    }

    fn approximate_graph() -> DiskANNIndex<provider::Provider> {
        let config =
            provider::Config::new(Metric::L2, 4, provider::StartPoint::new(10, vec![10.0]))
                .unwrap();
        let provider = provider::Provider::new_from(
            config,
            [(10, AdjacencyList::from_iter_untrusted([0, 1]))],
            [
                (0, vec![0.0], AdjacencyList::from_iter_untrusted([2])),
                (1, vec![0.1], AdjacencyList::from_iter_untrusted([8])),
                (2, vec![2.0], AdjacencyList::from_iter_untrusted([4])),
                (4, vec![4.0], AdjacencyList::from_iter_untrusted([6])),
                (6, vec![6.0], AdjacencyList::new()),
                (8, vec![0.01], AdjacencyList::new()),
            ],
        )
        .unwrap();
        let config = crate::graph::config::Builder::new(
            4,
            crate::graph::config::MaxDegree::Same,
            10,
            Metric::L2.into(),
        )
        .build()
        .unwrap();
        DiskANNIndex::new(config, provider, None)
    }

    #[test]
    fn exact_reranking_retains_expanded_winner_evicted_from_pq_frontier() {
        let index = approximate_graph();
        let trace = Mutex::new(Trace::default());
        let mut results = Vec::<Neighbor<u32>>::new();
        let stats = current_thread_runtime()
            .block_on(index.search_with(
                SingleHopFilterSearch::new(Knn::new_default(2).unwrap(), &[10]),
                &Filtered::new(ApproximateStrategy(&trace, approximate_distance), &Evens),
                ExactRerank(&trace),
                &provider::Context::new(),
                &[0.0][..],
                &mut results,
            ))
            .unwrap();
        assert_eq!(*results[0].id(), 0, "the exact winner was already expanded");
        assert_eq!(trace.lock().unwrap().expanded, [10, 0, 2, 4, 6]);
        assert_eq!(trace.lock().unwrap().candidates, [6, 4, 2, 0, 10]);
        assert_eq!(stats.cmps, 5);
        assert_eq!(stats.hops, 5);
        assert_eq!(index.provider().metrics().get_vector, 5);

        let frontier_index = approximate_graph();
        let frontier_trace = Mutex::new(Trace::default());
        let mut frontier_results = Vec::<Neighbor<u32>>::new();
        let frontier_stats = current_thread_runtime()
            .block_on(
                frontier_index.search_with(
                    SingleHopFilterSearch::new(Knn::new_default(2).unwrap(), &[10])
                        .with_frontier_reranking(),
                    &Filtered::new(
                        ApproximateStrategy(&frontier_trace, approximate_distance),
                        &Evens,
                    ),
                    ExactRerank(&frontier_trace),
                    &provider::Context::new(),
                    &[0.0][..],
                    &mut frontier_results,
                ),
            )
            .unwrap();
        assert_eq!(*frontier_results[0].id(), 4);
        assert_eq!(frontier_trace.lock().unwrap().candidates, [6, 4]);
        assert_eq!(
            frontier_trace.lock().unwrap().expanded,
            trace.lock().unwrap().expanded,
            "retaining expanded candidates must not widen or change traversal",
        );
        assert_eq!(frontier_stats.cmps, stats.cmps);
        assert_eq!(frontier_stats.hops, stats.hops);
        assert_eq!(frontier_index.provider().metrics().get_vector, 5);
    }

    #[test]
    fn dataset_entry_points_do_not_increase_strict_frontier_capacity() {
        for params in [
            SingleHopFilterSearch::new(Knn::new_default(1).unwrap(), &[10]),
            SingleHopFilterSearch::new(Knn::new_default(1).unwrap(), &[10])
                .with_frontier_reranking(),
        ] {
            let index = graph(10);
            let trace = Mutex::new(Trace::default());
            let mut results = Vec::<Neighbor<u32>>::new();
            let stats = current_thread_runtime()
                .block_on(index.search_with(
                    params,
                    &Filtered::new(ApproximateStrategy(&trace, approximate_distance), &Evens),
                    ExactRerank(&trace),
                    &provider::Context::new(),
                    &[0.0][..],
                    &mut results,
                ))
                .unwrap();
            assert_eq!(trace.lock().unwrap().expanded, [10, 2]);
            assert_eq!(stats.hops, 2, "the lower-PQ-ranked ID 0 must be evicted");
            assert_eq!(stats.cmps, 3);
        }
    }

    #[test]
    fn equal_pq_distances_expand_lower_dataset_id_first() {
        let tied_distance = |id| match id {
            10 => 100.0,
            0 | 2 => 1.0,
            _ => panic!("rejected neighbors must not reach PQ evaluation"),
        };
        for params in [
            SingleHopFilterSearch::new(Knn::new_default(1).unwrap(), &[10]),
            SingleHopFilterSearch::new(Knn::new_default(1).unwrap(), &[10])
                .with_frontier_reranking(),
        ] {
            let index = graph(10);
            let trace = Mutex::new(Trace::default());
            let mut results = Vec::<Neighbor<u32>>::new();
            let stats = current_thread_runtime()
                .block_on(index.search_with(
                    params,
                    &Filtered::new(ApproximateStrategy(&trace, tied_distance), &Evens),
                    ExactRerank(&trace),
                    &provider::Context::new(),
                    &[0.0][..],
                    &mut results,
                ))
                .unwrap();
            assert_eq!(trace.lock().unwrap().expanded, [10, 0]);
            assert_eq!(*results[0].id(), 0);
            assert_eq!(stats.hops, 2);
            assert_eq!(stats.cmps, 3);
        }
    }

    #[derive(Debug)]
    struct Evens;

    impl QueryLabelProvider<u32> for Evens {
        fn is_match(&self, id: u32) -> bool {
            id.is_multiple_of(2)
        }
    }

    fn graph(start_id: u32) -> DiskANNIndex<provider::Provider> {
        let config = provider::Config::new(
            Metric::L2,
            4,
            provider::StartPoint::new(start_id, vec![10.0]),
        )
        .unwrap();
        let provider = provider::Provider::new_from(
            config,
            [(start_id, AdjacencyList::from_iter_untrusted([0, 1, 2]))],
            [
                (0, vec![0.0], AdjacencyList::from_iter_untrusted([2])),
                // A rejected neighbor has a closer matching child. Strict search must
                // neither compute the rejected vector's distance nor traverse this edge.
                (1, vec![0.1], AdjacencyList::from_iter_untrusted([4])),
                (2, vec![2.0], AdjacencyList::from_iter_untrusted([0])),
                (4, vec![0.01], AdjacencyList::new()),
            ],
        )
        .unwrap();
        let config = crate::graph::config::Builder::new(
            4,
            crate::graph::config::MaxDegree::Same,
            10,
            Metric::L2.into(),
        )
        .build()
        .unwrap();
        DiskANNIndex::new(config, provider, None)
    }

    #[test]
    fn strict_search_does_not_cross_rejected_neighbors() {
        let index = graph(10);
        let mut results = Vec::<Neighbor<u32>>::new();
        let stats = current_thread_runtime()
            .block_on(index.search_with(
                SingleHopFilterSearch::new(Knn::new_default(10).unwrap(), &[10]),
                &Filtered::new(provider::Strategy::new(), &Evens),
                glue::CopyIds,
                &provider::Context::new(),
                &[0.0][..],
                &mut results,
            ))
            .unwrap();
        let mut ids: Vec<_> = results.iter().map(|n| *n.id()).collect();
        ids.sort_unstable();
        assert_eq!(ids, [0, 2, 10]);
        assert_eq!(stats.cmps, 3, "only the start and two accepted distances");
        assert_eq!(
            stats.hops, 3,
            "rejected nodes must never become frontier nodes"
        );
        assert_eq!(index.provider().metrics().get_vector, 3);
    }

    #[test]
    fn mismatching_start_is_an_error() {
        let index = graph(11);
        let mut results = Vec::<Neighbor<u32>>::new();
        let error = current_thread_runtime()
            .block_on(index.search_with(
                SingleHopFilterSearch::new(Knn::new_default(10).unwrap(), &[11]),
                &Filtered::new(provider::Strategy::new(), &Evens),
                glue::CopyIds,
                &provider::Context::new(),
                &[0.0][..],
                &mut results,
            ))
            .unwrap_err();
        assert!(error.to_string().contains("starting point"));
        assert!(results.is_empty());
    }

    #[test]
    fn copy_ids_returns_nearest_candidate_with_batched_expansion() {
        for beam in [1, 4] {
            let index = graph(10);
            let mut ids = [u32::MAX];
            let mut distances = [f32::MAX];
            let mut output =
                crate::graph::search_output_buffer::IdDistance::new(&mut ids, &mut distances);
            let mut recorder = super::super::record::VisitedSearchRecord::new(3);
            let stats = current_thread_runtime()
                .block_on(
                    index.search_with(
                        SingleHopFilterSearch::new(Knn::new(10, Some(beam)).unwrap(), &[10])
                            .with_recorder(&mut recorder),
                        &Filtered::new(provider::Strategy::new(), &Evens),
                        glue::CopyIds,
                        &provider::Context::new(),
                        &[0.0][..],
                        &mut output,
                    ),
                )
                .unwrap();
            assert_eq!(ids, [0]);
            assert_eq!(distances, [0.0]);
            assert_eq!(stats.result_count, 1);
            assert_eq!(recorder.ids().collect::<Vec<_>>(), [10, 0, 2]);
        }
    }

    #[test]
    fn early_stop_keeps_sorted_frontier_without_duplicates() {
        for limit in [0, 1, 2] {
            let index = graph(10);
            let trace = Mutex::new(Trace {
                stop_after: Some(limit),
                ..Default::default()
            });
            let mut results = Vec::<Neighbor<u32>>::new();
            let stats = current_thread_runtime()
                .block_on(index.search_with(
                    SingleHopFilterSearch::new(Knn::new(10, Some(4)).unwrap(), &[10, 10]),
                    &Filtered::new(ApproximateStrategy(&trace, approximate_distance), &Evens),
                    glue::CopyIds,
                    &provider::Context::new(),
                    &[0.0][..],
                    &mut results,
                ))
                .unwrap();
            let ids: Vec<_> = results.iter().map(|n| *n.id()).collect();
            if limit == 0 {
                assert_eq!(ids, [10]);
                assert_eq!(stats.hops, 0);
            } else {
                assert_eq!(ids, [2, 0, 10]);
                assert_eq!(stats.hops, if limit == 1 { 1 } else { 3 });
            }
        }
    }

    #[test]
    fn explicit_starts_bypass_rejected_frozen_point_and_allow_empty_search() {
        let index = graph(11);
        for starts in [&[2, 2][..], &[][..]] {
            let mut results = Vec::<Neighbor<u32>>::new();
            let stats = current_thread_runtime()
                .block_on(index.search_with(
                    SingleHopFilterSearch::new(Knn::new_default(10).unwrap(), starts),
                    &Filtered::new(provider::Strategy::new(), &Evens),
                    glue::CopyIds,
                    &provider::Context::new(),
                    &[0.0][..],
                    &mut results,
                ))
                .unwrap();
            let ids: Vec<_> = results.iter().map(|n| *n.id()).collect();
            assert_eq!(
                ids,
                if starts.is_empty() {
                    vec![]
                } else {
                    vec![0, 2]
                }
            );
            assert_eq!(stats.hops as usize, ids.len());
        }
    }

    fn nearest_search(
        starts: &[u32],
        fallback: Option<u32>,
        distance: fn(u32) -> f32,
    ) -> (ANNResult<SearchStats>, Vec<u32>, Vec<u32>) {
        let index = graph(10);
        let trace = Mutex::new(Trace::default());
        let mut results = Vec::<Neighbor<u32>>::new();
        let stats = current_thread_runtime().block_on(
            index.search_with(
                SingleHopFilterSearch::new(Knn::new(10, Some(1)).unwrap(), starts)
                    .with_start_selection(StartSelection::Nearest { fallback }),
                &Filtered::new(ApproximateStrategy(&trace, distance), &Evens),
                glue::CopyIds,
                &provider::Context::new(),
                &[0.0][..],
                &mut results,
            ),
        );
        let expanded = trace.into_inner().unwrap().expanded;
        (stats, expanded, results.iter().map(|n| *n.id()).collect())
    }

    #[test]
    fn nearest_start_seeds_only_the_nearest_pool_member_and_its_fallback() {
        let (stats, expanded, _) = nearest_search(&[10, 2], None, approximate_distance);
        let stats = stats.unwrap();
        assert_eq!(expanded, [2, 0], "the farther pool member is never seeded");
        assert_eq!(stats.cmps, 3, "both pool members plus one neighbor");
        assert_eq!(stats.hops, 2);

        let (stats, expanded, ids) = nearest_search(&[10, 2], Some(10), approximate_distance);
        let stats = stats.unwrap();
        assert_eq!(expanded, [2, 0, 10]);
        assert_eq!(ids, [2, 0, 10]);
        assert_eq!(stats.cmps, 3);

        // Selecting the nearest point as its own fallback seeds it only once.
        let (stats, expanded, _) = nearest_search(&[10, 2], Some(2), approximate_distance);
        assert_eq!(expanded, [2, 0]);
        assert_eq!(stats.unwrap().cmps, 3);
    }

    #[test]
    fn nearest_start_breaks_ties_by_id_and_counts_duplicates_once() {
        let tied = |id| match id {
            0 | 2 => 1.0,
            _ => panic!("only pool members and their matching neighbors are scored"),
        };
        let (stats, expanded, _) = nearest_search(&[2, 0, 2], None, tied);
        let stats = stats.unwrap();
        assert_eq!(
            expanded,
            [0, 2],
            "ID 0 wins the tie; 2 is found as a neighbor"
        );
        assert_eq!(stats.cmps, 3, "two distinct pool members plus one neighbor");
    }

    #[test]
    fn nearest_start_validates_the_pool() {
        for (starts, fallback) in [(&[2, 0][..], Some(4)), (&[][..], Some(0))] {
            let index = graph(10);
            let error = current_thread_runtime()
                .block_on(
                    index.search_with(
                        SingleHopFilterSearch::new(Knn::new_default(10).unwrap(), starts)
                            .with_start_selection(StartSelection::Nearest { fallback }),
                        &Filtered::new(provider::Strategy::new(), &Evens),
                        glue::CopyIds,
                        &provider::Context::new(),
                        &[0.0][..],
                        &mut Vec::<Neighbor<u32>>::new(),
                    ),
                )
                .unwrap_err();
            assert!(error.to_string().contains("fallback starting point"));
            assert_eq!(
                index.provider().metrics().get_vector,
                0,
                "the pool must not be scored after a fallback error"
            );
        }

        // An unselected pool member must still satisfy the strict filter.
        let (stats, _, _) = nearest_search(&[2, 1], None, |id| id as f32);
        assert!(
            stats
                .unwrap_err()
                .to_string()
                .contains("every starting point")
        );

        let (stats, expanded, ids) = nearest_search(&[], None, approximate_distance);
        assert_eq!(stats.unwrap().cmps, 0);
        assert!(expanded.is_empty() && ids.is_empty());
    }
}
