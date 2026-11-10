/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Single-hop filtered traversal for graphs built to preserve label connectivity.

use diskann_utils::future::SendFuture;

use super::{
    Knn, Search,
    record::{NoopSearchRecord, SearchRecord},
};
use crate::{
    ANNError, ANNResult,
    error::IntoANNResult,
    graph::{
        glue::{
            self, FilteredAccessor, SearchPostProcess, SearchStrategy, SeedableFilteredAccessor,
        },
        index::{DiskANNIndex, SearchStats},
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::Neighbor,
    provider::DataProvider,
    utils::VectorId,
};

/// A seed is stale or incompatible with the query's filter.
///
/// Search's existing `ANNResult` boundary preserves this type. Callers can use
/// [`ANNError::downcast_ref`] to refresh seeds and retry without matching error text.
#[derive(Debug, thiserror::Error)]
pub enum SeedError<I> {
    /// The seed does not name an initialized, live dataset vector.
    #[error("seed ID {0} is outside the dataset or no longer available")]
    InvalidId(I),
    /// Single-hop traversal cannot begin at a rejected seed.
    #[error("single-hop seed {0} does not match the filter")]
    Rejected(I),
}

impl<I: VectorId> From<SeedError<I>> for ANNError {
    #[track_caller]
    fn from(error: SeedError<I>) -> Self {
        Self::new(error)
    }
}

/// Which supplied seeds enter the bounded search frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedSelection {
    /// Use every distinct seed, sharing the same L slots as discovered candidates.
    All,
    /// Use the closest seed by traversal distance; input order breaks ties.
    Closest,
}

/// Candidates passed to post-processing after single-hop traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultPool {
    /// Only the final bounded frontier. Appropriate for exact traversal distances.
    Frontier,
    /// Expanded nodes and the remaining frontier, ordered by distance.
    /// Retains potential winners when traversal distances are approximate.
    Expanded,
}

/// Single-hop traversal for graphs that preserve connectivity within the filter.
///
/// Only matching neighbors enter the frontier or have their distance computed.
/// Seeds are required internal dataset IDs and must stay live for the query. Every seed must
/// match, including seeds not selected by [`SeedSelection::Closest`]. Ordinary
/// unfiltered graphs can give poor recall. The provider chooses the result pool
/// unless explicitly overridden; post-processing may omit candidates it cannot load.
///
/// A filtered accessor alone is insufficient; the seed capability is a compile-time
/// requirement. Seed distances never pass through the provider start-point hooks.
///
/// ```compile_fail,E0277
/// use diskann::{graph::{glue::{FilteredAccessor, SearchStrategy},
///     search::{Knn, Search, SingleHopFilterSearch}}, provider::DataProvider};
/// fn require_search<'a, DP: DataProvider, S: SearchStrategy<'a, DP, &'a [f32]>>(
///     _: impl Search<'a, DP, S, &'a [f32]>) {}
/// fn unsupported<'a, DP, S>(init_ids: &[DP::InternalId])
/// where DP: DataProvider,
///       S: SearchStrategy<'a, DP, &'a [f32], SearchAccessor: FilteredAccessor>,
/// {
///     require_search::<DP, S>(SingleHopFilterSearch::new(Knn::new_default(10).unwrap(), init_ids));
/// }
/// ```
///
/// Omitting seeds is a compile-time error:
///
/// ```compile_fail,E0061
/// use diskann::graph::search::{Knn, SingleHopFilterSearch};
/// let search = SingleHopFilterSearch::<u32>::new(Knn::new_default(10).unwrap());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SingleHopFilterSearch<'s, I> {
    pub inner: Knn,
    init_ids: &'s [I],
    seed_selection: SeedSelection,
    result_pool: Option<ResultPool>,
}

impl<'s, I> SingleHopFilterSearch<'s, I> {
    pub fn new(inner: Knn, init_ids: &'s [I]) -> Self {
        Self {
            inner,
            init_ids,
            seed_selection: SeedSelection::All,
            result_pool: None,
        }
    }

    pub fn with_seed_selection(mut self, selection: SeedSelection) -> Self {
        self.seed_selection = selection;
        self
    }

    pub fn with_result_pool(mut self, pool: ResultPool) -> Self {
        self.result_pool = Some(pool);
        self
    }

    /// Record nodes selected for expansion. A provider may truncate a selected beam
    /// when its I/O budget is exhausted.
    pub fn with_recorder<SR: ?Sized>(self, recorder: &mut SR) -> RecordedSingleHop<'s, '_, I, SR> {
        RecordedSingleHop {
            inner: self,
            recorder,
        }
    }
}

/// Single-hop search with a statically dispatched traversal recorder.
#[derive(Debug)]
pub struct RecordedSingleHop<'s, 'r, I, SR: ?Sized> {
    pub inner: SingleHopFilterSearch<'s, I>,
    pub recorder: &'r mut SR,
}

impl<'a, DP, S, T> Search<'a, DP, S, T> for SingleHopFilterSearch<'_, DP::InternalId>
where
    DP: DataProvider,
    S: SearchStrategy<'a, DP, T, SearchAccessor: SeedableFilteredAccessor>,
    T: Copy + Send + Sync,
{
    type Output = SearchStats;

    async fn search<O, PP, OB>(
        self,
        index: &'a DiskANNIndex<DP>,
        strategy: &'a S,
        processor: PP,
        context: &'a DP::Context,
        query: T,
        output: &mut OB,
    ) -> ANNResult<Self::Output>
    where
        O: Send,
        PP: SearchPostProcess<S::SearchAccessor, T, O> + Send + Sync,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        self.with_recorder(&mut NoopSearchRecord::new())
            .search(index, strategy, processor, context, query, output)
            .await
    }
}

impl<'a, DP, S, T, SR> Search<'a, DP, S, T> for RecordedSingleHop<'_, '_, DP::InternalId, SR>
where
    DP: DataProvider,
    S: SearchStrategy<'a, DP, T, SearchAccessor: SeedableFilteredAccessor>,
    T: Copy + Send + Sync,
    SR: SearchRecord<DP::InternalId> + ?Sized,
{
    type Output = SearchStats;

    fn search<O, PP, OB>(
        self,
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
            // Seeds are ordinary matching dataset IDs. They share the L frontier slots with other candidates.
            let mut scratch = index.search_scratch(self.inner.inner.l_value().get(), 0);
            let scratch = &mut *scratch;
            let mut rejected_seed = None;
            let mut closest: Option<Neighbor<DP::InternalId>> = None;
            accessor
                .filtered_distances_to_ids(self.inner.init_ids, |decision, distance| {
                    if let glue::Decision::Accept(id) = decision {
                        let id = id.into_inner();
                        if scratch.visited.insert(id) {
                            match self.inner.seed_selection {
                                SeedSelection::All => {
                                    scratch.best.insert(Neighbor::new(id, distance));
                                }
                                SeedSelection::Closest => {
                                    if closest
                                        .as_ref()
                                        .is_none_or(|best| distance < *best.distance())
                                    {
                                        closest = Some(Neighbor::new(id, distance));
                                    }
                                }
                            }
                            scratch.cmps += 1;
                        }
                    } else {
                        rejected_seed = Some(decision.into_inner());
                    }
                })
                .await?;
            if let Some(id) = rejected_seed {
                return Err(SeedError::Rejected(id).into());
            }

            if let Some(seed) = closest {
                // Other seeds remain discoverable through graph traversal.
                scratch.visited.clear();
                scratch.visited.insert(*seed.id());
                scratch.best.insert(seed);
            }

            let mut neighbors = Vec::with_capacity(index.max_degree_with_slack());
            // This does not widen the traversal queue. Each expanded ID is retained
            // once, so the additional storage is O(hops).
            let mut expanded = (self
                .inner
                .result_pool
                .unwrap_or_else(|| accessor.default_result_pool())
                == ResultPool::Expanded)
                .then(|| Vec::with_capacity(self.inner.inner.l_value().get()));
            while scratch.best.has_notvisited_node() && !accessor.terminate_early() {
                scratch.beam_nodes.clear();
                while scratch.beam_nodes.len() < self.inner.inner.beam_width().get()
                    && let Some(node) = scratch.best.closest_notvisited()
                {
                    scratch.beam_nodes.push(*node.id());
                    self.recorder.record(node, scratch.hops, scratch.cmps);
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
                    scratch.best.insert(*neighbor);
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
                expanded.sort_unstable_by(crate::neighbor::ord::fast_distance);
                processor
                    .post_process(&mut accessor, query, expanded.into_iter(), output)
                    .await
                    .into_ann_result()?
            } else {
                processor
                    .post_process(
                        &mut accessor,
                        query,
                        scratch.best.iter().take(self.inner.inner.l_value().get()),
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

    impl glue::SeedableAccessor for ApproximateAccessor<'_> {
        async fn distances_to_ids<F>(&mut self, ids: &[u32], mut f: F) -> ANNResult<()>
        where
            F: FnMut(u32, f32) + Send,
        {
            let distance = self.distance;
            glue::SeedableAccessor::distances_to_ids(&mut self.inner, ids, |id, _| {
                f(id, distance(id));
            })
            .await
        }
        fn default_result_pool(&self) -> ResultPool {
            ResultPool::Expanded
        }
    }

    impl glue::SearchAccessor for ApproximateAccessor<'_> {
        fn terminate_early(&mut self) -> bool {
            let trace = self.trace.lock().unwrap();
            trace
                .stop_after
                .is_some_and(|limit| trace.expanded.len() >= limit)
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
    fn post_processing_retains_expanded_winner_evicted_from_approximate_frontier() {
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
                        .with_result_pool(ResultPool::Frontier),
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
    fn dataset_seeds_do_not_increase_frontier_capacity() {
        for params in [
            SingleHopFilterSearch::new(Knn::new_default(1).unwrap(), &[10]),
            SingleHopFilterSearch::new(Knn::new_default(1).unwrap(), &[10])
                .with_result_pool(ResultPool::Frontier),
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
            assert_eq!(stats.hops, 2, "the lower-ranked ID 0 must be evicted");
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
    fn single_hop_does_not_cross_rejected_neighbors() {
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
        assert_eq!(stats.cmps, 3, "only the seed and two accepted distances");
        assert_eq!(
            stats.hops, 3,
            "rejected nodes must never become frontier nodes"
        );
        assert_eq!(index.provider().metrics().get_vector, 3);
    }

    #[test]
    fn closest_seed_uses_distance_and_preserves_input_order_on_ties() {
        for (init_ids, query, expected) in [
            (vec![4, 0], 1.0, vec![4]),   // disconnected 4 is closer
            (vec![4, 0], 0.005, vec![4]), // exact tie retains first seed
            (vec![0, 4], 0.005, vec![0, 2]),
        ] {
            let index = graph(10);
            let mut results = Vec::<Neighbor<u32>>::new();
            current_thread_runtime()
                .block_on(
                    index.search_with(
                        SingleHopFilterSearch::new(Knn::new_default(10).unwrap(), &init_ids)
                            .with_seed_selection(SeedSelection::Closest),
                        &Filtered::new(provider::Strategy::new(), &Evens),
                        glue::CopyIds,
                        &provider::Context::new(),
                        &[query][..],
                        &mut results,
                    ),
                )
                .unwrap();
            assert_eq!(
                results.iter().map(|n| *n.id()).collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn mismatching_seed_is_an_error() {
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
        assert!(matches!(
            error.downcast_ref::<SeedError<u32>>(),
            Some(SeedError::Rejected(11))
        ));
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
    fn explicit_seeds_bypass_rejected_frozen_point_and_allow_empty_search() {
        let index = graph(11);
        for init_ids in [&[2, 2][..], &[][..]] {
            let mut results = Vec::<Neighbor<u32>>::new();
            let stats = current_thread_runtime()
                .block_on(index.search_with(
                    SingleHopFilterSearch::new(Knn::new_default(10).unwrap(), init_ids),
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
                if init_ids.is_empty() {
                    vec![]
                } else {
                    vec![0, 2]
                }
            );
            assert_eq!(stats.hops as usize, ids.len());
        }
    }

    #[test]
    fn closest_seed_still_validates_every_seed() {
        for seeds in [[0, 1], [1, 0]] {
            let index = graph(10);
            let strategy = Filtered::new(provider::Strategy::new(), &Evens);
            let mut results = Vec::<Neighbor<u32>>::new();
            let error = current_thread_runtime()
                .block_on(
                    index.search_with(
                        SingleHopFilterSearch::new(Knn::new_default(1).unwrap(), &seeds)
                            .with_seed_selection(SeedSelection::Closest),
                        &strategy,
                        glue::CopyIds,
                        &provider::Context::new(),
                        &[0.0][..],
                        &mut results,
                    ),
                )
                .unwrap_err();
            assert!(matches!(
                error.downcast_ref::<SeedError<u32>>(),
                Some(SeedError::Rejected(1))
            ));
            assert!(results.is_empty());
        }
    }

    #[test]
    fn more_seeds_than_l_share_the_bounded_frontier() {
        let index = graph(11);
        let strategy = Filtered::new(provider::Strategy::new(), &Evens);
        let mut recorder = super::super::record::VisitedSearchRecord::new(1);
        let mut results = Vec::<Neighbor<u32>>::new();
        let stats = current_thread_runtime()
            .block_on(
                index.search_with(
                    SingleHopFilterSearch::new(Knn::new_default(1).unwrap(), &[2, 4, 0, 4])
                        .with_recorder(&mut recorder),
                    &strategy,
                    glue::CopyIds,
                    &provider::Context::new(),
                    &[0.0][..],
                    &mut results,
                ),
            )
            .unwrap();
        assert_eq!(recorder.ids().collect::<Vec<_>>(), [0]);
        assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [0]);
        assert_eq!(stats.hops, 1);
    }

    #[test]
    fn provider_default_avoids_retaining_evicted_exact_candidates() {
        let index = graph(10);
        let strategy = Filtered::new(provider::Strategy::new(), &Evens);
        let mut results = Vec::<Neighbor<u32>>::new();
        current_thread_runtime()
            .block_on(index.search_with(
                SingleHopFilterSearch::new(Knn::new_default(1).unwrap(), &[10]),
                &strategy,
                glue::CopyIds,
                &provider::Context::new(),
                &[0.0][..],
                &mut results,
            ))
            .unwrap();
        assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [0]);
    }

    #[test]
    fn approximate_beam_retains_evicted_winner_without_changing_traversal() {
        for pool in [ResultPool::Expanded, ResultPool::Frontier] {
            let index = approximate_graph();
            let trace = Mutex::new(Trace::default());
            let strategy = Filtered::new(ApproximateStrategy(&trace, approximate_distance), &Evens);
            let mut results = Vec::<Neighbor<u32>>::new();
            let stats = current_thread_runtime()
                .block_on(
                    index.search_with(
                        SingleHopFilterSearch::new(Knn::new(2, Some(4)).unwrap(), &[10, 0])
                            .with_result_pool(pool),
                        &strategy,
                        ExactRerank(&trace),
                        &provider::Context::new(),
                        &[0.0][..],
                        &mut results,
                    ),
                )
                .unwrap();
            assert_eq!(trace.lock().unwrap().expanded, [0, 10, 2, 4, 6]);
            assert_eq!(stats.hops, 5);
            assert_eq!(
                *results[0].id(),
                if pool == ResultPool::Expanded { 0 } else { 4 }
            );
        }
    }

    #[test]
    fn invalid_seed_is_typed_and_search_can_retry_with_refreshed_seeds() {
        let index = graph(10);
        let context = provider::Context::new();
        let mut results = Vec::<Neighbor<u32>>::new();
        for (seed, valid) in [(999, false), (4, true)] {
            let init_ids = [seed];
            let strategy = Filtered::new(provider::Strategy::new(), &Evens);
            let result = current_thread_runtime().block_on(index.search(
                SingleHopFilterSearch::new(Knn::new_default(1).unwrap(), &init_ids),
                &strategy,
                &context,
                &[0.0][..],
                &mut results,
            ));
            if valid {
                assert_eq!(result.unwrap().result_count, 1);
                assert_eq!(*results[0].id(), 4);
            } else {
                assert!(matches!(
                    result.unwrap_err().downcast::<SeedError<u32>>(),
                    Ok(SeedError::InvalidId(999))
                ));
                assert!(results.is_empty());
            }
        }
    }

    #[test]
    fn filter_start_points_keeps_dataset_seeds() {
        for pool in [ResultPool::Frontier, ResultPool::Expanded] {
            let index = graph(10);
            let mut results = Vec::<Neighbor<u32>>::new();
            current_thread_runtime()
                .block_on(
                    index.search_with(
                        SingleHopFilterSearch::new(Knn::new_default(10).unwrap(), &[4, 10, 4])
                            .with_result_pool(pool),
                        &Filtered::new(provider::Strategy::new(), &Evens),
                        glue::Pipeline::new(
                            crate::graph::ext::labeled::Unfiltered,
                            glue::Pipeline::new(glue::FilterStartPoints, glue::CopyIds),
                        ),
                        &provider::Context::new(),
                        &[0.01][..],
                        &mut results,
                    ),
                )
                .unwrap();
            assert_eq!(
                results.iter().map(|n| *n.id()).collect::<Vec<_>>(),
                [4, 0, 2]
            );
            assert_eq!(*results[0].distance(), 0.0);
        }
    }

    #[test]
    fn closest_seed_leaves_unselected_seeds_discoverable() {
        let index = graph(10);
        let mut results = Vec::<Neighbor<u32>>::new();
        let mut recorder = super::super::record::VisitedSearchRecord::new(2);
        current_thread_runtime()
            .block_on(
                index.search_with(
                    SingleHopFilterSearch::new(Knn::new_default(10).unwrap(), &[2, 0, 2, 0])
                        .with_seed_selection(SeedSelection::Closest)
                        .with_recorder(&mut recorder),
                    &Filtered::new(provider::Strategy::new(), &Evens),
                    glue::CopyIds,
                    &provider::Context::new(),
                    &[0.0][..],
                    &mut results,
                ),
            )
            .unwrap();
        assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [0, 2]);
        assert_eq!(recorder.ids().collect::<Vec<_>>(), [0, 2]);
    }

    #[test]
    fn paged_search_keeps_seeds_without_repeating_results() {
        current_thread_runtime().block_on(async {
            let index = graph(10);
            let strategy = provider::Strategy::new();
            let context = provider::Context::new();
            for (seeds, expected) in [
                (&[4, 4][..], vec![4]),       // isolated seed is the exact nearest neighbor
                (&[0, 0, 2][..], vec![0, 2]), // cycle and duplicate seeds
                (&[10, 4][..], vec![4, 0, 1, 2]), // frozen point is excluded
                (&[][..], vec![]),            // no fallback to provider start points
            ] {
                let mut search = index
                    .paged_search_with_init_ids(&strategy, &context, &[0.01][..], 10, Some(seeds))
                    .await
                    .unwrap();
                for id in expected {
                    let page = search.next_page(1).await.unwrap();
                    assert_eq!(page.len(), 1);
                    assert_eq!(*page[0].id(), id);
                }
                assert!(search.next_page(1).await.unwrap().is_empty());
                assert!(search.next_page(1).await.unwrap().is_empty());
            }
            // Default paged search continues to use the provider's frozen points.
            let mut search = index
                .paged_search(&strategy, &context, &[0.0][..], 10)
                .await
                .unwrap();
            assert_eq!(*search.next_page(1).await.unwrap()[0].id(), 0);
        });
    }

    #[test]
    fn paged_search_reports_invalid_seed() {
        let index = graph(10);
        let error = current_thread_runtime()
            .block_on(index.paged_search_with_init_ids(
                &provider::Strategy::new(),
                &provider::Context::new(),
                &[0.0][..],
                10,
                Some(&[999]),
            ))
            .unwrap_err();
        assert!(matches!(
            error.downcast_ref::<SeedError<u32>>(),
            Some(SeedError::InvalidId(999))
        ));
    }
}
