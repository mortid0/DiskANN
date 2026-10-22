/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Strict filtered traversal for graphs built to preserve label connectivity.

use diskann_utils::future::SendFuture;
use thiserror::Error;

use super::{Knn, Search};
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
/// All expanded candidates are passed to the post-processor, including candidates
/// evicted from the bounded approximate-distance frontier. The post-processor is
/// responsible for ranking this candidate stream.
#[derive(Debug)]
pub struct SingleHopFilterSearch {
    /// Base graph search parameters.
    pub inner: Knn,
    rerank_frontier: bool,
}

impl SingleHopFilterSearch {
    /// Create strict filtered search parameters.
    pub fn new(inner: Knn) -> Self {
        Self {
            inner,
            rerank_frontier: false,
        }
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

#[derive(Debug, Error)]
#[error("single-hop filtered search requires every starting point to match the filter")]
struct RejectedStartingPoint;

impl<'a, DP, S, T> Search<'a, DP, S, T> for SingleHopFilterSearch
where
    DP: DataProvider,
    S: SearchStrategy<'a, DP, T, SearchAccessor: FilteredAccessor>,
    T: Copy + Send + Sync,
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
            // Strict entry points are ordinary matching dataset IDs, not frozen
            // build points. They share the L frontier slots with other candidates.
            let mut scratch = index.search_scratch(self.inner.l_value().get(), 0);
            let scratch = &mut *scratch;
            let mut rejected_start = false;
            accessor
                .start_point_distances(|decision, distance| {
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
            let result_count = if let Some(expanded) = expanded {
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
                SingleHopFilterSearch::new(Knn::new_default(2).unwrap()),
                &Filtered::new(ApproximateStrategy(&trace, approximate_distance), &Evens),
                ExactRerank(&trace),
                &provider::Context::new(),
                &[0.0][..],
                &mut results,
            ))
            .unwrap();
        assert_eq!(*results[0].id(), 0, "the exact winner was already expanded");
        assert_eq!(trace.lock().unwrap().expanded, [10, 0, 2, 4, 6]);
        assert_eq!(trace.lock().unwrap().candidates, [10, 0, 2, 4, 6]);
        assert_eq!(stats.cmps, 5);
        assert_eq!(stats.hops, 5);
        assert_eq!(index.provider().metrics().get_vector, 5);

        let frontier_index = approximate_graph();
        let frontier_trace = Mutex::new(Trace::default());
        let mut frontier_results = Vec::<Neighbor<u32>>::new();
        let frontier_stats = current_thread_runtime()
            .block_on(frontier_index.search_with(
                SingleHopFilterSearch::new(Knn::new_default(2).unwrap()).with_frontier_reranking(),
                &Filtered::new(
                    ApproximateStrategy(&frontier_trace, approximate_distance),
                    &Evens,
                ),
                ExactRerank(&frontier_trace),
                &provider::Context::new(),
                &[0.0][..],
                &mut frontier_results,
            ))
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
            SingleHopFilterSearch::new(Knn::new_default(1).unwrap()),
            SingleHopFilterSearch::new(Knn::new_default(1).unwrap()).with_frontier_reranking(),
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
            SingleHopFilterSearch::new(Knn::new_default(1).unwrap()),
            SingleHopFilterSearch::new(Knn::new_default(1).unwrap()).with_frontier_reranking(),
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
                SingleHopFilterSearch::new(Knn::new_default(10).unwrap()),
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
                SingleHopFilterSearch::new(Knn::new_default(10).unwrap()),
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
}
