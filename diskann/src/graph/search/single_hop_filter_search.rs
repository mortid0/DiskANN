/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Experimental traversal for graphs with connectivity within each filter.

use diskann_utils::future::SendFuture;

use super::{
    Knn, Search,
    record::{NoopSearchRecord, SearchRecord},
};
use crate::{
    ANNResult,
    error::IntoANNResult,
    graph::{
        glue::{self, FilteredAccessor, SearchPostProcess, SearchStrategy},
        index::{DiskANNIndex, SearchStats},
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::Neighbor,
    provider::{DataProvider, HasId},
};

/// Experimental search that expands only neighbors matching the query filter.
///
/// Providers select matching start points through [`FilteredAccessor`]. Rejected start
/// points and start points with NaN distances are skipped; if none remain, search returns
/// no candidates. Disconnected matching components require their own start points.
/// Starting points share the L frontier slots with matching neighbors. Ordinary
/// unfiltered graphs can give poor recall with this traversal.
///
/// Post-processing remains the provider's responsibility, including reranking approximate
/// distances and retaining expanded candidates when needed.
#[derive(Debug, Clone, Copy)]
pub struct SingleHopFilterSearch {
    parameters: Knn,
}

impl SingleHopFilterSearch {
    pub fn new(parameters: Knn) -> Self {
        Self { parameters }
    }

    /// Record nodes selected for expansion. A provider may truncate a selected beam
    /// when its I/O budget is exhausted.
    pub fn with_recorder<SR: ?Sized>(self, recorder: &mut SR) -> RecordedSingleHop<'_, SR> {
        RecordedSingleHop {
            parameters: self.parameters,
            recorder,
        }
    }
}

/// Single-hop search with a statically dispatched traversal recorder.
#[derive(Debug)]
pub struct RecordedSingleHop<'r, SR: ?Sized> {
    parameters: Knn,
    recorder: &'r mut SR,
}

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
            self.with_recorder(&mut NoopSearchRecord::new())
                .search(index, strategy, processor, context, query, output)
                .await
        }
    }
}

impl<'a, DP, S, T, SR> Search<'a, DP, S, T> for RecordedSingleHop<'_, SR>
where
    DP: DataProvider,
    S: SearchStrategy<'a, DP, T, SearchAccessor: FilteredAccessor>,
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
            let mut scratch = index.search_scratch(self.parameters.l_value().get(), 0);
            let scratch = &mut *scratch;
            accessor
                .start_point_distances(|decision, distance| {
                    if let glue::Decision::Accept(id) = decision
                        && !distance.is_nan()
                        && scratch.visited.insert(id.into_inner())
                    {
                        scratch
                            .best
                            .insert(Neighbor::new(id.into_inner(), distance));
                        scratch.cmps += 1;
                    }
                })
                .await?;

            let stats = index
                .search_internal(
                    Some(self.parameters.beam_width().get()),
                    &mut Preseeded(&mut accessor),
                    scratch,
                    self.recorder,
                )
                .await?;
            let result_count = processor
                .post_process(&mut accessor, query, scratch.best.iter(), output)
                .await
                .into_ann_result()?;
            Ok(stats.finish(result_count as u32))
        }
    }
}

// Start points have already been installed in scratch. This adapter supplies only
// traversal, so an empty set stays empty instead of reinitializing the provider.
struct Preseeded<'a, A>(&'a mut A);

impl<A: HasId> HasId for Preseeded<'_, A> {
    type Id = A::Id;
}

impl<A: FilteredAccessor> glue::SearchAccessor for Preseeded<'_, A> {
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        std::future::ready(Ok(Vec::new()))
    }

    fn start_point_distances<F>(&mut self, _: F) -> impl Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        std::future::ready(Ok(()))
    }

    fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        pred: P,
        mut on_neighbors: F,
    ) -> impl Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Self::Id, f32) + Send,
    {
        self.0
            .expand_beam_accept_only(ids, Accepted(pred), move |id, distance| {
                on_neighbors(id.into_inner(), distance)
            })
    }

    fn terminate_early(&mut self) -> bool {
        self.0.terminate_early()
    }
}

struct Accepted<P>(P);

impl<I, P: glue::Predicate<I>> glue::Predicate<I> for Accepted<P> {
    fn eval(&self, item: &I) -> bool {
        self.0.eval(item)
    }
}

impl<I, P: glue::PredicateMut<I>> glue::PredicateMut<glue::Accept<I>> for Accepted<P> {
    fn eval_mut(&mut self, item: &glue::Accept<I>) -> bool {
        self.0.eval_mut(item.get())
    }
}
