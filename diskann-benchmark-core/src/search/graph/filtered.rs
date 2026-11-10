/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Benchmark single-hop traversal with provider-owned label entry points.

use std::sync::Arc;

use diskann::{
    ANNResult,
    graph::{self, glue},
    provider::DataProvider,
};
use diskann_utils::{
    future::AsyncFriendly,
    views::rowmajor::{self, Matrix},
};

use crate::search::{
    self,
    graph::{KnnParams, Strategy},
};

/// The strategy supplies filter decisions and label-specific entry points. Preparing
/// those strategies happens before timed queries; no internal IDs enter query inputs.
#[derive(Debug)]
pub struct FilteredSearch<DP: DataProvider, T, S> {
    index: Arc<graph::DiskANNIndex<DP>>,
    queries: Arc<rowmajor::Owned<T>>,
    strategy: Strategy<S>,
}

impl<DP: DataProvider, T, S> FilteredSearch<DP, T, S> {
    /// # Errors
    /// Returns an error when the strategy count differs from the query count.
    pub fn new(
        index: Arc<graph::DiskANNIndex<DP>>,
        queries: Arc<rowmajor::Owned<T>>,
        strategy: Strategy<S>,
    ) -> anyhow::Result<Arc<Self>> {
        strategy.length_compatible(queries.nrows())?;
        Ok(Arc::new(Self {
            index,
            queries,
            strategy,
        }))
    }
}

impl<DP, T, S> search::Search for FilteredSearch<DP, T, S>
where
    DP: DataProvider<Context: Default, ExternalId: search::Id>,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [T],
            DP::ExternalId,
            SearchAccessor: glue::FilteredAccessor,
        > + AsyncFriendly,
    T: AsyncFriendly + Clone,
{
    type Id = DP::ExternalId;
    type Parameters = KnnParams;
    type Output = super::knn::Metrics;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &KnnParams) -> search::IdCount {
        search::IdCount::Fixed(parameters.k_value())
    }

    async fn search<O>(
        &self,
        parameters: &KnnParams,
        buffer: &mut O,
        index: usize,
    ) -> ANNResult<Self::Output>
    where
        O: graph::SearchOutputBuffer<DP::ExternalId> + Send,
    {
        let context = DP::Context::default();
        let strategy = self.strategy.get(index)?;
        let query = self.queries.row(index);
        let stats = self
            .index
            .search(
                graph::search::SingleHopFilterSearch::new(parameters.knn),
                strategy,
                &context,
                query,
                buffer,
            )
            .await?;
        Ok(super::knn::Metrics {
            comparisons: stats.cmps,
            hops: stats.hops,
        })
    }
}
