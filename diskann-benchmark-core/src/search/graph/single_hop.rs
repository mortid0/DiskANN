/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use diskann::{
    ANNResult,
    graph::{self, ext::labeled, glue},
    provider,
};
use diskann_utils::{
    future::AsyncFriendly,
    views::rowmajor::{self, Matrix},
};

use crate::search::{self, Search, graph::KnnParams, graph::Strategy};

/// Benchmark single-hop search on a label-connected graph with explicit seeds for each
/// query. Seeds are prepared once and borrowed during timed searches.
/// Empty seed lists return zero results; short result rows are handled by the
/// standard recall aggregator without treating buffer padding as retrieved IDs.
///
/// This is intended to be used in conjunction with [`search::search`] or [`search::search_all`]
/// and provides some basic additional metrics for the latter. Result aggregation for
/// [`search::search_all`] is provided by the [`search::graph::knn::Aggregator`] type (same
/// aggregator as [`search::graph::knn::KNN`]).
///
/// The provided implementation of [`Search`] accepts [`KnnParams`]
/// and returns [`search::graph::knn::Metrics`] as additional output.
#[derive(Debug)]
pub struct SingleHop<DP, T, S>
where
    DP: provider::DataProvider,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    queries: Arc<rowmajor::Owned<T>>,
    strategy: Strategy<S>,
    labels: Arc<[Arc<dyn labeled::QueryLabelProvider<DP::InternalId>>]>,
    init_ids: Arc<[Arc<[DP::InternalId]>]>,
}

impl<DP, T, S> SingleHop<DP, T, S>
where
    DP: provider::DataProvider,
{
    /// Query rows, label providers and seed lists have a one-to-one mapping.
    /// A collection of strategies must also have one entry per query.
    ///
    /// # Errors
    ///
    /// Returns an error if any querywise collection has the wrong length.
    pub fn new(
        index: Arc<graph::DiskANNIndex<DP>>,
        queries: Arc<rowmajor::Owned<T>>,
        strategy: Strategy<S>,
        labels: Arc<[Arc<dyn labeled::QueryLabelProvider<DP::InternalId>>]>,
        init_ids: Arc<[Arc<[DP::InternalId]>]>,
    ) -> anyhow::Result<Arc<Self>> {
        strategy.length_compatible(queries.nrows())?;
        anyhow::ensure!(
            init_ids.len() == queries.nrows(),
            "number of seed lists ({}) must equal number of queries ({})",
            init_ids.len(),
            queries.nrows()
        );

        if labels.len() != queries.nrows() {
            Err(anyhow::anyhow!(
                "Number of label providers ({}) must be equal to the number of queries ({})",
                labels.len(),
                queries.nrows()
            ))
        } else {
            Ok(Arc::new(Self {
                index,
                queries,
                strategy,
                labels,
                init_ids,
            }))
        }
    }
}

impl<DP, T, S> Search for SingleHop<DP, T, S>
where
    DP: provider::DataProvider<Context: Default, ExternalId: search::Id>,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [T],
            DP::ExternalId,
            SearchAccessor: glue::SearchAccessor + glue::SeedableAccessor,
        > + Clone
        + AsyncFriendly,
    T: AsyncFriendly + Clone,
{
    type Id = DP::ExternalId;
    type Parameters = KnnParams;
    type Output = super::knn::Metrics;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &Self::Parameters) -> search::IdCount {
        search::IdCount::Fixed(parameters.k_value())
    }

    async fn search<O>(
        &self,
        parameters: &Self::Parameters,
        buffer: &mut O,
        index: usize,
    ) -> ANNResult<Self::Output>
    where
        O: graph::SearchOutputBuffer<DP::ExternalId> + Send,
    {
        let context = DP::Context::default();
        let knn = parameters.knn;
        let single_hop_search =
            graph::search::SingleHopFilterSearch::new(knn, &self.init_ids[index]);
        let strategy =
            labeled::Filtered::new(self.strategy.get(index)?.clone(), &*self.labels[index]);
        let stats = self
            .index
            .search(
                single_hop_search,
                &strategy,
                &context,
                self.queries.row(index),
                buffer,
            )
            .await?;

        Ok(super::knn::Metrics {
            comparisons: stats.cmps,
            hops: stats.hops,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recall::GroundTruthMode;
    use diskann::graph::{AdjacencyList, test::provider};
    use diskann_vector::distance::Metric;
    use std::num::NonZeroUsize;

    #[derive(Debug)]
    struct Evens;
    impl labeled::QueryLabelProvider<u32> for Evens {
        fn is_match(&self, id: u32) -> bool {
            id < 4 && id.is_multiple_of(2)
        }
    }

    fn index() -> Arc<graph::DiskANNIndex<provider::Provider>> {
        let provider = provider::Provider::new_from(
            provider::Config::new(Metric::L2, 4, provider::StartPoint::new(10, vec![10.0]))
                .unwrap(),
            [(10, AdjacencyList::from_iter_untrusted([0]))],
            [
                (0, vec![0.0], AdjacencyList::from_iter_untrusted([1, 2])),
                (1, vec![1.0], AdjacencyList::new()),
                (2, vec![2.0], AdjacencyList::from_iter_untrusted([0])),
            ],
        )
        .unwrap();
        let config =
            graph::config::Builder::new(4, graph::config::MaxDegree::Same, 10, Metric::L2.into())
                .build()
                .unwrap();
        Arc::new(graph::DiskANNIndex::new(config, provider, None))
    }

    #[test]
    fn single_hop_harness_uses_exact_groundtruth_and_preserves_short_rows() {
        let queries =
            Arc::new(rowmajor::Owned::try_from_data(vec![0.1f32, 2.1, 0.1].into(), 3, 1).unwrap());
        let searcher = SingleHop::new(
            index(),
            queries,
            Strategy::broadcast(provider::Strategy::new()),
            (0..3)
                .map(|_| Arc::new(Evens) as Arc<dyn labeled::QueryLabelProvider<u32>>)
                .collect(),
            [Arc::from([2]), Arc::from([0]), Arc::from([])].into(),
        )
        .unwrap();
        let params = KnnParams::new(5, 10).unwrap();
        let runtime = crate::tokio::runtime(2).unwrap();
        let results = search::search(
            searcher.clone(),
            params,
            NonZeroUsize::new(2).unwrap(),
            &runtime,
        )
        .unwrap();
        let rows = results.ids().as_rows();
        assert_eq!(rows.row(0), [0, 2]);
        assert_eq!(rows.row(1), [2, 0]);
        assert!(rows.row(2).is_empty());
        let truth = vec![vec![0, 2], vec![2, 0], vec![]];
        let two = NonZeroUsize::new(2).unwrap();
        let summaries = search::search_all(
            searcher,
            [search::Run::new(
                params,
                search::Setup {
                    threads: two,
                    tasks: two,
                    reps: two,
                },
            )],
            super::super::knn::Aggregator::new(&truth, 5, 5, GroundTruthMode::Flexible),
        )
        .unwrap();
        assert_eq!(summaries[0].recall.average, 1.0);
        assert!(summaries[0].mean_cmps > 0.0);
        assert!(summaries[0].mean_hops > 0.0);
    }

    #[test]
    fn single_hop_harness_rejects_misaligned_query_inputs() {
        for (labels, seeds) in [(0, 1), (1, 0)] {
            let result = SingleHop::new(
                index(),
                Arc::new(rowmajor::Owned::from_element(1, 1, 0.0f32)),
                Strategy::broadcast(provider::Strategy::new()),
                (0..labels)
                    .map(|_| Arc::new(Evens) as Arc<dyn labeled::QueryLabelProvider<u32>>)
                    .collect(),
                (0..seeds).map(|_| Arc::from([0])).collect(),
            );
            assert!(result.is_err());
        }
    }
}
