/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Provider-owned entry points and expanded-node reranking for experimental search.

use super::*;
use crate::label_metadata::LabelMetadata;
use diskann::graph::{glue::SearchAccessor, search::SingleHopFilterSearch};

#[derive(Debug)]
struct Label<'a>(Option<&'a roaring::RoaringBitmap>);

impl QueryLabelProvider<u32> for Label<'_> {
    fn is_match(&self, id: u32) -> bool {
        self.0.is_some_and(|posting| posting.contains(id))
    }
}

struct EntryStrategy<'a, Data, Factory>
where
    Data: GraphDataType<VectorIdType = u32>,
    Factory: VertexProviderFactory<Data>,
{
    inner: DiskSearchStrategy<'a, Data, Factory>,
    entry_points: &'a [u32],
}

struct EntryAccessor<'a, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    inner: DiskAccessor<'a, Data, VP>,
    entry_points: &'a [u32],
}

impl<Data, VP> HasId for EntryAccessor<'_, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    type Id = u32;
}

impl<Data, VP> SearchAccessor for EntryAccessor<'_, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    async fn starting_points(&self) -> ANNResult<Vec<u32>> {
        Ok(self.entry_points.to_vec())
    }

    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(u32, f32) + Send,
    {
        let batch_size = self.inner.scratch.pq_scratch.max_vectors().max(1);
        for ids in self.entry_points.chunks(batch_size) {
            self.inner
                .pq_distances(ids, |distance, id| f(id, distance))?;
        }
        Ok(())
    }

    async fn expand_beam<I, P, F>(&mut self, ids: I, pred: P, f: F) -> ANNResult<()>
    where
        I: Iterator<Item = u32> + Send,
        P: glue::HybridPredicate<u32> + Send + Sync,
        F: FnMut(u32, f32) + Send,
    {
        self.inner.expand_beam(ids, pred, f).await
    }

    fn terminate_early(&mut self) -> bool {
        self.inner.io_tracker.io_count() >= self.inner.provider.search_io_limit
    }
}

impl<'a, Data, Factory> SearchStrategy<'a, DiskProvider<Data>, &'a [Data::VectorDataType]>
    for EntryStrategy<'a, Data, Factory>
where
    Data: GraphDataType<VectorIdType = u32>,
    Factory: VertexProviderFactory<Data>,
{
    type SearchAccessor = EntryAccessor<'a, Data, Factory::VertexProviderType>;
    type SearchAccessorError = ANNError;

    fn search_accessor(
        &'a self,
        provider: &'a DiskProvider<Data>,
        context: &'a DefaultContext,
        query: &'a [Data::VectorDataType],
    ) -> ANNResult<Self::SearchAccessor> {
        Ok(EntryAccessor {
            inner: self.inner.search_accessor(provider, context, query)?,
            entry_points: self.entry_points,
        })
    }
}

struct ExpandedRerank;

impl<Data, VP>
    SearchPostProcess<
        EntryAccessor<'_, Data, VP>,
        &[Data::VectorDataType],
        SearchPayload<Data::AssociatedDataType, Data::VectorDataType>,
    > for ExpandedRerank
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    type Error = ANNError;

    async fn post_process<I, B>(
        &self,
        accessor: &mut EntryAccessor<'_, Data, VP>,
        _query: &[Data::VectorDataType],
        _candidates: I,
        output: &mut B,
    ) -> ANNResult<usize>
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: search_output_buffer::SearchOutputBuffer<
                SearchPayload<Data::AssociatedDataType, Data::VectorDataType>,
            > + Send
            + ?Sized,
    {
        // The cache contains precisely the nodes actually expanded, including ones
        // evicted from the PQ frontier. An I/O limit never triggers extra reads here.
        let mut results: Vec<_> = accessor
            .inner
            .scratch
            .distance_cache
            .iter()
            .filter(|(_, (distance, _, _))| distance.is_finite())
            .map(|(&id, &(distance, data, _))| Neighbor::new((id, data), distance))
            .collect();
        results.sort_unstable_by(|a, b| {
            a.distance()
                .total_cmp(b.distance())
                .then(a.id().0.cmp(&b.id().0))
        });
        extend_output(&mut accessor.inner, results, output)
    }
}

#[derive(Debug, thiserror::Error)]
enum QueryError {
    #[error("experimental stitched search supports squared L2 only, got {0:?}")]
    Metric(Metric),
    #[error("constructed graph has {constructed} rows but the loaded disk index has {loaded}")]
    Rows { constructed: usize, loaded: usize },
    #[error("query has {query} dimensions but the loaded disk index has {loaded}")]
    Dimensions { query: usize, loaded: usize },
    #[error("search list size {l} must be at least requested result count {k}")]
    SearchList { l: u32, k: u32 },
}

#[track_caller]
fn validate_query<Data: GraphDataType<VectorIdType = u32>>(
    provider: &DiskProvider<Data>,
    metadata: &LabelMetadata,
    query: &[Data::VectorDataType],
    k: u32,
    l: u32,
) -> ANNResult<()> {
    let error = if provider.metric != Metric::L2 {
        Some(QueryError::Metric(provider.metric))
    } else if metadata.rows() != provider.num_points {
        Some(QueryError::Rows {
            constructed: metadata.rows(),
            loaded: provider.num_points,
        })
    } else if query.len() != provider.graph_header.metadata().dims {
        Some(QueryError::Dimensions {
            query: query.len(),
            loaded: provider.graph_header.metadata().dims,
        })
    } else if l < k {
        Some(QueryError::SearchList { l, k })
    } else {
        None
    };
    error.map_or(Ok(()), |error| Err(ANNError::new(error)))
}

impl<Data, Factory> DiskIndexSearcher<Data, Factory>
where
    Data: GraphDataType<VectorIdType = u32>,
    Factory: VertexProviderFactory<Data>,
{
    /// Search an immutable stitched graph by atomic label.
    ///
    /// `metadata` must describe the graph loaded by this searcher, with the same row
    /// ordering. Entry points come from construction. Unknown labels return no
    /// results. Only expanded nodes are reranked, so an I/O limit may return fewer
    /// than `return_list_size` results. The returned vector contains valid results only.
    /// This synchronous method has the same runtime requirements as [`Self::search`].
    ///
    /// # Errors
    /// Returns an error for a shape or metric mismatch, invalid search parameters,
    /// or an underlying storage failure. The experimental builder supports L2 only.
    pub fn search_single_hop(
        &self,
        query: &[Data::VectorDataType],
        return_list_size: u32,
        search_list_size: u32,
        beam_width: Option<usize>,
        metadata: &LabelMetadata,
        label: u32,
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
        validate_query(
            self.index.provider(),
            metadata,
            query,
            return_list_size,
            search_list_size,
        )?;
        let knn = Knn::new(search_list_size as usize, beam_width)?;
        let timer = Instant::now();
        let tracker = IOTracker::default();
        let strategy = EntryStrategy {
            inner: self.search_strategy(&tracker, PostprocessStrategy::AcceptAll, false),
            entry_points: metadata.entry_points(label),
        };
        let filter = Label(metadata.posting(label));
        let strategy = labeled::Filtered::new(strategy, &filter);
        let count = return_list_size as usize;
        let mut ids = vec![0; count];
        let mut distances = vec![0.; count];
        let mut data = vec![Data::AssociatedDataType::default(); count];
        let mut output = SearchOutput::new(&mut ids, &mut distances, &mut data, None)?;
        let stats = self.runtime.block_on(self.index.search_with(
            SingleHopFilterSearch::new(knn),
            &strategy,
            glue::Pipeline::new(labeled::Unfiltered, ExpandedRerank),
            &DefaultContext,
            query,
            &mut output,
        ))?;
        let total_execution_time_us = timer.elapsed().as_micros();
        let io_time_us = u128::from(IOTracker::time(&tracker.io_time_us));
        let query_pq_preprocess_time_us = u128::from(IOTracker::time(&tracker.preprocess_time_us));
        let query_statistics = QueryStatistics {
            total_comparisons: stats.cmps,
            search_hops: stats.hops,
            total_execution_time_us,
            io_time_us,
            total_io_operations: tracker.io_count() as u32,
            total_vertices_loaded: tracker.io_count() as u32,
            query_pq_preprocess_time_us,
            cpu_time_us: total_execution_time_us
                .saturating_sub(io_time_us + query_pq_preprocess_time_us),
            ..Default::default()
        };
        let results = ids
            .into_iter()
            .zip(distances)
            .zip(data)
            .take(stats.result_count as usize)
            .map(|((vertex_id, distance), data)| SearchResultItem {
                vertex_id,
                distance,
                data,
            })
            .collect();
        Ok(SearchResult {
            results,
            stats: SearchResultStats {
                cmps: stats.cmps,
                result_count: stats.result_count,
                query_statistics,
            },
        })
    }
}
