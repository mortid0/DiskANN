/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Top-level disk search mode.
//!
//! Encodes the algorithm + filter combination for `DiskIndexSearcher::search`
//! as a sum type so that invalid combinations (flat scan with adaptive L,
//! inline filter without a predicate) are unrepresentable at the API boundary.

use diskann::graph::ext::labeled::QueryLabelProvider;
use diskann::graph::search::{AdaptiveL, StartSelection};
use diskann_providers::model::graph::provider::DeterminantDiversityParams;

/// Owned closure used to filter vector IDs during disk search.
///
/// The `&u32` argument is the disk-side internal/external ID (they coincide
/// on the disk path by construction).
pub type SearchPredicate<'a> = Box<dyn Fn(&u32) -> bool + Send + Sync + 'a>;

struct FnLabelProvider<F>(F);

impl<F> std::fmt::Debug for FnLabelProvider<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnLabelProvider").finish_non_exhaustive()
    }
}

impl<F> QueryLabelProvider<u32> for FnLabelProvider<F>
where
    F: Fn(&u32) -> bool + Send + Sync,
{
    fn is_match(&self, id: u32) -> bool {
        (self.0)(&id)
    }
}

/// Top-level disk search mode.
///
/// Variants encode the algorithm + filter combination:
///
/// * `FlatScan` — brute-force linear scan, with or without an inline filter.
/// * `Graph` — plain greedy beam search; the optional filter is applied as a
///   hard post-filter during reranking (no traversal-time effect).
/// * `InlineFilter` — label-filtered graph search; the predicate is consulted
///   at visit time (not just during rerank). `adaptive_l = Some(_)` grows the
///   beam mid-search if the observed match specificity is low.
/// * `SingleHopFilter` — strict traversal within a label-connected graph.
/// * `DiverseGraph` — greedy graph search with determinant-diversity
///   post-processing; selects a maximally diverse top-k from the candidate
///   pool using `DeterminantDiversityParams`. Optional hard post-filter is
///   applied during the diversity selection step.
pub enum SearchMode<'a> {
    FlatScan {
        filter: Option<SearchPredicate<'a>>,
    },

    Graph {
        filter: Option<SearchPredicate<'a>>,
    },

    InlineFilter {
        filter: Box<dyn QueryLabelProvider<u32> + 'a>,
        adaptive_l: Option<AdaptiveL>,
    },

    /// Experimental strict traversal; requires a label-connected graph and matching starts.
    SingleHopFilter {
        starting_points: &'a [u32],
        start_selection: StartSelection<u32>,
        filter: Box<dyn QueryLabelProvider<u32> + 'a>,
    },

    /// Ablation: strict traversal with reranking limited to the final PQ frontier.
    SingleHopFilterFrontier {
        starting_points: &'a [u32],
        start_selection: StartSelection<u32>,
        filter: Box<dyn QueryLabelProvider<u32> + 'a>,
    },

    DiverseGraph {
        filter: Option<SearchPredicate<'a>>,
        params: DeterminantDiversityParams,
    },
}

impl<'a> SearchMode<'a> {
    /// Flat scan over all vectors. Recall baseline.
    pub fn flat() -> Self {
        Self::FlatScan { filter: None }
    }

    /// Flat scan restricted to vectors that satisfy `predicate`.
    pub fn flat_filtered<F>(predicate: F) -> Self
    where
        F: Fn(&u32) -> bool + Send + Sync + 'a,
    {
        Self::FlatScan {
            filter: Some(Box::new(predicate)),
        }
    }

    /// Plain greedy graph search; no filter.
    pub fn graph() -> Self {
        Self::Graph { filter: None }
    }

    /// Plain greedy graph search with a hard post-filter applied during
    /// reranking. Traversal is unaffected.
    pub fn graph_filtered<F>(predicate: F) -> Self
    where
        F: Fn(&u32) -> bool + Send + Sync + 'a,
    {
        Self::Graph {
            filter: Some(Box::new(predicate)),
        }
    }

    /// Inline label-filtered graph search. `adaptive_l = Some(_)` enables
    /// mid-search beam widening; `None` runs inline tracking only (no
    /// resizing).
    ///
    /// The closure is wrapped in a generic adapter (`FnLabelProvider<F>`)
    /// that implements `QueryLabelProvider<u32>`.
    pub fn inline_filter<F>(predicate: F, adaptive_l: Option<AdaptiveL>) -> Self
    where
        F: Fn(&u32) -> bool + Send + Sync + 'a,
    {
        Self::InlineFilter {
            filter: Box::new(FnLabelProvider(predicate)),
            adaptive_l,
        }
    }

    /// Experimental strict single-hop traversal. All starts must match the predicate.
    ///
    /// Use a graph built to retain connectivity for this label; restricting an ordinary
    /// graph can substantially reduce recall. Entry points are borrowed and required;
    /// an empty slice returns no results. Every expanded candidate is reranked
    /// using its cached full-precision distance, even if later evicted from the PQ frontier.
    /// An I/O limit can yield fewer than k results: uncached frontier nodes are omitted.
    pub fn single_hop_filter<F>(predicate: F, starting_points: &'a [u32]) -> Self
    where
        F: Fn(&u32) -> bool + Send + Sync + 'a,
    {
        Self::SingleHopFilter {
            starting_points,
            start_selection: StartSelection::All,
            filter: Box::new(FnLabelProvider(predicate)),
        }
    }

    /// Strict single-hop traversal seeded from the nearest member of an entry pool.
    ///
    /// The pool is ranked using the query's in-memory PQ distances, so selection
    /// reads no vectors from SSD; ties go to the lower ID. `fallback`, which must
    /// belong to the pool, is seeded as well. Every pool member must match the
    /// predicate, and each distinct member counts as one comparison.
    pub fn single_hop_filter_nearest_start<F>(
        predicate: F,
        pool: &'a [u32],
        fallback: Option<u32>,
    ) -> Self
    where
        F: Fn(&u32) -> bool + Send + Sync + 'a,
    {
        Self::SingleHopFilter {
            starting_points: pool,
            start_selection: StartSelection::Nearest { fallback },
            filter: Box::new(FnLabelProvider(predicate)),
        }
    }

    /// Ablation of strict search that reranks only the final best `L` PQ candidates.
    ///
    /// Traversal is identical to [`Self::single_hop_filter`], but expanded candidates
    /// evicted from the frontier are excluded from the result pool.
    pub fn single_hop_filter_frontier<F>(predicate: F, starting_points: &'a [u32]) -> Self
    where
        F: Fn(&u32) -> bool + Send + Sync + 'a,
    {
        Self::SingleHopFilterFrontier {
            starting_points,
            start_selection: StartSelection::All,
            filter: Box::new(FnLabelProvider(predicate)),
        }
    }

    /// Frontier-reranking ablation seeded like [`Self::single_hop_filter_nearest_start`].
    pub fn single_hop_filter_frontier_nearest_start<F>(
        predicate: F,
        pool: &'a [u32],
        fallback: Option<u32>,
    ) -> Self
    where
        F: Fn(&u32) -> bool + Send + Sync + 'a,
    {
        Self::SingleHopFilterFrontier {
            starting_points: pool,
            start_selection: StartSelection::Nearest { fallback },
            filter: Box::new(FnLabelProvider(predicate)),
        }
    }

    /// Greedy graph search with determinant-diversity post-processing.
    /// Selects a diverse top-k from the candidate pool found at L.
    pub fn diverse_graph(params: DeterminantDiversityParams) -> Self {
        Self::DiverseGraph {
            filter: None,
            params,
        }
    }

    /// Greedy graph search with determinant-diversity post-processing and a
    /// hard post-filter. The filter is honored during the diverse-selection
    /// step (non-matching IDs are excluded from the final top-k).
    pub fn diverse_graph_filtered<F>(predicate: F, params: DeterminantDiversityParams) -> Self
    where
        F: Fn(&u32) -> bool + Send + Sync + 'a,
    {
        Self::DiverseGraph {
            filter: Some(Box::new(predicate)),
            params,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_no_filter_constructor() {
        let mode = SearchMode::flat();
        assert!(matches!(mode, SearchMode::FlatScan { filter: None }));
    }

    #[test]
    fn flat_filtered_constructor() {
        let mode = SearchMode::flat_filtered(|id| *id == 5);
        match &mode {
            SearchMode::FlatScan { filter: Some(p) } => {
                assert!(p(&5));
                assert!(!p(&4));
            }
            _ => panic!("expected FlatScan with filter"),
        }
    }

    #[test]
    fn graph_no_filter_constructor() {
        let mode = SearchMode::graph();
        assert!(matches!(mode, SearchMode::Graph { filter: None }));
    }

    #[test]
    fn graph_filtered_constructor() {
        let mode = SearchMode::graph_filtered(|id| *id == 7);
        match &mode {
            SearchMode::Graph { filter: Some(p) } => {
                assert!(p(&7));
                assert!(!p(&6));
            }
            _ => panic!("expected Graph with filter"),
        }
    }

    #[test]
    fn inline_filter_constructor_without_adaptive_l() {
        let mode = SearchMode::inline_filter(|id| *id == 3, None);
        match &mode {
            SearchMode::InlineFilter {
                filter,
                adaptive_l: None,
            } => {
                assert!(filter.is_match(3));
                assert!(!filter.is_match(2));
            }
            _ => panic!("expected InlineFilter with adaptive_l = None"),
        }
    }

    #[test]
    fn inline_filter_constructor_with_adaptive_l() {
        let adaptive = AdaptiveL::new(10, 16.0).expect("valid AdaptiveL");
        let mode = SearchMode::inline_filter(|id| *id == 11, Some(adaptive));
        match &mode {
            SearchMode::InlineFilter {
                adaptive_l: Some(_),
                ..
            } => {}
            _ => panic!("expected InlineFilter with adaptive_l = Some"),
        }
    }
}
