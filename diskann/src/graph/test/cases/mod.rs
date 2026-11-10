/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod consolidate;
mod filtered_range_search;
mod grid_insert;
mod grid_search;
mod helpers;
mod index;
mod inline;
mod inplace_delete;
mod multihop;
mod paged_search;
mod range_search;
#[cfg(feature = "experimental_single_hop_search")]
mod single_hop;

/// Set to `true` and recompile to include full adjacency list state in participating
/// baselines.
///
/// Useful for debugging regressions, but produces large baseline files.
const DUMP_GRAPH_STATE: bool = false;
