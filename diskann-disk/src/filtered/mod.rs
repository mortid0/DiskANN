/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Native filtered index lifecycle.
#![doc = include_str!("README.md")]

mod index;
pub mod metadata;
mod native;
pub mod persistence;
pub mod updates;
mod view;

pub use index::{FilteredIndex, FilteredIndexConfig};
pub use native::NativeVector;
pub use view::{FilteredHit, FilteredQueryResult, FilteredSnapshot};
