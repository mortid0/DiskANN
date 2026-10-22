/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Snapshot-consistent queries over an immutable graph and newer records.

use std::{
    collections::BTreeSet,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    sync::Arc,
};

use diskann::{ANNError, ANNResult};

use super::{
    metadata::LabelMetadata,
    native::{self, NativeSearcher, NativeVector},
    updates::{DeltaSnapshot, Mutation},
};
use crate::search::search_mode::SearchMode;

#[track_caller]
pub(super) fn ann(error: impl std::fmt::Display) -> ANNError {
    ANNError::message(error.to_string())
}

/// An exact distance and stable application ID, with no padding entries.
#[derive(Clone, Debug, PartialEq)]
pub struct FilteredHit {
    pub id: u32,
    pub distance: f32,
}

/// A result from one consistent mutation sequence.
#[derive(Clone, Debug)]
pub struct FilteredQueryResult {
    pub hits: Vec<FilteredHit>,
    pub sequence: u64,
    /// True when a small base or an insufficient approximate result required a scan.
    pub exact_fallback: bool,
    pub base_searches: usize,
    pub delta_candidates: usize,
}

pub(super) struct Base<T: NativeVector> {
    _lease: Arc<super::persistence::GenerationStore>,
    pub directory: std::path::PathBuf,
    pub generation: u64,
    pub sequence: u64,
    pub dimensions: usize,
    pub labels: LabelMetadata,
    pub searcher: Option<NativeSearcher<T>>,
}

impl<T: NativeVector> Base<T> {
    pub fn open(
        directory: &Path,
        checkpoint: (u64, u64),
        dimensions: usize,
        labels: LabelMetadata,
        has_graph: bool,
        threads: usize,
        lease: Arc<super::persistence::GenerationStore>,
    ) -> ANNResult<Self> {
        let (generation, sequence) = checkpoint;
        let mut file = File::open(directory.join("vectors.bin"))?;
        let mut header = [0; 8];
        file.read_exact(&mut header)?;
        let rows = u32::from_le_bytes(header[..4].try_into().map_err(ann)?) as usize;
        let cols = u32::from_le_bytes(header[4..].try_into().map_err(ann)?) as usize;
        let expected = rows
            .checked_mul(cols)
            .and_then(|x| x.checked_mul(size_of::<T>()))
            .and_then(|x| x.checked_add(8))
            .ok_or_else(|| ann("vector file size overflow"))?;
        if rows != labels.point_count()
            || cols != dimensions
            || file.metadata()?.len() != expected as u64
        {
            return Err(ann(
                "filtered vector file does not match its generation metadata",
            ));
        }
        T::validate_stored_vectors(&directory.join("vectors.bin"), dimensions)?;
        Ok(Self {
            _lease: lease,
            directory: directory.to_path_buf(),
            generation,
            sequence,
            dimensions,
            labels,
            searcher: if has_graph {
                Some(native::open_searcher(directory, threads)?)
            } else {
                None
            },
        })
    }

    pub fn vector(&self, ordinal: u32) -> ANNResult<Vec<T>> {
        if ordinal as usize >= self.labels.point_count() {
            return Err(ann("base vector ordinal is outside this generation"));
        }
        let mut file = File::open(self.directory.join("vectors.bin"))?;
        file.seek(SeekFrom::Start(
            8 + ordinal as u64 * self.dimensions as u64 * size_of::<T>() as u64,
        ))?;
        let mut vector = vec![T::default(); self.dimensions];
        file.read_exact(bytemuck::cast_slice_mut(&mut vector))?;
        Ok(vector)
    }
}

pub(super) struct View<T: NativeVector> {
    pub base: Arc<Base<T>>,
    pub delta: DeltaSnapshot<T>,
    pub next_id: u64,
    shadowed_labels: Arc<BTreeSet<u32>>,
}

impl<T: NativeVector> View<T> {
    pub fn new(base: Arc<Base<T>>, delta: DeltaSnapshot<T>, next_id: u64) -> Self {
        let shadowed_labels = delta
            .iter()
            .filter_map(|(id, _)| base.labels.ordinal(id))
            .flat_map(|ordinal| base.labels.labels(ordinal).iter().copied())
            .collect();
        Self {
            base,
            delta,
            next_id,
            shadowed_labels: Arc::new(shadowed_labels),
        }
    }

    pub fn after_batch(
        &self,
        delta: DeltaSnapshot<T>,
        next_id: u64,
        mutations: &[Mutation<T>],
    ) -> Self {
        let mut shadowed_labels = self.shadowed_labels.clone();
        for mutation in mutations {
            if let Some(ordinal) = self.base.labels.ordinal(mutation.id()) {
                for &label in self.base.labels.labels(ordinal) {
                    if !shadowed_labels.contains(&label) {
                        Arc::make_mut(&mut shadowed_labels).insert(label);
                    }
                }
            }
        }
        Self {
            base: self.base.clone(),
            delta,
            next_id,
            shadowed_labels,
        }
    }
}

/// A leased, immutable view. Later mutations and compactions cannot change its results.
pub struct FilteredSnapshot<T: NativeVector> {
    pub(super) view: Arc<View<T>>,
}

impl<T: NativeVector> Clone for FilteredSnapshot<T> {
    fn clone(&self) -> Self {
        Self {
            view: self.view.clone(),
        }
    }
}

impl<T: NativeVector> FilteredSnapshot<T> {
    pub fn sequence(&self) -> u64 {
        self.view.delta.sequence()
    }

    /// Last mutation incorporated into the immutable SSD generation.
    pub fn base_sequence(&self) -> u64 {
        self.view.base.sequence
    }

    /// Run SSD search on Tokio's blocking pool without blocking an async executor.
    pub async fn search_async(
        &self,
        query: Vec<T>,
        label: u32,
        k: usize,
        l: u32,
    ) -> ANNResult<FilteredQueryResult> {
        let snapshot = self.clone();
        tokio::task::spawn_blocking(move || {
            native::on_blocking_worker(|| snapshot.search(&query, label, k, l))
        })
        .await
        .map_err(ann)?
    }

    /// Query one stable label. Deleted and relabeled base vertices remain routing bridges;
    /// their obsolete records never enter the returned result.
    ///
    /// The base remains approximate. A bounded overfetch retries after shadow filtering;
    /// if shadowing still hides useful base candidates, an exact posting scan establishes
    /// completeness. Matching delta hits cannot conceal a truncated live-base pool.
    pub fn search(
        &self,
        query: &[T],
        label: u32,
        k: usize,
        l: u32,
    ) -> ANNResult<FilteredQueryResult> {
        let view = &self.view;
        let base = &view.base;
        if query.len() != base.dimensions || k == 0 || (l as usize) < k || k > u32::MAX as usize {
            return Err(ann(
                "filtered search requires matching dimensions and 0 < k <= L <= u32::MAX",
            ));
        }
        T::validate(query)?;
        let mut delta_hits: Vec<_> = view
            .delta
            .iter_matching(label)
            .map(|(id, point)| FilteredHit {
                id,
                distance: T::exact_l2(query, &point.vector),
            })
            .collect();
        let delta_candidates = delta_hits.len();
        // A delta point below its own top-k cannot enter the merged top-k.
        // Keep the complete match count, but bound retry clones and merge inputs.
        sort_hits(&mut delta_hits, k);
        let postings = base.labels.posting(label);
        let shadowed = view.shadowed_labels.contains(&label);
        let mut base_searches = 0;
        let mut exact_fallback = false;
        let mut hits = delta_hits.clone();
        if !postings.is_empty() {
            if let Some(searcher) = &base.searcher {
                if postings.len() > k {
                    let mut breadth = (l as usize).min(postings.len());
                    let max_breadth = breadth.saturating_mul(4).min(postings.len());
                    loop {
                        let requested = if shadowed { breadth } else { k };
                        let mode =
                            SearchMode::single_hop_filter(|id| base.labels.contains(label, *id))
                                .with_starting_points(base.labels.entries(label).to_vec());
                        let found = searcher.search(
                            query,
                            requested as u32,
                            breadth as u32,
                            Some(4),
                            mode,
                        )?;
                        base_searches += 1;
                        hits = delta_hits.clone();
                        let mut live_base_hits = 0;
                        for item in found.results.iter().take(found.stats.result_count as usize) {
                            let id = base
                                .labels
                                .stable_id(item.vertex_id)
                                .ok_or_else(|| ann("search returned an invalid base ordinal"))?;
                            if view.delta.get(id).is_none() {
                                live_base_hits += 1;
                                hits.push(FilteredHit {
                                    id,
                                    distance: item.distance,
                                });
                            }
                        }
                        sort_hits(&mut hits, k);
                        // The exact-sorted output can be truncated before live base winners.
                        // A full set of far delta hits alone is not evidence we fetched enough.
                        let count = found.stats.result_count as usize;
                        let pool_sufficient = !shadowed
                            || live_base_hits >= k
                            || count < requested
                            || (count > 0
                                && hits.last().is_some_and(|worst| {
                                    worst.distance < found.results[count - 1].distance
                                }));
                        if hits.len() >= k && pool_sufficient {
                            break;
                        }
                        if breadth >= max_breadth {
                            exact_fallback = true;
                            break;
                        }
                        breadth = breadth.saturating_mul(2).min(max_breadth);
                    }
                } else {
                    exact_fallback = true;
                }
            } else {
                exact_fallback = true;
            }
            if exact_fallback {
                hits = delta_hits;
                // Open once for the scan rather than once per vector.
                let mut file = File::open(base.directory.join("vectors.bin"))?;
                let mut vector = vec![T::default(); base.dimensions];
                for &ordinal in postings {
                    let id = base
                        .labels
                        .stable_id(ordinal)
                        .ok_or_else(|| ann("invalid posting ordinal"))?;
                    if view.delta.get(id).is_some() {
                        continue;
                    }
                    file.seek(SeekFrom::Start(
                        8 + ordinal as u64 * base.dimensions as u64 * size_of::<T>() as u64,
                    ))?;
                    file.read_exact(bytemuck::cast_slice_mut(&mut vector))?;
                    hits.push(FilteredHit {
                        id,
                        distance: T::exact_l2(query, &vector),
                    });
                }
            }
        }
        sort_hits(&mut hits, k);
        Ok(FilteredQueryResult {
            hits,
            sequence: self.sequence(),
            exact_fallback,
            base_searches,
            delta_candidates,
        })
    }

    /// Read the current record by stable ID, including records newer than the SSD generation.
    pub fn get(&self, id: u32) -> ANNResult<Option<(Vec<T>, Vec<u32>)>> {
        if let Some(record) = self.view.delta.get(id) {
            return Ok(record.point().map(|p| (p.vector.clone(), p.labels.clone())));
        }
        let base = &self.view.base;
        base.labels
            .ordinal(id)
            .map(|ordinal| Ok((base.vector(ordinal)?, base.labels.labels(ordinal).to_vec())))
            .transpose()
    }
}

fn sort_hits(hits: &mut Vec<FilteredHit>, k: usize) {
    let compare = |a: &FilteredHit, b: &FilteredHit| {
        a.distance
            .total_cmp(&b.distance)
            .then_with(|| a.id.cmp(&b.id))
    };
    if hits.len() > k {
        hits.select_nth_unstable_by(k, compare);
        hits.truncate(k);
    }
    hits.sort_unstable_by(compare);
}
