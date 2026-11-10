/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Experimental in-memory StitchedVamana construction for immutable atomic labels.
//!
//! This prototype supports squared L2 and data that fit in RAM. It does not
//! implement FilteredVamana, universal labels, dense-point replication, or a
//! persistent filtered-index format. Use the supported benchmark job to evaluate
//! recall and distance comparisons before choosing it for a dataset.
//!
//! Per-label Vamana graphs supply geometric edges. A protected directed cycle for
//! each posting list makes label connectivity independent of geometric pruning.
//! Overlapping labels can require more protected edges than the degree budget;
//! construction fails rather than silently disconnecting such a label.

use std::{collections::BTreeMap, sync::Arc};

use diskann::{
    ANNError, ANNResult,
    error::ErrorContext,
    graph::{
        AdjacencyList, DiskANNIndex, config,
        glue::{PruneStrategy, SearchAccessor, SearchStrategy},
    },
    provider::{DataProvider, NeighborAccessor},
    utils::VectorRepr,
};
use diskann_inmem::{
    Context, Provider, Strategy,
    num::{Capacity, MaxDegree},
    repr::full::{Config as ProviderConfig, FullPrecision},
};
use diskann_providers::utils::{ParallelIteratorInPool, create_thread_pool};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::{DistanceFunction, distance::Metric};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rayon::prelude::*;
use thiserror::Error;
use tokio::task::JoinSet;

/// Construction settings for squared-L2 vector data.
#[derive(Debug, Clone)]
pub struct StitchedBuildConfig {
    label_degree: usize,
    max_degree: usize,
    l_build: usize,
    alpha: f32,
    /// Maximum simultaneous insertion tasks. The caller supplies the Tokio runtime.
    concurrency: usize,
    /// Controls insertion order; parallel graph updates need not be bit reproducible.
    seed: u64,
}

/// Invalid experimental construction settings.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("label degree must be in 1..=u32::MAX, got {0}")]
    LabelDegree(usize),
    #[error("maximum degree must be in 1..=u32::MAX, got {0}")]
    MaxDegree(usize),
    #[error("construction L {l_build} must be at least label degree {label_degree} and fit u32")]
    SearchList { l_build: usize, label_degree: usize },
    #[error("alpha must be finite and at least one, got {0}")]
    Alpha(f32),
    #[error("construction concurrency must be positive")]
    ZeroConcurrency,
}

#[derive(Debug, Error)]
#[error(
    "filtered build requires nonzero dimensions and 0 < rows < u32::MAX, got {rows} rows and {dimensions} dimensions"
)]
struct InvalidShape {
    rows: usize,
    dimensions: usize,
}

#[derive(Debug, Error)]
enum InvalidPosting {
    #[error("duplicate label {0}")]
    DuplicateLabel(u32),
    #[error("label {0} has an empty posting list")]
    Empty(u32),
    #[error("label {label} contains row {id}, outside a dataset of {rows} rows")]
    OutOfBounds { label: u32, id: u32, rows: usize },
    #[error("label {label} has non-increasing adjacent members {left} and {right}")]
    Unsorted { label: u32, left: u32, right: u32 },
}

#[derive(Debug, Error)]
#[error("non-finite vector at row {0}")]
struct NonFiniteVector(u32);

#[derive(Debug, Error)]
#[error("row {row} needs {required} protected label edges but max_degree is {available}")]
struct InsufficientDegree {
    row: u32,
    required: usize,
    available: usize,
}

#[derive(Debug, Error)]
enum InvalidEntryPoint {
    #[error("entry-point full dimension must be positive")]
    EmptyDimension,
    #[error("row {row} of label {label} has full dimension {actual}, expected {expected}")]
    Dimension {
        label: u32,
        row: u32,
        expected: usize,
        actual: usize,
    },
    #[error("non-finite full-precision entry-point row {row} of label {label}")]
    NonFiniteRow { label: u32, row: u32 },
    #[error("invalid centroid for label {0}")]
    NonFiniteCentroid(u32),
    #[error("invalid centroid distance {distance} for row {row} of label {label}")]
    NonFiniteDistance { label: u32, row: u32, distance: f64 },
}

#[derive(Debug, Error)]
#[error("membership count overflow at row {0}")]
struct MembershipOverflow(usize);

#[derive(Debug, Error)]
enum InvalidLabelGraph {
    #[error("construction entry {entry} is outside label {label}")]
    MissingEntry { label: u32, entry: u32 },
    #[error("graph for label {label} emitted unmapped internal ID {id}")]
    UnmappedId { label: u32, id: u32 },
    #[error(
        "per-label Vamana emitted local ID {id}, outside {members} members and their virtual start"
    )]
    OutOfBounds { id: u32, members: usize },
}

#[derive(Debug, Error)]
#[error("invalid squared L2 distance {distance} between rows {left} and {right}")]
struct InvalidDistance {
    left: u32,
    right: u32,
    distance: f32,
}

diskann::convert_error!(InvalidShape);
diskann::convert_error!(InvalidPosting);
diskann::convert_error!(NonFiniteVector);
diskann::convert_error!(InsufficientDegree);
diskann::convert_error!(InvalidEntryPoint);
diskann::convert_error!(MembershipOverflow);
diskann::convert_error!(InvalidLabelGraph);
diskann::convert_error!(InvalidDistance);

impl StitchedBuildConfig {
    /// Validate construction settings before starting a build.
    ///
    /// # Errors
    /// Rejects zero degrees/concurrency, L smaller than the label degree, and
    /// non-finite alpha or alpha below one.
    pub fn new(
        label_degree: usize,
        max_degree: usize,
        l_build: usize,
        alpha: f32,
        concurrency: usize,
        seed: u64,
    ) -> Result<Self, ConfigError> {
        let settings = Self {
            label_degree,
            max_degree,
            l_build,
            alpha,
            concurrency,
            seed,
        };
        settings.validate()?;
        Ok(settings)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if self.label_degree == 0 || self.label_degree > u32::MAX as usize {
            return Err(ConfigError::LabelDegree(self.label_degree));
        }
        if self.max_degree == 0 || self.max_degree > u32::MAX as usize {
            return Err(ConfigError::MaxDegree(self.max_degree));
        }
        if self.l_build < self.label_degree || self.l_build > u32::MAX as usize {
            return Err(ConfigError::SearchList {
                l_build: self.l_build,
                label_degree: self.label_degree,
            });
        }
        if !self.alpha.is_finite() || self.alpha < 1.0 {
            return Err(ConfigError::Alpha(self.alpha));
        }
        if self.concurrency == 0 {
            return Err(ConfigError::ZeroConcurrency);
        }
        Ok(())
    }
}

impl Default for StitchedBuildConfig {
    fn default() -> Self {
        Self {
            label_degree: 32,
            max_degree: 128,
            l_build: 100,
            alpha: 1.2,
            concurrency: 8,
            seed: 0,
        }
    }
}

/// Dataset IDs remain the original row IDs; no frozen point is included.
#[derive(Debug, Clone)]
pub struct StitchedGraph {
    neighbors: Vec<Vec<u32>>,
    postings: BTreeMap<u32, Vec<u32>>,
    entry_points: BTreeMap<u32, Vec<u32>>,
    max_degree: u32,
}

impl StitchedGraph {
    pub fn neighbors(&self) -> &[Vec<u32>] {
        &self.neighbors
    }
    pub fn entry_points(&self) -> &BTreeMap<u32, Vec<u32>> {
        &self.entry_points
    }
    pub fn postings(&self) -> &BTreeMap<u32, Vec<u32>> {
        &self.postings
    }
    pub fn max_degree(&self) -> u32 {
        self.max_degree
    }
    /// Sorted dataset row IDs with this atomic label. Unknown labels are empty.
    pub fn posting(&self, label: u32) -> &[u32] {
        self.postings.get(&label).map_or(&[], Vec::as_slice)
    }
}

/// Build a filtered graph using only Rust Vamana and vector kernels.
///
/// Each posting list must be nonempty, strictly increasing and in bounds, and
/// label IDs must be unique. Label IDs need not be dense. Rows without any
/// geometric or protected edges receive an edge to the next dataset row, so
/// unlabeled and singleton-only rows can be exported with the standard disk
/// writer. Filtered traversal rejects these edges when the label does not match.
/// A one-row dataset retains empty adjacency and cannot be exported to disk.
///
/// Membership storage is O(rows + memberships), not O(rows * labels). Only one
/// per-label vector provider is resident at a time. The stitched candidate graph
/// additionally needs O(sum of per-label edges) memory before final pruning.
///
/// # Errors
/// Returns an error for malformed postings, invalid settings or distances, an
/// insufficient protected-edge degree budget, or a failed Vamana insertion task.
pub async fn build_stitched<'a, T, I>(
    vectors: rowmajor::Ref<'_, T>,
    postings: I,
    settings: &StitchedBuildConfig,
) -> ANNResult<StitchedGraph>
where
    T: VectorRepr + FullPrecision,
    I: IntoIterator<Item = (u32, &'a [u32])>,
{
    validate_settings(vectors.nrows(), vectors.ncols(), settings)?;
    let mut postings: Vec<_> = postings.into_iter().collect();
    postings.sort_unstable_by_key(|(label, _)| *label);
    validate_postings(vectors.nrows(), &postings)?;
    let membership = Membership::new(vectors.nrows(), &postings)?;
    let distance = T::distance(Metric::L2, Some(vectors.ncols()));
    let pool = create_thread_pool(settings.concurrency)?;
    (0..vectors.nrows())
        .into_par_iter()
        .try_for_each_in_pool(pool.as_ref(), |id| {
            if !distance
                .evaluate_similarity(vectors.row(id), vectors.row(id))
                .is_finite()
            {
                return Err(ANNError::from(NonFiniteVector(id as u32)));
            }
            Ok(())
        })?;

    let mut neighbors: Vec<Vec<u32>> = (0..vectors.nrows()).map(|_| Vec::new()).collect();
    // Install all connectivity witnesses before expensive Vamana construction.
    for (_, members) in &postings {
        if members.len() < 2 {
            continue;
        }
        for (position, &id) in members.iter().enumerate() {
            let next = members[(position + 1) % members.len()];
            let row = &mut neighbors[id as usize];
            if !row.contains(&next) {
                row.push(next);
                if row.len() > settings.max_degree {
                    return Err(InsufficientDegree {
                        row: id,
                        required: row.len(),
                        available: settings.max_degree,
                    }
                    .into());
                }
            }
        }
    }
    let protected_counts: Vec<_> = neighbors.iter().map(|row| row.len()).collect();
    let mut entry_points = BTreeMap::new();
    let mut entry_scratch = EntryScratch::new(vectors.row(0))?;

    for &(label, members) in &postings {
        let entry = entry_scratch.select(vectors, label, members)?;
        entry_points.insert(label, vec![entry]);
        if members.len() <= settings.label_degree.saturating_add(1) {
            // Exact small-label graphs avoid a provider and Tokio task per tiny label.
            for &id in members {
                neighbors[id as usize].extend(members.iter().copied().filter(|&n| n != id));
            }
        } else {
            append_label_graph(vectors, label, members, entry, settings, &mut neighbors).await?;
        }
    }

    neighbors
        .par_iter_mut()
        .enumerate()
        .try_for_each_in_pool(pool.as_ref(), |(id, row)| {
            prune_row(
                id as u32,
                row,
                protected_counts[id],
                vectors,
                &membership,
                settings,
                &distance,
            )
        })?;
    if vectors.nrows() > 1 {
        for (id, row) in neighbors.iter_mut().enumerate() {
            if row.is_empty() {
                let neighbor = ((id + 1) % vectors.nrows()) as u32;
                checked_distance(&distance, vectors, id as u32, neighbor)?;
                row.push(neighbor);
            }
        }
    }
    Ok(StitchedGraph {
        neighbors,
        postings: postings
            .into_iter()
            .map(|(label, ids)| (label, ids.to_vec()))
            .collect(),
        entry_points,
        max_degree: u32::try_from(settings.max_degree)?,
    })
}

/// Scratch is shared across labels; conversion never allocates for an individual row.
struct EntryScratch {
    row: Vec<f32>,
    centroid: Vec<f64>,
}

impl EntryScratch {
    #[track_caller]
    fn new<T: VectorRepr>(first: &[T]) -> ANNResult<Self> {
        let dimensions = T::full_dimension(first).map_err(Into::<ANNError>::into)?;
        if dimensions == 0 {
            return Err(InvalidEntryPoint::EmptyDimension.into());
        }
        Ok(Self {
            row: vec![0.; dimensions],
            centroid: vec![0.; dimensions],
        })
    }

    #[track_caller]
    fn convert<T: VectorRepr>(&mut self, vector: &[T], label: u32, id: u32) -> ANNResult<()> {
        let dimensions = T::full_dimension(vector).map_err(Into::<ANNError>::into)?;
        if dimensions != self.row.len() {
            return Err(InvalidEntryPoint::Dimension {
                label,
                row: id,
                expected: self.row.len(),
                actual: dimensions,
            }
            .into());
        }
        T::as_f32_into(vector, &mut self.row)
            .map_err(Into::<ANNError>::into)
            .with_context(|| format!("converting entry-point row {id} of label {label}"))?;
        if self.row.iter().any(|value| !value.is_finite()) {
            return Err(InvalidEntryPoint::NonFiniteRow { label, row: id }.into());
        }
        Ok(())
    }

    // Members are validated sorted, unique base IDs. Keep accumulation sequential
    // for deterministic metadata, independently of concurrent Vamana construction.
    fn select<T: VectorRepr>(
        &mut self,
        vectors: rowmajor::Ref<'_, T>,
        label: u32,
        members: &[u32],
    ) -> ANNResult<u32> {
        if let [id] = members {
            self.convert(vectors.row(*id as usize), label, *id)?;
            return Ok(*id);
        }
        self.centroid.fill(0.);
        for &id in members {
            self.convert(vectors.row(id as usize), label, id)?;
            for (sum, &value) in self.centroid.iter_mut().zip(&self.row) {
                *sum += f64::from(value);
            }
        }
        for sum in &mut self.centroid {
            *sum /= members.len() as f64;
        }
        if self.centroid.iter().any(|value| !value.is_finite()) {
            return Err(InvalidEntryPoint::NonFiniteCentroid(label).into());
        }
        let mut best = (f64::INFINITY, u32::MAX);
        for &id in members {
            self.convert(vectors.row(id as usize), label, id)?;
            let distance: f64 = self
                .row
                .iter()
                .zip(&self.centroid)
                .map(|(&value, &center)| {
                    let delta = f64::from(value) - center;
                    delta * delta
                })
                .sum();
            if !distance.is_finite() {
                return Err(InvalidEntryPoint::NonFiniteDistance {
                    label,
                    row: id,
                    distance,
                }
                .into());
            }
            if distance < best.0 || (distance == best.0 && id < best.1) {
                best = (distance, id);
            }
        }
        Ok(best.1)
    }
}

#[track_caller]
fn validate_settings(rows: usize, dim: usize, settings: &StitchedBuildConfig) -> ANNResult<()> {
    if rows == 0 || rows >= u32::MAX as usize || dim == 0 {
        return Err(InvalidShape {
            rows,
            dimensions: dim,
        }
        .into());
    }
    settings.validate().map_err(ANNError::new)
}

fn validate_postings(rows: usize, postings: &[(u32, &[u32])]) -> Result<(), InvalidPosting> {
    let mut previous_label = None;
    for &(label, members) in postings {
        if previous_label == Some(label) {
            return Err(InvalidPosting::DuplicateLabel(label));
        }
        previous_label = Some(label);
        if members.is_empty() {
            return Err(InvalidPosting::Empty(label));
        }
        if let Some(&id) = members.iter().find(|&&id| id as usize >= rows) {
            return Err(InvalidPosting::OutOfBounds { label, id, rows });
        }
        if let Some(pair) = members.windows(2).find(|pair| pair[0] >= pair[1]) {
            return Err(InvalidPosting::Unsorted {
                label,
                left: pair[0],
                right: pair[1],
            });
        }
    }
    Ok(())
}

/// Point-to-label CSR used only for label-aware occlusion.
struct Membership {
    offsets: Vec<usize>,
    labels: Vec<u32>,
}

impl Membership {
    fn new(rows: usize, postings: &[(u32, &[u32])]) -> Result<Self, MembershipOverflow> {
        let mut offsets = vec![0usize; rows + 1];
        for (_, members) in postings {
            for &id in *members {
                offsets[id as usize + 1] = offsets[id as usize + 1]
                    .checked_add(1)
                    .ok_or(MembershipOverflow(id as usize))?;
            }
        }
        for i in 1..offsets.len() {
            offsets[i] = offsets[i]
                .checked_add(offsets[i - 1])
                .ok_or(MembershipOverflow(i - 1))?;
        }
        let mut cursor = offsets[..rows].to_vec();
        let mut labels = vec![0; offsets[rows]];
        for &(label, members) in postings {
            for &id in members {
                labels[cursor[id as usize]] = label;
                cursor[id as usize] += 1;
            }
        }
        Ok(Self { offsets, labels })
    }

    fn get(&self, id: u32) -> &[u32] {
        &self.labels[self.offsets[id as usize]..self.offsets[id as usize + 1]]
    }

    // The paper's condition: labels(source) intersect labels(candidate) is a
    // subset of labels(selected). Irrelevant candidate labels cannot block pruning.
    fn covered(&self, source: u32, candidate: u32, selected: u32) -> bool {
        let a = self.get(source);
        let b = self.get(candidate);
        let c = self.get(selected);
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    if c.binary_search(&a[i]).is_err() {
                        return false;
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        true
    }
}

async fn append_label_graph<T: VectorRepr + FullPrecision>(
    vectors: rowmajor::Ref<'_, T>,
    label: u32,
    members: &[u32],
    entry: u32,
    settings: &StitchedBuildConfig,
    stitched: &mut [Vec<u32>],
) -> ANNResult<()> {
    let local_entry = members
        .binary_search(&entry)
        .map_err(|_| InvalidLabelGraph::MissingEntry { label, entry })?;
    let entry_id = u32::try_from(local_entry)?;
    let entry_vector = vectors.row(entry as usize);
    let mut builder = config::Builder::new(
        settings.label_degree,
        config::MaxDegree::default_slack(),
        settings.l_build,
        Metric::L2.into(),
    );
    builder.alpha(settings.alpha).saturate_after_prune(false);
    let config = builder.build()?;
    let provider = Provider::<_, u32>::new(
        ProviderConfig::new(
            Capacity::new(members.len()),
            MaxDegree::new(config.max_degree().get()),
            Metric::L2,
            rowmajor::Owned::row_vector(entry_vector.to_vec().into()),
        )
        .map_err(ANNError::new)?,
    )?;
    let index = Arc::new(DiskANNIndex::new(config, provider, None));
    index
        .insert(&Strategy, &Context, &entry_id, entry_vector)
        .await?;

    let mut order: Vec<_> = (0..members.len())
        .filter(|&local| local != local_entry)
        .collect();
    order.shuffle(&mut StdRng::seed_from_u64(settings.seed ^ u64::from(label)));
    let mut tasks = JoinSet::new();
    for local in order {
        if tasks.len() == settings.concurrency {
            join_one(&mut tasks).await?;
        }
        let index = Arc::clone(&index);
        // At most `concurrency` row copies are live; do not copy a whole label dataset.
        let vector = vectors.row(members[local] as usize).to_vec();
        let local = u32::try_from(local)?;
        tasks.spawn(async move { index.insert(&Strategy, &Context, &local, &vector).await });
    }
    while !tasks.is_empty() {
        join_one(&mut tasks).await?;
    }
    // Query the provider mapping; insertion order and allocation policy are opaque.
    let mut internal_to_local = BTreeMap::new();
    for local in 0..members.len() {
        internal_to_local.insert(
            index.provider().to_internal_id(&Context, &(local as u32))?,
            local,
        );
    }
    let accessor = Strategy.search_accessor(index.provider(), &Context, entry_vector)?;
    let starts = accessor.starting_points().await?;
    drop(accessor);
    let mut prune_ids: Vec<_> = internal_to_local.keys().copied().collect();
    prune_ids.extend_from_slice(&starts);
    index.prune_range(&Strategy, &Context, prune_ids).await?;
    let mut accessor = Strategy.prune_accessor(index.provider(), &Context, 0)?;
    let mut virtual_neighbors = AdjacencyList::new();
    for start in &starts {
        let mut row = AdjacencyList::new();
        accessor.get_neighbors(*start, &mut row).await?;
        virtual_neighbors.extend_from_slice(&row);
    }
    for start in starts {
        internal_to_local.insert(start, members.len());
    }
    let remap = |ids: &[u32]| -> ANNResult<Vec<u32>> {
        ids.iter()
            .map(|id| {
                internal_to_local
                    .get(id)
                    .copied()
                    .map(|local| local as u32)
                    .ok_or_else(|| InvalidLabelGraph::UnmappedId { label, id: *id }.into())
            })
            .collect()
    };
    let virtual_neighbors = remap(&virtual_neighbors)?;
    let mut local_neighbors = AdjacencyList::new();
    for (local, &global) in members.iter().enumerate() {
        let internal = index.provider().to_internal_id(&Context, &(local as u32))?;
        accessor
            .get_neighbors(internal, &mut local_neighbors)
            .await?;
        append_remapped_label_neighbors(
            local,
            local_entry,
            members,
            &virtual_neighbors,
            &remap(&local_neighbors)?,
            &mut stitched[global as usize],
        )?;
    }
    Ok(())
}

fn append_remapped_label_neighbors(
    local: usize,
    local_entry: usize,
    members: &[u32],
    virtual_neighbors: &[u32],
    local_neighbors: &[u32],
    stitched: &mut Vec<u32>,
) -> Result<(), InvalidLabelGraph> {
    // The selected real centroid entry inherits the identical virtual start's
    // neighborhood. Every virtual edge must resolve to that same real member.
    let inherited = if local == local_entry {
        virtual_neighbors
    } else {
        &[]
    };
    for &neighbor in inherited.iter().chain(local_neighbors) {
        let neighbor = if neighbor as usize == members.len() {
            local_entry
        } else {
            neighbor as usize
        };
        if neighbor >= members.len() {
            return Err(InvalidLabelGraph::OutOfBounds {
                id: neighbor as u32,
                members: members.len(),
            });
        }
        if neighbor != local {
            stitched.push(members[neighbor]);
        }
    }
    Ok(())
}

async fn join_one(tasks: &mut JoinSet<ANNResult<()>>) -> ANNResult<()> {
    if let Some(result) = tasks.join_next().await {
        result
            .map_err(ANNError::new)
            .context("filtered Vamana insertion task failed")??;
    }
    Ok(())
}

fn prune_row<T: VectorRepr>(
    id: u32,
    row: &mut Vec<u32>,
    protected_count: usize,
    vectors: rowmajor::Ref<'_, T>,
    membership: &Membership,
    settings: &StitchedBuildConfig,
    distance: &T::Distance,
) -> ANNResult<()> {
    let mut selected = row[..protected_count].to_vec();
    for &neighbor in &selected {
        checked_distance(distance, vectors, id, neighbor)?;
    }
    let mut candidates = std::mem::take(row);
    candidates.sort_unstable();
    candidates.dedup();
    candidates.retain(|candidate| *candidate != id && !selected.contains(candidate));
    let mut scored = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let score = checked_distance(distance, vectors, id, candidate)?;
        scored.push((candidate, score));
    }
    scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    // The per-label graphs were already geometrically pruned. A second prune
    // would remove their routing edges even when the stitched union fits on SSD.
    // Keep the protected prefix and use distance/ID order for the other edges.
    if selected.len() + scored.len() <= settings.max_degree {
        selected.extend(scored.into_iter().map(|(candidate, _)| candidate));
        *row = selected;
        return Ok(());
    }
    for (candidate, score) in scored {
        if selected.len() == settings.max_degree {
            break;
        }
        let mut occluded = false;
        for &other in &selected {
            if membership.covered(id, candidate, other)
                && settings.alpha * checked_distance(distance, vectors, candidate, other)? <= score
            {
                occluded = true;
                break;
            }
        }
        if !occluded {
            selected.push(candidate);
        }
    }
    // Protected cycles remain a prefix and are never candidates for removal.
    *row = selected;
    Ok(())
}

fn checked_distance<T: VectorRepr>(
    distance: &T::Distance,
    vectors: rowmajor::Ref<'_, T>,
    left: u32,
    right: u32,
) -> Result<f32, InvalidDistance> {
    let value =
        distance.evaluate_similarity(vectors.row(left as usize), vectors.row(right as usize));
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(InvalidDistance {
            left,
            right,
            distance: value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, VecDeque};

    fn assert_connected(graph: &StitchedGraph, label: u32, members: &[u32]) {
        let entry = graph.entry_points[&label][0];
        assert!(members.contains(&entry));
        let mut seen = BTreeSet::from([entry]);
        let mut queue = VecDeque::from([entry]);
        while let Some(id) = queue.pop_front() {
            let row = &graph.neighbors[id as usize];
            assert!(row.len() <= graph.max_degree as usize);
            assert_eq!(
                row.iter().copied().collect::<BTreeSet<_>>().len(),
                row.len()
            );
            for &neighbor in row {
                assert_ne!(neighbor, id);
                assert!((neighbor as usize) < graph.neighbors.len());
                if members.binary_search(&neighbor).is_ok() && seen.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
        assert_eq!(seen.into_iter().collect::<Vec<_>>(), members);
    }

    #[tokio::test]
    async fn entry_is_actual_centroid_neighbor_instead_of_first_outlier() {
        let values = [100f32, 100., 35., 35., 0., 0., 5., 5.];
        let vectors = rowmajor::Ref::try_from_data(&values, 4, 2).unwrap();
        let graph = build_stitched(
            vectors,
            [(7, &[0, 1, 2, 3][..]), (99, &[0][..])],
            &Default::default(),
        )
        .await
        .unwrap();
        // The centroid is [35,35], exactly row1; row0 is an outlier.
        assert_eq!(graph.entry_points[&7], [1]);
        assert_eq!(graph.entry_points[&99], [0]);
        assert_connected(&graph, 7, &[0, 1, 2, 3]);
        assert_connected(&graph, 99, &[0]);
    }

    #[tokio::test]
    async fn centroid_ties_use_global_ids_and_sparse_labels_use_only_their_members() {
        let values = [
            90u8, 0, 1, 0, 0, 0, 10, 0, 120, 0, 20, 0, 255, 0, 2, 0, 30, 0,
        ];
        let vectors = rowmajor::Ref::try_from_data(&values, 9, 2).unwrap();
        let labels = [
            (u32::MAX, vec![2, 7]),
            (400_000_007, vec![3, 5, 8]),
            (99, vec![4]),
        ];
        let graph = build_stitched(
            vectors,
            labels.iter().map(|(label, ids)| (*label, ids.as_slice())),
            &Default::default(),
        )
        .await
        .unwrap();
        // Row1 is at this centroid but is not eligible. Rows2/7 tie, so ID2 wins.
        assert_eq!(graph.entry_points[&u32::MAX], [2]);
        assert_eq!(graph.entry_points[&400_000_007], [5]);
        assert_eq!(graph.entry_points[&99], [4]);
        for (label, members) in labels {
            assert_connected(&graph, label, &members);
        }
        assert_eq!(graph.neighbors[1], [2]);
    }

    #[test]
    fn centroid_accumulation_handles_f32_sum_overflow_and_rejects_nonfinite_conversion() {
        let values = [f32::MAX; 3];
        let vectors = rowmajor::Ref::column_vector(&values);
        let mut scratch = EntryScratch::new(vectors.row(0)).unwrap();
        assert_eq!(scratch.select(vectors, 7, &[0, 1, 2]).unwrap(), 0);
        for invalid in [f32::NAN, f32::INFINITY] {
            let values = [1., invalid];
            assert!(
                scratch
                    .select(rowmajor::Ref::column_vector(&values), 7, &[0, 1])
                    .is_err()
            );
        }
    }

    #[tokio::test]
    async fn sparse_overlapping_labels_keep_every_label_connected() {
        let values: Vec<_> = (0..16).map(|x| x as f32).collect();
        let vectors = rowmajor::Ref::try_from_data(&values, values.len() / 2, 2).unwrap();
        let labels = [
            (7, vec![0, 2, 4, 6]),
            (u32::MAX, vec![0, 1, 3]),
            (99, vec![5]),
        ];
        let settings = StitchedBuildConfig {
            label_degree: 2,
            max_degree: 4,
            l_build: 8,
            concurrency: 2,
            ..Default::default()
        };
        let graph = build_stitched(
            vectors,
            labels.iter().map(|(l, ids)| (*l, ids.as_slice())),
            &settings,
        )
        .await
        .unwrap();
        for (label, members) in &labels {
            assert_connected(&graph, *label, members);
        }
        assert_eq!(graph.neighbors[7], [0]);
        assert_eq!(graph.neighbors.len(), 8);
    }

    #[tokio::test]
    async fn unlabeled_and_singleton_rows_have_deterministic_disk_compatible_edges() {
        let values = [0f32, 1., 2., 3.];
        let vectors = rowmajor::Ref::column_vector(&values);
        let settings = StitchedBuildConfig {
            max_degree: 1,
            ..Default::default()
        };
        let unlabeled = build_stitched(vectors, [], &settings).await.unwrap();
        assert_eq!(unlabeled.neighbors, [vec![1], vec![2], vec![3], vec![0]]);
        assert!(unlabeled.postings().is_empty());
        assert!(unlabeled.entry_points().is_empty());

        let singleton_labels = [(7, &[0][..]), (9, &[2][..]), (99, &[3][..])];
        let singleton = build_stitched(vectors, singleton_labels, &settings)
            .await
            .unwrap();
        assert_eq!(singleton.neighbors, unlabeled.neighbors);
        for (label, members) in singleton_labels {
            assert_eq!(singleton.posting(label), members);
            assert_eq!(singleton.entry_points()[&label], members);
            assert_connected(&singleton, label, members);
        }
    }

    #[tokio::test]
    async fn one_row_graph_retains_valid_membership_without_a_self_edge() {
        let values = [1f32];
        let graph = build_stitched(
            rowmajor::Ref::column_vector(&values),
            [(7, &[0][..])],
            &Default::default(),
        )
        .await
        .unwrap();
        assert!(graph.neighbors[0].is_empty());
        assert_connected(&graph, 7, &[0]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_vamana_u8_preserves_cycles_after_stitch_and_prune() {
        let values: Vec<u8> = (0..64).flat_map(|i| [(i % 8) * 10, (i / 8) * 10]).collect();
        let vectors = rowmajor::Ref::try_from_data(&values, values.len() / 2, 2).unwrap();
        let first: Vec<u32> = (0..64).filter(|id| id % 2 == 0).collect();
        let second: Vec<u32> = (0..64).filter(|id| id % 3 == 0).collect();
        let settings = StitchedBuildConfig {
            label_degree: 4,
            max_degree: 8,
            l_build: 16,
            concurrency: 2,
            seed: 17,
            ..Default::default()
        };
        let graph = build_stitched(
            vectors,
            [(100_000_007, first.as_slice()), (2, second.as_slice())],
            &settings,
        )
        .await
        .unwrap();
        assert_connected(&graph, 100_000_007, &first);
        assert_connected(&graph, 2, &second);
    }

    #[tokio::test]
    async fn insufficient_degree_fails_instead_of_losing_a_label() {
        let values = [0.0f32, 1.0, 2.0, 3.0];
        let vectors = rowmajor::Ref::try_from_data(&values, values.len(), 1).unwrap();
        let settings = StitchedBuildConfig {
            max_degree: 2,
            ..Default::default()
        };
        let error = build_stitched(
            vectors,
            [(1, &[0, 1][..]), (2, &[0, 2][..]), (3, &[0, 3][..])],
            &settings,
        )
        .await
        .unwrap_err();
        let error = error.downcast::<InsufficientDegree>().unwrap();
        assert_eq!(error.row, 0);
        assert_eq!(error.required, 3);
        assert_eq!(error.available, 2);
    }

    #[tokio::test]
    async fn malformed_memberships_and_nonfinite_vectors_are_rejected() {
        let values = [0.0f32, 1.0, 2.0];
        let vectors = rowmajor::Ref::try_from_data(&values, values.len(), 1).unwrap();
        for members in [&[][..], &[0, 0][..], &[1, 0][..], &[0, 3][..]] {
            assert!(
                build_stitched(vectors, [(1, members)], &Default::default())
                    .await
                    .is_err()
            );
        }
        assert!(
            build_stitched(vectors, [(1, &[0][..]), (1, &[1][..])], &Default::default())
                .await
                .is_err()
        );
        let invalid = [f32::NAN];
        assert!(
            build_stitched(
                rowmajor::Ref::try_from_data(&invalid, invalid.len(), 1).unwrap(),
                [(1, &[0][..])],
                &Default::default()
            )
            .await
            .is_err()
        );
    }

    #[test]
    fn posting_validation_reports_the_invalid_member_and_label() {
        assert!(matches!(
            validate_postings(3, &[(7, &[0]), (7, &[1])]),
            Err(InvalidPosting::DuplicateLabel(7))
        ));
        assert!(matches!(
            validate_postings(3, &[(7, &[])]),
            Err(InvalidPosting::Empty(7))
        ));
        assert!(matches!(
            validate_postings(3, &[(7, &[0, 3])]),
            Err(InvalidPosting::OutOfBounds {
                label: 7,
                id: 3,
                rows: 3
            })
        ));
        for members in [&[0, 0][..], &[1, 0][..]] {
            assert!(matches!(
                validate_postings(3, &[(7, members)]),
                Err(InvalidPosting::Unsorted { label: 7, .. })
            ));
        }
    }

    #[tokio::test]
    async fn invalid_settings_and_overflowing_edge_distances_are_rejected() {
        let values = [0.0f32, 1.0];
        let vectors = rowmajor::Ref::column_vector(&values);
        let defaults = StitchedBuildConfig::default();
        for settings in [
            StitchedBuildConfig {
                concurrency: 0,
                ..defaults.clone()
            },
            StitchedBuildConfig {
                max_degree: 0,
                ..defaults.clone()
            },
            StitchedBuildConfig {
                label_degree: 0,
                ..defaults.clone()
            },
            StitchedBuildConfig {
                l_build: 1,
                ..defaults.clone()
            },
            StitchedBuildConfig {
                alpha: f32::NAN,
                ..defaults.clone()
            },
            StitchedBuildConfig {
                alpha: 0.5,
                ..defaults.clone()
            },
        ] {
            assert!(
                StitchedBuildConfig::new(
                    settings.label_degree,
                    settings.max_degree,
                    settings.l_build,
                    settings.alpha,
                    settings.concurrency,
                    settings.seed,
                )
                .is_err()
            );
            assert!(
                build_stitched(vectors, [(1, &[0, 1][..])], &settings)
                    .await
                    .is_err()
            );
        }
        let extreme = [f32::MAX, -f32::MAX];
        assert!(
            build_stitched(
                rowmajor::Ref::column_vector(&extreme),
                [(1, &[0, 1][..])],
                &defaults
            )
            .await
            .is_err()
        );
    }

    #[test]
    fn occlusion_uses_shared_source_labels() {
        let membership = Membership::new(4, &[(1, &[0, 1, 2]), (2, &[1, 3])]).unwrap();
        // Candidate1 has labels1+2; source0 only requires label1, which selected2 covers.
        assert!(membership.covered(0, 1, 2));
        assert!(!membership.covered(0, 1, 3));
    }

    #[test]
    fn virtual_edges_and_inherited_neighbors_resolve_to_the_nonfirst_centroid_entry() {
        let members = [10, 20, 30, 40];
        let mut other = Vec::new();
        append_remapped_label_neighbors(0, 2, &members, &[0, 1], &[4], &mut other).unwrap();
        // A non-entry row resolves its virtual neighbor to the centroid member,
        // and does not inherit the virtual start's outgoing neighborhood.
        assert_eq!(other, [30]);
        let mut entry = Vec::new();
        append_remapped_label_neighbors(2, 2, &members, &[0, 1], &[4, 3], &mut entry).unwrap();
        assert_eq!(entry, [10, 20, 40]);
        assert!(append_remapped_label_neighbors(0, 2, &members, &[], &[5], &mut other).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn vamana_constructs_from_a_nonfirst_centroid_and_exports_only_real_label_members() {
        let values = [255u8, 255, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0];
        let vectors = rowmajor::Ref::try_from_data(&values, 8, 2).unwrap();
        let members = [0, 1, 2, 3, 4, 5, 6, 7];
        let settings = StitchedBuildConfig {
            label_degree: 2,
            max_degree: 8,
            l_build: 8,
            concurrency: 2,
            ..Default::default()
        };
        // This forces real Vamana construction, rather than the exact-small-label
        // shortcut exercised by the separate centroid metadata tests.
        let graph = build_stitched(vectors, [(7, members.as_slice())], &settings)
            .await
            .unwrap();
        assert_eq!(graph.entry_points[&7][0], 7);
        assert_connected(&graph, 7, &members);
        assert!(graph.neighbors[7].iter().all(|&id| id != 7 && id < 8));
    }

    #[test]
    fn in_budget_union_keeps_geometric_edges_and_cleans_duplicate_candidates() {
        let values = [0f32, 1., 1.1, 2.];
        let vectors = rowmajor::Ref::column_vector(&values);
        let membership = Membership::new(4, &[(7, &[0, 1, 2, 3])]).unwrap();
        let settings = StitchedBuildConfig {
            max_degree: 3,
            ..Default::default()
        };
        let distance = f32::distance(Metric::L2, Some(1));
        let mut row = vec![3, 2, 1, 0, 2];
        prune_row(0, &mut row, 1, vectors, &membership, &settings, &distance).unwrap();
        // Edge0->2 is geometrically occluded by edge0->1, but still belongs to
        // the already-pruned per-label union. The protected edge stays first.
        assert_eq!(row, [3, 1, 2]);

        let mut overflow = vec![3, 2, 1, 0, 2];
        let settings = StitchedBuildConfig {
            max_degree: 2,
            ..settings
        };
        prune_row(
            0,
            &mut overflow,
            1,
            vectors,
            &membership,
            &settings,
            &distance,
        )
        .unwrap();
        assert_eq!(overflow, [3, 1]);
    }

    #[tokio::test]
    async fn stitching_preserves_in_budget_cliques_and_bounds_overflow_with_connectivity() {
        let values = [0f32, 1., 2., 3.];
        let vectors = rowmajor::Ref::column_vector(&values);
        let members = [0, 1, 2, 3];
        let settings = StitchedBuildConfig {
            label_degree: 3,
            max_degree: 3,
            l_build: 4,
            ..Default::default()
        };
        let graph = build_stitched(vectors, [(7, members.as_slice())], &settings)
            .await
            .unwrap();
        for (id, row) in graph.neighbors.iter().enumerate() {
            assert_eq!(row.len(), 3);
            assert!(
                members
                    .iter()
                    .filter(|&&member| member != id as u32)
                    .all(|member| row.contains(member))
            );
        }
        assert_connected(&graph, 7, &members);

        let settings = StitchedBuildConfig {
            max_degree: 2,
            ..settings
        };
        let overflow = build_stitched(vectors, [(7, members.as_slice())], &settings)
            .await
            .unwrap();
        assert_connected(&overflow, 7, &members);
        assert!(overflow.neighbors.iter().all(|row| row.len() <= 2));
        assert!(overflow.neighbors.iter().any(|row| row.len() < 3));
    }
}
