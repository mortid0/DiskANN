/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Rust StitchedVamana construction for immutable, atomic-label SSD graphs.
//!
//! Per-label Vamana graphs supply geometric edges. A protected directed cycle for
//! each posting list makes label connectivity independent of geometric pruning.
//! Overlapping labels can require more protected edges than the degree budget;
//! construction fails rather than silently disconnecting such a label.

use std::{collections::BTreeMap, sync::Arc};

use diskann::{
    error::ErrorContext,
    graph::{config, AdjacencyList},
    provider::DefaultContext,
    utils::VectorRepr,
    ANNError, ANNResult,
};
use diskann_providers::{
    index::diskann_async,
    model::graph::provider::async_::{
        common::{FullPrecision, NoDeletes},
        inmem::{DefaultProviderParameters, SetStartPoints},
    },
    utils::{create_thread_pool, ParallelIteratorInPool},
};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::{distance::Metric, DistanceFunction};
use rand::{
    rngs::StdRng,
    seq::{index, SliceRandom},
    SeedableRng,
};
use rayon::prelude::*;
use tokio::task::JoinSet;

/// Construction settings for squared-L2 vector data.
#[derive(Debug, Clone)]
pub struct StitchedBuildConfig {
    pub label_degree: usize,
    pub max_degree: usize,
    pub l_build: usize,
    pub alpha: f32,
    /// Maximum simultaneous insertion tasks. The caller supplies the Tokio runtime.
    pub concurrency: usize,
    /// Controls insertion order; parallel graph updates need not be bit reproducible.
    pub seed: u64,
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
    pub neighbors: Vec<Vec<u32>>,
    pub entry_points: BTreeMap<u32, Vec<u32>>,
    pub max_degree: u32,
}

/// Build a filtered graph using only Rust Vamana and vector kernels.
///
/// Each posting list must be nonempty, strictly increasing and in bounds, and
/// label IDs must be unique. Label IDs need not be dense. Unlabeled rows remain
/// present with empty adjacency lists. A caller may supply an explicit unqueried
/// label for those rows if it also requires unfiltered navigation.
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
    T: VectorRepr,
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
                return Err(ANNError::message(format!("non-finite vector at row {id}")));
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
                    return Err(ANNError::message(format!(
                        "row {id} needs {} protected label edges but max_degree is {}",
                        row.len(),
                        settings.max_degree
                    )));
                }
            }
        }
    }
    let protected_counts: Vec<_> = neighbors.iter().map(|row| row.len()).collect();
    let mut entry_scratch = EntryScratch::new(vectors.row(0))?;
    let entry_points: BTreeMap<_, _> = postings
        .iter()
        .map(|&(label, members)| Ok((label, vec![entry_scratch.select(vectors, label, members)?])))
        .collect::<ANNResult<_>>()?;

    for &(label, members) in &postings {
        // Keep ordinary builds on one centroid route. Additional corpus entry
        // pools remain available through the explicit selector below.
        let entry = entry_points[&label][0];
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
    Ok(StitchedGraph {
        neighbors,
        entry_points,
        max_degree: u32::try_from(settings.max_degree)?,
    })
}

const ENTRY_CLUSTERS: usize = 4;
const ENTRY_SAMPLE_LIMIT: usize = 4096;
const ENTRY_LLOYD_ITERATIONS: usize = 8;
const ENTRY_SAMPLE_SEED: u64 = 0x6c61_6265_6c5f_7633;

/// Select up to five corpus-only search representatives for each atomic label.
///
/// The first entry is the actual member nearest the full-label centroid and is
/// also the default construction seed. Labels with at most five members return all
/// members. For larger labels, four centers use deterministic farthest-first
/// initialization, followed by eight Lloyd iterations on at most 4096 members.
/// The uniform, without-replacement sample uses `StdRng` seeded by
/// `0x6c61_6265_6c5f_7633 ^ u64::from(label)`. Sorted sample indices include the
/// global centroid medoid, replacing the largest sampled index when necessary.
/// A final full-label assignment pass selects the actual nearest assigned member
/// to each cluster center. Empty clusters contribute no representative.
///
/// All distance and mean accumulation uses finite `f64` values. Distance ties
/// between real members use the smaller global ID; assignment ties use the
/// center's deterministic initialization order. Results keep the global centroid
/// first, then distinct cluster medoids in global-ID order. Only posting members
/// are converted, so calls for separate labels have the same result as one call.
/// Memory is bounded by the sample and vector dimension, independently of label
/// cardinality; graph edges and query data are not used.
///
/// # Errors
/// Returns an error for an empty or invalid vector shape, malformed postings,
/// non-finite eligible vectors, or non-finite mean/distance calculations.
pub fn select_label_entries<'a, T, I>(
    vectors: rowmajor::Ref<'_, T>,
    postings: I,
) -> ANNResult<BTreeMap<u32, Vec<u32>>>
where
    T: VectorRepr,
    I: IntoIterator<Item = (u32, &'a [u32])>,
{
    validate_vector_shape(vectors.nrows(), vectors.ncols())?;
    let mut postings: Vec<_> = postings.into_iter().collect();
    postings.sort_unstable_by_key(|(label, _)| *label);
    validate_postings(vectors.nrows(), &postings)?;
    let mut scratch = EntryScratch::new(vectors.row(0))?;
    postings
        .into_iter()
        .map(|(label, members)| Ok((label, scratch.select_pool(vectors, label, members)?)))
        .collect()
}

/// Scratch is shared across labels; conversion never allocates for an individual row.
struct EntryScratch {
    row: Vec<f32>,
    row64: Vec<f64>,
    centroid: Vec<f64>,
}

impl EntryScratch {
    #[track_caller]
    fn new<T: VectorRepr>(first: &[T]) -> ANNResult<Self> {
        let dimensions = T::full_dimension(first).map_err(Into::<ANNError>::into)?;
        if dimensions == 0 {
            return Err(ANNError::message(
                "entry-point full dimension must be positive",
            ));
        }
        Ok(Self {
            row: vec![0.; dimensions],
            row64: vec![0.; dimensions],
            centroid: vec![0.; dimensions],
        })
    }

    #[track_caller]
    fn convert<T: VectorRepr>(&mut self, vector: &[T], label: u32, id: u32) -> ANNResult<()> {
        let dimensions = T::full_dimension(vector).map_err(Into::<ANNError>::into)?;
        if dimensions != self.row.len() {
            return Err(ANNError::message(format!(
                "row {id} of label {label} has inconsistent full dimension"
            )));
        }
        T::as_f32_into(vector, &mut self.row)
            .map_err(Into::<ANNError>::into)
            .with_context(|| format!("converting entry-point row {id} of label {label}"))?;
        if self.row.iter().any(|value| !value.is_finite()) {
            return Err(ANNError::message(format!(
                "non-finite full-precision entry-point row {id} of label {label}"
            )));
        }
        for (destination, &value) in self.row64.iter_mut().zip(&self.row) {
            *destination = f64::from(value);
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
            return Err(ANNError::message(format!(
                "invalid centroid for label {label}"
            )));
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
                return Err(ANNError::message(format!(
                    "invalid centroid distance for row {id} of label {label}"
                )));
            }
            if distance < best.0 || (distance == best.0 && id < best.1) {
                best = (distance, id);
            }
        }
        Ok(best.1)
    }

    fn select_pool<T: VectorRepr>(
        &mut self,
        vectors: rowmajor::Ref<'_, T>,
        label: u32,
        members: &[u32],
    ) -> ANNResult<Vec<u32>> {
        let centroid = self.select(vectors, label, members)?;
        if members.len() <= ENTRY_CLUSTERS + 1 {
            let mut entries = vec![centroid];
            entries.extend(members.iter().copied().filter(|&id| id != centroid));
            return Ok(entries);
        }
        let centroid_local = members
            .binary_search(&centroid)
            .map_err(|_| ANNError::message("entry centroid is outside its label"))?;
        let mut sample = if members.len() > ENTRY_SAMPLE_LIMIT {
            let mut rng = StdRng::seed_from_u64(ENTRY_SAMPLE_SEED ^ u64::from(label));
            index::sample(&mut rng, members.len(), ENTRY_SAMPLE_LIMIT).into_vec()
        } else {
            (0..members.len()).collect()
        };
        sample.sort_unstable();
        if sample.binary_search(&centroid_local).is_err() {
            sample[ENTRY_SAMPLE_LIMIT - 1] = centroid_local;
            sample.sort_unstable();
        }
        let mut points = Vec::with_capacity(sample.len());
        for &local in &sample {
            self.convert(vectors.row(members[local] as usize), label, members[local])?;
            points.push(self.row64.clone());
        }
        let first = sample
            .binary_search(&centroid_local)
            .map_err(|_| ANNError::message("entry sample omitted its centroid"))?;
        let mut centers = vec![points[first].clone()];
        let mut nearest = vec![f64::INFINITY; points.len()];
        while centers.len() < ENTRY_CLUSTERS {
            let newest = &centers[centers.len() - 1];
            let mut farthest = (0., u32::MAX, 0);
            for (position, point) in points.iter().enumerate() {
                nearest[position] =
                    nearest[position].min(entry_squared_distance(point, newest, label)?);
                let id = members[sample[position]];
                if nearest[position] > farthest.0
                    || (nearest[position] == farthest.0 && id < farthest.1)
                {
                    farthest = (nearest[position], id, position);
                }
            }
            if farthest.0 == 0. {
                break;
            }
            centers.push(points[farthest.2].clone());
        }
        let mut sums = vec![vec![0.; self.row64.len()]; centers.len()];
        let mut counts = vec![0usize; centers.len()];
        for _ in 0..ENTRY_LLOYD_ITERATIONS {
            for sum in &mut sums {
                sum.fill(0.);
            }
            counts.fill(0);
            for point in &points {
                let (cluster, _) = nearest_entry_center(point, &centers, label)?;
                counts[cluster] += 1;
                for (sum, &value) in sums[cluster].iter_mut().zip(point) {
                    *sum += value;
                }
            }
            for ((center, sum), &count) in centers.iter_mut().zip(&sums).zip(&counts) {
                if count == 0 {
                    continue;
                }
                for (value, &total) in center.iter_mut().zip(sum) {
                    *value = total / count as f64;
                    if !value.is_finite() {
                        return Err(ANNError::message(format!(
                            "invalid entry cluster mean for label {label}"
                        )));
                    }
                }
            }
        }
        // The fit is bounded; representatives themselves are selected from every
        // eligible row, so a useful medoid need not occur in the training sample.
        let mut medoids = vec![(f64::INFINITY, u32::MAX); centers.len()];
        for &id in members {
            self.convert(vectors.row(id as usize), label, id)?;
            let (cluster, score) = nearest_entry_center(&self.row64, &centers, label)?;
            let best = &mut medoids[cluster];
            if score < best.0 || (score == best.0 && id < best.1) {
                *best = (score, id);
            }
        }
        let mut entries: Vec<_> = medoids
            .into_iter()
            .filter_map(|(_, id)| (id != u32::MAX && id != centroid).then_some(id))
            .collect();
        entries.sort_unstable();
        entries.dedup();
        entries.insert(0, centroid);
        Ok(entries)
    }
}

fn entry_squared_distance(left: &[f64], right: &[f64], label: u32) -> ANNResult<f64> {
    let value: f64 = left
        .iter()
        .zip(right)
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ANNError::message(format!(
            "invalid entry-pool distance for label {label}"
        )))
    }
}

fn nearest_entry_center(
    point: &[f64],
    centers: &[Vec<f64>],
    label: u32,
) -> ANNResult<(usize, f64)> {
    let mut best = (0, f64::INFINITY);
    for (cluster, center) in centers.iter().enumerate() {
        let score = entry_squared_distance(point, center, label)?;
        if score < best.1 {
            best = (cluster, score);
        }
    }
    Ok(best)
}

fn validate_vector_shape(rows: usize, dim: usize) -> ANNResult<()> {
    if rows == 0 || rows >= u32::MAX as usize || dim == 0 {
        return Err(ANNError::message(
            "filtered build requires nonzero dimensions and 0 < rows < u32::MAX",
        ));
    }
    Ok(())
}

fn validate_settings(rows: usize, dim: usize, settings: &StitchedBuildConfig) -> ANNResult<()> {
    validate_vector_shape(rows, dim)?;
    if settings.label_degree == 0
        || settings.max_degree == 0
        || settings.max_degree > u32::MAX as usize
        || settings.label_degree > u32::MAX as usize
        || settings.l_build > u32::MAX as usize
        || settings.l_build < settings.label_degree
        || settings.concurrency == 0
        || !settings.alpha.is_finite()
        || settings.alpha < 1.0
    {
        return Err(ANNError::message("invalid filtered build settings"));
    }
    Ok(())
}

fn validate_postings(rows: usize, postings: &[(u32, &[u32])]) -> ANNResult<()> {
    let mut previous_label = None;
    for &(label, members) in postings {
        if previous_label == Some(label) {
            return Err(ANNError::message(format!("duplicate label {label}")));
        }
        previous_label = Some(label);
        if members.is_empty()
            || members.last().is_some_and(|&id| id as usize >= rows)
            || members.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(ANNError::message(format!(
                "label {label} requires a nonempty, sorted, unique in-bounds posting list"
            )));
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
    fn new(rows: usize, postings: &[(u32, &[u32])]) -> ANNResult<Self> {
        let mut offsets = vec![0usize; rows + 1];
        for (_, members) in postings {
            for &id in *members {
                offsets[id as usize + 1] = offsets[id as usize + 1]
                    .checked_add(1)
                    .ok_or_else(|| ANNError::message("membership count overflow"))?;
            }
        }
        for i in 1..offsets.len() {
            offsets[i] = offsets[i]
                .checked_add(offsets[i - 1])
                .ok_or_else(|| ANNError::message("membership count overflow"))?;
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

async fn append_label_graph<T: VectorRepr>(
    vectors: rowmajor::Ref<'_, T>,
    label: u32,
    members: &[u32],
    entry: u32,
    settings: &StitchedBuildConfig,
    stitched: &mut [Vec<u32>],
) -> ANNResult<()> {
    let local_entry = members
        .binary_search(&entry)
        .map_err(|_| ANNError::message("construction entry is outside its label"))?;
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
    let params = DefaultProviderParameters::simple(
        members.len(),
        vectors.ncols(),
        Metric::L2,
        config.max_degree_u32().get(),
    );
    let index = diskann_async::new_index::<T, _>(config, params, NoDeletes)?;
    index
        .provider()
        .set_start_points(std::iter::once(entry_vector))?;
    index
        .insert(&FullPrecision, &DefaultContext, &entry_id, entry_vector)
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
        tasks.spawn(async move {
            index
                .insert(&FullPrecision, &DefaultContext, &local, &vector)
                .await
        });
    }
    while !tasks.is_empty() {
        join_one(&mut tasks).await?;
    }
    let total = members.len() + 1;
    let chunk = total.div_ceil(settings.concurrency);
    for start in (0..total).step_by(chunk) {
        let end = (start + chunk).min(total);
        let index = Arc::clone(&index);
        let range = u32::try_from(start)?..u32::try_from(end)?;
        tasks.spawn(async move {
            index
                .prune_range(&FullPrecision, &DefaultContext, range)
                .await
        });
    }
    while !tasks.is_empty() {
        join_one(&mut tasks).await?;
    }

    let mut virtual_neighbors = AdjacencyList::new();
    index
        .provider()
        .neighbors()
        .get_neighbors_sync(members.len(), &mut virtual_neighbors)?;
    let mut local_neighbors = AdjacencyList::new();
    for (local, &global) in members.iter().enumerate() {
        index
            .provider()
            .neighbors()
            .get_neighbors_sync(local, &mut local_neighbors)?;
        append_remapped_label_neighbors(
            local,
            local_entry,
            members,
            &virtual_neighbors,
            &local_neighbors,
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
) -> ANNResult<()> {
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
            return Err(ANNError::message("per-label Vamana emitted an invalid ID"));
        }
        if neighbor != local {
            stitched.push(members[neighbor]);
        }
    }
    Ok(())
}

async fn join_one(tasks: &mut JoinSet<ANNResult<()>>) -> ANNResult<()> {
    if let Some(result) = tasks.join_next().await {
        result.map_err(|error| {
            ANNError::message(format!("filtered Vamana insertion task failed: {error}"))
        })??;
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
) -> ANNResult<f32> {
    let value =
        distance.evaluate_similarity(vectors.row(left as usize), vectors.row(right as usize));
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(ANNError::message(format!(
            "invalid squared L2 distance between rows {left} and {right}"
        )))
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

    #[test]
    fn corpus_entry_pool_keeps_global_centroid_and_four_spatial_cluster_medoids() {
        let values = [
            -100f32, -99., -98., -10., -9., -8., 10., 11., 12., 100., 101., 102.,
        ];
        let vectors = rowmajor::Ref::column_vector(&values);
        let members: Vec<_> = (0..values.len() as u32).collect();
        let entries = select_label_entries(vectors, [(7, members.as_slice())]).unwrap();
        // Global mean is1; IDs5(-8) and6(10) tie, so the smaller ID remains
        // the construction fallback. Cluster means are -99,-9,11,101.
        assert_eq!(entries[&7], [5, 1, 4, 7, 10]);
    }

    #[test]
    fn corpus_entry_pool_small_labels_deduplicate_and_ignore_unrelated_vectors() {
        let values = [f32::NAN, 1., 9., 2., 10., 3., 11., 4.];
        let vectors = rowmajor::Ref::column_vector(&values);
        let entries =
            select_label_entries(vectors, [(7, &[1, 3, 5, 7][..]), (99, &[2][..])]).unwrap();
        assert_eq!(entries[&7], [3, 1, 5, 7]);
        assert_eq!(entries[&99], [2]);
        let identical = [f32::MAX; 8];
        let members: Vec<_> = (0..8).collect();
        let entries = select_label_entries(
            rowmajor::Ref::column_vector(&identical),
            [(7, members.as_slice())],
        )
        .unwrap();
        assert_eq!(entries[&7], [0]);
    }

    #[test]
    fn corpus_entry_pool_sampling_is_independent_of_label_enumeration_and_call_batching() {
        let values: Vec<_> = (0..5000).map(|id| ((id * 37) % 1009) as f32).collect();
        let vectors = rowmajor::Ref::column_vector(&values);
        let even: Vec<_> = (0..5000).step_by(2).collect();
        let all: Vec<_> = (0..5000).collect();
        let combined =
            select_label_entries(vectors, [(99, even.as_slice()), (7, all.as_slice())]).unwrap();
        let separate = select_label_entries(vectors, [(7, all.as_slice())]).unwrap();
        assert_eq!(combined[&7], separate[&7]);
        assert_eq!(
            combined[&99],
            select_label_entries(vectors, [(99, even.as_slice())]).unwrap()[&99]
        );
        for (label, members) in [(7, &all), (99, &even)] {
            let entries = &combined[&label];
            assert!(!entries.is_empty() && entries.len() <= ENTRY_CLUSTERS + 1);
            assert!(entries.iter().all(|id| members.binary_search(id).is_ok()));
            assert_eq!(entries.iter().collect::<BTreeSet<_>>().len(), entries.len());
            assert!(entries[1..].windows(2).all(|pair| pair[0] < pair[1]));
            let mut scratch = EntryScratch::new(vectors.row(0)).unwrap();
            assert_eq!(entries[0], scratch.select(vectors, label, members).unwrap());
        }
    }

    #[test]
    fn corpus_entry_pool_can_choose_a_medoid_outside_the_bounded_fit_sample() {
        let values: Vec<_> = (0..5000).map(|id| id as f32).collect();
        let vectors = rowmajor::Ref::column_vector(&values);
        let members: Vec<_> = (0..5000).collect();
        let mut found = false;
        for label in 0..16 {
            let entries = select_label_entries(vectors, [(label, members.as_slice())]).unwrap();
            let entries = &entries[&label];
            let mut rng = StdRng::seed_from_u64(ENTRY_SAMPLE_SEED ^ u64::from(label));
            let mut sample = index::sample(&mut rng, members.len(), ENTRY_SAMPLE_LIMIT).into_vec();
            sample.sort_unstable();
            if sample.binary_search(&(entries[0] as usize)).is_err() {
                sample[ENTRY_SAMPLE_LIMIT - 1] = entries[0] as usize;
                sample.sort_unstable();
            }
            if entries[1..]
                .iter()
                .any(|&id| sample.binary_search(&(id as usize)).is_err())
            {
                found = true;
                break;
            }
        }
        assert!(
            found,
            "full-label representatives must not be restricted to fit rows"
        );
    }

    #[test]
    fn corpus_entry_pool_uses_f64_for_extreme_values_and_rejects_invalid_members() {
        let values = [f32::MAX, -f32::MAX, f32::MAX, -f32::MAX, 0., 0., 0., 0.];
        let vectors = rowmajor::Ref::column_vector(&values);
        let members: Vec<_> = (0..8).collect();
        let entries = select_label_entries(vectors, [(7, members.as_slice())]).unwrap();
        assert_eq!(entries[&7][0], 4);
        assert!(entries[&7].contains(&0) && entries[&7].contains(&1));
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let values = [0., invalid];
            assert!(select_label_entries(
                rowmajor::Ref::column_vector(&values),
                [(7, &[0, 1][..])]
            )
            .is_err());
        }
        for invalid in [&[][..], &[0, 0][..], &[1, 0][..], &[8][..]] {
            assert!(select_label_entries(vectors, [(7, invalid)]).is_err());
        }
        assert!(select_label_entries(vectors, [(7, &[0][..]), (7, &[1][..])]).is_err());
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
        assert!(graph.neighbors[1].is_empty());
    }

    #[test]
    fn centroid_accumulation_handles_f32_sum_overflow_and_rejects_nonfinite_conversion() {
        let values = [f32::MAX; 3];
        let vectors = rowmajor::Ref::column_vector(&values);
        let mut scratch = EntryScratch::new(vectors.row(0)).unwrap();
        assert_eq!(scratch.select(vectors, 7, &[0, 1, 2]).unwrap(), 0);
        for invalid in [f32::NAN, f32::INFINITY] {
            let values = [1., invalid];
            assert!(scratch
                .select(rowmajor::Ref::column_vector(&values), 7, &[0, 1])
                .is_err());
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
        assert!(graph.neighbors[7].is_empty());
        assert_eq!(graph.neighbors.len(), 8);
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
        assert!(error.to_string().contains("protected label edges"));
    }

    #[tokio::test]
    async fn malformed_memberships_and_nonfinite_vectors_are_rejected() {
        let values = [0.0f32, 1.0, 2.0];
        let vectors = rowmajor::Ref::try_from_data(&values, values.len(), 1).unwrap();
        for members in [&[][..], &[0, 0][..], &[1, 0][..], &[0, 3][..]] {
            assert!(build_stitched(vectors, [(1, members)], &Default::default())
                .await
                .is_err());
        }
        assert!(
            build_stitched(vectors, [(1, &[0][..]), (1, &[1][..])], &Default::default())
                .await
                .is_err()
        );
        let invalid = [f32::NAN];
        assert!(build_stitched(
            rowmajor::Ref::try_from_data(&invalid, invalid.len(), 1).unwrap(),
            [(1, &[0][..])],
            &Default::default()
        )
        .await
        .is_err());
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
            assert!(build_stitched(vectors, [(1, &[0, 1][..])], &settings)
                .await
                .is_err());
        }
        let extreme = [f32::MAX, -f32::MAX];
        assert!(build_stitched(
            rowmajor::Ref::column_vector(&extreme),
            [(1, &[0, 1][..])],
            &defaults
        )
        .await
        .is_err());
    }

    #[test]
    fn occlusion_uses_shared_source_labels() {
        let membership = Membership::new(4, &[(1, &[0, 1, 2]), (2, &[1, 3])]).unwrap();
        // Candidate1 has labels1+2; source0 only requires label1, which selected2 covers.
        assert!(membership.covered(0, 1, 2));
        assert!(!membership.covered(0, 1, 3));
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
            assert!(members
                .iter()
                .filter(|&&member| member != id as u32)
                .all(|member| row.contains(member)));
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
