/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Durable immutable generations with a bounded, snapshot-consistent update delta.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Weak},
};

use arc_swap::ArcSwap;
use diskann::ANNResult;
use diskann_utils::views::rowmajor::{self, Matrix};
use serde::{Deserialize, Serialize};

use super::{
    metadata::LabelMetadata,
    native::{self, NativeVector},
    persistence::{GenerationStore, Journal, PreparedGeneration},
    updates::{DeltaLimits, DeltaSnapshot, Mutation},
    view::{ann, Base, FilteredSnapshot, View},
};
use crate::build::filtered::{build_stitched, StitchedBuildConfig};

/// Admission limits bound the mutable delta and its durable journal. Compact before
/// exceeding either limit. Compaction itself requires memory for the live base/build.
#[derive(Clone, Debug)]
pub struct FilteredIndexConfig {
    pub build: StitchedBuildConfig,
    pub pq_chunks: usize,
    /// Optional trained codebook to copy into the first generation. Compactions
    /// reuse the committed codebook; no external path is needed after publication.
    pub initial_pq_codebook: Option<PathBuf>,
    pub threads: usize,
    pub delta_limits: DeltaLimits,
    pub max_journal_bytes: u64,
    pub max_frame_bytes: usize,
}

impl Default for FilteredIndexConfig {
    fn default() -> Self {
        Self {
            build: StitchedBuildConfig::default(),
            pq_chunks: 32,
            initial_pq_codebook: None,
            threads: 8,
            delta_limits: DeltaLimits::default(),
            max_journal_bytes: 256 * 1024 * 1024,
            max_frame_bytes: 16 * 1024 * 1024,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    encoding: String,
    dimensions: usize,
    base_sequence: u64,
    next_id: u64,
    has_graph: bool,
    #[serde(default)]
    pq_reused: bool,
    // Persist the construction/admission settings used on the next compaction.
    label_degree: usize,
    max_degree: usize,
    l_build: usize,
    alpha: f32,
    concurrency: usize,
    seed: u64,
    pq_chunks: usize,
    max_records: usize,
    max_delta_bytes: usize,
    max_journal_bytes: u64,
    max_frame_bytes: usize,
}

impl Manifest {
    fn new<T: NativeVector>(
        dimensions: usize,
        sequence: u64,
        next_id: u64,
        has_graph: bool,
        config: &FilteredIndexConfig,
        pq_reused: bool,
    ) -> Self {
        Self {
            version: 1,
            encoding: T::ENCODING.into(),
            dimensions,
            base_sequence: sequence,
            next_id,
            has_graph,
            pq_reused,
            label_degree: config.build.label_degree,
            max_degree: config.build.max_degree,
            l_build: config.build.l_build,
            alpha: config.build.alpha,
            concurrency: config.build.concurrency,
            seed: config.build.seed,
            pq_chunks: config.pq_chunks,
            max_records: config.delta_limits.max_records,
            max_delta_bytes: config.delta_limits.max_bytes,
            max_journal_bytes: config.max_journal_bytes,
            max_frame_bytes: config.max_frame_bytes,
        }
    }
    fn config(&self, threads: usize) -> FilteredIndexConfig {
        FilteredIndexConfig {
            build: StitchedBuildConfig {
                label_degree: self.label_degree,
                max_degree: self.max_degree,
                l_build: self.l_build,
                alpha: self.alpha,
                concurrency: self.concurrency,
                seed: self.seed,
            },
            pq_chunks: self.pq_chunks,
            initial_pq_codebook: None,
            threads,
            delta_limits: DeltaLimits {
                max_records: self.max_records,
                max_bytes: self.max_delta_bytes,
            },
            max_journal_bytes: self.max_journal_bytes,
            max_frame_bytes: self.max_frame_bytes,
        }
    }
}

struct Writer<T> {
    journal: Journal<T>,
    poisoned: bool,
}

/// One writer owner with independent leased reader snapshots.
///
/// Building and `compact` perform CPU/SSD work on the calling task; invoke them on
/// a background runtime/thread in a serving process. Queries never take the writer
/// lock. Mutations are acknowledged only after their complete journal batch is fsynced.
/// The durable implementation currently requires Unix directory fsync semantics.
pub struct FilteredIndex<T: NativeVector> {
    store: Arc<GenerationStore>,
    current: ArcSwap<View<T>>,
    writer: Mutex<Writer<T>>,
    compaction: tokio::sync::Mutex<()>,
    retired: Mutex<Vec<Weak<Base<T>>>>,
    config: FilteredIndexConfig,
}

impl<T: NativeVector> FilteredIndex<T> {
    /// Build entirely with Rust Vamana, PQ training and SSD layout. Row IDs initially
    /// equal stable IDs. Empty posting lists are omitted; unlabeled rows are retained.
    pub async fn build(
        root: impl AsRef<Path>,
        vectors: rowmajor::Ref<'_, T>,
        mut postings: BTreeMap<u32, Vec<u32>>,
        config: FilteredIndexConfig,
    ) -> ANNResult<Self> {
        validate_config(&config, vectors.ncols())?;
        T::validate_vectors(vectors.as_slice(), vectors.ncols())?;
        postings.retain(|_, members| !members.is_empty());
        let store = Arc::new(GenerationStore::open(root).map_err(ann)?);
        if store.load_current().map_err(ann)?.is_some() {
            return Err(ann("filtered index already exists; use open"));
        }
        if store.root().join("updates.wal").try_exists()? {
            return Err(ann(
                "uncommitted journal exists; use a fresh index directory",
            ));
        }
        let next_id = vectors.nrows() as u64;
        let generation = next_generation(store.root())?;
        let (prepared, labels) = prepare(
            &store,
            (generation, 0, next_id),
            vectors,
            postings,
            None,
            &config,
            config.initial_pq_codebook.as_deref(),
        )
        .await?;
        let base = Arc::new(Base::open(
            prepared.directory(),
            (generation, 0),
            vectors.ncols(),
            labels,
            vectors.nrows() >= 256,
            config.threads,
            store.clone(),
        )?);
        let (journal, _) =
            Journal::create_new(store.root().join("updates.wal"), 0, config.max_frame_bytes)
                .map_err(ann)?;
        store.commit_prepared(prepared).map_err(ann)?;
        let delta = DeltaSnapshot::new(0, vectors.ncols()).map_err(ann)?;
        Ok(Self {
            store,
            current: ArcSwap::from_pointee(View::new(base, delta, next_id)),
            writer: Mutex::new(Writer {
                journal,
                poisoned: false,
            }),
            compaction: tokio::sync::Mutex::new(()),
            retired: Mutex::new(Vec::new()),
            config,
        })
    }

    /// Verify the committed generation and recover complete journal batches. Missing
    /// journals or complete corrupt frames are errors, never silently recreated.
    pub fn open(root: impl AsRef<Path>, threads: usize) -> ANNResult<Self> {
        let store = Arc::new(GenerationStore::open(root).map_err(ann)?);
        let loaded = store
            .load_current()
            .map_err(ann)?
            .ok_or_else(|| ann("no committed filtered index"))?;
        let manifest: Manifest = serde_json::from_slice(&loaded.manifest_payload).map_err(ann)?;
        if manifest.version != 1 || manifest.encoding != T::ENCODING {
            return Err(ann("incompatible filtered index format or scalar encoding"));
        }
        let config = manifest.config(threads);
        validate_config(&config, manifest.dimensions)?;
        let labels: LabelMetadata =
            bincode::deserialize(&fs::read(loaded.directory.join("labels.bin"))?).map_err(ann)?;
        if labels.stable_ids().is_some_and(|ids| {
            ids.last()
                .is_some_and(|&id| u64::from(id) >= manifest.next_id)
        }) || (labels.stable_ids().is_none() && labels.point_count() as u64 > manifest.next_id)
            || manifest.next_id > u64::from(u32::MAX) + 1
        {
            return Err(ann("invalid stable-ID allocation watermark"));
        }
        let base = Arc::new(Base::open(
            &loaded.directory,
            (loaded.generation, manifest.base_sequence),
            manifest.dimensions,
            labels,
            manifest.has_graph,
            threads,
            store.clone(),
        )?);
        let (journal, recovery) =
            Journal::open_existing(store.root().join("updates.wal"), config.max_frame_bytes)
                .map_err(ann)?;
        if recovery.base_sequence > manifest.base_sequence
            || recovery.last_sequence < manifest.base_sequence
        {
            return Err(ann("journal does not cover the committed base sequence"));
        }
        let mut delta =
            DeltaSnapshot::new(manifest.base_sequence, manifest.dimensions).map_err(ann)?;
        let mut next_id = manifest.next_id;
        for batch in recovery
            .batches
            .into_iter()
            .filter(|b| b.sequence > manifest.base_sequence)
        {
            validate_mutations(&batch.mutations)?;
            next_id = advance_ids(next_id, &batch.mutations)?;
            delta = delta
                .apply_batch(batch.sequence, &batch.mutations, config.delta_limits)
                .map_err(ann)?;
        }
        Ok(Self {
            store,
            current: ArcSwap::from_pointee(View::new(base, delta, next_id)),
            writer: Mutex::new(Writer {
                journal,
                poisoned: false,
            }),
            compaction: tokio::sync::Mutex::new(()),
            retired: Mutex::new(Vec::new()),
            config,
        })
    }

    pub fn snapshot(&self) -> FilteredSnapshot<T> {
        FilteredSnapshot {
            view: self.current.load_full(),
        }
    }

    /// Commit one atomic batch. IDs below the allocation watermark can be replaced;
    /// new explicit IDs advance it. Stable IDs are never reused by `insert`.
    pub fn apply(&self, mutations: &[Mutation<T>]) -> ANNResult<u64> {
        let mut writer = self.writer.lock().map_err(ann)?;
        self.apply_locked(&mut writer, mutations)
    }

    fn apply_locked(&self, writer: &mut Writer<T>, mutations: &[Mutation<T>]) -> ANNResult<u64> {
        if writer.poisoned {
            return Err(ann("writer has an uncertain durable outcome; reopen index"));
        }
        let current = self.current.load_full();
        validate_mutations(mutations)?;
        let sequence = current
            .delta
            .sequence()
            .checked_add(1)
            .ok_or_else(|| ann("sequence exhausted"))?;
        let next_id = advance_ids(current.next_id, mutations)?;
        let delta = current
            .delta
            .apply_batch(sequence, mutations, self.config.delta_limits)
            .map_err(ann)?;
        // Conservative bound, including framing; reject before an acknowledged state changes.
        let payload = bincode::serialized_size(mutations).map_err(ann)?;
        if payload > self.config.max_frame_bytes as u64
            || writer
                .journal
                .file_len()
                .map_err(ann)?
                .saturating_add(payload)
                .saturating_add(40)
                > self.config.max_journal_bytes
        {
            return Err(ann(
                "journal admission limit reached; compact before retrying",
            ));
        }
        let staged = Arc::new(current.after_batch(delta, next_id, mutations));
        if let Err(error) = writer.journal.append(sequence, mutations) {
            writer.poisoned = true;
            return Err(ann(error));
        }
        self.current.store(staged);
        Ok(sequence)
    }

    pub fn insert(&self, vector: Vec<T>, labels: Vec<u32>) -> ANNResult<(u32, u64)> {
        let mut writer = self.writer.lock().map_err(ann)?;
        let id = u32::try_from(self.current.load().next_id).map_err(ann)?;
        let sequence =
            self.apply_locked(&mut writer, &[Mutation::Upsert { id, vector, labels }])?;
        Ok((id, sequence))
    }

    pub fn replace_labels(&self, id: u32, labels: Vec<u32>) -> ANNResult<u64> {
        let mut writer = self.writer.lock().map_err(ann)?;
        let (vector, _) = self
            .snapshot()
            .get(id)?
            .ok_or_else(|| ann("stable ID does not exist"))?;
        self.apply_locked(&mut writer, &[Mutation::Upsert { id, vector, labels }])
    }

    pub fn replace_vector(&self, id: u32, vector: Vec<T>) -> ANNResult<u64> {
        let mut writer = self.writer.lock().map_err(ann)?;
        let (_, labels) = self
            .snapshot()
            .get(id)?
            .ok_or_else(|| ann("stable ID does not exist"))?;
        self.apply_locked(&mut writer, &[Mutation::Upsert { id, vector, labels }])
    }

    /// Build a fresh immutable base while queries/mutations continue, then replay
    /// changes newer than the captured sequence and atomically publish the generation.
    /// Journal checkpointing occurs only after the new base is durably committed.
    pub async fn compact(&self) -> ANNResult<u64> {
        self.compact_with_validation(|_| Ok(())).await
    }

    /// Validate the rebuilt snapshot before publication, for example against held-out
    /// per-label recall/latency floors. Rejection keeps CURRENT and acknowledged updates
    /// unchanged. The candidate represents the captured sequence; newer mutations are
    /// replayed only after validation. The callback must not mutate this index.
    pub async fn compact_with_validation<F>(&self, validate: F) -> ANNResult<u64>
    where
        F: FnOnce(&FilteredSnapshot<T>) -> ANNResult<()>,
    {
        let _compaction = self.compaction.lock().await;
        let captured = self.snapshot();
        let (values, stable_ids, postings) = materialize(&captured)?;
        let dimensions = captured.view.base.dimensions;
        let vectors = rowmajor::Ref::try_from_data(values.as_slice(), stable_ids.len(), dimensions)
            .map_err(ann)?;
        let generation = next_generation(self.store.root())?;
        let sequence = captured.sequence();
        let committed_codebook = captured.view.base.directory.join("index_pq_pivots.bin");
        let codebook = committed_codebook
            .try_exists()?
            .then_some(committed_codebook);
        let (prepared, labels) = prepare(
            &self.store,
            (generation, sequence, captured.view.next_id),
            vectors,
            postings,
            Some(stable_ids),
            &self.config,
            codebook.as_deref(),
        )
        .await?;
        let base = Arc::new(Base::open(
            prepared.directory(),
            (generation, sequence),
            dimensions,
            labels,
            vectors.nrows() >= 256,
            self.config.threads,
            self.store.clone(),
        )?);
        // Validation may retain a clone even when rejecting publication.
        self.retired
            .lock()
            .map_err(ann)?
            .push(Arc::downgrade(&base));
        validate(&FilteredSnapshot {
            view: Arc::new(View::new(
                base.clone(),
                DeltaSnapshot::new(sequence, dimensions).map_err(ann)?,
                captured.view.next_id,
            )),
        })?;
        let mut writer = self.writer.lock().map_err(ann)?;
        if writer.poisoned {
            return Err(ann("reopen index before compaction publication"));
        }
        let mut delta = DeltaSnapshot::new(sequence, dimensions).map_err(ann)?;
        let catchup = match writer.journal.replay_after(sequence) {
            Ok(batches) => batches,
            Err(error) => {
                writer.poisoned = true;
                return Err(ann(error));
            }
        };
        for batch in catchup {
            validate_mutations(&batch.mutations)?;
            delta = delta
                .apply_batch(batch.sequence, &batch.mutations, self.config.delta_limits)
                .map_err(ann)?;
        }
        let current = self.current.load_full();
        if delta.sequence() != current.delta.sequence() {
            return Err(ann("compaction catchup sequence mismatch"));
        }
        if let Err(error) = self.store.commit_prepared(prepared) {
            writer.poisoned = true;
            return Err(ann(error));
        }
        self.retired
            .lock()
            .map_err(ann)?
            .push(Arc::downgrade(&current.base));
        self.current
            .store(Arc::new(View::new(base, delta, current.next_id)));
        if let Err(error) = writer.journal.checkpoint(sequence) {
            writer.poisoned = true;
            return Err(ann(error));
        }
        Ok(generation)
    }

    /// Delete retired/orphan generations only after their final reader lease ends,
    /// and abandoned staging directories. The count includes both kinds of directory.
    /// Busy compactions leave garbage collection to a subsequent call.
    pub fn collect_garbage(&self) -> ANNResult<usize> {
        let _compaction = self.compaction.try_lock().map_err(ann)?;
        if self.writer.lock().map_err(ann)?.poisoned {
            return Err(ann(
                "reopen index before collecting generations after an uncertain durable write",
            ));
        }
        let current = self.current.load_full();
        let mut retired = self.retired.lock().map_err(ann)?;
        let leased: BTreeSet<_> = retired
            .iter()
            .filter_map(Weak::upgrade)
            .map(|base| base.generation)
            .collect();
        retired.retain(|base| base.strong_count() > 0);
        let mut removed = 0;
        for entry in fs::read_dir(self.store.root())? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if !entry.file_type()?.is_dir() {
                continue;
            }
            if is_staging_directory(name) {
                fs::remove_dir_all(entry.path())?;
                removed += 1;
                continue;
            }
            let Some(id) = name
                .strip_prefix("generation-")
                .and_then(|s| s.parse::<u64>().ok())
            else {
                continue;
            };
            if id != current.base.generation && !leased.contains(&id) && entry.file_type()?.is_dir()
            {
                fs::remove_dir_all(entry.path())?;
                removed += 1;
            }
        }
        Ok(removed)
    }
}

fn validate_config(config: &FilteredIndexConfig, dimensions: usize) -> ANNResult<()> {
    if dimensions == 0
        || dimensions > u32::MAX as usize
        || config.threads == 0
        || config.pq_chunks == 0
        || config.pq_chunks > dimensions
        || config.max_frame_bytes < 64
        || config.max_journal_bytes < (config.max_frame_bytes as u64).saturating_add(72)
        || config.delta_limits.max_records == 0
        || config.delta_limits.max_bytes == 0
        || config.build.label_degree == 0
        || config.build.max_degree == 0
        || config.build.max_degree > u32::MAX as usize
        || config.build.label_degree > u32::MAX as usize
        || config.build.l_build > u32::MAX as usize
        || config.build.l_build < config.build.label_degree
        || config.build.concurrency == 0
        || !config.build.alpha.is_finite()
        || config.build.alpha < 1.0
    {
        return Err(ann("invalid native filtered-index settings"));
    }
    Ok(())
}

fn advance_ids<T>(mut next_id: u64, mutations: &[Mutation<T>]) -> ANNResult<u64> {
    for mutation in mutations {
        if let Mutation::Upsert { id, .. } = mutation {
            next_id = next_id.max(u64::from(*id) + 1);
        } else if u64::from(mutation.id()) >= next_id {
            return Err(ann("cannot delete an unallocated ID"));
        }
    }
    Ok(next_id)
}

fn validate_mutations<T: NativeVector>(mutations: &[Mutation<T>]) -> ANNResult<()> {
    for mutation in mutations {
        if let Mutation::Upsert { vector, .. } = mutation {
            T::validate(vector)?;
        }
    }
    Ok(())
}

fn is_staging_directory(name: &str) -> bool {
    let Some(suffix) = name.strip_prefix(".staging-") else {
        return false;
    };
    let parts: Vec<_> = suffix.split('-').collect();
    parts.len() == 3
        && parts.iter().all(|part| {
            part.parse::<u64>()
                .is_ok_and(|value| value.to_string() == *part)
        })
}

fn next_generation(root: &Path) -> ANNResult<u64> {
    let mut next = 1;
    for entry in fs::read_dir(root)? {
        let name = entry?.file_name();
        if let Some(id) = name
            .to_str()
            .and_then(|n| n.strip_prefix("generation-"))
            .and_then(|s| s.parse::<u64>().ok())
        {
            next = next.max(
                id.checked_add(1)
                    .ok_or_else(|| ann("generation exhausted"))?,
            );
        }
    }
    Ok(next)
}

async fn prepare<T: NativeVector>(
    store: &GenerationStore,
    checkpoint: (u64, u64, u64),
    vectors: rowmajor::Ref<'_, T>,
    postings: BTreeMap<u32, Vec<u32>>,
    stable_ids: Option<Vec<u32>>,
    config: &FilteredIndexConfig,
    codebook: Option<&Path>,
) -> ANNResult<(PreparedGeneration, LabelMetadata)> {
    let (generation, sequence, next_id) = checkpoint;
    let graph = if vectors.nrows() == 0 {
        None
    } else {
        Some(
            build_stitched(
                vectors,
                postings.iter().map(|(&label, ids)| (label, ids.as_slice())),
                &config.build,
            )
            .await?,
        )
    };
    let entries = graph
        .as_ref()
        .map(|g| g.entry_points.clone())
        .unwrap_or_default();
    let mut labels =
        LabelMetadata::from_postings(vectors.nrows(), postings, entries).map_err(ann)?;
    if let Some(ids) = stable_ids {
        labels = labels.with_stable_ids(ids).map_err(ann)?;
    }
    let metadata = bincode::serialize(&labels).map_err(ann)?;
    let manifest = Manifest::new::<T>(
        vectors.ncols(),
        sequence,
        next_id,
        vectors.nrows() >= 256,
        config,
        codebook.is_some(),
    );
    let payload = serde_json::to_vec(&manifest).map_err(ann)?;
    let prepared = store
        .prepare_generation(generation, &payload, |directory| {
            let result = (|| -> ANNResult<()> {
                let mut file = BufWriter::new(File::create(directory.join("vectors.bin"))?);
                file.write_all(&u32::try_from(vectors.nrows()).map_err(ann)?.to_le_bytes())?;
                file.write_all(&u32::try_from(vectors.ncols()).map_err(ann)?.to_le_bytes())?;
                file.write_all(bytemuck::cast_slice(vectors.as_slice()))?;
                file.flush()?;
                fs::write(directory.join("labels.bin"), &metadata)?;
                if vectors.nrows() >= 256 {
                    native::write_native_with_codebook(
                        directory,
                        vectors,
                        graph.as_ref().ok_or_else(|| ann("missing graph"))?,
                        config.pq_chunks,
                        config.threads,
                        config.build.seed,
                        codebook,
                    )?;
                } else if let Some(codebook) = codebook {
                    native::copy_codebook(directory, vectors.ncols(), config.pq_chunks, codebook)?;
                }
                Ok(())
            })();
            result.map_err(std::io::Error::other)
        })
        .map_err(ann)?;
    Ok((prepared, labels))
}

type Materialized<T> = (Vec<T>, Vec<u32>, BTreeMap<u32, Vec<u32>>);

fn materialize<T: NativeVector>(snapshot: &FilteredSnapshot<T>) -> ANNResult<Materialized<T>> {
    let base = &snapshot.view.base;
    let mut ids = BTreeSet::new();
    for ordinal in 0..base.labels.point_count() {
        ids.insert(
            base.labels
                .stable_id(ordinal as u32)
                .ok_or_else(|| ann("missing stable ID"))?,
        );
    }
    for (id, record) in snapshot.view.delta.iter() {
        if record.is_deleted() {
            ids.remove(&id);
        } else {
            ids.insert(id);
        }
    }
    let mut values = Vec::new();
    let mut stable_ids = Vec::with_capacity(ids.len());
    let mut postings: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    // Keep one file descriptor; stable base IDs/ordinals are monotonic.
    use std::io::{Read, Seek, SeekFrom};
    let mut file = File::open(base.directory.join("vectors.bin"))?;
    let mut vector = vec![T::default(); base.dimensions];
    for id in ids {
        let labels = if let Some(point) = snapshot.view.delta.get(id).and_then(|r| r.point()) {
            values.extend_from_slice(&point.vector);
            point.labels.as_slice()
        } else {
            let ordinal = base
                .labels
                .ordinal(id)
                .ok_or_else(|| ann("live ID missing from base"))?;
            file.seek(SeekFrom::Start(
                8 + ordinal as u64 * base.dimensions as u64 * size_of::<T>() as u64,
            ))?;
            file.read_exact(bytemuck::cast_slice_mut(&mut vector))?;
            values.extend_from_slice(&vector);
            base.labels.labels(ordinal)
        };
        let ordinal = u32::try_from(stable_ids.len()).map_err(ann)?;
        for &label in labels {
            postings.entry(label).or_default().push(ordinal);
        }
        stable_ids.push(id);
    }
    Ok((values, stable_ids, postings))
}
