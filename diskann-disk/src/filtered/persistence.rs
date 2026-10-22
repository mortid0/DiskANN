/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Local durable journals and immutable generations. Checksums detect accidental
//! corruption; they do not authenticate files written by an untrusted party.
//!
//! Publication requires a local filesystem with atomic same-directory rename and
//! directory fsync. Old generations are retained: their reclamation must respect
//! reader leases. A failed commit can have an uncertain outcome and must be reloaded.

use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    marker::PhantomData,
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
};

use bincode::Options;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::updates::Mutation;

const VERSION: u32 = 1;
const JOURNAL_MAGIC: &[u8; 8] = b"DFWAL001";
const FRAME_MAGIC: &[u8; 8] = b"DFTXN001";
const CATALOG_MAGIC: &[u8; 8] = b"DFCAT001";
const CURRENT_MAGIC: &[u8; 8] = b"DFCUR001";
const JOURNAL_HEADER: usize = 32;
const FRAME_HEADER: usize = 40;
const ENVELOPE_HEADER: usize = 40;
const CATALOG_LIMIT: usize = 32 * 1024 * 1024;
const CURRENT_LIMIT: usize = 1024;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A complete, committed transaction. Sequence numbers advance once per batch.
#[derive(Debug)]
pub struct JournalBatch<T> {
    pub sequence: u64,
    pub mutations: Vec<Mutation<T>>,
}

#[derive(Debug)]
pub struct Recovery<T> {
    pub base_sequence: u64,
    pub last_sequence: u64,
    pub batches: Vec<JournalBatch<T>>,
    /// Only incomplete final headers/payloads are truncated, never bad checksums.
    pub truncated_bytes: u64,
}

/// An exclusively locked append journal. The enclosing lifecycle validates and
/// stages each batch before append, then publishes its reader snapshot after fsync.
/// Scalar type/dimension compatibility belongs to the generation manifest.
pub struct Journal<T> {
    file: File,
    path: PathBuf,
    last_sequence: u64,
    max_frame_bytes: usize,
    poisoned: bool,
    marker: PhantomData<T>,
}

impl<T: Serialize + DeserializeOwned> Journal<T> {
    /// Create or recover a journal. `base_sequence` is used only for a new file.
    /// Use `open_existing` when a committed manifest already references the file.
    pub fn open(
        path: impl AsRef<Path>,
        base_sequence: u64,
        max_frame_bytes: usize,
    ) -> io::Result<(Self, Recovery<T>)> {
        match Self::create_new(path.as_ref(), base_sequence, max_frame_bytes) {
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                Self::open_existing(path, max_frame_bytes)
            }
            result => result,
        }
    }

    pub fn create_new(
        path: impl AsRef<Path>,
        base_sequence: u64,
        max_frame_bytes: usize,
    ) -> io::Result<(Self, Recovery<T>)> {
        require_directory_sync()?;
        check_frame_limit(max_frame_bytes)?;
        let path = path.as_ref();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)?;
        lock(&file)?;
        let mut header = Vec::with_capacity(JOURNAL_HEADER);
        header.extend_from_slice(JOURNAL_MAGIC);
        header.extend_from_slice(&VERSION.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&base_sequence.to_le_bytes());
        header.extend_from_slice(&checksum(&header).to_le_bytes());
        file.write_all(&header)?;
        file.sync_all()?;
        sync_directory(parent(path))?;
        Ok((
            Self {
                file,
                path: fs::canonicalize(path)?,
                last_sequence: base_sequence,
                max_frame_bytes,
                poisoned: false,
                marker: PhantomData,
            },
            Recovery {
                base_sequence,
                last_sequence: base_sequence,
                batches: Vec::new(),
                truncated_bytes: 0,
            },
        ))
    }

    /// Recover complete frames and remove an incomplete final append. A complete
    /// frame with a corrupt checksum, length, encoding, or sequence is rejected.
    pub fn open_existing(
        path: impl AsRef<Path>,
        max_frame_bytes: usize,
    ) -> io::Result<(Self, Recovery<T>)> {
        check_frame_limit(max_frame_bytes)?;
        let path = path.as_ref();
        if !fs::symlink_metadata(path)?.is_file() {
            return Err(invalid_data("journal must be a regular file"));
        }
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        lock(&file)?;
        let recovery = recover(&mut file, max_frame_bytes, true)?;
        Ok((
            Self {
                file,
                path: fs::canonicalize(path)?,
                last_sequence: recovery.last_sequence,
                max_frame_bytes,
                poisoned: false,
                marker: PhantomData,
            },
            recovery,
        ))
    }

    pub fn last_sequence(&self) -> u64 {
        self.last_sequence
    }

    pub fn file_len(&self) -> io::Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    /// Append and fsync one atomic batch. If I/O fails, reopening is required
    /// because a complete frame may have reached durable storage without an ack.
    pub fn append(&mut self, sequence: u64, mutations: &[Mutation<T>]) -> io::Result<()> {
        if self.poisoned {
            return Err(io::Error::other(
                "journal has an uncertain write outcome; reopen it",
            ));
        }
        if self.last_sequence.checked_add(1) != Some(sequence) || mutations.is_empty() {
            return Err(invalid_input(
                "journal batch must be nonempty and have the next sequence",
            ));
        }
        let payload = codec(self.max_frame_bytes)
            .serialize(mutations)
            .map_err(|error| invalid_input(format!("encode journal batch: {error}")))?;
        if payload.len() > self.max_frame_bytes {
            return Err(invalid_input("journal batch exceeds max_frame_bytes"));
        }
        let mut header = Vec::with_capacity(FRAME_HEADER);
        header.extend_from_slice(FRAME_MAGIC);
        header.extend_from_slice(&sequence.to_le_bytes());
        header.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        header.extend_from_slice(&checksum(&payload).to_le_bytes());
        header.extend_from_slice(&checksum(&header).to_le_bytes());
        let result = (|| {
            self.file.seek(SeekFrom::End(0))?;
            self.file.write_all(&header)?;
            self.file.write_all(&payload)?;
            self.file.sync_all()
        })();
        if result.is_err() {
            self.poisoned = true;
        } else {
            self.last_sequence = sequence;
        }
        result
    }

    /// Catch up a compacted base snapshot from this journal, without discarding
    /// earlier frames that another persisted generation may still require.
    pub fn replay_after(&mut self, sequence: u64) -> io::Result<Vec<JournalBatch<T>>> {
        if self.poisoned {
            return Err(io::Error::other(
                "journal has an uncertain write outcome; reopen it",
            ));
        }
        let recovery = recover(&mut self.file, self.max_frame_bytes, false)?;
        if sequence < recovery.base_sequence || sequence > recovery.last_sequence {
            return Err(invalid_input(
                "requested replay sequence is outside the journal history",
            ));
        }
        Ok(recovery
            .batches
            .into_iter()
            .filter(|batch| batch.sequence > sequence)
            .collect())
    }

    /// Atomically discard history incorporated into a durably published base.
    /// All later committed batches are preserved. Call only after CURRENT's base
    /// sequence is durable, while holding the lifecycle's mutation lock. Every
    /// checkpoint error poisons this writer, requiring reload before further use.
    pub fn checkpoint(&mut self, base_sequence: u64) -> io::Result<()> {
        if self.poisoned {
            return Err(io::Error::other(
                "journal has an uncertain write outcome; reopen it",
            ));
        }
        let result = self.checkpoint_inner(base_sequence);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn checkpoint_inner(&mut self, base_sequence: u64) -> io::Result<()> {
        require_directory_sync()?;
        let recovery: Recovery<T> = recover(&mut self.file, self.max_frame_bytes, false)?;
        if base_sequence < recovery.base_sequence || base_sequence > recovery.last_sequence {
            return Err(invalid_input(
                "checkpoint sequence is outside the journal history",
            ));
        }
        if base_sequence == recovery.base_sequence {
            return Ok(());
        }
        let temporary = parent(&self.path).join(format!(
            ".journal-checkpoint-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let mut replacement = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        lock(&replacement)?;
        let mut header = Vec::with_capacity(JOURNAL_HEADER);
        header.extend_from_slice(JOURNAL_MAGIC);
        header.extend_from_slice(&VERSION.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&base_sequence.to_le_bytes());
        header.extend_from_slice(&checksum(&header).to_le_bytes());
        replacement.write_all(&header)?;
        for batch in recovery
            .batches
            .into_iter()
            .filter(|batch| batch.sequence > base_sequence)
        {
            let payload = codec(self.max_frame_bytes)
                .serialize(&batch.mutations)
                .map_err(|error| invalid_data(format!("encode checkpoint batch: {error}")))?;
            let mut header = Vec::with_capacity(FRAME_HEADER);
            header.extend_from_slice(FRAME_MAGIC);
            header.extend_from_slice(&batch.sequence.to_le_bytes());
            header.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            header.extend_from_slice(&checksum(&payload).to_le_bytes());
            header.extend_from_slice(&checksum(&header).to_le_bytes());
            replacement.write_all(&header)?;
            replacement.write_all(&payload)?;
        }
        replacement.sync_all()?;
        fs::rename(&temporary, &self.path)?;
        // The new inode was locked before rename, so the path never points to an
        // unlocked journal. Retain that lock even when the directory fsync fails.
        self.file = replacement;
        sync_directory(parent(&self.path))?;
        self.last_sequence = recovery.last_sequence;
        Ok(())
    }
}

fn recover<T: DeserializeOwned>(
    file: &mut File,
    max_frame_bytes: usize,
    truncate_tail: bool,
) -> io::Result<Recovery<T>> {
    file.seek(SeekFrom::Start(0))?;
    let mut header = [0; JOURNAL_HEADER];
    file.read_exact(&mut header)
        .map_err(|error| invalid_data(format!("incomplete journal header: {error}")))?;
    if &header[..8] != JOURNAL_MAGIC
        || number32(&header[8..12]) != VERSION
        || number32(&header[12..16]) != 0
        || checksum(&header[..24]) != number64(&header[24..32])
    {
        return Err(invalid_data("invalid journal header/version/checksum"));
    }
    let base_sequence = number64(&header[16..24]);
    let mut last_sequence = base_sequence;
    let mut batches = Vec::new();
    let total = file.metadata()?.len();
    let mut good_end = JOURNAL_HEADER as u64;
    loop {
        let remaining = total.saturating_sub(good_end);
        if remaining == 0 {
            break;
        }
        if remaining < FRAME_HEADER as u64 {
            break;
        }
        let mut header = [0; FRAME_HEADER];
        file.read_exact(&mut header)?;
        if &header[..8] != FRAME_MAGIC || checksum(&header[..32]) != number64(&header[32..40]) {
            return Err(invalid_data(format!(
                "corrupt journal frame header at byte {good_end}"
            )));
        }
        let sequence = number64(&header[8..16]);
        let length = number64(&header[16..24]);
        if last_sequence.checked_add(1) != Some(sequence) {
            return Err(invalid_data(format!(
                "noncontiguous journal sequence {sequence}"
            )));
        }
        if length > max_frame_bytes as u64 {
            return Err(invalid_data(format!(
                "journal frame {sequence} exceeds max_frame_bytes"
            )));
        }
        if length > remaining - FRAME_HEADER as u64 {
            break;
        }
        let mut payload = vec![0; length as usize];
        file.read_exact(&mut payload)?;
        if checksum(&payload) != number64(&header[24..32]) {
            return Err(invalid_data(format!(
                "corrupt journal frame {sequence} payload checksum"
            )));
        }
        let mutations: Vec<Mutation<T>> = codec(max_frame_bytes)
            .deserialize(&payload)
            .map_err(|error| invalid_data(format!("invalid journal frame {sequence}: {error}")))?;
        if mutations.is_empty() {
            return Err(invalid_data("journal contains an empty batch"));
        }
        batches.push(JournalBatch {
            sequence,
            mutations,
        });
        last_sequence = sequence;
        good_end += FRAME_HEADER as u64 + length;
    }
    let truncated_bytes = total - good_end;
    if truncated_bytes > 0 {
        if !truncate_tail {
            return Err(invalid_data(
                "journal has an incomplete tail; reopen it to recover",
            ));
        }
        file.set_len(good_end)?;
        file.sync_all()?;
    }
    file.seek(SeekFrom::End(0))?;
    Ok(Recovery {
        base_sequence,
        last_sequence,
        batches,
        truncated_bytes,
    })
}

/// File inventory stored in the generation's checksummed catalog.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Artifact {
    pub name: String,
    pub bytes: u64,
    pub checksum: u64,
}

#[derive(Debug)]
pub struct LoadedGeneration {
    pub generation: u64,
    pub manifest_payload: Vec<u8>,
    pub directory: PathBuf,
    pub artifacts: Vec<Artifact>,
}

/// Fully written and fsynced generation awaiting only the CURRENT swap. Preparing
/// can run while updates continue; commit follows the caller's catchup validation.
pub struct PreparedGeneration {
    root: PathBuf,
    loaded: LoadedGeneration,
    catalog_checksum: u64,
}

impl PreparedGeneration {
    pub fn generation(&self) -> u64 {
        self.loaded.generation
    }

    pub fn directory(&self) -> &Path {
        &self.loaded.directory
    }
}

#[derive(Serialize, Deserialize)]
struct Catalog {
    generation: u64,
    manifest_payload: Vec<u8>,
    artifacts: Vec<Artifact>,
}

/// Exclusive generation writer. The lock is held for this value's lifetime and
/// released by the OS on process exit. Readers can use `read_current` independently.
pub struct GenerationStore {
    root: PathBuf,
    _writer_lock: File,
    publication: Mutex<()>,
}

/// Remove only the unpublished staging name, including on callback errors/unwind.
/// After rename that name is absent; a prepared final generation is never removed.
struct StagingCleanup(PathBuf);

impl Drop for StagingCleanup {
    fn drop(&mut self) {
        // A failed cleanup can be retried by explicit crash-orphan collection.
        let _ = fs::remove_dir_all(&self.0);
    }
}

impl GenerationStore {
    pub fn open(root: impl AsRef<Path>) -> io::Result<Self> {
        require_directory_sync()?;
        let root = root.as_ref().to_path_buf();
        create_directory_tree(&root)?;
        // Reader snapshots open generation artifacts lazily, so their paths must
        // remain valid if the application later changes its working directory.
        let root = fs::canonicalize(root)?;
        let writer_lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join(".writer.lock"))?;
        lock(&writer_lock)?;
        sync_directory(&root)?;
        sync_directory(parent(&root))?;
        Ok(Self {
            root,
            _writer_lock: writer_lock,
            publication: Mutex::new(()),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Choose an unused monotonic generation after both published generations and
    /// crash-orphans. This is not a reservation; the enclosing lifecycle serializes
    /// compaction attempts. Orphaned directories are retained for explicit cleanup.
    pub fn next_generation(&self) -> io::Result<u64> {
        let mut maximum = 0u64;
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(number) = name.strip_prefix("generation-") else {
                continue;
            };
            if let Ok(generation) = number.parse::<u64>() {
                if generation_name(generation) == name {
                    maximum = maximum.max(generation);
                }
            }
        }
        maximum
            .checked_add(1)
            .ok_or_else(|| invalid_data("generation number space exhausted"))
    }

    pub fn load_current(&self) -> io::Result<Option<LoadedGeneration>> {
        Self::read_current(&self.root)
    }

    /// Read and verify one immutable generation. CURRENT is read once; a concurrent
    /// publication cannot change the generation already selected by this read.
    pub fn read_current(root: impl AsRef<Path>) -> io::Result<Option<LoadedGeneration>> {
        let root = match fs::canonicalize(root) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            result => result?,
        };
        Self::read_current_inner(&root, true)
    }

    fn read_current_inner(root: &Path, verify_files: bool) -> io::Result<Option<LoadedGeneration>> {
        let current = match read_envelope(&root.join("CURRENT"), CURRENT_MAGIC, CURRENT_LIMIT) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            result => result?,
        };
        if current.len() != 16 {
            return Err(invalid_data(
                "CURRENT payload must contain generation and catalog checksum",
            ));
        }
        let generation = number64(&current[..8]);
        let directory = root.join(generation_name(generation));
        ensure_directory(&directory)?;
        let catalog_path = directory.join("CATALOG");
        let catalog_bytes = read_envelope(&catalog_path, CATALOG_MAGIC, CATALOG_LIMIT)?;
        if checksum(&catalog_bytes) != number64(&current[8..16]) {
            return Err(invalid_data(
                "CURRENT and generation catalog checksums disagree",
            ));
        }
        let catalog: Catalog = codec(CATALOG_LIMIT)
            .deserialize(&catalog_bytes)
            .map_err(|error| invalid_data(format!("invalid generation catalog: {error}")))?;
        if catalog.generation != generation {
            return Err(invalid_data(
                "CURRENT and catalog generation numbers disagree",
            ));
        }
        if verify_files {
            verify_artifacts(&directory, &catalog.artifacts)?;
        }
        Ok(Some(LoadedGeneration {
            generation,
            manifest_payload: catalog.manifest_payload,
            directory,
            artifacts: catalog.artifacts,
        }))
    }

    /// Write artifacts into a private directory, fsync and checksum them, then
    /// atomically publish CURRENT. The callback must finish all file writes before
    /// returning. Existing generation directories are never overwritten or deleted.
    pub fn publish(
        &self,
        generation: u64,
        manifest_payload: &[u8],
        write_artifacts: impl FnOnce(&Path) -> io::Result<()>,
    ) -> io::Result<LoadedGeneration> {
        let prepared = self.prepare_generation(generation, manifest_payload, write_artifacts)?;
        self.commit_prepared(prepared)
    }

    /// Prepare and fsync a generation without changing CURRENT. This phase can be
    /// expensive and belongs outside the enclosing lifecycle's update lock.
    pub fn prepare_generation(
        &self,
        generation: u64,
        manifest_payload: &[u8],
        write_artifacts: impl FnOnce(&Path) -> io::Result<()>,
    ) -> io::Result<PreparedGeneration> {
        let _guard = self
            .publication
            .lock()
            .map_err(|_| io::Error::other("publication lock poisoned"))?;
        if let Some(current) = Self::read_current_inner(&self.root, false)? {
            if generation <= current.generation {
                return Err(invalid_input(
                    "published generation must increase monotonically",
                ));
            }
        }
        let final_directory = self.root.join(generation_name(generation));
        if final_directory.try_exists()? {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "generation directory already exists",
            ));
        }
        let suffix = format!(
            "{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let staging = self.root.join(format!(".staging-{generation}-{suffix}"));
        fs::create_dir(&staging)?;
        let _cleanup = StagingCleanup(staging.clone());
        write_artifacts(&staging)?;
        if staging.join("CATALOG").try_exists()? {
            return Err(invalid_input(
                "CATALOG is reserved for the generation store",
            ));
        }
        let artifacts = collect_artifacts(&staging, true)?;
        let catalog = Catalog {
            generation,
            manifest_payload: manifest_payload.to_vec(),
            artifacts: artifacts.clone(),
        };
        let catalog_bytes = codec(CATALOG_LIMIT)
            .serialize(&catalog)
            .map_err(|error| invalid_input(format!("encode generation catalog: {error}")))?;
        write_envelope(&staging.join("CATALOG"), CATALOG_MAGIC, &catalog_bytes)?;
        sync_directory(&staging)?;
        fs::rename(&staging, &final_directory)?;
        sync_directory(&self.root)?;

        Ok(PreparedGeneration {
            root: self.root.clone(),
            loaded: LoadedGeneration {
                generation,
                manifest_payload: manifest_payload.to_vec(),
                directory: final_directory,
                artifacts,
            },
            catalog_checksum: checksum(&catalog_bytes),
        })
    }

    /// Commit a previously prepared generation. No vector/graph artifacts are
    /// rehashed here. Caller must serialize its catchup check and this publication
    /// with mutations; readers retain access to all previous generation directories.
    pub fn commit_prepared(&self, prepared: PreparedGeneration) -> io::Result<LoadedGeneration> {
        let _guard = self
            .publication
            .lock()
            .map_err(|_| io::Error::other("publication lock poisoned"))?;
        if prepared.root != self.root {
            return Err(invalid_input(
                "prepared generation belongs to a different store",
            ));
        }
        let generation = prepared.loaded.generation;
        if Self::read_current_inner(&self.root, false)?
            .is_some_and(|current| generation <= current.generation)
        {
            return Err(invalid_input("prepared generation is older than CURRENT"));
        }

        let mut current = Vec::with_capacity(16);
        current.extend_from_slice(&generation.to_le_bytes());
        current.extend_from_slice(&prepared.catalog_checksum.to_le_bytes());
        let suffix = format!(
            "{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let temporary_current = self.root.join(format!(".CURRENT-{suffix}"));
        write_envelope(&temporary_current, CURRENT_MAGIC, &current)?;
        fs::rename(temporary_current, self.root.join("CURRENT"))?;
        sync_directory(&self.root)?;
        Ok(prepared.loaded)
    }
}

fn generation_name(generation: u64) -> String {
    format!("generation-{generation:020}")
}

fn collect_artifacts(directory: &Path, sync: bool) -> io::Result<Vec<Artifact>> {
    fn visit(
        root: &Path,
        directory: &Path,
        sync: bool,
        output: &mut Vec<Artifact>,
    ) -> io::Result<()> {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            let kind = entry.file_type()?;
            if kind.is_dir() {
                visit(root, &path, sync, output)?;
            } else if kind.is_file() {
                let relative = path.strip_prefix(root).map_err(io::Error::other)?;
                let name = relative
                    .to_str()
                    .ok_or_else(|| invalid_input("artifact path must be UTF-8"))?;
                if name == "CATALOG" {
                    continue;
                }
                validate_artifact_name(name)?;
                let mut file = File::open(&path)?;
                if sync {
                    file.sync_all()?;
                }
                let (bytes, checksum) = checksum_reader(&mut file)?;
                output.push(Artifact {
                    name: name.to_owned(),
                    bytes,
                    checksum,
                });
            } else {
                return Err(invalid_data(format!(
                    "artifact {} is not a regular file/directory",
                    path.display()
                )));
            }
        }
        if sync {
            sync_directory(directory)?;
        }
        Ok(())
    }
    let mut artifacts = Vec::new();
    visit(directory, directory, sync, &mut artifacts)?;
    artifacts.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    Ok(artifacts)
}

fn verify_artifacts(directory: &Path, expected: &[Artifact]) -> io::Result<()> {
    let mut names = BTreeSet::new();
    for artifact in expected {
        validate_artifact_name(&artifact.name)?;
        if !names.insert(&artifact.name) {
            return Err(invalid_data("duplicate artifact name in catalog"));
        }
    }
    let actual = collect_artifacts(directory, false)?;
    if actual.len() != expected.len() {
        return Err(invalid_data(
            "generation artifact count disagrees with catalog",
        ));
    }
    for file in &actual {
        let expected = expected
            .iter()
            .find(|artifact| artifact.name == file.name)
            .ok_or_else(|| invalid_data(format!("unlisted generation artifact {}", file.name)))?;
        if file.bytes != expected.bytes || file.checksum != expected.checksum {
            return Err(invalid_data(format!(
                "generation artifact {} size/checksum mismatch",
                file.name
            )));
        }
    }
    Ok(())
}

fn validate_artifact_name(name: &str) -> io::Result<()> {
    if name.is_empty()
        || name == "CATALOG"
        || name.contains('\\')
        || Path::new(name)
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(invalid_data(format!(
            "invalid artifact relative path {name:?}"
        )));
    }
    Ok(())
}

fn ensure_directory(path: &Path) -> io::Result<()> {
    if !fs::symlink_metadata(path)?.is_dir() {
        return Err(invalid_data(format!(
            "{} is not a regular directory",
            path.display()
        )));
    }
    Ok(())
}

fn write_envelope(path: &Path, magic: &[u8; 8], payload: &[u8]) -> io::Result<()> {
    let mut header = Vec::with_capacity(ENVELOPE_HEADER);
    header.extend_from_slice(magic);
    header.extend_from_slice(&VERSION.to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes());
    header.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    header.extend_from_slice(&checksum(payload).to_le_bytes());
    header.extend_from_slice(&checksum(&header).to_le_bytes());
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&header)?;
    file.write_all(payload)?;
    file.sync_all()
}

fn read_envelope(path: &Path, magic: &[u8; 8], limit: usize) -> io::Result<Vec<u8>> {
    if !fs::symlink_metadata(path)?.is_file() {
        return Err(invalid_data(format!(
            "{} is not a regular file",
            path.display()
        )));
    }
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    if length < ENVELOPE_HEADER as u64 || length - ENVELOPE_HEADER as u64 > limit as u64 {
        return Err(invalid_data(format!(
            "invalid envelope size in {}",
            path.display()
        )));
    }
    let mut header = [0; ENVELOPE_HEADER];
    file.read_exact(&mut header)?;
    if &header[..8] != magic
        || number32(&header[8..12]) != VERSION
        || number32(&header[12..16]) != 0
        || number64(&header[16..24]) != length - ENVELOPE_HEADER as u64
        || checksum(&header[..32]) != number64(&header[32..40])
    {
        return Err(invalid_data(format!(
            "invalid envelope header/version/checksum in {}",
            path.display()
        )));
    }
    let mut payload = vec![0; (length - ENVELOPE_HEADER as u64) as usize];
    file.read_exact(&mut payload)?;
    if checksum(&payload) != number64(&header[24..32]) {
        return Err(invalid_data(format!(
            "invalid payload checksum in {}",
            path.display()
        )));
    }
    Ok(payload)
}

fn check_frame_limit(limit: usize) -> io::Result<()> {
    if limit == 0 {
        return Err(invalid_input("max_frame_bytes must be positive"));
    }
    Ok(())
}

fn codec(limit: usize) -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .reject_trailing_bytes()
        .with_limit(limit as u64)
}

fn lock(file: &File) -> io::Result<()> {
    file.try_lock().map_err(|error| match error {
        fs::TryLockError::WouldBlock => io::Error::new(
            io::ErrorKind::WouldBlock,
            "another writer holds the store/journal lock",
        ),
        fs::TryLockError::Error(error) => error,
    })
}

fn parent(path: &Path) -> &Path {
    path.parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn require_directory_sync() -> io::Result<()> {
    if cfg!(unix) {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "durable generation publication currently requires Unix directory fsync",
        ))
    }
}

fn sync_directory(path: &Path) -> io::Result<()> {
    require_directory_sync()?;
    File::open(path)?.sync_all()
}

fn create_directory_tree(path: &Path) -> io::Result<()> {
    let mut missing = Vec::new();
    let mut directory = path;
    loop {
        match fs::metadata(directory) {
            Ok(metadata) if metadata.is_dir() => break,
            Ok(_) => {
                return Err(invalid_input(
                    "generation store ancestor is not a directory",
                ))
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                missing.push(directory);
                directory = parent(directory);
            }
            Err(error) => return Err(error),
        }
    }
    fs::create_dir_all(path)?;
    for directory in missing.into_iter().rev() {
        sync_directory(directory)?;
        sync_directory(parent(directory))?;
    }
    Ok(())
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn number32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("fixed-size format field"))
}

fn number64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("fixed-size format field"))
}

const fn crc_table() -> [u64; 256] {
    let mut table = [0; 256];
    let mut index = 0;
    while index < table.len() {
        let mut value = (index as u64) << 56;
        let mut bit = 0;
        while bit < 8 {
            value = if value & (1 << 63) == 0 {
                value << 1
            } else {
                (value << 1) ^ 0x42F0_E1EB_A9EA_3693
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

const CRC_TABLE: [u64; 256] = crc_table();

fn checksum_update(mut value: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        value = (value << 8) ^ CRC_TABLE[((value >> 56) as u8 ^ byte) as usize];
    }
    value
}

/// CRC-64/ECMA-182, unreflected, initial value and final xor both zero.
pub fn checksum(bytes: &[u8]) -> u64 {
    checksum_update(0, bytes)
}

fn checksum_reader(reader: &mut impl Read) -> io::Result<(u64, u64)> {
    let mut buffer = [0; 64 * 1024];
    let mut bytes = 0u64;
    let mut crc = 0u64;
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        bytes = bytes
            .checked_add(count as u64)
            .ok_or_else(|| invalid_data("artifact size overflow"))?;
        crc = checksum_update(crc, &buffer[..count]);
    }
    Ok((bytes, crc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn checksum_matches_ecma_known_vector_and_streaming() {
        assert_eq!(checksum(b"123456789"), 0x6c40_df5f_0b49_7347);
        assert_eq!(
            checksum_reader(&mut &b"123456789"[..]).unwrap(),
            (9, checksum(b"123456789"))
        );
    }

    #[test]
    #[cfg(unix)]
    fn journal_replays_atomic_batches_and_rejects_a_second_writer() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("journal");
        let (mut journal, _) = Journal::<f32>::create_new(&path, 8, 4096).unwrap();
        journal
            .append(
                9,
                &[
                    Mutation::Upsert {
                        id: 7,
                        vector: vec![1.5, 3.],
                        labels: vec![2],
                    },
                    Mutation::Delete { id: 9 },
                ],
            )
            .unwrap();
        assert!(journal.append(11, &[Mutation::Delete { id: 0 }]).is_err());
        assert_eq!(
            Journal::<f32>::open_existing(&path, 4096)
                .err()
                .unwrap()
                .kind(),
            io::ErrorKind::WouldBlock
        );
        assert!(journal.replay_after(9).unwrap().is_empty());
        assert_eq!(journal.replay_after(8).unwrap().len(), 1);
        drop(journal);
        let (journal, recovered) = Journal::<f32>::open_existing(&path, 4096).unwrap();
        assert_eq!(recovered.base_sequence, 8);
        assert_eq!(recovered.last_sequence, 9);
        assert_eq!(recovered.batches[0].mutations.len(), 2);
        assert!(
            matches!(&recovered.batches[0].mutations[0], Mutation::Upsert { vector, .. } if vector == &[1.5, 3.])
        );
        assert_eq!(journal.last_sequence(), 9);
    }

    #[test]
    #[cfg(unix)]
    fn incomplete_journal_tail_is_recovered_but_complete_corruption_is_rejected() {
        for truncate in [1, 41] {
            let directory = tempdir().unwrap();
            let path = directory.path().join("journal");
            let (mut journal, _) = Journal::<u8>::create_new(&path, 0, 4096).unwrap();
            journal.append(1, &[Mutation::Delete { id: 4 }]).unwrap();
            let committed_len = journal.file_len().unwrap();
            journal
                .append(
                    2,
                    &[Mutation::Upsert {
                        id: 7,
                        vector: vec![1; 100],
                        labels: vec![2],
                    }],
                )
                .unwrap();
            let length = journal.file_len().unwrap();
            drop(journal);
            let file = OpenOptions::new().write(true).open(&path).unwrap();
            file.set_len(committed_len + truncate).unwrap();
            drop(file);
            let (mut journal, recovered) = Journal::<u8>::open_existing(&path, 4096).unwrap();
            assert_eq!(recovered.last_sequence, 1);
            assert_eq!(recovered.truncated_bytes, truncate);
            assert!(length > committed_len + truncate);
            journal.append(2, &[Mutation::Delete { id: 7 }]).unwrap();
            drop(journal);
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            file.seek(SeekFrom::End(-1)).unwrap();
            file.write_all(&[0xff]).unwrap();
            drop(file);
            assert_eq!(
                Journal::<u8>::open_existing(&path, 4096)
                    .err()
                    .unwrap()
                    .kind(),
                io::ErrorKind::InvalidData
            );
        }
    }

    #[test]
    #[cfg(unix)]
    fn journal_rejects_corrupt_header_and_an_oversized_frame() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("journal");
        let (mut journal, _) = Journal::<u8>::create_new(&path, 0, 4096).unwrap();
        journal
            .append(
                1,
                &[Mutation::Upsert {
                    id: 0,
                    vector: vec![0; 512],
                    labels: vec![],
                }],
            )
            .unwrap();
        drop(journal);
        assert!(Journal::<u8>::open_existing(&path, 128).is_err());
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(JOURNAL_HEADER as u64 + 16))
            .unwrap();
        file.write_all(&[0xff]).unwrap();
        drop(file);
        assert!(Journal::<u8>::open_existing(&path, 4096).is_err());
        assert_eq!(
            Journal::<u8>::open_existing(directory.path().join("missing"), 4096)
                .err()
                .unwrap()
                .kind(),
            io::ErrorKind::NotFound
        );
    }

    #[test]
    #[cfg(unix)]
    fn checkpoint_preserves_catchup_batches_and_the_writer_lock() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("journal");
        let (mut journal, _) = Journal::<u8>::create_new(&path, 0, 4096).unwrap();
        journal.append(1, &[Mutation::Delete { id: 1 }]).unwrap();
        journal
            .append(
                2,
                &[Mutation::Upsert {
                    id: 2,
                    vector: vec![9],
                    labels: vec![7],
                }],
            )
            .unwrap();
        journal.append(3, &[Mutation::Delete { id: 3 }]).unwrap();
        let old_length = journal.file_len().unwrap();
        journal.checkpoint(2).unwrap();
        assert!(journal.file_len().unwrap() < old_length);
        assert_eq!(journal.last_sequence(), 3);
        assert_eq!(
            Journal::<u8>::open_existing(&path, 4096)
                .err()
                .unwrap()
                .kind(),
            io::ErrorKind::WouldBlock
        );
        let batches = journal.replay_after(2).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].sequence, 3);
        journal.append(4, &[Mutation::Delete { id: 4 }]).unwrap();
        drop(journal);
        let (mut journal, recovery) = Journal::<u8>::open_existing(&path, 4096).unwrap();
        assert_eq!(recovery.base_sequence, 2);
        assert_eq!(recovery.last_sequence, 4);
        assert_eq!(recovery.batches.len(), 2);
        journal.checkpoint(4).unwrap();
        assert_eq!(journal.file_len().unwrap(), JOURNAL_HEADER as u64);
        assert!(journal.checkpoint(5).is_err());
        assert!(journal.append(5, &[Mutation::Delete { id: 5 }]).is_err());
    }

    #[test]
    #[cfg(unix)]
    fn every_incomplete_frame_prefix_preserves_the_prior_committed_batch() {
        let directory = tempdir().unwrap();
        let template = directory.path().join("template");
        let (mut journal, _) = Journal::<u8>::create_new(&template, 0, 4096).unwrap();
        journal.append(1, &[Mutation::Delete { id: 1 }]).unwrap();
        let first_end = journal.file_len().unwrap() as usize;
        journal
            .append(
                2,
                &[Mutation::Upsert {
                    id: 2,
                    vector: vec![9; 8],
                    labels: vec![7],
                }],
            )
            .unwrap();
        drop(journal);
        let bytes = fs::read(&template).unwrap();
        for end in first_end..bytes.len() {
            let path = directory.path().join(format!("prefix-{end}"));
            fs::write(&path, &bytes[..end]).unwrap();
            let (journal, recovery) = Journal::<u8>::open_existing(path, 4096).unwrap();
            assert_eq!(recovery.last_sequence, 1);
            assert_eq!(recovery.batches.len(), 1);
            assert_eq!(recovery.truncated_bytes as usize, end - first_end);
            assert_eq!(journal.file_len().unwrap() as usize, first_end);
        }
    }

    #[test]
    #[cfg(unix)]
    fn relative_store_roots_and_generation_paths_become_absolute() {
        let directory = tempfile::Builder::new()
            .prefix(".filtered-relative-root-")
            .tempdir_in(std::env::current_dir().unwrap())
            .unwrap();
        let relative = PathBuf::from(directory.path().file_name().unwrap()).join("nested/store");
        assert!(relative.is_relative());
        let store = GenerationStore::open(&relative).unwrap();
        assert!(store.root().is_absolute());
        assert_eq!(store.root(), fs::canonicalize(&relative).unwrap());
        let prepared = store
            .prepare_generation(1, b"manifest", |path| {
                assert!(path.is_absolute());
                fs::write(path.join("vectors"), b"data")
            })
            .unwrap();
        assert!(prepared.directory().is_absolute());
        let committed = store.commit_prepared(prepared).unwrap();
        assert!(committed.directory.is_absolute());
        let loaded = store.load_current().unwrap().unwrap();
        assert!(loaded.directory.is_absolute());
        assert_eq!(loaded.directory, committed.directory);
        let loaded = GenerationStore::read_current(&relative).unwrap().unwrap();
        assert!(loaded.directory.is_absolute());
        assert_eq!(loaded.directory, committed.directory);
    }

    #[test]
    #[cfg(unix)]
    fn failed_preparation_removes_staging_and_preserves_current() {
        let directory = tempdir().unwrap();
        let store = GenerationStore::open(directory.path()).unwrap();
        let original = store
            .publish(1, b"original", |path| {
                fs::write(path.join("vectors"), b"old")
            })
            .unwrap();
        let error = store
            .prepare_generation(2, b"rejected", |path| {
                fs::write(path.join("partial-vectors"), vec![0; 1024])?;
                Err(io::Error::other("injected artifact failure"))
            })
            .err()
            .unwrap();
        assert_eq!(error.to_string(), "injected artifact failure");
        assert!(fs::read_dir(store.root()).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".staging-")));
        assert!(!store.root().join(generation_name(2)).exists());
        let loaded = store.load_current().unwrap().unwrap();
        assert_eq!(loaded.generation, 1);
        assert_eq!(loaded.manifest_payload, b"original");
        assert_eq!(
            fs::read(original.directory.join("vectors")).unwrap(),
            b"old"
        );

        // Errors after a successful callback also clean the unpublished directory.
        assert!(store
            .prepare_generation(2, b"invalid inventory", |path| {
                fs::write(path.join("CATALOG"), b"reserved")
            })
            .is_err());
        assert!(fs::read_dir(store.root()).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".staging-")));
        assert_eq!(store.load_current().unwrap().unwrap().generation, 1);
    }

    #[test]
    #[cfg(unix)]
    fn generation_publication_is_atomic_and_checks_all_artifacts() {
        let directory = tempdir().unwrap();
        let store = GenerationStore::open(directory.path()).unwrap();
        assert!(store.load_current().unwrap().is_none());
        assert_eq!(
            GenerationStore::open(directory.path())
                .err()
                .unwrap()
                .kind(),
            io::ErrorKind::WouldBlock
        );
        let first = store
            .publish(1, b"manifest 1", |path| {
                fs::create_dir(path.join("nested"))?;
                fs::write(path.join("nested/vectors"), b"data")
            })
            .unwrap();
        assert_eq!(first.artifacts.len(), 1);
        assert!(store
            .publish(2, b"manifest 2", |_path| Err(io::Error::other(
                "injected failure"
            )))
            .is_err());
        assert_eq!(store.load_current().unwrap().unwrap().generation, 1);
        assert!(store.publish(1, b"duplicate", |_| Ok(())).is_err());
        let second = store
            .publish(3, b"manifest 3", |path| {
                fs::write(path.join("vectors"), b"new")
            })
            .unwrap();
        assert!(first.directory.join("nested/vectors").exists());
        let loaded = GenerationStore::read_current(directory.path())
            .unwrap()
            .unwrap();
        assert_eq!(loaded.generation, 3);
        assert_eq!(loaded.manifest_payload, b"manifest 3");
        fs::write(second.directory.join("vectors"), b"bad").unwrap();
        assert!(store.load_current().is_err());
    }

    #[test]
    #[cfg(unix)]
    fn generation_rejects_symlinks_and_corrupt_current() {
        use std::os::unix::fs::symlink;

        let directory = tempdir().unwrap();
        let store = GenerationStore::open(directory.path()).unwrap();
        assert!(store
            .publish(1, b"bad", |path| symlink(
                "/etc/passwd",
                path.join("vectors")
            ))
            .is_err());
        store
            .publish(2, b"good", |path| fs::write(path.join("vectors"), b"data"))
            .unwrap();
        let mut file = OpenOptions::new()
            .write(true)
            .open(directory.path().join("CURRENT"))
            .unwrap();
        file.seek(SeekFrom::End(-1)).unwrap();
        file.write_all(&[0xff]).unwrap();
        drop(file);
        assert!(store.load_current().is_err());
    }

    #[test]
    #[cfg(unix)]
    fn prepared_generation_is_invisible_until_commit_and_cannot_roll_back_current() {
        let directory = tempdir().unwrap();
        let store = GenerationStore::open(directory.path().join("new/nested/store")).unwrap();
        assert_eq!(store.next_generation().unwrap(), 1);
        store.publish(1, b"one", |_| Ok(())).unwrap();
        let second = store
            .prepare_generation(2, b"two", |path| fs::write(path.join("data"), b"prepared"))
            .unwrap();
        assert_eq!(second.generation(), 2);
        assert_eq!(store.next_generation().unwrap(), 3);
        assert!(second.directory().join("data").exists());
        assert_eq!(store.load_current().unwrap().unwrap().generation, 1);
        let third = store.prepare_generation(3, b"three", |_| Ok(())).unwrap();
        assert_eq!(store.commit_prepared(third).unwrap().generation, 3);
        assert!(store.commit_prepared(second).is_err());
        assert_eq!(store.load_current().unwrap().unwrap().generation, 3);
    }
}
