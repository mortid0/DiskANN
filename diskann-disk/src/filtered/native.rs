/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Native Rust disk layout and PQ storage for immutable filtered indexes.

use std::{
    fs::File,
    io::{BufWriter, Read, Write},
    marker::PhantomData,
    path::Path,
};

use bytemuck::Pod;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use diskann::{utils::VectorRepr, ANNError, ANNResult};
use diskann_providers::{
    model::{MAX_PQ_TRAINING_SET_SIZE, NUM_KMEANS_REPS_PQ, NUM_PQ_CENTROIDS},
    storage::{
        get_compressed_pq_file, get_disk_index_file, get_mem_index_file, get_pq_pivot_file,
        FileStorageProvider, PQStorage,
    },
    utils::create_thread_pool,
};
use diskann_quantization::{
    error::Format,
    product::{BasicTable, TransposedTable},
    CompressInto,
};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::{distance::Metric, DistanceFunction};
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    build::filtered::StitchedGraph,
    data_model::{AdHoc, CachingStrategy},
    search::{
        provider::{
            disk_provider::{DiskIndexSearcher, SearchResult},
            disk_vertex_provider_factory::DiskVertexProviderFactory,
        },
        search_mode::SearchMode,
    },
    storage::{
        disk_index_reader::DiskIndexReader,
        quant::{
            pq::pq_generation::{PQGeneration as RustPQGeneration, PQGenerationContext},
            QuantCompressor, QuantDataGenerator,
        },
        DiskIndexWriter,
    },
};

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for f32 {}
}

/// Scalar types supported by the immutable native filtered-index format.
pub trait NativeVector: VectorRepr + Pod + Serialize + DeserializeOwned + sealed::Sealed {
    /// Stable manifest encoding for this scalar type.
    const ENCODING: &'static str;

    /// Validate one vector in the finite query-distance domain. For f32, its
    /// squared norm accumulated in f64 must not exceed `f32::MAX / 8`; this leaves
    /// headroom for pairwise subtraction, squaring and SIMD accumulation.
    fn validate(values: &[Self]) -> ANNResult<()>;

    /// Apply vector-domain checks per row, independently of corpus size.
    fn validate_vectors(values: &[Self], dimensions: usize) -> ANNResult<()> {
        validate_matrix_shape(values.len(), dimensions)?;
        for vector in values.chunks_exact(dimensions) {
            Self::validate(vector)?;
        }
        Ok(())
    }

    /// Validate persisted scalar values on reload. Byte-valued vectors always
    /// satisfy the numeric domain and require no additional file scan.
    fn validate_stored_vectors(_path: &Path, _dimensions: usize) -> ANNResult<()> {
        Ok(())
    }

    /// Compute squared L2 using DiskANN's native vector kernels.
    ///
    /// Both slices must have the same dimension, as checked by the index API.
    fn exact_l2(query: &[Self], vector: &[Self]) -> f32 {
        Self::distance(Metric::L2, Some(query.len())).evaluate_similarity(query, vector)
    }
}

impl NativeVector for u8 {
    const ENCODING: &'static str = "u8";

    fn validate(_values: &[Self]) -> ANNResult<()> {
        Ok(())
    }

    fn validate_vectors(values: &[Self], dimensions: usize) -> ANNResult<()> {
        validate_matrix_shape(values.len(), dimensions)
    }
}

impl NativeVector for f32 {
    const ENCODING: &'static str = "f32";

    fn validate(values: &[Self]) -> ANNResult<()> {
        let mut squared_norm = 0.;
        for &value in values {
            validate_f32_component(value, &mut squared_norm)?;
        }
        Ok(())
    }

    fn validate_stored_vectors(path: &Path, dimensions: usize) -> ANNResult<()> {
        let mut file = File::open(path)?;
        let rows = file.read_u32::<LittleEndian>()? as usize;
        let columns = file.read_u32::<LittleEndian>()? as usize;
        let elements = rows
            .checked_mul(columns)
            .ok_or_else(|| ANNError::message("stored f32 shape overflow"))?;
        validate_matrix_shape(elements, dimensions)?;
        let bytes = elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| ANNError::message("stored f32 length overflow"))?;
        if columns != dimensions || file.metadata()?.len() != (bytes as u64).saturating_add(8) {
            return Err(ANNError::message("stored f32 matrix shape/length mismatch"));
        }
        // Fixed memory even when one row exceeds the buffer. The running norm is
        // reset only at a real row boundary, including across read boundaries.
        let mut buffer = [0u8; 64 * 1024];
        let mut remaining = bytes;
        let mut coordinate = 0usize;
        let mut row = 0usize;
        let mut squared_norm = 0.;
        while remaining > 0 {
            let count = remaining.min(buffer.len());
            file.read_exact(&mut buffer[..count])?;
            for bytes in buffer[..count].chunks_exact(4) {
                let value = f32::from_le_bytes(bytes.try_into().expect("four-byte f32"));
                validate_f32_component(value, &mut squared_norm)
                    .map_err(|error| error.context(format!("stored f32 vector row {row}")))?;
                coordinate += 1;
                if coordinate == dimensions {
                    coordinate = 0;
                    squared_norm = 0.;
                    row += 1;
                }
            }
            remaining -= count;
        }
        Ok(())
    }
}

fn validate_matrix_shape(elements: usize, dimensions: usize) -> ANNResult<()> {
    if dimensions == 0 || !elements.is_multiple_of(dimensions) {
        return Err(ANNError::message(
            "matrix must have positive dimensions and complete rows",
        ));
    }
    Ok(())
}

#[track_caller]
fn validate_f32_component(value: f32, squared_norm: &mut f64) -> ANNResult<()> {
    if !value.is_finite() {
        return Err(ANNError::message("vectors must contain only finite values"));
    }
    *squared_norm += f64::from(value) * f64::from(value);
    if *squared_norm > f64::from(f32::MAX) / 8. {
        return Err(ANNError::message("f32 vector squared norm exceeds the supported finite query-distance domain (f32::MAX / 8)"));
    }
    Ok(())
}

const BLOCK_SIZE: usize = 4096;
const COMPRESSION_BLOCK_VECTORS: usize = 25_000;

fn index_prefix(directory: &Path) -> ANNResult<String> {
    directory
        .join("index")
        .into_os_string()
        .into_string()
        .map_err(|_| ANNError::message("native index directory must be a UTF-8 path"))
}

/// Write the native graph/PQ files alongside the caller's `vectors.bin`.
pub(crate) fn write_native<T: NativeVector>(
    directory: &Path,
    vectors: rowmajor::Ref<'_, T>,
    graph: &StitchedGraph,
    pq_chunks: usize,
    threads: usize,
    seed: u64,
) -> ANNResult<()> {
    write_native_impl(directory, vectors, graph, pq_chunks, threads, seed, None)
}

/// Copy and reuse a saved PQ codebook, or train a fresh Rust codebook when absent.
pub(crate) fn write_native_with_codebook<T: NativeVector>(
    directory: &Path,
    vectors: rowmajor::Ref<'_, T>,
    graph: &StitchedGraph,
    pq_chunks: usize,
    threads: usize,
    seed: u64,
    codebook: Option<&Path>,
) -> ANNResult<()> {
    match codebook {
        None => write_native(directory, vectors, graph, pq_chunks, threads, seed),
        Some(_) => write_native_impl(
            directory, vectors, graph, pq_chunks, threads, seed, codebook,
        ),
    }
}

struct SavedCodebookContext {
    path: String,
    dim: usize,
    chunks: usize,
}

struct SavedCodebook<T> {
    table: TransposedTable,
    chunks: usize,
    scalar: PhantomData<T>,
}

fn load_saved_pivots(context: &SavedCodebookContext) -> ANNResult<BasicTable> {
    // The loader validates the legacy format and folds any C++ centering vector
    // into effective absolute pivots, as it does on the disk-search path.
    let table = PQStorage::new(&context.path, "", None).load_pivots(&FileStorageProvider)?;
    if table.dim() != context.dim
        || table.nchunks() != context.chunks
        || table.ncenters() != NUM_PQ_CENTROIDS
        || table
            .view_pivots()
            .as_slice()
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err(ANNError::message(format!(
                "saved PQ codebook must have {} dimensions, {} chunks and {} finite centroids; found {} dimensions, {} chunks and {} centroids",
                context.dim, context.chunks, NUM_PQ_CENTROIDS,
                table.dim(), table.nchunks(), table.ncenters(),
            )));
    }
    Ok(table)
}

/// Retain an owned, validated codebook even for a small or empty exact-search base.
pub(crate) fn copy_codebook(
    directory: &Path,
    dimensions: usize,
    pq_chunks: usize,
    source: &Path,
) -> ANNResult<()> {
    let context = SavedCodebookContext {
        path: source
            .to_str()
            .ok_or_else(|| ANNError::message("saved PQ codebook path must be UTF-8"))?
            .into(),
        dim: dimensions,
        chunks: pq_chunks,
    };
    load_saved_pivots(&context)?;
    let pivot_path = get_pq_pivot_file(&index_prefix(directory)?);
    let destination = Path::new(&pivot_path);
    if !destination.try_exists()?
        || std::fs::canonicalize(source)? != std::fs::canonicalize(destination)?
    {
        std::fs::copy(source, destination)?;
    }
    Ok(())
}

impl<T: NativeVector> QuantCompressor<T> for SavedCodebook<T> {
    type CompressorContext = SavedCodebookContext;

    fn new(context: &SavedCodebookContext) -> ANNResult<Self> {
        let table = load_saved_pivots(context)?;
        let table =
            TransposedTable::from_parts(table.view_pivots(), table.view_offsets().to_owned())
                .map_err(|error| {
                    ANNError::message(format!("invalid saved PQ codebook: {}", Format(error)))
                })?;
        Ok(Self {
            table,
            chunks: context.chunks,
            scalar: PhantomData,
        })
    }

    fn compress(
        &self,
        vectors: rowmajor::Ref<'_, f32>,
        output: rowmajor::Mut<'_, u8>,
    ) -> ANNResult<()> {
        self.table.compress_into(vectors, output).map_err(|error| {
            ANNError::message(format!("saved PQ compression failed: {}", Format(error)))
        })
    }

    fn compressed_bytes(&self) -> usize {
        self.chunks
    }
}

fn write_native_impl<T: NativeVector>(
    directory: &Path,
    vectors: rowmajor::Ref<'_, T>,
    graph: &StitchedGraph,
    pq_chunks: usize,
    threads: usize,
    seed: u64,
    codebook: Option<&Path>,
) -> ANNResult<()> {
    if vectors.nrows() < NUM_PQ_CENTROIDS
        || vectors.nrows() > u32::MAX as usize
        || vectors.ncols() == 0
        || pq_chunks == 0
        || pq_chunks > vectors.ncols()
        || threads == 0
    {
        return Err(ANNError::message(
            "native PQ layout requires at least 256 rows, positive dimensions/threads, and 1..=dimensions PQ chunks",
        ));
    }
    T::validate_vectors(vectors.as_slice(), vectors.ncols())?;
    if graph.neighbors.len() != vectors.nrows()
        || graph.max_degree == 0
        || graph.neighbors.iter().any(|row| {
            row.len() > graph.max_degree as usize
                || row.iter().any(|&id| id as usize >= vectors.nrows())
        })
        || graph
            .entry_points
            .values()
            .flatten()
            .any(|&id| id as usize >= vectors.nrows())
    {
        return Err(ANNError::message(
            "invalid stitched graph for native layout",
        ));
    }

    let prefix = index_prefix(directory)?;
    let data_path = directory.join("vectors.bin");
    let mut data_file = File::open(&data_path)?;
    let stored_rows = data_file.read_u32::<LittleEndian>()? as usize;
    let stored_dims = data_file.read_u32::<LittleEndian>()? as usize;
    let expected_size = vectors
        .nrows()
        .checked_mul(vectors.ncols())
        .and_then(|size| size.checked_mul(std::mem::size_of::<T>()))
        .and_then(|size| size.checked_add(8))
        .ok_or_else(|| ANNError::message("native vector file size overflow"))?;
    if stored_rows != vectors.nrows()
        || stored_dims != vectors.ncols()
        || data_file.metadata()?.len() != expected_size as u64
    {
        return Err(ANNError::message(
            "vectors.bin shape or length differs from the native build input",
        ));
    }
    let data_path = data_path
        .into_os_string()
        .into_string()
        .map_err(|_| ANNError::message("native vector file must be a UTF-8 path"))?;

    let file_size = graph
        .neighbors
        .iter()
        .try_fold(24u64, |size, row| {
            size.checked_add(4u64.checked_mul(row.len() as u64 + 1)?)
        })
        .ok_or_else(|| ANNError::message("native graph file size overflow"))?;
    let mut writer = BufWriter::new(File::create(get_mem_index_file(&prefix))?);
    // Canonical in-memory graph header: length, max degree, medoid, frozen count.
    writer.write_u64::<LittleEndian>(file_size)?;
    writer.write_u32::<LittleEndian>(graph.max_degree)?;
    writer.write_u32::<LittleEndian>(
        graph
            .entry_points
            .values()
            .flatten()
            .copied()
            .next()
            .unwrap_or(0),
    )?;
    writer.write_u64::<LittleEndian>(0)?;
    for row in &graph.neighbors {
        writer.write_u32::<LittleEndian>(row.len() as u32)?;
        for &id in row {
            writer.write_u32::<LittleEndian>(id)?;
        }
    }
    writer.flush()?;
    drop(writer);

    let storage = FileStorageProvider;
    let pool = create_thread_pool(threads)?;
    let compressed_path = get_compressed_pq_file(&prefix);
    if let Some(codebook) = codebook {
        copy_codebook(directory, vectors.ncols(), pq_chunks, codebook)?;
        let pivot_path = get_pq_pivot_file(&prefix);
        let context = SavedCodebookContext {
            path: pivot_path,
            dim: vectors.ncols(),
            chunks: pq_chunks,
        };
        let generator = QuantDataGenerator::<T, SavedCodebook<T>>::new(
            data_path.clone(),
            compressed_path,
            &context,
        )?;
        generator.generate_data(&storage, pool.as_ref(), COMPRESSION_BLOCK_VECTORS)?;
    } else {
        let context = PQGenerationContext {
            pq_storage: PQStorage::new(
                &get_pq_pivot_file(&prefix),
                &compressed_path,
                Some(&data_path),
            ),
            num_chunks: pq_chunks,
            seed: Some(seed),
            p_val: (MAX_PQ_TRAINING_SET_SIZE / vectors.nrows() as f64).min(1.0),
            storage_provider: &storage,
            pool: pool.as_ref(),
            metric: Metric::L2,
            dim: vectors.ncols(),
            max_kmeans_reps: NUM_KMEANS_REPS_PQ,
            num_centers: NUM_PQ_CENTROIDS,
        };
        let generator = QuantDataGenerator::<T, RustPQGeneration<T, FileStorageProvider>>::new(
            data_path.clone(),
            compressed_path,
            &context,
        )?;
        generator.generate_data(&storage, pool.as_ref(), COMPRESSION_BLOCK_VECTORS)?;
    }
    let writer = DiskIndexWriter::new(data_path, prefix, None, BLOCK_SIZE)?;
    writer.create_disk_layout::<AdHoc<T>, _>(&storage)?;
    writer.index_build_cleanup(&storage)
}

#[cfg(any(miri, not(target_os = "macos")))]
pub(crate) type NativeReaderFactory =
    crate::search::provider::aligned_file_reader::AlignedFileReaderFactory;

#[cfg(all(not(miri), target_os = "macos"))]
pub(crate) use macos::NativeReaderFactory;

type InnerNativeSearcher<T> =
    DiskIndexSearcher<AdHoc<T>, DiskVertexProviderFactory<AdHoc<T>, NativeReaderFactory>>;

/// Owns the disk searcher's runtime without nesting it in a caller's Tokio runtime.
pub(crate) struct NativeSearcher<T: NativeVector> {
    inner: Option<InnerNativeSearcher<T>>,
}

thread_local! {
    static BLOCKING_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Call only within a known plain or Tokio `spawn_blocking` worker.
/// Queries can use the owned runtime directly here; never enter this scope while
/// polling an async task. Nested scopes restore the previous state on return/unwind.
pub(super) fn on_blocking_worker<R>(operation: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            BLOCKING_WORKER.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(BLOCKING_WORKER.with(|flag| flag.replace(true)));
    operation()
}

fn outside_runtime<const ALLOW_BLOCKING_WORKER: bool, R: Send>(
    operation: impl FnOnce() -> ANNResult<R> + Send,
) -> ANNResult<R> {
    if tokio::runtime::Handle::try_current().is_ok()
        && !(ALLOW_BLOCKING_WORKER && BLOCKING_WORKER.with(|flag| flag.get()))
    {
        std::thread::scope(|scope| {
            std::thread::Builder::new()
                .name("diskann-native".into())
                .spawn_scoped(scope, operation)?
                .join()
                .map_err(|_| ANNError::message("native disk operation panicked on its worker"))?
        })
    } else {
        operation()
    }
}

impl<T: NativeVector> NativeSearcher<T> {
    fn with_inner<R: Send>(
        &self,
        operation: impl FnOnce(&InnerNativeSearcher<T>) -> ANNResult<R> + Send,
    ) -> ANNResult<R> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| ANNError::message("native searcher has already been closed"))?;
        outside_runtime::<true, _>(|| operation(inner))
    }

    pub(crate) fn search(
        &self,
        query: &[T],
        k: u32,
        l: u32,
        beam_width: Option<usize>,
        mode: SearchMode<'_>,
    ) -> ANNResult<SearchResult<()>> {
        self.with_inner(|inner| inner.search(query, k, l, beam_width, mode))
    }

    #[cfg(test)]
    fn search_with_indexed_vectors(
        &self,
        query: &[T],
        k: u32,
        l: u32,
        beam_width: Option<usize>,
        mode: SearchMode<'_>,
    ) -> ANNResult<crate::search::provider::disk_provider::SearchResultWithVectors<(), T>> {
        self.with_inner(|inner| inner.search_with_indexed_vectors(query, k, l, beam_width, mode))
    }
}

impl<T: NativeVector> Drop for NativeSearcher<T> {
    fn drop(&mut self) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        if tokio::runtime::Handle::try_current().is_ok() {
            // If spawning fails, dropping the closure must not destroy the runtime
            // in this async context. Retain it in that exceptional resource failure.
            let inner = std::mem::ManuallyDrop::new(inner);
            match std::thread::Builder::new()
                .name("diskann-native-drop".into())
                .spawn(move || drop(std::mem::ManuallyDrop::into_inner(inner)))
            {
                Ok(worker) => {
                    if worker.join().is_err() {
                        tracing::error!("native searcher disposal worker panicked");
                    }
                }
                Err(error) => {
                    tracing::error!(%error, "unable to spawn native disposal worker; retaining runtime");
                }
            }
        } else {
            drop(inner);
        }
    }
}

/// Open native Rust graph/PQ files without a shared graph-node cache.
pub(crate) fn open_searcher<T: NativeVector>(
    directory: &Path,
    threads: usize,
) -> ANNResult<NativeSearcher<T>> {
    outside_runtime::<false, _>(|| open_searcher_inner(directory, threads))
}

fn open_searcher_inner<T: NativeVector>(
    directory: &Path,
    threads: usize,
) -> ANNResult<NativeSearcher<T>> {
    if threads == 0 {
        return Err(ANNError::message("native search threads must be positive"));
    }
    let prefix = index_prefix(directory)?;
    let storage = FileStorageProvider;
    let reader = DiskIndexReader::new(
        get_pq_pivot_file(&prefix),
        get_compressed_pq_file(&prefix),
        &storage,
    )?;
    let factory = DiskVertexProviderFactory::new(
        NativeReaderFactory::new(get_disk_index_file(&prefix)),
        CachingStrategy::None,
    )?;
    Ok(NativeSearcher {
        inner: Some(DiskIndexSearcher::new(
            threads,
            usize::MAX,
            &reader,
            factory,
            Metric::L2,
            None,
        )?),
    })
}

#[cfg(all(not(miri), target_os = "macos"))]
mod macos {
    use std::{
        fs::File,
        os::unix::{fs::FileExt, io::AsRawFd},
    };

    use diskann::ANNResult;

    use crate::search::provider::aligned_file_reader::{
        traits::{AlignedFileReader, AlignedReaderFactory},
        AlignedRead, A512,
    };

    pub(crate) struct NativeReaderFactory {
        path: String,
    }

    impl NativeReaderFactory {
        pub(crate) fn new(path: String) -> Self {
            Self { path }
        }
    }

    pub(crate) struct PositionalReader(File);

    impl AlignedReaderFactory for NativeReaderFactory {
        type AlignedReaderType = PositionalReader;

        fn build(&self) -> ANNResult<PositionalReader> {
            let file = File::open(&self.path)?;
            // SAFETY: fcntl receives a valid owned descriptor and the integer argument
            // required by F_NOCACHE; no ownership is transferred or memory accessed.
            if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
                return Err(std::io::Error::last_os_error().into());
            }
            Ok(PositionalReader(file))
        }
    }

    impl AlignedFileReader for PositionalReader {
        type Alignment = A512;

        fn read(&mut self, requests: &mut [AlignedRead<u8, A512>]) -> ANNResult<()> {
            for request in requests {
                let offset = request.offset();
                self.0.read_exact_at(request.aligned_buf_mut(), offset)?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use diskann_utils::io::write_bin;

    use super::*;

    #[test]
    fn native_scalar_validation_and_exact_l2() {
        assert!(f32::validate(&[0.0, f32::NAN]).is_err());
        assert!(f32::validate(&[f32::INFINITY]).is_err());
        assert!(f32::validate(&[f32::NEG_INFINITY]).is_err());
        assert!(f32::validate(&[f32::MAX]).is_err());
        f32::validate(&[-3.0, 4.0]).unwrap();
        u8::validate(&[0, 255]).unwrap();
        assert_eq!(f32::exact_l2(&[1.0, -2.0], &[4.0, 2.0]), 25.0);
        assert_eq!(u8::exact_l2(&[0, 255], &[255, 0]), 130_050.0);
    }

    #[test]
    fn f32_domain_validation_is_per_row_and_streams_across_buffer_boundaries() {
        let values = [4.0e18f32; 4];
        assert!(f32::validate(&values).is_err());
        f32::validate_vectors(&values, 1).unwrap();
        assert!(f32::validate_vectors(&values, 3).is_err());
        assert!(f32::validate_vectors(&values, 0).is_err());
        assert!(f32::exact_l2(&[4.0e18], &[-4.0e18]).is_finite());
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("vectors.bin");
        // 16384 floats per read is not divisible by seven: row norms must carry
        // across buffer boundaries, and reset independently of each read.
        let mut values = vec![1.0e18f32; 2500 * 7];
        write_bin(
            rowmajor::Ref::try_from_data(&values, 2500, 7).unwrap(),
            &mut File::create(&path).unwrap(),
        )
        .unwrap();
        f32::validate_stored_vectors(&path, 7).unwrap();
        assert!(f32::validate_stored_vectors(&path, 6).is_err());
        values[2300 * 7 + 1] = f32::MAX;
        write_bin(
            rowmajor::Ref::try_from_data(&values, 2500, 7).unwrap(),
            &mut File::create(&path).unwrap(),
        )
        .unwrap();
        assert!(f32::validate_stored_vectors(&path, 7).is_err());
    }

    fn native_fixture<T: NativeVector>(values: &[T]) -> tempfile::TempDir {
        native_fixture_with_codebook(values, None)
    }

    fn native_fixture_with_codebook<T: NativeVector>(
        values: &[T],
        codebook: Option<&Path>,
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        let vectors = rowmajor::Ref::try_from_data(values, 300, 4).unwrap();
        write_bin(
            vectors,
            &mut File::create(directory.path().join("vectors.bin")).unwrap(),
        )
        .unwrap();
        let graph = StitchedGraph {
            neighbors: (0..300)
                .map(|id| match id {
                    0 => Vec::new(),
                    299 => vec![1],
                    _ => vec![id + 1],
                })
                .collect(),
            entry_points: BTreeMap::from([(7, vec![0]), (99, vec![1])]),
            max_degree: 2,
        };
        write_native_with_codebook(
            directory.path(),
            vectors,
            &graph,
            2,
            2,
            if codebook.is_some() { 777 } else { 42 },
            codebook,
        )
        .unwrap();
        directory
    }

    fn frozen_roundtrip<T: NativeVector + PartialEq + std::fmt::Debug>(values: &[T]) {
        let original = native_fixture(values);
        let source = original.path().join("index_pq_pivots.bin");
        let pivot_bytes = std::fs::read(&source).unwrap();
        let codes = std::fs::read(original.path().join("index_pq_compressed.bin")).unwrap();
        let identical = native_fixture_with_codebook(values, Some(&source));
        assert_eq!(
            std::fs::read(identical.path().join("index_pq_pivots.bin")).unwrap(),
            pivot_bytes
        );
        assert_eq!(
            std::fs::read(identical.path().join("index_pq_compressed.bin")).unwrap(),
            codes
        );
        let row_codes = |row: usize| &codes[8 + row * 2..8 + (row + 1) * 2];
        let donor = (0..300)
            .find(|&row| row_codes(row) != row_codes(150))
            .unwrap();
        let mut changed = values.to_vec();
        changed[150 * 4..151 * 4].copy_from_slice(&values[donor * 4..(donor + 1) * 4]);
        let updated = native_fixture_with_codebook(&changed, Some(&source));
        assert_eq!(
            std::fs::read(updated.path().join("index_pq_pivots.bin")).unwrap(),
            pivot_bytes
        );
        let changed_codes = std::fs::read(updated.path().join("index_pq_compressed.bin")).unwrap();
        for row in 0..300 {
            assert_eq!(
                &changed_codes[8 + row * 2..8 + (row + 1) * 2],
                row_codes(if row == 150 { donor } else { row }),
                "PQ codes differ at row {row}"
            );
        }
        let searcher = open_searcher::<T>(updated.path(), 1).unwrap();
        let result = searcher
            .search_with_indexed_vectors(
                &changed[150 * 4..151 * 4],
                1,
                10,
                Some(4),
                SearchMode::single_hop_filter(|id| *id == 150).with_starting_points(vec![150]),
            )
            .unwrap();
        assert_eq!(result.results[0].vertex_id, 150);
        assert_eq!(result.results[0].distance, 0.0);
        assert_eq!(
            result.results[0].indexed_vector.as_ref(),
            &changed[150 * 4..151 * 4]
        );
    }

    #[test]
    fn frozen_codebook_preserves_bytes_and_unmodified_codes_for_native_scalars() {
        let values: Vec<u8> = (0..300)
            .flat_map(|id| {
                [
                    (id % 256) as u8,
                    (id / 256) as u8,
                    (id % 251) as u8,
                    (id % 239) as u8,
                ]
            })
            .collect();
        frozen_roundtrip(&values);
        let values: Vec<f32> = values.into_iter().map(f32::from).collect();
        frozen_roundtrip(&values);
    }

    #[test]
    fn copied_codebooks_validate_shape_centers_and_truncation_before_publication() {
        let values: Vec<f32> = (0..300).flat_map(|id| [id as f32, 0.0, 1.0, 2.0]).collect();
        let original = native_fixture(&values);
        let source = original.path().join("index_pq_pivots.bin");
        let target = tempfile::tempdir().unwrap();
        copy_codebook(target.path(), 4, 2, &source).unwrap();
        assert_eq!(
            std::fs::read(target.path().join("index_pq_pivots.bin")).unwrap(),
            std::fs::read(&source).unwrap()
        );
        // Copying the already owned asset is safe too, without truncating itself.
        copy_codebook(
            target.path(),
            4,
            2,
            &target.path().join("index_pq_pivots.bin"),
        )
        .unwrap();
        for (dimensions, chunks) in [(5, 2), (4, 1)] {
            let rejected = tempfile::tempdir().unwrap();
            assert!(copy_codebook(rejected.path(), dimensions, chunks, &source).is_err());
            assert!(!rejected.path().join("index_pq_pivots.bin").exists());
        }
        let bad = tempfile::tempdir().unwrap();
        let truncated = bad.path().join("truncated.bin");
        let bytes = std::fs::read(&source).unwrap();
        std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();
        assert!(copy_codebook(bad.path(), 4, 2, &truncated).is_err());
        let small = bad.path().join("small.bin");
        PQStorage::new(small.to_str().unwrap(), "", None)
            .write_pivot_data(
                &vec![0.0; 128 * 4],
                None,
                &[0, 2, 4],
                128,
                4,
                &FileStorageProvider,
            )
            .unwrap();
        assert!(copy_codebook(bad.path(), 4, 2, &small).is_err());
        assert!(!bad.path().join("index_pq_pivots.bin").exists());
    }

    #[test]
    fn frozen_legacy_cpp_fixture_reencodes_without_cpp_runtime() {
        use diskann_utils::{io::read_bin, test_data_root};
        let data = test_data_root().join("disk_index_search");
        let source = data.join("disk_index_sift_learn_R4_L50_A1.2_truth_search_pq_pivots.bin");
        let vectors = read_bin::<f32>(
            &mut File::open(data.join("disk_index_siftsmall_learn_256pts_data.fbin")).unwrap(),
        )
        .unwrap();
        let graph = StitchedGraph {
            neighbors: (0..256).map(|id| vec![(id + 1) % 256]).collect(),
            entry_points: BTreeMap::from([(7, vec![0])]),
            max_degree: 2,
        };
        let directory = tempfile::tempdir().unwrap();
        write_bin(
            vectors.as_view(),
            &mut File::create(directory.path().join("vectors.bin")).unwrap(),
        )
        .unwrap();
        write_native_with_codebook(
            directory.path(),
            vectors.as_view(),
            &graph,
            128,
            2,
            999,
            Some(&source),
        )
        .unwrap();
        assert_eq!(
            std::fs::read(directory.path().join("index_pq_pivots.bin")).unwrap(),
            std::fs::read(source).unwrap()
        );
        assert_eq!(
            std::fs::read(directory.path().join("index_pq_compressed.bin")).unwrap(),
            std::fs::read(
                data.join("disk_index_sift_learn_R4_L50_A1.2_truth_search_pq_compressed.bin")
            )
            .unwrap()
        );
        let searcher = open_searcher::<f32>(directory.path(), 1).unwrap();
        let result = searcher
            .search(
                vectors.row(0),
                1,
                10,
                Some(4),
                SearchMode::single_hop_filter(|id| *id == 0).with_starting_points(vec![0]),
            )
            .unwrap();
        assert_eq!(result.results[0].vertex_id, 0);
        assert_eq!(result.results[0].distance, 0.0);
    }

    fn native_roundtrip<T: NativeVector + PartialEq + std::fmt::Debug>(values: &[T]) {
        let directory = native_fixture(values);
        let vectors = rowmajor::Ref::try_from_data(values, 300, 4).unwrap();
        let searcher = open_searcher::<T>(directory.path(), 1).unwrap();
        let singleton = searcher
            .search_with_indexed_vectors(
                vectors.row(0),
                1,
                10,
                Some(4),
                SearchMode::single_hop_filter(|id| *id == 0).with_starting_points(vec![0]),
            )
            .unwrap();
        assert_eq!(singleton.stats.result_count, 1);
        assert_eq!(singleton.results[0].vertex_id, 0);
        assert_eq!(singleton.results[0].distance, 0.0);
        assert_eq!(singleton.results[0].indexed_vector.as_ref(), vectors.row(0));
        assert_eq!(singleton.stats.cmps, 1);
        assert_eq!(singleton.stats.query_statistics.total_vertices_loaded, 1);
        let matches = searcher
            .search(
                vectors.row(1),
                5,
                20,
                Some(4),
                SearchMode::single_hop_filter(|id| *id > 0).with_starting_points(vec![1]),
            )
            .unwrap();
        assert_eq!(matches.stats.result_count, 5);
        assert_eq!(matches.results[0].vertex_id, 1);
        assert_eq!(matches.results[0].distance, 0.0);
        assert!(matches
            .results
            .iter()
            .all(|result| (1..300).contains(&result.vertex_id)));
        assert!(matches
            .results
            .windows(2)
            .all(|pair| pair[0].distance <= pair[1].distance));
        assert!(!directory.path().join("index_mem.index").exists());
    }

    #[test]
    fn native_u8_pq_graph_and_vector_roundtrip_with_singleton_label() {
        let values: Vec<u8> = (0..300)
            .flat_map(|id| {
                [
                    (id % 256) as u8,
                    (id / 256) as u8,
                    (id % 251) as u8,
                    (id % 239) as u8,
                ]
            })
            .collect();
        native_roundtrip(&values);
    }

    #[test]
    fn native_f32_pq_graph_and_vector_roundtrip_with_singleton_label() {
        let values: Vec<f32> = (0..300)
            .flat_map(|id| [id as f32, id as f32 / 2.0, (id % 7) as f32, -1.0])
            .collect();
        native_roundtrip(&values);
    }

    #[tokio::test]
    async fn native_searcher_construct_search_drop_and_open_error_are_safe_inside_tokio() {
        let values: Vec<f32> = (0..300)
            .flat_map(|id| [id as f32, id as f32 / 2.0, (id % 7) as f32, -1.0])
            .collect();
        native_roundtrip(&values);

        let directory = native_fixture(&values);
        // Header failure happens after DiskIndexSearcher creates its runtime.
        // Cleanup on this error path must also occur outside the caller's runtime.
        File::options()
            .write(true)
            .open(directory.path().join("index_disk.index"))
            .unwrap()
            .write_all(&[0u8; BLOCK_SIZE])
            .unwrap();
        assert!(open_searcher::<f32>(directory.path(), 1).is_err());
    }

    #[tokio::test]
    async fn blocking_worker_queries_avoid_redispatch_and_restore_sync_protection() {
        let values: Vec<f32> = (0..300)
            .flat_map(|id| [id as f32, id as f32 / 2.0, (id % 7) as f32, -1.0])
            .collect();
        let directory = native_fixture(&values);
        let searcher = open_searcher::<f32>(directory.path(), 1).unwrap();
        tokio::task::spawn_blocking(move || {
            let _directory = directory;
            let worker_id = std::thread::current().id();
            assert!(tokio::runtime::Handle::try_current().is_ok());
            on_blocking_worker(|| {
                on_blocking_worker(|| {
                    let actual = searcher
                        .with_inner(|_| Ok(std::thread::current().id()))
                        .unwrap();
                    assert_eq!(actual, worker_id);
                });
                let actual = searcher
                    .with_inner(|_| Ok(std::thread::current().id()))
                    .unwrap();
                assert_eq!(
                    actual, worker_id,
                    "nested scope must preserve the outer guard"
                );
                let result = searcher
                    .search(
                        &values[..4],
                        1,
                        10,
                        Some(4),
                        SearchMode::single_hop_filter(|id| *id == 0).with_starting_points(vec![0]),
                    )
                    .unwrap();
                assert_eq!(result.stats.result_count, 1);
                assert_eq!(result.results[0].vertex_id, 0);
                assert_eq!(result.results[0].distance, 0.0);
            });
            let protected = searcher
                .with_inner(|_| Ok(std::thread::current().id()))
                .unwrap();
            assert_ne!(
                protected, worker_id,
                "sync safety must resume after leaving the guard"
            );
            drop(searcher);
        })
        .await
        .unwrap();
    }
}
