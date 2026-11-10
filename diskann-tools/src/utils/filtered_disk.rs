/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Immutable experimental disk bundles. Checksums bind graph, PQ, and labels on open.
//! See `rfcs/00000-experimental-filtered-disk-format.md` for the draft format contract.

use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use anyhow::Context;
use diskann::graph::search::Knn;
use diskann_disk::{
    build::disk_index_build_parameter::DISK_SECTOR_LEN,
    data_model::{AdHoc, CachingStrategy, GraphHeader},
    label_metadata::LabelMetadata,
    search::{
        provider::{
            disk_provider::{DiskIndexSearcher, SearchResult, SearchResultStats},
            disk_vertex_provider_factory::DiskVertexProviderFactory,
        },
        traits::VertexProviderFactory,
    },
    storage::disk_index_reader::DiskIndexReader,
};
use diskann_providers::storage::{
    get_disk_index_compressed_pq_file, get_disk_index_file, get_disk_index_pq_pivot_file,
    FileStorageProvider,
};
use diskann_vector::distance::Metric;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MANIFEST: &str = "bundle.json";
pub const LABELS_MANIFEST: &str = "labels.json";

/// The fixed prefix prevents manifest-controlled paths from escaping the directory.
pub fn index_prefix(directory: &Path) -> anyhow::Result<String> {
    directory
        .join("index")
        .into_os_string()
        .into_string()
        .map_err(|_| anyhow::anyhow!("index directory must have a UTF-8 path"))
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fingerprint {
    bytes: u64,
    sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format: String,
    version: u32,
    graph: Fingerprint,
    pq_pivots: Fingerprint,
    pq_vectors: Fingerprint,
    labels: Fingerprint,
}

fn fingerprint(path: &Path) -> anyhow::Result<Fingerprint> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    let mut bytes = 0;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        bytes += read as u64;
    }
    Ok(Fingerprint {
        bytes,
        sha256: format!("{:x}", hasher.finalize()),
    })
}

fn paths(directory: &Path) -> anyhow::Result<[PathBuf; 4]> {
    let prefix = index_prefix(directory)?;
    Ok([
        get_disk_index_file(&prefix).into(),
        get_disk_index_pq_pivot_file(&prefix).into(),
        get_disk_index_compressed_pq_file(&prefix).into(),
        directory.join(LABELS_MANIFEST),
    ])
}

/// Write the completion manifest after all artifacts have been closed.
///
/// # Errors
/// Fails if an artifact cannot be read or the completion manifest already exists.
/// This is an immutable export, without a crash-durability or update protocol.
pub fn finish_bundle(directory: &Path) -> anyhow::Result<()> {
    let [graph, pq_pivots, pq_vectors, labels] = paths(directory)?;
    let manifest = Manifest {
        format: "diskann-experimental-single-hop-f32-l2".into(),
        version: 1,
        graph: fingerprint(&graph)?,
        pq_pivots: fingerprint(&pq_pivots)?,
        pq_vectors: fingerprint(&pq_vectors)?,
        labels: fingerprint(&labels)?,
    };
    let mut file = File::create_new(directory.join(MANIFEST))?;
    serde_json::to_writer_pretty(&mut file, &manifest)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn verify_bundle(directory: &Path) -> anyhow::Result<()> {
    let file = File::open(directory.join(MANIFEST)).context("opening completed bundle")?;
    let manifest: Manifest = serde_json::from_reader(file)?;
    anyhow::ensure!(
        manifest.version == 1 && manifest.format == "diskann-experimental-single-hop-f32-l2",
        "unsupported experimental bundle version or encoding"
    );
    for (path, expected) in paths(directory)?.iter().zip([
        manifest.graph,
        manifest.pq_pivots,
        manifest.pq_vectors,
        manifest.labels,
    ]) {
        let actual = fingerprint(path)?;
        anyhow::ensure!(
            actual.bytes == expected.bytes && actual.sha256 == expected.sha256,
            "checksum or length mismatch for {}",
            path.display()
        );
    }
    Ok(())
}

/// An opened bundle keeps compressed memberships and PQ codes, but no graph adjacency.
/// Files must remain immutable while the index is open. Open and query this synchronous
/// API outside a Tokio runtime, as required by the underlying disk searcher.
pub struct FilteredDiskIndex {
    metadata: LabelMetadata,
    searcher: DiskIndexSearcher<AdHoc<f32>>,
    dimensions: usize,
}

impl FilteredDiskIndex {
    /// Verify all artifacts once, then open their disk-backed graph and compact labels.
    ///
    /// # Errors
    /// Rejects incomplete, unsupported, corrupt, or mismatched bundles and storage failures.
    pub fn open(
        directory: &Path,
        threads: usize,
        io_limit: usize,
        cache_nodes: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(threads > 0, "threads must be positive");
        verify_bundle(directory)?;
        let metadata = LabelMetadata::load(&directory.join(LABELS_MANIFEST), directory)?;
        let prefix = index_prefix(directory)?;
        let reader = DiskIndexReader::new(
            get_disk_index_pq_pivot_file(&prefix),
            get_disk_index_compressed_pq_file(&prefix),
            &FileStorageProvider,
        )?;
        anyhow::ensure!(
            metadata.rows() == reader.get_num_points(),
            "label/PQ row count mismatch"
        );
        let graph_path = get_disk_index_file(&prefix);
        // Validate the header before optional cache construction traverses graph nodes.
        let factory = DiskVertexProviderFactory::<AdHoc<f32>, _>::from_disk_index_path(
            graph_path.clone(),
            CachingStrategy::None,
        )?;
        let header = factory.get_header()?;
        let graph = header.metadata();
        let dimensions = reader.get_pq_data().get_dim();
        anyhow::ensure!(
            header.layout_version() == &GraphHeader::CURRENT_LAYOUT_VERSION,
            "unsupported experimental graph layout version"
        );
        anyhow::ensure!(
            graph.num_pts == metadata.rows() as u64,
            "graph/label/PQ row count mismatch"
        );
        anyhow::ensure!(
            graph.dims == dimensions && dimensions > 0,
            "graph/PQ dimension mismatch"
        );
        anyhow::ensure!(
            graph.vamana_frozen_num == 0
                && graph.vamana_frozen_loc == 0
                && graph.associated_data_length == 0,
            "experimental graph must have no frozen points or associated data"
        );
        anyhow::ensure!(
            graph.medoid < graph.num_pts,
            "graph entry point is out of bounds"
        );
        let degree = header.max_degree::<f32>()?;
        anyhow::ensure!(
            degree > 0 && degree <= u32::MAX as usize,
            "invalid graph degree"
        );
        let block_size = DISK_SECTOR_LEN as u64;
        anyhow::ensure!(
            header.block_size() == block_size
                && graph.node_len.is_multiple_of(4)
                && graph.num_nodes_per_block == block_size / graph.node_len,
            "invalid experimental graph block layout"
        );
        let data_blocks = if graph.num_nodes_per_block == 0 {
            graph
                .num_pts
                .checked_mul(graph.node_len.div_ceil(block_size))
        } else {
            Some(graph.num_pts.div_ceil(graph.num_nodes_per_block))
        };
        let expected_size = data_blocks
            .and_then(|blocks| blocks.checked_add(1))
            .and_then(|blocks| blocks.checked_mul(block_size));
        anyhow::ensure!(
            expected_size == Some(graph.disk_index_file_size)
                && graph.disk_index_file_size == std::fs::metadata(&graph_path)?.len(),
            "graph layout and file length mismatch"
        );
        let factory = if cache_nodes == 0 {
            factory
        } else {
            DiskVertexProviderFactory::from_disk_index_path(
                graph_path,
                CachingStrategy::StaticCacheWithBfsNodes(cache_nodes),
            )?
        };
        let searcher =
            DiskIndexSearcher::new(threads, io_limit, &reader, factory, Metric::L2, None)?;
        Ok(Self {
            metadata,
            searcher,
            dimensions,
        })
    }

    /// Search by external label name. Unknown labels produce an empty result.
    ///
    /// # Errors
    /// Returns invalid-query or disk I/O errors from the searcher.
    pub fn search(
        &self,
        query: &[f32],
        label: &str,
        k: u32,
        l: u32,
        beam: usize,
    ) -> anyhow::Result<SearchResult<()>> {
        Knn::new(l as usize, Some(beam))?;
        anyhow::ensure!(
            l >= k,
            "search list size {l} must be at least requested result count {k}"
        );
        anyhow::ensure!(query.len() == self.dimensions, "query dimension mismatch");
        let Some(label) = self.metadata.resolve_label(label) else {
            return Ok(SearchResult {
                results: Vec::new(),
                stats: SearchResultStats {
                    cmps: 0,
                    result_count: 0,
                    query_statistics: Default::default(),
                },
            });
        };
        Ok(self
            .searcher
            .search_single_hop(query, k, l, Some(beam), &self.metadata, label)?)
    }
}
