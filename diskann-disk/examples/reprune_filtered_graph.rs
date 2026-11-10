/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Quality-only counterfactual for the stitched-union preservation change.
//!
//! Reads a new native uint8 graph and emits a separate graph with the old
//! unconditional final prune. Entries, PQ assets, vectors, layout and construction
//! history stay fixed. The output is a diagnostic asset, never a published generation.

use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use anyhow::{ensure, Context, Result};
use diskann::utils::VectorRepr;
use diskann_disk::{data_model::GraphHeader, filtered::metadata::LabelMetadata};
use diskann_providers::utils::{create_thread_pool, ParallelIteratorInPool};
use diskann_vector::{distance::Metric, DistanceFunction};
use rayon::prelude::*;
use serde_json::json;

struct Settings {
    input_graph: PathBuf,
    labels: PathBuf,
    vectors: PathBuf,
    out_dir: PathBuf,
    threads: usize,
    alpha: f32,
}

#[derive(Default)]
struct Counts {
    rows: AtomicU64,
    changed_rows: AtomicU64,
    edges_before: AtomicU64,
    edges_after: AtomicU64,
}

fn labels_covered(metadata: &LabelMetadata, source: u32, candidate: u32, selected: u32) -> bool {
    let a = metadata.labels(source);
    let b = metadata.labels(candidate);
    let c = metadata.labels(selected);
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

fn protected_edges(metadata: &LabelMetadata, source: u32) -> Result<Vec<u32>> {
    let mut protected = Vec::with_capacity(metadata.labels(source).len());
    // The original builder iterates labels in sorted order and keeps one unique
    // ID-order cycle successor per label as the protected prefix.
    for &label in metadata.labels(source) {
        let posting = metadata.posting(label);
        if posting.len() < 2 {
            continue;
        }
        let position = posting
            .binary_search(&source)
            .map_err(|_| anyhow::anyhow!("CSR source absent from its posting list"))?;
        let next = posting[(position + 1) % posting.len()];
        if !protected.contains(&next) {
            protected.push(next);
        }
    }
    Ok(protected)
}

// This deliberately reproduces the old one-shot final prune rather than calling
// the fixed production builder. It is idempotent with its selected candidate set:
// rejected candidates never become occluders, so removing them changes no decision.
fn legacy_prune<D: for<'a> DistanceFunction<&'a [u8], &'a [u8], f32>>(
    source: u32,
    row: &[u32],
    vectors: &[u8],
    dimensions: usize,
    metadata: &LabelMetadata,
    alpha: f32,
    distance: &D,
) -> Result<Vec<u32>> {
    let mut selected = protected_edges(metadata, source)?;
    ensure!(
        row.starts_with(&selected),
        "protected-prefix mismatch at row {source}"
    );
    let vector = |id: u32| {
        let start = id as usize * dimensions;
        &vectors[start..start + dimensions]
    };
    let mut candidates = row.to_vec();
    candidates.sort_unstable();
    candidates.dedup();
    candidates.retain(|candidate| *candidate != source && !selected.contains(candidate));
    let mut scored = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        ensure!(
            (candidate as usize) < metadata.point_count(),
            "invalid graph ID"
        );
        let score = distance.evaluate_similarity(vector(source), vector(candidate));
        ensure!(
            score.is_finite() && score >= 0.,
            "invalid source-edge distance"
        );
        scored.push((candidate, score));
    }
    scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    // Input rows already fit the fixed SSD degree bound, so the old output cap
    // cannot truncate a newly accepted candidate. Keep every pruning decision.
    for (candidate, score) in scored {
        let mut occluded = false;
        for &other in &selected {
            if labels_covered(metadata, source, candidate, other) {
                let between = distance.evaluate_similarity(vector(candidate), vector(other));
                ensure!(
                    between.is_finite() && between >= 0.,
                    "invalid occlusion distance"
                );
                if alpha * between <= score {
                    occluded = true;
                    break;
                }
            }
        }
        if !occluded {
            selected.push(candidate);
        }
    }
    Ok(selected)
}

fn rewrite(settings: &Settings) -> Result<serde_json::Value> {
    ensure!(settings.threads > 0 && settings.alpha.is_finite() && settings.alpha >= 1.);
    ensure!(!settings.out_dir.exists(), "output directory must be fresh");
    let started = Instant::now();
    let metadata: LabelMetadata = bincode::deserialize(&std::fs::read(&settings.labels)?)?;
    let matrix = std::fs::read(&settings.vectors)?;
    ensure!(matrix.len() >= 8, "short vector matrix");
    let rows = u32::from_le_bytes(matrix[0..4].try_into()?) as usize;
    let dimensions = u32::from_le_bytes(matrix[4..8].try_into()?) as usize;
    ensure!(rows > 0 && dimensions > 0 && rows == metadata.point_count());
    ensure!(matrix.len() == rows.checked_mul(dimensions).context("matrix overflow")? + 8);
    let vectors = &matrix[8..];
    let mut input = File::open(&settings.input_graph)?;
    let mut header_bytes = vec![0; 8 + GraphHeader::get_size()];
    input.read_exact(&mut header_bytes)?;
    ensure!(u32::from_le_bytes(header_bytes[0..4].try_into()?) as usize == GraphHeader::get_size());
    ensure!(u32::from_le_bytes(header_bytes[4..8].try_into()?) == 1);
    let header = GraphHeader::try_from(&header_bytes[8..])?;
    let stored = header.metadata();
    let block = usize::try_from(header.block_size())?;
    let width = usize::try_from(stored.node_len)?;
    let per_block = usize::try_from(stored.num_nodes_per_block)?;
    ensure!(stored.num_pts as usize == rows && stored.dims == dimensions);
    ensure!(stored.associated_data_length == 0 && stored.vamana_frozen_num == 0);
    ensure!(width > dimensions + 4 && block > header_bytes.len());
    ensure!(
        per_block > 0 && per_block == block / width,
        "requires packed native uint8 nodes"
    );
    let degree_bound = header.max_degree::<u8>()?;
    let sectors = rows.div_ceil(per_block);
    ensure!(input.metadata()?.len() == (sectors as u64 + 1) * block as u64);
    ensure!(stored.disk_index_file_size == input.metadata()?.len());
    let mut first_sector = vec![0; block];
    first_sector[..header_bytes.len()].copy_from_slice(&header_bytes);
    input.read_exact(&mut first_sector[header_bytes.len()..])?;

    std::fs::create_dir(&settings.out_dir)?;
    let output_path = settings.out_dir.join("index_disk.index");
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&output_path)?;
    output.write_all(&first_sector)?;
    let pool = create_thread_pool(settings.threads)?;
    let distance = u8::distance(Metric::L2, Some(dimensions));
    let counts = Counts::default();
    let batch_sectors = (16 * 1024 * 1024 / block).max(1);
    let mut buffer = vec![0; batch_sectors * block];
    for start in (0..sectors).step_by(batch_sectors) {
        let count = batch_sectors.min(sectors - start);
        let batch = &mut buffer[..count * block];
        input.read_exact(batch)?;
        batch
            .par_chunks_mut(block)
            .enumerate()
            .try_for_each_in_pool(pool.as_ref(), |(local, sector)| -> Result<()> {
                let first = (start + local) * per_block;
                for within in 0..per_block.min(rows - first) {
                    let source = (first + within) as u32;
                    let node = &mut sector[within * width..(within + 1) * width];
                    let vector_start = source as usize * dimensions;
                    ensure!(
                        node[..dimensions] == vectors[vector_start..vector_start + dimensions],
                        "graph and vector file differ at row {source}"
                    );
                    let size =
                        u32::from_le_bytes(node[dimensions..dimensions + 4].try_into()?) as usize;
                    ensure!(size <= degree_bound, "degree exceeds fixed SSD layout");
                    let row: Vec<_> = node[dimensions + 4..dimensions + 4 + size * 4]
                        .chunks_exact(4)
                        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
                        .collect();
                    let selected = legacy_prune(
                        source,
                        &row,
                        vectors,
                        dimensions,
                        &metadata,
                        settings.alpha,
                        &distance,
                    )?;
                    ensure!(selected.len() <= size, "reprune must only remove edges");
                    counts.rows.fetch_add(1, Ordering::Relaxed);
                    counts
                        .changed_rows
                        .fetch_add(u64::from(row != selected), Ordering::Relaxed);
                    counts
                        .edges_before
                        .fetch_add(size as u64, Ordering::Relaxed);
                    counts
                        .edges_after
                        .fetch_add(selected.len() as u64, Ordering::Relaxed);
                    node[dimensions..dimensions + 4]
                        .copy_from_slice(&(selected.len() as u32).to_le_bytes());
                    for (slot, neighbor) in node[dimensions + 4..].chunks_exact_mut(4).zip(selected)
                    {
                        slot.copy_from_slice(&neighbor.to_le_bytes());
                    }
                }
                Ok(())
            })?;
        output.write_all(batch)?;
    }
    output.sync_all()?;
    let value = json!({
        "schema_version": 1,
        "scope": "quality-only legacy-prune counterfactual of the same per-label construction",
        "input_graph": settings.input_graph, "labels": settings.labels, "vectors": settings.vectors,
        "output_graph": output_path,
        "rows": counts.rows.load(Ordering::Relaxed), "dimensions": dimensions,
        "degree_bound": degree_bound, "alpha": settings.alpha, "threads": settings.threads,
        "changed_rows": counts.changed_rows.load(Ordering::Relaxed),
        "edges_before": counts.edges_before.load(Ordering::Relaxed),
        "edges_after": counts.edges_after.load(Ordering::Relaxed),
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "publication": "no generation manifest, CURRENT, CATALOG, labels, entries or PQ artifacts written",
        "precondition": "input came from the union-preserving builder with this alpha and protected-cycle policy",
    });
    std::fs::write(
        settings.out_dir.join("legacy-reprune.json"),
        serde_json::to_vec_pretty(&value)?,
    )?;
    Ok(value)
}

fn required(args: &BTreeMap<String, String>, name: &str) -> Result<PathBuf> {
    args.get(name)
        .map(PathBuf::from)
        .with_context(|| format!("--{name} is required"))
}

fn main() -> Result<()> {
    let mut args = BTreeMap::new();
    let mut values = std::env::args().skip(1);
    while let Some(key) = values.next() {
        ensure!(key.starts_with("--"), "arguments use --name value pairs");
        let value = values.next().context("missing argument value")?;
        ensure!(
            args.insert(key[2..].to_owned(), value).is_none(),
            "duplicate argument"
        );
    }
    let settings = Settings {
        input_graph: required(&args, "input-graph")?,
        labels: required(&args, "labels")?,
        vectors: required(&args, "vectors")?,
        out_dir: required(&args, "out-dir")?,
        threads: args.get("threads").map_or(Ok(8), |value| value.parse())?,
        alpha: args.get("alpha").map_or(Ok(1.2), |value| value.parse())?,
    };
    println!("{}", rewrite(&settings)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_disk::data_model::{GraphLayoutVersion, GraphMetadata};

    fn metadata(rows: usize) -> LabelMetadata {
        LabelMetadata::from_postings(
            rows,
            BTreeMap::from([(7, (0..rows as u32).collect())]),
            BTreeMap::from([(7, vec![0])]),
        )
        .unwrap()
    }

    #[test]
    fn legacy_counterfactual_removes_the_retained_edge_and_is_idempotent() {
        let values = [0u8, 10, 11, 20];
        let metadata = metadata(values.len());
        let distance = u8::distance(Metric::L2, Some(1));
        let selected = legacy_prune(0, &[1, 2, 3], &values, 1, &metadata, 1.2, &distance).unwrap();
        assert_eq!(selected, [1]);
        assert_eq!(
            legacy_prune(0, &selected, &values, 1, &metadata, 1.2, &distance).unwrap(),
            selected
        );
        assert!(legacy_prune(0, &[3, 2, 1], &values, 1, &metadata, 1.2, &distance).is_err());
    }

    #[test]
    fn legacy_prune_is_idempotent_for_overlapping_label_cycles_and_alphas() {
        let vectors: Vec<_> = (0u8..32).flat_map(|id| [id * 7, (id % 5) * 30]).collect();
        let metadata = LabelMetadata::from_postings(
            32,
            BTreeMap::from([(7, (0..32).collect()), (99, (0..32).step_by(2).collect())]),
            BTreeMap::from([(7, vec![0]), (99, vec![0])]),
        )
        .unwrap();
        let distance = u8::distance(Metric::L2, Some(2));
        for source in 0..32 {
            let protected = protected_edges(&metadata, source).unwrap();
            let mut row = protected.clone();
            row.extend(
                (0..32)
                    .rev()
                    .filter(|id| *id != source && !protected.contains(id)),
            );
            for alpha in [1., 1.2, 1.44] {
                let selected =
                    legacy_prune(source, &row, &vectors, 2, &metadata, alpha, &distance).unwrap();
                assert!(selected.starts_with(&protected));
                assert_eq!(
                    legacy_prune(source, &selected, &vectors, 2, &metadata, alpha, &distance)
                        .unwrap(),
                    selected
                );
            }
        }
    }

    #[test]
    fn sector_export_preserves_input_header_vectors_and_layout() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        let vectors = [0u8, 10, 11, 20];
        let block = 4096usize;
        let width = 1 + 4 + 4 * 3usize;
        let mut graph = vec![0; block * 2];
        let header = GraphHeader::new(
            GraphMetadata::new(
                4,
                1,
                0,
                width as u64,
                (block / width) as u64,
                0,
                0,
                graph.len() as u64,
                0,
            ),
            block as u64,
            GraphLayoutVersion::new(1, 0),
        )
        .to_bytes()
        .unwrap();
        graph[..4].copy_from_slice(&(header.len() as u32).to_le_bytes());
        graph[4..8].copy_from_slice(&1u32.to_le_bytes());
        graph[8..8 + header.len()].copy_from_slice(&header);
        for source in 0..4 {
            let offset = block as usize + source * width as usize;
            graph[offset] = vectors[source];
            let mut row = vec![((source + 1) % 4) as u32];
            let protected = row[0];
            row.extend((0..4).filter(|id| *id != source as u32 && *id != protected));
            graph[offset + 1..offset + 5].copy_from_slice(&3u32.to_le_bytes());
            for (index, id) in row.iter().enumerate() {
                graph[offset + 5 + index * 4..offset + 9 + index * 4]
                    .copy_from_slice(&id.to_le_bytes());
            }
        }
        let mut matrix = [4u32.to_le_bytes(), 1u32.to_le_bytes()].concat();
        matrix.extend(vectors);
        let settings = Settings {
            input_graph: root.join("original.index"),
            labels: root.join("labels.bin"),
            vectors: root.join("vectors.bin"),
            out_dir: root.join("counterfactual"),
            threads: 2,
            alpha: 1.2,
        };
        std::fs::write(&settings.input_graph, &graph).unwrap();
        std::fs::write(&settings.vectors, &matrix).unwrap();
        std::fs::write(&settings.labels, bincode::serialize(&metadata(4)).unwrap()).unwrap();
        let report = rewrite(&settings).unwrap();
        assert_eq!(report["rows"], 4);
        assert!(report["changed_rows"].as_u64().unwrap() > 0);
        assert_eq!(std::fs::read(&settings.input_graph).unwrap(), graph);
        assert_eq!(std::fs::read(&settings.vectors).unwrap(), matrix);
        let output = std::fs::read(settings.out_dir.join("index_disk.index")).unwrap();
        assert_eq!(output.len(), graph.len());
        assert_eq!(&output[..block as usize], &graph[..block as usize]);
        for (source, &value) in vectors.iter().enumerate() {
            assert_eq!(output[block as usize + source * width as usize], value);
        }
        assert!(!settings.out_dir.join("CURRENT").exists());
        assert!(rewrite(&settings).is_err());
    }
}
