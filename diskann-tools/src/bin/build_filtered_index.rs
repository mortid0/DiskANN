/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Construct an immutable experimental filtered disk bundle from JSON-lines labels.

use std::{
    fs::{self, File},
    path::PathBuf,
};

use clap::Parser;
use diskann_disk::{
    build::{
        filtered::{build_stitched, StitchedBuildConfig},
        filtered_export::write_stitched,
    },
    label_metadata::LabelMetadata,
};
use diskann_providers::{
    model::NUM_PQ_CENTROIDS,
    storage::{get_mem_index_file, FileStorageProvider},
};
use diskann_tools::utils::{
    build_pq,
    filtered_disk::{finish_bundle, index_prefix, LABELS_MANIFEST},
    filtered_labels::load_labels,
    BuildPQParameters,
};
use diskann_utils::{io::read_bin, views::rowmajor::Matrix};
use diskann_vector::distance::Metric;

/// Build an immutable experimental f32/L2 stitched disk index in RAM.
#[derive(Parser)]
struct Args {
    #[arg(long)]
    data_file: PathBuf,
    #[arg(long)]
    labels: PathBuf,
    #[arg(long, default_value = "/label")]
    label_field: String,
    /// New directory; existing directories are never overwritten.
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 16)]
    label_degree: usize,
    #[arg(long, default_value_t = 32)]
    max_degree: usize,
    #[arg(long, default_value_t = 64)]
    l_build: usize,
    #[arg(long, default_value_t = 1.2)]
    alpha: f32,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    pq_bytes: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let vectors = read_bin::<f32>(&mut File::open(&args.data_file)?)?;
    anyhow::ensure!(
        vectors.nrows() >= NUM_PQ_CENTROIDS,
        "PQ training requires at least {NUM_PQ_CENTROIDS} rows"
    );
    anyhow::ensure!(
        args.pq_bytes > 0 && args.pq_bytes <= vectors.ncols(),
        "pq-bytes must be in 1..=dimensions"
    );
    let config = StitchedBuildConfig::new(
        args.label_degree,
        args.max_degree,
        args.l_build,
        args.alpha,
        args.threads,
        args.seed,
    )?;
    let labels = load_labels(&args.labels, vectors.nrows(), &args.label_field)?;
    let prefix = index_prefix(&args.output_dir)?;
    let data_path = args
        .data_file
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("data path must be UTF-8"))?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.threads)
        .build()?;
    let graph = runtime.block_on(build_stitched(
        vectors.as_view(),
        labels
            .postings()
            .iter()
            .map(|(&label, ids)| (label, ids.as_slice())),
        &config,
    ))?;
    let metadata = LabelMetadata::from_graph(&graph, labels.names())?;
    fs::create_dir(&args.output_dir)?;
    write_stitched(&graph, data_path, &prefix, &FileStorageProvider)?;
    drop(graph);
    drop(vectors);
    drop(runtime);
    fs::remove_file(get_mem_index_file(&prefix))?;
    build_pq::<f32>(
        &FileStorageProvider,
        BuildPQParameters {
            metric: Metric::L2,
            data_path,
            index_path_prefix: &prefix,
            num_threads: args.threads,
            p_val: 1.0,
            pq_bytes: args.pq_bytes as f64,
        },
    )?;
    metadata.save(&args.output_dir, args.output_dir.join(LABELS_MANIFEST))?;
    finish_bundle(&args.output_dir)?;
    eprintln!(
        "Wrote experimental filtered disk index to {}",
        args.output_dir.display()
    );
    Ok(())
}
