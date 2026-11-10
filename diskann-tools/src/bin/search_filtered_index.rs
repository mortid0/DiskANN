/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Reopen an experimental filtered disk bundle and emit labeled query results.

use std::{
    fs::File,
    io::{self, Write},
    path::PathBuf,
};

use clap::Parser;
use diskann_tools::utils::filtered_disk::FilteredDiskIndex;
use diskann_utils::{io::read_bin, views::rowmajor::Matrix};
use serde::Serialize;

/// Reopen an experimental disk bundle and search an atomic label for each query.
#[derive(Parser)]
struct Args {
    #[arg(long)]
    index_dir: PathBuf,
    #[arg(long)]
    query_file: PathBuf,
    #[arg(long)]
    label: String,
    #[arg(long, default_value_t = 10)]
    k: u32,
    #[arg(long, default_value_t = 64)]
    l: u32,
    #[arg(long, default_value_t = 4)]
    beam: usize,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    #[arg(long, default_value_t = usize::MAX)]
    io_limit: usize,
    #[arg(long, default_value_t = 0)]
    cache_nodes: usize,
}

#[derive(Serialize)]
struct Row {
    query_id: usize,
    ids: Vec<u32>,
    squared_distances: Vec<f32>,
    distance_comparisons: u32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let index = FilteredDiskIndex::open(
        &args.index_dir,
        args.threads,
        args.io_limit,
        args.cache_nodes,
    )?;
    let queries = read_bin::<f32>(&mut File::open(args.query_file)?)?;
    let mut output = io::stdout().lock();
    for query_id in 0..queries.nrows() {
        let result = index.search(
            queries.row(query_id),
            &args.label,
            args.k,
            args.l,
            args.beam,
        )?;
        serde_json::to_writer(
            &mut output,
            &Row {
                query_id,
                ids: result.results.iter().map(|r| r.vertex_id).collect(),
                squared_distances: result.results.iter().map(|r| r.distance).collect(),
                distance_comparisons: result.stats.cmps,
            },
        )?;
        output.write_all(b"\n")?;
    }
    Ok(())
}
