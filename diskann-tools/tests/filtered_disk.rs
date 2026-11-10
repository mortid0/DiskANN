/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![cfg(feature = "experimental_single_hop_search")]

use std::{
    fs::{self, File},
    io::Write,
    path::PathBuf,
    process::{Command, Output},
};

use diskann_disk::data_model::{GraphHeader, GraphLayoutVersion};
use diskann_providers::storage::{
    get_disk_index_compressed_pq_file, get_disk_index_file, get_disk_index_pq_pivot_file,
};
use diskann_tools::utils::filtered_disk::{
    finish_bundle, index_prefix, FilteredDiskIndex, LABELS_MANIFEST,
};
use diskann_utils::{
    io::{read_bin, write_bin},
    views::rowmajor::{self, Matrix},
};
use serde::Deserialize;
use serde_json::json;

const QUERIES: [u32; 3] = [42, 120, 252];

struct Fixture {
    _temporary: tempfile::TempDir,
    data: PathBuf,
    labels: PathBuf,
    queries: PathBuf,
    index: PathBuf,
}

fn successful(command: &mut Command) -> Output {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{command:?} failed: {}\nstdout: {}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    output
}

impl Fixture {
    fn build() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let data = temporary.path().join("vectors.bin");
        let labels = temporary.path().join("labels.jsonl");
        let queries = temporary.path().join("queries.bin");
        let index = temporary.path().join("index");
        let vectors: Vec<f32> = (0..256)
            .flat_map(|row| [row as f32, (row * 2) as f32, (row * 3) as f32])
            .collect();
        write_bin(
            rowmajor::Ref::try_from_data(&vectors, 256, 3).unwrap(),
            &mut File::create(&data).unwrap(),
        )
        .unwrap();
        let query_vectors: Vec<f32> = QUERIES
            .into_iter()
            .flat_map(|row| [row as f32, (row * 2) as f32, (row * 3) as f32])
            .collect();
        write_bin(
            rowmajor::Ref::try_from_data(&query_vectors, QUERIES.len(), 3).unwrap(),
            &mut File::create(&queries).unwrap(),
        )
        .unwrap();
        let mut documents = File::create(&labels).unwrap();
        for row in 0..256 {
            let mut labels = Vec::new();
            if row < 254 {
                if row % 2 == 0 {
                    labels.push("red");
                }
                if row % 3 == 0 {
                    labels.push("blue");
                }
            } else if row == 254 {
                labels.push("singleton");
            }
            // There is deliberately no label covering all rows: unlabeled and
            // singleton-only rows must pass the ordinary disk writer too.
            serde_json::to_writer(&mut documents, &json!({"doc_id": row, "label": labels}))
                .unwrap();
            documents.write_all(b"\n").unwrap();
        }
        drop(documents);
        let fixture = Self {
            _temporary: temporary,
            data,
            labels,
            queries,
            index,
        };
        // Wait for the builder to exit before any search process starts. No
        // construction graph or process-local metadata can survive this boundary.
        successful(&mut fixture.builder());
        fixture
    }

    fn builder(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_build_filtered_index"));
        command
            .arg("--data-file")
            .arg(&self.data)
            .arg("--labels")
            .arg(&self.labels)
            .arg("--output-dir")
            .arg(&self.index)
            .args([
                "--label-degree",
                "4",
                "--max-degree",
                "8",
                "--l-build",
                "16",
                "--pq-bytes",
                "2",
                "--threads",
                "1",
            ]);
        command
    }

    fn searcher(&self, label: &str, cache_nodes: usize) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_search_filtered_index"));
        command
            .arg("--index-dir")
            .arg(&self.index)
            .arg("--query-file")
            .arg(&self.queries)
            .arg("--label")
            .arg(label)
            .arg("--cache-nodes")
            .arg(cache_nodes.to_string())
            .args(["--k", "3", "--l", "256", "--beam", "4", "--threads", "1"]);
        command
    }

    fn search(&self, label: &str, cache_nodes: usize) -> Vec<Row> {
        let output = successful(&mut self.searcher(label, cache_nodes));
        let text = String::from_utf8(output.stdout).unwrap();
        let rows: Vec<Row> = text
            .lines()
            .map(|line| {
                serde_json::from_str(line).unwrap_or_else(|error| panic!("{error}: {line}"))
            })
            .collect();
        assert_eq!(rows.len(), QUERIES.len());
        rows
    }

    fn rejects_open(&self) {
        let output = self.searcher("red", 0).output().unwrap();
        assert!(
            !output.status.success(),
            "corrupt or incomplete bundle was accepted: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert!(!output.stderr.is_empty());
    }

    fn artifacts(&self) -> Vec<PathBuf> {
        let prefix = index_prefix(&self.index).unwrap();
        let label_manifest = self.index.join(LABELS_MANIFEST);
        let labels: serde_json::Value =
            serde_json::from_slice(&fs::read(&label_manifest).unwrap()).unwrap();
        let label_artifact = self
            .index
            .join(labels["value"]["artifact"]["$handle"].as_str().unwrap());
        vec![
            get_disk_index_file(&prefix).into(),
            get_disk_index_pq_pivot_file(&prefix).into(),
            get_disk_index_compressed_pq_file(&prefix).into(),
            label_manifest,
            label_artifact,
        ]
    }

    fn graph_header(&self) -> GraphHeader {
        let path = get_disk_index_file(&index_prefix(&self.index).unwrap());
        let bytes = read_bin::<u8>(&mut File::open(path).unwrap()).unwrap();
        GraphHeader::try_from(bytes.as_slice()).unwrap()
    }

    fn replace_header(&self, header: &GraphHeader) {
        let path = get_disk_index_file(&index_prefix(&self.index).unwrap());
        let bytes = header.to_bytes().unwrap();
        // The canonical writer stores its GraphHeader inside the standard binary
        // matrix wrapper. Rewrite that wrapper without modifying the graph blocks.
        write_bin(
            rowmajor::Ref::column_vector(&bytes),
            &mut File::options().write(true).open(path).unwrap(),
        )
        .unwrap();
        fs::remove_file(self.index.join("bundle.json")).unwrap();
        finish_bundle(&self.index).unwrap();
    }
}

#[derive(Debug, Deserialize)]
struct Row {
    query_id: usize,
    ids: Vec<u32>,
    squared_distances: Vec<f32>,
    distance_comparisons: u32,
}

#[test]
fn subprocess_build_and_reopen_preserve_membership_and_node_cache_results() {
    let fixture = Fixture::build();
    for label in ["red", "blue"] {
        let ordinary = fixture.search(label, 0);
        let cached = fixture.search(label, 32);
        let reopened = fixture.search(label, 0);
        for (query_id, ((ordinary, cached), reopened)) in
            ordinary.iter().zip(&cached).zip(&reopened).enumerate()
        {
            assert_eq!(ordinary.query_id, query_id);
            assert_eq!(ordinary.ids.len(), 3);
            assert_eq!(ordinary.ids.len(), ordinary.squared_distances.len());
            assert_eq!(ordinary.ids[0], QUERIES[query_id]);
            assert_eq!(ordinary.squared_distances[0], 0.);
            assert!(ordinary.distance_comparisons > 0);
            assert!(ordinary.ids.iter().all(|&id| {
                id < 254
                    && if label == "red" {
                        id % 2 == 0
                    } else {
                        id % 3 == 0
                    }
            }));
            assert_eq!(ordinary.ids, cached.ids);
            assert_eq!(ordinary.squared_distances, cached.squared_distances);
            assert_eq!(ordinary.ids, reopened.ids);
            assert_eq!(ordinary.squared_distances, reopened.squared_distances);
        }
    }
    for row in fixture.search("singleton", 0) {
        assert_eq!(row.ids, [254]);
        assert_eq!(row.squared_distances.len(), 1);
        assert!(row.distance_comparisons > 0);
    }
    for row in fixture.search("unknown", 32) {
        assert!(row.ids.is_empty());
        assert!(row.squared_distances.is_empty());
        assert_eq!(row.distance_comparisons, 0);
    }
}

#[test]
fn subprocess_open_rejects_corruption_and_incomplete_exports_without_overwriting() {
    let fixture = Fixture::build();
    for artifact in fixture.artifacts() {
        let original = fs::read(&artifact).unwrap();
        assert!(!original.is_empty());
        let mut changed = original.clone();
        changed[original.len() / 2] ^= 0x80;
        fs::write(&artifact, changed).unwrap();
        fixture.rejects_open();
        let mut zero_tail = original.clone();
        zero_tail.extend_from_slice(&[0; 24]);
        fs::write(&artifact, zero_tail).unwrap();
        fixture.rejects_open();
        fs::write(&artifact, original).unwrap();
    }

    let completion = fixture.index.join("bundle.json");
    let pending = fixture.index.join("bundle.pending");
    fs::rename(&completion, &pending).unwrap();
    fixture.rejects_open();
    fs::rename(&pending, &completion).unwrap();

    let original = fs::read(&completion).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&original).unwrap();
    value["version"] = json!(999);
    fs::write(&completion, serde_json::to_vec(&value).unwrap()).unwrap();
    fixture.rejects_open();
    fs::write(&completion, &original).unwrap();

    let artifacts = fixture.artifacts();
    let contents: Vec<_> = artifacts
        .iter()
        .map(|path| fs::read(path).unwrap())
        .collect();
    let sentinel = fixture.index.join("keep.txt");
    fs::write(&sentinel, b"existing output must remain unchanged").unwrap();
    let output = fixture.builder().output().unwrap();
    assert!(
        !output.status.success(),
        "builder overwrote an existing index"
    );
    assert_eq!(fs::read(completion).unwrap(), original);
    assert_eq!(
        fs::read(sentinel).unwrap(),
        b"existing output must remain unchanged"
    );
    for (path, original) in artifacts.iter().zip(contents) {
        assert_eq!(
            fs::read(path).unwrap(),
            original,
            "changed {}",
            path.display()
        );
    }
    let rows = fixture.search("red", 0);
    assert!(rows
        .iter()
        .zip(QUERIES)
        .all(|(row, expected)| row.ids[0] == expected));
}

#[test]
fn open_rejects_mismatched_graph_headers_even_with_valid_checksums() {
    let fixture = Fixture::build();
    let original = fixture.graph_header();
    for invalid in [
        "rows",
        "dimensions",
        "version",
        "frozen",
        "associated",
        "degree",
        "block",
        "packing",
        "size",
        "entry",
    ] {
        let mut metadata = original.metadata().clone();
        let mut version = original.layout_version().clone();
        let mut block_size = original.block_size();
        match invalid {
            "rows" => metadata.num_pts -= 1,
            "dimensions" => metadata.dims += 1,
            "version" => version = GraphLayoutVersion::new(99, 0),
            "frozen" => metadata.vamana_frozen_num = 1,
            "associated" => metadata.associated_data_length = 4,
            "degree" => metadata.node_len = 4 * (metadata.dims as u64 + 1),
            "block" => block_size *= 2,
            "packing" => metadata.num_nodes_per_block += 1,
            "size" => metadata.disk_index_file_size += block_size,
            "entry" => metadata.medoid = metadata.num_pts,
            _ => unreachable!(),
        }
        fixture.replace_header(&GraphHeader::new(metadata, block_size, version));
        // Cache creation also has to wait until the header has been validated.
        for cache_nodes in [0, 32] {
            assert!(
                FilteredDiskIndex::open(&fixture.index, 1, usize::MAX, cache_nodes).is_err(),
                "accepted {invalid}"
            );
        }
    }
    fixture.replace_header(&original);
    assert!(FilteredDiskIndex::open(&fixture.index, 1, usize::MAX, 0).is_ok());
}

#[test]
fn unknown_labels_validate_query_parameters_before_returning_empty_results() {
    let fixture = Fixture::build();
    let index = FilteredDiskIndex::open(&fixture.index, 1, usize::MAX, 0).unwrap();
    for label in ["red", "unknown"] {
        assert!(index.search(&[0.0; 3], label, 1, 10, 0).is_err());
        assert!(index.search(&[0.0; 3], label, 0, 0, 1).is_err());
        assert!(index.search(&[0.0; 3], label, 11, 10, 1).is_err());
        assert!(index.search(&[0.0; 2], label, 1, 10, 1).is_err());
    }
    assert!(index
        .search(&[0.0; 3], "unknown", 1, 10, 1)
        .unwrap()
        .results
        .is_empty());
}
