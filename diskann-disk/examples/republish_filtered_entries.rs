/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Install corpus-derived entry pools in a fresh immutable filtered generation.
//!
//! The graph, vectors, PQ pair and original construction manifest are copied
//! byte-for-byte. Only label entry metadata changes. Python 3's standard hashlib
//! provides streaming SHA-256 provenance without production dependency changes.

use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use anyhow::{ensure, Context, Result};
use diskann_disk::{
    build::filtered::select_label_entries,
    filtered::{
        metadata::LabelMetadata,
        persistence::{checksum, GenerationStore, Journal},
        FilteredIndex,
    },
};
use diskann_providers::utils::{create_thread_pool, ParallelIteratorInPool};
use diskann_utils::views::rowmajor;
use rayon::prelude::*;
use serde_json::{json, Value};

const PROVENANCE: &str = "entry-pool-provenance.json";
const HASHER: &str = r#"
import hashlib,json,sys
result={}
for name in sys.argv[1:]:
    digest=hashlib.sha256()
    with open(name,'rb') as stream:
        while True:
            block=stream.read(8*1024*1024)
            if not block: break
            digest.update(block)
    result[name]=digest.hexdigest()
print(json.dumps(result))
"#;

struct Settings {
    root: PathBuf,
    out_root: PathBuf,
    threads: usize,
}

fn hashes(paths: &[PathBuf]) -> Result<BTreeMap<PathBuf, String>> {
    let output = Command::new("python3")
        .args(["-c", HASHER])
        .args(paths)
        .output()
        .context("running Python 3 streaming SHA-256")?;
    ensure!(
        output.status.success(),
        "hashing failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn empty_journal(root: &Path) -> Result<Vec<u8>> {
    let bytes = fs::read(root.join("updates.wal"))?;
    ensure!(
        bytes.len() == 32 && &bytes[..8] == b"DFWAL001",
        "source WAL is not empty"
    );
    ensure!(u32::from_le_bytes(bytes[8..12].try_into()?) == 1);
    ensure!(
        bytes[12..24].iter().all(|value| *value == 0),
        "source WAL is not sequence zero"
    );
    ensure!(checksum(&bytes[..24]) == u64::from_le_bytes(bytes[24..].try_into()?));
    Ok(bytes)
}

fn require_identity_ids(metadata: &LabelMetadata) -> Result<()> {
    ensure!(
        (0..metadata.point_count() as u32).all(|id| metadata.stable_id(id) == Some(id)),
        "this experiment requires identity ordinal/stable IDs"
    );
    Ok(())
}

fn new_metadata(directory: &Path, manifest: &Value, threads: usize) -> Result<(Vec<u8>, Value)> {
    let original: LabelMetadata = bincode::deserialize(&fs::read(directory.join("labels.bin"))?)?;
    require_identity_ids(&original)?;
    let matrix = fs::read(directory.join("vectors.bin"))?;
    ensure!(matrix.len() >= 8, "short vector matrix");
    let rows = u32::from_le_bytes(matrix[..4].try_into()?) as usize;
    let dimensions = u32::from_le_bytes(matrix[4..8].try_into()?) as usize;
    ensure!(rows > 0 && rows == original.point_count() && dimensions > 0);
    ensure!(Some(dimensions as u64) == manifest["dimensions"].as_u64());
    ensure!(
        matrix.len()
            == rows
                .checked_mul(dimensions)
                .context("vector size overflow")?
                + 8
    );
    let vectors = rowmajor::Ref::try_from_data(&matrix[8..], rows, dimensions)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let pool = create_thread_pool(threads)?;
    let selected: Result<Vec<_>> = original
        .postings()
        .par_iter()
        .map(|(&label, members)| {
            let mut entries = select_label_entries(vectors, [(label, members.as_slice())])?;
            let entries = entries.remove(&label).context("selector omitted a label")?;
            ensure!(
                !entries.is_empty() && entries.len() <= 5,
                "invalid entry pool size"
            );
            ensure!(
                entries.first() == original.entries(label).first(),
                "global centroid changed for label {label}"
            );
            Ok((label, entries))
        })
        .collect_in_pool(pool.as_ref());
    let entries: BTreeMap<_, _> = selected?.into_iter().collect();
    let mut replacement = LabelMetadata::from_postings(rows, original.postings().clone(), entries)?;
    if let Some(ids) = original.stable_ids() {
        replacement = replacement.with_stable_ids(ids.to_vec())?;
    }
    ensure!(
        replacement.postings() == original.postings()
            && replacement.stable_ids() == original.stable_ids(),
        "metadata binding changed"
    );
    ensure!(
        (0..rows as u32).all(|id| replacement.labels(id) == original.labels(id)),
        "point-to-label membership changed"
    );
    let summary = json!({"rows": rows, "dimensions": dimensions,
        "point_count_preserved": true, "point_labels_preserved": true,
        "postings_preserved": true, "stable_ids_preserved": true, "source_centroids_preserved": true,
        "old_entry_points": original.entry_points(), "new_entry_points": replacement.entry_points(),
        "selector": "diskann_disk::build::filtered::select_label_entries",
        "policy": {"spatial_clusters": 4, "maximum_entries": 5, "maximum_fit_members": 4096,
            "lloyd_iterations": 8, "sample_seed_base": 0x6c61_6265_6c5f_7633u64,
            "sample_seed_mixing": "XOR u64(label)",
            "global_centroid_first": true,
            "medoids": "actual full-posting members selected after fitting deterministic corpus sample"},
        "corpus_only": true, "query_results_used": false,
        "parallelism": "independent labels in an explicit bounded Rayon pool"});
    Ok((bincode::serialize(&replacement)?, summary))
}

fn republish(settings: &Settings) -> Result<Value> {
    ensure!(
        settings.threads > 0 && !settings.out_root.exists(),
        "fresh output root and positive threads required"
    );
    let started = Instant::now();
    let source_wal = empty_journal(&settings.root)?;
    let source_current = fs::read(settings.root.join("CURRENT"))?;
    let source =
        GenerationStore::read_current(&settings.root)?.context("source CURRENT missing")?;
    let manifest: Value = serde_json::from_slice(&source.manifest_payload)?;
    ensure!(
        manifest["version"] == 1 && manifest["encoding"] == "u8" && manifest["base_sequence"] == 0,
        "requires immutable sequence-zero uint8 generation"
    );
    ensure!(
        !source
            .artifacts
            .iter()
            .any(|artifact| artifact.name == PROVENANCE),
        "source already contains an entry-pool intervention"
    );
    let input_paths: Vec<_> = source
        .artifacts
        .iter()
        .map(|artifact| source.directory.join(&artifact.name))
        .chain([
            source.directory.join("CATALOG"),
            settings.root.join("CURRENT"),
            settings.root.join("updates.wal"),
        ])
        .collect();
    let source_hashes = hashes(&input_paths)?;
    let (labels, selection) = new_metadata(&source.directory, &manifest, settings.threads)?;
    let provenance = json!({"schema_version": 1, "operation": "republish corpus-derived label entry pools",
        "input_root": settings.root, "source_root": settings.root, "output_root": settings.out_root,
        "source_generation": source.generation,
        "original_construction_manifest": manifest, "manifest_payload_preserved_verbatim": true,
        "source_files_sha256": source_hashes, "entry_selection": selection,
        "only_replaced_artifact": "labels.bin", "added_artifact": PROVENANCE,
        "graph_vectors_PQ_preserved_exactly": true, "query_workload_not_read": true});
    let store = GenerationStore::open(&settings.out_root)?;
    let prepared = store.prepare_generation(
        store.next_generation()?,
        &source.manifest_payload,
        |directory| {
            for artifact in &source.artifacts {
                if artifact.name == "labels.bin" {
                    continue;
                }
                let output = directory.join(&artifact.name);
                if let Some(parent) = output.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::copy(source.directory.join(&artifact.name), output)?;
            }
            fs::write(directory.join("labels.bin"), &labels)?;
            let paths: Vec<_> = source
                .artifacts
                .iter()
                .filter(|artifact| artifact.name != "labels.bin")
                .map(|artifact| directory.join(&artifact.name))
                .collect();
            let copied = hashes(&paths).map_err(io::Error::other)?;
            for artifact in source
                .artifacts
                .iter()
                .filter(|artifact| artifact.name != "labels.bin")
            {
                if copied[&directory.join(&artifact.name)]
                    != source_hashes[&source.directory.join(&artifact.name)]
                {
                    return Err(io::Error::other(format!(
                        "copied SHA-256 differs: {}",
                        artifact.name
                    )));
                }
            }
            fs::write(
                directory.join(PROVENANCE),
                serde_json::to_vec_pretty(&provenance)?,
            )?;
            Ok(())
        },
    )?;
    ensure!(
        fs::read(settings.root.join("CURRENT"))? == source_current
            && empty_journal(&settings.root)? == source_wal,
        "source changed during selection/publication"
    );
    let max_frame = manifest["max_frame_bytes"]
        .as_u64()
        .context("missing max_frame_bytes")? as usize;
    let (journal, _) = Journal::<u8>::create_new(store.root().join("updates.wal"), 0, max_frame)?;
    let loaded = store.commit_prepared(prepared)?;
    ensure!(loaded.manifest_payload == source.manifest_payload);
    drop(journal);
    drop(store);
    let reopened = FilteredIndex::<u8>::open(&settings.out_root, settings.threads)?;
    ensure!(reopened.snapshot().sequence() == 0 && reopened.snapshot().base_sequence() == 0);
    drop(reopened);
    let output_paths: Vec<_> = loaded
        .artifacts
        .iter()
        .map(|artifact| loaded.directory.join(&artifact.name))
        .chain([
            loaded.directory.join("CATALOG"),
            settings.out_root.join("CURRENT"),
            settings.out_root.join("updates.wal"),
        ])
        .collect();
    let output_hashes = hashes(&output_paths)?;
    let current = fs::read(settings.out_root.join("CURRENT"))?;
    ensure!(current.len() == 56, "unexpected CURRENT envelope");
    let provenance_artifact = loaded
        .artifacts
        .iter()
        .find(|artifact| artifact.name == PROVENANCE)
        .context("entry provenance omitted from published catalog")?;
    let result = json!({"schema_version": 1, "status": "published and reopened successfully",
        "input_root": settings.root, "source_root": settings.root, "output_root": settings.out_root,
        "generation": loaded.generation, "generation_directory": loaded.directory,
        "source_files_sha256": source_hashes, "output_files_sha256": output_hashes,
        "generation_artifact_crc64_ecma182": loaded.artifacts,
        "installed_files": loaded.artifacts,
        "entry_provenance_artifact": provenance_artifact,
        "source_manifest_before": manifest,
        "source_manifest_after": serde_json::from_slice::<Value>(&loaded.manifest_payload)?,
        "manifest_payload_crc64_ecma182": {"source": checksum(&source.manifest_payload), "installed": checksum(&loaded.manifest_payload)},
        "entry_points": provenance["entry_selection"]["new_entry_points"],
        "entry_policy": provenance["entry_selection"]["policy"],
        "verified_bindings": {"point_count_equal": true, "point_labels_equal": true,
            "postings_equal": true, "stable_ids_equal": true, "global_centroid_first": true},
        "published_catalog_payload_crc64_ecma182": u64::from_le_bytes(current[48..56].try_into()?),
        "entry_selection": provenance["entry_selection"], "manifest_payload_preserved_verbatim": true,
        "validated_sequence": 0, "elapsed_seconds": started.elapsed().as_secs_f64()});
    fs::write(
        settings.out_root.join("entry-pool-result.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(result)
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
    let path = |name| {
        args.get(name)
            .map(PathBuf::from)
            .with_context(|| format!("--{name} required"))
    };
    println!(
        "{}",
        republish(&Settings {
            root: path("root")?,
            out_root: path("out-root")?,
            threads: args.get("threads").map_or(Ok(8), |value| value.parse())?
        })?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_disk::filtered::{updates::Mutation, FilteredIndexConfig};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn authentic_republication_preserves_graph_vectors_pq_manifest_and_validates_entries() {
        let temporary = tempfile::tempdir().unwrap();
        let settings = Settings {
            root: temporary.path().join("original"),
            out_root: temporary.path().join("spatial-pool"),
            threads: 2,
        };
        let values: Vec<u8> = (0..256)
            .flat_map(|id| [id as u8, (id % 13) as u8])
            .collect();
        let vectors = rowmajor::Ref::try_from_data(&values, 256, 2).unwrap();
        let postings =
            BTreeMap::from([(7, (0..256).collect()), (99, (0..256).step_by(2).collect())]);
        let mut config = FilteredIndexConfig {
            threads: 2,
            pq_chunks: 2,
            ..Default::default()
        };
        config.build.label_degree = 8;
        config.build.l_build = 16;
        config.build.concurrency = 2;
        let index = FilteredIndex::<u8>::build(&settings.root, vectors, postings.clone(), config)
            .await
            .unwrap();
        drop(index);
        // Make an authentic prior-style generation with one centroid entry per
        // label and an explicit identity stable-ID map. Its checksums/publication
        // come from the same public store API as an ordinary compaction.
        let initial = GenerationStore::read_current(&settings.root)
            .unwrap()
            .unwrap();
        let initial_labels: LabelMetadata =
            bincode::deserialize(&fs::read(initial.directory.join("labels.bin")).unwrap()).unwrap();
        let single_entries = initial_labels
            .entry_points()
            .iter()
            .map(|(&label, entries)| (label, vec![entries[0]]))
            .collect();
        let single_labels = LabelMetadata::from_postings_with_ids(
            256,
            postings.clone(),
            single_entries,
            (0..256).collect(),
        )
        .unwrap();
        let store = GenerationStore::open(&settings.root).unwrap();
        store
            .publish(2, &initial.manifest_payload, |directory| {
                for artifact in &initial.artifacts {
                    if artifact.name != "labels.bin" {
                        fs::copy(
                            initial.directory.join(&artifact.name),
                            directory.join(&artifact.name),
                        )?;
                    }
                }
                fs::write(
                    directory.join("labels.bin"),
                    bincode::serialize(&single_labels).unwrap(),
                )
            })
            .unwrap();
        drop(store);
        let before = GenerationStore::read_current(&settings.root)
            .unwrap()
            .unwrap();
        let source_current = fs::read(settings.root.join("CURRENT")).unwrap();
        let source_catalog = fs::read(before.directory.join("CATALOG")).unwrap();
        let report = republish(&settings).unwrap();
        assert_eq!(report["validated_sequence"], 0);
        let after = GenerationStore::read_current(&settings.out_root)
            .unwrap()
            .unwrap();
        assert_eq!(after.manifest_payload, before.manifest_payload);
        for name in [
            "index_disk.index",
            "index_pq_pivots.bin",
            "index_pq_compressed.bin",
            "vectors.bin",
        ] {
            assert_eq!(
                fs::read(before.directory.join(name)).unwrap(),
                fs::read(after.directory.join(name)).unwrap()
            );
        }
        assert_eq!(
            fs::read(settings.root.join("CURRENT")).unwrap(),
            source_current
        );
        assert_eq!(
            fs::read(before.directory.join("CATALOG")).unwrap(),
            source_catalog
        );
        let metadata: LabelMetadata =
            bincode::deserialize(&fs::read(after.directory.join("labels.bin")).unwrap()).unwrap();
        let expected = select_label_entries(
            vectors,
            postings
                .iter()
                .map(|(&label, members)| (label, members.as_slice())),
        )
        .unwrap();
        assert_eq!(metadata.entry_points(), &expected);
        assert_eq!(metadata.postings(), &postings);
        assert_eq!(metadata.stable_ids(), single_labels.stable_ids());
        assert_ne!(
            fs::read(before.directory.join("labels.bin")).unwrap(),
            fs::read(after.directory.join("labels.bin")).unwrap()
        );
        for (&label, entries) in metadata.entry_points() {
            assert!(entries.len() > 1 && entries.len() <= 5);
            assert!(entries.iter().all(|&entry| metadata.contains(label, entry)));
        }
        assert!(after
            .artifacts
            .iter()
            .any(|artifact| artifact.name == PROVENANCE));
        assert!(republish(&settings).is_err());
    }

    #[test]
    fn nonempty_journal_and_nonidentity_stable_ids_are_rejected() {
        let temporary = tempfile::tempdir().unwrap();
        let (mut journal, _) =
            Journal::<u8>::create_new(temporary.path().join("updates.wal"), 0, 4096).unwrap();
        journal.append(1, &[Mutation::Delete { id: 0 }]).unwrap();
        drop(journal);
        assert!(empty_journal(temporary.path()).is_err());
        let metadata = LabelMetadata::from_postings_with_ids(
            2,
            BTreeMap::from([(7, vec![0, 1])]),
            BTreeMap::from([(7, vec![0])]),
            vec![5, 9],
        )
        .unwrap();
        assert!(require_identity_ids(&metadata).is_err());
        let identity = metadata.with_stable_ids(vec![0, 1]).unwrap();
        assert!(require_identity_ids(&identity).is_ok());
    }
}
