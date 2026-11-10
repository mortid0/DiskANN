/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Clone an immutable uint8 generation with an explicitly substituted PQ pair.
//!
//! This artifact intervention preserves the graph and original build manifest.
//! Python 3's standard hashlib supplies streaming SHA-256 provenance without
//! changing production dependencies. Both source roots remain read-only.

use std::{
    collections::BTreeMap,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use anyhow::{ensure, Context, Result};
use diskann_disk::filtered::{
    metadata::LabelMetadata,
    persistence::{checksum, GenerationStore, Journal, LoadedGeneration},
    FilteredIndex,
};
use serde_json::{json, Value};

const PQ_FILES: [&str; 2] = ["index_pq_pivots.bin", "index_pq_compressed.bin"];
const HASHER: &str = r#"
import hashlib,json,sys
result={}
for name in sys.argv[1:]:
    h=hashlib.sha256()
    with open(name,'rb') as stream:
        while True:
            block=stream.read(8*1024*1024)
            if not block: break
            h.update(block)
    result[name]=h.hexdigest()
print(json.dumps(result))
"#;

struct Settings {
    root: PathBuf,
    pq_root: PathBuf,
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

fn manifest(loaded: &LoadedGeneration) -> Result<Value> {
    let value: Value = serde_json::from_slice(&loaded.manifest_payload)?;
    ensure!(value["version"] == 1 && value["encoding"] == "u8");
    ensure!(
        value["base_sequence"] == 0,
        "only immutable sequence-zero fixtures supported"
    );
    Ok(value)
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

fn bindings(
    source: &LoadedGeneration,
    donor: &LoadedGeneration,
    source_manifest: &Value,
    donor_manifest: &Value,
) -> Result<Value> {
    ensure!(source_manifest["dimensions"] == donor_manifest["dimensions"]);
    ensure!(source_manifest["pq_chunks"] == donor_manifest["pq_chunks"]);
    let source_labels: LabelMetadata =
        bincode::deserialize(&fs::read(source.directory.join("labels.bin"))?)?;
    let donor_labels: LabelMetadata =
        bincode::deserialize(&fs::read(donor.directory.join("labels.bin"))?)?;
    ensure!(source_labels.point_count() == donor_labels.point_count());
    ensure!(
        source_labels.postings() == donor_labels.postings(),
        "label postings differ"
    );
    ensure!(
        (0..source_labels.point_count() as u32)
            .all(|id| source_labels.stable_id(id) == Some(id)
                && donor_labels.stable_id(id) == Some(id)),
        "PQ replacement requires identity ordinal/stable-ID ordering"
    );
    for directory in [&source.directory, &donor.directory] {
        let mut file = fs::File::open(directory.join("vectors.bin"))?;
        let mut header = [0; 8];
        file.read_exact(&mut header)?;
        let rows = u32::from_le_bytes(header[..4].try_into()?) as usize;
        let dims = u32::from_le_bytes(header[4..].try_into()?) as usize;
        ensure!(rows == source_labels.point_count());
        ensure!(Some(dims as u64) == source_manifest["dimensions"].as_u64());
        ensure!(file.metadata()?.len() == 8 + rows as u64 * dims as u64);
    }
    Ok(
        json!({"rows": source_labels.point_count(), "dimensions": source_manifest["dimensions"],
              "postings_equal": true, "stable_ids": "identity 0..N in both generations",
              "source_entries_preserved": true}),
    )
}

fn clone_generation(settings: &Settings) -> Result<Value> {
    ensure!(
        settings.threads > 0 && !settings.out_root.exists(),
        "fresh output root and positive threads required"
    );
    let started = Instant::now();
    let source =
        GenerationStore::read_current(&settings.root)?.context("source CURRENT missing")?;
    let donor =
        GenerationStore::read_current(&settings.pq_root)?.context("PQ donor CURRENT missing")?;
    let source_manifest = manifest(&source)?;
    let donor_manifest = manifest(&donor)?;
    let binding = bindings(&source, &donor, &source_manifest, &donor_manifest)?;
    let source_wal = empty_journal(&settings.root)?;
    let donor_wal = empty_journal(&settings.pq_root)?;
    let source_current = fs::read(settings.root.join("CURRENT"))?;
    let donor_current = fs::read(settings.pq_root.join("CURRENT"))?;
    for name in ["vectors.bin", "labels.bin"].into_iter().chain(PQ_FILES) {
        ensure!(
            source
                .artifacts
                .iter()
                .any(|artifact| artifact.name == name),
            "missing source {name}"
        );
        ensure!(
            donor.artifacts.iter().any(|artifact| artifact.name == name),
            "missing donor {name}"
        );
    }
    let paths: Vec<_> = [&source, &donor]
        .into_iter()
        .flat_map(|loaded| {
            loaded
                .artifacts
                .iter()
                .map(|artifact| loaded.directory.join(&artifact.name))
                .chain(std::iter::once(loaded.directory.join("CATALOG")))
        })
        .chain([
            settings.root.join("CURRENT"),
            settings.root.join("updates.wal"),
            settings.pq_root.join("CURRENT"),
            settings.pq_root.join("updates.wal"),
        ])
        .collect();
    let input_hashes = hashes(&paths)?;
    ensure!(
        input_hashes[&source.directory.join("vectors.bin")]
            == input_hashes[&donor.directory.join("vectors.bin")],
        "complete vector-file SHA-256 differs; PQ row binding is unsafe"
    );
    let provenance = json!({
        "schema_version": 1, "operation": "replace complete PQ pair in a fresh immutable generation",
        "source_root": settings.root, "source_generation": source.generation,
        "pq_donor_root": settings.pq_root, "pq_donor_generation": donor.generation,
        "source_manifest_preserved_verbatim": true,
        "manifest_semantics": "original build configuration and pq_reused describe original construction; this explicit intervention changes only installed PQ files",
        "original_source_manifest": source_manifest, "original_donor_manifest": donor_manifest,
        "bindings": binding, "input_sha256": input_hashes, "replaced_files": PQ_FILES,
    });
    let mut inventory = Vec::new();
    for artifact in &source.artifacts {
        let origin = if PQ_FILES.contains(&artifact.name.as_str()) {
            &donor.directory
        } else {
            &source.directory
        };
        inventory.push((artifact.name.clone(), origin.join(&artifact.name)));
    }
    let store = GenerationStore::open(&settings.out_root)?;
    let generation = store.next_generation()?;
    let prepared = store.prepare_generation(generation, &source.manifest_payload, |directory| {
        for (name, input) in &inventory {
            let output = directory.join(name);
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(input, output)?;
        }
        let copied_paths: Vec<_> = inventory
            .iter()
            .map(|(name, _)| directory.join(name))
            .collect();
        let copied_hashes = hashes(&copied_paths).map_err(io::Error::other)?;
        for (name, input) in &inventory {
            if copied_hashes[&directory.join(name)] != input_hashes[input] {
                return Err(io::Error::other(format!(
                    "copied file SHA-256 differs: {name}"
                )));
            }
        }
        fs::write(
            directory.join("clone-provenance.json"),
            serde_json::to_vec_pretty(&provenance)?,
        )?;
        Ok(())
    })?;
    ensure!(
        fs::read(settings.root.join("CURRENT"))? == source_current
            && fs::read(settings.root.join("updates.wal"))? == source_wal,
        "source changed during clone"
    );
    ensure!(
        fs::read(settings.pq_root.join("CURRENT"))? == donor_current
            && fs::read(settings.pq_root.join("updates.wal"))? == donor_wal,
        "PQ donor changed during clone"
    );
    let max_frame = source_manifest["max_frame_bytes"]
        .as_u64()
        .context("missing max_frame_bytes")? as usize;
    let (journal, _) = Journal::<u8>::create_new(store.root().join("updates.wal"), 0, max_frame)?;
    let loaded = store.commit_prepared(prepared)?;
    let output_directory = loaded.directory.clone();
    let inventory_crc = loaded.artifacts.clone();
    ensure!(loaded.manifest_payload == source.manifest_payload);
    drop(journal);
    drop(store);
    let reopened = FilteredIndex::<u8>::open(&settings.out_root, settings.threads)?;
    ensure!(reopened.snapshot().sequence() == 0 && reopened.snapshot().base_sequence() == 0);
    drop(reopened);
    let output_paths: Vec<_> = loaded
        .artifacts
        .iter()
        .map(|artifact| output_directory.join(&artifact.name))
        .chain([
            output_directory.join("CATALOG"),
            settings.out_root.join("CURRENT"),
            settings.out_root.join("updates.wal"),
        ])
        .collect();
    let output_hashes = hashes(&output_paths)?;
    let current = fs::read(settings.out_root.join("CURRENT"))?;
    ensure!(current.len() == 56, "unexpected CURRENT envelope");
    let report = json!({"schema_version": 1, "status": "published and reopened successfully",
        "source_root": settings.root, "pq_donor_root": settings.pq_root, "output_root": settings.out_root,
        "generation": loaded.generation, "generation_directory": output_directory,
        "bindings": provenance["bindings"], "source_manifest_preserved_verbatim": true,
        "input_sha256": input_hashes, "output_sha256": output_hashes,
        "generation_artifact_crc64_ecma182": inventory_crc,
        "published_catalog_payload_crc64_ecma182": u64::from_le_bytes(current[48..56].try_into()?),
        "validated_sequence": 0, "elapsed_seconds": started.elapsed().as_secs_f64()});
    fs::write(
        settings.out_root.join("clone-result.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
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
    let settings = Settings {
        root: path("root")?,
        pq_root: path("pq-root")?,
        out_root: path("out-root")?,
        threads: args.get("threads").map_or(Ok(8), |value| value.parse())?,
    };
    println!("{}", clone_generation(&settings)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_disk::filtered::FilteredIndexConfig;
    use diskann_utils::views::rowmajor;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fresh_clone_preserves_graph_manifest_labels_and_installs_exact_donor_pair() {
        let temporary = tempfile::tempdir().unwrap();
        let settings = Settings {
            root: temporary.path().join("source"),
            pq_root: temporary.path().join("donor"),
            out_root: temporary.path().join("clone"),
            threads: 2,
        };
        let values: Vec<u8> = (0..256)
            .flat_map(|id| [id as u8, (id % 13) as u8])
            .collect();
        let vectors = rowmajor::Ref::try_from_data(&values, 256, 2).unwrap();
        let postings = BTreeMap::from([(7, (0..256).collect())]);
        for (root, seed) in [(&settings.root, 0), (&settings.pq_root, 19)] {
            let mut config = FilteredIndexConfig {
                threads: 2,
                pq_chunks: 2,
                ..Default::default()
            };
            config.build.label_degree = 8;
            config.build.l_build = 16;
            config.build.concurrency = 2;
            config.build.seed = seed;
            let index = FilteredIndex::<u8>::build(root, vectors, postings.clone(), config)
                .await
                .unwrap();
            drop(index);
        }
        let report = clone_generation(&settings).unwrap();
        assert_eq!(report["validated_sequence"], 0);
        let source = GenerationStore::read_current(&settings.root)
            .unwrap()
            .unwrap();
        let donor = GenerationStore::read_current(&settings.pq_root)
            .unwrap()
            .unwrap();
        let clone = GenerationStore::read_current(&settings.out_root)
            .unwrap()
            .unwrap();
        assert_eq!(clone.manifest_payload, source.manifest_payload);
        for name in ["index_disk.index", "vectors.bin", "labels.bin"] {
            assert_eq!(
                fs::read(clone.directory.join(name)).unwrap(),
                fs::read(source.directory.join(name)).unwrap()
            );
        }
        for name in PQ_FILES {
            assert_eq!(
                fs::read(clone.directory.join(name)).unwrap(),
                fs::read(donor.directory.join(name)).unwrap()
            );
        }
        assert!(clone
            .artifacts
            .iter()
            .any(|artifact| artifact.name == "clone-provenance.json"));
        assert!(clone_generation(&settings).is_err());
    }

    #[test]
    fn empty_journal_rejects_a_nonzero_checkpoint_or_extra_bytes() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("updates.wal");
        let (journal, _) = Journal::<u8>::create_new(&path, 1, 4096).unwrap();
        drop(journal);
        assert!(empty_journal(temporary.path()).is_err());
        fs::write(path, [0u8; 33]).unwrap();
        assert!(empty_journal(temporary.path()).is_err());
    }
}
