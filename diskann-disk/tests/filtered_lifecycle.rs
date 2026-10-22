/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![cfg(all(unix, not(miri)))]

use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use diskann::ANNError;
use diskann_disk::{
    build::filtered::StitchedBuildConfig,
    filtered::{
        metadata::LabelMetadata,
        persistence::{GenerationStore, Journal},
        updates::{DeltaLimits, Mutation},
        FilteredIndex, FilteredIndexConfig, FilteredSnapshot,
    },
};
use diskann_utils::views::rowmajor;
use tempfile::tempdir;

const ROWS: usize = 300; // Exercises native PQ and the SSD graph, not the <256 fallback.
const DIMS: usize = 8;

fn config() -> FilteredIndexConfig {
    FilteredIndexConfig {
        build: StitchedBuildConfig {
            label_degree: 8,
            max_degree: 32,
            l_build: 32,
            alpha: 1.2,
            concurrency: 2,
            seed: 777,
        },
        pq_chunks: 4,
        threads: 2,
        ..Default::default()
    }
}

fn values() -> Vec<u8> {
    (0..ROWS)
        .flat_map(|id| {
            (0..DIMS).map(move |column| match column {
                0 => id as u8,
                1 => (id >> 8) as u8,
                _ => ((id * 37 + column * 53 + (id / 251) * (column + 1) * 17) % 256) as u8,
            })
        })
        .collect()
}

async fn build(path: &Path, config: FilteredIndexConfig) -> FilteredIndex<u8> {
    let values = values();
    let vectors = rowmajor::Ref::try_from_data(&values, ROWS, DIMS).unwrap();
    let postings = BTreeMap::from([
        (7, (0..ROWS as u32).filter(|id| id % 2 == 0).collect()),
        (9, (0..ROWS as u32).filter(|id| id % 3 == 0).collect()),
        (11, (0..ROWS as u32).filter(|id| id % 2 == 1).collect()),
    ]);
    FilteredIndex::build(path, vectors, postings, config)
        .await
        .unwrap()
}

fn verify_results(snapshot: &FilteredSnapshot<u8>, label: u32) {
    let result = snapshot.search(&[0; DIMS], label, 5, 200).unwrap();
    assert_eq!(result.sequence, snapshot.sequence());
    assert_eq!(result.hits.len(), 5);
    let mut ids: Vec<_> = result.hits.iter().map(|hit| hit.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), result.hits.len());
    for hit in result.hits {
        let (vector, labels) = snapshot.get(hit.id).unwrap().unwrap();
        assert!(labels.contains(&label));
        let expected: f32 = vector.iter().map(|&value| (value as f32).powi(2)).sum();
        assert_eq!(hit.distance, expected);
    }
}

#[tokio::test]
async fn matching_far_delta_does_not_hide_live_base_neighbors_after_mass_deletion() {
    let directory = tempdir().unwrap();
    let index = build(directory.path(), config()).await;
    let original = index.snapshot();
    let mut nearest = Vec::new();
    for id in 0..ROWS as u32 {
        let (vector, labels) = original.get(id).unwrap().unwrap();
        if labels.contains(&7) {
            let distance: u64 = vector.iter().map(|&value| u64::from(value).pow(2)).sum();
            nearest.push((distance, id));
        }
    }
    nearest.sort_unstable();
    assert_eq!(nearest.len(), ROWS / 2);
    let deleted: Vec<_> = nearest.iter().take(60).map(|&(_, id)| id).collect();
    let initial_candidates = original.search(&[0; DIMS], 7, 10, 10).unwrap();
    assert!(initial_candidates.base_searches > 0 && !initial_candidates.exact_fallback);
    assert!(
        initial_candidates
            .hits
            .iter()
            .all(|hit| deleted.contains(&hit.id)),
        "the initial low-breadth base pool must consist entirely of records being deleted"
    );
    index
        .apply(
            &deleted
                .iter()
                .map(|&id| Mutation::Delete { id })
                .collect::<Vec<_>>(),
        )
        .unwrap();
    for _ in 0..5 {
        index.insert(vec![255; DIMS], vec![7]).unwrap();
    }
    let current = index.snapshot();
    let mut live_truth = Vec::new();
    for id in 0..(ROWS + 5) as u32 {
        if let Some((vector, labels)) = current.get(id).unwrap() {
            if labels.contains(&7) {
                let distance: u64 = vector.iter().map(|&value| u64::from(value).pow(2)).sum();
                live_truth.push((distance, id));
            }
        }
    }
    live_truth.sort_unstable();
    live_truth.truncate(5);
    assert!(
        live_truth.iter().all(|&(_, id)| id < ROWS as u32),
        "far delta points must be worse than the remaining live base neighbors"
    );
    let found = current.search(&[0; DIMS], 7, 5, 10).unwrap();
    assert_eq!(found.delta_candidates, 5);
    let actual: Vec<_> = found
        .hits
        .iter()
        .map(|hit| (hit.distance as u64, hit.id))
        .collect();
    assert_eq!(
        actual, live_truth,
        "matching delta hits cannot stop base retries while the base result pool is shadowed"
    );
    let before_relabel_hits = found.hits.clone();

    // Relabel the next closest base records without changing their vectors. They
    // remain routing bridges for label 7, but belong only to label 11 in new results.
    let relabeled: Vec<_> = nearest
        .iter()
        .skip(60)
        .take(30)
        .map(|&(_, id)| id)
        .collect();
    let mutations: Vec<_> = relabeled
        .iter()
        .map(|&id| Mutation::Upsert {
            id,
            vector: current.get(id).unwrap().unwrap().0,
            labels: vec![11],
        })
        .collect();
    index.apply(&mutations).unwrap();
    let after_relabel = index.snapshot();
    let mut relabeled_truth = Vec::new();
    for id in 0..(ROWS + 5) as u32 {
        if let Some((vector, labels)) = after_relabel.get(id).unwrap() {
            if labels.contains(&7) {
                let distance: u64 = vector.iter().map(|&value| u64::from(value).pow(2)).sum();
                relabeled_truth.push((distance, id));
            }
        }
    }
    relabeled_truth.sort_unstable();
    relabeled_truth.truncate(5);
    let found = after_relabel.search(&[0; DIMS], 7, 5, 10).unwrap();
    assert_eq!(found.delta_candidates, 5);
    assert_eq!(
        found
            .hits
            .iter()
            .map(|hit| (hit.distance as u64, hit.id))
            .collect::<Vec<_>>(),
        relabeled_truth
    );
    assert!(current.get(relabeled[0]).unwrap().unwrap().1.contains(&7));
    assert_eq!(after_relabel.get(relabeled[0]).unwrap().unwrap().1, [11]);
    assert_eq!(
        current.search(&[0; DIMS], 7, 5, 10).unwrap().hits,
        before_relabel_hits
    );
}

#[tokio::test]
async fn native_build_mutation_batches_reload_and_stable_ids() {
    let directory = tempdir().unwrap();
    let index = build(directory.path(), config()).await;
    let original = index.snapshot();
    let first_vector = values()[..DIMS].to_vec();
    let original_query = original.search(&first_vector, 7, 5, 200).unwrap();
    assert!(
        original_query.base_searches > 0,
        "must exercise SSD graph search"
    );
    assert!(!original_query.exact_fallback);
    assert_eq!(original_query.hits[0].id, 0);
    let asynchronous = original
        .search_async(first_vector.clone(), 7, 5, 200)
        .await
        .unwrap();
    assert_eq!(asynchronous.hits, original_query.hits);
    let generation = GenerationStore::read_current(directory.path())
        .unwrap()
        .unwrap();
    assert!(generation
        .artifacts
        .iter()
        .any(|file| file.name.ends_with("_disk.index")));

    let metadata: LabelMetadata =
        bincode::deserialize(&fs::read(generation.directory.join("labels.bin")).unwrap()).unwrap();
    let entry = metadata.entries(7)[0];
    let mut batch = vec![
        Mutation::Upsert {
            id: 0,
            vector: vec![0; DIMS],
            labels: vec![11, 11],
        },
        Mutation::Delete { id: 2 },
    ];
    if entry != 0 && entry != 2 {
        batch.push(Mutation::Delete { id: entry });
    }
    assert_eq!(index.apply(&batch).unwrap(), 1);
    let after_batch = index.snapshot();
    assert_eq!(
        after_batch.get(0).unwrap().unwrap(),
        (vec![0; DIMS], vec![11])
    );
    assert!(after_batch.get(2).unwrap().is_none());
    let after_entry_change = after_batch.search(&first_vector, 7, 5, 200).unwrap();
    assert!(after_entry_change.base_searches > 0);
    assert!(after_entry_change.hits.iter().all(|hit| hit.id != entry));
    assert_eq!(original.sequence(), 0);
    assert_eq!(original.get(0).unwrap().unwrap().0, first_vector);
    assert!(original.get(2).unwrap().is_some());
    assert_eq!(
        after_batch.search(&[0; DIMS], 11, 1, 200).unwrap().hits[0].id,
        0
    );

    let before_failure = index.snapshot();
    assert!(index
        .apply(&[
            Mutation::Delete { id: 5 },
            Mutation::Upsert {
                id: 6,
                vector: vec![1; DIMS - 1],
                labels: vec![7]
            },
        ])
        .is_err());
    assert_eq!(index.snapshot().sequence(), before_failure.sequence());
    assert!(index.snapshot().get(5).unwrap().is_some());
    assert!(index
        .apply(&[Mutation::Delete { id: 6 }, Mutation::Delete { id: 6 }])
        .is_err());
    assert_eq!(index.snapshot().sequence(), before_failure.sequence());

    index.replace_labels(3, vec![7]).unwrap();
    index.replace_vector(3, vec![1; DIMS]).unwrap();
    let (inserted, sequence) = index.insert(vec![0; DIMS], vec![7]).unwrap();
    assert_eq!(inserted, ROWS as u32);
    let current = index.snapshot();
    assert_eq!(current.sequence(), sequence);
    assert_eq!(
        current.search(&[0; DIMS], 7, 1, 200).unwrap().hits[0].id,
        inserted
    );
    verify_results(&current, 7);
    verify_results(&current, 11);
    assert!(current
        .search(&[0; DIMS], u32::MAX, 5, 200)
        .unwrap()
        .hits
        .is_empty());
    assert!(current.search(&[0; DIMS - 1], 7, 5, 200).is_err());
    drop((original, after_batch, before_failure, current, index));

    let reopened = FilteredIndex::<u8>::open(directory.path(), 2).unwrap();
    assert_eq!(reopened.snapshot().sequence(), sequence);
    assert_eq!(
        reopened.snapshot().get(3).unwrap().unwrap(),
        (vec![1; DIMS], vec![7])
    );
    assert!(reopened.snapshot().get(2).unwrap().is_none());
    assert_eq!(
        reopened.insert(vec![3; DIMS], vec![7]).unwrap().0,
        inserted + 1
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repeated_concurrent_compaction_preserves_updates_and_snapshot_leases() {
    let directory = tempdir().unwrap();
    let index = Arc::new(build(directory.path(), config()).await);
    let original = index.snapshot();
    let original_zero = original.get(0).unwrap();
    let initial_generation = GenerationStore::read_current(directory.path())
        .unwrap()
        .unwrap();
    let codebook_bytes =
        fs::read(initial_generation.directory.join("index_pq_pivots.bin")).unwrap();
    for round in 0..3 {
        let active = Arc::new(AtomicBool::new(true));
        let updater_index = index.clone();
        let updater_active = active.clone();
        let updater = std::thread::spawn(move || {
            let mut count = 0;
            // Replacing one record bounds the current delta while the durable log
            // still records every acknowledged sequence during the background build.
            while updater_active.load(Ordering::Acquire) || count < 4 {
                updater_index
                    .apply(&[Mutation::Upsert {
                        id: 0,
                        vector: vec![(round * 20 + count % 20) as u8; DIMS],
                        labels: vec![7, 19],
                    }])
                    .unwrap();
                count += 1;
                std::thread::yield_now();
            }
            count
        });
        let compacted = index.compact().await;
        active.store(false, Ordering::Release);
        assert!(updater.join().unwrap() >= 4);
        let generation = compacted.unwrap();
        let current = index.snapshot();
        assert!(current.get(0).unwrap().unwrap().1.contains(&19));
        assert_eq!(original.get(0).unwrap(), original_zero);
        assert_eq!(original.sequence(), 0);
        verify_results(&current, 7);
        let loaded = GenerationStore::read_current(directory.path())
            .unwrap()
            .unwrap();
        assert_eq!(loaded.generation, generation);
        assert_eq!(
            fs::read(loaded.directory.join("index_pq_pivots.bin")).unwrap(),
            codebook_bytes
        );
        assert!(index.collect_garbage().unwrap() <= round);
        assert!(directory
            .path()
            .join("generation-00000000000000000001")
            .exists());
    }
    let final_snapshot = index.snapshot();
    let sequence = final_snapshot.sequence();
    let expected = final_snapshot.get(0).unwrap();
    drop(original);
    assert!(index.collect_garbage().unwrap() >= 1);
    assert!(!directory
        .path()
        .join("generation-00000000000000000001")
        .exists());
    drop(index);
    assert!(FilteredIndex::<u8>::open(directory.path(), 2).is_err());
    verify_results(&final_snapshot, 7);
    drop(final_snapshot);
    let reopened = FilteredIndex::<u8>::open(directory.path(), 2).unwrap();
    assert_eq!(reopened.snapshot().sequence(), sequence);
    assert_eq!(reopened.snapshot().get(0).unwrap(), expected);
    verify_results(&reopened.snapshot(), 7);
    let loaded = GenerationStore::read_current(directory.path())
        .unwrap()
        .unwrap();
    assert_eq!(
        fs::read(loaded.directory.join("index_pq_pivots.bin")).unwrap(),
        codebook_bytes
    );

    // A trained table must survive a generation with no graph or vectors, so a
    // future populated generation need not silently train a different codebook.
    reopened
        .apply(
            &(0..ROWS as u32)
                .map(|id| Mutation::Delete { id })
                .collect::<Vec<_>>(),
        )
        .unwrap();
    reopened.compact().await.unwrap();
    let empty = GenerationStore::read_current(directory.path())
        .unwrap()
        .unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(&empty.manifest_payload).unwrap();
    assert_eq!(manifest["has_graph"], false);
    assert_eq!(manifest["pq_reused"], true);
    assert_eq!(
        fs::read(empty.directory.join("index_pq_pivots.bin")).unwrap(),
        codebook_bytes
    );
    assert!(reopened
        .snapshot()
        .search(&[0; DIMS], 7, 5, 10)
        .unwrap()
        .hits
        .is_empty());
    drop(reopened);
    let reopened = FilteredIndex::<u8>::open(directory.path(), 2).unwrap();
    assert_eq!(
        reopened.insert(vec![0; DIMS], vec![7]).unwrap().0,
        ROWS as u32
    );
    reopened.compact().await.unwrap();
    let singleton = GenerationStore::read_current(directory.path())
        .unwrap()
        .unwrap();
    assert_eq!(
        fs::read(singleton.directory.join("index_pq_pivots.bin")).unwrap(),
        codebook_bytes
    );
    drop(reopened);
    let reopened = FilteredIndex::<u8>::open(directory.path(), 2).unwrap();
    assert_eq!(
        reopened.snapshot().get(ROWS as u32).unwrap().unwrap(),
        (vec![0; DIMS], vec![7])
    );
}

#[tokio::test]
async fn rejected_compaction_validation_preserves_current_and_collects_its_orphan() {
    let directory = tempdir().unwrap();
    let index = build(directory.path(), config()).await;
    index
        .apply(&[
            Mutation::Upsert {
                id: 0,
                vector: vec![0; DIMS],
                labels: vec![19],
            },
            Mutation::Delete { id: 2 },
        ])
        .unwrap();
    let inserted = index.insert(vec![1; DIMS], vec![19]).unwrap().0;
    let before = index.snapshot();
    let before_generation = GenerationStore::read_current(directory.path())
        .unwrap()
        .unwrap()
        .generation;
    let current_bytes = fs::read(directory.path().join("CURRENT")).unwrap();
    let journal_bytes = fs::read(directory.path().join("updates.wal")).unwrap();
    let callback_ran = AtomicBool::new(false);
    let mut retained_candidate = None;
    let error = index
        .compact_with_validation(|candidate| {
            callback_ran.store(true, Ordering::Release);
            assert_eq!(candidate.sequence(), before.sequence());
            assert_eq!(candidate.base_sequence(), before.sequence());
            assert_eq!(
                candidate.get(0).unwrap().unwrap(),
                (vec![0; DIMS], vec![19])
            );
            assert!(candidate.get(2).unwrap().is_none());
            assert_eq!(
                candidate.get(inserted).unwrap().unwrap(),
                (vec![1; DIMS], vec![19])
            );
            let filtered = candidate.search(&[0; DIMS], 19, 2, 200).unwrap();
            assert_eq!(
                filtered.hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
                [0, inserted]
            );
            verify_results(candidate, 7);
            retained_candidate = Some(candidate.clone());
            Err(ANNError::message(
                "application quality gate rejected candidate",
            ))
        })
        .await
        .unwrap_err();
    assert!(callback_ran.load(Ordering::Acquire));
    assert!(error
        .to_string()
        .contains("application quality gate rejected candidate"));
    assert_eq!(
        fs::read(directory.path().join("CURRENT")).unwrap(),
        current_bytes
    );
    assert_eq!(
        fs::read(directory.path().join("updates.wal")).unwrap(),
        journal_bytes
    );
    assert_eq!(index.snapshot().sequence(), before.sequence());
    assert_eq!(index.snapshot().base_sequence(), before.base_sequence());
    assert_eq!(index.snapshot().get(0).unwrap(), before.get(0).unwrap());
    assert!(index.snapshot().get(2).unwrap().is_none());
    assert_eq!(
        GenerationStore::read_current(directory.path())
            .unwrap()
            .unwrap()
            .generation,
        before_generation
    );
    let orphan = directory
        .path()
        .join(format!("generation-{:020}", before_generation + 1));
    assert!(orphan.is_dir());
    assert_eq!(index.collect_garbage().unwrap(), 0);
    verify_results(retained_candidate.as_ref().unwrap(), 7);
    drop(retained_candidate);
    assert_eq!(index.collect_garbage().unwrap(), 1);
    assert!(!orphan.exists());
    assert_eq!(
        GenerationStore::read_current(directory.path())
            .unwrap()
            .unwrap()
            .generation,
        before_generation
    );
    let sequence = before.sequence();
    drop((before, index));
    let reopened = FilteredIndex::<u8>::open(directory.path(), 2).unwrap();
    assert_eq!(reopened.snapshot().sequence(), sequence);
    assert!(reopened.snapshot().get(2).unwrap().is_none());
    assert_eq!(
        reopened.snapshot().get(inserted).unwrap().unwrap(),
        (vec![1; DIMS], vec![19])
    );
    assert_eq!(
        reopened.insert(vec![2; DIMS], vec![19]).unwrap().0,
        inserted + 1
    );
}

#[tokio::test]
async fn admission_failures_leave_the_committed_sequence_and_wal_unchanged() {
    let directory = tempdir().unwrap();
    let mut settings = config();
    settings.delta_limits = DeltaLimits {
        max_records: 1,
        max_bytes: 1024,
    };
    settings.max_frame_bytes = 128;
    settings.max_journal_bytes = 256;
    let index = build(directory.path(), settings).await;
    index.apply(&[Mutation::Delete { id: 0 }]).unwrap();
    let before = fs::metadata(directory.path().join("updates.wal"))
        .unwrap()
        .len();
    assert!(index.apply(&[Mutation::Delete { id: 1 }]).is_err());
    assert_eq!(index.snapshot().sequence(), 1);
    assert!(index.snapshot().get(1).unwrap().is_some());
    assert_eq!(
        fs::metadata(directory.path().join("updates.wal"))
            .unwrap()
            .len(),
        before
    );
    // Repeated replacement fits the delta but eventually reaches the journal cap.
    let mut successful = 0;
    while index.apply(&[Mutation::Delete { id: 0 }]).is_ok() {
        successful += 1;
        assert!(successful < 10);
    }
    let sequence = index.snapshot().sequence();
    drop(index);
    let reopened = FilteredIndex::<u8>::open(directory.path(), 2).unwrap();
    assert_eq!(reopened.snapshot().sequence(), sequence);
    assert!(reopened.snapshot().get(0).unwrap().is_none());
}

#[tokio::test]
async fn reload_rejects_wrong_scalar_corrupt_artifacts_and_missing_journal() {
    let directory = tempdir().unwrap();
    let index = build(directory.path(), config()).await;
    drop(index);
    assert!(FilteredIndex::<f32>::open(directory.path(), 2).is_err());
    let loaded = GenerationStore::read_current(directory.path())
        .unwrap()
        .unwrap();
    let pivot_bytes = fs::read(loaded.directory.join("index_pq_pivots.bin")).unwrap();
    let malformed = tempdir().unwrap();
    let bad_codebook = malformed.path().join("truncated-pq.bin");
    fs::write(&bad_codebook, &pivot_bytes[..16]).unwrap();
    let rejected_root = malformed.path().join("rejected-index");
    let mut settings = config();
    settings.initial_pq_codebook = Some(bad_codebook);
    let values = values();
    assert!(FilteredIndex::build(
        &rejected_root,
        rowmajor::Ref::try_from_data(&values, ROWS, DIMS).unwrap(),
        BTreeMap::from([(7, (0..ROWS as u32).collect())]),
        settings
    )
    .await
    .is_err());
    assert!(!rejected_root.join("CURRENT").exists());
    assert!(!rejected_root.join("updates.wal").exists());
    let path = loaded.directory.join("vectors.bin");
    let mut file = OpenOptions::new().write(true).open(path).unwrap();
    file.seek(SeekFrom::Start(8)).unwrap();
    file.write_all(&[255]).unwrap();
    drop(file);
    assert!(FilteredIndex::<u8>::open(directory.path(), 2).is_err());

    let second = tempdir().unwrap();
    drop(build(second.path(), config()).await);
    fs::remove_file(second.path().join("updates.wal")).unwrap();
    assert!(FilteredIndex::<u8>::open(second.path(), 2).is_err());
    assert!(!second.path().join("updates.wal").exists());
}

#[tokio::test]
async fn f32_empty_compaction_keeps_allocation_watermark_and_rejects_invalid_mutations() {
    let directory = tempdir().unwrap();
    let values: Vec<f32> = (0..5).flat_map(|row| [row as f32, 0., 0., 0.]).collect();
    let vectors = rowmajor::Ref::try_from_data(values.as_slice(), 5, 4).unwrap();
    let index = FilteredIndex::build(
        directory.path(),
        vectors,
        BTreeMap::from([(42, vec![0, 2, 4]), (7, vec![1, 3])]),
        config(),
    )
    .await
    .unwrap();
    let old = index.snapshot();
    let result = old.search(&[0.; 4], 42, 3, 8).unwrap();
    assert!(result.exact_fallback);
    assert_eq!(
        result.hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        [0, 2, 4]
    );
    assert_eq!(
        result
            .hits
            .iter()
            .map(|hit| hit.distance)
            .collect::<Vec<_>>(),
        [0., 4., 16.]
    );
    for vector in [vec![f32::NAN; 4], vec![f32::INFINITY; 4], vec![0.; 3]] {
        assert!(index
            .apply(&[
                Mutation::Delete { id: 0 },
                Mutation::Upsert {
                    id: 1,
                    vector,
                    labels: vec![42]
                }
            ])
            .is_err());
        assert_eq!(index.snapshot().sequence(), 0);
        assert!(index.snapshot().get(0).unwrap().is_some());
    }
    assert!(old.search(&[f32::NAN; 4], 42, 1, 8).is_err());
    index
        .apply(&(0..5).map(|id| Mutation::Delete { id }).collect::<Vec<_>>())
        .unwrap();
    index.compact().await.unwrap();
    let crashed_stage = directory.path().join(".staging-999-123-0");
    fs::create_dir(&crashed_stage).unwrap();
    fs::write(crashed_stage.join("partial"), b"abandoned preparation").unwrap();
    let unknown = directory.path().join(".staging-unknown");
    fs::create_dir(&unknown).unwrap();
    let leased_generation = directory.path().join("generation-00000000000000000001");
    let stage_symlink = directory.path().join(".staging-998-123-0");
    std::os::unix::fs::symlink(&leased_generation, &stage_symlink).unwrap();
    assert_eq!(index.collect_garbage().unwrap(), 1);
    assert!(!crashed_stage.exists());
    assert!(unknown.is_dir());
    assert!(fs::symlink_metadata(&stage_symlink)
        .unwrap()
        .file_type()
        .is_symlink());
    assert!(leased_generation.is_dir());
    assert!(index
        .snapshot()
        .search(&[0.; 4], 42, 3, 8)
        .unwrap()
        .hits
        .is_empty());
    let loaded = GenerationStore::read_current(directory.path())
        .unwrap()
        .unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(&loaded.manifest_payload).unwrap();
    assert_eq!(manifest["has_graph"], false);
    assert_eq!(manifest["next_id"], 5);
    assert_eq!(
        fs::metadata(loaded.directory.join("vectors.bin"))
            .unwrap()
            .len(),
        8
    );
    let (inserted, sequence) = index.insert(vec![0.5; 4], vec![42]).unwrap();
    assert_eq!(inserted, 5);
    let result = index.snapshot().search(&[0.; 4], 42, 3, 8).unwrap();
    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.hits[0].id, inserted);
    assert_eq!(result.hits[0].distance, 1.);
    assert!(old.get(0).unwrap().is_some());
    drop((old, index));
    let reopened = FilteredIndex::<f32>::open(directory.path(), 2).unwrap();
    assert_eq!(reopened.snapshot().sequence(), sequence);
    assert_eq!(
        reopened.snapshot().get(5).unwrap().unwrap(),
        (vec![0.5; 4], vec![42])
    );
    assert_eq!(reopened.insert(vec![1.; 4], vec![7]).unwrap().0, 6);
}

#[tokio::test]
async fn f32_overflow_domain_is_rejected_before_wal_and_during_reload() {
    let directory = tempdir().unwrap();
    let mut settings = config();
    settings.pq_chunks = 1;
    let rejected_root = directory.path().join("rejected");
    let bad = [f32::MAX];
    assert!(FilteredIndex::build(
        &rejected_root,
        rowmajor::Ref::column_vector(&bad),
        BTreeMap::from([(7, vec![0])]),
        settings.clone()
    )
    .await
    .is_err());
    assert!(!rejected_root.join("CURRENT").exists());
    // Each row is supported; treating the whole matrix as one vector is not.
    let values = [4.0e18f32; 4];
    let root = directory.path().join("valid");
    let index = FilteredIndex::build(
        &root,
        rowmajor::Ref::column_vector(&values),
        BTreeMap::from([(7, vec![0, 1, 2, 3]), (99, vec![])]),
        settings.clone(),
    )
    .await
    .unwrap();
    let old = index.snapshot();
    let wal = fs::read(root.join("updates.wal")).unwrap();
    assert!(index
        .apply(&[
            Mutation::Delete { id: 0 },
            Mutation::Upsert {
                id: 1,
                vector: vec![f32::MAX],
                labels: vec![7],
            }
        ])
        .is_err());
    assert!(index.insert(vec![f32::MAX], vec![7]).is_err());
    assert!(index.replace_vector(0, vec![f32::MAX]).is_err());
    assert_eq!(index.snapshot().sequence(), 0);
    assert_eq!(fs::read(root.join("updates.wal")).unwrap(), wal);
    assert!(index.snapshot().get(0).unwrap().is_some());
    assert!(index.snapshot().search(&[f32::MAX], 7, 2, 4).is_err());
    assert!(index
        .snapshot()
        .search(&[0.], 7, 2, 4)
        .unwrap()
        .hits
        .iter()
        .all(|hit| hit.distance.is_finite()));
    assert!(index
        .snapshot()
        .search(&[0.], 99, 2, 4)
        .unwrap()
        .hits
        .is_empty());
    assert_eq!(index.insert(vec![2.0e18], vec![7]).unwrap().0, 4);
    index.compact().await.unwrap();
    drop((old, index));
    let index = FilteredIndex::<f32>::open(&root, 2).unwrap();
    assert_eq!(index.snapshot().get(4).unwrap().unwrap().0, [2.0e18]);
    drop(index);
    let loaded = GenerationStore::read_current(&root).unwrap().unwrap();
    let mut manifest: serde_json::Value = serde_json::from_slice(&loaded.manifest_payload).unwrap();
    // A validly checksummed older journal can contain a previously admitted value.
    let (mut journal, recovery) =
        Journal::<f32>::open_existing(root.join("updates.wal"), settings.max_frame_bytes).unwrap();
    journal
        .append(
            recovery.last_sequence + 1,
            &[Mutation::Upsert {
                id: 0,
                vector: vec![f32::MAX],
                labels: vec![7],
            }],
        )
        .unwrap();
    drop(journal);
    assert!(FilteredIndex::<f32>::open(&root, 2).is_err());
    // Likewise reject an old unsafe base despite valid generation checksums.
    let legacy_root = directory.path().join("legacy");
    let store = GenerationStore::open(&legacy_root).unwrap();
    let (journal, _) =
        Journal::<f32>::create_new(legacy_root.join("updates.wal"), 0, settings.max_frame_bytes)
            .unwrap();
    manifest["base_sequence"] = serde_json::json!(0);
    manifest["next_id"] = serde_json::json!(1);
    manifest["has_graph"] = serde_json::json!(false);
    let labels = LabelMetadata::from_postings(
        1,
        BTreeMap::from([(7, vec![0])]),
        BTreeMap::from([(7, vec![0])]),
    )
    .unwrap();
    store
        .publish(1, &serde_json::to_vec(&manifest).unwrap(), |path| {
            let mut vectors = Vec::new();
            vectors.extend_from_slice(&1u32.to_le_bytes());
            vectors.extend_from_slice(&1u32.to_le_bytes());
            vectors.extend_from_slice(&f32::MAX.to_le_bytes());
            fs::write(path.join("vectors.bin"), vectors)?;
            fs::write(
                path.join("labels.bin"),
                bincode::serialize(&labels).unwrap(),
            )
        })
        .unwrap();
    drop((journal, store));
    assert!(FilteredIndex::<f32>::open(&legacy_root, 2).is_err());
}
