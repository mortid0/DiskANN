/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![cfg(all(unix, not(miri)))]

use std::collections::BTreeMap;

use diskann_disk::{
    build::filtered::StitchedBuildConfig,
    filtered::{updates::Mutation, FilteredIndex, FilteredIndexConfig},
};
use diskann_utils::views::rowmajor;

const BASE_ROWS: u32 = 300;
const DELTA_ROWS: u32 = 1_000;
const DIMS: usize = 4;
const HOT: u32 = 7;
const OVERLAP: u32 = 9;
const FRESH: u32 = 101;

type Point = (Vec<u8>, Vec<u32>);

fn exact_truth(
    model: &BTreeMap<u32, Point>,
    query: &[u8],
    label: u32,
    k: usize,
) -> Vec<(u64, u32)> {
    let mut truth: Vec<_> = model
        .iter()
        .filter(|(_, (_, labels))| labels.contains(&label))
        .map(|(&id, (vector, _))| {
            let distance = vector
                .iter()
                .zip(query)
                .map(|(&value, &target)| {
                    let difference = i64::from(value) - i64::from(target);
                    (difference * difference) as u64
                })
                .sum();
            (distance, id)
        })
        .collect();
    truth.sort_unstable();
    truth.truncate(k);
    truth
}

#[tokio::test]
async fn large_tied_delta_merges_with_native_base_and_reports_all_matching_candidates() {
    let directory = tempfile::tempdir().unwrap();
    let mut model = BTreeMap::new();
    let mut values = Vec::with_capacity(BASE_ROWS as usize * DIMS);
    for id in 0..BASE_ROWS {
        let vector = vec![(id % 31) as u8, (id / 31) as u8, ((id * 7) % 23) as u8, 0];
        let labels = if id % 2 == 1 {
            vec![HOT, OVERLAP]
        } else {
            vec![HOT]
        };
        values.extend_from_slice(&vector);
        model.insert(id, (vector, labels));
    }
    let config = FilteredIndexConfig {
        build: StitchedBuildConfig {
            label_degree: 8,
            max_degree: 32,
            l_build: 32,
            concurrency: 2,
            seed: 777,
            ..Default::default()
        },
        pq_chunks: 2,
        threads: 2,
        ..Default::default()
    };
    let index = FilteredIndex::build(
        directory.path(),
        rowmajor::Ref::try_from_data(&values, BASE_ROWS as usize, DIMS).unwrap(),
        BTreeMap::from([
            (HOT, (0..BASE_ROWS).collect()),
            (OVERLAP, (0..BASE_ROWS).filter(|id| id % 2 == 1).collect()),
        ]),
        config,
    )
    .await
    .unwrap();
    let before = index.snapshot();

    // Sequential stable IDs span all 256 shards. Query iteration therefore visits
    // them in a different order from their stable-ID tie order.
    let mut mutations = Vec::new();
    let mut delta_labels = BTreeMap::<u32, Vec<u32>>::new();
    for offset in 0..DELTA_ROWS {
        let id = BASE_ROWS + offset;
        // Repeated vectors place many stable IDs exactly on each top-k boundary.
        let vector = vec![(offset % 31) as u8, 0, 0, 0];
        let mut labels = vec![FRESH];
        if offset % 3 != 0 {
            labels.push(HOT);
        }
        if offset % 5 == 0 {
            labels.push(OVERLAP);
        }
        labels.sort_unstable();
        model.insert(id, (vector.clone(), labels.clone()));
        delta_labels.insert(id, labels.clone());
        mutations.push(Mutation::Upsert { id, vector, labels });
    }
    mutations.push(Mutation::Delete { id: 0 });
    model.remove(&0);
    let replacement = (vec![0; DIMS], vec![HOT, OVERLAP]);
    model.insert(31, replacement.clone());
    delta_labels.insert(31, replacement.1.clone());
    mutations.push(Mutation::Upsert {
        id: 31,
        vector: replacement.0,
        labels: replacement.1,
    });
    let sequence = index.apply(&mutations).unwrap();
    let snapshot = index.snapshot();
    assert_eq!(sequence, 1);
    assert!(before
        .search(&[0; DIMS], FRESH, 5, 350)
        .unwrap()
        .hits
        .is_empty());
    assert!(before.get(0).unwrap().is_some());
    assert!(snapshot.get(0).unwrap().is_none());

    for (label, ks) in [
        (FRESH, vec![1, 17, 64, 257, 1_001]),
        (HOT, vec![1, 17, 64, 1_000]),
        (OVERLAP, vec![1, 17, 64, 400]),
        (u32::MAX, vec![1, 17]),
    ] {
        let all_delta_matches = delta_labels
            .values()
            .filter(|labels| labels.contains(&label))
            .count();
        if label == FRESH {
            assert_eq!(all_delta_matches, DELTA_ROWS as usize);
        }
        for query in [[0, 0, 0, 0], [15, 0, 0, 0], [31, 0, 0, 0]] {
            for &k in &ks {
                // Exhaust the small native base so the oracle tests exact merging,
                // rather than making an ANN recall assumption.
                let l = k.max(350) as u32;
                let result = snapshot.search(&query, label, k, l).unwrap();
                let actual: Vec<_> = result
                    .hits
                    .iter()
                    .map(|hit| (hit.distance, hit.id))
                    .collect();
                let truth = exact_truth(&model, &query, label, k);
                let expected: Vec<_> = truth
                    .iter()
                    .map(|&(distance, id)| (distance as f32, id))
                    .collect();
                assert_eq!(actual, expected, "query {query:?}, label {label}, k {k}");
                assert_eq!(result.hits.len(), truth.len());
                assert_eq!(result.sequence, sequence);
                assert_eq!(
                    result.delta_candidates, all_delta_matches,
                    "candidate accounting must include matches discarded before merging"
                );
                for hit in &result.hits {
                    assert!(model[&hit.id].1.contains(&label));
                    assert!(hit.distance.is_finite());
                }
                if label == FRESH || label == u32::MAX {
                    assert_eq!(result.base_searches, 0);
                } else if k < 150 {
                    assert!(
                        result.base_searches > 0,
                        "must exercise the native SSD result pool"
                    );
                }
            }
        }
    }
}
