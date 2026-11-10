/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Immutable, bounded current-version overrides for an immutable base generation.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use diskann::utils::VectorRepr;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const SHARDS: usize = 256;
const RECORD_BYTES: usize = 128;
const MEMBERSHIP_BYTES: usize = 64;

/// A batch carries complete replacements so vector and label changes are atomic.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Mutation<T> {
    Upsert {
        id: u32,
        vector: Vec<T>,
        labels: Vec<u32>,
    },
    Delete {
        id: u32,
    },
}

impl<T> Mutation<T> {
    pub fn id(&self) -> u32 {
        match self {
            Self::Upsert { id, .. } | Self::Delete { id } => *id,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeltaPoint<T> {
    pub vector: Vec<T>,
    /// Sorted and unique current labels, independent of original base labels.
    pub labels: Vec<u32>,
}

impl<T> DeltaPoint<T> {
    pub fn matches(&self, label: u32) -> bool {
        self.labels.binary_search(&label).is_ok()
    }
}

#[derive(Clone, Debug)]
pub enum DeltaRecord<T> {
    Live(Arc<DeltaPoint<T>>),
    Deleted,
}

impl<T> DeltaRecord<T> {
    pub fn point(&self) -> Option<&DeltaPoint<T>> {
        match self {
            Self::Live(point) => Some(point),
            Self::Deleted => None,
        }
    }

    pub fn is_deleted(&self) -> bool {
        matches!(self, Self::Deleted)
    }
}

/// Limits on the current delta, including tombstones.
///
/// Byte accounting includes vector payload and conservative fixed record/membership
/// allowances. This is an admission budget, not measured process RSS. Retained reader
/// snapshots and temporary copy-on-write allocations can consume additional memory.
#[derive(Clone, Copy, Debug)]
pub struct DeltaLimits {
    pub max_records: usize,
    pub max_bytes: usize,
}

impl Default for DeltaLimits {
    fn default() -> Self {
        Self {
            max_records: 10_000,
            max_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug)]
struct Shard<T> {
    records: BTreeMap<u32, DeltaRecord<T>>,
    postings: BTreeMap<u32, Arc<BTreeSet<u32>>>,
}

impl<T> Default for Shard<T> {
    fn default() -> Self {
        Self {
            records: BTreeMap::new(),
            postings: BTreeMap::new(),
        }
    }
}

/// Reader-owned snapshots never observe half an update batch.
///
/// Each batch copies the fixed shard directory and only the touched shards/postings;
/// it never clones the immutable base or the complete delta record map.
#[derive(Clone, Debug)]
pub struct DeltaSnapshot<T> {
    sequence: u64,
    dimensions: usize,
    records: usize,
    bytes: usize,
    shards: Arc<[Arc<Shard<T>>]>,
}

#[derive(Debug, Error)]
pub enum UpdateError {
    #[error("vector dimension must be positive")]
    EmptyDimension,
    #[error("batch sequence must be exactly one after {current}, got {actual}")]
    Sequence { current: u64, actual: u64 },
    #[error("an update batch must not be empty")]
    EmptyBatch,
    #[error("stable ID {0} occurs more than once in one batch")]
    DuplicateId(u32),
    #[error("vector {id} has dimension {actual}, expected {expected}")]
    Dimension {
        id: u32,
        actual: usize,
        expected: usize,
    },
    #[error("vector {0} contains a non-finite or unsupported value")]
    InvalidVector(u32),
    #[error("delta size accounting overflow")]
    SizeOverflow,
    #[error("delta admission budget exceeded: {records} records and {bytes} accounted bytes")]
    Limit { records: usize, bytes: usize },
}

impl<T> DeltaSnapshot<T> {
    pub fn new(sequence: u64, dimensions: usize) -> Result<Self, UpdateError> {
        if dimensions == 0 {
            return Err(UpdateError::EmptyDimension);
        }
        Ok(Self {
            sequence,
            dimensions,
            records: 0,
            bytes: 0,
            shards: (0..SHARDS)
                .map(|_| Arc::new(Shard::default()))
                .collect::<Vec<_>>()
                .into(),
        })
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn record_count(&self) -> usize {
        self.records
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.records == 0
    }

    /// Any returned record shadows the same stable ID in the immutable base.
    pub fn get(&self, id: u32) -> Option<&DeltaRecord<T>> {
        self.shards[shard_index(id)].records.get(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (u32, &DeltaRecord<T>)> {
        self.shards
            .iter()
            .flat_map(|shard| shard.records.iter().map(|(&id, record)| (id, record)))
    }

    /// Visit only live overrides in this label's sparse postings.
    pub fn iter_matching(&self, label: u32) -> impl Iterator<Item = (u32, &DeltaPoint<T>)> {
        self.shards.iter().flat_map(move |shard| {
            shard.postings.get(&label).into_iter().flat_map(move |ids| {
                ids.iter().filter_map(move |id| {
                    shard
                        .records
                        .get(id)
                        .and_then(DeltaRecord::point)
                        .map(|point| (*id, point))
                })
            })
        })
    }

    pub fn label_ids(&self, label: u32) -> Vec<u32> {
        let mut ids: Vec<_> = self.iter_matching(label).map(|(id, _)| id).collect();
        ids.sort_unstable();
        ids
    }
}

impl<T: VectorRepr> DeltaSnapshot<T> {
    /// Stage a complete immutable batch before its journal frame is committed.
    ///
    /// Duplicate IDs in a batch are rejected. Upsert label order/duplicates are
    /// canonicalized; empty label sets are allowed. Allocation and base-ID policy
    /// belong to the enclosing lifecycle, which persists the next unused stable ID.
    pub fn apply_batch(
        &self,
        sequence: u64,
        mutations: &[Mutation<T>],
        limits: DeltaLimits,
    ) -> Result<Self, UpdateError> {
        if self.sequence.checked_add(1) != Some(sequence) {
            return Err(UpdateError::Sequence {
                current: self.sequence,
                actual: sequence,
            });
        }
        if mutations.is_empty() {
            return Err(UpdateError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        for mutation in mutations {
            let id = mutation.id();
            if !seen.insert(id) {
                return Err(UpdateError::DuplicateId(id));
            }
            if let Mutation::Upsert { vector, .. } = mutation {
                if vector.len() != self.dimensions {
                    return Err(UpdateError::Dimension {
                        id,
                        actual: vector.len(),
                        expected: self.dimensions,
                    });
                }
                let full = T::as_f32(vector).map_err(|_| UpdateError::InvalidVector(id))?;
                if full.iter().any(|value| !value.is_finite()) {
                    return Err(UpdateError::InvalidVector(id));
                }
            }
        }

        let mut shards = self.shards.to_vec();
        let mut records = self.records;
        let mut bytes = self.bytes;
        for mutation in mutations {
            let id = mutation.id();
            let shard = Arc::make_mut(&mut shards[shard_index(id)]);
            if let Some(previous) = shard.records.remove(&id) {
                records -= 1;
                bytes -= record_bytes(&previous)?;
                if let Some(point) = previous.point() {
                    for label in &point.labels {
                        if let Some(ids) = shard.postings.get_mut(label) {
                            Arc::make_mut(ids).remove(&id);
                            if ids.is_empty() {
                                shard.postings.remove(label);
                            }
                        }
                    }
                }
            }
            let record = match mutation {
                Mutation::Upsert { vector, labels, .. } => {
                    let mut labels = labels.clone();
                    labels.sort_unstable();
                    labels.dedup();
                    for &label in &labels {
                        Arc::make_mut(shard.postings.entry(label).or_default()).insert(id);
                    }
                    DeltaRecord::Live(Arc::new(DeltaPoint {
                        vector: vector.clone(),
                        labels,
                    }))
                }
                Mutation::Delete { .. } => DeltaRecord::Deleted,
            };
            records = records.checked_add(1).ok_or(UpdateError::SizeOverflow)?;
            bytes = bytes
                .checked_add(record_bytes(&record)?)
                .ok_or(UpdateError::SizeOverflow)?;
            shard.records.insert(id, record);
        }
        if records > limits.max_records || bytes > limits.max_bytes {
            return Err(UpdateError::Limit { records, bytes });
        }
        Ok(Self {
            sequence,
            dimensions: self.dimensions,
            records,
            bytes,
            shards: shards.into(),
        })
    }
}

fn shard_index(id: u32) -> usize {
    id as usize % SHARDS
}

fn record_bytes<T>(record: &DeltaRecord<T>) -> Result<usize, UpdateError> {
    let Some(point) = record.point() else {
        return Ok(RECORD_BYTES);
    };
    point
        .vector
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| {
            point
                .labels
                .len()
                .checked_mul(MEMBERSHIP_BYTES)
                .and_then(|labels| bytes.checked_add(labels))
        })
        .and_then(|bytes| bytes.checked_add(RECORD_BYTES))
        .ok_or(UpdateError::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_keep_old_vectors_labels_and_sparse_postings() {
        let initial = DeltaSnapshot::<u8>::new(10, 2).unwrap();
        let first = initial
            .apply_batch(
                11,
                &[Mutation::Upsert {
                    id: 99,
                    vector: vec![1, 2],
                    labels: vec![7, 3, 7],
                }],
                DeltaLimits::default(),
            )
            .unwrap();
        let second = first
            .apply_batch(
                12,
                &[
                    Mutation::Upsert {
                        id: 99,
                        vector: vec![8, 9],
                        labels: vec![8],
                    },
                    Mutation::Delete { id: 4 },
                ],
                DeltaLimits::default(),
            )
            .unwrap();
        assert!(initial.is_empty());
        assert_eq!(first.label_ids(3), [99]);
        assert_eq!(first.get(99).unwrap().point().unwrap().vector, [1, 2]);
        assert!(second.label_ids(3).is_empty());
        assert_eq!(second.label_ids(8), [99]);
        assert!(second.get(4).unwrap().is_deleted());
        assert!(Arc::ptr_eq(&first.shards[0], &second.shards[0]));
        assert!(!Arc::ptr_eq(&first.shards[99], &second.shards[99]));
    }

    #[test]
    fn invalid_or_oversized_batches_do_not_publish_partial_state() {
        let initial = DeltaSnapshot::<f32>::new(0, 1).unwrap();
        assert!(initial
            .apply_batch(
                1,
                &[
                    Mutation::Delete { id: 1 },
                    Mutation::Upsert {
                        id: 2,
                        vector: vec![f32::NAN],
                        labels: vec![3],
                    },
                ],
                DeltaLimits::default(),
            )
            .is_err());
        assert!(initial
            .apply_batch(
                1,
                &[Mutation::Delete { id: 1 }],
                DeltaLimits {
                    max_records: 0,
                    max_bytes: usize::MAX,
                },
            )
            .is_err());
        assert!(initial.is_empty());
        assert_eq!(initial.sequence(), 0);
    }

    #[test]
    fn deleting_a_live_override_removes_it_from_label_postings() {
        let initial = DeltaSnapshot::<u8>::new(0, 1).unwrap();
        let first = initial
            .apply_batch(
                1,
                &[Mutation::Upsert {
                    id: 256,
                    vector: vec![4],
                    labels: vec![7],
                }],
                DeltaLimits::default(),
            )
            .unwrap();
        let second = first
            .apply_batch(2, &[Mutation::Delete { id: 256 }], DeltaLimits::default())
            .unwrap();
        assert!(second.label_ids(7).is_empty());
        assert_eq!(second.record_count(), 1);
        assert!(first.get(256).unwrap().point().is_some());
    }
}
