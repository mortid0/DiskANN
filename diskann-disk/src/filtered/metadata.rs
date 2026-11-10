/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Immutable base membership. Postings and entry points always use base ordinals.

use std::collections::{BTreeMap, BTreeSet};

use diskann_utils::views::rowmajor::{self, Matrix};

mod sidecars;
use thiserror::Error;

/// Sparse base labels and the optional mapping from base ordinals to stable IDs.
///
/// Deleted or relabeled base rows retain this original membership for navigation.
/// The current delta determines whether their stable IDs may be returned.
#[derive(Clone, Debug)]
pub struct LabelMetadata {
    point_count: u32,
    offsets: Vec<u64>,
    labels: Vec<u32>,
    postings: BTreeMap<u32, Vec<u32>>,
    entry_points: BTreeMap<u32, Vec<u32>>,
    stable_ids: Option<Vec<u32>>,
    label_map: BTreeMap<String, u32>,
    universal_label: Option<u32>,
    dummy_map: BTreeMap<u32, u32>,
    medoids: Vec<u32>,
    centroids: Option<rowmajor::Owned<f32>>,
}

#[derive(Debug, Error)]
pub enum MetadataError {
    #[error("base row count exceeds the supported u32 ordinal range")]
    TooManyPoints,
    #[error("invalid filtered metadata: {0}")]
    Invalid(String),
}

impl LabelMetadata {
    /// Construct CSR membership from sorted, unique ordinal postings.
    pub fn from_postings(
        point_count: usize,
        postings: BTreeMap<u32, Vec<u32>>,
        entry_points: BTreeMap<u32, Vec<u32>>,
    ) -> Result<Self, MetadataError> {
        let rows = u32::try_from(point_count).map_err(|_| MetadataError::TooManyPoints)?;
        let offset_count = point_count
            .checked_add(1)
            .ok_or(MetadataError::TooManyPoints)?;
        let mut offsets = vec![0u64; offset_count];
        for (&label, members) in &postings {
            validate_members(label, members, rows)?;
            for &id in members {
                offsets[id as usize + 1] += 1;
            }
        }
        for row in 0..point_count {
            offsets[row + 1] = offsets[row + 1]
                .checked_add(offsets[row])
                .ok_or_else(|| MetadataError::Invalid("membership count overflow".into()))?;
        }
        let count = usize::try_from(offsets[point_count])
            .map_err(|_| MetadataError::Invalid("membership count exceeds usize".into()))?;
        let mut labels = vec![0; count];
        let mut cursors = offsets[..point_count].to_vec();
        for (&label, members) in &postings {
            for &id in members {
                labels[cursors[id as usize] as usize] = label;
                cursors[id as usize] += 1;
            }
        }
        let metadata = Self {
            point_count: rows,
            offsets,
            labels,
            label_map: postings.keys().map(|id| (id.to_string(), *id)).collect(),
            postings,
            entry_points,
            stable_ids: None,
            universal_label: None,
            dummy_map: BTreeMap::new(),
            medoids: Vec::new(),
            centroids: None,
        };
        metadata.validate()?;
        Ok(metadata)
    }

    /// Construct metadata for live rows stored in ascending stable-ID order.
    pub fn from_postings_with_ids(
        point_count: usize,
        postings: BTreeMap<u32, Vec<u32>>,
        entry_points: BTreeMap<u32, Vec<u32>>,
        stable_ids: Vec<u32>,
    ) -> Result<Self, MetadataError> {
        Self::from_postings(point_count, postings, entry_points)?.with_stable_ids(stable_ids)
    }

    pub fn with_stable_ids(mut self, stable_ids: Vec<u32>) -> Result<Self, MetadataError> {
        self.stable_ids = Some(stable_ids);
        self.validate()?;
        Ok(self)
    }

    pub fn point_count(&self) -> usize {
        self.point_count as usize
    }

    pub fn labels(&self, ordinal: u32) -> &[u32] {
        if ordinal >= self.point_count {
            return &[];
        }
        &self.labels
            [self.offsets[ordinal as usize] as usize..self.offsets[ordinal as usize + 1] as usize]
    }

    pub fn posting(&self, label: u32) -> &[u32] {
        self.postings.get(&label).map_or(&[], Vec::as_slice)
    }

    pub fn entries(&self, label: u32) -> &[u32] {
        self.entry_points.get(&label).map_or(&[], Vec::as_slice)
    }

    pub fn contains(&self, label: u32, ordinal: u32) -> bool {
        self.labels(ordinal).binary_search(&label).is_ok()
    }

    /// C++ admits universal points for every filter, including the universal filter itself.
    pub fn matches(&self, label: u32, ordinal: u32) -> bool {
        self.contains(label, ordinal)
            || self
                .universal_label
                .is_some_and(|universal| self.contains(universal, ordinal))
    }

    /// Convert an original label name, falling back to the universal label as C++ does.
    pub fn resolve_label(&self, name: &str) -> Result<u32, MetadataError> {
        self.label_map
            .get(name)
            .copied()
            .or(self.universal_label)
            .ok_or_else(|| MetadataError::Invalid(format!("unknown label {name:?}")))
    }

    pub fn is_dummy(&self, ordinal: u32) -> bool {
        self.dummy_map.contains_key(&ordinal)
    }

    /// Translate after ranking; C++ does not deduplicate mapped result IDs.
    pub fn real_id(&self, ordinal: u32) -> u32 {
        self.dummy_map.get(&ordinal).copied().unwrap_or(ordinal)
    }

    pub fn medoids(&self) -> &[u32] {
        &self.medoids
    }

    pub fn centroids(&self) -> Option<rowmajor::Ref<'_, f32>> {
        self.centroids.as_ref().map(|matrix| matrix.as_view())
    }

    pub(crate) fn set_global_starts(&mut self, medoids: Vec<u32>, centroids: rowmajor::Owned<f32>) {
        self.medoids = medoids;
        self.centroids = Some(centroids);
    }

    pub fn postings(&self) -> &BTreeMap<u32, Vec<u32>> {
        &self.postings
    }

    pub fn entry_points(&self) -> &BTreeMap<u32, Vec<u32>> {
        &self.entry_points
    }

    pub fn stable_ids(&self) -> Option<&[u32]> {
        self.stable_ids.as_deref()
    }

    pub fn stable_id(&self, ordinal: u32) -> Option<u32> {
        if ordinal >= self.point_count {
            return None;
        }
        Some(
            self.stable_ids
                .as_ref()
                .map_or(ordinal, |ids| ids[ordinal as usize]),
        )
    }

    pub fn ordinal(&self, stable_id: u32) -> Option<u32> {
        match &self.stable_ids {
            Some(ids) => ids.binary_search(&stable_id).ok().map(|id| id as u32),
            None => (stable_id < self.point_count).then_some(stable_id),
        }
    }

    /// Check loaded CSR, inverse membership, entries, and stable-ID mapping.
    pub fn validate(&self) -> Result<(), MetadataError> {
        let invalid = |message: &str| MetadataError::Invalid(message.to_owned());
        if self.point_count().checked_add(1) != Some(self.offsets.len())
            || self.offsets.first() != Some(&0)
            || self.offsets.last().copied() != Some(self.labels.len() as u64)
            || self.offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(invalid(
                "CSR offsets do not match the rows and membership data",
            ));
        }
        for ordinal in 0..self.point_count {
            if !strictly_sorted(self.labels(ordinal)) {
                return Err(invalid("a CSR label row is not sorted and unique"));
            }
        }
        let mut memberships = 0usize;
        for (&label, members) in &self.postings {
            validate_members(label, members, self.point_count)?;
            memberships = memberships
                .checked_add(members.len())
                .ok_or_else(|| invalid("membership count overflow"))?;
            if members.iter().any(|&id| !self.contains(label, id)) {
                return Err(invalid("postings disagree with CSR membership"));
            }
            if !members.is_empty() && self.entries(label).is_empty() {
                return Err(invalid("a nonempty label has no entry point"));
            }
        }
        if memberships != self.labels.len() {
            return Err(invalid("CSR membership has no corresponding posting"));
        }
        for (&label, entries) in &self.entry_points {
            let mut seen = BTreeSet::new();
            for &entry in entries {
                if !seen.insert(entry) || !self.matches(label, entry) {
                    return Err(invalid("entry points are duplicate or outside their label"));
                }
            }
        }
        if let Some(ids) = &self.stable_ids {
            if ids.len() != self.point_count() || !strictly_sorted(ids) {
                return Err(invalid(
                    "stable IDs must be sorted, unique, and match base row count",
                ));
            }
        }
        Ok(())
    }
}

fn strictly_sorted(ids: &[u32]) -> bool {
    ids.windows(2).all(|pair| pair[0] < pair[1])
}

fn validate_members(label: u32, ids: &[u32], points: u32) -> Result<(), MetadataError> {
    if !strictly_sorted(ids) || ids.last().is_some_and(|&id| id >= points) {
        return Err(MetadataError::Invalid(format!(
            "label {label} postings must be sorted, unique base ordinals"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> LabelMetadata {
        LabelMetadata::from_postings_with_ids(
            3,
            BTreeMap::from([(7, vec![0, 2]), (9, vec![1, 2])]),
            BTreeMap::from([(7, vec![2]), (9, vec![1])]),
            vec![2, 8, 100],
        )
        .unwrap()
    }

    #[test]
    fn sparse_membership_preserves_ordinals_and_stable_ids() {
        let metadata = example();
        assert_eq!(metadata.labels(2), [7, 9]);
        assert_eq!(metadata.posting(7), [0, 2]);
        assert_eq!(metadata.entries(7), [2]);
        assert_eq!(metadata.stable_id(2), Some(100));
        assert_eq!(metadata.ordinal(100), Some(2));
        assert!(!metadata.contains(7, 100));
        assert_eq!(metadata.ordinal(3), None);
    }

    #[test]
    fn invalid_loaded_membership_and_mapping_are_rejected() {
        let mut metadata = example();
        metadata.labels[0] = 99;
        assert!(metadata.validate().is_err());
        assert!(example().with_stable_ids(vec![2, 2, 100]).is_err());
        assert!(example().with_stable_ids(vec![2, 8]).is_err());
    }

    #[test]
    fn missing_or_nonmatching_entries_are_rejected() {
        let postings = BTreeMap::from([(7, vec![0, 2])]);
        assert!(LabelMetadata::from_postings(3, postings.clone(), BTreeMap::new()).is_err());
        assert!(LabelMetadata::from_postings(3, postings, BTreeMap::from([(7, vec![1])])).is_err());
    }
}
