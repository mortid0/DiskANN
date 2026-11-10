/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Immutable label membership and entry points for experimental disk search.
//!
//! Metadata is independent of graph adjacency and is persisted through
//! `diskann-record`. Its single binary artifact contains compressed postings and
//! exact UTF-8 label names. A SHA-256 digest detects artifact corruption; this is
//! an experimental export format, not a mutable store or durability protocol.

use std::{
    collections::BTreeMap,
    io::{Cursor, Read, Write},
    path::Path,
};

use bincode::Options;
use diskann_record::{Version, load, save};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::build::filtered::StitchedGraph;

const ENCODING: &str = "diskann-single-hop-labels/bincode-fixed-le-roaring";
// Bound corrupt manifest lengths before allocating or decoding. This prototype
// supports metadata artifacts up to 1 GiB; larger builds need another format.
const MAX_ARTIFACT_BYTES: u64 = 1 << 30;

#[derive(Debug, Clone)]
struct Label {
    members: RoaringBitmap,
    entry_points: Vec<u32>,
}

/// Compact metadata that remains usable after construction adjacency is dropped.
#[derive(Debug, Clone)]
pub struct LabelMetadata {
    rows: u32,
    labels: BTreeMap<u32, Label>,
    names: BTreeMap<String, u32>,
}

/// Invalid membership or entry-point metadata.
#[derive(Debug, Error)]
pub enum MetadataError {
    #[error("metadata requires 0 < rows < u32::MAX, got {0}")]
    RowCount(usize),
    #[error("labels must be strictly increasing, found {previous} followed by {next}")]
    LabelOrder { previous: u32, next: u32 },
    #[error("label names must be strictly increasing, found {previous:?} followed by {next:?}")]
    NameOrder { previous: String, next: String },
    #[error("label {0} has no members")]
    EmptyPosting(u32),
    #[error("label {label} contains row {row}, outside a dataset of {rows} rows")]
    MemberOutOfBounds { label: u32, row: u32, rows: u32 },
    #[error("label {0} has no entry points")]
    EmptyEntryPoints(u32),
    #[error("entry points for label {0} must be strictly increasing")]
    EntryPointOrder(u32),
    #[error("entry point {row} is not a member of label {label}")]
    EntryOutsidePosting { label: u32, row: u32 },
    #[error("label name {name:?} refers to unknown label {label}")]
    UnknownName { name: String, label: u32 },
}

#[derive(Debug, Error)]
enum ArtifactError {
    #[error("unsupported label metadata encoding {0:?}")]
    Encoding(String),
    #[error("label metadata length {0} is outside 1..=1073741824 bytes")]
    LengthLimit(u64),
    #[error("label metadata length differs from {expected} bytes (read {actual} bytes)")]
    Length { expected: u64, actual: u64 },
    #[error("label metadata SHA-256 digest has {0} bytes, expected 32")]
    DigestLength(usize),
    #[error("label metadata SHA-256 digest does not match its artifact")]
    Checksum,
    #[error("label {label} bitmap has {remaining} trailing bytes")]
    BitmapTrailingBytes { label: u32, remaining: u64 },
}

// Explicit wire structs keep map ordering and duplicate keys observable during
// validation, instead of silently replacing them during map deserialization.
#[derive(Serialize, Deserialize)]
struct WireMetadata {
    rows: u32,
    labels: Vec<WireLabel>,
    names: Vec<WireName>,
}

#[derive(Serialize, Deserialize)]
struct WireLabel {
    label: u32,
    bitmap: Vec<u8>,
    entry_points: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
struct WireName {
    name: String,
    label: u32,
}

fn codec(limit: u64) -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .reject_trailing_bytes()
        .with_limit(limit)
}

impl LabelMetadata {
    /// Detach compressed membership and entry points from a constructed graph.
    /// Names may be omitted, or multiple names may refer to the same label.
    ///
    /// # Errors
    /// Rejects invalid postings or entry points and names for unknown labels.
    pub fn from_graph(
        graph: &StitchedGraph,
        names: &BTreeMap<String, u32>,
    ) -> Result<Self, MetadataError> {
        let labels = graph
            .postings()
            .iter()
            .map(|(&label, members)| {
                (
                    label,
                    Label {
                        members: members.iter().copied().collect(),
                        entry_points: graph
                            .entry_points()
                            .get(&label)
                            .cloned()
                            .unwrap_or_default(),
                    },
                )
            })
            .collect();
        Self::validated(graph.neighbors().len(), labels, names.clone())
    }

    pub fn rows(&self) -> usize {
        self.rows as usize
    }

    /// Unknown labels have no entry points.
    pub fn entry_points(&self, label: u32) -> &[u32] {
        self.labels
            .get(&label)
            .map_or(&[], |value| value.entry_points.as_slice())
    }

    pub fn contains(&self, label: u32, row: u32) -> bool {
        self.labels
            .get(&label)
            .is_some_and(|value| value.members.contains(row))
    }

    pub fn posting(&self, label: u32) -> Option<&RoaringBitmap> {
        self.labels.get(&label).map(|value| &value.members)
    }

    pub fn resolve_label(&self, name: &str) -> Option<u32> {
        self.names.get(name).copied()
    }

    /// Write a record manifest and one binary artifact to an existing directory.
    ///
    /// # Errors
    /// Returns the underlying record or I/O error, or rejects metadata above the
    /// experimental 1 GiB artifact limit. Existing artifacts are not overwritten.
    pub fn save(
        &self,
        directory: impl AsRef<Path>,
        manifest: impl AsRef<Path>,
    ) -> save::Result<()> {
        save::save_to_disk(self, directory, manifest)
    }

    /// Load an independent record, validating its checksum before decoding.
    ///
    /// # Errors
    /// Rejects unknown versions/encodings, truncation, trailing data, corruption,
    /// metadata above 1 GiB, or invalid memberships and entry points.
    pub fn load(manifest: &Path, directory: &Path) -> load::Result<Self> {
        load::load_from_disk(manifest, directory)
    }

    fn validated(
        rows: usize,
        labels: BTreeMap<u32, Label>,
        names: BTreeMap<String, u32>,
    ) -> Result<Self, MetadataError> {
        if rows == 0 || rows >= u32::MAX as usize {
            return Err(MetadataError::RowCount(rows));
        }
        let rows = rows as u32;
        for (&label, value) in &labels {
            let maximum = value
                .members
                .max()
                .ok_or(MetadataError::EmptyPosting(label))?;
            if maximum >= rows {
                return Err(MetadataError::MemberOutOfBounds {
                    label,
                    row: maximum,
                    rows,
                });
            }
            if value.entry_points.is_empty() {
                return Err(MetadataError::EmptyEntryPoints(label));
            }
            if value.entry_points.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(MetadataError::EntryPointOrder(label));
            }
            for &row in &value.entry_points {
                if !value.members.contains(row) {
                    return Err(MetadataError::EntryOutsidePosting { label, row });
                }
            }
        }
        for (name, &label) in &names {
            if !labels.contains_key(&label) {
                return Err(MetadataError::UnknownName {
                    name: name.clone(),
                    label,
                });
            }
        }
        Ok(Self {
            rows,
            labels,
            names,
        })
    }

    fn encode(&self) -> save::Result<Vec<u8>> {
        let labels = self
            .labels
            .iter()
            .map(|(&label, value)| {
                let mut bitmap = Vec::new();
                value
                    .members
                    .serialize_into(&mut bitmap)
                    .map_err(save::Error::new)?;
                Ok(WireLabel {
                    label,
                    bitmap,
                    entry_points: value.entry_points.clone(),
                })
            })
            .collect::<save::Result<_>>()?;
        let names = self
            .names
            .iter()
            .map(|(name, &label)| WireName {
                name: name.clone(),
                label,
            })
            .collect();
        codec(MAX_ARTIFACT_BYTES)
            .serialize(&WireMetadata {
                rows: self.rows,
                labels,
                names,
            })
            .map_err(save::Error::new)
    }

    fn decode(bytes: &[u8]) -> load::Result<Self> {
        let wire: WireMetadata = codec(bytes.len() as u64)
            .deserialize(bytes)
            .map_err(load::Error::new)?;
        for pair in wire.labels.windows(2) {
            if pair[0].label >= pair[1].label {
                return Err(load::Error::new(MetadataError::LabelOrder {
                    previous: pair[0].label,
                    next: pair[1].label,
                }));
            }
        }
        for pair in wire.names.windows(2) {
            if pair[0].name >= pair[1].name {
                return Err(load::Error::new(MetadataError::NameOrder {
                    previous: pair[0].name.clone(),
                    next: pair[1].name.clone(),
                }));
            }
        }
        let mut labels = BTreeMap::new();
        for value in wire.labels {
            let mut input = Cursor::new(&value.bitmap);
            let members = RoaringBitmap::deserialize_from(&mut input).map_err(load::Error::new)?;
            let remaining = value.bitmap.len() as u64 - input.position();
            if remaining != 0 {
                return Err(load::Error::new(ArtifactError::BitmapTrailingBytes {
                    label: value.label,
                    remaining,
                }));
            }
            labels.insert(
                value.label,
                Label {
                    members,
                    entry_points: value.entry_points,
                },
            );
        }
        Self::validated(
            wire.rows as usize,
            labels,
            wire.names
                .into_iter()
                .map(|value| (value.name, value.label))
                .collect(),
        )
        .map_err(load::Error::new)
    }
}

impl save::Save for LabelMetadata {
    const VERSION: Version = Version::new(0, 0);

    fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
        let bytes = self.encode()?;
        let mut writer = context.write(Some("labels.bin"))?;
        writer.write_all(&bytes).map_err(save::Error::new)?;
        let artifact = &writer.finish()?;
        let encoding = ENCODING;
        let byte_len = &(bytes.len() as u64);
        let digest = &Sha256::digest(&bytes).to_vec();
        Ok(
            diskann_record::save_fields!(context, [encoding, artifact, byte_len, digest])
                .into_owned(),
        )
    }
}

impl load::Load<'_> for LabelMetadata {
    const VERSION: Version = Version::new(0, 0);

    fn load(object: load::Object<'_>) -> load::Result<Self> {
        diskann_record::load_fields!(object, [
            encoding: String,
            artifact: save::Handle,
            byte_len: u64,
            digest: Vec<u8>,
        ]);
        if encoding != ENCODING {
            return Err(load::Error::new(ArtifactError::Encoding(encoding)));
        }
        if byte_len == 0 || byte_len > MAX_ARTIFACT_BYTES {
            return Err(load::Error::new(ArtifactError::LengthLimit(byte_len)));
        }
        if digest.len() != 32 {
            return Err(load::Error::new(ArtifactError::DigestLength(digest.len())));
        }
        let mut bytes = Vec::new();
        object
            .read(&artifact)?
            .take(byte_len + 1)
            .read_to_end(&mut bytes)
            .map_err(load::Error::new)?;
        if bytes.len() as u64 != byte_len {
            return Err(load::Error::new(ArtifactError::Length {
                expected: byte_len,
                actual: bytes.len() as u64,
            }));
        }
        if Sha256::digest(&bytes).as_slice() != digest.as_slice() {
            return Err(load::Error::new(ArtifactError::Checksum));
        }
        Self::decode(&bytes)
    }

    fn load_legacy(_: load::Object<'_>) -> load::Result<Self> {
        Err(load::error::Kind::UnknownVersion.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{error::Error, fs};

    use diskann_utils::views::rowmajor;

    use crate::build::filtered::build_stitched;

    fn metadata() -> LabelMetadata {
        LabelMetadata::validated(
            4,
            BTreeMap::from([
                (
                    7,
                    Label {
                        members: [0, 2].into_iter().collect(),
                        entry_points: vec![0],
                    },
                ),
                (
                    19,
                    Label {
                        members: [1, 2, 3].into_iter().collect(),
                        entry_points: vec![2],
                    },
                ),
            ]),
            BTreeMap::from([("/label/雪\"\n\\".into(), 7), ("a,b".into(), 19)]),
        )
        .unwrap()
    }

    fn assert_membership(metadata: &LabelMetadata) {
        assert_eq!(metadata.rows(), 4);
        assert_eq!(
            metadata.posting(7).unwrap().iter().collect::<Vec<_>>(),
            [0, 2]
        );
        assert_eq!(
            metadata.posting(19).unwrap().iter().collect::<Vec<_>>(),
            [1, 2, 3]
        );
        assert_eq!(metadata.entry_points(7), [0]);
        assert_eq!(metadata.entry_points(19), [2]);
        assert!(metadata.contains(7, 0));
        assert!(!metadata.contains(7, 1));
        assert!(!metadata.contains(7, u32::MAX));
        assert!(!metadata.contains(99, 0));
        assert!(metadata.entry_points(99).is_empty());
        assert!(metadata.posting(99).is_none());
        assert_eq!(metadata.resolve_label("/label/雪\"\n\\"), Some(7));
        assert_eq!(metadata.resolve_label("a,b"), Some(19));
        assert_eq!(metadata.resolve_label("missing"), None);
    }

    #[tokio::test]
    async fn roundtrip_remains_independent_of_construction_adjacency() {
        let values = [0f32, 1., 2., 3.];
        let graph = build_stitched(
            rowmajor::Ref::column_vector(&values),
            [(7, &[0, 2][..]), (19, &[1, 2, 3][..])],
            &Default::default(),
        )
        .await
        .unwrap();
        assert!(matches!(
            LabelMetadata::from_graph(&graph, &BTreeMap::from([("unknown".into(), 99)])),
            Err(MetadataError::UnknownName { label: 99, .. })
        ));
        let metadata = LabelMetadata::from_graph(&graph, &metadata().names).unwrap();
        drop(graph);
        let directory = tempfile::tempdir().unwrap();
        let manifest = directory.path().join("labels.json");
        metadata.save(directory.path(), &manifest).unwrap();
        drop(metadata);
        assert_membership(&LabelMetadata::load(&manifest, directory.path()).unwrap());
    }

    fn saved() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let manifest = directory.path().join("labels.json");
        metadata().save(directory.path(), &manifest).unwrap();
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
        let artifact = directory
            .path()
            .join(value["value"]["artifact"]["$handle"].as_str().unwrap());
        (directory, manifest, artifact)
    }

    #[test]
    fn corruption_truncation_and_zero_tails_are_rejected() {
        let (directory, manifest, artifact) = saved();
        let original = fs::read(&artifact).unwrap();
        for cut in 0..original.len() {
            fs::write(&artifact, &original[..cut]).unwrap();
            assert!(LabelMetadata::load(&manifest, directory.path()).is_err());
        }
        for extra in [1, 24, 32, 256] {
            let mut corrupt = original.clone();
            corrupt.resize(corrupt.len() + extra, 0);
            fs::write(&artifact, corrupt).unwrap();
            assert!(LabelMetadata::load(&manifest, directory.path()).is_err());
        }
        let mut corrupt = original.clone();
        corrupt[0] ^= 1;
        fs::write(&artifact, corrupt).unwrap();
        let error = LabelMetadata::load(&manifest, directory.path()).unwrap_err();
        assert!(error.source().unwrap().is::<ArtifactError>());
        assert!(error.to_string().contains("SHA-256"));
        fs::write(&artifact, original).unwrap();
        assert_membership(&LabelMetadata::load(&manifest, directory.path()).unwrap());
    }

    #[test]
    fn unknown_versions_encodings_and_unbounded_lengths_are_rejected() {
        for (field, replacement) in [
            ("$version", serde_json::json!("1.0")),
            ("encoding", serde_json::json!("other")),
            ("byte_len", serde_json::json!(MAX_ARTIFACT_BYTES + 1)),
            ("byte_len", serde_json::json!(0)),
            ("digest", serde_json::json!([1, 2, 3])),
        ] {
            let (directory, manifest, _) = saved();
            let mut value: serde_json::Value =
                serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
            value["value"][field] = replacement;
            fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();
            assert!(LabelMetadata::load(&manifest, directory.path()).is_err());
        }
    }

    fn wire() -> WireMetadata {
        codec(MAX_ARTIFACT_BYTES)
            .deserialize(&metadata().encode().unwrap())
            .unwrap()
    }

    fn rejects(wire: WireMetadata) {
        let bytes = codec(MAX_ARTIFACT_BYTES).serialize(&wire).unwrap();
        assert!(LabelMetadata::decode(&bytes).is_err());
    }

    #[test]
    fn decoding_rejects_ambiguous_or_invalid_membership_metadata() {
        let mut value = wire();
        value.labels[1].label = value.labels[0].label;
        rejects(value);
        let mut value = wire();
        value.labels.reverse();
        rejects(value);
        let mut value = wire();
        value.names[1].name = value.names[0].name.clone();
        rejects(value);
        let mut value = wire();
        value.names.reverse();
        rejects(value);
        let mut value = wire();
        value.names[0].label = 99;
        rejects(value);
        let mut value = wire();
        value.labels[0].entry_points = vec![1];
        rejects(value);
        let mut value = wire();
        value.labels[0].entry_points = vec![0, 0];
        rejects(value);
        let mut value = wire();
        value.labels[0].entry_points.clear();
        rejects(value);
        for rows in [0, 3, u32::MAX] {
            let mut value = wire();
            value.rows = rows;
            rejects(value);
        }
        let mut value = wire();
        value.labels[0].bitmap.clear();
        RoaringBitmap::new()
            .serialize_into(&mut value.labels[0].bitmap)
            .unwrap();
        rejects(value);
    }

    #[test]
    fn trailing_binary_and_bitmap_bytes_are_rejected_even_with_a_valid_checksum() {
        let mut value = wire();
        value.labels[0].bitmap.push(0);
        rejects(value);
        let (directory, manifest, artifact) = saved();
        let mut bytes = fs::read(&artifact).unwrap();
        bytes.extend_from_slice(&[0; 24]);
        fs::write(&artifact, &bytes).unwrap();
        let mut record: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
        record["value"]["byte_len"] = serde_json::json!(bytes.len());
        record["value"]["digest"] = serde_json::json!(Sha256::digest(&bytes).to_vec());
        fs::write(&manifest, serde_json::to_vec(&record).unwrap()).unwrap();
        assert!(LabelMetadata::load(&manifest, directory.path()).is_err());
    }
}
