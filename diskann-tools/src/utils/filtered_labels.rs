/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! String label membership shared by experimental filtered construction tools.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};

use diskann_label_filter::{
    attribute::AttributeValue,
    kv_index::GenericIndex,
    parser::format::Document,
    read_baselabels,
    stores::bftree_store::BfTreeStore,
    traits::{
        inverted_index_trait::InvertedIndexProvider,
        key_codec::DefaultKeyCodec,
        posting_list_trait::{PostingList, PostingListAccessor, RoaringPostingList},
    },
    Attributes,
};

/// Deterministic label IDs and deduplicated, sorted external vector-row memberships.
#[derive(Debug)]
pub struct LabelData {
    names: BTreeMap<String, u32>,
    postings: BTreeMap<u32, Vec<u32>>,
}

impl LabelData {
    pub fn names(&self) -> &BTreeMap<String, u32> {
        &self.names
    }

    pub fn postings(&self) -> &BTreeMap<u32, Vec<u32>> {
        &self.postings
    }

    pub fn into_parts(self) -> (BTreeMap<String, u32>, BTreeMap<u32, Vec<u32>>) {
        (self.names, self.postings)
    }
}

/// Read the existing JSON-lines label format, selecting a string or string array at
/// `label_field` (an RFC 6901 JSON pointer). Missing fields and empty arrays mean no labels.
/// An array is interpreted as membership in every listed label, independent of position.
///
/// # Errors
/// Returns an error for malformed data, non-string labels, invalid document IDs, a
/// document count different from `rows`, or storage errors from the inverted index.
pub fn load_labels(
    path: impl AsRef<Path>,
    rows: usize,
    label_field: &str,
) -> anyhow::Result<LabelData> {
    labels_from_documents(read_baselabels(path)?, rows, label_field)
}

fn labels_from_documents(
    documents: Vec<Document>,
    rows: usize,
    field: &str,
) -> anyhow::Result<LabelData> {
    anyhow::ensure!(field.starts_with('/'), "label_field must be a JSON pointer");
    for token in field.split('/').skip(1) {
        let mut chars = token.chars();
        while let Some(c) = chars.next() {
            if c == '~' {
                anyhow::ensure!(
                    matches!(chars.next(), Some('0' | '1')),
                    "invalid escape in label_field"
                );
            }
        }
    }
    anyhow::ensure!(
        documents.len() == rows,
        "one metadata document is required per vector"
    );
    anyhow::ensure!(rows <= u32::MAX as usize, "too many document rows");
    let store = Arc::new(BfTreeStore::memory()?);
    let mut inverted = GenericIndex::<BfTreeStore, RoaringPostingList, DefaultKeyCodec>::new(store);
    // The shared parser uses distinct keys for array elements. Union those postings by
    // string value, so position never changes membership and duplicate labels disappear.
    let mut value_fields: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for document in documents {
        anyhow::ensure!(
            document.doc_id < rows && seen.insert(document.doc_id),
            "duplicate or out-of-range doc_id {}",
            document.doc_id
        );
        let value = document.label.pointer(field);
        let mut attributes = Attributes::new();
        let mut insert = |key: String, value: &serde_json::Value| -> anyhow::Result<()> {
            let name = value.as_str().ok_or_else(|| {
                anyhow::anyhow!(
                    "label_field {field} of doc_id {} must contain a string or an array of strings",
                    document.doc_id
                )
            })?;
            value_fields
                .entry(name.to_owned())
                .or_default()
                .insert(key.clone());
            attributes.insert(key, AttributeValue::String(name.to_owned()));
            Ok(())
        };
        match value {
            None => {}
            Some(serde_json::Value::Array(values)) => {
                for (position, value) in values.iter().enumerate() {
                    insert(format!("{field}/{position}"), value)?;
                }
            }
            Some(value) => insert(field.to_owned(), value)?,
        }
        inverted.insert(document.doc_id, &attributes)?;
    }
    let mut names = BTreeMap::new();
    let mut postings = BTreeMap::new();
    for (name, fields) in value_fields {
        let label = u32::try_from(names.len())?;
        anyhow::ensure!(label < u32::MAX, "too many labels");
        let mut members = RoaringPostingList::empty();
        for key in fields {
            let found = inverted
                .get_posting_list(&key, &AttributeValue::String(name.clone()))?
                .ok_or_else(|| anyhow::anyhow!("missing posting list for {name}"))?;
            members = members.union(&found);
        }
        postings.insert(label, members.rb.iter().collect());
        names.insert(name, label);
    }
    Ok(LabelData { names, postings })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn documents(values: Vec<serde_json::Value>) -> Vec<Document> {
        values
            .into_iter()
            .enumerate()
            .map(|(doc_id, label)| Document { doc_id, label })
            .collect()
    }

    #[test]
    fn combines_scalar_and_array_labels_and_preserves_escapes() {
        let docs = documents(vec![
            json!({"a/b~": "red\n\"escaped\""}),
            json!({"a/b~": ["blue", "red\n\"escaped\"", "blue"]}),
            json!({"a/b~": ["red\n\"escaped\"", "blue"]}),
            json!({"a/b~": []}),
            json!({}),
        ]);
        let labels = labels_from_documents(docs, 5, "/a~1b~0").unwrap();
        assert_eq!(labels.postings[&labels.names["blue"]], [1, 2]);
        assert_eq!(
            labels.postings[&labels.names["red\n\"escaped\""]],
            [0, 1, 2]
        );
    }

    #[test]
    fn rejects_malformed_labels_and_document_ids() {
        for value in [
            json!(12),
            json!(null),
            json!({"nested":"red"}),
            json!(["red", 1]),
            json!([["red"]]),
        ] {
            let error = labels_from_documents(documents(vec![json!({"label":value})]), 1, "/label")
                .unwrap_err();
            assert!(error.to_string().contains("doc_id 0"));
        }
        let duplicate = vec![
            Document {
                doc_id: 0,
                label: json!({})
            };
            2
        ];
        assert!(labels_from_documents(duplicate, 2, "/label").is_err());
        let outside = vec![Document {
            doc_id: 1,
            label: json!({}),
        }];
        assert!(labels_from_documents(outside, 1, "/label").is_err());
        assert!(labels_from_documents(vec![], 1, "/label").is_err());
        for pointer in ["label", "/label~", "/label~2"] {
            assert!(labels_from_documents(vec![], 0, pointer).is_err());
        }
    }
}
