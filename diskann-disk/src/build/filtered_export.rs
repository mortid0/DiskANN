/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Export experimental stitched graphs through the standard disk serializers.

use diskann::{ANNResult, error::ErrorContext};
use diskann_providers::{
    model::graph::provider::async_::SimpleNeighborProviderAsync,
    storage::{StorageReadProvider, StorageWriteProvider, get_mem_index_file},
};
use diskann_utils::io::Metadata;
use thiserror::Error;

use super::{disk_index_build_parameter::DISK_SECTOR_LEN, filtered::StitchedGraph};
use crate::{data_model::AdHoc, storage::DiskIndexWriter};

#[derive(Debug, Error)]
enum ExportError {
    #[error("a disk graph needs at least two rows, got {0}")]
    TooFewRows(usize),
    #[error("stitched graph has {graph_rows} rows, but vector data has {vector_rows}")]
    RowCount {
        graph_rows: usize,
        vector_rows: usize,
    },
    #[error("vector data must have nonzero dimensions")]
    EmptyDimension,
    #[error("row {0} has no neighbors and cannot be exported to the disk graph format")]
    EmptyAdjacency(usize),
}

diskann::convert_error!(ExportError);

/// Write the canonical memory graph and disk layout for an f32 stitched dataset.
///
/// `data_path` must contain the original row-ordered vectors in DiskANN's binary
/// matrix format. PQ data and label metadata are separate artifacts. The ordinary
/// disk entry point is row zero; filtered searches use the saved label entry points.
///
/// # Errors
/// Rejects fewer than two rows, a graph/vector row-count mismatch, zero dimensions,
/// and empty adjacency lists before writing. Propagates storage and serializer errors;
/// a failed write can leave partial output and does not publish a complete index.
pub fn write_stitched(
    graph: &StitchedGraph,
    data_path: &str,
    index_prefix: &str,
    storage: &(impl StorageReadProvider + StorageWriteProvider),
) -> ANNResult<()> {
    let rows = graph.neighbors().len();
    if rows < 2 {
        return Err(ExportError::TooFewRows(rows).into());
    }
    let metadata = Metadata::read(
        &mut storage
            .open_reader(data_path)
            .with_context(|| format!("opening vector data {data_path}"))?,
    )
    .with_context(|| format!("reading vector metadata from {data_path}"))?;
    if metadata.npoints() != rows {
        return Err(ExportError::RowCount {
            graph_rows: rows,
            vector_rows: metadata.npoints(),
        }
        .into());
    }
    if metadata.ndims() == 0 {
        return Err(ExportError::EmptyDimension.into());
    }
    if let Some(row) = graph.neighbors().iter().position(Vec::is_empty) {
        return Err(ExportError::EmptyAdjacency(row).into());
    }

    let neighbors = SimpleNeighborProviderAsync::new(rows, 0, graph.max_degree(), 1.0);
    for (row, adjacency) in graph.neighbors().iter().enumerate() {
        neighbors.set_neighbors_sync(row, adjacency)?;
    }
    neighbors.save_direct(storage, 0, &get_mem_index_file(index_prefix))?;
    DiskIndexWriter::new(data_path.into(), index_prefix.into(), None, DISK_SECTOR_LEN)?
        .create_disk_layout::<AdHoc<f32>, _>(storage)
}

#[cfg(test)]
mod tests {
    use diskann_providers::storage::{VirtualStorageProvider, get_disk_index_file};
    use diskann_utils::{
        io::{read_bin, write_bin},
        views::rowmajor::{self, Matrix},
    };

    use super::*;
    use crate::{build::filtered::build_stitched, data_model::GraphHeader};

    const DATA: &str = "/vectors.bin";
    const PREFIX: &str = "/stitched";

    fn write_vectors(storage: &impl StorageWriteProvider, rows: usize) -> rowmajor::Owned<f32> {
        let data = rowmajor::Owned::column_vector((0..rows).map(|row| row as f32).collect());
        write_bin(data.as_view(), &mut storage.create_for_write(DATA).unwrap()).unwrap();
        data
    }

    #[tokio::test]
    async fn exports_unlabeled_singleton_and_mixed_rows_with_canonical_headers() {
        let cases = [
            vec![],
            vec![(1, vec![0]), (2, vec![1]), (3, vec![2]), (4, vec![3])],
            vec![(1, vec![0, 1]), (2, vec![2])],
        ];
        for postings in cases {
            let storage = VirtualStorageProvider::new_memory();
            let data = write_vectors(&storage, 4);
            let graph = build_stitched(
                data.as_view(),
                postings
                    .iter()
                    .map(|(label, rows)| (*label, rows.as_slice())),
                &Default::default(),
            )
            .await
            .unwrap();
            write_stitched(&graph, DATA, PREFIX, &storage).unwrap();
            let bytes =
                read_bin::<u8>(&mut storage.open_reader(&get_disk_index_file(PREFIX)).unwrap())
                    .unwrap();
            let header = GraphHeader::try_from(bytes.as_slice()).unwrap();
            assert_eq!(header.metadata().num_pts, 4);
            assert_eq!(header.metadata().dims, 1);
            assert_eq!(header.metadata().vamana_frozen_num, 0);
            assert_eq!(header.block_size(), DISK_SECTOR_LEN as u64);
            assert!(graph.neighbors().iter().enumerate().all(|(id, neighbors)| {
                !neighbors.is_empty() && !neighbors.contains(&(id as u32))
            }));
        }
    }

    #[tokio::test]
    async fn rejects_one_row_and_mismatched_vector_rows_before_writing() {
        for (rows, vector_rows) in [(1, 1), (4, 3)] {
            let storage = VirtualStorageProvider::new_memory();
            let data = write_vectors(&storage, rows);
            let graph = build_stitched(data.as_view(), [], &Default::default())
                .await
                .unwrap();
            write_vectors(&storage, vector_rows);
            let error = write_stitched(&graph, DATA, PREFIX, &storage).unwrap_err();
            let cause = error.downcast_ref::<ExportError>().unwrap();
            if rows == 1 {
                assert!(matches!(cause, ExportError::TooFewRows(1)));
            } else {
                assert!(matches!(
                    cause,
                    ExportError::RowCount {
                        graph_rows: 4,
                        vector_rows: 3
                    }
                ));
            }
            assert!(!storage.exists(&get_mem_index_file(PREFIX)));
            assert!(!storage.exists(&get_disk_index_file(PREFIX)));
        }
    }

    #[tokio::test]
    async fn rejects_zero_dimensions_and_preserves_storage_errors() {
        let storage = VirtualStorageProvider::new_memory();
        let data = write_vectors(&storage, 4);
        let graph = build_stitched(data.as_view(), [], &Default::default())
            .await
            .unwrap();
        Metadata::new(4, 0)
            .unwrap()
            .write(&mut storage.create_for_write(DATA).unwrap())
            .unwrap();
        let error = write_stitched(&graph, DATA, PREFIX, &storage).unwrap_err();
        assert!(matches!(
            error.downcast_ref::<ExportError>(),
            Some(ExportError::EmptyDimension)
        ));
        storage.delete(DATA).unwrap();
        let error = write_stitched(&graph, DATA, PREFIX, &storage).unwrap_err();
        assert!(error.downcast_ref::<std::io::Error>().is_some());
        assert!(!storage.exists(&get_mem_index_file(PREFIX)));
    }
}
