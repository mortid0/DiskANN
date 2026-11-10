/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The suffixes and delimiters here follow cpp_main's PQFlashIndex::load.

use std::{
    fs::{self, File},
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use diskann_utils::{
    io::{read_bin, write_bin},
    views::rowmajor::{self, Matrix},
};

use super::LabelMetadata;

fn sidecar(index: &Path, suffix: &str) -> PathBuf {
    let mut name = index.as_os_str().to_os_string();
    name.push(suffix);
    name.into()
}

fn invalid(error: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
}

fn optional(path: &Path) -> io::Result<Option<File>> {
    match File::open(path) {
        Ok(file) => Ok(Some(file)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn lines(path: &Path) -> io::Result<Option<io::Lines<BufReader<File>>>> {
    Ok(optional(path)?.map(|file| BufReader::new(file).lines()))
}

fn number(value: &str) -> io::Result<u32> {
    value.trim().parse().map_err(invalid)
}

fn csv(line: &str) -> io::Result<Vec<u32>> {
    // C++ writers may include a trailing comma. Interior empty fields are invalid.
    let line = line.trim().trim_end_matches(',');
    if line.is_empty() {
        return Ok(Vec::new());
    }
    line.split(',').map(number).collect()
}

impl LabelMetadata {
    /// Read sidecars appended to the full disk filename, e.g. `index_disk.index`.
    /// Missing labels are allowed for an unfiltered index. Present files are checked
    /// against the graph shape; malformed optional files are errors, not ignored.
    pub fn load_cpp(index: impl AsRef<Path>, points: usize, dimensions: usize) -> io::Result<Self> {
        let index = index.as_ref();
        let mut postings = std::collections::BTreeMap::<u32, Vec<u32>>::new();
        if let Some(rows) = lines(&sidecar(index, "_labels.txt"))? {
            let mut count = 0;
            for (id, row) in rows.enumerate() {
                if id >= points {
                    return Err(invalid("too many label rows"));
                }
                count += 1;
                let mut labels = csv(&row?)?;
                labels.sort_unstable();
                labels.dedup();
                for label in labels {
                    postings.entry(label).or_default().push(id as u32);
                }
            }
            if count != points {
                return Err(invalid("label row count does not match graph"));
            }
        }
        // Construct CSR first, then validate the actual entries after loading universal labels.
        let entries = postings
            .iter()
            .map(|(&label, ids)| (label, vec![ids[0]]))
            .collect();
        let mut metadata = Self::from_postings(points, postings, entries).map_err(invalid)?;
        metadata.entry_points.clear();
        metadata.label_map.clear();
        if let Some(rows) = lines(&sidecar(index, "_labels_map.txt"))? {
            for row in rows {
                let row = row?;
                let (name, id) = row
                    .split_once('\t')
                    .ok_or_else(|| invalid("label map requires name<TAB>id"))?;
                if name.is_empty()
                    || metadata
                        .label_map
                        .insert(name.to_owned(), number(id)?)
                        .is_some()
                {
                    return Err(invalid("empty or duplicate label name"));
                }
            }
        }
        if let Some(rows) = lines(&sidecar(index, "_universal_label.txt"))? {
            metadata.universal_label =
                Some(number(&rows.collect::<io::Result<Vec<_>>>()?.join("\n"))?);
        }
        if let Some(rows) = lines(&sidecar(index, "_labels_to_medoids.txt"))? {
            for row in rows {
                let row = row?;
                let values = csv(&row)?;
                if values.len() < 2
                    || metadata
                        .entry_points
                        .insert(values[0], values[1..].to_vec())
                        .is_some()
                {
                    return Err(invalid("invalid or duplicate label entry-point row"));
                }
            }
        }
        if let Some(rows) = lines(&sidecar(index, "_dummy_map.txt"))? {
            for row in rows {
                let row = row?;
                let values = csv(&row)?;
                if values.len() != 2
                    || values.iter().any(|&id| id as usize >= points)
                    || values[0] == values[1]
                    || metadata.dummy_map.insert(values[0], values[1]).is_some()
                {
                    return Err(invalid("invalid or duplicate dummy mapping"));
                }
            }
            if metadata
                .dummy_map
                .values()
                .any(|id| metadata.dummy_map.contains_key(id))
            {
                return Err(invalid("dummy mapping must target a real point"));
            }
        }
        if let Some(mut file) = optional(&sidecar(index, "_medoids.bin"))? {
            let medoids = read_bin::<u32>(&mut file).map_err(invalid)?;
            if medoids.ncols() != 1
                || medoids.nrows() == 0
                || medoids.as_slice().iter().any(|&id| id as usize >= points)
            {
                return Err(invalid("medoids must be an m-by-1 matrix of graph IDs"));
            }
            metadata.medoids = medoids.as_slice().to_vec();
        }
        if let Some(mut file) = optional(&sidecar(index, "_centroids.bin"))? {
            let centroids = read_bin::<f32>(&mut file).map_err(invalid)?;
            if centroids.nrows() != metadata.medoids.len()
                || centroids.ncols() != dimensions
                || centroids.as_slice().iter().any(|value| !value.is_finite())
            {
                return Err(invalid(
                    "centroid shape must match medoids and graph dimensions",
                ));
            }
            metadata.centroids = Some(centroids);
        }
        metadata.validate().map_err(invalid)?;
        Ok(metadata)
    }

    /// Write the C++ sidecars. Stable application IDs belong to the optional
    /// update lifecycle and are deliberately not encoded as C++ dummy points.
    /// Empty label rows are a Rust extension; C++ requires a label on every row.
    pub fn save_cpp(&self, index: impl AsRef<Path>) -> io::Result<()> {
        self.validate().map_err(invalid)?;
        let index = index.as_ref();
        let mut labels = BufWriter::new(File::create(sidecar(index, "_labels.txt"))?);
        for id in 0..self.point_count {
            write_csv(&mut labels, self.labels(id).iter().copied())?;
        }
        labels.flush()?;
        let mut entries = BufWriter::new(File::create(sidecar(index, "_labels_to_medoids.txt"))?);
        for (&label, ids) in &self.entry_points {
            write_csv(
                &mut entries,
                std::iter::once(label).chain(ids.iter().copied()),
            )?;
        }
        entries.flush()?;
        let mut names = BufWriter::new(File::create(sidecar(index, "_labels_map.txt"))?);
        for (name, id) in &self.label_map {
            writeln!(names, "{name}\t{id}")?;
        }
        names.flush()?;
        write_optional(
            index,
            "_universal_label.txt",
            self.universal_label.is_some(),
            |file| writeln!(file, "{}", self.universal_label.unwrap()),
        )?;
        write_optional(
            index,
            "_dummy_map.txt",
            !self.dummy_map.is_empty(),
            |file| {
                for (dummy, real) in &self.dummy_map {
                    writeln!(file, "{dummy},{real}")?;
                }
                Ok(())
            },
        )?;
        write_optional(index, "_medoids.bin", !self.medoids.is_empty(), |file| {
            let matrix =
                rowmajor::Ref::try_from_data(self.medoids.as_slice(), self.medoids.len(), 1)
                    .map_err(invalid)?;
            write_bin(matrix, file).map_err(invalid)?;
            Ok(())
        })?;
        write_optional(index, "_centroids.bin", self.centroids.is_some(), |file| {
            write_bin(self.centroids.as_ref().unwrap().as_view(), file).map_err(invalid)?;
            Ok(())
        })
    }
}

fn write_csv(writer: &mut impl Write, values: impl Iterator<Item = u32>) -> io::Result<()> {
    for (index, value) in values.enumerate() {
        if index != 0 {
            write!(writer, ",")?;
        }
        write!(writer, "{value}")?;
    }
    writeln!(writer)
}

fn write_optional(
    index: &Path,
    suffix: &str,
    present: bool,
    write: impl FnOnce(&mut BufWriter<File>) -> io::Result<()>,
) -> io::Result<()> {
    let path = sidecar(index, suffix);
    if present {
        let mut file = BufWriter::new(File::create(path)?);
        write(&mut file)?;
        file.flush()
    } else {
        match fs::remove_file(path) {
            Err(error) if error.kind() != io::ErrorKind::NotFound => Err(error),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpp_sidecars_roundtrip_and_universal_dummy_semantics() {
        let dir = tempfile::tempdir().unwrap();
        let index = dir.path().join("index_disk.index");
        for (suffix, contents) in [
            ("_labels.txt", "7,9,\r\n0\r\n7\r\n"),
            ("_labels_map.txt", "sports\t7\nnews\t9\nall\t0\n"),
            ("_labels_to_medoids.txt", "7,2,1,\n9,0\n0,1\n"),
            ("_universal_label.txt", "0\n"),
            ("_dummy_map.txt", "2,0\n"),
        ] {
            fs::write(sidecar(&index, suffix), contents).unwrap();
        }
        // Literal C++ bin headers, independent of the Rust writer.
        fs::write(
            sidecar(&index, "_medoids.bin"),
            [2u32, 1, 0, 1]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut centroids = [2u32, 2]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        centroids.extend([1f32, 2., 3., 4.].into_iter().flat_map(f32::to_le_bytes));
        fs::write(sidecar(&index, "_centroids.bin"), centroids).unwrap();
        let loaded = LabelMetadata::load_cpp(&index, 3, 2).unwrap();
        assert_eq!(loaded.labels(0), &[7, 9]);
        assert_eq!(loaded.entries(7), &[2, 1]);
        assert_eq!(loaded.resolve_label("sports").unwrap(), 7);
        assert_eq!(loaded.resolve_label("unknown").unwrap(), 0);
        assert!(loaded.matches(9, 1));
        assert!(!loaded.matches(0, 0));
        assert_eq!(loaded.real_id(2), 0);
        assert_eq!(loaded.medoids(), &[0, 1]);
        assert_eq!(loaded.centroids().unwrap().as_slice(), &[1., 2., 3., 4.]);
        let output = dir.path().join("copy_disk.index");
        loaded.save_cpp(&output).unwrap();
        assert_eq!(
            fs::read_to_string(sidecar(&output, "_labels.txt")).unwrap(),
            "7,9\n0\n7\n"
        );
        assert_eq!(
            fs::read_to_string(sidecar(&output, "_labels_map.txt")).unwrap(),
            "all\t0\nnews\t9\nsports\t7\n"
        );
        let copy = LabelMetadata::load_cpp(output, 3, 2).unwrap();
        assert_eq!(copy.entries(7), &[2, 1]);
        assert_eq!(copy.real_id(2), 0);
        assert_eq!(copy.centroids().unwrap().as_slice(), &[1., 2., 3., 4.]);
    }

    #[test]
    fn rejects_bad_shapes_ids_and_missing_label_entries() {
        let dir = tempfile::tempdir().unwrap();
        let index = dir.path().join("index_disk.index");
        fs::write(sidecar(&index, "_labels.txt"), "7\n7\n").unwrap();
        assert!(LabelMetadata::load_cpp(&index, 2, 2).is_err());
        fs::write(sidecar(&index, "_labels_to_medoids.txt"), "7,2\n").unwrap();
        assert!(LabelMetadata::load_cpp(&index, 2, 2).is_err());
        fs::write(sidecar(&index, "_labels_to_medoids.txt"), "7,1\n").unwrap();
        let valid = LabelMetadata::load_cpp(&index, 2, 2).unwrap();
        assert!(valid.resolve_label("missing").is_err());
        assert!(LabelMetadata::load_cpp(&index, 3, 2).is_err());
        fs::write(sidecar(&index, "_dummy_map.txt"), "0,1\n1,0\n").unwrap();
        assert!(LabelMetadata::load_cpp(&index, 2, 2).is_err());
        fs::remove_file(sidecar(&index, "_dummy_map.txt")).unwrap();
        fs::write(
            sidecar(&index, "_medoids.bin"),
            [1u32, 2, 0, 1]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(LabelMetadata::load_cpp(&index, 2, 2).is_err());
    }
}
