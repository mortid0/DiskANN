/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fs::File, os::unix::fs::FileExt, path::Path};

use diskann::ANNResult;

use crate::search::provider::aligned_file_reader::{traits::AlignedFileReader, AlignedRead, A512};

/// Blocking positional reads without loading the entire index into memory.
pub struct PositionalAlignedFileReader(File);

impl PositionalAlignedFileReader {
    /// Open a file using the operating system's page cache.
    pub fn new(path: impl AsRef<Path>) -> ANNResult<Self> {
        Ok(Self(File::open(path)?))
    }

    /// Open a macOS file with page-cache bypass enabled.
    #[cfg(all(not(miri), target_os = "macos"))]
    pub fn uncached(path: impl AsRef<Path>) -> ANNResult<Self> {
        use std::os::fd::AsRawFd;
        let reader = Self::new(path)?;
        // SAFETY: fcntl receives a valid owned descriptor and the integer argument
        // required by F_NOCACHE; ownership is retained and no memory is accessed.
        if unsafe { libc::fcntl(reader.0.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(reader)
    }
}

impl AlignedFileReader for PositionalAlignedFileReader {
    type Alignment = A512;

    fn read(&mut self, requests: &mut [AlignedRead<u8, A512>]) -> ANNResult<()> {
        for request in requests {
            let offset = request.offset();
            self.0.read_exact_at(request.aligned_buf_mut(), offset)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[repr(align(512))]
    struct Buffer([u8; 512]);

    #[test]
    fn positional_reads_support_out_of_order_requests_and_report_eof() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&[7; 512]).unwrap();
        file.write_all(&[9; 512]).unwrap();
        let mut reader = PositionalAlignedFileReader::new(file.path()).unwrap();
        let mut first = Buffer([0; 512]);
        let mut second = Buffer([0; 512]);
        reader
            .read(&mut [
                AlignedRead::new(512, &mut second.0).unwrap(),
                AlignedRead::new(0, &mut first.0).unwrap(),
            ])
            .unwrap();
        assert_eq!(first.0, [7; 512]);
        assert_eq!(second.0, [9; 512]);
        assert!(reader
            .read(&mut [AlignedRead::new(1024, &mut first.0).unwrap()])
            .is_err());
        #[cfg(all(not(miri), target_os = "macos"))]
        PositionalAlignedFileReader::uncached(file.path())
            .unwrap()
            .read(&mut [AlignedRead::new(0, &mut first.0).unwrap()])
            .unwrap();
    }
}
