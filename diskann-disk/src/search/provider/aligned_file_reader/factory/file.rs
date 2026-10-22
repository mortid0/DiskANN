/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;

#[cfg(all(not(miri), target_os = "macos"))]
use crate::search::provider::aligned_file_reader::reader::PositionalAlignedFileReader;

#[cfg(all(not(miri), target_os = "linux"))]
use crate::search::provider::aligned_file_reader::reader::LinuxAlignedFileReader;
#[cfg(miri)]
use crate::search::provider::aligned_file_reader::reader::StorageProviderAlignedFileReader;
#[cfg(all(not(miri), target_os = "windows"))]
use crate::search::provider::aligned_file_reader::reader::WindowsAlignedFileReader;
use crate::search::provider::aligned_file_reader::traits::AlignedReaderFactory;

#[cfg(miri)]
use diskann_providers::storage::FileStorageProvider;

pub struct AlignedFileReaderFactory {
    pub file_path: String,
}

impl AlignedReaderFactory for AlignedFileReaderFactory {
    // Miri cannot execute native I/O syscalls used by the platform readers.
    #[cfg(miri)]
    type AlignedReaderType = StorageProviderAlignedFileReader;

    #[cfg(all(not(miri), target_os = "linux"))]
    type AlignedReaderType = LinuxAlignedFileReader;

    #[cfg(all(not(miri), target_os = "windows"))]
    type AlignedReaderType = WindowsAlignedFileReader;

    #[cfg(all(not(miri), target_os = "macos"))]
    type AlignedReaderType = PositionalAlignedFileReader;

    fn build(&self) -> ANNResult<Self::AlignedReaderType> {
        #[cfg(all(not(miri), target_os = "macos"))]
        return PositionalAlignedFileReader::uncached(&self.file_path);

        #[cfg(miri)]
        return StorageProviderAlignedFileReader::new(
            &FileStorageProvider,
            self.file_path.as_str(),
        );

        #[cfg(all(not(miri), target_os = "windows"))]
        return WindowsAlignedFileReader::new(self.file_path.as_str());

        #[cfg(all(not(miri), target_os = "linux"))]
        return LinuxAlignedFileReader::new(self.file_path.as_str());
    }
}

impl AlignedFileReaderFactory {
    pub fn new(file_path: String) -> Self {
        Self { file_path }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_file_reader_factory_new() {
        let path = "/tmp/test.bin".to_string();
        let factory = AlignedFileReaderFactory::new(path.clone());
        assert_eq!(factory.file_path, path);
    }
}
