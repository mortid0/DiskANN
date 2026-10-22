/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Same-artifact filtered-search experiment. See tools/single-hop-benchmark/README.md.

#[cfg(unix)]
mod experiment {
    use std::{
        collections::{BTreeMap, HashMap},
        fs::{File, OpenOptions},
        io::{Read, Write},
        os::unix::fs::FileExt,
        path::{Path, PathBuf},
        sync::{
            atomic::{AtomicU64, AtomicUsize, Ordering},
            Arc,
        },
        time::Instant,
    };

    #[cfg(target_os = "macos")]
    use std::os::unix::io::AsRawFd;

    use anyhow::{ensure, Context, Result};
    use diskann::{graph::search::AdaptiveL, ANNResult};
    use diskann_disk::{
        data_model::{AdHoc, CachingStrategy, GraphHeader, GraphLayoutVersion, GraphMetadata},
        filtered::metadata::LabelMetadata,
        search::{
            provider::{
                aligned_file_reader::{
                    traits::{AlignedFileReader, AlignedReaderFactory},
                    AlignedRead,
                },
                disk_provider::{DiskIndexSearcher, SearchResultItem},
                disk_vertex_provider_factory::DiskVertexProviderFactory,
            },
            search_mode::SearchMode,
            traits::VertexProviderFactory,
        },
        storage::disk_index_reader::DiskIndexReader,
    };
    use diskann_providers::storage::FileStorageProvider;
    use diskann_vector::distance::Metric;
    use serde_json::json;

    #[cfg(target_os = "linux")]
    use diskann_disk::search::provider::aligned_file_reader::{
        LinuxAlignedFileReader, A512 as ReaderAlignment,
    };
    #[cfg(not(target_os = "linux"))]
    type ReaderAlignment = diskann_disk::search::provider::aligned_file_reader::A512;

    #[derive(Default)]
    struct Counters {
        requests: AtomicU64,
        nanos: AtomicU64,
        bytes: AtomicU64,
    }

    struct PreadFactory {
        path: PathBuf,
        bypass: bool,
        counters: Arc<Counters>,
    }
    struct PreadReader {
        file: File,
        counters: Arc<Counters>,
        #[cfg(target_os = "linux")]
        native: Option<LinuxAlignedFileReader>,
    }

    impl AlignedReaderFactory for PreadFactory {
        type AlignedReaderType = PreadReader;
        fn build(&self) -> ANNResult<PreadReader> {
            let file = File::open(&self.path)?;
            #[cfg(target_os = "macos")]
            if self.bypass {
                // SAFETY: fcntl receives a live owned file descriptor and the integer argument
                // required by F_NOCACHE. It neither transfers ownership nor accesses memory.
                if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
                    return Err(std::io::Error::last_os_error().into());
                }
            }
            #[cfg(not(any(target_os = "macos", target_os = "linux")))]
            if self.bypass {
                return Err(diskann::ANNError::message(
                    "this prototype's cache bypass requires macOS; use --bypass false elsewhere",
                ));
            }
            Ok(PreadReader {
                file,
                counters: self.counters.clone(),
                #[cfg(target_os = "linux")]
                native: if self.bypass {
                    Some(LinuxAlignedFileReader::new(&self.path.to_string_lossy())?)
                } else {
                    None
                },
            })
        }
    }

    impl AlignedFileReader for PreadReader {
        type Alignment = ReaderAlignment;
        fn read(&mut self, requests: &mut [AlignedRead<u8, ReaderAlignment>]) -> ANNResult<()> {
            let begin = Instant::now();
            #[cfg(target_os = "linux")]
            if let Some(native) = &mut self.native {
                native.read(requests)?;
                self.counters
                    .requests
                    .fetch_add(requests.len() as u64, Ordering::Relaxed);
                self.counters.bytes.fetch_add(
                    requests.iter().map(|r| r.aligned_buf().len() as u64).sum(),
                    Ordering::Relaxed,
                );
                self.counters
                    .nanos
                    .fetch_add(begin.elapsed().as_nanos() as u64, Ordering::Relaxed);
                return Ok(());
            }
            for request in requests {
                let offset = request.offset();
                let size = request.aligned_buf().len() as u64;
                self.file.read_exact_at(request.aligned_buf_mut(), offset)?;
                self.counters.requests.fetch_add(1, Ordering::Relaxed);
                self.counters.bytes.fetch_add(size, Ordering::Relaxed);
            }
            self.counters
                .nanos
                .fetch_add(begin.elapsed().as_nanos() as u64, Ordering::Relaxed);
            Ok(())
        }
    }

    // Native headers use the provider parser. Legacy C++ metadata is translated in memory.
    struct BenchmarkFactory {
        inner: DiskVertexProviderFactory<AdHoc<u8>, PreadFactory>,
        cpp_meta: Option<Vec<u64>>,
    }

    impl VertexProviderFactory<AdHoc<u8>> for BenchmarkFactory {
        type VertexProviderType =
            <DiskVertexProviderFactory<AdHoc<u8>, PreadFactory> as VertexProviderFactory<
                AdHoc<u8>,
            >>::VertexProviderType;
        fn get_header(&self) -> ANNResult<GraphHeader> {
            let Some(m) = &self.cpp_meta else {
                return self.inner.get_header();
            };
            Ok(GraphHeader::new(
                GraphMetadata::new(m[0], m[1] as usize, m[2], m[3], m[4], m[5], m[6], m[8], 0),
                4096,
                GraphLayoutVersion::new(0, 0),
            ))
        }
        fn create_vertex_provider(
            &self,
            max_batch_size: usize,
            header: &GraphHeader,
        ) -> ANNResult<Self::VertexProviderType> {
            self.inner.create_vertex_provider(max_batch_size, header)
        }
    }

    fn metadata_fingerprint(bytes: &[u8]) -> String {
        // This reproducible content fingerprint is an experiment identity, not a security hash.
        let hash = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        });
        format!("fnv1a64:{hash:016x}")
    }

    fn matrix(path: &Path, width: usize) -> Result<(usize, usize, Vec<u8>)> {
        let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
        ensure!(bytes.len() >= 8, "short matrix {}", path.display());
        let n = u32::from_le_bytes(bytes[0..4].try_into()?) as usize;
        let d = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
        ensure!(
            bytes.len() == 8 + n * d * width,
            "invalid matrix size {}",
            path.display()
        );
        Ok((n, d, bytes[8..].to_vec()))
    }

    fn u32_matrix(path: &Path) -> Result<(usize, usize, Vec<u32>)> {
        let (n, d, bytes) = matrix(path, 4)?;
        Ok((
            n,
            d,
            bytes
                .chunks_exact(4)
                .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
                .collect(),
        ))
    }

    fn entries(prefix: &str) -> Result<BTreeMap<u32, Vec<u32>>> {
        let mut raw_to_internal = HashMap::new();
        let map_path = format!("{prefix}_disk.index_labels_map.txt");
        for line in std::fs::read_to_string(&map_path)
            .with_context(|| map_path.clone())?
            .lines()
        {
            let fields: Vec<_> = line
                .split([',', '\t', ' '])
                .filter(|s| !s.is_empty())
                .collect();
            ensure!(fields.len() == 2, "invalid label map row {line}");
            raw_to_internal.insert(fields[1].parse::<u32>()?, fields[0].parse::<u32>()?);
        }
        let mut result = BTreeMap::new();
        for line in
            std::fs::read_to_string(format!("{prefix}_disk.index_labels_to_medoids.txt"))?.lines()
        {
            let fields = line
                .split(',')
                .map(str::trim)
                .map(str::parse::<u32>)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            ensure!(fields.len() >= 2, "label has no medoid");
            let raw = *raw_to_internal
                .get(&fields[0])
                .context("medoid label missing from map")?;
            result.insert(raw, fields[1..].to_vec());
        }
        Ok(result)
    }

    #[derive(Clone)]
    struct Query {
        vector: Vec<u8>,
        label: u32,
        truth: Vec<u32>,
    }
    struct Observation {
        results: Vec<SearchResultItem<()>>,
        micros: f64,
        recall: f64,
        cmps: u32,
        hops: u32,
        returned: usize,
        violations: usize,
        duplicates: usize,
        label: u32,
    }

    fn run_batch<F>(queries: &[Query], threads: usize, execute: &F) -> Result<Vec<Observation>>
    where
        F: Fn(&Query) -> Result<Observation> + Sync,
    {
        let workers = threads.min(queries.len());
        let barrier = std::sync::Barrier::new(workers);
        let next = AtomicUsize::new(0);
        let observations: Vec<_> = (0..queries.len())
            .map(|_| std::sync::OnceLock::new())
            .collect();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(|| -> Result<()> {
                        barrier.wait();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= queries.len() {
                                break;
                            }
                            observations[i]
                                .set(execute(&queries[i])?)
                                .map_err(|_| anyhow::anyhow!("query executed more than once"))?;
                        }
                        Ok(())
                    })
                })
                .collect();
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("worker panicked"))??;
            }
            Ok::<_, anyhow::Error>(())
        })?;
        observations
            .into_iter()
            .map(|slot| slot.into_inner().context("query was not executed"))
            .collect()
    }

    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        ensure!(
            args.len().is_multiple_of(2),
            "arguments must be --name value pairs"
        );
        let args: HashMap<_, _> = args
            .chunks_exact(2)
            .map(|x| (x[0].trim_start_matches('-').to_owned(), x[1].clone()))
            .collect();
        let get =
            |key: &str, default: &str| args.get(key).cloned().unwrap_or_else(|| default.to_owned());
        let prefix = args.get("prefix").context("--prefix required")?;
        let pq_prefix = args.get("pq-prefix").unwrap_or(prefix);
        let entry_prefix = args.get("entry-prefix");
        let header_kind = get("header", "cpp");
        ensure!(
            ["cpp", "native"].contains(&header_kind.as_str()),
            "--header must be cpp or native"
        );
        let label_metadata_path = args.get("label-metadata").map(PathBuf::from);
        ensure!(
            header_kind != "native" || label_metadata_path.is_some(),
            "--header native requires --label-metadata pointing to its LabelMetadata bincode file"
        );
        let (label_metadata, label_metadata_identity) = if let Some(path) = &label_metadata_path {
            let bytes = std::fs::read(path)?;
            let metadata: LabelMetadata = bincode::deserialize(&bytes)?;
            ensure!(
                metadata.stable_ids().is_none_or(|ids| ids.iter().enumerate().all(|(ordinal, &id)| ordinal == id as usize)),
                "diagnostic graph/PQ pairing requires identity stable IDs; remapped generations are unsupported"
            );
            let identity = json!({"path":path,"bytes":bytes.len(),"content_fingerprint":metadata_fingerprint(&bytes)});
            (Some(metadata), Some(identity))
        } else {
            (None, None)
        };
        let prepared = PathBuf::from(args.get("prepared").context("--prepared required")?);
        let prepared_manifest: Option<serde_json::Value> =
            if prepared.join("manifest.json").exists() {
                Some(serde_json::from_slice(&std::fs::read(
                    prepared.join("manifest.json"),
                )?)?)
            } else {
                None
            };
        let prepared_identity = if let Some(manifest) = &prepared_manifest {
            [
                "query.u8bin",
                "query_labels.u32bin",
                "groundtruth.bin",
                "labels.txt",
            ]
            .iter()
            .map(|name| {
                manifest["artifacts"][name]["sha256"]
                    .as_str()
                    .context("prepared artifact SHA-256 missing")
                    .map(str::to_owned)
            })
            .collect::<Result<Vec<_>>>()?
            .join(":")
        } else {
            format!("unmanifested-smoke:{}", prepared.canonicalize()?.display())
        };
        let bypass = get("bypass", "true").parse::<bool>()?;
        let timing_authoritative = get("timing-authoritative", "false").parse::<bool>()?;
        let starts = get("starts", "label");
        ensure!(
            starts == "label" || starts == "global",
            "--starts must be label or global"
        );
        let modes = get("modes", "inline,multihop,single");
        let repetitions: usize = get("reps", "3").parse()?;
        let limit: usize = get("limit", "2000").parse()?;
        let k: usize = get("k", "10").parse()?;
        let adaptive_sample: usize = get("adaptive-sample", "1000").parse()?;
        let adaptive_scale: f64 = get("adaptive-scale", "16").parse()?;
        ensure!(repetitions > 0 && k > 0, "reps and k must be positive");
        let mut output = OpenOptions::new()
            .create(true)
            .append(true)
            .open(args.get("out").context("--out required")?)?;
        let mut result_dump = args
            .get("result-dump")
            .map(|path| OpenOptions::new().create_new(true).write(true).open(path))
            .transpose()?;

        let (nq, dim, vectors) = matrix(&prepared.join("query.u8bin"), 1)?;
        let (nl, dl, labels) = u32_matrix(&prepared.join("query_labels.u32bin"))?;
        ensure!(nq == nl && dl == 1, "query label count mismatch");
        let truth_bytes = std::fs::read(prepared.join("groundtruth.bin"))?;
        ensure!(truth_bytes.len() >= 8, "short ground truth");
        let ng = u32::from_le_bytes(truth_bytes[0..4].try_into()?) as usize;
        let kg = u32::from_le_bytes(truth_bytes[4..8].try_into()?) as usize;
        ensure!(
            ng == nq && k == kg && truth_bytes.len() == 8 + ng * kg * 8,
            "ground truth size/k mismatch"
        );
        let queries: Vec<_> = (0..nq.min(limit))
            .map(|i| Query {
                vector: vectors[i * dim..(i + 1) * dim].to_vec(),
                label: labels[i],
                truth: (0..k)
                    .map(|j| {
                        let p = 8 + (i * kg + j) * 4;
                        u32::from_le_bytes(truth_bytes[p..p + 4].try_into().unwrap())
                    })
                    .collect(),
            })
            .collect();
        ensure!(!queries.is_empty(), "no queries selected");
        let mut postings = BTreeMap::new();
        for q in &queries {
            if let std::collections::btree_map::Entry::Vacant(entry) = postings.entry(q.label) {
                let (_, d, ids) = u32_matrix(&prepared.join(format!("label_{}.u32bin", q.label)))?;
                ensure!(
                    d == 1 && ids.windows(2).all(|w| w[0] < w[1]),
                    "postings must be unique and sorted"
                );
                entry.insert(ids);
            }
        }
        let label_entries = if let Some(prefix) = entry_prefix {
            entries(prefix)?
        } else if let Some(metadata) = &label_metadata {
            metadata.entry_points().clone()
        } else {
            entries(prefix)?
        };
        ensure!(
            !Path::new(&format!("{prefix}_disk.index_universal_label.txt")).exists(),
            "universal-label indices require an explicit membership adapter"
        );
        let mut file = File::open(format!("{prefix}_disk.index"))?;
        let mut header = [0u8; 4096];
        file.read_exact(&mut header)?;
        let (meta, cpp_meta) = if header_kind == "cpp" {
            let rows = u32::from_le_bytes(header[0..4].try_into()?) as usize;
            ensure!(
                rows == 9 && u32::from_le_bytes(header[4..8].try_into()?) == 1,
                "unsupported C++ header; require nine-field uncompressed disk layout"
            );
            let meta: Vec<_> = header[8..8 + rows * 8]
                .chunks_exact(8)
                .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
                .collect();
            (meta.clone(), Some(meta))
        } else {
            let inner = DiskVertexProviderFactory::<AdHoc<u8>, _>::new(
                PreadFactory {
                    path: PathBuf::from(format!("{prefix}_disk.index")),
                    bypass,
                    counters: Arc::new(Counters::default()),
                },
                CachingStrategy::None,
            )?;
            let native_header = inner.get_header()?;
            ensure!(
                native_header.block_size() == 4096,
                "native graph block size must be4096"
            );
            let m = native_header.metadata();
            ensure!(
                m.associated_data_length == 0,
                "associated vector data is unsupported in this diagnostic"
            );
            native_header.max_degree::<u8>()?;
            (
                vec![
                    m.num_pts,
                    m.dims as u64,
                    m.medoid,
                    m.node_len,
                    m.num_nodes_per_block,
                    m.vamana_frozen_num,
                    m.vamana_frozen_loc,
                    0,
                    m.disk_index_file_size,
                ],
                None,
            )
        };
        ensure!(
            meta[0] > 0 && meta[2] < meta[0] && meta[5] == 0,
            "invalid medoid or unsupported frozen nodes"
        );
        ensure!(
            meta[3] >= meta[1] + 4 && (meta[3] - meta[1] - 4).is_multiple_of(4),
            "invalid node layout"
        );
        ensure!(
            meta[3] > 0 && meta[4] == 4096 / meta[3] && file.metadata()?.len() == meta[8],
            "disk layout or size mismatch"
        );
        if let Some(manifest) = &prepared_manifest {
            ensure!(
                manifest["base_vectors"].as_u64() == Some(meta[0]),
                "prepared base count does not match graph"
            );
        }
        ensure!(
            meta[1] as usize == dim && meta[7] == 0 && meta[4] > 0,
            "dimension, reorder, or multi-sector layout unsupported"
        );
        if let Some(metadata) = &label_metadata {
            ensure!(
                metadata.point_count() as u64 == meta[0],
                "label metadata/graph row mismatch"
            );
            for (&label, ids) in &postings {
                ensure!(
                    metadata.posting(label) == ids,
                    "native metadata differs from prepared membership for label {label}"
                );
            }
        }
        let membership: BTreeMap<u32, Vec<u64>> = postings
            .iter()
            .map(|(&label, ids)| {
                let mut bits = vec![0u64; (meta[0] as usize).div_ceil(64)];
                for &id in ids {
                    ensure!((id as u64) < meta[0], "posting outside graph");
                    bits[id as usize / 64] |= 1 << (id % 64);
                }
                Ok((label, bits))
            })
            .collect::<Result<_>>()?;
        for (&label, ids) in &postings {
            let seeds = label_entries
                .get(&label)
                .context("query label has no entry point")?;
            ensure!(
                seeds.len() == 1,
                "benchmark requires one entry per label to match C++ start selection"
            );
            ensure!(
                seeds.iter().all(|id| ids.binary_search(id).is_ok()),
                "entry point does not match raw label {label}"
            );
        }
        if prepared.join("base.u8bin").exists() {
            let base = File::open(prepared.join("base.u8bin"))?;
            let mut shape = [0u8; 8];
            base.read_exact_at(&mut shape, 0)?;
            ensure!(
                u32::from_le_bytes(shape[0..4].try_into()?) as u64 == meta[0]
                    && u32::from_le_bytes(shape[4..8].try_into()?) as usize == dim,
                "prepared base shape mismatch"
            );
            let mut expected = vec![0u8; dim];
            let mut indexed = vec![0u8; dim];
            for seed in label_entries.values().flatten().take(64) {
                base.read_exact_at(&mut expected, 8 + *seed as u64 * dim as u64)?;
                let offset =
                    (*seed as u64 / meta[4] + 1) * 4096 + (*seed as u64 % meta[4]) * meta[3];
                file.read_exact_at(&mut indexed, offset)?;
                ensure!(
                    expected == indexed,
                    "indexed vectors/ID mapping do not match prepared base at seed {seed}"
                );
            }
        }
        let reader = DiskIndexReader::new(
            format!("{pq_prefix}_pq_pivots.bin"),
            format!("{pq_prefix}_pq_compressed.bin"),
            &FileStorageProvider,
        )?;
        ensure!(
            reader.get_num_points() == meta[0] as usize,
            "PQ/graph row mismatch"
        );
        ensure!(
            reader.get_pq_data().get_dim() == dim,
            "PQ/graph dimension mismatch"
        );
        let ls = get("ls", "20,40,80,160,320")
            .split(',')
            .map(str::parse::<u32>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let thread_counts = get("threads", "1,8")
            .split(',')
            .map(str::parse::<usize>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for threads in thread_counts {
            ensure!(threads > 0, "threads must be positive");
            let counters = Arc::new(Counters::default());
            let factory = BenchmarkFactory {
                inner: DiskVertexProviderFactory::new(
                    PreadFactory {
                        path: PathBuf::from(format!("{prefix}_disk.index")),
                        bypass,
                        counters: counters.clone(),
                    },
                    CachingStrategy::None,
                )?,
                cpp_meta: cpp_meta.clone(),
            };
            let searcher = DiskIndexSearcher::<AdHoc<u8>, _>::new(
                threads,
                usize::MAX,
                &reader,
                factory,
                Metric::L2,
                None,
            )?;
            for &l in &ls {
                ensure!(l as usize >= k, "L must be >= k");
                for mode in modes.split(',') {
                    ensure!(
                        ["inline", "multihop", "single", "single_frontier"].contains(&mode),
                        "unknown mode {mode}"
                    );
                    ensure!(
                        !mode.starts_with("single") || starts == "label",
                        "single-hop requires matching label starts"
                    );
                    let execute = |q: &Query| -> Result<Observation> {
                        let bits = &membership[&q.label];
                        let predicate = |id: &u32| {
                            bits.get(*id as usize / 64)
                                .is_some_and(|word| word & (1u64 << (id % 64)) != 0)
                        };
                        let search = match mode {
                            "single" => SearchMode::single_hop_filter(predicate),
                            "single_frontier" => SearchMode::single_hop_filter_frontier(predicate),
                            "multihop" => SearchMode::multihop_filter(predicate),
                            _ => SearchMode::inline_filter(
                                predicate,
                                Some(AdaptiveL::new(adaptive_sample, adaptive_scale)?),
                            ),
                        };
                        let search = if starts == "label" {
                            search.with_starting_points(label_entries[&q.label].clone())
                        } else {
                            search
                        };
                        let start = Instant::now();
                        let result = searcher.search(&q.vector, k as u32, l, Some(4), search)?;
                        let micros = start.elapsed().as_secs_f64() * 1e6;
                        let returned = result.stats.result_count as usize;
                        Ok(Observation {
                            results: result.results,
                            micros,
                            recall: 0.0,
                            cmps: result.stats.cmps,
                            hops: result.stats.query_statistics.search_hops,
                            returned,
                            violations: 0,
                            duplicates: 0,
                            label: q.label,
                        })
                    };
                    // Allocate scratch and warm PQ/code paths. The graph reader still bypasses
                    // the OS file cache when requested; these are not forced cold-device reads.
                    run_batch(&queries, threads, &execute)?;
                    for repeat in 0..repetitions {
                        counters.requests.store(0, Ordering::Relaxed);
                        counters.nanos.store(0, Ordering::Relaxed);
                        counters.bytes.store(0, Ordering::Relaxed);
                        let start = Instant::now();
                        let mut observations = run_batch(&queries, threads, &execute)?;
                        let elapsed = start.elapsed().as_secs_f64();
                        // Validate after stopping the throughput timer, as in the C++ driver.
                        for (observation, query) in observations.iter_mut().zip(&queries) {
                            let ids: Vec<_> = observation
                                .results
                                .iter()
                                .take(observation.returned)
                                .map(|r| r.vertex_id)
                                .collect();
                            let unique: std::collections::HashSet<_> =
                                ids.iter().copied().collect();
                            observation.duplicates = ids.len() - unique.len();
                            observation.recall =
                                query.truth.iter().filter(|id| ids.contains(id)).count() as f64
                                    / k as f64;
                            observation.violations = ids
                                .iter()
                                .filter(|id| postings[&query.label].binary_search(id).is_err())
                                .count();
                        }
                        if let Some(dump) = &mut result_dump {
                            for (query_index, observation) in observations.iter().enumerate() {
                                let ids: Vec<_> = observation
                                    .results
                                    .iter()
                                    .take(observation.returned)
                                    .map(|r| r.vertex_id)
                                    .collect();
                                let distances: Vec<_> = observation
                                    .results
                                    .iter()
                                    .take(observation.returned)
                                    .map(|r| r.distance)
                                    .collect();
                                writeln!(
                                    dump,
                                    "{}",
                                    json!({"query":query_index,
                                    "label":observation.label,"mode":mode,"l":l,
                                    "threads":threads,"repeat":repeat,"ids":ids,
                                    "distances":distances,"returned":observation.returned})
                                )?;
                            }
                            dump.flush()?;
                        }
                        let mut latencies: Vec<_> = observations.iter().map(|x| x.micros).collect();
                        latencies.sort_by(f64::total_cmp);
                        let percentile = |p: f64| {
                            latencies[((latencies.len() as f64 * p).ceil() as usize)
                                .saturating_sub(1)
                                .min(latencies.len() - 1)]
                        };
                        let mut by_label: BTreeMap<u32, Vec<&Observation>> = BTreeMap::new();
                        for o in &observations {
                            by_label.entry(o.label).or_default().push(o);
                        }
                        let slices: Vec<_> = by_label.into_iter().map(|(label, obs)| {
                            let mut times: Vec<_> = obs.iter().map(|o| o.micros).collect();
                            times.sort_by(f64::total_cmp);
                            let percentile = |p: f64| times[((times.len() as f64 * p).ceil() as usize).saturating_sub(1).min(times.len() - 1)];
                            json!({"label":label,"eligible":postings[&label].len(),"n":obs.len(),"recall":obs.iter().map(|x|x.recall).sum::<f64>()/obs.len() as f64,
                                "mean_us":times.iter().sum::<f64>()/times.len() as f64,"p50_us":percentile(0.5),"p95_us":percentile(0.95)})
                        }).collect();
                        let n = observations.len() as f64;
                        let mut row = json!({"implementation":"rust","index_prefix":prefix,"prepared_identity":prepared_identity,"prepared":prepared,"timing_authoritative":timing_authoritative,"latency_contract":"public_search_call","query_scheduling":"dynamic_one","mode":mode,"starts":starts,"n_base":meta[0],"dimensions":dim,
                        "queries":observations.len(),"rerank_candidates":match mode {"single"=>Some("expanded"),"single_frontier"=>Some("frontier"),_=>None},"adaptive_sample":if mode=="inline" {Some(adaptive_sample)} else {None},"adaptive_scale":if mode=="inline" {Some(adaptive_scale)} else {None},"k":k,"l":l,"beam":4,"threads":threads,"repeat":repeat,
                        "qps":n/elapsed,"mean_us":latencies.iter().sum::<f64>()/n,"p50_us":percentile(0.5),"p95_us":percentile(0.95),"p99_us":percentile(0.99),
                        "recall":observations.iter().map(|x|x.recall).sum::<f64>()/n,"mean_cmps":observations.iter().map(|x|x.cmps as f64).sum::<f64>()/n,
                        "mean_hops":observations.iter().map(|x|x.hops as f64).sum::<f64>()/n,"short_results":observations.iter().filter(|x|x.returned<k).count(),
                        "filter_violations":observations.iter().map(|x|x.violations).sum::<usize>(),"duplicate_results":observations.iter().map(|x|x.duplicates).sum::<usize>(),"reader_requests":counters.requests.load(Ordering::Relaxed),
                            "reader_time_us":counters.nanos.load(Ordering::Relaxed) as f64 / 1000.0,"reader_bytes":counters.bytes.load(Ordering::Relaxed),"cache_bypass":bypass,"shared_cache_nodes":0,"storage":if cfg!(target_os="linux") && bypass {"native_io_uring"} else {"positional_pread"},"labels":slices});
                        row["pq_prefix"] = json!(pq_prefix);
                        row["header"] = json!(header_kind);
                        row["label_metadata"] = json!(label_metadata_path);
                        row["label_metadata_identity"] = json!(label_metadata_identity);
                        row["entry_prefix"] = json!(entry_prefix);
                        row["entry_source"] = json!(if entry_prefix.is_some() {
                            "cpp_sidecar_override"
                        } else if label_metadata.is_some() {
                            "native_label_metadata"
                        } else {
                            "cpp_sidecars"
                        });
                        writeln!(output, "{row}")?;
                        output.flush()?;
                        eprintln!("{mode} starts={starts} T={threads} L={l} repeat={repeat}: recall={:.4} QPS={:.1} p99={:.1}us", row["recall"].as_f64().unwrap(), row["qps"].as_f64().unwrap(), row["p99_us"].as_f64().unwrap());
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(unix)]
fn main() -> anyhow::Result<()> {
    experiment::run()
}

#[cfg(not(unix))]
fn main() {
    eprintln!("This prototype benchmark requires positional Unix file reads.");
}
