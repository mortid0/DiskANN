/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Native build, durable-update, reopen and compaction experiment.
//!
//! Arguments are `--name value` pairs. `--action all` builds an index, benchmarks
//! its natural prepared workload, then exercises updates and compaction. The
//! update scenario uses a new label and rows outside every queried natural label,
//! preserving the supplied natural ground truth. A separate probe workload has
//! independently recomputed integer squared-L2 ground truth for changed records.
//! `--action delta` instead adds synthetic far vectors to every queried label,
//! measuring the worst case in which every query must exact-scan the live delta.
//! Query-only `--label-l plan.json` supplies a complete label-to-L object for
//! empirical tuning on this workload; it requires one nominal `--l` floor.
//! Held-out subsets can retain the exact frozen plan with
//! `--allow-unused-plan-labels true --evaluation-scope frozen_policy_heldout`.
//! `--query-details-dir` retains each measured query after timing completes.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Barrier,
    },
    time::Instant,
};

use anyhow::{ensure, Context, Result};
use diskann_disk::{
    build::filtered::StitchedBuildConfig,
    filtered::{
        persistence::GenerationStore,
        updates::{DeltaLimits, DeltaSnapshot, Mutation},
        FilteredHit, FilteredIndex, FilteredIndexConfig, FilteredSnapshot,
    },
};
use diskann_utils::views::rowmajor;
use serde_json::{json, Value};

struct Args(HashMap<String, String>);

impl Args {
    fn parse() -> Result<Self> {
        let values: Vec<_> = std::env::args().skip(1).collect();
        ensure!(
            values.len().is_multiple_of(2),
            "arguments must be --name value pairs"
        );
        let mut args = HashMap::new();
        for pair in values.chunks_exact(2) {
            ensure!(pair[0].starts_with("--"), "expected --name argument");
            ensure!(
                args.insert(pair[0][2..].to_owned(), pair[1].clone())
                    .is_none(),
                "duplicate argument {}",
                pair[0]
            );
        }
        Ok(Self(args))
    }

    fn value(&self, name: &str, default: &str) -> String {
        self.0
            .get(name)
            .cloned()
            .unwrap_or_else(|| default.to_owned())
    }

    fn required(&self, name: &str) -> Result<PathBuf> {
        Ok(PathBuf::from(
            self.0
                .get(name)
                .with_context(|| format!("--{name} required"))?,
        ))
    }
}

fn header(path: &Path) -> Result<(usize, usize)> {
    let mut bytes = [0; 8];
    File::open(path)?.read_exact(&mut bytes)?;
    Ok((
        u32::from_le_bytes(bytes[..4].try_into()?) as usize,
        u32::from_le_bytes(bytes[4..].try_into()?) as usize,
    ))
}

fn matrix_u8(path: &Path) -> Result<(usize, usize, Vec<u8>)> {
    let (rows, dimensions) = header(path)?;
    let count = rows
        .checked_mul(dimensions)
        .context("matrix dimensions overflow")?;
    let mut file = File::open(path)?;
    ensure!(
        file.metadata()?.len() == 8 + count as u64,
        "invalid uint8 matrix size: {}",
        path.display()
    );
    file.read_exact(&mut [0; 8])?;
    let mut values = vec![0; count];
    file.read_exact(&mut values)?;
    Ok((rows, dimensions, values))
}

fn matrix_u32(path: &Path) -> Result<(usize, usize, Vec<u32>)> {
    let (rows, columns) = header(path)?;
    let bytes = std::fs::read(path)?;
    ensure!(
        bytes.len() == 8 + rows * columns * 4,
        "invalid uint32 matrix size: {}",
        path.display()
    );
    let values = bytes[8..]
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().expect("four-byte chunk")))
        .collect();
    Ok((rows, columns, values))
}

fn read_postings(path: &Path, rows: usize, skip: Option<u32>) -> Result<BTreeMap<u32, Vec<u32>>> {
    let mut postings = BTreeMap::<u32, Vec<u32>>::new();
    let mut count = 0;
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        ensure!(count < rows, "more label rows than base vectors");
        let mut labels = BTreeSet::new();
        for token in line
            .split([',', ' ', '\t'])
            .filter(|token| !token.is_empty())
        {
            let label = token.parse::<u32>()?;
            if Some(label) != skip {
                labels.insert(label);
            }
        }
        for label in labels {
            postings.entry(label).or_default().push(count as u32);
        }
        count += 1;
    }
    ensure!(count == rows, "label row count differs from base vectors");
    Ok(postings)
}

#[derive(Clone)]
struct Query {
    vector: Vec<u8>,
    label: u32,
    truth: Vec<u32>,
    kth_distance: f64,
}

struct Workload {
    queries: Vec<Query>,
    postings: BTreeMap<u32, Vec<u32>>,
    k: usize,
    dimensions: usize,
    identity: String,
}

fn load_workload(prepared: &Path, limit: usize) -> Result<Workload> {
    let (rows, dimensions, vectors) = matrix_u8(&prepared.join("query.u8bin"))?;
    let (label_rows, label_width, labels) = matrix_u32(&prepared.join("query_labels.u32bin"))?;
    let truth_path = prepared.join("groundtruth.bin");
    let (truth_rows, k) = header(&truth_path)?;
    let truth_bytes = std::fs::read(truth_path)?;
    ensure!(
        rows == label_rows && label_width == 1 && rows == truth_rows && k > 0,
        "prepared query/label/truth shapes differ"
    );
    ensure!(
        truth_bytes.len() == 8 + rows * k * 8,
        "ground truth must contain IDs and distances"
    );
    let queries: Vec<_> = (0..rows.min(limit))
        .map(|i| -> Result<Query> {
            let distances: Vec<_> = (0..k)
                .map(|j| {
                    let offset = 8 + rows * k * 4 + (i * k + j) * 4;
                    f32::from_le_bytes(
                        truth_bytes[offset..offset + 4]
                            .try_into()
                            .expect("four-byte distance"),
                    )
                })
                .collect();
            ensure!(
                distances.iter().all(|d| d.is_finite() && *d >= 0.)
                    && distances.windows(2).all(|pair| pair[0] <= pair[1]),
                "ground-truth distances must be finite, nonnegative and sorted (query {i})"
            );
            Ok(Query {
                vector: vectors[i * dimensions..(i + 1) * dimensions].to_vec(),
                label: labels[i],
                truth: (0..k)
                    .map(|j| {
                        let offset = 8 + (i * k + j) * 4;
                        u32::from_le_bytes(
                            truth_bytes[offset..offset + 4]
                                .try_into()
                                .expect("four-byte truth ID"),
                        )
                    })
                    .collect(),
                kth_distance: f64::from(distances[k - 1]),
            })
        })
        .collect::<Result<_>>()?;
    ensure!(!queries.is_empty(), "query workload is empty");
    let mut postings = BTreeMap::new();
    for label in queries.iter().map(|q| q.label).collect::<BTreeSet<_>>() {
        let (n, width, ids) = matrix_u32(&prepared.join(format!("label_{label}.u32bin")))?;
        ensure!(
            width == 1 && ids.len() == n && ids.windows(2).all(|w| w[0] < w[1]),
            "invalid label postings"
        );
        postings.insert(label, ids);
    }
    let manifest_path = prepared.join("manifest.json");
    let identity = if manifest_path.exists() {
        let manifest: Value = serde_json::from_slice(&std::fs::read(manifest_path)?)?;
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
                .context("missing prepared hash")
                .map(str::to_owned)
        })
        .collect::<Result<Vec<_>>>()?
        .join(":")
    } else {
        format!("unmanifested:{}", prepared.canonicalize()?.display())
    };
    Ok(Workload {
        queries,
        postings,
        k,
        dimensions,
        identity,
    })
}

fn emit(output: &mut File, row: Value) -> Result<()> {
    writeln!(output, "{row}")?;
    output.flush()?;
    Ok(())
}

struct Observation {
    query: usize,
    hits: Vec<FilteredHit>,
    micros: f64,
    sequence: u64,
    exact_fallback: bool,
    base_searches: usize,
    delta_candidates: usize,
}

struct LabelLPlan {
    path: PathBuf,
    content_sha256: String,
    entries: BTreeMap<u32, u32>,
    unused_labels: Vec<u32>,
}

impl LabelLPlan {
    fn load(
        path: &Path,
        queried: &BTreeSet<u32>,
        baseline: u32,
        k: usize,
        allow_unused: bool,
    ) -> Result<Self> {
        ensure!(
            baseline as usize >= k,
            "label-L baseline must be at least k"
        );
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading label-L plan {}", path.display()))?;
        let entries: BTreeMap<u32, u32> = serde_json::from_slice(&bytes)
            .context("label-L plan must be a JSON object mapping u32 label IDs to u32 breadths")?;
        for (&label, &breadth) in &entries {
            ensure!(
                allow_unused || queried.contains(&label),
                "label-L plan contains unknown queried label {label}"
            );
            ensure!(breadth >= baseline && breadth as usize >= k,
                "label-L breadth {breadth} for label {label} must be >= nominal L {baseline} and k {k}");
        }
        let missing: Vec<_> = queried
            .iter()
            .filter(|label| !entries.contains_key(label))
            .copied()
            .collect();
        ensure!(
            missing.is_empty(),
            "label-L plan does not cover queried labels {missing:?}"
        );
        Ok(Self {
            path: path.canonicalize()?,
            content_sha256: sha256(&bytes),
            unused_labels: entries
                .keys()
                .filter(|label| !queried.contains(label))
                .copied()
                .collect(),
            entries,
        })
    }

    fn metadata(&self) -> Value {
        json!({"path":self.path,"content_sha256":self.content_sha256,"unused_plan_labels":self.unused_labels,
            "entries":self.entries.iter().map(|(&label,&l)| json!({"label":label,"l":l})).collect::<Vec<_>>()})
    }
}

fn actual_l(plan: Option<&LabelLPlan>, label: u32, baseline: u32) -> u32 {
    plan.and_then(|plan| plan.entries.get(&label))
        .copied()
        .unwrap_or(baseline)
}

struct QueryMeasurement<'a> {
    phase: &'a str,
    threads: usize,
    l: u32,
    repeat: usize,
    label_l_plan: Option<&'a LabelLPlan>,
}

// Retained observations are serialized only after the timed worker loop.
fn write_query_details(
    directory: &Path,
    observations: &[Observation],
    workload: &Workload,
    measurement: &QueryMeasurement<'_>,
) -> Result<Value> {
    let path = directory.join(format!(
        "{}.t{}.L{}.r{}.jsonl",
        measurement.phase, measurement.threads, measurement.l, measurement.repeat
    ));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .with_context(|| format!("creating query details {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for observation in observations {
        let query = &workload.queries[observation.query];
        let ids: Vec<_> = observation.hits.iter().map(|hit| hit.id).collect();
        let distances: Vec<_> = observation.hits.iter().map(|hit| hit.distance).collect();
        let strict_id_hits = query.truth.iter().filter(|id| ids.contains(id)).count();
        let row = json!({"kind":"query_detail","phase":measurement.phase,
            "query_index":observation.query,"label":query.label,"l":measurement.l,
            "actual_l":actual_l(measurement.label_l_plan,query.label,measurement.l),
            "repeat":measurement.repeat,"k":workload.k,"latency_us":observation.micros,
            "ids":ids,"distances":distances,"strict_id_hits":strict_id_hits,
            "sequence":observation.sequence,"exact_fallback":observation.exact_fallback,
            "base_searches":observation.base_searches,"delta_candidates":observation.delta_candidates});
        writeln!(writer, "{row}")?;
    }
    writer.flush()?;
    drop(writer);
    let bytes = std::fs::read(&path)?;
    Ok(json!({"path":path,"sha256":sha256(&bytes),"queries":observations.len()}))
}

// SHA-256 of exact plan bytes, including whitespace/newline. Kept local to this
// benchmark so adding provenance does not change the library's dependencies.
fn sha256(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut padded = bytes.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&(bytes.len() as u64).wrapping_mul(8).to_be_bytes());
    for block in padded.chunks_exact(64) {
        let mut words = [0u32; 64];
        for (word, bytes) in words[..16].iter_mut().zip(block.chunks_exact(4)) {
            *word = u32::from_be_bytes(bytes.try_into().expect("four-byte SHA word"));
        }
        for i in 16..64 {
            let a = words[i - 15].rotate_right(7)
                ^ words[i - 15].rotate_right(18)
                ^ (words[i - 15] >> 3);
            let b = words[i - 2].rotate_right(17)
                ^ words[i - 2].rotate_right(19)
                ^ (words[i - 2] >> 10);
            words[i] = words[i - 16]
                .wrapping_add(a)
                .wrapping_add(words[i - 7])
                .wrapping_add(b);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for i in 0..64 {
            let s = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s)
                .wrapping_add(choose)
                .wrapping_add(K[i])
                .wrapping_add(words[i]);
            let s = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (state, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *state = state.wrapping_add(value);
        }
    }
    state.iter().map(|value| format!("{value:08x}")).collect()
}

fn benchmark(
    snapshot: &FilteredSnapshot<u8>,
    workload: &Workload,
    args: &Args,
    phase: &str,
    output: &mut File,
) -> Result<()> {
    let threads: Vec<usize> = args
        .value("query-threads", &args.value("threads", "8"))
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let breadths: Vec<u32> = args
        .value("l", "60,144")
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let repeats: usize = args.value("reps", "3").parse()?;
    let authoritative: bool = args.value("timing-authoritative", "false").parse()?;
    let allow_unused_plan_labels: bool = args.value("allow-unused-plan-labels", "false").parse()?;
    let evaluation_scope = args.value(
        "evaluation-scope",
        "empirical_on_existing_prepared_workload_not_held_out",
    );
    let query_details = if let Some(path) = args.0.get("query-details-dir") {
        std::fs::create_dir_all(path)?;
        Some(Path::new(path).canonicalize()?)
    } else {
        None
    };
    ensure!(
        repeats > 0
            && threads.iter().all(|&n| n > 0)
            && breadths.iter().all(|&l| l as usize >= workload.k),
        "positive threads/repeats and L >= K required"
    );
    let label_l_plan = if let Some(path) = args.0.get("label-l") {
        ensure!(
            breadths.len() == 1,
            "--label-l requires exactly one nominal --l value"
        );
        let queried = workload.queries.iter().map(|query| query.label).collect();
        Some(LabelLPlan::load(
            Path::new(path),
            &queried,
            breadths[0],
            workload.k,
            allow_unused_plan_labels,
        )?)
    } else {
        None
    };
    let plan_metadata = label_l_plan
        .as_ref()
        .map(LabelLPlan::metadata)
        .unwrap_or(Value::Null);
    for &threads in &threads {
        for &l in &breadths {
            for repeat in 0..repeats {
                let warmup_next = AtomicUsize::new(0);
                let measured_next = AtomicUsize::new(0);
                let ready = Barrier::new(threads + 1);
                let start = Barrier::new(threads + 1);
                let (mut observations, elapsed) = std::thread::scope(|scope| -> Result<_> {
                    let workers: Vec<_> = (0..threads)
                        .map(|_| {
                            scope.spawn(|| -> Result<Vec<Observation>> {
                                let mut observed =
                                    Vec::with_capacity(workload.queries.len().div_ceil(threads));
                                let warmup = (|| -> Result<()> {
                                    loop {
                                        let query = warmup_next.fetch_add(1, Ordering::Relaxed);
                                        let Some(query) = workload.queries.get(query) else {
                                            break;
                                        };
                                        snapshot.search(
                                            &query.vector,
                                            query.label,
                                            workload.k,
                                            actual_l(label_l_plan.as_ref(), query.label, l),
                                        )?;
                                    }
                                    Ok(())
                                })();
                                ready.wait();
                                start.wait();
                                warmup?;
                                loop {
                                    let query_index = measured_next.fetch_add(1, Ordering::Relaxed);
                                    let Some(query) = workload.queries.get(query_index) else {
                                        break;
                                    };
                                    let before = Instant::now();
                                    let found = snapshot.search(
                                        &query.vector,
                                        query.label,
                                        workload.k,
                                        actual_l(label_l_plan.as_ref(), query.label, l),
                                    )?;
                                    let micros = before.elapsed().as_secs_f64() * 1e6;
                                    observed.push(Observation {
                                        query: query_index,
                                        hits: found.hits,
                                        micros,
                                        sequence: found.sequence,
                                        exact_fallback: found.exact_fallback,
                                        base_searches: found.base_searches,
                                        delta_candidates: found.delta_candidates,
                                    });
                                }
                                Ok(observed)
                            })
                        })
                        .collect();
                    ready.wait();
                    let begin = Instant::now();
                    start.wait();
                    let mut observed = Vec::with_capacity(workload.queries.len());
                    for worker in workers {
                        observed.extend(
                            worker
                                .join()
                                .map_err(|_| anyhow::anyhow!("query worker panicked"))??,
                        );
                    }
                    Ok((observed, begin.elapsed()))
                })?;
                observations.sort_unstable_by_key(|x| x.query);
                ensure!(
                    observations.len() == workload.queries.len(),
                    "incomplete measured workload"
                );
                let mut violations = 0;
                let mut duplicates = 0;
                let mut short = 0;
                let mut hits = 0;
                let mut by_label = BTreeMap::<u32, (usize, usize, Vec<f64>)>::new();
                for (i, observation) in observations.iter().enumerate() {
                    let query = &workload.queries[i];
                    ensure!(
                        observation.query == i && observation.sequence == snapshot.sequence(),
                        "inconsistent query snapshot"
                    );
                    let ids: BTreeSet<_> = observation.hits.iter().map(|h| h.id).collect();
                    duplicates += observation.hits.len() - ids.len();
                    short += usize::from(observation.hits.len() < workload.k);
                    violations += ids
                        .iter()
                        .filter(|id| workload.postings[&query.label].binary_search(id).is_err())
                        .count();
                    let correct = query.truth.iter().filter(|id| ids.contains(id)).count();
                    hits += correct;
                    let entry = by_label.entry(query.label).or_default();
                    entry.0 += 1;
                    entry.1 += correct;
                    entry.2.push(observation.micros);
                }
                let mut latencies: Vec<_> = observations.iter().map(|o| o.micros).collect();
                latencies.sort_unstable_by(f64::total_cmp);
                let percentile =
                    |p: f64| latencies[(latencies.len() as f64 * p).ceil() as usize - 1];
                let labels: Vec<_> = by_label.into_iter().map(|(label, (queries, hits, mut latencies))| {
                    latencies.sort_unstable_by(f64::total_cmp);
                    json!({"label":label,"queries":queries,"actual_l":actual_l(label_l_plan.as_ref(),label,l),"recall":hits as f64/(queries*workload.k) as f64,
                        "mean_us":latencies.iter().sum::<f64>()/queries as f64,
                        "p95_us":latencies[(queries as f64*0.95).ceil() as usize-1],
                        "p99_us":latencies[(queries as f64*0.99).ceil() as usize-1]})
                }).collect();
                let mut row = json!({"kind":"query","implementation":"rust_native_lifecycle","phase":phase,
                    "sequence":snapshot.sequence(),"queries":observations.len(),"warmup_queries":observations.len(),
                    "label_l_plan":plan_metadata,"label_l_tuning_scope":if label_l_plan.is_some() {Some(evaluation_scope.as_str())} else {None},
                    "unused_plan_labels":label_l_plan.as_ref().map(|plan|plan.unused_labels.as_slice()).unwrap_or(&[]),
                    "threads":threads,"l":l,"k":workload.k,"repeat":repeat,"prepared_identity":workload.identity,
                    "timing_authoritative":authoritative,"latency_contract":"public_snapshot_search_call",
                    "qps":observations.len() as f64/elapsed.as_secs_f64(),"wall_seconds":elapsed.as_secs_f64(),
                    "mean_us":latencies.iter().sum::<f64>()/latencies.len() as f64,
                    "p50_us":percentile(0.5),"p95_us":percentile(0.95),"p99_us":percentile(0.99),
                    "recall":hits as f64/(observations.len()*workload.k) as f64,"strict_id_recall":true,
                    "filter_violations":violations,"duplicate_results":duplicates,"short_results":short,
                    "exact_fallback_queries":observations.iter().filter(|o| o.exact_fallback).count(),
                    "base_searches":observations.iter().map(|o| o.base_searches).sum::<usize>(),
                    "delta_candidates":observations.iter().map(|o| o.delta_candidates).sum::<usize>(),"labels":labels});
                if let Some(directory) = &query_details {
                    row["raw_queries"] = write_query_details(
                        directory,
                        &observations,
                        workload,
                        &QueryMeasurement {
                            phase,
                            threads,
                            l,
                            repeat,
                            label_l_plan: label_l_plan.as_ref(),
                        },
                    )?;
                }
                if let Some(size) = phase
                    .strip_prefix("delta_")
                    .and_then(|size| size.parse::<usize>().ok())
                {
                    row["synthetic_delta_records"] = json!(size);
                    row["stress_scope"] =
                        json!("new_far_vectors_matching_every_queried_natural_label");
                    row["groundtruth_unchanged_by_inserts"] = json!(true);
                    ensure!(
                        observations.iter().all(|o| o.delta_candidates == size),
                        "synthetic stress requires every query to match all {size} delta vectors"
                    );
                }
                emit(output, row)?;
                ensure!(
                    violations == 0 && duplicates == 0 && short == 0,
                    "invalid or incomplete filtered results in {phase}"
                );
            }
        }
    }
    Ok(())
}

fn exact_l2(left: &[u8], right: &[u8]) -> u64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| {
            let d = i64::from(a) - i64::from(b);
            (d * d) as u64
        })
        .sum()
}

fn delta_sizes(text: &str) -> Result<Vec<usize>> {
    let sizes: Vec<usize> = text
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    ensure!(
        !sizes.is_empty()
            && sizes.iter().all(|&size| size > 0 && size <= 10_000)
            && sizes.windows(2).all(|pair| pair[0] < pair[1]),
        "--delta-sizes must be strictly increasing positive counts no greater than 10000"
    );
    Ok(sizes)
}

fn prove_far_delta(workload: &Workload) -> Result<(u64, f64)> {
    let far = vec![255; workload.dimensions];
    let mut minimum = u64::MAX;
    let mut minimum_margin = f64::INFINITY;
    for (query, item) in workload.queries.iter().enumerate() {
        let distance = exact_l2(&item.vector, &far);
        ensure!(distance as f64 > item.kth_distance,
            "far vector does not preserve published top-k for query {query}: integer distance {distance} <= kth {}", item.kth_distance);
        minimum = minimum.min(distance);
        minimum_margin = minimum_margin.min(distance as f64 - item.kth_distance);
    }
    Ok((minimum, minimum_margin))
}

fn delta_batch_size(
    dimensions: usize,
    labels: &[u32],
    max_frame_bytes: usize,
) -> Result<(usize, u64)> {
    let empty: &[Mutation<u8>] = &[];
    let overhead = bincode::serialized_size(empty)?;
    let single = bincode::serialized_size(&[Mutation::Upsert {
        id: 0,
        vector: vec![255u8; dimensions],
        labels: labels.to_vec(),
    }])?;
    let limit = max_frame_bytes.min(16 * 1024 * 1024) as u64;
    ensure!(
        single <= limit,
        "one synthetic insert exceeds the persisted journal frame cap or 16MiB"
    );
    let per_record = single
        .checked_sub(overhead)
        .context("invalid mutation encoding size")?;
    ensure!(per_record > 0, "empty synthetic mutation encoding");
    let batch_size = usize::try_from((limit - overhead) / per_record)?.min(1000);
    ensure!(
        batch_size > 0,
        "journal frame cap cannot hold a synthetic insert"
    );
    Ok((batch_size, single))
}

/// Synthetic matching-delta stress: no base vectors/labels/graph are modified.
fn stress_delta(
    index: FilteredIndex<u8>,
    root: &Path,
    mut workload: Workload,
    args: &Args,
    output: &mut File,
) -> Result<()> {
    let sizes = delta_sizes(&args.value("delta-sizes", "100,1000,10000"))?;
    let maximum = *sizes.last().context("empty delta grid")?;
    let initial = index.snapshot();
    ensure!(initial.sequence() == initial.base_sequence(),
        "delta stress requires an empty live delta (sequence == base_sequence); start from a fresh/compacted index");
    let labels: Vec<_> = workload.postings.keys().copied().collect();
    let far = vec![255; workload.dimensions];
    let (minimum_far_distance, minimum_kth_margin) = prove_far_delta(&workload)?;
    let loaded = GenerationStore::read_current(root)?.context("no committed generation")?;
    let manifest: Value = serde_json::from_slice(&loaded.manifest_payload)?;
    let number = |name: &str| {
        manifest[name]
            .as_u64()
            .with_context(|| format!("persisted manifest missing {name}"))
    };
    let limits = DeltaLimits {
        max_records: usize::try_from(number("max_records")?)?,
        max_bytes: usize::try_from(number("max_delta_bytes")?)?,
    };
    let max_frame = usize::try_from(number("max_frame_bytes")?)?;
    let max_journal = number("max_journal_bytes")?;
    let first_id = u32::try_from(number("next_id")?)?;
    ensure!(
        workload
            .postings
            .values()
            .all(|posting| posting.last().is_none_or(|&id| id < first_id)),
        "prepared natural postings overlap the index's next stable ID"
    );
    ensure!(
        u64::from(first_id) + maximum as u64 <= u64::from(u32::MAX) + 1,
        "synthetic inserts would exhaust stable IDs"
    );
    ensure!(
        maximum <= limits.max_records,
        "delta grid exceeds persisted max_records"
    );
    let (batch_size, single_payload) = delta_batch_size(workload.dimensions, &labels, max_frame)?;
    let initial_journal_bytes = std::fs::metadata(root.join("updates.wal"))?.len();
    let worst_journal_bytes = (maximum as u64)
        .checked_mul(single_payload + 40)
        .and_then(|growth| growth.checked_add(initial_journal_bytes))
        .context("journal size overflow")?;
    ensure!(
        worst_journal_bytes <= max_journal,
        "synthetic grid's conservative journal bound exceeds persisted max_journal_bytes"
    );
    // Use the actual delta admission implementation before any persistent mutation.
    let staged_mutations: Vec<_> = (0..maximum)
        .map(|id| Mutation::Upsert {
            id: id as u32,
            vector: far.clone(),
            labels: labels.clone(),
        })
        .collect();
    let staged = DeltaSnapshot::new(initial.sequence(), workload.dimensions)?.apply_batch(
        initial
            .sequence()
            .checked_add(1)
            .context("sequence exhausted")?,
        &staged_mutations,
        limits,
    )?;
    let accounted_bytes = staged.bytes();
    drop((staged, staged_mutations));
    let mut measured_args = Args(args.0.clone());
    measured_args
        .0
        .entry("l".into())
        .or_insert_with(|| "60,64,144".into());
    emit(
        output,
        json!({"kind":"delta_stress","action":"delta","sizes":sizes,
        "scope":"synthetic_worst_case_all_queried_natural_labels_match_every_insert",
        "far_vector_value":255,"dimensions":workload.dimensions,"labels":labels,
        "queried_label_count":labels.len(),"queries":workload.queries.len(),"k":workload.k,
        "prepared_identity":workload.identity,"generation":loaded.generation,"initial_sequence":initial.sequence(),"base_sequence":initial.base_sequence(),
        "minimum_exact_integer_far_distance":minimum_far_distance,"minimum_far_minus_published_kth":minimum_kth_margin,
        "groundtruth_unchanged_by_inserts":true,"proof":"every exact integer squared-L2 far distance strictly exceeds that query's published kth distance",
        "max_delta_records":limits.max_records,"max_delta_bytes":limits.max_bytes,"largest_phase_accounted_delta_bytes":accounted_bytes,
        "max_frame_bytes":max_frame,"effective_frame_cap":max_frame.min(16*1024*1024),"max_batch_records":batch_size,
        "max_journal_bytes":max_journal,"conservative_final_journal_bytes":worst_journal_bytes,
        "first_stable_id":first_id,"compaction_performed":false,"graph_unchanged":true}),
    )?;
    benchmark(&initial, &workload, &measured_args, "delta_0", output)?;
    drop(initial);
    let mut inserted = 0usize;
    for target in sizes {
        let phase_begin = Instant::now();
        let previous = inserted;
        let mut batches = 0;
        if inserted == 0 {
            let (id, _) = index.insert(far.clone(), labels.clone())?;
            ensure!(id == first_id, "unexpected allocation watermark");
            inserted = 1;
            batches += 1;
        }
        while inserted < target {
            let count = batch_size.min(target - inserted);
            let mutations: Vec<_> = (inserted..inserted + count)
                .map(|offset| Mutation::Upsert {
                    id: first_id + offset as u32,
                    vector: far.clone(),
                    labels: labels.clone(),
                })
                .collect();
            ensure!(
                bincode::serialized_size(&mutations)? <= max_frame.min(16 * 1024 * 1024) as u64,
                "synthetic batch exceeds frame cap"
            );
            index.apply(&mutations)?;
            inserted += count;
            batches += 1;
        }
        let mutation_seconds = phase_begin.elapsed().as_secs_f64();
        for posting in workload.postings.values_mut() {
            posting.extend((previous..inserted).map(|offset| first_id + offset as u32));
            ensure!(
                posting.windows(2).all(|pair| pair[0] < pair[1]),
                "new stable IDs must follow prepared natural postings"
            );
        }
        let snapshot = index.snapshot();
        for id in [first_id, first_id + (inserted - 1) as u32] {
            let (vector, current_labels) = snapshot.get(id)?.context("synthetic insert absent")?;
            ensure!(
                vector == far && current_labels == labels,
                "synthetic record differs from stress definition"
            );
        }
        emit(
            output,
            json!({"kind":"delta_apply","phase":format!("delta_{target}"),
            "scope":"synthetic_worst_case_all_queried_natural_labels_match_every_insert",
            "inserted_this_phase":inserted-previous,"cumulative_delta_records":inserted,"durable_batches":batches,
            "seconds":mutation_seconds,"mutations_per_second":(inserted-previous) as f64/mutation_seconds,
            "sequence":snapshot.sequence(),"base_sequence":snapshot.base_sequence(),
            "journal_bytes":std::fs::metadata(root.join("updates.wal"))?.len(),"compaction_performed":false}),
        )?;
        benchmark(
            &snapshot,
            &workload,
            &measured_args,
            &format!("delta_{target}"),
            output,
        )?;
    }
    Ok(())
}

struct Scenario {
    mutations: Vec<Mutation<u8>>,
    before: BTreeMap<u32, (Vec<u8>, Vec<u32>)>,
    live: BTreeMap<u32, Vec<u8>>,
    deleted: Vec<u32>,
    label: u32,
    dimensions: usize,
}

fn scenario(
    snapshot: &FilteredSnapshot<u8>,
    workload: &Workload,
    rows: usize,
    count: usize,
) -> Result<Scenario> {
    ensure!(
        count >= 4 && count.is_multiple_of(4),
        "--updates must be a positive multiple of four"
    );
    let quarter = count / 4;
    let mut queried = vec![false; rows];
    for posting in workload.postings.values() {
        for &id in posting {
            ensure!((id as usize) < rows, "posting outside base");
            queried[id as usize] = true;
        }
    }
    let ids: Vec<_> = queried
        .iter()
        .enumerate()
        .filter_map(|(id, &matches)| (!matches).then_some(id as u32))
        .take(quarter * 3)
        .collect();
    ensure!(
        ids.len() == quarter * 3,
        "not enough rows outside queried labels for this scenario"
    );
    let label = (0..=u32::MAX)
        .rev()
        .find(|label| !workload.postings.contains_key(label))
        .context("no unused probe label")?;
    ensure!(
        snapshot
            .search(&workload.queries[0].vector, label, 1, 1)?
            .hits
            .is_empty(),
        "probe label already exists in the index"
    );
    let mut mutations = Vec::with_capacity(count);
    let mut before = BTreeMap::new();
    let mut live = BTreeMap::new();
    let mut deleted = Vec::new();
    for (offset, &id) in ids.iter().enumerate() {
        let original = snapshot
            .get(id)?
            .context("selected base row does not exist")?;
        before.insert(id, original.clone());
        if offset < quarter * 2 {
            let vector = if offset < quarter {
                workload.queries[offset % workload.queries.len()]
                    .vector
                    .clone()
            } else {
                original.0
            };
            live.insert(id, vector.clone());
            mutations.push(Mutation::Upsert {
                id,
                vector,
                labels: vec![label],
            });
        } else {
            deleted.push(id);
            mutations.push(Mutation::Delete { id });
        }
    }
    for offset in 0..quarter {
        let id = u32::try_from(rows.checked_add(offset).context("insert ID overflow")?)?;
        ensure!(
            snapshot.get(id)?.is_none(),
            "insert scenario requires an original uncompacted index with dense IDs"
        );
        let vector = workload.queries[(offset + quarter) % workload.queries.len()]
            .vector
            .clone();
        live.insert(id, vector.clone());
        mutations.push(Mutation::Upsert {
            id,
            vector,
            labels: vec![label],
        });
    }
    Ok(Scenario {
        mutations,
        before,
        live,
        deleted,
        label,
        dimensions: workload.dimensions,
    })
}

fn verify_changed(
    snapshot: &FilteredSnapshot<u8>,
    scenario: &Scenario,
    sequence: u64,
    phase: &str,
    output: &mut File,
) -> Result<()> {
    ensure!(
        snapshot.sequence() == sequence,
        "unexpected lifecycle sequence"
    );
    for (&id, vector) in &scenario.live {
        let (found, labels) = snapshot.get(id)?.context("live changed record absent")?;
        ensure!(
            &found == vector && labels == [scenario.label],
            "changed vector/labels mismatch for {id}"
        );
    }
    for &id in &scenario.deleted {
        ensure!(
            snapshot.get(id)?.is_none(),
            "deleted record reappeared: {id}"
        );
    }
    let k = 10.min(scenario.live.len());
    let l = u32::try_from(scenario.live.len())?;
    let mut correct = 0;
    let mut fallbacks = 0;
    for vector in scenario.live.values() {
        let mut truth: Vec<_> = scenario
            .live
            .iter()
            .map(|(&id, value)| (exact_l2(vector, value), id))
            .collect();
        truth.sort_unstable();
        let expected: BTreeSet<_> = truth.iter().take(k).map(|(_, id)| *id).collect();
        let found = snapshot.search(vector, scenario.label, k, l)?;
        ensure!(
            found.sequence == sequence && found.hits.len() == k,
            "probe results incomplete"
        );
        let mut unique = BTreeSet::new();
        for hit in found.hits {
            ensure!(unique.insert(hit.id), "duplicate probe result");
            let stored = scenario
                .live
                .get(&hit.id)
                .context("probe returned a nonmatching record")?;
            ensure!(
                f64::from(hit.distance) == exact_l2(vector, stored) as f64,
                "probe distance differs from exact integer L2"
            );
            correct += usize::from(expected.contains(&hit.id));
        }
        fallbacks += usize::from(found.exact_fallback);
    }
    emit(
        output,
        json!({"kind":"probe","phase":phase,"sequence":sequence,"label":scenario.label,
        "live_records":scenario.live.len(),"deleted_records":scenario.deleted.len(),"queries":scenario.live.len(),
        "k":k,"l":l,"dimensions":scenario.dimensions,"recall":correct as f64/(scenario.live.len()*k) as f64,
        "truth":"independent_integer_squared_l2_then_stable_id","exact_fallback_queries":fallbacks,
        "point_record_checks":true,"filter_violations":0,"duplicate_results":0,"short_results":0}),
    )?;
    ensure!(
        correct == scenario.live.len() * k,
        "exhaustive-breadth probe differs from independent exact truth"
    );
    Ok(())
}

async fn exercise(
    mut index: FilteredIndex<u8>,
    root: &Path,
    rows: usize,
    workload: &Workload,
    args: &Args,
    output: &mut File,
    threads: usize,
) -> Result<()> {
    let old = index.snapshot();
    benchmark(&old, workload, args, "before_updates", output)?;
    let changes = scenario(&old, workload, rows, args.value("updates", "100").parse()?)?;
    let begin = Instant::now();
    let sequence = index.apply(&changes.mutations)?;
    emit(
        output,
        json!({"kind":"lifecycle","action":"apply","sequence":sequence,"mutations":changes.mutations.len(),
        "seconds":begin.elapsed().as_secs_f64(),"scope":"rows_outside_queried_natural_labels_plus_new_probe_label"}),
    )?;
    for (&id, expected) in &changes.before {
        ensure!(
            old.get(id)?.as_ref() == Some(expected),
            "old snapshot record changed"
        );
    }
    ensure!(
        old.search(&workload.queries[0].vector, changes.label, 1, 1)?
            .hits
            .is_empty(),
        "old snapshot sees new probe label"
    );
    let updated = index.snapshot();
    verify_changed(&updated, &changes, sequence, "after_updates", output)?;
    benchmark(&updated, workload, args, "after_updates", output)?;
    let begin = Instant::now();
    let generation = index.compact().await?;
    emit(
        output,
        json!({"kind":"lifecycle","action":"compact","generation":generation,
        "seconds":begin.elapsed().as_secs_f64()}),
    )?;
    // These leases still refer to the generations preceding compaction.
    verify_changed(
        &updated,
        &changes,
        sequence,
        "old_updated_snapshot_after_compact",
        output,
    )?;
    for (&id, expected) in &changes.before {
        ensure!(
            old.get(id)?.as_ref() == Some(expected),
            "pre-update snapshot changed after compact"
        );
    }
    let compacted = index.snapshot();
    verify_changed(&compacted, &changes, sequence, "after_compact", output)?;
    benchmark(&compacted, workload, args, "after_compact", output)?;
    drop((old, updated, compacted, index));
    let begin = Instant::now();
    index = FilteredIndex::open(root, threads)?;
    emit(
        output,
        json!({"kind":"lifecycle","action":"reopen","seconds":begin.elapsed().as_secs_f64(),"sequence":sequence}),
    )?;
    verify_changed(
        &index.snapshot(),
        &changes,
        sequence,
        "after_reopen",
        output,
    )?;
    benchmark(&index.snapshot(), workload, args, "after_reopen", output)?;
    let removed = index.collect_garbage()?;
    emit(
        output,
        json!({"kind":"lifecycle","action":"collect_garbage","removed":removed,
        "old_snapshots_consistent":true,"durable_reopen_verified":true}),
    )?;
    Ok(())
}

async fn run(args: Args) -> Result<()> {
    let root = args.required("root")?;
    let prepared = args.required("prepared")?;
    let out = args.required("out")?;
    if let Some(parent) = out.parent().filter(|path| !path.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut output = OpenOptions::new().create(true).append(true).open(out)?;
    let threads: usize = args.value("threads", "8").parse()?;
    ensure!(threads > 0, "--threads must be positive");
    let action = args.value("action", "all");
    ensure!(
        ["build", "query", "exercise", "all", "delta"].contains(&action.as_str()),
        "--action must be build,query,exercise,all,or delta"
    );
    ensure!(
        !args.0.contains_key("label-l") || action == "query",
        "--label-l is supported only with --action query"
    );
    ensure!(
        ![
            "query-details-dir",
            "allow-unused-plan-labels",
            "evaluation-scope"
        ]
        .iter()
        .any(|name| args.0.contains_key(*name))
            || action == "query",
        "query details and held-out policy options are supported only with --action query"
    );
    let (rows, dimensions) = header(&prepared.join("base.u8bin"))?;
    ensure!(
        rows > 0 && rows < u32::MAX as usize,
        "base rows must fit stable IDs"
    );
    emit(
        &mut output,
        json!({"kind":"experiment","action":action,"root":root,"prepared":prepared,
        "n_base":rows,"dimensions":dimensions,"index_threads":threads,"beam":4,"shared_cache_nodes":0,
        "storage":if cfg!(target_os="linux") {"native_io_uring"} else {"positional_pread"},
        "cache_bypass":true,"query_scheduling":"dynamic_one","warmup":"full_same_process_same_worker_count",
        "update_scope":if action=="delta" {"synthetic_far_inserts_matching_every_queried_natural_label; exact_distance_proof_required"}
            else {"unqueried_natural_rows_and_fresh_label; supplied_natural_GT_remains_current"}}),
    )?;
    let index = if action == "build" || action == "all" {
        let begin = Instant::now();
        let skip = args.value("skip-label", "200386");
        let skip = if skip == "none" {
            None
        } else {
            Some(skip.parse()?)
        };
        let postings = read_postings(&prepared.join("labels.txt"), rows, skip)?;
        let (read_rows, read_dimensions, vectors) = matrix_u8(&prepared.join("base.u8bin"))?;
        ensure!(
            (rows, dimensions) == (read_rows, read_dimensions),
            "base changed while loading"
        );
        let build = StitchedBuildConfig {
            label_degree: args.value("label-degree", "32").parse()?,
            max_degree: args.value("degree", "128").parse()?,
            l_build: args.value("build-l", "100").parse()?,
            alpha: args.value("alpha", "1.2").parse()?,
            concurrency: threads,
            seed: args.value("seed", "0").parse()?,
        };
        let config = FilteredIndexConfig {
            build,
            pq_chunks: args
                .value("pq-chunks", &dimensions.min(32).to_string())
                .parse()?,
            initial_pq_codebook: args.0.get("pq-codebook").map(PathBuf::from),
            threads,
            delta_limits: DeltaLimits {
                max_records: args.value("delta-records", "10000").parse()?,
                max_bytes: args.value("delta-bytes", "67108864").parse()?,
            },
            ..Default::default()
        };
        let labels = postings.len();
        let memberships: usize = postings.values().map(Vec::len).sum();
        let recorded_config = json!({"label_degree":config.build.label_degree,"max_degree":config.build.max_degree,
            "l_build":config.build.l_build,"alpha":config.build.alpha,"seed":config.build.seed,
            "pq_chunks":config.pq_chunks,"initial_pq_codebook":config.initial_pq_codebook,
            "max_delta_records":config.delta_limits.max_records,
            "max_delta_bytes":config.delta_limits.max_bytes});
        let index = FilteredIndex::build(
            &root,
            rowmajor::Ref::try_from_data(&vectors, rows, dimensions)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?,
            postings,
            config,
        )
        .await?;
        emit(
            &mut output,
            json!({"kind":"build","implementation":"rust_native_lifecycle","rows":rows,
            "dimensions":dimensions,"labels":labels,"memberships":memberships,"skipped_label":skip,
            "threads":threads,"seconds_including_input_load":begin.elapsed().as_secs_f64(),"root":root,
            "config":recorded_config}),
        )?;
        index
    } else {
        FilteredIndex::open(&root, threads)?
    };
    if action == "build" {
        return Ok(());
    }
    let workload = load_workload(&prepared, args.value("limit", "2000").parse()?)?;
    ensure!(
        workload.dimensions == dimensions,
        "base/query dimensions differ"
    );
    if action == "delta" {
        stress_delta(index, &root, workload, &args, &mut output)
    } else if action == "query" {
        benchmark(&index.snapshot(), &workload, &args, "query", &mut output)
    } else {
        exercise(index, &root, rows, &workload, &args, &mut output, threads).await
    }
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let threads: usize = args.value("threads", "8").parse()?;
    ensure!(threads > 0, "--threads must be positive");
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(threads)
        .enable_all()
        .build()?
        .block_on(run(args))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn label_plan_hash_matches_standard_short_and_multiblock_vectors() {
        assert_eq!(
            sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn label_plan_is_complete_bounded_and_loaded_from_exact_file_bytes_once() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("plan.json");
        let bytes = b"{\"9\":64,\"7\":80}\n";
        std::fs::write(&path, bytes).unwrap();
        let queried = BTreeSet::from([7, 9]);
        let plan = LabelLPlan::load(&path, &queried, 64, 10, false).unwrap();
        assert_eq!(actual_l(Some(&plan), 7, 64), 80);
        assert_eq!(actual_l(Some(&plan), 9, 64), 64);
        assert_eq!(actual_l(None, 7, 64), 64);
        assert_eq!(plan.content_sha256, sha256(bytes));
        assert_eq!(
            plan.metadata()["entries"],
            json!([{"label":7,"l":80},{"label":9,"l":64}])
        );
        std::fs::write(&path, b"changed after load").unwrap();
        assert_eq!(actual_l(Some(&plan), 7, 64), 80);
        assert_eq!(plan.content_sha256, sha256(bytes));
        for invalid in [
            r#"{"7":80}"#,
            r#"{"7":80,"9":64,"11":80}"#,
            r#"{"7":63,"9":64}"#,
            r#"{"7":64,"9":1.5}"#,
            r#"{"7":64,"9":4294967296}"#,
        ] {
            std::fs::write(&path, invalid).unwrap();
            assert!(
                LabelLPlan::load(&path, &queried, 64, 10, false).is_err(),
                "accepted {invalid}"
            );
        }
        std::fs::write(&path, bytes).unwrap();
        assert!(LabelLPlan::load(&path, &queried, 8, 10, false).is_err());
    }

    #[test]
    fn frozen_label_plan_allows_only_explicit_complete_queried_subsets() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("frozen.json");
        let bytes = b"{\"7\":80,\"9\":64,\"11\":96}\n";
        std::fs::write(&path, bytes).unwrap();
        let queried = BTreeSet::from([7, 9]);
        assert!(LabelLPlan::load(&path, &queried, 64, 10, false).is_err());
        let plan = LabelLPlan::load(&path, &queried, 64, 10, true).unwrap();
        assert_eq!(plan.unused_labels, [11]);
        assert_eq!(plan.metadata()["unused_plan_labels"], json!([11]));
        assert_eq!(plan.content_sha256, sha256(bytes));
        assert_eq!(actual_l(Some(&plan), 7, 64), 80);
        assert_eq!(plan.entries.len(), 3);
        assert!(LabelLPlan::load(&path, &BTreeSet::from([7, 13]), 64, 10, true).is_err());
        std::fs::write(&path, b"{\"7\":80,\"9\":64,\"11\":63}").unwrap();
        assert!(LabelLPlan::load(&path, &queried, 64, 10, true).is_err());
    }

    #[test]
    fn query_details_preserve_sorted_outputs_counts_and_post_write_hash() {
        let directory = tempdir().unwrap();
        let workload = Workload {
            queries: vec![Query {
                vector: vec![0, 0],
                label: 7,
                truth: vec![2, 4],
                kth_distance: 2.,
            }],
            postings: BTreeMap::from([(7, vec![2, 4, 8])]),
            k: 2,
            dimensions: 2,
            identity: String::new(),
        };
        let observations = vec![Observation {
            query: 0,
            hits: vec![
                FilteredHit {
                    id: 2,
                    distance: 0.,
                },
                FilteredHit {
                    id: 8,
                    distance: 3.,
                },
            ],
            micros: 1234.5,
            sequence: 17,
            exact_fallback: false,
            base_searches: 1,
            delta_candidates: 0,
        }];
        let measurement = QueryMeasurement {
            phase: "query",
            threads: 8,
            l: 64,
            repeat: 0,
            label_l_plan: None,
        };
        let manifest =
            write_query_details(directory.path(), &observations, &workload, &measurement).unwrap();
        let path = Path::new(manifest["path"].as_str().unwrap());
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(manifest["sha256"], sha256(&bytes));
        assert_eq!(manifest["queries"], 1);
        let row: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(row["kind"], "query_detail");
        assert_eq!(row["query_index"], 0);
        assert_eq!(row["label"], 7);
        assert_eq!(row["l"], 64);
        assert_eq!(row["actual_l"], 64);
        assert_eq!(row["ids"], json!([2, 8]));
        assert_eq!(row["distances"], json!([0., 3.]));
        assert_eq!(row["strict_id_hits"], 1);
        assert_eq!(row["latency_us"], 1234.5);
        assert_eq!(row["sequence"], 17);
        assert_eq!(row["exact_fallback"], false);
        assert_eq!(row["base_searches"], 1);
        assert_eq!(row["delta_candidates"], 0);
        assert!(
            write_query_details(directory.path(), &observations, &workload, &measurement).is_err()
        );
    }

    #[test]
    fn delta_grids_are_cumulative_bounded_and_strictly_increasing() {
        assert_eq!(delta_sizes("100,1000,10000").unwrap(), [100, 1000, 10000]);
        for invalid in ["", "0", "10001", "10,10", "100,10", "-1"] {
            assert!(delta_sizes(invalid).is_err(), "accepted {invalid:?}");
        }
    }

    #[test]
    fn far_proof_rejects_a_closer_vector_and_a_kth_distance_tie() {
        let mut workload = Workload {
            queries: vec![Query {
                vector: vec![0, 0],
                label: 7,
                truth: vec![0],
                kth_distance: 2.,
            }],
            postings: BTreeMap::from([(7, vec![0])]),
            k: 1,
            dimensions: 2,
            identity: String::new(),
        };
        assert_eq!(prove_far_delta(&workload).unwrap(), (130050, 130048.));
        workload.queries[0].vector = vec![254, 254];
        assert!(prove_far_delta(&workload).is_err());
        workload.queries[0].vector = vec![255, 255];
        assert!(prove_far_delta(&workload).is_err());
    }

    #[test]
    fn delta_batches_respect_the_serialized_frame_budget() {
        let labels: Vec<_> = (0..64).collect();
        let (size, single) = delta_batch_size(192, &labels, 16 * 1024 * 1024).unwrap();
        assert_eq!(size, 1000);
        assert_eq!(
            delta_batch_size(192, &labels, single as usize).unwrap().0,
            1
        );
        assert!(delta_batch_size(192, &labels, single as usize - 1).is_err());
        let mutations: Vec<_> = (0..size)
            .map(|id| Mutation::Upsert {
                id: id as u32,
                vector: vec![255u8; 192],
                labels: labels.clone(),
            })
            .collect();
        assert!(bincode::serialized_size(&mutations).unwrap() <= 16 * 1024 * 1024);
    }

    #[test]
    fn workload_reads_kth_distances_after_the_complete_id_matrix() {
        let directory = tempdir().unwrap();
        let mut query = Vec::new();
        query.extend_from_slice(&2u32.to_le_bytes());
        query.extend_from_slice(&2u32.to_le_bytes());
        query.extend_from_slice(&[0, 1, 2, 3]);
        std::fs::write(directory.path().join("query.u8bin"), query).unwrap();
        let matrix = |rows: u32, columns: u32, values: &[u32]| {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&rows.to_le_bytes());
            bytes.extend_from_slice(&columns.to_le_bytes());
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            bytes
        };
        std::fs::write(
            directory.path().join("query_labels.u32bin"),
            matrix(2, 1, &[7, 7]),
        )
        .unwrap();
        std::fs::write(
            directory.path().join("label_7.u32bin"),
            matrix(4, 1, &[4, 5, 6, 7]),
        )
        .unwrap();
        let mut truth = matrix(2, 2, &[4, 5, 6, 7]);
        for distance in [0f32, 7., 2., 11.] {
            truth.extend_from_slice(&distance.to_le_bytes());
        }
        let path = directory.path().join("groundtruth.bin");
        std::fs::write(&path, &truth).unwrap();
        let workload = load_workload(directory.path(), 2).unwrap();
        assert_eq!(workload.queries[0].kth_distance, 7.);
        assert_eq!(workload.queries[1].kth_distance, 11.);
        assert_eq!(workload.queries[1].truth, [6, 7]);
        let end = truth.len();
        truth[end - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        std::fs::write(path, truth).unwrap();
        assert!(load_workload(directory.path(), 2).is_err());
    }
}
