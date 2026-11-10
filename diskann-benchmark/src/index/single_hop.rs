/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! An opt-in construction and comparison job for atomic string labels.

use std::{collections::BTreeMap, io::Write, num::NonZeroUsize, sync::Arc};

use diskann_benchmark_core::{
    build,
    recall::GroundTruthMode,
    search::{
        self,
        graph::{filtered::FilteredSearch, InlineFilterSearch, KnnParams, MultiHop, Strategy},
    },
};
use diskann_benchmark_runner::{
    benchmark::{MatchContext, Score},
    files::InputFile,
    Benchmark, Checker, Checkpoint, Registry,
};
use diskann_disk::build::filtered::{build_stitched, StitchedBuildConfig};
use diskann_label_filter::{parse_query_filter, read_queries, ASTExpr, CompareOp};
use diskann_tools::utils::filtered_labels::load_labels;
use diskann_utils::views::rowmajor::{self, Matrix};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(
    tag = "type",
    content = "path",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum QueryFilters {
    QueryPredicates(InputFile),
    QueryLabels(InputFile),
}

impl QueryFilters {
    fn path_mut(&mut self) -> &mut InputFile {
        match self {
            Self::QueryPredicates(path) | Self::QueryLabels(path) => path,
        }
    }
}

use crate::{
    index::result::SearchResults,
    inputs::Example,
    utils::datafiles::{self, BinFile},
};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    data: InputFile,
    data_labels: InputFile,
    label_field: String,
    queries: InputFile,
    query_filters: QueryFilters,
    groundtruth: InputFile,
    label_degree: usize,
    max_degree: usize,
    l_build: usize,
    alpha: f32,
    build_threads: usize,
    seed: u64,
    k: usize,
    search_l: Vec<usize>,
    threads: Vec<NonZeroUsize>,
    reps: NonZeroUsize,
}

impl diskann_benchmark_runner::Input for Input {
    type Raw = Self;
    fn tag() -> &'static str {
        Self::tag()
    }
    fn from_raw(mut raw: Self::Raw, checker: &mut Checker) -> anyhow::Result<Self> {
        raw.validate(checker)?;
        Ok(raw)
    }
    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }
    fn example() -> Self {
        <Self as Example>::example()
    }
}

impl Input {
    fn tag() -> &'static str {
        "experimental-single-hop"
    }

    fn build_config(
        &self,
    ) -> Result<StitchedBuildConfig, diskann_disk::build::filtered::ConfigError> {
        StitchedBuildConfig::new(
            self.label_degree,
            self.max_degree,
            self.l_build,
            self.alpha,
            self.build_threads,
            self.seed,
        )
    }

    fn validate(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        for path in [
            &mut self.data,
            &mut self.data_labels,
            &mut self.queries,
            self.query_filters.path_mut(),
            &mut self.groundtruth,
        ] {
            path.resolve(checker)?;
        }
        self.build_config()?;
        anyhow::ensure!(
            self.label_field.starts_with('/'),
            "label_field must be a JSON pointer"
        );
        anyhow::ensure!(
            !self.search_l.is_empty() && !self.threads.is_empty(),
            "search sweeps cannot be empty"
        );
        for &l in &self.search_l {
            KnnParams::new(self.k, l)?;
        }
        Ok(())
    }
}

impl Example for Input {
    fn example() -> Self {
        Self {
            data: InputFile::new("data.f32.bin"),
            data_labels: InputFile::new("labels.jsonl"),
            label_field: "/label".into(),
            queries: InputFile::new("queries.f32.bin"),
            query_filters: QueryFilters::QueryPredicates(InputFile::new("query-predicates.jsonl")),
            groundtruth: InputFile::new("filtered-truth.bin"),
            label_degree: 16,
            max_degree: 64,
            l_build: 100,
            alpha: 1.2,
            build_threads: 4,
            seed: 0,
            k: 10,
            search_l: vec![20, 40, 80],
            threads: vec![NonZeroUsize::MIN],
            reps: NonZeroUsize::new(3).unwrap(),
        }
    }
}

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register(Input::tag(), Job)?;
    Ok(())
}

#[derive(Debug)]
struct Job;

#[derive(Debug, Serialize)]
struct Results {
    single_hop: Comparison,
    multihop: Comparison,
    inline: Comparison,
}

#[derive(Debug, Serialize)]
struct Comparison {
    graph: &'static str,
    build_seconds: f64,
    prepare_seconds: f64,
    runs: Vec<SearchResults>,
}

impl Benchmark for Job {
    type Input = Input;
    type Output = Results;
    fn try_match(&self, _: &Input, context: &MatchContext) -> Score {
        context.success(1)
    }
    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Experimental stitched construction and filtered traversal comparison")
    }
    fn run(
        &self,
        input: &Input,
        _: Checkpoint<'_>,
        output: &mut dyn diskann_benchmark_runner::Output,
    ) -> anyhow::Result<Results> {
        let vectors: Arc<rowmajor::Owned<f32>> =
            Arc::new(datafiles::load_dataset(BinFile(&input.data))?);
        let queries: Arc<rowmajor::Owned<f32>> =
            Arc::new(datafiles::load_dataset(BinFile(&input.queries))?);
        anyhow::ensure!(
            vectors.ncols() == queries.ncols(),
            "query dimension mismatch"
        );
        anyhow::ensure!(queries.nrows() > 0, "queries cannot be empty");
        let (names, postings) =
            load_labels(&*input.data_labels, vectors.nrows(), &input.label_field)?.into_parts();
        let query_labels = load_query_labels(
            &input.query_filters,
            &input.label_field,
            queries.nrows(),
            &names,
        )?;
        let truth = datafiles::load_range_groundtruth(BinFile(&input.groundtruth))?;
        anyhow::ensure!(
            truth.len() == queries.nrows(),
            "groundtruth row count mismatch"
        );
        let runtime = diskann_benchmark_core::tokio::runtime(input.build_threads)?;
        let start = std::time::Instant::now();
        let graph = runtime.block_on(build_stitched(
            vectors.as_view(),
            postings.iter().map(|(&l, ids)| (l, ids.as_slice())),
            &input.build_config()?,
        ))?;
        let build_seconds = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let index = Arc::new(runtime.block_on(import_graph(&vectors, &graph))?);
        let labels = diskann_inmem::provider::filtered::LabelIndex::new(
            index.provider(),
            &postings,
            graph.entry_points(),
        )?;
        let prepare_seconds = start.elapsed().as_secs_f64();
        drop(graph);
        let strategies = query_labels
            .iter()
            .map(|&label| labels.strategy(&label))
            .collect::<Vec<_>>();
        let single_hop = measure(
            FilteredSearch::new(index, queries.clone(), Strategy::collection(strategies))?,
            Comparison::new("stitched_vamana", build_seconds, prepare_seconds),
            "single_hop",
            input,
            &truth,
            output,
        )?;
        // The baseline algorithms use the ordinary builder and its global virtual start
        // point. They do not inherit label-specific entry points from the stitched graph.
        let start = std::time::Instant::now();
        let ordinary = build_ordinary(vectors.clone(), input, &runtime)?;
        let build_seconds = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let filters = ordinary_filters(ordinary.provider(), &query_labels, &postings)?;
        let prepare_seconds = start.elapsed().as_secs_f64();
        let multihop = measure(
            MultiHop::new(
                ordinary.clone(),
                queries.clone(),
                Strategy::broadcast(diskann_inmem::Strategy),
                filters.clone(),
            )?,
            Comparison::new("vamana", build_seconds, prepare_seconds),
            "multihop",
            input,
            &truth,
            output,
        )?;
        let inline = measure(
            InlineFilterSearch::new(
                ordinary,
                queries,
                Strategy::broadcast(diskann_inmem::Strategy),
                filters,
                None,
            )?,
            Comparison::new("vamana", build_seconds, prepare_seconds),
            "inline",
            input,
            &truth,
            output,
        )?;
        Ok(Results {
            single_hop,
            multihop,
            inline,
        })
    }
}

fn measure<S>(
    searcher: Arc<S>,
    mut comparison: Comparison,
    algorithm: &str,
    input: &Input,
    truth: &dyn diskann_benchmark_core::recall::Rows<u32>,
    mut output: &mut dyn diskann_benchmark_runner::Output,
) -> anyhow::Result<Comparison>
where
    S: search::Search<Id = u32, Parameters = KnnParams, Output = search::graph::knn::Metrics>,
{
    let mut runs = Vec::new();
    for &threads in &input.threads {
        for &l in &input.search_l {
            runs.push(search::Run::new(
                KnnParams::new(input.k, l)?,
                search::Setup {
                    threads,
                    tasks: threads,
                    reps: input.reps,
                },
            ));
        }
    }
    let summaries = search::search_all(
        searcher,
        runs,
        search::graph::knn::Aggregator::new(truth, input.k, input.k, GroundTruthMode::Flexible),
    )?;
    for row in &summaries {
        writeln!(
            output,
            "{algorithm} on {}: L={} recall={:.4} comparisons={:.1} hops={:.1}",
            comparison.graph,
            row.parameters.knn.l_value(),
            row.recall.average,
            row.mean_cmps,
            row.mean_hops
        )?;
    }
    comparison.runs = summaries.into_iter().map(SearchResults::new).collect();
    Ok(comparison)
}

impl Comparison {
    fn new(graph: &'static str, build_seconds: f64, prepare_seconds: f64) -> Self {
        Self {
            graph,
            build_seconds,
            prepare_seconds,
            runs: Vec::new(),
        }
    }
}

type Postings = BTreeMap<u32, Vec<u32>>;

fn load_query_labels(
    filters: &QueryFilters,
    field: &str,
    rows: usize,
    names: &BTreeMap<String, u32>,
) -> anyhow::Result<Vec<u32>> {
    let labels = match filters {
        QueryFilters::QueryLabels(path) => {
            let labels: Vec<String> =
                serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(&**path)?))?;
            anyhow::ensure!(labels.len() == rows, "query label count mismatch");
            labels
        }
        QueryFilters::QueryPredicates(path) => {
            let predicates = read_queries(&**path)?;
            anyhow::ensure!(predicates.len() == rows, "query predicate count mismatch");
            let mut labels = vec![None; rows];
            for predicate in predicates {
                let id = predicate.query_id;
                anyhow::ensure!(
                    id < rows && labels[id].is_none(),
                    "duplicate or out-of-range query_id {id}"
                );
                // The upstream parser ignores some non-operator fields; validate the
                // complete expression first so unsupported predicates never get weakened.
                let object = predicate
                    .filter
                    .as_object()
                    .ok_or_else(|| anyhow::anyhow!("query {id}: filter must be an object"))?;
                let condition = object
                    .iter()
                    .next()
                    .filter(|(name, _)| predicate_field(name) == field)
                    .and_then(|(_, value)| value.as_object());
                anyhow::ensure!(
                    object.len() == 1
                        && condition.is_some_and(|c| c.len() == 1
                            && c.get("$eq").is_some_and(serde_json::Value::is_string)),
                    "query {id}: single-hop requires exactly {{\"{field}\":{{\"$eq\":\"label\"}}}}; compound, range and other-field predicates are unsupported"
                );
                match parse_query_filter(&predicate.filter)? {
                    ASTExpr::Compare {
                        field: actual,
                        op: CompareOp::Eq(serde_json::Value::String(name)),
                    } if predicate_field(&actual) == field => labels[id] = Some(name),
                    _ => anyhow::bail!("query {id}: expected atomic string equality on {field}"),
                }
            }
            labels
                .into_iter()
                .map(|name| name.ok_or_else(|| anyhow::anyhow!("missing query ID")))
                .collect::<anyhow::Result<Vec<_>>>()?
        }
    };
    Ok(labels
        .iter()
        .map(|name| names.get(name).copied().unwrap_or(u32::MAX))
        .collect())
}

// Match the upstream query parser's dot notation as well as explicit JSON pointers.
fn predicate_field(field: &str) -> String {
    if field.starts_with('/') {
        field.to_owned()
    } else {
        field
            .split('.')
            .map(|part| format!("/{}", part.replace('~', "~0").replace('/', "~1")))
            .collect()
    }
}

type InMemoryIndex =
    diskann::graph::DiskANNIndex<diskann_inmem::Provider<diskann_inmem::repr::Full<f32>, u32>>;
type QueryMembership = Arc<[Arc<dyn diskann::graph::ext::labeled::QueryLabelProvider<u32>>]>;

fn ordinary_filters(
    provider: &diskann_inmem::Provider<diskann_inmem::repr::Full<f32>, u32>,
    queries: &[u32],
    postings: &Postings,
) -> anyhow::Result<QueryMembership> {
    use diskann::provider::DataProvider;
    // Translate once after construction. Frozen benchmark providers cannot invalidate
    // these IDs. A virtual start point intentionally does not match any label.
    let mut filters = BTreeMap::new();
    for (label, members) in postings {
        let mut bitmap = bit_set::BitSet::new();
        for id in members {
            bitmap.insert(provider.to_internal_id(&diskann_inmem::Context, id)? as usize);
        }
        filters.insert(
            *label,
            crate::utils::filters::as_query_label_provider(bitmap),
        );
    }
    let empty = crate::utils::filters::as_query_label_provider(bit_set::BitSet::new());
    Ok(queries
        .iter()
        .map(|label| filters.get(label).unwrap_or(&empty).clone())
        .collect())
}

fn build_ordinary(
    vectors: Arc<rowmajor::Owned<f32>>,
    input: &Input,
    runtime: &tokio::runtime::Runtime,
) -> anyhow::Result<Arc<InMemoryIndex>> {
    use diskann_inmem::{
        num::{Capacity, MaxDegree},
        repr::full::Config,
        Provider,
    };
    use diskann_vector::distance::Metric;
    let mut builder = diskann::graph::config::Builder::new(
        input.max_degree,
        diskann::graph::config::MaxDegree::Same,
        input.l_build,
        Metric::L2.into(),
    );
    builder.alpha(input.alpha);
    let config = builder.build()?;
    // A dataset centroid is the ordinary Vamana global start point.
    let mut centroid = vec![0.0f64; vectors.ncols()];
    for row in vectors.rows() {
        for (sum, &value) in centroid.iter_mut().zip(row) {
            *sum += f64::from(value);
        }
    }
    let centroid = centroid
        .into_iter()
        .map(|x| (x / vectors.nrows() as f64) as f32)
        .collect::<Vec<_>>();
    let provider = Provider::new(Config::new(
        Capacity::new(vectors.nrows()),
        MaxDegree::new(config.max_degree().get()),
        Metric::L2,
        rowmajor::Owned::row_vector(centroid.into()),
    )?)?;
    let index = Arc::new(diskann::graph::DiskANNIndex::new(config, provider, None));
    build::build(
        build::graph::SingleInsert::new(
            index.clone(),
            vectors,
            diskann_inmem::Strategy,
            build::ids::Identity::<u32>::new(),
        ),
        build::Parallelism::dynamic(
            NonZeroUsize::MIN,
            NonZeroUsize::new(input.build_threads).unwrap(),
        ),
        runtime,
    )?;
    Ok(index)
}

// Import through provider mappings; graph row IDs are external construction IDs.
async fn import_graph(
    vectors: &rowmajor::Owned<f32>,
    graph: &diskann_disk::build::filtered::StitchedGraph,
) -> anyhow::Result<
    diskann::graph::DiskANNIndex<diskann_inmem::Provider<diskann_inmem::repr::Full<f32>, u32>>,
> {
    use diskann::{
        graph::glue::PruneStrategy,
        provider::{DataProvider, NeighborAccessorMut, SetElement},
    };
    use diskann_inmem::{
        num::{Capacity, MaxDegree},
        repr::full::Config,
        Context, Provider, Strategy,
    };
    let provider = Provider::new(Config::new(
        Capacity::new(vectors.nrows()),
        MaxDegree::new(graph.max_degree() as usize),
        diskann_vector::distance::Metric::L2,
        rowmajor::Owned::from_element(0, vectors.ncols(), 0.0),
    )?)?;
    for id in 0..vectors.nrows() {
        provider
            .set_element(&Context, &(id as u32), vectors.row(id))
            .await?;
    }
    {
        let mut accessor = Strategy.prune_accessor(&provider, &Context, 0)?;
        for (id, neighbors) in graph.neighbors().iter().enumerate() {
            let internal = provider.to_internal_id(&Context, &(id as u32))?;
            let translated = neighbors
                .iter()
                .map(|id| provider.to_internal_id(&Context, id))
                .collect::<Result<Vec<_>, _>>()?;
            accessor.set_neighbors(internal, &translated).await?;
        }
    }
    let config = diskann::graph::config::Builder::new(
        graph.max_degree() as usize,
        diskann::graph::config::MaxDegree::Same,
        100,
        diskann_vector::distance::Metric::L2.into(),
    )
    .build()?;
    Ok(diskann::graph::DiskANNIndex::new(config, provider, None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_benchmark_runner::{app::Commands, output::Memory};
    use rand::{Rng, SeedableRng};

    #[test]
    fn constructed_graph_benchmark_compares_traversals() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let mut rng = rand::rngs::StdRng::seed_from_u64(139);
        let data: Vec<f32> = (0..384 * 8).map(|_| rng.random_range(-1.0..1.0)).collect();
        let vectors = rowmajor::Owned::try_from_data(data.into(), 384, 8).unwrap();
        diskann_utils::io::write_bin(
            vectors.as_view(),
            &mut std::fs::File::create(root.join("data.bin")).unwrap(),
        )
        .unwrap();
        let names = ["blue", "red\n\"escaped\"", "green"];
        let mut labels = std::fs::File::create(root.join("labels.jsonl")).unwrap();
        for id in 0..vectors.nrows() {
            writeln!(
                labels,
                "{}",
                serde_json::json!({"doc_id": id, "label": if id % 4 == 0 { serde_json::json!(names[id % 3]) } else { serde_json::json!([names[id % 3], names[(id + 1) % 3], names[id % 3]]) }})
            )
            .unwrap();
        }
        let queries = rowmajor::Owned::try_from_data(
            (0..13)
                .flat_map(|i| vectors.row(i * 29).iter().map(|x| x + 0.001))
                .collect::<Vec<_>>()
                .into(),
            13,
            8,
        )
        .unwrap();
        diskann_utils::io::write_bin(
            queries.as_view(),
            &mut std::fs::File::create(root.join("queries.bin")).unwrap(),
        )
        .unwrap();
        let query_labels: Vec<_> = (0..13)
            .map(|i| {
                if i == 12 {
                    "missing"
                } else {
                    names[(i * 29) % 3]
                }
            })
            .collect();
        let mut predicates = std::fs::File::create(root.join("query-predicates.jsonl")).unwrap();
        for (id, name) in query_labels.iter().enumerate().rev() {
            writeln!(
                predicates,
                "{}",
                serde_json::json!({"query_id":id,"filter":{"/label":{"$eq":name}}})
            )
            .unwrap();
        }
        let truth: Vec<Vec<u32>> = (0..13)
            .map(|q| {
                if q == 12 {
                    return Vec::new();
                }
                let mut distances: Vec<_> = (0..384)
                    .filter(|id| {
                        names[id % 3] == query_labels[q]
                            || (id % 4 != 0 && names[(id + 1) % 3] == query_labels[q])
                    })
                    .map(|id| {
                        let d: f32 = vectors
                            .row(id)
                            .iter()
                            .zip(queries.row(q))
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum();
                        (id as u32, d)
                    })
                    .collect();
                distances.sort_by(|a, b| a.1.total_cmp(&b.1));
                distances.into_iter().take(5).map(|(id, _)| id).collect()
            })
            .collect();
        let mut truth_file = std::fs::File::create(root.join("truth.bin")).unwrap();
        for value in [13u32, truth.iter().map(Vec::len).sum::<usize>() as u32]
            .into_iter()
            .chain(truth.iter().map(|v| v.len() as u32))
            .chain(truth.iter().flatten().copied())
        {
            truth_file.write_all(&value.to_le_bytes()).unwrap();
        }
        let mut input = Input::example();
        input.data = InputFile::new(root.join("data.bin"));
        input.data_labels = InputFile::new(root.join("labels.jsonl"));
        input.queries = InputFile::new(root.join("queries.bin"));
        input.query_filters =
            QueryFilters::QueryPredicates(InputFile::new(root.join("query-predicates.jsonl")));
        input.groundtruth = InputFile::new(root.join("truth.bin"));
        input.k = 5;
        input.search_l = vec![32, 64];
        input.label_degree = 12;
        input.max_degree = 32;
        input.build_threads = 2;
        input.reps = NonZeroUsize::MIN;
        let input_path = root.join("input.json");
        std::fs::write(
            &input_path,
            serde_json::to_vec(&serde_json::json!({
                "search_directories": [root],
                "jobs": [{"type": Input::tag(), "content": input}],
            }))
            .unwrap(),
        )
        .unwrap();
        let output_path = root.join("output.json");
        let cli = crate::Cli::from_commands(
            Commands::Run {
                input_file: input_path,
                output_file: output_path.clone(),
                dry_run: false,
                allow_debug: true,
            },
            true,
        );
        let mut output = Memory::new();
        cli.run(&mut output).unwrap();
        println!("{}", String::from_utf8(output.into_inner()).unwrap());
        let result: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output_path).unwrap()).unwrap();
        let result = &result[0]["results"];
        for algorithm in ["single_hop", "multihop", "inline"] {
            assert_eq!(
                result[algorithm]["graph"],
                if algorithm == "single_hop" {
                    "stitched_vamana"
                } else {
                    "vamana"
                }
            );
            let runs = result[algorithm]["runs"].as_array().unwrap();
            assert_eq!(runs.len(), 2);
            for run in runs {
                assert!(run["recall"]["average"].as_f64().unwrap() >= 0.95, "{run}");
                assert!(run["mean_cmps"].as_f64().unwrap() > 0.0);
                assert!(run["mean_hops"].as_f64().unwrap() > 0.0);
            }
        }
    }

    #[test]
    fn query_filter_formats_and_validation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("queries.json");
        let names = BTreeMap::from([("red\n\"escaped\"".to_owned(), 7)]);
        std::fs::write(
            &path,
            serde_json::to_vec(&["red\n\"escaped\"", "missing"]).unwrap(),
        )
        .unwrap();
        let filters = QueryFilters::QueryLabels(InputFile::new(&path));
        assert_eq!(
            load_query_labels(&filters, "/label", 2, &names).unwrap(),
            [7, u32::MAX]
        );
        assert!(load_query_labels(&filters, "/label", 1, &names).is_err());

        let filters = QueryFilters::QueryPredicates(InputFile::new(&path));
        let predicate = |id, filter| serde_json::json!({"query_id": id, "filter": filter});
        let write = |values: Vec<serde_json::Value>| {
            std::fs::write(
                &path,
                values.iter().map(|v| format!("{v}\n")).collect::<String>(),
            )
            .unwrap();
        };
        write(vec![
            predicate(1, serde_json::json!({"label":{"$eq":"missing"}})),
            predicate(0, serde_json::json!({"/label":{"$eq":"red\n\"escaped\""}})),
        ]);
        assert_eq!(
            load_query_labels(&filters, "/label", 2, &names).unwrap(),
            [7, u32::MAX]
        );
        assert!(load_query_labels(&filters, "/label", 1, &names).is_err());
        for filter in [
            serde_json::json!({"/label":{"$gt":3}}),
            serde_json::json!({"/label":{"$eq":7}}),
            serde_json::json!({"/label":{"$eq":["red"]}}),
            serde_json::json!({"/other":{"$eq":"red"}}),
            serde_json::json!({"$or":[{"/label":{"$eq":"red"}},{"/label":{"$eq":"blue"}}]}),
            serde_json::json!({"/label":{"$eq":"red"},"ignored":true}),
            serde_json::json!({"/label":{"$eq":"red","$ne":"blue"}}),
            serde_json::json!(null),
        ] {
            write(vec![predicate(0, filter)]);
            assert!(load_query_labels(&filters, "/label", 1, &names).is_err());
        }
        let valid = serde_json::json!({"/label":{"$eq":"red"}});
        write(vec![predicate(1, valid.clone())]);
        assert!(load_query_labels(&filters, "/label", 1, &names).is_err());
        write(vec![predicate(0, valid.clone()), predicate(0, valid)]);
        assert!(load_query_labels(&filters, "/label", 2, &names).is_err());
    }

    #[test]
    fn input_rejects_unknown_fields() {
        let mut raw = serde_json::to_value(Input::example()).unwrap();
        raw["starting_points"] = serde_json::json!([0]);
        assert!(serde_json::from_value::<Input>(raw).is_err());
    }
}
