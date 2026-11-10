#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Plot audited centroid-plus-spatial-entry V4 results in separate workloads.

Both final inputs are required. Natural-query recall is exposed development;
the fresh atomic-derived partition has a different five-label workload. No
calibration result or quality-only timing can enter either figure.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys

TARGETS = (9, 24, 89, 98, 2495)
ENGINES = ('cpp', 'baseline', 'candidate')
FROZEN_PLAN = 'df42e65440af0a891185913e91f5f22cb6808c9251b2c1d8f6ca968d5bda2e59'
FROZEN_PROTOCOL = 'fe754328702d29a4925d421becd9303ac35888f7b652268cdaf78abe858dbeb3'
FROZEN_BINARY = 'ddd4277e9bb52f4cc42b4b9b66c6927a841ad6a96b534926389c7eb91a86625f'
FROZEN_RUNTIME_ARCHIVE = '0d19211cb5ff2e65a559520414760a49fb0cef742ec53e9335c6df755ad3e2e0'
ENTRY_POLICY = {'mode': 'global_centroid_plus_nearest_pq', 'maximum_initial_seeds': 2,
                'deduplicate_identical_seeds': True, 'original_search_l_unchanged': True,
                'maximum_additional_ssd_seed_reads': 1}
SPECS = {
    'development': {
        'phase': 'development-recheck', 'freshness': 'previously_exposed_development',
        'scope': 'Exposed natural-query development recheck', 'queries': 16600, 'labels': 64,
        'target_counts': {9: 141, 24: 167, 89: 864, 98: 50, 2495: 56},
    },
    'validation': {
        'phase': 'secondary-validation', 'freshness': 'previously_unsearched_partition',
        'scope': 'Fresh dual-entry atomic-label validation; separate five-label shifted workload',
        'queries': 4316, 'labels': 5,
        'target_counts': {9: 811, 24: 177, 89: 3017, 98: 175, 2495: 136},
    },
}
COLORS = {'cpp': '#A6ADB4', 'baseline': '#697783', 'candidate': '#126782'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return type(value) is str and len(value) == 64 and all(character in '0123456789abcdef' for character in value)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def load_summary(path, context):
    summary = json.loads(path.read_text())
    expected = SPECS[context]
    require(summary.get('schema_version') == 4 and summary.get('phase') == expected['phase']
            and summary.get('query_freshness') == expected['freshness']
            and summary.get('scope') == expected['scope'] and summary.get('timing_excluded') is False,
            'Expected an audited V4 timing summary with the correct development/freshness scope; calibration is excluded')
    require(set(summary.get('targets', [])) == set(TARGETS), 'Target label set differs')
    bootstrap = summary.get('bootstrap', {})
    require(bootstrap.get('unit') == 'paired query' and bootstrap.get('replications', 0) >= 10000,
            'Expected at least 10K paired-query bootstrap replications')
    jobs = summary.get('jobs_proof', {})
    orders = jobs.get('orders', [])
    require(jobs.get('blocks') == 6 and len(orders) == 6
            and {tuple(order) for order in orders} == set(itertools.permutations(ENGINES))
            and jobs.get('merged_rows_match_original_aggregates') is True
            and jobs.get('declared_timing_jobs_do_not_overlap') is True,
            'Missing six authoritative serial timing blocks or raw aggregate bindings')
    identity_proof = summary.get('identity_proof', {})
    identity = identity_proof.get('declarations', {})
    require(identity.get('phase') == expected['phase'] and identity.get('query_freshness') == expected['freshness']
            and identity.get('cpu_affinity') == list(range(8))
            and identity.get('plan_before_sha256') == identity.get('plan_after_sha256') == FROZEN_PLAN,
            'Phase, freshness, physical-core affinity or original L policy differs')
    require(identity.get('protocol_version') == 4 and identity.get('entry_policy') == ENTRY_POLICY
            and identity_proof.get('entry_policy') == ENTRY_POLICY
            and identity.get('protocol_source_freeze_sha256') == FROZEN_PROTOCOL
            and identity_proof.get('protocol_source_freeze', {}).get('sha256') == FROZEN_PROTOCOL,
            'Frozen dual-entry policy or protocol source identity differs')
    require(context != 'validation' or identity_proof.get('calibration_identity') is not None,
            'Fresh validation lacks frozen development-candidate binding')
    if context == 'validation':
        claim = identity.get('final_validation_claim', {})
        require(digest(claim.get('sha256')) and claim.get('sha256')
                == identity_proof.get('final_validation_claim', {}).get('sha256'),
                'Fresh validation lacks its exclusive final-launch claim')
    else:
        require(identity.get('final_validation_claim') is None
                and identity_proof.get('final_validation_claim') is None,
                'Exposed natural development must not claim fresh validation')
    entries = identity_proof.get('spatial_entry_proof', {})
    require(entries.get('parent_graph_and_PQ_preserved') is True
            and entries.get('source_membership_preserved') is True
            and entries.get('all_entry_memberships_independently_checked') is True
            and entries.get('labels') == 64 and 64 <= entries.get('entry_count', 0) <= 320,
            'Spatial-entry corpus/publication proof is missing')
    policy = entries.get('policy', {})
    require(policy.get('spatial_clusters') == 4 and policy.get('maximum_entries') == 5
            and policy.get('maximum_fit_members') == 4096 and policy.get('lloyd_iterations') == 8
            and policy.get('global_centroid_first') is True
            and policy.get('sample_seed_base') == 0x6c6162656c5f7633
            and policy.get('sample_seed_mixing') == 'XOR u64(label)',
            'Spatial-entry representative policy differs from the frozen corpus-only policy')
    rows = summary.get('per_label', [])
    require(type(rows) is list and len(rows) == expected['labels'], 'Incorrect label coverage')
    seen = set()
    for row in rows:
        require(type(row.get('label')) is int and row['label'] not in seen
                and type(row.get('queries')) is int and row['queries'] > 0,
                'Invalid or duplicate label/sample count')
        seen.add(row['label'])
        require(type(row.get('rust_l')) is int and row['rust_l'] >= 64 and row.get('cpp_l') == 64,
                'Unexpected search breadth')
        for engine in ENGINES:
            recall = row.get(engine + '_recall_percent')
            require(number(recall) and 0 <= recall <= 100, 'Invalid per-label recall')
            p99 = row.get(engine + '_p99_ms')
            low = row.get(engine + '_p99_range_min_ms')
            high = row.get(engine + '_p99_range_max_ms')
            require(all(number(value) and value > 0 for value in (p99, low, high))
                    and low <= p99 <= high, 'Invalid per-label p99 median or observed range')
        for engine in ('baseline', 'candidate'):
            key = engine + '_minus_cpp'
            mean = row.get(key + '_pp')
            lower = row.get(key + '_marginal_95_lower_pp')
            upper = row.get(key + '_marginal_95_upper_pp')
            require(all(number(value) for value in (mean, lower, upper)) and -100 <= lower <= upper <= 100
                    and close(mean, row[engine + '_recall_percent'] - row['cpp_recall_percent']),
                    'Invalid recall difference or marginal 95% paired-query interval')
    require(sum(row['queries'] for row in rows) == expected['queries'] and set(TARGETS) <= seen,
            'Summary query counts differ from the frozen cohort')
    lookup = {row['label']: row for row in rows}
    require(all(lookup[label]['queries'] == count for label, count in expected['target_counts'].items()),
            'Target sample counts differ from the frozen cohort')
    require(context != 'validation' or seen == set(TARGETS), 'Fresh validation contains unintended labels')
    split = summary.get('split_proof', {})
    require(split.get('all_pool_rows_partitioned_once') is True
            and split.get('calibration_validation_embedding_overlap') == 0
            and split.get('partitions', {}).get('secondary-validation', {}).get('queries') == 4316,
            'Fresh partition is not bound to the precommitted disjoint split')
    engines = summary.get('engines', {})
    declared = identity.get('engines', {})
    require(set(engines) == set(declared) == set(ENGINES), 'Missing measured engine or asset identity')
    require(declared['candidate'].get('binary_sha256') == FROZEN_BINARY
            and declared['candidate'].get('source_archive', {}).get('sha256') == FROZEN_RUNTIME_ARCHIVE,
            'Candidate is not the precommitted V9 centroid-plus-nearest runtime')
    for engine, record in engines.items():
        require(record.get('repetitions') == 6, 'Expected six timing repetitions')
        weighted_recall = sum(row['queries'] * row[engine + '_recall_percent'] for row in rows) / expected['queries'] / 100
        require(number(record.get('strict_id_recall')) and close(record['strict_id_recall'], weighted_recall),
                'Per-label recall does not match measured aggregate recall')
        repeats = record.get('repeat_metrics', [])
        require(len(repeats) == 6 and [row.get('repeat') for row in repeats] == list(range(6)),
                'Timing repetitions are incomplete or duplicated')
        for metric in ('qps', 'p99_us'):
            values = [row.get(metric) for row in repeats]
            bounds = record.get('ranges', {}).get(metric)
            require(number(record.get(metric)) and type(bounds) is list and len(bounds) == 2
                    and all(number(value) and value > 0 for value in (*values, *bounds))
                    and close(record[metric], statistics.median(values))
                    and close(bounds[0], min(values)) and close(bounds[1], max(values)),
                    'Invalid authoritative median, repeat values or observed range')
        assets = declared[engine]
        require(assets.get('index_before') and assets.get('index_before') == assets.get('index_after')
                and digest(assets.get('binary_sha256'))
                and assets.get('binary_before_sha256') == assets.get('binary_sha256'),
                'Measured engine assets changed during the evaluation')
    return summary


def check_pair(development, validation):
    first = development['identity_proof']['declarations']
    second = validation['identity_proof']['declarations']
    require(first.get('host') == second.get('host'), 'Development/validation hosts differ')
    for engine in ENGINES:
        a, b = first['engines'][engine], second['engines'][engine]
        require(a['binary_sha256'] == b['binary_sha256'] and a['index_after'] == b['index_before'],
                'Engine assets differ between exposed natural and fresh atomic validation')
        if engine != 'cpp':
            require(a['source_archive']['sha256'] == b['source_archive']['sha256'],
                    'Native sources differ between contexts')
    first_rows = {row['label']: row for row in development['per_label']}
    require(all(row['rust_l'] == first_rows[row['label']]['rust_l'] for row in validation['per_label']),
            'Per-label L changed between workload contexts')


def setup_matplotlib(path):
    if path:
        require(path.is_dir(), 'Matplotlib dependency directory is unavailable')
        sys.path.insert(0, str(path))
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/diskann-spatial-entry-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 13, 'axes.titleweight': 'bold',
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    return plt


def clean_axis(axis):
    axis.set_facecolor('#FAFBFC')
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_axisbelow(True)
    axis.tick_params(length=0)


def save(figure, output, suffix, sources):
    description = '; '.join(name + ' SHA-256 ' + item['sha256'] for name, item in sources.items())
    paths = [Path(str(output) + '-' + suffix + '.' + extension) for extension in ('png', 'svg')]
    for path in paths:
        figure.savefig(path, dpi=190, metadata={'Title': 'DiskANN spatial-entry evaluation',
                                               'Description': description,
                                               'Creator': 'DiskANN plot_spatial_entries.py'})
    return [{'path': str(path.resolve()), 'sha256': sha256(path)} for path in paths]


def recall_plot(plt, development, validation, output, sources):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    contexts = (development, validation)
    lookup = [{row['label']: row for row in summary['per_label']} for summary in contexts]
    values = [value for rows in lookup for label in TARGETS for engine in ('baseline', 'candidate')
              for value in (rows[label][engine + '_minus_cpp_pp'],
                            rows[label][engine + '_minus_cpp_marginal_95_lower_pp'],
                            rows[label][engine + '_minus_cpp_marginal_95_upper_pp'])]
    lower, upper = min(min(values), 0), max(max(values), 0)
    width = max(upper - lower, .4)
    lower, upper = lower - .12 * width, upper + .12 * width
    figure, axes = plt.subplots(1, 2, figsize=(15.4, 6.8))
    figure.subplots_adjust(left=.10, right=.90, top=.70, bottom=.235, wspace=.54)
    titles = ('Natural queries: development recheck', 'Fresh queries: derived atomic filters')
    for axis, summary, rows, title in zip(axes, contexts, lookup, titles):
        clean_axis(axis)
        axis.set_xlim(lower, upper)
        axis.set_ylim(4.65, -.75)
        axis.axvspan(lower, 0, color='#C64A3A', alpha=.04, zorder=0)
        axis.axvline(0, color='#697580', linewidth=.9, linestyle=(0, (4, 3)))
        axis.grid(axis='x', color='#DDE3E8', linewidth=.7)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
        for position, label in enumerate(TARGETS):
            row = rows[label]
            for engine, offset, marker in (('baseline', -.14, 'o'), ('candidate', .14, 's')):
                key = engine + '_minus_cpp'
                low = row[key + '_marginal_95_lower_pp']
                high = row[key + '_marginal_95_upper_pp']
                y = position + offset
                axis.hlines(y, low, high, color=COLORS[engine], linewidth=1.4)
                axis.plot((low, high), (y, y), linestyle='none', marker='|', markersize=6, color=COLORS[engine])
                axis.plot(row[key + '_pp'], y, linestyle='none', marker=marker, markersize=5.5,
                          color=COLORS[engine], markerfacecolor='white' if engine == 'baseline' else COLORS[engine],
                          markeredgewidth=1.2, zorder=4)
            axis.text(1.025, position + .14, f"{row['candidate_minus_cpp_pp']:+.3f}",
                      transform=axis.get_yaxis_transform(), va='center', ha='left', fontsize=10, color=COLORS['candidate'])
        axis.set_yticks(range(5))
        axis.set_yticklabels([f"Label {label}\nn={rows[label]['queries']:,}; L={rows[label]['rust_l']}" +
                              (' *' if rows[label]['queries'] < 100 else '') for label in TARGETS], fontsize=10)
        axis.tick_params(axis='y', pad=10)
        axis.tick_params(axis='x', pad=7)
        axis.set_xlabel('Recall@10 difference from C++ (percentage points)', labelpad=10, fontsize=10)
        count = sum(row['queries'] for row in summary['per_label'])
        axis.set_title(title + f"\n{count:,} queries; {len(summary['per_label'])} labels evaluated", loc='left', pad=17)
        axis.text(1.025, 1.025, 'New Rust Δ', transform=axis.transAxes, ha='left', fontsize=9, color=COLORS['baseline'])
    figure.suptitle('Recall with centroid + nearest spatial starts', x=.055, y=.975, ha='left',
                   fontsize=21, fontweight='bold', color='#23313C')
    figure.text(.055, .917, 'Five originally declining labels · unchanged L policy · both Rust variants compared with C++ L64',
                fontsize=12, color='#59636A')
    handles = [Line2D([], [], marker='o', linestyle='-', markerfacecolor='white', color=COLORS['baseline'],
                      label='Previous Rust: paired-query 95% CI'),
               Line2D([], [], marker='s', linestyle='-', color=COLORS['candidate'],
                      label='Rust centroid + nearest: paired-query 95% CI')]
    figure.legend(handles=handles, loc='upper left', bbox_to_anchor=(.05, .887), frameon=False,
                  ncol=2, fontsize=11, columnspacing=2)
    figure.text(.10, .805, 'Previously exposed queries; recall is exploratory', fontsize=10.5, color='#A75228')
    figure.text(.585, .805, 'Fresh at freeze; original two-tag filters become one label', fontsize=10, color='#59636A')
    natural_deficits = sum(row['candidate_minus_cpp_pp'] < -1e-9 for row in development['per_label'])
    fresh_deficits = sum(row['candidate_minus_cpp_pp'] < -1e-9 for row in validation['per_label'])
    fresh_old_deficits = sum(row['candidate_recall_percent'] < row['baseline_recall_percent'] - 1e-9
                            for row in validation['per_label'])
    natural_delta = 100 * (development['engines']['candidate']['strict_id_recall']
                           - development['engines']['baseline']['strict_id_recall'])
    figure.text(.055, .151,
                'Intervals are marginal paired-query bootstrap 95% intervals; they do not prove per-label parity.\n'
                f'Natural: {natural_deficits}/64 labels trail C++; recall vs previous Rust {natural_delta:+.3f} pp. Fresh: {fresh_deficits}/5 trail C++, {fresh_old_deficits}/5 trail previous Rust.\n'
                'Fresh atomic-derived results use a different predicate/query mixture and remain separate; natural recall influenced development.\n'
                '* Fewer than 100 queries. Same horizontal scale; the accompanying report discloses all 64 natural-label outcomes.',
                fontsize=9.4, color='#59636A', linespacing=1.4, va='top')
    figure.text(.055, .025, 'Source hashes: ' + ' · '.join(name + ' ' + item['sha256'][:12] + '…' for name, item in sources.items()),
                fontsize=8, color='#87939D')
    outputs = save(figure, output, 'recall', sources)
    plt.close(figure)
    return outputs


def timing_plot(plt, development, validation, output, sources):
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    contexts = (development, validation)
    figure, axes = plt.subplots(2, 2, figsize=(13.1, 10.1))
    figure.subplots_adjust(left=.105, right=.96, top=.795, bottom=.16, hspace=.86, wspace=.38)
    headings = ('Natural queries: exposed development recheck', 'Fresh queries: shifted atomic-label validation')
    for row_index, (summary, heading) in enumerate(zip(contexts, headings)):
        for axis, metric, scale, title in zip(axes[row_index], ('qps', 'p99_us'), (1, 1000),
                                             ('Throughput (queries/second)', 'p99 search service time (ms)')):
            clean_axis(axis)
            values = [summary['engines'][engine][metric] / scale for engine in ENGINES]
            bounds = [[value / scale for value in summary['engines'][engine]['ranges'][metric]] for engine in ENGINES]
            axis.bar(range(3), values, color=[COLORS[engine] for engine in ENGINES], width=.62, zorder=2, alpha=.85)
            axis.errorbar(range(3), values,
                          yerr=([value - interval[0] for value, interval in zip(values, bounds)],
                                [interval[1] - value for value, interval in zip(values, bounds)]),
                          fmt='none', ecolor='#23313C', capsize=4, linewidth=1, zorder=4)
            for position, engine in enumerate(ENGINES):
                repeated = [row[metric] / scale for row in summary['engines'][engine]['repeat_metrics']]
                offsets = [-.14, -.084, -.028, .028, .084, .14]
                axis.scatter([position + offset for offset in offsets], repeated, s=15, color='#23313C',
                             edgecolor='white', linewidth=.4, alpha=.75, zorder=3)
            maximum = max(high for _, high in bounds)
            axis.set_ylim(0, maximum * 1.23)
            for position, value, interval in zip(range(3), values, bounds):
                axis.text(position, interval[1] + maximum * .048,
                          f'{value:,.0f}' if metric == 'qps' else f'{value:.3f}', ha='center', va='bottom',
                          fontsize=11.5, fontweight='bold', color='#23313C')
            axis.set_xticks(range(3), ['C++', 'Previous Rust', 'Rust centroid\n+ nearest'], fontsize=10)
            axis.tick_params(axis='x', pad=9)
            axis.grid(axis='y', color='#DDE3E8', linewidth=.7)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if metric == 'qps':
                axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
            axis.set_title(title, loc='left', pad=11, fontsize=11)
        count = sum(row['queries'] for row in summary['per_label'])
        axes[row_index, 0].text(0, 1.29, heading + f'\n{count:,} queries; {len(summary["per_label"])} labels',
                                transform=axes[row_index, 0].transAxes, fontsize=12, fontweight='bold',
                                color='#A75228' if row_index == 0 else '#23313C', ha='left', linespacing=1.4)
    figure.suptitle('Performance on two separate query mixtures', x=.055, y=.975, ha='left',
                   fontsize=21, fontweight='bold', color='#23313C')
    figure.text(.055, .921, 'Eight physical cores · unchanged L policy · six serial engine-order permutations in each mixture',
                fontsize=11.5, color='#59636A')
    worst = max(validation['per_label'], key=lambda row: row['candidate_p99_ms'] / row['baseline_p99_ms'])
    tail_note = (f"Fresh label {worst['label']} p99 median: candidate {worst['candidate_p99_ms']:.3f} ms; "
                 f"previous Rust {worst['baseline_p99_ms']:.3f}, C++ {worst['cpp_p99_ms']:.3f}; "
                 f"candidate repeat range {worst['candidate_p99_range_min_ms']:.3f}–{worst['candidate_p99_range_max_ms']:.3f} ms.")
    figure.text(.055, .111,
                'Bars: six-repeat medians. Whiskers: observed min–max. Dots: individual repetitions. Each row is a separate workload.\n'
                'Natural queries were exposed during development. Fresh secondary queries use derived atomic labels and a different five-label mixture.\n'
                + tail_note + '\n'
                'Search service times describe this host/workload. At most two initial seeds; total traversal/I/O may change. Timing repeats reuse query vectors.',
                fontsize=9.2, color='#59636A', linespacing=1.4, va='top')
    figure.text(.055, .025, 'Source hashes: ' + ' · '.join(name + ' ' + item['sha256'][:12] + '…' for name, item in sources.items()),
                fontsize=8, color='#87939D')
    outputs = save(figure, output, 'performance', sources)
    plt.close(figure)
    return outputs


def plot(development_path, validation_path, output, dependencies=None, check_only=False, overwrite=False):
    development = load_summary(development_path, 'development')
    validation = load_summary(validation_path, 'validation')
    check_pair(development, validation)
    sources = {name: {'path': str(path.resolve()), 'sha256': sha256(path)} for name, path in
               (('natural_development', development_path), ('fresh_atomic_validation', validation_path))}
    if check_only:
        print(json.dumps({'validated': True, 'sources': sources, 'plots_generated': False}, indent=2))
        return []
    output = output.with_suffix('') if output.suffix.lower() in ('.png', '.svg') else output
    prospective = [Path(str(output) + '-' + suffix + '.' + extension) for suffix in ('recall', 'performance') for extension in ('png', 'svg')]
    proof_path = Path(str(output) + '-provenance.json')
    require(all(path.resolve() not in (development_path.resolve(), validation_path.resolve()) for path in (*prospective, proof_path)),
            'Figure output cannot overwrite an input summary')
    require(overwrite or not any(path.exists() for path in (*prospective, proof_path)), 'Plot output exists; use --overwrite to replace these figures')
    output.parent.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib(dependencies)
    outputs = recall_plot(plt, development, validation, output, sources)
    outputs += timing_plot(plt, development, validation, output, sources)
    proof = {'schema_version': 1, 'sources': sources, 'plotter': {'path': str(Path(__file__).resolve()), 'sha256': sha256(Path(__file__))},
             'outputs': outputs, 'target_labels': list(TARGETS), 'development_queries': 16600, 'fresh_validation_queries': 4316,
             'natural_recall_scope': 'Previously exposed development; exploratory recall',
             'fresh_recall_scope': 'Previously unsearched atomic-derived five-label partition; explicit distribution shift',
             'entry_policy': ENTRY_POLICY,
             'recall_uncertainty': 'Marginal 95% paired-query bootstrap intervals for previous and centroid-plus-nearest Rust versus C++',
             'timing_uncertainty': 'Six-repeat medians, observed min-max ranges, individual dots; no pooled workloads or per-label QPS'}
    proof_path.write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({'provenance': str(proof_path), 'outputs': outputs}, indent=2))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-summary', type=Path, required=True)
    parser.add_argument('--validation-summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='Output prefix for recall/performance PNG/SVG figures')
    parser.add_argument('--matplotlib-path', type=Path, default=Path('/private/tmp/diskann-heldout-plot-deps') if Path('/private/tmp/diskann-heldout-plot-deps').is_dir() else None)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    plot(args.development_summary.resolve(), args.validation_summary.resolve(), args.out.resolve(),
         dependencies=args.matplotlib_path, check_only=args.check_only, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
