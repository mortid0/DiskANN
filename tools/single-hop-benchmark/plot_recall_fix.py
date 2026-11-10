#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Plot only audited fresh-validation results for the unchanged-L recall fix.

Writes separate target-label recall and performance figures. Three-context mode
keeps candidate1 natural validation, candidate2 exposed natural development, and
candidate2 fresh atomic validation distinct. Quality-only timings are excluded.
Matplotlib is loaded only when rendering; --check-only performs no plotting.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

TARGETS = (9, 24, 89, 98, 2495)
ENGINES = ('cpp', 'baseline', 'candidate')
FROZEN_PLAN = 'df42e65440af0a891185913e91f5f22cb6808c9251b2c1d8f6ca968d5bda2e59'
SCOPES = {
    'main': 'Fresh sealed natural query validation; original L policy unchanged',
    'supplemental': 'Supplemental shifted atomic-label validation; separate from natural one-tag workload',
}
COLORS = {'cpp': '#A6ADB4', 'baseline': '#596773', 'candidate': '#126782'}
LABELS = {'cpp': 'C++', 'baseline': 'Previous Rust', 'candidate': 'Updated Rust'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def load_summary(path, supplemental=False):
    result = json.loads(path.read_text())
    phase = 'secondary-validation' if supplemental else 'validation'
    scope = SCOPES['supplemental' if supplemental else 'main']
    require(result.get('schema_version') == 1 and result.get('scope') == scope
            and result.get('timing_excluded') is supplemental,
            'Expected an audited final validation summary; calibration summaries cannot be plotted')
    require(set(result.get('targets', [])) == set(TARGETS), 'Target label set differs')
    bootstrap = result.get('bootstrap', {})
    require(bootstrap.get('unit') == 'paired query' and bootstrap.get('replications', 0) >= 10000,
            'Expected at least 10K paired-query bootstrap replications')
    jobs = result.get('jobs_proof', {})
    require(jobs.get('blocks') == (1 if supplemental else 6)
            and jobs.get('merged_rows_match_original_aggregates') is True
            and jobs.get('declared_timing_jobs_do_not_overlap') is True,
            'Missing validated serial timing/job bindings')
    evidence = result.get('identity_proof', {}).get('declared_identity', {})
    require(evidence.get('phase') == phase and evidence.get('cpu_affinity') == list(range(8))
            and evidence.get('plan_before_sha256') == evidence.get('plan_after_sha256') == FROZEN_PLAN,
            'Evaluation phase, eight-core affinity or original L policy differs')
    require(result.get('identity_proof', {}).get('candidate_freeze_declaration', {}).get('phase') == phase,
            'Candidate freeze phase differs')
    rows = result.get('per_label', [])
    require(type(rows) is list and len(rows) == (5 if supplemental else 64),
            'Expected all 64 natural labels or exactly five supplemental labels')
    seen = set()
    for row in rows:
        require(type(row.get('label')) is int and row['label'] not in seen
                and type(row.get('queries')) is int and row['queries'] > 0,
                'Invalid label identity or sample count')
        seen.add(row['label'])
        require(type(row.get('rust_l')) is int and row['rust_l'] >= 64 and row.get('cpp_l') == 64,
                'Unexpected search breadth')
        for engine in ENGINES:
            recall = row.get(engine + '_recall_percent')
            require(number(recall) and 0 <= recall <= 100, 'Invalid per-label recall')
        for engine in ('baseline', 'candidate'):
            contrast = engine + '_minus_cpp'
            mean = row.get(contrast + '_pp')
            low = row.get(contrast + '_marginal_95_lower_pp')
            high = row.get(contrast + '_marginal_95_upper_pp')
            require(all(number(value) for value in (mean, low, high)) and -100 <= low <= high <= 100
                    and close(mean, row[engine + '_recall_percent'] - row['cpp_recall_percent']),
                    'Invalid recall delta or marginal 95% confidence interval')
    require(set(TARGETS) <= seen and (not supplemental or seen == set(TARGETS)), 'Missing target label')
    engines = result.get('engines', {})
    require(set(engines) == set(ENGINES), 'Missing benchmark engine')
    query_count = sum(row['queries'] for row in rows)
    proof = result.get('fresh_query_proof', {})
    coverage = proof.get('supplemental', {}) if supplemental else proof
    require(coverage.get('queries' if supplemental else 'validation_queries') == query_count,
            'Per-label query counts differ from audited fresh-query coverage')
    for engine, record in engines.items():
        require(record.get('repetitions') == (1 if supplemental else 6), 'Unexpected engine repetition count')
        aggregate = sum(row['queries'] * row[engine + '_recall_percent'] for row in rows) / query_count / 100
        require(number(record.get('strict_id_recall')) and close(aggregate, record['strict_id_recall']),
                'Per-label recall does not match aggregate')
        if supplemental:
            require(not any(key in record for key in ('qps', 'p99_us', 'repeat_metrics')),
                    'Supplemental timing must be excluded')
            continue
        repeats = record.get('repeat_metrics', [])
        require(len(repeats) == 6 and [item.get('repeat') for item in repeats] == list(range(6)),
                'Expected six distinct timing repetitions')
        for metric in ('qps', 'p99_us'):
            value = record.get(metric)
            limits = record.get('ranges', {}).get(metric)
            values = [item.get(metric) for item in repeats]
            require(number(value) and value > 0 and type(limits) is list and len(limits) == 2
                    and all(number(item) and item > 0 for item in (*limits, *values))
                    and close(statistics.median(values), value)
                    and close(min(values), limits[0]) and close(max(values), limits[1]),
                    f'Invalid authoritative {metric} median/range')
    return result


def validate_pair(main, supplemental):
    first = main['identity_proof']['declared_identity']
    second = supplemental['identity_proof']['declared_identity']
    for engine in ENGINES:
        a, b = first['engines'][engine], second['engines'][engine]
        require(a['binary_sha256'] == b['binary_sha256'] and a['index_after'] == b['index_before'],
                'Engine assets differ between natural and supplemental validation')
    main_rows = {row['label']: row for row in main['per_label']}
    for row in supplemental['per_label']:
        require(row['rust_l'] == main_rows[row['label']]['rust_l'],
                'Actual per-label search breadth differs between workloads')


def style(plt):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 13, 'axes.titleweight': 'bold',
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})


def clean_axis(axis):
    axis.set_facecolor('#FAFBFC')
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_axisbelow(True)
    axis.tick_params(length=0)


def save(figure, output, suffix, sources):
    paths = [Path(str(output) + '-' + suffix + '.' + extension) for extension in ('png', 'svg')]
    description = '; '.join(name + ' SHA-256 ' + value['sha256'] for name, value in sources.items())
    for path in paths:
        figure.savefig(path, dpi=190, metadata={'Title': 'DiskANN unchanged-L recall-fix validation',
                                               'Description': description,
                                               'Creator': 'DiskANN plot_recall_fix.py'})
    return [{'path': str(path.resolve()), 'sha256': sha256(path)} for path in paths]


def recall_plot(plt, main, supplemental, output, sources):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    contexts = (main, supplemental)
    rows_by_context = [{row['label']: row for row in summary['per_label']} for summary in contexts]
    values = [value for rows in rows_by_context for label in TARGETS for value in
              (rows[label]['baseline_minus_cpp_pp'], rows[label]['candidate_minus_cpp_pp'],
               rows[label]['candidate_minus_cpp_marginal_95_lower_pp'],
               rows[label]['candidate_minus_cpp_marginal_95_upper_pp'])]
    lower = min(min(values), 0)
    upper = max(max(values), 0)
    span = max(upper - lower, .4)
    lower, upper = lower - .12 * span, upper + .12 * span
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.3))
    figure.subplots_adjust(left=.095, right=.89, top=.73, bottom=.235, wspace=.54)
    subtitles = ('Natural one-tag validation', 'Supplemental derived atomic labels')
    for axis, rows, summary, title in zip(axes, rows_by_context, contexts, subtitles):
        clean_axis(axis)
        axis.set_xlim(lower, upper)
        axis.set_ylim(4.65, -.75)
        axis.axvspan(lower, 0, color='#C64A3A', alpha=.04, zorder=0)
        axis.axvline(0, color='#697580', linewidth=.9, linestyle=(0, (4, 3)))
        axis.grid(axis='x', color='#DDE3E8', linewidth=.7)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
        for position, label in enumerate(TARGETS):
            row = rows[label]
            baseline_y, candidate_y = position - .13, position + .13
            axis.plot(row['baseline_minus_cpp_pp'], baseline_y, marker='o', markersize=5.5,
                      markerfacecolor='white', markeredgewidth=1.3, color=COLORS['baseline'], zorder=4)
            low, high = (row['candidate_minus_cpp_marginal_95_' + side + '_pp'] for side in ('lower', 'upper'))
            axis.hlines(candidate_y, low, high, color=COLORS['candidate'], linewidth=1.6)
            axis.plot((low, high), (candidate_y, candidate_y), marker='|', linestyle='none',
                      markersize=7, color=COLORS['candidate'])
            axis.plot(row['candidate_minus_cpp_pp'], candidate_y, marker='s', markersize=5.5,
                      markeredgecolor='white', markeredgewidth=.6, color=COLORS['candidate'], zorder=4)
            axis.text(1.025, candidate_y, f"{row['candidate_minus_cpp_pp']:+.3f}",
                      transform=axis.get_yaxis_transform(), va='center', ha='left', fontsize=10,
                      color=COLORS['candidate'])
        axis.set_yticks(range(5))
        axis.set_yticklabels([f"Label {label}\nn={rows[label]['queries']:,}" +
                              (' *' if rows[label]['queries'] < 100 else '') for label in TARGETS],
                             fontsize=10, linespacing=1.4)
        axis.tick_params(axis='y', pad=10)
        axis.tick_params(axis='x', pad=7)
        axis.set_xlabel('Recall@10 difference from C++ (percentage points)', labelpad=10, fontsize=10)
        n = sum(row['queries'] for row in summary['per_label'])
        axis.set_title(title + f"\n{n:,} queries; {len(summary['per_label'])} labels evaluated", loc='left', pad=17)
        axis.text(1.025, 1.025, 'Updated Δ', transform=axis.transAxes, ha='left', fontsize=9, color=COLORS['baseline'])
    figure.suptitle('Recall at unchanged search breadth', x=.055, y=.972, ha='left',
                   fontsize=20, fontweight='bold', color='#23313C')
    figure.text(.055, .913, 'Five originally declining labels · C++ L64 · both Rust indexes use the original frozen per-label L policy',
                fontsize=11, color='#59636A')
    handles = [Line2D([], [], marker='o', linestyle='none', markerfacecolor='white', color=COLORS['baseline'],
                      markeredgewidth=1.3, label='Previous Rust: observed difference'),
               Line2D([], [], marker='s', linestyle='-', color=COLORS['candidate'],
                      label='Updated Rust: difference and marginal 95% CI')]
    figure.legend(handles=handles, loc='upper left', bbox_to_anchor=(.05, .884), frameon=False,
                  ncol=2, fontsize=10, columnspacing=2.0)
    figure.text(.055, .138,
                'Intervals use paired-query bootstrap resampling; they are marginal, not simultaneous. An interval crossing zero does not establish parity.\n'
                'Supplemental queries replace an original two-tag AND filter with one constituent label; keep their results separate from natural one-tag queries.\n'
                '* Fewer than 100 natural queries. Five targets are shown here; the accompanying report discloses all 64 natural labels.',
                fontsize=9.5, color='#59636A', linespacing=1.5, va='top')
    figure.text(.055, .027, 'Sources: ' + ' · '.join(name + ' ' + item['sha256'][:12] + '…' for name, item in sources.items()),
                fontsize=8, color='#87939D')
    paths = save(figure, output, 'recall', sources)
    plt.close(figure)
    return paths


def performance_plot(plt, main, output, sources):
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    figure, axes = plt.subplots(1, 2, figsize=(12.3, 5.6))
    figure.subplots_adjust(left=.09, right=.96, top=.73, bottom=.225, wspace=.38)
    for axis, metric, scale, title in zip(axes, ('qps', 'p99_us'), (1, 1000),
                                          ('Throughput (queries/second)', 'p99 search service time (ms)')):
        clean_axis(axis)
        values = [main['engines'][engine][metric] / scale for engine in ENGINES]
        ranges = [[number / scale for number in main['engines'][engine]['ranges'][metric]] for engine in ENGINES]
        positions = range(3)
        axis.bar(positions, values, color=[COLORS[engine] for engine in ENGINES], width=.62, zorder=2)
        axis.errorbar(list(positions), values,
                      yerr=([value - bounds[0] for value, bounds in zip(values, ranges)],
                            [bounds[1] - value for value, bounds in zip(values, ranges)]),
                      fmt='none', ecolor='#23313C', capsize=4, linewidth=1, zorder=3)
        maximum = max(high for _, high in ranges)
        axis.set_ylim(0, maximum * 1.21)
        for position, value, bounds in zip(positions, values, ranges):
            label = f'{value:,.0f}' if metric == 'qps' else f'{value:.3f}'
            axis.text(position, bounds[1] + maximum * .045, label, ha='center', va='bottom',
                      fontsize=12, fontweight='bold', color='#23313C')
        axis.set_xticks(list(positions), [LABELS[engine] for engine in ENGINES], fontsize=10)
        axis.tick_params(axis='x', pad=10)
        axis.grid(axis='y', color='#DDE3E8', linewidth=.7)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        if metric == 'qps':
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
        axis.set_title(title, loc='left', pad=17)
    n = sum(row['queries'] for row in main['per_label'])
    figure.suptitle('Performance on fresh natural queries', x=.055, y=.972, ha='left',
                   fontsize=20, fontweight='bold', color='#23313C')
    figure.text(.055, .906, f'{n:,} queries · all 64 labels · eight physical cores · six serial engine-order permutations · unchanged L policy',
                fontsize=11, color='#59636A')
    figure.text(.055, .128,
                'Bars: medians of six authoritative timing repetitions. Whiskers: observed min–max across those repetitions, not confidence intervals.\n'
                'Timing repeats reuse the same queries. These are mixed-workload search service times; supplemental and calibration timings are excluded.',
                fontsize=9.5, color='#59636A', linespacing=1.5, va='top')
    figure.text(.055, .026, 'Source: main summary ' + sources['main']['sha256'][:16] + '…', fontsize=8, color='#87939D')
    paths = save(figure, output, 'performance', {'main': sources['main']})
    plt.close(figure)
    return paths


def load_v2_summary(path, development=False):
    result = json.loads(path.read_text())
    phase = 'development-recheck' if development else 'secondary-validation'
    freshness = 'previously_exposed_development' if development else 'previously_unsearched_partition'
    scope = ('Exposed natural-query development recheck' if development else
             'Fresh secondary atomic-derived validation; separate five-label shifted workload')
    require(result.get('schema_version') == 2 and result.get('phase') == phase
            and result.get('query_freshness') == freshness and result.get('scope') == scope,
            'Expected the explicit candidate2 development or fresh atomic validation scope; calibration is excluded')
    excluded = result.get('timing_excluded')
    require(type(excluded) is bool and (development or excluded is False),
            'Fresh secondary validation requires authoritative timing')
    repeats = 1 if excluded else 6
    jobs = result.get('jobs_proof', {})
    require(jobs.get('blocks') == repeats and jobs.get('merged_rows_match_original_aggregates') is True
            and jobs.get('declared_timing_jobs_do_not_overlap') is True, 'Missing audited serial job proof')
    identity = result.get('identity_proof', {})
    evidence = identity.get('declarations', {})
    require(evidence.get('phase') == phase and evidence.get('query_freshness') == freshness
            and evidence.get('cpu_affinity') == list(range(8))
            and evidence.get('plan_before_sha256') == evidence.get('plan_after_sha256') == FROZEN_PLAN,
            'Candidate 2 phase, freshness, affinity or L policy differs')
    require(development or identity.get('calibration_identity') is not None,
            'Fresh candidate2 validation lacks frozen calibration identity')
    bootstrap = result.get('bootstrap', {})
    require(bootstrap.get('unit') == 'paired query' and bootstrap.get('replications', 0) >= 10000,
            'Candidate 2 uncertainty must resample paired queries')
    require(set(result.get('targets', [])) == set(TARGETS), 'Candidate 2 target set differs')
    rows = result.get('per_label', [])
    require(type(rows) is list and len(rows) == (64 if development else 5), 'Candidate 2 label coverage differs')
    seen = set()
    for row in rows:
        require(type(row.get('label')) is int and row['label'] not in seen
                and type(row.get('queries')) is int and row['queries'] > 0, 'Invalid candidate2 label/sample count')
        seen.add(row['label'])
        require(row.get('cpp_l') == 64 and type(row.get('rust_l')) is int and row['rust_l'] >= 64,
                'Candidate 2 search breadth differs')
        for engine in ENGINES:
            recall = row.get(engine + '_recall_percent')
            require(number(recall) and 0 <= recall <= 100, 'Invalid candidate2 per-label recall')
        for engine in ('baseline', 'candidate'):
            key = engine + '_minus_cpp'
            values = [row.get(key + '_pp'), row.get(key + '_marginal_95_lower_pp'), row.get(key + '_marginal_95_upper_pp')]
            require(all(number(value) for value in values) and -100 <= values[1] <= values[2] <= 100
                    and close(values[0], row[engine + '_recall_percent'] - row['cpp_recall_percent']),
                    'Invalid candidate2 recall delta or marginal interval')
    require(set(TARGETS) <= seen and (development or seen == set(TARGETS)), 'Missing candidate2 target labels')
    if not development:
        count = result.get('split_proof', {}).get('partitions', {}).get('secondary-validation', {}).get('queries')
        require(count == sum(row['queries'] for row in rows), 'Fresh atomic validation sample count differs from sealed split')
    records = result.get('engines', {})
    require(set(records) == set(ENGINES), 'Missing candidate2 engine')
    count = sum(row['queries'] for row in rows)
    for engine, record in records.items():
        require(record.get('repetitions') == repeats, 'Candidate 2 timing repetition count differs')
        aggregate = sum(row['queries'] * row[engine + '_recall_percent'] for row in rows) / count / 100
        require(number(record.get('strict_id_recall')) and close(aggregate, record['strict_id_recall']),
                'Candidate 2 per-label and aggregate recall differ')
        if excluded:
            require(not any(key in record for key in ('qps', 'p99_us', 'repeat_metrics')), 'Quality-only development must exclude timing')
            continue
        values = record.get('repeat_metrics', [])
        require(len(values) == 6 and [item.get('repeat') for item in values] == list(range(6)),
                'Candidate 2 authoritative timing requires six repeats')
        for metric in ('qps', 'p99_us'):
            observed = [item.get(metric) for item in values]
            limits = record.get('ranges', {}).get(metric)
            require(all(number(value) and value > 0 for value in observed) and number(record.get(metric))
                    and type(limits) is list and len(limits) == 2
                    and close(statistics.median(observed), record[metric])
                    and close(min(observed), limits[0]) and close(max(observed), limits[1]),
                    'Invalid candidate2 authoritative timing median or range')
    return result


def validate_three(original, development, fresh):
    require(original['prepared_identity'] == development['prepared_identity'],
            'Candidate 2 natural recheck is not the same exposed query workload')
    previous = original['identity_proof']['declared_identity']['engines']
    later = development['identity_proof']['declarations']['engines']
    final = fresh['identity_proof']['declarations']['engines']
    for engine in ('cpp', 'baseline'):
        require(previous[engine]['binary_sha256'] == later[engine]['binary_sha256']
                and previous[engine]['index_after'] == later[engine]['index_before'],
                'Original C++ or previous-Rust benchmark assets changed')
    for engine in ENGINES:
        require(later[engine]['binary_sha256'] == final[engine]['binary_sha256']
                and later[engine]['index_after'] == final[engine]['index_before'],
                'Candidate 2 assets differ between development and fresh atomic validation')
    rows = [{row['label']: row for row in summary['per_label']} for summary in (original, development, fresh)]
    for label in TARGETS:
        require(len({context[label]['rust_l'] for context in rows}) == 1,
                'Actual per-label search breadth changed between contexts')


def recall_plot_three(plt, original, development, fresh, output, sources):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    contexts = (original, development, fresh)
    lookup = [{row['label']: row for row in context['per_label']} for context in contexts]
    values = [value for rows in lookup for label in TARGETS for value in
              (rows[label]['baseline_minus_cpp_pp'], rows[label]['candidate_minus_cpp_pp'],
               rows[label]['candidate_minus_cpp_marginal_95_lower_pp'],
               rows[label]['candidate_minus_cpp_marginal_95_upper_pp'])]
    low, high = min(min(values), 0), max(max(values), 0)
    span = max(high - low, .4)
    low, high = low - .12 * span, high + .12 * span
    figure, axes = plt.subplots(1, 3, figsize=(19.5, 6.9))
    figure.subplots_adjust(left=.07, right=.92, top=.68, bottom=.225, wspace=.55)
    titles = ('Candidate 1: natural validation', 'Candidate 2: natural development recheck', 'Candidate 2: fresh atomic validation')
    captions = ('Fresh for candidate1 at its evaluation', 'Exploratory recall; previously exposed queries', 'Derived from two-tag vectors; shifted workload')
    for axis, context, rows, title, caption in zip(axes, contexts, lookup, titles, captions):
        clean_axis(axis)
        axis.set_xlim(low, high)
        axis.set_ylim(4.65, -.75)
        axis.axvspan(low, 0, color='#C64A3A', alpha=.04, zorder=0)
        axis.axvline(0, color='#697580', linewidth=.9, linestyle=(0, (4, 3)))
        axis.grid(axis='x', color='#DDE3E8', linewidth=.7)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        for position, label in enumerate(TARGETS):
            row = rows[label]
            old_y, new_y = position - .13, position + .13
            axis.plot(row['baseline_minus_cpp_pp'], old_y, marker='o', markersize=5.5,
                      markerfacecolor='white', markeredgewidth=1.3, color=COLORS['baseline'], zorder=4)
            lower, upper = (row['candidate_minus_cpp_marginal_95_' + side + '_pp'] for side in ('lower', 'upper'))
            axis.hlines(new_y, lower, upper, color=COLORS['candidate'], linewidth=1.6)
            axis.plot((lower, upper), (new_y, new_y), marker='|', linestyle='none', markersize=7, color=COLORS['candidate'])
            axis.plot(row['candidate_minus_cpp_pp'], new_y, marker='s', markersize=5.5,
                      markeredgecolor='white', markeredgewidth=.6, color=COLORS['candidate'], zorder=4)
            axis.text(1.025, new_y, f"{row['candidate_minus_cpp_pp']:+.3f}", transform=axis.get_yaxis_transform(),
                      va='center', ha='left', fontsize=10, color=COLORS['candidate'])
        axis.set_yticks(range(5))
        axis.set_yticklabels([f"Label {label}\nn={rows[label]['queries']:,}" +
                              (' *' if rows[label]['queries'] < 100 else '') for label in TARGETS], fontsize=10)
        axis.tick_params(axis='y', pad=9)
        axis.tick_params(axis='x', pad=7)
        axis.set_xlabel('Recall@10 difference from C++ (percentage points)', labelpad=10, fontsize=9.5)
        count = sum(row['queries'] for row in context['per_label'])
        axis.set_title(title + f'\n{count:,} queries; {len(context["per_label"])} labels evaluated', loc='left', pad=18, fontsize=12)
        axis.text(0, 1.22, caption, transform=axis.transAxes, fontsize=10,
                  color='#A75228' if context is development else '#59636A', ha='left')
        axis.text(1.025, 1.025, 'Candidate Δ', transform=axis.transAxes, ha='left', fontsize=9, color=COLORS['baseline'])
    figure.suptitle('Recall at unchanged search breadth', x=.045, y=.975, ha='left',
                   fontsize=21, fontweight='bold', color='#23313C')
    figure.text(.045, .918, 'Five originally declining labels · C++ L64 · all Rust configurations use the original frozen per-label L policy',
                fontsize=12, color='#59636A')
    handles = [Line2D([], [], marker='o', linestyle='none', markerfacecolor='white', color=COLORS['baseline'],
                      markeredgewidth=1.3, label='Previous Rust: observed difference'),
               Line2D([], [], marker='s', linestyle='-', color=COLORS['candidate'],
                      label='Candidate in panel: difference and marginal 95% CI')]
    figure.legend(handles=handles, loc='upper left', bbox_to_anchor=(.04, .892), frameon=False,
                  ncol=2, fontsize=11, columnspacing=2.0)
    figure.text(.045, .14,
                'Recall intervals use paired-query bootstrap resampling; they are marginal, not simultaneous. An interval crossing zero does not establish parity.\n'
                'Candidate 2 natural-query recall is exploratory because those queries influenced development. Fresh atomic results cannot establish natural-query parity.\n'
                '* Fewer than 100 queries. The same horizontal scale is used throughout. All 64 natural labels are disclosed in the accompanying report.',
                fontsize=10, color='#59636A', linespacing=1.5, va='top')
    figure.text(.045, .025, 'Sources: ' + ' · '.join(name + ' ' + item['sha256'][:10] + '…' for name, item in sources.items()),
                fontsize=8, color='#87939D')
    outputs = save(figure, output, 'recall', sources)
    plt.close(figure)
    return outputs


def performance_plot_three(plt, original, development, fresh, output, sources):
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    contexts = (original, development, fresh)
    headings = ('Candidate 1 · natural validation', 'Candidate 2 · natural development recheck', 'Candidate 2 · fresh atomic-derived validation')
    figure, axes = plt.subplots(3, 2, figsize=(13.1, 13.4))
    figure.subplots_adjust(left=.105, right=.96, top=.855, bottom=.12, hspace=.85, wspace=.38)
    for row_index, (context, heading) in enumerate(zip(contexts, headings)):
        for axis, metric, scale, title in zip(axes[row_index], ('qps', 'p99_us'), (1, 1000),
                                             ('Throughput (queries/second)', 'p99 search service time (ms)')):
            clean_axis(axis)
            if context['timing_excluded']:
                axis.set_axis_off()
                axis.text(.5, .5, 'Quality-only development check\nNo authoritative timing measured',
                          transform=axis.transAxes, ha='center', va='center', color='#59636A')
                continue
            values = [context['engines'][engine][metric] / scale for engine in ENGINES]
            bounds = [[value / scale for value in context['engines'][engine]['ranges'][metric]] for engine in ENGINES]
            axis.bar(range(3), values, color=[COLORS[engine] for engine in ENGINES], width=.62, zorder=2)
            axis.errorbar(range(3), values,
                          yerr=([value - interval[0] for value, interval in zip(values, bounds)],
                                [interval[1] - value for value, interval in zip(values, bounds)]),
                          fmt='none', ecolor='#23313C', capsize=4, linewidth=1, zorder=3)
            maximum = max(upper for _, upper in bounds)
            axis.set_ylim(0, maximum * 1.22)
            for position, value, interval in zip(range(3), values, bounds):
                axis.text(position, interval[1] + maximum * .045,
                          f'{value:,.0f}' if metric == 'qps' else f'{value:.3f}',
                          ha='center', va='bottom', fontsize=11, fontweight='bold', color='#23313C')
            axis.set_xticks(range(3), ['C++', 'Previous Rust', f'Candidate{1 if row_index == 0 else 2}'], fontsize=10)
            axis.tick_params(axis='x', pad=9)
            axis.grid(axis='y', color='#DDE3E8', linewidth=.7)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if metric == 'qps':
                axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
            axis.set_title(title, loc='left', pad=10, fontsize=11)
        count = sum(row['queries'] for row in context['per_label'])
        axes[row_index, 0].text(0, 1.29, heading + f' · {count:,} queries',
                                transform=axes[row_index, 0].transAxes, fontsize=12, fontweight='bold',
                                color='#A75228' if row_index == 1 else '#23313C', ha='left')
    figure.suptitle('Performance by evaluation context', x=.055, y=.975, ha='left',
                   fontsize=21, fontweight='bold', color='#23313C')
    figure.text(.055, .932, 'Eight physical cores · unchanged L policy · each measured row uses six serial engine-order permutations',
                fontsize=11, color='#59636A')
    figure.text(.055, .078,
                'Bars: six-repeat medians. Whiskers: observed min–max, not confidence intervals. Every row is a separate workload/run.\n'
                'Natural development queries were previously exposed; fresh secondary queries use derived atomic labels and a different five-label mixture.\n'
                'These are mixed-workload service times on this host. Calibration timings are excluded; timing repetitions reuse the same query vectors.',
                fontsize=9.5, color='#59636A', linespacing=1.5, va='top')
    figure.text(.055, .018, 'Sources: ' + ' · '.join(item['sha256'][:10] + '…' for item in sources.values()),
                fontsize=8, color='#87939D')
    outputs = save(figure, output, 'performance', sources)
    plt.close(figure)
    return outputs



def plot(main_path, supplemental_path, output, matplotlib_path=None, check_only=False, overwrite=False, development_path=None):
    main = load_summary(main_path)
    if development_path is None:
        supplemental = load_summary(supplemental_path, supplemental=True)
        development = None
        validate_pair(main, supplemental)
        inputs = (('main', main_path), ('supplemental', supplemental_path))
    else:
        development = load_v2_summary(development_path, development=True)
        supplemental = load_v2_summary(supplemental_path)
        validate_three(main, development, supplemental)
        inputs = (('candidate1_natural', main_path), ('candidate2_natural_development', development_path),
                  ('candidate2_atomic_validation', supplemental_path))
    sources = {name: {'path': str(path.resolve()), 'sha256': sha256(path)} for name, path in inputs}
    if check_only:
        print(json.dumps({'validated': True, 'sources': sources, 'plots_generated': False}, indent=2))
        return []
    output = output.with_suffix('') if output.suffix.lower() in ('.png', '.svg') else output
    paths = [Path(str(output) + '-' + name + '.' + extension)
             for name in ('recall', 'performance') for extension in ('png', 'svg')]
    proof_path = Path(str(output) + '-provenance.json')
    require(all(path.resolve() not in tuple(source.resolve() for _, source in inputs) for path in (*paths, proof_path)),
            'Figure cannot overwrite an input summary')
    require(overwrite or not any(path.exists() for path in (*paths, proof_path)), 'Figure output already exists; use --overwrite to replace these figures')
    output.parent.mkdir(parents=True, exist_ok=True)
    if matplotlib_path:
        require(matplotlib_path.is_dir(), 'Matplotlib dependency directory does not exist')
        sys.path.insert(0, str(matplotlib_path))
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/diskann-recall-fix-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style(plt)
    if development is None:
        outputs = recall_plot(plt, main, supplemental, output, sources)
        outputs += performance_plot(plt, main, output, sources)
    else:
        outputs = recall_plot_three(plt, main, development, supplemental, output, sources)
        outputs += performance_plot_three(plt, main, development, supplemental, output, sources)
    record = {'schema_version': 1, 'sources': sources, 'plotter': {'path': str(Path(__file__).resolve()),
              'sha256': sha256(Path(__file__))}, 'outputs': outputs,
              'target_labels': list(TARGETS), 'main_queries': sum(row['queries'] for row in main['per_label']),
              'supplemental_queries': sum(row['queries'] for row in supplemental['per_label']),
              'recall_interval': 'marginal 95% paired-query bootstrap; previous Rust displayed as observed point only',
              'performance_scope': ('natural validation only, six repetitions, median and observed min–max' if development is None
                                    else 'Each context kept separate: candidate1 natural; candidate2 exposed natural development; candidate2 fresh shifted atomic validation. Only authoritative timing is shown.'),
              'candidate2_natural_recall_is_exploratory': development is not None}
    proof_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'provenance': str(proof_path), 'outputs': outputs}, indent=2))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-summary', type=Path, required=True, help='Audited candidate1 fresh natural validation summary')
    parser.add_argument('--development-summary', type=Path, help='Audited candidate2 exposed natural-query recheck; activates the three-context layout')
    parser.add_argument('--supplemental-summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='Output prefix for recall/performance PNG/SVG figures')
    parser.add_argument('--matplotlib-path', type=Path,
                        default=Path('/private/tmp/diskann-heldout-plot-deps') if Path('/private/tmp/diskann-heldout-plot-deps').is_dir() else None)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    plot(args.main_summary.resolve(), args.supplemental_summary.resolve(), args.out.resolve(),
         matplotlib_path=args.matplotlib_path, check_only=args.check_only, overwrite=args.overwrite,
         development_path=args.development_summary.resolve() if args.development_summary else None)


if __name__ == '__main__':
    main()
