#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Run only exposed calibration/natural queries to verify a conservative default rollback.

These jobs are quality-only output parity controls, not a new held-out evaluation
or a throughput benchmark. The reserved final query partition is never opened.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tarfile

from run_heldout import FROZEN_PLAN, dump, now, sha
from run_recall_fix_v4 import native_identity, validate_pool_identity

CORE = ('diskann-disk/src/build/filtered.rs', 'diskann-disk/src/filtered/view.rs',
        'diskann-disk/src/search/provider/disk_provider.rs', 'diskann-disk/src/search/search_mode.rs')
WORKLOADS = (
    ('calibration', 'recall-fix/data/secondary-two-tag/prepared-secondary-atomic-calibration-v2', 1436),
    ('natural', 'recall-fix/data/natural-private/prepared-private-natural-validation', 16600),
)


def archive_core(path):
    with tarfile.open(path, 'r:gz') as source:
        files = {}
        for member in source.getmembers():
            if not member.isfile():
                continue
            name = member.name.removeprefix('./')
            if name in files:
                raise ValueError('Duplicate archive source path')
            files[name] = hashlib.sha256(source.extractfile(member).read()).hexdigest()
    if not set(CORE) <= set(files):
        raise ValueError('Rollback source archive omits a core source')
    return {name: files[name] for name in CORE}


def source_core(root):
    return {name: sha(root / 'rust' / name) for name in CORE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'binary', 'source-archive', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root, binary, archive, out = (getattr(args, name).resolve() for name in ('root', 'binary', 'source_archive', 'out'))
    if sorted(os.sched_getaffinity(0)) != list(range(8)):
        parser.error('Requires physical cores 0..7')
    if out.exists():
        parser.error('Refusing to overwrite parity outputs')
    plan = root / 'results/migration/native-label-l-plan.json'
    if sha(plan) != FROZEN_PLAN:
        parser.error('Original frozen L policy changed')
    archived = archive_core(archive)
    live = source_core(root)
    if archived != live:
        parser.error('Rollback source archive differs from the live compiled source checkout')
    index_root = root / 'indexes/rust-native-10m-spatial-entry'
    before = native_identity(index_root)
    publication_path = index_root / 'entry-pool-result.json'
    validate_pool_identity(before, index_root, json.loads(publication_path.read_text()))
    out.mkdir(parents=True, exist_ok=False)
    evidence = {'schema_version': 1, 'status': 'running', 'scope': 'exposed_output_parity_only',
                'started': now(), 'index_root': str(index_root), 'output_root': str(out),
                'cpu_affinity': list(range(8)), 'runner_sha256': sha(Path(__file__)),
                'binary': {'path': str(binary), 'before_sha256': sha(binary)},
                'source_archive': {'path': str(archive), 'before_sha256': sha(archive)},
                'live_core_before_sha256': live, 'archive_core_sha256': archived,
                'pool_publication_sha256': sha(publication_path),
                'plan': {'path': str(plan), 'before_sha256': sha(plan)},
                'index_before': before, 'jobs': []}
    path = out / 'evidence.json'
    dump(path, evidence)
    try:
        for name, relative, expected_count in WORKLOADS:
            prepared = root / relative
            manifest = prepared / 'manifest.json'
            count, dimensions = struct.unpack('<II', (prepared / 'query.u8bin').read_bytes()[:8])
            if (count, dimensions) != (expected_count, 192):
                raise ValueError('Exposed parity workload shape differs')
            aggregate, details = out / (name + '.jsonl'), out / (name + '-details/query.t8.L64.r0.jsonl')
            command = [str(binary), '--action', 'query', '--root', str(index_root), '--prepared', str(prepared),
                       '--out', str(aggregate), '--threads', '8', '--query-threads', '8', '--l', '64',
                       '--reps', '1', '--limit', str(count), '--timing-authoritative', 'false',
                       '--query-details-dir', str(details.parent), '--evaluation-scope', 'frozen_policy_heldout',
                       '--label-l', str(plan), '--allow-unused-plan-labels', 'true']
            job = {'workload': name, 'query_freshness': 'previously_exposed_development',
                   'timing_authoritative': False, 'command': command, 'start': now(),
                   'prepared': str(prepared), 'prepared_manifest_sha256': sha(manifest),
                   'prepared_artifacts_sha256': {filename: sha(prepared / filename) for filename in
                                                ('query.u8bin', 'query_ids.u32bin', 'query_labels.u32bin')},
                   'aggregate_path': str(aggregate), 'details_path': str(details),
                   'stdout_log': str(out / (name + '.log'))}
            with Path(job['stdout_log']).open('x') as output:
                result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
            job.update(end=now(), exit_code=result.returncode)
            if result.returncode:
                evidence['jobs'].append(job)
                raise RuntimeError(f'{name} query control failed')
            job.update(aggregate_sha256=sha(aggregate), details_sha256=sha(details),
                       prepared_manifest_after_sha256=sha(manifest),
                       prepared_artifacts_after_sha256={filename: sha(prepared / filename)
                                                       for filename in job['prepared_artifacts_sha256']})
            evidence['jobs'].append(job)
            dump(path, evidence)
            if (job['prepared_manifest_after_sha256'] != job['prepared_manifest_sha256']
                    or job['prepared_artifacts_after_sha256'] != job['prepared_artifacts_sha256']):
                raise ValueError('Prepared exposed workload changed during the control')
            print(json.dumps({'workload': name, 'status': 'query complete', 'queries': count,
                              'scope': 'output parity only; timings not authoritative'}), flush=True)
        evidence['binary']['after_sha256'] = sha(binary)
        evidence['source_archive']['after_sha256'] = sha(archive)
        evidence['plan']['after_sha256'] = sha(plan)
        evidence['index_after'] = native_identity(index_root)
        evidence['live_core_after_sha256'] = source_core(root)
        if (evidence['binary']['before_sha256'] != evidence['binary']['after_sha256']
                or evidence['source_archive']['before_sha256'] != evidence['source_archive']['after_sha256']
                or evidence['plan']['after_sha256'] != FROZEN_PLAN
                or evidence['index_after'] != before or evidence['live_core_after_sha256'] != live
                or sha(publication_path) != evidence['pool_publication_sha256']):
            raise ValueError('Binary/source/index/policy identity changed during parity controls')
        evidence.update(status='complete', ended=now())
    except Exception as error:
        evidence.update(status='failed', failure=str(error), ended=now())
        dump(path, evidence)
        raise
    dump(path, evidence)


if __name__ == '__main__':
    main()
