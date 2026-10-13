"""Baselines: every dependent's suite result at a scan's pins, and diffs between them.

A baseline records, per dependent: status, the exact command (stored, not
assumed), duration, pass/fail/error/skip counts, and **the set of failing test
ids**, so comparing two runs is a set difference that names the regression.
It is deliberately not ``-x``: fail-fast hides the shape of a suite.

Every dependent is run, none is filtered by a "has tests" guess: pytest's own
exit code 5 ("no tests collected") is recorded as ``NO_TESTS``.

The command is the dependents gate's, from priv's ``dep_graph`` (CI doctest
flags plus each repo's ``[tool.wads.ci.testing]`` exclusions), so a baseline and
``priv test-dependents`` measure the same thing. ``gate_args=`` is the seam.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import fleet, store

#: Arguments the ledger adds to the gate's (quiet, no cache files left in repos).
DFLT_PYTEST_ARGS = ('-q', '--tb=line', '-p', 'no:cacheprovider')
DFLT_TIMEOUT = 900
JUNIT_PLACEHOLDER = '<junitxml>'
PYTHON_PLACEHOLDER = '<python>'

_STATUS_OF_RC = {0: 'PASS', 1: 'FAIL', 2: 'INTERRUPTED', 5: 'NO_TESTS'}


def priv_gate_args(repo: Path) -> tuple[bool, list[str]]:
    """``(enabled, pytest args)`` exactly as ``priv test-dependents`` builds them."""
    from priv import dep_graph as dg

    cfg = dg._ci_testing_config(Path(repo))
    if cfg.get('enabled', True) is False:
        return False, []
    excludes = cfg.get('exclude_paths', ['examples', 'scrap'])
    ignores = [f'--ignore={p}' for p in (*excludes, *dg.GATE_EXTRA_IGNORES)]
    return True, [*dg.CI_DOCTEST_ARGS, *ignores]


def failing_ids(junit_path: Path) -> tuple[dict, list[str]]:
    """Counts and sorted ``classname::name`` ids of failed/errored cases in a JUnit file."""
    tree = ET.parse(junit_path)
    counts = dict(tests=0, failures=0, errors=0, skipped=0)
    failing = set()
    for case in tree.iter('testcase'):
        counts['tests'] += 1
        tags = {child.tag for child in case}
        # one outcome per case: a failure with a teardown error is one failure
        if 'failure' in tags:
            counts['failures'] += 1
        elif 'error' in tags:
            counts['errors'] += 1
        elif 'skipped' in tags:
            counts['skipped'] += 1
        if tags & {'failure', 'error'}:
            failing.add(f"{case.get('classname', '')}::{case.get('name', '')}")
    counts['passed'] = (
        counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped']
    )
    return counts, sorted(failing)


def run_suite(
    repo: Path,
    *,
    python: str = sys.executable,
    gate_args: Callable[[Path], tuple[bool, list[str]]] = priv_gate_args,
    pytest_args: tuple[str, ...] = DFLT_PYTEST_ARGS,
    timeout: float = DFLT_TIMEOUT,
) -> dict:
    """Run one repo's suite; return its baseline entry (no paths in it)."""
    enabled, args = gate_args(repo)
    template = [PYTHON_PLACEHOLDER, '-m', 'pytest', *args, *pytest_args,
                f'--junitxml={JUNIT_PLACEHOLDER}']
    if not enabled:
        return dict(status='DISABLED', command=template, seconds=0.0)
    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / 'junit.xml'
        cmd = [python if a == PYTHON_PLACEHOLDER else a.replace(JUNIT_PLACEHOLDER, str(junit))
               for a in template]
        t0 = time.monotonic()
        rc, out = _run_killable(cmd, cwd=repo, timeout=timeout)
        seconds = round(time.monotonic() - t0, 1)
        if rc is None:
            return dict(status='TIMEOUT', command=template, seconds=seconds, timeout=timeout)
        entry = dict(
            status=_STATUS_OF_RC.get(rc, 'ERROR'),
            returncode=rc,
            command=template,
            seconds=seconds,
        )
        if junit.exists():
            try:
                counts, failing = failing_ids(junit)
                entry.update(counts=counts, failing=failing)
            except ET.ParseError:
                entry['junit'] = 'unparseable'
        if entry['status'] not in ('PASS', 'NO_TESTS'):
            entry['tail'] = [store.redact(store.scrub(l)) for l in out.strip().splitlines()[-15:]]
    entry['failing'] = [store.redact(store.scrub(f)) for f in entry.get('failing', [])]
    return entry


def _run_killable(cmd, *, cwd, timeout) -> tuple[int | None, str]:
    """Run *cmd*; on timeout kill its whole process group. ``(None, '')`` on timeout."""
    posix = os.name == 'posix'
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        errors='replace',
        start_new_session=posix,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        if posix:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        return None, ''


def _imported(python: str, package: str) -> dict:
    """Where *package* resolves for the gate's interpreter (from a neutral cwd)."""
    code = (
        f'import {package}, json, os; '
        f'print(json.dumps({{"file": {package}.__file__, '
        f'"version": getattr({package}, "__version__", None)}}))'
    )
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run([python, '-c', code], cwd=tmp, capture_output=True,
                              text=True, check=False)
    if proc.returncode:
        return dict(error=proc.stderr.strip().splitlines()[-1:])
    info = json.loads(proc.stdout)
    pkg_dir = Path(info['file']).resolve().parents[1]
    info['pin'] = fleet.git_pin(pkg_dir) if (pkg_dir / '.git').exists() else None
    return info


def run_baseline(
    root,
    *,
    package: str = 'dol',
    scan_id: str | None = None,
    repos: Mapping | None = None,
    jobs: int = 1,
    python: str = sys.executable,
    gate_args: Callable[[Path], tuple[bool, list[str]]] = priv_gate_args,
    timeout: float = DFLT_TIMEOUT,
    only=None,
    note: str | None = None,
    progress: Callable[[str, dict], None] | None = None,
) -> str:
    """Run every dependent of a scan, record a baseline, return its id.

    ``jobs`` runs suites concurrently (1 by default, like the gate: some suites
    share on-disk state). ``only`` restricts to the given dependent names.
    """
    from .pins import require_same_source

    root = Path(root)
    scan_id = scan_id or store.latest_id(root, 'scans')
    require_same_source(root, scan_id, repos)
    pins = store.read_json(root / 'scans' / scan_id / 'pins.json')['pins']
    names = sorted(n for n in pins if only is None or n in set(only))
    paths = fleet.resolve_repos(names, repos=repos)

    def one(name):
        path = paths[name]
        if path is None:
            entry = dict(status='NOT_FOUND')
        else:
            try:  # one dependent's trouble must not cost the whole baseline
                entry = run_suite(path, python=python, gate_args=gate_args, timeout=timeout)
                now = fleet.git_pin(path)
                entry['sha'] = now.get('sha')
                entry['at_pin'] = now.get('sha') == pins[name].get('sha')
                entry['dirty'] = now.get('dirty')
            except Exception as e:
                entry = dict(status='ERROR', error=store.redact(store.scrub(
                    f'{type(e).__name__}: {e}'))[:300])
        if progress:
            progress(name, entry)
        return name, entry

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        results = dict(ex.map(one, names))
    base_dir = root / 'baselines'
    bid = store.new_record_id(base_dir)
    created = store.utc_now().isoformat()
    record = dict(
        schema_version=store.SCHEMA_VERSION,
        kind='baseline',
        id=bid,
        scan_id=scan_id,
        created=created,
        package=package,
        package_imported=store.scrub_obj(_imported(python, package)),
        jobs=jobs,
        timeout=timeout,
        wall_seconds=round(time.monotonic() - t0, 1),
        results=results,
        note=note,
    )
    store.write_json(base_dir / bid / 'baseline.json', record)
    store.add_to_index(
        root,
        'baselines',
        dict(id=bid, created=created, scan_id=scan_id, n=len(results), note=note,
             statuses=summarize(results)),
    )
    return bid


def summarize(results: Mapping[str, dict]) -> dict:
    """``{status: count}``."""
    out: dict[str, int] = {}
    for r in results.values():
        out[r['status']] = out.get(r['status'], 0) + 1
    return dict(sorted(out.items()))


#: Statuses whose failing-id set covers the whole suite, so ids can be compared.
COMPLETE_RUNS = frozenset({'PASS', 'FAIL', 'NO_TESTS'})


class IncomparableBaselines(ValueError):
    """Two baselines ran a dependent with different commands."""


def diff(a: dict, b: dict) -> dict:
    """Per dependent: tests newly failing in *b*, fixed in *b*, and status changes.

    Refuses (``IncomparableBaselines``) when a dependent present in both ran with
    a different command: changing the flags changes what gets reported, which
    would read exactly like a regression. A dependent where either side did not
    run to completion (``INTERRUPTED``, ``TIMEOUT``, ``ERROR``...) is reported as
    ``incomparable`` when its status changed, never as tests fixed.
    """
    ra, rb = a['results'], b['results']
    mismatched = [
        n for n in sorted(set(ra) & set(rb))
        if 'command' in ra[n] and 'command' in rb[n] and ra[n]['command'] != rb[n]['command']
    ]
    if mismatched:
        raise IncomparableBaselines(
            f'{len(mismatched)} dependent(s) ran with different commands, '
            f'e.g. {mismatched[0]}: {ra[mismatched[0]]["command"]} vs '
            f'{rb[mismatched[0]]["command"]}'
        )
    out = {}
    for name in sorted(set(ra) | set(rb)):
        ea, eb = ra.get(name, {}), rb.get(name, {})
        status = (ea.get('status'), eb.get('status'))
        if not (set(status) <= COMPLETE_RUNS):
            # one side did not run its whole suite (collection error, timeout...):
            # failures absent from it are not fixed, just not run
            if status[0] != status[1]:
                out[name] = dict(status=status, incomparable=True,
                                 new_failures=[], fixed=[])
            continue
        fa, fb = set(ea.get('failing', ())), set(eb.get('failing', ()))
        change = dict(
            status=status,
            incomparable=False,
            new_failures=sorted(fb - fa),
            fixed=sorted(fa - fb),
        )
        if status[0] != status[1] or fb != fa:
            out[name] = change
    return dict(a=a['id'], b=b['id'], changed=out)


def load_baseline(root, baseline_id: str | None = None) -> dict:
    root = Path(root)
    baseline_id = baseline_id or store.latest_id(root, 'baselines')
    return store.read_json(root / 'baselines' / baseline_id / 'baseline.json')
