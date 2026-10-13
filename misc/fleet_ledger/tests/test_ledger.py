"""End to end on throwaway git repos: scan, pins (restore/undo safety), baselines, diff.

Nothing here touches the real fleet: every dependent is passed explicitly
(``repos=``) and every repo is created under ``tmp_path``.
"""

import json
import subprocess
import sys

import pytest

from fleet_ledger import baseline, pins, report, scan, store


def git(repo, *args):
    return subprocess.run(
        ['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@t', *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def make_repo(path, files):
    path.mkdir(parents=True)
    git(path, 'init', '-q')
    for rel, text in files.items():
        f = path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    git(path, 'add', '-A')
    git(path, 'commit', '-q', '-m', 'init')
    return path


@pytest.fixture
def fleet(tmp_path):
    """A target package `tgt` and two dependents, all in git."""
    src = make_repo(
        tmp_path / 'tgt_src',
        {
            'tgt/__init__.py': 'import os\nfrom tgt.base import Store, unused\n',
            'tgt/base.py': 'class Store: ...\ndef unused(): ...\ndef hidden(): ...\n',
        },
    )
    a = make_repo(
        tmp_path / 'dep_a',
        {
            'dep_a/__init__.py': 'from tgt import Store\nfrom tgt.base import hidden\n'
            'class S(Store): ...\n',
            'test_a.py': 'def test_ok(): pass\ndef test_flaky(): assert FLAG\nFLAG = True\n',
        },
    )
    b = make_repo(tmp_path / 'dep_b', {'dep_b.py': 'import tgt\ntgt.Store\n'})
    return dict(src=src, repos={'dep_a': a, 'dep_b': b}, root=tmp_path / 'ledger')


def test_scan_writes_an_immutable_indexed_record(fleet):
    sid = scan.run_scan('tgt', source_root=fleet['src'], repos=fleet['repos'], root=fleet['root'])
    data = scan.load_scan(fleet['root'], sid)
    assert data['meta']['schema_version'] == store.SCHEMA_VERSION
    assert data['meta']['import_resolved']['from_source']
    assert len(data['pins']['pins']['dep_a']['sha']) == 40
    agg = data['census']['aggregates']
    assert agg['symbols']['Store']['packages'] == ['dep_a', 'dep_b']
    assert agg['subclasses']['Store']['packages'] == ['dep_a']
    assert list(agg['compat_names']) == ['hidden']
    z = report.zero_usage(data)
    assert z['incidental'] == ['os'] and [c['name'] for c in z['recent']] == ['unused']
    md = report.zero_usage_md(data)
    assert md.count('Scope: "zero usage"') == 1 and 'hidden' in md
    # successive scans accumulate; records are never overwritten
    sid2 = scan.run_scan('tgt', source_root=fleet['src'], repos=fleet['repos'], root=fleet['root'])
    assert sid2 != sid and [e['id'] for e in store.read_index(fleet['root'])['scans']] == [sid, sid2]
    with pytest.raises(FileExistsError):
        store.write_json(fleet['root'] / 'scans' / sid / 'meta.json', {})
    store.mark_superseded(fleet['root'], sid2, reason='test')
    assert store.latest_id(fleet['root']) == sid
    assert (fleet['root'] / 'scans' / sid2).exists()


def test_no_local_path_is_ever_written(fleet):
    sid = scan.run_scan('tgt', source_root=fleet['src'], repos=fleet['repos'], root=fleet['root'])
    for f in (fleet['root'] / 'scans' / sid).iterdir():
        text = f.read_text()
        assert str(fleet['src']) not in text and str(fleet['repos']['dep_a']) not in text
    with pytest.raises(store.LocalPathLeak):
        store.write_json(fleet['root'] / 'x.json', {'p': '/home/someone/secret'})


def test_restore_refuses_dirty_writes_undo_first_and_undo_puts_it_back(fleet):
    root, repos = fleet['root'], fleet['repos']
    a, b = repos['dep_a'], repos['dep_b']
    scan.run_scan('tgt', source_root=fleet['src'], repos=repos, root=root)
    pinned_a = git(a, 'rev-parse', 'HEAD')
    for repo in (a, b):  # both move past their pins
        (repo / 'new.txt').write_text('x')
        git(repo, 'add', '-A')
        git(repo, 'commit', '-q', '-m', 'later')
    (b / 'dep_b.py').write_text('changed')  # b is dirty: must be refused
    assert {n: s['state'] for n, s in pins.check(root, repos=repos).items()} == {
        'dep_a': 'moved', 'dep_b': 'moved'}

    dry = pins.restore(root, repos=repos)
    assert not dry['applied'] and git(a, 'rev-parse', 'HEAD') != pinned_a
    assert dry['plan']['dep_b']['action'] == 'skip'

    done = pins.restore(root, repos=repos, apply=True)
    assert done['applied'] and git(a, 'rev-parse', 'HEAD') == pinned_a
    assert (b / 'dep_b.py').read_text() == 'changed'
    rec = store.read_json(root / 'pins' / done['undo_record'])
    assert list(rec['repos']) == ['dep_a'] and rec['repos']['dep_a']['branch']

    pins.undo(root, repos=repos, apply=True)
    assert git(a, 'symbolic-ref', '--short', 'HEAD') == rec['repos']['dep_a']['branch']
    with pytest.raises(LookupError):  # fully undone: nothing pending
        pins.undo(root, repos=repos)


def test_restore_never_fetches_a_missing_commit(fleet, tmp_path):
    root, repos = fleet['root'], fleet['repos']
    scan.run_scan('tgt', source_root=fleet['src'], repos=repos, root=root)
    pins_path = root / 'scans' / store.latest_id(root) / 'pins.json'
    data = json.loads(pins_path.read_text())
    data['pins']['dep_a']['sha'] = 'f' * 40
    pins_path.write_text(json.dumps(data))
    plan = pins.plan_restore(root, repos=repos)['plan']
    assert plan['dep_a'] == dict(action='skip', reason='pinned commit not present locally (never fetches)')


def no_priv(repo):
    return True, ['-o', 'addopts=']


def test_baseline_records_failing_ids_and_diff_localises_the_regression(fleet):
    root, repos = fleet['root'], fleet['repos']
    scan.run_scan('tgt', source_root=fleet['src'], repos=repos, root=root)
    kw = dict(package='tgt', repos=repos, gate_args=no_priv, python=sys.executable)
    b1 = baseline.load_baseline(root, baseline.run_baseline(root, **kw))
    assert b1['results']['dep_a']['status'] == 'PASS'
    assert b1['results']['dep_b']['status'] == 'NO_TESTS'
    assert '<junitxml>' in b1['results']['dep_a']['command'][-1]

    test_a = repos['dep_a'] / 'test_a.py'
    test_a.write_text(test_a.read_text().replace('FLAG = True', 'FLAG = False'))
    b2 = baseline.load_baseline(root, baseline.run_baseline(root, **kw))
    r = b2['results']['dep_a']
    assert r['status'] == 'FAIL' and r['counts']['failures'] == 1 and r['dirty'] == 1
    d = baseline.diff(b1, b2)
    assert d['changed']['dep_a']['new_failures'] == ['test_a::test_flaky']
    assert 'dep_b' not in d['changed']


def test_diff_refuses_baselines_run_with_different_commands():
    a = dict(id='a', results={'x': dict(status='PASS', command=['pytest', '-rf'])})
    b = dict(id='b', results={'x': dict(status='FAIL', command=['pytest', '-rfE'])})
    with pytest.raises(baseline.IncomparableBaselines):
        baseline.diff(a, b)


def test_suite_timeout_is_recorded_not_raised(tmp_path):
    (tmp_path / 'test_slow.py').write_text('import time\ndef test_x(): time.sleep(30)\n')
    r = baseline.run_suite(tmp_path, gate_args=no_priv, timeout=3)
    assert r['status'] == 'TIMEOUT'
