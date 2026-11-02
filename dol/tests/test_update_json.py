"""Tests for ``dol.atomic.update_json``: a locked read-modify-write of one JSON file."""

import functools
import json
import multiprocessing as mp
import os
import warnings

import pytest

from dol import FileLock, update_json


def test_creates_from_default_and_returns_changes_result(tmp_path):
    p = tmp_path / "m.json"
    assert update_json(p, lambda m: m.update(a=1) or "done") == "done"
    assert json.loads(p.read_text()) == {"a": 1}


def test_custom_default(tmp_path):
    p = tmp_path / "l.json"
    update_json(p, lambda xs: xs.append(1), default=list)
    assert json.loads(p.read_text()) == [1]


def test_noop_neither_creates_nor_touches(tmp_path):
    p = tmp_path / "m.json"
    update_json(p, lambda m: None)
    assert not p.exists()
    p.write_text('{ "a" :  1 }')  # hand formatted
    mtime = os.stat(p).st_mtime_ns
    update_json(p, lambda m: m.get("a"))
    assert p.read_text() == '{ "a" :  1 }' and os.stat(p).st_mtime_ns == mtime


def test_a_raising_change_writes_nothing(tmp_path):
    p = tmp_path / "m.json"
    p.write_text('{"a": 1}')

    def change(m):
        m["a"] = 2
        raise RuntimeError("abort")

    with pytest.raises(RuntimeError):
        update_json(p, change)
    assert json.loads(p.read_text()) == {"a": 1}


def test_invalid_json_raises_and_keeps_the_file(tmp_path):
    p = tmp_path / "m.json"
    p.write_text('{"a": 1')  # torn
    with pytest.raises(ValueError, match="nothing was written"):
        update_json(p, lambda m: m.update(b=2))
    assert p.read_text() == '{"a": 1'


def test_a_read_error_other_than_missing_propagates(tmp_path, monkeypatch):
    p = tmp_path / "m.json"
    p.write_text('{"a": 1}')
    real_open = open

    def flaky_open(file, *a, **k):
        if os.fspath(file) == str(p):
            raise PermissionError("transient")
        return real_open(file, *a, **k)

    monkeypatch.setattr("builtins.open", flaky_open)
    with pytest.raises(PermissionError):
        update_json(p, lambda m: m.update(b=2))
    monkeypatch.undo()
    assert json.loads(p.read_text()) == {"a": 1}  # not replaced by default + change


def test_custom_codec(tmp_path):
    p = tmp_path / "m.json"
    update_json(
        p, lambda m: m.update(a=1), dumps=functools.partial(json.dumps, indent=2)
    )
    assert p.read_text() == '{\n  "a": 1\n}'


def test_warns_when_the_lock_is_not_held(tmp_path):
    p = tmp_path / "m.json"
    lock = FileLock(p)
    lock._try_lock = lambda: False  # as on a filesystem without locks
    with pytest.warns(UserWarning, match="could not lock"):
        update_json(p, lambda m: m.update(a=1), lock=lock)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        update_json(p, lambda m: m.update(a=2), lock=lock, warn_unlocked=False)
    assert json.loads(p.read_text()) == {"a": 2}
    lock.close()


def test_an_injected_lock_is_used_and_left_open(tmp_path):
    p = tmp_path / "m.json"
    lock = FileLock(p, sidecar=".mylock")
    update_json(p, lambda m: m.update(a=1), lock=lock)
    assert (tmp_path / "m.json.mylock").exists() and lock._fh is not None
    lock.close()


def _increment_n(path, n, start):
    start.wait()
    for _ in range(n):
        update_json(path, lambda m: m.update(n=m.get("n", 0) + 1))


def test_processes_never_lose_an_update(tmp_path):
    p = str(tmp_path / "count.json")
    start = mp.Barrier(4)
    procs = [mp.Process(target=_increment_n, args=(p, 25, start)) for _ in range(4)]
    [pr.start() for pr in procs]
    [pr.join(60) for pr in procs]
    assert all(pr.exitcode == 0 for pr in procs)
    assert json.loads(open(p).read()) == {"n": 100}
