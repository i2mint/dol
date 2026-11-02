"""Tests for ``dol.atomic.update_json``: a locked read-modify-write of one JSON file."""

import contextlib
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


@contextlib.contextmanager
def _unlockable():
    yield False  # like file_lock on a filesystem without locks


def test_warns_at_the_caller_when_the_lock_is_not_held(tmp_path):
    p = tmp_path / "m.json"
    with pytest.warns(UserWarning, match="could not lock") as record:
        update_json(p, lambda m: m.update(a=1), lock=_unlockable())
    assert record[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        update_json(p, lambda m: m.update(a=2), lock=_unlockable(), warn_unlocked=False)
    assert json.loads(p.read_text()) == {"a": 2}


def test_any_context_manager_is_a_lock(tmp_path):
    p = tmp_path / "m.json"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # yields None: assumed held
        update_json(p, lambda m: m.update(a=1), lock=contextlib.nullcontext())
    assert json.loads(p.read_text()) == {"a": 1}


@pytest.mark.parametrize(
    "default, change",
    [
        (int, lambda n: n + 1),  # a scalar cannot be changed in place
        (dict, lambda m: {**m, "z": 1}),  # returning a new object is not writing it
    ],
)
def test_an_update_is_never_silently_lost(tmp_path, default, change):
    p = tmp_path / "m.json"
    if default is int:
        with pytest.raises(TypeError, match="in place"):
            update_json(p, change, default=default)
    else:
        assert update_json(p, change) == {"z": 1}
        assert not p.exists()  # documented: what change returns is not written


def test_a_scalar_file_is_refused(tmp_path):
    p = tmp_path / "m.json"
    p.write_text("5")
    with pytest.raises(TypeError):
        update_json(p, lambda n: n)
    assert p.read_text() == "5"


def test_nested_mutation_is_detected(tmp_path):
    p = tmp_path / "m.json"
    p.write_text('{"a": {"b": [1]}}')
    update_json(p, lambda m: m["a"]["b"].append(2))
    assert json.loads(p.read_text()) == {"a": {"b": [1, 2]}}


def test_dumps_may_return_bytes(tmp_path):
    p = tmp_path / "m.json"
    update_json(p, lambda m: m.update(a=1), dumps=lambda o: json.dumps(o).encode())
    assert json.loads(p.read_bytes()) == {"a": 1}


def test_missing_folder_is_named_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="folder"):
        update_json(tmp_path / "nope" / "m.json", lambda m: None)


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
def test_two_links_to_one_file_share_its_lock(tmp_path):
    real = tmp_path / "real.json"
    real.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    update_json(link, lambda m: m.update(a=1))
    assert (tmp_path / "real.json.lock").exists()
    assert not (tmp_path / "link.json.lock").exists()
    assert json.loads(real.read_text()) == {"a": 1}


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
