"""Tests for ``dol.atomic.update_json``: a locked read-modify-write of one JSON file."""

import contextlib
import functools
import json
import multiprocessing as mp
import os
import stat
import warnings

import pytest

from dol import FileLock, update_json
from dol.atomic import InvalidJSONError, atomic_write


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


def test_invalid_json_error_is_distinct_from_changes_value_errors(tmp_path):
    p = tmp_path / "m.json"
    p.write_text("{")
    with pytest.raises(InvalidJSONError):
        update_json(p, lambda m: None)
    p.write_text("{}")

    def change(m):
        raise ValueError("the caller's own refusal")

    with pytest.raises(ValueError) as info:
        update_json(p, change)
    assert not isinstance(info.value, InvalidJSONError)


@pytest.mark.parametrize("bad", ["", "folder" + os.sep])
def test_a_path_that_names_no_file_is_refused(tmp_path, monkeypatch, bad):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="file path"):
        update_json(bad, lambda m: m.update(a=1))
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize("yielded", [0, False])
def test_a_falsy_yield_counts_as_not_held(tmp_path, yielded):
    @contextlib.contextmanager
    def lock():
        yield yielded

    with pytest.warns(UserWarning, match="could not lock"):
        update_json(tmp_path / "m.json", lambda m: None, lock=lock())


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
def test_a_symlink_takes_the_same_lock_as_file_lock(tmp_path):
    real = tmp_path / "real.json"
    real.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    update_json(link, lambda m: m.update(a=1))
    # the lock sits beside the real file, as file_lock(link) puts it (#188)
    assert (tmp_path / "real.json.lock").exists()
    assert not (tmp_path / "link.json.lock").exists()
    assert json.loads(real.read_text()) == {"a": 1}


def test_write_is_how_the_bytes_land_under_the_lock_and_only_on_a_change(tmp_path):
    calls = []

    def write(path, data):
        with pytest.raises(RuntimeError):  # this thread holds path's lock right now
            with FileLock(path):
                pass
        calls.append((path, data))
        atomic_write(path, data, perms=0o600)

    p = tmp_path / "m.json"
    update_json(p, lambda m: m.update(a=1), write=write)
    update_json(p, lambda m: None, write=write)  # no change: nothing written
    assert calls == [(str(p), b'{"a": 1}')]
    if os.name != "nt":
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
