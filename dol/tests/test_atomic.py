"""Tests for ``dol.atomic``: ``atomic_write``, ``atomic_delete`` and ``Retry``."""

import os
import stat
import sys

import pytest

from dol import atomic_delete, atomic_write
from dol.atomic import Retry

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX mode bits/symlinks"
)


def _mode(p):
    return stat.S_IMODE(os.stat(p).st_mode)


def test_bytes_and_chunks_and_overwrite(tmp_path):
    p = tmp_path / "f"
    atomic_write(p, b"one")
    atomic_write(p, iter([b"t", bytearray(b"w"), memoryview(b"o")]))
    assert p.read_bytes() == b"two"
    assert os.listdir(tmp_path) == ["f"]


def test_bytes_path(tmp_path):
    p = tmp_path / "f"
    atomic_write(os.fsencode(p), b"x")
    assert p.read_bytes() == b"x" and os.listdir(tmp_path) == ["f"]
    atomic_delete(os.fsencode(p))
    assert os.listdir(tmp_path) == []


def test_empty_data(tmp_path):
    atomic_write(tmp_path / "f", iter([]))
    assert (tmp_path / "f").read_bytes() == b""


def test_str_is_refused_clearly(tmp_path):
    with pytest.raises(TypeError, match="encode"):
        atomic_write(tmp_path / "f", "text")
    assert os.listdir(tmp_path) == []


def test_failure_keeps_old_content_and_cleans_tmp(tmp_path):
    p = tmp_path / "f"
    atomic_write(p, b"old")

    def gen():
        yield b"new"
        raise ValueError

    with pytest.raises(ValueError):
        atomic_write(p, gen())
    assert p.read_bytes() == b"old" and os.listdir(tmp_path) == ["f"]


def test_missing_parent_raises_and_leaves_nothing(tmp_path):
    with pytest.raises(OSError):
        atomic_write(tmp_path / "nope" / "f", b"x")
    assert os.listdir(tmp_path) == []


def test_tmp_is_hidden_unique_and_in_same_folder(tmp_path, monkeypatch):
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append(a), real(a, b)))
    atomic_write(tmp_path / "f", b"x")
    atomic_write(tmp_path / "f", b"x")
    assert len(set(seen)) == 2
    for s in seen:
        assert os.path.dirname(s) == os.path.realpath(tmp_path)
        assert os.path.basename(s).startswith(".f.")


def test_long_names_do_not_overflow(tmp_path):
    name = "n" * 240
    atomic_write(tmp_path / name, b"x")
    assert (tmp_path / name).read_bytes() == b"x"


def test_long_non_ascii_names_do_not_overflow(tmp_path, monkeypatch):
    name = "\u6587" * 85  # 85 characters, 255 UTF-8 bytes: a legal name
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append(a), real(a, b)))
    atomic_write(tmp_path / name, b"x")
    assert (tmp_path / name).read_bytes() == b"x"
    tmp_name = os.fsencode(os.path.basename(seen[0]))
    assert len(tmp_name) <= 255  # ext4 refuses longer (ENAMETOOLONG)
    assert tmp_name.decode()  # cut on a character boundary: still valid UTF-8


def test_fsync_only_when_asked(tmp_path, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd)))
    atomic_write(tmp_path / "f", b"x")
    assert calls == []
    atomic_write(tmp_path / "f", b"x", fsync=True)
    assert len(calls) >= 1 and (tmp_path / "f").read_bytes() == b"x"


@posix_only
def test_perms_exact_despite_umask(tmp_path):
    old = os.umask(0o077)
    try:
        atomic_write(tmp_path / "f", b"x", perms=0o644)
    finally:
        os.umask(old)
    assert _mode(tmp_path / "f") == 0o644


@posix_only
def test_perms_set_before_data_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "fchmod", lambda *a: None)  # creation alone must do it
    seen = []

    def gen():
        (tmp,) = [p for p in os.listdir(tmp_path) if p.startswith(".f.")]
        seen.append(_mode(tmp_path / tmp))
        yield b"secret"

    old = os.umask(0)
    try:
        atomic_write(tmp_path / "f", gen(), perms=0o600)
    finally:
        os.umask(old)
    assert seen == [0o600]


@posix_only
def test_existing_mode_preserved(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"old")
    os.chmod(p, 0o600)
    atomic_write(p, b"new")
    assert _mode(p) == 0o600 and p.read_bytes() == b"new"


@posix_only
def test_symlink_is_followed(tmp_path):
    target, link = tmp_path / "target", tmp_path / "link"
    target.write_bytes(b"old")
    link.symlink_to(target)
    atomic_write(link, b"new")
    assert link.is_symlink() and target.read_bytes() == b"new"


# ---------------------------------------------------------------------------
# retry (Windows sharing violations, #167)


def _refuse(n, real, exc=PermissionError):
    """A stand-in for ``real`` that raises ``exc`` on its first ``n`` calls."""
    calls = []

    def op(*args):
        calls.append(args)
        if len(calls) <= n:
            raise exc("file in use")
        return real(*args)

    return op, calls


NO_SLEEP = Retry(on=(PermissionError,), backoff_s=0)


def test_default_retries_only_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr("dol.atomic.time.sleep", lambda s: None)
    op, calls = _refuse(99, os.replace)
    monkeypatch.setattr(os, "replace", op)
    with pytest.raises(PermissionError):
        atomic_write(tmp_path / "f", b"x")  # default retry
    assert len(calls) == (Retry().attempts if os.name == "nt" else 1)
    assert os.listdir(tmp_path) == []


def test_interrupt_during_backoff_keeps_old_and_cleans_tmp(tmp_path, monkeypatch):
    p = tmp_path / "f"
    atomic_write(p, b"old")

    def interrupted(s):
        raise KeyboardInterrupt

    monkeypatch.setattr("dol.atomic.time.sleep", interrupted)
    op, _ = _refuse(99, os.replace)
    monkeypatch.setattr(os, "replace", op)
    with pytest.raises(KeyboardInterrupt):
        atomic_write(p, b"new", retry=Retry(on=(PermissionError,)))
    assert p.read_bytes() == b"old" and os.listdir(tmp_path) == ["f"]


def test_retry_backoff_doubles_and_reraises_the_last(monkeypatch):
    sleeps = []
    monkeypatch.setattr("dol.atomic.time.sleep", sleeps.append)
    op, calls = _refuse(99, lambda: None)
    with pytest.raises(PermissionError):
        Retry(on=(PermissionError,), attempts=4, backoff_s=0.01)(op)
    assert len(calls) == 4 and sleeps == [0.01, 0.02, 0.04]


def test_retry_does_not_catch_other_errors():
    op, calls = _refuse(1, lambda: None, exc=FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        NO_SLEEP(op)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(attempts=0), ValueError),
        (dict(attempts=2.5), ValueError),
        (dict(backoff_s=-1), ValueError),
        (dict(backoff_s=float("nan")), ValueError),
        (dict(on=(PermissionError, "nope")), TypeError),
        (dict(on=(int,)), TypeError),
    ],
)
def test_retry_refuses_bad_config_at_construction(kwargs, error):
    with pytest.raises(error):
        Retry(**kwargs)


def test_retry_on_is_normalized_to_a_tuple():
    assert Retry(on=PermissionError).on == (PermissionError,)
    assert Retry(on=[PermissionError]).on == (PermissionError,)
    hash(Retry(on=[PermissionError]))  # still a hashable value


def test_write_retries_a_refused_replace(tmp_path, monkeypatch):
    p = tmp_path / "f"
    atomic_write(p, b"old")
    op, calls = _refuse(2, os.replace)
    monkeypatch.setattr(os, "replace", op)
    atomic_write(p, b"new", retry=NO_SLEEP)
    assert len(calls) == 3 and p.read_bytes() == b"new"
    assert os.listdir(tmp_path) == ["f"]


def test_write_gives_up_keeps_old_and_cleans_tmp(tmp_path, monkeypatch):
    p = tmp_path / "f"
    atomic_write(p, b"old")
    op, calls = _refuse(99, os.replace)
    monkeypatch.setattr(os, "replace", op)
    with pytest.raises(PermissionError):
        atomic_write(
            p, b"new", retry=Retry(on=(PermissionError,), attempts=3, backoff_s=0)
        )
    assert len(calls) == 3 and p.read_bytes() == b"old"
    assert os.listdir(tmp_path) == ["f"]


def test_single_attempt_calls_replace_once(tmp_path, monkeypatch):
    op, calls = _refuse(1, os.replace)
    monkeypatch.setattr(os, "replace", op)
    with pytest.raises(PermissionError):
        atomic_write(
            tmp_path / "f", b"x", retry=Retry(on=(PermissionError,), attempts=1)
        )
    assert len(calls) == 1 and os.listdir(tmp_path) == []


def test_delete_retries_and_removes(tmp_path, monkeypatch):
    p = tmp_path / "f"
    atomic_write(p, b"x")
    op, calls = _refuse(2, os.unlink)
    monkeypatch.setattr(os, "unlink", op)
    atomic_delete(p, retry=NO_SLEEP)
    assert len(calls) == 3 and not p.exists()


def test_delete_missing_raises_without_retrying(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr("dol.atomic.time.sleep", sleeps.append)
    with pytest.raises(FileNotFoundError):
        atomic_delete(tmp_path / "nope", retry=Retry(on=(PermissionError,)))
    assert sleeps == []


@posix_only
def test_delete_removes_the_link_not_its_target(tmp_path):
    target, link = tmp_path / "target", tmp_path / "link"
    target.write_bytes(b"x")
    link.symlink_to(target)
    atomic_delete(link)
    assert not link.is_symlink() and target.read_bytes() == b"x"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing semantics")
def test_windows_reader_holding_file_open_is_outlasted(tmp_path):
    import threading

    p = tmp_path / "f"
    atomic_write(p, b"old")
    fh = open(p, "rb")  # no FILE_SHARE_DELETE: replace/unlink are refused
    threading.Timer(0.05, fh.close).start()
    atomic_write(p, b"new")
    assert p.read_bytes() == b"new"
    fh = open(p, "rb")
    threading.Timer(0.05, fh.close).start()
    atomic_delete(p)
    assert not p.exists()
