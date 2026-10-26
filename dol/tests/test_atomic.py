"""Tests for ``dol.atomic.atomic_write``."""

import os
import stat
import sys

import pytest

from dol import atomic_write

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
def test_perms_set_before_data_is_written(tmp_path):
    seen = []

    def gen():
        (tmp,) = [p for p in os.listdir(tmp_path) if p.startswith(".f.")]
        seen.append(_mode(tmp_path / tmp))
        yield b"secret"

    old = os.umask(0o022)
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
