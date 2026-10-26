"""Tests for ``dol.atomic.atomic_write``."""

import os
import stat
import sys

import pytest

from dol import atomic_write


def test_bytes_and_chunks_and_overwrite(tmp_path):
    p = tmp_path / "f"
    atomic_write(p, b"one")
    atomic_write(p, iter([b"t", bytearray(b"w"), memoryview(b"o")]))
    assert p.read_bytes() == b"two"
    assert os.listdir(tmp_path) == ["f"]


@pytest.mark.parametrize("exclusive", [True, False])
def test_failure_keeps_old_content_and_cleans_tmp(tmp_path, exclusive):
    p = tmp_path / "f"
    atomic_write(p, b"old")

    def gen():
        yield b"new"
        raise ValueError

    with pytest.raises(ValueError):
        atomic_write(p, gen(), exclusive=exclusive)
    assert p.read_bytes() == b"old" and os.listdir(tmp_path) == ["f"]


def test_non_exclusive_uses_fixed_tmp_name(tmp_path, monkeypatch):
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append(a), real(a, b)))
    atomic_write(tmp_path / "f", b"x", exclusive=False, tmp_suffix=".part")
    assert seen == [str(tmp_path / "f") + ".part"]


def test_tmp_names_are_unique_when_exclusive(tmp_path, monkeypatch):
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append(a), real(a, b)))
    atomic_write(tmp_path / "f", b"x")
    atomic_write(tmp_path / "f", b"x")
    assert len(set(seen)) == 2 and all(s.endswith(".tmp") for s in seen)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_perms_exact_despite_umask(tmp_path):
    old = os.umask(0o077)
    try:
        atomic_write(tmp_path / "f", b"x", perms=0o644)
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "f").stat().st_mode) == 0o644


def test_fsync_runs(tmp_path):
    atomic_write(tmp_path / "f", b"x", fsync=True)
    assert (tmp_path / "f").read_bytes() == b"x"


def test_missing_parent_raises_and_leaves_nothing(tmp_path):
    with pytest.raises(OSError):
        atomic_write(tmp_path / "nope" / "f", b"x")
    assert os.listdir(tmp_path) == []
