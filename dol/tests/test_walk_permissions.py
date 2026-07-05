"""Regression test for Issue #3: the recursive file walk must skip unreadable dirs.

``FileBytesReader(gettempdir())`` used to raise ``PermissionError`` because ``os.listdir``
hits OS-protected subdirectories (e.g. macOS's ``.../com.apple.*/TemporaryItems/``). The
walk now skips directories it cannot read instead of crashing.
"""

import os
import sys

import pytest

from dol.filesys import FileBytesReader, paths_in_dir


def test_recursive_walk_skips_unreadable_dirs(tmp_path):
    # a readable file at the root
    (tmp_path / "readable.txt").write_bytes(b"hi")
    # a subdirectory we will make unreadable
    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / "secret.txt").write_bytes(b"nope")

    if sys.platform.startswith("win"):
        pytest.skip("POSIX directory permissions don't apply on Windows")

    os.chmod(protected, 0o000)
    try:
        # If we can still list it (e.g. running as root), the scenario is unreachable.
        try:
            os.listdir(protected)
            pytest.skip("cannot make a directory unreadable in this environment")
        except PermissionError:
            pass

        # The walk must not raise, and must skip the unreadable subdir's contents.
        s = FileBytesReader(str(tmp_path))
        keys = list(s)  # was: PermissionError
        assert any(k.endswith("readable.txt") for k in keys)
        assert not any(k.endswith("secret.txt") for k in keys)
    finally:
        os.chmod(protected, 0o755)  # restore so tmp cleanup can remove it


def test_paths_in_dir_on_missing_dir_is_empty(tmp_path):
    missing = str(tmp_path / "does_not_exist")
    assert list(paths_in_dir(missing)) == []
