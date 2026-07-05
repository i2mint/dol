"""Combinatorial regression tests for ``mk_dirs_if_missing`` (Issue #14).

Issue #14 reported two fragilities, both of which are resolved on current dol; these
tests lock the behavior in (the issue explicitly asked for "more combinatorial wrapper
testing"):

1. **Read ops on a fresh store whose root dir does not exist yet** return empty /
   ``KeyError`` instead of raising a filesystem error (``list`` -> ``[]``, ``len`` -> 0,
   ``in`` -> ``False``, ``__getitem__`` -> ``KeyError``).
2. **``mk_dirs_if_missing`` composes with ``wrap_kvs`` in BOTH orders** (the
   ``mk_dirs_if_missing(wrap_kvs(G))`` ordering previously raised ``FileNotFoundError: ''``).
"""

import os

import pytest

from dol import Files, JsonFiles, mk_dirs_if_missing, wrap_kvs


def _nonexistent_rootdir(tmp_path):
    """A rootdir two levels below tmp_path that does NOT exist yet."""
    return os.path.join(str(tmp_path), "A", "B")


def test_read_ops_on_fresh_missing_dir(tmp_path):
    rootdir = _nonexistent_rootdir(tmp_path)
    assert not os.path.exists(rootdir)
    s = mk_dirs_if_missing(Files)(rootdir)
    assert list(s) == []
    assert len(s) == 0
    assert "x" not in s
    with pytest.raises(KeyError):
        s["missing"]


class _G(Files):
    """A trivial Files subclass, to exercise wrapping a user-defined store class."""


@pytest.mark.parametrize(
    "factory",
    [
        lambda: mk_dirs_if_missing(Files),
        lambda: mk_dirs_if_missing(_G),
        lambda: wrap_kvs(mk_dirs_if_missing(_G)),
        lambda: mk_dirs_if_missing(wrap_kvs(_G)),  # used to raise FileNotFoundError: ''
        lambda: mk_dirs_if_missing(mk_dirs_if_missing(Files)),
    ],
    ids=[
        "mk_dirs(Files)",
        "mk_dirs(G)",
        "wrap_kvs(mk_dirs(G))",
        "mk_dirs(wrap_kvs(G))",
        "mk_dirs(mk_dirs(Files))",
    ],
)
def test_wrap_ordering_writes_and_makes_dirs(tmp_path, factory):
    s = factory()(_nonexistent_rootdir(tmp_path))
    # write to a key at the (missing) root: the root dirs are created on write
    s["foo"] = b"bar"
    assert s["foo"] == b"bar"
    # write to a key whose intermediate subdirs also don't exist yet
    deep = os.path.join("sub", "deep", "foo")
    s[deep] = b"baz"
    assert s[deep] == b"baz"
    assert set(s) >= {"foo", deep}


def test_mk_dirs_with_value_transform(tmp_path):
    """mk_dirs_if_missing composes with a value codec (JsonFiles) + deep keys."""
    s = mk_dirs_if_missing(JsonFiles)(_nonexistent_rootdir(tmp_path))
    s["a.json"] = {"x": 1}
    assert s["a.json"] == {"x": 1}
    deep = os.path.join("sub", "deep", "b.json")
    s[deep] = {"y": 2}
    assert s[deep] == {"y": 2}
