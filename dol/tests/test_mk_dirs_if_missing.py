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

from dol import Files, JsonFiles, filt_iter, mk_dirs_if_missing, wrap_kvs


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


# --- Issue #112: the preset only makes dirs; the wrapper does the one real write ---


class _SubOfDecorated(mk_dirs_if_missing(Files)):
    """A subclass of a decorated class (the graze ``LocalGrazed`` shape). The old preset
    wrote via ``super(type(self), self)``, which re-entered itself for such subclasses."""


def test_subclass_of_decorated_class_writes_once(tmp_path, monkeypatch):
    from dol.filesys import FileBytesPersister

    calls = []
    orig = FileBytesPersister.__setitem__
    monkeypatch.setattr(
        FileBytesPersister,
        "__setitem__",
        lambda self, k, v: (calls.append(k), orig(self, k, v))[1],
    )
    s = _SubOfDecorated(_nonexistent_rootdir(tmp_path))
    s["x/y/z"] = b"1"
    assert s["x/y/z"] == b"1"
    assert len(calls) == 1  # one write, not one per recursion level (or two per set)


@pytest.mark.parametrize("store_cls", [mk_dirs_if_missing(Files), _SubOfDecorated])
def test_write_errors_propagate(tmp_path, store_cls):
    s = store_cls(str(tmp_path))
    os.makedirs(os.path.join(str(tmp_path), "a_dir"))
    with pytest.raises(KeyError):  # writing to a path that is a directory
        s["a_dir"] = b"nope"
    with pytest.raises(KeyError):  # bytes store given a str: used to be swallowed
        s["sub/file"] = "not bytes"


def test_no_dirs_for_non_path_inner_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = mk_dirs_if_missing(wrap_kvs(dict, id_of_key=lambda k: k))()
    s["rel/dir/key"] = 1
    assert s["rel/dir/key"] == 1
    assert not (tmp_path / "rel").exists()  # never mkdir under the CWD


@pytest.mark.parametrize(
    "factory, rootdir, key",
    [
        (mk_dirs_if_missing(Files), "rel/root", "sub/deep/x"),
        (_SubOfDecorated, "rel/root", "sub/deep/x"),
        (mk_dirs_if_missing(Files), "", "rel/root/e/f/x"),
    ],
    ids=["relative-rootdir", "relative-rootdir-subclass", "empty-rootdir"],
)
def test_dirs_made_for_relative_paths(tmp_path, monkeypatch, factory, rootdir, key):
    monkeypatch.chdir(tmp_path)
    os.makedirs("rel/root")
    s = factory(rootdir)
    s[key] = b"hi"
    assert s[key] == b"hi"
    assert (tmp_path / "rel" / "root").is_dir()


# ---- a key a layer below refuses leaves no stray directory (i2mint/dol#122) -------------


def _txt_only(k):
    return k.endswith(".txt")


@pytest.mark.parametrize(
    "factory",
    [
        lambda root: mk_dirs_if_missing(filt_iter(Files, filt=_txt_only))(root),
        lambda root: filt_iter(mk_dirs_if_missing(Files), filt=_txt_only)(root),
        lambda root: mk_dirs_if_missing(filt_iter(Files(root), filt=_txt_only)),
        lambda root: mk_dirs_if_missing(
            mk_dirs_if_missing(filt_iter(Files, filt=_txt_only))
        )(root),
    ],
    ids=["mk(filt(cls))", "filt(mk(cls))", "mk(filt(inst))", "mk(mk(filt))"],
)
def test_rejected_key_makes_no_directory(tmp_path, factory):
    root = str(tmp_path) + os.sep
    s = factory(root)
    s["ok/a.txt"] = b"x"
    with pytest.raises(KeyError):
        s["stray/deep/b.bin"] = b"x"
    assert sorted(os.listdir(root)) == ["ok"]


def test_failed_write_removes_only_the_directories_it_made(tmp_path):
    root = str(tmp_path) + os.sep
    os.makedirs(root + "keep/sub")
    s = mk_dirs_if_missing(filt_iter(Files, filt=_txt_only))(root)
    with pytest.raises(KeyError):
        s["keep/sub/new1/new2/b.bin"] = b"x"
    assert os.path.isdir(root + "keep/sub") and not os.path.exists(root + "keep/sub/new1")


def test_failed_write_keeps_a_directory_that_got_content_meanwhile(tmp_path):
    root = str(tmp_path) + os.sep

    class Racy(Files):
        def __setitem__(self, k, v):
            open(os.path.join(root, "d", "other.txt"), "wb").close()  # someone else writes
            raise KeyError(k)

    s = mk_dirs_if_missing(Racy)(root)
    with pytest.raises(KeyError):
        s["d/x.txt"] = b"x"
    assert os.listdir(root + "d") == ["other.txt"]  # not empty: not removed


def test_a_directory_made_meanwhile_by_someone_else_is_not_removed(tmp_path, monkeypatch):
    """Only directories this call made are cleaned up, not ones that appeared in between."""
    import dol.filesys as fs

    root = str(tmp_path) + os.sep
    real_mkdir = os.mkdir
    calls = []

    def mkdir_after_a_rival(path, *a, **k):
        if not calls:  # a rival creates the first directory just before we do
            calls.append(path)
            real_mkdir(path)
        return real_mkdir(path, *a, **k)

    monkeypatch.setattr(fs.os, "mkdir", mkdir_after_a_rival)
    s = mk_dirs_if_missing(filt_iter(Files, filt=_txt_only))(root)
    with pytest.raises(KeyError):
        s["rival/mine/b.bin"] = b"x"
    assert os.path.isdir(root + "rival") and not os.path.exists(root + "rival/mine")
