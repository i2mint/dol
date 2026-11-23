"""The ``Files`` walk (i2mint/dol#219): ``os.scandir`` entries, and walked keys that
skip the containment check unless a subclass customises validation."""

import os

import pytest

from dol.filesys import DirCollection, FileCollection


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "x.txt").write_text("1")
    (tmp_path / "y.json").write_text("2")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "h.txt").write_text("3")
    return tmp_path


def _rel(store, root):
    return sorted(os.path.relpath(k, root).replace(os.sep, "/") for k in store)


def test_files_and_dirs_are_walked_recursively(tree):
    assert _rel(FileCollection(str(tree)), tree) == ["a/b/x.txt", "y.json"]
    assert _rel(DirCollection(str(tree)), tree) == ["a", "a/b"]


def test_dirs_keep_their_trailing_separator(tree):
    assert all(k.endswith(os.sep) for k in DirCollection(str(tree)))


def test_hidden_entries_only_on_request(tree):
    assert _rel(FileCollection(str(tree), include_hidden=True), tree) == [
        ".hidden/h.txt",
        "a/b/x.txt",
        "y.json",
    ]


def test_max_levels_bounds_the_walk(tree):
    assert _rel(FileCollection(str(tree), max_levels=0), tree) == ["y.json"]
    assert _rel(DirCollection(str(tree), max_levels=0), tree) == ["a"]


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_symlinks_are_followed_and_broken_ones_skipped(tree, tmp_path_factory):
    target = tmp_path_factory.mktemp("target")
    (target / "t.txt").write_text("5")
    os.symlink(target, tree / "linkdir")
    os.symlink(tree / "missing", tree / "broken")
    assert _rel(FileCollection(str(tree)), tree) == [
        "a/b/x.txt",
        "linkdir/t.txt",
        "y.json",
    ]


def test_the_key_pattern_still_filters_walked_keys(tree):
    s = FileCollection(str(tree), subpath=os.path.join("a", "{rest}"))
    assert _rel(s, tree) == ["a/b/x.txt"]


def test_a_subclass_validation_applies_to_walked_keys(tree):
    class OnlyJson(FileCollection):
        def is_valid_key(self, k):
            return super().is_valid_key(k) and k.endswith(".json")

    assert _rel(OnlyJson(str(tree)), tree) == ["y.json"]


def test_a_subclass_containment_applies_to_walked_keys(tree):
    class NotUnderA(FileCollection):
        def is_in_root(self, k):
            return super().is_in_root(k) and os.sep + "a" + os.sep not in k

    assert _rel(NotUnderA(str(tree)), tree) == ["y.json"]


def test_files_lists_relative_keys(tree):
    from dol import Files

    assert sorted(Files(str(tree))) == ["a/b/x.txt".replace("/", os.sep), "y.json"]


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
@pytest.mark.parametrize(
    "make_link",
    [
        pytest.param(lambda root: os.symlink(root / "loop", root / "loop"), id="self-loop"),
        pytest.param(lambda root: os.symlink(root / "y.json" / "x", root / "through_file"), id="enotdir"),
    ],
)
def test_unresolvable_symlinks_are_skipped(tree, make_link):
    make_link(tree)
    assert _rel(FileCollection(str(tree)), tree) == ["a/b/x.txt", "y.json"]
    assert _rel(DirCollection(str(tree)), tree) == ["a", "a/b"]


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_a_directory_cycle_ends(tree):
    os.symlink(tree, tree / "a" / "back")
    keys = list(FileCollection(str(tree)))  # ends once the OS refuses the depth
    assert len(keys) > 2


def test_files_written_during_iteration_are_not_yielded(tmp_path):
    n = 3000  # large enough that a lazy directory read would see new entries
    for i in range(n):
        (tmp_path / f"f{i}").write_text("")
    seen = 0
    for k in FileCollection(str(tmp_path)):
        seen += 1
        with open(k + ".bak", "w"):
            pass
    assert seen == n


def test_an_instance_level_validator_applies_to_walked_keys(tree):
    s = FileCollection(str(tree))
    s.is_valid_key = lambda k: k.endswith(".json")
    assert _rel(s, tree) == ["y.json"]
