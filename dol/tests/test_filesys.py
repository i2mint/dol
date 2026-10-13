"""Test filesys objects."""

import os
from functools import partial
import tempfile
from pathlib import Path
from collections.abc import Mapping
import pytest

from dol.tests.utils_for_tests import mk_test_store_from_keys, mk_tmp_local_store
from dol.filesys import mk_dirs_if_missing, TextFiles, process_path


# --------------------------------------------------------------------------------------
# Utils


def all_folder_paths_under_folder(rootpath: str, include_rootpath=False):
    """Return all folder paths under folderpath."""
    from pathlib import Path

    rootpath = Path(rootpath)
    folderpaths = (str(p) for p in rootpath.glob("**/") if p.is_dir())
    if not include_rootpath:
        folderpaths = filter(lambda x: x != str(rootpath), folderpaths)
    return folderpaths


def delete_all_folders_under_folder(rootpath: str, include_rootpath=False):
    """Delete all folders under folderpath."""
    import shutil

    rootpath = Path(rootpath)
    if Path(rootpath).is_dir():
        for p in all_folder_paths_under_folder(
            rootpath, include_rootpath=include_rootpath
        ):
            p = Path(p)
            if p.is_dir():
                shutil.rmtree(p)


def empty_directory(s, path_must_include=("test_mk_dirs_if_missing",)):
    if isinstance(path_must_include, str):
        path_must_include = (path_must_include,)

    if not all(substr in s for substr in path_must_include):
        raise ValueError(
            f"Path '{s}' does not include any of the substrings: {path_must_include}.\n"
            "This is a safeguard. For your safety, I will delete nothing!"
        )

    import os, shutil

    try:
        for item in os.scandir(s):
            if item.is_dir():
                shutil.rmtree(item.path)
            else:
                os.remove(item.path)
    except FileNotFoundError:
        pass


# TODO: Should have a more general version of this that works with any MutableMapping
#  store as target store (instead of dirpath).
#  That's easy -- but then we need to also be able to make a filesys target store
#  that it works with (need to make folders on write, etc.)
def populate_folder(dirpath, contents: Mapping):
    """Populate a folder with the given (Mapping) contents."""
    for key, content in contents.items():
        path = os.path.join(dirpath, key)
        if isinstance(content, Mapping):
            os.makedirs(path, exist_ok=True)
            populate_folder(path, content)
        else:
            if isinstance(content, str):
                data_type = "s"
            elif isinstance(content, bytes):
                data_type = "b"
            else:
                raise ValueError(f"Unsupported type: {type(content)}")
            with open(path, "w" + data_type) as f:
                f.write(content)


# --------------------------------------------------------------------------------------
# Tests


def test_process_path():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Separate components (not "foo/bar"): process_path normalizes separators
        # (abspath), so a literal "/" would become "\\" on Windows and break the
        # equality assertions below.
        temp_path = os.path.join(temp_dir, "foo", "bar")

        output_path = process_path(temp_path)
        assert output_path == temp_path
        assert not os.path.exists(output_path)

        output_path = process_path(temp_path, expanduser=False)
        assert output_path == temp_path
        assert not os.path.exists(output_path)

        with pytest.raises(AssertionError):
            output_path = process_path(temp_path, assert_exists=True)

        output_path = process_path(temp_path, ensure_dir_exists=True)
        assert output_path == temp_path
        assert os.path.exists(output_path)

        output_path = process_path(temp_path, assert_exists=True)
        assert output_path == temp_path
        assert os.path.exists(output_path)

        # If path doesn't end with a (system file separator) slash, add one:
        output_path = process_path(temp_path, ensure_endswith_slash=True)
        assert output_path == temp_path + os.path.sep

        # If path ends with a (system file separator) slash, remove it.
        output_path = process_path(
            temp_path + os.path.sep, ensure_does_not_end_with_slash=True
        )
        assert output_path == temp_path


def test_json_files():
    from dol import JsonFiles, Jsons
    from pathlib import Path
    import os

    t = mk_tmp_local_store("test_mk_dirs_if_missing", make_dirs_if_missing=False)
    empty_directory(t.rootdir, path_must_include="test_mk_dirs_if_missing")
    rootdir = t.rootdir

    s = JsonFiles(rootdir)
    s["foo"] = {"bar": 1}
    assert s["foo"] == {"bar": 1}
    foo_path = Path(os.path.join(rootdir, "foo"))
    assert foo_path.is_file(), "Should have created a file"
    assert (
        foo_path.read_text() == '{\n    "bar": 1\n}'
    ), "Should be json encoded, with indent, so on multiple lines"

    ss = Jsons(rootdir)
    assert "foo" not in ss, "foo should be filtered out because no .json extension"
    ss["apple"] = {"crumble": True}
    assert "apple" in ss
    assert "apple" in set(ss)  # which is different than 'apple' in ss
    assert ss["apple"] == {"crumble": True}
    apple_path = Path(os.path.join(rootdir, "apple.json"))
    assert apple_path.is_file(), "Should have created a file (with .json extension)"
    assert (
        apple_path.read_text() == '{\n    "crumble": true\n}'
    ), "Should be json encoded, with indent, so on multiple lines"


def test_mk_dirs_if_missing():
    s = mk_tmp_local_store("test_mk_dirs_if_missing", make_dirs_if_missing=False)
    empty_directory(s.rootdir, path_must_include="test_mk_dirs_if_missing")
    with pytest.raises(KeyError):
        s["this/path/does/not/exist"] = "hello"
    ss = mk_dirs_if_missing(s)
    ss["this/path/does/not/exist"] = "hello"  # this should work now
    assert ss["this/path/does/not/exist"] == "hello"

    # # It works on classes too:
    # TextFilesWithAutoMkdir = mk_tmp_local_store(TextFiles)
    # sss = TextFilesWithAutoMkdir(s.rootdir)
    # assert sss["another/path/that/does/not/exist"] == "hello"


def test_subfolder_stores():
    import os
    import tempfile
    from dol import Files
    from dol.tests.utils_for_tests import mk_tmp_local_store

    # from dol.kv_codecs import KeyCodecs
    # from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define the folder structure and contents
        data = {
            "folder1": {
                "subfolder": {
                    "apple.p": b"pie",
                },
                "day.doc": b"time",
            },
            "folder2": {
                "this.txt": b"that",
                "over.json": b"there",
            },
        }

        # Create the directory structure in the temporary directory
        populate_folder(temp_dir, data)

        # Now import the function to be tested
        from dol.filesys import subfolder_stores

        # Invoke subfolder_stores with the temporary directory
        stores = subfolder_stores(
            root_folder=temp_dir,
            max_levels=None,
            include_hidden=False,
            relative_paths=True,
            slash_suffix=False,
            folder_to_store=Files,
        )

        # Collect the keys (subfolder paths)
        store_keys = set(stores.keys())

        # Expected subfolder paths (relative to temp_dir)
        expected_subfolders = {
            "folder1",
            os.path.join("folder1", "subfolder"),
            "folder2",
        }

        # Assert that the discovered subfolders match the expected ones
        assert (
            store_keys == expected_subfolders
        ), f"Expected {expected_subfolders}, got {store_keys}"

        # Test that the stores can access the files in their respective folders
        # Testing folder1
        folder1_store = stores["folder1"]
        assert isinstance(folder1_store, Files)
        assert set(folder1_store.keys()) == {"day.doc", os.path.join("subfolder", "apple.p")}
        assert folder1_store["day.doc"] == b"time"

        # Testing folder1/subfolder
        subfolder_store = stores[os.path.join("folder1", "subfolder")]
        assert isinstance(subfolder_store, Files)
        assert set(subfolder_store.keys()) == {"apple.p"}
        assert subfolder_store["apple.p"] == b"pie"

        # Testing folder2
        folder2_store = stores["folder2"]
        assert isinstance(folder2_store, Files)
        assert set(folder2_store.keys()) == {"this.txt", "over.json"}
        assert folder2_store["this.txt"] == b"that"
        assert folder2_store["over.json"] == b"there"


def test_is_valid_key_sees_the_inner_key(tmpdir):
    """``Files`` is ``mk_relative_path_store(FileBytesPersister)``, so ``is_valid_key`` was
    reached through ``Store.__getattr__`` with the RELATIVE key while the leaf matched it
    against a pattern built from the absolute path -- returning False for keys that exist."""
    import os

    from dol import Files, TextFiles

    rootdir = str(tmpdir)
    os.makedirs(os.path.join(rootdir, "sub"), exist_ok=True)
    with open(os.path.join(rootdir, "a.txt"), "w") as fp:
        fp.write("x")
    with open(os.path.join(rootdir, "sub", "b.txt"), "w") as fp:
        fp.write("y")

    s = Files(rootdir)
    nested_key = os.path.join("sub", "b.txt")  # separator differs on Windows
    assert sorted(s) == ["a.txt", nested_key]
    # the invariant that was broken: every key the store yields is a valid key
    assert all(s.is_valid_key(k) for k in s)
    assert s.is_valid_key("a.txt")
    assert s.is_valid_key(nested_key)
    s.validate_key("a.txt")  # must not raise
    assert TextFiles(rootdir).is_valid_key("a.txt")
    assert s["a.txt"] == b"x"  # reads unaffected


# --- i2mint/dol#107: keys must not escape rootdir through '..' --------------------


def _sandbox(tmp_path):
    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    (root / "a" / "in.txt").write_bytes(b"inside")
    (tmp_path / "outside.txt").write_bytes(b"secret")
    return root


@pytest.mark.parametrize("key", ["../outside.txt", "a/../../outside.txt"])
def test_files_reject_keys_escaping_rootdir(tmp_path, key):
    from dol.filesys import Files, FilesReader, TextFiles

    root = _sandbox(tmp_path)
    for store_cls in (Files, FilesReader, TextFiles):
        s = store_cls(str(root))
        assert key not in s
        with pytest.raises(KeyError):
            s[key]
        assert s.get(key) is None
    s = Files(str(root))
    with pytest.raises(KeyError):
        s[key] = b"pwned"
    with pytest.raises(KeyError):
        del s[key]
    assert (tmp_path / "outside.txt").read_bytes() == b"secret"
    assert sorted(s) == [os.path.join("a", "in.txt")]


def test_files_dotdot_staying_inside_rootdir_still_works(tmp_path):
    from dol.filesys import Files

    root = _sandbox(tmp_path)
    s = Files(str(root))
    assert s["a/../a/in.txt"] == b"inside"
    s["a/../new.txt"] = b"new"
    assert (root / "new.txt").read_bytes() == b"new"
    del s["a/../new.txt"]
    assert not (root / "new.txt").exists()


def test_files_allow_escape_opt_in(tmp_path):
    from dol.filesys import Files

    root = _sandbox(tmp_path)
    s = Files(str(root), allow_escape=True)
    assert s["../outside.txt"] == b"secret"
    assert "allow_escape=True" in repr(s)
    assert "allow_escape=" not in repr(Files(str(root)))


def test_mk_dirs_if_missing_makes_no_dirs_outside_rootdir(tmp_path):
    from dol.filesys import Files, mk_dirs_if_missing

    root = _sandbox(tmp_path)
    s = mk_dirs_if_missing(Files(str(root)))
    with pytest.raises(KeyError):
        s["../escaped/dir/x.txt"] = b"pwned"
    assert not (tmp_path / "escaped").exists()
    s["made/here.txt"] = b"ok"  # valid keys still get their dirs made
    assert (root / "made" / "here.txt").read_bytes() == b"ok"


def test_dir_and_zip_readers_reject_keys_escaping_rootdir(tmp_path):
    from dol.filesys import DirReader, PickleStores
    from dol.zipfiledol import ZipFilesReader, ZipFilesReaderAndBytesWriter

    root = _sandbox(tmp_path)
    (tmp_path / "outdir").mkdir()
    d = DirReader(str(root))  # absolute keys
    with pytest.raises(KeyError):
        d[str(root) + "/../outdir"]
    d[str(root) + "/a"]  # inside the root: still fine
    p = PickleStores(str(root))  # relative keys
    with pytest.raises(KeyError):
        p["../outdir"]
    p["a"]
    if os.name == "nt":  # ZipFilesReader's default subpath regex is POSIX-only
        return
    with pytest.raises(KeyError):
        ZipFilesReader(str(root))[str(root) + "/../out.zip"]
    with pytest.raises(KeyError):
        ZipFilesReaderAndBytesWriter(str(root))[str(root) + "/../out.zip"] = b""
    assert not (tmp_path / "out.zip").exists()


def test_rootdir_spellings_still_list_their_keys(tmp_path, monkeypatch):
    from dol.filesys import Files

    root = _sandbox(tmp_path)
    key = os.path.join("a", "in.txt")
    if os.name != "nt":
        assert list(Files("/" + str(root))) == [key]  # '//...' rootdir
    monkeypatch.chdir(root)
    assert list(Files(os.path.join("a", ".."))) == [key]  # rootdir normalising to '.'


# --- sprint review (i2mint/dol#123): coverage of the #107 containment surface --------


def test_path_is_under_mixed_absolute_and_relative_is_false():
    from dol.filesys import path_is_under

    # commonpath raises ValueError on a mix of absolute and relative paths
    assert path_is_under(os.path.abspath("x"), "rel/root") is False
    assert path_is_under("rel/root/x", os.path.abspath("root")) is False


def test_zip_readers_read_and_write_keys_inside_rootdir(tmp_path):
    import zipfile
    from dol.zipfiledol import ZipFilesReader, ZipFilesReaderAndBytesWriter

    if os.name == "nt":  # ZipFilesReader's default subpath regex is POSIX-only
        return
    root = tmp_path / "root"
    root.mkdir()
    zip_path = str(root / "in.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("member.txt", b"hello")
    reader = ZipFilesReader(str(root))
    assert reader[zip_path]["member.txt"] == b"hello"
    with pytest.raises(KeyError):  # inside the root, but no such file
        reader[str(root / "missing.zip")]
    writer = ZipFilesReaderAndBytesWriter(str(root))
    copy_path = str(root / "copy.zip")
    writer[copy_path] = (root / "in.zip").read_bytes()
    assert writer[copy_path]["member.txt"] == b"hello"


def test_make_missing_dirs_mixin_reraises_out_of_root_keys(tmp_path):
    from dol.filesys import FileBytesPersister, MakeMissingDirsStoreMixin

    class S(MakeMissingDirsStoreMixin, FileBytesPersister):
        pass

    root = _sandbox(tmp_path)
    s = S(str(root))
    with pytest.raises(KeyError):
        s[str(root) + "/../escaped/x.txt"] = b"pwned"
    assert not (tmp_path / "escaped").exists()


# POSIX follows a symlink before applying the next '..'; Windows normalizes '..'
# lexically first, so there 'lnk/../x' IS 'root/x' and nothing escapes to refuse.
_DOTDOT_FOLLOWS_SYMLINKS = os.name != "nt"


def _symlink_sandbox(tmp_path):
    """``root/lnk`` -> ``outside/deep``; ``outside/secret.txt`` is outside the root."""
    root = _sandbox(tmp_path)
    deep = tmp_path / "outside" / "deep"
    deep.mkdir(parents=True)
    (tmp_path / "outside" / "secret.txt").write_bytes(b"secret")
    try:
        os.symlink(str(deep), str(root / "lnk"), target_is_directory=True)
    except (OSError, NotImplementedError):  # e.g. Windows without the privilege
        pytest.skip("cannot create symlinks here")
    return root


def test_files_symlink_then_dotdot_does_not_escape_rootdir(tmp_path):
    from dol.filesys import Files

    root = _symlink_sandbox(tmp_path)
    s = Files(str(root))
    escaping_key = os.path.join("lnk", "..", "secret.txt")
    assert escaping_key not in s
    assert s.get(escaping_key) is None
    pwn_key = os.path.join("lnk", "..", "pwned.txt")
    if _DOTDOT_FOLLOWS_SYMLINKS:
        with pytest.raises(KeyError):
            s[pwn_key] = b"x"
    else:
        s[pwn_key] = b"x"
        assert (root / "pwned.txt").read_bytes() == b"x"  # contained in the root
    assert not (tmp_path / "outside" / "pwned.txt").exists()


def test_symlinked_subdir_of_rootdir_stays_addressable(tmp_path):
    # the design #107 kept: a symlink inside the root is followed, and a '..' that
    # does not come after a symlink still resolves lexically
    from dol.filesys import Files

    root = _symlink_sandbox(tmp_path)
    s = Files(str(root))
    s[os.path.join("lnk", "f.txt")] = b"via link"
    assert (tmp_path / "outside" / "deep" / "f.txt").read_bytes() == b"via link"
    assert s[os.path.join("lnk", "f.txt")] == b"via link"
    assert s[os.path.join("a", "..", "lnk", "f.txt")] == b"via link"
    assert s[os.path.join("a", "..", "a", "in.txt")] == b"inside"


def test_symlink_then_dotdot_is_refused_by_every_containment_user(tmp_path):
    from dol.filesys import DirReader, Files, mk_dirs_if_missing

    root = _symlink_sandbox(tmp_path)
    if not _DOTDOT_FOLLOWS_SYMLINKS:
        pytest.skip("'..' never follows a symlink here: nothing escapes to refuse")
    (tmp_path / "outside" / "sib").mkdir()
    with pytest.raises(KeyError):
        DirReader(str(root))[os.path.join(str(root), "lnk", "..", "sib")]
    s = mk_dirs_if_missing(Files(str(root)))
    with pytest.raises(KeyError):
        s[os.path.join("lnk", "..", "made", "x.txt")] = b"pwned"
    assert not (tmp_path / "outside" / "made").exists()
    # allow_escape=True is the explicit opt-out
    assert Files(str(root), allow_escape=True)[
        os.path.join("lnk", "..", "secret.txt")
    ] == b"secret"


def test_rootdir_with_dotdot_after_a_symlink_lists_and_reads(tmp_path):
    # e.g. join(dirname(__file__), '..', 'data') in a symlink-installed package: the
    # rootdir's own '..' is the OS's business, only the key's part beyond it is checked
    from dol.filesys import Files

    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "data" / "sub").mkdir(parents=True)
    (src / "data" / "a.txt").write_bytes(b"A")
    site = tmp_path / "site"
    site.mkdir()
    try:
        os.symlink(str(src / "pkg"), str(site / "pkg"), target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")
    s = Files(os.path.join(str(site), "pkg", "..", "data"))
    if os.name != "nt":  # Windows resolves '..' lexically: site/data doesn't exist
        assert list(s) == ["a.txt"]
        assert s["a.txt"] == b"A"
        assert s[os.path.join("sub", "..", "a.txt")] == b"A"
