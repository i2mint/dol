"""Test filesys objects."""

import os
from functools import partial
import tempfile
from pathlib import Path
from typing import Mapping
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
        temp_path = os.path.join(temp_dir, "foo/bar")

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
    assert foo_path.read_text() == '{"bar": 1}', "Should be json encoded"

    ss = Jsons(rootdir)
    assert "foo" not in ss, "foo should be filtered out because no .json extension"
    ss["apple"] = {"crumble": True}
    assert "apple" in ss
    assert "apple" in set(ss)  # which is different than 'apple' in ss
    assert ss["apple"] == {"crumble": True}
    apple_path = Path(os.path.join(rootdir, "apple.json"))
    assert apple_path.is_file(), "Should have created a file (with .json extension)"
    assert apple_path.read_text() == '{"crumble": true}', "Should be json encoded"


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
        assert set(folder1_store.keys()) == {"day.doc", "subfolder/apple.p"}
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
