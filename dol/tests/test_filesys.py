"""Test filesys objects."""

from functools import partial
from pathlib import Path
from typing import MutableMapping
import pytest

from dol.tests.utils_for_tests import mk_test_store_from_keys, mk_tmp_local_store
from dol.filesys import mk_dirs_if_missing, TextFiles


# --------------------------------------------------------------------------------------
# Utils


def all_folder_paths_under_folder(rootpath: str, include_rootpath=False):
    """Return all folder paths under folderpath."""
    from pathlib import Path

    rootpath = Path(rootpath)
    folderpaths = (str(p) for p in rootpath.glob('**/') if p.is_dir())
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


def empty_directory(s, path_must_include=('test_mk_dirs_if_missing',)):
    if isinstance(path_must_include, str):
        path_must_include = (path_must_include,)

    if not all(substr in s for substr in path_must_include):
        raise ValueError(
            f"Path '{s}' does not include any of the substrings: {path_must_include}.\n"
            'This is a safeguard. For your safety, I will delete nothing!'
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


# --------------------------------------------------------------------------------------
# Tests


def test_mk_dirs_if_missing():
    s = mk_tmp_local_store('test_mk_dirs_if_missing', make_dirs_if_missing=False)
    empty_directory(s.rootdir, path_must_include='test_mk_dirs_if_missing')
    with pytest.raises(KeyError):
        s['this/path/does/not/exist'] = 'hello'
    ss = mk_dirs_if_missing(s)
    ss['this/path/does/not/exist'] = 'hello'  # this should work now
    assert ss['this/path/does/not/exist'] == 'hello'

    # # It works on classes too:
    # TextFilesWithAutoMkdir = mk_tmp_local_store(TextFiles)
    # sss = TextFilesWithAutoMkdir(s.rootdir)
    # assert sss["another/path/that/does/not/exist"] == "hello"
