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


def empty_directory(
    s: MutableMapping, delete_folders: bool = True, delete_rootpath: bool = False
):
    for k in s:
        del s[k]
    import time

    time.sleep(0.4)
    if delete_folders:
        rootdir = getattr(  # I hate this as much as you do, but s.rootdir didn't work!!
            s, 'rootdir', None
        ) or getattr(s, '_prefix', None)
        delete_all_folders_under_folder(rootdir, include_rootpath=delete_rootpath)


# --------------------------------------------------------------------------------------
# Tests


def test_mk_dirs_if_missing():
    s = mk_tmp_local_store('test_mk_dirs_if_missing', make_dirs_if_missing=False)
    empty_directory(s)
    with pytest.raises(KeyError):
        s['this/path/does/not/exist'] = 'hello'
    ss = mk_dirs_if_missing(s)
    ss['this/path/does/not/exist'] = 'hello'  # this should work now
    assert ss['this/path/does/not/exist'] == 'hello'

    # # It works on classes too:
    # TextFilesWithAutoMkdir = mk_tmp_local_store(TextFiles)
    # sss = TextFilesWithAutoMkdir(s.rootdir)
    # assert sss["another/path/that/does/not/exist"] == "hello"
