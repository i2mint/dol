"""Test filesys objects."""

import os
from functools import partial
import tempfile
from pathlib import Path
from typing import MutableMapping
import pytest

from dol.tests.utils_for_tests import mk_test_store_from_keys, mk_tmp_local_store
from dol.filesys import mk_dirs_if_missing, TextFiles, process_path


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


def test_process_path():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, 'foo/bar')

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

    t = mk_tmp_local_store('test_mk_dirs_if_missing', make_dirs_if_missing=False)
    empty_directory(t.rootdir, path_must_include='test_mk_dirs_if_missing')
    rootdir = t.rootdir

    s = JsonFiles(rootdir)
    s['foo'] = {'bar': 1}
    assert s['foo'] == {'bar': 1}
    foo_path = Path(os.path.join(rootdir, 'foo'))
    assert foo_path.is_file(), 'Should have created a file'
    assert foo_path.read_text() == '{"bar": 1}', 'Should be json encoded'

    ss = Jsons(rootdir)
    assert 'foo' not in ss, 'foo should be filtered out because no .json extension'
    ss['apple'] = {'crumble': True}
    assert 'apple' in ss
    assert 'apple' in set(ss)  # which is different than 'apple' in ss
    assert ss['apple'] == {'crumble': True}
    apple_path = Path(os.path.join(rootdir, 'apple.json'))
    assert apple_path.is_file(), 'Should have created a file (with .json extension)'
    assert apple_path.read_text() == '{"crumble": true}', 'Should be json encoded'


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
