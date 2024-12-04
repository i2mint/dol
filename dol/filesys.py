"""File system access"""

import os
from os import stat as os_stat
from functools import wraps, partial
from typing import Union, Callable, Iterable, Optional

from dol.base import Collection, KvReader, KvPersister
from dol.trans import wrap_kvs, store_decorator, filt_iter
from dol.naming import mk_pattern_from_template_and_format_dict
from dol.paths import mk_relative_path_store

file_sep = os.path.sep
inf = float("infinity")


def ensure_slash_suffix(path: str):
    r"""Add a file separation (/ or \) at the end of path str, if not already present."""
    if not path.endswith(file_sep):
        return path + file_sep
    else:
        return path


def paths_in_dir(rootdir, include_hidden=False):
    for name in os.listdir(rootdir):
        if include_hidden or not name.startswith(
            "."
        ):  # TODO: is dot a platform independent marker for hidden file?
            filepath = os.path.join(rootdir, name)
            if os.path.isdir(filepath):
                yield ensure_slash_suffix(filepath)
            else:
                yield filepath


def iter_filepaths_in_folder_recursively(
    root_folder, max_levels=None, _current_level=0, include_hidden=False
):
    """Recursively generates filepaths of folder (and subfolders, etc.) up to a given level"""
    if max_levels is None:
        max_levels = inf
    for full_path in paths_in_dir(root_folder, include_hidden):
        if os.path.isdir(full_path):
            if _current_level < max_levels:
                for entry in iter_filepaths_in_folder_recursively(
                    full_path, max_levels, _current_level + 1, include_hidden
                ):
                    yield entry
        else:
            if os.path.isfile(full_path):
                yield full_path


def iter_dirpaths_in_folder_recursively(
    root_folder, max_levels=None, _current_level=0, include_hidden=False
):
    """Recursively generates dirpaths of folder (and subfolders, etc.) up to a given level"""
    if max_levels is None:
        max_levels = inf
    for full_path in paths_in_dir(root_folder, include_hidden):
        if os.path.isdir(full_path):
            yield full_path
            if _current_level < max_levels:
                for entry in iter_dirpaths_in_folder_recursively(
                    full_path, max_levels, _current_level + 1, include_hidden
                ):
                    yield entry


def create_directories(dirpath, max_dirs_to_make: Optional[int] = None):
    """
    Create directories up to a specified limit.

    Parameters:
    dirpath (str): The directory path to create.
    max_dirs_to_make (int, optional): The maximum number of directories to create. If None, there's no limit.

    Returns:
    bool: True if the directory was created successfully, False otherwise.

    Raises:
    ValueError: If max_dirs_to_make is negative.

    Examples:
    >>> import tempfile, shutil
    >>> temp_dir = tempfile.mkdtemp()
    >>> target_dir = os.path.join(temp_dir, 'a', 'b', 'c')
    >>> create_directories(target_dir, max_dirs_to_make=2)
    False
    >>> create_directories(target_dir, max_dirs_to_make=3)
    True
    >>> os.path.isdir(target_dir)
    True
    >>> shutil.rmtree(temp_dir)  # Cleanup

    >>> temp_dir = tempfile.mkdtemp()
    >>> target_dir = os.path.join(temp_dir, 'a', 'b', 'c', 'd')
    >>> create_directories(target_dir)
    True
    >>> os.path.isdir(target_dir)
    True
    >>> shutil.rmtree(temp_dir)  # Cleanup
    """
    if max_dirs_to_make is not None and max_dirs_to_make < 0:
        raise ValueError("max_dirs_to_make must be non-negative or None")

    if os.path.exists(dirpath):
        return True

    if max_dirs_to_make is None:
        os.makedirs(dirpath, exist_ok=True)
        return True

    # Calculate the number of directories to create
    dirs_to_make = []
    current_path = dirpath

    while not os.path.exists(current_path):
        dirs_to_make.append(current_path)
        current_path, _ = os.path.split(current_path)

    if len(dirs_to_make) > max_dirs_to_make:
        return False

    # Create directories from the top level down
    for dir_to_make in reversed(dirs_to_make):
        os.mkdir(dir_to_make)

    return True


def process_path(
    *path: Iterable[str],
    ensure_dir_exists: Union[int, bool] = False,
    assert_exists: bool = False,
    ensure_endswith_slash: bool = False,
    ensure_does_not_end_with_slash: bool = False,
    expanduser: bool = True,
    expandvars: bool = True,
    abspath: bool = True,
    rootdir: str = "",
) -> str:
    """
    Process a path string, ensuring it exists, and optionally expanding user.

    Args:
        path (Iterable[str]): The path to process. Can be multiple components of a path.
        ensure_dir_exists (bool): Whether to ensure the path exists.
        assert_exists (bool): Whether to assert that the path exists.
        ensure_endswith_slash (bool): Whether to ensure the path ends with a slash.
        ensure_does_not_end_with_slash (bool): Whether to ensure the path does not end with a slash.
        expanduser (bool): Whether to expand the user in the path.
        expandvars (bool): Whether to expand environment variables in the path.
        abspath (bool): Whether to convert the path to an absolute path.
        rootdir (str): The root directory to prepend to the path.

    Returns:
        str: The processed path.

    >>> process_path('a', 'b', 'c')  # doctest: +ELLIPSIS
    '...a/b/c'
    >>> from functools import partial
    >>> process_path('a', 'b', 'c', rootdir='/root/dir/', ensure_endswith_slash=True)
    '/root/dir/a/b/c/'

    """
    path = os.path.join(*path)
    if ensure_endswith_slash and ensure_does_not_end_with_slash:
        raise ValueError(
            "Cannot ensure both ends with slash and does not end with slash."
        )
    if rootdir:
        path = os.path.join(rootdir, path)
    if expanduser:
        path = os.path.expanduser(path)
    if expandvars:
        path = os.path.expandvars(path)
    if abspath:
        path = os.path.abspath(path)
    if ensure_endswith_slash:
        if not path.endswith("/"):
            path = path + "/"
    if ensure_does_not_end_with_slash:
        if path.endswith("/"):
            path = path[:-1]
    if ensure_dir_exists:
        if ensure_dir_exists is True:
            ensure_dir_exists = None  # max_dirs_to_make
        create_directories(path, max_dirs_to_make=ensure_dir_exists)
    if assert_exists:
        assert os.path.exists(path), f"Path does not exist: {path}"
    return path


def ensure_dir(
    dirpath,
    *,
    max_dirs_to_make: Optional[int] = None,
    verbose: Union[bool, str, Callable] = False,
):
    """Ensure that a directory exists, creating it if necessary.

    :param dirpath: path to the directory to create
    :param max_dirs_to_make: the maximum number of directories to create.
        If None, there's no limit.
    :param verbose: controls verbosity (the noise ensure_dir makes if it make folder)
    :return: the path to the directory

    When the path does not exist, if ``verbose`` is:

    - a ``bool``' a standard message will be printed

    - a ``callable``; will be called on dirpath before directory is created -- you
    can use this to ask the user for confirmation for example

    - a ''string``; this string will be printed


    Usage note: If you want to string or the (argument-less) callable to be dependent
    on ``dirpath``, you need make them so when calling ensure_dir.

    """
    if not os.path.exists(dirpath):
        if verbose:
            if isinstance(verbose, bool):
                print(f"Making the directory: {dirpath}")
            elif isinstance(verbose, Callable):
                callaback = verbose
                callaback(dirpath)
            else:
                string_to_print = verbose
                print(string_to_print)
        create_directories(dirpath, max_dirs_to_make=max_dirs_to_make)
    return dirpath


def temp_dir(dirname="", make_it_if_necessary=True, verbose=False):
    from tempfile import gettempdir

    tmpdir = os.path.join(gettempdir(), dirname)
    if make_it_if_necessary:
        ensure_dir(tmpdir, verbose=verbose)
    return tmpdir


mk_tmp_dol_dir = temp_dir  # for backward compatibility


def mk_absolute_path(path_format):
    if path_format.startswith("~"):
        path_format = os.path.expanduser(path_format)
    elif path_format.startswith("."):
        path_format = os.path.abspath(path_format)
    return path_format


# TODO: subpath: Need to be able to allow named and unnamed file format markers (i.e {} and {named})

_dflt_not_valid_error_msg = (
    "Key not valid (usually because does not exist or access not permitted): {}"
)
_dflt_not_found_error_msg = "Key not found: {}"


class KeyValidationError(KeyError):
    pass


# TODO: The validate and try/except is a frequent pattern. Make it a decorator.
def validate_key_and_raise_key_error_on_exception(func):
    @wraps(func)
    def wrapped_method(self, k, *args, **kwargs):
        self.validate_key(k)
        try:
            return func(self, k, *args, **kwargs)
        except Exception as e:
            raise KeyError(e)

    return wrapped_method


def resolve_path(path, assert_existence: bool = False):
    """Resolve a path to a full, real, (file or folder) path (opt assert existence).
    That is, resolve situations where ~ and . prefix the paths.
    """
    if path.startswith("."):
        path = os.path.abspath(path)
    elif path.startswith("~"):
        path = os.path.expanduser(path)
    if assert_existence:
        assert os.path.exists(path), f"This path (file or folder) wasn't found: {path}"
    return path


def resolve_dir(
    dirpath: str, assert_existence: bool = False, ensure_existence: bool = False
):
    """Resolve a path to a full, real, path to a directory"""
    dirpath = resolve_path(dirpath)
    if ensure_existence and not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    if assert_existence:
        assert os.path.isdir(dirpath), f"This directory wasn't found: {dirpath}"
    return dirpath


def _for_repr(obj, quote="'"):
    """
    >>> _for_repr('a string')
    "'a string'"
    >>> _for_repr(10)
    10
    >>> _for_repr(None)
    'None'
    """
    if isinstance(obj, str):
        obj = f"{quote}{obj}{quote}"
    elif obj is None:
        obj = "None"
    return obj


class FileSysCollection(Collection):
    # rootdir = None  # mentioning here so that the attribute is seen as an attribute before instantiation.

    def __init__(
        self,
        rootdir,
        subpath="",
        pattern_for_field=None,
        max_levels=None,
        *,
        include_hidden=False,
        assert_rootdir_existence=False,
    ):
        self._init_kwargs = {k: v for k, v in locals().items() if k != "self"}
        rootdir = resolve_dir(rootdir, assert_existence=assert_rootdir_existence)
        if max_levels is None:
            max_levels = inf
        subpath_implied_min_levels = len(subpath.split(os.path.sep)) - 1
        assert (
            max_levels >= subpath_implied_min_levels
        ), f"max_levels is {max_levels}, but subpath {subpath} would imply at least {subpath_implied_min_levels}"
        pattern_for_field = pattern_for_field or {}
        self.rootdir = ensure_slash_suffix(rootdir)
        self.subpath = subpath
        self._key_pattern = mk_pattern_from_template_and_format_dict(
            os.path.join(rootdir, subpath), pattern_for_field
        )
        self._max_levels = max_levels
        self.include_hidden = include_hidden

    def is_valid_key(self, k):
        return bool(self._key_pattern.match(k))

    def validate_key(
        self,
        k,
        err_msg_format=_dflt_not_valid_error_msg,
        err_type=KeyValidationError,
    ):
        if not self.is_valid_key(k):
            raise err_type(err_msg_format.format(k))

    def __repr__(self):
        input_str = ", ".join(
            f"{k}={_for_repr(v)}" for k, v in self._init_kwargs.items()
        )
        return f"{type(self).__name__}({input_str})"

    def with_relative_paths(self):
        """Return a copy of self with relative paths"""
        return with_relative_paths(self)


class DirCollection(FileSysCollection):
    def __iter__(self):
        yield from filter(
            self.is_valid_key,
            iter_dirpaths_in_folder_recursively(
                self.rootdir,
                max_levels=self._max_levels,
                include_hidden=self.include_hidden,
            ),
        )

    def __contains__(self, k):
        return self.is_valid_key(k) and os.path.isdir(k)


class FileCollection(FileSysCollection):
    def __iter__(self):
        """
        Iterator of valid filepaths.

        >>> import os
        >>> filepath = __file__  # path to this module
        >>> dirpath = os.path.dirname(__file__)  # path of the directory where I (the module file) am
        >>> s = FileCollection(dirpath, max_levels=0)
        >>>
        >>> files_in_this_dir = list(s)
        >>> filepath in files_in_this_dir
        True
        """
        yield from filter(
            self.is_valid_key,
            iter_filepaths_in_folder_recursively(
                self.rootdir,
                max_levels=self._max_levels,
                include_hidden=self.include_hidden,
            ),
        )

    def __contains__(self, k):
        """
        Checks if k is valid and contained in the store

        >>> import os
        >>> filepath = __file__  # path to this module
        >>> dirpath = os.path.dirname(__file__)  # path of the directory where I (the module file) am
        >>> s = FileCollection(dirpath, max_levels=0)
        >>>
        >>> filepath in s
        True
        >>> '_this_filepath_will_never_be_valid_' in s
        False
        """
        return self.is_valid_key(k) and os.path.isfile(k)


class FileInfoReader(FileCollection, KvReader):
    def __getitem__(self, k):
        self.validate_key(k)
        return os_stat(k)


class FileBytesReader(FileCollection, KvReader):
    _read_open_kwargs = dict(
        mode="rb",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    )

    @validate_key_and_raise_key_error_on_exception
    def __getitem__(self, k):
        '''
        Gets the bytes contents of the file k.

        >>> import os
        >>> filepath = __file__
        >>> dirpath = os.path.dirname(__file__)  # path of the directory where I (the module file) am
        >>> s = FileBytesReader(dirpath, max_levels=0)
        >>>
        >>> ####### Get the first 9 characters (as bytes) of this module #####################
        >>> s[filepath][:9]
        b'"""File s'
        >>>
        >>> ####### Test key validation #####################
        >>> # this key is not valid since not under the dirpath folder, so should give an exception
        >>> # Skipped because filesys.KeyValidationError vs dol.filesys.KeyValidationError on different systems
        >>> s['not_a_valid_key']  # doctest: +SKIP
        Traceback (most recent call last):
            ...
        filesys.KeyValidationError: 'Key not valid (usually because does not exist or access not permitted): not_a_valid_key'
        >>>
        >>> ####### Test further exceptions (that should be wrapped in KeyError) #####################
        >>> # this key is valid, since under dirpath, but the file itself doesn't exist (hopefully for this test)
        >>> non_existing_file = os.path.join(dirpath, 'non_existing_file')
        >>> try:
        ...     s[non_existing_file]
        ... except KeyError:
        ...     print("KeyError (not FileNotFoundError) was raised.")
        KeyError (not FileNotFoundError) was raised.
        '''
        with open(k, **self._read_open_kwargs) as fp:
            return fp.read()


class LocalFileDeleteMixin:
    @validate_key_and_raise_key_error_on_exception
    def __delitem__(self, k):
        os.remove(k)


class FileBytesPersister(FileBytesReader, KvPersister):
    _write_open_kwargs = dict(
        mode="wb",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    )
    # _make_dirs_if_missing = False

    @validate_key_and_raise_key_error_on_exception
    def __setitem__(self, k, v):
        # TODO: Make this work with validate_key_and_raise_key_error_on_exception
        # if self._make_dirs_if_missing:
        #     dirname = os.path.dirname(k)
        #     os.makedirs(dirname, exist_ok=True)
        with open(k, **self._write_open_kwargs) as fp:
            return fp.write(v)

    @validate_key_and_raise_key_error_on_exception
    def __delitem__(self, k):
        os.remove(k)


# ---------------------------------------------------------------------------------------
# TODO: Once test coverage sufficient, apply this pattern to all other convenience stores

with_relative_paths = partial(mk_relative_path_store, prefix_attr="rootdir")


@with_relative_paths
class FilesReader(FileBytesReader):
    """FileBytesReader with relative paths"""


@with_relative_paths
class Files(FileBytesPersister):
    """FileBytesPersister with relative paths"""


RelPathFileBytesReader = FilesReader
RelPathFileBytesPersister = Files  # back-compatibility alias

# ---------------------------------------------------------------------------------------


class FileStringReader(FileBytesReader):
    _read_open_kwargs = dict(FileBytesReader._read_open_kwargs, mode="rt")


class FileStringPersister(FileBytesPersister):
    _read_open_kwargs = dict(FileBytesReader._read_open_kwargs, mode="rt")
    _write_open_kwargs = dict(FileBytesPersister._write_open_kwargs, mode="wt")


@with_relative_paths(prefix_attr="rootdir")
class TextFilesReader(FileStringReader):
    """FileStringReader with relative paths"""


@with_relative_paths(prefix_attr="rootdir")
class TextFiles(FileStringPersister):
    """FileStringPersister with relative paths"""


RelPathFileStringReader = TextFilesReader
RelPathFileStringPersister = TextFiles


# ------------------------------------ misc --------------------------------------------
import pickle
import json

# TODO: Want to replace with use of ValueCodecs but need to resolve circular imports
pickle_bytes_wrap = wrap_kvs(value_decoder=pickle.loads, value_encoder=pickle.dumps)
json_bytes_wrap = wrap_kvs(value_decoder=json.loads, value_encoder=json.dumps)


# And two factories to make the above more configurable:
def mk_pickle_bytes_wrap(
    *, loads_kwargs: Optional[dict] = None, dumps_kwargs: Optional[dict] = None
) -> Callable:
    """"""
    return wrap_kvs(
        value_decoder=partial(pickle.loads, **(loads_kwargs or {})),
        value_encoder=partial(pickle.dumps, **(dumps_kwargs or {})),
    )


def mk_json_bytes_wrap(
    *, loads_kwargs: Optional[dict] = None, dumps_kwargs: Optional[dict] = None
) -> Callable:
    return wrap_kvs(
        value_decoder=partial(json.loads, **(loads_kwargs or {})),
        value_encoder=partial(json.dumps, **(dumps_kwargs or {})),
    )


@pickle_bytes_wrap
class PickleFiles(Files):
    """A store of pickles"""


PickleStore = PickleFiles  # back-compatibility alias


@json_bytes_wrap
class JsonFiles(TextFiles):
    """A store of json files"""


from dol.trans import affix_key_codec


@affix_key_codec(suffix=".json")
@filt_iter.suffixes(".json")
class Jsons(JsonFiles):
    """Like JsonFiles, but with added .json extension handling
    Namely: filtering for `.json` extensions but not showing the extension in keys"""


# @wrap_kvs(key_of_id=lambda x: x[:-1], id_of_key=lambda x: x + path_sep)
@mk_relative_path_store(prefix_attr="rootdir")
class PickleStores(DirCollection):
    def __getitem__(self, k):
        return PickleFiles(k)

    def __repr__(self):
        return f"{type(self).__name__}('{self.rootdir}', ...)"


class DirReader(DirCollection, KvReader):
    def __getitem__(self, k):
        return DirReader(k)


def mk_dirs_if_missing_preset(
    self, k, v, *, max_dirs_to_make: Optional[int] = None, verbose=False
):
    # TODO: I'm not thrilled in the way I'm doing this; find alternatives
    try:
        super(type(self), self).__setitem__(k, v)
    except Exception:  # general on purpose...
        # TODO: ... But perhaps a more precise (but sufficient) exception list better?
        from dol.dig import inner_most_key

        # get the inner most key, which should be a full path
        _id = inner_most_key(self, k)
        # get the full path of directory needed for this file
        dirname = os.path.dirname(_id)
        # make all the directories needed
        ensure_dir(dirname, max_dirs_to_make=max_dirs_to_make, verbose=verbose)
        # os.makedirs(dirname, exist_ok=True)  # TODO: ensure_dir does this already, no?
        # try writing again
        super(type(self), self).__setitem__(k, v)
        # TODO: Undesirable here: If the setitem still fails, we created dirs
        #  already, for nothing, and are not cleaning up (if clean up need to make
        #  sure to not delete dirs that already existed!)
    finally:
        return v


# TODO: Add more control over mk dir condition (e.g. number of levels, or any key cond)
#   Also, add a verbose option to print the dirs that are being made
#   (see dol.filesys.ensure_dir)
@store_decorator
def mk_dirs_if_missing(
    store_cls=None,
    *,
    max_dirs_to_make: Optional[int] = None,
    verbose: Union[bool, str, Callable] = False,
    key_condition=None,  # TODO: not used! Should use! Add to ensure_dir
):
    """Store decorator that will make the store create directories on write as
    needed.

    Note that it'll only effect paths relative to the rootdir, which needs to be
    ensured to exist separatedly.
    """
    _mk_dirs_if_missing_preset = partial(
        mk_dirs_if_missing_preset, max_dirs_to_make=max_dirs_to_make, verbose=verbose
    )
    return wrap_kvs(store_cls, preset=_mk_dirs_if_missing_preset)


# DEPRECATED!!
# This one really smells.
class MakeMissingDirsStoreMixin:
    """Will make a local file store automatically create the directories needed to create a file.
    Should be placed before the concrete perisister in the mro but in such a manner so that it receives full paths.
    """

    _verbose: Union[bool, str, Callable] = False  # eek! Can't set in init.

    def __setitem__(self, k, v):
        print(
            f"Deprecating message: Consider using the mk_dirs_if_missing decorator instead."
        )
        # TODO: I'm not thrilled in the way I'm doing this; find alternatives
        try:
            super().__setitem__(k, v)
        except Exception:  # general on purpose...
            # TODO: ... But perhaps a more precise (but sufficient) exception list better?
            from dol.dig import inner_most_key

            # get the inner most key, which should be a full path
            _id = inner_most_key(self, k)
            # get the full path of directory needed for this file
            dirname = os.path.dirname(_id)
            # make all the directories needed
            ensure_dir(dirname, self._verbose)
            os.makedirs(dirname, exist_ok=True)
            # try writing again
            super().__setitem__(k, v)
            # TODO: Undesirable here: If the setitem still fails, we created dirs
            #  already, for nothing, and are not cleaning up (if clean up need to make
            #  sure to not delete dirs that already existed!)


# -------------------------------------------------------------------------------------

from dol.kv_codecs import KeyCodecs


def subfolder_stores(
    root_folder,
    *,
    max_levels: Optional[int] = None,
    include_hidden: bool = False,
    relative_paths: bool = True,
    slash_suffix: bool = False,
    folder_to_store=Files,
):
    """
    Create a store of subfolders of a given folder, where the keys are the subfolder
    paths (by default, relative and slash-less) and the values are stores of these
    subfolders.

    By default, all subfolders will be taken, recursively, but this can be controlled by
    the `max_levels` parameter.
    """
    root_folder = ensure_slash_suffix(root_folder)
    wrap = KeyCodecs.affixed(
        prefix=root_folder if relative_paths else "",
        suffix="/" if not slash_suffix else "",
    )
    folders = iter_dirpaths_in_folder_recursively(
        root_folder, max_levels=max_levels, include_hidden=include_hidden
    )
    return wrap({path: folder_to_store(path) for path in folders})
