# dol.filesys

File system access: dict-like stores over folders and files.

`Files` gives a folder a `MutableMapping` interface: keys are paths relative to the
root folder, values are the files’ bytes. `TextFiles`, `JsonFiles` and `PickleFiles`
add the corresponding value codecs. Writing under a sub-folder that does not exist raises
`KeyError`; wrap the store with `mk_dirs_if_missing` to create folders on write.

Main entry points:

- `Files`: bytes of the files under a root folder
- `TextFiles`: same, with text values
- `JsonFiles`: same, with JSON-decoded values
- `PickleFiles`: same, with pickled values
- `mk_dirs_if_missing`: make a file store create missing directories on write
  ```pycon
  >>> import tempfile
  >>> s = Files(tempfile.mkdtemp())
  >>> s['hello.txt'] = b'world'
  >>> s['hello.txt']
  b'world'
  >>> list(s)
  ['hello.txt']
  ```

### Functions

| [`create_directories`](#dol.filesys.create_directories)(dirpath[, max_dirs_to_make])     | Create directories up to a specified limit.                                                                                                                                 |
|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`ensure_dir`](#dol.filesys.ensure_dir)(dirpath, \*[, max_dirs_to_make, ...])    | Ensure that a directory exists, creating it if necessary.                                                                                                                   |
| [`ensure_slash_suffix`](#dol.filesys.ensure_slash_suffix)(path)                           | Add a file separation (/ or ) at the end of path str, if not already present.                                                                                               |
| [`iter_dirpaths_in_folder_recursively`](#dol.filesys.iter_dirpaths_in_folder_recursively)(root_folder)    | Recursively generates dirpaths of folder (and subfolders, etc.) up to a given level                                                                                         |
| [`iter_filepaths_in_folder_recursively`](#dol.filesys.iter_filepaths_in_folder_recursively)(root_folder)   | Recursively generates filepaths of folder (and subfolders, etc.) up to a given level                                                                                        |
| [`mk_absolute_path`](#dol.filesys.mk_absolute_path)(path_format)                       | Expand a leading `~`, or make a leading `.` path absolute; other paths are returned as is.                                                                                  |
| [`mk_dirs_if_missing`](#dol.filesys.mk_dirs_if_missing)([store_cls, ...])                | Store decorator that will make the store create directories on write as needed.                                                                                             |
| [`mk_dirs_if_missing_preset`](#dol.filesys.mk_dirs_if_missing_preset)(self, k, v, \*[, ...])    | Preset function that will make the store create directories on write as needed.                                                                                             |
| [`mk_json_bytes_wrap`](#dol.filesys.mk_json_bytes_wrap)(\*[, loads_kwargs, ...])         | Make a `wrap_kvs` value-codec wrapper for JSON, with kwargs for `json.loads`/`json.dumps`.                                                                                  |
| [`mk_pickle_bytes_wrap`](#dol.filesys.mk_pickle_bytes_wrap)(\*[, loads_kwargs, ...])       | Make a `wrap_kvs` value-codec wrapper for pickle, with kwargs for `pickle.loads`/`pickle.dumps`.                                                                            |
| [`mk_tmp_dol_dir`](#dol.filesys.mk_tmp_dol_dir)([dirname, ...])                      | Create and return a path to a temporary directory that's guaranteed to be accessible to the user.                                                                           |
| [`paths_in_dir`](#dol.filesys.paths_in_dir)(rootdir[, include_hidden])             | Yield the paths of the entries of `rootdir` (directories with a trailing separator), skipping hidden ones unless `include_hidden`.                                          |
| [`process_path`](#dol.filesys.process_path)(\*path[, ensure_dir_exists, ...])      | Process a path string, ensuring it exists, and optionally expanding user.                                                                                                   |
| [`resolve_dir`](#dol.filesys.resolve_dir)(dirpath[, assert_existence, ...])       | Resolve a path to a full, real, path to a directory                                                                                                                         |
| [`resolve_path`](#dol.filesys.resolve_path)(path[, assert_existence])              | Resolve a path to a full, real, (file or folder) path (opt assert existence).                                                                                               |
| [`subfolder_stores`](#dol.filesys.subfolder_stores)(root_folder, \*[, ...])            | Create a store of subfolders of a given folder, where the keys are the subfolder paths (by default, relative and slash-less) and the values are stores of these subfolders. |
| [`temp_dir`](#dol.filesys.temp_dir)([dirname, make_it_if_necessary, ...])      | Create and return a path to a temporary directory that's guaranteed to be accessible to the user.                                                                           |
| [`validate_key_and_raise_key_error_on_exception`](#dol.filesys.validate_key_and_raise_key_error_on_exception)(func) | Method decorator: validate the key first, and re-raise any exception of the method as a `KeyError`.                                                                         |

### Classes

| [`DirCollection`](#dol.filesys.DirCollection)(rootdir[, subpath, ...])     | Collection of the directory paths under `rootdir`.                                                                                     |
|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| [`DirReader`](#dol.filesys.DirReader)(rootdir[, subpath, ...])         | Reader mapping each sub-directory of `rootdir` to a `DirReader` of it.                                                                 |
| [`FileBytesPersister`](#dol.filesys.FileBytesPersister)(\*args[, delete_func])  | File persistence with configurable deletion.                                                                                           |
| [`FileBytesReader`](#dol.filesys.FileBytesReader)(rootdir[, subpath, ...])   | Reader mapping file paths under `rootdir` to the files' bytes.                                                                         |
| [`FileCollection`](#dol.filesys.FileCollection)(rootdir[, subpath, ...])    | Collection of the file paths under `rootdir`.                                                                                          |
| [`FileInfoReader`](#dol.filesys.FileInfoReader)(rootdir[, subpath, ...])    | Reader mapping file paths to their `os.stat` result.                                                                                   |
| [`FileStringPersister`](#dol.filesys.FileStringPersister)(\*args[, delete_func]) | Persister mapping file paths to the files' text (files opened in text mode).                                                           |
| [`FileStringReader`](#dol.filesys.FileStringReader)(rootdir[, subpath, ...])  | Reader mapping file paths to the files' text (files opened in text mode).                                                              |
| [`FileSysCollection`](#dol.filesys.FileSysCollection)(rootdir[, subpath, ...]) | Base collection of file-system paths under `rootdir`, optionally restricted by `subpath`, `max_levels` and hidden-file inclusion.      |
| [`Files`](#dol.filesys.Files)(\*args[, delete_func])               | FileBytesPersister with relative paths                                                                                                 |
| [`FilesReader`](#dol.filesys.FilesReader)(rootdir[, subpath, ...])       | FileBytesReader with relative paths                                                                                                    |
| [`JsonFiles`](#dol.filesys.JsonFiles)(\*args[, delete_func])           | A store of json files                                                                                                                  |
| [`Jsons`](#dol.filesys.Jsons)(\*args[, delete_func])               | Like JsonFiles, but with added .json extension handling Namely: filtering for `.json` extensions but not showing the extension in keys |
| [`LocalFileDeleteMixin`](#dol.filesys.LocalFileDeleteMixin)()                     | Mixin providing configurable file deletion.                                                                                            |
| [`MakeMissingDirsStoreMixin`](#dol.filesys.MakeMissingDirsStoreMixin)()                | Will make a local file store automatically create the directories needed to create a file.                                             |
| [`PickleFiles`](#dol.filesys.PickleFiles)(\*args[, delete_func])         | A store of pickles                                                                                                                     |
| [`PickleStore`](#dol.filesys.PickleStore)                                |                                                                                                                                        |
| [`PickleStores`](#dol.filesys.PickleStores)(rootdir[, subpath, ...])      | Reader mapping each sub-directory of `rootdir` to a `PickleFiles` store of it.                                                         |
| [`RelPathFileBytesPersister`](#dol.filesys.RelPathFileBytesPersister)                  |                                                                                                                                        |
| [`RelPathFileBytesReader`](#dol.filesys.RelPathFileBytesReader)                     |                                                                                                                                        |
| [`RelPathFileStringPersister`](#dol.filesys.RelPathFileStringPersister)                 |                                                                                                                                        |
| [`RelPathFileStringReader`](#dol.filesys.RelPathFileStringReader)                    |                                                                                                                                        |
| [`ReprMixin`](#dol.filesys.ReprMixin)()                                | A `__repr__` showing the `_init_kwargs` the instance was created with.                                                                 |
| [`TextFiles`](#dol.filesys.TextFiles)(\*args[, delete_func])           | FileStringPersister with relative paths                                                                                                |
| [`TextFilesReader`](#dol.filesys.TextFilesReader)(rootdir[, subpath, ...])   | FileStringReader with relative paths                                                                                                   |

### Exceptions

| [`KeyValidationError`](#dol.filesys.KeyValidationError)   | A `KeyError` for keys that fail a file-system store's validation.   |
|-----------------------------------------------------------------------|---------------------------------------------------------------------|

### *class* dol.filesys.DirCollection(rootdir, subpath='', pattern_for_field=None, max_levels=None, , include_hidden=False, assert_rootdir_existence=False)

Bases: [`FileSysCollection`](#dol.filesys.FileSysCollection)

Collection of the directory paths under `rootdir`.

### *class* dol.filesys.DirReader(rootdir, subpath='', pattern_for_field=None, max_levels=None, , include_hidden=False, assert_rootdir_existence=False)

Bases: [`DirCollection`](#dol.filesys.DirCollection), [`KvReader`](dol.base.html.md#dol.base.KvReader)

Reader mapping each sub-directory of `rootdir` to a `DirReader` of it.

### *class* dol.filesys.FileBytesPersister(\*args, delete_func=None, \*\*kwargs)

Bases: [`LocalFileDeleteMixin`](#dol.filesys.LocalFileDeleteMixin), [`FileBytesReader`](#dol.filesys.FileBytesReader), [`KvPersister`](dol.base.html.md#dol.base.KvPersister)

File persistence with configurable deletion.

Supports custom deletion functions via delete_func parameter in \_\_init_\_.

By default, tries to move files to trash with fallback to os.remove.
See dol.trash module for deletion strategies: permanent_delete, trash_only, etc.

### *class* dol.filesys.FileBytesReader(rootdir, subpath='', pattern_for_field=None, max_levels=None, , include_hidden=False, assert_rootdir_existence=False)

Bases: [`FileCollection`](#dol.filesys.FileCollection), [`KvReader`](dol.base.html.md#dol.base.KvReader)

Reader mapping file paths under `rootdir` to the files’ bytes.

### *class* dol.filesys.FileCollection(rootdir, subpath='', pattern_for_field=None, max_levels=None, , include_hidden=False, assert_rootdir_existence=False)

Bases: [`FileSysCollection`](#dol.filesys.FileSysCollection)

Collection of the file paths under `rootdir`.

### *class* dol.filesys.FileInfoReader(rootdir, subpath='', pattern_for_field=None, max_levels=None, , include_hidden=False, assert_rootdir_existence=False)

Bases: [`FileCollection`](#dol.filesys.FileCollection), [`KvReader`](dol.base.html.md#dol.base.KvReader)

Reader mapping file paths to their `os.stat` result.

### *class* dol.filesys.FileStringPersister(\*args, delete_func=None, \*\*kwargs)

Bases: [`FileBytesPersister`](#dol.filesys.FileBytesPersister)

Persister mapping file paths to the files’ text (files opened in text mode).

Reads and writes as UTF-8 explicitly, rather than inheriting
`locale.getpreferredencoding()` (see i2mint/dol#97): a store is a serialization
boundary, and one whose format silently depends on an ambient environment
variable isn’t really specified. Without this, a write can raise on a
non-ASCII-locale machine, or a store synced between two machines with different
locales can silently corrupt on round trip.

### *class* dol.filesys.FileStringReader(rootdir, subpath='', pattern_for_field=None, max_levels=None, , include_hidden=False, assert_rootdir_existence=False)

Bases: [`FileBytesReader`](#dol.filesys.FileBytesReader)

Reader mapping file paths to the files’ text (files opened in text mode).

Reads as UTF-8 explicitly, rather than inheriting `locale.getpreferredencoding()`
(see i2mint/dol#97): a store is a serialization boundary, and one whose format
silently depends on an ambient environment variable isn’t really specified. This
also matches how `FileStringPersister` writes (below), so a round trip is safe
regardless of which locale reads or writes.

### *class* dol.filesys.FileSysCollection(rootdir, subpath='', pattern_for_field=None, max_levels=None, , include_hidden=False, assert_rootdir_existence=False)

Bases: [`Collection`](dol.base.html.md#dol.base.Collection)

Base collection of file-system paths under `rootdir`, optionally restricted by `subpath`, `max_levels` and hidden-file inclusion.

#### with_relative_paths()

Return a copy of self with relative paths

### *class* dol.filesys.Files(\*args, delete_func=None, \*\*kwargs)

Bases: [`PrefixRelativizationMixin`](dol.paths.html.md#dol.paths.PrefixRelativizationMixin), [`Store`](dol.base.html.md#dol.base.Store)

FileBytesPersister with relative paths

#### is_valid_key(k, \*args, \_\_name='is_valid_key', \*\*kwargs)

`is_valid_key` on the inner key – see `mk_relative_path_store`.

#### validate_key(k, \*args, \_\_name='validate_key', \*\*kwargs)

`validate_key` on the inner key – see `mk_relative_path_store`.

### *class* dol.filesys.FilesReader(rootdir, subpath='', pattern_for_field=None, max_levels=None, , include_hidden=False, assert_rootdir_existence=False)

Bases: [`PrefixRelativizationMixin`](dol.paths.html.md#dol.paths.PrefixRelativizationMixin), [`Store`](dol.base.html.md#dol.base.Store)

FileBytesReader with relative paths

#### is_valid_key(k, \*args, \_\_name='is_valid_key', \*\*kwargs)

`is_valid_key` on the inner key – see `mk_relative_path_store`.

#### validate_key(k, \*args, \_\_name='validate_key', \*\*kwargs)

`validate_key` on the inner key – see `mk_relative_path_store`.

### *class* dol.filesys.JsonFiles(\*args, delete_func=None, \*\*kwargs)

Bases: [`Store`](dol.base.html.md#dol.base.Store)

A store of json files

#### is_valid_key(k, \*args, \_\_name='is_valid_key', \*\*kwargs)

`is_valid_key` on the inner key – see `mk_relative_path_store`.

#### validate_key(k, \*args, \_\_name='validate_key', \*\*kwargs)

`validate_key` on the inner key – see `mk_relative_path_store`.

### *class* dol.filesys.Jsons(\*args, delete_func=None, \*\*kwargs)

Bases: [`Store`](dol.base.html.md#dol.base.Store)

Like JsonFiles, but with added .json extension handling
Namely: filtering for `.json` extensions but not showing the extension in keys

#### is_valid_key(k, \*args, \_\_name='is_valid_key', \*\*kwargs)

`is_valid_key` on the inner key – see `mk_relative_path_store`.

#### validate_key(k, \*args, \_\_name='validate_key', \*\*kwargs)

`validate_key` on the inner key – see `mk_relative_path_store`.

### *exception* dol.filesys.KeyValidationError

Bases: [`KeyError`](https://docs.python.org/3/builtins/exceptions.html#KeyError)

A `KeyError` for keys that fail a file-system store’s validation.

### *class* dol.filesys.LocalFileDeleteMixin

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Mixin providing configurable file deletion.

The deletion function can be configured either at class level by setting
the \_delete_func class attribute, or at instance level by setting the
\_delete_func instance attribute.

By default, uses safe deletion that tries to move to trash with fallback
to os.remove (with warning).

See dol.trash module for available deletion strategies:

- default_delete_func: Safe trash with warning on fallback
- permanent_delete: Direct os.remove (no warnings)
- trash_only: Error if trash unavailable

### *class* dol.filesys.MakeMissingDirsStoreMixin

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Will make a local file store automatically create the directories needed to create a file.
Should be placed before the concrete perisister in the mro but in such a manner so that it receives full paths.

### *class* dol.filesys.PickleFiles(\*args, delete_func=None, \*\*kwargs)

Bases: [`Store`](dol.base.html.md#dol.base.Store)

A store of pickles

#### is_valid_key(k, \*args, \_\_name='is_valid_key', \*\*kwargs)

`is_valid_key` on the inner key – see `mk_relative_path_store`.

#### validate_key(k, \*args, \_\_name='validate_key', \*\*kwargs)

`validate_key` on the inner key – see `mk_relative_path_store`.

### dol.filesys.PickleStore

alias of [`PickleFiles`](#dol.filesys.PickleFiles)

### *class* dol.filesys.PickleStores(rootdir, subpath='', pattern_for_field=None, max_levels=None, , include_hidden=False, assert_rootdir_existence=False)

Bases: [`PrefixRelativizationMixin`](dol.paths.html.md#dol.paths.PrefixRelativizationMixin), [`Store`](dol.base.html.md#dol.base.Store)

Reader mapping each sub-directory of `rootdir` to a `PickleFiles` store of it.

#### is_valid_key(k, \*args, \_\_name='is_valid_key', \*\*kwargs)

`is_valid_key` on the inner key – see `mk_relative_path_store`.

#### validate_key(k, \*args, \_\_name='validate_key', \*\*kwargs)

`validate_key` on the inner key – see `mk_relative_path_store`.

### dol.filesys.RelPathFileBytesPersister

alias of [`Files`](#dol.filesys.Files)

### dol.filesys.RelPathFileBytesReader

alias of [`FilesReader`](#dol.filesys.FilesReader)

### dol.filesys.RelPathFileStringPersister

alias of [`TextFiles`](#dol.filesys.TextFiles)

### dol.filesys.RelPathFileStringReader

alias of [`TextFilesReader`](#dol.filesys.TextFilesReader)

### *class* dol.filesys.ReprMixin

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A `__repr__` showing the `_init_kwargs` the instance was created with.

### *class* dol.filesys.TextFiles(\*args, delete_func=None, \*\*kwargs)

Bases: [`PrefixRelativizationMixin`](dol.paths.html.md#dol.paths.PrefixRelativizationMixin), [`Store`](dol.base.html.md#dol.base.Store)

FileStringPersister with relative paths

#### is_valid_key(k, \*args, \_\_name='is_valid_key', \*\*kwargs)

`is_valid_key` on the inner key – see `mk_relative_path_store`.

#### validate_key(k, \*args, \_\_name='validate_key', \*\*kwargs)

`validate_key` on the inner key – see `mk_relative_path_store`.

### *class* dol.filesys.TextFilesReader(rootdir, subpath='', pattern_for_field=None, max_levels=None, , include_hidden=False, assert_rootdir_existence=False)

Bases: [`PrefixRelativizationMixin`](dol.paths.html.md#dol.paths.PrefixRelativizationMixin), [`Store`](dol.base.html.md#dol.base.Store)

FileStringReader with relative paths

#### is_valid_key(k, \*args, \_\_name='is_valid_key', \*\*kwargs)

`is_valid_key` on the inner key – see `mk_relative_path_store`.

#### validate_key(k, \*args, \_\_name='validate_key', \*\*kwargs)

`validate_key` on the inner key – see `mk_relative_path_store`.

### dol.filesys.create_directories(dirpath, max_dirs_to_make=None)

Create directories up to a specified limit.

* **Parameters:**
  * **dirpath** – The directory path to create.
  * **max_dirs_to_make** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The maximum number of directories to create. If None,
    there’s no limit.
* **Returns:**
  True if the directory exists (already, or after creation); False if creating
  it would need more than `max_dirs_to_make` new directories (none are made).
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If max_dirs_to_make is negative.

### Examples

```pycon
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
```

```pycon
>>> temp_dir = tempfile.mkdtemp()
>>> target_dir = os.path.join(temp_dir, 'a', 'b', 'c', 'd')
>>> create_directories(target_dir)
True
>>> os.path.isdir(target_dir)
True
>>> shutil.rmtree(temp_dir)  # Cleanup
```

### dol.filesys.ensure_dir(dirpath, , max_dirs_to_make=None, verbose=False)

Ensure that a directory exists, creating it if necessary.

* **Parameters:**
  * **dirpath** – path to the directory to create
  * **max_dirs_to_make** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the maximum number of directories to create.
    If None, there’s no limit.
  * **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – controls verbosity (the noise ensure_dir makes if it make folder)
* **Returns:**
  the path to the directory

When the path does not exist, if `verbose` is:

- a `bool`’ a standard message will be printed
- a `callable`; will be called on dirpath before directory is created – you
  can use this to ask the user for confirmation for example
- a ‘’string\`\`; this string will be printed

Usage note: If you want to string or the (argument-less) callable to be dependent
on `dirpath`, you need make them so when calling ensure_dir.

### dol.filesys.ensure_slash_suffix(path)

Add a file separation (/ or ) at the end of path str, if not already present.

An empty path stays empty: an empty prefix has no slash to “ensure”, and turning
it into a bare separator anchors otherwise-absolute keys to the filesystem root.
On Windows that produces invalid paths like `\C:\Users\...` (a separator before
the drive letter -> `OSError: [Errno 22]`); e.g. `Files("")` used with
absolute keys, as in `dol.misc.get_obj`.

### dol.filesys.iter_dirpaths_in_folder_recursively(root_folder, max_levels=None, \_current_level=0, include_hidden=False)

Recursively generates dirpaths of folder (and subfolders, etc.) up to a given level

### dol.filesys.iter_filepaths_in_folder_recursively(root_folder, max_levels=None, \_current_level=0, include_hidden=False)

Recursively generates filepaths of folder (and subfolders, etc.) up to a given level

### dol.filesys.mk_absolute_path(path_format)

Expand a leading `~`, or make a leading `.` path absolute; other paths are returned as is.

### dol.filesys.mk_dirs_if_missing(store_cls=None, , max_dirs_to_make=None, verbose=False, key_condition=None, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Store decorator that will make the store create directories on write as
needed.

Note that it’ll only effect paths relative to the rootdir, which needs to be
ensured to exist separatedly.

### dol.filesys.mk_dirs_if_missing_preset(self, k, v, , max_dirs_to_make=None, verbose=False)

Preset function that will make the store create directories on write as needed.

### dol.filesys.mk_json_bytes_wrap(, loads_kwargs=None, dumps_kwargs=None)

Make a `wrap_kvs` value-codec wrapper for JSON, with kwargs for `json.loads`/`json.dumps`.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

### dol.filesys.mk_pickle_bytes_wrap(, loads_kwargs=None, dumps_kwargs=None)

Make a `wrap_kvs` value-codec wrapper for pickle, with kwargs for `pickle.loads`/`pickle.dumps`.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

### dol.filesys.mk_tmp_dol_dir(dirname='', make_it_if_necessary=True, verbose=False)

Create and return a path to a temporary directory that’s guaranteed to be
accessible to the user.

* **Parameters:**
  * **dirname** – Optional subdirectory name to append to the temporary directory path
  * **make_it_if_necessary** – Whether to create the directory if it doesn’t exist
  * **verbose** – Controls verbosity when creating directories
* **Returns:**
  Path to a temporary directory that the user has access to

#### NOTE
This function creates a user-specific temporary directory to avoid permission
issues with system-wide temporary directories.

### dol.filesys.paths_in_dir(rootdir, include_hidden=False)

Yield the paths of the entries of `rootdir` (directories with a trailing separator), skipping hidden ones unless `include_hidden`.

### dol.filesys.process_path(\*path, ensure_dir_exists=False, assert_exists=False, ensure_endswith_slash=False, ensure_does_not_end_with_slash=False, expanduser=True, expandvars=True, abspath=True, rootdir='')

Process a path string, ensuring it exists, and optionally expanding user.

* **Parameters:**
  * **path** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The path to process. Can be multiple components of a path.
  * **ensure_dir_exists** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to ensure the path exists.
  * **assert_exists** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to assert that the path exists.
  * **ensure_endswith_slash** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to ensure the path ends with a slash.
  * **ensure_does_not_end_with_slash** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to ensure the path does not end with a slash.
  * **expanduser** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to expand the user in the path.
  * **expandvars** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to expand environment variables in the path.
  * **abspath** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to convert the path to an absolute path.
  * **rootdir** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The root directory to prepend to the path.
* **Returns:**
  The processed path.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

The result uses the running OS’s native separator, so these examples assert
OS-independently (the literal forward-slash form is what you get on POSIX):

```pycon
>>> import os
>>> process_path('a', 'b', 'c').endswith(os.path.join('a', 'b', 'c'))
True
>>> p = process_path(
...     'a', 'b', 'c', rootdir='root_dir',
...     ensure_endswith_slash=True, abspath=False, expanduser=False, expandvars=False,
... )
>>> p == os.path.join('root_dir', 'a', 'b', 'c') + os.sep
True
```

### dol.filesys.resolve_dir(dirpath, assert_existence=False, ensure_existence=False)

Resolve a path to a full, real, path to a directory

### dol.filesys.resolve_path(path, assert_existence=False)

Resolve a path to a full, real, (file or folder) path (opt assert existence).
That is, resolve situations where ~ and . prefix the paths.

### dol.filesys.subfolder_stores(root_folder, \*, max_levels=None, include_hidden=False, relative_paths=True, slash_suffix=False, folder_to_store=<class 'dol.filesys.Files'>)

Create a store of subfolders of a given folder, where the keys are the subfolder
paths (by default, relative and slash-less) and the values are stores of these
subfolders.

By default, all subfolders will be taken, recursively, but this can be controlled by
the `max_levels` parameter.

### dol.filesys.temp_dir(dirname='', make_it_if_necessary=True, verbose=False)

Create and return a path to a temporary directory that’s guaranteed to be
accessible to the user.

* **Parameters:**
  * **dirname** – Optional subdirectory name to append to the temporary directory path
  * **make_it_if_necessary** – Whether to create the directory if it doesn’t exist
  * **verbose** – Controls verbosity when creating directories
* **Returns:**
  Path to a temporary directory that the user has access to

#### NOTE
This function creates a user-specific temporary directory to avoid permission
issues with system-wide temporary directories.

### dol.filesys.validate_key_and_raise_key_error_on_exception(func)

Method decorator: validate the key first, and re-raise any exception of the method as a `KeyError`.
