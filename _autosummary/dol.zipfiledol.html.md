# dol.zipfiledol

Data object layers and other utils to work with zip files.

### Functions

| [`zip_compress`](#dol.zipfiledol.zip_compress)(b[, filename, compression, ...])   | Compress input bytes, returning the compressed bytes                                                                           |
|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| [`zip_decompress`](#dol.zipfiledol.zip_decompress)(b, \*[, allowZip64, ...])        | Decompress input bytes of a single file zip, returning the uncompressed bytes                                                  |
| [`to_zip_file`](#dol.zipfiledol.to_zip_file)(b, zip_filepath[, filename, ...])   | Zip input bytes and save to a single-file zip file.                                                                            |
| [`file_or_folder_to_zip_file`](#dol.zipfiledol.file_or_folder_to_zip_file)(src_path[, ...])     | Zip input bytes and save to a single-file zip file.                                                                            |
| [`if_i_zipped_stats`](#dol.zipfiledol.if_i_zipped_stats)(b)                            | Compress and decompress bytes with four different methods and return a dictionary of (size and time) stats.                    |
| [`mk_flatzips_store`](#dol.zipfiledol.mk_flatzips_store)(dir_of_zips[, ...])           | A store so that you can work with a folder that has a bunch of zip files, as if they've all been extracted in the same folder. |
| [`remove_some_entries_from_zip`](#dol.zipfiledol.remove_some_entries_from_zip)(zip_source, ...)   | Removes specific keys from a zip file.                                                                                         |
| [`remove_mac_junk_from_zip`](#dol.zipfiledol.remove_mac_junk_from_zip)(zip_source, \*[, ...]) | Removes mac junk keys from zip                                                                                                 |

### Classes

| `COMPRESSION`()                                                                                   |                                                                                                              |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| [`ZipReader`](#dol.zipfiledol.ZipReader)(zip_file[, prefix, open_kws, ...])     | A KvReader to read the contents of a zip file.                                                               |
| [`ZipInfoReader`](#dol.zipfiledol.ZipInfoReader)(zip_file[, prefix, open_kws, ...]) |                                                                                                              |
| [`ZipFilesReader`](#dol.zipfiledol.ZipFilesReader)(rootdir[, subpath, ...])          | A local file reader whose keys are the zip filepaths of the rootdir and values are corresponding ZipReaders. |
| [`ZipFilesReaderAndBytesWriter`](#dol.zipfiledol.ZipFilesReaderAndBytesWriter)(rootdir[, ...])     | Like ZipFilesReader, but the ability to write bytes (assumed to be valid bytes of the zip format) to a key   |
| [`FlatZipFilesReader`](#dol.zipfiledol.FlatZipFilesReader)(rootdir[, subpath, ...])      | Read the union of the contents of multiple zip files.                                                        |
| [`FilesOfZip`](#dol.zipfiledol.FilesOfZip)(zip_file[, prefix, open_kws])         |                                                                                                              |
| [`FileStreamsOfZip`](#dol.zipfiledol.FileStreamsOfZip)(zip_file[, prefix, open_kws])   | Like FilesOfZip, but object returns are file streams instead.                                                |
| [`ZipFileStreamsReader`](#dol.zipfiledol.ZipFileStreamsReader)(rootdir[, subpath, ...])    | Like ZipFilesReader, but objects returned are file streams instead.                                          |
| [`ZipStore`](#dol.zipfiledol.ZipStore)                                         |                                                                                                              |
| [`ZipFiles`](#dol.zipfiledol.ZipFiles)(zip_filepath[, compression, ...])       | Zip read and writing.                                                                                        |

### Exceptions

| [`OverwriteNotAllowed`](#dol.zipfiledol.OverwriteNotAllowed)   |    |
|------------------------------------------------------------------------|----|
| [`EmptyZipError`](#dol.zipfiledol.EmptyZipError)         |    |

### *exception* dol.zipfiledol.EmptyZipError

Bases: [`KeyError`](https://docs.python.org/3/library/exceptions.html#KeyError), [`FileNotFoundError`](https://docs.python.org/3/library/exceptions.html#FileNotFoundError)

### *class* dol.zipfiledol.FileStreamsOfZip(zip_file, prefix='', open_kws=None)

Bases: [`FilesOfZip`](#dol.zipfiledol.FilesOfZip)

Like FilesOfZip, but object returns are file streams instead.
So you use it like this:

```default
z = FileStreamsOfZip(rootdir)
with z[relpath] as fp:
    ...  # do stuff with fp, like fp.readlines() or such...
```

### *class* dol.zipfiledol.FilesOfZip(zip_file, prefix='', open_kws=None)

Bases: [`ZipReader`](#dol.zipfiledol.ZipReader)

### *class* dol.zipfiledol.FlatZipFilesReader(rootdir, subpath='.+\\\\.zip', pattern_for_field=None, max_levels=0, zip_reader=<class 'dol.zipfiledol.ZipReader'>, \*\*zip_reader_kwargs)

Bases: [`FlatReader`](dol.sources.html.md#dol.sources.FlatReader), [`ZipFilesReader`](#dol.zipfiledol.ZipFilesReader)

Read the union of the contents of multiple zip files.
A local file reader whose keys are the zip filepaths of the rootdir and values are
corresponding ZipReaders.

Example use case:

A remote data provider creates snapshots of whatever changed (modified files and new
ones…) since the last snapshot, dumping snapshot zip files in a specic
accessible location.

You make `remote` and `local` stores and can update your local. Then you can perform
syncing actions such as:

```python
missing_keys = remote.keys() - local.keys()
local.update({k: remote[k] for k in missing_keys})  # downloads missing snapshots
```

The data will look something like this:

```python
dump_folder/
   2021_09_11.zip
   2021_09_12.zip
   2021_09_13.zip
   etc.
```

both on remote and local.

What should then local do to use this data?
Unzip and merge?

Well, one solution, provided through FlatZipFilesReader, is to not unzip at all,
but instead, give you a store that provides you a view “as if you unzipped and
merged”.

### *exception* dol.zipfiledol.OverwriteNotAllowed

Bases: [`FileExistsError`](https://docs.python.org/3/library/exceptions.html#FileExistsError), [`OverWritesNotAllowedError`](dol.errors.html.md#dol.errors.OverWritesNotAllowedError)

### *class* dol.zipfiledol.ZipFileStreamsReader(rootdir, subpath='.+\\\\.zip', pattern_for_field=None, max_levels=0, \*, zip_reader=<class 'dol.zipfiledol.FileStreamsOfZip'>, \*\*zip_reader_kwargs)

Bases: [`PrefixRelativizationMixin`](dol.paths.html.md#dol.paths.PrefixRelativizationMixin), [`Store`](dol.base.html.md#dol.base.Store)

Like ZipFilesReader, but objects returned are file streams instead.

#### is_valid_key(k, \*args, \_\_name='is_valid_key', \*\*kwargs)

`is_valid_key` on the inner key – see `mk_relative_path_store`.

#### validate_key(k, \*args, \_\_name='validate_key', \*\*kwargs)

`validate_key` on the inner key – see `mk_relative_path_store`.

### *class* dol.zipfiledol.ZipFiles(zip_filepath, compression=8, allow_overwrites=True, pwd=None)

Bases: [`KvPersister`](dol.base.html.md#dol.base.KvPersister)

Zip read and writing.
When you want to read zips, there’s the `FilesOfZip`, `ZipReader`, or `ZipFilesReader` we
know and love.

Sometimes though, you want to write to zips too. For this, we have `ZipFiles`.

Since ZipFiles can write to a zip, it’s read functionality is not going to assume static data,
and cache things, as your favorite zip readers did.
This, and the acrobatics need to disguise the weird zipfile into something more… key-value
natural,
makes for a not so efficient store, out of the box.

I advise using one of the zip readers if all you need to do is read, or subclassing or
: wrapping ZipFiles with caching layers if it is appropriate to you.

Let’s verify that a ZipFiles can indeed write data. First, we’ll set things up!

```pycon
>>> from tempfile import gettempdir
>>> import os
>>>
>>> rootdir = gettempdir()
>>>
>>> # preparation
>>> test_zipfile = os.path.join(rootdir, 'zipstore_test_file.zip')
>>> if os.path.isfile(test_zipfile):
...     os.remove(test_zipfile)
>>> assert not os.path.isfile(test_zipfile)
```

Okay, test_zipfile doesn’t exist (but will soon…)

```pycon
>>> z = ZipFiles(test_zipfile)
```

See that the file still doesn’t exist (it will only be created when we start writing)

```pycon
>>> assert not os.path.isfile(test_zipfile)
>>> list(z)  # z "is" empty (which makes sense?)
[]
```

Now let’s write something interesting (notice, it has to be in bytes):

```pycon
>>> z['foo'] = b'bar'
>>> list(z)  # now we have something in z
['foo']
>>> z['foo']  # and that thing is what we put there
b'bar'
```

And indeed we have a zip file now:

```pycon
>>> assert os.path.isfile(test_zipfile)
```

### *class* dol.zipfiledol.ZipFilesReader(rootdir, subpath='.+\\\\.zip', pattern_for_field=None, max_levels=0, zip_reader=<class 'dol.zipfiledol.ZipReader'>, \*\*zip_reader_kwargs)

Bases: [`FileCollection`](dol.filesys.html.md#dol.filesys.FileCollection), [`KvReader`](dol.base.html.md#dol.base.KvReader)

A local file reader whose keys are the zip filepaths of the rootdir and values are
corresponding ZipReaders.

### *class* dol.zipfiledol.ZipFilesReaderAndBytesWriter(rootdir, subpath='.+\\\\.zip', pattern_for_field=None, max_levels=0, zip_reader=<class 'dol.zipfiledol.ZipReader'>, \*\*zip_reader_kwargs)

Bases: [`ZipFilesReader`](#dol.zipfiledol.ZipFilesReader)

Like ZipFilesReader, but the ability to write bytes (assumed to be valid bytes of
the zip format) to a key

### *class* dol.zipfiledol.ZipInfoReader(zip_file, prefix='', , open_kws=None, file_info_filt=None)

Bases: [`ZipReader`](#dol.zipfiledol.ZipReader)

### *class* dol.zipfiledol.ZipReader(zip_file, prefix='', , open_kws=None, file_info_filt=None)

Bases: [`KvReader`](dol.base.html.md#dol.base.KvReader)

A KvReader to read the contents of a zip file.
Provides a KV perspective of [https://docs.python.org/3/library/zipfile.html](https://docs.python.org/3/library/zipfile.html)

`ZipReader` has two value categories: Directories and Files.
Both categories are distinguishable by the keys, through the “ends with slash” convention.

When a file, the value return is bytes, as usual.

When a directory, the value returned is a `ZipReader` itself, with all params the same,
except for the `prefix`

> which serves `to specify the subfolder (that is, ``prefix`` acts as a filter).

#### NOTE
If you get data zipped by a mac, you might get some junk along with it.
Namely `__MACOSX` folders `.DS_Store` files. I won’t rant about it, since others have.
But you might find it useful to remove them from view. One choice is to use
`dol.trans.filt_iter`
to get a filtered view of the zips contents. In most cases, this should do the job:

```default
# applied to store instance or class:
store = filt_iter(filt=lambda x: not x.startswith('__MACOSX') and '.DS_Store' not in x)(store)
```

Another option is just to remove these from the zip file once and for all. In unix-like systems:

```default
zip -d filename.zip __MACOSX/\*
zip -d filename.zip \*/.DS_Store
```

### Examples

```default
# >>> s = ZipReader('/path/to/some_zip_file.zip')
# >>> len(s)
# 53432
# >>> list(s)[:3]  # the first 3 elements (well... their keys)
# ['odir/', 'odir/app/', 'odir/app/data/']
# >>> list(s)[-3:]  # the last 3 elements (well... their keys)
# ['odir/app/data/audio/d/1574287049078391/m/Ctor.json',
#  'odir/app/data/audio/d/1574287049078391/m/intensity.json',
#  'odir/app/data/run/status.json']
# >>> # getting a file (note that by default, you get bytes, so need to decode)
# >>> s['odir/app/data/run/status.json'].decode()
# b'{"test_phase_number": 9, "test_phase": "TestActions.IGNORE_TEST", "session_id": 0}'
# >>> # when you ask for the contents for a key that's a directory,
# >>> # you get a ZipReader filtered for that prefix:
# >>> s['odir/app/data/audio/']
# ZipReader('/path/to/some_zip_file.zip', 'odir/app/data/audio/', {}, <function
take_everything at 0x1538999e0>)
# >>> # Often, you only want files (not directories)
# >>> # You can filter directories out using the file_info_filt argument
# >>> s = ZipReader('/path/to/some_zip_file.zip', file_info_filt=ZipReader.FILES_ONLY)
# >>> len(s)  # compare to the 53432 above, that contained dirs too
# 53280
# >>> list(s)[:3]  # first 3 keys are all files now
# ['odir/app/data/plc/d/1574304926795633/d/1574305026895702',
#  'odir/app/data/plc/d/1574304926795633/d/1574305276853053',
#  'odir/app/data/plc/d/1574304926795633/d/1574305159343326']
# >>>
# >>> # ZipReader.FILES_ONLY and ZipReader.DIRS_ONLY are just convenience filt functions
# >>> # Really, you can provide any custom one yourself.
# >>> # This filter function should take a ZipInfo object, and return True or False.
# >>> # (https://docs.python.org/3/library/zipfile.html#zipfile.ZipInfo)
# >>>
# >>> import re
# >>> p = re.compile('audio.*\.json$')
# >>> my_filt_func = lambda fileinfo: bool(p.search(fileinfo.filename))
# >>> s = ZipReader('/Users/twhalen/Downloads/2019_11_21.zip', file_info_filt=my_filt_func)
# >>> len(s)
# 48
# >>> list(s)[:3]
# ['odir/app/data/audio/d/1574333557263758/m/Ctor.json',
#  'odir/app/data/audio/d/1574333557263758/m/intensity.json',
#  'odir/app/data/audio/d/1574288084739961/m/Ctor.json']
```

### dol.zipfiledol.ZipStore

alias of [`ZipFiles`](#dol.zipfiledol.ZipFiles)

### dol.zipfiledol.file_or_folder_to_zip_file(src_path, zip_filepath=None, filename=None, , compression=8, allow_overwrites=True, pwd=None)

Zip input bytes and save to a single-file zip file.

### dol.zipfiledol.if_i_zipped_stats(b)

Compress and decompress bytes with four different methods and return a dictionary
of (size and time) stats.

```pycon
>>> b = b'x' * 1000 + b'y' * 1000  # 2000 (quite compressible) bytes
>>> if_i_zipped_stats(b)
{'uncompressed': {'bytes': 2000,
  'comp_time': 0,
  'uncomp_time': 0},
 'deflated': {'bytes': 137,
  'comp_time': 0.00015592575073242188,
  'uncomp_time': 0.00012612342834472656},
 'bzip2': {'bytes': 221,
  'comp_time': 0.0013129711151123047,
  'uncomp_time': 0.0011119842529296875},
 'lzma': {'bytes': 206,
  'comp_time': 0.0058901309967041016,
  'uncomp_time': 0.0005228519439697266}}
```

### dol.zipfiledol.mk_flatzips_store(dir_of_zips, zip_pair_path_preproc=<built-in function sorted>, mk_store=<class 'dol.zipfiledol.FlatZipFilesReader'>, \*\*extra_mk_store_kwargs)

A store so that you can work with a folder that has a bunch of zip files,
as if they’ve all been extracted in the same folder.
Note that `zip_pair_path_preproc` can be used to control how to resolve key conflicts
(i.e. when you get two different zip files that have a same path in their contents).
The last path encountered by `zip_pair_path_preproc(zip_path_pairs)` is the one that
will be used, so one should make `zip_pair_path_preproc` act accordingly.

### dol.zipfiledol.remove_mac_junk_from_zip(zip_source, \*, keys_to_be_removed=<function is_a_mac_junk_path>, ask_before_before_deleting=False, remove_action='filter')

Removes mac junk keys from zip

### dol.zipfiledol.remove_some_entries_from_zip(zip_source, keys_to_be_removed, ask_before_before_deleting=True, , remove_action='filter')

Removes specific keys from a zip file.

* **Parameters:**
  * **zip_source** – zip filepath, bytes, or whatever a `ZipFiles` can take
  * **keys_to_be_removed** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/library/stdtypes.html#str)], [`bool`](https://docs.python.org/3/library/functions.html#bool)] | [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/library/stdtypes.html#str)]) – An iterable of keys or a boolean filter function
  * **ask_before_before_deleting** – True (default) if the user should be
    presented with the keys first, and asked permission to delete.
* **Returns:**
  The ZipFiles (in case you want to do further work with it)

#### TIP
If you want to delete with no questions asked, use currying:

```pycon
>>> from functools import partial
>>> rm_keys_without_asking = partial(
...     remove_some_entries_from_zip,
...     ask_before_before_deleting=False
... )
```

### dol.zipfiledol.to_zip_file(b, zip_filepath, filename=None, , compression=8, allow_overwrites=True, pwd=None, encoding='utf-8')

Zip input bytes and save to a single-file zip file.

* **Parameters:**
  * **b** ([`bytes`](https://docs.python.org/3/library/stdtypes.html#bytes) | [`str`](https://docs.python.org/3/library/stdtypes.html#str)) – Input bytes or string
  * **zip_filepath** – zip filepath to save the zipped input to
  * **filename** – The name/path of the zip entry we want to save to
  * **encoding** – In case the input is str, the encoding to use to convert to bytes

### dol.zipfiledol.zip_compress(b, filename='some_bytes', , compression=8, allowZip64=True, compresslevel=None, strict_timestamps=True, encoding='utf-8')

Compress input bytes, returning the compressed bytes

* **Return type:**
  [`bytes`](https://docs.python.org/3/library/stdtypes.html#bytes)

```pycon
>>> b = b'x' * 1000 + b'y' * 1000  # 2000 (quite compressible) bytes
>>> len(b)
2000
>>>
>>> zipped_bytes = zip_compress(b)
>>> # Note: Compression details will be system dependent
>>> len(zipped_bytes)
137
>>> unzipped_bytes = zip_decompress(zipped_bytes)
>>> unzipped_bytes == b  # verify that unzipped bytes are the same as the original
True
>>>
>>> from dol.zipfiledol import compression_methods
>>>
>>> zipped_bytes = zip_compress(b, compression=compression_methods['bzip2'])
>>> # Note: Compression details will be system dependent
>>> len(zipped_bytes)
221
>>> unzipped_bytes = zip_decompress(zipped_bytes)
>>> unzipped_bytes == b  # verify that unzipped bytes are the same as the original
True
```

### dol.zipfiledol.zip_decompress(b, , allowZip64=True, compresslevel=None, strict_timestamps=True)

Decompress input bytes of a single file zip, returning the uncompressed bytes

See `zip_compress` for usage examples.

* **Return type:**
  [`bytes`](https://docs.python.org/3/library/stdtypes.html#bytes)
