"""
Data object layers and other utils to work with zip files.
"""

import os
from pathlib import Path
from io import BytesIO
from functools import partial, wraps
from typing import Callable, Union, Iterable, Mapping, Literal
import zipfile
from zipfile import (
    ZipFile,
    BadZipFile,
    ZIP_STORED,
    ZIP_DEFLATED,
    ZIP_BZIP2,
    ZIP_LZMA,
)
from dol.base import KvReader, KvPersister
from dol.trans import filt_iter
from dol.filesys import FileCollection, Files
from dol.util import lazyprop, fullpath
from dol.sources import FlatReader

__all__ = [
    "COMPRESSION",
    "DFLT_COMPRESSION",
    "compression_methods",
    "zip_compress",
    "zip_decompress",
    "to_zip_file",
    "file_or_folder_to_zip_file",
    "if_i_zipped_stats",
    "ZipReader",
    "ZipInfoReader",
    "ZipFilesReader",
    "ZipFilesReaderAndBytesWriter",
    "FlatZipFilesReader",
    "mk_flatzips_store",
    "FilesOfZip",
    "FileStreamsOfZip",
    "ZipFileStreamsReader",
    "OverwriteNotAllowed",
    "EmptyZipError",
    "ZipStore",
    "remove_some_entries_from_zip",
    "remove_mac_junk_from_zip",
]

# TODO: Do all systems have this? If not, need to choose dflt carefully
#  (choose dynamically?)
DFLT_COMPRESSION = zipfile.ZIP_DEFLATED
DFLT_ENCODING = "utf-8"


class COMPRESSION:
    # The numeric constant for an uncompressed archive member.
    ZIP_STORED = ZIP_STORED
    # The numeric constant for the usual ZIP compression method. Requires zlib module.
    ZIP_DEFLATED = ZIP_DEFLATED
    # The numeric constant for the BZIP2 compression method. Requires the bz2 module:
    ZIP_BZIP2 = ZIP_BZIP2
    # The numeric constant for the LZMA compression method. Requires the lzma module:
    ZIP_LZMA = ZIP_LZMA


compression_methods = {
    "stored": zipfile.ZIP_STORED,  # doesn't even compress
    "deflated": zipfile.ZIP_DEFLATED,  # usual zip compression method
    "bzip2": zipfile.ZIP_BZIP2,  # BZIP2 compression method.
    "lzma": zipfile.ZIP_LZMA,  # LZMA compression method
}


def take_everything(fileinfo):
    return True


def zip_compress(
    b: Union[bytes, str],
    filename="some_bytes",
    *,
    compression=DFLT_COMPRESSION,
    allowZip64=True,
    compresslevel=None,
    strict_timestamps=True,
    encoding=DFLT_ENCODING,
) -> bytes:
    """Compress input bytes, returning the compressed bytes

    >>> b = b'x' * 1000 + b'y' * 1000  # 2000 (quite compressible) bytes
    >>> len(b)
    2000
    >>>
    >>> zipped_bytes = zip_compress(b)
    >>> # Note: Compression details will be system dependent
    >>> len(zipped_bytes)  # doctest: +SKIP
    137
    >>> unzipped_bytes = zip_decompress(zipped_bytes)
    >>> unzipped_bytes == b  # verify that unzipped bytes are the same as the original
    True
    >>>
    >>> from dol.zipfiledol import compression_methods
    >>>
    >>> zipped_bytes = zip_compress(b, compression=compression_methods['bzip2'])
    >>> # Note: Compression details will be system dependent
    >>> len(zipped_bytes)  # doctest: +SKIP
    221
    >>> unzipped_bytes = zip_decompress(zipped_bytes)
    >>> unzipped_bytes == b  # verify that unzipped bytes are the same as the original
    True
    """
    kwargs = dict(
        compression=compression,
        allowZip64=allowZip64,
        compresslevel=compresslevel,
        strict_timestamps=strict_timestamps,
    )
    bytes_buffer = BytesIO()
    if isinstance(b, str):  # if b is a string, need to convert to bytes
        b = b.encode(encoding)
    with ZipFile(bytes_buffer, "w", **kwargs) as fp:
        fp.writestr(filename, b)
    return bytes_buffer.getvalue()


def zip_decompress(
    b: bytes,
    *,
    allowZip64=True,
    compresslevel=None,
    strict_timestamps=True,
) -> bytes:
    """Decompress input bytes of a single file zip, returning the uncompressed bytes

    See ``zip_compress`` for usage examples.
    """
    kwargs = dict(
        allowZip64=allowZip64,
        compresslevel=compresslevel,
        strict_timestamps=strict_timestamps,
    )
    bytes_buffer = BytesIO(b)
    with ZipFile(bytes_buffer, "r", **kwargs) as zip_file:
        file_list = zip_file.namelist()
        if len(file_list) != 1:
            raise RuntimeError("zip_decompress only works with single file zips")
        filename = file_list[0]
        with zip_file.open(filename, "r") as fp:
            file_bytes = fp.read()
    return file_bytes


def _filename_from_zip_path(path):
    filename = path  # default
    if path.endswith(".zip"):
        filename, _ = os.path.splitext(os.path.basename(path))
    return filename


# TODO: Look into pwd: Should we use it for setting pwd when pwd doesn't exist?
def to_zip_file(
    b: Union[bytes, str],
    zip_filepath,
    filename=None,
    *,
    compression=DFLT_COMPRESSION,
    allow_overwrites=True,
    pwd=None,
    encoding=DFLT_ENCODING,
):
    """Zip input bytes and save to a single-file zip file.

    :param b: Input bytes or string
    :param zip_filepath: zip filepath to save the zipped input to
    :param filename: The name/path of the zip entry we want to save to
    :param encoding: In case the input is str, the encoding to use to convert to bytes

    """
    z = ZipStore(
        zip_filepath,
        compression=compression,
        allow_overwrites=allow_overwrites,
        pwd=pwd,
    )
    filename = filename or _filename_from_zip_path(zip_filepath)
    if isinstance(b, str):  # if b is a string, need to convert to bytes
        b = b.encode(encoding)
    z[filename] = b


def file_or_folder_to_zip_file(
    src_path: str,
    zip_filepath=None,
    filename=None,
    *,
    compression=DFLT_COMPRESSION,
    allow_overwrites=True,
    pwd=None,
):
    """Zip input bytes and save to a single-file zip file."""

    if zip_filepath is None:
        zip_filepath = os.path.basename(src_path) + ".zip"

    z = ZipStore(
        zip_filepath,
        compression=compression,
        allow_overwrites=allow_overwrites,
        pwd=pwd,
    )

    if os.path.isfile(src_path):
        filename = filename or os.path.basename(src_path)
        z[filename] = Path(src_path).read_bytes()
    elif os.path.isdir(src_path):
        src = Files(src_path)
        for k, v in src.items():
            z[k] = v
    else:
        raise FileNotFoundError(f"{src_path}")


def if_i_zipped_stats(b: bytes):
    """Compress and decompress bytes with four different methods and return a dictionary
    of (size and time) stats.

    >>> b = b'x' * 1000 + b'y' * 1000  # 2000 (quite compressible) bytes
    >>> if_i_zipped_stats(b)  # doctest: +SKIP
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
    """
    import time

    stats = dict()
    stats["uncompressed"] = {"bytes": len(b), "comp_time": 0, "uncomp_time": 0}
    for name, compression in compression_methods.items():
        if name != "stored":
            try:
                stats[name] = dict.fromkeys(stats["uncompressed"])
                tic = time.time()
                compressed = zip_compress(b, compression=compression)
                elapsed = time.time() - tic
                stats[name]["bytes"] = len(compressed)
                stats[name]["comp_time"] = elapsed
                tic = time.time()
                uncompressed = zip_decompress(compressed)
                elapsed = time.time() - tic
                assert (
                    uncompressed == b
                ), "the uncompressed bytes were different than the original"
                stats[name]["uncomp_time"] = elapsed
            except Exception:
                raise
                pass
    return stats


class ZipReader(KvReader):
    r"""A KvReader to read the contents of a zip file.
    Provides a KV perspective of https://docs.python.org/3/library/zipfile.html

    ``ZipReader`` has two value categories: Directories and Files.
    Both categories are distinguishable by the keys, through the "ends with slash" convention.

    When a file, the value return is bytes, as usual.

    When a directory, the value returned is a ``ZipReader`` itself, with all params the same,
    except for the ``prefix``
     which serves `to specify the subfolder (that is, ``prefix`` acts as a filter).

    Note: If you get data zipped by a mac, you might get some junk along with it.
    Namely `__MACOSX` folders `.DS_Store` files. I won't rant about it, since others have.
    But you might find it useful to remove them from view. One choice is to use
    `dol.trans.filt_iter`
    to get a filtered view of the zips contents. In most cases, this should do the job:

    .. code-block::

        # applied to store instance or class:
        store = filt_iter(filt=lambda x: not x.startswith('__MACOSX') and '.DS_Store' not in x)(store)


    Another option is just to remove these from the zip file once and for all. In unix-like systems:

    .. code-block::

        zip -d filename.zip __MACOSX/\*
        zip -d filename.zip \*/.DS_Store


    Examples:

    .. code-block::

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
    """

    def __init__(
        self,
        zip_file,
        prefix="",
        *,
        open_kws=None,
        file_info_filt=None,
    ):
        """

        Args:
            zip_file: A path to make ZipFile(zip_file)
            prefix: A prefix to filter by.
            open_kws:  To be used when doing a ZipFile(...).open
            file_info_filt: Filter for the FileInfo objects (see
            https://docs.python.org/3/library/zipfile.html)
                of the paths listed in the zip file
        """
        self.open_kws = open_kws or {}
        self.file_info_filt = file_info_filt or ZipReader.EVERYTHING
        self.prefix = prefix
        if not isinstance(zip_file, ZipFile):
            if isinstance(zip_file, str):
                zip_file = fullpath(zip_file)
            if isinstance(zip_file, dict):
                zip_file = ZipFile(**zip_file)
            elif isinstance(zip_file, (tuple, list)):
                zip_file = ZipFile(*zip_file)
            elif isinstance(zip_file, bytes):
                zip_file = ZipFile(BytesIO(zip_file))
            else:
                zip_file = ZipFile(zip_file)
        self.zip_file = zip_file

    @classmethod
    def for_files_only(cls, zip_file, prefix="", open_kws=None, file_info_filt=None):
        if file_info_filt is None:
            file_info_filt = ZipReader.FILES_ONLY
        else:
            _file_info_filt = file_info_filt

            def file_info_filt(x):
                return ZipReader.FILES_ONLY(x) and _file_info_filt(x)

        return cls(zip_file, prefix, open_kws, file_info_filt)

    # TODO: Unaware of trans (filters, key trans, etc.)
    @lazyprop
    def info_for_key(self):
        return {
            x.filename: x
            for x in self.zip_file.infolist()
            if x.filename.startswith(self.prefix) and self.file_info_filt(x)
        }

    def __iter__(self):
        # using zip_file.infolist(), we could also filter for info (like directory/file)
        yield from self.info_for_key.keys()

    def __getitem__(self, k):
        if not self.info_for_key[k].is_dir():
            with self.zip_file.open(k, **self.open_kws) as fp:
                return fp.read()
        else:  # is a directory
            return self.__class__(self.zip_file, k, self.open_kws, self.file_info_filt)

    def __len__(self):
        return len(self.info_for_key)

    @staticmethod
    def FILES_ONLY(fileinfo):
        return not fileinfo.is_dir()

    @staticmethod
    def DIRS_ONLY(fileinfo):
        return fileinfo.is_dir()

    @staticmethod
    def EVERYTHING(fileinfo):
        return True

    def __repr__(self):
        args_str = ", ".join(
            (
                f"'{self.zip_file.filename}'",
                f"'{self.prefix}'",
                f"{self.open_kws}",
                f"{self.file_info_filt}",
            )
        )
        return f"{self.__class__.__name__}({args_str})"

    # TODO: Unaware of trans (filters, key trans, etc.)
    def get_info_reader(self):
        return ZipInfoReader(
            zip_file=self.zip_file,
            prefix=self.prefix,
            open_kws=self.open_kws,
            file_info_filt=self.file_info_filt,
        )


class ZipInfoReader(ZipReader):
    def __getitem__(self, k):
        return self.zip_file.getinfo(k)


class FilesOfZip(ZipReader):
    def __init__(self, zip_file, prefix="", open_kws=None):
        super().__init__(
            zip_file,
            prefix=prefix,
            open_kws=open_kws,
            file_info_filt=ZipReader.FILES_ONLY,
        )


# TODO: This file object item is more fundemental than file contents.
#  Should it be at the base?
class FileStreamsOfZip(FilesOfZip):
    """Like FilesOfZip, but object returns are file streams instead.
    So you use it like this:

    .. code-block::

        z = FileStreamsOfZip(rootdir)
        with z[relpath] as fp:
            ...  # do stuff with fp, like fp.readlines() or such...

    """

    def __getitem__(self, k):
        return self.zip_file.open(k, **self.open_kws)


class ZipFilesReader(FileCollection, KvReader):
    """A local file reader whose keys are the zip filepaths of the rootdir and values are
    corresponding ZipReaders.
    """

    def __init__(
        self,
        rootdir,
        subpath=r".+\.zip",
        pattern_for_field=None,
        max_levels=0,
        zip_reader=ZipReader,
        **zip_reader_kwargs,
    ):
        super().__init__(rootdir, subpath, pattern_for_field, max_levels)
        self.zip_reader = zip_reader
        self.zip_reader_kwargs = zip_reader_kwargs
        if self.zip_reader is ZipReader:
            self.zip_reader_kwargs = dict(
                dict(
                    prefix="",
                    open_kws=None,
                    file_info_filt=ZipReader.FILES_ONLY,
                ),
                **self.zip_reader_kwargs,
            )

    def __getitem__(self, k):
        try:
            return self.zip_reader(k, **self.zip_reader_kwargs)
        except FileNotFoundError as e:
            raise KeyError(f"FileNotFoundError: {e}")


class ZipFilesReaderAndBytesWriter(ZipFilesReader):
    """Like ZipFilesReader, but the ability to write bytes (assumed to be valid bytes of
    the zip format) to a key
    """

    def __setitem__(self, k, v):
        with open(k, "wb") as fp:
            fp.write(v)


ZipFileReader = ZipFilesReader  # back-compatibility alias


# TODO: Add easy connection to ExplicitKeymapReader and other path trans and cache useful
#  for the folder of zips context
# TODO: The "injection" of _readers to be able to use FlatReader stinks.
class FlatZipFilesReader(FlatReader, ZipFilesReader):
    """Read the union of the contents of multiple zip files.
    A local file reader whose keys are the zip filepaths of the rootdir and values are
    corresponding ZipReaders.

    Example use case:

    A remote data provider creates snapshots of whatever changed (modified files and new
    ones...) since the last snapshot, dumping snapshot zip files in a specic
    accessible location.

    You make `remote` and `local` stores and can update your local. Then you can perform
    syncing actions such as:

    .. code-block:: python

        missing_keys = remote.keys() - local.keys()
        local.update({k: remote[k] for k in missing_keys})  # downloads missing snapshots


    The data will look something like this:

    .. code-block:: python

        dump_folder/
           2021_09_11.zip
           2021_09_12.zip
           2021_09_13.zip
           etc.

    both on remote and local.

    What should then local do to use this data?
    Unzip and merge?

    Well, one solution, provided through FlatZipFilesReader, is to not unzip at all,
    but instead, give you a store that provides you a view "as if you unzipped and
    merged".

    """

    __init__ = ZipFilesReader.__init__

    @lazyprop
    def _readers(self):
        rootdir_len = len(self.rootdir)
        return {
            path[rootdir_len:]: ZipFilesReader.__getitem__(self, path)
            for path in ZipFilesReader.__iter__(self)
        }

    _zip_readers = _readers  # back-compatibility alias


# TODO: Refactor zipfiledol to make it possible to design FlatZipFilesReaderFromBytes
#  better than the following.
#  * init doesn't use super, but super is locked to rootdir specification
#  * perhaps better making _readers a lazy mapping (not precompute all FilesOfZip(v))?
#  * Should ZipFilesReader be generalized to take bytes instead of rootdir?
#  * Using .zips to delegate the what in is
class FlatZipFilesReaderFromBytes(FlatReader, FilesOfZip):
    """Like FlatZipFilesReader but instead of sourcing with folder of zips, we source
    with the bytes of a zipped folder of zips"""

    @wraps(FilesOfZip.__init__)
    def __init__(self, *args, **kwargs):
        self.zips = FilesOfZip(*args, **kwargs)

    @lazyprop
    def _readers(self):
        return {k: FilesOfZip(v) for k, v in self.zips.items()}


def mk_flatzips_store(
    dir_of_zips,
    zip_pair_path_preproc=sorted,
    mk_store=FlatZipFilesReader,
    **extra_mk_store_kwargs,
):
    """A store so that you can work with a folder that has a bunch of zip files,
    as if they've all been extracted in the same folder.
    Note that `zip_pair_path_preproc` can be used to control how to resolve key conflicts
    (i.e. when you get two different zip files that have a same path in their contents).
    The last path encountered by `zip_pair_path_preproc(zip_path_pairs)` is the one that
    will be used, so one should make `zip_pair_path_preproc` act accordingly.
    """
    from dol.explicit import ExplicitKeymapReader

    z = mk_store(dir_of_zips, **extra_mk_store_kwargs)
    path_to_pair = {pair[1]: pair for pair in zip_pair_path_preproc(z)}
    return ExplicitKeymapReader(z, id_of_key=path_to_pair)


from dol.paths import mk_relative_path_store
from dol.util import partialclass

ZipFileStreamsReader = mk_relative_path_store(
    partialclass(ZipFilesReader, zip_reader=FileStreamsOfZip),
    prefix_attr="rootdir",
)
ZipFileStreamsReader.__name__ = "ZipFileStreamsReader"
ZipFileStreamsReader.__qualname__ = "ZipFileStreamsReader"
ZipFileStreamsReader.__doc__ = (
    """Like ZipFilesReader, but objects returned are file streams instead."""
)

from dol.errors import OverWritesNotAllowedError


class OverwriteNotAllowed(FileExistsError, OverWritesNotAllowedError): ...


class EmptyZipError(KeyError, FileNotFoundError): ...


class _EmptyZipReader(KvReader):
    def __init__(self, zip_filepath):
        self.zip_filepath = zip_filepath

    def __iter__(self):
        yield from ()

    def infolist(self):
        return []

    def __getitem__(self, k):
        raise EmptyZipError(
            "The store is empty: ZipStore(zip_filepath={self.zip_filepath})"
        )

    def open(self, *args, **kwargs):
        raise EmptyZipError(
            f"The zip file doesn't exist yet! Nothing was written in it: {self.zip_filepath}"
        )
        #
        # class OpenedNotExistingFile:
        #     zip_filepath = self.zip_filepath
        #
        #     def read(self):
        #         raise EmptyZipError(
        #             f"The zip file doesn't exist yet! Nothing was written in it: {
        #             self.zip_filepath}")
        #
        #     def __enter__(self, ):
        #         return self
        #
        #     def __exit__(self, *exc):
        #         return False
        #
        # return OpenedNotExistingFile()


# TODO: Revise ZipReader and ZipFilesReader architecture and make ZipStore be a subclass of
#  Reader if poss
# TODO: What if I just want to zip a (single) file. What does dol offer for that?
# TODO: How about set_obj (in misc.py)? Make it recognize the .zip extension and subextension (
#  e.g. .txt.zip) serialize
class ZipStore(KvPersister):
    """Zip read and writing.
    When you want to read zips, there's the `FilesOfZip`, `ZipReader`, or `ZipFilesReader` we
    know and love.

    Sometimes though, you want to write to zips too. For this, we have `ZipStore`.

    Since ZipStore can write to a zip, it's read functionality is not going to assume static data,
    and cache things, as your favorite zip readers did.
    This, and the acrobatics need to disguise the weird zipfile into something more... key-value
    natural,
    makes for a not so efficient store, out of the box.

    I advise using one of the zip readers if all you need to do is read, or subclassing or
     wrapping ZipStore with caching layers if it is appropriate to you.

    Let's verify that a ZipStore can indeed write data. First, we'll set things up!

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

    Okay, test_zipfile doesn't exist (but will soon...)

    >>> z = ZipStore(test_zipfile)

    See that the file still doesn't exist (it will only be created when we start writing)

    >>> assert not os.path.isfile(test_zipfile)
    >>> list(z)  # z "is" empty (which makes sense?)
    []

    Now let's write something interesting (notice, it has to be in bytes):

    >>> z['foo'] = b'bar'
    >>> list(z)  # now we have something in z
    ['foo']
    >>> z['foo']  # and that thing is what we put there
    b'bar'

    And indeed we have a zip file now:

    >>> assert os.path.isfile(test_zipfile)

    """

    _zipfile_init_kw = dict(
        compression=DFLT_COMPRESSION,
        allowZip64=True,
        compresslevel=None,
        strict_timestamps=True,
    )
    _open_kw = dict(pwd=None, force_zip64=False)
    _writestr_kw = dict(compress_type=None, compresslevel=None)
    zip_writer = None

    # @wraps(ZipReader.__init__)
    def __init__(
        self,
        zip_filepath,
        compression=DFLT_COMPRESSION,
        allow_overwrites=True,
        pwd=None,
    ):
        self.zip_filepath = fullpath(zip_filepath)
        self.zip_filepath = zip_filepath
        self.zip_writer_opened = False
        self.allow_overwrites = allow_overwrites
        self._zipfile_init_kw = dict(self._zipfile_init_kw, compression=compression)
        self._open_kw = dict(self._open_kw, pwd=pwd)

    @staticmethod
    def files_only_filt(fileinfo):
        return not fileinfo.is_dir()

    @property
    def zip_reader(self):
        if os.path.isfile(self.zip_filepath):
            return ZipFile(self.zip_filepath, mode="r", **self._zipfile_init_kw)
        else:
            return _EmptyZipReader(self.zip_filepath)

    def __iter__(self):
        # using zip_file.infolist(), we could also filter for info (like directory/file)
        yield from (
            fi.filename for fi in self.zip_reader.infolist() if self.files_only_filt(fi)
        )

    def __getitem__(self, k):
        with self.zip_reader.open(k, **dict(self._open_kw, mode="r")) as fp:
            return fp.read()

    def __repr__(self):
        args_str = ", ".join(
            (
                f"'{self.zip_filepath}'",
                f"'allow_overwrites={self.allow_overwrites}'",
            )
        )
        return f"{self.__class__.__name__}({args_str})"

    def __contains__(self, k):
        try:
            with self.zip_reader.open(k, **dict(self._open_kw, mode="r")) as fp:
                pass
            return True
        except (
            KeyError,
            BadZipFile,
        ):  # BadZipFile is to catch when zip file exists, but is empty.
            return False

    # # TODO: Find better way to avoid duplicate keys!
    # # TODO: What's the right Error to raise
    # def _assert_non_existing_key(self, k):
    #     # if self.zip_writer is not None:
    #     if not self.zip_writer_opened:
    #         try:
    #             self.zip_reader.open(k)
    #             raise OverwriteNotAllowed(f"You're not allowed to overwrite an existing key: {k}")
    #         except KeyError as e:
    #             if isinstance(e, EmptyZipError) or e.args[-1].endswith('archive'):
    #                 pass  #
    #             else:
    #                 raise OverwriteNotAllowed(f"You're not allowed to overwrite an existing
    #                 key: {k}")

    # TODO: Repeated with zip_writer logic. Consider DRY possibilities.
    def __setitem__(self, k, v):
        if k in self:
            if self.allow_overwrites and not self.zip_writer_opened:
                del self[k]  # remove key so it can be overwritten
            else:
                if self.zip_writer_opened:
                    raise OverwriteNotAllowed(
                        f"When using the context mode, you're not allowed to overwrite an "
                        f"existing key: {k}"
                    )
                else:
                    raise OverwriteNotAllowed(
                        f"You're not allowed to overwrite an existing key: {k}"
                    )

        if self.zip_writer_opened:
            with self.zip_writer.open(k, **dict(self._open_kw, mode="w")) as fp:
                return fp.write(v)
        else:
            with ZipFile(
                self.zip_filepath, mode="a", **self._zipfile_init_kw
            ) as zip_writer:
                with zip_writer.open(k, **dict(self._open_kw, mode="w")) as fp:
                    return fp.write(v)

    def __delitem__(self, k):
        try:
            os.system(f"zip -d {self.zip_filepath} {k}")
        except Exception as e:
            raise KeyError(f"{e.__class__}: {e.args}")
        # raise NotImplementedError("zipfile, the backend of ZipStore, doesn't support deletion,
        # so neither will we.")

    def open(self):
        self.zip_writer = ZipFile(self.zip_filepath, mode="a", **self._zipfile_init_kw)
        self.zip_writer_opened = True
        return self

    def close(self):
        if self.zip_writer is not None:
            self.zip_writer.close()
        self.zip_writer_opened = False

    __enter__ = open

    def __exit__(self, *exc):
        self.close()
        return False


PathString = str
PathFilterFunc = Callable[[PathString], bool]


def _not_in(excluded, obj=None):
    if obj is None:
        return partial(_not_in, excluded)
    return obj not in excluded


def remove_some_entries_from_zip(
    zip_source,
    keys_to_be_removed: Union[PathFilterFunc, Iterable[PathString]],
    ask_before_before_deleting=True,
    *,
    remove_action: Literal["delete", "filter"] = "filter",
):
    """Removes specific keys from a zip file.

    :param zip_source: zip filepath, bytes, or whatever a ``ZipStore`` can take
    :param keys_to_be_removed: An iterable of keys or a boolean filter function
    :param ask_before_before_deleting: True (default) if the user should be
        presented with the keys first, and asked permission to delete.
    :return: The ZipStore (in case you want to do further work with it)

    Tip: If you want to delete with no questions asked, use currying:

    >>> from functools import partial
    >>> rm_keys_without_asking = partial(
    ...     remove_some_entries_from_zip,
    ...     ask_before_before_deleting=False
    ... )

    """
    z = zip_source
    if not isinstance(z, Mapping):
        z = ZipStore(z)
    if not isinstance(keys_to_be_removed, Callable):
        if isinstance(keys_to_be_removed, str):
            keys_to_be_removed = [keys_to_be_removed]
        assert isinstance(keys_to_be_removed, Iterable)
        keys_to_be_removed = lambda x: x in set(keys_to_be_removed)
    keys_that_will_be_deleted = list(filter(keys_to_be_removed, z))
    if keys_that_will_be_deleted:
        if remove_action == "delete":
            if ask_before_before_deleting:
                print("These keys will be removed:\n\r")
                print(*keys_that_will_be_deleted, sep="\n")
                n = len(keys_that_will_be_deleted)
                answer = input(f"\nShould I go ahead and delete these {n} keys? (y/N)")
                if not answer == "y":
                    print("Okay, I will NOT delete these.")
                    return
            for k in keys_that_will_be_deleted:
                del z[k]
        else:  # remove_action == 'filter'
            z = filt_iter(z, filt=_not_in(keys_that_will_be_deleted))

    return z


from dol.util import not_a_mac_junk_path


def is_a_mac_junk_path(path):
    return not not_a_mac_junk_path(path)


remove_mac_junk_from_zip = partial(
    remove_some_entries_from_zip,
    keys_to_be_removed=is_a_mac_junk_path,
    ask_before_before_deleting=False,
)
remove_mac_junk_from_zip.__doc__ = "Removes mac junk keys from zip"

# TODO: The way prefix and file_info_filt is handled is not efficient
# TODO: prefix is silly: less general than filename_filt would be, and not even producing
#  relative paths
#  (especially when getitem returns subdirs)


# trans alternative:
# from dol.trans import mk_kv_reader_from_kv_collection, wrap_kvs
#
# ZipFileReader = wrap_kvs(mk_kv_reader_from_kv_collection(FileCollection, name='_ZipFileReader'),
#                          name='ZipFileReader',
#                          obj_of_data=ZipReader)


# ----------------------------- Extras -------------------------------------------------


def tar_compress(data_bytes, file_name="data.bin"):
    import tarfile
    import io

    with io.BytesIO() as tar_buffer:
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            data_file = io.BytesIO(data_bytes)
            tarinfo = tarfile.TarInfo(name=file_name)
            tarinfo.size = len(data_bytes)
            tar.addfile(tarinfo, fileobj=data_file)
        return tar_buffer.getvalue()


def tar_decompress(tar_bytes):
    import tarfile
    import io

    with io.BytesIO(tar_bytes) as tar_buffer:
        with tarfile.open(fileobj=tar_buffer, mode="r:") as tar:
            for member in tar.getmembers():
                extracted_file = tar.extractfile(member)
                if extracted_file:
                    return extracted_file.read()
    return None
