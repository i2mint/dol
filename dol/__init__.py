"""Core tools to build simple interfaces to complex data sources and bend the interface to your will (and need)"""

import os

# from contextlib import suppress

file_sep = os.path.sep


def kvhead(store, n=1):
    """Get the first item of a kv store, or a list of the first n items"""
    if n == 1:
        for k in store:
            return k, store[k]
    else:
        return [(k, store[k]) for i, k in enumerate(store) if i < n]


def ihead(store, n=1):
    """Get the first item of an iterable, or a list of the first n items"""
    if n == 1:
        for item in iter(store):
            return item
    else:
        return [item for i, item in enumerate(store) if i < n]


# from dol.base import (
#     Collection,  # base class for collections (adds to collections.abc.Collection)
#     MappingViewMixin,
#     KvReader,  # base class for kv readers (adds to collections.abc.Mapping)
#     KvPersister,  # base for kv persisters (adds to collections.abc.MutableMapping)
#     Reader,  # TODO: deprecate? (now KvReader)
#     Persister,  # TODO: deprecate? (now KvPersister)
#     kv_walk,  # walk a kv store
#     Store,  # base class for stores (adds hooks for key and value transforms)
#     BaseKeysView,  # base class for keys views
#     BaseValuesView,  # base class for values views
#     BaseItemsView,  # base class for items views
#     KT,  # Key type,
#     VT,  # Value type
# )


from dol.tools import (
    cache_this,  # cache the result of "property" methods in a store
    add_extension,  # a helper (for cache_this) to make key functions
    lru_cache_method,  # A decorator to cache the result of a method, ignoring the first argument
    store_aggregate,  # aggregate stores keys and values into an aggregate object (e.g. string concatenation)
)

from dol.kv_codecs import ValueCodecs, KeyCodecs

from dol.base import (
    Collection,  # base class for collections (adds to collections.abc.Collection)
    MappingViewMixin,
    KvReader,  # base class for kv readers (adds to collections.abc.Mapping)
    KvPersister,  # base for kv persisters (adds to collections.abc.MutableMapping)
    Reader,  # TODO: deprecate? (now KvReader)
    Persister,  # TODO: deprecate? (now KvPersister)
    kv_walk,  # walk a kv store
    Store,  # base class for stores (adds hooks for key and value transforms)
)


from dol.base import KT, VT, BaseKeysView, BaseValuesView, BaseItemsView


from dol.zipfiledol import (
    zip_compress,
    zip_decompress,
    to_zip_file,
    ZipReader,
    ZipInfoReader,
    FilesOfZip,
    FileStreamsOfZip,
    FlatZipFilesReader,
    ZipStore,
    ZipFileStreamsReader,
    remove_mac_junk_from_zip,
    tar_compress,
    tar_decompress,
)

from dol.filesys import (
    Files,  # read-write-delete access to files; relative paths, bytes values
    FilesReader,  # read-only version of LocalFiles,
    TextFiles,  # read-write-delete access to text files; relative paths, str values
    ensure_dir,  # function to create a directory, if missing
    mk_dirs_if_missing,  # store deco to create directories on write, when missing
    MakeMissingDirsStoreMixin,  # Mixin to enable auto-dir-making on write
    resolve_path,  # to get a full path (resolve ~ and .),
    resolve_dir,  # to get a full path (resolve ~ and .) and ensure it is a directory
    DirReader,  # recursive read-only access to directories,
    temp_dir,  # make a temporary directory,
    PickleFiles,  # CRUD access to pickled files
    JsonFiles,  # CRUD access to jsob files,
    Jsons,  # Same as JsonFiles, but with added .json extension handling
    create_directories,
    process_path,
    subfolder_stores,  # a store of stores, each store corresponding to a subfolder
)

from dol.util import (
    Pipe,  # chain functions
    lazyprop,  # lazy evaluation of properties
    partialclass,  # partial class instantiation
    groupby,  # group items according to group keys
    regroupby,  # recursive version of groupby
    igroupby,
    not_a_mac_junk_path,  # filter function to filter out mac junk paths
    instance_checker,  # make filter function that checks the type of an object
    chain_get,  # a function to perform chained get operations (i.e. path keys get)
    written_bytes,  # transform a file-writing function into a bytes-writing function
    written_key,  # writes an object to a key and returns the key.
    read_from_bytes,  # transform a file-reading function into a bytes-reading function
)

from dol.trans import (
    wrap_kvs,  # transform store key and/or value
    filt_iter,  # filter store keys (and contains ready to use filters as attributes)
    cached_keys,  # cache store keys
    add_decoder,  # add a decoder (i.e. outcomming value transformer) to a store
    add_ipython_key_completions,  # add ipython key completions
    insert_hash_method,  # add a hash method to store
    add_path_get,  # add a path_get method to store
    add_path_access,  # add path_get and path_set methods to store
    flatten,  # flatten a nested store
    kv_wrap,  # different interface to wrap_kvs
    disable_delitem,  # disable ability to delete
    disable_setitem,  # disable ability to write to a store
    mk_read_only,  # disable ability to write to a store or delete its keys
    add_aliases,  # delegation-wrap any object and add aliases for its methods
    insert_aliases,  # insert aliases for special (dunder) store methods,
    add_missing_key_handling,  # add a missing key handler to a store
    cache_iter,  # being deprecated
    store_decorator,  # Helper to make store decorators
    redirect_getattr_to_getitem,  # redirect attribute access to __getitem__
)

from dol.caching import (
    WriteBackChainMap,  # write-back cache
    mk_cached_store,  # (old alias of cache_vals) wrap a store so it uses a cache
    cache_vals,  # wrap a store so it uses a cache
    store_cached,  # func memorizer using a specific store as its "memory"
    store_cached_with_single_key,
    ensure_clear_to_kv_store,  # add a clear method to a store (removed by default)
    flush_on_exit,  # make a store become a context manager that flushes cache on exit
    mk_write_cached_store,
)

from dol.appendable import mk_item2kv_for, appendable

from dol.naming import (
    StrTupleDict,  # convert from and to strings, tuples, and dicts.
    mk_store_from_path_format_store_cls,
)

from dol.paths import (
    flatten_dict,  # flatten a nested Mapping, getting a dict
    leaf_paths,  # get the paths to the leaves of a Mapping
    KeyTemplate,  # express strings, tuples, and dict keys from a string template
    mk_relative_path_store,  # transform path store into relative path store
    KeyPath,  # a class to represent a path to a key
    paths_getter,  # to make mapping extractors that use path_get
    path_get,  # get a value from a path
    path_set,  # set a value from a path
    path_filter,  # search through paths of a Mapping
    add_prefix_filtering,  # add a prefix filtering method to a store
    # PathMappedData,  # A mapping that extracts data from a mapping according to paths
)

from dol.dig import trace_getitem  # trace getitem calls, stepping through the layers

from dol.explicit import ExplicitKeyMap, invertible_maps, KeysReader


from dol.sources import (
    FlatReader,
    SequenceKvReader,
    FuncReader,
    Attrs,
    ObjReader,
    FanoutReader,
    FanoutPersister,
    CascadedStores,  # multi-store writes to all stores and reads from first store.
)
