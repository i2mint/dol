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


# TODO: On (my) pycharm IDE, these show up greyed out and grey out all
#  of the base objects when included (pycharm bug?), so separating them out.
from dol.base import KT, VT, BaseKeysView, BaseValuesView, BaseItemsView


# TODO: Check usage and replace star import with explicit imports
from dol.zipfiledol import *

from dol.filesys import (
    Files,  # read-write-delete access to files; relative paths, bytes values
    FilesReader,  # read-only version of LocalFiles,
    TextFiles,  # read-write-delete access to text files; relative paths, str values
    ensure_dir,  # function to create a directory, if missing
    mk_dirs_if_missing,  # store deco to create directories on write, when missing
    MakeMissingDirsStoreMixin,  # Mixin to enable auto-dir-making on write
    resolve_path,  # to get a full path (resolve ~ and .),
    resolve_dir,  # to get a full path (resolve ~ and .) and ensure it is a directory
    DirReader,  # recursive read-only access to directories
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
)

from dol.trans import (
    wrap_kvs,  # transform store key and/or value
    cached_keys,  # cache store keys
    filt_iter,  # filter store keys
    add_ipython_key_completions,  # add ipython key completions
    insert_hash_method,  # add a hash method to store
    add_path_get,  # add a path_get method to store
    add_path_access,  # add path_get and path_set methods to store
    flatten,  # flatten a nested store
    kv_wrap,  # different interface to wrap_kvs
    disable_delitem,  # disable ability to delete
    disable_setitem,  # disable ability to write to a store
    mk_read_only,  # disable ability to write to a store or delete its keys
    insert_aliases,  # insert aliases for store methods
    cache_iter,  # being deprecated
)

from dol.caching import (
    WriteBackChainMap,  # write-back cache
    mk_cached_store,  # wrap a store so it uses a cache
    store_cached,  # func memorizer using a specific store as its "memory"
    store_cached_with_single_key,
    ensure_clear_to_kv_store,  # add a clear method to a store (removed by default)
    flush_on_exit,  # make a store become a context manager that flushes cache on exit
    mk_write_cached_store,
)

from dol.appendable import appendable

from dol.naming import (
    StrTupleDict,  # convert from and to strings, tuples, and dicts.
    mk_store_from_path_format_store_cls,
)

from dol.paths import (
    mk_relative_path_store,  # transform path store into relative path store
    KeyPath,  # a class to represent a path to a key
    path_get,  # get a value from a path
    path_set,  # set a value from a path
    path_filter,  # search through paths of a Mapping
)

from dol.explicit import ExplicitKeyMap, invertible_maps


from dol.sources import FlatReader, SequenceKvReader, FuncReader, Attrs
