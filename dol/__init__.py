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


from dol.util import lazyprop, partialclass, groupby, regroupby, igroupby

from dol.base import (
    Collection,
    KvReader,
    KvPersister,
    Reader,
    Persister,
    kv_walk,
    Store,
)

from dol.trans import (
    wrap_kvs,
    disable_delitem,
    disable_setitem,
    mk_read_only,
    kv_wrap,
    cached_keys,
    filt_iter,
    add_path_get,
    insert_aliases,
    add_ipython_key_completions,
    cache_iter,  # being deprecated
)

from dol.caching import (
    WriteBackChainMap,
    mk_cached_store,
    store_cached,
    store_cached_with_single_key,
    ensure_clear_to_kv_store,
    flush_on_exit,
    mk_write_cached_store,
)

from dol.appendable import appendable

from dol.naming import StrTupleDict, mk_store_from_path_format_store_cls
from dol.paths import mk_relative_path_store
