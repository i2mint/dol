"""Utils for tests."""

from dol import TextFiles
import os
from functools import partial


_dflt_keys = (
    "pluto",
    "planets/mercury",
    "planets/venus",
    "planets/earth",
    "planets/mars",
    "fruit/apple",
    "fruit/banana",
    "fruit/cherry",
)


def mk_test_store_from_keys(
    keys=_dflt_keys,
    *,
    mk_store=dict,
    obj_of_key=lambda k: f"Content of {k}",
    empty_store_before_writing=False,
):
    """Make some test data for a store from a list of keys.

    None of the arguments are required, for the convenience of getting test stores
    quickly:

    >>> store = mk_test_store_from_keys()
    >>> store = mk_test_store_from_keys.for_local()  # makes files in temp local dir
    >>> store = mk_test_store_from_keys(keys=['one', 'two', 'three'])
    """
    if isinstance(mk_store, str):
        mk_store = mk_tmp_local_store(mk_store)
    store = mk_store()
    if empty_store_before_writing:
        for k in store:
            del store[k]
    for k in keys:
        store[k] = obj_of_key(k)
    return store


def mk_tmp_local_store(
    tmp_name="temp_local_store", mk_store=TextFiles, make_dirs_if_missing=True
):
    from dol import temp_dir, mk_dirs_if_missing

    store = mk_store(temp_dir(tmp_name))
    if make_dirs_if_missing:
        store = mk_dirs_if_missing(store)
    return store


mk_test_store_from_keys.for_local = partial(
    mk_test_store_from_keys, mk_store=mk_tmp_local_store
)
