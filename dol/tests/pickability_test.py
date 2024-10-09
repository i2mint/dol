"""
Test the pickability of stores when they're wrapped.
"""

import pytest
import pickle
from functools import partial

from dol.base import Store
from dol.trans import wrap_kvs, filt_iter, cached_keys

# TODO: Make it work


def test_pickling_w_dict():
    """To show that a dict pickles and unpickles just fine!"""
    s = {"a": 1, "b": 2}
    assert_dict_of_unpickled_is_the_same(s)


def test_pickling_w_simple_store():
    s = Store({"a": 1, "b": 2})
    assert_dict_of_unpickled_is_the_same(s)


def test_pickling_with_store_wrap():
    D = Store.wrap(dict)
    d = {"a": 1, "b": 2}
    s = D(d)
    b = pickle.dumps(s)
    ss = pickle.loads(b)
    assert dict(s) == dict(ss)


def test_pickling_with_wrap_kvs_class():
    WrappedDict = wrap_kvs(key_of_id=add_tag, id_of_key=remove_tag)(dict)
    s = WrappedDict({"a": 1, "b": 2})
    assert_dict_of_unpickled_is_the_same(s)


# @pytest.mark.xfail
def test_pickling_with_wrap_kvs_instance():
    d = {"a": 1, "b": 2}
    s = wrap_kvs(d, key_of_id=add_tag, id_of_key=remove_tag)
    assert_dict_of_unpickled_is_the_same(s)


def test_pickling_with_filt_iter_class():
    filt_func = partial(is_below_max_len, max_len=3)
    WrappedDict = filt_iter(dict, filt=filt_func)
    s = WrappedDict({"a": 1, "bb": 2, "ccc": 3})
    assert dict(s) == {"a": 1, "bb": 2}
    assert_dict_of_unpickled_is_the_same(s)


def test_pickling_with_filt_iter_instance():
    d = {"a": 1, "bb": 2, "ccc": 3}
    filt_func = partial(is_below_max_len, max_len=3)
    s = filt_iter(d, filt=filt_func)
    assert dict(s) == {"a": 1, "bb": 2}
    assert_dict_of_unpickled_is_the_same(s)


# @pytest.mark.xfail
def test_pickling_with_cached_keys_class():
    WrappedDict = cached_keys(dict, keys_cache=sorted)
    s = WrappedDict({"b": 2, "a": 1})  # Note: b comes before a here
    assert list(s) == ["a", "b"]  # but here, things are sorted
    # assert list(dict(s)) == ['a', 'b']  # TODO: This fails! Why?
    assert list(dict(s.items())) == ["a", "b"]  # ... yet this one sees the cache
    assert dict(s.items()) == {"a": 1, "b": 2}
    assert_dict_of_unpickled_is_the_same(s)


def test_pickling_with_cached_keys_instance():
    d = {"b": 2, "a": 1}  # Note: b comes before a here
    s = cached_keys(d, keys_cache=sorted)
    assert list(s) == ["a", "b"]  # but here, things are sorted
    # assert list(dict(s)) == ['a', 'b']  # TODO: This fails! Why?
    assert list(dict(s.items())) == ["a", "b"]  # ... yet this one sees the cache
    assert dict(s.items()) == {"a": 1, "b": 2}
    assert_dict_of_unpickled_is_the_same(s)


# ------------------------ utils -------------------------------------------------------------------

pup = lambda obj: pickle.loads(pickle.dumps(obj))


def assert_dict_of_unpickled_is_the_same(original_obj):
    pickled = pickle._dumps(original_obj)
    unpickled = pickle.loads(pickled)
    assert dict(unpickled) == dict(original_obj)


def add_tag(k):
    return k + "__tag"


def remove_tag(k):
    assert k.endswith("__tag")
    return k[: -len("__tag")]


def is_below_max_len(x, max_len=3):
    return len(x) < max_len
