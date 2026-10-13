"""Tests for the introspection attributes of a class wrapper (dol#131):
``_wrapped_class``, ``_wrapper_class`` and ``_delegation_attr``."""

import pickle

from dol import Store, wrap_kvs, filt_iter, KeyCodecs


def test_wrap_kvs_exposes_wrapped_class():
    class D(dict):
        pass

    W = wrap_kvs(D, key_of_id=str.lower, id_of_key=str.upper)
    assert W._wrapped_class is D
    assert W._delegation_attr == "store"
    assert issubclass(W, W._wrapper_class)
    assert isinstance(W().store, W._wrapped_class)


def test_store_wrap_exposes_attrs():
    W = Store.wrap(dict)
    assert W._wrapped_class is dict
    assert W._wrapper_class is Store
    assert W._delegation_attr == "store"


def test_nested_wraps_are_not_shadowed():
    """The inner wrapper's copies must not be delegated over the outer's."""
    inner = wrap_kvs(dict, key_of_id=str.lower, id_of_key=str.upper)
    outer = filt_iter(inner, filt=lambda k: True)
    assert outer._wrapped_class is inner
    assert outer._delegation_attr == "store"
    deeper = wrap_kvs(inner, obj_of_data=int, data_of_obj=str)
    assert deeper._wrapped_class is inner
    assert deeper._wrapped_class is not dict


def test_class_decorators_keep_attrs():
    for deco in (
        filt_iter(filt=lambda k: True),
        KeyCodecs.suffixed(".x"),
    ):
        W = deco(dict)
        assert W._wrapped_class is dict


def test_pickle_roundtrip_unaffected():
    W = wrap_kvs(dict, key_of_id=str.lower, id_of_key=str.upper)
    w = W()
    w["a"] = 1
    w2 = pickle.loads(pickle.dumps(w))
    assert w2._wrapped_class is dict
