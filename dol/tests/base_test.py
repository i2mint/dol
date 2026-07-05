"""Testing base.py objects"""

import math
import pickle
from typing import KT, VT, Tuple
from collections.abc import Iterable
import pytest
from dol import (
    MappingViewMixin,
    Store,
    wrap_kvs,
    filt_iter,
    cached_keys,
    wrapped_self,
)
from dol.base import BaseItemsView, BaseKeysView, BaseValuesView
from dol.trans import take_everything


class WrappedDict(MappingViewMixin, dict):
    keys_iterated = False

    # you can modify the mapping object
    class KeysView(BaseKeysView):
        def __iter__(self) -> Iterable[KT]:
            self._mapping.keys_iterated = True
            return super().__iter__()

    # You can add functionality:
    class ValuesView(BaseValuesView):
        def distinct(self) -> Iterable[VT]:
            return set(super().__iter__())

    # you can modify existing functionality:
    class ItemsView(BaseItemsView):
        """Just like BaseKeysView, but yields the [key,val] pairs as lists instead of tuples"""

        def __iter__(self) -> Iterable[tuple[KT, VT]]:
            return map(list, super().__iter__())


@pytest.mark.parametrize(
    "source_dict, key_input_mapper, key_output_mapper, value_input_mapper, value_output_mapper, postget, key_filter",
    [
        ({"a": 1, "b": 2, "c": 3}, None, None, None, None, None, None),
        (
            {"a": 3, "b": 1, "c": 3},  # source_dict
            lambda k: k.lower(),  # key_input_mapper
            lambda k: k.upper(),  # key_output_mapper
            lambda v: v // 10,  # value_input_mapper
            lambda v: v * 10,  # value_output_mapper
            lambda k, v: f"{k}{v}",  # postget
            lambda k: k in {"a", "c"},  # key_filter
        ),
    ],
)
def test_mapping_views(
    source_dict,
    key_input_mapper,
    key_output_mapper,
    value_input_mapper,
    value_output_mapper,
    postget,
    key_filter,
):
    def assert_store_functionality(
        store,
        key_output_mapper=None,
        value_output_mapper=None,
        postget=None,
        key_filter=None,
        collection=list,
    ):
        key_output_mapper = key_output_mapper or (lambda k: k)
        value_output_mapper = value_output_mapper or (lambda v: v)
        postget = postget or (lambda k, v: v)
        key_filter = key_filter or (lambda k: True)
        assert collection(store) == collection(
            [key_output_mapper(k) for k in source_dict if key_filter(k)]
        )
        assert not store.keys_iterated
        assert collection(store.keys()) == collection(
            [key_output_mapper(k) for k in source_dict.keys() if key_filter(k)]
        )
        assert store.keys_iterated
        assert collection(store.values()) == collection(
            [
                postget(key_output_mapper(k), value_output_mapper(v))
                for k, v in source_dict.items()
                if key_filter(k)
            ]
        )
        assert sorted(store.values().distinct()) == sorted(
            {
                postget(key_output_mapper(k), value_output_mapper(v))
                for k, v in source_dict.items()
                if key_filter(k)
            }
        )
        assert collection(store.items()) == collection(
            [
                [
                    key_output_mapper(k),
                    postget(key_output_mapper(k), value_output_mapper(v)),
                ]
                for k, v in source_dict.items()
                if key_filter(k)
            ]
        )

    wd = WrappedDict(**source_dict)
    assert_store_functionality(wd)

    wwd = Store.wrap(WrappedDict(**source_dict))
    assert_store_functionality(wwd)

    WWD = Store.wrap(WrappedDict)
    wwd = WWD(**source_dict)
    assert_store_functionality(wwd)

    wwd = wrap_kvs(
        WrappedDict(**source_dict),
        id_of_key=key_input_mapper,
        key_of_id=key_output_mapper,
        data_of_obj=value_input_mapper,
        obj_of_data=value_output_mapper,
        postget=postget,
    )
    assert_store_functionality(
        wwd,
        key_output_mapper=key_output_mapper,
        value_output_mapper=value_output_mapper,
        postget=postget,
    )

    wwd = filt_iter(WrappedDict(**source_dict), filt=key_filter or take_everything)
    assert_store_functionality(wwd, key_filter=key_filter)

    wwd = cached_keys(WrappedDict(**source_dict), keys_cache=set)
    assert wwd._keys_cache == set(source_dict)
    assert isinstance(wwd.values().distinct(), set)
    assert_store_functionality(wwd, collection=sorted)


def test_wrap_kvs_vs_class_and_static_methods():
    """Adding wrap_kvs breaks methods when called from class

    That is, when you call Klass.method() (where method is a normal, class, or static)

    See issue "dol.base.Store.wrap breaks unbound method calls":
    https://github.com/i2mint/dol/issues/17

    """

    @Store.wrap
    class MyFiles:
        y = 2

        def normal_method(self, x=3):
            return self.y * x

        @classmethod
        def hello(cls):
            pass

        @staticmethod
        def hi():
            pass

    errors = []

    # This works fine!
    instance = MyFiles()
    assert instance.normal_method() == 6

    # But calling the method as a class...
    try:
        MyFiles.normal_method(instance)
    except Exception as e:
        print("method normal_method is broken by wrap_kvs decorator")
        print(f"{type(e).__name__}: {e}")
        errors.append(e)

    try:
        MyFiles.hello()
    except Exception as e:
        print("classmethod hello is broken by wrap_kvs decorator")
        print(f"{type(e).__name__}: {e}")
        errors.append(e)

    try:
        MyFiles.hi()
    except Exception as e:
        print("staticmethod hi is broken by wrap_kvs decorator")
        print(f"{type(e).__name__}: {e}")
        errors.append(e)

    if errors:
        first_error, *_ = errors
        raise first_error


# ---------------------------------------------------------------------------------------
# wrapped_self / Issue #18 ("self is the unwrapped inner store" in delegation-wrapped
# classes). See misc/docs/dol_issue18_design.md.
#
# Module-level named transforms + a non-rebound base class so the wrapped store is
# picklable (lambdas and name-rebound classes are not — pre-existing dol limitations).

def _square(x):
    return x * x


def _sqrt(x):
    return math.sqrt(x)


_square_wrap = wrap_kvs(data_of_obj=_square, obj_of_data=_sqrt)


class _SquareStoreBase(dict):
    """A store whose stored values are the square of the value written."""

    def via_self(self, k):
        return self[k]  # self is the INNER store -> Issue #18: NOT transformed

    def via_wrapped_self(self, k):
        return wrapped_self(self)[k]  # outer store -> transform applied


# NOTE: assign to a NEW name (do not rebind _SquareStoreBase) so the __reduce__ pickle
# path can still resolve the original wrapped class by qualified name.
_SquareStore = _square_wrap(_SquareStoreBase)


def test_issue18_reproduced_and_wrapped_self_fixes_it():
    """The #18 bug is real, and wrapped_self(self)[k] recovers the transformed value."""
    s = _SquareStore()
    s["2"] = 2
    assert s["2"] == 2.0  # external access is transformed (sqrt of stored 4)
    assert s.via_self("2") == 4  # Issue #18: inner-bound self bypasses the transform
    assert s.via_wrapped_self("2") == 2.0  # wrapped_self recovers the transformed value


def test_wrapped_self_climbs_to_outermost_when_nested():
    """Under stacked/Pipe wrapping, wrapped_self returns the OUTERMOST wrapper."""
    outer_wrap = wrap_kvs(obj_of_data=lambda x: x + 100, data_of_obj=lambda x: x - 100)

    @outer_wrap
    @_square_wrap
    class Nested(dict):
        def via_wrapped_self(self, k):
            return wrapped_self(self)[k]

    s = Nested()
    s["2"] = 2
    # via_wrapped_self must match the fully-transformed external read, not a partial one
    assert s.via_wrapped_self("2") == s["2"]


def test_wrapped_self_is_noop_on_direct_store_subclass():
    """A direct Store subclass already has self == the store, so wrapped_self is a no-op."""

    class Direct(Store):
        def _obj_of_data(self, data):
            return data * 10

        def via_wrapped_self(self, k):
            return wrapped_self(self)[k]

    d = Direct()
    d.store["a"] = 5
    assert d["a"] == 50
    assert d.via_wrapped_self("a") == 50  # wrapped_self(self) is self here
    assert wrapped_self(d) is d


def test_wrapped_self_is_noop_on_plain_objects():
    class Plain:
        pass

    p = Plain()
    assert wrapped_self(p) is p
    assert wrapped_self(123) == 123
    assert wrapped_self("hello") == "hello"


def test_wrapped_self_covers_instance_wrap():
    """wrap_kvs(instance) also registers (via Store.__init__), so wrapped_self resolves."""
    inner = {"a": 1}
    s = wrap_kvs(inner, obj_of_data=lambda x: x * 10)
    assert s["a"] == 10
    assert wrapped_self(s.store) is s  # inner store climbs back to the wrapper


def test_wrapped_self_survives_pickle_round_trip():
    """__setstate__ re-registers the backref (unpickling bypasses __init__)."""
    s = _SquareStore()
    s["2"] = 2
    assert s.via_wrapped_self("2") == 2.0

    s2 = pickle.loads(pickle.dumps(s))
    assert s2["2"] == 2.0
    # Without the __setstate__ re-registration this would silently revert to 4 (Issue #18):
    assert s2.via_wrapped_self("2") == 2.0


def test_wrapped_self_registration_is_zero_behavior_change():
    """A store that never calls wrapped_self behaves exactly as before."""
    s = wrap_kvs(dict(), key_of_id=str.upper, id_of_key=str.lower)
    s["foo"] = 1
    s["bar"] = 2
    # key_of_id=str.upper => iterating yields upper-cased keys; id_of_key=str.lower on write
    assert dict(s.items()) == {"FOO": 1, "BAR": 2}
    assert list(s) == ["FOO", "BAR"]
    assert "foo" in s and "nope" not in s  # __contains__ lowercases the key on lookup
    assert len(s) == 2


def test_wrapped_self_copy_copy_preserves_original():
    """A shallow copy shares the inner store; GC of the copy must NOT break the original.

    Regression for the single-slot id-registry clobber found in adversarial review: the
    copy re-registered id(inner) then its finalizer popped the entry, silently reverting the
    still-alive original to raw Issue #18 behavior.
    """
    import copy
    import gc

    s = _SquareStore()
    s["2"] = 2
    c = copy.copy(s)
    assert c.store is s.store  # shallow copy shares the inner store
    assert c.via_wrapped_self("2") == 2.0
    assert s.via_wrapped_self("2") == 2.0
    del c
    gc.collect()
    assert s.via_wrapped_self("2") == 2.0  # would have been 4 before the multi-valued fix


def test_wrapped_self_deepcopy_is_independent():
    import copy

    s = _SquareStore()
    s["2"] = 2
    d = copy.deepcopy(s)
    assert d.store is not s.store  # deepcopy builds a fresh inner
    assert d.via_wrapped_self("2") == 2.0
    assert s.via_wrapped_self("2") == 2.0


def test_wrapped_self_registry_does_not_leak():
    import gc
    from dol.base import _wrapper_backrefs

    gc.collect()
    before = len(_wrapper_backrefs)
    for _ in range(1000):
        t = _SquareStore()
        t["1"] = 1
        assert t.via_wrapped_self("1") == 1.0
    gc.collect()
    after = len(_wrapper_backrefs)
    assert after - before <= 5  # short-lived wrappers' entries are cleaned up


def test_wrapped_self_shared_inner_returns_a_valid_wrapper():
    """Two live wrappers over one shared inner: ambiguous, but never returns the raw inner."""
    base = {}
    a = wrap_kvs(base, obj_of_data=lambda x: x * 2, data_of_obj=lambda x: x // 2)
    b = wrap_kvs(base, obj_of_data=lambda x: x * 3, data_of_obj=lambda x: x // 3)
    resolved = wrapped_self(base)
    assert resolved is a or resolved is b  # a real wrapper...
    assert resolved is not base  # ...never the raw inner
