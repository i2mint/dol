"""Testing base.py objects"""

from typing import Iterable, KT, VT, Tuple
import pytest
from dol import (
    MappingViewMixin,
    Store,
    wrap_kvs,
    filt_iter,
    cached_keys,
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

        def __iter__(self) -> Iterable[Tuple[KT, VT]]:
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
