"""Testing base.py objects"""

from dol import MappingViewMixin, BaseKeysView, BaseValuesView, BaseItemsView


def test_mapping_views():
    class WrappedDict(MappingViewMixin, dict):
        # you can log method calls
        class KeysView(BaseKeysView):
            def __iter__(self):
                print('A KeysView is being iterated on...')
                return super().__iter__()

        # You can add functionality:
        class ValuesView(BaseValuesView):
            def distinct(self):
                return set(self._mapping.values())

        # you can modify existing functionality:
        class ItemsView(BaseItemsView):
            """Just like BaseKeysView, but yields the [key,val] pairs as lists instead of tuples"""

            def __iter__(self):
                return map(list, super().__iter__())

    wd = WrappedDict(a=3, b=1, c=3)

    def assert_store_functionality(s):
        assert list(s) == ['a', 'b', 'c']
        assert list(s.values()) == [3, 1, 3]
        assert list(s.keys()) == ['a', 'b', 'c']
        assert isinstance(s.values().distinct(), set)
        assert sorted(s.values().distinct()) == [1, 3]
        assert list(s.items()) == [['a', 3], ['b', 1], ['c', 3]]

    assert_store_functionality(wd)

    from dol import Store

    # testing wrapping an instance:
    wwd = Store.wrap(wd)
    assert_store_functionality(wwd)
    WWD = Store.wrap(WrappedDict)
    wwd = WWD(a=3, b=1, c=3)
    assert_store_functionality(wwd)

    from dol.trans import wrap_kvs, filt_iter, cached_keys

    wwd = wrap_kvs(
        wd,
        key_of_id=lambda x: x.upper(),
        id_of_key=lambda x: x.lower(),
        obj_of_data=lambda x: x * 10,
        data_of_obj=lambda x: x // 10,
    )
    assert list(wwd) == ['A', 'B', 'C']
    assert list(wwd.keys()) == ['A', 'B', 'C']
    assert list(wwd.values()) == [30, 10, 30]
    assert sorted(wwd.values().distinct()) == [10, 30]
    assert list(wwd.items()) == [['A', 30], ['B', 10], ['C', 30]]

    wwd = filt_iter(wd, filt=lambda k: k in {'a', 'c'})
    assert list(wwd) == ['a', 'c']
    assert list(wwd.values()) == [3, 3]
    assert list(wwd.keys()) == ['a', 'c']
    assert sorted(wwd.values().distinct()) == [3]
    assert list(wwd.items()) == [['a', 3], ['c', 3]]

    wwd = cached_keys(wd, keys_cache=set)
    assert wwd._keys_cache == {'a', 'b', 'c'}  # the wrapped store has a cache
    # But other things still work
    # Note: Using sorted instead of list here, to not be sensitive to ordering differences that occur with set.
    assert sorted(wwd) == ['a', 'b', 'c']
    assert sorted(wwd.values()) == [
        1,
        3,
        3,
    ]  # different than in assert_store_functionality
    assert sorted(wwd.keys()) == ['a', 'b', 'c']
    assert isinstance(wwd.values().distinct(), set)
    assert sorted(wwd.values().distinct()) == [1, 3]
    assert sorted(wwd.items()) == [['a', 3], ['b', 1], ['c', 3]]
