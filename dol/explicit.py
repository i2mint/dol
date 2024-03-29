"""
utils to make stores based on a the input data itself
"""
from collections.abc import Mapping
from typing import Callable, Collection as CollectionType

from dol.base import Collection, KvReader, Store
from dol.trans import kv_wrap
from dol.paths import PrefixRelativizationMixin
from dol.util import max_common_prefix
from dol.sources import ObjReader  # because it used to be here


# TODO: Revisit ExplicitKeys and ExplicitKeysWithPrefixRelativization. Not extendible to full store!
class ExplicitKeys(Collection):
    """
    dol.base.Keys implementation that gets it's keys explicitly from a collection given
    at initialization time.
    The key_collection must be a collections.abc.Collection
    (such as list, tuple, set, etc.)

    >>> keys = ExplicitKeys(key_collection=['foo', 'bar', 'alice'])
    >>> 'foo' in keys
    True
    >>> 'not there' in keys
    False
    >>> list(keys)
    ['foo', 'bar', 'alice']
    """

    __slots__ = ('_keys_cache',)

    def __init__(
        self, key_collection: CollectionType
    ):  # don't remove this init: Don't. Need for _keys_cache init
        assert isinstance(key_collection, CollectionType), (
            'key_collection must be a collections.abc.Collection, i.e. have a __len__, __contains__, and __len__.'
            'The key_collection you gave me was a {}'.format(type(key_collection))
        )
        # self._key_collection = key_collection
        self._keys_cache = key_collection

    def __iter__(self):
        yield from self._keys_cache

    def __len__(self):
        return len(self._keys_cache)

    def __contains__(self, k):
        return k in self._keys_cache


# TODO: Should we deprecate or replace with recipe?
class ExplicitKeysSource(ExplicitKeys, ObjReader, KvReader):
    """
    An object source that uses an explicit keys collection and a specified function to
    read contents for a key.

    >>> s = ExplicitKeysSource([1, 2, 3], str)
    >>> list(s)
    [1, 2, 3]
    >>> list(s.values())
    ['1', '2', '3']

    Main functionality equivalent to recipe:

    >>> def explicit_keys_source(key_collection, _obj_of_key):
    ...     from dol.trans import wrap_kvs
    ...     return wrap_kvs({k: k for k in key_collection}, obj_of_data=_obj_of_key)

    >>> s = explicit_keys_source([1, 2, 3], str)
    >>> list(s)
    [1, 2, 3]
    >>> list(s.values())
    ['1', '2', '3']

    """

    def __init__(self, key_collection: CollectionType, _obj_of_key: Callable):
        """

        :param key_collection: The collection of keys that this source handles
        :param _obj_of_key: The function that returns the contents for a key
        """
        ObjReader.__init__(self, _obj_of_key)
        self._keys_cache = key_collection


class ExplicitKeysStore(ExplicitKeys, Store):
    """Wrap a store (instance) so that it gets it's keys from an explicit iterable of keys.

    >>> s = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    >>> list(s)
    ['a', 'b', 'c', 'd']
    >>> ss = ExplicitKeysStore(s, ['d', 'a'])
    >>> len(ss)
    2
    >>> list(ss)
    ['d', 'a']
    >>> list(ss.values())
    [4, 1]
    >>> ss.head()
    ('d', 4)
    """

    def __init__(self, store, key_collection):
        Store.__init__(self, store)
        self._keys_cache = key_collection


def invertible_maps(mapping=None, inv_mapping=None):
    """Returns two maps that are inverse of each other.
    Raises an AssertionError iif both maps are None, or if the maps are not inverse of each other

    Get a pair of invertible maps
    >>> invertible_maps({1: 11, 2: 22})
    ({1: 11, 2: 22}, {11: 1, 22: 2})
    >>> invertible_maps(None, {11: 1, 22: 2})
    ({1: 11, 2: 22}, {11: 1, 22: 2})

    If two maps are given and invertible, you just get them back
    >>> invertible_maps({1: 11, 2: 22}, {11: 1, 22: 2})
    ({1: 11, 2: 22}, {11: 1, 22: 2})

    Or if they're not invertible
    >>> invertible_maps({1: 11, 2: 22}, {11: 1, 22: 'ha, not what you expected!'})
    Traceback (most recent call last):
      ...
    AssertionError: mapping and inv_mapping are not inverse of each other!

    >>> invertible_maps(None, None)
    Traceback (most recent call last):
      ...
    ValueError: You need to specify one or both maps
    """
    if inv_mapping is None and mapping is None:
        raise ValueError('You need to specify one or both maps')
    if inv_mapping is None:
        assert hasattr(mapping, 'items')
        inv_mapping = {v: k for k, v in mapping.items()}
        assert len(inv_mapping) == len(
            mapping
        ), 'The values of mapping are not unique, so the mapping is not invertible'
    elif mapping is None:
        assert hasattr(inv_mapping, 'items')
        mapping = {v: k for k, v in inv_mapping.items()}
        assert len(mapping) == len(
            inv_mapping
        ), 'The values of inv_mapping are not unique, so the mapping is not invertible'
    else:
        assert (len(mapping) == len(inv_mapping)) and (
            mapping == {v: k for k, v in inv_mapping.items()}
        ), 'mapping and inv_mapping are not inverse of each other!'

    return mapping, inv_mapping


class ExplicitKeyMap:
    def __init__(self, *, key_of_id: Mapping = None, id_of_key: Mapping = None):
        """

        :param key_of_id:
        :param id_of_key:

        >>> km = ExplicitKeyMap(key_of_id={'a': 1, 'b': 2})
        >>> km.id_of_key = {1: 'a', 2: 'b'}
        >>> km._key_of_id('b')
        2
        >>> km._id_of_key(1)
        'a'
        >>> # You can specify id_of_key instead
        >>> km = ExplicitKeyMap(id_of_key={1: 'a', 2: 'b'})
        >>> assert km.key_of_id_map == {'a': 1, 'b': 2}
        >>> # You can specify both key_of_id and id_of_key
        >>> km = ExplicitKeyMap(key_of_id={'a': 1, 'b': 2}, id_of_key={1: 'a', 2: 'b'})
        >>> assert km._key_of_id(km._id_of_key(2)) == 2
        >>> assert km._id_of_key(km._key_of_id('b')) == 'b'
        >>> # But they better be inverse of each other!
        >>> km = ExplicitKeyMap(key_of_id={'a': 1, 'b': 2, 'c': 2})
        Traceback (most recent call last):
          ...
        AssertionError: The values of inv_mapping are not unique, so the mapping is not invertible
        >>> km = ExplicitKeyMap(key_of_id={'a': 1, 'b': 2}, id_of_key={1: 'a', 2: 'oh no!!!!'})
        Traceback (most recent call last):
          ...
        AssertionError: mapping and inv_mapping are not inverse of each other!
        """
        id_of_key, key_of_id = invertible_maps(id_of_key, key_of_id)
        self.key_of_id_map = key_of_id
        self.id_of_key_map = id_of_key

    def _key_of_id(self, _id):
        return self.key_of_id_map[_id]

    def _id_of_key(self, k):
        return self.id_of_key_map[k]


class ExplicitKeymapReader(ExplicitKeys, Store):
    """Wrap a store (instance) so that it gets it's keys from an explicit iterable of keys.

    >>> s = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    >>> id_of_key = {'A': 'a', 'C': 'c'}
    >>> ss = ExplicitKeymapReader(s, id_of_key=id_of_key)
    >>> list(ss)
    ['A', 'C']
    >>> ss['C']  # will look up 'C', find 'c', and call the store on that.
    3
    """

    def __init__(self, store, key_of_id=None, id_of_key=None):
        key_trans = ExplicitKeyMap(key_of_id=key_of_id, id_of_key=id_of_key)
        Store.__init__(self, kv_wrap(key_trans)(store))
        ExplicitKeys.__init__(self, key_trans.id_of_key_map.keys())


class ExplicitKeysWithPrefixRelativization(PrefixRelativizationMixin, Store):
    """
    dol.base.Keys implementation that gets it's keys explicitly from a collection given at initialization time.
    The key_collection must be a collections.abc.Collection (such as list, tuple, set, etc.)

    >>> from dol.base import Store
    >>> s = ExplicitKeysWithPrefixRelativization(key_collection=['/root/of/foo', '/root/of/bar', '/root/for/alice'])
    >>> keys = Store(store=s)
    >>> 'of/foo' in keys
    True
    >>> 'not there' in keys
    False
    >>> list(keys)
    ['of/foo', 'of/bar', 'for/alice']
    """

    __slots__ = ('_key_collection',)

    def __init__(self, key_collection, _prefix=None):
        if _prefix is None:
            _prefix = max_common_prefix(key_collection)
        store = ExplicitKeys(key_collection=key_collection)
        self._prefix = _prefix
        super().__init__(store=store)


class ObjDumper(object):
    def __init__(self, save_data_to_key, data_of_obj=None):
        self.save_data_to_key = save_data_to_key
        if data_of_obj is not None or not callable(data_of_obj):
            raise TypeError('serializer must be None or a callable')
        self.data_of_obj = data_of_obj

    def __call__(self, k, v):
        if self.data_of_obj is not None:
            return self.save_data_to_key(k, self.data_of_obj(v))
        else:
            return self.save_data_to_key(k, v)
