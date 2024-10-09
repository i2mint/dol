"""
utils to make stores based on a the input data itself
"""

from collections.abc import Mapping
from typing import Callable, Collection as CollectionType, KT, VT, TypeVar, Iterator

from dol.base import Collection, KvReader, Store
from dol.trans import kv_wrap
from dol.util import max_common_prefix
from dol.sources import ObjReader  # because it used to be here


Source = TypeVar("Source")  # the source of some values
Getter = Callable[
    [Source, KT], VT
]  # a function that gets a value from a source and a key
# TODO: Might want to make the Getter by generic, so that we can do things like
#   Getter[Mapping] or Getter[Mapping, KeyType] or Getter[Any, KeyType]


class KeysReader(Mapping):
    """
    Mapping defined by keys with a getter function that gets values from keys.

    `KeysReader` is particularly useful in cases where you want to have a mapping
    that lazy-load values for keys from an explicit collection.

    Keywords: Lazy-evaluation, Mapping

    Args:
        src: The source where values will be extracted from.
        key_collection: A collection of keys that will be used to extract values from `src`.
        getter: A function that takes a source and a key, and returns the value for that key.
        key_error_msg: A function that takes a source and a key, and returns an error message.


    Example::

    >>> src = {'apple': 'pie', 'banana': 'split', 'carrot': 'cake'}
    >>> key_collection = ['carrot', 'apple']
    >>> getter = lambda src, key: src[key]
    >>> key_reader = KeysReader(src, key_collection, getter)

    Note that the only the keys mentioned by `key_collection` will be iterated through,
    and in the order they are mentioned in `key_collection`.

    >>> list(key_reader)
    ['carrot', 'apple']

    >>> key_reader['apple']
    'pie'
    >>> key_reader['banana']  # doctest: +ELLIPSIS +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    KeyError: "Key 'banana' was not found....key_collection attribute)"

    Let's take the same `src` and `key_collection`, but with a different getter and
    key_error_msg:

    Note that a key_error_msg must be a function that takes a `src` and a `key`,
    in that order and with those argument names. Say you wanted to not use the `src`
    in your message. You would still have to write a function that takes `src` as the
    first argument.

    >>> key_error_msg = lambda src, key: f"Key {key} was not found"  # no source information

    >>> getter = lambda src, key: f"Value for {key} in {src}: {src[key]}"
    >>> key_reader = KeysReader(src, key_collection, getter, key_error_msg=key_error_msg)
    >>> list(key_reader)
    ['carrot', 'apple']
    >>> key_reader['apple']
    "Value for apple in {'apple': 'pie', 'banana': 'split', 'carrot': 'cake'}: pie"
    >>> key_reader['banana']  # doctest: +ELLIPSIS +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    KeyError: "Key banana was not found"

    """

    def __init__(
        self,
        src: Source,
        key_collection: CollectionType[KT],
        getter: Callable[[Source, KT], VT],
        *,
        key_error_msg: Callable[
            [Source, KT], str
        ] = "Key {key} was not found in {src} should be in .key_collection attribute)".format,
    ) -> None:
        self.src = src
        self.key_collection = key_collection
        self.getter = getter
        self.key_error_msg = key_error_msg

    def __getitem__(self, key: KT) -> VT:
        if key in self:
            return self.getter(self.src, key)
        else:
            raise KeyError(self.key_error_msg(src=self.src, key=key))

    def __iter__(self) -> Iterator[KT]:
        yield from self.key_collection

    def __len__(self) -> int:
        return len(self.key_collection)

    def __contains__(self, key: KT) -> bool:
        return key in self.key_collection


# --------------------------------------------------------------------------------------
# Older stuff:


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

    __slots__ = ("_keys_cache",)

    def __init__(
        self, key_collection: CollectionType
    ):  # don't remove this init: Don't. Need for _keys_cache init
        assert isinstance(key_collection, CollectionType), (
            "key_collection must be a collections.abc.Collection, i.e. have a __len__, __contains__, and __len__."
            "The key_collection you gave me was a {}".format(type(key_collection))
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


from dol.util import invertible_maps


# TODO: Put on the path of deprecation, since KeyCodecs.mapped_keys is a better way to do this.
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


# ExplicitKeysWithPrefixRelativization: Moved to dol.paths


class ObjDumper(object):
    def __init__(self, save_data_to_key, data_of_obj=None):
        self.save_data_to_key = save_data_to_key
        if data_of_obj is not None or not callable(data_of_obj):
            raise TypeError("serializer must be None or a callable")
        self.data_of_obj = data_of_obj

    def __call__(self, k, v):
        if self.data_of_obj is not None:
            return self.save_data_to_key(k, self.data_of_obj(v))
        else:
            return self.save_data_to_key(k, v)
