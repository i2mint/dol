"""
This module contains key-value views of disparate sources.
"""

from typing import Iterator, Mapping, Iterable, Callable, Union, Any
from operator import itemgetter
from itertools import groupby as itertools_groupby

from dol.base import KvReader, KvPersister
from dol.trans import cached_keys
from dol.caching import mk_cached_store
from dol.util import copy_attrs
from dol.signatures import Sig


# ignore_if_module_not_found = suppress(ModuleNotFoundError)
#
# with ignore_if_module_not_found:
#     # To install: pip install mongodol
#     from mongodol.stores import (
#         MongoStore,
#         MongoTupleKeyStore,
#         MongoAnyKeyStore,
#     )


def identity_func(x):
    return x


def inclusive_subdict(d, include):
    return {k: d[k] for k in d.keys() & include}


def exclusive_subdict(d, exclude):
    return {k: d[k] for k in d.keys() - exclude}


class NotUnique(ValueError):
    """Raised when an iterator was expected to have only one element, but had more"""


NoMoreElements = type("NoMoreElements", (object,), {})()


def unique_element(iterator):
    element = next(iterator)
    if next(iterator, NoMoreElements) is not NoMoreElements:
        raise NotUnique("iterator had more than one element")
    return element


KvSpec = Union[Callable, Iterable[Union[str, int]], str, int]


def _kv_spec_to_func(kv_spec: KvSpec) -> Callable:
    if isinstance(kv_spec, (str, int)):
        return itemgetter(kv_spec)
    elif isinstance(kv_spec, Iterable):
        return itemgetter(*kv_spec)
    elif kv_spec is None:
        return identity_func
    return kv_spec


# TODO: This doesn't work
# KvSpec.from = _kv_spec_to_func  # I'd like to be able to couple KvSpec and it's
# conversion function (even more: __call__ instead of from)


# TODO: Generalize to several layers
#   Need a general tool for flattening views.
#   What we're doing here is giving access to a nested/tree structure through a key-value
#   view where keys specify tree paths.
#   Should handle situations where number layers are not fixed in advanced,
#   but determined by some rules executed dynamically.
#   Related DirStore and kv_walk.
class FlatReader(KvReader):
    """Get a 'flat view' of a store of stores.
    That is, where keys are `(first_level_key, second_level_key)` pairs.

    >>> readers = {
    ...     'fr': {1: 'un', 2: 'deux'},
    ...     'it': {1: 'uno', 2: 'due', 3: 'tre'},
    ... }
    >>> s = FlatReader(readers)
    >>> list(s)
    [('fr', 1), ('fr', 2), ('it', 1), ('it', 2), ('it', 3)]
    >>> s[('fr', 1)]
    'un'
    >>> s['it', 2]
    'due'
    """

    def __init__(self, readers):
        self._readers = readers

    def __iter__(self):
        # go through the first level paths:
        for first_level_path, reader in self._readers.items():
            for second_level_path in reader:  # go through the keys of the reader
                yield first_level_path, second_level_path

    def __getitem__(self, k):
        first_level_path, second_level_path = k
        return self._readers[first_level_path][second_level_path]


from collections import ChainMap


from collections import ChainMap
from typing import Callable, Iterable, Iterator, Mapping, TypedDict, Union

from dol.base import KvPersister, KvReader
from dol.trans import wrap_kvs


class FanoutReader(KvReader):
    """Get a 'fanout view' of a store of stores.
    That is, when a key is requested, the key is passed to all the stores, and results
    accumulated in a dict that is then returned.

    param stores: A mapping of store keys to stores.
    param default: The value to return if the key is not in any of the stores.
    param get_existing_values_only: If True, only return values for stores that contain
        the key.

    Let's define the following sub-stores:

    >>> bytes_store = dict(
    ...     a=b'a',
    ...     b=b'b',
    ...     c=b'c',
    ... )
    >>> metadata_store = dict(
    ...     b=dict(x=2),
    ...     c=dict(x=3),
    ...     d=dict(x=4),
    ... )

    We can create a fan-out reader from these stores:

    >>> stores = dict(bytes_store=bytes_store, metadata_store=metadata_store)
    >>> reader = FanoutReader(stores)
    >>> reader['b']
    {'bytes_store': b'b', 'metadata_store': {'x': 2}}

    The reader returns a dict with the values from each store, keyed by the name of the
    store.

    We can also pass a default value to return if the key is not in the store:

    >>> reader = FanoutReader(
    ...     stores=stores,
    ...     default='no value in this store for this key',
    ... )
    >>> reader['a']
    {'bytes_store': b'a', 'metadata_store': 'no value in this store for this key'}

    If the key is not in any of the stores, a KeyError is raised:

    >>> reader['z']
    Traceback (most recent call last):
        ...
    KeyError: 'z'

    We can also pass `get_existing_values_only=True` to only return values for stores
    that contain the key:

    >>> reader = FanoutReader(
    ...     stores=stores,
    ...     get_existing_values_only=True,
    ... )
    >>> reader['a']
    {'bytes_store': b'a'}
    """

    def __init__(
        self,
        stores: Mapping[Any, Mapping],
        default: Any = None,
        *,
        get_existing_values_only: bool = False,
    ):
        if not isinstance(stores, Mapping):
            if isinstance(stores, Iterable):
                stores = dict(enumerate(stores))
            else:
                raise ValueError(
                    f"stores must be a Mapping or an Iterable, not {type(stores)}"
                )
        self._stores = stores
        self._default = default
        self._get_existing_values_only = get_existing_values_only

    @classmethod
    def from_variadics(cls, *args, **kwargs):
        """A way to create a fan-out store from a mix of args and kwargs, instead of a
        single dict.

        param args: sub-stores used to fan-out the data. These stores will be
            represented by their index in the tuple.
        param kwargs: sub-stores used to fan-out the data. These stores will be
            represented by their name in the dict. __init__ arguments can also be passed
            as kwargs (i.e. `default`, `get_existing_values_only`, and any other subclass
            specific arguments).

        Let's use the same sub-stores:

        >>> bytes_store = dict(
        ...     a=b'a',
        ...     b=b'b',
        ...     c=b'c',
        ... )
        >>> metadata_store = dict(
        ...     b=dict(x=2),
        ...     c=dict(x=3),
        ...     d=dict(x=4),
        ... )

        We can create a fan-out reader from these stores, using args:

        >>> reader = FanoutReader.from_variadics(bytes_store, metadata_store)
        >>> reader['b']
        {0: b'b', 1: {'x': 2}}

        The reader returns a dict with the values from each store, keyed by the index of
        the store in the `args` tuple.

        We can also create a fan-out reader passing the stores in kwargs:

        >>> reader = FanoutReader.from_variadics(
        ...     bytes_store=bytes_store,
        ...     metadata_store=metadata_store
        ... )
        >>> reader['b']
        {'bytes_store': b'b', 'metadata_store': {'x': 2}}

        This way, the returned value is keyed by the name of the store.

        We can also mix args and kwargs:

        >>> reader = FanoutReader.from_variadics(bytes_store, metadata_store=metadata_store)
        >>> reader['b']
        {0: b'b', 'metadata_store': {'x': 2}}

        Note that the order of the stores is determined by the order of the args and
        kwargs.
        """

        def extract_init_kwargs():
            for p in cls_sig.parameters:
                if p in kwargs:
                    yield p, kwargs.pop(p)

        cls_sig = Sig(cls)
        cls_kwargs = dict(extract_init_kwargs())
        stores = dict({i: store for i, store in enumerate(args)}, **kwargs)
        return cls(stores=stores, **cls_kwargs)

    @property
    def _keys(self):
        return ChainMap(*self._stores.values())

    def __getitem__(self, k):
        v = {
            store_key: store.get(k, self._default)
            for store_key, store in self._stores.items()
        }
        if all(v == self._default for v in v.values()):
            raise KeyError(k)
        if self._get_existing_values_only:
            v = {k: v for k, v in v.items() if v != self._default}
        return v

    def __iter__(self) -> Iterator:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, k) -> int:
        return k in self._keys


class FanoutPersister(FanoutReader, KvPersister):
    """
    A fanout persister is a fanout reader that can also set and delete items.

    param stores: A mapping of store keys to stores.
    param default: The value to return if the key is not in any of the stores.
    param get_existing_values_only: If True, only return values for stores that contain
        the key.
    param need_to_set_all_stores: If True, all stores must be set when setting a value.
        If False, only the stores that are set will be updated.
    param ignore_non_existing_store_keys: If True, ignore store keys from the value that
        are not in the persister. If False, a ValueError is raised.

    Let's create a persister from in-memory stores:

    >>> bytes_store = dict()
    >>> metadata_store = dict()
    >>> persister = FanoutPersister(
    ...     stores = dict(bytes_store=bytes_store, metadata_store=metadata_store)
    ... )

    The persister sets the values in each store, based on the store key in the value dict.

    >>> persister['a'] = dict(bytes_store=b'a', metadata_store=dict(x=1))
    >>> persister['a']
    {'bytes_store': b'a', 'metadata_store': {'x': 1}}

    By default, not all stores must be set when setting a value:

    >>> persister['b'] = dict(bytes_store=b'b')
    >>> persister['b']
    {'bytes_store': b'b', 'metadata_store': None}

    This allow to update a subset of the stores whithout having to set all the stores.

    >>> persister['a'] = dict(bytes_store=b'A')
    >>> persister['a']
    {'bytes_store': b'A', 'metadata_store': {'x': 1}}

    This behavior can be changed by passing `need_to_set_all_stores=True`:

    >>> persister_all_stores = FanoutPersister(
    ...     stores=dict(bytes_store=dict(), metadata_store=dict()),
    ...     need_to_set_all_stores=True,
    ... )
    >>> persister_all_stores['a'] = dict(bytes_store=b'a')
    Traceback (most recent call last):
        ...
    ValueError: All stores must be set when setting a value. Missing stores: {'metadata_store'}

    By default, if a store key from the value is not in the persister, a ValueError is
    raised:

    >>> persister['a'] = dict(
    ...     bytes_store=b'a', metadata_store=dict(y=1), other_store='some value'
    ... )
    Traceback (most recent call last):
        ...
    ValueError: The value contains some invalid store keys: {'other_store'}

    This behavior can be changed by passing `ignore_non_existing_store_keys=True`:

    >>> persister_ignore_non_existing_store_keys = FanoutPersister(
    ...     stores=dict(bytes_store=dict(), metadata_store=dict()),
    ...     ignore_non_existing_store_keys=True,
    ... )
    >>> persister_ignore_non_existing_store_keys['a'] = dict(
    ...     bytes_store=b'a', metadata_store=dict(y=1), other_store='some value'
    ... )
    >>> persister_ignore_non_existing_store_keys['a']
    {'bytes_store': b'a', 'metadata_store': {'y': 1}}

    Note that the value of the non-existing store key is ignored! So, be careful when
    using this option, to avoid losing data.

    Let's delete items now:

    >>> del persister['a']
    >>> 'a' in persister
    False

    The key as been deleted from all the stores:

    >>> 'a' in bytes_store
    False
    >>> 'a' in metadata_store
    False

    As expected, if the key is not in any of the stores, a KeyError is raised:

    >>> del persister['z']
    Traceback (most recent call last):
        ...
    KeyError: 'z'

    However, if the key is in some of the stores, but not in others, the key is deleted
    from the stores where it is present:

    >>> bytes_store=dict(a=b'a')
    >>> persister = FanoutPersister(
    ...     stores=dict(bytes_store=bytes_store, metadata_store=dict()),
    ... )
    >>> del persister['a']
    >>> 'a' in persister
    False
    >>> 'a' in bytes_store
    False
    """

    def __init__(
        self,
        stores: Mapping[Any, Mapping],
        default: Any = None,
        *,
        get_existing_values_only: bool = False,
        need_to_set_all_stores: bool = False,
        ignore_non_existing_store_keys: bool = False,
        **kwargs,
    ):
        super().__init__(
            stores=stores,
            default=default,
            get_existing_values_only=get_existing_values_only,
        )
        self._need_to_set_all_stores = need_to_set_all_stores
        self._ignore_non_existing_store_keys = ignore_non_existing_store_keys

    def __setitem__(self, k, v: Mapping):
        if self._need_to_set_all_stores and not set(self._stores).issubset(set(v)):
            missing_stores = set(self._stores) - set(v)
            raise ValueError(
                f"All stores must be set when setting a value. Missing stores: {missing_stores}"
            )
        if not self._ignore_non_existing_store_keys and not set(v).issubset(
            set(self._stores)
        ):
            invalid_store_keys = set(v) - set(self._stores)
            raise ValueError(
                f"The value contains some invalid store keys: {invalid_store_keys}"
            )
        for store_key, vv in v.items():
            if store_key in self._stores:
                self._stores[store_key][k] = vv

    def __delitem__(self, k):
        stores_to_delete_from = {
            store_key: store for store_key, store in self._stores.items() if k in store
        }
        if not stores_to_delete_from:
            raise KeyError(k)
        for store in stores_to_delete_from.values():
            del store[k]


NotFound = type("NotFound", (object,), {})()


@wrap_kvs(value_encoder=lambda self, v: {k: v for k in self._stores.keys()})
class CascadedStores(FanoutPersister):
    """
    A MutableMapping interface to a collection of stores that will write a value in
    all the stores it contains, read it from the first store it finds that has it, and
    write it back to all the stores up to the store where it found it.

    This is useful, for example, when you want to, say, write something to disk,
    and possibly to a remote backup or shared store, but also keep that value in memory.

    The name `CascadedStores` comes from "Cascaded Caches", which is a common pattern in
    caching systems
    (e.g. https://philipwalton.com/articles/cascading-cache-invalidation/)

    To demo this, let's create a couple of stores that print when they get a value:


    >>> from collections import UserDict
    >>> class LoggedDict(UserDict):
    ...     def __init__(self, name: str):
    ...        self.name = name
    ...        super().__init__()
    ...     def __getitem__(self, k):
    ...         print(f"Getting {k} from {self.name}")
    ...         return super().__getitem__(k)
    >>> cache = LoggedDict('cache')
    >>> disk = LoggedDict('disk')
    >>> remote = LoggedDict('remote')

    Now we can create a CascadedStores instance with these stores and write a
    value to it:

    >>> stores = CascadedStores([cache, disk, remote])
    >>> stores['f'] = 42

    See that it's in both stores:

    >>> cache['f']
    Getting f from cache
    42
    >>> disk['f']
    Getting f from disk
    42
    >>> remote['f']
    Getting f from remote
    42

    See how it reads from the first store only, because it found the `f` key there:

    >>> stores['f']
    Getting f from cache
    42

    Let's write something in disk only:

    >>> disk['g'] = 43

    Now if you ask for `g`, it won't find it in cache, but will find it in `disk`
    and return it. The reason you see the "Getting g from cache" message is because
    the `stores` object first tries to get it in `cache`, and only if it doesn't find
    it there, it tries to get it from `disk`.

    >>> stores['g']
    Getting g from cache
    Getting g from disk
    43

    Here's the thing though. Now, `g` is also in `cache`:

    >>> cache
    {'f': 42, 'g': 43}

    But `remote` still only has `f`:

    >>> remote
    {'f': 42}


    """

    # Note: Need to overwrite FanoutPersister's getitem to not read values from all stores
    def __getitem__(self, k):
        """Returns the value of the first store for that key"""
        for store_ref, store in self._stores.items():
            if (v := store.get(k, NotFound)) is not NotFound:
                # value found, now let's write it to all the stores up to the store_ref
                for _store_ref, store in self._stores.items():
                    if _store_ref != store_ref:
                        store[k] = v
                    else:
                        break
                # now return the value
                return v
        raise KeyError(k)


class SequenceKvReader(KvReader):
    """
    A KvReader that sources itself in an iterable of elements from which keys and values
    will be extracted and grouped by key.

    >>> docs = [{'_id': 0, 's': 'a', 'n': 1},
    ...  {'_id': 1, 's': 'b', 'n': 2},
    ...  {'_id': 2, 's': 'b', 'n': 3}]
    >>>

    Out of the box, SequenceKvReader gives you enumerated integer indices as keys,
    and the sequence items as is, as vals

    >>> s = SequenceKvReader(docs)
    >>> list(s)
    [0, 1, 2]
    >>> s[1]
    {'_id': 1, 's': 'b', 'n': 2}
    >>> assert s.get('not_a_key') is None

    You can make it more interesting by specifying a val function to compute the vals
    from the sequence elements

    >>> s = SequenceKvReader(docs, val=lambda x: (x['_id'] + x['n']) * x['s'])
    >>> assert list(s) == [0, 1, 2]  # as before
    >>> list(s.values())
    ['a', 'bbb', 'bbbbb']

    But where it becomes more useful is when you specify a key as well.
    SequenceKvReader will then compute the keys with that function, group them,
    and return as the value, the list of sequence elements that match that key.

    >>> s = SequenceKvReader(docs,
    ...         key=lambda x: x['s'],
    ...         val=lambda x: {k: x[k] for k in x.keys() - {'s'}})
    >>> assert list(s) == ['a', 'b']
    >>> assert s['a'] == [{'_id': 0, 'n': 1}]
    >>> assert s['b'] == [{'_id': 1, 'n': 2}, {'_id': 2, 'n': 3}]

    The cannonical form of key and val is a function, but if you specify a str, int,
    or iterable thereof,
    SequenceKvReader will make an itemgetter function from it, for your convenience.

    >>> s = SequenceKvReader(docs, key='_id')
    >>> assert list(s) == [0, 1, 2]
    >>> assert s[1] == [{'_id': 1, 's': 'b', 'n': 2}]

    The ``val_postproc`` argument is ``list`` by default, but what if we don't specify
    any?
    Well then you'll get an unconsumed iterable of matches

    >>> s = SequenceKvReader(docs, key='_id', val_postproc=None)
    >>> assert isinstance(s[1], Iterable)

    The ``val_postproc`` argument specifies what to apply to this iterable of matches.
    For example, you can specify ``val_postproc=next`` to simply get the first matched
    element:


    >>> s = SequenceKvReader(docs, key='_id', val_postproc=next)
    >>> assert list(s) == [0, 1, 2]
    >>> assert s[1] == {'_id': 1, 's': 'b', 'n': 2}

    We got the whole dict there. What if we just want we didn't want the _id, which is
    used by the key, in our val?

    >>> from functools import partial
    >>> all_but_s = partial(exclusive_subdict, exclude=['s'])
    >>> s = SequenceKvReader(docs, key='_id', val=all_but_s, val_postproc=next)
    >>> assert list(s) == [0, 1, 2]
    >>> assert s[1] == {'_id': 1, 'n': 2}

    Suppose we want to have the pair of ('_id', 'n') values as a key, and only 's'
    as a value...

    >>> s = SequenceKvReader(docs, key=('_id', 'n'), val='s', val_postproc=next)
    >>> assert list(s) == [(0, 1), (1, 2), (2, 3)]
    >>> assert s[1, 2] == 'b'

    But remember that using ``val_postproc=next`` will only give you the first match
    as a val.

    >>> s = SequenceKvReader(docs, key='s', val=all_but_s, val_postproc=next)
    >>> assert list(s) == ['a', 'b']
    >>> assert s['a'] == {'_id': 0, 'n': 1}
    >>> assert s['b'] == {'_id': 1, 'n': 2}   # note that only the first match is returned.

    If you do want to only grab the first match, but want to additionally assert
    that there is no more than one,
    you can specify this with ``val_postproc=unique_element``:

    >>> s = SequenceKvReader(docs, key='s', val=all_but_s, val_postproc=unique_element)
    >>> assert s['a'] == {'_id': 0, 'n': 1}
    >>> # The following should raise an exception since there's more than one match
    >>> s['b']  # doctest: +SKIP
    Traceback (most recent call last):
      ...
    sources.NotUnique: iterator had more than one element

    """

    def __init__(
        self,
        sequence: Iterable,
        key: KvSpec = None,
        val: KvSpec = None,
        val_postproc=list,
    ):
        """Make a SequenceKvReader instance,

        :param sequence: The iterable to source the keys and values from.
        :param key: Specification of how to extract a key from an iterable element.
            If None, will use integer keys from key, val = enumerate(iterable).
            key can be a callable, a str or int, or an iterable of strs and ints.
        :param val: Specification of how to extract a value from an iterable element.
            If None, will use the element as is, as the value.
            val can be a callable, a str or int, or an iterable of strs and ints.
        :param val_postproc: Function to apply to the iterable of vals.
            Default is ``list``, which will have the effect of values being lists of all
            vals matching a key.
            Another popular choice is ``next`` which will have the effect of values
            being the first matched to the key
        """
        self.sequence = sequence
        if key is not None:
            self.key = _kv_spec_to_func(key)
        else:
            self.key = None
        self.val = _kv_spec_to_func(val)
        self.val_postproc = val_postproc or identity_func
        assert isinstance(self.val_postproc, Callable)

    def kv_items(self):
        if self.key is not None:
            for k, v in itertools_groupby(self.sequence, key=self.key):
                yield k, self.val_postproc(map(self.val, v))
        else:
            for i, v in enumerate(self.sequence):
                yield i, self.val(v)

    def __getitem__(self, k):
        for kk, vv in self.kv_items():
            if kk == k:
                return vv
        raise KeyError(f"Key not found: {k}")

    def __iter__(self):
        yield from map(itemgetter(0), self.kv_items())


@cached_keys
class CachedKeysSequenceKvReader(SequenceKvReader):
    """SequenceKvReader but with keys cached. Use this one if you will perform multiple
    accesses to only some of the keys of the store"""


@mk_cached_store
class CachedSequenceKvReader(SequenceKvReader):
    """SequenceKvReader but with the whole mapping cached as a dict. Use this one if
    you will perform multiple accesses to the store"""


# TODO: Basically same could be acheived with
#  wrap_kvs(obj_of_data=methodcaller('__call__'))
class FuncReader(KvReader):
    """Reader that seeds itself from a data fetching function list
    Uses the function list names as the keys, and their returned value as the values.

    For example: You have a list of urls that contain the data you want to have access
    to.
    You can write functions that bare the names you want to give to each dataset,
    and have the function fetch the data from the url, extract the data from the
    response and possibly prepare it (we advise minimally, since you can always
    transform from the raw source, but the opposite can be impossible).

    >>> def foo():
    ...     return 'bar'
    >>> def pi():
    ...     return 3.14159
    >>> s = FuncReader([foo, pi])
    >>> list(s)
    ['foo', 'pi']
    >>> s['foo']
    'bar'
    >>> s['pi']
    3.14159

    You might want to give your own names to the functions.
    You might even have to (because the callable you're using doesn't have a `__name__`).
    In that case, you can specify a ``{name: func, ...}`` dict instead of a simple
    iterable.

    >>> s = FuncReader({'FU': foo, 'Pie': pi})
    >>> list(s)
    ['FU', 'Pie']
    >>> s['FU']
    'bar'

    """

    def __init__(self, funcs: Union[Mapping[str, Callable], Iterable[Callable]]):
        # TODO: assert no free arguments (arguments are allowed but must all have
        #  defaults)
        if isinstance(funcs, Mapping):
            self.funcs = dict(funcs)
        else:
            self.funcs = {func.__name__: func for func in funcs}

    def __contains__(self, k):
        return k in self.funcs

    def __iter__(self):
        yield from self.funcs

    def __len__(self):
        return len(self.funcs)

    def __getitem__(self, k):
        return self.funcs[k]()  # call the func


class FuncDag(FuncReader):
    def __init__(self, funcs, **kwargs):
        super().__init__(funcs)
        self._sig = {fname: Sig(func) for fname, func in self._func.items()}
        # self._input_names = sum(self._sig)

    def __getitem__(self, k):
        return self._func_of_name[k]()  # call the func


import os

psep = os.path.sep

ddir = lambda o: [x for x in dir(o) if not x.startswith("_")]


def not_underscore_prefixed(x):
    return not x.startswith("_")


def _path_to_module_str(path, root_path):
    assert path.endswith(".py")
    path = path[:-3]
    if root_path.endswith(psep):
        root_path = root_path[:-1]
    root_path = os.path.dirname(root_path)
    len_root = len(root_path) + 1
    path_parts = path[len_root:].split(psep)
    if path_parts[-1] == "__init__.py":
        path_parts = path_parts[:-1]
    return ".".join(path_parts)


# class SourceReader(KvReader):
#     def __getitem__(self, k):
#         return getsource(k)

# class NestedObjReader(ObjReader):
#     def __init__(self, obj, src_to_key, key_filt=None, ):


class ObjLoader(object):
    def __init__(self, data_of_key, obj_of_data=None):
        self.data_of_key = data_of_key
        if obj_of_data is not None or not callable(obj_of_data):
            raise TypeError("serializer must be None or a callable")
        self.obj_of_data = obj_of_data

    def __call__(self, k):
        if self.obj_of_data is not None:
            return self.obj_of_data(self.data_of_key(k))
        else:
            return self.data_of_key(k)


# TODO: See explicit.py module and FuncReader above for near duplicates!
# TODO: Add an obj_of_key argument to wrap_kvs? (Or should it be data_of_key?)
# TODO: Another near-duplicate found: dol.paths.PathMappedData
# Note: Older version commmented below
class ObjReader:
    """
    A reader that uses a specified function to get the contents for a given key.

    >>> # define a contents_of_key that reads stuff from a dict
    >>> data = {'foo': 'bar', 42: "everything"}
    >>> def read_dict(k):
    ...     return data[k]
    >>> pr = ObjReader(_obj_of_key=read_dict)
    >>> pr['foo']
    'bar'
    >>> pr[42]
    'everything'
    >>>
    >>> # define contents_of_key that reads stuff from a file given it's path
    >>> def read_file(path):
    ...     with open(path) as fp:
    ...         return fp.read()
    >>> pr = ObjReader(_obj_of_key=read_file)
    >>> file_where_this_code_is = __file__

    ``file_where_this_code_is`` should be the file where this doctest is written,
    therefore should contain what I just said:

    >>> 'therefore should contain what I just said' in pr[file_where_this_code_is]
    True

    """

    def __init__(self, _obj_of_key: Callable):
        self._obj_of_key = _obj_of_key

    @classmethod
    def from_composition(cls, data_of_key, obj_of_data=None):
        return cls(
            _obj_of_key=ObjLoader(data_of_key=data_of_key, obj_of_data=obj_of_data)
        )

    def __getitem__(self, k):
        try:
            return self._obj_of_key(k)
        except Exception as e:
            raise KeyError(
                "KeyError in {} when trying to __getitem__({}): {}".format(
                    e.__class__.__name__, k, e
                )
            )


# Pattern: Recursive navigation
# Note: Moved dev to independent package called "guide"
@cached_keys(keys_cache=set, name="Attrs")
class Attrs(ObjReader):
    """A simple recursive KvReader for the attributes of a python object.
    Keys are attr names, values are Attrs(attr_val) instances.

    Note: A more significant version of Attrs, along with many tools based on it,
    was moved to pypi package: guide.


        pip install guide
    """

    def __init__(self, obj, key_filt=not_underscore_prefixed, getattrs=dir):
        super().__init__(obj)
        self._key_filt = key_filt
        self.getattrs = getattrs

    @classmethod
    def module_from_path(
        cls, path, key_filt=not_underscore_prefixed, name=None, root_path=None
    ):
        import importlib.util

        if name is None:
            if root_path is not None:
                try:
                    name = _path_to_module_str(path, root_path)
                except Exception:
                    name = "fake.module.name"
        spec = importlib.util.spec_from_file_location(name, path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        return cls(foo, key_filt)

    def __iter__(self):
        yield from filter(self._key_filt, self.getattrs(self.src))

    def __getitem__(self, k):
        return self.__class__(getattr(self.src, k), self._key_filt, self.getattrs)

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.src}, {self._key_filt})"


Ddir = Attrs  # for back-compatibility, temporarily

import re


def _extract_first_identifier(string: str) -> str:
    m = re.match(r"\w+", string)
    if m:
        return m.group(0)
    else:
        return ""


def _dflt_object_namer(obj, dflt_name: str = "name_not_found"):
    return (
        getattr(obj, "__name__", None)
        or _extract_first_identifier(getattr(obj, "__doc__"))
        or dflt_name
    )


class AttrContainer:
    """Convenience class to hold Key-Val pairs as attribute-val pairs, with all the
    magic methods of mappings.

    On the other hand, you will not get the usuall non-dunders (non magic methods) of
    ``Mappings``. This is so that you can use tab completion to access only the keys
    the container has, and not any of the non-dunder methods like ``get``, ``items``,
    etc.

    >>> da = AttrContainer(foo='bar', life=42)
    >>> da.foo
    'bar'
    >>> da['life']
    42
    >>> da.true = 'love'
    >>> len(da)  # count the number of fields
    3
    >>> da['friends'] = 'forever'  # write as dict
    >>> da.friends  # read as attribute
    'forever'
    >>> list(da)  # list fields (i.e. keys i.e. attributes)
    ['foo', 'life', 'true', 'friends']
    >>> 'life' in da  # check containement
    True

    >>> del da['friends']  # delete as dict
    >>> del da.foo # delete as attribute
    >>> list(da)
    ['life', 'true']
    >>> da._source  # the hidden Mapping (here dict) that is wrapped
    {'life': 42, 'true': 'love'}

    If you don't specify a name for some objects, ``AttrContainer`` will use the
    ``__name__`` attribute of the objects:

    >>> d = AttrContainer(map, tuple, obj='objects')
    >>> list(d)
    ['map', 'tuple', 'obj']

    You can also specify a different way of auto naming the objects:

    >>> d = AttrContainer('an', 'example', _object_namer=lambda x: f"_{len(x)}")
    >>> {k: getattr(d, k) for k in d}
    {'_2': 'an', '_7': 'example'}

    .. seealso:: Objects in ``py2store.utils.attr_dict`` module
    """

    _source = None

    def __init__(
        self,
        *objects,
        _object_namer: Callable[[Any], str] = _dflt_object_namer,
        **named_objects,
    ):
        if objects:
            auto_named_objects = {_object_namer(obj): obj for obj in objects}
            self._validate_named_objects(auto_named_objects, named_objects)
            named_objects = dict(auto_named_objects, **named_objects)

        super().__setattr__("_source", {})
        for k, v in named_objects.items():
            setattr(self, k, v)

    @staticmethod
    def _validate_named_objects(auto_named_objects, named_objects):
        if not all(map(str.isidentifier, auto_named_objects)):
            raise ValueError(
                "All names produced by _object_namer should be valid python identifiers:"
                f" {', '.join(x for x in auto_named_objects if not x.isidentifier())}"
            )
        clashing_names = auto_named_objects.keys() & named_objects.keys()
        if clashing_names:
            raise ValueError(
                "Some auto named objects clashed with named ones: "
                f"{', '.join(clashing_names)}"
            )

    def __getitem__(self, k):
        return self._source[k]

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __delitem__(self, k):
        delattr(self, k)

    def __iter__(self):
        return iter(self._source.keys())

    def __len__(self):
        return len(self._source)

    def __setattr__(self, k, v):
        self._source[k] = v
        super().__setattr__(k, v)

    def __delattr__(self, k):
        del self._source[k]
        super().__delattr__(k)

    def __contains__(self, k):
        return k in self._source

    def __repr__(self):
        return super().__repr__()


# TODO: Make it work with a store, without having to load and store the values explicitly.
class AttrDict(AttrContainer, KvPersister):
    """Convenience class to hold Key-Val pairs with both a dict-like and struct-like
    interface.

    The dict-like interface has just the basic get/set/del/iter/len
    (all "dunders": none visible as methods). There is no get, update, etc.
    This is on purpose, so that the only visible attributes
    (those you get by tab-completion for instance) are the those you injected.

    >>> da = AttrDict(foo='bar', life=42)

    You get the "keys as attributes" that you get with ``AttrContainer``:

    >>> da.foo
    'bar'

    But additionally, you get the extra ``Mapping`` methods:

    >>> list(da.keys())
    ['foo', 'life']
    >>> list(da.values())
    ['bar', 42]
    >>> da.get('foo')
    'bar'
    >>> da.get('not_a_key', 'default')
    'default'

    You can assign through key or attribute assignment:

    >>> da['true'] = 'love'
    >>> da.friends = 'forever'
    >>> list(da.items())
    [('foo', 'bar'), ('life', 42), ('true', 'love'), ('friends', 'forever')]


    etc.

    .. seealso:: Objects in ``py2store.utils.attr_dict`` module
    """


# class ObjReader(KvReader):
#     def __init__(self, obj):
#         self.src = obj
#         copy_attrs(
#             target=self,
#             source=self.src,
#             attrs=('__name__', '__qualname__', '__module__'),
#             raise_error_if_an_attr_is_missing=False,
#         )

#     def __repr__(self):
#         return f'{self.__class__.__qualname__}({self.src})'

#     @property
#     def _source(self):
#         from warnings import warn

#         warn('Deprecated: Use .src instead of ._source', DeprecationWarning, 2)
#         return self.src
