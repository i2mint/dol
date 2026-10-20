"""Store add-ons: path access, flattening, method aliases and hashing.

Each tool here takes a store (class or instance) and gives it an extra capability,
without changing its keys or values:

- ``add_path_get`` / ``add_path_access`` / ``autoviv``: read (and write) nested stores
  through key paths such as ``s['a', 'b', 'c']``
- ``flatten``: a flat view of a nested store, keyed by key paths
- ``insert_aliases`` / ``insert_load_dump_aliases``: alias the mapping methods
  (``write``, ``read``, ``delete``, ``list``, ``count``, ``load``, ``dump``)
- ``insert_hash_method``: make a store hashable

These used to live in ``dol.trans``, which still re-exports them, so
``from dol.trans import add_path_access`` keeps working (i2mint/dol#70).

    >>> from dol.store_addons import add_path_access
    >>> s = add_path_access({'a': {'b': {'c': 42}}})
    >>> s['a', 'b', 'c']
    42
"""

import types
from typing import Any
from collections.abc import Callable

from dol.base import kv_walk
from dol._paths_core import (
    path_get_core,
    PathCreationError,
    path_set_writeback,
    path_del_writeback,
    _warn_on_create,
)
from dol.trans import (
    store_decorator,
    cached_keys,
    kv_wrap_persister_cls,
    _wrap_store,
)


@store_decorator
def insert_hash_method(
    store=None,
    *,
    hash_method: Callable[[Any], int] = id,
):
    """Make a store hashable using the specified ``hash_method``.
    Will add (or overwrite) a ``__hash__`` method to the store that uses the
    hash_method to hash the store and an ``__eq__`` method that compares the store to
    another based on the hash_method.

    The ``hash_method``, which will be used as the ``__hash__`` method of a class
    should return an integer value that represents the hash of the object.
    To remain sane, the hash value must be the same for an object every time the
    ``__hash__`` method is called during the lifetime of the object, and objects
    that compare equal (using the __eq__ method) must have the same hash value.

    It's also important that the hash function has the property of being deterministic
    and returning a hash value that is uniformly distributed across the range of
    possible integers for the given data. This is important for the hash table
    data structure to work efficiently.

    See `This issue <https://github.com/i2mint/dol/issues/7>`_ for further information.

    >>> d = {1: 2}  # not hashable!
    >>> dd = insert_hash_method(d)
    >>> assert isinstance(hash(dd), int)  # now hashable!

    It looks the same:

    >>> dd
    {1: 2}

    But don't be fooled: dd is not equal to the original ``d`` (since
    insert_hash_method``overwrote the ``__eq__`` method to compare based on the
    hash value):

    >>> d == dd
    False

    But if you cast both to dicts and then compare, you'll be using the key and value
    based comparison of dicts, which makes these two equal.

    >>> dict(d) == dict(dd)
    True

    The default ``hash_method`` is ``id``, so two hashable wrappers won't be equal
    to eachother:

    >>> insert_hash_method(d) == insert_hash_method(d)
    False

    In the following we show two things: That you can specify your own custom
    ``hash_method``, and that you can use ``insert_hash_method`` to wrap classes

    >>> class D(dict):
    ...     pass
    >>> DD = insert_hash_method(D, hash_method=lambda x: 42)
    >>> hash(DD(d))
    42

    You can also use it as a decorator, without arguments,

    >>> @insert_hash_method
    ... class E(dict):
    ...     pass
    >>> assert isinstance(hash(E({1: 2})), int)

    or with arguments (which you must specify as keyword arguments):

    >>> @insert_hash_method(hash_method=lambda x: sum(x.values()))
    ... class F(dict):
    ...     pass
    >>> hash(F({1: 2, 3: 4}))
    6
    """
    return _wrap_store(_insert_hash_method, locals())


def _is_hashable(store):
    return getattr(store, "__hash__", None) is not None


def _insert_hash_method(store, hash_method):
    class hashable_cls(store):
        """A hashable wrapper to stores"""

        def __hash__(self):
            return hash_method(self)

        def __eq__(self, other):
            return hash_method(self) == hash_method(other)

    return hashable_cls


########################################################################################################################
# Aliasing

_method_name_for = {
    "write": "__setitem__",
    "read": "__getitem__",
    "delete": "__delitem__",
    "list": "__iter__",
    "count": "__len__",
}


@store_decorator
def add_path_get(store=None, *, name=None, path_type: type = tuple):
    """
    Make nested stores accessible through key paths.
    In a way "flatten the nested keys access".
    By default, the path object will be a tuple (e.g. ``('a', 'b', 'c')``, but you can
    make it whatever you want, and/or use `dol.paths.KeyPath` to map to and from
    forms like ``'a.b.c'``, ``'a/b/c'``, etc.

    (Warning: ``path_type`` only effects the first level.
    That is, it doesn't work recursively.
    See issue: https://github.com/i2mint/dol/issues/10.)

    Say you have some nested stores.
    You know... like a `ZipFileReader` store whose values are `ZipReader` instances,
    whose values are bytes of the zipped files
    (and you can go on... whose (json) values are...).

    For our example, let's take a nested dict instead:

    >>> s = {'a': {'b': {'c': 42}}}

    Well, you can access any node of this nested tree of stores like this:

    >>> s['a']['b']['c']
    42

    And that's fine. But maybe you'd like to do it this way instead:

    >>> s = add_path_get(s)
    >>> s['a', 'b', 'c']
    42

    You might also want to access 42 with `a.b.c` or `a/b/c` etc.
    To do that you can use `dol.paths.KeyPath` in combination with

    Args:
        store: The store (class or instance) you're wrapping.
            If not specified, the function will return a decorator.

        name: The name to give the class (not applicable to instance wrapping)
        path_type: The type that paths are expressed as. Needs to be an Iterable type.
            By default, a tuple.
            This is used to decide whether the key should be taken as a "normal"
            key of the store,
            or should be used to iterate through, recursively getting values.

    Returns:
        A wrapped store (class or instance), or a store wrapping decorator
        (if store is not specified)

    .. seealso::

        ``KeyPath`` in ``dol.paths``

    Wrapping an instance

    >>> s = add_path_get({'a': {'b': {'c': 42}}})
    >>> s['a']
    {'b': {'c': 42}}
    >>> s['a', 'b']
    {'c': 42}
    >>> s['a', 'b', 'c']
    42

    Wrapping a class

    >>> S = add_path_get(dict)
    >>> s = S(a={'b': {'c': 42}})
    >>> assert s['a'] == {'b': {'c': 42}};
    >>> assert s['a', 'b'] == {'c': 42};
    >>> assert s['a', 'b', 'c'] == 42

    Using add_path_get as a decorator

    >>> @add_path_get
    ... class S(dict):
    ...    pass
    >>> s = S(a={'b': {'c': 42}})
    >>> assert s['a'] == {'b': {'c': 42}};
    >>> assert s['a', 'b'] == s['a']['b'] == {'c': 42};
    >>> assert s['a', 'b', 'c'] == s['a']['b']['c'] == 42

    A different kind of path?
    You can choose a different path_type, but sometimes (say both keys and key paths are strings)
    You need to involve more tools. Like dol.paths.KeyPath...

    >>> from dol.paths import KeyPath
    >>> from dol.trans import kv_wrap
    >>> SS = kv_wrap(KeyPath(path_sep='.'))(S)
    >>> s = SS({'a': {'b': {'c': 42}}})
    >>> assert s['a'] == {'b': {'c': 42}};
    >>> assert s['a.b'] == s['a']['b'];
    >>> assert s['a.b.c'] == s['a']['b']['c']
    """
    name = name or store.__qualname__ + "WithPathGet"

    # TODO: This is not the best way to handle this. Investigate another way. ######################
    global_names = set(globals()).union(locals())
    if name in global_names:
        raise NameError("That name is already in use")
    # TODO: ########################################################################################

    store_cls = kv_wrap_persister_cls(store, name=name)
    store_cls._path_type = path_type

    def __getitem__(self, k):
        if isinstance(k, self._path_type):
            return path_get_core(self, k, caught_errors=())
        else:  # do things normally if the key is not a _path_type
            return super(store_cls, self).__getitem__(k)

    store_cls.__getitem__ = __getitem__

    return store_cls


# TODO: Should we keep add_path_get, or add "read_only" flag to add_path_access?
# TODO: See https://github.com/i2mint/dol/issues/10
@store_decorator
def add_path_access(
    store=None,
    *,
    name=None,
    path_type: type = tuple,
    create_missing: bool = False,
    mk_missing=None,
    explore_further=None,
    may_create=None,
    on_create=_warn_on_create,
    max_created=None,
    max_levels=20,
    verify_writeback: bool = False,
    writeback_lock=None,
):
    """Make nested stores (read/write) accessible through key paths (iterable of keys).

    Like ``add_path_get``, but with write and delete accessible through key paths.

    In a way "flatten the nested keys access".
    (Warning: ``path_type`` only effects the first level.
    That is, it doesn't work recursively.
    See issue: https://github.com/i2mint/dol/issues/10.)

    By default, the path object will be a tuple (e.g. ``('a', 'b', 'c')``, but you can
    make it whatever you want, and/or use `dol.paths.KeyPath` to map to and from
    forms like ``'a.b.c'``, ``'a/b/c'``, etc.

    Say you have some nested stores.
    You know... like a `ZipFileReader` store whose values are `ZipReader` instances,
    whose values are bytes of the zipped files
    (and you can go on... whose (json) values are...).

    For our example, let's take a nested dict instead:

    >>> s = {'a': {'b': {'c': 42}}}

    Well, you can access any node of this nested tree of stores like this:

    >>> s['a']['b']['c']
    42

    And that's fine. But maybe you'd like to do it this way instead:

    >>> s = add_path_access(s)
    >>> s['a', 'b', 'c']
    42

    So far, this is what ``add_path_get`` does. With ``add_path_access`` though you
    can also write and delete that way too:

    >>> s['a', 'b', 'c'] = 3.14
    >>> s['a', 'b', 'c']
    3.14
    >>> del s['a', 'b', 'c']
    >>> s
    {'a': {'b': {}}}

    You might also want to access 42 with `a.b.c` or `a/b/c` etc.
    To do that you can use `dol.paths.KeyPath` in combination with

    Args:
        store: The store (class or instance) you're wrapping.
            If not specified, the function will return a decorator.

        name: The name to give the class (not applicable to instance wrapping)
        path_type: The type that paths are expressed as. Needs to be an Iterable type.
            By default, a tuple.
            This is used to decide whether the key should be taken as a "normal"
            key of the store,
            or should be used to iterate through, recursively getting values.

    Returns:
        A wrapped store (class or instance), or a store wrapping decorator
        (if store is not specified)

    .. seealso::

        ``KeyPath`` in ``dol.paths``

    Wrapping a class

    >>> S = add_path_access(dict)
    >>> s = S(a={'b': {'c': 42}})
    >>> assert s['a'] == {'b': {'c': 42}};
    >>> assert s['a', 'b'] == {'c': 42};
    >>> assert s['a', 'b', 'c'] == 42
    >>> s['a', 'b', 'c'] = 3.14
    >>> s['a', 'b', 'c']
    3.14
    >>> del s['a', 'b', 'c']
    >>> s
    {'a': {'b': {}}}

    Using add_path_get as a decorator

    >>> @add_path_access
    ... class S(dict):
    ...    pass
    >>> s = S(a={'b': {'c': 42}})
    >>> assert s['a'] == {'b': {'c': 42}};
    >>> assert s['a', 'b'] == s['a']['b'] == {'c': 42};
    >>> assert s['a', 'b', 'c'] == s['a']['b']['c'] == 42
    >>> s['a', 'b', 'c'] = 3.14
    >>> s['a', 'b', 'c']
    3.14
    >>> del s['a', 'b', 'c']
    >>> s
    {'a': {'b': {}}}

    A different kind of path?
    You can choose a different path_type, but sometimes (say both keys and key paths are strings)
    You need to involve more tools. Like dol.paths.KeyPath...

    >>> from dol.paths import KeyPath
    >>> from dol.trans import kv_wrap
    >>> SS = kv_wrap(KeyPath(path_sep='.'))(S)
    >>> s = SS({'a': {'b': {'c': 42}}})
    >>> assert s['a'] == {'b': {'c': 42}};
    >>> assert s['a.b'] == s['a']['b'];
    >>> assert s['a.b.c'] == s['a']['b']['c']
    >>> s['a.b.c'] = 3.14
    >>> s
    {'a': {'b': {'c': 3.14}}}
    >>> del s['a.b.c']
    >>> s
    {'a': {'b': {}}}

    Note:
        The add_path_access doesn't carry on to values.

    >>> s = add_path_access({'a': {'b': {'c': 42}}})
    >>> s['a', 'b', 'c']
    42
    >>> # but
    >>> s['a']['b', 'c']
    Traceback (most recent call last):
      ...
    KeyError: ('b', 'c')

    That said,

    >>> add_path_access(s['a'])['b', 'c']
    42

    The reason why we don't do this automatically is that it may not always be desirable.
    If one wanted to though, one could use ``wrap_kvs(obj_of_data=...)`` to wrap
    specific values with ``add_path_access``.
    For example, if you wanted to wrap all mappings recursively, you could:

    >>> from typing import Mapping
    >>> from dol.util import instance_checker
    >>> from dol.trans import conditional_data_trans
    >>> add_path_access_if_mapping = conditional_data_trans(
    ...     condition=instance_checker(Mapping), data_trans=add_path_access
    ... )
    >>> s = add_path_access_if_mapping({'a': {'b': {'c': 42}}})
    >>> s['a', 'b', 'c']
    42
    >>> # But now this works:
    >>> s['a']['b', 'c']
    42
    """
    store_cls = kv_wrap_persister_cls(store, name=name)
    store_cls = add_path_get(store_cls, name=name, path_type=path_type)

    # Any callable write-through option (or verify_writeback) implies create_missing.
    _create_missing = (
        create_missing
        or verify_writeback
        or any(
            opt is not None
            for opt in (mk_missing, explore_further, may_create, writeback_lock)
        )
    )
    # Stash the scalar flag on the class for introspection / dispatch. The *callables*
    # are closure-captured below (NOT stashed as attributes), so they are never invoked
    # as bound methods (which would inject `self` as an extra positional argument).
    store_cls._create_missing = _create_missing

    def __setitem__(self, k, v):
        if isinstance(k, self._path_type):
            if not k:
                raise ValueError("empty key path")
            if getattr(self, "_create_missing", False):
                # Collision guard: don't silently shadow a genuine literal key that
                # happens to equal this path (stores whose real keys are tuples/paths).
                try:
                    _is_literal_key = k in getattr(self, "store", ())
                except Exception:
                    _is_literal_key = False
                if _is_literal_key:
                    raise PathCreationError(
                        f"refusing to autoviv path {k!r}: a literal key {k!r} already "
                        f"exists (create_missing is unsafe on stores whose real keys "
                        f"are tuples/paths)."
                    )
                path_set_writeback(
                    self,
                    list(k),
                    v,
                    mk_missing=mk_missing,
                    explore_further=explore_further,
                    may_create=may_create,
                    on_create=on_create,
                    max_created=max_created,
                    max_levels=max_levels,
                    verify_writeback=verify_writeback,
                    writeback_lock=writeback_lock,
                )
            else:  # backward-compatible legacy walk (unchanged behavior)
                *path_head, last_key = k
                penultimate_level = path_get_core(self, path_head, caught_errors=())
                penultimate_level[last_key] = v
        else:  # do things normally if the key is not a _path_type
            return super(store_cls, self).__setitem__(k, v)

    def __delitem__(self, k):
        if isinstance(k, self._path_type):
            if not k:
                raise ValueError("empty key path")
            if getattr(self, "_create_missing", False):
                path_del_writeback(self, list(k), writeback_lock=writeback_lock)
            else:  # backward-compatible legacy walk (unchanged behavior)
                *path_head, last_key = k
                penultimate_level = path_get_core(self, path_head, caught_errors=())
                del penultimate_level[last_key]
        else:  # do things normally if the key is not a _path_type
            return super(store_cls, self).__delitem__(k)

    store_cls.__setitem__ = __setitem__
    store_cls.__delitem__ = __delitem__

    return store_cls


def autoviv(store=None, **kwargs):
    """Opt-in write-through autovivification for key-paths.

    Shorthand for ``add_path_access(store, create_missing=True, **kwargs)``: writing
    through a key-path (e.g. ``s['a', 'b', 'c'] = v``) creates any missing intermediate
    levels on the way, and the change persists correctly even through persistent /
    copy-semantics stores (``Files``, ``wrap_kvs``-wrapped stores). See
    ``add_path_access`` for the full set of options (``mk_missing``, ``may_create``,
    ``max_created``, ``on_create``, ...). Missing-key creation is announced via
    ``warnings.warn`` by default, so a typo is never silent.

    >>> from dol import autoviv
    >>> s = autoviv({})
    >>> s['a', 'b', 'c'] = 42
    >>> s['a', 'b', 'c']
    42
    >>> s['a']['b']['c']
    42

    Like ``add_path_access``, it also works as a class decorator / factory:

    >>> S = autoviv(dict)
    >>> s = S()
    >>> s['x', 'y'] = 1
    >>> s['x', 'y']
    1
    """
    return add_path_access(store, create_missing=True, **kwargs)


@store_decorator
def flatten(store=None, *, levels=None, cache_keys=False):
    """
    Give a nested store a flat view whose keys are the ``(a, b, c)`` key paths.

    Not to be confused with ``dol.flatten_dict``, which flattens a plain nested
    ``Mapping`` into a new ``dict`` (eagerly, with joined-string keys). This one wraps
    a *store of stores* in a lazy, levels-aware view. ``flatten_store`` is the
    unambiguous name for this function (``flatten`` stays as its alias).

    Say you have a store that has three levels (or more), that is, that you can always
    ask for the value ``store[a][b][c]`` if ``a`` is a valid key of ``store``,
    ``b`` is a valid key of ``store[a]`` and ``c`` is a valid key of ``store[a][b]``.

    What ``flattened_store = flatten(store, levels=3)`` will give you is the ability
    to access the ``store[a][b][c]`` as ``store[a, b, c]``, while still being able
    to access these stores "normally".

    If that's all you need, you can just use the ``add_get_path`` wrapper for this.

    Why would you use ``flatten``? Because ``add_get_path(store)`` would still only
    give you the ``KvReader`` point of view of the root ``store``.
    If you ``list(store)``, you'd only get the first level keys,
    or if you ask if ``(a, b, c)`` is in the store, it will tell you it's not
    (though you can access data with such a key.

    Instead, a flattened store will consider that the keys are those ``(a, b, c)``
    key paths.

    Further, when flattening a store, you can ask for the view to cache the keys,
    specifying ``cache_keys=True`` or give it an explicit place to cache or
    factory to make a cache (see ``cached_keys`` wrapper for more details).
    Though caching keys is not the default it's highly recommended to do so in most
    cases. The only reason it is not the default is because if you have millions of
    keys, but little memory, that's not what you might want.

    Note:
        Flattening just provides a wrapper giving you a "flattened view". It doesn't
        change the store itself, or it's contents.

    :param store: The store instance or class to be wrapped
    :param levels: The number of nested levels to flatten
    :param cache_keys: Whether to cache the keys, or a cache factory or instance.

    >>> from dol import flatten
    >>> d = {
    ...     'a': {'b': {'c': 42}},
    ...     'aa': {'bb': {'cc': 'dragon_con'}}
    ... }

    You can get a flattened view of an instance:

    >>> m = flatten(d, levels=3, cache_keys=True)
    >>> assert (
    ...         list(m.items())
    ...         == [
    ...             (('a', 'b', 'c'), 42),
    ...             (('aa', 'bb', 'cc'), 'dragon_con')
    ...         ]
    ... )

    You can make a flattener and apply it to an instance (or a class):

    >>> my_flattener = flatten(levels=2)
    >>> m = my_flattener(d)
    >>> assert (
    ...         list(m.items())
    ...         == [
    ...             (('a', 'b'), {'c': 42}),
    ...             (('aa', 'bb'), {'cc': 'dragon_con'})
    ...         ]
    ... )

    Finally, you can wrap a class itself.

    >>> @flatten(levels=1)
    ... class MyFlatDict(dict):
    ...     pass
    >>> m = MyFlatDict(d)
    >>> assert (
    ...         list(m.items())
    ...         == [
    ...             (('a',), {'b': {'c': 42}}),
    ...             (('aa',), {'bb': {'cc': 'dragon_con'}})
    ...         ]
    ... )
    """
    return _wrap_store(_flatten, locals())
    #
    # arguments = {k: v for k, v in locals().items() if k != "arguments"}
    # store = arguments.pop("store")
    #
    # class_trans = partial(_flatten, **arguments)
    # return Store.wrap(store, class_trans=class_trans)


# The unambiguous name (vs ``dol.paths.flatten_dict``); ``flatten`` is kept.
flatten_store = flatten


def _flatten(store, *, levels, cache_keys):
    store._levels = levels

    def __iter__(self):
        yield from leveled_paths_walk(self.store, self._levels)

    store.__iter__ = __iter__

    if cache_keys:
        if cache_keys is True:
            cache_keys = list
        return add_path_get(cached_keys(store, keys_cache=cache_keys))
    else:

        def __len__(self):
            i = 0
            for i, _ in enumerate(self, 1):
                pass
            return i

        def __contains__(self, k):
            if isinstance(k, tuple):
                assert len(k) < self._levels
                return super(store, self).__contains__(k[0]) and all(
                    k[i] in self[k[i - 1]] for i in range(1, self._levels)
                )
            else:
                return super(store, self).__contains__(k)

        store.__len__ = __len__
        store.__contains__ = __contains__

        # TODO: This adds read access to all levels, not limited to levels
        return add_path_get(store)


def mk_level_walk_filt(levels):
    """Makes a ``walk_filt`` function for ``kv_walk`` based on some level logic.
    If ``levels`` is an integer, will consider it as the max path length,
    if not it will just assert that ``levels`` is callable, and return it
    """
    if isinstance(levels, int):
        return lambda p, k, v: len(p) < levels - 1
    else:
        assert callable(levels), f"levels must be a callable or an integer: {levels=}"
        return levels


def leveled_paths_walk(m, levels):
    """Yield the key paths of ``m``, down to ``levels`` levels."""
    yield from kv_walk(
        m, leaf_yield=lambda p, k, v: p, walk_filt=mk_level_walk_filt(levels)
    )


def _insert_alias(store, method_name, alias=None):
    if isinstance(alias, str) and hasattr(store, method_name):
        setattr(store, alias, getattr(store, method_name))


@store_decorator
def insert_aliases(
    store=None, *, write=None, read=None, delete=None, list=None, count=None
):
    """Insert method aliases of CRUD operations of a store (class or instance).
    If store is a class, you'll get a copy of the class with those methods added.
    If store is an instance, the methods will be added in place (no copy will be made).

    Note:
        If an operation (write, read, delete, list, count) is not specified, no alias will be created for
        that operation.

    IMPORTANT NOTE: The signatures of the methods the aliases will point to will not change.
    We say this because, you can call the write method "dump", but you'll have to use it as
    `store.dump(key, val)`, not `store.dump(val, key)`, which is the signature you're probably used to
    (it's the one used by json.dump or pickle.dump for example). If you want that familiar interface,
    using the insert_load_dump_aliases function.

    See also (and not to be confused with): ``add_aliases``

    Args:
        store: The store to extend with aliases.
        write: Desired method name for __setitem__
        read: Desired method name for __getitem__
        delete: Desired method name for __delitem__
        list: Desired method name for __iter__
        count: Desired method name for __len__

    Returns:
        A store with the desired aliases.

    >>> # Example of extending a class
    >>> mydict = insert_aliases(dict, write='dump', read='load', delete='rm', list='peek', count='size')
    >>> s = mydict(true='love')
    >>> s.dump('friends', 'forever')
    >>> s
    {'true': 'love', 'friends': 'forever'}
    >>> s.load('true')
    'love'
    >>> list(s.peek())
    ['true', 'friends']
    >>> s.size()
    2
    >>> s.rm('true')
    >>> s
    {'friends': 'forever'}
    >>>
    >>> # Example of extending an instance
    >>> from collections import UserDict
    >>> s = UserDict(true='love')  # make (and instance) of a UserDict (can't modify a dict instance)
    >>> # make aliases of note that you don't need
    >>> s = insert_aliases(s, write='put', read='retrieve', count='num_of_items')
    >>> s.put('friends', 'forever')
    >>> s
    {'true': 'love', 'friends': 'forever'}
    >>> s.retrieve('true')
    'love'
    >>> s.num_of_items()
    2
    """

    if isinstance(store, type):
        store = type(store.__qualname__, (store,), {})
    for alias, method_name in _method_name_for.items():
        _insert_alias(store, method_name, alias=locals().get(alias))
    return store


@store_decorator
def insert_load_dump_aliases(store=None, *, delete=None, list=None, count=None):
    """Insert load and dump methods, with familiar dump(obj, location) signature.

    Args:
        store: The store to extend with aliases.
        delete: Desired method name for __delitem__
        list: Desired method name for __iter__
        count: Desired method name for __len__

    Returns:
        A store with the desired aliases.

    >>> mydict = insert_load_dump_aliases(dict)
    >>> s = mydict()
    >>> s.dump(obj='love', key='true')
    >>> s
    {'true': 'love'}
    """
    store = insert_aliases(store, read="load", delete=delete, list=list, count=count)

    def dump(self, obj, key):
        return self.__setitem__(key, obj)

    if isinstance(store, type):
        store.dump = dump
    else:
        store.dump = types.MethodType(dump, store)

    return store
