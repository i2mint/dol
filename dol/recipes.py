"""Recipes using dol

Examples::

    >>> d = {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
    >>> list(search_paths(d, lambda p, k, v: v == 2))
    [('a', 'b', 'd')]

"""

from functools import partial
from typing import Callable, Mapping, KT, VT, TypeVar, Iterator, Union, Literal
from dol import kv_walk

from dol.util import add_as_attribute_of

__all__ = ['search_paths']

PT = TypeVar('PT')  # Path Type
PkvFilt = Callable[[PT, KT, VT], bool]


def search_paths(d: Mapping, pkv_filt: Callable[[PT, KT, VT], bool]) -> Iterator[PT]:
    """Walk a dict, yielding paths to values that pass the ``pkv_filt``

    :param d: The ``Mapping`` to walk (scan through)
    :param pkv_filt: A function that takes a path, key, and value, and returns
        ``True`` if the path should be yielded, and ``False`` otherwise
    :return: An iterator of paths to values that pass the ``pkv_filt``


    Example::

    >>> d = {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
    >>> list(search_paths(d, lambda p, k, v: v == 2))
    [('a', 'b', 'd')]

    >>> mm = {
    ...     'a': {'b': {'c': 42}},
    ...     'aa': {'bb': {'cc': 'meaning of life'}},
    ...     'aaa': {'bbb': 314},
    ... }
    >>> return_path_if_int_leaf = lambda p, k, v: (p, v) if isinstance(v, int) else None
    >>> paths = list(search_paths(mm, return_path_if_int_leaf))
    >>> paths  # only the paths to the int leaves are returned
    [('a', 'b', 'c'), ('aaa', 'bbb')]

    The ``pkv_filt`` argument can use path, key, and/or value to define your search
    query. For example, let's extract all the paths that have depth at least 3.

    >>> paths = list(search_paths(mm, lambda p, k, v: len(p) >= 3))
    >>> paths
    [('a', 'b', 'c'), ('aa', 'bb', 'cc')]

    The reason it makes sense to have a paths search function, is that if you have the
    paths, you can than get the keys and values with them.

    >>> from functools import partial, reduce
    >>> path_get = lambda m, k: reduce(lambda m, k: m[k], k, m)
    >>> extract_paths = lambda m, paths: map(partial(path_get, m), paths)
    >>> vals = list(extract_paths(mm, paths))
    >>> vals
    [42, 'meaning of life']

    """
    _yield_func = partial(_path_matcher_yield_func, pkv_filt, None)
    walker = kv_walk(d, yield_func=_yield_func)
    yield from filter(None, walker)


def _path_matcher_yield_func(pkv_filt: PkvFilt, sentinel, p: PT, k: KT, v: VT):
    """Helper to make (picklable) yield_funcs for paths_matching (through partial)"""
    if pkv_filt(p, k, v):
        return p
    else:
        return sentinel


@add_as_attribute_of(search_paths)
def _mk_path_matcher(pkv_filt: PkvFilt, sentinel=None):
    """Make a yield_func that only yields paths that pass the pkv_filt,
    and a sentinel (by default, ``None``) otherwise"""
    return partial(_path_matcher_yield_func, pkv_filt, sentinel)


@add_as_attribute_of(search_paths)
def _mk_pkv_filt(
    filt: Callable[[Union[PT, KT, VT]], bool], kind: Literal['path', 'key', 'value']
) -> PkvFilt:
    """pkv_filt based on a ``filt`` that matches EITHER path, key, or value."""
    return partial(_pkv_filt, filt, kind)


def _pkv_filt(
    filt: Callable[[Union[PT, KT, VT]], bool],
    kind: Literal['path', 'key', 'value'],
    p: PT,
    k: KT,
    v: VT,
):
    """Helper to make (picklable) pkv_filt based on a ``filt`` that matches EITHER
    path, key, or value."""
    if kind == 'path':
        return filt(p)
    elif kind == 'key':
        return filt(k)
    elif kind == 'value':
        return filt(v)
    else:
        raise ValueError(f'Invalid kind: {kind}')
