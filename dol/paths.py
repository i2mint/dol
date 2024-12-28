"""Module for path (and path-like) object manipulation


Examples::

    >>> d = {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
    >>> list(path_filter(lambda p, k, v: v == 2, d))
    [('a', 'b', 'd')]
    >>> path_get(d, ('a', 'b', 'd'))
    2
    >>> path_set(d, ('a', 'b', 'd'), 4)
    >>> d
    {'a': {'b': {'c': 1, 'd': 4}, 'e': 3}}
    >>> path_set(d, ('a', 'b', 'new_ab_key'), 42)
    >>> d
    {'a': {'b': {'c': 1, 'd': 4, 'new_ab_key': 42}, 'e': 3}}

"""

from functools import wraps, partial
from dataclasses import dataclass
from typing import (
    Optional,
    Union,
    Callable,
    Any,
    Mapping,
    Iterable,
    Tuple,
    Literal,
    Iterator,
    KT,
    VT,
    TypeVar,
    Generator,
    TypeAlias,
    List,
    Dict,
)

from operator import getitem
import os

from dol.base import Store
from dol.util import lazyprop, add_as_attribute_of, max_common_prefix, safe_compile
from dol.trans import (
    store_decorator,
    kv_wrap,
    add_path_access,
    filt_iter,
    wrap_kvs,
    add_missing_key_handling,
)
from dol.dig import recursive_get_attr


KeyValueGenerator = Generator[tuple[KT, VT], None, None]
Path = TypeVar("Path")
PathExtenderFunc = Callable[[Path, KT], Path]
PathExtenderSpec = Union[str, PathExtenderFunc]
NestedMapping: TypeAlias = Mapping[KT, Union[VT, "NestedMapping[KT, VT]"]]


def separator_based_path_extender(path: Path, key: KT, sep: str) -> Path:
    """
    Extends a given path with a new key using the specified separator.
    If the path is empty, the key is returned as is.
    """
    return f"{path}{sep}{key}" if path else key


def ensure_path_extender_func(path_extender: PathExtenderSpec) -> PathExtenderFunc:
    """
    Ensure that the path_extender is a function that takes a path and a key and returns
    a new path."""
    if isinstance(path_extender, str):
        return partial(separator_based_path_extender, sep=path_extender)
    return path_extender


def flattened_dict_items(
    d,
    sep: PathExtenderSpec = ".",
    *,
    parent_path: Optional[Path] = None,
    visit_nested: Callable = lambda obj: isinstance(obj, Mapping),
) -> KeyValueGenerator:
    """
    Yield flattened key-value pairs from a nested dictionary.
    """
    path_extender = ensure_path_extender_func(sep)

    stable_kwargs = dict(sep=sep, visit_nested=visit_nested)

    for k, v in d.items():
        new_path = path_extender(parent_path, k)
        if visit_nested(v):
            yield from flattened_dict_items(v, parent_path=new_path, **stable_kwargs)
        else:
            yield new_path, v


def flatten_dict(
    d,
    sep: PathExtenderSpec = ".",
    *,
    parent_path: Optional[Path] = None,
    visit_nested: Callable = lambda obj: isinstance(obj, Mapping),
    egress: Callable[[KeyValueGenerator], Mapping] = dict,
):
    r"""
    Flatten a nested dictionary into a flat one, using key-paths as keys.

    See also `leaf_paths` for a related function that returns paths to leaf values.

    Args:
        d: The dictionary to flatten
        sep: The separator to use for joining keys, or a function that takes a path and
            a key and returns a new path.
        parent_path: The path to the parent of the current dict
        visit_nested: A function that returns True if a value should be visited
        egress: A function that takes a generator of key-value pairs and returns a mapping

    >>> d = {'a': {'b': 2}, 'c': 3}
    >>> flatten_dict(d)
    {'a.b': 2, 'c': 3}
    >>> flatten_dict(d, sep='/')
    {'a/b': 2, 'c': 3}

    """
    return egress(
        flattened_dict_items(
            d, sep=sep, parent_path=parent_path, visit_nested=visit_nested
        )
    )


def leaf_paths(
    d: NestedMapping,
    sep: PathExtenderSpec = ".",
    *,
    parent_path: Optional[Path] = None,
    egress: Callable[[KeyValueGenerator], Mapping] = dict,
) -> Dict[KT, Union[KT, Path]]:
    """
    Get a dictionary of leaf paths of a nested dictionary.

    Given a nested dictionary, returns a similarly structured dictionary where each
    leaf value is replaced by its flattened path. The 'sep' parameter can be either
    a string or a callable.

    Original use case: You used flatten_dict to flatten a nested dictionary, referencing
    your values with paths, but maybe you'd like to know what the paths that your
    nested dictionary is going to flatten to are. This function does that.
    The output is a dict with the same keys and structure as the input, but the leaf
    values are replaced by the paths that would be used to access them in a flat dict.

    Args:
        d: The nested dictionary to get the leaf paths from
        sep: The separator to use for joining keys, or a function that takes a path and
            a key and returns a new path.
        parent_path: The path to the parent of the current dict
        egress: A function that takes a generator of key-value pairs and returns a mapping

    Example:
    >>> leaf_paths({'a': {'b': 2}, 'c': 3})
    {'a': {'b': 'a.b'}, 'c': 'c'}

    >>> leaf_paths({'a': {'b': 2}, 'c': 3}, sep="/")
    {'a': {'b': 'a/b'}, 'c': 'c'}

    >>> leaf_paths({'a': {'b': 2}, 'c': 3}, sep=lambda p, k: f"{p}-{k}" if p else k)
    {'a': {'b': 'a-b'}, 'c': 'c'}
    """
    path_extender = ensure_path_extender_func(sep)

    return egress(_leaf_paths_recursive(d, path_extender, parent_path=parent_path))


def _leaf_paths_recursive(
    d: NestedMapping,
    path_extender: PathExtenderFunc,
    parent_path: Optional[Path] = None,
    *,
    visit_nested: Callable[[Any], bool] = lambda x: isinstance(x, dict),
) -> KeyValueGenerator:
    """
    A recursive generator that yields (key, value) pairs.
    A helper for leaf_paths.
    """
    for k, v in d.items():
        current_path = path_extender(parent_path, k)
        if visit_nested(v):
            yield k, dict(_leaf_paths_recursive(v, path_extender, current_path))
        else:
            yield k, current_path


path_sep = os.path.sep


def raise_on_error(d: dict):
    raise


def return_none_on_error(d: dict):
    return None


def return_empty_tuple_on_error(d: dict):
    return ()


OnErrorType = Union[Callable[[dict], Any], str]


# TODO: Could extend OnErrorType to be a dict with error class keys and callables or
#  strings as values. Then, the error class could be used to determine the error
#  handling strategy.
def _path_get(
    obj: Any,
    path,
    on_error: OnErrorType = raise_on_error,
    *,
    path_to_keys: Callable[[Any], Iterable] = None,
    get_value: Callable = getitem,
    caught_errors=(KeyError, IndexError),
):
    """Get elements of a mapping through a path to be called recursively.

    >>> _path_get({'a': {'b': 2}}, 'a')
    {'b': 2}
    >>> _path_get({'a': {'b': 2}}, ['a', 'b'])
    2
    >>> _path_get({'a': {'b': 2}}, ['a', 'c'])
    Traceback (most recent call last):
        ...
    KeyError: 'c'
    >>> _path_get({'a': {'b': 2}}, ['a', 'c'], lambda x: x)
    {'obj': {'a': {'b': 2}}, 'path': ['a', 'c'], 'result': {'b': 2}, 'k': 'c', 'error': KeyError('c')}

    # >>> assert _path_get({'a': {'b': 2}}, ['a', 'c'], lambda x: x) == {
    # ...     'mapping': {'a': {'b': 2}},
    # ...     'path': ['a', 'c'],
    # ...     'result': {'b': 2},
    # ...     'k': 'c',
    # ...     'error': KeyError('c')
    # ... }

    """

    if path_to_keys is not None:
        keys = path_to_keys(path)
    else:
        keys = path

    result = obj

    for k in keys:
        try:
            result = get_value(result, k)
        except caught_errors as error:
            if callable(on_error):
                return on_error(
                    dict(
                        obj=obj,
                        path=path,
                        result=result,
                        k=k,
                        error=error,
                    )
                )
            elif isinstance(on_error, str):
                # use on_error as a message, raising the same error class
                raise type(error)(on_error)
            else:
                raise ValueError(
                    f"on_error should be a callable (input is a dict) or a string. "
                    f"Was: {on_error}"
                )
    return result


def split_if_str(obj, sep="."):
    if isinstance(obj, str):
        return obj.split(sep)
    return obj


def separate_keys_with_separator(obj, sep="."):
    return map(cast_to_int_if_numeric_str, split_if_str(obj, sep))


def getitem(obj, k):
    return obj[k]


def get_attr_or_item(obj, k):
    """If ``k`` is a string, tries to get ``k`` as an attribute of ``obj`` first,
    and if that fails, gets it as ``obj[k]``"""
    if isinstance(k, str):
        try:
            return getattr(obj, k)
        except AttributeError:
            pass
        if str.isnumeric(k) and not isinstance(obj, Mapping):
            # if obj is not a mapping, and k is numeric, consider it to be int index
            k = int(k)
    return obj[k]


# ------------------------------------------------------------------------------
# key-path operations


# TODO: Needs a lot more documentation and tests to show how versatile it is
def path_get(
    obj: Any,
    path,
    on_error: OnErrorType = raise_on_error,
    *,
    sep=".",
    key_transformer=None,
    get_value: Callable = get_attr_or_item,
    caught_errors=(Exception,),
):
    """
    Get elements of a mapping through a path to be called recursively.

    :param obj: The object to get the path from
    :param path: The path to get
    :param on_error: The error handler to use (default: raise_on_error)
    :param sep: The separator to use if the path is a string
    :param key_transformer: A function to transform the keys of the path
    :param get_value: A function to get the value of a key in a mapping
    :param caught_errors: The errors to catch (default: Exception)

    It will

    - split a path into keys if it is a string, using the specified seperator ``sep``

    - consider string keys that are numeric as ints (convenient for lists)

    - get items also as attributes (attributes are checked for first for string keys)

    - catch all exceptions (that are subclasses of ``Exception``)

    >>> class A:
    ...      an_attribute = 42
    >>> path_get([1, [4, 5, {'a': A}], 3], [1, 2, 'a', 'an_attribute'])
    42

    By default, if ``path`` is a string, it will be split on ``sep``,
    which is ``'.'`` by default.

    >>> path_get([1, [4, 5, {'a': A}], 3], '1.2.a.an_attribute')
    42

    Note: The underlying function is ``_path_get``, but `path_get` has defaults and
    flexible input processing for more convenience.

    Note: ``path_get`` contains some ready-made ``OnErrorType`` functions in its
    attributes. For example, see how we can make ``path_get`` have the same behavior
    as ``dict.get`` by passing ``path_get.return_none_on_error`` as ``on_error``:

    >>> dd = path_get({}, 'no.keys', on_error=path_get.return_none_on_error)
    >>> dd is None
    True

    For example, ``path_get.raise_on_error``,
    ``path_get.return_none_on_error``, and ``path_get.return_empty_tuple_on_error``.

    """
    if isinstance(path, str) and sep is not None:
        path_to_keys = lambda x: x.split(sep)
    else:
        path_to_keys = lambda x: x
    if key_transformer is not None:
        _path_to_keys = path_to_keys
        path_to_keys = lambda path: map(key_transformer, _path_to_keys(path))

    return _path_get(
        obj,
        path,
        on_error=on_error,
        path_to_keys=path_to_keys,
        get_value=get_value,
        caught_errors=caught_errors,
    )


path_get.split_if_str = split_if_str
path_get.separate_keys_with_separator = separate_keys_with_separator
path_get.get_attr_or_item = get_attr_or_item
path_get.get_item = getitem
path_get.get_attr = getattr


@add_as_attribute_of(path_get)
def paths_getter(
    paths,
    obj=None,
    *,
    egress=dict,
    on_error: OnErrorType = raise_on_error,
    sep=".",
    key_transformer=None,
    get_value: Callable = get_attr_or_item,
    caught_errors=(Exception,),
):
    """
    Returns (path, values) pairs of the given paths in the given object.
    This is the "fan-out" version of ``path_get``, specifically designed to
    get multiple paths, returning the (path, value) pairs in a dict (by default),
    or via any pairs aggregator (``egress``) function.

    :param paths: The paths to get
    :param obj: The object to get the paths from
    :param egress: The egress function to use (default: dict)
    :param on_error: The error handler to use (default: raise_on_error)
    :param sep: The separator to use if the path is a string
    :param key_transformer: A function to transform the keys of the path
    :param get_value: A function to get the value of a key in a mapping
    :param caught_errors: The errors to catch (default: Exception)

    >>> obj = {'a': {'b': 1, 'c': 2}, 'd': 3}
    >>> paths = ['a.c', 'd']
    >>> paths_getter(paths, obj=obj)
    {'a.c': 2, 'd': 3}
    >>> path_extractor = paths_getter(paths)
    >>> path_extractor(obj)
    {'a.c': 2, 'd': 3}

    See that the paths are used as the keys of the returned dict.
    If you want to specify your own keys, you can simply specify `paths` as a dict
    whose keys are the keys you want, and whose values are the paths to get:

    >>> path_extractor_2 = paths_getter({'california': 'a.c', 'dreaming': 'd'})
    >>> path_extractor_2(obj)
    {'california': 2, 'dreaming': 3}

    """
    kwargs = dict(
        on_error=on_error,
        sep=sep,
        key_transformer=key_transformer,
        get_value=get_value,
        caught_errors=caught_errors,
    )
    if obj is None:
        return partial(paths_getter, paths, egress=egress, **kwargs)

    if isinstance(paths, Mapping):

        def pairs():
            for key, path in paths.items():
                yield key, path_get(obj, path=path, **kwargs)

    else:

        def pairs():
            for path in paths:
                yield path, path_get(obj, path=path, **kwargs)

    return egress(pairs())


@add_as_attribute_of(path_get)
def chain_of_getters(
    getters: Iterable[Callable], obj=None, k=None, *, caught_errors=(Exception,)
):
    """If ``k`` is a string, tries to get ``k`` as an attribute of ``obj`` first,
    and if that fails, gets it as ``obj[k]``"""
    if obj is None and k is None:
        return partial(chain_of_getters, getters, caught_errors=caught_errors)
    for getter in getters:
        try:
            return getter(obj, k)
        except caught_errors:
            pass


@add_as_attribute_of(path_get)
def cast_to_int_if_numeric_str(k):
    if isinstance(k, str) and str.isnumeric(k):
        return int(k)
    return k


@add_as_attribute_of(path_get)
def _raise_on_error(d: Any):
    """Raise the error that was caught."""
    raise


@add_as_attribute_of(path_get)
def _return_none_on_error(d: Any):
    """Return None if an error was caught."""
    return None


@add_as_attribute_of(path_get)
def _return_empty_tuple_on_error(d: Any):
    """Return an empty tuple if an error was caught."""
    return ()


@add_as_attribute_of(path_get)
def _return_new_dict_on_error(d: Any):
    """Return a new dict if an error was caught."""
    return dict()


from dol.explicit import KeysReader


# TODO: Nothing particular about paths here. It's just a collection of keys
# (see dol.explicit.ExplicitKeys) with a key_to_value function.
# TODO: Yet another "explicit" pattern, found in dol.explicit, dol.sources
# (e.g. ObjReader), and which can (but perhaps not should) really be completely
# implemented with a value decoder (the getter) in a wrap_kvs over a {k: k...} mapping.
class PathMappedData(KeysReader):
    """
    A collection of keys with a key_to_value function to lazy load values.

    `PathMappedData` is particularly useful in cases where you want to have a mapping
    that lazy-loads values for keys from an explicit collection.

    Keywords: Lazy-evaluation, Mapping

    Args:
        data: The mapping to extract data from
        paths: The paths to extract data from the mapping

    Example::

    >>> data = {
    ...     'a': {
    ...         'b': [{'c': 1}, {'c': 2}],
    ...         'd': 'bar'
    ...     }
    ... }
    >>> paths = ['a.d', 'a.b.0.c']
    >>>
    >>> d = PathMappedData(data, paths)
    >>> list(d)
    ['a.d', 'a.b.0.c']
    >>> d['a.d']
    'bar'
    >>> d['a.b.0.c']
    1

    Now, data does contain a key path for 'a.b.1.c':

    >>> d.getter(d.src, 'a.b.1.c')
    2

    But since we didn't mention it in our paths parameter, it will raise a KeyError
    if we try to access it via the `PathMappedData` object:

    >>> d['a.b.1.c']  # doctest: +ELLIPSIS +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    KeyError: "Key a.b.1.c was not found....key_collection attribute)"

    """

    def __init__(
        self,
        src: Mapping,
        key_collection,
        getter: Callable[[Mapping, Path], VT] = path_get,
        *,
        key_to_value: Callable[[Path], VT] = None,
    ) -> None:
        super().__init__(src, key_collection, getter)

    # def __getitem__(self, path: Path) -> VT:
    #     if path in self:
    #         return self.getter(self.data, path)
    #     else:
    #         raise KeyError(f'Path not found (in .paths attribute): {path}')

    # def __iter__(self) -> Iterator[Path]:
    #     yield from self.paths

    # def __len__(self) -> int:
    #     return len(self.paths)

    # def __contains__(self, path: Path) -> bool:
    #     return path in self.paths


# Note: Purposely didn't include any path validation to favor efficiency.
# Validation such as:
# if not key_path or not isinstance(key_path, Iterable):
#     raise ValueError(
#         f"Not a valid key path (should be an iterable with at least one element:"
#         f" {key_path}"
#     )
# TODO: Add possibility of producing different mappings according to the path/level.
#  For example, the new_mapping factory could be a list of factories, one for each
#  level, and/or take a path as an argument.
def path_set(
    d: Mapping,
    key_path: Iterable[KT],
    val: VT,
    *,
    sep: str = ".",
    new_mapping: Callable[[], VT] = dict,
):
    """
    Sets a val to a path of keys.

    :param d: The mapping to set the value in
    :param key_path: The path of keys to set the value to
    :param val: The value to set
    :param sep: The separator to use if the path is a string
    :param new_mapping: callable that returns a new mapping to use when key is not found
    :return:

    >>> d = {'a': 1, 'b': {'c': 2}}
    >>> path_set(d, ['b', 'e'], 42)
    >>> d
    {'a': 1, 'b': {'c': 2, 'e': 42}}

    >>> input_dict = {
    ...   "a": {
    ...     "c": "val of a.c",
    ...     "b": 1,
    ...   },
    ...   "10": 10,
    ...   "b": {
    ...     "B": {
    ...       "AA": 3
    ...     }
    ...   }
    ... }
    >>>
    >>> path_set(input_dict, ('new', 'key', 'path'), 7)
    >>> input_dict  # doctest: +NORMALIZE_WHITESPACE
    {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}},
    'new': {'key': {'path': 7}}}

    You can also use a string as a path, with a separator:

    >>> path_set(input_dict, 'new/key/old/path', 8, sep='/')
    >>> input_dict  # doctest: +NORMALIZE_WHITESPACE
    {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}},
    'new': {'key': {'path': 7, 'old': {'path': 8}}}}

    If you specify a string path and a non-None separator, the separator will be used
    to split the string into a list of keys. The default separator is ``sep='.'``.

    >>> path_set(input_dict, 'new.key', 'new val')
    >>> input_dict  # doctest: +NORMALIZE_WHITESPACE
    {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}},
    'new': {'key': 'new val'}}

    You can also specify a different ``new_mapping`` factory, which will be used to
    create new mappings when a key is missing. The default is ``dict``.

    >>> from collections import OrderedDict
    >>> input_dict = {}
    >>> path_set(input_dict, 'new.key', 42, new_mapping=OrderedDict)
    >>> input_dict  # doctest: +NORMALIZE_WHITESPACE
    {'new': OrderedDict([('key', 42)])}

    """
    if isinstance(key_path, str) and sep is not None:
        key_path = key_path.split(sep)

    first_key, *remaining_keys = key_path
    if len(key_path) == 1:  # base case
        d[first_key] = val
    else:
        if first_key not in d:
            d[first_key] = new_mapping()
        path_set(d[first_key], remaining_keys, val)


# TODO: Nice to have: Edits can be a nested dict, not necessarily a flat path-value one.
Edits = Union[Mapping[Path, VT], Iterable[Tuple[Path, VT]]]


def path_edit(d: Mapping, edits: Edits = ()) -> Mapping:
    """Make a series of (in place) edits to a Mapping, specifying `(path, value)` pairs.


    Args:
        d (Mapping): The mapping to edit.
        edits: An iterable of ``(path, value)`` tuples, or ``path: value`` Mapping.

    Returns:
        Mapping: The edited mapping.

    >>> d = {'a': 1}
    >>> path_edit(d, [(['b', 'c'], 2), ('d.e.f', 3)])
    {'a': 1, 'b': {'c': 2}, 'd': {'e': {'f': 3}}}

    Changes happened also inplace (so if you don't want that, make a deepcopy first):

    >>> d
    {'a': 1, 'b': {'c': 2}, 'd': {'e': {'f': 3}}}

    You can also pass a dict of edits.

    >>> path_edit(d, {'a': 4, 'd.e.f': 5})
    {'a': 4, 'b': {'c': 2}, 'd': {'e': {'f': 5}}}

    """

    if isinstance(edits, Mapping):
        edits = list(edits.items())
    for path, value in edits:
        path_set(d, path, value)
    return d


from dol.base import kv_walk


PT = TypeVar("PT")  # Path Type
PkvFilt = Callable[[PT, KT, VT], bool]


#
def path_filter(
    pkv_filt: PkvFilt,
    d: Mapping,
    *,
    leafs_only: bool = True,
    breadth_first: bool = False,
) -> Iterator[PT]:
    """Walk a dict, yielding paths to values that pass the ``pkv_filt``

    :param pkv_filt: A function that takes a path, key, and value, and returns
        ``True`` if the path should be yielded, and ``False`` otherwise
    :param d: The ``Mapping`` to walk (scan through)
    :param leafs_only: Whether to yield only paths to leafs (default), or to yield
        paths to all values that pass the ``pkv_filt``.
    :param breadth_first: Whether to perform breadth-first traversal
        (instead of the default depth-first traversal).
    :return: An iterator of paths to values that pass the ``pkv_filt``

    Example::

    >>> d = {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
    >>> list(path_filter(lambda p, k, v: v == 2, d))
    [('a', 'b', 'd')]

    >>> mm = {
    ...     'a': {'b': {'c': 42}},
    ...     'aa': {'bb': {'cc': 'meaning of life'}},
    ...     'aaa': {'bbb': 314},
    ... }
    >>> return_path_if_int_leaf = lambda p, k, v: (p, v) if isinstance(v, int) else None
    >>> paths = list(path_filter(return_path_if_int_leaf, mm))
    >>> paths  # only the paths to the int leaves are returned
    [('a', 'b', 'c'), ('aaa', 'bbb')]

    The ``pkv_filt`` argument can use path, key, and/or value to define your search
    query. For example, let's extract all the paths that have depth at least 3.

    >>> paths = list(path_filter(lambda p, k, v: len(p) >= 3, mm))
    >>> paths
    [('a', 'b', 'c'), ('aa', 'bb', 'cc')]

    The rationale for ``path_filter`` yielding matching paths, and not values or keys,
    is that if you have the paths, you can than get the keys and values with them,
    using ``path_get``.

    >>> from functools import partial, reduce
    >>> path_get = lambda m, k: reduce(lambda m, k: m[k], k, m)
    >>> extract_paths = lambda m, paths: map(partial(path_get, m), paths)
    >>> vals = list(extract_paths(mm, paths))
    >>> vals
    [42, 'meaning of life']

    Note: pkv_filt is first to match the order of the arguments of the
    builtin filter function.
    """
    _leaf_yield = partial(_path_matcher_leaf_yield, pkv_filt, None)
    kwargs = dict(leaf_yield=_leaf_yield, breadth_first=breadth_first)
    if not leafs_only:
        kwargs["branch_yield"] = _leaf_yield
    walker = kv_walk(d, **kwargs)
    yield from filter(None, walker)


# backwards compatibility quasi-alias (arguments are flipped)
def search_paths(
    d: Mapping,
    pkv_filt: PkvFilt,
    *,
    leafs_only: bool = True,
    breadth_first: bool = False,
) -> Iterator[PT]:
    """backwards compatibility quasi-alias (arguments are flipped)
    Use path_filter instead, since search_paths will be deprecated.
    """
    return path_filter(pkv_filt, d, leafs_only=leafs_only, breadth_first=breadth_first)


def _path_matcher_leaf_yield(pkv_filt: PkvFilt, sentinel, p: PT, k: KT, v: VT):
    """Helper to make (picklable) leaf_yields for paths_matching (through partial)"""
    if pkv_filt(p, k, v):
        return p
    else:
        return sentinel


@add_as_attribute_of(path_filter)
def _mk_path_matcher(pkv_filt: PkvFilt, sentinel=None):
    """Make a leaf_yield that only yields paths that pass the pkv_filt,
    and a sentinel (by default, ``None``) otherwise"""
    return partial(_path_matcher_leaf_yield, pkv_filt, sentinel)


@add_as_attribute_of(path_filter)
def _mk_pkv_filt(
    filt: Callable[[Union[PT, KT, VT]], bool], kind: Literal["path", "key", "value"]
) -> PkvFilt:
    """pkv_filt based on a ``filt`` that matches EITHER path, key, or value."""
    return partial(_pkv_filt, filt, kind)


def _pkv_filt(
    filt: Callable[[Union[PT, KT, VT]], bool],
    kind: Literal["path", "key", "value"],
    p: PT,
    k: KT,
    v: VT,
):
    """Helper to make (picklable) pkv_filt based on a ``filt`` that matches EITHER
    path, key, or value."""
    if kind == "path":
        return filt(p)
    elif kind == "key":
        return filt(k)
    elif kind == "value":
        return filt(v)
    else:
        raise ValueError(f"Invalid kind: {kind}")


@dataclass
class KeyPath:
    """
    A key mapper that converts from an iterable key (default tuple) to a string
    (given a path-separator str)

    Args:
        path_sep: The path separator (used to make string paths from iterable paths and
            visa versa
        _path_type: The type of the outcoming (inner) path. But really, any function to
        convert from a list to
            the outer path type we want.

    With ``'/'`` as a separator:

    >>> kp = KeyPath(path_sep='/')
    >>> kp._key_of_id(('a', 'b', 'c'))
    'a/b/c'
    >>> kp._id_of_key('a/b/c')
    ('a', 'b', 'c')

    With ``'.'`` as a separator:

    >>> kp = KeyPath(path_sep='.')
    >>> kp._key_of_id(('a', 'b', 'c'))
    'a.b.c'
    >>> kp._id_of_key('a.b.c')
    ('a', 'b', 'c')
    >>> kp = KeyPath(path_sep=':::', _path_type=dict.fromkeys)
    >>> _id = dict.fromkeys('abc')
    >>> _id
    {'a': None, 'b': None, 'c': None}
    >>> kp._key_of_id(_id)
    'a:::b:::c'
    >>> kp._id_of_key('a:::b:::c')
    {'a': None, 'b': None, 'c': None}

    Calling a ``KeyPath`` instance on a store wraps it so we can have path access to
    it.

    >>> s = {'a': {'b': {'c': 42}}}
    >>> s['a']['b']['c']
    42
    >>> # Now let's wrap the store
    >>> s = KeyPath('.')(s)
    >>> s['a.b.c']
    42
    >>> s['a.b.c'] = 3.14
    >>> s['a.b.c']
    3.14
    >>> del s['a.b.c']
    >>> s
    {'a': {'b': {}}}

    Note: ``KeyPath`` enables you to read with paths when all the keys of the paths
    are valid (i.e. have a value), but just as with a ``dict``, it will not create
    intermediate nested values for you (as for example, you could make for yourself
    using  ``collections.defaultdict``).

    """

    path_sep: str = path_sep
    _path_type: Union[type, callable] = tuple

    def _key_of_id(self, _id):
        if not isinstance(_id, str):
            return self.path_sep.join(_id)
        else:
            return _id

    def _id_of_key(self, k):
        return self._path_type(k.split(self.path_sep))

    def __call__(self, store):
        path_accessible_store = add_path_access(store, path_type=self._path_type)
        return kv_wrap(self)(path_accessible_store)


# ------------------------------------------------------------------------------


class PrefixRelativizationMixin:
    """
    Mixin that adds a intercepts the _id_of_key an _key_of_id methods, transforming absolute keys to relative ones.
    Designed to work with string keys, where absolute and relative are relative to a _prefix attribute
    (assumed to exist).
    The cannonical use case is when keys are absolute file paths, but we want to identify data through relative paths.
    Instead of referencing files through an absolute path such as
        /A/VERY/LONG/ROOT/FOLDER/the/file/we.want
    we can instead reference the file as
        the/file/we.want

    Note though, that PrefixRelativizationMixin can be used, not only for local paths,
    but when ever a string reference is involved.
    In fact, not only strings, but any key object that has a __len__, __add__, and subscripting.

    When subclassed, should be placed before the class defining _id_of_key an _key_of_id.
    Also, assumes that a (string) _prefix attribute will be available.

    >>> from dol.base import Store
    >>> from collections import UserDict
    >>>
    >>> class MyStore(PrefixRelativizationMixin, Store):
    ...     def __init__(self, store, _prefix='/root/of/data/'):
    ...         super().__init__(store)
    ...         self._prefix = _prefix
    ...
    >>> s = MyStore(store=dict())  # using a dict as our store
    >>> s['foo'] = 'bar'
    >>> assert s['foo'] == 'bar'
    >>> s['too'] = 'much'
    >>> assert list(s.keys()) == ['foo', 'too']
    >>> # Everything looks normal, but are the actual keys behind the hood?
    >>> s._id_of_key('foo')
    '/root/of/data/foo'
    >>> # see when iterating over s.items(), we get the interface view:
    >>> list(s.items())
    [('foo', 'bar'), ('too', 'much')]
    >>> # but if we ask the store we're actually delegating the storing to, we see what the keys actually are.
    >>> s.store.items()
    dict_items([('/root/of/data/foo', 'bar'), ('/root/of/data/too', 'much')])
    """

    _prefix_attr_name = "_prefix"

    @lazyprop
    def _prefix_length(self):
        return len(getattr(self, self._prefix_attr_name))

    def _id_of_key(self, k):
        return getattr(self, self._prefix_attr_name) + k

    def _key_of_id(self, _id):
        return _id[self._prefix_length :]


class PrefixRelativization(PrefixRelativizationMixin):
    """A key wrap that allows one to interface with absolute paths through relative paths.
    The original intent was for local files. Instead of referencing files through an absolute path such as:

        */A/VERY/LONG/ROOT/FOLDER/the/file/we.want*

    we can instead reference the file as:

        *the/file/we.want*

    But PrefixRelativization can be used, not only for local paths, but when ever a string reference is involved.
    In fact, not only strings, but any key object that has a __len__, __add__, and subscripting.
    """

    def __init__(self, _prefix=""):
        self._prefix = _prefix


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

    __slots__ = ("_key_collection",)

    def __init__(self, key_collection, _prefix=None):
        # TODO: Find a better way to avoid the circular import
        from dol.explicit import ExplicitKeys  # here to avoid circular imports

        if _prefix is None:
            _prefix = max_common_prefix(key_collection)
        store = ExplicitKeys(key_collection=key_collection)
        self._prefix = _prefix
        super().__init__(store=store)


@store_decorator
def mk_relative_path_store(
    store_cls=None,
    *,
    name=None,
    with_key_validation=False,
    prefix_attr="_prefix",
):
    """

    Args:
        store_cls: The base store to wrap (subclass)
        name: The name of the new store (by default 'RelPath' + store_cls.__name__)
        with_key_validation: Whether keys should be validated upon access (store_cls must have an is_valid_key method

    Returns: A new class that uses relative paths (i.e. where _prefix is automatically added to incoming keys,
        and the len(_prefix) first characters are removed from outgoing keys.

    >>> # The dynamic way (if you try this at home, be aware of the pitfalls of the dynamic way
    >>> # -- but don't just believe the static dogmas).
    >>> MyStore = mk_relative_path_store(dict)  # wrap our favorite store: A dict.
    >>> s = MyStore()  # make such a store
    >>> s._prefix = '/ROOT/'
    >>> s['foo'] = 'bar'
    >>> dict(s.items())  # gives us what you would expect
    {'foo': 'bar'}
    >>>  # but under the hood, the dict we wrapped actually contains the '/ROOT/' prefix
    >>> dict(s.store)
    {'/ROOT/foo': 'bar'}
    >>>
    >>> # The static way: Make a class that will integrate the _prefix at construction time.
    >>> class MyStore(mk_relative_path_store(dict)):  # Indeed, mk_relative_path_store(dict) is a class you can subclass
    ...     def __init__(self, _prefix, *args, **kwargs):
    ...         self._prefix = _prefix

    You can choose the name you want that prefix to have as an attribute (we'll still make
    a hidden '_prefix' attribute for internal use, but at least you can have an attribute with the
    name you want.

    >>> MyRelStore = mk_relative_path_store(dict, prefix_attr='rootdir')
    >>> s = MyRelStore()
    >>> s.rootdir = '/ROOT/'

    >>> s['foo'] = 'bar'
    >>> dict(s.items())  # gives us what you would expect
    {'foo': 'bar'}
    >>>  # but under the hood, the dict we wrapped actually contains the '/ROOT/' prefix
    >>> dict(s.store)
    {'/ROOT/foo': 'bar'}

    """
    # name = name or ("RelPath" + store_cls.__name__)
    # __module__ = __module__ or getattr(store_cls, "__module__", None)

    if name is not None:
        from warnings import warn

        warn(
            f"The use of name argumment is deprecated. Use __name__ instead",
            DeprecationWarning,
        )

    cls = type(store_cls.__name__, (PrefixRelativizationMixin, Store), {})

    @wraps(store_cls.__init__)
    def __init__(self, *args, **kwargs):
        Store.__init__(self, store=store_cls(*args, **kwargs))
        prefix = recursive_get_attr(self.store, prefix_attr, "")
        setattr(
            self, prefix_attr, prefix
        )  # TODO: Might need descriptor to enable assignment

    cls.__init__ = __init__

    if prefix_attr != "_prefix":
        assert not hasattr(store_cls, "_prefix"), (
            f"You already have a _prefix attribute, "
            f"but want the prefix name to be {prefix_attr}. "
            f"That's not going to be easy for me."
        )

        # if not hasattr(cls, prefix_attr):
        #     warn(f"You said you wanted prefix_attr='{prefix_attr}', "
        #          f"but {cls} (the wrapped class) doesn't have a '{prefix_attr}'. "
        #          f"I'll let it slide because perhaps the attribute is dynamic. But I'm warning you!!")

        @property
        def _prefix(self):
            return getattr(self, prefix_attr)

        cls._prefix = _prefix

    if with_key_validation:
        assert hasattr(store_cls, "is_valid_key"), (
            "If you want with_key_validation=True, "
            "you'll need a method called is_valid_key to do the validation job"
        )

        def _id_of_key(self, k):
            _id = super(cls, self)._id_of_key(k)
            if self.store.is_valid_key(_id):
                return _id
            else:
                raise KeyError(
                    f"Key not valid (usually because does not exist or access not permitted): {k}"
                )

        cls._id_of_key = _id_of_key

    # if __module__ is not None:
    #     cls.__module__ = __module__

    # print(callable(cls))

    return cls


# TODO: Intended to replace the init-less PrefixRelativizationMixin
#  (but should change name if so, since Mixins shouldn't have inits)
class RelativePathKeyMapper:
    def __init__(self, prefix):
        self._prefix = prefix
        self._prefix_length = len(self._prefix)

    def _id_of_key(self, k):
        return self._prefix + k

    def _key_of_id(self, _id):
        return _id[self._prefix_length :]


@store_decorator
def prefixless_view(store=None, *, prefix=None):
    key_mapper = RelativePathKeyMapper(prefix)
    return wrap_kvs(
        store, id_of_key=key_mapper._id_of_key, key_of_id=key_mapper._key_of_id
    )


def _fallback_startswith(iterable, prefix):
    """Returns True iff iterable starts with prefix.
    Compares the first items of iterable and prefix iteratively.
    It can be terribly inefficient though, so it's best to use it only when you have to.
    """
    iter_iterable = iter(iterable)
    iter_prefix = iter(prefix)

    for prefix_item in iter_prefix:
        try:
            # Get the next item from iterable
            item = next(iter_iterable)
        except StopIteration:
            # If we've reached the end of iterable, return False
            return False

        if item != prefix_item:
            # If any pair of items are unequal, return False
            return False

    # If we've checked every item in prefix without returning, return True
    return True


# TODO: Routing pattern. Make plugin architecture.
# TODO: Add faster option for lists and tuples that are sizable and sliceable
def _startswith(iterable, prefix):
    """Returns True iff iterable starts with prefix.
    If prefix is a string, `str.startswith` is used, otherwise, the function
    will compare the first items of iterable and prefix iteratively.

    >>> _startswith('apple', 'app')
    True
    >>> _startswith('crapple', 'app')
    False
    >>> _startswith([1,2,3,4], [1,2])
    True
    >>> _startswith([0, 1,2,3,4], [1,2])
    False
    >>> _startswith([1,2,3,4], [])
    True
    """
    if isinstance(prefix, str):
        return iterable.startswith(prefix)
    else:
        return _fallback_startswith(iterable, prefix)


def _prefix_filter(store, prefix: str):
    """Filter the store to have only keys that start with prefix"""
    return filt_iter(store, filt=partial(_startswith, prefix=prefix))


def _prefix_filter_with_relativization(store, prefix: str):
    """Filter the store to have only keys that start with prefix"""
    return prefixless_view(_prefix_filter(store, prefix), prefix=prefix)


@store_decorator
def add_prefix_filtering(store=None, *, relativize_prefix: bool = False):
    """Add prefix filtering to a store.

    >>> d = {'a/b': 1, 'a/c': 2, 'd/e': 3, 'f': 4}
    >>> s = add_prefix_filtering(d)
    >>> assert s['a/'] == {'a/b': 1, 'a/c': 2}

    Demo usage on a `Mapping` type:

    >>> from collections import UserDict
    >>> D = add_prefix_filtering(UserDict)
    >>> s = D(d)
    >>> assert s['a/'] == {'a/b': 1, 'a/c': 2}

    """
    __prefix_filter = _prefix_filter
    if relativize_prefix:
        __prefix_filter = _prefix_filter_with_relativization
    return add_missing_key_handling(store, missing_key_callback=__prefix_filter)


@store_decorator
def handle_prefixes(
    store=None,
    *,
    prefix=None,
    filter_prefix: bool = True,
    relativize_prefix: bool = True,
    default_prefix="",
):
    """A store decorator that handles prefixes.

    If aggregates several prefix-related functionalities. It will (by default)

    - Filter the store so that only the keys starting with given prefix are accessible.

    - Relativize the keys (provide a view where the prefix is removed from the keys)

    Args:
        store: The store to wrap
        prefix: The prefix to use. If None and the store is an instance (not type),
                will take the longest common prefix as the prefix.
        filter_prefix: Whether to filter out keys that don't start with the prefix
        relativize_prefix: Whether to relativize the prefix
        default_prefix: The default prefix to use if no prefix is given and the store
                        is a type (not instance)

    >>> d = {'/ROOT/of/every/thing': 42, '/ROOT/of/this/too': 0}
    >>> dd = handle_prefixes(d, prefix='/ROOT/of/')
    >>> dd['foo'] = 'bar'
    >>> dict(dd.items())  # gives us what you would expect
    {'every/thing': 42, 'this/too': 0, 'foo': 'bar'}
    >>> dict(dd.store)  # but see where the underlying store actually wrote 'bar':
    {'/ROOT/of/every/thing': 42, '/ROOT/of/this/too': 0, '/ROOT/of/foo': 'bar'}

    """
    if prefix is None:
        if isinstance(store, type):
            raise TypeError(
                f"I can only infer prefix from a store instance, not a type: {store}"
            )
        prefix = max_common_prefix(store, default=default_prefix)
    if filter_prefix:
        store = filt_iter(store, filt=lambda k: k.startswith(prefix))
    if relativize_prefix:
        store = prefixless_view(store, prefix=prefix)
    return store


# TODO: Enums introduce a ridiculous level of complexity here.
#  Learn them of remove them!!

from dol.naming import StrTupleDict
from enum import Enum


class PathKeyTypes(Enum):
    str = "str"
    dict = "dict"
    tuple = "tuple"
    namedtuple = "namedtuple"


path_key_type_for_type = {
    str: PathKeyTypes.str,
    dict: PathKeyTypes.dict,
    tuple: PathKeyTypes.tuple,
}

_method_names_for_path_type = {
    PathKeyTypes.str: {
        "_id_of_key": StrTupleDict.simple_str_to_str,
        "_key_of_id": StrTupleDict.str_to_simple_str,
    },
    PathKeyTypes.dict: {
        "_id_of_key": StrTupleDict.dict_to_str,
        "_key_of_id": StrTupleDict.str_to_dict,
    },
    PathKeyTypes.tuple: {
        "_id_of_key": StrTupleDict.tuple_to_str,
        "_key_of_id": StrTupleDict.str_to_tuple,
    },
    PathKeyTypes.namedtuple: {
        "_id_of_key": StrTupleDict.namedtuple_to_str,
        "_key_of_id": StrTupleDict.str_to_namedtuple,
    },
}


#
# def str_to_simple_str(self, s: str):
#     return self.sep.join(*self.str_to_tuple(s))
#
#
# def simple_str_to_str(self, ss: str):
#     self.tuple_to_str(self.si)


# TODO: Add key and id type validation
def str_template_key_trans(
    template: str,
    key_type: Union[PathKeyTypes, type],
    format_dict=None,
    process_kwargs=None,
    process_info_dict=None,
    named_tuple_type_name="NamedTuple",
    sep: str = path_sep,
):
    """Make a key trans object that translates from a string _id to a dict, tuple, or namedtuple key (and back)"""

    assert (
        key_type in PathKeyTypes
    ), f"key_type was {key_type}. Needs to be one of these: {', '.join(PathKeyTypes)}"

    class PathKeyMapper(StrTupleDict): ...

    setattr(
        PathKeyMapper,
        "_id_of_key",
        _method_names_for_path_type[key_type]["_id_of_key"],
    )
    setattr(
        PathKeyMapper,
        "_key_of_id",
        _method_names_for_path_type[key_type]["_key_of_id"],
    )

    key_trans = PathKeyMapper(
        template,
        format_dict,
        process_kwargs,
        process_info_dict,
        named_tuple_type_name,
        sep,
    )

    return key_trans


str_template_key_trans.method_names_for_path_type = _method_names_for_path_type
str_template_key_trans.key_types = PathKeyTypes


# TODO: Merge with mk_relative_path_store
def rel_path_wrap(o, _prefix):
    """
    Args:
        o: An object to be wrapped
        _prefix: The _prefix to use for key wrapping (will remove it from outcoming keys and add to ingoing keys.

    >>> # The dynamic way (if you try this at home, be aware of the pitfalls of the dynamic way
    >>> # -- but don't just believe the static dogmas).
    >>> d = {'/ROOT/of/every/thing': 42, '/ROOT/of/this/too': 0}
    >>> dd = rel_path_wrap(d, '/ROOT/of/')
    >>> dd['foo'] = 'bar'
    >>> dict(dd.items())  # gives us what you would expect
    {'every/thing': 42, 'this/too': 0, 'foo': 'bar'}
    >>>  # but under the hood, the dict we wrapped actually contains the '/ROOT/' prefix
    >>> dict(dd.store)
    {'/ROOT/of/every/thing': 42, '/ROOT/of/this/too': 0, '/ROOT/of/foo': 'bar'}
    >>>
    >>> # The static way: Make a class that will integrate the _prefix at construction time.
    >>> class MyStore(mk_relative_path_store(dict)):  # Indeed, mk_relative_path_store(dict) is a class you can subclass
    ...     def __init__(self, _prefix, *args, **kwargs):
    ...         self._prefix = _prefix

    """

    from dol import kv_wrap

    trans_obj = RelativePathKeyMapper(_prefix)
    return kv_wrap(trans_obj)(o)


# mk_relative_path_store_cls = mk_relative_path_store  # alias

## Alternative to mk_relative_path_store that doesn't make lint complain (but the repr shows MyStore, not name)
# def mk_relative_path_store_alt(store_cls, name=None):
#     if name is None:
#         name = 'RelPath' + store_cls.__name__
#
#     class MyStore(PrefixRelativizationMixin, Store):
#         @wraps(store_cls.__init__)
#         def __init__(self, *args, **kwargs):
#             super().__init__(store=store_cls(*args, **kwargs))
#             self._prefix = self.store._prefix
#     MyStore.__name__ = name
#
#     return MyStore


## Alternative to StrTupleDict (staging here for now, but should replace when ready)

import re
import string
from collections import namedtuple
from functools import wraps


def _return_none_if_none_input(func):
    """Wraps a method function, making it return `None` if the input is `None`.

    (More precisely, it will return `None` if the first (non-instance) input is `None`.

    >>> class Foo:
    ...     @_return_none_if_none_input
    ...     def bar(self, x, y=1):
    ...         return x + y
    >>> foo = Foo()
    >>> foo.bar(2)
    3
    >>> assert foo.bar(None) is None
    >>> assert foo.bar(x=None) is None

    Note: On the other hand, this will not return `None`, but should:
    ``foo.bar(y=3, x=None)``. To achieve this, we'd need to look into the signature,
    which seems like overkill and I might not want that systematic overhead in my
    methods.
    """

    @wraps(func)
    def _func(self, *args, **kwargs):
        if args and args[0] is None:
            return None
        elif kwargs and next(iter(kwargs.values())) is None:
            return None
        else:
            return func(self, *args, **kwargs)

    return _func


from typing import Iterable, Tuple

string_formatter = string.Formatter()


def string_unparse(parsing_result: Iterable[Tuple[str, str, str, str]]):
    """The inverse of string.Formatter.parse

    Will ravel

    >>> import string
    >>> formatter = string.Formatter()
    >>> string_unparse(formatter.parse('literal{name!c:spec}'))
    'literal{name!c:spec}'
    """
    reconstructed = ""
    for literal_text, field_name, format_spec, conversion in parsing_result:
        reconstructed += literal_text
        if field_name is not None:
            field = f"{{{field_name}"
            if conversion:
                assert (
                    len(conversion) == 1
                ), f"conversion can only be a single character: {conversion=}"
                field += f"!{conversion}"
            if format_spec:
                field += f":{format_spec}"
            field += "}"
            reconstructed += field
    return reconstructed


def _field_names(string_template):
    """
    Returns the field names in a string template.

    >>> _field_names("{name} is {age} years old.")
    ('name', 'age')
    """
    parsing_result = string_formatter.parse(string_template)
    return tuple(
        field_name for _, field_name, _, _ in parsing_result if field_name is not None
    )


def identity(x):
    return x


from dol.trans import KeyCodec, filt_iter
from inspect import signature

# Codec = namedtuple('Codec', 'encoder decoder')
FieldTypeNames = Literal["str", "dict", "tuple", "namedtuple", "simple_str", "single"]


# TODO: Make and use _return_none_if_none_input or not?
# TODO: Change to dataclass with 3.10+ (to be able to do KW_ONLY)
# TODO: Should be refactored and generalized to be able to automatically handle
#   all combinations of FieldTypeNames (and possibly open-close these as well?)
#   It's a "path finder" meshed pattern.
# TODO: Do we really want to allow field_patterns to be included in the template (the `{name:pattern}` options)?
#  Normally, this is used for string GENERATION as `{name:format}`, which is still useful for us here too.
#  The counter argument is that the main usage of KeyTemplate is not actually
#  generation, but extraction. Further, the format language is not as general as simply
#  using a format_field = {field: cast_function, ...} argument.
#  My decision would be to remove any use of the `{name:X}` form in the base class,
#  and have classmethods specialized for short-hand versions that use `name:regex` or
#  `name:format`, ...
class KeyTemplate:
    """A class for parsing and generating keys based on a template.

    Args:
        template: A template string with fields to be extracted or filled in.
        field_patterns: A dictionary of field names and their regex patterns.
        simple_str_sep: A separator string for simple strings (i.e. strings without
            fields).
        namedtuple_type_name: The name of the namedtuple type to use for namedtuple
            fields.
        dflt_pattern: The default pattern to use for fields that don't have a pattern
            specified.
        to_str_funcs: A dictionary of field names and their functions to convert them
            to strings.
        from_str_funcs: A dictionary of field names and their functions to convert
            them from strings.

    Examples:

    >>> st = KeyTemplate(
    ...     'root/{name}/v_{version}.json',
    ...     field_patterns={'version': r'\d+'},
    ...     from_str_funcs={'version': int},
    ... )

    And now you have a template that can be used to convert between various
    representations of the template: You can extract fields from strings, generate
    strings from fields, etc.

    >>> st.str_to_dict("root/dol/v_9.json")
    {'name': 'dol', 'version': 9}
    >>> st.dict_to_str({'name': 'meshed', 'version': 42})
    'root/meshed/v_42.json'
    >>> st.dict_to_tuple({'name': 'meshed', 'version': 42})
    ('meshed', 42)
    >>> st.tuple_to_dict(('i2', 96))
    {'name': 'i2', 'version': 96}
    >>> st.str_to_tuple("root/dol/v_9.json")
    ('dol', 9)
    >>> st.tuple_to_str(('front', 11))
    'root/front/v_11.json'
    >>> st.str_to_namedtuple("root/dol/v_9.json")
    NamedTuple(name='dol', version=9)
    >>> st.str_to_simple_str("root/dol/v_9.json")
    'dol,9'
    >>> st_clone = st.clone(simple_str_sep='/')
    >>> st_clone.str_to_simple_str("root/dol/v_9.json")
    'dol/9'


    With ``st.key_codec``, you can make a ``KeyCodec`` for the given source (decoded)
    and target (encoded) types.
    A `key_codec` is a codec; it has an encoder and a decoder.

    >>> key_codec = st.key_codec('tuple', 'str')
    >>> encoder, decoder = key_codec
    >>> decoder('root/dol/v_9.json')
    ('dol', 9)
    >>> encoder(('dol', 9))
    'root/dol/v_9.json'

    If you have a ``Mapping``, you can use ``key_codec`` as a decorator to wrap
    the mapping with a key mappings.

    >>> store = {
    ...     'root/meshed/v_151.json': '{"downloads": 41, "type": "productivity"}',
    ...     'root/dol/v_9.json': '{"downloads": 132, "type": "utility"}',
    ... }
    >>>
    >>> accessor = key_codec(store)
    >>> list(accessor)
    [('meshed', 151), ('dol', 9)]
    >>> accessor['i2', 4] = '{"downloads": 274, "type": "utility"}'
    >>> list(store)
    ['root/meshed/v_151.json', 'root/dol/v_9.json', 'root/i2/v_4.json']
    >>> store['root/i2/v_4.json']
    '{"downloads": 274, "type": "utility"}'

    Note: If your store contains keys that don't fit the format, key_codec will
    raise a ``ValueError``. To remedy this, you can use the ``st.filt_iter`` to
    filter out keys that don't fit the format, before you wrap the store with
    ``st.key_codec``.

    >>> store = {
    ...     'root/meshed/v_151.json': '{"downloads": 41, "type": "productivity"}',
    ...     'root/dol/v_9.json': '{"downloads": 132, "type": "utility"}',
    ...     'root/not/the/right/format': "something else"
    ... }
    >>> accessor = st.filt_iter('str')(store)
    >>> list(accessor)
    ['root/meshed/v_151.json', 'root/dol/v_9.json']
    >>> accessor = st.key_codec('tuple', 'str')(st.filt_iter('str')(store))
    >>> list(accessor)
    [('meshed', 151), ('dol', 9)]
    >>> accessor['dol', 9]
    '{"downloads": 132, "type": "utility"}'

    You can also ask any (handled) combination of field types:

    >>> key_codec = st.key_codec('tuple', 'dict')
    >>> key_codec.encoder(('i2', 96))
    {'name': 'i2', 'version': 96}
    >>> key_codec.decoder({'name': 'fantastic', 'version': 4})
    ('fantastic', 4)

    """

    _formatter = string_formatter

    def __init__(
        self,
        template: str,
        *,
        field_patterns: dict = None,
        to_str_funcs: dict = None,
        from_str_funcs: dict = None,
        simple_str_sep: str = ",",
        namedtuple_type_name: str = "NamedTuple",
        dflt_pattern: str = ".*",
        dflt_field_name: Callable[[str], str] = "i{:02.0f}_".format,
    ):
        self._init_kwargs = dict(
            template=template,
            field_patterns=field_patterns,
            to_str_funcs=to_str_funcs,
            from_str_funcs=from_str_funcs,
            simple_str_sep=simple_str_sep,
            namedtuple_type_name=namedtuple_type_name,
            dflt_pattern=dflt_pattern,
            dflt_field_name=dflt_field_name,
        )
        self._original_template = template
        self.simple_str_sep = simple_str_sep
        self.namedtuple_type_name = namedtuple_type_name
        self.dflt_pattern = dflt_pattern
        self.dflt_field_name = dflt_field_name

        (
            self.template,
            self._fields,
            _to_str_funcs,
            field_patterns_,
        ) = self._extract_template_info(template)

        self._field_patterns = dict(
            {field: self.dflt_pattern for field in self._fields},
            **dict(field_patterns_, **(field_patterns or {})),
        )
        self._to_str_funcs = dict(
            {field: str for field in self._fields},
            **dict(_to_str_funcs, **(to_str_funcs or {})),
        )
        self._from_str_funcs = dict(
            {field: identity for field in self._fields}, **(from_str_funcs or {})
        )
        self._n_fields = len(self._fields)
        self._regex = self._compile_regex(self.template)

    def clone(self, **kwargs):
        return type(self)(**{**self._init_kwargs, **kwargs})

    clone.__signature__ = signature(__init__)

    def key_codec(
        self, decoded: FieldTypeNames = "tuple", encoded: FieldTypeNames = "str"
    ):
        r"""Makes a ``KeyCodec`` for the given source and target types.

        >>> st = KeyTemplate(
        ...     'root/{name}/v_{version}.json',
        ...     field_patterns={'version': r'\d+'},
        ...     from_str_funcs={'version': int},
        ... )

        A `key_codec` is a codec; it has an encoder and a decoder.

        >>> key_codec = st.key_codec('tuple', 'str')
        >>> encoder, decoder = key_codec
        >>> decoder('root/dol/v_9.json')
        ('dol', 9)
        >>> encoder(('dol', 9))
        'root/dol/v_9.json'

        If you have a ``Mapping``, you can use ``key_codec`` as a decorator to wrap
        the mapping with a key mappings.

        >>> store = {
        ...     'root/meshed/v_151.json': '{"downloads": 41, "type": "productivity"}',
        ...     'root/dol/v_9.json': '{"downloads": 132, "type": "utility"}',
        ... }
        >>>
        >>> accessor = key_codec(store)
        >>> list(accessor)
        [('meshed', 151), ('dol', 9)]
        >>> accessor['i2', 4] = '{"downloads": 274, "type": "utility"}'
        >>> list(store)
        ['root/meshed/v_151.json', 'root/dol/v_9.json', 'root/i2/v_4.json']
        >>> store['root/i2/v_4.json']
        '{"downloads": 274, "type": "utility"}'

        Note: If your store contains keys that don't fit the format, key_codec will
        raise a ``ValueError``. To remedy this, you can use the ``st.filt_iter`` to
        filter out keys that don't fit the format, before you wrap the store with
        ``st.key_codec``.

        """
        self._assert_field_type(decoded, "decoded")
        self._assert_field_type(encoded, "encoded")
        coder = getattr(self, f"{decoded}_to_{encoded}")
        decoder = getattr(self, f"{encoded}_to_{decoded}")
        return KeyCodec(coder, decoder)

    def filt_iter(self, field_type: FieldTypeNames = "str"):
        r"""
        Makes a store decorator that filters out keys that don't match the template
        given field type.

        >>> store = {
        ...     'root/meshed/v_151.json': '{"downloads": 41, "type": "productivity"}',
        ...     'root/dol/v_9.json': '{"downloads": 132, "type": "utility"}',
        ...     'root/not/the/right/format': "something else"
        ... }
        >>> filt = KeyTemplate('root/{pkg}/v_{version}.json')
        >>> filtered_store = filt.filt_iter('str')(store)
        >>> list(filtered_store)
        ['root/meshed/v_151.json', 'root/dol/v_9.json']

        """
        if isinstance(field_type, Mapping):
            # The user wants to filter a store with the default
            return self.filt_iter()(field_type)
        self._assert_field_type(field_type, "field_type")
        filt_func = getattr(self, f"match_{field_type}")
        return filt_iter(filt=filt_func)

    # @_return_none_if_none_input
    def str_to_dict(self, s: str) -> dict:
        r"""Parses the input string and returns a dictionary of extracted values.

        >>> st = KeyTemplate(
        ...     'root/{}/v_{ver:03.0f:\d+}.json',
        ...     from_str_funcs={'ver': int},
        ... )
        >>> st.str_to_dict('root/life/v_30.json')
        {'i01_': 'life', 'ver': 30}

        """
        if s is None:
            return None
        match = self._regex.match(s)
        if match:
            return {k: self._from_str_funcs[k](v) for k, v in match.groupdict().items()}
        else:
            raise ValueError(f"String '{s}' does not match the template.")

    # @_return_none_if_none_input
    def dict_to_str(self, params: dict) -> str:
        r"""Generates a string from the dictionary values based on the template.

        >>> st = KeyTemplate(
        ...     'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.dict_to_str({'i01_': 'life', 'ver': 42})
        'root/life/v_042.json'

        """
        if params is None:
            return None
        params = {k: self._to_str_funcs[k](v) for k, v in params.items()}
        return self.template.format(**params)

    # @_return_none_if_none_input
    def dict_to_tuple(self, params: dict) -> tuple:
        r"""Generates a tuple from the dictionary values based on the template.

        >>> st = KeyTemplate(
        ...     'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.str_to_tuple('root/life/v_42.json')
        ('life', 42)

        """
        if params is None:
            return None
        return tuple(params.get(field_name) for field_name in self._fields)

    # @_return_none_if_none_input
    def tuple_to_dict(self, param_vals: tuple) -> dict:
        r"""Generates a dictionary from the tuple values based on the template.

        >>> st = KeyTemplate(
        ...     'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.tuple_to_dict(('life', 42))
        {'i01_': 'life', 'ver': 42}
        """
        if param_vals is None:
            return None
        return {
            field_name: value for field_name, value in zip(self._fields, param_vals)
        }

    # @_return_none_if_none_input
    def str_to_tuple(self, s: str) -> tuple:
        r"""Parses the input string and returns a tuple of extracted values.

        >>> st = KeyTemplate(
        ...     'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.str_to_tuple('root/life/v_42.json')
        ('life', 42)
        """
        if s is None:
            return None
        return self.dict_to_tuple(self.str_to_dict(s))

    # @_return_none_if_none_input
    def tuple_to_str(self, param_vals: tuple) -> str:
        r"""Generates a string from the tuple values based on the template.

        >>> st = KeyTemplate(
        ...     'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.tuple_to_str(('life', 42))
        'root/life/v_042.json'
        """
        if param_vals is None:
            return None
        return self.dict_to_str(self.tuple_to_dict(param_vals))

    # @_return_none_if_none_input
    def str_to_single(self, s: str) -> Any:
        r"""Parses the input string and returns a single value.

        >>> st = KeyTemplate(
        ...     'root/life/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.str_to_single('root/life/v_42.json')
        42
        """
        if s is None:
            return None
        return self.str_to_tuple(s)[0]

    # @_return_none_if_none_input
    def single_to_str(self, k: Any) -> str:
        r"""Generates a string from the single value based on the template.

        >>> st = KeyTemplate(
        ...     'root/life/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.single_to_str(42)
        'root/life/v_042.json'
        """
        if k is None:
            return None
        return self.tuple_to_str((k,))

    # @_return_none_if_none_input
    def dict_to_namedtuple(
        self,
        params: dict,
    ):
        r"""Generates a namedtuple from the dictionary values based on the template.

        >>> st = KeyTemplate(
        ...     'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> App = st.dict_to_namedtuple({'i01_': 'life', 'ver': 42})
        >>> App
        NamedTuple(i01_='life', ver=42)
        """
        if params is None:
            return None
        return namedtuple(self.namedtuple_type_name, params.keys())(**params)

    # @_return_none_if_none_input
    def namedtuple_to_dict(self, nt):
        r"""Converts a namedtuple to a dictionary.

        >>> st = KeyTemplate(
        ...     'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> App = st.dict_to_namedtuple({'i01_': 'life', 'ver': 42})
        >>> st.namedtuple_to_dict(App)
        {'i01_': 'life', 'ver': 42}
        """
        if nt is None:
            return None
        return dict(nt._asdict())  # TODO: Find way that doesn't involve private method

    def str_to_namedtuple(self, s: str):
        r"""Converts a string to a namedtuple.

        >>> st = KeyTemplate(
        ...     'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> App = st.str_to_namedtuple('root/life/v_042.json')
        >>> App
        NamedTuple(i01_='life', ver=42)
        """
        if s is None:
            return None
        return self.dict_to_namedtuple(self.str_to_dict(s))

    # @_return_none_if_none_input
    def str_to_simple_str(self, s: str):
        r"""Converts a string to a simple string (i.e. a simple character-delimited string).

        >>> st = KeyTemplate(
        ...     'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.str_to_simple_str('root/life/v_042.json')
        'life,042'
        >>> st_clone = st.clone(simple_str_sep='-')
        >>> st_clone.str_to_simple_str('root/life/v_042.json')
        'life-042'
        """
        if s is None:
            return None
        return self.simple_str_sep.join(
            self._to_str_funcs[k](v) for k, v in self.str_to_dict(s).items()
        )

    # @_return_none_if_none_input
    def simple_str_to_tuple(self, ss: str):
        r"""Converts a simple character-delimited string to a dict.

        >>> st = KeyTemplate(
        ...     'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ...     simple_str_sep='-',
        ... )
        >>> st.simple_str_to_tuple('life-042')
        ('life', 42)
        """
        if ss is None:
            return None
        if self.simple_str_sep:
            field_values = ss.split(self.simple_str_sep)
        else:
            field_values = (ss,)
        if len(field_values) != self._n_fields:
            raise ValueError(
                f"String '{ss}' has does not have the right number of field values. "
                f"Expected {self._n_fields}, got {len(field_values)} "
                f"(namely: {field_values}.)"
            )
        return tuple(f(x) for f, x in zip(self._from_str_funcs.values(), field_values))

    # @_return_none_if_none_input
    def simple_str_to_str(self, ss: str):
        r"""Converts a simple character-delimited string to a string.

        >>> st = KeyTemplate(
        ...     'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ...     simple_str_sep='-',
        ... )
        >>> st.simple_str_to_str('life-042')
        'root/life/v_042.json'
        """
        if ss is None:
            return None
        return self.tuple_to_str(self.simple_str_to_tuple(ss))

    def match_str(self, s: str) -> bool:
        r"""
        Returns True iff the string matches the template.

        >>> st = KeyTemplate(
        ...     'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.match_str('root/life/v_042.json')
        True
        >>> st.match_str('this/does/not_match')
        False
        """
        return self._regex.match(s) is not None

    def match_dict(self, params: dict) -> bool:
        return self.match_str(self.dict_to_str(params))
        # Note: Could do:
        #  return all(self._field_patterns[k].match(v) for k, v in params.items())
        # but not sure that's even quicker (given regex is compiled)

    def match_tuple(self, param_vals: tuple) -> bool:
        return self.match_str(self.tuple_to_str(param_vals))

    def match_namedtuple(self, params: namedtuple) -> bool:
        return self.match_str(self.namedtuple_to_str(params))

    def match_simple_str(self, params: str) -> bool:
        return self.match_str(self.simple_str_to_str(params))

    def _extract_template_info(self, template):
        r"""Extracts information from the template. Namely:

        - normalized_template: A template where each placeholder has a field name
        (if not given, dflt_field_name will be used, which by default is
        'i{:02.0f}_'.format)

        - field_names: The tuple of field names in the order they appear in template

        - to_str_funcs: A dict of field names and their corresponding to_str functions,
        which will be used to convert the field values to strings when generating a
        string.

        - field_patterns_: A dict of field names and their corresponding regex patterns,
        which will be used to extract the field values from a string.

        These four values are used in the init to compute the parameters of the
        instance.

        >>> st = KeyTemplate('{:03.0f}/{name::\w+}')
        >>> st.template
        '{i01_}/{name}'
        >>> st._fields
        ('i01_', 'name')
        >>> st._field_patterns
        {'i01_': '.*', 'name': '\\w+'}
        >>> st._regex.pattern
        '(?P<i01_>.*)/(?P<name>\\w+)'
        >>> to_str_funcs = st._to_str_funcs
        >>> to_str_funcs['i01_'](3)
        '003'
        >>> to_str_funcs['name']('life')
        'life'

        """

        field_names = []
        field_patterns_ = {}
        to_str_funcs = {}

        def parse_and_transform():
            for index, (literal_text, field_name, format_spec, conversion) in enumerate(
                self._formatter.parse(template), 1
            ):
                field_name = (
                    self.dflt_field_name(index) if field_name == "" else field_name
                )
                if field_name is not None:
                    field_names.append(field_name)  # remember the field name
                    # extract format and pattern information:
                    if ":" not in format_spec:
                        format_spec += ":"
                    to_str_func_format, pattern = format_spec.split(":")
                    if to_str_func_format:
                        to_str_funcs[field_name] = (
                            "{" + f":{to_str_func_format}" + "}"
                        ).format
                    field_patterns_[field_name] = pattern or self.dflt_pattern
                # At this point you should have a valid field_name and empty format_spec
                yield (
                    literal_text,
                    field_name,
                    "",
                    conversion,
                )

        normalized_template = string_unparse(parse_and_transform())
        return normalized_template, tuple(field_names), to_str_funcs, field_patterns_

    def _compile_regex(self, template):
        r"""Parses the template, generating regex for matching the template.
        Essentially, it weaves together the literal text parts and the format_specs
        parts, transformed into name-caputuring regex patterns.

        Note that the literal text parts are regex-escaped so that they are not
        interpreted as regex. For example, if the template is "{name}.txt", the
        literal text part is replaced with "\\.txt", to avoid that the "." is
        interpreted as a regex wildcard. This would otherwise match any character.
        Instead, the escaped dot is matched literally.
        See https://docs.python.org/3/library/re.html#re.escape for more information.

        >>> KeyTemplate('{}.ext')._regex.pattern
        '(?P<i01_>.*)\\.ext'
        >>> KeyTemplate('{name}.ext')._regex.pattern
        '(?P<name>.*)\\.ext'
        >>> KeyTemplate('{::\w+}.ext')._regex.pattern
        '(?P<i01_>\\w+)\\.ext'
        >>> KeyTemplate('{name::\w+}.ext')._regex.pattern
        '(?P<name>\\w+)\\.ext'
        >>> KeyTemplate('{:0.02f:\w+}.ext')._regex.pattern
        '(?P<i01_>\\w+)\\.ext'
        >>> KeyTemplate('{name:0.02f:\w+}.ext')._regex.pattern
        '(?P<name>\\w+)\\.ext'
        """

        def mk_named_capture_group(field_name):
            if field_name:
                return f"(?P<{field_name}>{self._field_patterns[field_name]})"
            else:
                return ""

        def generate_pattern_parts(template):
            parts = self._formatter.parse(template)
            for literal_text, field_name, _, _ in parts:
                yield re.escape(literal_text) + mk_named_capture_group(field_name)

        return safe_compile("".join(generate_pattern_parts(template)))

    @staticmethod
    def _assert_field_type(field_type: FieldTypeNames, name="field_type"):
        if field_type not in FieldTypeNames.__args__:
            raise ValueError(
                f"{name} must be one of {FieldTypeNames}. Was: {field_type}"
            )
