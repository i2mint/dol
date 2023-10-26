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
from typing import Union, Callable, Any, Mapping, Iterable, Tuple, Literal
from operator import getitem
import os

from dol.base import Store
from dol.util import lazyprop, add_as_attribute_of, max_common_prefix
from dol.trans import (
    store_decorator,
    kv_wrap,
    add_path_access,
    filt_iter,
    wrap_kvs,
    add_missing_key_handling,
)
from dol.dig import recursive_get_attr

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
                    dict(obj=obj, path=path, result=result, k=k, error=error,)
                )
            elif isinstance(on_error, str):
                # use on_error as a message, raising the same error class
                raise type(error)(on_error)
            else:
                raise ValueError(
                    f'on_error should be a callable (input is a dict) or a string. '
                    f'Was: {on_error}'
                )
    return result


def split_if_str(obj, sep='.'):
    if isinstance(obj, str):
        return obj.split(sep)
    return obj


def cast_to_int_if_numeric_str(k):
    if isinstance(k, str) and str.isnumeric(k):
        return int(k)
    return k


def separate_keys_with_separator(obj, sep='.'):
    return map(cast_to_int_if_numeric_str, split_if_str(obj, sep))


def get_attr_or_item(obj, k):
    """If ``k`` is a string, tries to get ``k`` as an attribute of ``obj`` first,
    and if that fails, gets it as ``obj[k]``"""
    if isinstance(k, str):
        try:
            return getattr(obj, k)
        except AttributeError:
            pass
    return obj[k]


# ------------------------------------------------------------------------------
# key-path operations


from typing import Iterable, KT, VT, Callable, Mapping, Union

Path = Union[Iterable[KT], str]


# TODO: Needs a lot more documentation and tests to show how versatile it is
def path_get(
    obj: Any,
    path,
    on_error: OnErrorType = raise_on_error,
    *,
    sep='.',
    key_transformer=cast_to_int_if_numeric_str,
    get_value: Callable = get_attr_or_item,
    caught_errors=(Exception,),
):
    """
    Get elements of a mapping through a path to be called recursively.

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
    sep: str = '.',
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


from typing import Callable, Mapping, KT, VT, TypeVar, Iterator, Union, Literal
from dol.base import kv_walk


PT = TypeVar('PT')  # Path Type
PkvFilt = Callable[[PT, KT, VT], bool]


def path_filter(pkv_filt: PkvFilt, d: Mapping,) -> Iterator[PT]:
    """Walk a dict, yielding paths to values that pass the ``pkv_filt``

    :param pkv_filt: A function that takes a path, key, and value, and returns
        ``True`` if the path should be yielded, and ``False`` otherwise
    :param d: The ``Mapping`` to walk (scan through)

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

    """
    _yield_func = partial(_path_matcher_yield_func, pkv_filt, None)
    walker = kv_walk(d, yield_func=_yield_func)
    yield from filter(None, walker)


# backwards compatibility quasi-alias (arguments are flipped)
def search_paths(d: Mapping, pkv_filt: PkvFilt) -> Iterator[PT]:
    """backwards compatibility quasi-alias (arguments are flipped)
    Use path_filter instead, since search_paths will be deprecated.
    """
    return path_filter(pkv_filt, d)


def _path_matcher_yield_func(pkv_filt: PkvFilt, sentinel, p: PT, k: KT, v: VT):
    """Helper to make (picklable) yield_funcs for paths_matching (through partial)"""
    if pkv_filt(p, k, v):
        return p
    else:
        return sentinel


@add_as_attribute_of(path_filter)
def _mk_path_matcher(pkv_filt: PkvFilt, sentinel=None):
    """Make a yield_func that only yields paths that pass the pkv_filt,
    and a sentinel (by default, ``None``) otherwise"""
    return partial(_path_matcher_yield_func, pkv_filt, sentinel)


@add_as_attribute_of(path_filter)
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

    _prefix_attr_name = '_prefix'

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

    def __init__(self, _prefix=''):
        self._prefix = _prefix


@store_decorator
def mk_relative_path_store(
    store_cls=None, *, name=None, with_key_validation=False, prefix_attr='_prefix',
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
            f'The use of name argumment is deprecated. Use __name__ instead',
            DeprecationWarning,
        )

    cls = type(store_cls.__name__, (PrefixRelativizationMixin, Store), {})

    @wraps(store_cls.__init__)
    def __init__(self, *args, **kwargs):
        Store.__init__(self, store=store_cls(*args, **kwargs))
        prefix = recursive_get_attr(self.store, prefix_attr, '')
        setattr(
            self, prefix_attr, prefix
        )  # TODO: Might need descriptor to enable assignment

    cls.__init__ = __init__

    if prefix_attr != '_prefix':
        assert not hasattr(store_cls, '_prefix'), (
            f'You already have a _prefix attribute, '
            f'but want the prefix name to be {prefix_attr}. '
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
        assert hasattr(store_cls, 'is_valid_key'), (
            'If you want with_key_validation=True, '
            "you'll need a method called is_valid_key to do the validation job"
        )

        def _id_of_key(self, k):
            _id = super(cls, self)._id_of_key(k)
            if self.store.is_valid_key(_id):
                return _id
            else:
                raise KeyError(
                    f'Key not valid (usually because does not exist or access not permitted): {k}'
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
    default_prefix='',
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
                f'I can only infer prefix from a store instance, not a type: {store}'
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
    str = 'str'
    dict = 'dict'
    tuple = 'tuple'
    namedtuple = 'namedtuple'


path_key_type_for_type = {
    str: PathKeyTypes.str,
    dict: PathKeyTypes.dict,
    tuple: PathKeyTypes.tuple,
}

_method_names_for_path_type = {
    PathKeyTypes.str: {
        '_id_of_key': StrTupleDict.simple_str_to_str,
        '_key_of_id': StrTupleDict.str_to_simple_str,
    },
    PathKeyTypes.dict: {
        '_id_of_key': StrTupleDict.dict_to_str,
        '_key_of_id': StrTupleDict.str_to_dict,
    },
    PathKeyTypes.tuple: {
        '_id_of_key': StrTupleDict.tuple_to_str,
        '_key_of_id': StrTupleDict.str_to_tuple,
    },
    PathKeyTypes.namedtuple: {
        '_id_of_key': StrTupleDict.namedtuple_to_str,
        '_key_of_id': StrTupleDict.str_to_namedtuple,
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
    named_tuple_type_name='NamedTuple',
    sep: str = path_sep,
):
    """Make a key trans object that translates from a string _id to a dict, tuple, or namedtuple key (and back)"""

    assert (
        key_type in PathKeyTypes
    ), f"key_type was {key_type}. Needs to be one of these: {', '.join(PathKeyTypes)}"

    class PathKeyMapper(StrTupleDict):
        ...

    setattr(
        PathKeyMapper,
        '_id_of_key',
        _method_names_for_path_type[key_type]['_id_of_key'],
    )
    setattr(
        PathKeyMapper,
        '_key_of_id',
        _method_names_for_path_type[key_type]['_key_of_id'],
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


Codec = namedtuple('Codec', 'encoder decoder')
FieldTypeNames = Literal['str', 'dict', 'tuple', 'namedtuple', 'simple_str']


# TODO: Make and use _return_none_if_none_input or not?
# TODO: Change to dataclass with 3.10+ (to be able to do KW_ONLY)
# TODO: Should be refactored and generalized to be able to automatically handle
#   all combinations of FieldTypeNames (and possibly open-close these as well?)
#   It's a "path finder" meshed pattern.
# TODO: Do we really want to allow field_patterns to be included in the template (the `{name:pattern}` options)?
#  Normally, this is used for string GENERATION as `{name:format}`, which is still useful for us here too.
#  The counter argument is that the main usage of StringTemplate is not actually
#  generation, but extraction. Further, the format language is not as general as simply
#  using a format_field = {field: cast_function, ...} argument.
#  My decision would be to remove any use of the `{name:X}` form in the base class,
#  and have classmethods specialized for short-hand versions that use `name:regex` or
#  `name:format`, ...
class StringTemplate:
    """A class for parsing and generating strings based on a template.

    Args:
        template: A template string with fields to be extracted or filled in.
        field_patterns: A dictionary of field names and their regex patterns.
        simple_str_sep: A separator string for simple strings (i.e. strings without
            fields). If `None`, the template string will be used as the separator.

    Examples:
        >>> st = StringTemplate(
        ...     "{name} is {age} years old.",
        ...     field_patterns={"name": r"\w+", "age": r"\d+"}
        ... )
        >>> st.str_to_dict("Alice is 30 years old.")
        {'name': 'Alice', 'age': '30'}
        >>> st.dict_to_str({'name': 'Alice', 'age': '30'})
        'Alice is 30 years old.'
        >>> st.dict_to_tuple({'name': 'Alice', 'age': '30'})
        ('Alice', '30')
        >>> st.tuple_to_dict(('Alice', '30'))
        {'name': 'Alice', 'age': '30'}
        >>> st.str_to_tuple("Alice is 30 years old.")
        ('Alice', '30')

        You can also ask any (handled) combination of field types:
        >>> coder, encoder = st.codec('tuple', 'dict')
        >>> coder(('Alice', '30'))
        {'name': 'Alice', 'age': '30'}
        >>> encoder({'name': 'Alice', 'age': '30'})
        ('Alice', '30')
    """

    def __init__(
        self,
        template: str,
        *,
        field_patterns: dict = None,
        simple_str_sep: str = None,
        namedtuple_type_name: str = 'NamedTuple',
    ):
        self.template = template
        self.field_patterns = field_patterns or {}
        self.simple_str_sep = simple_str_sep
        self.namedtuple_type_name = namedtuple_type_name
        self.regex = None
        self._construct_regex()

    # TODO: For now we harded coded the "interpret format_specs as field patterns",
    #   but we could make this more general and allow for other types of
    #   "interpretations" (e.g. "interpret format_specs as cast functions").
    #   Note: We removed the cast functions (that existed in legacy StrTupleDict class)
    #   to keep it simple, for now. The idea being that if you want to cast, you can
    #   do it yourself by adding egress/ingress to codecs.
    def _construct_regex(self):
        formatter = string.Formatter()
        pattern = self.template
        self.field_names = []
        for literal_text, field_name, format_spec, conversion in formatter.parse(
            self.template
        ):
            # Check if the field_name has either a format_spec (regex) in the template
            # or a matching regex in the field_patterns dictionary before adding it
            # to the field_names list.
            if field_name and (format_spec or field_name in self.field_patterns):
                self.field_names.append(field_name)
                regex = format_spec or self.field_patterns.get(field_name, '.*?')
                to_replace = (
                    '{' + field_name + (':' + format_spec if format_spec else '') + '}'
                )
                pattern = pattern.replace(to_replace, f'(?P<{field_name}>{regex})')
        self.regex = re.compile(pattern)

    def codec(self, source: FieldTypeNames, target: FieldTypeNames):
        """Makes a ``(coder, decoder)`` pair for the given source and target types.

        >>> st = StringTemplate(
        ...     "{name} is {age} years old.",
        ...     field_patterns={"name": r"\w+", "age": r"\d+"}
        ... )
        >>> coder, encoder = st.codec('tuple', 'dict')
        >>> coder(('Alice', '30'))
        {'name': 'Alice', 'age': '30'}
        >>> encoder({'name': 'Alice', 'age': '30'})
        ('Alice', '30')
        """
        coder = getattr(self, f'{source}_to_{target}')
        decoder = getattr(self, f'{target}_to_{source}')
        return Codec(coder, decoder)

    # @_return_none_if_none_input
    def str_to_dict(self, s: str) -> dict:
        """Parses the input string and returns a dictionary of extracted values.

        >>> st = StringTemplate(
        ...     "{name} is {age} years old.",
        ...     field_patterns={"name": r"\w+", "age": r"\d+"}
        ... )

        >>> st.str_to_dict("Alice is 30 years old.")
        {'name': 'Alice', 'age': '30'}
        """
        if s is None:
            return None
        match = self.regex.match(s)
        if match:
            return match.groupdict()
        else:
            raise ValueError(f"String '{s}' does not match the template.")

    # @_return_none_if_none_input
    def dict_to_str(self, params: dict) -> str:
        """Generates a string from the dictionary values based on the template.

        >>> st = StringTemplate(
        ...     "{name} is {age} years old.",
        ...     field_patterns={"name": r"\w+", "age": r"\d+"}
        ... )
        >>> st.dict_to_str({'name': 'Alice', 'age': '30'})
        'Alice is 30 years old.'

        """
        if params is None:
            return None
        return self.template.format(**params)

    # @_return_none_if_none_input
    def dict_to_tuple(self, params: dict) -> tuple:
        """Generates a tuple from the dictionary values based on the template.

        >>> st = StringTemplate(
        ...     "{name} is {age} years old.",
        ...     field_patterns={"name": r"\w+", "age": r"\d+"}
        ... )
        >>> st.dict_to_tuple({'name': 'Alice', 'age': '30'})
        ('Alice', '30')
        """
        if params is None:
            return None
        return tuple(params.get(field_name) for field_name in self.field_names)

    # @_return_none_if_none_input
    def tuple_to_dict(self, param_vals: tuple) -> dict:
        """Generates a dictionary from the tuple values based on the template.

        >>> st = StringTemplate(
        ...     "{name} is {age} years old.",
        ...     field_patterns={"name": r"\w+", "age": r"\d+"}
        ... )
        >>> st.tuple_to_dict(('Alice', '30'))
        {'name': 'Alice', 'age': '30'}
        """
        if param_vals is None:
            return None
        return {
            field_name: value for field_name, value in zip(self.field_names, param_vals)
        }

    # @_return_none_if_none_input
    def str_to_tuple(self, s: str) -> tuple:
        """Parses the input string and returns a tuple of extracted values.

        >>> st = StringTemplate(
        ...     "{name} is {age} years old.",
        ...     field_patterns={"name": r"\w+", "age": r"\d+"}
        ... )
        >>> st.str_to_tuple("Alice is 30 years old.")
        ('Alice', '30')
        """
        if s is None:
            return None
        return self.dict_to_tuple(self.str_to_dict(s))

    # @_return_none_if_none_input
    def tuple_to_str(self, params: tuple) -> str:
        """Generates a string from the tuple values based on the template.

        >>> st = StringTemplate(
        ...     "{name} is {age} years old.",
        ...     field_patterns={"name": r"\w+", "age": r"\d+"}
        ... )
        >>> st.tuple_to_str(('Alice', '30'))
        'Alice is 30 years old.'
        """
        if params is None:
            return None
        return self.dict_to_str(self.tuple_to_dict(params))

    # @_return_none_if_none_input
    def dict_to_namedtuple(
        self, params: dict,
    ):
        """Generates a namedtuple from the dictionary values based on the template.

        >>> st = StringTemplate(
        ...     "{name} is {age} years old.",
        ...     field_patterns={"name": r"\w+", "age": r"\d+"}
        ... )
        >>> Person = st.dict_to_namedtuple({'name': 'Alice', 'age': '30'})
        >>> Person
        NamedTuple(name='Alice', age='30')
        """
        if params is None:
            return None
        return namedtuple(self.namedtuple_type_name, params.keys())(**params)

    # @_return_none_if_none_input
    def namedtuple_to_dict(self, nt):
        """Converts a namedtuple to a dictionary.

        >>> st = StringTemplate(
        ...     "{name} is {age} years old.",
        ...     field_patterns={"name": r"\w+", "age": r"\d+"}
        ... )
        >>> Person = st.dict_to_namedtuple({'name': 'Alice', 'age': '30'})
        >>> st.namedtuple_to_dict(Person)
        {'name': 'Alice', 'age': '30'}
        """
        if nt is None:
            return None
        return dict(nt._asdict())  # TODO: Find way that doesn't involve private method

    # @_return_none_if_none_input
    def str_to_simple_str(self, s: str, sep: str):
        """Converts a string to a simple string (i.e. a simple character-delimited string).

        >>> st = StringTemplate(
        ...     "{name} is {age} years old.",
        ...     field_patterns={"name": r"\w+", "age": r"\d+"}
        ... )
        >>> st.str_to_simple_str("Alice is 30 years old.", '-')
        'Alice-30'
        """
        if s is None:
            return None
        elif sep is None:
            if self.simple_str_sep is None:
                raise ValueError(
                    'Need to specify a sep (at method call time), or a simple_str_sep '
                    '(at instiantiation time) to use str_to_simple_str'
                )
        return sep.join(self.str_to_tuple(s))

    # @_return_none_if_none_input
    def simple_str_to_str(self, ss: str, sep: str):
        """Converts a simple character-delimited string to a string.

        >>> st = StringTemplate(
        ...     "{name} is {age} years old.",
        ...     field_patterns={"name": r"\w+", "age": r"\d+"}
        ... )
        >>> st.simple_str_to_str('Alice-30', '-')
        'Alice is 30 years old.'
        """
        if ss is None:
            return None
        return self.tuple_to_str(tuple(ss.split(sep)))
