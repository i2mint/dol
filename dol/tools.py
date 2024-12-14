"""
Various tools to add functionality to stores
"""

import os
from typing import Optional, Callable, KT, VT, Any, Union, T
from collections.abc import Mapping

from dol.base import Store
from dol.trans import store_decorator
from dol.caching import cache_vals

NoSuchKey = type("NoSuchKey", (), {})


from functools import RLock, cached_property
from types import GenericAlias
from collections.abc import MutableMapping

_NOT_FOUND = object()

Instance = Any
PropertyFunc = Callable[[Instance], VT]
MethodName = str
Cache = Union[MethodName, MutableMapping[KT, VT]]
KeyType = Union[KT, Callable[[MethodName], KT]]


def identity(x: T) -> T:
    return x


class CachedProperty:
    """Descriptor that caches the result of the first call to a method.

    Note: Usually, you'd want to use the convenience decorator `cache_this` instead of
    using this class directly.

    It generalizes the builtin functools.cached_property class, enabling the user to
    specify a cache object and a key to store the cache value.
    """

    def __init__(
        self,
        func: PropertyFunc,
        cache: Optional[Cache] = None,
        key: Optional[KeyType] = None,
        *,
        allow_none_keys: bool = False,
        lock_factory: Callable = RLock,
        pre_cache: Union[bool, MutableMapping] = False,
    ):
        """
        Initialize the cached property.

        :param func: The function whose result needs to be cached.
        :param cache: The cache storage, can be a MutableMapping or an attribute name.
        :param key: The key to store the cache value, can be a callable or a string.
        :param pre_cache: Default is False. If True, adds an in-memory cache to the method
            to (also) cache the results in memory. If a MutableMapping is given,
            it will be used as the pre-cache.
            This is useful when you want a persistent cache but also want to speed up
            access to the method in the same session.
        """
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = lock_factory()
        self.cache = cache
        self.key = key if key else lambda name: name
        self.allow_none_keys = allow_none_keys

        if pre_cache is not False:
            if pre_cache is True:
                pre_cache = dict()
            else:
                assert isinstance(pre_cache, MutableMapping), (
                    f"`pre_cache` must be a bool or a MutableMapping, "
                    f"Was a {type(pre_cache)}: {pre_cache}"
                )
            self.wrap_cache = partial(cache_vals, cache=pre_cache)
        else:
            self.wrap_cache = identity

    def __set_name__(self, owner, name):
        """
        Set the name of the property.

        :param owner: The class owning the property.
        :param name: The name of the property.
        """
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same CachedProperty to two different names "
                f"({self.attrname!r} and {name!r})."
            )
        if isinstance(self.key, str):
            self.cache_key = self.key
        else:
            assert callable(
                self.key
            ), f"The key must be a callable or a string, not {type(self.key).__name__}."
            self.cache_key = self.key(self.attrname)
            if self.cache_key is None and not self.allow_none_keys:
                raise TypeError("The key returned by the key function cannot be None.")

    def __get_cache(self, instance):
        """
        Get the cache for the instance.

        :param instance: The instance of the class.
        :return: The cache storage.
        """
        if isinstance(self.cache, str):
            cache = getattr(instance, self.cache, None)
            if cache is None:
                raise TypeError(
                    f"No attribute named '{self.cache}' found on {type(instance).__name__!r} instance."
                )
            if not isinstance(cache, MutableMapping):
                raise TypeError(
                    f"Attribute '{self.cache}' on {type(instance).__name__!r} instance is not a MutableMapping."
                )
            __cache = cache
        elif isinstance(self.cache, MutableMapping):
            __cache = self.cache
        else:
            __cache = instance.__dict__

        return self.wrap_cache(__cache)

    def __get__(self, instance, owner=None):
        """
        Get the value of the cached property.

        :param instance: The instance of the class.
        :param owner: The owner class.
        :return: The cached value or computed value if not cached.
        """
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use CachedProperty instance without calling __set_name__ on it."
            )
        if self.cache is False:
            # If cache is False, always compute the value
            return self.func(instance)

        cache = self._get_cache(instance)

        return self._get_or_compute(instance, cache)

    def _get_cache(self, instance):
        try:
            cache = self.__get_cache(instance)
        except (
            AttributeError
        ):  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None
        return cache

    def _get_or_compute(self, instance, cache):
        val = cache.get(self.cache_key, _NOT_FOUND)
        if val is _NOT_FOUND:
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(self.cache_key, _NOT_FOUND)
                if val is _NOT_FOUND:
                    val = self.func(instance)
                    try:
                        cache[self.cache_key] = val
                    except TypeError as e:
                        msg = (
                            f"The cache on {type(instance).__name__!r} instance "
                            f"does not support item assignment for caching {self.cache_key!r} property.\n"
                            f"Error: {e}"
                        )
                        raise TypeError(msg) from None
        return val

    __class_getitem__ = classmethod(GenericAlias)

    # TODO: Time-boxed attempt to get a __call__ method to work with the class
    #    (so that you can chain two cache_this decorators, (problem is that the outer
    #    expects the inner to be a function, not an instance of CachedProperty, so
    #    tried to make CachedProperty callable).
    # def __call__(self, instance):
    #     """
    #     Call the cached property.

    #     :param func: The function to be called.
    #     :return: The cached property.
    #     """
    #     cache = self._get_cache(instance)

    #     return self._get_or_compute(instance, cache)


def cache_this(
    func: PropertyFunc = None,
    *,
    cache: Optional[Cache] = None,
    key: Optional[KeyType] = None,
    pre_cache: Union[bool, MutableMapping] = False,
):
    r"""
    Transforms a method into a cached property with control over cache object and key.

    :param func: The function to be decorated (usually left empty).
    :param cache: The cache storage, can be a `MutableMapping` or the name of an
        instance attribute that is a `MutableMapping`.
    :param key: The key to store the cache value, can be a callable that will be
        applied to the method name to make a key, or an explicit string.
    :param pre_cache: Default is False. If True, adds an in-memory cache to the method
        to (also) cache the results in memory. If a MutableMapping is given, it will be
        used as the pre-cache.
        This is useful when you want a persistent cache but also want to speed up
        access to the method in the same session.
    :return: The decorated function.

    Used with no arguments, `cache_this` will cache just as the builtin
    `cached_property` does -- in the instance's `__dict__` attribute.

    >>> class SameAsCachedProperty:
    ...     @cache_this
    ...     def foo(self):
    ...         print("In SameAsCachedProperty.foo...")
    ...         return 42
    ...
    >>> obj = SameAsCachedProperty()
    >>> obj.__dict__  # the cache is empty
    {}
    >>> obj.foo  # when we access foo, it's computed and returned...
    In SameAsCachedProperty.foo...
    42
    >>> obj.__dict__  # ... but also cached
    {'foo': 42}
    >>> obj.foo  # so that the next time we access foo, it's returned from the cache.
    42

    Not that if you specify `cache=False`, you get a property that is computed
    every time it's accessed:

    >>> class NoCache:
    ...     @cache_this(cache=False)
    ...     def foo(self):
    ...         print("In NoCache.foo...")
    ...         return 42
    ...
    >>> obj = NoCache()
    >>> obj.foo
    In NoCache.foo...
    42
    >>> obj.foo
    In NoCache.foo...
    42

    Specify the cache as a dictionary that lives outside the instance:

    >>> external_cache = {}
    >>>
    >>> class CacheWithExternalMapping:
    ...     @cache_this(cache=external_cache)
    ...     def foo(self):
    ...         print("In CacheWithExternalMapping.foo...")
    ...         return 42
    ...
    >>> obj = CacheWithExternalMapping()
    >>> external_cache
    {}
    >>> obj.foo
    In CacheWithExternalMapping.foo...
    42
    >>> external_cache
    {'foo': 42}
    >>> obj.foo
    42

    Specify the cache as an attribute of the instance, and an explicit key:

    >>> class WithCacheInInstanceAttribute:
    ...
    ...     def __init__(self):
    ...         self.my_cache = {}
    ...
    ...     @cache_this(cache='my_cache', key='key_for_foo')
    ...     def foo(self):
    ...         print("In WithCacheInInstanceAttribute.foo...")
    ...         return 42
    ...
    >>> obj = WithCacheInInstanceAttribute()
    >>> obj.my_cache
    {}
    >>> obj.foo
    In WithCacheInInstanceAttribute.foo...
    42
    >>> obj.my_cache
    {'key_for_foo': 42}
    >>> obj.foo
    42

    Now let's see a more involved example that exhibits how `cache_this` would be used
    in real life. Note two things in the example below.

    First, that we use `functools.partial` to fix the parameters of our `cache_this`.
    This enables us to reuse the same `cache_this` in multiple places without all
    the verbosity. We fix that the cache is the attribute `cache` of the instance,
    and that the key is a function that will be computed from the name of the method
    adding a `'.pkl'` extension to it.

    Secondly, we use the `ValueCodecs` from `dol` to provide a pickle codec for storying
    values. The backend store used here is a dictionary, so we don't really need a
    codec to store values, but in real life you would use a persistent storage that
    would require a codec, such as files or a database.

    Thirdly, we'll use a `pre_cache` to store the values in a different cache "before"
    (setting and getting) them in the main cache.
    This is useful, for instance, when you want to persist the values (in the main
    cache), but keep them in memory for faster access in the same session
    (the pre-cache, a dict() instance usually). It can also be used to store and
    use things locally (pre-cache) while sharing them with others by storing them in
    a remote store (main cache).

    Finally, we'll use a dict that logs any setting and getting of values to show
    how the caches are being used.

    >>> from dol import cache_this
    >>>
    >>> from functools import partial
    >>> from dol import ValueCodecs
    >>> from collections import UserDict
    >>>
    >>>
    >>> class LoggedCache(UserDict):
    ...     name = 'cache'
    ...
    ...     def __setitem__(self, key, value):
    ...         print(f"In {self.name}: setting {key} to {value}")
    ...         return super().__setitem__(key, value)
    ...
    ...     def __getitem__(self, key):
    ...         print(f"In {self.name}: getting value of {key}")
    ...         return super().__getitem__(key)
    ...
    >>>
    >>> class CacheA(LoggedCache):
    ...     name = 'CacheA'
    ...
    >>>
    >>> class CacheB(LoggedCache):
    ...     name = 'CacheB'
    ...
    >>>
    >>> cache_with_pickle = partial(
    ...     cache_this,
    ...     cache='cache',  # the cache can be found on the instance attribute `cache`
    ...     key=lambda x: f"{x}.pkl",  # the key is the method name with a '.pkl' extension
    ...     pre_cache=CacheB(),
    ... )
    >>>
    >>>
    >>> class PickleCached:
    ...     def __init__(self, backend_store_factory=CacheA):
    ...         # usually this would be a mapping interface to persistent storage:
    ...         self._backend_store = backend_store_factory()
    ...         self.cache = ValueCodecs.default.pickle(self._backend_store)
    ...
    ...     @cache_with_pickle
    ...     def foo(self):
    ...         print("In PickleCached.foo...")
    ...         return 42
    ...

    >>> obj = PickleCached()
    >>> list(obj.cache)
    []

    >>> obj.foo
    In CacheA: getting value of foo.pkl
    In CacheA: getting value of foo.pkl
    In PickleCached.foo...
    In CacheA: setting foo.pkl to b'\x80\x04K*.'
    42
    >>> obj.foo
    In CacheA: getting value of foo.pkl
    In CacheB: setting foo.pkl to 42
    42

    As usual, it's because the cache now holds something that has to do with `foo`:

    >>> list(obj.cache)
    ['foo.pkl']
    >>> # == ['foo.pkl']

    The value of `'foo.pkl'` is indeed `42`:

    >>> obj.cache['foo.pkl']
    In CacheA: getting value of foo.pkl
    42


    But note that the actual way it's stored in the `_backend_store` is as pickle bytes:

    >>> obj._backend_store['foo.pkl']
    In CacheA: getting value of foo.pkl
    b'\x80\x04K*.'
    >>> # == b'\x80\x04K*.'


    """

    # the cache is False case, where we just want a property, computed by func
    if cache is False:
        if func is None:

            def wrapper(f):
                return property(f)

            return wrapper
        else:
            return property(func)

    # The general case
    else:  #   If func is not given, we want a decorator
        if func is None:

            def wrapper(f):
                return CachedProperty(f, cache=cache, key=key, pre_cache=pre_cache)

            return wrapper

        else:  #   If func is given, we want to return the CachedProperty instance
            return CachedProperty(func, cache=cache, key=key, pre_cache=pre_cache)


extsep = os.path.extsep


def add_extension(ext=None, name=None):
    """
    Add an extension to a name.

    If name is None, return a partial function that will add the extension to a
    name when called.

    add_extension is a useful helper for making key functions, namely for cache_this.

    >>> add_extension('txt', 'file')
    'file.txt'
    >>> add_txt_ext = add_extension('txt')
    >>> add_txt_ext('file')
    'file.txt'

    Note: If you want to add an extension to a name that already has an extension,
    you can do that, but it will add the extension to the end of the name,
    not replace the existing extension.

    >>> add_txt_ext('file.txt')
    'file.txt.txt'

    Also, bare in mind that if ext starts with the system's extension separator,
    (os.path.extsep), it will be removed.

    >>> add_extension('.txt', 'file') == add_extension('txt', 'file') == 'file.txt'
    True

    """
    if ext.startswith(extsep):
        ext = ext[1:]
    if name is None:
        return partial(add_extension, ext)
    if ext:
        return f"{name}{extsep}{ext}"
    else:
        return name


from functools import lru_cache, partial, wraps


def cached_method(func=None, *, maxsize=128, typed=False):
    """
    A decorator to cache the result of a method, ignoring the first argument (usually `self`).

    This decorator uses `functools.lru_cache` to cache the method result based on the arguments passed
    to the method, excluding the first argument (typically `self`). This allows methods of a class to
    be cached while ignoring the instance (`self`) in the cache key.

    Parameters:
    - func (callable, optional): The method to be decorated. If not provided, a partially applied decorator
      will be returned for later application.
    - maxsize (int, optional): The maximum size of the cache. Defaults to 128.
    - typed (bool, optional): If True, cache entries will be different based on argument types, such as
      distinguishing between `1` and `1.0`. Defaults to False.

    Returns:
    - callable: A wrapped function with LRU caching applied, ignoring the first argument (`self`).

    Example:
    >>> class MyClass:
    ...     @cached_method(maxsize=2, typed=True)
    ...     def add(self, x, y):
    ...         print(f"Computing {x} + {y}")
    ...         return x + y
    ...
    >>> obj = MyClass()
    >>> obj.add(1, 2)
    Computing 1 + 2
    3
    >>> obj.add(1, 2)  # Cached result, no recomputation
    3
    >>> obj.add(1.0, 2.0)  # Different types, recomputation occurs
    Computing 1.0 + 2.0
    3.0
    """
    if func is None:
        # Parametrize cached_method and return a decorator to be applied to a function directly
        return partial(cached_method, maxsize=maxsize, typed=typed)

    # Create a cache, ignoring the first argument (`self`)
    cache = lru_cache(maxsize=maxsize, typed=typed)(
        lambda _, *args, **kwargs: func(_, *args, **kwargs)
    )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Call the cache but don't include `self` in the arguments for caching
        return cache(None, *args, **kwargs)

    return wrapper


from functools import lru_cache, partial, wraps


def lru_cache_method(func=None, *, maxsize=128, typed=False):
    """
    A decorator to cache the result of a method, ignoring the first argument
    (usually `self`).

    This decorator uses `functools.lru_cache` to cache the method result based on the arguments passed
    to the method, excluding the first argument (typically `self`). This allows methods of a class to
    be cached while ignoring the instance (`self`) in the cache key.

    Parameters:
    - func (callable, optional): The method to be decorated. If not provided, a partially applied decorator
      will be returned for later application.
    - maxsize (int, optional): The maximum size of the cache. Defaults to 128.
    - typed (bool, optional): If True, cache entries will be different based on argument types, such as
      distinguishing between `1` and `1.0`. Defaults to False.

    Returns:
    - callable: A wrapped function with LRU caching applied, ignoring the first argument (`self`).

    Example:

    >>> class MyClass:
    ...     @lru_cache_method
    ...     def add(self, x, y):
    ...         print(f"Computing {x} + {y}")
    ...         return x + y
    >>> obj = MyClass()
    >>> obj.add(1, 2)
    Computing 1 + 2
    3
    >>> obj.add(1, 2)  # Cached result, no recomputation
    3

    Like `lru_cache`, you can specify the `maxsize` and `typed` parameters:

    >>> class MyOtherClass:
    ...     @lru_cache_method(maxsize=2, typed=True)
    ...     def add(self, x, y):
    ...         print(f"Computing {x} + {y}")
    ...         return x + y
    ...
    >>> obj = MyOtherClass()
    >>> obj.add(1, 2)
    Computing 1 + 2
    3
    >>> obj.add(1, 2)  # Cached result, no recomputation
    3
    >>> obj.add(1.0, 2.0)  # Different types, recomputation occurs
    Computing 1.0 + 2.0
    3.0
    """
    if func is None:
        # Parametrize lru_cache_method and return a decorator to be applied to a function directly
        return partial(lru_cache_method, maxsize=maxsize, typed=typed)

    # Create a cache, ignoring the first argument (`self`)
    cache = lru_cache(maxsize=maxsize, typed=typed)(
        lambda _, *args, **kwargs: func(_, *args, **kwargs)
    )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Call the cache but don't include `self` in the arguments for caching
        return cache(None, *args, **kwargs)

    return wrapper


def cache_property_method(
    cls=None, method_name: MethodName = None, *, cache_decorator: Callable = cache_this
):
    """
    Converts a method of a class into a CachedProperty.

    Essentially, it does what `A.method = cache_this(A.method)` would do, taking care of
    the `__set_name__` problem that you'd run into doing it that way.
    Note that here, you need to say `cache_property_method(A, 'method')`.

    Args:
        cls (type): The class containing the method.
        method_name (str): The name of the method to convert to a cached property.
        cache_decorator (Callable): The decorator to use to cache the method. Defaults to
            `cache_this`. One frequent use case would be to use `functools.partial` to
            fix the cache and key parameters of `cache_this` and inject that.

    Example:

    >>> @cache_property_method(['normal_method', 'property_method'])
    ... class TestClass:
    ...     def normal_method(self):
    ...         print('normal_method called')
    ...         return 1
    ...
    ...     @property
    ...     def property_method(self):
    ...         print('property_method called')
    ...         return 2
    >>>
    >>> c = TestClass()
    >>> c.normal_method
    normal_method called
    1
    >>> c.normal_method
    1
    >>> c.property_method
    property_method called
    2
    >>> c.property_method
    2


    You can also use it like this:

    >>> class TestClass:
    ...     def normal_method(self):
    ...         print('normal_method called')
    ...         return 1
    ...
    ...     @property
    ...     def property_method(self):
    ...         print('property_method called')
    ...         return 2
    >>>
    >>> cache_property_method(
    ...     TestClass,
    ...     [
    ...         'normal_method',
    ...         'property_method',
    ...     ],
    ... )  # doctest: +ELLIPSIS
    <class ...TestClass'>
    >>> c = TestClass()
    >>> c.normal_method
    normal_method called
    1
    >>> c.normal_method
    1
    >>> c.property_method
    property_method called
    2
    >>> c.property_method
    2


    """
    if method_name is None:
        assert cls is not None, (
            "If method_name is None, cls (which will play the role of method_name in "
            "a decorator factory) must not be None."
        )
        method_name = cls
        return partial(
            cache_property_method,
            method_name=method_name,
            cache_decorator=cache_decorator,
        )
    if not isinstance(method_name, str) and isinstance(method_name, Iterable):
        for name in method_name:
            cache_property_method(cls, name, cache_decorator=cache_decorator)
        return cls

    method = getattr(cls, method_name)

    if isinstance(method, property):
        method = method.fget  # Get the original method from the property
    elif isinstance(method, (cached_property, CachedProperty)):
        method = method.func
    # not sure we want to handle (staticmethod, classmethod, but in case:
    # elif isinstance(method, (staticmethod, classmethod)):
    #     method = method.__func__

    cached_method = cache_decorator(method)
    cached_method.__set_name__(cls, method_name)
    setattr(cls, method_name, cached_method)
    return cls


# ------------ useful trans functions to be used with wrap_kvs etc. ---------------------
# TODO: Consider typing or decorating functions to indicate their role (e.g. id_of_key,
#   key_of_id, data_of_obj, obj_of_data, preset, postget...)


_dflt_confirm_overwrite_user_input_msg = (
    "The key {k} already exists and has value {existing_v}. "
    "If you want to overwrite it with {v}, confirm by typing {v} here: "
)


# TODO: Parametrize user messages (bring to interface)
# role: preset
def confirm_overwrite(
    mapping, k, v, user_input_msg=_dflt_confirm_overwrite_user_input_msg
):
    """A preset function you can use in wrap_kvs to ask the user to confirm if
    they're writing a value in a key that already has a different value under it.

    >>> from dol.trans import wrap_kvs
    >>> d = {'a': 'apple', 'b': 'banana'}
    >>> d = wrap_kvs(d, preset=confirm_overwrite)

    Overwriting ``a`` with the same value it already has is fine (not really an
    over-write):

    >>> d['a'] = 'apple'

    Creating new values is also fine:

    >>> d['c'] = 'coconut'
    >>> assert d == {'a': 'apple', 'b': 'banana', 'c': 'coconut'}

    But if we tried to do ``d['a'] = 'alligator'``, we'll get a user input request:

    .. code-block::

        The key a already exists and has value apple.
        If you want to overwrite it with alligator, confirm by typing alligator here:

    And we'll have to type `alligator` and press RETURN to make the write go through.

    """
    if (existing_v := mapping.get(k, NoSuchKey)) is not NoSuchKey and existing_v != v:
        user_input = input(user_input_msg.format(k=k, v=v, existing_v=existing_v))
        if user_input != v:
            print(f"--> User confirmation failed: I won't overwrite {k}")
            # this will have the effect of rewriting the same value that's there already:
            return existing_v
    return v


# -------------------------------- Aggregate a store -----------------------------------

from typing import (
    Callable,
    Optional,
    KT,
    VT,
    Tuple,
    Mapping,
    Union,
    TypeVar,
    Any,
    Iterable,
)
import os
from functools import partial
from pathlib import Path
from dol.trans import wrap_kvs
from dol.filesys import Files


def decode_as_latin1(b: bytes) -> str:
    return b.decode("latin1")


def markdown_section(k: KT, v: VT) -> str:
    return f"## {k}\n\n{v.strip()}\n\n"


def save_string_to_filepath(filepath: str, string: str):
    filepath = Path(filepath).expanduser().absolute()
    filepath.write_text(string)
    return string


def identity(x):
    return x


Latin1TextFiles = wrap_kvs(Files, value_decoder=decode_as_latin1)

Item = TypeVar("Item")
Aggregate = TypeVar("Aggregate")


def store_aggregate(
    content_store: Union[Mapping[KT, VT], str],  # Path to the folder or dol store
    *,
    kv_to_item: Callable[
        [KT, VT], Item
    ] = markdown_section,  # Function to convert key-value pairs to text
    aggregator: Callable[
        [Iterable[Item]], Aggregate
    ] = "\n\n".join,  # How to aggregate the item's into an aggregate
    egress: Union[
        Callable[[Aggregate], Any], str
    ] = identity,  # function to apply to the aggregate before returning
    key_filter: Optional[Callable[[KT], bool]] = None,  # Filter function for keys
    value_filter: Optional[Callable[[VT], bool]] = None,  # Filter function for values
    kv_filter: Optional[
        Callable[[Tuple[KT, VT]], bool]
    ] = None,  # Filter function for key-value pairs
    local_store_factory: Callable[
        [str], Mapping[KT, VT]
    ] = Latin1TextFiles,  # Factory function for the local store
) -> Any:
    r'''
    Create an aggregate object of a store's content.

    The function is written to be able to aggregate the keys and/or values of a store,
    no matter their type, and concatenate them into an object of arbitrary type.
    That said, the defaults are setup assuming the store's keys and values are text,
    and you want to concatenate them into a single string.
    This is useful, for example, when you have several files in a folder,
    and you want to create a single text/markdown file with all the content therein.

    This function filters content from a given content store, converts the key-value
    pairs to items (usually text), and (if you specify a filepath as the `egress`)
    saves the aggregate (text) before returning it.

    Args:
        content_store (Union[Mapping[KT, VT], str]): Path to the folder or dol store to read from.
        kv_to_item (Callable[[KT, VT], Item]):
            Function to convert key-value pairs to an Item (usually a string).
        aggregator (Callable[[Iterable[Item]], Aggregate]):
            The function that will aggregate the items that `kv_to_item` produces.
            Defaults to '\n\n'.join.
        egress (Union[Callable[[Aggregate], Any], str]):
            The function that will be called on the aggregate before returning it.
            Defaults to identity.
            Note that if you provide a string, the function will save the aggregate
            text to a file, assuming it is indeed text.
        key_filter (Optional[Callable[[KT], bool]]):
            Optional filter for keys. Defaults to None (no filtering).
        value_filter (Optional[Callable[[VT], bool]]):
            Optional filter for values. Defaults to None (no filtering).
        kv_filter (Optional[Callable[[Tuple[KT, VT]], bool]]):
            Optional filter for key-value pairs. Defaults to None (no filtering).
        local_store_factory (Callable[[str], Mapping[KT, VT]]): Factory function for the local store,
            used only if `content_store` is an existing folder path. Defaults to Latin1TextFiles.

    Returns:
        Any: Usually the aggregate object, which is usually the concatenated text.

    Normally, you'd specify your content store by specifying a root folder
    (the function will create a Mapping-view of the contents of the folder for you),
    or make a content store yourself (a Mapping object providing the key-value pairs).

    To provide a small example, we'll take a dict as our content store:

    >>> content_store = {
    ...     'file1.py': '"""Module docstring."""',
    ...     'file2.py': 'def foo(): pass',
    ...     'file3.py': '"""Another docstring."""',
    ...     'file4.md': 'Markdown content here.',
    ...     'file5.py': '"""If I mention file5.py, I will be excluded."""',
    ... }

    Define the filters:

    >>> key_filter = lambda k: k.endswith('.py')  # Only include keys that end with '.py'
    >>> value_filter = lambda v: v.startswith(
    ...     '"""'
    ... )  # Only include values that start with """ (marking a module docstring)
    >>> kv_filter = (
    ...     lambda kv: kv[0] not in kv[1]
    ... )  # Exclude key-value pairs where the value mentions the key

    Call the function with the provided filters and settings

    >>> result = store_aggregate(
    ...     content_store=content_store,  # The content_store dict
    ...     kv_to_item="{} -> {}".format,  # Format key-value pairs as "key -> value"
    ...     key_filter=key_filter,  # Key filter: Include only .py files
    ...     value_filter=value_filter,  # Value filter: Include only values starting with """
    ...     kv_filter=kv_filter,  # KV filter: Exclude if value contains the key
    ...     aggregator=', '.join,
    ...     egress='~/test.md'
    ... )
    >>> result
    'file1.py -> """Module docstring.""", file3.py -> """Another docstring."""'

    Here, you got the string as the result. If you want to save it to a file,
    you can provide the save_filepath argument, and it will save the text to the file,
    and return the save_filepath to you (which )

    Recipe: You can do a lot with the `kv_to_text` argument. For example, if your
    content store doesn't have string keys or values, you can always extract whatever
    information you need from them to produce the text that will represent that item.
    '''
    # Convert content_store to a dol store if it's a directory path
    if isinstance(content_store, str) and os.path.isdir(content_store):
        content_store = local_store_factory(content_store)

    if isinstance(egress, str):
        save_filepath = egress
        # make an egress that will save the string to a file (then return the string)
        egress = partial(save_string_to_filepath, save_filepath)

    # Define default filters if not provided
    key_filter = key_filter or (lambda key: True)
    value_filter = value_filter or (lambda value: True)
    kv_filter = kv_filter or (lambda kv: True)

    def actual_kv_filter(kv):
        k, v = kv
        return key_filter(k) and value_filter(v) and kv_filter(kv)

    # Create the string by applying filters and kv_to_text conversion
    filtered_kv_pairs = filter(actual_kv_filter, content_store.items())
    aggregate = aggregator(kv_to_item(k, v) for k, v in filtered_kv_pairs)

    return egress(aggregate)


# --------------------------------------- Misc ------------------------------------------

_dflt_ask_user_for_value_when_missing_msg = (
    "No such key was found. You can enter a value for it here "
    "or simply hit enter to leave the slot empty"
)


def convert_to_numerical_if_possible(s: str):
    """To be used with ``ask_user_for_value_when_missing`` ``value_preprocessor`` arg

    >>> convert_to_numerical_if_possible("123")
    123
    >>> convert_to_numerical_if_possible("123.4")
    123.4
    >>> convert_to_numerical_if_possible("one")
    'one'

    Border case: The strings "infinity" and "inf" actually convert to a valid float.

    >>> convert_to_numerical_if_possible("infinity")
    inf
    """
    try:
        s = int(s)
    except ValueError:
        try:
            s = float(s)
        except ValueError:
            pass
    return s


@store_decorator
def ask_user_for_value_when_missing(
    store=None,
    *,
    value_preprocessor: Optional[Callable] = None,
    on_missing_msg: str = _dflt_ask_user_for_value_when_missing_msg,
):
    """Wrap a store so if a value is missing when the user asks for it, they will be
    given a chance to enter the value they want to write.

    :param store: The store (instance or class) to wrap
    :param value_preprocessor: Function to transform the user value before trying to
        write it (bearing in mind all user specified values are strings)
    :param on_missing_msg: String that will be displayed to prompt the user to enter a
        value
    :return:
    """

    store = Store.wrap(store)

    def __missing__(self, k):
        user_value = input(on_missing_msg + f" Value for {k}:\n")

        if user_value:
            if value_preprocessor:
                user_value = value_preprocessor(user_value)
            self[k] = user_value
        else:
            super(type(self), self).__missing__(k)

    store.__missing__ = __missing__
    return store


class iSliceStore(Mapping):
    """
    Wraps a store to make a reader that acts as if the store was a list
    (with integer keys, and that can be sliced).
    I say "list", but it should be noted that the behavior is more that of range,
    that outputs an element of the list
    when keying with an integer, but returns an iterable object (a range) if sliced.

    Here, a map object is returned when the sliceable store is sliced.

    >>> s = {'foo': 'bar', 'hello': 'world', 'alice': 'bob'}
    >>> sliceable_s = iSliceStore(s)

    The read-only functionalities of the underlying mapping are still available:

    >>> list(sliceable_s)
    ['foo', 'hello', 'alice']
    >>> 'hello' in sliceable_s
    True
    >>> sliceable_s['hello']
    'world'

    But now you can get slices as well:

    >>> list(sliceable_s[0:2])
    ['bar', 'world']
    >>> list(sliceable_s[-2:])
    ['world', 'bob']
    >>> list(sliceable_s[:-1])
    ['bar', 'world']

    Now, you can't do `sliceable_s[1]` because `1` isn't a valid key.
    But if you really wanted "item number 1", you can do:

    >>> next(sliceable_s[1:2])
    'world'

    Note that `sliceable_s[i:j]` is an iterable that needs to be consumed
    (here, with list) to actually get the data. If you want your data in a different
    format, you can use `dol.trans.wrap_kvs` for that.

    >>> from dol import wrap_kvs
    >>> ss = wrap_kvs(sliceable_s, obj_of_data=list)
    >>> ss[1:3]
    ['world', 'bob']
    >>> sss = wrap_kvs(sliceable_s, obj_of_data=sorted)
    >>> sss[1:3]
    ['bob', 'world']
    """

    def __init__(self, store):
        self.store = store

    def _get_islice(self, k: slice):
        start, stop, step = k.start, k.stop, k.step

        assert (step is None) or (step > 0), "step of slice can't be negative"
        negative_start = start is not None and start < 0
        negative_stop = stop is not None and stop < 0
        if negative_start or negative_stop:
            n = self.__len__()
            if negative_start:
                start = n + start
            if negative_stop:
                stop = n + stop

        return islice(self.store.keys(), start, stop, step)

    def __getitem__(self, k):
        if not isinstance(k, slice):
            return self.store[k]
        else:
            return map(self.store.__getitem__, self._get_islice(k))

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __contains__(self, k):
        return k in self.store


from dol.base import KvReader
from functools import partial
from typing import Any, Iterable, Union, Callable
from itertools import islice

Src = Any
Key = Any
Val = Any

key_error_flag = type("KeyErrorFlag", (), {})()


def _isinstance(obj, class_or_tuple):
    """Same as builtin isinstance, but without the position only limitation
    that prevents from partializing class_or_tuple"""
    return isinstance(obj, class_or_tuple)


def type_check_if_type(filt):
    if isinstance(filt, type) or isinstance(filt, tuple):
        class_or_tuple = filt
        filt = partial(_isinstance, class_or_tuple=class_or_tuple)
    return filt


def return_input(x):
    return x


# TODO: Try dataclass
# TODO: Generalize forest_type to include mappings too
# @dataclass
class Forest(KvReader):
    """Provides a key-value forest interface to objects.

    A `<tree https://en.wikipedia.org/wiki/Tree_(data_structure)>`_
    is a nested data structure. A tree has a root, which is the parent of children,
    who themselves can be parents of further subtrees, or not; in which case they're
    called leafs.
    For more information, see
    `<wikipedia on trees https://en.wikipedia.org/wiki/Tree_(data_structure)>`_

    Here we allow one to construct a tree view of any python object, using a
    key-value interface to the parent-child relationship.

    A forest is a collection of trees.

    Arguably, a dictionnary might not be the most impactful example to show here, since
    it is naturally a tree (therefore a forest), and naturally key-valued: But it has
    the advantage of being easy to demo with.
    Where Forest would really be useful is when you (1) want to give a consistent
    key-value interface to the many various forms that trees and forest objects come
    in, or even more so when (2) your object's tree/forest structure is not obvious,
    so you need to "extract" that view from it (plus give it a consistent key-value
    interface, so that you can build an ecosystem of tools around it.

    Anyway, here's our dictionary example:

    >>> d = {
    ...     'apple': {
    ...         'kind': 'fruit',
    ...         'types': {
    ...             'granny': {'color': 'green'},
    ...             'fuji': {'color': 'red'}
    ...         },
    ...         'tasty': True
    ...     },
    ...     'acrobat': {
    ...         'kind': 'person',
    ...         'nationality': 'french',
    ...         'brave': True,
    ...     },
    ...     'ball': {
    ...         'kind': 'toy'
    ...     }
    ... }

    Must of the time, you'll want to curry ``Forest`` to make an ``object_to_forest``
    constructor for a given class of objects. In the case of dictionaries as the one
    above, this might look like this:

    >>> from functools import partial
    >>> a_forest = partial(
    ...     Forest,
    ...     is_leaf=lambda k, v: not isinstance(v, dict),
    ...     get_node_keys=lambda v: [vv for vv in iter(v) if not vv.startswith('b')],
    ...     get_src_item=lambda src, k: src[k]
    ... )
    >>>
    >>> f = a_forest(d)
    >>> list(f)
    ['apple', 'acrobat']

    Note that we specified in ``get_node_keys``that we didn't want to include items
    whose keys start with ``b`` as valid children. Therefore we don't have our
    ``'ball'`` in the list above.

    Note below which nodes are themselves ``Forests``, and whic are leafs:

    >>> ff = f['apple']
    >>> isinstance(ff, Forest)
    True
    >>> list(ff)
    ['kind', 'types', 'tasty']
    >>> ff['kind']
    'fruit'
    >>> fff = ff['types']
    >>> isinstance(fff, Forest)
    True
    >>> list(fff)
    ['granny', 'fuji']

    """

    def __init__(
        self,
        src: Src,
        *,
        get_node_keys: Callable[[Src], Iterable[Key]],
        get_src_item: Callable[[Src, Key], bool],
        is_leaf: Callable[[Key, Val], bool],
        forest_type: Union[type, Callable] = list,
        leaf_trans: Callable[[Val], Any] = return_input,
    ):
        """Initialize a ``Forest``

        :param src: The source of the ``Forest``. This could be any object you want.
            The following arguments should know how to handle it.
        :param get_node_keys: How to get the keys of the children of ``src``.
        :param get_src_item: How to get the value of a child of ``src`` from its key
        :param is_leaf: Determines if a ``(k, v)`` pair (child) is a leaf.
        :param forest_type: The type of a forest. Used both to determine if an object
            (must be iterable) is to be considered a forest (i.e. an iterable of sources
            that are roots of trees
        :param leaf_trans:
        """
        self.src = src
        self.get_node_keys = get_node_keys
        self.get_src_item = get_src_item
        self.is_leaf = is_leaf
        self.leaf_trans = leaf_trans
        if isinstance(forest_type, type):
            self.is_forest = isinstance(src, forest_type)
        else:
            self.is_forest = forest_type(src)
        self._forest_maker = partial(
            type(self),
            get_node_keys=get_node_keys,
            get_src_item=get_src_item,
            is_leaf=is_leaf,
            forest_type=forest_type,
            leaf_trans=leaf_trans,
        )

    def is_forest_type(self, obj):
        return isinstance(obj, list)

    def __iter__(self):
        if not self.is_forest:
            yield from self.get_node_keys(self.src)
        else:
            for i, _ in enumerate(self.src):
                yield i

    def __getitem__(self, k):
        if self.is_forest:
            assert isinstance(k, int), (
                f"When the src is a forest, you should key with an "
                f"integer. The key was {k}"
            )
            v = next(
                islice(self.src, k, k + 1), key_error_flag
            )  # TODO: raise KeyError if
            if v is key_error_flag:
                raise KeyError(f"No value for {k=}")
        else:
            v = self.get_src_item(self.src, k)
        if self.is_leaf(k, v):
            return self.leaf_trans(v)
        else:
            return self._forest_maker(v)

    def to_dict(self):
        def gen():
            for k, v in self.items():
                if isinstance(v, Forest):
                    yield k, v.to_dict()
                else:
                    yield k, v

        return dict(gen())

    def __repr__(self):
        return f"{type(self).__name__}({self.src})"
