"""
Tools to add caching layers to stores and methods.

This module provides comprehensive caching functionality for Python applications,
offering flexible and powerful caching solutions for both data stores and method calls.

Main Use Cases:

- Property caching: Cache expensive computations that only need to be run once
- Method caching: Cache method results based on arguments, with smart key generation
- Store caching: Add caching layers to data stores for improved performance
- Custom caching strategies: Flexible key generation and cache storage options

Key Tools:

- ``cache_this``: The main decorator for caching properties and methods.
  Automatically detects whether to use property or method caching based on
  function signature. Supports custom cache storage, key functions, parameter
  ignoring, and serialization hooks.
- ``CachedProperty``: A descriptor for caching property values with flexible
  cache storage and key generation strategies.
- ``CachedMethod``: A descriptor for caching method results based on arguments,
  with support for parameter filtering and custom key functions.
- ``KeyStrategy`` protocol: Extensible system for defining how cache keys are
  generated, including strategies for explicit keys, instance properties,
  method arguments, and composite keys.
- Store decorators: Tools like ``cache_vals``, ``mk_sourced_store``, and
  ``store_cached`` for adding caching layers to data stores.

The key strategies are defined in ``dol.cache_keys`` and the store caching layers in
``dol.store_caching``; this module re-exports them (i2mint/dol#70).

Examples:

    Basic property caching:

    >>> class MyClass:
    ...     @cache_this
    ...     def expensive_computation(self):
    ...         return sum(range(1000000))

    Method caching with argument-based keys:

    >>> class Calculator:
    ...     @cache_this(cache={})
    ...     def multiply(self, x, y):
    ...         return x * y

    Custom cache storage and key functions:

    >>> class DataProcessor:
    ...     def __init__(self):
    ...         self.cache = {}
    ...     @cache_this(cache='cache', ignore={'verbose'})
    ...     def process(self, data, mode='fast', verbose=False):
    ...         return len(data) if mode == 'fast' else sum(data)
"""

# -------------------------------------------------------------------------------------

import os
import types
from functools import RLock, cached_property, partial, wraps
from types import GenericAlias
from typing import Any, Optional, Protocol, TypeVar, Union
from collections.abc import Callable, Mapping, MutableMapping

from dol.base import Store
from dol.trans import store_decorator

# The key strategies moved to ``dol.cache_keys`` and the store caching layers to
# ``dol.store_caching``; they are re-exported here so ``from dol.caching import X``
# keeps working (i2mint/dol#70).
from dol.cache_keys import (
    KeyStrategy,
    register_key_strategy,
    ExplicitKey,
    ApplyToMethodName,
    InstanceProp,
    ApplyToInstance,
    FromMethodArgs,
    CompositeKey,
    _resolve_key_for_cached_prop,
    _resolve_key_for_cached_method,
    _default_method_key,
)
from dol.store_caching import (
    is_a_cache,
    get_cache,
    mk_memoizer,
    _mk_cache_instance,
    cache_vals,
    mk_cached_store,
    mk_sourced_store,
    _pre_condition_containment,
    _slow_but_somewhat_general_hash,
    store_cached,
    store_cached_with_single_key,
    ensure_clear_to_kv_store,
    flush_on_exit,
    mk_write_cached_store,
    WriteBackChainMap,
    _mk_cache_method_local_path_key,
    HashableDict,
    cache_func_outputs,
)

# Imported here before the split; kept so the same names stay importable from here.
from collections import ChainMap, deque
from collections.abc import Iterable
from inspect import signature
from dol.util import HashableMixin, has_enabled_clear_method

# Type variables
KT = TypeVar("KT")  # Key type
VT = TypeVar("VT")  # Value type
T = TypeVar("T")  # Generic type

#: Sentinel marking "no value cached yet". Distinct from every user value, so a
#: legitimately cached ``None`` is not mistaken for a cache miss.
_NOT_FOUND = object()

# Type definitions
Instance = Any
PropertyFunc = Callable[[Instance], VT]
MethodName = str
Cache = Union[MethodName, MutableMapping[KT, VT]]
KeyType = Union[KT, Callable[[MethodName], KT]]


from dol.util import identity_func as identity  # the one definition (#70)


class CachedProperty:
    """
    Descriptor that caches the result of the first call to a method.

    It generalizes the builtin functools.cached_property class, enabling the user to
    specify a cache object and a key to store the cache value.
    """

    def __init__(
        self,
        func: PropertyFunc,
        cache: Cache | None = None,
        key: str | Callable | KeyStrategy | None = None,
        *,
        allow_none_keys: bool = False,
        lock_factory: Callable = RLock,
        pre_cache: bool | MutableMapping = False,
        serialize: Callable[[Any], Any] | None = None,
        deserialize: Callable[[Any], Any] | None = None,
    ):
        """
        Initialize the cached property.

        Args:
            func: The function whose result needs to be cached.
            cache: The cache storage. Can be:

                - A MutableMapping instance (shared across all instances)
                - A string naming an instance attribute that is a MutableMapping
                - A callable that takes the instance and returns a MutableMapping
                  (allows per-instance cache customization),
                  e.g. ``cache=lambda self: Files(f'/tmp/cache_{self.user_id}/')``

            key: The key to store the cache value. Can be:

                - A string (treated as an explicit key)
                - A function (interpreted based on its signature)
                - A KeyStrategy instance

            allow_none_keys: Whether to allow None as a valid key.
            lock_factory: Factory function to create a lock.
            pre_cache: If True or a MutableMapping, adds in-memory caching.
            serialize: Optional function to serialize values before storing in cache.
                       Signature: ``(value) -> serialized_value``, e.g. ``pickle.dumps``.
            deserialize: Optional function to deserialize values retrieved from cache.
                         Signature: ``(serialized_value) -> value``, e.g. ``pickle.loads``.
        """
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = lock_factory()
        self.cache = cache
        self.key_strategy = _resolve_key_for_cached_prop(key)
        self.allow_none_keys = allow_none_keys
        self.cache_key = (
            None  # Will be set in __set_name__ if resolvable at definition time
        )
        self.serialize = serialize if serialize is not None else identity
        self.deserialize = deserialize if deserialize is not None else identity

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

        Args:
            owner: The class owning the property.
            name: The name of the property.
        """
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same CachedProperty to two different names "
                f"({self.attrname!r} and {name!r})."
            )

        # If func is a descriptor (stacked cache_this), propagate __set_name__ to it
        if isinstance(self.func, (CachedProperty, CachedMethod)):
            self.func.__set_name__(owner, name)

        # Try to resolve the key at definition time
        key = self.key_strategy.resolve_at_definition(self.attrname)

        if key is not None:
            if key is None and not self.allow_none_keys:
                raise TypeError("The key resolved at definition time cannot be None.")
            self.cache_key = key

    def _get_cache_key(self, instance):
        """
        Get the cache key for the instance.

        Args:
            instance: The instance of the class.

        Returns:
            The cache key to use.
        """
        # If we already have a cache_key from definition time, use it
        if self.cache_key is not None:
            return self.cache_key

        # Otherwise, resolve at runtime
        key = self.key_strategy.resolve_at_runtime(instance, self.attrname)

        if key is None and not self.allow_none_keys:
            raise TypeError(
                f"The key resolved at runtime for {self.attrname!r} cannot be None."
            )

        return key

    def __get_cache(self, instance):
        """
        Get the cache for the instance.

        This method handles the three main cache specification patterns:

        1. Cache factories (functions that create cache instances)
        2. Attribute names (strings referring to instance attributes)
        3. Direct cache objects (MutableMapping instances)

        Args:
            instance: The instance of the class.

        Returns:
            The cache storage.
        """
        # Pattern 1: Callable cache factories
        # These enable dynamic, instance-specific cache creation
        # Example: cache=lambda self: Files(f'/cache/{self.user_id}/')
        if callable(self.cache) and not isinstance(self.cache, type):
            # It's a factory function - call it with the instance
            import inspect

            sig = inspect.signature(self.cache)

            if len(sig.parameters) >= 1:
                # Factory expects the instance
                cache = self.cache(instance)
            else:
                # No-argument factory
                cache = self.cache()

            if not isinstance(cache, MutableMapping):
                raise TypeError(
                    f"Cache factory must return a MutableMapping, got {type(cache)}"
                )
            __cache = cache

        elif isinstance(self.cache, str):
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

        Args:
            instance: The instance of the class.
            owner: The owner class.

        Returns:
            The cached value or computed value if not cached.
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

        # Main caching flow: get cache storage, compute key, then get or compute value
        cache = self._get_cache(instance)
        cache_key = self._get_cache_key(instance)

        return self._get_or_compute(instance, cache, cache_key)

    def _get_cache(self, instance):
        """Get the cache for the instance, handling potential errors."""
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

    def _get_or_compute(self, instance, cache, cache_key):
        """Get cached value or compute it if not found."""
        val = cache.get(cache_key, _NOT_FOUND)
        if val is _NOT_FOUND:
            with self.lock:
                # Double-check pattern: verify cache is still empty after acquiring lock
                # This prevents multiple threads from computing the same value simultaneously
                val = cache.get(cache_key, _NOT_FOUND)
                if val is _NOT_FOUND:
                    # Check if func is actually a descriptor (e.g., another CachedProperty)
                    # This enables stacking of cache_this decorators
                    if isinstance(self.func, (CachedProperty, CachedMethod)):
                        # Use descriptor protocol to get the value
                        val = self.func.__get__(instance, type(instance))
                    else:
                        # Normal function call
                        val = self.func(instance)
                    try:
                        # Serialize before storing
                        cache[cache_key] = self.serialize(val)
                    except TypeError as e:
                        msg = (
                            f"The cache on {type(instance).__name__!r} instance "
                            f"does not support item assignment for caching {cache_key!r} property.\n"
                            f"Error: {e}"
                        )
                        raise TypeError(msg) from None
                    return val  # Return the unserialized value
                else:
                    # Deserialize when retrieving from cache
                    return self.deserialize(val)
        else:
            # Deserialize when retrieving from cache
            return self.deserialize(val)

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


class CachedMethod:
    """
    Descriptor that caches the result of method calls based on their arguments.

    Similar to CachedProperty but handles methods with arguments, caching results
    based on unique combinations of arguments (excluding self).
    """

    def __init__(
        self,
        func: Callable,
        cache: Cache | None = None,
        key: Callable | None = None,
        *,
        ignore: str | list[str] | None = None,
        allow_none_keys: bool = False,
        lock_factory: Callable = RLock,
        pre_cache: bool | MutableMapping = False,
        serialize: Callable[[Any], Any] | None = None,
        deserialize: Callable[[Any], Any] | None = None,
    ):
        """
        Initialize the cached method.

        Args:
            func: The function whose results need to be cached.
            cache: The cache storage. Can be:

                - A MutableMapping instance (shared across instances)
                - A string naming an instance attribute containing a MutableMapping
                - A callable taking (instance) and returning a MutableMapping
                  This enables instance-specific caching, e.g.:
                  cache=lambda self: Files(f'/cache/{self.user_id}/')
            key: Callable that takes ``(self, *args, **kwargs)`` and returns a cache key.
                 Defaults to a function that converts args/kwargs to a string.
            ignore: Parameter name(s) to exclude from cache key computation.
                Can be a string (single parameter) or list of strings (multiple parameters).
                Commonly used to ignore 'self' or parameters like 'verbose' that don't
                affect the result.

            allow_none_keys: Whether to allow None as a valid key.
            lock_factory: Factory function to create a lock.
            pre_cache: If True or a MutableMapping, adds in-memory caching.
            serialize: Optional function to serialize values before storing in cache.
                       Signature: ``(value) -> serialized_value``, e.g. ``pickle.dumps``.
            deserialize: Optional function to deserialize values retrieved from cache.
                         Signature: ``(serialized_value) -> value``, e.g. ``pickle.loads``.
        """
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = lock_factory()
        self.cache = cache

        # Handle ignore parameter
        if isinstance(ignore, str):
            ignore = [ignore]
        self.ignore = set(ignore or [])

        # Set up key strategy
        self.key_strategy = _resolve_key_for_cached_method(
            key, ignore=self.ignore, func=self.func
        )
        self.allow_none_keys = allow_none_keys
        self.serialize = serialize if serialize is not None else identity
        self.deserialize = deserialize if deserialize is not None else identity

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
        Set the name of the method.

        Args:
            owner: The class owning the method.
            name: The name of the method.
        """
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same CachedMethod to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def _get_cache_key(self, instance, *args, **kwargs):
        """
        Get the cache key for the method call.

        Args:
            instance: The instance of the class.
            *args: Arguments passed to the method.
            **kwargs: Keyword arguments passed to the method.

        Returns:
            The cache key to use.
        """
        # Resolve key using the strategy
        if hasattr(self.key_strategy, "resolve_at_runtime"):
            # Check if it's a method-aware strategy
            try:
                key = self.key_strategy.resolve_at_runtime(
                    instance, self.attrname, *args, **kwargs
                )
            except TypeError:
                # Fallback for property-style strategies
                key = self.key_strategy.resolve_at_runtime(instance, self.attrname)
        else:
            key = self.key_strategy.resolve_at_definition(self.attrname)

        if key is None and not self.allow_none_keys:
            raise TypeError(f"The key resolved for {self.attrname!r} cannot be None.")

        return key

    def __get_cache(self, instance):
        """
        Get the cache for the instance.

        Args:
            instance: The instance of the class.

        Returns:
            The cache storage.
        """
        # Handle callable cache factories
        if callable(self.cache) and not isinstance(self.cache, type):
            # It's a factory function - call it with the instance
            import inspect

            sig = inspect.signature(self.cache)

            if len(sig.parameters) >= 1:
                # Factory expects the instance
                cache = self.cache(instance)
            else:
                # No-argument factory
                cache = self.cache()

            if not isinstance(cache, MutableMapping):
                raise TypeError(
                    f"Cache factory must return a MutableMapping, got {type(cache)}"
                )
            __cache = cache

        elif isinstance(self.cache, str):
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
        Get the bound method wrapper that provides caching.

        Args:
            instance: The instance of the class.
            owner: The owner class.

        Returns:
            A bound wrapper function that caches method results.
        """
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use CachedMethod instance without calling __set_name__ on it."
            )

        if self.cache is False:
            # If cache is False, always compute the value
            return partial(self.func, instance)

        cache = self._get_cache(instance)

        @wraps(self.func)
        def cached_method_wrapper(*args, **kwargs):
            cache_key = self._get_cache_key(instance, *args, **kwargs)
            return self._get_or_compute(instance, cache, cache_key, args, kwargs)

        return cached_method_wrapper

    def _get_cache(self, instance):
        """Get the cache for the instance, handling potential errors."""
        try:
            cache = self.__get_cache(instance)
        except (
            AttributeError
        ):  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} method."
            )
            raise TypeError(msg) from None
        return cache

    def _get_or_compute(self, instance, cache, cache_key, args, kwargs):
        """Get cached value or compute it if not found."""
        val = cache.get(cache_key, _NOT_FOUND)
        if val is _NOT_FOUND:
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(cache_key, _NOT_FOUND)
                if val is _NOT_FOUND:
                    val = self.func(instance, *args, **kwargs)
                    try:
                        # Serialize before storing
                        cache[cache_key] = self.serialize(val)
                    except TypeError as e:
                        msg = (
                            f"The cache on {type(instance).__name__!r} instance "
                            f"does not support item assignment for caching {cache_key!r} method.\n"
                            f"Error: {e}"
                        )
                        raise TypeError(msg) from None
                    return val  # Return the unserialized value
                else:
                    # Deserialize when retrieving from cache
                    return self.deserialize(val)
        else:
            # Deserialize when retrieving from cache
            return self.deserialize(val)

    __class_getitem__ = classmethod(GenericAlias)


def cache_this(
    func: PropertyFunc = None,
    *,
    cache: Cache | None = None,
    key: KeyType | None = None,
    pre_cache: bool | MutableMapping = False,
    as_property: bool | None = None,
    ignore: str | list[str] | None = None,
    serialize: Callable[[Any], Any] | None = None,
    deserialize: Callable[[Any], Any] | None = None,
):
    r"""
    Unified caching decorator for properties and methods with persistent storage support.

    `cache_this` extends the capabilities of Python's built-in `functools.cached_property`
    and `functools.lru_cache` by providing:

    - **Persistent caching**: Store cached values in files, databases, or any MutableMapping
    - **Flexible cache backends**: Use instance attributes, external stores, or cache factories
    - **Smart key generation**: Automatic argument-based keys for methods with parameter filtering
    - **Serialization support**: Custom serialize/deserialize functions for complex data
    - **Auto-detection**: Automatically chooses property vs method caching based on signature
    - **No LRU eviction**: Unlike lru_cache, values persist until explicitly removed

    Unlike functools.cached_property (properties only) and lru_cache (memory-only with eviction),
    cache_this provides a unified interface for both use cases with persistent storage options.

    :param func: The function to be decorated (usually left empty).
    :param cache: The cache storage. Can be:

        - A MutableMapping instance (shared across instances)
        - A string naming an instance attribute containing a MutableMapping
        - A callable taking (instance) and returning a MutableMapping
          This enables instance-specific caching, e.g.:
          cache=lambda self: Files(f'/cache/{self.user_id}/')

    :param key: For properties: the key to store the cache value, can be a callable
        that will be applied to the method name to make a key, or an explicit string.
        For methods: a callable that takes ``(self, *args, **kwargs)`` and returns a cache key.
    :param pre_cache: Default is False. If True, adds an in-memory cache to the method
        to (also) cache the results in memory. If a MutableMapping is given, it will be
        used as the pre-cache.
        This is useful when you want a persistent cache but also want to speed up
        access to the method in the same session.
    :param as_property: If True, force use of CachedProperty. If False, force use of
        CachedMethod. If None (default), auto-detect based on function signature.
    :param ignore: Parameter name(s) to exclude from cache key computation.
        Can be a string (single parameter) or list of strings (multiple parameters).
        Commonly used to ignore 'self' or parameters like 'verbose' that don't
        affect the result.
    :param serialize: Optional function to serialize values before caching
        (e.g. ``pickle.dumps`` for binary file storage).
    :param deserialize: Optional function to deserialize cached values
        (e.g. ``pickle.loads``).
    :return: The decorated function.

    .. rubric:: Comprehensive Example

    Here's a complete example showcasing all major features of cache_this:

    >>> import tempfile
    >>> import os
    >>> from pathlib import Path
    >>>
    >>> class DataProcessor:
    ...     def __init__(self, user_id="user123"):
    ...         self.user_id = user_id
    ...         self.memory_cache = {}  # In-memory cache
    ...         self.call_counts = {}   # Track function calls for demo
    ...
    ...     # 1. Basic property caching (like functools.cached_property)
    ...     @cache_this
    ...     def basic_property(self):
    ...         '''Cached in instance.__dict__ by default'''
    ...         self.call_counts['basic_property'] = self.call_counts.get('basic_property', 0) + 1
    ...         return f"computed_value_{self.call_counts['basic_property']}"
    ...
    ...     # 2. Property with custom cache and key
    ...     @cache_this(cache='memory_cache', key='custom_prop_key')
    ...     def custom_cached_property(self):
    ...         '''Cached in instance.memory_cache with custom key'''
    ...         self.call_counts['custom_cached_property'] = self.call_counts.get('custom_cached_property', 0) + 1
    ...         return f"custom_value_{self.call_counts['custom_cached_property']}"
    ...
    ...     # 3. Method caching with argument-based keys
    ...     @cache_this(cache='memory_cache')
    ...     def compute_result(self, x, y, mode='fast'):
    ...         '''Cached based on arguments (x, y, mode)'''
    ...         key = ('compute_result', x, y, mode)
    ...         self.call_counts[key] = self.call_counts.get(key, 0) + 1
    ...         return x * y * (2 if mode == 'fast' else 3)
    ...
    ...     # 4. Method caching with ignored parameters
    ...     @cache_this(cache='memory_cache', ignore={'verbose', 'debug'})
    ...     def process_data(self, data, algorithm='default', verbose=False, debug=False):
    ...         '''Cache ignores verbose and debug parameters'''
    ...         key = ('process_data', tuple(data), algorithm)
    ...         self.call_counts[key] = self.call_counts.get(key, 0) + 1
    ...         if verbose: print(f"Processing {data} with {algorithm}")
    ...         return sum(data) * (2 if algorithm == 'default' else 3)
    ...
    ...     # 5. Instance-specific cache factory
    ...     @cache_this(cache=lambda self: {f'{self.user_id}_cache': {}}.get(f'{self.user_id}_cache'))
    ...     def user_specific_computation(self, value):
    ...         '''Each instance gets its own cache based on user_id'''
    ...         key = ('user_specific_computation', value)
    ...         self.call_counts[key] = self.call_counts.get(key, 0) + 1
    ...         return value ** 2

    Now let's test all the features:

    >>> processor = DataProcessor("alice")
    >>>
    >>> # Test basic property caching
    >>> result1 = processor.basic_property
    >>> result2 = processor.basic_property  # Should use cache
    >>> assert result1 == result2 == "computed_value_1"
    >>> assert 'basic_property' in processor.__dict__  # Cached in instance dict
    >>>
    >>> # Test custom cache and key
    >>> result1 = processor.custom_cached_property
    >>> result2 = processor.custom_cached_property  # Should use cache
    >>> assert result1 == result2 == "custom_value_1"
    >>> assert 'custom_prop_key' in processor.memory_cache
    >>>
    >>> # Test method caching with arguments
    >>> result1 = processor.compute_result(3, 4, 'fast')
    >>> result2 = processor.compute_result(3, 4, 'fast')  # Should use cache
    >>> result3 = processor.compute_result(3, 4, 'slow')  # Different args, new computation
    >>> assert result1 == result2 == 24  # 3 * 4 * 2
    >>> assert result3 == 36  # 3 * 4 * 3
    >>>
    >>> # Test parameter ignoring
    >>> result1 = processor.process_data([1, 2, 3], verbose=True)
    Processing [1, 2, 3] with default
    >>> result2 = processor.process_data([1, 2, 3], verbose=False)  # Should use same cache
    >>> result3 = processor.process_data([1, 2, 3], debug=True)     # Should use same cache
    >>> assert result1 == result2 == result3 == 12  # sum([1,2,3]) * 2
    >>>
    >>> # Test instance-specific caching
    >>> result1 = processor.user_specific_computation(5)
    >>> result2 = processor.user_specific_computation(5)  # Should use cache
    >>> assert result1 == result2 == 25  # 5 ** 2
    >>>
    >>> # Different instance should have separate cache
    >>> processor2 = DataProcessor("bob")
    >>> result3 = processor2.user_specific_computation(5)  # Fresh computation
    >>> assert result3 == 25

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

    import inspect

    def _should_use_property(func, as_property):
        """
        Determine whether to use CachedProperty or CachedMethod based on function signature.

        This is the core auto-detection logic that makes cache_this work seamlessly
        for both properties (no arguments) and methods (with arguments).
        """
        # Explicit override takes precedence
        if as_property is not None:
            return as_property

        # If func is already a CachedProperty or CachedMethod, treat it as a property
        # (since these are descriptors that work like properties)
        if isinstance(func, (CachedProperty, CachedMethod)):
            return True

        # Auto-detect based on function signature
        try:
            sig = inspect.signature(func)
            # Count all parameters except 'self', including variadic ones (*args, **kwargs)
            non_self_params = [
                p for name, p in sig.parameters.items() if name != "self"
            ]

            # Key insight: If there are any parameters beyond 'self', it's a method
            # that takes arguments and needs argument-based cache keys.
            # If only 'self' parameter exists, it's a property-like function.
            return len(non_self_params) == 0
        except (ValueError, TypeError):
            # If we can't get signature, default to property behavior for backward compatibility
            return True

    # Special case: cache=False means disable caching entirely (compute every time)
    if cache is False:
        if func is None:
            # Decorator factory: @cache_this(cache=False)
            def wrapper(f):
                return property(f)  # Just a plain property, no caching

            return wrapper
        else:
            # Direct decoration: cache_this(some_func, cache=False)
            return property(func)

    # The main case: cache is enabled (default or explicitly set)
    else:
        if func is None:
            # Decorator factory case: @cache_this() or @cache_this(cache=..., key=...)
            def wrapper(f):
                # This is where the magic happens: auto-detect property vs method
                if _should_use_property(f, as_property):
                    # Function has no arguments beyond 'self' -> use property caching
                    return CachedProperty(
                        f,
                        cache=cache,
                        key=key,
                        pre_cache=pre_cache,
                        serialize=serialize,
                        deserialize=deserialize,
                    )
                else:
                    # Function has arguments beyond 'self' -> use method caching
                    return CachedMethod(
                        f,
                        cache=cache,
                        key=key,
                        pre_cache=pre_cache,
                        ignore=ignore,
                        serialize=serialize,
                        deserialize=deserialize,
                    )

            return wrapper

        else:  #   If func is given, we want to return the appropriate cached instance
            if _should_use_property(func, as_property):
                return CachedProperty(
                    func,
                    cache=cache,
                    key=key,
                    pre_cache=pre_cache,
                    serialize=serialize,
                    deserialize=deserialize,
                )
            else:
                return CachedMethod(
                    func,
                    cache=cache,
                    key=key,
                    pre_cache=pre_cache,
                    ignore=ignore,
                    serialize=serialize,
                    deserialize=deserialize,
                )


# add the key strategies as attributes of cache_this to have them easily accessible
for _key_strategy in KeyStrategy.registered_key_strategies:
    setattr(cache_this, _key_strategy.__name__, _key_strategy)


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

    Note:
        If you want to add an extension to a name that already has an extension,
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

    Args:
        func: The method to be decorated. If not provided, a partially applied
            decorator will be returned for later application.
        maxsize: The maximum size of the cache.
        typed: If True, cache entries will be different based on argument types,
            such as distinguishing between `1` and `1.0`.

    Returns:
        A wrapped function with LRU caching applied, ignoring the first argument (`self`).

    .. rubric:: Example

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

    Args:
        func: The method to be decorated. If not provided, a partially applied
            decorator will be returned for later application.
        maxsize: The maximum size of the cache.
        typed: If True, cache entries will be different based on argument types,
            such as distinguishing between `1` and `1.0`.

    Returns:
        A wrapped function with LRU caching applied, ignoring the first argument (`self`).

    .. rubric:: Example

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

    .. rubric:: Example

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
