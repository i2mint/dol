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

cache_this:
    The main decorator for caching properties and methods. Automatically detects
    whether to use property or method caching based on function signature.
    Supports custom cache storage, key functions, parameter ignoring, and
    serialization hooks.

CachedProperty:
    A descriptor for caching property values with flexible cache storage and
    key generation strategies.

CachedMethod:
    A descriptor for caching method results based on arguments, with support
    for parameter filtering and custom key functions.

KeyStrategy Protocol:
    Extensible system for defining how cache keys are generated, including
    strategies for explicit keys, instance properties, method arguments, and
    composite keys.

Store Decorators:
    Tools like cache_vals, mk_sourced_store, and store_cached for adding
    caching layers to data stores.

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
from typing import Optional, Callable, KT, VT, Any, Union, T
from collections.abc import Mapping

from dol.base import Store
from dol.trans import store_decorator

from functools import RLock, cached_property
from types import GenericAlias
from collections.abc import MutableMapping

Instance = Any
PropertyFunc = Callable[[Instance], VT]
MethodName = str
Cache = Union[MethodName, MutableMapping[KT, VT]]
KeyType = Union[KT, Callable[[MethodName], KT]]


def identity(x: T) -> T:
    """Identity function that returns its input unchanged.

    >>> identity(42)
    42
    >>> identity("hello")
    'hello'
    >>> identity([1, 2, 3])
    [1, 2, 3]
    """
    return x


from functools import RLock, partial, wraps
from types import GenericAlias
from collections.abc import MutableMapping
from typing import Optional, Callable, TypeVar, Union, Any, Protocol

# Type variables
KT = TypeVar("KT")  # Key type
VT = TypeVar("VT")  # Value type
T = TypeVar("T")  # Generic type

# Constants
_NOT_FOUND = object()

# Type definitions
Instance = Any
PropertyFunc = Callable[[Instance], VT]
MethodName = str
Cache = Union[MethodName, MutableMapping[KT, VT]]


def identity(x: T) -> T:
    """Identity function that returns its input unchanged."""
    return x


class KeyStrategy(Protocol):
    """Protocol defining how a key strategy should behave."""

    registered_key_strategies = set()

    def resolve_at_definition(self, method_name: str) -> Optional[Any]:
        """
        Attempt to resolve the key at class definition time.

        Args:
            method_name: The name of the method being decorated.

        Returns:
            The resolved key or None if it can't be resolved at definition time.
        """
        ...

    def resolve_at_runtime(self, instance: Any, method_name: str) -> Any:
        """
        Resolve the key at runtime.
        By default, this will call resolve_at_definition on method_name.

        Args:
            instance: The instance the property is being accessed on.
            method_name: The name of the method being decorated.

        Returns:
            The resolved key.
        """
        return self.resolve_at_definition(method_name)


def register_key_strategy(cls):
    """Register a class as a KeyStrategy."""
    KeyStrategy.registered_key_strategies.add(cls)
    return cls


@register_key_strategy
class ExplicitKey:
    """Use an explicitly provided key value.

    >>> strategy = ExplicitKey("my_key")
    >>> strategy.resolve_at_definition("method_name")
    'my_key'
    """

    def __init__(self, key: Any):
        """
        Initialize with an explicit key value.

        Args:
            key: The explicit key to use.
        """
        self.key = key

    def resolve_at_definition(self, method_name: str) -> Any:
        """Return the explicit key value at definition time."""
        return self.key


@register_key_strategy
class ApplyToMethodName:
    """Apply a function to the method name to generate the key.

    >>> strategy = ApplyToMethodName(lambda name: f"{name}.cache")
    >>> strategy.resolve_at_definition("my_method")
    'my_method.cache'
    """

    def __init__(self, func: Callable[[str], Any]):
        """
        Initialize with a function to apply to the method name.

        Args:
            func: A function that takes a method name and returns a key.
        """
        self.func = func

    def resolve_at_definition(self, method_name: str) -> Any:
        """Apply the function to the method name at definition time."""
        return self.func(method_name)


@register_key_strategy
class InstanceProp:
    """Get a key from an instance property."""

    def __init__(self, prop_name: str):
        """
        Initialize with the name of the instance property to use as a key.

        Args:
            prop_name: The name of the property to get from the instance.
        """
        self.prop_name = prop_name

    def resolve_at_definition(self, method_name: str) -> None:
        """Cannot resolve at definition time, need the instance."""
        return None

    def resolve_at_runtime(self, instance: Any, method_name: str) -> Any:
        """Get the property value from the instance at runtime."""
        return getattr(instance, self.prop_name)


@register_key_strategy
class ApplyToInstance:
    """Apply a function to the instance to generate the key."""

    def __init__(self, func: Callable[[Any], Any]):
        """
        Initialize with a function to apply to the instance.

        Args:
            func: A function that takes an instance and returns a key.
        """
        self.func = func

    def resolve_at_definition(self, method_name: str) -> None:
        """Cannot resolve at definition time, need the instance."""
        return None

    def resolve_at_runtime(self, instance: Any, method_name: str) -> Any:
        """Apply the function to the instance at runtime."""
        return self.func(instance)


@register_key_strategy
class FromMethodArgs:
    """
    Apply a function to method arguments to generate the key.

    The function receives (self, *args, **kwargs) and should return a cache key.
    """

    def __init__(self, func: Callable):
        """
        Initialize with a function to apply to method arguments.

        Args:
            func: A function that takes (self, *args, **kwargs) and returns a key.
                  Example: lambda self, x, y: f'{x}_{y}'
        """
        self.func = func

    def resolve_at_definition(self, method_name: str) -> None:
        """Cannot resolve at definition time, need the arguments."""
        return None

    def resolve_at_runtime(
        self, instance: Any, method_name: str, *args, **kwargs
    ) -> Any:
        """Apply the function to the instance and method arguments at runtime."""
        return self.func(instance, *args, **kwargs)


@register_key_strategy
class CompositeKey:
    """
    Combine multiple key strategies into a single composite key.

    Useful for creating keys that depend on both instance properties and method arguments.
    """

    def __init__(self, *strategies, separator: str = "_"):
        """
        Initialize with multiple key strategies to combine.

        Args:
            *strategies: KeyStrategy instances to combine
            separator: String to use when joining key parts (default: '_')
        """
        self.strategies = strategies
        self.separator = separator

    def resolve_at_definition(self, method_name: str) -> Optional[Any]:
        """Try to resolve all strategies at definition time."""
        parts = []
        any_unresolved = False

        for strategy in self.strategies:
            part = strategy.resolve_at_definition(method_name)
            if part is None:
                any_unresolved = True
            else:
                parts.append(str(part))

        # If any strategy can't be resolved at definition time, return None
        if any_unresolved:
            return None

        return self.separator.join(parts) if parts else None

    def resolve_at_runtime(
        self, instance: Any, method_name: str, *args, **kwargs
    ) -> Any:
        """Resolve all strategies at runtime and combine them."""
        import inspect

        parts = []

        for strategy in self.strategies:
            # Try runtime resolution first (works for both property and method strategies)
            if hasattr(strategy, "resolve_at_runtime"):
                try:
                    # Check if it's a method-aware strategy by inspecting signature
                    sig = inspect.signature(strategy.resolve_at_runtime)
                    if len(sig.parameters) > 2:
                        # Method-aware strategy - pass args and kwargs
                        part = strategy.resolve_at_runtime(
                            instance, method_name, *args, **kwargs
                        )
                    else:
                        # Property strategy - only pass instance and method_name
                        part = strategy.resolve_at_runtime(instance, method_name)
                except TypeError:
                    # Fallback to definition-time resolution
                    part = strategy.resolve_at_definition(method_name)
            else:
                # Fallback to definition-time resolution
                part = strategy.resolve_at_definition(method_name)

            if part is not None:
                parts.append(str(part))

        return self.separator.join(parts) if parts else None


def _resolve_key_for_cached_prop(key: Any) -> KeyStrategy:
    """
    Convert a key specification to a KeyStrategy instance.

    Args:
        key: The key specification, can be a string, function, or KeyStrategy.

    Returns:
        A KeyStrategy instance.
    """
    if key is None:
        # Default to using the method name as the key
        return ApplyToMethodName(lambda x: x)

    if isinstance(key, tuple(KeyStrategy.registered_key_strategies)):
        # Already a KeyStrategy instance
        return key

    if isinstance(key, str):
        # Explicit string key
        return ExplicitKey(key)

    if callable(key):
        # Check the signature to determine the right strategy
        if hasattr(key, "__code__"):
            co_varnames = key.__code__.co_varnames

            if (
                key.__code__.co_argcount > 0
                and co_varnames
                and co_varnames[0] in ("instance", "self")
            ):
                # Function that takes an instance as first arg
                return ApplyToInstance(key)
            else:
                # Function that operates on something else (like method name)
                return ApplyToMethodName(key)
        else:
            # Callable without a __code__ attribute (like partial)
            return ApplyToMethodName(key)

    # For any other type, treat as an explicit key
    return ExplicitKey(key)


def _resolve_key_for_cached_method(
    key: Any, ignore: set = None, func: Callable = None
) -> KeyStrategy:
    """
    Convert a key specification to a KeyStrategy instance for methods.

    This function is the heart of the flexible key generation system. It takes
    various types of key specifications and converts them into a standardized
    KeyStrategy object that knows how to generate cache keys.

    Args:
        key: The key specification, can be a string, function, or KeyStrategy.
        ignore: Set of parameter names to ignore when using default key function.
        func: The function being cached (needed for default key function).

    Returns:
        A KeyStrategy instance suitable for methods with arguments.
    """
    ignore = ignore or set()

    if key is None:
        # Default case: Auto-generate keys from method arguments
        # This creates keys like "x=1;y=2;mode=fast" from method arguments
        def default_key_func(inst, *args, **kwargs):
            return _default_method_key(func, inst, *args, ignore=ignore, **kwargs)

        return FromMethodArgs(default_key_func)

    if isinstance(key, tuple(KeyStrategy.registered_key_strategies)):
        # User provided a pre-built KeyStrategy instance - use it directly
        return key

    if isinstance(key, str):
        # User provided a fixed string - all method calls use the same cache key
        # Useful when you want to cache only the latest call result
        return ExplicitKey(key)

    if callable(key):
        # User provided a custom function to generate keys from method arguments
        # We need to inspect the function to understand how to use it
        import inspect

        if hasattr(key, "__code__"):
            sig = inspect.signature(key)
            params = list(sig.parameters.keys())

            # Check what the callable expects
            if len(params) == 0:
                # No parameters - use as explicit key
                return ExplicitKey(key())
            elif len(params) == 1 and params[0] in ("self", "instance"):
                # Only takes instance - wrap in ApplyToInstance
                return ApplyToInstance(key)
            else:
                # Takes arguments - use FromMethodArgs
                return FromMethodArgs(key)
        else:
            # Callable without __code__ (like partial) - assume it takes full args
            return FromMethodArgs(key)

    # For any other type, treat as an explicit key
    return ExplicitKey(key)


class CachedProperty:
    """
    Descriptor that caches the result of the first call to a method.

    It generalizes the builtin functools.cached_property class, enabling the user to
    specify a cache object and a key to store the cache value.
    """

    def __init__(
        self,
        func: PropertyFunc,
        cache: Optional[Cache] = None,
        key: Optional[Union[str, Callable, KeyStrategy]] = None,
        *,
        allow_none_keys: bool = False,
        lock_factory: Callable = RLock,
        pre_cache: Union[bool, MutableMapping] = False,
        serialize: Optional[Callable[[Any], Any]] = None,
        deserialize: Optional[Callable[[Any], Any]] = None,
    ):
        """
        Initialize the cached property.

        Args:
            func: The function whose result needs to be cached.
            cache: The cache storage. Can be:
                - A MutableMapping instance (shared across all instances)
                - A string naming an instance attribute that is a MutableMapping
                - A callable that takes the instance and returns a MutableMapping
                  (allows per-instance cache customization)
                Example: cache=lambda self: Files(f'/tmp/cache_{self.user_id}/')
            key: The key to store the cache value. Can be:
                - A string (treated as an explicit key)
                - A function (interpreted based on its signature)
                - A KeyStrategy instance
            allow_none_keys: Whether to allow None as a valid key.
            lock_factory: Factory function to create a lock.
            pre_cache: If True or a MutableMapping, adds in-memory caching.
            serialize: Optional function to serialize values before storing in cache.
                       Signature: (value) -> serialized_value
                       Example: serialize=pickle.dumps
            deserialize: Optional function to deserialize values retrieved from cache.
                         Signature: (serialized_value) -> value
                         Example: deserialize=pickle.loads
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


def _default_method_key(func, self, *args, ignore=None, **kwargs):
    """
    Default key function for CachedMethod.

    Uses inspect.signature to bind all arguments to their parameter names,
    then creates a semicolon-separated string of param_name=value pairs.

    Args:
        func: The function being cached
        self: The instance (first argument of the method)
        *args: Positional arguments passed to the method
        **kwargs: Keyword arguments passed to the method
        ignore: Set of parameter names to exclude from the key

    Returns:
        A string like "x=1;y=2;mode=fast" representing all arguments

    Examples:
        >>> def sample_method(self, x, y, mode='fast'): pass
        >>> _default_method_key(sample_method, None, 1, 2)
        'x=1;y=2;mode=fast'
        >>> _default_method_key(sample_method, None, 1, y=3, mode='slow')
        'x=1;y=3;mode=slow'
        >>> _default_method_key(sample_method, None, 1, 2, ignore={'mode'})
        'x=1;y=2'
    """
    import inspect

    ignore = ignore or set()

    # Get the signature of the wrapped function
    sig = inspect.signature(func)

    # Bind the arguments (excluding 'self')
    # We pass args and kwargs as if calling the function without self
    try:
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
    except TypeError:
        # Fallback if binding fails - shouldn't happen but be defensive
        key_parts = [str(arg) for arg in args]
        key_parts.extend(
            f"{k}={v}" for k, v in sorted(kwargs.items()) if k not in ignore
        )
        return ";".join(key_parts) if key_parts else ""

    # Build key from bound arguments
    key_parts = []
    for param_name, param_value in bound.arguments.items():
        # Skip 'self' and any ignored parameters
        if param_name == "self" or param_name in ignore:
            continue

        # Handle VAR_POSITIONAL (*args)
        if sig.parameters[param_name].kind == inspect.Parameter.VAR_POSITIONAL:
            if param_value:  # Only include if non-empty
                key_parts.append(f"{param_name}={param_value}")

        # Handle VAR_KEYWORD (**kwargs)
        elif sig.parameters[param_name].kind == inspect.Parameter.VAR_KEYWORD:
            # Add each kwarg as a separate key=value pair
            for k, v in sorted(param_value.items()):
                if k not in ignore:
                    key_parts.append(f"{k}={v}")

        # Handle regular parameters
        else:
            key_parts.append(f"{param_name}={param_value}")

    return ";".join(key_parts) if key_parts else ""


class CachedMethod:
    """
    Descriptor that caches the result of method calls based on their arguments.

    Similar to CachedProperty but handles methods with arguments, caching results
    based on unique combinations of arguments (excluding self).
    """

    def __init__(
        self,
        func: Callable,
        cache: Optional[Cache] = None,
        key: Optional[Callable] = None,
        *,
        ignore: Optional[Union[str, list[str]]] = None,
        allow_none_keys: bool = False,
        lock_factory: Callable = RLock,
        pre_cache: Union[bool, MutableMapping] = False,
        serialize: Optional[Callable[[Any], Any]] = None,
        deserialize: Optional[Callable[[Any], Any]] = None,
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
            key: Callable that takes (self, *args, **kwargs) and returns a cache key.
                 Defaults to a function that converts args/kwargs to a string.
            ignore: Parameter name(s) to exclude from cache key computation.
                Can be a string (single parameter) or list of strings (multiple parameters).
                Commonly used to ignore 'self' or parameters like 'verbose' that don't
                affect the result.
            allow_none_keys: Whether to allow None as a valid key.
            lock_factory: Factory function to create a lock.
            pre_cache: If True or a MutableMapping, adds in-memory caching.
            serialize: Optional function to serialize values before storing in cache.
                       Signature: (value) -> serialized_value
                       Example: serialize=pickle.dumps
            deserialize: Optional function to deserialize values retrieved from cache.
                         Signature: (serialized_value) -> value
                         Example: deserialize=pickle.loads
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
    cache: Optional[Cache] = None,
    key: Optional[KeyType] = None,
    pre_cache: Union[bool, MutableMapping] = False,
    as_property: Optional[bool] = None,
    ignore: Optional[Union[str, list[str]]] = None,
    serialize: Optional[Callable[[Any], Any]] = None,
    deserialize: Optional[Callable[[Any], Any]] = None,
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
        For methods: a callable that takes (self, *args, **kwargs) and returns a cache key.
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
    :param serialize: Optional function to serialize values before caching.
        Example: serialize=pickle.dumps for binary file storage
    :param deserialize: Optional function to deserialize cached values.
        Example: deserialize=pickle.loads
    :return: The decorated function.

    ## Comprehensive Example

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


# -------------------------------------------------------------------------------------
import os
from functools import wraps, partial
from typing import Iterable, Callable
from inspect import signature

from dol.trans import store_decorator


def is_a_cache(obj):
    """Check if an object implements the cache interface.

    A cache object must have __contains__, __getitem__, and __setitem__ methods.

    >>> is_a_cache({})  # dict is a valid cache
    True
    >>> is_a_cache([])  # list has these methods but for indexed access
    True
    >>> is_a_cache("string")  # string is not (immutable)
    False
    """
    return all(
        map(
            partial(hasattr, obj),
            ("__contains__", "__getitem__", "__setitem__"),
        )
    )


def get_cache(cache):
    """Convenience function to get a cache (whether it's already an instance, or needs to be validated).

    >>> get_cache({'a': 1})  # Return existing cache instance
    {'a': 1}
    >>> get_cache(dict)()  # Return result of calling cache factory
    {}
    """
    if is_a_cache(cache):
        return cache
    elif callable(cache) and len(signature(cache).parameters) == 0:
        return cache()  # consider it to be a cache factory, and call to make factory


# -------------------------------------------------------------------------------------
# Read caching


# The following is a "Cache-Aside" read-cache with NO builtin cache update or refresh mechanism.
def mk_memoizer(cache):
    """
    Make a memoizer that caches the output of a getter function in a cache.

    Note: This is a specialized memoizer for getter functions/methods, i.e.
    functions/methods that have the signature (instance, key) and return a value.

    :param cache: The cache to use. Must have __getitem__ and __setitem__ methods.
    :return: A memoizer that caches the output of the function in the cache.

    >>> cache = dict()
    >>> @mk_memoizer(cache)
    ... def getter(self, k):
    ...     print(f"getting value for {k}...")
    ...     return k * 10
    ...
    >>> getter(None, 2)
    getting value for 2...
    20
    >>> getter(None, 2)
    20

    """

    def memoize(method):
        @wraps(method)
        def memoizer(self, k):
            if k not in cache:
                val = method(self, k)
                cache[k] = val  # cache it
                return val
            else:
                return cache[k]

        return memoizer

    return memoize


def _mk_cache_instance(cache=None, assert_attrs=()):
    """Make a cache store (if it's not already) from a type or a callable, or just return dict.
    Also assert the presence of given attributes

    >>> _mk_cache_instance(dict(a=1, b=2))
    {'a': 1, 'b': 2}
    >>> _mk_cache_instance(None)
    {}
    >>> _mk_cache_instance(dict)
    {}
    >>> _mk_cache_instance(list, ('__getitem__', '__setitem__'))
    []
    >>> _mk_cache_instance(tuple, ('__getitem__', '__setitem__'))
    Traceback (most recent call last):
        ...
    AssertionError: cache should have the __setitem__ method, but does not: ()

    """
    if isinstance(assert_attrs, str):
        assert_attrs = (assert_attrs,)
    if cache is None:
        cache = {}  # use a dict (memory caching) by default
    elif isinstance(cache, type) or (  # if caching_store is a type...
        not hasattr(cache, "__getitem__")  # ... or is a callable without a __getitem__
        and callable(cache)
    ):
        cache = (
            cache()
        )  # ... assume it's a no-argument callable that makes the instance
    for method in assert_attrs or ():
        assert hasattr(
            cache, method
        ), f"cache should have the {method} method, but does not: {cache}"
    return cache


# TODO: Make it so that the resulting store gets arguments to construct it's own cache
#   right now, only cache instances or no-argument cache types can be used.
#


@store_decorator
def cache_vals(store=None, *, cache=dict):
    """

    Args:
        store: The class of the store you want to cache
        cache: The store you want to use to cache. Anything with a __setitem__(k, v) and a __getitem__(k).
            By default, it will use a dict

    Returns: A subclass of the input store, but with caching (to the cache store)

    >>> from dol.caching import cache_vals
    >>> import time
    >>> class SlowDict(dict):
    ...     sleep_s = 0.2
    ...     def __getitem__(self, k):
    ...         time.sleep(self.sleep_s)
    ...         return super().__getitem__(k)
    ...
    ...
    >>> d = SlowDict({'a': 1, 'b': 2, 'c': 3})
    >>>
    >>> d['a']  # Wow! Takes a long time to get 'a'
    1
    >>> cache = dict()
    >>> CachedSlowDict = cache_vals(store=SlowDict, cache=cache)
    >>>
    >>> s = CachedSlowDict({'a': 1, 'b': 2, 'c': 3})
    >>> print(f"store: {list(s)}\\ncache: {list(cache)}")
    store: ['a', 'b', 'c']
    cache: []
    >>> # This will take a LONG time because it's the first time we ask for 'a'
    >>> v = s['a']
    >>> print(f"store: {list(s)}\\ncache: {list(cache)}")
    store: ['a', 'b', 'c']
    cache: ['a']
    >>> # This will take very little time because we have 'a' in the cache
    >>> v = s['a']
    >>> print(f"store: {list(s)}\\ncache: {list(cache)}")
    store: ['a', 'b', 'c']
    cache: ['a']
    >>> # But we don't have 'b'
    >>> v = s['b']
    >>> print(f"store: {list(s)}\\ncache: {list(cache)}")
    store: ['a', 'b', 'c']
    cache: ['a', 'b']
    >>> # But now we have 'b'
    >>> v = s['b']
    >>> print(f"store: {list(s)}\\ncache: {list(cache)}")
    store: ['a', 'b', 'c']
    cache: ['a', 'b']
    >>> s['d'] = 4  # and we can do things normally (like put stuff in the store)
    >>> print(f"store: {list(s)}\\ncache: {list(cache)}")
    store: ['a', 'b', 'c', 'd']
    cache: ['a', 'b']
    >>> s['d']  # if we ask for it again though, it will take time (the first time)
    4
    >>> print(f"store: {list(s)}\\ncache: {list(cache)}")
    store: ['a', 'b', 'c', 'd']
    cache: ['a', 'b', 'd']
    >>> # Of course, we could write 'd' in the cache as well, to get it quicker,
    >>> # but that's another story: The story of write caches!
    >>>
    >>> # And by the way, your "cache wrapped" store hold a pointer to the cache it's using,
    >>> # so you can take a peep there if needed:
    >>> s._cache
    {'a': 1, 'b': 2, 'd': 4}
    """

    # cache = _mk_cache_instance(cache, assert_attrs=('__getitem__', '__setitem__'))
    assert isinstance(
        store, type
    ), f"store should be a type, was a {type(store)}: {store}"

    class CachedStore(store):
        @wraps(store)
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._cache = _mk_cache_instance(
                cache,
                assert_attrs=("__getitem__", "__contains__", "__setitem__"),
            )
            # self.__getitem__ = mk_memoizer(self._cache)(self.__getitem__)

        def __getitem__(self, k):
            if k not in self._cache:
                val = super(type(self), self).__getitem__(k)
                self._cache[k] = val  # cache it
                return val
            else:
                return self._cache[k]

    return CachedStore


mk_cached_store = cache_vals  # backwards compatibility alias


@store_decorator
def mk_sourced_store(store=None, *, source=None, return_source_data=True):
    """

    Args:
        store: The class of the store you want to cache
        cache: The store you want to use to cache. Anything with a __setitem__(k, v) and a __getitem__(k).
            By default, it will use a dict
        return_source_data:
    Returns: A subclass of the input store, but with caching (to the cache store)


    :param store: The class of the store you're talking to. This store acts as the cache
    :param source: The store that is used to populate the store (cache) when a key is missing there.
    :param return_source_data:
        If True, will return ``source[k]`` as is. This should be used only if ``store[k]`` would return the same.
        If False, will first write to cache (``store[k] = source[k]``) then return ``store[k]``.
        The latter introduces a performance hit (we write and then read again from the cache),
        but ensures consistency (and is useful if the writing or the reading to/from store
        transforms the data in some way.
    :return: A decorated store

    Here are two stores pretending to be local and remote data stores respectively.

    >>> from dol.caching import mk_sourced_store
    >>>
    >>> class Local(dict):
    ...     def __getitem__(self, k):
    ...         print(f"looking for {k} in Local")
    ...         return super().__getitem__(k)
    >>>
    >>> class Remote(dict):
    ...     def __getitem__(self, k):
    ...         print(f"looking for {k} in Remote")
    ...         return super().__getitem__(k)


    Let's make a remote store with two elements in it, and a local store class that asks the remote store for stuff
    if it can't find it locally.

    >>> remote = Remote({'foo': 'bar', 'hello': 'world'})
    >>> SourcedLocal = mk_sourced_store(Local, source=remote)
    >>> s = SourcedLocal({'some': 'local stuff'})
    >>> list(s)  # the local store has one key
    ['some']

    # but if we ask for a key that is in the remote store, it provides it

    >>> assert s['foo'] == 'bar'
    looking for foo in Local
    looking for foo in Remote

    >>> list(s)
    ['some', 'foo']

    See that next time we ask for the 'foo' key, the local store provides it:

    >>> assert s['foo'] == 'bar'
    looking for foo in Local

    >>> assert s['hello'] == 'world'
    looking for hello in Local
    looking for hello in Remote
    >>> list(s)
    ['some', 'foo', 'hello']

    We can still add stuff (locally)...

    >>> s['something'] = 'else'
    >>> list(s)
    ['some', 'foo', 'hello', 'something']
    """
    assert source is not None, "You need to specify a source"

    source = _mk_cache_instance(source, assert_attrs=("__getitem__",))

    assert isinstance(
        store, type
    ), f"store should be a type, was a {type(store)}: {store}"

    if return_source_data:

        class SourcedStore(store):
            _src = source

            def __missing__(self, k):
                # if you didn't have it "locally", ask src for it
                v = self._src[k]  # ... get it from _src,
                self[k] = v  # ... store it in self
                return v  # ... and return it.

    else:

        class SourcedStore(store):
            _src = source

            def __missing__(self, k):
                # if you didn't have it "locally", ask src for it
                v = self._src[k]  # ... get it from _src,
                self[k] = v  # ... store it in self
                return self[k]  # retrieve it again and return

    return SourcedStore


# cache = _mk_cache_instance(cache, assert_attrs=('__getitem__',))
# assert isinstance(store, type), f"store should be a type, was a {type(store)}: {store}"
#
# class CachedStore(store):
#     _cache = cache
#
#     @mk_memoizer(cache)
#     def __getitem__(self, k):
#         return super().__getitem__(k)
#
# return CachedStore


# TODO: Didn't finish this. Finish, doctest, and remove underscore
def _pre_condition_containment(store=None, *, bool_key_func):
    """Adds a custom boolean key function `bool_key_func` before the store_cls.__contains__ check is performed.

    It is meant to be used to create smart read caches.

    This can be used, for example, to perform TTL caching by having `bool_key_func` check on how long
    ago a cache item has been created, and returning False if the item is past it's expiry time.
    """

    class PreContaimentStore(store):
        def __contains__(self, k):
            return bool_key_func(k) and super().__contains__(k)

    return PreContaimentStore


def _slow_but_somewhat_general_hash(*args, **kwargs):
    """
    Attempts to create a hash of the inputs, recursively resolving the most common hurdles (dicts, sets, lists)
    Returns: A hash value for the input

    >>> _slow_but_somewhat_general_hash(1, [1, 2], a_set={1,2}, a_dict={'a': 1, 'b': [1,2]})
    ((1, (1, 2)), (('a_set', (1, 2)), ('a_dict', (('a', 1), ('b', (1, 2))))))
    """
    if len(kwargs) == 0 and len(args) == 1:
        single_val = args[0]
        if hasattr(single_val, "items"):
            return tuple(
                (k, _slow_but_somewhat_general_hash(v)) for k, v in single_val.items()
            )
        elif isinstance(single_val, (set, list)):
            return tuple(single_val)
        else:
            return single_val
    else:
        return (
            tuple(_slow_but_somewhat_general_hash(x) for x in args),
            tuple((k, _slow_but_somewhat_general_hash(v)) for k, v in kwargs.items()),
        )


# TODO: Could add an empty_cache function attribute.
#  Wrap the store cache to track new keys, and delete those (and only those!!) when emptying the store.
def store_cached(store, key_func: Callable):
    """
    Function output memorizer but using a specific (usually persisting) store as it's
    memory and a key_func to compute the key under which to store the output.

    The key can be
    - a single value under which the output should be stored, regardless of the input.
    - a key function that is called on the inputs to create a hash under which the function's output should be stored.

    Args:
        store: The key-value store to use for caching. Must support __getitem__ and __setitem__.
        key_func: The key function that is called on the input of the function to create the key value.

    Note: Union[Callable, Any] is equivalent to just Any, but reveals the two cases of a key more clearly.
    Note: No, Union[Callable, Hashable] is not better. For one, general store keys are not restricted to hashable keys.
    Note: No, they shouldn't.

    See Also: store_cached_with_single_key (for a version where the cache store key doesn't depend on function's args)

    >>> # Note: Our doc test will use dict as the store, but to make the functionality useful beyond existing
    >>> # RAM-memorizer, you should use actual "persisting" stores that store in local files, or DBs, etc.
    >>> store = dict()
    >>> @store_cached(store, lambda *args: args)
    ... def my_data(x, y):
    ...     print("Pretend this is a long computation")
    ...     return x + y
    >>> t = my_data(1, 2)  # note the print below (because the function is called
    Pretend this is a long computation
    >>> tt = my_data(1, 2)  # note there's no print (because the function is NOT called)
    >>> assert t == tt
    >>> tt
    3
    >>> my_data(3, 4)  # but different inputs will trigger the actual function again
    Pretend this is a long computation
    7
    >>> my_data._cache
    {(1, 2): 3, (3, 4): 7}
    """
    assert callable(key_func), (
        "key_func should be a callable: "
        "It's called on the wrapped function's input to make a key for the caching store."
    )

    def func_wrapper(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            key = key_func(*args, **kwargs)
            if key in store:  # if the store has that key...
                return store[key]  # ... just return the data cached under this key
            else:  # if the store doesn't have it...
                output = func(
                    *args, **kwargs
                )  # ... call the function and get the output
                store[key] = output  # store the output under the key
                return output

        wrapped_func._cache = store
        return wrapped_func

    return func_wrapper


def store_cached_with_single_key(store, key):
    """
    Function output memorizer but using a specific store and key as its memory.

    Use in situations where you have a argument-less function or bound method that computes some data whose dependencies
    are static enough that there's enough advantage to make the data refresh explicit (by deleting the cache entry)
    instead of making it implicit (recomputing/refetching the data every time).

    The key should be a single value under which the output should be stored, regardless of the input.

    Note: The wrapped function comes with a empty_cache attribute, which when called, empties the cache (i.e. removes
    the key from the store)

    Note: The wrapped function has a hidden `_cache` attribute pointing to the store in case you need to peep into it.

    Args:
        store: The cache. The key-value store to use for caching. Must support __getitem__ and __setitem__.
        key: The store key under which to store the output of the function.

    Note: Union[Callable, Any] is equivalent to just Any, but reveals the two cases of a key more clearly.
    Note: No, Union[Callable, Hashable] is not better. For one, general store keys are not restricted to hashable keys.
    Note: No, they shouldn't.

    See Also: store_cached (for a version whose keys are computed from the wrapped function's input.

    >>> # Note: Our doc test will use dict as the store, but to make the functionality useful beyond existing
    >>> # RAM-memorizer, you should use actual "persisting" stores that store in local files, or DBs, etc.
    >>> store = dict()
    >>> @store_cached_with_single_key(store, 'whatevs')
    ... def my_data():
    ...     print("Pretend this is a long computation")
    ...     return [1, 2, 3]
    >>> t = my_data()  # note the print below (because the function is called
    Pretend this is a long computation
    >>> tt = my_data()  # note there's no print (because the function is NOT called)
    >>> assert t == tt
    >>> tt
    [1, 2, 3]
    >>> my_data._cache  # peep in the cache
    {'whatevs': [1, 2, 3]}
    >>> # let's empty the cache
    >>> my_data.empty_cache_entry()
    >>> assert 'whatevs' not in my_data._cache  # see that the cache entry is gone.
    >>> t = my_data()  # so when you call the function again, it prints again!d
    Pretend this is a long computation
    """

    def func_wrapper(func):
        # TODO: Enforce that the func is argument-less or a bound method here?

        # TODO: WhyTF doesn't this work: (unresolved reference)
        # if key is None:
        #     key = '.'.join([func.__module__, func.__qualname___])

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if key in store:  # if the store has that key...
                return store[key]  # ... just return the data cached under this key
            else:
                output = func(*args, **kwargs)
                store[key] = output
                return output

        wrapped_func._cache = store
        wrapped_func.empty_cache_entry = lambda: wrapped_func._cache.__delitem__(key)
        return wrapped_func

    return func_wrapper


def ensure_clear_to_kv_store(store):
    """
    Ensures the store has a working clear method.

    If the store doesn't have a clear method or has the disabled version,
    adds a proper implementation that safely removes all items.

    Args:
        store: A Store class or instance

    Returns:
        The same store with guaranteed clear functionality

    >>> class NoClearing(dict):
    ...     clear = None
    >>> d = NoClearing({'a': 1, 'b': 2})
    >>> d = ensure_clear_to_kv_store(d)
    >>> len(d)
    2
    >>> d.clear()
    >>> len(d)
    0
    """

    def _needs_clear_method(obj):
        """Check if the object needs a clear method added."""
        has_clear = hasattr(obj, "clear")
        if not has_clear:
            return True

        clear_attr = getattr(obj, "clear")
        if clear_attr is None:
            return True

        if (
            hasattr(clear_attr, "__name__")
            and clear_attr.__name__ == "_disabled_clear_method"
        ):
            return True

        return False

    if not _needs_clear_method(store):
        return store

    def _clear_method(self):
        """Remove all items from the store."""
        # Create a separate list to avoid modification during iteration
        keys = list(self.keys())
        for k in keys:
            del self[k]

    # Apply the appropriate clear method
    if isinstance(store, type):
        store.clear = _clear_method
    else:
        store.clear = types.MethodType(_clear_method, store)

    return store


# TODO: Normalize using store_decorator and add control over flush_cache method name
def flush_on_exit(cls):
    new_cls = type(cls.__name__, (cls,), {})

    if not hasattr(new_cls, "__enter__"):

        def __enter__(self):
            return self

        new_cls.__enter__ = __enter__

    if not hasattr(new_cls, "__exit__"):

        def __exit__(self, *args, **kwargs):
            return self.flush_cache()

    else:  # TODO: Untested case where the class already has an __exit__, which we want to call after flush

        @wraps(new_cls.__exit__)
        def __exit__(self, *args, **kwargs):
            self.flush_cache()
            return super(new_cls, self).__exit__(*args, **kwargs)

    new_cls.__exit__ = __exit__

    return new_cls


from dol.util import has_enabled_clear_method


@store_decorator
def mk_write_cached_store(store=None, *, w_cache=dict, flush_cache_condition=None):
    """Wrap a write cache around a store.

    Args:
        w_cache: The store to (write) cache to
        flush_cache_condition: The condition to apply to the cache
            to decide whether it's contents should be flushed or not

    A ``w_cache`` must have a clear method (that clears the cache's contents).
    If you know what you're doing and want to add one to your input kv store,
    you can do so by calling ``ensure_clear_to_kv_store(store)``
    -- this will add a ``clear`` method inplace AND return the resulting store as well.

    We didn't add this automatically because the first thing ``mk_write_cached_store`` will do is call clear,
    to remove all the contents of the store.
    You don't want to do this unwittingly and delete a bunch of precious data!!

    >>> from dol.caching import mk_write_cached_store, ensure_clear_to_kv_store
    >>> from dol.base import Store
    >>>
    >>> def print_state(store):
    ...     print(f"store: {store} ----- store._w_cache: {store._w_cache}")
    ...
    >>> class MyStore(dict): ...
    >>> MyCachedStore = mk_write_cached_store(MyStore, w_cache={})  # wrap MyStore with a (dict) write cache
    >>> s = MyCachedStore()  # make a MyCachedStore instance
    >>> print_state(s)  # print the contents (both store and cache), see that it's empty
    store: {} ----- store._w_cache: {}
    >>> s['hello'] = 'world'  # write 'world' in 'hello'
    >>> print_state(s)  # see that it hasn't been written
    store: {} ----- store._w_cache: {'hello': 'world'}
    >>> s['ding'] = 'dong'
    >>> print_state(s)
    store: {} ----- store._w_cache: {'hello': 'world', 'ding': 'dong'}
    >>> s.flush_cache()  # manually flush the cache
    >>> print_state(s)  # note that store._w_cache is empty, but store has the data now
    store: {'hello': 'world', 'ding': 'dong'} ----- store._w_cache: {}
    >>>
    >>> # But you usually want to use the store as a context manager
    >>> MyCachedStore = mk_write_cached_store(
    ...     MyStore, w_cache={},
    ...     flush_cache_condition=None)
    >>>
    >>> the_persistent_dict = dict()
    >>>
    >>> s = MyCachedStore(the_persistent_dict)
    >>> with s:
    ...     print("===> Before writing data:")
    ...     print_state(s)
    ...     s['hello'] = 'world'
    ...     print("===> Before exiting the with block:")
    ...     print_state(s)
    ...
    ===> Before writing data:
    store: {} ----- store._w_cache: {}
    ===> Before exiting the with block:
    store: {} ----- store._w_cache: {'hello': 'world'}
    >>>
    >>> print("===> After exiting the with block:"); print_state(s)  # Note that the cache store flushed!
    ===> After exiting the with block:
    store: {'hello': 'world'} ----- store._w_cache: {}
    >>>
    >>> # Example of auto-flushing when there's at least two elements
    >>> class MyStore(dict): ...
    ...
    >>> MyCachedStore = mk_write_cached_store(
    ...     MyStore, w_cache={},
    ...     flush_cache_condition=lambda w_cache: len(w_cache) >= 3)
    >>>
    >>> s = MyCachedStore()
    >>> with s:
    ...     for i in range(7):
    ...         s[i] = i * 10
    ...         print_state(s)
    ...
    store: {} ----- store._w_cache: {0: 0}
    store: {} ----- store._w_cache: {0: 0, 1: 10}
    store: {0: 0, 1: 10, 2: 20} ----- store._w_cache: {}
    store: {0: 0, 1: 10, 2: 20} ----- store._w_cache: {3: 30}
    store: {0: 0, 1: 10, 2: 20} ----- store._w_cache: {3: 30, 4: 40}
    store: {0: 0, 1: 10, 2: 20, 3: 30, 4: 40, 5: 50} ----- store._w_cache: {}
    store: {0: 0, 1: 10, 2: 20, 3: 30, 4: 40, 5: 50} ----- store._w_cache: {6: 60}
    >>> # There was still something left in the cache before exiting the with block. But now...
    >>> print_state(s)
    store: {0: 0, 1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60} ----- store._w_cache: {}
    """

    w_cache = _mk_cache_instance(w_cache, ("clear", "__setitem__", "items"))

    if not has_enabled_clear_method(w_cache):
        raise TypeError(
            """w_cache needs to have an enabled clear method to be able to act as a write cache.
        You can wrap w_cache in dol.trans.ensure_clear_method to inject a clear method, 
        but BE WARNED: mk_write_cached_store will immediately delete all contents of `w_cache`!
        So don't give it your filesystem or important DB to delete!
        """
        )
    w_cache.clear()  # assure the cache is empty, by emptying it.

    @flush_on_exit
    class WriteCachedStore(store):
        _w_cache = w_cache
        _flush_cache_condition = staticmethod(flush_cache_condition)

        if flush_cache_condition is None:

            def __setitem__(self, k, v):
                return self._w_cache.__setitem__(k, v)

        else:
            assert callable(flush_cache_condition), (
                "flush_cache_condition must be None or a callable ",
                "taking the (write) cache store as an input and returning"
                "True if and only if the cache should be flushed.",
            )

            def __setitem__(self, k, v):
                r = self._w_cache.__setitem__(k, v)
                if self._flush_cache_condition(self._w_cache):
                    self.flush_cache()
                return r

        if not hasattr(store, "flush"):

            def flush(self, items: Iterable = tuple()):
                for k, v in items:
                    super().__setitem__(k, v)

        def flush_cache(self):
            self.flush(self._w_cache.items())
            return self._w_cache.clear()

    return WriteCachedStore


from collections import ChainMap, deque


# Note: A (big) generalization of this is a set of graphs that determines how to
# operate with multiple (mutuable) mappings: The order in which to search, the stores
# that should be "written back" to according to where the key was found, the stores that
# should be synced with other stores (possibly even when searched), etc.
class WriteBackChainMap(ChainMap):
    """A collections.ChainMap that also 'writes back' when a key is found.

    >>> from dol.caching import WriteBackChainMap
    >>>
    >>> d = WriteBackChainMap({'a': 1, 'b': 2}, {'b': 22, 'c': 33}, {'d': 444})

    In a ``ChainMap``, when you ask for the value for a key, each mapping in the
    sequence is checked for, and the first mapping found that contains it will be
    the one determining the value.

    So here if you look for `b`, though the first mapping will give you the value,
    though the second mapping also contains a `b` with a different value:

    >>> d['b']
    2

    if you ask for `c`, it's the second mapping that will give you the value:

    >>> d['c']
    33

    But unlike with the builtin ``ChainMap``, something else is going to happen here:

    >>> d
    WriteBackChainMap({'a': 1, 'b': 2, 'c': 33}, {'b': 22, 'c': 33}, {'d': 444})

    See that now the first mapping also has the ``('c', 33)`` key-value pair:

    That is what we call "write back".

    When a key is found in a mapping, all previous mappings (which by definition of
    ``ChainMap`` did not have a value for that key) will be revisited and that key-value
    pair will be written in it.

    As in with ``ChainMap``, all writes will be carried out in the first mapping,
    and only the first mapping:

    >>> d['e'] = 5
    >>> d
    WriteBackChainMap({'a': 1, 'b': 2, 'c': 33, 'e': 5}, {'b': 22, 'c': 33}, {'d': 444})

    Example use cases:

    - You're working with a local and a remote source of data. You'd like to list the
    keys available in both, and use the local item if it's available, and if it's not,
    you want it to be sourced from remote, but written in local for quicker access
    next time.

    - You have several sources to look for configuration values: a sequence of
    configuration files/folders to look through (like a unix search path for command
    resolution) and environment variables.
    """

    max_key_search_depth = 1

    def __getitem__(self, key):
        q = deque([])  # a queue to remember the "failed" mappings
        for mapping in self.maps:  # for each mapping
            try:  # try getting  a value for that key
                v = mapping[key]  # Note: can't use 'key in mapping' with defaultdict
                # if that mapping had that key
                for d in q:  # make sure all other previous mappings
                    d[key] = v  # get that value too (* this is the "write back")
                return v  # and then return the value
            except KeyError:  # if you get a key error for that mapping
                q.append(mapping)  # remember that mapping, so you can write back (*)
        # if no such key was found in any of the self.maps...
        return self.__missing__(key)  # ... call __missing__

    def __len__(self):
        return len(
            set().union(*self.maps[: self.max_key_search_depth])
        )  # reuses stored hash values if possible

    def __iter__(self):
        d = {}
        for mapping in reversed(self.maps[: self.max_key_search_depth]):
            d.update(dict.fromkeys(mapping))  # reuses stored hash values if possible
        return iter(d)

    def __contains__(self, key):
        return any(key in m for m in self.maps[: self.max_key_search_depth])


# Experimental #########################################################################################################


def _mk_cache_method_local_path_key(
    method, args, kwargs, ext=".p", path_sep=os.path.sep
):
    """"""
    return (
        method.__module__
        + path_sep
        + method.__qualname__
        + path_sep
        + (
            ",".join(map(str, args))
            + ",".join(f"{k}={v}" for k, v in kwargs.items())
            + ext
        )
    )


class HashableMixin:
    def __hash__(self):
        return id(self)


class HashableDict(HashableMixin, dict):
    """Just a dict, but hashable"""


# NOTE: cache uses (func, args, kwargs). Don't want to make more complex with a bind cast to (func, kwargs) only
def cache_func_outputs(cache=HashableDict):
    cache = get_cache(cache)

    def cache_method_decorator(func):
        @wraps(func)
        def _func(*args, **kwargs):
            k = (func, args, HashableDict(kwargs))
            if k not in cache:
                val = func(*args, **kwargs)
                cache[k] = val  # cache it
                return val
            else:
                return cache[k]

        return _func

    return cache_method_decorator


# from dol import StrTupleDict
#
# def process_fak(module, qualname, args, kwargs):
# #     func, args, kwargs = map(fak_dict.get, ('func', 'args', 'kwargs'))
#     return {
#         'module': module,
#         'qualname': qualname,
#         'args': ",".join(map(str, args)),
#         'kwargs': ",".join(f"{k}={v}" for k, v in kwargs.items())
#     }
#
# t = StrTupleDict(os.path.join("{module}", "{qualname}", "{args},{kwargs}.p"), process_kwargs=process_fak)
#
# t.tuple_to_str((StrTupleDict.__module__, StrTupleDict.__qualname__, (1, 'one'), {'mode': 'lydian'}))
