"""Key strategies: how ``cache_this`` turns an instance and a call into a cache key.

A key strategy (the ``KeyStrategy`` protocol) resolves a key from the method name when
the class is defined (``resolve_at_definition``) or, if it needs the instance, when the
cached attribute is accessed (``resolve_at_runtime``). The ready-made ones:

- ``ExplicitKey``: a fixed key
- ``ApplyToMethodName``: a function of the method name (e.g. add an extension)
- ``InstanceProp``: the value of an attribute of the instance
- ``ApplyToInstance``: a function of the instance
- ``FromMethodArgs``: a function of the method's arguments
- ``CompositeKey``: several strategies combined

``cache_this`` accepts a strategy, or anything ``_resolve_key_for_cached_prop`` /
``_resolve_key_for_cached_method`` can turn into one (a string, a callable, ...).
The strategies are also attributes of ``cache_this`` (``cache_this.ExplicitKey``).

These used to live in ``dol.caching``, which still re-exports them (i2mint/dol#70).

    >>> from dol.cache_keys import ExplicitKey, ApplyToMethodName
    >>> ExplicitKey('data.json').resolve_at_definition('load')
    'data.json'
    >>> ApplyToMethodName(lambda name: name + '.pkl').resolve_at_definition('load')
    'load.pkl'
"""

from typing import Any, Protocol
from collections.abc import Callable


class KeyStrategy(Protocol):
    """Protocol defining how a key strategy should behave."""

    registered_key_strategies = set()

    def resolve_at_definition(self, method_name: str) -> Any | None:
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

    The function receives ``(self, *args, **kwargs)`` and should return a cache key.
    """

    def __init__(self, func: Callable):
        """
        Initialize with a function to apply to method arguments.

        Args:
            func: A function that takes ``(self, *args, **kwargs)`` and returns a key,
                for example ``lambda self, x, y: f'{x}_{y}'``.
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

    def resolve_at_definition(self, method_name: str) -> Any | None:
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
