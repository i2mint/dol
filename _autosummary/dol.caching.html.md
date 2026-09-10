# dol.caching

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

```default
The main decorator for caching properties and methods. Automatically detects
whether to use property or method caching based on function signature.
Supports custom cache storage, key functions, parameter ignoring, and
serialization hooks.
```

CachedProperty:

```default
A descriptor for caching property values with flexible cache storage and
key generation strategies.
```

CachedMethod:

```default
A descriptor for caching method results based on arguments, with support
for parameter filtering and custom key functions.
```

KeyStrategy Protocol:

```default
Extensible system for defining how cache keys are generated, including
strategies for explicit keys, instance properties, method arguments, and
composite keys.
```

Store Decorators:

```default
Tools like cache_vals, mk_sourced_store, and store_cached for adding
caching layers to data stores.
```

### Examples

Basic property caching:

```pycon
>>> class MyClass:
...     @cache_this
...     def expensive_computation(self):
...         return sum(range(1000000))
```

Method caching with argument-based keys:

```pycon
>>> class Calculator:
...     @cache_this(cache={})
...     def multiply(self, x, y):
...         return x * y
```

Custom cache storage and key functions:

```pycon
>>> class DataProcessor:
...     def __init__(self):
...         self.cache = {}
...     @cache_this(cache='cache', ignore={'verbose'})
...     def process(self, data, mode='fast', verbose=False):
...         return len(data) if mode == 'fast' else sum(data)
```

### Functions

| [`add_extension`](#dol.caching.add_extension)([ext, name])                          | Add an extension to a name.                                                                                                                                 |
|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `cache_func_outputs`([cache])                                                                        |                                                                                                                                                             |
| [`cache_property_method`](#dol.caching.cache_property_method)([cls, method_name, ...])      | Converts a method of a class into a CachedProperty.                                                                                                         |
| [`cache_this`](#dol.caching.cache_this)([func, cache, key, pre_cache, ...])      | Unified caching decorator for properties and methods with persistent storage support.                                                                       |
| [`cache_vals`](#dol.caching.cache_vals)([store, cache, \_\_module_\_, ...])      |                                                                                                                                                             |
| [`cached_method`](#dol.caching.cached_method)([func, maxsize, typed])               | A decorator to cache the result of a method, ignoring the first argument (usually `self`).                                                                  |
| [`ensure_clear_to_kv_store`](#dol.caching.ensure_clear_to_kv_store)(store)                     | Ensures the store has a working clear method.                                                                                                               |
| `flush_on_exit`(cls)                                                                                 |                                                                                                                                                             |
| [`get_cache`](#dol.caching.get_cache)(cache)                                    | Convenience function to get a cache (whether it's already an instance, or needs to be validated).                                                           |
| [`identity`](#dol.caching.identity)(x)                                         | Identity function that returns its input unchanged.                                                                                                         |
| [`is_a_cache`](#dol.caching.is_a_cache)(obj)                                     | Check if an object implements the cache interface.                                                                                                          |
| [`lru_cache_method`](#dol.caching.lru_cache_method)([func, maxsize, typed])            | A decorator to cache the result of a method, ignoring the first argument (usually `self`).                                                                  |
| [`mk_cached_store`](#dol.caching.mk_cached_store)([store, cache, \_\_module_\_, ...]) |                                                                                                                                                             |
| [`mk_memoizer`](#dol.caching.mk_memoizer)(cache)                                  | Make a memoizer that caches the output of a getter function in a cache.                                                                                     |
| [`mk_sourced_store`](#dol.caching.mk_sourced_store)([store, source, ...])              |                                                                                                                                                             |
| [`mk_write_cached_store`](#dol.caching.mk_write_cached_store)([store, w_cache, ...])        | Wrap a write cache around a store.                                                                                                                          |
| [`register_key_strategy`](#dol.caching.register_key_strategy)(cls)                          | Register a class as a KeyStrategy.                                                                                                                          |
| [`store_cached`](#dol.caching.store_cached)(store, key_func)                       | Function output memorizer but using a specific (usually persisting) store as it's memory and a key_func to compute the key under which to store the output. |
| [`store_cached_with_single_key`](#dol.caching.store_cached_with_single_key)(store, key)            | Function output memorizer but using a specific store and key as its memory.                                                                                 |

### Classes

| [`ApplyToInstance`](#dol.caching.ApplyToInstance)(func)                         | Apply a function to the instance to generate the key.                       |
|------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`ApplyToMethodName`](#dol.caching.ApplyToMethodName)(func)                       | Apply a function to the method name to generate the key.                    |
| [`CachedMethod`](#dol.caching.CachedMethod)(func[, cache, key, ignore, ...]) | Descriptor that caches the result of method calls based on their arguments. |
| [`CachedProperty`](#dol.caching.CachedProperty)(func[, cache, key, ...])       | Descriptor that caches the result of the first call to a method.            |
| [`CompositeKey`](#dol.caching.CompositeKey)(\*strategies[, separator])       | Combine multiple key strategies into a single composite key.                |
| [`ExplicitKey`](#dol.caching.ExplicitKey)(key)                              | Use an explicitly provided key value.                                       |
| [`FromMethodArgs`](#dol.caching.FromMethodArgs)(func)                          | Apply a function to method arguments to generate the key.                   |
| [`HashableDict`](#dol.caching.HashableDict)                                  | Just a dict, but hashable                                                   |
| `HashableMixin`()                                                                              |                                                                             |
| [`InstanceProp`](#dol.caching.InstanceProp)(prop_name)                       | Get a key from an instance property.                                        |
| [`KeyStrategy`](#dol.caching.KeyStrategy)(\*args, \*\*kwargs)               | Protocol defining how a key strategy should behave.                         |
| [`WriteBackChainMap`](#dol.caching.WriteBackChainMap)(\*maps)                     | A collections.ChainMap that also 'writes back' when a key is found.         |

### *class* dol.caching.ApplyToInstance(func)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Apply a function to the instance to generate the key.

#### resolve_at_definition(method_name)

Cannot resolve at definition time, need the instance.

* **Return type:**
  [`None`](https://docs.python.org/3/library/constants.html#None)

#### resolve_at_runtime(instance, method_name)

Apply the function to the instance at runtime.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### *class* dol.caching.ApplyToMethodName(func)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Apply a function to the method name to generate the key.

```pycon
>>> strategy = ApplyToMethodName(lambda name: f"{name}.cache")
>>> strategy.resolve_at_definition("my_method")
'my_method.cache'
```

#### resolve_at_definition(method_name)

Apply the function to the method name at definition time.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### *class* dol.caching.CachedMethod(func, cache=None, key=None, \*, ignore=None, allow_none_keys=False, lock_factory=<class '_thread.RLock'>, pre_cache=False, serialize=None, deserialize=None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Descriptor that caches the result of method calls based on their arguments.

Similar to CachedProperty but handles methods with arguments, caching results
based on unique combinations of arguments (excluding self).

### *class* dol.caching.CachedProperty(func, cache=None, key=None, \*, allow_none_keys=False, lock_factory=<class '_thread.RLock'>, pre_cache=False, serialize=None, deserialize=None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Descriptor that caches the result of the first call to a method.

It generalizes the builtin functools.cached_property class, enabling the user to
specify a cache object and a key to store the cache value.

### *class* dol.caching.CompositeKey(\*strategies, separator='_')

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Combine multiple key strategies into a single composite key.

Useful for creating keys that depend on both instance properties and method arguments.

#### resolve_at_definition(method_name)

Try to resolve all strategies at definition time.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any) | [`None`](https://docs.python.org/3/library/constants.html#None)

#### resolve_at_runtime(instance, method_name, \*args, \*\*kwargs)

Resolve all strategies at runtime and combine them.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### *class* dol.caching.ExplicitKey(key)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Use an explicitly provided key value.

```pycon
>>> strategy = ExplicitKey("my_key")
>>> strategy.resolve_at_definition("method_name")
'my_key'
```

#### resolve_at_definition(method_name)

Return the explicit key value at definition time.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### *class* dol.caching.FromMethodArgs(func)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Apply a function to method arguments to generate the key.

The function receives (self, \*args, \*\*kwargs) and should return a cache key.

#### resolve_at_definition(method_name)

Cannot resolve at definition time, need the arguments.

* **Return type:**
  [`None`](https://docs.python.org/3/library/constants.html#None)

#### resolve_at_runtime(instance, method_name, \*args, \*\*kwargs)

Apply the function to the instance and method arguments at runtime.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### *class* dol.caching.HashableDict

Bases: `HashableMixin`, [`dict`](https://docs.python.org/3/library/stdtypes.html#dict)

Just a dict, but hashable

### *class* dol.caching.InstanceProp(prop_name)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Get a key from an instance property.

#### resolve_at_definition(method_name)

Cannot resolve at definition time, need the instance.

* **Return type:**
  [`None`](https://docs.python.org/3/library/constants.html#None)

#### resolve_at_runtime(instance, method_name)

Get the property value from the instance at runtime.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### *class* dol.caching.KeyStrategy(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Protocol defining how a key strategy should behave.

#### resolve_at_definition(method_name)

Attempt to resolve the key at class definition time.

* **Parameters:**
  **method_name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – The name of the method being decorated.
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any) | [`None`](https://docs.python.org/3/library/constants.html#None)
* **Returns:**
  The resolved key or None if it can’t be resolved at definition time.

#### resolve_at_runtime(instance, method_name)

Resolve the key at runtime.
By default, this will call resolve_at_definition on method_name.

* **Parameters:**
  * **instance** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The instance the property is being accessed on.
  * **method_name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – The name of the method being decorated.
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  The resolved key.

### *class* dol.caching.WriteBackChainMap(\*maps)

Bases: [`ChainMap`](https://docs.python.org/3/library/collections.html#collections.ChainMap)

A collections.ChainMap that also ‘writes back’ when a key is found.

```pycon
>>> from dol.caching import WriteBackChainMap
>>>
>>> d = WriteBackChainMap({'a': 1, 'b': 2}, {'b': 22, 'c': 33}, {'d': 444})
```

In a `ChainMap`, when you ask for the value for a key, each mapping in the
sequence is checked for, and the first mapping found that contains it will be
the one determining the value.

So here if you look for `b`, though the first mapping will give you the value,
though the second mapping also contains a `b` with a different value:

```pycon
>>> d['b']
2
```

if you ask for `c`, it’s the second mapping that will give you the value:

```pycon
>>> d['c']
33
```

But unlike with the builtin `ChainMap`, something else is going to happen here:

```pycon
>>> d
WriteBackChainMap({'a': 1, 'b': 2, 'c': 33}, {'b': 22, 'c': 33}, {'d': 444})
```

See that now the first mapping also has the `('c', 33)` key-value pair:

That is what we call “write back”.

When a key is found in a mapping, all previous mappings (which by definition of
`ChainMap` did not have a value for that key) will be revisited and that key-value
pair will be written in it.

As in with `ChainMap`, all writes will be carried out in the first mapping,
and only the first mapping:

```pycon
>>> d['e'] = 5
>>> d
WriteBackChainMap({'a': 1, 'b': 2, 'c': 33, 'e': 5}, {'b': 22, 'c': 33}, {'d': 444})
```

Example use cases:

- You’re working with a local and a remote source of data. You’d like to list the
  keys available in both, and use the local item if it’s available, and if it’s not,
  you want it to be sourced from remote, but written in local for quicker access
  next time.
- You have several sources to look for configuration values: a sequence of
  configuration files/folders to look through (like a unix search path for command
  resolution) and environment variables.

### dol.caching.add_extension(ext=None, name=None)

Add an extension to a name.

If name is None, return a partial function that will add the extension to a
name when called.

add_extension is a useful helper for making key functions, namely for cache_this.

```pycon
>>> add_extension('txt', 'file')
'file.txt'
>>> add_txt_ext = add_extension('txt')
>>> add_txt_ext('file')
'file.txt'
```

#### NOTE
If you want to add an extension to a name that already has an extension,
you can do that, but it will add the extension to the end of the name,
not replace the existing extension.

```pycon
>>> add_txt_ext('file.txt')
'file.txt.txt'
```

Also, bare in mind that if ext starts with the system’s extension separator,
(os.path.extsep), it will be removed.

```pycon
>>> add_extension('.txt', 'file') == add_extension('txt', 'file') == 'file.txt'
True
```

### dol.caching.cache_property_method(cls=None, method_name=None, \*, cache_decorator=<function cache_this>)

Converts a method of a class into a CachedProperty.

Essentially, it does what `A.method = cache_this(A.method)` would do, taking care of
the `__set_name__` problem that you’d run into doing it that way.
Note that here, you need to say `cache_property_method(A, 'method')`.

* **Parameters:**
  * **cls** ([*type*](https://docs.python.org/3/library/functions.html#type)) – The class containing the method.
  * **method_name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – The name of the method to convert to a cached property.
  * **cache_decorator** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – The decorator to use to cache the method. Defaults to
    `cache_this`. One frequent use case would be to use `functools.partial` to
    fix the cache and key parameters of `cache_this` and inject that.

### Example

```pycon
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
```

You can also use it like this:

```pycon
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
... )
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
```

### dol.caching.cache_this(func=None, , cache=None, key=None, pre_cache=False, as_property=None, ignore=None, serialize=None, deserialize=None)

Unified caching decorator for properties and methods with persistent storage support.

`cache_this` extends the capabilities of Python’s built-in `functools.cached_property`
and `functools.lru_cache` by providing:

- **Persistent caching**: Store cached values in files, databases, or any MutableMapping
- **Flexible cache backends**: Use instance attributes, external stores, or cache factories
- **Smart key generation**: Automatic argument-based keys for methods with parameter filtering
- **Serialization support**: Custom serialize/deserialize functions for complex data
- **Auto-detection**: Automatically chooses property vs method caching based on signature
- **No LRU eviction**: Unlike lru_cache, values persist until explicitly removed

Unlike functools.cached_property (properties only) and lru_cache (memory-only with eviction),
cache_this provides a unified interface for both use cases with persistent storage options.

* **Parameters:**
  * **func** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)]) – The function to be decorated (usually left empty).
  * **cache** (`Union`[[`str`](https://docs.python.org/3/library/stdtypes.html#str), [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)], [`None`](https://docs.python.org/3/library/constants.html#None)]) – 

    The cache storage. Can be:
    - A MutableMapping instance (shared across instances)
    - A string naming an instance attribute containing a MutableMapping
    - A callable taking (instance) and returning a MutableMapping
      This enables instance-specific caching, e.g.:
      cache=lambda self: Files(f’/cache/{self.user_id}/’)
  * **key** (`Union`[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/library/stdtypes.html#str)], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)], [`None`](https://docs.python.org/3/library/constants.html#None)]) – For properties: the key to store the cache value, can be a callable
    that will be applied to the method name to make a key, or an explicit string.
    For methods: a callable that takes (self, \*args, \*\*kwargs) and returns a cache key.
  * **pre_cache** ([`bool`](https://docs.python.org/3/library/functions.html#bool) | [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)) – Default is False. If True, adds an in-memory cache to the method
    to (also) cache the results in memory. If a MutableMapping is given, it will be
    used as the pre-cache.
    This is useful when you want a persistent cache but also want to speed up
    access to the method in the same session.
  * **as_property** ([`bool`](https://docs.python.org/3/library/functions.html#bool) | [`None`](https://docs.python.org/3/library/constants.html#None)) – If True, force use of CachedProperty. If False, force use of
    CachedMethod. If None (default), auto-detect based on function signature.
  * **ignore** ([`str`](https://docs.python.org/3/library/stdtypes.html#str) | [`list`](https://docs.python.org/3/library/stdtypes.html#list)[[`str`](https://docs.python.org/3/library/stdtypes.html#str)] | [`None`](https://docs.python.org/3/library/constants.html#None)) – Parameter name(s) to exclude from cache key computation.
    Can be a string (single parameter) or list of strings (multiple parameters).
    Commonly used to ignore ‘self’ or parameters like ‘verbose’ that don’t
    affect the result.
  * **serialize** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/library/constants.html#None)) – 

    Optional function to serialize values before caching.

    Example:
    : serialize=pickle.dumps for binary file storage
  * **deserialize** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/library/constants.html#None)) – 

    Optional function to deserialize cached values.

    Example:
    : deserialize=pickle.loads
* **Returns:**
  The decorated function.

### Comprehensive Example

Here’s a complete example showcasing all major features of cache_this:

```pycon
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
```

Now let’s test all the features:

```pycon
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
```

Used with no arguments, `cache_this` will cache just as the builtin
`cached_property` does – in the instance’s `__dict__` attribute.

```pycon
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
```

Not that if you specify `cache=False`, you get a property that is computed
every time it’s accessed:

```pycon
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
```

Specify the cache as a dictionary that lives outside the instance:

```pycon
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
```

Specify the cache as an attribute of the instance, and an explicit key:

```pycon
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
```

Now let’s see a more involved example that exhibits how `cache_this` would be used
in real life. Note two things in the example below.

First, that we use `functools.partial` to fix the parameters of our `cache_this`.
This enables us to reuse the same `cache_this` in multiple places without all
the verbosity. We fix that the cache is the attribute `cache` of the instance,
and that the key is a function that will be computed from the name of the method
adding a `'.pkl'` extension to it.

Secondly, we use the `ValueCodecs` from `dol` to provide a pickle codec for storying
values. The backend store used here is a dictionary, so we don’t really need a
codec to store values, but in real life you would use a persistent storage that
would require a codec, such as files or a database.

Thirdly, we’ll use a `pre_cache` to store the values in a different cache “before”
(setting and getting) them in the main cache.
This is useful, for instance, when you want to persist the values (in the main
cache), but keep them in memory for faster access in the same session
(the pre-cache, a dict() instance usually). It can also be used to store and
use things locally (pre-cache) while sharing them with others by storing them in
a remote store (main cache).

Finally, we’ll use a dict that logs any setting and getting of values to show
how the caches are being used.

```pycon
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
```

```pycon
>>> obj = PickleCached()
>>> list(obj.cache)
[]
```

```pycon
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
```

As usual, it’s because the cache now holds something that has to do with `foo`:

```pycon
>>> list(obj.cache)
['foo.pkl']
>>> # == ['foo.pkl']
```

The value of `'foo.pkl'` is indeed `42`:

```pycon
>>> obj.cache['foo.pkl']
In CacheA: getting value of foo.pkl
42
```

But note that the actual way it’s stored in the `_backend_store` is as pickle bytes:

```pycon
>>> obj._backend_store['foo.pkl']
In CacheA: getting value of foo.pkl
b'\x80\x04K*.'
>>> # == b'\x80\x04K*.'
```

### dol.caching.cache_vals(store=None, \*, cache=<class 'dict'>, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

* **Parameters:**
  * **store** – The class of the store you want to cache
  * **cache** – The store you want to use to cache. Anything with a \_\_setitem_\_(k, v) and a \_\_getitem_\_(k).
    By default, it will use a dict
* **Returns:**
  A subclass of the input store, but with caching (to the cache store)

```pycon
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
>>> print(f"store: {list(s)}\ncache: {list(cache)}")
store: ['a', 'b', 'c']
cache: []
>>> # This will take a LONG time because it's the first time we ask for 'a'
>>> v = s['a']
>>> print(f"store: {list(s)}\ncache: {list(cache)}")
store: ['a', 'b', 'c']
cache: ['a']
>>> # This will take very little time because we have 'a' in the cache
>>> v = s['a']
>>> print(f"store: {list(s)}\ncache: {list(cache)}")
store: ['a', 'b', 'c']
cache: ['a']
>>> # But we don't have 'b'
>>> v = s['b']
>>> print(f"store: {list(s)}\ncache: {list(cache)}")
store: ['a', 'b', 'c']
cache: ['a', 'b']
>>> # But now we have 'b'
>>> v = s['b']
>>> print(f"store: {list(s)}\ncache: {list(cache)}")
store: ['a', 'b', 'c']
cache: ['a', 'b']
>>> s['d'] = 4  # and we can do things normally (like put stuff in the store)
>>> print(f"store: {list(s)}\ncache: {list(cache)}")
store: ['a', 'b', 'c', 'd']
cache: ['a', 'b']
>>> s['d']  # if we ask for it again though, it will take time (the first time)
4
>>> print(f"store: {list(s)}\ncache: {list(cache)}")
store: ['a', 'b', 'c', 'd']
cache: ['a', 'b', 'd']
>>> # Of course, we could write 'd' in the cache as well, to get it quicker,
>>> # but that's another story: The story of write caches!
>>>
>>> # And by the way, your "cache wrapped" store hold a pointer to the cache it's using,
>>> # so you can take a peep there if needed:
>>> s._cache
{'a': 1, 'b': 2, 'd': 4}
```

### dol.caching.cached_method(func=None, , maxsize=128, typed=False)

A decorator to cache the result of a method, ignoring the first argument (usually `self`).

This decorator uses `functools.lru_cache` to cache the method result based on the arguments passed
to the method, excluding the first argument (typically `self`). This allows methods of a class to
be cached while ignoring the instance (`self`) in the cache key.

### Parameters

- func (callable, optional): The method to be decorated. If not provided, a partially applied decorator
  will be returned for later application.
- maxsize (int, optional): The maximum size of the cache. Defaults to 128.
- typed (bool, optional): If True, cache entries will be different based on argument types, such as
  distinguishing between `1` and `1.0`. Defaults to False.

### Returns

- callable: A wrapped function with LRU caching applied, ignoring the first argument (`self`).

### Example

```pycon
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
```

### dol.caching.ensure_clear_to_kv_store(store)

Ensures the store has a working clear method.

If the store doesn’t have a clear method or has the disabled version,
adds a proper implementation that safely removes all items.

* **Parameters:**
  **store** – A Store class or instance
* **Returns:**
  The same store with guaranteed clear functionality

```pycon
>>> class NoClearing(dict):
...     clear = None
>>> d = NoClearing({'a': 1, 'b': 2})
>>> d = ensure_clear_to_kv_store(d)
>>> len(d)
2
>>> d.clear()
>>> len(d)
0
```

### dol.caching.get_cache(cache)

Convenience function to get a cache (whether it’s already an instance, or needs to be validated).

```pycon
>>> get_cache({'a': 1})  # Return existing cache instance
{'a': 1}
>>> get_cache(dict)()  # Return result of calling cache factory
{}
```

### dol.caching.identity(x)

Identity function that returns its input unchanged.

* **Return type:**
  [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)

```pycon
>>> identity(42)
42
>>> identity("hello")
'hello'
>>> identity([1, 2, 3])
[1, 2, 3]
```

### dol.caching.is_a_cache(obj)

Check if an object implements the cache interface.

A cache object must have \_\_contains_\_, \_\_getitem_\_, and \_\_setitem_\_ methods.

```pycon
>>> is_a_cache({})  # dict is a valid cache
True
>>> is_a_cache([])  # list has these methods but for indexed access
True
>>> is_a_cache("string")  # string is not (immutable)
False
```

### dol.caching.lru_cache_method(func=None, , maxsize=128, typed=False)

A decorator to cache the result of a method, ignoring the first argument
(usually `self`).

This decorator uses `functools.lru_cache` to cache the method result based on the arguments passed
to the method, excluding the first argument (typically `self`). This allows methods of a class to
be cached while ignoring the instance (`self`) in the cache key.

### Parameters

- func (callable, optional): The method to be decorated. If not provided, a partially applied decorator
  will be returned for later application.
- maxsize (int, optional): The maximum size of the cache. Defaults to 128.
- typed (bool, optional): If True, cache entries will be different based on argument types, such as
  distinguishing between `1` and `1.0`. Defaults to False.

### Returns

- callable: A wrapped function with LRU caching applied, ignoring the first argument (`self`).

### Example

```pycon
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
```

Like `lru_cache`, you can specify the `maxsize` and `typed` parameters:

```pycon
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
```

### dol.caching.mk_cached_store(store=None, \*, cache=<class 'dict'>, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

* **Parameters:**
  * **store** – The class of the store you want to cache
  * **cache** – The store you want to use to cache. Anything with a \_\_setitem_\_(k, v) and a \_\_getitem_\_(k).
    By default, it will use a dict
* **Returns:**
  A subclass of the input store, but with caching (to the cache store)

```pycon
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
>>> print(f"store: {list(s)}\ncache: {list(cache)}")
store: ['a', 'b', 'c']
cache: []
>>> # This will take a LONG time because it's the first time we ask for 'a'
>>> v = s['a']
>>> print(f"store: {list(s)}\ncache: {list(cache)}")
store: ['a', 'b', 'c']
cache: ['a']
>>> # This will take very little time because we have 'a' in the cache
>>> v = s['a']
>>> print(f"store: {list(s)}\ncache: {list(cache)}")
store: ['a', 'b', 'c']
cache: ['a']
>>> # But we don't have 'b'
>>> v = s['b']
>>> print(f"store: {list(s)}\ncache: {list(cache)}")
store: ['a', 'b', 'c']
cache: ['a', 'b']
>>> # But now we have 'b'
>>> v = s['b']
>>> print(f"store: {list(s)}\ncache: {list(cache)}")
store: ['a', 'b', 'c']
cache: ['a', 'b']
>>> s['d'] = 4  # and we can do things normally (like put stuff in the store)
>>> print(f"store: {list(s)}\ncache: {list(cache)}")
store: ['a', 'b', 'c', 'd']
cache: ['a', 'b']
>>> s['d']  # if we ask for it again though, it will take time (the first time)
4
>>> print(f"store: {list(s)}\ncache: {list(cache)}")
store: ['a', 'b', 'c', 'd']
cache: ['a', 'b', 'd']
>>> # Of course, we could write 'd' in the cache as well, to get it quicker,
>>> # but that's another story: The story of write caches!
>>>
>>> # And by the way, your "cache wrapped" store hold a pointer to the cache it's using,
>>> # so you can take a peep there if needed:
>>> s._cache
{'a': 1, 'b': 2, 'd': 4}
```

### dol.caching.mk_memoizer(cache)

Make a memoizer that caches the output of a getter function in a cache.

#### NOTE
This is a specialized memoizer for getter functions/methods, i.e.
functions/methods that have the signature (instance, key) and return a value.

* **Parameters:**
  **cache** – The cache to use. Must have \_\_getitem_\_ and \_\_setitem_\_ methods.
* **Returns:**
  A memoizer that caches the output of the function in the cache.

```pycon
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
```

### dol.caching.mk_sourced_store(store=None, , source=None, return_source_data=True, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

* **Parameters:**
  * **store** – The class of the store you want to cache
  * **cache** – The store you want to use to cache. Anything with a \_\_setitem_\_(k, v) and a \_\_getitem_\_(k).
    By default, it will use a dict
  * **return_source_data**
  * **store** – The class of the store you’re talking to. This store acts as the cache
  * **source** – The store that is used to populate the store (cache) when a key is missing there.
  * **return_source_data** – If True, will return `source[k]` as is. This should be used only if `store[k]` would return the same.
    If False, will first write to cache (`store[k] = source[k]`) then return `store[k]`.
    The latter introduces a performance hit (we write and then read again from the cache),
    but ensures consistency (and is useful if the writing or the reading to/from store
    transforms the data in some way.
* **Returns:**
  A subclass of the input store, but with caching (to the cache store)
* **Returns:**
  A decorated store

Here are two stores pretending to be local and remote data stores respectively.

```pycon
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
```

Let’s make a remote store with two elements in it, and a local store class that asks the remote store for stuff
if it can’t find it locally.

```pycon
>>> remote = Remote({'foo': 'bar', 'hello': 'world'})
>>> SourcedLocal = mk_sourced_store(Local, source=remote)
>>> s = SourcedLocal({'some': 'local stuff'})
>>> list(s)  # the local store has one key
['some']
```

### but if we ask for a key that is in the remote store, it provides it

```pycon
>>> assert s['foo'] == 'bar'
looking for foo in Local
looking for foo in Remote
```

```pycon
>>> list(s)
['some', 'foo']
```

See that next time we ask for the ‘foo’ key, the local store provides it:

```pycon
>>> assert s['foo'] == 'bar'
looking for foo in Local
```

```pycon
>>> assert s['hello'] == 'world'
looking for hello in Local
looking for hello in Remote
>>> list(s)
['some', 'foo', 'hello']
```

We can still add stuff (locally)…

```pycon
>>> s['something'] = 'else'
>>> list(s)
['some', 'foo', 'hello', 'something']
```

### dol.caching.mk_write_cached_store(store=None, \*, w_cache=<class 'dict'>, flush_cache_condition=None, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Wrap a write cache around a store.

* **Parameters:**
  * **w_cache** – The store to (write) cache to
  * **flush_cache_condition** – The condition to apply to the cache
    to decide whether it’s contents should be flushed or not

A `w_cache` must have a clear method (that clears the cache’s contents).
If you know what you’re doing and want to add one to your input kv store,
you can do so by calling `ensure_clear_to_kv_store(store)`
– this will add a `clear` method inplace AND return the resulting store as well.

We didn’t add this automatically because the first thing `mk_write_cached_store` will do is call clear,
to remove all the contents of the store.
You don’t want to do this unwittingly and delete a bunch of precious data!!

```pycon
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
```

### dol.caching.register_key_strategy(cls)

Register a class as a KeyStrategy.

### dol.caching.store_cached(store, key_func)

Function output memorizer but using a specific (usually persisting) store as it’s
memory and a key_func to compute the key under which to store the output.

The key can be

- a single value under which the output should be stored, regardless of the input.
- a key function that is called on the inputs to create a hash under which the function’s output should be stored.

* **Parameters:**
  * **store** – The key-value store to use for caching. Must support \_\_getitem_\_ and \_\_setitem_\_.
  * **key_func** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – The key function that is called on the input of the function to create the key value.

#### NOTE
Union[Callable, Any] is equivalent to just Any, but reveals the two cases of a key more clearly.

#### NOTE
No, Union[Callable, Hashable] is not better. For one, general store keys are not restricted to hashable keys.

#### NOTE
No, they shouldn’t.

#### SEE ALSO
store_cached_with_single_key (for a version where the cache store key doesn’t depend on function’s args)

```pycon
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
```

### dol.caching.store_cached_with_single_key(store, key)

Function output memorizer but using a specific store and key as its memory.

Use in situations where you have a argument-less function or bound method that computes some data whose dependencies
are static enough that there’s enough advantage to make the data refresh explicit (by deleting the cache entry)
instead of making it implicit (recomputing/refetching the data every time).

The key should be a single value under which the output should be stored, regardless of the input.

#### NOTE
The wrapped function comes with a empty_cache attribute, which when called, empties the cache (i.e. removes
the key from the store)

#### NOTE
The wrapped function has a hidden `_cache` attribute pointing to the store in case you need to peep into it.

* **Parameters:**
  * **store** – The cache. The key-value store to use for caching. Must support \_\_getitem_\_ and \_\_setitem_\_.
  * **key** – The store key under which to store the output of the function.

#### NOTE
Union[Callable, Any] is equivalent to just Any, but reveals the two cases of a key more clearly.

#### NOTE
No, Union[Callable, Hashable] is not better. For one, general store keys are not restricted to hashable keys.

#### NOTE
No, they shouldn’t.

#### SEE ALSO
store_cached (for a version whose keys are computed from the wrapped function’s input.

```pycon
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
```
