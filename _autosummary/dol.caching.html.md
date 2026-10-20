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

- `cache_this`: The main decorator for caching properties and methods.
  Automatically detects whether to use property or method caching based on
  function signature. Supports custom cache storage, key functions, parameter
  ignoring, and serialization hooks.
- `CachedProperty`: A descriptor for caching property values with flexible
  cache storage and key generation strategies.
- `CachedMethod`: A descriptor for caching method results based on arguments,
  with support for parameter filtering and custom key functions.
- `KeyStrategy` protocol: Extensible system for defining how cache keys are
  generated, including strategies for explicit keys, instance properties,
  method arguments, and composite keys.
- Store decorators: Tools like `cache_vals`, `mk_sourced_store`, and
  `store_cached` for adding caching layers to data stores.

The key strategies are defined in `dol.cache_keys` and the store caching layers in
`dol.store_caching`; this module re-exports them (i2mint/dol#70).

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

| [`add_extension`](#dol.caching.add_extension)([ext, name])                     | Add an extension to a name.                                                                |
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| [`cache_property_method`](#dol.caching.cache_property_method)([cls, method_name, ...]) | Converts a method of a class into a CachedProperty.                                        |
| [`cache_this`](#dol.caching.cache_this)([func, cache, key, pre_cache, ...]) | Unified caching decorator for properties and methods with persistent storage support.      |
| [`cached_method`](#dol.caching.cached_method)([func, maxsize, typed])          | A decorator to cache the result of a method, ignoring the first argument (usually `self`). |
| [`lru_cache_method`](#dol.caching.lru_cache_method)([func, maxsize, typed])       | A decorator to cache the result of a method, ignoring the first argument (usually `self`). |

### Classes

| [`CachedMethod`](#dol.caching.CachedMethod)(func[, cache, key, ignore, ...])   | Descriptor that caches the result of method calls based on their arguments.   |
|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`CachedProperty`](#dol.caching.CachedProperty)(func[, cache, key, ...])         | Descriptor that caches the result of the first call to a method.              |

### *class* dol.caching.CachedMethod(func, cache=None, key=None, \*, ignore=None, allow_none_keys=False, lock_factory=<class '_thread.RLock'>, pre_cache=False, serialize=None, deserialize=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Descriptor that caches the result of method calls based on their arguments.

Similar to CachedProperty but handles methods with arguments, caching results
based on unique combinations of arguments (excluding self).

### *class* dol.caching.CachedProperty(func, cache=None, key=None, \*, allow_none_keys=False, lock_factory=<class '_thread.RLock'>, pre_cache=False, serialize=None, deserialize=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Descriptor that caches the result of the first call to a method.

It generalizes the builtin functools.cached_property class, enabling the user to
specify a cache object and a key to store the cache value.

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
  * **cls** ([*type*](https://docs.python.org/3/builtins/functions.html#type)) – The class containing the method.
  * **method_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The name of the method to convert to a cached property.
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
  * **func** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], `VT`]) – The function to be decorated (usually left empty).
  * **cache** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[`KT`, `VT`], [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – 

    The cache storage. Can be:
    - A MutableMapping instance (shared across instances)
    - A string naming an instance attribute containing a MutableMapping
    - A callable taking (instance) and returning a MutableMapping
      This enables instance-specific caching, e.g.:
      cache=lambda self: Files(f’/cache/{self.user_id}/’)
  * **key** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[`KT`, [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], `KT`], [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – For properties: the key to store the cache value, can be a callable
    that will be applied to the method name to make a key, or an explicit string.
    For methods: a callable that takes `(self, *args, **kwargs)` and returns a cache key.
  * **pre_cache** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool) | [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)) – Default is False. If True, adds an in-memory cache to the method
    to (also) cache the results in memory. If a MutableMapping is given, it will be
    used as the pre-cache.
    This is useful when you want a persistent cache but also want to speed up
    access to the method in the same session.
  * **as_property** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – If True, force use of CachedProperty. If False, force use of
    CachedMethod. If None (default), auto-detect based on function signature.
  * **ignore** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Parameter name(s) to exclude from cache key computation.
    Can be a string (single parameter) or list of strings (multiple parameters).
    Commonly used to ignore ‘self’ or parameters like ‘verbose’ that don’t
    affect the result.
  * **serialize** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional function to serialize values before caching
    (e.g. `pickle.dumps` for binary file storage).
  * **deserialize** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional function to deserialize cached values
    (e.g. `pickle.loads`).
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

### dol.caching.cached_method(func=None, , maxsize=128, typed=False)

A decorator to cache the result of a method, ignoring the first argument (usually `self`).

This decorator uses `functools.lru_cache` to cache the method result based on the arguments passed
to the method, excluding the first argument (typically `self`). This allows methods of a class to
be cached while ignoring the instance (`self`) in the cache key.

* **Parameters:**
  * **func** – The method to be decorated. If not provided, a partially applied
    decorator will be returned for later application.
  * **maxsize** – The maximum size of the cache.
  * **typed** – If True, cache entries will be different based on argument types,
    such as distinguishing between `1` and `1.0`.
* **Returns:**
  A wrapped function with LRU caching applied, ignoring the first argument (`self`).

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

### dol.caching.lru_cache_method(func=None, , maxsize=128, typed=False)

A decorator to cache the result of a method, ignoring the first argument
(usually `self`).

This decorator uses `functools.lru_cache` to cache the method result based on the arguments passed
to the method, excluding the first argument (typically `self`). This allows methods of a class to
be cached while ignoring the instance (`self`) in the cache key.

* **Parameters:**
  * **func** – The method to be decorated. If not provided, a partially applied
    decorator will be returned for later application.
  * **maxsize** – The maximum size of the cache.
  * **typed** – If True, cache entries will be different based on argument types,
    such as distinguishing between `1` and `1.0`.
* **Returns:**
  A wrapped function with LRU caching applied, ignoring the first argument (`self`).

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
