# dol.store_caching

Caching layers for stores: read caches, write caches and memoizers.

- `cache_vals`: cache the values read from a store (in a dict or any other store)
- `mk_sourced_store`: a store that fetches missing keys from a source store
- `store_cached` / `store_cached_with_single_key`: cache a function’s outputs in a store
- `mk_write_cached_store` / `WriteBackChainMap`: buffer writes and flush them later
- `ensure_clear_to_kv_store` / `flush_on_exit`: helpers for those write caches
- `cache_func_outputs`, `mk_memoizer`, `is_a_cache`, `get_cache`: smaller tools

These used to live in `dol.caching`, which still re-exports them (i2mint/dol#70).

```pycon
>>> from dol.store_caching import cache_vals
>>> source = {'a': 1}
>>> s = cache_vals(source)
>>> s['a']
1
>>> s._cache
{'a': 1}
```

### Functions

| [`cache_func_outputs`](#dol.store_caching.cache_func_outputs)([cache])                         | Decorator factory that caches a function's outputs in `cache`, keyed by `(func, args, kwargs)`.                                                                                                         |
|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`cache_vals`](#dol.store_caching.cache_vals)([store, cache, \_\_module_\_, ...])      |                                                                                                                                                                                                         |
| [`ensure_clear_to_kv_store`](#dol.store_caching.ensure_clear_to_kv_store)(store)                     | Ensures the store has a working clear method.                                                                                                                                                           |
| [`flush_on_exit`](#dol.store_caching.flush_on_exit)(cls)                                  | Class decorator: a subclass whose `__exit__` calls `flush_cache()` (adding a trivial `__enter__` if the class has none), so a write-cached store can be used as a context manager that flushes on exit. |
| [`get_cache`](#dol.store_caching.get_cache)(cache)                                    | Convenience function to get a cache (whether it's already an instance, or needs to be validated).                                                                                                       |
| [`is_a_cache`](#dol.store_caching.is_a_cache)(obj)                                     | Check if an object implements the cache interface.                                                                                                                                                      |
| [`mk_cached_store`](#dol.store_caching.mk_cached_store)([store, cache, \_\_module_\_, ...]) |                                                                                                                                                                                                         |
| [`mk_memoizer`](#dol.store_caching.mk_memoizer)(cache)                                  | Make a memoizer that caches the output of a getter function in a cache.                                                                                                                                 |
| [`mk_sourced_store`](#dol.store_caching.mk_sourced_store)([store, source, ...])              |                                                                                                                                                                                                         |
| [`mk_write_cached_store`](#dol.store_caching.mk_write_cached_store)([store, w_cache, ...])        | Wrap a write cache around a store.                                                                                                                                                                      |
| [`store_cached`](#dol.store_caching.store_cached)(store, key_func)                       | Function output memorizer but using a specific (usually persisting) store as it's memory and a key_func to compute the key under which to store the output.                                             |
| [`store_cached_with_single_key`](#dol.store_caching.store_cached_with_single_key)(store, key)            | Function output memorizer but using a specific store and key as its memory.                                                                                                                             |

### Classes

| [`HashableDict`](#dol.store_caching.HashableDict)              | Just a dict, but hashable                                           |
|----------------------------------------------------------------------------|---------------------------------------------------------------------|
| [`WriteBackChainMap`](#dol.store_caching.WriteBackChainMap)(\*maps) | A collections.ChainMap that also 'writes back' when a key is found. |

### *class* dol.store_caching.HashableDict

Bases: [`HashableMixin`](dol.util.html.md#dol.util.HashableMixin), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

Just a dict, but hashable

### *class* dol.store_caching.WriteBackChainMap(\*maps)

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

### dol.store_caching.cache_func_outputs(cache=<class 'dol.store_caching.HashableDict'>)

Decorator factory that caches a function’s outputs in `cache`, keyed by
`(func, args, kwargs)`.

* **Parameters:**
  **cache** – A cache instance (with `__contains__`/`__getitem__`/
  `__setitem__`), or a zero-arg factory/class (e.g. the default,
  `HashableDict`) used to make one.

#### NOTE
like `args`, every `kwargs` value must be hashable (they’re part of
the cache key). Passing an unhashable value (e.g. a `list`) raises
`TypeError` rather than silently skipping the cache.

```pycon
>>> @cache_func_outputs()
... def f(x, y=2):
...     print(f"computing f({x}, {y})")
...     return x + y
>>> f(1)
computing f(1, 2)
3
>>> f(1)  # cached: no "computing" print
3
>>> f(1, y=3)  # different kwargs: not a cache hit
computing f(1, 3)
4
>>> f(1, y=3)  # this one is now cached too
4
```

### dol.store_caching.cache_vals(store=None, \*, cache=<class 'dict'>, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

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

### dol.store_caching.ensure_clear_to_kv_store(store)

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

### dol.store_caching.flush_on_exit(cls)

Class decorator: a subclass whose `__exit__` calls `flush_cache()` (adding a
trivial `__enter__` if the class has none), so a write-cached store can be used as
a context manager that flushes on exit. Used by `mk_write_cached_store`.

### dol.store_caching.get_cache(cache)

Convenience function to get a cache (whether it’s already an instance, or needs to be validated).

```pycon
>>> get_cache({'a': 1})  # Return existing cache instance
{'a': 1}
>>> get_cache(dict)()  # Return result of calling cache factory
{}
```

### dol.store_caching.is_a_cache(obj)

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

### dol.store_caching.mk_cached_store(store=None, \*, cache=<class 'dict'>, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

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

### dol.store_caching.mk_memoizer(cache)

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

### dol.store_caching.mk_sourced_store(store=None, , source=None, return_source_data=True, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

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

But if we ask for a key that is in the remote store, it provides it:

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

### dol.store_caching.mk_write_cached_store(store=None, \*, w_cache=<class 'dict'>, flush_cache_condition=None, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

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

### dol.store_caching.store_cached(store, key_func)

Function output memorizer but using a specific (usually persisting) store as it’s
memory and a key_func to compute the key under which to store the output.

The key can be

- a single value under which the output should be stored, regardless of the input.
- a key function that is called on the inputs to create a hash under which the function’s output should be stored.

* **Parameters:**
  * **store** – The key-value store to use for caching. Must support \_\_getitem_\_ and \_\_setitem_\_.
  * **key_func** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – The key function that is called on the input of the function to create the key value.

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

### dol.store_caching.store_cached_with_single_key(store, key)

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
