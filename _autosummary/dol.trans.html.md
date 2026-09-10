# dol.trans

Transformation/wrapping tools

### Module Attributes

| [`confirm_overwrite`](#dol.trans.confirm_overwrite)(self, k, v)   | A ready-to-use `wrap_kvs` `preset` that asks (via the builtin `input`) to confirm before overwriting an existing key with a different value (Issue #13).   |
|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|

### Functions

| [`add_aliases`](#dol.trans.add_aliases)(obj, \*\*aliases)                      | A function that wraps the object instance and adds aliases.                                                                                              |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`add_decoder`](#dol.trans.add_decoder)([store_cls, decoder, name, ...])       | Add a decoder layer to a store.                                                                                                                          |
| [`add_ipython_key_completions`](#dol.trans.add_ipython_key_completions)(store)                 | Add tab completion that shows you the keys of the store.                                                                                                 |
| [`add_missing_key_handling`](#dol.trans.add_missing_key_handling)([store, ...])             | Overrides the `__missing__` method of a store with a custom callback.                                                                                    |
| [`add_path_access`](#dol.trans.add_path_access)([store, name, path_type, ...])     | Make nested stores (read/write) accessible through key paths (iterable of keys).                                                                         |
| [`add_path_get`](#dol.trans.add_path_get)([store, name, path_type, ...])        | Make nested stores accessible through key paths.                                                                                                         |
| [`add_store_method`](#dol.trans.add_store_method)(store, \*, method_func[, ...])    | Add methods to store classes or instances                                                                                                                |
| [`add_wrapper_method`](#dol.trans.add_wrapper_method)([wrap_cls, method_name])        | Decorator that adds a wrapper method (itself a decorator) to a wrapping class Clear? See `mk_wrapper` function and doctest example if not.               |
| [`affix_key_codec`](#dol.trans.affix_key_codec)([prefix, suffix])                  | A factory that creates a key codec that affixes a prefix and suffix to the key                                                                           |
| [`assert_min_num_of_args`](#dol.trans.assert_min_num_of_args)(func, num_of_args)          | Assert that a function can be a store method.                                                                                                            |
| [`autoviv`](#dol.trans.autoviv)([store])                                   | Opt-in write-through autovivification for key-paths.                                                                                                     |
| [`cache_iter`](#dol.trans.cache_iter)([store, keys_cache, ...])               | Make a class that wraps input class's \_\_iter_\_ becomes cached.                                                                                        |
| [`cached_keys`](#dol.trans.cached_keys)([store, keys_cache, ...])              | Make a class that wraps input class's \_\_iter_\_ becomes cached.                                                                                        |
| [`catch_and_cache_error_keys`](#dol.trans.catch_and_cache_error_keys)([store, ...])           | Store that will cache keys as they're accessed, separating those that raised errors and those that didn't.                                               |
| `condition_function_call`([func, condition, ...])                                                   |                                                                                                                                                          |
| `conditional_data_trans`([store, \_\_module_\_, ...])                                               |                                                                                                                                                          |
| [`confirm_overwrite`](#dol.trans.confirm_overwrite)(self, k, v)                      | A ready-to-use `wrap_kvs` `preset` that asks (via the builtin `input`) to confirm before overwriting an existing key with a different value (Issue #13). |
| [`constant_output`](#dol.trans.constant_output)([return_val])                      | Function that returns a constant value no matter what the inputs are.                                                                                    |
| `disable_delitem`(o)                                                                                |                                                                                                                                                          |
| `disable_setitem`(o)                                                                                |                                                                                                                                                          |
| `disallow_overwrites`(store, \*[, error_msg, ...])                                                  |                                                                                                                                                          |
| [`double_up_as_factory`](#dol.trans.double_up_as_factory)(decorator_func)               | Repurpose a decorator both as it's original form, and as a decorator factory.                                                                            |
| [`ensure_clear_method`](#dol.trans.ensure_clear_method)([store, clear_method])         | If obj doesn't have an enabled clear method, will add one (a slow one that runs through keys and deletes them                                            |
| `ensure_set`(x)                                                                                     |                                                                                                                                                          |
| [`filt_iter`](#dol.trans.filt_iter)([store, filt, name, \_\_module_\_, ...]) | Make a wrapper that will transform a store (class or instance thereof) into a sub-store (i.e. subset of keys).                                           |
| [`filter_prefixes`](#dol.trans.filter_prefixes)(prefixes)                          | Make a filter that returns True if a string starts with one of the given prefixes                                                                        |
| [`filter_regex`](#dol.trans.filter_regex)(regex, \*[, return_search_func])      | Make a filter that returns True if a string matches the given regex                                                                                      |
| [`filter_suffixes`](#dol.trans.filter_suffixes)(suffixes)                          | Make a filter that returns True if a string ends with one of the given suffixes                                                                          |
| [`flatten`](#dol.trans.flatten)([store, levels, cache_keys, ...])          | Flatten a nested store.                                                                                                                                  |
| `get_class_name`(cls[, dflt_name])                                                                  |                                                                                                                                                          |
| `ignore_if_error`([store, errors])                                                                  |                                                                                                                                                          |
| [`insert_aliases`](#dol.trans.insert_aliases)([store, write, read, delete, ...])  | Insert method aliases of CRUD operations of a store (class or instance).                                                                                 |
| [`insert_hash_method`](#dol.trans.insert_hash_method)([store, hash_method, ...])      | Make a store hashable using the specified `hash_method`.                                                                                                 |
| [`insert_load_dump_aliases`](#dol.trans.insert_load_dump_aliases)([store, delete, ...])     | Insert load and dump methods, with familiar dump(obj, location) signature.                                                                               |
| `is_iterable`(x)                                                                                    |                                                                                                                                                          |
| `iterate_values_and_accumulate_non_error_keys`(...)                                                 |                                                                                                                                                          |
| [`kv_wrap`](#dol.trans.kv_wrap)(trans_obj)                                 | A function that makes a wrapper (a decorator) that will get the wrappers from methods of the input object.                                               |
| [`kv_wrap_persister_cls`](#dol.trans.kv_wrap_persister_cls)(persister_cls[, name])       | Make a class that wraps a persister into a dol.base.Store,                                                                                               |
| `leveled_paths_walk`(m, levels)                                                                     |                                                                                                                                                          |
| [`mk_confirm_overwrite_preset`](#dol.trans.mk_confirm_overwrite_preset)(\*[, get_input, ...])  | Make a `wrap_kvs` `preset` that asks for confirmation before overwriting an existing key that holds a *different* value.                                 |
| [`mk_kv_reader_from_kv_collection`](#dol.trans.mk_kv_reader_from_kv_collection)(kv_collection)     | Make a KvReader class from a Collection class.                                                                                                           |
| [`mk_level_walk_filt`](#dol.trans.mk_level_walk_filt)(levels)                         | Makes a `walk_filt` function for `kv_walk` based on some level logic.                                                                                    |
| `mk_read_only`(o)                                                                                   |                                                                                                                                                          |
| [`mk_trans_obj`](#dol.trans.mk_trans_obj)(\*\*kwargs)                           | Convenience method to quickly make a trans_obj (just an object holding some trans functions                                                              |
| [`mk_wrapper`](#dol.trans.mk_wrapper)(wrap_cls)                               | You have a wrapper class and you want to make a wrapper out of it, that is, a decorator factory with which you can make wrappers, like this:             |
| `raise_disabled_error`(functionality)                                                               |                                                                                                                                                          |
| [`redirect_getattr_to_getitem`](#dol.trans.redirect_getattr_to_getitem)([cls, ...])            | A mapping decorator that redirects attribute access to \_\_getitem_\_.                                                                                   |
| `return_default_if_error`([store, default, errors])                                                 |                                                                                                                                                          |
| [`store_decorator`](#dol.trans.store_decorator)(func)                              | Helper to make store decorators.                                                                                                                         |
| `store_wrap`(obj)                                                                                   |                                                                                                                                                          |
| `take_everything`(key)                                                                              |                                                                                                                                                          |
| `transparent_key_method`(self, k)                                                                   |                                                                                                                                                          |
| `warn_and_ignore_if_error`([store, errors, ...])                                                    |                                                                                                                                                          |
| [`wrap_kvs`](#dol.trans.wrap_kvs)([store, wrapper, name, key_of_id, ...])   | Make a Store that is wrapped with the given key/val transformers.                                                                                        |

### Classes

| [`CachedInvertibleTrans`](#dol.trans.CachedInvertibleTrans)(trans_func)   |                                                                              |
|--------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| [`Codec`](#dol.trans.Codec)(encoder, decoder)             |                                                                              |
| `FiltIter`(\*args, \*\*kwargs)                                                       |                                                                              |
| [`FirstArgIsMapping`](#dol.trans.FirstArgIsMapping)(val)              | Mark a transform so its first argument is the store (mapping), not the data. |
| [`KeyCodec`](#dol.trans.KeyCodec)(encoder, decoder)          |                                                                              |
| [`KeyValueCodec`](#dol.trans.KeyValueCodec)(encoder, decoder)     |                                                                              |
| [`OverWritesNotAllowedMixin`](#dol.trans.OverWritesNotAllowedMixin)()         | Mixin for only allowing a write to a key if they key doesn't already exist.  |
| `SimpleDelegator`(obj)                                                               |                                                                              |
| [`ValueCodec`](#dol.trans.ValueCodec)(encoder, decoder)        |                                                                              |

### Exceptions

| [`MapInvertabilityError`](#dol.trans.MapInvertabilityError)   | To be used to indicate that a mapping isn't, or wouldn't be, invertible   |
|--------------------------------------------------------------------------|---------------------------------------------------------------------------|

### *class* dol.trans.CachedInvertibleTrans(trans_func)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

```pycon
>>> t = CachedInvertibleTrans(lambda x: x[1])
>>> t.ingress('ab')
'b'
>>> t.ingress((1, 2))
2
>>> t.egress('b')
'ab'
>>> t.egress(2)
(1, 2)
```

### *class* dol.trans.Codec(encoder, decoder)

Bases: [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`DecodedType`, `EncodedType`]

#### invert()

Return a codec that is the inverse of this one.
That is, encoder and decoder will be swapped.

### *class* dol.trans.FirstArgIsMapping(val)

Bases: [`LiteralVal`](dol.util.html.md#dol.util.LiteralVal)

Mark a transform so its first argument is the store (mapping), not the data.

Use this to explicitly opt a transform function into the `f(self, data)`
calling convention in wrappers such as `wrap_kvs` – instead of relying on the
name/arity heuristic (`_has_unbound_self()`). This is the escape hatch for
functions the heuristic can’t (or shouldn’t) infer, and the recommended, explicit
alternative to naming a transform’s first parameter `self`/`store`/`mapping`.

```pycon
>>> from dol import wrap_kvs, FirstArgIsMapping
>>> def prefix_with_name(self, data):
...     return f"{getattr(self, 'name', '?')}:{data}"
>>> S = wrap_kvs(dict, obj_of_data=FirstArgIsMapping(prefix_with_name))
>>> s = S({'a': 'x'}); s.name = 'ns'
>>> s['a']
'ns:x'
```

### *class* dol.trans.KeyCodec(encoder, decoder)

Bases: [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`DecodedType`, `EncodedType`], [`Codec`](#dol.trans.Codec)[`DecodedType`, `EncodedType`]

### *class* dol.trans.KeyValueCodec(encoder, decoder)

Bases: [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`DecodedType`, `EncodedType`], [`Codec`](#dol.trans.Codec)[`DecodedType`, `EncodedType`]

### *exception* dol.trans.MapInvertabilityError

Bases: [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)

To be used to indicate that a mapping isn’t, or wouldn’t be, invertible

### *class* dol.trans.OverWritesNotAllowedMixin

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Mixin for only allowing a write to a key if they key doesn’t already exist.

#### NOTE
Should be before the persister in the MRO.

```pycon
>>> class TestPersister(OverWritesNotAllowedMixin, dict):
...     pass
>>> p = TestPersister()
>>> p['foo'] = 'bar'
>>> #p['foo'] = 'bar2'  # will raise error
>>> p['foo'] = 'this value should not be stored'
Traceback (most recent call last):
  ...
dol.errors.OverWritesNotAllowedError: key foo already exists and cannot be overwritten.
    If you really want to write to that key, delete it before writing
>>> p['foo']  # foo is still bar
'bar'
>>> del p['foo']
>>> p['foo'] = 'this value WILL be stored'
>>> p['foo']
'this value WILL be stored'
```

### *class* dol.trans.ValueCodec(encoder, decoder)

Bases: [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`DecodedType`, `EncodedType`], [`Codec`](#dol.trans.Codec)[`DecodedType`, `EncodedType`]

### dol.trans.add_aliases(obj, \*\*aliases)

A function that wraps the object instance and adds aliases.

See also, and not to be confused with `insert_aliases`, which adds aliases to
dunder mapping methods (like `__iter__`, `__getitem__`) etc.

### dol.trans.add_decoder(store_cls=None, , decoder=None, name=None, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Add a decoder layer to a store.

#### NOTE
This is a convenience function for `wrap_kvs(..., obj_of_data=decoder)`.

```pycon
>>> s = {'a': "42"}
>>> ss = add_decoder(s, decoder=int)
>>> ss['a']
42
```

If there’s only one callable argument, it is assumed to be the decoder:

```pycon
>>> wrapper = add_decoder(int)
>>> S = wrapper(dict)
>>> sss = S({'a': "42"})
>>> dict(sss) == {'a': 42}
True
```

### dol.trans.add_ipython_key_completions(store)

Add tab completion that shows you the keys of the store.

#### NOTE
ipython already adds local path listing automatically,

so you’ll still get those along with your valid store keys.

### dol.trans.add_missing_key_handling(store=None, \*, missing_key_callback, errors_that_trigger_missing=(<class 'KeyError'>, ), \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Overrides the `__missing__` method of a store with a custom callback.

#### NOTE
The callback must have two arguments: the store and the key.

* **Parameters:**
  * **store** – The store class to wrap.
  * **missing_key_callback** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – Function(store, key) -> value for missing keys.
  * **errors_that_trigger_missing** – Tuple of exceptions that trigger \_\_missing_\_.

In the following example, we endow a store to return a sub-store when a key is
missing. This substore will contain only keys that start with that missing key.
This is useful, for example, to get “subfolder filtering” on a store.

```pycon
>>> def prefix_filter(store, prefix: str):
...     '''Filter the store to have only keys that start with prefix'''
...     from dol import filt_iter
...     return filt_iter(store, filt=lambda x: x.startswith(prefix))
...
>>> @add_missing_key_handling(missing_key_callback=prefix_filter)
... class D(dict):
...     pass
>>>
>>> s = D({'a/b': 1, 'a/c': 2, 'd/e': 3, 'f': 4})
>>> sorted(s)
['a/b', 'a/c', 'd/e', 'f']
>>> 'a/' not in s
True
>>> # yet
>>> v = s['a/']
>>> assert dict(v) == {'a/b': 1, 'a/c': 2}
```

### dol.trans.add_path_access(store=None, \*, name=None, path_type=<class 'tuple'>, create_missing=False, mk_missing=None, explore_further=None, may_create=None, on_create=<function \_warn_on_create>, max_created=None, max_levels=20, verify_writeback=False, writeback_lock=None, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Make nested stores (read/write) accessible through key paths (iterable of keys).

Like `add_path_get`, but with write and delete accessible through key paths.

In a way “flatten the nested keys access”.
(Warning: `path_type` only effects the first level.
That is, it doesn’t work recursively.
See issue: [https://github.com/i2mint/dol/issues/10](https://github.com/i2mint/dol/issues/10).)

By default, the path object will be a tuple (e.g. `('a', 'b', 'c')`, but you can
make it whatever you want, and/or use `dol.paths.KeyPath` to map to and from
forms like `'a.b.c'`, `'a/b/c'`, etc.

Say you have some nested stores.
You know… like a `ZipFileReader` store whose values are 

```
`
```

ZipReader\`s,
whose values are bytes of the zipped files
(and you can go on… whose (json) values are…).

For our example, let’s take a nested dict instead:

```pycon
>>> s = {'a': {'b': {'c': 42}}}
```

Well, you can access any node of this nested tree of stores like this:

```pycon
>>> s['a']['b']['c']
42
```

And that’s fine. But maybe you’d like to do it this way instead:

```pycon
>>> s = add_path_access(s)
>>> s['a', 'b', 'c']
42
```

So far, this is what `add_path_get` does. With `add_path_access` though you
can also write and delete that way too:

```pycon
>>> s['a', 'b', 'c'] = 3.14
>>> s['a', 'b', 'c']
3.14
>>> del s['a', 'b', 'c']
>>> s
{'a': {'b': {}}}
```

You might also want to access 42 with `a.b.c` or `a/b/c` etc.
To do that you can use `dol.paths.KeyPath` in combination with

* **Parameters:**
  * **store** – The store (class or instance) you’re wrapping.
    If not specified, the function will return a decorator.
  * **name** – The name to give the class (not applicable to instance wrapping)
  * **path_type** ([`type`](https://docs.python.org/3/library/functions.html#type)) – The type that paths are expressed as. Needs to be an Iterable type.
    By default, a tuple.
    This is used to decide whether the key should be taken as a “normal”
    key of the store,
    or should be used to iterate through, recursively getting values.
* **Returns:**
  A wrapped store (class or instance), or a store wrapping decorator
  (if store is not specified)

#### SEE ALSO
`KeyPath` in paths

Wrapping a class

```pycon
>>> S = add_path_access(dict)
>>> s = S(a={'b': {'c': 42}})
>>> assert s['a'] == {'b': {'c': 42}};
>>> assert s['a', 'b'] == {'c': 42};
>>> assert s['a', 'b', 'c'] == 42
>>> s['a', 'b', 'c'] = 3.14
>>> s['a', 'b', 'c']
3.14
>>> del s['a', 'b', 'c']
>>> s
{'a': {'b': {}}}
```

Using add_path_get as a decorator

```pycon
>>> @add_path_access
... class S(dict):
...    pass
>>> s = S(a={'b': {'c': 42}})
>>> assert s['a'] == {'b': {'c': 42}};
>>> assert s['a', 'b'] == s['a']['b'] == {'c': 42};
>>> assert s['a', 'b', 'c'] == s['a']['b']['c'] == 42
>>> s['a', 'b', 'c'] = 3.14
>>> s['a', 'b', 'c']
3.14
>>> del s['a', 'b', 'c']
>>> s
{'a': {'b': {}}}
```

A different kind of path?
You can choose a different path_type, but sometimes (say both keys and key paths are strings)
You need to involve more tools. Like dol.paths.KeyPath…

```pycon
>>> from dol.paths import KeyPath
>>> from dol.trans import kv_wrap
>>> SS = kv_wrap(KeyPath(path_sep='.'))(S)
>>> s = SS({'a': {'b': {'c': 42}}})
>>> assert s['a'] == {'b': {'c': 42}};
>>> assert s['a.b'] == s['a']['b'];
>>> assert s['a.b.c'] == s['a']['b']['c']
>>> s['a.b.c'] = 3.14
>>> s
{'a': {'b': {'c': 3.14}}}
>>> del s['a.b.c']
>>> s
{'a': {'b': {}}}
```

#### NOTE
The add_path_access doesn’t carry on to values.

```pycon
>>> s = add_path_access({'a': {'b': {'c': 42}}})
>>> s['a', 'b', 'c']
42
>>> # but
>>> s['a']['b', 'c']
Traceback (most recent call last):
  ...
KeyError: ('b', 'c')
```

That said,

```pycon
>>> add_path_access(s['a'])['b', 'c']
42
```

The reason why we don’t do this automatically is that it may not always be desirable.
If one wanted to though, one could use `wrap_kvs(obj_of_data=...)` to wrap
specific values with `add_path_access`.
For example, if you wanted to wrap all mappings recursively, you could:

```pycon
>>> from typing import Mapping
>>> from dol.util import instance_checker
>>> add_path_access_if_mapping = conditional_data_trans(
...     condition=instance_checker(Mapping), data_trans=add_path_access
... )
>>> s = add_path_access_if_mapping({'a': {'b': {'c': 42}}})
>>> s['a', 'b', 'c']
42
>>> # But now this works:
>>> s['a']['b', 'c']
42
```

### dol.trans.add_path_get(store=None, \*, name=None, path_type=<class 'tuple'>, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Make nested stores accessible through key paths.
In a way “flatten the nested keys access”.
By default, the path object will be a tuple (e.g. `('a', 'b', 'c')`, but you can
make it whatever you want, and/or use `dol.paths.KeyPath` to map to and from
forms like `'a.b.c'`, `'a/b/c'`, etc.

(Warning: `path_type` only effects the first level.
That is, it doesn’t work recursively.
See issue: [https://github.com/i2mint/dol/issues/10](https://github.com/i2mint/dol/issues/10).)

Say you have some nested stores.
You know… like a `ZipFileReader` store whose values are 

```
`
```

ZipReader\`s,
whose values are bytes of the zipped files
(and you can go on… whose (json) values are…).

For our example, let’s take a nested dict instead:

```pycon
>>> s = {'a': {'b': {'c': 42}}}
```

Well, you can access any node of this nested tree of stores like this:

```pycon
>>> s['a']['b']['c']
42
```

And that’s fine. But maybe you’d like to do it this way instead:

```pycon
>>> s = add_path_get(s)
>>> s['a', 'b', 'c']
42
```

You might also want to access 42 with `a.b.c` or `a/b/c` etc.
To do that you can use `dol.paths.KeyPath` in combination with

* **Parameters:**
  * **store** – The store (class or instance) you’re wrapping.
    If not specified, the function will return a decorator.
  * **name** – The name to give the class (not applicable to instance wrapping)
  * **path_type** ([`type`](https://docs.python.org/3/library/functions.html#type)) – The type that paths are expressed as. Needs to be an Iterable type.
    By default, a tuple.
    This is used to decide whether the key should be taken as a “normal”
    key of the store,
    or should be used to iterate through, recursively getting values.
* **Returns:**
  A wrapped store (class or instance), or a store wrapping decorator
  (if store is not specified)

#### SEE ALSO
`KeyPath` in paths

Wrapping an instance

```pycon
>>> s = add_path_get({'a': {'b': {'c': 42}}})
>>> s['a']
{'b': {'c': 42}}
>>> s['a', 'b']
{'c': 42}
>>> s['a', 'b', 'c']
42
```

Wrapping a class

```pycon
>>> S = add_path_get(dict)
>>> s = S(a={'b': {'c': 42}})
>>> assert s['a'] == {'b': {'c': 42}};
>>> assert s['a', 'b'] == {'c': 42};
>>> assert s['a', 'b', 'c'] == 42
```

Using add_path_get as a decorator

```pycon
>>> @add_path_get
... class S(dict):
...    pass
>>> s = S(a={'b': {'c': 42}})
>>> assert s['a'] == {'b': {'c': 42}};
>>> assert s['a', 'b'] == s['a']['b'] == {'c': 42};
>>> assert s['a', 'b', 'c'] == s['a']['b']['c'] == 42
```

A different kind of path?
You can choose a different path_type, but sometimes (say both keys and key paths are strings)
You need to involve more tools. Like dol.paths.KeyPath…

```pycon
>>> from dol.paths import KeyPath
>>> from dol.trans import kv_wrap
>>> SS = kv_wrap(KeyPath(path_sep='.'))(S)
>>> s = SS({'a': {'b': {'c': 42}}})
>>> assert s['a'] == {'b': {'c': 42}};
>>> assert s['a.b'] == s['a']['b'];
>>> assert s['a.b.c'] == s['a']['b']['c']
```

### dol.trans.add_store_method(store, , method_func, method_name=None, validator=None, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Add methods to store classes or instances

* **Parameters:**
  * **store** ([`type`](https://docs.python.org/3/library/functions.html#type)) – A store type or instance
  * **method_func** – The function of the method to be added
  * **method_name** – The name of the store attribute this function should be written to
  * **validator** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`type`](https://docs.python.org/3/library/functions.html#type), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)], [`bool`](https://docs.python.org/3/library/functions.html#bool)] | [`None`](https://docs.python.org/3/library/constants.html#None)) – An optional validator. If not None, `validator(store, method_func)` will be called.
    If it doesn’t return True, a `SetattrNotAllowed` will be raised.
    Note that `validator` can also raise its own exception.
* **Returns:**
  A store with the added (or modified) method

### dol.trans.add_wrapper_method(wrap_cls=None, , method_name='wrapper')

Decorator that adds a wrapper method (itself a decorator) to a wrapping class
Clear?
See `mk_wrapper` function and doctest example if not.

What `add_wrapper_method` does is just to add a `"wrapper"` method
(or another name if you ask for it) to `wrap_cls`, so that you can use that
class for it’s purpose of transforming stores more conveniently.

* **Parameters:**
  * **wrap_cls** – The wrapper class (the definitioin of the transformation.
    If None, the functiion will make a decorator to decorate wrap_cls later
  * **method_name** – The method name you want to use (default is ‘wrapper’)

```pycon
>>>
>>> @add_wrapper_method
... class RelPath:
...     def __init__(self, root):
...         self.root = root
...         self._root_length = len(root)
...     def _key_of_id(self, _id):
...         return _id[self._root_length:]
...     def _id_of_key(self, k):
...         return self.root + k
...
>>> RelDict = RelPath.wrapper(root='foo/')(dict)
>>> s = RelDict()
>>> s['bar'] = 42
>>> assert list(s) == ['bar']
>>> assert s['bar'] == 42
>>> assert str(s) == "{'foo/bar': 42}"  # reveals that actually, behind the scenes, there's a "foo/" prefix
```

### dol.trans.affix_key_codec(prefix='', suffix='')

A factory that creates a key codec that affixes a prefix and suffix to the key

```pycon
>>> codec = affix_key_codec(prefix='/folder/', suffix='.txt')
>>> codec.encoder('name')
'/folder/name.txt'
>>> codec.decoder('/folder/name.txt')
'name'
```

### dol.trans.assert_min_num_of_args(func, num_of_args)

Assert that a function can be a store method.
That is, it should have a signature that takes the store as the first argument

### dol.trans.autoviv(store=None, \*\*kwargs)

Opt-in write-through autovivification for key-paths.

Shorthand for `add_path_access(store, create_missing=True, **kwargs)`: writing
through a key-path (e.g. `s['a', 'b', 'c'] = v`) creates any missing intermediate
levels on the way, and the change persists correctly even through persistent /
copy-semantics stores (`Files`, `wrap_kvs`-wrapped stores). See
`add_path_access` for the full set of options (`mk_missing`, `may_create`,
`max_created`, `on_create`, …). Missing-key creation is announced via
`warnings.warn` by default, so a typo is never silent.

```pycon
>>> from dol import autoviv
>>> s = autoviv({})
>>> s['a', 'b', 'c'] = 42
>>> s['a', 'b', 'c']
42
>>> s['a']['b']['c']
42
```

Like `add_path_access`, it also works as a class decorator / factory:

```pycon
>>> S = autoviv(dict)
>>> s = S()
>>> s['x', 'y'] = 1
>>> s['x', 'y']
1
```

### dol.trans.cache_iter(store=None, \*, keys_cache=<class 'list'>, iter_to_container=None, cache_update_method='update', name=None, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Make a class that wraps input class’s \_\_iter_\_ becomes cached.

Quite often we have a lot of keys, that we get from a remote data source, and don’t want to have to ask for
them again and again, having them be fetched, sent over the network, etc.
So we need caching.

But this caching is not the typical read caching, since it’s \_\_iter_\_ we want to cache, and that’s a generator.
So we’ll implement a store class decorator specialized for this.

The following decorator, when applied to a class (that has an \_\_iter_\_), will perform the \_\_iter_\_ code, consuming
all items of the generator and storing them in \_keys_cache, and then will yield from there every subsequent call.

It is assumed, if you’re using the cached_keys transformation, that you’re dealing with static data
(or data that can be considered static for the life of the store – for example, when conducting analytics).
If you ever need to refresh the cache during the life of the store, you can to delete \_keys_cache like this:

```text
del your_store._keys_cache
```

Once you do that, the next time you try to ask something about the contents of the store, it will actually do
a live query again, as for the first time.

#### NOTE
The default keys_cache is list though in many cases, you’d probably should use set, or an explicitly
computer set instead. The reason list is used as the default is because (1) we didn’t want to assume that
order did not matter (maybe it does to you) and (2) we didn’t want to assume that your keys were hashable.
That said, if you’re keys are hashable, and order does not matter, use set. That’ll give you two things:
(a) your `key in store` checks will be faster (O(1) instead of O(n)) and (b) you’ll enforce unicity of keys.

Know also that if you precompute the keys you want to cache with a container that has an update
method (by default `update`) your cache updates will be faster and if the container you use has
a `remove` method, you’ll be able to delete as well.

* **Parameters:**
  * **store** – The store instance or class to wrap (must have an \_\_iter_\_), or None if you want a decorator.
  * **keys_cache** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`Collection`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Collection)) – An explicit collection of keys
  * **iter_to_container** – The function that will be applied to existing \_\_iter_\_() and assigned to cache.
    The default is list. Another useful one is the sorted function.
  * **cache_update_method** – 

    Name of the keys_cache update method to use, if it is an attribute of keys_cache.
    Note that this cache_update_method will be used only
    > if keys_cache is an explicit iterable and has that attribute
    > if keys_cache is a callable and has that attribute.

    The default None
  * **name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – The name of the new class
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`KvReader`](dol.base.html.md#dol.base.KvReader)
* **Returns:**
  If store is:
  ```default
  None: Will return a decorator that can be applied to a store
  a store class: Will return a wrapped class that caches it's keys
  a store instance: Will return a wrapped instance that caches it's keys
  ```

  The instances of such key-cached classes have some extra attributes:
  ```default
  _explicit_keys: The actual cache. An iterable container
  update_keys_cache: Is called if a user uses the instance to mutate the store (i.e. write or delete).
  ```

You have two ways of caching keys:

- By providing the explicit list of keys you want cache (and use)
- By providing a callable that will iterate through your store and collect an explicit list of keys

Let’s take a simple dict as our original store.

```pycon
>>> source = dict(c=3, b=2, a=1)
```

Specify an iterable, and it will be used as the cached keys

```pycon
>>> cached = cached_keys(source, keys_cache='bc')
>>> list(cached.items())  # notice that the order you get things is also ruled by the cache
[('b', 2), ('c', 3)]
```

Specify a callable, and it will apply it to the existing keys to make your cache

```pycon
>>> list(cached_keys(source, keys_cache=sorted))
['a', 'b', 'c']
```

You can use the callable keys_cache specification to filter as well!
Oh, and let’s demo the fact that if you don’t specify the store, it will make a store decorator for you:

```pycon
>>> cache_my_keys = cached_keys(keys_cache=lambda keys: list(filter(lambda k: k >= 'b', keys)))
>>> d = cache_my_keys(source)  # used as to transform an instance
>>> list(d)
['c', 'b']
```

Let’s use that same `cache_my_keys` to decorate a class instead:

```pycon
>>> cached_dict = cache_my_keys(dict)
>>> d = cached_dict(c=3, b=2, a=1)
>>> list(d)
['c', 'b']
```

Note that there’s still an underlying store (dict) that has the data:

```pycon
>>> repr(d)  # repr isn't wrapped, so you can still see your underlying dict
"{'c': 3, 'b': 2, 'a': 1}"
```

And yes, you can still add elements,

```pycon
>>> d['z'] = 26
>>> list(d.items())
[('c', 3), ('b', 2), ('z', 26)]
```

do bulk updates,

```pycon
>>> d.update({'more': 'of this'}, more_of='that')
>>> list(d.items())
[('c', 3), ('b', 2), ('z', 26), ('more', 'of this'), ('more_of', 'that')]
```

and delete…

```pycon
>>> del d['more']
>>> list(d.items())
[('c', 3), ('b', 2), ('z', 26), ('more_of', 'that')]
```

But careful! Know what you’re doing if you try to get creative. Have a look at this:

```pycon
>>> d['a'] = 100  # add an 'a' item
>>> d.update(and_more='of that')  # update to add yet another item
>>> list(d.items())
[('c', 3), ('b', 2), ('z', 26), ('more_of', 'that')]
```

Indeed: No ‘a’ or ‘and_more’.

Now… they were indeed added. Or to be more precise, the value of the already existing a was changed,
and a new (‘and_more’, ‘of that’) item was indeed added in the underlying store:

```pycon
>>> repr(d)
"{'c': 3, 'b': 2, 'a': 100, 'z': 26, 'more_of': 'that', 'and_more': 'of that'}"
```

But you’re not seeing it.

Why?

Because you chose to use a callable keys_cache that doesn’t have an ‘update’ method.
When your \_keys_cache attribute (the iterable cache) is not updatable itself, the
way updates work is that we iterate through the underlying store (where the updates actually took place),
and apply the keys_cache (callable) to that iterable.

So what happened here was that you have your new ‘a’ and ‘and_more’ items, but your cached version of the
store doesn’t see it because it’s filtered out. On the other hand, check out what happens if you have
an updateable cache.

Using `set` instead of `list`, after the `filter`.

```pycon
>>> cache_my_keys = cached_keys(keys_cache=set)
>>> d = cache_my_keys(source)  # used as to transform an instance
>>> sorted(d)  # using sorted because a set's order is not always the same
['a', 'b', 'c']
>>> d['a'] = 100
>>> d.update(and_more='of that')  # update to add yet another item
>>> sorted(d.items())
[('a', 100), ('and_more', 'of that'), ('b', 2), ('c', 3)]
```

This example was to illustrate a more subtle aspect of cached_keys. You would probably deal with
the filter concern in a different way in this case. But the rope is there – it’s your choice on how
to use it.

And here’s some more examples if that wasn’t enough!

```pycon
>>> # Lets cache the keys of a dict.
>>> cached_dict = cached_keys(dict)
>>> d = cached_dict(a=1, b=2, c=3)
>>> # And you get a store that behaves as expected (but more speed and RAM)
>>> list(d)
['a', 'b', 'c']
>>> list(d.items())  # whether you iterate with .keys(), .values(), or .items()
[('a', 1), ('b', 2), ('c', 3)]
```

This is where the keys are stored:

```pycon
>>> d._keys_cache
['a', 'b', 'c']
```

```pycon
>>> # Let's demo the iter_to_container argument. The default is "list", which will just consume the iter in order
>>> sorted_dict = cached_keys(dict, keys_cache=list)
>>> s = sorted_dict({'b': 3, 'a': 2, 'c': 1})
>>> list(s)  # keys will be in the order they were defined
['b', 'a', 'c']
>>> sorted_dict = cached_keys(dict, keys_cache=sorted)
>>> s = sorted_dict({'b': 3, 'a': 2, 'c': 1})
>>> list(s)  # keys will be sorted
['a', 'b', 'c']
>>> sorted_dict = cached_keys(dict, keys_cache=lambda x: sorted(x, key=len))
>>> s = sorted_dict({'bbb': 3, 'aa': 2, 'c': 1})
>>> list(s)  # keys will be sorted according to their length
['c', 'aa', 'bbb']
```

If you change the keys (adding new ones with \_\_setitem_\_ or update, or removing with pop or popitem)
then the cache is recomputed (the first time you use an operation that iterates over keys)

```pycon
>>> d.update(d=4)  # let's add an element (try d['d'] = 4 as well)
>>> list(d)
['a', 'b', 'c', 'd']
>>> d['e'] = 5
>>> list(d.items())  # whether you iterate with .keys(), .values(), or .items()
[('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5)]
```

```pycon
>>> @cached_keys
... class A:
...     def __iter__(self):
...         yield from [1, 2, 3]
>>> # Note, could have also used this form: AA = cached_keys(A)
>>> a = A()
>>> list(a)
[1, 2, 3]
>>> a._keys_cache = ['a', 'b', 'c']  # changing the cache, to prove that subsequent listing will read from there
>>> list(a)  # proof:
['a', 'b', 'c']
>>>
```

```pycon
>>> # Let's demo the iter_to_container argument. The default is "list", which will just consume the iter in order
>>> sorted_dict = cached_keys(dict, keys_cache=list)
>>> s = sorted_dict({'b': 3, 'a': 2, 'c': 1})
>>> list(s)  # keys will be in the order they were defined
['b', 'a', 'c']
>>> sorted_dict = cached_keys(dict, keys_cache=sorted)
>>> s = sorted_dict({'b': 3, 'a': 2, 'c': 1})
>>> list(s)  # keys will be sorted
['a', 'b', 'c']
>>> sorted_dict = cached_keys(dict, keys_cache=lambda x: sorted(x, key=len))
>>> s = sorted_dict({'bbb': 3, 'aa': 2, 'c': 1})
>>> list(s)  # keys will be sorted according to their length
['c', 'aa', 'bbb']
```

### dol.trans.cached_keys(store=None, \*, keys_cache=<class 'list'>, iter_to_container=None, cache_update_method='update', name=None, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Make a class that wraps input class’s \_\_iter_\_ becomes cached.

Quite often we have a lot of keys, that we get from a remote data source, and don’t want to have to ask for
them again and again, having them be fetched, sent over the network, etc.
So we need caching.

But this caching is not the typical read caching, since it’s \_\_iter_\_ we want to cache, and that’s a generator.
So we’ll implement a store class decorator specialized for this.

The following decorator, when applied to a class (that has an \_\_iter_\_), will perform the \_\_iter_\_ code, consuming
all items of the generator and storing them in \_keys_cache, and then will yield from there every subsequent call.

It is assumed, if you’re using the cached_keys transformation, that you’re dealing with static data
(or data that can be considered static for the life of the store – for example, when conducting analytics).
If you ever need to refresh the cache during the life of the store, you can to delete \_keys_cache like this:

```text
del your_store._keys_cache
```

Once you do that, the next time you try to ask something about the contents of the store, it will actually do
a live query again, as for the first time.

#### NOTE
The default keys_cache is list though in many cases, you’d probably should use set, or an explicitly
computer set instead. The reason list is used as the default is because (1) we didn’t want to assume that
order did not matter (maybe it does to you) and (2) we didn’t want to assume that your keys were hashable.
That said, if you’re keys are hashable, and order does not matter, use set. That’ll give you two things:
(a) your `key in store` checks will be faster (O(1) instead of O(n)) and (b) you’ll enforce unicity of keys.

Know also that if you precompute the keys you want to cache with a container that has an update
method (by default `update`) your cache updates will be faster and if the container you use has
a `remove` method, you’ll be able to delete as well.

* **Parameters:**
  * **store** – The store instance or class to wrap (must have an \_\_iter_\_), or None if you want a decorator.
  * **keys_cache** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`Collection`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Collection)) – An explicit collection of keys
  * **iter_to_container** – The function that will be applied to existing \_\_iter_\_() and assigned to cache.
    The default is list. Another useful one is the sorted function.
  * **cache_update_method** – 

    Name of the keys_cache update method to use, if it is an attribute of keys_cache.
    Note that this cache_update_method will be used only
    > if keys_cache is an explicit iterable and has that attribute
    > if keys_cache is a callable and has that attribute.

    The default None
  * **name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – The name of the new class
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`KvReader`](dol.base.html.md#dol.base.KvReader)
* **Returns:**
  If store is:
  ```default
  None: Will return a decorator that can be applied to a store
  a store class: Will return a wrapped class that caches it's keys
  a store instance: Will return a wrapped instance that caches it's keys
  ```

  The instances of such key-cached classes have some extra attributes:
  ```default
  _explicit_keys: The actual cache. An iterable container
  update_keys_cache: Is called if a user uses the instance to mutate the store (i.e. write or delete).
  ```

You have two ways of caching keys:

- By providing the explicit list of keys you want cache (and use)
- By providing a callable that will iterate through your store and collect an explicit list of keys

Let’s take a simple dict as our original store.

```pycon
>>> source = dict(c=3, b=2, a=1)
```

Specify an iterable, and it will be used as the cached keys

```pycon
>>> cached = cached_keys(source, keys_cache='bc')
>>> list(cached.items())  # notice that the order you get things is also ruled by the cache
[('b', 2), ('c', 3)]
```

Specify a callable, and it will apply it to the existing keys to make your cache

```pycon
>>> list(cached_keys(source, keys_cache=sorted))
['a', 'b', 'c']
```

You can use the callable keys_cache specification to filter as well!
Oh, and let’s demo the fact that if you don’t specify the store, it will make a store decorator for you:

```pycon
>>> cache_my_keys = cached_keys(keys_cache=lambda keys: list(filter(lambda k: k >= 'b', keys)))
>>> d = cache_my_keys(source)  # used as to transform an instance
>>> list(d)
['c', 'b']
```

Let’s use that same `cache_my_keys` to decorate a class instead:

```pycon
>>> cached_dict = cache_my_keys(dict)
>>> d = cached_dict(c=3, b=2, a=1)
>>> list(d)
['c', 'b']
```

Note that there’s still an underlying store (dict) that has the data:

```pycon
>>> repr(d)  # repr isn't wrapped, so you can still see your underlying dict
"{'c': 3, 'b': 2, 'a': 1}"
```

And yes, you can still add elements,

```pycon
>>> d['z'] = 26
>>> list(d.items())
[('c', 3), ('b', 2), ('z', 26)]
```

do bulk updates,

```pycon
>>> d.update({'more': 'of this'}, more_of='that')
>>> list(d.items())
[('c', 3), ('b', 2), ('z', 26), ('more', 'of this'), ('more_of', 'that')]
```

and delete…

```pycon
>>> del d['more']
>>> list(d.items())
[('c', 3), ('b', 2), ('z', 26), ('more_of', 'that')]
```

But careful! Know what you’re doing if you try to get creative. Have a look at this:

```pycon
>>> d['a'] = 100  # add an 'a' item
>>> d.update(and_more='of that')  # update to add yet another item
>>> list(d.items())
[('c', 3), ('b', 2), ('z', 26), ('more_of', 'that')]
```

Indeed: No ‘a’ or ‘and_more’.

Now… they were indeed added. Or to be more precise, the value of the already existing a was changed,
and a new (‘and_more’, ‘of that’) item was indeed added in the underlying store:

```pycon
>>> repr(d)
"{'c': 3, 'b': 2, 'a': 100, 'z': 26, 'more_of': 'that', 'and_more': 'of that'}"
```

But you’re not seeing it.

Why?

Because you chose to use a callable keys_cache that doesn’t have an ‘update’ method.
When your \_keys_cache attribute (the iterable cache) is not updatable itself, the
way updates work is that we iterate through the underlying store (where the updates actually took place),
and apply the keys_cache (callable) to that iterable.

So what happened here was that you have your new ‘a’ and ‘and_more’ items, but your cached version of the
store doesn’t see it because it’s filtered out. On the other hand, check out what happens if you have
an updateable cache.

Using `set` instead of `list`, after the `filter`.

```pycon
>>> cache_my_keys = cached_keys(keys_cache=set)
>>> d = cache_my_keys(source)  # used as to transform an instance
>>> sorted(d)  # using sorted because a set's order is not always the same
['a', 'b', 'c']
>>> d['a'] = 100
>>> d.update(and_more='of that')  # update to add yet another item
>>> sorted(d.items())
[('a', 100), ('and_more', 'of that'), ('b', 2), ('c', 3)]
```

This example was to illustrate a more subtle aspect of cached_keys. You would probably deal with
the filter concern in a different way in this case. But the rope is there – it’s your choice on how
to use it.

And here’s some more examples if that wasn’t enough!

```pycon
>>> # Lets cache the keys of a dict.
>>> cached_dict = cached_keys(dict)
>>> d = cached_dict(a=1, b=2, c=3)
>>> # And you get a store that behaves as expected (but more speed and RAM)
>>> list(d)
['a', 'b', 'c']
>>> list(d.items())  # whether you iterate with .keys(), .values(), or .items()
[('a', 1), ('b', 2), ('c', 3)]
```

This is where the keys are stored:

```pycon
>>> d._keys_cache
['a', 'b', 'c']
```

```pycon
>>> # Let's demo the iter_to_container argument. The default is "list", which will just consume the iter in order
>>> sorted_dict = cached_keys(dict, keys_cache=list)
>>> s = sorted_dict({'b': 3, 'a': 2, 'c': 1})
>>> list(s)  # keys will be in the order they were defined
['b', 'a', 'c']
>>> sorted_dict = cached_keys(dict, keys_cache=sorted)
>>> s = sorted_dict({'b': 3, 'a': 2, 'c': 1})
>>> list(s)  # keys will be sorted
['a', 'b', 'c']
>>> sorted_dict = cached_keys(dict, keys_cache=lambda x: sorted(x, key=len))
>>> s = sorted_dict({'bbb': 3, 'aa': 2, 'c': 1})
>>> list(s)  # keys will be sorted according to their length
['c', 'aa', 'bbb']
```

If you change the keys (adding new ones with \_\_setitem_\_ or update, or removing with pop or popitem)
then the cache is recomputed (the first time you use an operation that iterates over keys)

```pycon
>>> d.update(d=4)  # let's add an element (try d['d'] = 4 as well)
>>> list(d)
['a', 'b', 'c', 'd']
>>> d['e'] = 5
>>> list(d.items())  # whether you iterate with .keys(), .values(), or .items()
[('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5)]
```

```pycon
>>> @cached_keys
... class A:
...     def __iter__(self):
...         yield from [1, 2, 3]
>>> # Note, could have also used this form: AA = cached_keys(A)
>>> a = A()
>>> list(a)
[1, 2, 3]
>>> a._keys_cache = ['a', 'b', 'c']  # changing the cache, to prove that subsequent listing will read from there
>>> list(a)  # proof:
['a', 'b', 'c']
>>>
```

```pycon
>>> # Let's demo the iter_to_container argument. The default is "list", which will just consume the iter in order
>>> sorted_dict = cached_keys(dict, keys_cache=list)
>>> s = sorted_dict({'b': 3, 'a': 2, 'c': 1})
>>> list(s)  # keys will be in the order they were defined
['b', 'a', 'c']
>>> sorted_dict = cached_keys(dict, keys_cache=sorted)
>>> s = sorted_dict({'b': 3, 'a': 2, 'c': 1})
>>> list(s)  # keys will be sorted
['a', 'b', 'c']
>>> sorted_dict = cached_keys(dict, keys_cache=lambda x: sorted(x, key=len))
>>> s = sorted_dict({'bbb': 3, 'aa': 2, 'c': 1})
>>> list(s)  # keys will be sorted according to their length
['c', 'aa', 'bbb']
```

### dol.trans.catch_and_cache_error_keys(store=None, \*, errors_caught=<class 'Exception'>, error_callback=None, use_cached_keys_after_completed_iter=True, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Store that will cache keys as they’re accessed, separating those that raised errors and those that didn’t.
Getting a key will still through an error, but the access attempts will be collected in an ._error_keys attribute.
Successfful attemps will be stored in \_keys_cache.
Retrieval iteration (items() or values()) will on the other hand, skip the error (while still caching it).
If the iteration completes (and use_cached_keys_after_completed_iter), the use_cached_keys flag is turned on,
which will result in the store now getting it’s keys from the \_keys_cache.

```pycon
>>> @catch_and_cache_error_keys(
...     error_callback=lambda store, key, err: print(f"Error with {key} key: {err}"))
... class Blacklist(dict):
...     _black_list = {'black', 'list'}
...
...     def __getitem__(self, k):
...         if k not in self._black_list:
...             return super().__getitem__(k)
...         else:
...             raise KeyError(f"Nope, that's from the black list!")
>>>
>>> s = Blacklist(black=7,  friday=20, frenzy=13)
>>> list(s)
['black', 'friday', 'frenzy']
>>> list(s.items())
Error with black key: "Nope, that's from the black list!"
[('friday', 20), ('frenzy', 13)]
>>> sorted(s)  # sorting to get consistent output
['frenzy', 'friday']
```

See that? First we had three keys, then we iterated and got only 2 items (fortunately,
we specified an `error_callback` so we ccould see that the iteration actually
dropped a key).

That’s strange. And even stranger is the fact that when we list our keys again,
we get only two.

You don’t like it? Neither do I. But

- It’s not a completely outrageous behavior – if you’re talking to live data, it
  : often happens that you get more, or less, from one second to another.
- This store isn’t meant to be long living, but rather meant to solve the problem of
  : skiping items that are problematic (for example, malformatted files),
    with a trace of what was skipped and what’s valid (in case we need to iterate
    again and don’t want to bear the hit of requesting values for keys we already
    know are problematic.

Here’s a little peep of what is happening under the hood.
Meet `_keys_cache` and `_error_keys` sets (yes, unordered – so know it) that are meant
to acccumulate valid and problematic keys respectively.

```pycon
>>> s = Blacklist(black=7,  friday=20, frenzy=13)
>>> list(s)
['black', 'friday', 'frenzy']
>>> s._keys_cache, s._error_keys
(set(), set())
>>> s['friday']
20
>>> s._keys_cache, s._error_keys
({'friday'}, set())
>>> s['black']
Traceback (most recent call last):
  ...
KeyError: "Nope, that's from the black list!"
>>> s._keys_cache, s._error_keys
({'friday'}, {'black'})
```

But see that we still have the full list:

```pycon
>>> list(s)
['black', 'friday', 'frenzy']
```

Meet `use_cached_keys`: He’s the culprit. It’s a flag that indicates whether
we should be using the cached keys or not. Obviously, it’ll start off being
`False`:

```pycon
>>> s.use_cached_keys
False
```

Now we could set it to `True` manually to change the mode.
But know that this switch happens automatically (UNLESS you specify otherwise by
saying:`use_cached_keys_after_completed_iter=False`) when ever you got through a
VALUE-PRODUCING iteration (i.e. entirely consuming `items()` or `values()`).

```pycon
>>> sorted(s.values())  # sorting to get consistent output
Error with black key: "Nope, that's from the black list!"
[13, 20]
```

### dol.trans.confirm_overwrite(self, k, v)

A ready-to-use `wrap_kvs` `preset` that asks (via the builtin `input`) to confirm
before overwriting an existing key with a different value (Issue #13). For customization
(e.g. a non-interactive confirmation policy), use [`mk_confirm_overwrite_preset()`](#dol.trans.mk_confirm_overwrite_preset).

```pycon
>>> from dol import wrap_kvs, confirm_overwrite
>>> d = wrap_kvs(dict(a='apple', b='banana'), preset=confirm_overwrite)
>>> d['a'] = 'apple'      # same value -> no prompt, no change
>>> d['c'] = 'coconut'    # new key   -> no prompt, written
>>> dict(d) == {'a': 'apple', 'b': 'banana', 'c': 'coconut'}
True
```

### dol.trans.constant_output(return_val=None, \*args, \*\*kwargs)

Function that returns a constant value no matter what the inputs are.
Is meant to be used with functools.partial to create custom versions.

```pycon
>>> from functools import partial
>>> always_true = partial(constant_output, True)
>>> always_true('regardless', 'of', the='input', will='return True')
True
```

### dol.trans.double_up_as_factory(decorator_func)

Repurpose a decorator both as it’s original form, and as a decorator factory.
That is, from a decorator that is defined do `wrapped_func = decorator(func, **params)`,
make it also be able to do `wrapped_func = decorator(**params)(func)`.

#### NOTE
You’ll only be able to do this if all but the first argument are keyword-only,
and the first argument (the function to decorate) has a default of `None` (this is for your own good).
This is validated before making the “double up as factory” decorator.

```pycon
>>> @double_up_as_factory
... def decorator(func=None, *, multiplier=2):
...     def _func(x):
...         return func(x) * multiplier
...     return _func
...
>>> def foo(x):
...     return x + 1
...
>>> foo(2)
3
>>> wrapped_foo = decorator(foo, multiplier=10)
>>> wrapped_foo(2)
30
>>>
>>> multiply_by_3 = decorator(multiplier=3)
>>> wrapped_foo = multiply_by_3(foo)
>>> wrapped_foo(2)
9
>>>
>>> @decorator(multiplier=3)
... def foo(x):
...     return x + 1
...
>>> foo(2)
9
```

Note that to be able to use double_up_as_factory, your first argument (the object to be wrapped) needs to default
to None and be the only argument that is not keyword-only (i.e. all other arguments need to be keyword only).

```pycon
>>> @double_up_as_factory
... def decorator_2(func, *, multiplier=2):
...     '''Should not be able to be transformed with double_up_as_factory'''
Traceback (most recent call last):
  ...
AssertionError: First argument of the decorator function needs to default to None. Was <class 'inspect._empty'>
>>> @double_up_as_factory
... def decorator_3(func=None, multiplier=2):
...     '''Should not be able to be transformed with double_up_as_factory'''
Traceback (most recent call last):
  ...
AssertionError: All arguments (besides the first) need to be keyword-only
```

### dol.trans.ensure_clear_method(store=None, \*, clear_method=<function \_delete_keys_one_by_one>)

If obj doesn’t have an enabled clear method, will add one (a slow one that runs through keys and deletes them

### dol.trans.filt_iter(store=None, \*, filt=<function take_everything>, name=None, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Make a wrapper that will transform a store (class or instance thereof) into a sub-store (i.e. subset of keys).

* **Parameters:**
  * **filt** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)) – 

    A callable or iterable:
    ```default
    callable: Boolean filter function. A func taking a key and and returns True iff the key should be included.
    iterable: The collection of keys you want to filter "in"
    ```
  * **name** – The name to give the wrapped class
* **Returns:**
  A wrapper (that then needs to be applied to a store instance or class.

```pycon
>>> filtered_dict = filt_iter(filt=lambda k: (len(k) % 2) == 1)(dict)  # keep only odd length keys
>>>
>>> s = filtered_dict({'a': 1, 'bb': object, 'ccc': 'a string', 'dddd': [1, 2]})
>>>
>>> list(s)
['a', 'ccc']
>>> 'a' in s  # True because odd (length) key
True
>>> 'bb' in s  # False because odd (length) key
False
>>> assert s.get('bb', None) == None
>>> len(s)
2
>>> list(s.keys())
['a', 'ccc']
>>> list(s.values())
[1, 'a string']
>>> list(s.items())
[('a', 1), ('ccc', 'a string')]
>>> s.get('a')
1
>>> assert s.get('bb') is None
>>> s['x'] = 10
>>> list(s.items())
[('a', 1), ('ccc', 'a string'), ('x', 10)]
>>> try:
...     s['xx'] = 'not an odd key'
...     raise ValueError("This should have failed")
... except KeyError:
...     pass
```

### dol.trans.filter_prefixes(prefixes)

Make a filter that returns True if a string starts with one of the given prefixes

```pycon
>>> starts_with_test = filter_prefixes('test')
>>> starts_with_test("test.txt")
True
>>> starts_with_test("report.doc")
False
>>> is_test_or_report = filter_prefixes(['test', 'report'])
>>> is_test_or_report("test.txt")
True
>>> is_test_or_report("report.doc")
True
>>> is_test_or_report("image.jpg")
False
```

### dol.trans.filter_regex(regex, , return_search_func=False)

Make a filter that returns True if a string matches the given regex

```pycon
>>> is_txt = filter_regex(r'.*\.txt')
>>> is_txt("test.txt")
True
>>> is_txt("report.doc")
False
```

The argument is a *regular expression*, so it is compiled with `re.compile`
– NOT `safe_compile` (which is for file-path templates and `re.escape``s its
input on Windows). Using ``safe_compile` here silently broke every regex filter
on Windows: e.g. `filter_suffixes('.json')` (used by `Jsons`) had its
`(\.json)$` pattern escaped into a literal string matcher, so no `*.json` key
matched and the store raised `KeyError: 'Key not in store: <key>.json'`.

```pycon
>>> is_json = filter_regex(r"(\.json)$")  # works identically on every OS
>>> is_json("doc-001.json")
True
>>> is_json("doc-001.txt")
False
```

### dol.trans.filter_suffixes(suffixes)

Make a filter that returns True if a string ends with one of the given suffixes

```pycon
>>> ends_with_txt = filter_suffixes('.txt')
>>> ends_with_txt("test.txt")
True
>>> ends_with_txt("report.doc")
False
>>> is_text = filter_suffixes(['.txt', '.doc', '.pdf'])
>>> is_text("test.txt")
True
>>> is_text("report.doc")
True
>>> is_text("image.jpg")
False
```

### dol.trans.flatten(store=None, , levels=None, cache_keys=False, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Flatten a nested store.

Say you have a store that has three levels (or more), that is, that you can always
ask for the value `store[a][b][c]` if `a` is a valid key of `store`,
`b` is a valid key of `store[a]` and `c` is a valid key of `store[a][b]`.

What `flattened_store = flatten(store, levels=3)` will give you is the ability
to access the `store[a][b][c]` as `store[a, b, c]`, while still being able
to access these stores “normally”.

If that’s all you need, you can just use the `add_get_path` wrapper for this.

Why would you use `flatten`? Because `add_get_path(store)` would still only
give you the `KvReader` point of view of the root `store`.
If you `list(store)`, you’d only get the first level keys,
or if you ask if `(a, b, c)` is in the store, it will tell you it’s not
(though you can access data with such a key.

Instead, a flattened store will consider that the keys are those `(a, b, c)`
key paths.

Further, when flattening a store, you can ask for the view to cache the keys,
specifying `cache_keys=True` or give it an explicit place to cache or
factory to make a cache (see `cached_keys` wrapper for more details).
Though caching keys is not the default it’s highly recommended to do so in most
cases. The only reason it is not the default is because if you have millions of
keys, but little memory, that’s not what you might want.

#### NOTE
Flattening just provides a wrapper giving you a “flattened view”. It doesn’t
change the store itself, or it’s contents.

* **Parameters:**
  * **store** – The store instance or class to be wrapped
  * **levels** – The number of nested levels to flatten
  * **cache_keys** – Whether to cache the keys, or a cache factory or instance.

```pycon
>>> from dol import flatten
>>> d = {
...     'a': {'b': {'c': 42}},
...     'aa': {'bb': {'cc': 'dragon_con'}}
... }
```

You can get a flattened view of an instance:

```pycon
>>> m = flatten(d, levels=3, cache_keys=True)
>>> assert (
...         list(m.items())
...         == [
...             (('a', 'b', 'c'), 42),
...             (('aa', 'bb', 'cc'), 'dragon_con')
...         ]
... )
```

You can make a flattener and apply it to an instance (or a class):

```pycon
>>> my_flattener = flatten(levels=2)
>>> m = my_flattener(d)
>>> assert (
...         list(m.items())
...         == [
...             (('a', 'b'), {'c': 42}),
...             (('aa', 'bb'), {'cc': 'dragon_con'})
...         ]
... )
```

Finally, you can wrap a class itself.

```pycon
>>> @flatten(levels=1)
... class MyFlatDict(dict):
...     pass
>>> m = MyFlatDict(d)
>>> assert (
...         list(m.items())
...         == [
...             (('a',), {'b': {'c': 42}}),
...             (('aa',), {'bb': {'cc': 'dragon_con'}})
...         ]
... )
```

### dol.trans.insert_aliases(store=None, , write=None, read=None, delete=None, list=None, count=None, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Insert method aliases of CRUD operations of a store (class or instance).
If store is a class, you’ll get a copy of the class with those methods added.
If store is an instance, the methods will be added in place (no copy will be made).

#### NOTE
If an operation (write, read, delete, list, count) is not specified, no alias will be created for
that operation.

IMPORTANT NOTE: The signatures of the methods the aliases will point to will not change.
We say this because, you can call the write method “dump”, but you’ll have to use it as
`store.dump(key, val)`, not `store.dump(val, key)`, which is the signature you’re probably used to
(it’s the one used by json.dump or pickle.dump for example). If you want that familiar interface,
using the insert_load_dump_aliases function.

See also (and not to be confused with): `add_aliases`

* **Parameters:**
  * **store** – The store to extend with aliases.
  * **write** – Desired method name for \_\_setitem_\_
  * **read** – Desired method name for \_\_getitem_\_
  * **delete** – Desired method name for \_\_delitem_\_
  * **list** – Desired method name for \_\_iter_\_
  * **count** – Desired method name for \_\_len_\_
* **Returns:**
  A store with the desired aliases.

```pycon
>>> # Example of extending a class
>>> mydict = insert_aliases(dict, write='dump', read='load', delete='rm', list='peek', count='size')
>>> s = mydict(true='love')
>>> s.dump('friends', 'forever')
>>> s
{'true': 'love', 'friends': 'forever'}
>>> s.load('true')
'love'
>>> list(s.peek())
['true', 'friends']
>>> s.size()
2
>>> s.rm('true')
>>> s
{'friends': 'forever'}
>>>
>>> # Example of extending an instance
>>> from collections import UserDict
>>> s = UserDict(true='love')  # make (and instance) of a UserDict (can't modify a dict instance)
>>> # make aliases of note that you don't need
>>> s = insert_aliases(s, write='put', read='retrieve', count='num_of_items')
>>> s.put('friends', 'forever')
>>> s
{'true': 'love', 'friends': 'forever'}
>>> s.retrieve('true')
'love'
>>> s.num_of_items()
2
```

### dol.trans.insert_hash_method(store=None, \*, hash_method=<built-in function id>, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Make a store hashable using the specified `hash_method`.
Will add (or overwrite) a `__hash__` method to the store that uses the
hash_method to hash the store and an `__eq__` method that compares the store to
another based on the hash_method.

The `hash_method`, which will be used as the `__hash__` method of a class
should return an integer value that represents the hash of the object.
To remain sane, the hash value must be the same for an object every time the
`__hash__` method is called during the lifetime of the object, and objects
that compare equal (using the \_\_eq_\_ method) must have the same hash value.

It’s also important that the hash function has the property of being deterministic
and returning a hash value that is uniformly distributed across the range of
possible integers for the given data. This is important for the hash table
data structure to work efficiently.

See [This issue](https://github.com/i2mint/dol/issues/7) for further information.

```pycon
>>> d = {1: 2}  # not hashable!
>>> dd = insert_hash_method(d)
>>> assert isinstance(hash(dd), int)  # now hashable!
```

It looks the same:

```pycon
>>> dd
{1: 2}
```

But don’t be fooled: dd is not equal to the original `d` (since
insert_hash_method\`\`overwrote the `__eq__` method to compare based on the
hash value):

```pycon
>>> d == dd
False
```

But if you cast both to dicts and then compare, you’ll be using the key and value
based comparison of dicts, which makes these two equal.

```pycon
>>> dict(d) == dict(dd)
True
```

The default `hash_method` is `id`, so two hashable wrappers won’t be equal
to eachother:

```pycon
>>> insert_hash_method(d) == insert_hash_method(d)
False
```

In the following we show two things: That you can specify your own custom
`hash_method`, and that you can use `insert_hash_method` to wrap classes

```pycon
>>> class D(dict):
...     pass
>>> DD = insert_hash_method(D, hash_method=lambda x: 42)
>>> hash(DD(d))
42
```

You can also use it as a decorator, without arguments,

```pycon
>>> @insert_hash_method
... class E(dict):
...     pass
>>> assert isinstance(hash(E({1: 2})), int)
```

or with arguments (which you must specify as keyword arguments):

```pycon
>>> @insert_hash_method(hash_method=lambda x: sum(x.values()))
... class F(dict):
...     pass
>>> hash(F({1: 2, 3: 4}))
6
```

### dol.trans.insert_load_dump_aliases(store=None, , delete=None, list=None, count=None, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Insert load and dump methods, with familiar dump(obj, location) signature.

* **Parameters:**
  * **store** – The store to extend with aliases.
  * **delete** – Desired method name for \_\_delitem_\_
  * **list** – Desired method name for \_\_iter_\_
  * **count** – Desired method name for \_\_len_\_
* **Returns:**
  A store with the desired aliases.

```pycon
>>> mydict = insert_load_dump_aliases(dict)
>>> s = mydict()
>>> s.dump(obj='love', key='true')
>>> s
{'true': 'love'}
```

### dol.trans.kv_wrap(trans_obj)

A function that makes a wrapper (a decorator) that will get the wrappers from
methods of the input object.

* **Parameters:**
  **trans_obj** – An object that contains (as attributes) the collection of
  transformation functions. The attribute names that are used, natively, to make
  the wrapper are `_key_of_id`, `_id_of_key`, `_obj_of_data`,
  `_data_of_obj`, `_preset`, and `_postget`.

If your `trans_obj` uses different names for these functions, you can use the
`add_aliases` function. We’ll demo the use of `add_aliases` here:

```pycon
>>> from dol import kv_wrap, add_aliases, Pipe
>>> from functools import partial
>>>
>>> class SeparatorTrans:
...     def __init__(self, sep: str):
...         self.sep = sep
...     def string_to_tuple(self, string: str):
...         return tuple(string.split(self.sep))
...     def tuple_to_string(self, tup: tuple):
...         return self.sep.join(tup)
>>>
>>> _add_aliases = partial(
...     add_aliases, _key_of_id='string_to_tuple', _id_of_key='tuple_to_string'
... )
>>> mk_sep_trans = Pipe(SeparatorTrans, _add_aliases, kv_wrap)
>>> sep_trans = mk_sep_trans('/')
>>> d = sep_trans({'a/b/c': 1, 'd/e': 2})
>>> list(d)
[('a', 'b', 'c'), ('d', 'e')]
>>> d['d', 'e']
2
```

`kv_wrap` also has convenience attributes:

```default
``outcoming_keys``, ``ingoing_keys``, ``outcoming_vals``, ``ingoing_vals``,
and ``val_reads_wrt_to_keys``
```

which will only add a single specific wrapper (specified as a function),
when that’s what you need.

### dol.trans.kv_wrap_persister_cls(persister_cls, name=None)

Make a class that wraps a persister into a dol.base.Store,

* **Parameters:**
  **persister_cls** – The persister class to wrap
* **Returns:**
  A Store wrapping the persister (see dol.base)

```pycon
>>> A = kv_wrap_persister_cls(dict)
>>> a = A()
>>> a['one'] = 1
>>> a['two'] = 2
>>> a['three'] = 3
>>> list(a.items())
[('one', 1), ('two', 2), ('three', 3)]
>>> assert hasattr(a, '_obj_of_data')  # for example, it has this magic method
>>> # If you overwrite the _obj_of_data method, you'll transform outcomming values with it.
>>> # For example, say the data you stored were minutes, but you want to get then in secs...
>>> a._obj_of_data = lambda data: data * 60
>>> list(a.items())
[('one', 60), ('two', 120), ('three', 180)]
>>>
>>> # And if you want to have class that has this weird "store minutes, retrieve seconds", you can do this:
>>> class B(kv_wrap_persister_cls(dict)):
...     def _obj_of_data(self, data):
...         return data * 60
>>> b = B()
>>> b.update({'one': 1, 'two': 2, 'three': 3})  # you can write several key-value pairs at once this way!
>>> list(b.items())
[('one', 60), ('two', 120), ('three', 180)]
>>> # Warning! Advanced under-the-hood chat coming up.... Note this:
>>> print(b)
{'one': 1, 'two': 2, 'three': 3}
>>> # What?!? Well, remember, printing an object calls the objects __str__, which usually calls __repr__
>>> # The wrapper doesn't wrap those methods, since they don't have consistent behaviors.
>>> # Here you're getting the __repr__ of the underlying dict store, without the key and value transforms.
>>>
>>> # Say you wanted to transform the incoming minute-unit data, converting to secs BEFORE they were stored...
>>> class C(kv_wrap_persister_cls(dict)):
...     def _data_of_obj(self, obj):
...         return obj * 60
>>> c = C()
>>> c.update(one=1, two=2, three=3)  # yet another way you can write multiple key-vals at once
>>> list(c.items())
[('one', 60), ('two', 120), ('three', 180)]
>>> print(c)  # but notice that unlike when we printed b, here the stored data is actually transformed!
{'one': 60, 'two': 120, 'three': 180}
>>>
>>> # Now, just to demonstrate key transformation, let's say that we need internal (stored) keys to be upper case,
>>> # but external (the keys you see when listed) ones to be lower case, for some reason...
>>> class D(kv_wrap_persister_cls(dict)):
...     _data_of_obj = staticmethod(lambda obj: obj * 60)  # to demonstrated another way of doing this
...     _key_of_id = lambda self, _id: _id.lower()  # note if you don't specify staticmethod, 1st arg must be self
...     def _id_of_key(self, k):  # a function definition like you're used to
...         return k.upper()
>>> d = D()
>>> d['oNe'] = 1
>>> d.update(TwO=2, tHrEE=3)
>>> list(d.items())  # you see clean lower cased keys at the interface of the store
[('one', 60), ('two', 120), ('three', 180)]
>>> # but internally, the keys are all upper case
>>> print(d)  # equivalent to print(d.store), so keys and values not wrapped (values were transformed before stored)
{'ONE': 60, 'TWO': 120, 'THREE': 180}
>>>
>>> # On the other hand, careful, if you gave the data directly to D, you wouldn't get that.
>>> d = D({'one': 1, 'two': 2, 'three': 3})
>>> print(d)
{'one': 1, 'two': 2, 'three': 3}
>>> # Thus is because when you construct a D with the dict, it initializes the dicts data with it directly
>>> # before the key/val transformers are in place to do their jobs.
```

### dol.trans.mk_confirm_overwrite_preset(\*, get_input=None, prompt=<function \_dflt_overwrite_prompt>)

Make a `wrap_kvs` `preset` that asks for confirmation before overwriting an
existing key that holds a *different* value.

The returned preset writes `v` unchanged unless `k` already maps to a different
value; in that case it asks `get_input` to confirm (by typing the new value), and
keeps the existing value if the confirmation doesn’t match. Use `get_input` to
customize how confirmation is obtained (`None` -> the builtin `input`, resolved at
call time), e.g. for testing or for a non-interactive policy. See
[`confirm_overwrite()`](#dol.trans.confirm_overwrite).

```pycon
>>> store = dict(a='apple', b='banana')
>>> # simulate a user who always types the exact new value (confirms every overwrite)
>>> always_yes = wrap_kvs(
...     store, preset=mk_confirm_overwrite_preset(get_input=lambda prompt: 'alligator')
... )
>>> always_yes['a'] = 'alligator'   # confirmation matches -> overwrites
>>> store['a']
'alligator'
>>> # simulate a user who declines (types something that doesn't match)
>>> always_no = wrap_kvs(
...     store, preset=mk_confirm_overwrite_preset(get_input=lambda prompt: 'nope')
... )
>>> always_no['a'] = 'anteater'     # declined -> existing value kept
>>> store['a']
'alligator'
```

### dol.trans.mk_kv_reader_from_kv_collection(kv_collection, name=None, getitem=<function transparent_key_method>)

Make a KvReader class from a Collection class.

* **Parameters:**
  * **kv_collection** – The Collection class
  * **name** – The name to give the KvReader class (by default, it will be kv_collection._\_qualname_\_ + ‘Reader’)
  * **getitem** – The method that will be assigned to \_\_getitem_\_. Should have the (self, k) signature.
    By default, getitem will be transparent_key_method, returning the key as is.
    This default is useful when you want to delegate the actual getting to a \_obj_of_data wrapper.
* **Returns:**
  A KvReader class that subclasses the input kv_collection

### dol.trans.mk_level_walk_filt(levels)

Makes a `walk_filt` function for `kv_walk` based on some level logic.
If `levels` is an integer, will consider it as the max path length,
if not it will just assert that `levels` is callable, and return it

### dol.trans.mk_trans_obj(\*\*kwargs)

Convenience method to quickly make a trans_obj (just an object holding some trans functions

### dol.trans.mk_wrapper(wrap_cls)

You have a wrapper class and you want to make a wrapper out of it,
that is, a decorator factory with which you can make wrappers, like this:

```text
wrapper = mk_wrapper(wrap_cls)
```

that you can then use to transform stores like thiis:

```text
MyStore = wrapper(**wrapper_kwargs)(StoreYouWantToTransform)
```

* **Parameters:**
  **wrap_cls**
* **Returns:**

```pycon
>>> class RelPath:
...     def __init__(self, root):
...         self.root = root
...         self._root_length = len(root)
...     def _key_of_id(self, _id):
...         return _id[self._root_length:]
...     def _id_of_key(self, k):
...         return self.root + k
>>> relpath_wrap = mk_wrapper(RelPath)
>>> RelDict = relpath_wrap(root='foo/')(dict)
>>> s = RelDict()
>>> s['bar'] = 42
>>> assert list(s) == ['bar']
>>> assert s['bar'] == 42
>>> assert str(s) == "{'foo/bar': 42}"  # reveals that actually, behind the scenes, there's a "foo/" prefix
```

### dol.trans.redirect_getattr_to_getitem(cls=None, , keys_have_priority_over_attributes=False, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

A mapping decorator that redirects attribute access to \_\_getitem_\_.

#### WARNING
This decorator will make your class un-pickleable.

* **Parameters:**
  **keys_have_priority_over_attributes** – If True, keys will have priority over existing attributes.

```pycon
>>> @redirect_getattr_to_getitem
... class MyDict(dict):
...     pass
>>> d = MyDict(a=1, b=2)
>>> d.a
1
>>> d.b
2
>>> list(d)
['a', 'b']
```

### dol.trans.store_decorator(func)

Helper to make store decorators.

You provide a class-decorating function `func` that takes a store type (and possibly additional params)
and returns another decorated store type.

`store_decorator` takes that `func` and provides an enhanced class decorator specialized for stores.
Namely it will:

- Add `__module__`, `__qualname__`, `__name__` and `__doc__` arguments to it
- Copy the aforementioned arguments to the decorated class, or copy the attributes of the original if not specified.
- Output a decorator that can be used in four different ways: a class/instance decorator/factory.

By class/instance decorator/factory we mean that if `A` is a class, `a` an instance of it,
and `deco` a decorator obtained with `store_decorator(func)`,
we can use `deco` to

- class decorator: decorate a class
- class decorator factory: make a function that decorates classes
- instance decorator: decorate an instance of a store
- instancce decorator factor: make a function that decorates instances of stores

For example, say we have the following `deco` that we made with `store_decorator`:

```pycon
>>> @store_decorator
... def deco(cls=None, *, x=1):
...     # do stuff to cls, or a copy of it...
...     cls.x = x  # like this for example
...     return cls
```

And a class that has nothing to it:

```pycon
>>> class A: ...
```

Nammely, it doesn’t have an `x`

```pycon
>>> hasattr(A, 'x')
False
```

We make a `decorated_A` with `deco` (class decorator example)

```pycon
>>> t = deco(A, x=42)
>>> assert isinstance(t, type)
```

and we see that we now have an `x` and it’s 42

```pycon
>>> hasattr(A, 'x')
True
>>> A.x
42
```

But we could have also made a factory to decorate `A` and anything else that comes our way.

```pycon
>>> paint_it_42 = deco(x=42)
>>> decorated_A = paint_it_42(A)
>>> assert decorated_A.x == 42
>>> class B:
...     x = 'destined to disappear'
>>> assert paint_it_42(B).x == 42
```

To be fair though, you’ll probably see the factory usage appear in the following form,
where the class is decorated at definition time.

```pycon
>>> @deco(x=42)
... class B:
...     pass
>>> assert B.x == 42
```

If your exists already, and you want to keep it as is (with the same name), you can
use subclassing to transform a copy of `A` instead, as below.
Also note in the following example, that `deco` was used without parentheses,
which is equivalent to `@deco()`,
and yes, store_decorator makes that possible to, as long as your params have defaults

```pycon
>>> @deco
... class decorated_A(A):
...     pass
>>> assert decorated_A.x == 1
>>> assert A.x == 42
```

Finally, you can also decorate instances:

```pycon
>>> class A: ...
>>> a = A()
>>> hasattr(a, 'x')
False
>>> b = deco(a); assert b.x == 1; # b has an x and it's 1
>>> b = deco()(a); assert b.x == 1; # b has an x and it's 1
>>> b = deco(a, x=42); assert b.x == 42  # b has an x and it's 42
>>> b = deco(x=42)(a); assert b.x == 42; # b has an x and it's 42
```

#### WARNING
Note though that the type of `b` is not the same type as `a`

```pycon
>>> isinstance(b, a.__class__)
False
```

No, `b` is an instance of a `dol.base.Store`, which is a class containing an
instance of a store (here, `a`).

```pycon
>>> type(b)
<class 'dol.base.Store'>
>>> b.store == a
True
```

Now, here’s some more example, slightly closer to real usage

```pycon
>>> from dol.trans import store_decorator
>>> from inspect import signature
>>>
>>> def rm_deletion(store=None, *, msg='Deletions not allowed.'):
...     name = getattr(store, '__name__', 'Something') + '_w_sommething'
...     assert isinstance(store, type), f"Should be a type, was {type(store)}: {store}"
...     wrapped_store = type(name, (store,), {})
...     wrapped_store.__delitem__ = lambda self, k: msg
...     return wrapped_store
...
>>> remove_deletion = store_decorator(rm_deletion)
```

See how the signature of the wrapper has some extra inputs that were injected (_\_module_\_, \_\_qualname_\_, etc.):

```pycon
>>> print(str(signature(remove_deletion)))
(store=None, *, msg='Deletions not allowed.', __module__=None, __name__=None, __qualname__=None, __doc__=None, __annotations__=None, __defaults__=None, __kwdefaults__=None)
```

Using it as a class decorator factory (the most common way):

As a class decorator “factory”, without parameters (and without ()):

```pycon
>>> from collections import UserDict
>>> @remove_deletion
... class WD(UserDict):
...     "Here's the doc"
...     pass
>>> wd = WD(x=5, y=7)
>>> assert wd == UserDict(x=5, y=7)  # same as far as dict comparison goes
>>> assert wd.__delitem__('x') == 'Deletions not allowed.'
>>> assert wd.__doc__ == "Here's the doc"
```

As a class decorator “factory”, with parameters:

```pycon
>>> @remove_deletion(msg='No way. I do not trust you!!')
... class WD(UserDict): ...
>>> wd = WD(x=5, y=7)
>>> assert wd == UserDict(x=5, y=7)  # same as far as dict comparison goes
>>> assert wd.__delitem__('x') == 'No way. I do not trust you!!'
```

The \_\_doc_\_ is empty:

```pycon
>>> assert WD.__doc__ == None
```

But we could specify a doc if we wanted to:

```pycon
>>> @remove_deletion(__doc__="Hi, I'm a doc.")
... class WD(UserDict):
...     "This is the original doc, that will be overritten"
>>> assert WD.__doc__ == "Hi, I'm a doc."
```

The class decorations above are equivalent to the two following:

```pycon
>>> WD = remove_deletion(UserDict)
>>> wd = WD(x=5, y=7)
>>> assert wd == UserDict(x=5, y=7)  # same as far as dict comparison goes
>>> assert wd.__delitem__('x') == 'Deletions not allowed.'
>>>
>>> WD = remove_deletion(UserDict, msg='No way. I do not trust you!!')
>>> wd = WD(x=5, y=7)
>>> assert wd == UserDict(x=5, y=7)  # same as far as dict comparison goes
>>> assert wd.__delitem__('x') == 'No way. I do not trust you!!'
```

But we can also decorate instances. In this case they will be wrapped in a Store class
before being passed on to the actual decorator.

```pycon
>>> d = UserDict(x=5, y=7)
>>> wd = remove_deletion(d)
>>> assert wd == d  # same as far as dict comparison goes
>>> assert wd.__delitem__('x') == 'Deletions not allowed.'
>>>
>>> d = UserDict(x=5, y=7)
>>> wd = remove_deletion(d, msg='No way. I do not trust you!!')
>>> assert wd == d  # same as far as dict comparison goes
>>> assert wd.__delitem__('x') == 'No way. I do not trust you!!'
```

### dol.trans.wrap_kvs(store=None, , wrapper=None, name=None, key_of_id=None, id_of_key=None, obj_of_data=None, data_of_obj=None, preset=None, postget=None, key_codec=None, value_codec=None, key_encoder=None, key_decoder=None, value_encoder=None, value_decoder=None, \_\_module_\_=None, outcoming_key_methods=(), outcoming_value_methods=(), ingoing_key_methods=(), ingoing_value_methods=(), \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Make a Store that is wrapped with the given key/val transformers.

Naming convention:

```default
Morphemes:
    key: outer key
    _id: inner key
    obj: outer value
    data: inner value
Grammar:
    Y_of_X: means that you get a Y output when giving an X input. Also known as X_to_Y.
```

* **Parameters:**
  * **store** – Store class or instance
  * **name** – Name to give the wrapper class
  * **key_of_id** – The outcoming key transformation function.
    Forms are `k = key_of_id(_id)` or `k = key_of_id(self, _id)`
  * **id_of_key** – The ingoing key transformation function.
    Forms are `_id = id_of_key(k)` or `_id = id_of_key(self, k)`
  * **obj_of_data** – The outcoming val transformation function.
    Forms are `obj = obj_of_data(data)` or `obj = obj_of_data(self, data)`
  * **data_of_obj** – The ingoing val transformation function.
    Forms are `data = data_of_obj(obj)` or `data = data_of_obj(self, obj)`
  * **preset** – 

    A function that is called before doing a `__setitem__`.
    The function is called with both `k` and `v` as inputs, and should output a transformed value.
    The intent use is to do ingoing value transformations conditioned on the key.
    For example, you may want to serialize an object depending on if you’re writing to a
    > ’.csv’, or ‘.json’, or ‘.pickle’ file.

    Forms are `preset(k, obj)` or `preset(self, k, obj)`
  * **postget** – A function that is called after the value `v` for a key `k` is be `__getitem__`.
    The function is called with both `k` and `v` as inputs, and should output a transformed value.
    The intent use is to do outcoming value transformations conditioned on the key.
    We already have `obj_of_data` for outcoming value trans, but cannot condition it’s behavior on k.
    For example, you may want to deserialize the bytes of a ‘.csv’, or ‘.json’, or ‘.pickle’ in different ways.
    Forms are `obj = postget(k, data)` or `obj = postget(self, k, data)`
* **Returns:**
  A key and/or value transformed wrapped (or wrapper) class (or instance).

```pycon
>>> def key_of_id(_id):
...     return _id.upper()
>>> def id_of_key(k):
...     return k.lower()
>>> def obj_of_data(data):
...     return data - 100
>>> def data_of_obj(obj):
...     return obj + 100
>>>
>>> A = wrap_kvs(dict, name='A',
...             key_of_id=key_of_id, id_of_key=id_of_key, obj_of_data=obj_of_data, data_of_obj=data_of_obj)
>>> a = A()
>>> a['KEY'] = 1
>>> a  # repr is just the base class (dict) repr, so shows "inside" the store (lower case keys and +100)
{'key': 101}
>>> a['key'] = 2
>>> print(a)  # repr is just the base class (dict) repr, so shows "inside" the store (lower case keys and +100)
{'key': 102}
>>> a['kEy'] = 3
>>> a  # repr is just the base class (dict) repr, so shows "inside" the store (lower case keys and +100)
{'key': 103}
>>> list(a)  # but from the point of view of the interface the keys are all upper case
['KEY']
>>> list(a.items())  # and the values are those we put there.
[('KEY', 3)]
>>>
>>> # And now this: Showing how to condition the value transform (like obj_of_data), but conditioned on key.
>>> B = wrap_kvs(dict, name='B', postget=lambda k, v: f'upper {v}' if k[0].isupper() else f'lower {v}')
>>> b = B()
>>> b['BIG'] = 'letters'
>>> b['small'] = 'text'
>>> list(b.items())
[('BIG', 'upper letters'), ('small', 'lower text')]
>>>
>>>
>>> # Let's try preset and postget. We'll wrap a dict and write the same list of lists object to
>>> # keys ending with .csv, .json, and .pkl, specifying the obvious extension-dependent
>>> # serialization/deserialization we want to associate with it.
>>>
>>> # First, some very simple csv transformation functions
>>> to_csv = lambda LoL: '\\n'.join(map(','.join, map(lambda L: (x for x in L), LoL)))
>>> from_csv = lambda csv: list(map(lambda x: x.split(','), csv.split('\\n')))
>>> LoL = [['a','b','c'],['d','e','f']]
>>> assert from_csv(to_csv(LoL)) == LoL
>>>
>>> import json, pickle
>>>
>>> def preset(k, v):
...     if k.endswith('.csv'):
...         return to_csv(v)
...     elif k.endswith('.json'):
...         return json.dumps(v)
...     elif k.endswith('.pkl'):
...         return pickle.dumps(v)
...     else:
...         return v  # as is
...
...
>>> def postget(k, v):
...     if k.endswith('.csv'):
...         return from_csv(v)
...     elif k.endswith('.json'):
...         return json.loads(v)
...     elif k.endswith('.pkl'):
...         return pickle.loads(v)
...     else:
...         return v  # as is
...
>>> mydict = wrap_kvs(dict, preset=preset, postget=postget)
>>>
>>> obj = [['a','b','c'],['d','e','f']]
>>> d = mydict()
>>> d['foo.csv'] = obj  # store the object as csv
>>> d  # "printing" a dict by-passes the transformations, so we see the data in the "raw" format it is stored in.
{'foo.csv': 'a,b,c\\nd,e,f'}
>>> d['foo.csv']  # but if we actually ask for the data, it deserializes to our original object
[['a', 'b', 'c'], ['d', 'e', 'f']]
>>> d['bar.json'] = obj  # store the object as json
>>> d
{'foo.csv': 'a,b,c\\nd,e,f', 'bar.json': '[["a", "b", "c"], ["d", "e", "f"]]'}
>>> d['bar.json']
[['a', 'b', 'c'], ['d', 'e', 'f']]
>>> d['bar.json'] = {'a': 1, 'b': [1, 2], 'c': 'normal json'}  # let's write a normal json instead.
>>> d
{'foo.csv': 'a,b,c\\nd,e,f', 'bar.json': '{"a": 1, "b": [1, 2], "c": "normal json"}'}
>>> del d['foo.csv']
>>> del d['bar.json']
>>> d['foo.pkl'] = obj  # 'save' obj as pickle
>>> d['foo.pkl']
[['a', 'b', 'c'], ['d', 'e', 'f']]
```

### TODO: Add tests for outcoming_key_methods etc.
