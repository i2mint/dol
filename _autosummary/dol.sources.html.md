# dol.sources

Key-value views of disparate sources.

Readers and persisters over things that are not stores to begin with: several stores
at once (fan-out and cascades), sequences, functions, and the attributes of objects.

Main entry points:

- `FanoutReader`, `FanoutPersister`: one key, read from (written to) several stores
- `CascadedStores`: write to all stores, read from the first one that has the key
- `SequenceKvReader`: an iterable of elements, keyed by a key function (index by default)
- `FuncReader`: functions as a store, keyed by name
- `Attrs`: the attributes of an object as a (recursive) reader
  ```pycon
  >>> from dol.sources import FuncReader
  >>> def foo():
  ...     return 'bar'
  >>> r = FuncReader([foo])
  >>> list(r), r['foo']
  (['foo'], 'bar')
  ```

### Functions

| `ddir`(o)                       |    |
|---------------------------------|----|
| `exclusive_subdict`(d, exclude) |    |
| `identity_func`(x)              |    |
| `inclusive_subdict`(d, include) |    |
| `not_underscore_prefixed`(x)    |    |
| `unique_element`(iterator)      |    |

### Classes

| [`AttrContainer`](#dol.sources.AttrContainer)(\*objects[, \_object_namer])        | Convenience class to hold Key-Val pairs as attribute-val pairs, with all the magic methods of mappings.                                                                                                                               |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`AttrDict`](#dol.sources.AttrDict)(\*objects[, \_object_namer])             | Convenience class to hold Key-Val pairs with both a dict-like and struct-like interface.                                                                                                                                              |
| [`Attrs`](#dol.sources.Attrs)(obj[, key_filt, getattrs])                  | A simple recursive KvReader for the attributes of a python object.                                                                                                                                                                    |
| [`CachedKeysSequenceKvReader`](#dol.sources.CachedKeysSequenceKvReader)(sequence[, key, ...])  | SequenceKvReader but with keys cached.                                                                                                                                                                                                |
| [`CachedSequenceKvReader`](#dol.sources.CachedSequenceKvReader)(sequence[, key, val, ...]) | SequenceKvReader but with the whole mapping cached as a dict.                                                                                                                                                                         |
| [`CascadedStores`](#dol.sources.CascadedStores)(stores[, default, ...])            | A MutableMapping interface to a collection of stores that will write a value in all the stores it contains, read it from the first store it finds that has it, and write it back to all the stores up to the store where it found it. |
| [`Ddir`](#dol.sources.Ddir)                                              |                                                                                                                                                                                                                                       |
| [`FanoutPersister`](#dol.sources.FanoutPersister)(stores[, default, ...])           | A fanout persister is a fanout reader that can also set and delete items.                                                                                                                                                             |
| [`FanoutReader`](#dol.sources.FanoutReader)(stores[, default, ...])              | Get a 'fanout view' of a store of stores.                                                                                                                                                                                             |
| [`FlatReader`](#dol.sources.FlatReader)(readers)                               | Get a 'flat view' of a store of stores.                                                                                                                                                                                               |
| [`FuncDag`](#dol.sources.FuncDag)(funcs, \*\*kwargs)                        |                                                                                                                                                                                                                                       |
| [`FuncReader`](#dol.sources.FuncReader)(funcs)                                 | Reader that seeds itself from a data fetching function list Uses the function list names as the keys, and their returned value as the values.                                                                                         |
| [`MultiSource`](#dol.sources.MultiSource)(\*sources)                            | A read-only Mapping that composes multiple sources, tried in order.                                                                                                                                                                   |
| `ObjLoader`(data_of_key[, obj_of_data])                                                            |                                                                                                                                                                                                                                       |
| [`ObjReader`](#dol.sources.ObjReader)(_obj_of_key)                            | A reader that uses a specified function to get the contents for a given key.                                                                                                                                                          |
| [`SequenceKvReader`](#dol.sources.SequenceKvReader)(sequence[, key, val, ...])       | A KvReader that sources itself in an iterable of elements from which keys and values will be extracted and grouped by key.                                                                                                            |

### Exceptions

| [`NotUnique`](#dol.sources.NotUnique)   | Raised when an iterator was expected to have only one element, but had more   |
|--------------------------------------------------------------|-------------------------------------------------------------------------------|

### *class* dol.sources.AttrContainer(\*objects, \_object_namer=<function \_dflt_object_namer>, \*\*named_objects)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Convenience class to hold Key-Val pairs as attribute-val pairs, with all the
magic methods of mappings.

On the other hand, you will not get the usuall non-dunders (non magic methods) of
`Mappings`. This is so that you can use tab completion to access only the keys
the container has, and not any of the non-dunder methods like `get`, `items`,
etc.

```pycon
>>> da = AttrContainer(foo='bar', life=42)
>>> da.foo
'bar'
>>> da['life']
42
>>> da.true = 'love'
>>> len(da)  # count the number of fields
3
>>> da['friends'] = 'forever'  # write as dict
>>> da.friends  # read as attribute
'forever'
>>> list(da)  # list fields (i.e. keys i.e. attributes)
['foo', 'life', 'true', 'friends']
>>> 'life' in da  # check containement
True
```

```pycon
>>> del da['friends']  # delete as dict
>>> del da.foo # delete as attribute
>>> list(da)
['life', 'true']
>>> da._source  # the hidden Mapping (here dict) that is wrapped
{'life': 42, 'true': 'love'}
```

If you don’t specify a name for some objects, `AttrContainer` will use the
`__name__` attribute of the objects:

```pycon
>>> d = AttrContainer(map, tuple, obj='objects')
>>> list(d)
['map', 'tuple', 'obj']
```

You can also specify a different way of auto naming the objects:

```pycon
>>> d = AttrContainer('an', 'example', _object_namer=lambda x: f"_{len(x)}")
>>> {k: getattr(d, k) for k in d}
{'_2': 'an', '_7': 'example'}
```

#### SEE ALSO
Objects in `py2store.utils.attr_dict` module

### *class* dol.sources.AttrDict(\*objects, \_object_namer=<function \_dflt_object_namer>, \*\*named_objects)

Bases: [`AttrContainer`](#dol.sources.AttrContainer), [`KvPersister`](dol.base.html.md#dol.base.KvPersister)

Convenience class to hold Key-Val pairs with both a dict-like and struct-like
interface.

The dict-like interface has just the basic get/set/del/iter/len
(all “dunders”: none visible as methods). There is no get, update, etc.
This is on purpose, so that the only visible attributes
(those you get by tab-completion for instance) are the those you injected.

```pycon
>>> da = AttrDict(foo='bar', life=42)
```

You get the “keys as attributes” that you get with `AttrContainer`:

```pycon
>>> da.foo
'bar'
```

But additionally, you get the extra `Mapping` methods:

```pycon
>>> list(da.keys())
['foo', 'life']
>>> list(da.values())
['bar', 42]
>>> da.get('foo')
'bar'
>>> da.get('not_a_key', 'default')
'default'
```

You can assign through key or attribute assignment:

```pycon
>>> da['true'] = 'love'
>>> da.friends = 'forever'
>>> list(da.items())
[('foo', 'bar'), ('life', 42), ('true', 'love'), ('friends', 'forever')]
```

etc.

#### SEE ALSO
Objects in `py2store.utils.attr_dict` module

### *class* dol.sources.Attrs(obj, key_filt=<function not_underscore_prefixed>, getattrs=<built-in function dir>)

Bases: [`Store`](dol.base.html.md#dol.base.Store)

A simple recursive KvReader for the attributes of a python object.
Keys are attr names, values are Attrs(attr_val) instances.

#### NOTE
A more significant version of Attrs, along with many tools based on it,
was moved to pypi package: guide.

pip install guide

#### update(\*\*F) → None.  Update D from mapping/iterable E and F.

If E present and has a .keys() method, does:     for k in E.keys(): D[k] = E[k]
If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v
In either case, this is followed by: for k, v in F.items(): D[k] = v

#### update_keys_cache(keys)

Updates the \_keys_cache by calling its {} method

### *class* dol.sources.CachedKeysSequenceKvReader(sequence, key=None, val=None, val_postproc=<class 'list'>)

Bases: [`Store`](dol.base.html.md#dol.base.Store)

SequenceKvReader but with keys cached. Use this one if you will perform multiple
accesses to only some of the keys of the store

#### update(\*\*F) → None.  Update D from mapping/iterable E and F.

If E present and has a .keys() method, does:     for k in E.keys(): D[k] = E[k]
If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v
In either case, this is followed by: for k, v in F.items(): D[k] = v

#### update_keys_cache(keys)

Updates the \_keys_cache by deleting the attribute

### *class* dol.sources.CachedSequenceKvReader(sequence, key=None, val=None, val_postproc=<class 'list'>)

Bases: [`CachedSequenceKvReader`](#dol.sources.CachedSequenceKvReader)

SequenceKvReader but with the whole mapping cached as a dict. Use this one if
you will perform multiple accesses to the store

### *class* dol.sources.CascadedStores(stores, default=None, , get_existing_values_only=False, need_to_set_all_stores=False, ignore_non_existing_store_keys=False, \*\*kwargs)

Bases: [`Store`](dol.base.html.md#dol.base.Store)

A MutableMapping interface to a collection of stores that will write a value in
all the stores it contains, read it from the first store it finds that has it, and
write it back to all the stores up to the store where it found it.

This is useful, for example, when you want to, say, write something to disk,
and possibly to a remote backup or shared store, but also keep that value in memory.

The name `CascadedStores` comes from “Cascaded Caches”, which is a common pattern in
caching systems
(e.g. [https://philipwalton.com/articles/cascading-cache-invalidation/](https://philipwalton.com/articles/cascading-cache-invalidation/))

To demo this, let’s create a couple of stores that print when they get a value:

```pycon
>>> from collections import UserDict
>>> class LoggedDict(UserDict):
...     def __init__(self, name: str):
...        self.name = name
...        super().__init__()
...     def __getitem__(self, k):
...         print(f"Getting {k} from {self.name}")
...         return super().__getitem__(k)
>>> cache = LoggedDict('cache')
>>> disk = LoggedDict('disk')
>>> remote = LoggedDict('remote')
```

Now we can create a CascadedStores instance with these stores and write a
value to it:

```pycon
>>> stores = CascadedStores([cache, disk, remote])
>>> stores['f'] = 42
```

See that it’s in both stores:

```pycon
>>> cache['f']
Getting f from cache
42
>>> disk['f']
Getting f from disk
42
>>> remote['f']
Getting f from remote
42
```

See how it reads from the first store only, because it found the `f` key there:

```pycon
>>> stores['f']
Getting f from cache
42
```

Let’s write something in disk only:

```pycon
>>> disk['g'] = 43
```

Now if you ask for `g`, it won’t find it in cache, but will find it in `disk`
and return it.

```pycon
>>> stores['g']
Getting g from disk
43
```

Here’s the thing though. Now, `g` is also in `cache`:

```pycon
>>> cache
{'f': 42, 'g': 43}
```

But `remote` still only has `f`:

```pycon
>>> remote
{'f': 42}
```

#### from_variadics(\*args, \*\*kwargs)

A way to create a fan-out store from a mix of args and kwargs, instead of a
single dict.

* **Parameters:**
  * **args** – sub-stores used to fan-out the data. These stores will be
    represented by their index in the tuple.
  * **kwargs** – sub-stores used to fan-out the data. These stores will be
    represented by their name in the dict. \_\_init_\_ arguments can also be passed
    as kwargs (i.e. `default`, `get_existing_values_only`, and any other subclass
    specific arguments).

Let’s use the same sub-stores:

```pycon
>>> bytes_store = dict(
...     a=b'a',
...     b=b'b',
...     c=b'c',
... )
>>> metadata_store = dict(
...     b=dict(x=2),
...     c=dict(x=3),
...     d=dict(x=4),
... )
```

We can create a fan-out reader from these stores, using args:

```pycon
>>> reader = FanoutReader.from_variadics(bytes_store, metadata_store)
>>> reader['b']
{0: b'b', 1: {'x': 2}}
```

The reader returns a dict with the values from each store, keyed by the index of
the store in the `args` tuple.

We can also create a fan-out reader passing the stores in kwargs:

```pycon
>>> reader = FanoutReader.from_variadics(
...     bytes_store=bytes_store,
...     metadata_store=metadata_store
... )
>>> reader['b']
{'bytes_store': b'b', 'metadata_store': {'x': 2}}
```

This way, the returned value is keyed by the name of the store.

We can also mix args and kwargs:

```pycon
>>> reader = FanoutReader.from_variadics(bytes_store, metadata_store=metadata_store)
>>> reader['b']
{0: b'b', 'metadata_store': {'x': 2}}
```

Note that the order of the stores is determined by the order of the args and
kwargs.

### dol.sources.Ddir

alias of [`Attrs`](#dol.sources.Attrs)

### *class* dol.sources.FanoutPersister(stores, default=None, , get_existing_values_only=False, need_to_set_all_stores=False, ignore_non_existing_store_keys=False, \*\*kwargs)

Bases: [`FanoutReader`](#dol.sources.FanoutReader), [`KvPersister`](dol.base.html.md#dol.base.KvPersister)

A fanout persister is a fanout reader that can also set and delete items.

* **Parameters:**
  * **stores** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]) – A mapping of store keys to stores.
  * **default** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The value to return if the key is not in any of the stores.
  * **get_existing_values_only** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, only return values for stores that contain
    the key.
  * **need_to_set_all_stores** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, all stores must be set when setting a value.
    If False, only the stores that are set will be updated.
  * **ignore_non_existing_store_keys** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, ignore store keys from the value that
    are not in the persister. If False, a ValueError is raised.

Let’s create a persister from in-memory stores:

```pycon
>>> bytes_store = dict()
>>> metadata_store = dict()
>>> persister = FanoutPersister(
...     stores = dict(bytes_store=bytes_store, metadata_store=metadata_store)
... )
```

The persister sets the values in each store, based on the store key in the value dict.

```pycon
>>> persister['a'] = dict(bytes_store=b'a', metadata_store=dict(x=1))
>>> persister['a']
{'bytes_store': b'a', 'metadata_store': {'x': 1}}
```

By default, not all stores must be set when setting a value:

```pycon
>>> persister['b'] = dict(bytes_store=b'b')
>>> persister['b']
{'bytes_store': b'b', 'metadata_store': None}
```

This allow to update a subset of the stores whithout having to set all the stores.

```pycon
>>> persister['a'] = dict(bytes_store=b'A')
>>> persister['a']
{'bytes_store': b'A', 'metadata_store': {'x': 1}}
```

This behavior can be changed by passing `need_to_set_all_stores=True`:

```pycon
>>> persister_all_stores = FanoutPersister(
...     stores=dict(bytes_store=dict(), metadata_store=dict()),
...     need_to_set_all_stores=True,
... )
>>> persister_all_stores['a'] = dict(bytes_store=b'a')
Traceback (most recent call last):
    ...
ValueError: All stores must be set when setting a value. Missing stores: {'metadata_store'}
```

By default, if a store key from the value is not in the persister, a ValueError is
raised:

```pycon
>>> persister['a'] = dict(
...     bytes_store=b'a', metadata_store=dict(y=1), other_store='some value'
... )
Traceback (most recent call last):
    ...
ValueError: The value contains some invalid store keys: {'other_store'}
```

This behavior can be changed by passing `ignore_non_existing_store_keys=True`:

```pycon
>>> persister_ignore_non_existing_store_keys = FanoutPersister(
...     stores=dict(bytes_store=dict(), metadata_store=dict()),
...     ignore_non_existing_store_keys=True,
... )
>>> persister_ignore_non_existing_store_keys['a'] = dict(
...     bytes_store=b'a', metadata_store=dict(y=1), other_store='some value'
... )
>>> persister_ignore_non_existing_store_keys['a']
{'bytes_store': b'a', 'metadata_store': {'y': 1}}
```

Note that the value of the non-existing store key is ignored! So, be careful when
using this option, to avoid losing data.

Let’s delete items now:

```pycon
>>> del persister['a']
>>> 'a' in persister
False
```

The key as been deleted from all the stores:

```pycon
>>> 'a' in bytes_store
False
>>> 'a' in metadata_store
False
```

As expected, if the key is not in any of the stores, a KeyError is raised:

```pycon
>>> del persister['z']
Traceback (most recent call last):
    ...
KeyError: 'z'
```

However, if the key is in some of the stores, but not in others, the key is deleted
from the stores where it is present:

```pycon
>>> bytes_store=dict(a=b'a')
>>> persister = FanoutPersister(
...     stores=dict(bytes_store=bytes_store, metadata_store=dict()),
... )
>>> del persister['a']
>>> 'a' in persister
False
>>> 'a' in bytes_store
False
```

### *class* dol.sources.FanoutReader(stores, default=None, , get_existing_values_only=False)

Bases: [`KvReader`](dol.base.html.md#dol.base.KvReader)

Get a ‘fanout view’ of a store of stores.
That is, when a key is requested, the key is passed to all the stores, and results
accumulated in a dict that is then returned.

* **Parameters:**
  * **stores** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]) – A mapping of store keys to stores.
  * **default** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The value to return if the key is not in any of the stores.
  * **get_existing_values_only** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, only return values for stores that contain
    the key.

Let’s define the following sub-stores:

```pycon
>>> bytes_store = dict(
...     a=b'a',
...     b=b'b',
...     c=b'c',
... )
>>> metadata_store = dict(
...     b=dict(x=2),
...     c=dict(x=3),
...     d=dict(x=4),
... )
```

We can create a fan-out reader from these stores:

```pycon
>>> stores = dict(bytes_store=bytes_store, metadata_store=metadata_store)
>>> reader = FanoutReader(stores)
>>> reader['b']
{'bytes_store': b'b', 'metadata_store': {'x': 2}}
```

The reader returns a dict with the values from each store, keyed by the name of the
store.

We can also pass a default value to return if the key is not in the store:

```pycon
>>> reader = FanoutReader(
...     stores=stores,
...     default='no value in this store for this key',
... )
>>> reader['a']
{'bytes_store': b'a', 'metadata_store': 'no value in this store for this key'}
```

If the key is not in any of the stores, a KeyError is raised:

```pycon
>>> reader['z']
Traceback (most recent call last):
    ...
KeyError: 'z'
```

We can also pass `get_existing_values_only=True` to only return values for stores
that contain the key:

```pycon
>>> reader = FanoutReader(
...     stores=stores,
...     get_existing_values_only=True,
... )
>>> reader['a']
{'bytes_store': b'a'}
```

#### *classmethod* from_variadics(\*args, \*\*kwargs)

A way to create a fan-out store from a mix of args and kwargs, instead of a
single dict.

* **Parameters:**
  * **args** – sub-stores used to fan-out the data. These stores will be
    represented by their index in the tuple.
  * **kwargs** – sub-stores used to fan-out the data. These stores will be
    represented by their name in the dict. \_\_init_\_ arguments can also be passed
    as kwargs (i.e. `default`, `get_existing_values_only`, and any other subclass
    specific arguments).

Let’s use the same sub-stores:

```pycon
>>> bytes_store = dict(
...     a=b'a',
...     b=b'b',
...     c=b'c',
... )
>>> metadata_store = dict(
...     b=dict(x=2),
...     c=dict(x=3),
...     d=dict(x=4),
... )
```

We can create a fan-out reader from these stores, using args:

```pycon
>>> reader = FanoutReader.from_variadics(bytes_store, metadata_store)
>>> reader['b']
{0: b'b', 1: {'x': 2}}
```

The reader returns a dict with the values from each store, keyed by the index of
the store in the `args` tuple.

We can also create a fan-out reader passing the stores in kwargs:

```pycon
>>> reader = FanoutReader.from_variadics(
...     bytes_store=bytes_store,
...     metadata_store=metadata_store
... )
>>> reader['b']
{'bytes_store': b'b', 'metadata_store': {'x': 2}}
```

This way, the returned value is keyed by the name of the store.

We can also mix args and kwargs:

```pycon
>>> reader = FanoutReader.from_variadics(bytes_store, metadata_store=metadata_store)
>>> reader['b']
{0: b'b', 'metadata_store': {'x': 2}}
```

Note that the order of the stores is determined by the order of the args and
kwargs.

### *class* dol.sources.FlatReader(readers)

Bases: [`KvReader`](dol.base.html.md#dol.base.KvReader)

Get a ‘flat view’ of a store of stores.
That is, where keys are `(first_level_key, second_level_key)` pairs.
This is useful, for instance, to make a union of stores (you’ll get all the values).

```pycon
>>> readers = {
...     'fr': {1: 'un', 2: 'deux'},
...     'it': {1: 'uno', 2: 'due', 3: 'tre'},
... }
>>> s = FlatReader(readers)
>>> list(s)
[('fr', 1), ('fr', 2), ('it', 1), ('it', 2), ('it', 3)]
>>> s[('fr', 1)]
'un'
>>> s['it', 2]
'due'
```

### *class* dol.sources.FuncDag(funcs, \*\*kwargs)

Bases: [`FuncReader`](#dol.sources.FuncReader)

### *class* dol.sources.FuncReader(funcs)

Bases: [`KvReader`](dol.base.html.md#dol.base.KvReader)

Reader that seeds itself from a data fetching function list
Uses the function list names as the keys, and their returned value as the values.

For example: You have a list of urls that contain the data you want to have access
to.
You can write functions that bare the names you want to give to each dataset,
and have the function fetch the data from the url, extract the data from the
response and possibly prepare it (we advise minimally, since you can always
transform from the raw source, but the opposite can be impossible).

```pycon
>>> def foo():
...     return 'bar'
>>> def pi():
...     return 3.14159
>>> s = FuncReader([foo, pi])
>>> list(s)
['foo', 'pi']
>>> s['foo']
'bar'
>>> s['pi']
3.14159
```

You might want to give your own names to the functions.
You might even have to (because the callable you’re using doesn’t have a `__name__`).
In that case, you can specify a `{name: func, ...}` dict instead of a simple
iterable.

```pycon
>>> s = FuncReader({'FU': foo, 'Pie': pi})
>>> list(s)
['FU', 'Pie']
>>> s['FU']
'bar'
```

### *class* dol.sources.MultiSource(\*sources)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

A read-only Mapping that composes multiple sources, tried in order.

On key lookup, sources are tried left-to-right until one has the key.
On iteration, keys are yielded from all sources (deduplicated, order-preserved).

This is useful as the `source` argument to [`dol.caching.mk_sourced_store()`](dol.caching.html.md#dol.caching.mk_sourced_store)
when you need fallback across multiple read-only backends.

```pycon
>>> s1 = {'a': 1, 'b': 2}
>>> s2 = {'b': 20, 'c': 3}
>>> ms = MultiSource(s1, s2)
>>> ms['a']
1
>>> ms['b']
2
>>> ms['c']
3
>>> sorted(ms)
['a', 'b', 'c']
>>> len(ms)
3
>>> 'c' in ms
True
>>> 'z' in ms
False
>>> ms['z']
Traceback (most recent call last):
    ...
KeyError: 'z'
```

### *exception* dol.sources.NotUnique

Bases: [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

Raised when an iterator was expected to have only one element, but had more

### *class* dol.sources.ObjReader(\_obj_of_key)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A reader that uses a specified function to get the contents for a given key.

```pycon
>>> # define a contents_of_key that reads stuff from a dict
>>> data = {'foo': 'bar', 42: "everything"}
>>> def read_dict(k):
...     return data[k]
>>> pr = ObjReader(_obj_of_key=read_dict)
>>> pr['foo']
'bar'
>>> pr[42]
'everything'
>>>
>>> # define contents_of_key that reads stuff from a file given it's path
>>> def read_file(path):
...     with open(path) as fp:
...         return fp.read()
>>> pr = ObjReader(_obj_of_key=read_file)
>>> file_where_this_code_is = __file__
```

`file_where_this_code_is` should be the file where this doctest is written,
therefore should contain what I just said:

```pycon
>>> 'therefore should contain what I just said' in pr[file_where_this_code_is]
True
```

### *class* dol.sources.SequenceKvReader(sequence, key=None, val=None, val_postproc=<class 'list'>)

Bases: [`KvReader`](dol.base.html.md#dol.base.KvReader)

A KvReader that sources itself in an iterable of elements from which keys and values
will be extracted and grouped by key.

```pycon
>>> docs = [{'_id': 0, 's': 'a', 'n': 1},
...  {'_id': 1, 's': 'b', 'n': 2},
...  {'_id': 2, 's': 'b', 'n': 3}]
>>>
```

Out of the box, SequenceKvReader gives you enumerated integer indices as keys,
and the sequence items as is, as vals

```pycon
>>> s = SequenceKvReader(docs)
>>> list(s)
[0, 1, 2]
>>> s[1]
{'_id': 1, 's': 'b', 'n': 2}
>>> assert s.get('not_a_key') is None
```

You can make it more interesting by specifying a val function to compute the vals
from the sequence elements

```pycon
>>> s = SequenceKvReader(docs, val=lambda x: (x['_id'] + x['n']) * x['s'])
>>> assert list(s) == [0, 1, 2]  # as before
>>> list(s.values())
['a', 'bbb', 'bbbbb']
```

But where it becomes more useful is when you specify a key as well.
SequenceKvReader will then compute the keys with that function, group them,
and return as the value, the list of sequence elements that match that key.

```pycon
>>> s = SequenceKvReader(docs,
...         key=lambda x: x['s'],
...         val=lambda x: {k: x[k] for k in x.keys() - {'s'}})
>>> assert list(s) == ['a', 'b']
>>> assert s['a'] == [{'_id': 0, 'n': 1}]
>>> assert s['b'] == [{'_id': 1, 'n': 2}, {'_id': 2, 'n': 3}]
```

The cannonical form of key and val is a function, but if you specify a str, int,
or iterable thereof,
SequenceKvReader will make an itemgetter function from it, for your convenience.

```pycon
>>> s = SequenceKvReader(docs, key='_id')
>>> assert list(s) == [0, 1, 2]
>>> assert s[1] == [{'_id': 1, 's': 'b', 'n': 2}]
```

The `val_postproc` argument is `list` by default, but what if we don’t specify
any?
Well then you’ll get an unconsumed iterable of matches

```pycon
>>> s = SequenceKvReader(docs, key='_id', val_postproc=None)
>>> assert isinstance(s[1], Iterable)
```

The `val_postproc` argument specifies what to apply to this iterable of matches.
For example, you can specify `val_postproc=next` to simply get the first matched
element:

```pycon
>>> s = SequenceKvReader(docs, key='_id', val_postproc=next)
>>> assert list(s) == [0, 1, 2]
>>> assert s[1] == {'_id': 1, 's': 'b', 'n': 2}
```

We got the whole dict there. What if we just want we didn’t want the \_id, which is
used by the key, in our val?

```pycon
>>> from functools import partial
>>> all_but_s = partial(exclusive_subdict, exclude=['s'])
>>> s = SequenceKvReader(docs, key='_id', val=all_but_s, val_postproc=next)
>>> assert list(s) == [0, 1, 2]
>>> assert s[1] == {'_id': 1, 'n': 2}
```

Suppose we want to have the pair of (‘_id’, ‘n’) values as a key, and only ‘s’
as a value…

```pycon
>>> s = SequenceKvReader(docs, key=('_id', 'n'), val='s', val_postproc=next)
>>> assert list(s) == [(0, 1), (1, 2), (2, 3)]
>>> assert s[1, 2] == 'b'
```

But remember that using `val_postproc=next` will only give you the first match
as a val.

```pycon
>>> s = SequenceKvReader(docs, key='s', val=all_but_s, val_postproc=next)
>>> assert list(s) == ['a', 'b']
>>> assert s['a'] == {'_id': 0, 'n': 1}
>>> assert s['b'] == {'_id': 1, 'n': 2}   # note that only the first match is returned.
```

If you do want to only grab the first match, but want to additionally assert
that there is no more than one,
you can specify this with `val_postproc=unique_element`:

```pycon
>>> s = SequenceKvReader(docs, key='s', val=all_but_s, val_postproc=unique_element)
>>> assert s['a'] == {'_id': 0, 'n': 1}
>>> # The following should raise an exception since there's more than one match
>>> s['b']
Traceback (most recent call last):
  ...
sources.NotUnique: iterator had more than one element
```
