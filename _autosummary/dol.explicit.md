# dol.explicit

Stores whose keys are given explicitly, with values fetched lazily from a source.

Main entry points:

- `KeysReader`: a collection of keys plus a `getter(src, key)`
- `ExplicitKeysSource`: explicit keys plus a function reading the value for a key
- `ExplicitKeysStore`: wrap a store so that its keys come from an explicit iterable
- `ExplicitKeyMap`: a key mapper given as explicit dicts
  ```pycon
  >>> from dol.explicit import KeysReader
  >>> r = KeysReader({'apple': 'pie', 'banana': 'split'}, ['banana'], lambda src, k: src[k])
  >>> list(r), r['banana']
  (['banana'], 'split')
  ```

### Classes

| [`ExplicitKeyMap`](#dol.explicit.ExplicitKeyMap)(\*[, key_of_id, id_of_key])         | A key mapper given as explicit `key_of_id`/`id_of_key` dicts (one is enough; the other is derived, and both are checked to be inverse of each other).   |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`ExplicitKeymapReader`](#dol.explicit.ExplicitKeymapReader)(store[, key_of_id, ...])      | Wrap a store (instance) so that it gets it's keys from an explicit iterable of keys.                                                                    |
| [`ExplicitKeys`](#dol.explicit.ExplicitKeys)(key_collection)                       | dol.base.Keys implementation that gets it's keys explicitly from a collection given at initialization time.                                             |
| [`ExplicitKeysSource`](#dol.explicit.ExplicitKeysSource)(key_collection, \_obj_of_key)   | An object source that uses an explicit keys collection and a specified function to read contents for a key.                                             |
| [`ExplicitKeysStore`](#dol.explicit.ExplicitKeysStore)(store, key_collection)           | Wrap a store (instance) so that it gets it's keys from an explicit iterable of keys.                                                                    |
| [`KeysReader`](#dol.explicit.KeysReader)(src, key_collection, getter, \*[, ...]) | Mapping defined by keys with a getter function that gets values from keys.                                                                              |
| `ObjDumper`(save_data_to_key[, data_of_obj])                                                        |                                                                                                                                                         |

### *class* dol.explicit.ExplicitKeyMap(, key_of_id=None, id_of_key=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A key mapper given as explicit `key_of_id`/`id_of_key` dicts (one is enough;
the other is derived, and both are checked to be inverse of each other).
Provides the `_key_of_id`/`_id_of_key` methods that `kv_wrap` looks for.

### *class* dol.explicit.ExplicitKeymapReader(store, key_of_id=None, id_of_key=None)

Bases: [`ExplicitKeys`](#dol.explicit.ExplicitKeys), [`Store`](dol.base.md#dol.base.Store)

Wrap a store (instance) so that it gets it’s keys from an explicit iterable of keys.

```pycon
>>> s = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
>>> id_of_key = {'A': 'a', 'C': 'c'}
>>> ss = ExplicitKeymapReader(s, id_of_key=id_of_key)
>>> list(ss)
['A', 'C']
>>> ss['C']  # will look up 'C', find 'c', and call the store on that.
3
```

### *class* dol.explicit.ExplicitKeys(key_collection)

Bases: [`Collection`](dol.base.md#dol.base.Collection)

dol.base.Keys implementation that gets it’s keys explicitly from a collection given
at initialization time.
The key_collection must be a collections.abc.Collection
(such as list, tuple, set, etc.)

```pycon
>>> keys = ExplicitKeys(key_collection=['foo', 'bar', 'alice'])
>>> 'foo' in keys
True
>>> 'not there' in keys
False
>>> list(keys)
['foo', 'bar', 'alice']
```

### *class* dol.explicit.ExplicitKeysSource(key_collection, \_obj_of_key)

Bases: [`ExplicitKeys`](#dol.explicit.ExplicitKeys), [`ObjReader`](dol.sources.md#dol.sources.ObjReader), [`KvReader`](dol.base.md#dol.base.KvReader)

An object source that uses an explicit keys collection and a specified function to
read contents for a key.

```pycon
>>> s = ExplicitKeysSource([1, 2, 3], str)
>>> list(s)
[1, 2, 3]
>>> list(s.values())
['1', '2', '3']
```

Main functionality equivalent to recipe:

```pycon
>>> def explicit_keys_source(key_collection, _obj_of_key):
...     from dol.trans import wrap_kvs
...     return wrap_kvs({k: k for k in key_collection}, obj_of_data=_obj_of_key)
```

```pycon
>>> s = explicit_keys_source([1, 2, 3], str)
>>> list(s)
[1, 2, 3]
>>> list(s.values())
['1', '2', '3']
```

### *class* dol.explicit.ExplicitKeysStore(store, key_collection)

Bases: [`ExplicitKeys`](#dol.explicit.ExplicitKeys), [`Store`](dol.base.md#dol.base.Store)

Wrap a store (instance) so that it gets it’s keys from an explicit iterable of keys.

```pycon
>>> s = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
>>> list(s)
['a', 'b', 'c', 'd']
>>> ss = ExplicitKeysStore(s, ['d', 'a'])
>>> len(ss)
2
>>> list(ss)
['d', 'a']
>>> list(ss.values())
[4, 1]
>>> ss.head()
('d', 4)
```

### *class* dol.explicit.KeysReader(src, key_collection, getter, \*, key_error_msg=<built-in method format of str object>)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

Mapping defined by keys with a getter function that gets values from keys.

`KeysReader` is particularly useful in cases where you want to have a mapping
that lazy-load values for keys from an explicit collection.

Keywords: Lazy-evaluation, Mapping

* **Parameters:**
  * **src** ([`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Source`)) – The source where values will be extracted from.
  * **key_collection** ([`Collection`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Collection)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)]) – A collection of keys that will be used to extract values from `src`.
  * **getter** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Source`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)]) – A function that takes a source and a key, and returns the value for that key.
  * **key_error_msg** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Source`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – A function that takes a source and a key, and returns an error message.

### Example

```pycon
>>> src = {'apple': 'pie', 'banana': 'split', 'carrot': 'cake'}
>>> key_collection = ['carrot', 'apple']
>>> getter = lambda src, key: src[key]
>>> key_reader = KeysReader(src, key_collection, getter)
```

Note that the only the keys mentioned by `key_collection` will be iterated through,
and in the order they are mentioned in `key_collection`.

```pycon
>>> list(key_reader)
['carrot', 'apple']
```

```pycon
>>> key_reader['apple']
'pie'
>>> key_reader['banana']
Traceback (most recent call last):
...
KeyError: "Key 'banana' was not found....key_collection attribute)"
```

Let’s take the same `src` and `key_collection`, but with a different getter and
key_error_msg:

Note that a key_error_msg must be a function that takes a `src` and a `key`,
in that order and with those argument names. Say you wanted to not use the `src`
in your message. You would still have to write a function that takes `src` as the
first argument.

```pycon
>>> key_error_msg = lambda src, key: f"Key {key} was not found"  # no source information
```

```pycon
>>> getter = lambda src, key: f"Value for {key} in {src}: {src[key]}"
>>> key_reader = KeysReader(src, key_collection, getter, key_error_msg=key_error_msg)
>>> list(key_reader)
['carrot', 'apple']
>>> key_reader['apple']
"Value for apple in {'apple': 'pie', 'banana': 'split', 'carrot': 'cake'}: pie"
>>> key_reader['banana']
Traceback (most recent call last):
...
KeyError: "Key banana was not found"
```
