# dol.dig

Layers introspection: walk the layers of a wrapped store and trace a key through them.

Main entry points:

- `layers`: the list of nested layers (found through the `store` attribute), outermost first
- `trace_getitem`, `print_trace_info`: the (layer, method, value) steps of a `__getitem__`

### Functions

| `dig_up`(store, attr[, default])                                                           |                                                                                 |
|--------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `get_first_attr_found`(store, attrs[, default])                                            |                                                                                 |
| [`inner_most`](#dol.dig.inner_most)(store, arg, method[, default]) | The value of `arg` after every layer's `method` has been applied.               |
| [`key_in_scope`](#dol.dig.key_in_scope)(store, k)                    | Whether the key `k` of `store` is inside every `filt_iter` scope of its layers. |
| `last_element`(gen)                                                                        |                                                                                 |
| `layers`(store[, layer_attrs])                                                             |                                                                                 |
| `next_layer`(store[, layer_attrs])                                                         |                                                                                 |
| `print_trace_info`(trace[, item_info])                                                     |                                                                                 |
| `print_trans_path`(store, arg, method[, with_type])                                        |                                                                                 |
| `re_get_attr`(store, attr[, default])                                                      |                                                                                 |
| `recursive_calls`(func, x[, sentinel])                                                     |                                                                                 |
| `recursive_get_attr`(store, attr[, default])                                               |                                                                                 |
| [`store_trans_path`](#dol.dig.store_trans_path)(store, arg, method)      | Yield `arg` transformed by `method` at each layer, outermost first.             |
| [`trace_getitem`](#dol.dig.trace_getitem)(store, k[, layer_attrs])    | A generator of layered steps to inspect a store.                                |
| `trace_info`(trace[, item_func])                                                           |                                                                                 |

### dol.dig.inner_most(store, arg, method, default=<dol.dig.NoDefault object>)

The value of `arg` after every layer’s `method` has been applied.

```pycon
>>> from dol import wrap_kvs
>>> s = wrap_kvs({'a.txt': 1}, id_of_key=lambda k: k + '.txt')
>>> inner_most(s, 'a', '_id_of_key')
'a.txt'
```

\*\*Raises when no layer defines `method``**, instead of silently returning ``None`:

```pycon
>>> inner_most({}, 'a', '_id_of_key')
Traceback (most recent call last):
  ...
AttributeError: No layer of dict defines '_id_of_key', so 'a' cannot be resolved. ...
```

A `None` here is the worst possible answer: callers use the result as a key or a
path, so it surfaces far from its cause – as `https://.../None`, or as
`TypeError: expected str, bytes or os.PathLike object, not NoneType`.

Pass `default` to opt out of raising:

```pycon
>>> inner_most({}, 'a', '_id_of_key', default=None) is None
True
```

### dol.dig.inner_most_key(store, arg, \*, method='_id_of_key', default=<dol.dig.NoDefault object>)

The value of `arg` after every layer’s `method` has been applied.

```pycon
>>> from dol import wrap_kvs
>>> s = wrap_kvs({'a.txt': 1}, id_of_key=lambda k: k + '.txt')
>>> inner_most(s, 'a', '_id_of_key')
'a.txt'
```

\*\*Raises when no layer defines `method``**, instead of silently returning ``None`:

```pycon
>>> inner_most({}, 'a', '_id_of_key')
Traceback (most recent call last):
  ...
AttributeError: No layer of dict defines '_id_of_key', so 'a' cannot be resolved. ...
```

A `None` here is the worst possible answer: callers use the result as a key or a
path, so it surfaces far from its cause – as `https://.../None`, or as
`TypeError: expected str, bytes or os.PathLike object, not NoneType`.

Pass `default` to opt out of raising:

```pycon
>>> inner_most({}, 'a', '_id_of_key', default=None) is None
True
```

### dol.dig.inner_most_val(store, arg, \*, method='_data_of_obj', default=<dol.dig.NoDefault object>)

The value of `arg` after every layer’s `method` has been applied.

```pycon
>>> from dol import wrap_kvs
>>> s = wrap_kvs({'a.txt': 1}, id_of_key=lambda k: k + '.txt')
>>> inner_most(s, 'a', '_id_of_key')
'a.txt'
```

\*\*Raises when no layer defines `method``**, instead of silently returning ``None`:

```pycon
>>> inner_most({}, 'a', '_id_of_key')
Traceback (most recent call last):
  ...
AttributeError: No layer of dict defines '_id_of_key', so 'a' cannot be resolved. ...
```

A `None` here is the worst possible answer: callers use the result as a key or a
path, so it surfaces far from its cause – as `https://.../None`, or as
`TypeError: expected str, bytes or os.PathLike object, not NoneType`.

Pass `default` to opt out of raising:

```pycon
>>> inner_most({}, 'a', '_id_of_key', default=None) is None
True
```

### dol.dig.key_in_scope(store, k)

Whether the key `k` of `store` is inside every `filt_iter` scope of its layers.

A side-effect-free probe: nothing is read or written. `k` is followed through the layers
outermost first; each layer’s `filt_iter` predicates are applied to the key \*as it
enters that layer\*, which is what `store[k] = v` would do, then `_id_of_key` maps it
for the layer below. A leaf that has an `is_valid_key` method (e.g. `Files`) is asked
about the innermost key last.

```pycon
>>> from dol import filt_iter, KeyCodecs
>>> s = KeyCodecs.prefixed('data/')(filt_iter(dict, filt=lambda k: k.endswith('.txt')))()
>>> key_in_scope(s, 'a.txt'), key_in_scope(s, 'a.bin')
(True, False)
```

Here the `filt_iter` sits *below* the prefix, so it sees `'data/a.txt'`, not
`'a.txt'`:

```pycon
>>> s = KeyCodecs.prefixed('data/')(filt_iter(dict, filt=lambda k: k.startswith('data/a')))()
>>> key_in_scope(s, 'a.txt'), key_in_scope(s, 'b.txt')
(True, False)
```

The predicate is the layer’s own, so a predicate that raises on a key raises here too
(as `s[k]` would), and so does a layer whose `_id_of_key` raises for the key (e.g. a
key mapping that does not know it): the answer is a `bool` only where a write would be
accepted or refused, not failed. “Side-effect-free” holds as far as the layers’
`_id_of_key` and the leaf’s `is_valid_key` are.

Limits (the probe reads what the layers *declare*): a layer that maps keys but has no
`_id_of_key` cannot be followed (`TypeError`); a layer that looks keys up by its own
means while keeping the do-nothing `_id_of_key` (`add_path_access`, `flatten`, whose
keys are tuples walked through the inner store) gets a silently wrong answer; so does a
subclass that changes the key inside `__setitem__` before calling `super()`.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### dol.dig.store_trans_path(store, arg, method)

Yield `arg` transformed by `method` at each layer, outermost first.

Walks the `.store` chain, applying `store.<method>` at every layer that defines it.

```pycon
>>> from dol import wrap_kvs
>>> s = wrap_kvs({'a.txt': 1}, id_of_key=lambda k: k + '.txt')
>>> list(store_trans_path(s, 'a', '_id_of_key'))
['a.txt', 'a.txt']
```

Yields nothing when no layer defines `method` – which is why [`inner_most()`](#dol.dig.inner_most)
raises rather than returning the `None` that an empty walk would otherwise produce.

```pycon
>>> list(store_trans_path({}, 'a', '_id_of_key'))
[]
```

### dol.dig.trace_getitem(store, k, layer_attrs=('store',))

A generator of layered steps to inspect a store.

* **Parameters:**
  * **store** – An instance that has the base.Store interface
  * **k** – A key
  * **layer_attrs** – The attribute names that should be checked to get the next layer.
* **Returns:**
  A generator of (layer, method, value)

We start with a small dict:

```pycon
>>> d = {'a.num': '1000', 'b.num': '2000'}
```

Now let’s add layers to it. For example, with wrap_kvs:

```pycon
>>> from dol.trans import wrap_kvs
```

Say that we want the interface to not see the `'.num'` strings, and deal with numerical values, not strings.

```pycon
>>> s = wrap_kvs(d,
...              key_of_id=lambda x: x[:-len('.num')],
...              id_of_key=lambda x: x + '.num',
...              obj_of_data=lambda x: int(x),
...              data_of_obj=lambda x: str(x)
...             )
>>>
```

Oh, and we want the interface to display upper case keys.

```pycon
>>> ss = wrap_kvs(s,
...              key_of_id=lambda x: x.upper(),
...              id_of_key=lambda x: x.lower(),
...             )
```

And we want the numerical unit to be the kilo (that’s 1000):

```pycon
>>> sss = wrap_kvs(ss,
...                obj_of_data=lambda x: x / 1000,
...                data_of_obj=lambda x: x * 1000
...               )
>>>
>>> dict(sss.items())
{'A': 1.0, 'B': 2.0}
```

Well, if we had bugs, we’d like to inspect the various layers, and how they transform the data.

```python
# Here's how to do that:

# >>> for layer, method, value in trace_getitem(sss, 'A'):
# ...     print(layer, method, value)
# ...
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# NameError: name 'trace_getitem' is not defined
```

```pycon
>>> from dol.dig import trace_getitem
>>>
>>> for layer, method, value in trace_getitem(sss, 'A'):
...     print(layer, method, value)
...
{'a.num': '1000', 'b.num': '2000'} _id_of_key A
{'a.num': '1000', 'b.num': '2000'} _id_of_key A
{'a.num': '1000', 'b.num': '2000'} _id_of_key a
{'a.num': '1000', 'b.num': '2000'} _id_of_key a
{'a.num': '1000', 'b.num': '2000'} _id_of_key a.num
{'a.num': '1000', 'b.num': '2000'} _id_of_key a.num
{'a.num': '1000', 'b.num': '2000'} __getitem__ 1000
{'a.num': '1000', 'b.num': '2000'} _obj_of_data 1000
{'a.num': '1000', 'b.num': '2000'} _obj_of_data 1000
{'a.num': '1000', 'b.num': '2000'} _obj_of_data 1000
{'a.num': '1000', 'b.num': '2000'} _obj_of_data 1000
{'a.num': '1000', 'b.num': '2000'} _obj_of_data 1000
{'a.num': '1000', 'b.num': '2000'} _obj_of_data 1.0
```

### dol.dig.unravel_key(store, arg, , method='_id_of_key')

Yield `arg` transformed by `method` at each layer, outermost first.

Walks the `.store` chain, applying `store.<method>` at every layer that defines it.

```pycon
>>> from dol import wrap_kvs
>>> s = wrap_kvs({'a.txt': 1}, id_of_key=lambda k: k + '.txt')
>>> list(store_trans_path(s, 'a', '_id_of_key'))
['a.txt', 'a.txt']
```

Yields nothing when no layer defines `method` – which is why [`inner_most()`](#dol.dig.inner_most)
raises rather than returning the `None` that an empty walk would otherwise produce.

```pycon
>>> list(store_trans_path({}, 'a', '_id_of_key'))
[]
```

### dol.dig.unravel_val(store, arg, , method='_data_of_obj')

Yield `arg` transformed by `method` at each layer, outermost first.

Walks the `.store` chain, applying `store.<method>` at every layer that defines it.

```pycon
>>> from dol import wrap_kvs
>>> s = wrap_kvs({'a.txt': 1}, id_of_key=lambda k: k + '.txt')
>>> list(store_trans_path(s, 'a', '_id_of_key'))
['a.txt', 'a.txt']
```

Yields nothing when no layer defines `method` – which is why [`inner_most()`](#dol.dig.inner_most)
raises rather than returning the `None` that an empty walk would otherwise produce.

```pycon
>>> list(store_trans_path({}, 'a', '_id_of_key'))
[]
```
