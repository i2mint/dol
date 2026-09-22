# dol.paths

Module for path (and path-like) object manipulation

### Examples

```pycon
>>> d = {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
>>> list(path_filter(lambda p, k, v: v == 2, d))
[('a', 'b', 'd')]
>>> path_get(d, ('a', 'b', 'd'))
2
>>> path_set(d, ('a', 'b', 'd'), 4)
>>> d
{'a': {'b': {'c': 1, 'd': 4}, 'e': 3}}
>>> path_set(d, ('a', 'b', 'new_ab_key'), 42)
>>> d
{'a': {'b': {'c': 1, 'd': 4, 'new_ab_key': 42}, 'e': 3}}
```

### Functions

| [`add_prefix_filtering`](#dol.paths.add_prefix_filtering)([store, ...])                   | Make a missing key that is a prefix of existing keys return the sub-mapping of those keys (so `s['a/']` lists everything "under" `a/`).   |
|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| [`cast_to_int_if_numeric_str`](#dol.paths.cast_to_int_if_numeric_str)(k)                        | Cast `k` to `int` if it is a numeric string; return it unchanged otherwise.                                                               |
| [`chain_of_getters`](#dol.paths.chain_of_getters)(getters[, obj, k, ...])             | If `k` is a string, tries to get `k` as an attribute of `obj` first, and if that fails, gets it as `obj[k]`                               |
| [`ensure_path_extender_func`](#dol.paths.ensure_path_extender_func)(path_extender)             | Ensure that the path_extender is a function that takes a path and a key and returns a new path.                                           |
| [`flatten_dict`](#dol.paths.flatten_dict)(d[, sep, parent_path, ...])             | Flatten a nested dictionary into a flat one, using key-paths as keys.                                                                     |
| [`flattened_dict_items`](#dol.paths.flattened_dict_items)(d[, sep, parent_path, ...])     | Yield flattened key-value pairs from a nested dictionary.                                                                                 |
| [`get_attr_or_item`](#dol.paths.get_attr_or_item)(obj, k)                             | If `k` is a string, tries to get `k` as an attribute of `obj` first, and if that fails, gets it as `obj[k]`                               |
| [`getitem`](#dol.paths.getitem)(obj, k)                                      | Return `obj[k]`.                                                                                                                          |
| [`handle_prefixes`](#dol.paths.handle_prefixes)([store, prefix, ...])                | A store decorator that handles prefixes.                                                                                                  |
| [`identity`](#dol.paths.identity)(x)                                          | Return `x`.                                                                                                                               |
| [`keys_and_indices_path`](#dol.paths.keys_and_indices_path)(str_path, \*[, sep, ...])      | Transforms a string path separated by a specified separator into a tuple of keys and indices.                                             |
| [`leaf_paths`](#dol.paths.leaf_paths)(d[, sep, parent_path, egress])            | Get a dictionary of leaf paths of a nested dictionary.                                                                                    |
| [`mk_relative_path_store`](#dol.paths.mk_relative_path_store)([store_cls, name, ...])       |                                                                                                                                           |
| [`path_edit`](#dol.paths.path_edit)(d[, edits])                                | Make a series of (in place) edits to a Mapping, specifying `(path, value)` pairs.                                                         |
| [`path_filter`](#dol.paths.path_filter)(pkv_filt, d, \*[, leafs_only, ...])      | Walk a dict, yielding paths to values that pass the `pkv_filt`                                                                            |
| [`path_get`](#dol.paths.path_get)(obj, path[, on_error, sep, ...])            | Get elements of a mapping through a path to be called recursively.                                                                        |
| [`paths_getter`](#dol.paths.paths_getter)(paths[, obj, egress, on_error, ...])    | Returns (path, values) pairs of the given paths in the given object.                                                                      |
| [`prefixless_view`](#dol.paths.prefixless_view)([store, prefix, \_\_module_\_, ...]) | Wrap `store` so that keys are seen without `prefix` (added back on access).                                                               |
| [`raise_on_error`](#dol.paths.raise_on_error)(d)                                    | `on_error` policy for `path_get`: re-raise the caught error.                                                                              |
| [`rel_path_wrap`](#dol.paths.rel_path_wrap)(o, \_prefix)                           |                                                                                                                                           |
| [`return_empty_tuple_on_error`](#dol.paths.return_empty_tuple_on_error)(d)                       | `on_error` policy for `path_get`: return `()`.                                                                                            |
| [`return_none_on_error`](#dol.paths.return_none_on_error)(d)                              | `on_error` policy for `path_get`: return `None`.                                                                                          |
| [`search_paths`](#dol.paths.search_paths)(d, pkv_filt, \*[, leafs_only, ...])     | backwards compatibility quasi-alias (arguments are flipped) Use path_filter instead, since search_paths will be deprecated.               |
| [`separate_keys_with_separator`](#dol.paths.separate_keys_with_separator)(obj[, sep])             | Split a string path on `sep` and cast numeric parts to `int`; a non-string iterable is only cast element-wise.                            |
| [`separator_based_path_extender`](#dol.paths.separator_based_path_extender)(path, key, sep)        | Extends a given path with a new key using the specified separator.                                                                        |
| [`split_if_str`](#dol.paths.split_if_str)(obj[, sep])                             | Split `obj` on `sep` if it is a string; return it unchanged otherwise.                                                                    |
| [`str_template_key_trans`](#dol.paths.str_template_key_trans)(template, key_type[, ...])    | Make a key trans object that translates from a string \_id to a dict, tuple, or namedtuple key (and back)                                 |
| [`string_unparse`](#dol.paths.string_unparse)(parsing_result)                       | The inverse of string.Formatter.parse                                                                                                     |

### Classes

| [`ExplicitKeysWithPrefixRelativization`](#dol.paths.ExplicitKeysWithPrefixRelativization)(...[, ...])   | dol.base.Keys implementation that gets it's keys explicitly from a collection given at initialization time.       |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| [`KeyPath`](#dol.paths.KeyPath)([path_sep, \_path_type, ...])              | A key mapper that converts from an iterable key (default tuple) to a string (given a path-separator str)          |
| [`KeyTemplate`](#dol.paths.KeyTemplate)(template, \*[, field_patterns, ...])   | A class for parsing and generating keys based on a template.                                                      |
| [`PathKeyTypes`](#dol.paths.PathKeyTypes)(\*values)                             | Enum of the path key forms: `str`, `dict`, `tuple`, `namedtuple`.                                                 |
| [`PathMappedData`](#dol.paths.PathMappedData)(src, key_collection[, ...])         | A collection of keys with a key_to_value function to lazy load values.                                            |
| [`PrefixRelativization`](#dol.paths.PrefixRelativization)([_prefix])                    | A key wrap that allows one to interface with absolute paths through relative paths.                               |
| [`PrefixRelativizationMixin`](#dol.paths.PrefixRelativizationMixin)()                        | Mixin that adds a intercepts the \_id_of_key an \_key_of_id methods, transforming absolute keys to relative ones. |
| [`RelativePathKeyMapper`](#dol.paths.RelativePathKeyMapper)(prefix)                      | Key mapper adding `prefix` on the way in and removing it on the way out.                                          |

### *class* dol.paths.ExplicitKeysWithPrefixRelativization(key_collection, \_prefix=None)

Bases: [`PrefixRelativizationMixin`](#dol.paths.PrefixRelativizationMixin), [`Store`](dol.base.md#dol.base.Store)

dol.base.Keys implementation that gets it’s keys explicitly from a collection given at initialization time.
The key_collection must be a collections.abc.Collection (such as list, tuple, set, etc.)

```pycon
>>> from dol.base import Store
>>> s = ExplicitKeysWithPrefixRelativization(key_collection=['/root/of/foo', '/root/of/bar', '/root/for/alice'])
>>> keys = Store(store=s)
>>> 'of/foo' in keys
True
>>> 'not there' in keys
False
>>> list(keys)
['of/foo', 'of/bar', 'for/alice']
```

### *class* dol.paths.KeyPath(path_sep='/', \_path_type=<class 'tuple'>, \*, create_missing=False, mk_missing=None, explore_further=None, may_create=None, on_create=<function \_warn_on_create>, max_created=None, max_levels=20, verify_writeback=False, writeback_lock=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A key mapper that converts from an iterable key (default tuple) to a string
(given a path-separator str)

* **Parameters:**
  * **path_sep** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The path separator (used to make string paths from iterable paths and
    visa versa
  * **\_path_type** ([`type`](https://docs.python.org/3/builtins/functions.html#type) | [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – The type of the outcoming (inner) path. But really, any function to
  * **to** (*convert from a list*) – the outer path type we want.

With `'/'` as a separator:

```pycon
>>> kp = KeyPath(path_sep='/')
>>> kp._key_of_id(('a', 'b', 'c'))
'a/b/c'
>>> kp._id_of_key('a/b/c')
('a', 'b', 'c')
```

With `'.'` as a separator:

```pycon
>>> kp = KeyPath(path_sep='.')
>>> kp._key_of_id(('a', 'b', 'c'))
'a.b.c'
>>> kp._id_of_key('a.b.c')
('a', 'b', 'c')
>>> kp = KeyPath(path_sep=':::', _path_type=dict.fromkeys)
>>> _id = dict.fromkeys('abc')
>>> _id
{'a': None, 'b': None, 'c': None}
>>> kp._key_of_id(_id)
'a:::b:::c'
>>> kp._id_of_key('a:::b:::c')
{'a': None, 'b': None, 'c': None}
```

Calling a `KeyPath` instance on a store wraps it so we can have path access to
it.

```pycon
>>> s = {'a': {'b': {'c': 42}}}
>>> s['a']['b']['c']
42
>>> # Now let's wrap the store
>>> s = KeyPath('.')(s)
>>> s['a.b.c']
42
>>> s['a.b.c'] = 3.14
>>> s['a.b.c']
3.14
>>> del s['a.b.c']
>>> s
{'a': {'b': {}}}
```

#### NOTE
By default `KeyPath` reads with paths only when all the keys of the path
are valid (i.e. have a value), and, just like a `dict`, will *not* create
intermediate nested values for you on write. Pass `create_missing=True` to opt
into write-through autovivification: missing intermediates are created on write
(like `collections.defaultdict`, but with an optional contextual per-level
`mk_missing(ctx)` factory), and the change persists correctly even through
persistent / copy-semantics stores. See `misc/docs/dol_issue16_design.md`.

```pycon
>>> s = KeyPath('.', create_missing=True)({})
>>> s['a.b.c'] = 42
>>> s['a.b.c']
42
```

#### on_create()

Default `on_create` hook: announce a fabricated intermediate.

A structurally-valid *typo* on write would otherwise silently create a bogus
branch; warning keeps opted-in creation from being silent. Pass
`on_create=None` to silence (e.g. bulk tree building).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* dol.paths.KeyTemplate(template, \*, field_patterns=None, to_str_funcs=None, from_str_funcs=None, simple_str_sep=', ', namedtuple_type_name='NamedTuple', dflt_pattern='.\*', dflt_field_name=<built-in method format of str object>, normalize_paths=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A class for parsing and generating keys based on a template.

* **Parameters:**
  * **template** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A template string with fields to be extracted or filled in.
  * **field_patterns** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – A dictionary of field names and their regex patterns.
  * **simple_str_sep** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A separator string for simple strings (i.e. strings without
    fields).
  * **namedtuple_type_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The name of the namedtuple type to use for namedtuple
    fields.
  * **dflt_pattern** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The default pattern to use for fields that don’t have a pattern
    specified.
  * **to_str_funcs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – A dictionary of field names and their functions to convert them
    to strings.
  * **from_str_funcs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – A dictionary of field names and their functions to convert
    them from strings.

### Examples

```pycon
>>> st = KeyTemplate(
...     'root/{name}/v_{version}.json',
...     field_patterns={'version': r'\d+'},
...     from_str_funcs={'version': int},
... )
```

And now you have a template that can be used to convert between various
representations of the template: You can extract fields from strings, generate
strings from fields, etc.

```pycon
>>> st.str_to_dict("root/dol/v_9.json")
{'name': 'dol', 'version': 9}
>>> st.dict_to_str({'name': 'meshed', 'version': 42})
'root/meshed/v_42.json'
>>> st.dict_to_tuple({'name': 'meshed', 'version': 42})
('meshed', 42)
>>> st.tuple_to_dict(('i2', 96))
{'name': 'i2', 'version': 96}
>>> st.str_to_tuple("root/dol/v_9.json")
('dol', 9)
>>> st.tuple_to_str(('front', 11))
'root/front/v_11.json'
>>> st.str_to_namedtuple("root/dol/v_9.json")
NamedTuple(name='dol', version=9)
>>> st.str_to_simple_str("root/dol/v_9.json")
'dol,9'
>>> st_clone = st.clone(simple_str_sep='/')
>>> st_clone.str_to_simple_str("root/dol/v_9.json")
'dol/9'
```

With `st.key_codec`, you can make a `KeyCodec` for the given source (decoded)
and target (encoded) types.
A `key_codec` is a codec; it has an encoder and a decoder.

```pycon
>>> key_codec = st.key_codec('tuple', 'str')
>>> encoder, decoder = key_codec
>>> decoder('root/dol/v_9.json')
('dol', 9)
>>> encoder(('dol', 9))
'root/dol/v_9.json'
```

If you have a `Mapping`, you can use `key_codec` as a decorator to wrap
the mapping with a key mappings.

```pycon
>>> store = {
...     'root/meshed/v_151.json': '{"downloads": 41, "type": "productivity"}',
...     'root/dol/v_9.json': '{"downloads": 132, "type": "utility"}',
... }
>>>
>>> accessor = key_codec(store)
>>> list(accessor)
[('meshed', 151), ('dol', 9)]
>>> accessor['i2', 4] = '{"downloads": 274, "type": "utility"}'
>>> list(store)
['root/meshed/v_151.json', 'root/dol/v_9.json', 'root/i2/v_4.json']
>>> store['root/i2/v_4.json']
'{"downloads": 274, "type": "utility"}'
```

#### NOTE
If your store contains keys that don’t fit the format, key_codec will
raise a `ValueError`. To remedy this, you can use the `st.filt_iter` to
filter out keys that don’t fit the format, before you wrap the store with
`st.key_codec`.

```pycon
>>> store = {
...     'root/meshed/v_151.json': '{"downloads": 41, "type": "productivity"}',
...     'root/dol/v_9.json': '{"downloads": 132, "type": "utility"}',
...     'root/not/the/right/format': "something else"
... }
>>> accessor = st.filt_iter('str')(store)
>>> list(accessor)
['root/meshed/v_151.json', 'root/dol/v_9.json']
>>> accessor = st.key_codec('tuple', 'str')(st.filt_iter('str')(store))
>>> list(accessor)
[('meshed', 151), ('dol', 9)]
>>> accessor['dol', 9]
'{"downloads": 132, "type": "utility"}'
```

You can also ask any (handled) combination of field types:

```pycon
>>> key_codec = st.key_codec('tuple', 'dict')
>>> key_codec.encoder(('i2', 96))
{'name': 'i2', 'version': 96}
>>> key_codec.decoder({'name': 'fantastic', 'version': 4})
('fantastic', 4)
```

#### dict_to_namedtuple(params)

Generates a namedtuple from the dictionary values based on the template.

```pycon
>>> st = KeyTemplate(
...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
... )
>>> App = st.dict_to_namedtuple({'i01_': 'life', 'ver': 42})
>>> App
NamedTuple(i01_='life', ver=42)
```

#### dict_to_str(params)

Generates a string from the dictionary values based on the template.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> st = KeyTemplate(
...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
... )
>>> st.dict_to_str({'i01_': 'life', 'ver': 42})
'root/life/v_042.json'
```

#### dict_to_tuple(params)

Generates a tuple from the dictionary values based on the template.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)

```pycon
>>> st = KeyTemplate(
...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
... )
>>> st.str_to_tuple('root/life/v_42.json')
('life', 42)
```

#### filt_iter(field_type='str')

Makes a store decorator that filters out keys that don’t match the template
given field type.

```pycon
>>> store = {
...     'root/meshed/v_151.json': '{"downloads": 41, "type": "productivity"}',
...     'root/dol/v_9.json': '{"downloads": 132, "type": "utility"}',
...     'root/not/the/right/format': "something else"
... }
>>> filt = KeyTemplate('root/{pkg}/v_{version}.json')
>>> filtered_store = filt.filt_iter('str')(store)
>>> list(filtered_store)
['root/meshed/v_151.json', 'root/dol/v_9.json']
```

#### key_codec(decoded='tuple', encoded='str')

Makes a `KeyCodec` for the given source and target types.

```pycon
>>> st = KeyTemplate(
...     'root/{name}/v_{version}.json',
...     field_patterns={'version': r'\d+'},
...     from_str_funcs={'version': int},
... )
```

A `key_codec` is a codec; it has an encoder and a decoder.

```pycon
>>> key_codec = st.key_codec('tuple', 'str')
>>> encoder, decoder = key_codec
>>> decoder('root/dol/v_9.json')
('dol', 9)
>>> encoder(('dol', 9))
'root/dol/v_9.json'
```

If you have a `Mapping`, you can use `key_codec` as a decorator to wrap
the mapping with a key mappings.

```pycon
>>> store = {
...     'root/meshed/v_151.json': '{"downloads": 41, "type": "productivity"}',
...     'root/dol/v_9.json': '{"downloads": 132, "type": "utility"}',
... }
>>>
>>> accessor = key_codec(store)
>>> list(accessor)
[('meshed', 151), ('dol', 9)]
>>> accessor['i2', 4] = '{"downloads": 274, "type": "utility"}'
>>> list(store)
['root/meshed/v_151.json', 'root/dol/v_9.json', 'root/i2/v_4.json']
>>> store['root/i2/v_4.json']
'{"downloads": 274, "type": "utility"}'
```

#### NOTE
If your store contains keys that don’t fit the format, key_codec will
raise a `ValueError`. To remedy this, you can use the `st.filt_iter` to
filter out keys that don’t fit the format, before you wrap the store with
`st.key_codec`.

#### match_str(s)

Returns True iff the string matches the template.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

```pycon
>>> st = KeyTemplate(
...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
... )
>>> st.match_str('root/life/v_042.json')
True
>>> st.match_str('this/does/not_match')
False
```

#### namedtuple_to_dict(nt)

Converts a namedtuple to a dictionary.

```pycon
>>> st = KeyTemplate(
...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
... )
>>> App = st.dict_to_namedtuple({'i01_': 'life', 'ver': 42})
>>> st.namedtuple_to_dict(App)
{'i01_': 'life', 'ver': 42}
```

#### simple_str_to_str(ss)

Converts a simple character-delimited string to a string.

```pycon
>>> st = KeyTemplate(
...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
...     simple_str_sep='-',
... )
>>> st.simple_str_to_str('life-042')
'root/life/v_042.json'
```

#### simple_str_to_tuple(ss)

Converts a simple character-delimited string to a dict.

```pycon
>>> st = KeyTemplate(
...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
...     simple_str_sep='-',
... )
>>> st.simple_str_to_tuple('life-042')
('life', 42)
```

#### single_to_str(k)

Generates a string from the single value based on the template.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> st = KeyTemplate(
...     r'root/life/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
... )
>>> st.single_to_str(42)
'root/life/v_042.json'
```

#### str_to_dict(s)

Parses the input string and returns a dictionary of extracted values.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

```pycon
>>> st = KeyTemplate(
...     r'root/{}/v_{ver:03.0f:\d+}.json',
...     from_str_funcs={'ver': int},
... )
>>> st.str_to_dict('root/life/v_30.json')
{'i01_': 'life', 'ver': 30}
```

#### str_to_namedtuple(s)

Converts a string to a namedtuple.

```pycon
>>> st = KeyTemplate(
...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
... )
>>> App = st.str_to_namedtuple('root/life/v_042.json')
>>> App
NamedTuple(i01_='life', ver=42)
```

#### str_to_simple_str(s)

Converts a string to a simple string (i.e. a simple character-delimited string).

```pycon
>>> st = KeyTemplate(
...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
... )
>>> st.str_to_simple_str('root/life/v_042.json')
'life,042'
>>> st_clone = st.clone(simple_str_sep='-')
>>> st_clone.str_to_simple_str('root/life/v_042.json')
'life-042'
```

#### str_to_single(s)

Parses the input string and returns a single value.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

```pycon
>>> st = KeyTemplate(
...     r'root/life/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
... )
>>> st.str_to_single('root/life/v_42.json')
42
```

#### str_to_tuple(s)

Parses the input string and returns a tuple of extracted values.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)

```pycon
>>> st = KeyTemplate(
...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
... )
>>> st.str_to_tuple('root/life/v_42.json')
('life', 42)
```

#### tuple_to_dict(param_vals)

Generates a dictionary from the tuple values based on the template.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

```pycon
>>> st = KeyTemplate(
...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
... )
>>> st.tuple_to_dict(('life', 42))
{'i01_': 'life', 'ver': 42}
```

#### tuple_to_str(param_vals)

Generates a string from the tuple values based on the template.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> st = KeyTemplate(
...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
... )
>>> st.tuple_to_str(('life', 42))
'root/life/v_042.json'
```

### *class* dol.paths.PathKeyTypes(\*values)

Bases: [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

Enum of the path key forms: `str`, `dict`, `tuple`, `namedtuple`.

### *class* dol.paths.PathMappedData(src, key_collection, getter=<function path_get>, \*, key_to_value=None)

Bases: [`KeysReader`](dol.explicit.md#dol.explicit.KeysReader)

A collection of keys with a key_to_value function to lazy load values.

`PathMappedData` is particularly useful in cases where you want to have a mapping
that lazy-loads values for keys from an explicit collection.

Keywords: Lazy-evaluation, Mapping

* **Parameters:**
  * **data** – The mapping to extract data from
  * **paths** – The paths to extract data from the mapping

### Example

```pycon
>>> data = {
...     'a': {
...         'b': [{'c': 1}, {'c': 2}],
...         'd': 'bar'
...     }
... }
>>> paths = ['a.d', 'a.b.0.c']
>>>
>>> d = PathMappedData(data, paths)
>>> list(d)
['a.d', 'a.b.0.c']
>>> d['a.d']
'bar'
>>> d['a.b.0.c']
1
```

Now, data does contain a key path for ‘a.b.1.c’:

```pycon
>>> d.getter(d.src, 'a.b.1.c')
2
```

But since we didn’t mention it in our paths parameter, it will raise a KeyError
if we try to access it via the `PathMappedData` object:

```pycon
>>> d['a.b.1.c']
Traceback (most recent call last):
...
KeyError: "Key a.b.1.c was not found....key_collection attribute)"
```

### *class* dol.paths.PrefixRelativization(\_prefix='')

Bases: [`PrefixRelativizationMixin`](#dol.paths.PrefixRelativizationMixin)

A key wrap that allows one to interface with absolute paths through relative paths.
The original intent was for local files. Instead of referencing files through an absolute path such as:

>  */A/VERY/LONG/ROOT/FOLDER/the/file/we.want*

we can instead reference the file as:

> *the/file/we.want*

But PrefixRelativization can be used, not only for local paths, but when ever a string reference is involved.
In fact, not only strings, but any key object that has a \_\_len_\_, \_\_add_\_, and subscripting.

### *class* dol.paths.PrefixRelativizationMixin

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Mixin that adds a intercepts the \_id_of_key an \_key_of_id methods, transforming absolute keys to relative ones.
Designed to work with string keys, where absolute and relative are relative to a \_prefix attribute
(assumed to exist).
The cannonical use case is when keys are absolute file paths, but we want to identify data through relative paths.
Instead of referencing files through an absolute path such as
`/A/VERY/LONG/ROOT/FOLDER/the/file/we.want` we can instead reference the file
as `the/file/we.want`.

Note though, that PrefixRelativizationMixin can be used, not only for local paths,
but when ever a string reference is involved.
In fact, not only strings, but any key object that has a \_\_len_\_, \_\_add_\_, and subscripting.

When subclassed, should be placed before the class defining \_id_of_key an \_key_of_id.
Also, assumes that a (string) \_prefix attribute will be available.

```pycon
>>> from dol.base import Store
>>> from collections import UserDict
>>>
>>> class MyStore(PrefixRelativizationMixin, Store):
...     def __init__(self, store, _prefix='/root/of/data/'):
...         super().__init__(store)
...         self._prefix = _prefix
...
>>> s = MyStore(store=dict())  # using a dict as our store
>>> s['foo'] = 'bar'
>>> assert s['foo'] == 'bar'
>>> s['too'] = 'much'
>>> assert list(s.keys()) == ['foo', 'too']
>>> # Everything looks normal, but are the actual keys behind the hood?
>>> s._id_of_key('foo')
'/root/of/data/foo'
>>> # see when iterating over s.items(), we get the interface view:
>>> list(s.items())
[('foo', 'bar'), ('too', 'much')]
>>> # but if we ask the store we're actually delegating the storing to, we see what the keys actually are.
>>> s.store.items()
dict_items([('/root/of/data/foo', 'bar'), ('/root/of/data/too', 'much')])
```

### *class* dol.paths.RelativePathKeyMapper(prefix)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Key mapper adding `prefix` on the way in and removing it on the way out.

### dol.paths.add_prefix_filtering(store=None, , relativize_prefix=False, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Make a missing key that is a prefix of existing keys return the sub-mapping of
those keys (so `s['a/']` lists everything “under” `a/`).

```pycon
>>> d = {'a/b': 1, 'a/c': 2, 'd/e': 3, 'f': 4}
>>> s = add_prefix_filtering(d)
>>> assert s['a/'] == {'a/b': 1, 'a/c': 2}
```

Demo usage on a `Mapping` type:

```pycon
>>> from collections import UserDict
>>> D = add_prefix_filtering(UserDict)
>>> s = D(d)
>>> assert s['a/'] == {'a/b': 1, 'a/c': 2}
```

### dol.paths.cast_to_int_if_numeric_str(k)

Cast `k` to `int` if it is a numeric string; return it unchanged otherwise.

### dol.paths.chain_of_getters(getters, obj=None, k=None, \*, caught_errors=(<class 'Exception'>, ))

If `k` is a string, tries to get `k` as an attribute of `obj` first,
and if that fails, gets it as `obj[k]`

### dol.paths.ensure_path_extender_func(path_extender)

Ensure that the path_extender is a function that takes a path and a key and returns
a new path.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Path`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Path`)]

### dol.paths.flatten_dict(d, sep='.', \*, parent_path=None, visit_nested=<function <lambda>>, egress=<class 'dict'>)

Flatten a nested dictionary into a flat one, using key-paths as keys.

See also `leaf_paths` for a related function that returns paths to leaf values.

* **Parameters:**
  * **d** – The dictionary to flatten
  * **sep** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Path`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Path`)]]) – The separator to use for joining keys, or a function that takes a path and
    a key and returns a new path.
  * **parent_path** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Path`)]) – The path to the parent of the current dict
  * **visit_nested** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – A function that returns True if a value should be visited
  * **egress** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Generator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)], [`None`](https://docs.python.org/3/builtins/constants.html#None), [`None`](https://docs.python.org/3/builtins/constants.html#None)]], [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]) – A function that takes a generator of key-value pairs and returns a mapping

```pycon
>>> d = {'a': {'b': 2}, 'c': 3}
>>> flatten_dict(d)
{'a.b': 2, 'c': 3}
>>> flatten_dict(d, sep='/')
{'a/b': 2, 'c': 3}
```

### dol.paths.flattened_dict_items(d, sep='.', \*, parent_path=None, visit_nested=<function <lambda>>)

Yield flattened key-value pairs from a nested dictionary.

* **Return type:**
  [`Generator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)], [`None`](https://docs.python.org/3/builtins/constants.html#None), [`None`](https://docs.python.org/3/builtins/constants.html#None)]

### dol.paths.get_attr_or_item(obj, k)

If `k` is a string, tries to get `k` as an attribute of `obj` first,
and if that fails, gets it as `obj[k]`

#### WARNING
The hardcoded priority choices of this function regarding when to try
k as an item, index, or attribute, don’t apply to every case, so you may want to
use an explicit value getter to be more robust!

# >>> d = {‘a’: [1, {‘items’: 2, ‘3’: 33, 3: 42}]}

```pycon
>>> get_attr_or_item({'items': 2}, 'items')
2
```

But if “items” is not there as a key of the object, the attribute is found:

```pycon
>>> get_attr_or_item({'not_items': 2}, 'items')
<built-in method items of dict object...>
```

Both integers and string integers will work to get an item if obj is not a Mapping.

```pycon
>>> get_attr_or_item([7, 21, 42], 2)
42
>>> get_attr_or_item([7, 21, 42], '2')
42
```

If you’re dealling with a Mapping, you can get both integer and string keys, and
if you have both types in your Mapping, you’ll get the right one!

```pycon
>>> get_attr_or_item({2: 'numerical key', '2': 'string key'}, 2)
'numerical key'
>>> get_attr_or_item({2: 'numerical key', '2': 'string key'}, '2')
'string key'
```

If you don’t have the numerical version, the string version will still find your
numerical key.

```pycon
>>> get_attr_or_item({2: 'numerical key'}, '2')
'numerical key'
```

The opposite is not true though: If you ask for an integer key, it will not find
a string version of it.

```pycon
>>> get_attr_or_item({'2': 'string key'}, 2) # +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
...
KeyError: 2
```

### dol.paths.getitem(obj, k)

Return `obj[k]`.

### dol.paths.handle_prefixes(store=None, , prefix=None, filter_prefix=True, relativize_prefix=True, default_prefix='', \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

A store decorator that handles prefixes.

If aggregates several prefix-related functionalities. It will (by default)

- Filter the store so that only the keys starting with given prefix are accessible.
- Relativize the keys (provide a view where the prefix is removed from the keys)

* **Parameters:**
  * **store** – The store to wrap
  * **prefix** – The prefix to use. If None and the store is an instance (not type),
    will take the longest common prefix as the prefix.
  * **filter_prefix** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to filter out keys that don’t start with the prefix
  * **relativize_prefix** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to relativize the prefix
  * **default_prefix** – The default prefix to use if no prefix is given and the store
    is a type (not instance)

```pycon
>>> d = {'/ROOT/of/every/thing': 42, '/ROOT/of/this/too': 0}
>>> dd = handle_prefixes(d, prefix='/ROOT/of/')
>>> dd['foo'] = 'bar'
>>> dict(dd.items())  # gives us what you would expect
{'every/thing': 42, 'this/too': 0, 'foo': 'bar'}
>>> dict(dd.store)  # but see where the underlying store actually wrote 'bar':
{'/ROOT/of/every/thing': 42, '/ROOT/of/this/too': 0, '/ROOT/of/foo': 'bar'}
```

### dol.paths.identity(x)

Return `x`.

### dol.paths.keys_and_indices_path(str_path, , sep='.', index_pattern='\\\\[(\\\\d+)\\\\]')

Transforms a string path separated by a specified separator into a tuple
of keys and indices. Bracketed indices are extracted as integers.

This function is meant to be used in as the key_transformer argument of path_get etc.

* **Parameters:**
  * **path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The input path string, e.g., “a21-59c.message[2].user”.
  * **sep** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The separator used to split the path, default is ‘.’.
  * **index_pattern** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The regular expression pattern to match bracketed indices
* **Returns:**
  A tuple representation of the path, e.g., (“a21-59c”, “message”, 2, “user”).
* **Return type:**
  [*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)

### Example

```pycon
>>> keys_and_indices_path("a21-59c.message[2].user")
('a21-59c', 'message', 2, 'user')
```

### dol.paths.leaf_paths(d, sep='.', \*, parent_path=None, egress=<class 'dict'>)

Get a dictionary of leaf paths of a nested dictionary.

Given a nested dictionary, returns a similarly structured dictionary where each
leaf value is replaced by its flattened path. The ‘sep’ parameter can be either
a string or a callable.

Original use case: You used flatten_dict to flatten a nested dictionary, referencing
your values with paths, but maybe you’d like to know what the paths that your
nested dictionary is going to flatten to are. This function does that.
The output is a dict with the same keys and structure as the input, but the leaf
values are replaced by the paths that would be used to access them in a flat dict.

* **Parameters:**
  * **d** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), `Union`[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), `Union`[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`), NestedMapping[KT, VT]]]]]) – The nested dictionary to get the leaf paths from
  * **sep** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Path`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Path`)]]) – The separator to use for joining keys, or a function that takes a path and
    a key and returns a new path.
  * **parent_path** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Path`)]) – The path to the parent of the current dict
  * **egress** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Generator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)], [`None`](https://docs.python.org/3/builtins/constants.html#None), [`None`](https://docs.python.org/3/builtins/constants.html#None)]], [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]) – A function that takes a generator of key-value pairs and returns a mapping
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), `Union`[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Path`)]]

### Example

```pycon
>>> leaf_paths({'a': {'b': 2}, 'c': 3})
{'a': {'b': 'a.b'}, 'c': 'c'}
```

```pycon
>>> leaf_paths({'a': {'b': 2}, 'c': 3}, sep="/")
{'a': {'b': 'a/b'}, 'c': 'c'}
```

```pycon
>>> leaf_paths({'a': {'b': 2}, 'c': 3}, sep=lambda p, k: f"{p}-{k}" if p else k)
{'a': {'b': 'a-b'}, 'c': 'c'}
```

### dol.paths.mk_relative_path_store(store_cls=None, , name=None, with_key_validation=False, prefix_attr='_prefix', \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

* **Parameters:**
  * **store_cls** – The base store to wrap (subclass)
  * **name** – The name of the new store (by default ‘RelPath’ + store_cls._\_name_\_)
  * **with_key_validation** – Whether keys should be validated upon access (store_cls must have an is_valid_key method
* **Returns:**
  A new class that uses relative paths (i.e. where \_prefix is automatically added to incoming keys,
  and the len(_prefix) first characters are removed from outgoing keys.

```pycon
>>> # The dynamic way (if you try this at home, be aware of the pitfalls of the dynamic way
>>> # -- but don't just believe the static dogmas).
>>> MyStore = mk_relative_path_store(dict)  # wrap our favorite store: A dict.
>>> s = MyStore()  # make such a store
>>> s._prefix = '/ROOT/'
>>> s['foo'] = 'bar'
>>> dict(s.items())  # gives us what you would expect
{'foo': 'bar'}
>>>  # but under the hood, the dict we wrapped actually contains the '/ROOT/' prefix
>>> dict(s.store)
{'/ROOT/foo': 'bar'}
>>>
>>> # The static way: Make a class that will integrate the _prefix at construction time.
>>> class MyStore(mk_relative_path_store(dict)):  # Indeed, mk_relative_path_store(dict) is a class you can subclass
...     def __init__(self, _prefix, *args, **kwargs):
...         self._prefix = _prefix
```

You can choose the name you want that prefix to have as an attribute (we’ll still make
a hidden ‘_prefix’ attribute for internal use, but at least you can have an attribute with the
name you want.

```pycon
>>> MyRelStore = mk_relative_path_store(dict, prefix_attr='rootdir')
>>> s = MyRelStore()
>>> s.rootdir = '/ROOT/'
```

```pycon
>>> s['foo'] = 'bar'
>>> dict(s.items())  # gives us what you would expect
{'foo': 'bar'}
>>>  # but under the hood, the dict we wrapped actually contains the '/ROOT/' prefix
>>> dict(s.store)
{'/ROOT/foo': 'bar'}
```

### dol.paths.path_edit(d, edits=())

Make a series of (in place) edits to a Mapping, specifying `(path, value)` pairs.

* **Parameters:**
  * **d** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – The mapping to edit.
  * **edits** (`Union`[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Path`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Path`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)]]]) – An iterable of `(path, value)` tuples, or `path: value` Mapping.
* **Returns:**
  The edited mapping.
* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

```pycon
>>> d = {'a': 1}
>>> path_edit(d, [(['b', 'c'], 2), ('d.e.f', 3)])
{'a': 1, 'b': {'c': 2}, 'd': {'e': {'f': 3}}}
```

Changes happened also inplace (so if you don’t want that, make a deepcopy first):

```pycon
>>> d
{'a': 1, 'b': {'c': 2}, 'd': {'e': {'f': 3}}}
```

You can also pass a dict of edits.

```pycon
>>> path_edit(d, {'a': 4, 'd.e.f': 5})
{'a': 4, 'b': {'c': 2}, 'd': {'e': {'f': 5}}}
```

### dol.paths.path_filter(pkv_filt, d, , leafs_only=True, breadth_first=False)

Walk a dict, yielding paths to values that pass the `pkv_filt`

* **Parameters:**
  * **pkv_filt** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`PT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – A function that takes a path, key, and value, and returns
    `True` if the path should be yielded, and `False` otherwise
  * **d** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – The `Mapping` to walk (scan through)
  * **leafs_only** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to yield only paths to leafs (default), or to yield
    paths to all values that pass the `pkv_filt`.
  * **breadth_first** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to perform breadth-first traversal
    (instead of the default depth-first traversal).
* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`PT`)]
* **Returns:**
  An iterator of paths to values that pass the `pkv_filt`

### Example

```pycon
>>> d = {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
>>> list(path_filter(lambda p, k, v: v == 2, d))
[('a', 'b', 'd')]
```

```pycon
>>> mm = {
...     'a': {'b': {'c': 42}},
...     'aa': {'bb': {'cc': 'meaning of life'}},
...     'aaa': {'bbb': 314},
... }
>>> return_path_if_int_leaf = lambda p, k, v: (p, v) if isinstance(v, int) else None
>>> paths = list(path_filter(return_path_if_int_leaf, mm))
>>> paths  # only the paths to the int leaves are returned
[('a', 'b', 'c'), ('aaa', 'bbb')]
```

The `pkv_filt` argument can use path, key, and/or value to define your search
query. For example, let’s extract all the paths that have depth at least 3.

```pycon
>>> paths = list(path_filter(lambda p, k, v: len(p) >= 3, mm))
>>> paths
[('a', 'b', 'c'), ('aa', 'bb', 'cc')]
```

The rationale for `path_filter` yielding matching paths, and not values or keys,
is that if you have the paths, you can than get the keys and values with them,
using `path_get`.

```pycon
>>> from functools import partial, reduce
>>> path_get = lambda m, k: reduce(lambda m, k: m[k], k, m)
>>> extract_paths = lambda m, paths: map(partial(path_get, m), paths)
>>> vals = list(extract_paths(mm, paths))
>>> vals
[42, 'meaning of life']
```

#### NOTE
pkv_filt is first to match the order of the arguments of the
builtin filter function.

### dol.paths.path_get(obj, path, on_error=<function raise_on_error>, \*, sep=None, key_transformer=None, get_value=<function get_attr_or_item>, caught_errors=(<class 'Exception'>, ))

Get elements of a mapping through a path to be called recursively.

* **Parameters:**
  * **obj** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The object to get the path from
  * **path** – The path to get
  * **on_error** (`Union`[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The error handler to use (default: raise_on_error)
  * **sep** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Determines a path is transforms into a tuple of keys.
    If it’s a string, `lambda path: path.split(sep)` is used.
    If not, it should be a function which takes in a path object and returns an iterable of keys.
  * **key_transformer** – A function to transform the keys of the path
  * **get_value** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – A function to get the value of a key in a mapping
  * **caught_errors** – The errors to catch (default: Exception)

It will

- split a path into keys (if sep is given, or if path is a string, will use ‘.’ as a separator by default)
- if key_transformer is given, apply to each key
- consider string keys that are numeric as ints (convenient for lists)
- get items also as attributes (attributes are checked for first for string keys)
- catch all exceptions (that are subclasses of `Exception`)

```pycon
>>> class A:
...      an_attribute = 42
>>> path_get([1, [4, 5, {'a': A}], 3], [1, 2, 'a', 'an_attribute'])
42
```

By default, if `path` is a string, it will be split on `sep`,
which is `'.'` by default.

```pycon
>>> path_get([1, [4, 5, {'a': A}], 3], '1.2.a.an_attribute')
42
```

#### NOTE
The underlying function is `_path_get`, but `path_get` has defaults and
flexible input processing for more convenience.

#### NOTE
`path_get` contains some ready-made `OnErrorType` functions in its
attributes. For example, see how we can make `path_get` have the same behavior
as `dict.get` by passing `path_get.return_none_on_error` as `on_error`:

```pycon
>>> dd = path_get({}, 'no.keys', on_error=path_get.return_none_on_error)
>>> dd is None
True
```

For example, `path_get.raise_on_error`,
`path_get.return_none_on_error`, and `path_get.return_empty_tuple_on_error`.

### dol.paths.paths_getter(paths, obj=None, \*, egress=<class 'dict'>, on_error=<function raise_on_error>, sep=None, key_transformer=None, get_value=<function get_attr_or_item>, caught_errors=(<class 'Exception'>, ))

Returns (path, values) pairs of the given paths in the given object.
This is the “fan-out” version of `path_get`, specifically designed to
get multiple paths, returning the (path, value) pairs in a dict (by default),
or via any pairs aggregator (`egress`) function.

#### NOTE
For reasons who’s clarity is burried in historical legacy, the order of
obj and path are the opposite of path_get.

* **Parameters:**
  * **paths** – The paths to get
  * **obj** – The object to get the paths from
  * **egress** – The egress function to use (default: dict)
  * **on_error** (`Union`[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The error handler to use (default: raise_on_error)
  * **sep** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The separator to use if the path is a string
  * **key_transformer** – A function to transform the keys of the path
  * **get_value** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – A function to get the value of a key in a mapping
  * **caught_errors** – The errors to catch (default: Exception)

```pycon
>>> obj = {'a': {'b': 1, 'c': 2}, 'd': 3}
>>> paths = ['a.c', 'd']
>>> paths_getter(paths, obj=obj)
{'a.c': 2, 'd': 3}
>>> path_extractor = paths_getter(paths)
>>> path_extractor(obj)
{'a.c': 2, 'd': 3}
```

See that the paths are used as the keys of the returned dict.
If you want to specify your own keys, you can simply specify `paths` as a dict
whose keys are the keys you want, and whose values are the paths to get:

```pycon
>>> path_extractor_2 = paths_getter({'california': 'a.c', 'dreaming': 'd'})
>>> path_extractor_2(obj)
{'california': 2, 'dreaming': 3}
```

### dol.paths.prefixless_view(store=None, , prefix=None, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Wrap `store` so that keys are seen without `prefix` (added back on access).

### dol.paths.raise_on_error(d)

`on_error` policy for `path_get`: re-raise the caught error.

### dol.paths.rel_path_wrap(o, \_prefix)

* **Parameters:**
  * **o** – An object to be wrapped
  * **\_prefix** – The \_prefix to use for key wrapping (will remove it from outcoming keys and add to ingoing keys.

```pycon
>>> # The dynamic way (if you try this at home, be aware of the pitfalls of the dynamic way
>>> # -- but don't just believe the static dogmas).
>>> d = {'/ROOT/of/every/thing': 42, '/ROOT/of/this/too': 0}
>>> dd = rel_path_wrap(d, '/ROOT/of/')
>>> dd['foo'] = 'bar'
>>> dict(dd.items())  # gives us what you would expect
{'every/thing': 42, 'this/too': 0, 'foo': 'bar'}
>>>  # but under the hood, the dict we wrapped actually contains the '/ROOT/' prefix
>>> dict(dd.store)
{'/ROOT/of/every/thing': 42, '/ROOT/of/this/too': 0, '/ROOT/of/foo': 'bar'}
>>>
>>> # The static way: Make a class that will integrate the _prefix at construction time.
>>> class MyStore(mk_relative_path_store(dict)):  # Indeed, mk_relative_path_store(dict) is a class you can subclass
...     def __init__(self, _prefix, *args, **kwargs):
...         self._prefix = _prefix
```

### dol.paths.return_empty_tuple_on_error(d)

`on_error` policy for `path_get`: return `()`.

### dol.paths.return_none_on_error(d)

`on_error` policy for `path_get`: return `None`.

### dol.paths.search_paths(d, pkv_filt, , leafs_only=True, breadth_first=False)

backwards compatibility quasi-alias (arguments are flipped)
Use path_filter instead, since search_paths will be deprecated.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`PT`)]

### dol.paths.separate_keys_with_separator(obj, sep='.')

Split a string path on `sep` and cast numeric parts to `int`; a non-string iterable is only cast element-wise.

### dol.paths.separator_based_path_extender(path, key, sep)

Extends a given path with a new key using the specified separator.
If the path is empty, the key is returned as is.

* **Return type:**
  [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Path`)

### dol.paths.split_if_str(obj, sep='.')

Split `obj` on `sep` if it is a string; return it unchanged otherwise.

### dol.paths.str_template_key_trans(template, key_type, format_dict=None, process_kwargs=None, process_info_dict=None, named_tuple_type_name='NamedTuple', sep='/')

Make a key trans object that translates from a string \_id to a dict, tuple, or namedtuple key (and back)

### dol.paths.string_unparse(parsing_result)

The inverse of string.Formatter.parse

Will ravel

```pycon
>>> import string
>>> formatter = string.Formatter()
>>> string_unparse(formatter.parse('literal{name!c:spec}'))
'literal{name!c:spec}'
```
