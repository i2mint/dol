# dol.util

General util objects: function composition, grouping, partial classes, file helpers.

Main entry points:

- `Pipe`: compose functions left to right
- `partialclass`: `functools.partial` for classes
- `groupby`, `regroupby`, `igroupby`: group items by a key function
- `chain_get`: first value found for a sequence of keys
- `written_bytes`, `read_from_bytes`: turn file-writing/reading functions into bytes codecs
  ```pycon
  >>> from dol.util import Pipe
  >>> Pipe(lambda x: x + 1, str)(1)
  '2'
  ```

### Functions

| [`add_as_attribute_of`](#dol.util.add_as_attribute_of)(obj[, name])                   | Decorator that adds a function as an attribute of a container object `obj`.                                                                                                                                                                                                                                                                                                                                                  |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`add_attrs`](#dol.util.add_attrs)([remember_added_attrs, if_attr_exists])  | Make a function that will add attributes to an obj.                                                                                                                                                                                                                                                                                                                                                                          |
| `attrs_of`(obj)                                                                                     |                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [`chain_get`](#dol.util.chain_get)(d, keys[, default])                      | Returns the `d[key]` value for the first `key` in `keys` that is in `d`, and default if none are found                                                                                                                                                                                                                                                                                                                       |
| [`copy_attrs`](#dol.util.copy_attrs)(target, source, attrs[, ...])           | Copy attributes from one object to another.                                                                                                                                                                                                                                                                                                                                                                                  |
| `copy_attrs_from`(from_obj, to_obj, attrs)                                                          |                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [`decorate_callables`](#dol.util.decorate_callables)(decorator[, cls])               | Decorate all (non-underscored) callables in a class with a decorator.                                                                                                                                                                                                                                                                                                                                                        |
| `delegate_as`(delegate_cls[, to, include, exclude])                                                 |                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [`fill_with_dflts`](#dol.util.fill_with_dflts)(d[, dflt_dict])                    | Fed up with multiline handling of dict arguments? Fed up of repeating the if d is None: d = {} lines ad nauseam (because defaults can't be dicts as a default because dicts are mutable blah blah, and the python kings don't seem to think a mutable dict is useful enough)? Well, my favorite solution would be a built-in handling of the problem of complex/smart defaults, that is visible in the code and in the docs. |
| [`flatten_pipe`](#dol.util.flatten_pipe)(pipe)                                 | Unravel nested Pipes to get a flat 'sequence of functions' version of input.                                                                                                                                                                                                                                                                                                                                                 |
| [`format_invocation`](#dol.util.format_invocation)([name, args, kwargs])            | Given a name, positional arguments, and keyword arguments, format a basic Python-style function call.                                                                                                                                                                                                                                                                                                                        |
| `fullpath`(path)                                                                                    |                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `function_info_string`(func)                                                                        |                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [`get_app_folder`](#dol.util.get_app_folder)([folder_kind])                      | Get the full path of a directory suitable for storing application-specific configs, (or data, or cache, or state or runtime)                                                                                                                                                                                                                                                                                                 |
| [`groupby`](#dol.util.groupby)(items, key[, val, group_factory])          | Groups items according to group keys updated from those items through the given `key` function (mapping an item to its group key).                                                                                                                                                                                                                                                                                           |
| [`has_enabled_clear_method`](#dol.util.has_enabled_clear_method)(store)                    | Returns True iff obj has a clear method that is enabled (i.e. not disabled).                                                                                                                                                                                                                                                                                                                                                 |
| `identity_func`(x)                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [`igroupby`](#dol.util.igroupby)(items, key[, val, group_factory, ...])    | The generator version of dol groupby.                                                                                                                                                                                                                                                                                                                                                                                        |
| [`inject_method`](#dol.util.inject_method)(obj, method_function[, ...])         | method_function could be:                                                                                                                                                                                                                                                                                                                                                                                                    |
| [`instance_checker`](#dol.util.instance_checker)(\*types)                          | Makes a filter function that checks the type of an object.                                                                                                                                                                                                                                                                                                                                                                   |
| [`invertible_maps`](#dol.util.invertible_maps)([mapping, inv_mapping])            | Returns two maps that are inverse of each other.                                                                                                                                                                                                                                                                                                                                                                             |
| [`is_classmethod`](#dol.util.is_classmethod)(obj)                                | Checks if an object is a classmethod.                                                                                                                                                                                                                                                                                                                                                                                        |
| [`is_unbound_method`](#dol.util.is_unbound_method)(obj)                             | Determines if the given object is an unbound method.                                                                                                                                                                                                                                                                                                                                                                         |
| [`max_common_prefix`](#dol.util.max_common_prefix)(a, \*[, default])                | Given a list of strings (or other sliceable seq), returns the longest common prefix                                                                                                                                                                                                                                                                                                                                          |
| `move_files_of_folder_to_trash`(folder)                                                             |                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [`named_partial`](#dol.util.named_partial)(func, \*args[, \_\_name_\_])         | functools.partial, but with a \_\_name_\_                                                                                                                                                                                                                                                                                                                                                                                    |
| `nest_in_dict`(keys, values)                                                                        |                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [`non_colliding_key`](#dol.util.non_colliding_key)(key, exclude, \*[, ...])         | Return a key not present in the exclude container.                                                                                                                                                                                                                                                                                                                                                                           |
| [`norm_kv_filt`](#dol.util.norm_kv_filt)(kv_filt)                              | Prepare a boolean function to be used with `filter` when fed an iterable of (k, v) pairs.                                                                                                                                                                                                                                                                                                                                    |
| [`not_a_mac_junk_path`](#dol.util.not_a_mac_junk_path)(path)                          | A function that will tell you if the path is not a mac junk path/ More precisely, doesn't end with '.DS_Store' or have a `__MACOSX` folder somewhere on it's way.                                                                                                                                                                                                                                                            |
| `ntup`(\*\*kwargs)                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [`num_of_args`](#dol.util.num_of_args)(func)                                  | Number of arguments (parameters) of the function.                                                                                                                                                                                                                                                                                                                                                                            |
| [`num_of_required_args`](#dol.util.num_of_required_args)(func)                         | Number or REQUIRED arguments of a function.                                                                                                                                                                                                                                                                                                                                                                                  |
| [`partialclass`](#dol.util.partialclass)(cls, \*args, \*\*kwargs)              | What `partial(cls, *args, **kwargs)` does, but returning a class instead of an object.                                                                                                                                                                                                                                                                                                                                       |
| [`read_from_bytes`](#dol.util.read_from_bytes)(file_reader[, obj, ...])           | Takes a file reading function that expects a file-like object, and returns a function that instead of reading from a file, reads from bytes.                                                                                                                                                                                                                                                                                 |
| [`regroupby`](#dol.util.regroupby)(items, \*key_funcs, \*\*named_key_funcs) | Recursive groupby.                                                                                                                                                                                                                                                                                                                                                                                                           |
| [`safe_compile`](#dol.util.safe_compile)(path[, normalize_path])               | Compile a *literal file path* into a regex pattern that matches that path, normalizing separators and escaping regex-special characters on Windows.                                                                                                                                                                                                                                                                          |
| `signature_string_or_default`(func[, default])                                                      |                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `single_nest_in_dict`(key, value)                                                                   |                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `static_identity_method`(x)                                                                         |                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [`str_to_var_str`](#dol.util.str_to_var_str)(s)                                  | Make a valid python variable string from the input string.                                                                                                                                                                                                                                                                                                                                                                   |
| [`truncate_string_with_marker`](#dol.util.truncate_string_with_marker)(s, \*[, ...])          | Return a string with a limited length.                                                                                                                                                                                                                                                                                                                                                                                       |
| `write_to_file`(obj, key)                                                                           |                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [`written_bytes`](#dol.util.written_bytes)(file_writer[, obj, ...])             | Takes a file writing function that expects an object and a file-like object, and returns a function that instead of writing to a file, returns the bytes that would have been written.                                                                                                                                                                                                                                       |
| [`written_key`](#dol.util.written_key)([obj, writer, key, ...])               | Writes an object to a key and returns the key.                                                                                                                                                                                                                                                                                                                                                                               |

### Classes

| [`AttributeMapping`](#dol.util.AttributeMapping)                  | A read-only mapping with attribute access.                                  |
|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`AttributeMutableMapping`](#dol.util.AttributeMutableMapping)           | A mutable mapping that provides both attribute and dictionary-style access. |
| `DelegatedAttribute`(delegate_name, attr_name)                                     |                                                                             |
| [`FolderSpec`](#dol.util.FolderSpec)(env_var, default_path) |                                                                             |
| `HashableMixin`()                                                                  |                                                                             |
| `ImmutableMixin`()                                                                 |                                                                             |
| [`LiteralVal`](#dol.util.LiteralVal)(val)                   | An object to indicate that the value should be considered literally.        |
| `ModuleNotFoundErrorNiceMessage`([msg])                                            |                                                                             |
| `ModuleNotFoundIgnore`()                                                           |                                                                             |
| `ModuleNotFoundWarning`([msg])                                                     |                                                                             |
| `MutableStruct`(\*\*attr_val_dict)                                                 |                                                                             |
| [`Pipe`](#dol.util.Pipe)(\*funcs, \*\*named_funcs)    | Simple function composition.                                                |
| `SimpleProperty`()                                                                 |                                                                             |
| `Struct`(\*\*attr_val_dict)                                                        |                                                                             |
| [`imdict`](#dol.util.imdict)                            | A frozen hashable dict                                                      |
| [`lazyprop`](#dol.util.lazyprop)(func)                    | A descriptor implementation of lazyprop (cached property).                  |
| [`lazyprop_w_sentinel`](#dol.util.lazyprop_w_sentinel)(func)         | A descriptor implementation of lazyprop (cached property).                  |
| [`staticproperty`](#dol.util.staticproperty)(function)          | A decorator for defining static properties in classes.                      |

### *class* dol.util.AttributeMapping

Bases: [`SimpleNamespace`](https://docs.python.org/3/library/types.html#types.SimpleNamespace), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

A read-only mapping with attribute access.

Useful when you want mapping interface but don’t need mutation.

### Examples

```pycon
>>> ns = AttributeMapping(x=10, y=20)
>>> ns.x
10
>>> ns['y']
20
>>> list(ns)
['x', 'y']
```

#### *classmethod* from_mapping(mapping)

Create an AttributeMapping from a regular mapping.

This is useful when you want to convert a dictionary or other mapping
into an AttributeMapping for attribute-style access.

* **Return type:**
  [`AttributeMapping`](#dol.util.AttributeMapping)

### *class* dol.util.AttributeMutableMapping

Bases: [`AttributeMapping`](#dol.util.AttributeMapping), [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

A mutable mapping that provides both attribute and dictionary-style access.

Extends AttributeMapping with mutation capabilities,
ensuring proper error handling and protocol compliance.

### Examples

```pycon
>>> ns = AttributeMutableMapping(apple=1, banana=2)
>>> ns.apple
1
>>> ns['banana']
2
>>> ns['cherry'] = 3
>>> ns.cherry
3
>>> list(ns)
['apple', 'banana', 'cherry']
>>> len(ns)
3
>>> 'apple' in ns
True
>>> del ns['banana']
>>> 'banana' in ns
False
```

### *class* dol.util.FolderSpec(env_var, default_path)

Bases: [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)

#### default_path

Alias for field number 1

#### env_var

Alias for field number 0

### *class* dol.util.LiteralVal(val)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

An object to indicate that the value should be considered literally.

```pycon
>>> t = LiteralVal(42)
>>> t.get_val()
42
>>> t()
42
```

#### get_val()

Get the value wrapped by LiteralVal instance.

One might want to use `literal.get_val()` instead `literal()` to get the
value a `LiteralVal` is wrapping because `.get_val` is more explicit.

That said, with a bit of hesitation, we allow the `literal()` form as well
since it is useful in situations where we need to use a callback function to
get a value.

### *class* dol.util.Pipe(\*funcs, \*\*named_funcs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Simple function composition. That is, gives you a callable that implements input -> f_1 -> … -> f_n -> output.

```pycon
>>> def foo(a, b=2):
...     return a + b
>>> f = Pipe(foo, lambda x: print(f"x: {x}"))
>>> f(3)
x: 5
>>> len(f)
2
```

You can name functions, but this would just be for documentation purposes.
The names are completely ignored.

```pycon
>>> g = Pipe(
...     add_numbers = lambda x, y: x + y,
...     multiply_by_2 = lambda x: x * 2,
...     stringify = str
... )
>>> g(2, 3)
'10'
>>> len(g)
3
```

### Notes

- Pipe instances don’t have a \_\_name_\_ etc. So some expectations of normal functions are not met.
- Pipe instance are pickalable (as long as the functions that compose them are)

You can specify a single functions:

```pycon
>>> Pipe(lambda x: x + 1)(2)
3
```

but

```pycon
>>> Pipe()
Traceback (most recent call last):
  ...
ValueError: You need to specify at least one function!
```

You can specify an instance name and/or doc with the special (reserved) argument
names `__name__` and `__doc__` (which therefore can’t be used as function names):

```pycon
>>> f = Pipe(map, add_it=sum, __name__='map_and_sum', __doc__='Apply func and add')
>>> f(lambda x: x * 10, [1, 2, 3])
60
>>> f.__name__
'map_and_sum'
>>> f.__doc__
'Apply func and add'
```

### dol.util.add_as_attribute_of(obj, name=None)

Decorator that adds a function as an attribute of a container object `obj`.

If no `name` is given, the `__name__` of the function will be used, with a
leading underscore removed. This is useful for adding helper functions to main
“container” functions without polluting the namespace of the module, at least
from the point of view of imports and tab completion.

```pycon
>>> def foo():
...    pass
>>>
>>> @add_as_attribute_of(foo)
... def _helper():
...    pass
>>> hasattr(foo, 'helper')
True
>>> callable(foo.helper)
True
```

In reality, any object that has a `__name__` can be added to the attribute of
`obj`, but the intention is to add helper functions to main “container” functions.

### dol.util.add_attrs(remember_added_attrs=True, if_attr_exists='raise', \*\*attrs)

Make a function that will add attributes to an obj.
Originally meant to be used as a decorator of a function, to inject

```pycon
>>> from dol.util import add_attrs
>>> @add_attrs(bar='bituate', hello='world')
... def foo():
...     pass
>>> [x for x in dir(foo) if not x.startswith('_')]
['bar', 'hello']
>>> foo.bar
'bituate'
>>> foo.hello
'world'
>>> foo._added_attrs  # Another attr was added to hold the list of attributes added (in case we need to remove them
['bar', 'hello']
```

### dol.util.chain_get(d, keys, default=None)

Returns the `d[key]` value for the first `key` in `keys` that is in `d`, and default if none are found

#### NOTE
Think of `collections.ChainMap` where you can look for a single key in a sequence of maps until we find it.
Here we look for a sequence of keys in a single map, stopping as soon as we find a key that the map has.

```pycon
>>> d = {'here': '&', 'there': 'and', 'every': 'where'}
>>> chain_get(d, ['not there', 'not there either', 'there', 'every'])
'and'
```

Notice how `'not there'` and `'not there either'` are skipped, `'there'` is found and used to retrieve
the value, and `'every'` is not even checked (because `'there'` was found).
If non of the keys are found, `None` is returned by default.

```pycon
>>> assert chain_get(d, ('none', 'of', 'these')) is None
```

You can change this default though:

```pycon
>>> chain_get(d, ('none', 'of', 'these'), default='Not Found')
'Not Found'
```

### dol.util.copy_attrs(target, source, attrs, raise_error_if_an_attr_is_missing=True)

Copy attributes from one object to another.

```pycon
>>> class A:
...     x = 0
>>> class B:
...     x = 1
...     yy = 2
...     zzz = 3
>>> dict_of = lambda o: {a: getattr(o, a) for a in dir(A) if not a.startswith('_')}
>>> dict_of(A)
{'x': 0}
>>> copy_attrs(A, B, 'yy')
>>> dict_of(A)
{'x': 0, 'yy': 2}
>>> copy_attrs(A, B, ['x', 'zzz'])
>>> dict_of(A)
{'x': 1, 'yy': 2, 'zzz': 3}
```

But if you try to copy something that `B` (the source) doesn’t have, copy_attrs will complain:

```pycon
>>> copy_attrs(A, B, 'this_is_not_an_attr')
Traceback (most recent call last):
    ...
AttributeError: type object 'B' has no attribute 'this_is_not_an_attr'
```

If you tell it not to complain, it’ll just ignore attributes that are not in source.

```pycon
>>> copy_attrs(A, B, ['nothing', 'here', 'exists'], raise_error_if_an_attr_is_missing=False)
>>> dict_of(A)
{'x': 1, 'yy': 2, 'zzz': 3}
```

### dol.util.decorate_callables(decorator, cls=None)

Decorate all (non-underscored) callables in a class with a decorator.

```pycon
>>> from dol.util import LiteralVal
>>> @decorate_callables(property)
... class A:
...     def wet(self):
...         return 'dry'
...     @LiteralVal
...     def big(self):
...         return 'small'
>>> a = A()
>>> a.wet
'dry'
>>> a.big()
'small'
```

### dol.util.fill_with_dflts(d, dflt_dict=None)

Fed up with multiline handling of dict arguments?
Fed up of repeating the if d is None: d = {} lines ad nauseam (because defaults can’t be dicts as a default
because dicts are mutable blah blah, and the python kings don’t seem to think a mutable dict is useful enough)?
Well, my favorite solution would be a built-in handling of the problem of complex/smart defaults,
that is visible in the code and in the docs. But for now, here’s one of the tricks I use.

Main use is to handle defaults of function arguments. Say you have a function `func(d=None)` and you want
`d` to be a dict that has at least the keys `foo` and `bar` with default values 7 and 42 respectively.
Then, in the beginning of your function code you’ll say:

> d = fill_with_dflts(d, {‘a’: 7, ‘b’: 42})

See examples to know how to use it.

#### ATTENTION
A shallow copy of the dict is made. Know how that affects you (or not).

#### ATTENTION
This is not recursive: It won’t be filling any nested fields with defaults.

* **Parameters:**
  * **d** – The dict you want to “fill”
  * **dflt_dict** – What to fill it with (a {k: v, …} dict where if k is missing in d, you’ll get a new field k, with
    value v.
* **Returns:**
  val entries (if the key was missing in d).
* **Return type:**
  a dict with the new key

```pycon
>>> fill_with_dflts(None)
{}
>>> fill_with_dflts(None, {'a': 7, 'b': 42})
{'a': 7, 'b': 42}
>>> fill_with_dflts({}, {'a': 7, 'b': 42})
{'a': 7, 'b': 42}
>>> fill_with_dflts({'b': 1000}, {'a': 7, 'b': 42})
{'a': 7, 'b': 1000}
```

### dol.util.flatten_pipe(pipe)

Unravel nested Pipes to get a flat ‘sequence of functions’ version of input.

```pycon
>>> def f(x): return x + 1
>>> def g(x): return x * 2
>>> def h(x): return x - 3
>>> a = Pipe(f, g, h)
>>> b = Pipe(f, Pipe(g, h))
>>> len(a)
3
>>> len(b)
2
>>> c = flatten_pipe(b)
>>> len(c)
3
>>> assert a(10) == b(10) == c(10) == 19
```

### dol.util.format_invocation(name='', args=(), kwargs=None)

Given a name, positional arguments, and keyword arguments, format
a basic Python-style function call.

```pycon
>>> print(format_invocation('func', args=(1, 2), kwargs={'c': 3}))
func(1, 2, c=3)
>>> print(format_invocation('a_func', args=(1,)))
a_func(1)
>>> print(format_invocation('kw_func', kwargs=[('a', 1), ('b', 2)]))
kw_func(a=1, b=2)
```

### dol.util.get_app_config_folder(, folder_kind='config')

Get the full path of a directory suitable for storing application-specific configs,
(or data, or cache, or state or runtime)

On Windows, this is typically %APPDATA%.
On macOS, this is typically ~/.config.
On Linux, this is typically ~/.config.

* **Parameters:**
  **folder_kind** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The kind of folder to get. One of ‘config’, ‘data’, ‘cache’, ‘state’, ‘runtime’.
  Defaults to ‘config’.
  Here are concise explanations for each folder kind:
  **config**: User preferences and settings files (e.g., API keys, theme preferences, editor settings). Files users might edit manually or that define how the app behaves.
  **data**: Essential user-created content and application state (e.g., databases, saved games, user documents, session files). Data that should be backed up and persists across updates.
  **cache**: Temporary, regeneratable files (e.g., downloaded images, compiled assets, web cache). Can be safely deleted to free space without losing user work.
  **state**: Application state and logs that persist between sessions but aren’t critical user data (e.g., command history, undo history, recently opened files, log files). Unlike cache, shouldn’t be auto-deleted.
  **runtime**: Temporary runtime files that only exist while the app runs (e.g., PID files, Unix sockets, lock files, named pipes). Typically cleared on logout/reboot.
  **TL;DR**: config = settings, data = user files, cache = disposable, state = logs/history, runtime = process files.
* **Returns:**
  The full path of the app data folder.
* **Return type:**
  [*str*](https://docs.python.org/3/builtins/stdtypes.html#str)

See [https://github.com/i2mint/i2mint/issues/1](https://github.com/i2mint/i2mint/issues/1).

### dol.util.get_app_data_folder(, folder_kind='data')

Get the full path of a directory suitable for storing application-specific configs,
(or data, or cache, or state or runtime)

On Windows, this is typically %APPDATA%.
On macOS, this is typically ~/.config.
On Linux, this is typically ~/.config.

* **Parameters:**
  **folder_kind** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The kind of folder to get. One of ‘config’, ‘data’, ‘cache’, ‘state’, ‘runtime’.
  Defaults to ‘config’.
  Here are concise explanations for each folder kind:
  **config**: User preferences and settings files (e.g., API keys, theme preferences, editor settings). Files users might edit manually or that define how the app behaves.
  **data**: Essential user-created content and application state (e.g., databases, saved games, user documents, session files). Data that should be backed up and persists across updates.
  **cache**: Temporary, regeneratable files (e.g., downloaded images, compiled assets, web cache). Can be safely deleted to free space without losing user work.
  **state**: Application state and logs that persist between sessions but aren’t critical user data (e.g., command history, undo history, recently opened files, log files). Unlike cache, shouldn’t be auto-deleted.
  **runtime**: Temporary runtime files that only exist while the app runs (e.g., PID files, Unix sockets, lock files, named pipes). Typically cleared on logout/reboot.
  **TL;DR**: config = settings, data = user files, cache = disposable, state = logs/history, runtime = process files.
* **Returns:**
  The full path of the app data folder.
* **Return type:**
  [*str*](https://docs.python.org/3/builtins/stdtypes.html#str)

See [https://github.com/i2mint/i2mint/issues/1](https://github.com/i2mint/i2mint/issues/1).

### dol.util.get_app_folder(folder_kind='config')

Get the full path of a directory suitable for storing application-specific configs,
(or data, or cache, or state or runtime)

On Windows, this is typically %APPDATA%.
On macOS, this is typically ~/.config.
On Linux, this is typically ~/.config.

* **Parameters:**
  **folder_kind** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'config'`, `'data'`, `'cache'`, `'state'`, `'runtime'`]) – The kind of folder to get. One of ‘config’, ‘data’, ‘cache’, ‘state’, ‘runtime’.
  Defaults to ‘config’.
  Here are concise explanations for each folder kind:
  **config**: User preferences and settings files (e.g., API keys, theme preferences, editor settings). Files users might edit manually or that define how the app behaves.
  **data**: Essential user-created content and application state (e.g., databases, saved games, user documents, session files). Data that should be backed up and persists across updates.
  **cache**: Temporary, regeneratable files (e.g., downloaded images, compiled assets, web cache). Can be safely deleted to free space without losing user work.
  **state**: Application state and logs that persist between sessions but aren’t critical user data (e.g., command history, undo history, recently opened files, log files). Unlike cache, shouldn’t be auto-deleted.
  **runtime**: Temporary runtime files that only exist while the app runs (e.g., PID files, Unix sockets, lock files, named pipes). Typically cleared on logout/reboot.
  **TL;DR**: config = settings, data = user files, cache = disposable, state = logs/history, runtime = process files.
* **Returns:**
  The full path of the app data folder.
* **Return type:**
  [*str*](https://docs.python.org/3/builtins/stdtypes.html#str)

See [https://github.com/i2mint/i2mint/issues/1](https://github.com/i2mint/i2mint/issues/1).

### dol.util.groupby(items, key, val=None, group_factory=<class 'list'>)

Groups items according to group keys updated from those items through the given
`key` function (mapping an item to its group key).

* **Parameters:**
  * **items** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – iterable of items
  * **key** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Hashable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable)]) – The function that computes a key from an item. Needs to return a hashable.
  * **val** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – An optional function that computes a val from an item. If not given, the item itself will be taken.
  * **group_factory** – The function to make new (empty) group objects and accumulate group items.
    group_items = group_factory() will be called to make a new empty group collection
    group_items.append(x) will be called to add x to that collection
    The default is `list`
* **Returns:**
  items_in_that_group, …}
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### SEE ALSO
regroupby, itertools.groupby, and dol.source.SequenceKvReader

```pycon
>>> groupby(range(11), key=lambda x: x % 3)
{0: [0, 3, 6, 9], 1: [1, 4, 7, 10], 2: [2, 5, 8]}
>>>
>>> tokens = ['the', 'fox', 'is', 'in', 'a', 'box']
>>> groupby(tokens, len)
{3: ['the', 'fox', 'box'], 2: ['is', 'in'], 1: ['a']}
>>> key_map = {1: 'one', 2: 'two'}
>>> groupby(tokens, lambda x: key_map.get(len(x), 'more'))
{'more': ['the', 'fox', 'box'], 'two': ['is', 'in'], 'one': ['a']}
>>> stopwords = {'the', 'in', 'a', 'on'}
>>> groupby(tokens, lambda w: w in stopwords)
{True: ['the', 'in', 'a'], False: ['fox', 'is', 'box']}
>>> groupby(tokens, lambda w: ['words', 'stopwords'][int(w in stopwords)])
{'stopwords': ['the', 'in', 'a'], 'words': ['fox', 'is', 'box']}
```

### dol.util.has_enabled_clear_method(store)

Returns True iff obj has a clear method that is enabled (i.e. not disabled)

### dol.util.igroupby(items, key, val=None, group_factory=<class 'list'>, group_release_cond=<function <lambda>>, release_remainding=True, append_to_group_items=<method 'append' of 'list' objects>, grouper_mapping=<class 'collections.defaultdict'>)

The generator version of dol groupby.
Groups items according to group keys updated from those items through the given `key` function (mapping an item to its group key),
yielding the groups according to a logic defined by `group_release_cond`

* **Parameters:**
  * **items** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – iterable of items
  * **key** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Hashable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable)]) – The function that computes a key from an item. Needs to return a hashable.
  * **val** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – An optional function that computes a val from an item. If not given, the item itself will be taken.
  * **group_factory** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – The function to make new (empty) group objects and accumulate group items.
    group_items = group_collector() will be called to make a new empty group collection
    group_items.append(x) will be called to add x to that collection
    The default is `list`
  * **group_release_cond** (`Union`[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Hashable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)], [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict), [`Hashable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]]) – A boolean function that will be applied, at every iteration,
    to the accumulated items of the group that was just updated,
    and determines (if True) if the (group_key, group_items) should be yielded.
    The default is False, which results in
    `lambda group_key, group_items: False` being used.
  * **release_remainding** – Once the input items have been consumed, there may still be some
    items in the grouping “cache”. `release_remainding` is a boolean that indicates whether
    the contents of this cache should be released or not.
* **Yields:**
  `(group_key, items_in_that_group)` pairs

The following will group numbers according to their parity (0 for even, 1 for odd),
releasing a list of numbers collected when that list reaches length 3:

```pycon
>>> g = igroupby(items=range(11),
...             key=lambda x: x % 2,
...             group_release_cond=lambda k, v: len(v) == 3)
>>> list(g)
[(0, [0, 2, 4]), (1, [1, 3, 5]), (0, [6, 8, 10]), (1, [7, 9])]
```

If we specify `release_remainding=False` though, we won’t get

```pycon
>>> g = igroupby(items=range(11),
...             key=lambda x: x % 2,
...             group_release_cond=lambda k, v: len(v) == 3,
...             release_remainding=False)
>>> list(g)
[(0, [0, 2, 4]), (1, [1, 3, 5]), (0, [6, 8, 10])]
```

# >>> grps = partial(igroupby, group_release_cond=False, release_remainding=True)

Below we show that, with the default `group_release_cond = lambda k, v: False`
and release_remainding=True\`\` we have `dict(igroupby(...)) == groupby(...)`

```pycon
>>> from functools import partial
>>> from dol import groupby
>>>
>>> kws = dict(items=range(11), key=lambda x: x % 3)
>>> assert (dict(igroupby(**kws)) == groupby(**kws)
...         == {0: [0, 3, 6, 9], 1: [1, 4, 7, 10], 2: [2, 5, 8]})
>>>
>>> tokens = ['the', 'fox', 'is', 'in', 'a', 'box']
>>> kws = dict(items=tokens, key=len)
>>> assert (dict(igroupby(**kws)) == groupby(**kws)
...         == {3: ['the', 'fox', 'box'], 2: ['is', 'in'], 1: ['a']})
>>>
>>> key_map = {1: 'one', 2: 'two'}
>>> kws.update(key=lambda x: key_map.get(len(x), 'more'))
>>> assert (dict(igroupby(**kws)) == groupby(**kws)
...         == {'more': ['the', 'fox', 'box'], 'two': ['is', 'in'], 'one': ['a']})
>>>
>>> stopwords = {'the', 'in', 'a', 'on'}
>>> kws.update(key=lambda w: w in stopwords)
>>> assert (dict(igroupby(**kws)) == groupby(**kws)
...         == {True: ['the', 'in', 'a'], False: ['fox', 'is', 'box']})
>>> kws.update(key=lambda w: ['words', 'stopwords'][int(w in stopwords)])
>>> assert (dict(igroupby(**kws)) == groupby(**kws)
...         == {'stopwords': ['the', 'in', 'a'], 'words': ['fox', 'is', 'box']})
```

### *class* dol.util.imdict

Bases: `ImmutableMixin`, [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict), `HashableMixin`

A frozen hashable dict

### dol.util.inject_method(obj, method_function, method_name=None)

method_function could be:

> * a function
> * a {method_name: function, …} dict (for multiple injections)
> * a list of functions or (function, method_name) pairs

### dol.util.instance_checker(\*types)

Makes a filter function that checks the type of an object.

```pycon
>>> f = instance_checker(int, float)
>>> f(1)
True
>>> f(1.0)
True
>>> f('1.0')
False
```

### dol.util.invertible_maps(mapping=None, inv_mapping=None)

Returns two maps that are inverse of each other.
Raises an AssertionError iif both maps are None, or if the maps are not inverse of
each other.

Get a pair of invertible maps

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]

```pycon
>>> invertible_maps({1: 11, 2: 22})
({1: 11, 2: 22}, {11: 1, 22: 2})
>>> invertible_maps(None, {11: 1, 22: 2})
({1: 11, 2: 22}, {11: 1, 22: 2})
```

You can specify one argument as an iterable (of keys for the mapping) and the
other as a function (to be applied to the keys to get the inverse mapping).
The function acts similarly to a `Mapping.__getitem__`, transforming each key to
its associated value. The iterable defines the keys for the mapping, while the
function is applied to each key to produce the values.

```pycon
>>> invertible_maps([1,2,3], lambda x: x * 10)
({10: 1, 20: 2, 30: 3}, {1: 10, 2: 20, 3: 30})
>>> invertible_maps(lambda x: x * 10, [1,2,3])
({1: 10, 2: 20, 3: 30}, {10: 1, 20: 2, 30: 3})
```

If two maps are given and invertible, you just get them back

```pycon
>>> invertible_maps({1: 11, 2: 22}, {11: 1, 22: 2})
({1: 11, 2: 22}, {11: 1, 22: 2})
```

Or if they’re not invertible

```pycon
>>> invertible_maps({1: 11, 2: 22}, {11: 1, 22: 'ha, not what you expected!'})
Traceback (most recent call last):
  ...
AssertionError: mapping and inv_mapping are not inverse of each other!
```

```pycon
>>> invertible_maps(None, None)
Traceback (most recent call last):
  ...
ValueError: You need to specify one or both maps
```

### dol.util.is_classmethod(obj)

Checks if an object is a classmethod.

* **Parameters:**
  **obj** – The object to check.
* **Returns:**
  True if the object is a classmethod, False otherwise.

Example usage:

```pycon
>>> class MyClass:
...     @classmethod
...     def class_method(cls):
...         pass
...
...     def instance_method(self):
...         pass
>>> obj1 = MyClass.class_method
>>> obj2 = MyClass().instance_method
>>> is_classmethod(obj1)
True
>>> is_classmethod(obj2)
False
```

### dol.util.is_unbound_method(obj)

Determines if the given object is an unbound method.

* **Parameters:**
  **obj** – The object to check.
* **Returns:**
  True if obj is an unbound method, False otherwise.

### Examples

```pycon
>>> import sys
>>> import types
>>> def function():
...     pass
>>> class MyClass:
...     def method(self):
...         pass
>>> is_unbound_method(MyClass.method)
True
>>> is_unbound_method(MyClass().method)
False
>>> is_unbound_method(function)
False
```

### *class* dol.util.lazyprop(func)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A descriptor implementation of lazyprop (cached property).
Made based on David Beazley’s “Python Cookbook” book and enhanced with boltons.cacheutils ideas.

```pycon
>>> class Test:
...     def __init__(self, a):
...         self.a = a
...     @lazyprop
...     def len(self):
...         print('generating "len"')
...         return len(self.a)
>>> t = Test([0, 1, 2, 3, 4])
>>> t.__dict__
{'a': [0, 1, 2, 3, 4]}
>>> t.len
generating "len"
5
>>> t.__dict__
{'a': [0, 1, 2, 3, 4], 'len': 5}
>>> t.len
5
>>> # But careful when using lazyprop that no one will change the value of a without deleting the property first
>>> t.a = [0, 1, 2]  # if we change a...
>>> t.len  # ... we still get the old cached value of len
5
>>> del t.len  # if we delete the len prop
>>> t.len  # ... then len being recomputed again
generating "len"
3
```

### *class* dol.util.lazyprop_w_sentinel(func)

Bases: [`lazyprop`](#dol.util.lazyprop)

A descriptor implementation of lazyprop (cached property).
Inserts a `self.func.__name__ + '__cache_active'` attribute

```pycon
>>> class Test:
...     def __init__(self, a):
...         self.a = a
...     @lazyprop_w_sentinel
...     def len(self):
...         print('generating "len"')
...         return len(self.a)
>>> t = Test([0, 1, 2, 3, 4])
>>> lazyprop_w_sentinel.cache_is_active(t, 'len')
False
>>> t.__dict__  # let's look under the hood
{'a': [0, 1, 2, 3, 4]}
>>> t.len
generating "len"
5
>>> lazyprop_w_sentinel.cache_is_active(t, 'len')
True
>>> t.len  # notice there's no 'generating "len"' print this time!
5
>>> t.__dict__  # let's look under the hood
{'a': [0, 1, 2, 3, 4], 'len': 5, 'sentinel_of__len': True}
>>> # But careful when using lazyprop that no one will change the value of a without deleting the property first
>>> t.a = [0, 1, 2]  # if we change a...
>>> t.len  # ... we still get the old cached value of len
5
>>> del t.len  # if we delete the len prop
>>> t.len  # ... then len being recomputed again
generating "len"
3
```

### dol.util.max_common_prefix(a, , default='')

Given a list of strings (or other sliceable seq), returns the longest common prefix

* **Parameters:**
  **a** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)) – list-like of strings
* **Returns:**
  the smallest common prefix of all strings in a

```pycon
>>> max_common_prefix(['absolutely', 'abc', 'abba'])
'ab'
>>> max_common_prefix(['absolutely', 'not', 'abc', 'abba'])
''
>>> max_common_prefix([[3,2,1], [3,2,0]])
[3, 2]
>>> max_common_prefix([[3,2,1], [3,2,0], [1,2,3]])
[]
```

If the input is empty, will return default (which defaults to ‘’).

```pycon
>>> max_common_prefix([])
''
```

If you want a different default, you can specify it with the default
keyword argument.

```pycon
>>> from functools import partial
>>> my_max_common_prefix = partial(max_common_prefix, default=[])
>>> my_max_common_prefix([])
[]
```

### dol.util.named_partial(func, \*args, \_\_name_\_=None, \*\*keywords)

functools.partial, but with a \_\_name_\_

```pycon
>>> f = named_partial(print, sep='\n')
>>> f.__name__
'print'
```

```pycon
>>> f = named_partial(print, sep='\n', __name__='now_partial_has_a_name')
>>> f.__name__
'now_partial_has_a_name'
```

### dol.util.non_colliding_key(key, exclude, , collision_handler=None, max_attempts=10000)

Return a key not present in the exclude container.

If the input key is already unique, it’s returned as-is.
Otherwise, applies a collision_handler until a unique key is found.

* **Parameters:**
  * **key** ([`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)) – The candidate key to check/modify
  * **exclude** ([`Container`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Container)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)]) – Container of keys to avoid
  * **collision_handler** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)]) – Function taking (key, attempt_number) and returning a modified key.
    For strings, defaults to appending “ (N)” suffix before extension.
    For other types, must be provided.
  * **max_attempts** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number of transformation attempts
* **Return type:**
  [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)
* **Returns:**
  A key not present in the exclude container
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If no unique key found within max_attempts, or if collision_handler
      is None for non-string keys

```pycon
>>> non_colliding_key("file.txt", set())
'file.txt'
>>> non_colliding_key("file.txt", {"file.txt"})
'file (1).txt'
>>> non_colliding_key("file.txt", {"file.txt", "file (1).txt"})
'file (2).txt'
>>> non_colliding_key(42, {42}, collision_handler=lambda k, n: k + n)
43
```

### dol.util.norm_kv_filt(kv_filt)

Prepare a boolean function to be used with `filter` when fed an iterable of (k, v) pairs.

So you have a mapping. Say a dict `d`. Now you want to go through d.items(),
filtering based on the keys, or the values, or both.

It’s not hard to do, really. If you’re using a dict you might use a dict comprehension,
or in the general case you might do a `filter(lambda kv: my_filt(kv[0], kv[1]), d.items())`
if you have a `my_filt` that works wiith k and v, etc.

But thought simple, it can become a bit muddled.
`norm_kv_filt` simplifies this by allowing you to bring your own filtering boolean function,
whether it’s a key-based, value-based, or key-value-based one, and it will make a
ready-to-use with `filter` function for you.

Only thing: Your function needs to call a key `k` and a value `v`.
But hey, it’s alright, if you have a function that calls things differently, just do
something like

```python
new_filt_func = lambda k, v: your_filt_func(..., key=k, ..., value=v, ...)
```

and all will be fine.

* **Parameters:**
  **kv_filt** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – callable (starting with signature (k), (v), or (k, v)), and returning  a boolean
* **Returns:**
  A normalized callable.

```pycon
>>> d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
>>> list(filter(norm_kv_filt(lambda k: k in {'b', 'd'}), d.items()))
[('b', 2), ('d', 4)]
>>> list(filter(norm_kv_filt(lambda v: v > 2), d.items()))
[('c', 3), ('d', 4)]
>>> list(filter(norm_kv_filt(lambda k, v: (v > 1) & (k != 'c')), d.items()))
[('b', 2), ('d', 4)]
```

### dol.util.not_a_mac_junk_path(path)

A function that will tell you if the path is not a mac junk path/
More precisely, doesn’t end with ‘.DS_Store’ or have a `__MACOSX` folder somewhere
on it’s way.

This is usually meant to be used with `filter` or `filt_iter` to “filter in” only
those actually wanted files (not the junk that mac writes to your filesystem).

These files annoyingly show up often in zip files, and are usually unwanted.

See [https://apple.stackexchange.com/questions/239578/compress-without-ds-store-and-macosx](https://apple.stackexchange.com/questions/239578/compress-without-ds-store-and-macosx)

```pycon
>>> paths = ['A/normal/path', 'A/__MACOSX/path', 'path/ending/in/.DS_Store', 'foo/b']
>>> list(filter(not_a_mac_junk_path, paths))
['A/normal/path', 'foo/b']
```

### dol.util.num_of_args(func)

Number of arguments (parameters) of the function.

Contrast the behavior below with that of `num_of_required_args`.

```pycon
>>> num_of_args(lambda a, b, c: None)
3
>>> num_of_args(lambda a, b, c=3: None)
3
>>> num_of_args(lambda a, *args, b, c=1, d=2, **kwargs: None)
6
```

### dol.util.num_of_required_args(func)

Number or REQUIRED arguments of a function.

Contrast the behavior below with that of `num_of_args`, which counts all
parameters, including the variadics and defaulted ones.

```pycon
>>> num_of_required_args(lambda a, b, c: None)
3
>>> num_of_required_args(lambda a, b, c=3: None)
2
>>> num_of_required_args(lambda a, *args, b, c=1, d=2, **kwargs: None)
2
```

### dol.util.partialclass(cls, \*args, \*\*kwargs)

What `partial(cls, *args, **kwargs)` does, but returning a class instead of an object.

* **Parameters:**
  * **cls** – Class to get the partial of
  * **kwargs** – The kwargs to fix

The raison d’être of partialclass is that it returns a type, so let’s have a look at that with
a useless class.

```pycon
>>> from inspect import signature
>>> class A:
...     pass
>>> assert isinstance(A, type) == isinstance(partialclass(A), type) == True
```

```pycon
>>> class A:
...     def __init__(self, a=0, b=1):
...         self.a, self.b = a, b
...     def mysum(self):
...         return self.a + self.b
...     def __repr__(self):
...         return f"{self.__class__.__name__}(a={self.a}, b={self.b})"
>>>
>>> assert isinstance(A, type) == isinstance(partialclass(A), type) == True
>>>
>>> assert str(signature(A)) == '(a=0, b=1)'
>>>
>>> a = A()
>>> assert a.mysum() == 1
>>> assert str(a) == 'A(a=0, b=1)'
>>>
>>> assert A(a=10).mysum() == 11
>>> assert str(A()) == 'A(a=0, b=1)'
>>>
>>>
>>> AA = partialclass(A, b=2)
>>> assert str(signature(AA)) == '(a=0, *, b=2)'
>>> aa = AA()
>>> assert aa.mysum() == 2
>>> assert str(aa) == 'A(a=0, b=2)'
>>> assert AA(a=1, b=3).mysum() == 4
>>> assert str(AA(3)) == 'A(a=3, b=2)'
>>>
>>> AA = partialclass(A, a=7)
>>> assert str(signature(AA)) == '(*, a=7, b=1)'
>>> assert AA().mysum() == 8
>>> assert str(AA(a=3)) == 'A(a=3, b=1)'
```

Note in the last partial that since `a` was fixed, you need to specify the keyword `AA(a=3)`.
`AA(3)` won’t work:

```pycon
>>> AA(3)
Traceback (most recent call last):
  ...
TypeError: __init__() got multiple values for argument 'a'
```

On the other hand, you can use `*args` to specify the fixtures:

```pycon
>>> AA = partialclass(A, 22)
>>> assert str(AA()) == 'A(a=22, b=1)'
>>> assert str(signature(AA)) == '(b=1)'
>>> assert str(AA(3)) == 'A(a=22, b=3)'
```

### dol.util.read_from_bytes(file_reader, obj=None, \*, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>, \*\*kwargs)

Takes a file reading function that expects a file-like object,
and returns a function that instead of reading from a file, reads from bytes.

This is the read version of the `written_bytes` function of the same module.

#### NOTE
If obj is not given, read_from_bytes will return a “bytes reader” function that
takes obj as the first argument, and uses the file_reader to read the bytes.

* **Parameters:**
  * **file_reader** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – A function that reads from a file-like object.
  * **obj** ([`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)) – The bytes to read.
  * **buffer_arg_position** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – The position of the file-like object in file_reader’s arguments.
  * **buffer_arg_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The name of the file-like object argument in file_reader.
* **Returns:**
  The result of reading from the bytes.

Example usage:

Using `json.load` to read a JSON object from bytes:

```pycon
>>> import json
>>> data = {'a': 1, 'b': 2}
>>> json_bytes = json.dumps(data).encode('utf-8')
>>> read_json_from_bytes = read_from_bytes(json.load)
>>> data_loaded = read_json_from_bytes(json_bytes)
>>> data_loaded == data
True
```

Using `pickle.load` to read an object from bytes:

```pycon
>>> import pickle
>>> obj = {'x': [1, 2, 3], 'y': ('a', 'b')}
>>> pickle_bytes = pickle.dumps(obj)
>>> read_pickle_from_bytes = read_from_bytes(pickle.load)
>>> obj_loaded = read_pickle_from_bytes(pickle_bytes)
>>> obj_loaded == obj
True
```

### dol.util.regroupby(items, \*key_funcs, \*\*named_key_funcs)

Recursive groupby. Applies the groupby function recursively, using a sequence of key functions.

#### NOTE
The named_key_funcs argument names don’t have any external effect.

They just give a name to the key function, for code reading clarity purposes.

#### SEE ALSO
groupby, itertools.groupby, and dol.source.SequenceKvReader

```pycon
>>> # group by how big the number is, then by it's mod 3 value
>>> # note that named_key_funcs argument names doesn't have any external effect (but give a name to the function)
>>> regroupby([1, 2, 3, 4, 5, 6, 7], lambda x: 'big' if x > 5 else 'small', mod3=lambda x: x % 3)
{'small': {1: [1, 4], 2: [2, 5], 0: [3]}, 'big': {0: [6], 1: [7]}}
>>>
>>> tokens = ['the', 'fox', 'is', 'in', 'a', 'box']
>>> stopwords = {'the', 'in', 'a', 'on'}
>>> word_category = lambda x: 'stopwords' if x in stopwords else 'words'
>>> regroupby(tokens, word_category, len)
{'stopwords': {3: ['the'], 2: ['in'], 1: ['a']}, 'words': {3: ['fox', 'box'], 2: ['is']}}
>>> regroupby(tokens, len, word_category)
{3: {'stopwords': ['the'], 'words': ['fox', 'box']}, 2: {'words': ['is'], 'stopwords': ['in']}, 1: {'stopwords': ['a']}}
```

### dol.util.safe_compile(path, normalize_path=True)

Compile a *literal file path* into a regex pattern that matches that path,
normalizing separators and escaping regex-special characters on Windows.

#### WARNING
This is for **path templates only**, NOT for general regexes. It
`re.escape`-s its argument on Windows, which turns any regex into a
literal-string matcher there. To compile an actual regex, use
`re.compile` (see `dol.trans.filter_regex`, fixed to do exactly that).
Its output is intentionally platform-dependent (Windows paths get escaped),
so callers must not rely on a specific `.pattern` across OSes.

* **Parameters:**
  **path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The file path to be compiled into a regex pattern.
* **Returns:**
  A compiled regular expression object for the given path.
* **Return type:**
  [*Pattern*](https://docs.python.org/3/library/re.html#re.Pattern)

### Examples

```pycon
>>> import re
>>> isinstance(safe_compile("/fun/paths/are/awesome"), re.Pattern)
True
>>> isinstance(safe_compile(r"C:\folder\file.txt"), re.Pattern)
True
```

### *class* dol.util.staticproperty(function)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A decorator for defining static properties in classes.

```pycon
>>> class A:
...     @staticproperty
...     def foo():
...         return 2
>>> A.foo
2
>>> A().foo
2
```

### dol.util.str_to_var_str(s)

Make a valid python variable string from the input string.
Left untouched if already valid.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> str_to_var_str('this_is_a_valid_var_name')
'this_is_a_valid_var_name'
>>> str_to_var_str('not valid  #)*(&434')
'not_valid_______434'
>>> str_to_var_str('99_ballons')
'_99_ballons'
```

### dol.util.truncate_string_with_marker(s, , left_limit=15, right_limit=15, middle_marker='...')

Return a string with a limited length.

If the string is longer than the sum of the left_limit and right_limit,
the string is truncated and the middle_marker is inserted in the middle.

If the string is shorter than the sum of the left_limit and right_limit,
the string is returned as is.

```pycon
>>> truncate_string_with_marker('1234567890')
'1234567890'
```

But if the string is longer than the sum of the limits, it is truncated:

```pycon
>>> truncate_string_with_marker('1234567890', left_limit=3, right_limit=3)
'123...890'
>>> truncate_string_with_marker('1234567890', left_limit=3, right_limit=0)
'123...'
>>> truncate_string_with_marker('1234567890', left_limit=0, right_limit=3)
'...890'
```

If you’re using a specific parametrization of the function often, you can
create a partial function with the desired parameters:

```pycon
>>> from functools import partial
>>> truncate_string = partial(truncate_string_with_marker, left_limit=2, right_limit=2, middle_marker='---')
>>> truncate_string('1234567890')
'12---90'
>>> truncate_string('supercalifragilisticexpialidocious')
'su---us'
```

### dol.util.written_bytes(file_writer, obj=None, \*, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>)

Takes a file writing function that expects an object and a file-like object,
and returns a function that instead of writing to a file, returns the bytes that
would have been written.

This is the write version of the `read_from_bytes` function of the same module.

#### NOTE
If obj is not given, `write_bytes` will return a “bytes writer” function that
takes obj as the first argument, and uses the file_writer to write the bytes.

* **Parameters:**
  * **file_writer** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`), `Union`[[`BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO), [`StringIO`](https://docs.python.org/3/library/io.html#io.StringIO)]], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – A function that writes an object to a file-like object.
  * **obj** ([`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)) – The object to write.
* **Returns:**
  The bytes that would have been written to a file.

Use case: When you have a function that writes to files, and you want to get an
equivalent function but that gives you what bytes or string WOULD have been written
to a file, so you can better reuse (to write elsewhere, for example, or because
you need to pipe those bytes to another function).

Example usage: Yes, we have json.dumps to get the JSON string, but what if
(like is often the case) you just have a function that writes to a file-like object,
like the `json.dump(obj, fp)` function? You can use `written_bytes` to get a
function that will act as `json.dumps` like so:

```pycon
>>> import json
>>> get_json_bytes = written_bytes(json.dump, io_buffer_cls=io.StringIO)
>>> get_json_bytes({'a': 1, 'b': 2})
'{"a": 1, "b": 2}'
```

Here’s another example with pandas DataFrame.to_parquet:

```python
import pandas as pd
df = pd.DataFrame({'column1': [1, 2, 3], 'column2': ['A', 'B', 'C']})
# Get a function that converts DataFrame to Parquet bytes
df_to_parquet_bytes = written_bytes(pd.DataFrame.to_parquet)
# Get the bytes of the DataFrame in Parquet format
parquet_bytes = df_to_parquet_bytes(df)
all(pd.read_parquet(io.BytesIO(parquet_bytes)) == df)
```

### dol.util.written_key(obj=None, writer=<function write_to_file>, \*, key=None, obj_arg_position_in_writer=0, encoder=<function identity_func>)

Writes an object to a key and returns the key.
If key is not given, a temporary file is created and its path is returned.

* **Parameters:**
  * **obj** ([`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)) – The object to write.
  * **writer** (`Union`[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – A function that writes an object to a file.
  * **key** (`Union`[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – The key (by default, filepath) to write to.
    If None, a temporary file is created.
    If a string starting with ‘\*’, the ‘\*’ is replaced with a unique temporary filename.
    If a string that has a ‘\*’ somewhere in the middle, what’s on the left of if is used as a directory
    and the ‘\*’ is replaced with a unique temporary filename. For example
    `'/tmp/*_file.ext'` would be replaced with `'/tmp/oiu8fj9873_file.ext'`.
    If a callable, it will be called with obj as input to get the key. One use case
    is to use a function that generates a key based on the object.
  * **obj_arg_position_in_writer** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Position of the object argument in writer function (0 or 1).
  * **encoder** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – A function that encodes the object before writing it.
* **Returns:**
  The file path where the object was written.

Example usage:

Let’s make a store and a writer for that store.

```pycon
>>> store = dict()
>>> writer = writer=lambda obj, key: store.__setitem__(key, obj)
```

Note the order a writer expects is (obj, key), or we’d just be able to use
`store.__setitem__` as our writer.

If we specify a key, the object will be written to that key in the store
and the key is output.

```pycon
>>> written_key(42, writer=writer, key='my_key')
'my_key'
>>> store
{'my_key': 42}
```

Often, you’ll want to fix your writer (and possibly your key).
You can do so with `functools.partial`, but for convenience, you can also
just specify a writer, without an input object, and get a function that
will write an object to a key.

```pycon
>>> write_to_store = written_key(writer=writer, key='another_key')
>>> write_to_store(99)
'another_key'
>>> store
{'my_key': 42, 'another_key': 99}
```

If you don’t specify a key, a temporary file is created and the key is output.

```pycon
>>> write_to_store = written_key(writer=writer)
>>> key = write_to_store(43)
>>> key
'/var/folders/mc/c070wfh51kxd9lft8dl74q1r0000gn/T/tmp8yaczd8b'
>>> store[key]
43
```

If the key you specify is a string with a ‘\*’, the ‘\*’ is replaced with a
unique temporary filename, or the full path of the temporary file if the \*
is at the start.

```pycon
>>> write_to_store = written_key(writer=writer, key='*.ext')
>>> key = write_to_store(44)
>>> key
'....ext'
>>> store[key]
44
```

One useful use case is when you want to pipe the output of one function into
another function that expects a file path.
What you need to do then is just pipe your written_key function into that
function that expects to work with a file path, and it’ll be like piping the
value of your input object into that function (just via a temp file).

```pycon
>>> from dol.util import Pipe
>>> store.clear()
>>> key_func = lambda key: store.get(key) * 10
>>> pipe_obj_to_reader = Pipe(written_key(writer=writer), key_func)
>>> pipe_obj_to_reader(45)
450
>>> store
{...: 45}
```

The default writer is `write_to_file`, which can write bytes or strings to a file.
If your object is not a bytes or string, you can specify an encoder to encode it
before calling the writer.

```pycon
>>> import json, pathlib
>>> json_written_temp_filepath = written_key(key='*.json', encoder=json.dumps)
>>> filepath = json_written_temp_filepath({'a': 1, 'b': 2})
>>> filepath
'/var/folders/mc/c070wfh51kxd9lft8dl74q1r0000gn/T/tmp8yaczd8b.json'
>>> json.loads(open(filepath).read())
{'a': 1, 'b': 2}
```
