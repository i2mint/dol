# dol.tools

Various tools to add functionality to stores.

Main entry points:

- `store_aggregate`: aggregate a store’s items into one object (a Markdown text by default)
- `confirm_overwrite`: a `wrap_kvs` preset that asks before overwriting a value
- `Forest`: a key-value tree view of nested objects
  ```pycon
  >>> from dol.tools import store_aggregate
  >>> print(store_aggregate({'a': 'x', 'b': 'y'}))
  ## a

  x



  ## b

  y

  ```

### Functions

| [`ask_user_for_value_when_missing`](#dol.tools.ask_user_for_value_when_missing)([store, ...])   | Wrap a store so if a value is missing when the user asks for it, they will be given a chance to enter the value they want to write.                   |
|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`confirm_overwrite`](#dol.tools.confirm_overwrite)(mapping, k, v[, ...])         | A preset function you can use in wrap_kvs to ask the user to confirm if they're writing a value in a key that already has a different value under it. |
| [`convert_to_numerical_if_possible`](#dol.tools.convert_to_numerical_if_possible)(s)             | To be used with `ask_user_for_value_when_missing` `value_preprocessor` arg                                                                            |
| `decode_as_latin1`(b)                                                                            |                                                                                                                                                       |
| `identity`(x)                                                                                    |                                                                                                                                                       |
| `markdown_section`(k, v)                                                                         |                                                                                                                                                       |
| `return_input`(x)                                                                                |                                                                                                                                                       |
| `save_string_to_filepath`(filepath, string)                                                      |                                                                                                                                                       |
| [`store_aggregate`](#dol.tools.store_aggregate)(content_store, \*[, ...])       | Create an aggregate object of a store's (a Mapping of strings) content                                                                                |
| `type_check_if_type`(filt)                                                                       |                                                                                                                                                       |

### Classes

| [`Forest`](#dol.tools.Forest)(src, \*, get_node_keys, get_src_item, ...)   | Provides a key-value forest interface to objects.                                                                |
|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `NoSuchKey`()                                                                                        |                                                                                                                  |
| [`iSliceStore`](#dol.tools.iSliceStore)(store)                                  | Wraps a store to make a reader that acts as if the store was a list (with integer keys, and that can be sliced). |

### *class* dol.tools.Forest(src, \*, get_node_keys, get_src_item, is_leaf, forest_type=<class 'list'>, leaf_trans=<function return_input>)

Bases: [`KvReader`](dol.base.html.md#dol.base.KvReader)

Provides a key-value forest interface to objects.

A [treehttps://en.wikipedia.org/wiki/Tree_(data_structure)](treehttps://en.wikipedia.org/wiki/Tree_(data_structure))
is a nested data structure. A tree has a root, which is the parent of children,
who themselves can be parents of further subtrees, or not; in which case they’re
called leafs.
For more information, see
[wikipediaontreeshttps://en.wikipedia.org/wiki/Tree_(data_structure)](wikipediaontreeshttps://en.wikipedia.org/wiki/Tree_(data_structure))

Here we allow one to construct a tree view of any python object, using a
key-value interface to the parent-child relationship.

A forest is a collection of trees.

Arguably, a dictionnary might not be the most impactful example to show here, since
it is naturally a tree (therefore a forest), and naturally key-valued: But it has
the advantage of being easy to demo with.
Where Forest would really be useful is when you (1) want to give a consistent
key-value interface to the many various forms that trees and forest objects come
in, or even more so when (2) your object’s tree/forest structure is not obvious,
so you need to “extract” that view from it (plus give it a consistent key-value
interface, so that you can build an ecosystem of tools around it.

Anyway, here’s our dictionary example:

```pycon
>>> d = {
...     'apple': {
...         'kind': 'fruit',
...         'types': {
...             'granny': {'color': 'green'},
...             'fuji': {'color': 'red'}
...         },
...         'tasty': True
...     },
...     'acrobat': {
...         'kind': 'person',
...         'nationality': 'french',
...         'brave': True,
...     },
...     'ball': {
...         'kind': 'toy'
...     }
... }
```

Must of the time, you’ll want to curry `Forest` to make an `object_to_forest`
constructor for a given class of objects. In the case of dictionaries as the one
above, this might look like this:

```pycon
>>> from functools import partial
>>> a_forest = partial(
...     Forest,
...     is_leaf=lambda k, v: not isinstance(v, dict),
...     get_node_keys=lambda v: [vv for vv in iter(v) if not vv.startswith('b')],
...     get_src_item=lambda src, k: src[k]
... )
>>>
>>> f = a_forest(d)
>>> list(f)
['apple', 'acrobat']
```

Note that we specified in `get_node_keys``that we didn't want to include items
whose keys start with ``b` as valid children. Therefore we don’t have our
`'ball'` in the list above.

Note below which nodes are themselves `Forests`, and whic are leafs:

```pycon
>>> ff = f['apple']
>>> isinstance(ff, Forest)
True
>>> list(ff)
['kind', 'types', 'tasty']
>>> ff['kind']
'fruit'
>>> fff = ff['types']
>>> isinstance(fff, Forest)
True
>>> list(fff)
['granny', 'fuji']
```

### dol.tools.ask_user_for_value_when_missing(store=None, , value_preprocessor=None, on_missing_msg='No such key was found. You can enter a value for it here or simply hit enter to leave the slot empty', \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Wrap a store so if a value is missing when the user asks for it, they will be
given a chance to enter the value they want to write.

* **Parameters:**
  * **store** – The store (instance or class) to wrap
  * **value_preprocessor** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Function to transform the user value before trying to
    write it (bearing in mind all user specified values are strings)
  * **on_missing_msg** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – String that will be displayed to prompt the user to enter a
    value
* **Returns:**

### dol.tools.confirm_overwrite(mapping, k, v, user_input_msg='The key {k} already exists and has value {existing_v}. If you want to overwrite it with {v}, confirm by typing {v} here: ')

A preset function you can use in wrap_kvs to ask the user to confirm if
they’re writing a value in a key that already has a different value under it.

```pycon
>>> from dol.trans import wrap_kvs
>>> d = {'a': 'apple', 'b': 'banana'}
>>> d = wrap_kvs(d, preset=confirm_overwrite)
```

Overwriting `a` with the same value it already has is fine (not really an
over-write):

```pycon
>>> d['a'] = 'apple'
```

Creating new values is also fine:

```pycon
>>> d['c'] = 'coconut'
>>> assert d == {'a': 'apple', 'b': 'banana', 'c': 'coconut'}
```

But if we tried to do `d['a'] = 'alligator'`, we’ll get a user input request:

```default
The key a already exists and has value apple.
If you want to overwrite it with alligator, confirm by typing alligator here:
```

And we’ll have to type `alligator` and press RETURN to make the write go through.

### dol.tools.convert_to_numerical_if_possible(s)

To be used with `ask_user_for_value_when_missing` `value_preprocessor` arg

```pycon
>>> convert_to_numerical_if_possible("123")
123
>>> convert_to_numerical_if_possible("123.4")
123.4
>>> convert_to_numerical_if_possible("one")
'one'
```

Border case: The strings “infinity” and “inf” actually convert to a valid float.

```pycon
>>> convert_to_numerical_if_possible("infinity")
inf
```

### *class* dol.tools.iSliceStore(store)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

Wraps a store to make a reader that acts as if the store was a list
(with integer keys, and that can be sliced).
I say “list”, but it should be noted that the behavior is more that of range,
that outputs an element of the list
when keying with an integer, but returns an iterable object (a range) if sliced.

Here, a map object is returned when the sliceable store is sliced.

```pycon
>>> s = {'foo': 'bar', 'hello': 'world', 'alice': 'bob'}
>>> sliceable_s = iSliceStore(s)
```

The read-only functionalities of the underlying mapping are still available:

```pycon
>>> list(sliceable_s)
['foo', 'hello', 'alice']
>>> 'hello' in sliceable_s
True
>>> sliceable_s['hello']
'world'
```

But now you can get slices as well:

```pycon
>>> list(sliceable_s[0:2])
['bar', 'world']
>>> list(sliceable_s[-2:])
['world', 'bob']
>>> list(sliceable_s[:-1])
['bar', 'world']
```

Now, you can’t do `sliceable_s[1]` because `1` isn’t a valid key.
But if you really wanted “item number 1”, you can do:

```pycon
>>> next(sliceable_s[1:2])
'world'
```

Note that `sliceable_s[i:j]` is an iterable that needs to be consumed
(here, with list) to actually get the data. If you want your data in a different
format, you can use `dol.trans.wrap_kvs` for that.

```pycon
>>> from dol import wrap_kvs
>>> ss = wrap_kvs(sliceable_s, obj_of_data=list)
>>> ss[1:3]
['world', 'bob']
>>> sss = wrap_kvs(sliceable_s, obj_of_data=sorted)
>>> sss[1:3]
['bob', 'world']
```

### dol.tools.store_aggregate(content_store, \*, kv_to_item=<function markdown_section>, aggregator=<built-in method join of str object>, egress=<function identity>, key_filter=None, value_filter=None, kv_filter=None, local_store_factory=<class 'dol.filesys.Files'>)

Create an aggregate object of a store’s (a Mapping of strings) content

The function is written to be able to aggregate the keys and/or values of a store,
no matter their type, and concatenate them into an object of arbitrary type.
That said, the defaults are setup assuming the store’s keys and values are text,
and you want to concatenate them into a single string.
This is useful, for example, when you have several files in a folder,
and you want to create a single text/markdown file with all the content therein.

This function filters content from a given content store, converts the key-value
pairs to items (usually text), and (if you specify a filepath as the `egress`)
saves the aggregate (text) before returning it.

* **Parameters:**
  * **content_store** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)] | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to the folder or dol store to read from.
  * **kv_to_item** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Item`)]) – Function to convert key-value pairs to an Item (usually a string).
  * **aggregator** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Item`)]], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Aggregate`)]) – The function that will aggregate the items that `kv_to_item` produces.
    Defaults to ‘nn’.join.
  * **egress** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Aggregate`)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The function that will be called on the aggregate before returning it.
    Defaults to identity.
    Note that if you provide a string, the function will save the aggregate
    text to a file, assuming it is indeed text.
  * **key_filter** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional filter for keys. Defaults to None (no filtering).
  * **value_filter** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional filter for values. Defaults to None (no filtering).
  * **kv_filter** ([`None`](https://docs.python.org/3/builtins/constants.html#None) | [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)]], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – Optional filter for key-value pairs. Defaults to None (no filtering).
  * **local_store_factory** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)]]) – Factory function for the local store,
    used only if `content_store` is an existing folder path. Defaults to Latin1TextFiles.
* **Returns:**
  Usually the aggregate object, which is usually the concatenated text.
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

Normally, you’d specify your content store by specifying a root folder
(the function will create a Mapping-view of the contents of the folder for you),
or make a content store yourself (a Mapping object providing the key-value pairs).

To provide a small example, we’ll take a dict as our content store:

```pycon
>>> content_store = {
...     'file1.py': '"""Module docstring."""',
...     'file2.py': 'def foo(): pass',
...     'file3.py': '"""Another docstring."""',
...     'file4.md': 'Markdown content here.',
...     'file5.py': '"""If I mention file5.py, I will be excluded."""',
... }
```

Define the filters:

```pycon
>>> key_filter = lambda k: k.endswith('.py')  # Only include keys that end with '.py'
>>> value_filter = lambda v: v.startswith(
...     '"""'
... )  # Only include values that start with """ (marking a module docstring)
>>> kv_filter = (
...     lambda kv: kv[0] not in kv[1]
... )  # Exclude key-value pairs where the value mentions the key
```

Call the function with the provided filters and settings

```pycon
>>> result = store_aggregate(
...     content_store=content_store,  # The content_store dict
...     kv_to_item="{} -> {}".format,  # Format key-value pairs as "key -> value"
...     key_filter=key_filter,  # Key filter: Include only .py files
...     value_filter=value_filter,  # Value filter: Include only values starting with """
...     kv_filter=kv_filter,  # KV filter: Exclude if value contains the key
...     aggregator=', '.join,
...     egress='~/test.md'
... )
>>> result
'file1.py -> """Module docstring.""", file3.py -> """Another docstring."""'
```

Here, you got the string as the result. If you want to save it to a file,
you can provide the save_filepath argument, and it will save the text to the file,
and return the save_filepath to you (which )

Recipe: You can do a lot with the `kv_to_text` argument. For example, if your
content store doesn’t have string keys or values, you can always extract whatever
information you need from them to produce the text that will represent that item.
