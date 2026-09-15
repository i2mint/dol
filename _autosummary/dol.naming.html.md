# dol.naming

This module is about generating, validating, and operating on (parametrized) fields (i.e. strings, e.g. paths).

Main entry points:

- `StrTupleDict`: convert a templated name between string, tuple and dict forms
- `mk_pattern_from_template_and_format_dict`: a compiled regex from a template
- `get_fields_from_template`: the field names of a template
  ```pycon
  >>> from dol.naming import get_fields_from_template
  >>> get_fields_from_template('this{is}an{example}')
  ['is', 'example']
  ```

### Functions

| [`dict_to_namedtuple`](#dol.naming.dict_to_namedtuple)(d[, namedtuple_obj])           |                                                                                                                                                                                         |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`get_fields_from_template`](#dol.naming.get_fields_from_template)(template)                | Get list from {item} items of template string                                                                                                                                           |
| `mk_capture_patterns`(mapping_dict)                                                                |                                                                                                                                                                                         |
| `mk_extract_pattern`(template[, format_dict, ...])                                                 |                                                                                                                                                                                         |
| `mk_format_mapping_dict`(format_dict, ...[, sep])                                                  |                                                                                                                                                                                         |
| [`mk_kwargs_trans`](#dol.naming.mk_kwargs_trans)(\*\*trans_func_for_key)           | Make a dict transformer from functions that depends solely on keys (of the dict to be transformed) Used to easily make process_kwargs and process_info_dict arguments for LinearNaming. |
| `mk_named_capture_patterns`(mapping_dict)                                                          |                                                                                                                                                                                         |
| [`mk_pattern_from_template_and_format_dict`](#dol.naming.mk_pattern_from_template_and_format_dict)(...)     | Make a compiled regex to match template                                                                                                                                                 |
| `mk_prefix_templates_dicts`(template)                                                              |                                                                                                                                                                                         |
| [`mk_store_from_path_format_store_cls`](#dol.naming.mk_store_from_path_format_store_cls)([store, ...]) | Wrap a store (instance or class) that uses string keys to make it into a store that uses a specific key format.                                                                         |
| [`mk_tupled_store_from_path_format_store_cls`](#dol.naming.mk_tupled_store_from_path_format_store_cls)([...]) | Wrap a store (instance or class) that uses string keys to make it into a store that uses a specific key format.                                                                         |
| [`namedtuple_to_dict`](#dol.naming.namedtuple_to_dict)(nt)                            |                                                                                                                                                                                         |
| [`template_to_pattern`](#dol.naming.template_to_pattern)(mapping_dict, template)       | Weave a `{field}` template into a regex, substituting each field with its capture pattern and **regex-escaping the literal text between fields**.                                       |
| [`update_fields_of_namedtuple`](#dol.naming.update_fields_of_namedtuple)(nt, \*[, ...])        | Replace fields of namedtuple                                                                                                                                                            |
| [`validate_kwargs`](#dol.naming.validate_kwargs)(kwargs_to_validate, ...[, ...])   | Utility to validate a dict.                                                                                                                                                             |

### Classes

| [`BigDocTest`](#dol.naming.BigDocTest)()                               | Naming-scheme example holder whose (large) doctest is currently disabled.                   |
|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| [`KeyMapNames`](#dol.naming.KeyMapNames)                                |                                                                                             |
| [`KeyMaps`](#dol.naming.KeyMaps)(key_of_id, id_of_key)              |                                                                                             |
| [`LinearNaming`](#dol.naming.LinearNaming)                               |                                                                                             |
| `NamingInterface`([params, validation_funs, ...])                                           |                                                                                             |
| [`ParametricKeyStore`](#dol.naming.ParametricKeyStore)(store[, keymap])        |                                                                                             |
| [`PartialFormatter`](#dol.naming.PartialFormatter)()                         | A string formatter that won't complain if the fields are only partially formatted.          |
| [`StoreWithDictKeys`](#dol.naming.StoreWithDictKeys)(store[, keymap])         |                                                                                             |
| [`StoreWithNamedTupleKeys`](#dol.naming.StoreWithNamedTupleKeys)(store[, keymap])   |                                                                                             |
| [`StoreWithTupleKeys`](#dol.naming.StoreWithTupleKeys)(store[, keymap])        |                                                                                             |
| [`StrTupleDict`](#dol.naming.StrTupleDict)(template[, format_dict, ...]) | Convert a parametrized name between its string, tuple and dict forms.                       |
| [`StrTupleDictWithPrefix`](#dol.naming.StrTupleDictWithPrefix)(template[, ...])    | Converting from and to strings, tuples, and dicts, but with partial "prefix" specs allowed. |

### *class* dol.naming.BigDocTest

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Naming-scheme example holder whose (large) doctest is currently disabled.

The former doctest is kept as comments in the class body.

### dol.naming.KeyMapNames

alias of [`KeyMaps`](#dol.naming.KeyMaps)

### *class* dol.naming.KeyMaps(key_of_id, id_of_key)

Bases: [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)

#### id_of_key

Alias for field number 1

#### key_of_id

Alias for field number 0

### dol.naming.LinearNaming

alias of [`StrTupleDictWithPrefix`](#dol.naming.StrTupleDictWithPrefix)

### *class* dol.naming.ParametricKeyStore(store, keymap=None)

Bases: [`Store`](dol.base.html.md#dol.base.Store)

### *class* dol.naming.PartialFormatter

Bases: [`Formatter`](https://docs.python.org/3/library/string.html#string.Formatter)

A string formatter that won’t complain if the fields are only partially formatted.
But note that you will lose the spec part of your template (e.g. in {foo:1.2f}, you’ll loose the 1.2f
if not foo is given – but {foo} will remain).

```pycon
>>> partial_formatter = PartialFormatter()
>>> str_template = 'foo:{foo} bar={bar} a={a} b={b:0.02f} c={c}'
>>> partial_formatter.format(str_template, bar="BAR", b=34)
'foo:{foo} bar=BAR a={a} b=34.00 c={c}'
```

#### NOTE
If you only need a formatting function (not the transformed formatting string), a simpler solution may be:

```python
import functools
format_str = functools.partial(str_template.format, bar="BAR", b=34)
```

See [https://stackoverflow.com/questions/11283961/partial-string-formatting](https://stackoverflow.com/questions/11283961/partial-string-formatting) for more options and discussions.

### *class* dol.naming.StoreWithDictKeys(store, keymap=None)

Bases: [`ParametricKeyStore`](#dol.naming.ParametricKeyStore)

### *class* dol.naming.StoreWithNamedTupleKeys(store, keymap=None)

Bases: [`ParametricKeyStore`](#dol.naming.ParametricKeyStore)

### *class* dol.naming.StoreWithTupleKeys(store, keymap=None)

Bases: [`ParametricKeyStore`](#dol.naming.ParametricKeyStore)

### *class* dol.naming.StrTupleDict(template, format_dict=None, process_kwargs=None, process_info_dict=None, named_tuple_type_name='NamedTuple', sep='/')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Convert a parametrized name between its string, tuple and dict forms.

Built from a string template with `{field}` placeholders (and optional regexes
for the fields). See `__init__` for the parameters and an example.

#### extract(field, s)

Extract a single item from an name

* **Parameters:**
  * **field** – field of the item to extract
  * **s** – the string from which to extract it
* **Returns:**
  the value for name

#### info_dict(s)

Get a dict with the arguments of an name (for example group, user, subuser, etc.)

* **Parameters:**
  **s** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str))
* **Returns:**
  a dict holding the argument fields and values

#### is_valid(s)

Check if the name has the “upload format” (i.e. the kind of fields that are \_ids of fv_mgc, and what
name means in most of the iatis system.

* **Parameters:**
  **s** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the string to check
* **Returns:**
  True iff name has the upload format

#### replace_name_elements(s, \*\*elements_kwargs)

Replace specific name argument values with others

* **Parameters:**
  * **s** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the string to replace
  * **elements_kwargs** – the arguments to replace (and their values)
* **Returns:**
  a new name

#### str_to_dict(s)

Get a dict with the arguments of an name (for example group, user, subuser, etc.)

* **Parameters:**
  **s** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str))
* **Returns:**
  a dict holding the argument fields and values

#### super_dict_to_str(d)

Like dict_to_str, but the input dict can have extra keys that are not used by dict_to_str

### *class* dol.naming.StrTupleDictWithPrefix(template, format_dict=None, process_kwargs=None, process_info_dict=None, named_tuple_type_name='NamedTuple', sep='/')

Bases: [`StrTupleDict`](#dol.naming.StrTupleDict)

Converting from and to strings, tuples, and dicts, but with partial “prefix” specs allowed.

* **Parameters:**
  * **template** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple) | [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)) – The string format template
  * **format_dict** – A {field_name: field_value_format_regex, …} dict
  * **process_kwargs** – A function taking the field=value pairs and producing a dict of processed
    {field: value,…} dict (where both fields and values could have been processed.
    This is useful when we need to process (format, default, etc.) fields, or their values,
    according to the other fields of values in the collection.
    A specification of {field: function_to_process_this_value,…} wouldn’t allow the full powers
    we are allowing here.
  * **process_info_dict** – A sort of converse of format_dict.
    This is a {field_name: field_conversion_func, …} dict that is used to convert info_dict values
    before returning them.
  * **name_separator** – Used

```pycon
>>> ln = StrTupleDictWithPrefix('/home/{user}/fav/{num}.txt',
...                   format_dict={'user': '[^/]+', 'num': r'\d+'},
...                   process_info_dict={'num': int},
...                   sep='/'
...                  )
>>> ln.mk('USER', num=123)  # making a string (with args or kwargs)
'/home/USER/fav/123.txt'
>>> ####### prefix methods #######
>>> ln.is_valid_prefix('/home/USER/fav/')
True
>>> ln.is_valid_prefix('/home/USER/fav/12')  # False because too long
False
>>> ln.is_valid_prefix('/home/USER/fav')  # False because too short
False
>>> ln.is_valid_prefix('/home/')  # True because just right
True
>>> ln.is_valid_prefix('/home/USER/fav/123.txt')  # full path, so output same as is_valid() method
True
>>>
>>> ln.mk_prefix('ME')
'/home/ME/fav/'
>>> ln.mk_prefix(user='YOU', num=456)  # full specification, so output same as same as mk() method
'/home/YOU/fav/456.txt'
```

#### is_valid_prefix(s)

Check if name is a valid prefix.

* **Parameters:**
  **s** – a string (that might or might not be a valid prefix)
* **Returns:**
  True iff name is a valid prefix

### dol.naming.dict_to_namedtuple(d, namedtuple_obj=None)

```pycon
>>> from collections import namedtuple
>>> NT = namedtuple('MyTuple', ('foo', 'hello'))
>>> nt = NT(1, 42)
>>> nt
MyTuple(foo=1, hello=42)
>>> d = namedtuple_to_dict(nt)
>>> d
{'foo': 1, 'hello': 42}
>>> dict_to_namedtuple(d)
NamedTupleFromDict(foo=1, hello=42)
>>> dict_to_namedtuple(d, nt)
MyTuple(foo=1, hello=42)
```

### dol.naming.get_fields_from_template(template)

Get list from {item} items of template string

* **Parameters:**
  **template** – a “template” string (a string with {item} items
  – the kind that is used to mark token for str.format)
* **Returns:**
  a list of the token items of the string, in the order they appear

```pycon
>>> get_fields_from_template('this{is}an{example}of{a}template')
['is', 'example', 'a']
```

### dol.naming.mk_kwargs_trans(\*\*trans_func_for_key)

Make a dict transformer from functions that depends solely on keys (of the dict to be transformed)
Used to easily make process_kwargs and process_info_dict arguments for LinearNaming.

### dol.naming.mk_pattern_from_template_and_format_dict(template, format_dict=None, sep='/')

Make a compiled regex to match template

* **Parameters:**
  * **template** – A format string
  * **format_dict** – A dict whose keys are template fields and values are regex strings to capture them
* **Returns:**
  a compiled regex

Assert on *behavior* (matching) rather than the exact pattern string, so the
examples hold on every OS (the field separator – and therefore the default
capture class – is `/` on POSIX and `\` on Windows):

```pycon
>>> p = mk_pattern_from_template_and_format_dict('{here}/and/{there}')
>>> type(p)
<class 're.Pattern'>
>>> p.match('HERE/and/1234').groupdict()
{'here': 'HERE', 'there': '1234'}
>>> p = mk_pattern_from_template_and_format_dict('{here}/and/{there}', {'there': r'\d+'})
>>> p.match('HERE/and/1234').groupdict()
{'here': 'HERE', 'there': '1234'}
>>> p.match('HERE/and/not_digits') is None  # 'there' must be digits
True
```

### dol.naming.mk_store_from_path_format_store_cls(store=None, \*, subpath='', store_cls_kwargs=None, key_type=<function namedtuple>, keymap=<class 'dol.naming.StrTupleDict'>, keymap_kwargs=None, name=None, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Wrap a store (instance or class) that uses string keys to make it into a store that uses a specific key format.

* **Parameters:**
  * **store** – The instance or class to wrap
  * **subpath** – The subpath (defining the subset of the data pointed at by the URI
  * **store_cls_kwargs** – # if store is a class, the kwargs that you would have given the store_cls to make itself
  * **key_type** – The key type you want to interface with: `dict`, `tuple`,
    `namedtuple`, `str`, or one of those names as a string
  * **keymap** – # the keymap instance or class you want to use to map keys
  * **keymap_kwargs** – # if keymap is a cls, the kwargs to give it (besides the subpath)
  * **name** – The name to give the class the function will make here
* **Returns:**
  An instance of a wrapped class

### Example

```python
# Get a (session, bt) indexed LocalJsonStore
s = mk_store_from_path_format_store_cls(LocalJsonStore,
                                               os.path.join(root_dir, 'd'),
                                               subpath='{session}/d/{bt}',
                                               keymap_kwargs=dict(process_info_dict={'session': int, 'bt': int}))
```

### dol.naming.mk_tupled_store_from_path_format_store_cls(store=None, \*, subpath='', store_cls_kwargs=None, key_type=<function namedtuple>, keymap=<class 'dol.naming.StrTupleDict'>, keymap_kwargs=None, name=None, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Wrap a store (instance or class) that uses string keys to make it into a store that uses a specific key format.

* **Parameters:**
  * **store** – The instance or class to wrap
  * **subpath** – The subpath (defining the subset of the data pointed at by the URI
  * **store_cls_kwargs** – # if store is a class, the kwargs that you would have given the store_cls to make itself
  * **key_type** – The key type you want to interface with: `dict`, `tuple`,
    `namedtuple`, `str`, or one of those names as a string
  * **keymap** – # the keymap instance or class you want to use to map keys
  * **keymap_kwargs** – # if keymap is a cls, the kwargs to give it (besides the subpath)
  * **name** – The name to give the class the function will make here
* **Returns:**
  An instance of a wrapped class

### Example

```python
# Get a (session, bt) indexed LocalJsonStore
s = mk_store_from_path_format_store_cls(LocalJsonStore,
                                               os.path.join(root_dir, 'd'),
                                               subpath='{session}/d/{bt}',
                                               keymap_kwargs=dict(process_info_dict={'session': int, 'bt': int}))
```

### dol.naming.namedtuple_to_dict(nt)

```pycon
>>> from collections import namedtuple
>>> NT = namedtuple('MyTuple', ('foo', 'hello'))
>>> nt = NT(1, 42)
>>> nt
MyTuple(foo=1, hello=42)
>>> d = namedtuple_to_dict(nt)
>>> d
{'foo': 1, 'hello': 42}
```

### dol.naming.template_to_pattern(mapping_dict, template)

Weave a `{field}` template into a regex, substituting each field with its
capture pattern and **regex-escaping the literal text between fields**.

Escaping the literals is what makes this OS-independent: a template that is (or
contains) a real filesystem path has backslashes on Windows (`C:\Users\...`),
which are regex metacharacters – compiling them unescaped raises
`re.error: incomplete escape \U`. Escaping also makes a literal `.` match a
literal dot rather than any character. (This mirrors `KeyTemplate._compile_regex`
and is why the result is never routed through `safe_compile`, which would
re.escape the *whole* pattern on Windows and corrupt the capture groups.)

### dol.naming.update_fields_of_namedtuple(nt, , name_of_output_type=None, remove_fields=(), \*\*kwargs)

Replace fields of namedtuple

```pycon
>>> from collections import namedtuple
>>> NT = namedtuple('NT', ('a', 'b', 'c'))
>>> nt = NT(1,2,3)
>>> nt
NT(a=1, b=2, c=3)
>>> update_fields_of_namedtuple(nt, c=3000)  # replacing a single field
NT(a=1, b=2, c=3000)
>>> update_fields_of_namedtuple(nt, c=3000, a=1000)  # replacing two fields
NT(a=1000, b=2, c=3000)
>>> update_fields_of_namedtuple(nt, a=1000, c=3000)  # see that the original order doesn't change
NT(a=1000, b=2, c=3000)
>>> update_fields_of_namedtuple(nt, b=2000, d='hello')  # replacing one field and adding a new one
UpdatedNT(a=1, b=2000, c=3, d='hello')
>>> # Now let's try controlling the name of the output type, remove fields, and add new ones
>>> update_fields_of_namedtuple(nt, name_of_output_type='NewGuy', remove_fields=('a', 'c'), hello='world')
NewGuy(b=2, hello='world')
```

### dol.naming.validate_kwargs(kwargs_to_validate, validation_dict, validation_funs=None, all_kwargs_should_be_in_validation_dict=False, ignore_misunderstood_validation_instructions=False)

Utility to validate a dict. It’s main use is to validate function arguments (expressing the validation checks
in validation_dict) by doing validate_kwargs(locals()), usually in the beginning of the function
(to avoid having more accumulated variables than we need in locals())

* **Parameters:**
  * **kwargs_to_validate** – as the name implies…
  * **validation_dict** – A dict specifying what to validate. Keys are usually name of variables (when feeding
    locals()) and values are dicts, themselves specifying check:check_val pairs where check is a string that
    points to a function (see validation_funs argument) and check_val is an object that the kwargs_to_validate
    value will be checked against.
  * **validation_funs** – A dict of check:check_function(val, check_val) where check_function is a function returning
    True if val is valid (with respect to check_val).
  * **all_kwargs_should_be_in_validation_dict** – If True, will raise an error if kwargs_to_validate contains
    keys that are not in validation_dict.
  * **ignore_misunderstood_validation_instructions** – If True, will raise an error if validation_dict contains
    a key that is not in validation_funs (safer, since if you mistype a key in validation_dict, the function will
    tell you so!
* **Returns:**
  True if all the validations passed.

```pycon
>>> validation_dict = {
...     'system': {
...         'be in': {'darwin', 'linux'}
...     },
...     'fv_version': {
...         'be a': int,
...         'be at least': 5
...     }
... }
>>> validate_kwargs({'system': 'darwin'}, validation_dict)
True
>>> try:
...     validate_kwargs({'system': 'windows'}, validation_dict)
... except AssertionError as e:
...     assert str(e).startswith('system must be in')  # omitting the set because inconsistent order
>>> try:
...     validate_kwargs({'fv_version': 9.9}, validation_dict)
... except AssertionError as e:
...     print(e)
fv_version must be a <class 'int'>
>>> try:
...     validate_kwargs({'fv_version': 4}, validation_dict)
... except AssertionError as e:
...     print(e)
fv_version must be at least 5
>>> validate_kwargs({'fv_version': 6}, validation_dict)
True
```
