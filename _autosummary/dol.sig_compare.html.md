# dol.sig_compare

Compare signatures, and check whether calls to one function are valid calls to another.

- `compare_signatures`, `param_comparator`, `keyed_comparator`: comparison tools
- `param_differences_dict`, `SigPair`: describe how two signatures differ
- `is_call_compatible_with`: is every valid call to one signature a valid call to the
  other? (`Sig.is_call_compatible_with` calls it)

It used to be part of `dol.signatures`, which still re-exports all of it
(i2mint/dol#70).

```pycon
>>> from dol.sig_compare import is_call_compatible_with
>>> from dol.signatures import Sig
>>> is_call_compatible_with(Sig('(a, b)'), Sig('(a, b, c=3)'))
True
>>> is_call_compatible_with(Sig('(a, b, c=3)'), Sig('(a, b)'))
False
```

### Functions

| `compare_signatures`(func1, func2[, ...])                                                           |                                                                              |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| [`defaults_are_the_same_when_not_empty`](#dol.sig_compare.defaults_are_the_same_when_not_empty)(dflt1, ...)   | Check if two defaults are the same when they are not empty.                  |
| [`dflt1_is_empty_or_dflt2_is_not`](#dol.sig_compare.dflt1_is_empty_or_dflt2_is_not)(dflt1, dflt2)       | Why such a strange default comparison function?                              |
| `ignore_any_differences`(x, y)                                                                      |                                                                              |
| [`is_call_compatible_with`](#dol.sig_compare.is_call_compatible_with)(sig1, sig2, \*[, ...])     | Return True if `sig1` is compatible with `sig2`.                             |
| [`keyed_comparator`](#dol.sig_compare.keyed_comparator)(comparator, key)                  | Create a key-function enabled binary operator.                               |
| `mk_func_comparator_based_on_signature_comparator`(...)                                             |                                                                              |
| `param_attribute_dict`(...)                                                                         |                                                                              |
| [`param_binary_func`](#dol.sig_compare.param_binary_func)(param1, param2, \*[, name, ...]) | Compare two parameters.                                                      |
| [`param_comparator`](#dol.sig_compare.param_comparator)(param1, param2, \*[, name, ...])  | Compare two parameters.                                                      |
| [`param_differences_dict`](#dol.sig_compare.param_differences_dict)(param1, param2, \*[, ...])  | Makes a dictionary exibiting the differences between two parameters.         |
| [`postprocess`](#dol.sig_compare.postprocess)(egress)                                | A decorator that will process the output of the wrapped function with egress |
| `return_tuple`(x, y)                                                                                |                                                                              |

### Classes

| [`SigPair`](#dol.sig_compare.SigPair)(sig1, sig2)   | Class that operates on a pair of signatures.   |
|------------------------------------------------------------------------|------------------------------------------------|

### *class* dol.sig_compare.SigPair(sig1, sig2)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Class that operates on a pair of signatures.

For example, offers methods to compare two signatures in various ways.

* **Parameters:**
  * **sig1** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`Sig`](dol.signatures.html.md#dol.signatures.Sig)) – First signature or signature-able object.
  * **sig2** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`Sig`](dol.signatures.html.md#dol.signatures.Sig)) – Second signature or signature-able object.

```pycon
>>> from pprint import pprint
>>> def three(a, b: int, c=3): ...
>>> def little(a, *, b=2, d=4) -> int: ...
>>> def pigs(a, b) -> int: ...
>>> sig_pair = SigPair(three, little)
>>>
>>> sig_pair.shared_names
['a', 'b']
>>> sig_pair.names_missing_in_sig1
['d']
>>> sig_pair.names_missing_in_sig2
['c']
>>> sig_pair.param_comparison()
False
>>> pprint(sig_pair.diff())
{'names_missing_in_sig1': ['d'],
'names_missing_in_sig2': ['c'],
'param_differences': {'b': {'annotation': (<class 'int'>,
                                            <class 'inspect._empty'>),
                            'default': (<class 'inspect._empty'>, 2),
                            'kind': (<_ParameterKind.POSITIONAL_OR_KEYWORD: 1>,
                                    <_ParameterKind.KEYWORD_ONLY: 3>)}},
'return_annotation': (<class 'inspect._empty'>, <class 'int'>)}
```

Call compatibility says that any arguments leading to a valid call to a function
having the first signature, will also lead to a valid call to a function having the
second signature. This is not the case for the signatures of `three` and `little`:

```pycon
>>> sig_pair.are_call_compatible()
False
```

But we don’t need to have equal signatures to have call compatibility. For example,

```pycon
>>> SigPair(three, lambda a, b=2, c=30: None).are_call_compatible()
True
```

Note that call-compatibility is not symmetric. For example, `pigs` is call
compatible with `three`, since any arguments that are valid for `pigs` are valid
for `three`:

```pycon
>>> SigPair(pigs, three).are_call_compatible()
True
```

But `three` is not call-compatible with `pigs` since `three` requires could include
a `c` argument, which `pigs` would choke on.

```pycon
>>> SigPair(three, pigs).are_call_compatible()
False
```

#### are_call_compatible(param_comparator=None)

Check if the signatures are call-compatible.

Returns True if sig1 can be used to call sig2 or vice versa.

```pycon
>>> sig1 = Sig(lambda a, b, c=3: None)
>>> sig2 = Sig(lambda a, b: None)
>>> comp = SigPair(sig1, sig2)
>>> comp.are_call_compatible()
False
```

```pycon
>>> comp = SigPair(sig2, sig1)
>>> comp.are_call_compatible()
True
```

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### diff()

Get a dictionary of differences between the two signatures.

```pycon
>>> from pprint import pprint
>>> def three(a, b: int, c=3): ...
>>> def little(a, *, b=2, d=4) -> int: ...
>>> def pigs(a, b: int = 2) -> int: ...
>>> pprint(SigPair(three, little).diff())
{'names_missing_in_sig1': ['d'],
'names_missing_in_sig2': ['c'],
'param_differences': {'b': {'annotation': (<class 'int'>,
                                            <class 'inspect._empty'>),
                            'default': (<class 'inspect._empty'>, 2),
                            'kind': (<_ParameterKind.POSITIONAL_OR_KEYWORD: 1>,
                                    <_ParameterKind.KEYWORD_ONLY: 3>)}},
'return_annotation': (<class 'inspect._empty'>, <class 'int'>)}
>>> pprint(SigPair(three, pigs).diff())
{'names_missing_in_sig2': ['c'],
'param_differences': {'b': {'default': (<class 'inspect._empty'>, 2)}},
'return_annotation': (<class 'inspect._empty'>, <class 'int'>)}
>>> pprint(SigPair(three, three).diff())
{}
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### diff_str()

Get a string representation of the differences between the two signatures.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### *property* names_missing_in_sig1

List of names that are in the sig2 signature but not in sig1.

```pycon
>>> sig1 = Sig(lambda a, b, c: None)
>>> sig2 = Sig(lambda b, c, d: None)
>>> comp = SigPair(sig1, sig2)
>>> comp.names_missing_in_sig1
['d']
```

#### *property* names_missing_in_sig2

List of names that are in the sig1 signature but not in sig2.

```pycon
>>> sig1 = Sig(lambda a, b, c: None)
>>> sig2 = Sig(lambda b, c, d: None)
>>> comp = SigPair(sig1, sig2)
>>> comp.names_missing_in_sig2
['a']
```

#### param_comparison(comparator=<function param_comparator>, aggregation=<built-in function all>)

Compare parameters between the two signatures using the provided comparator function.

* **Parameters:**
  * **comparator** – A function to compare two parameters.
  * **aggregation** – A function to aggregate the results of the comparisons.
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  Boolean result of the aggregated comparisons.

```pycon
>>> sig1 = Sig('(a, b: int, c=3)')
>>> sig2 = Sig('(a, *, b=2, d=4)')
>>> comp = SigPair(sig1, sig2)
>>> comp.param_comparison()
False
```

#### param_differences()

Get a dictionary of parameter differences between the two signatures.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  A dict containing differences for each shared param that has any.

```pycon
>>> sig1 = Sig('(a, b: int, c=3)')
>>> sig2 = Sig('(a, *, b=2, d=4)')
>>> comp = SigPair(sig1, sig2)
>>> result = comp.param_differences()
>>> expected = {
...     'b': {
...         'kind': (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY),
...         'default': (Parameter.empty, 2),
...         'annotation': (int, Parameter.empty),
...     }
... }
>>> result == expected
True
```

#### *property* shared_names

List of names that are common to both signatures, in the order of sig1.

```pycon
>>> sig1 = Sig(lambda a, b, c: None)
>>> sig2 = Sig(lambda b, c, d: None)
>>> comp = SigPair(sig1, sig2)
>>> comp.shared_names
['b', 'c']
```

### dol.sig_compare.defaults_are_the_same_when_not_empty(dflt1, dflt2)

Check if two defaults are the same when they are not empty.

```pycon
>>> defaults_are_the_same_when_not_empty(1, 1)
True
>>> defaults_are_the_same_when_not_empty(1, 2)
False
>>> defaults_are_the_same_when_not_empty(1, None)
False
>>> defaults_are_the_same_when_not_empty(1, Parameter.empty)
True
```

### dol.sig_compare.dflt1_is_empty_or_dflt2_is_not(dflt1, dflt2)

Why such a strange default comparison function?

This is to be used as a default in is_call_compatible_with.

Consider two functions func1 and func2 with a parameter p with default values
dflt1 and dflt2 respectively.
If dflt1 was not empty and dflt2 was, this would mean that func1 could be called
without specifying p, but func2 couldn’t.

So to avoid this situation, we use dflt1_is_empty_or_dflt2_is_not as the default

### dol.sig_compare.dflt1_is_empty_or_dflt2_is_not_param_comparator(param1, param2, \*, name=<function ignore_any_differences>, kind=<function ignore_any_differences>, default=<function dflt1_is_empty_or_dflt2_is_not>, annotation=<function ignore_any_differences>, aggreg=<built-in function all>)

Permissive version of param_comparator that ignores any differences of parameter
attributes.

It is meant to be used with partial, but with a permissive base, contrary to the
base param_comparator which requires strict equality (`eq`) for all attributes.

* **Return type:**
  `Comparison`

### dol.sig_compare.is_call_compatible_with(sig1, sig2, , param_comparator=None)

Return True if `sig1` is compatible with `sig2`. Meaning that all valid ways
to call `sig1` are valid for `sig2`.

* **Parameters:**
  * **sig1** ([`Sig`](dol.signatures.html.md#dol.signatures.Sig)) – The main signature.
  * **sig2** ([`Sig`](dol.signatures.html.md#dol.signatures.Sig)) – The signature to be compared with.
  * **param_comparator** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter), [`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)], `Comparison`] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The function used to compare two parameters
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

```pycon
>>> is_call_compatible_with(
...     Sig('(a, /, b, *, c)'),
...     Sig('(a, b, c)')
... )
True
>>> is_call_compatible_with(
...     Sig('()'),
...     Sig('(a)')
... )
False
>>> is_call_compatible_with(
...     Sig('()'),
...     Sig('(a=0)')
... )
True
>>> is_call_compatible_with(
...     Sig('(a, /, *, c)'),
...     Sig('(a, /, b, *, c)')
... )
False
>>> is_call_compatible_with(
...     Sig('(a, /, *, c)'),
...     Sig('(a, /, b=0, *, c)')
... )
True
>>> is_call_compatible_with(
...     Sig('(a, /, b)'),
...     Sig('(a, /, b, *, c)')
... )
False
>>> is_call_compatible_with(
...     Sig('(a, /, b)'),
...     Sig('(a, /, b, *, c=0)')
... )
True
>>> is_call_compatible_with(
...     Sig('(a, /, b, *, c)'),
...     Sig('(*args, **kwargs)')
... )
True
```

### dol.sig_compare.keyed_comparator(comparator, key)

Create a key-function enabled binary operator.

In various places in python functionality is extended by allowing a key function.
For example, the `sorted` function allows a key function to be passed, which is
applied to each element before sorting. The keyed_comparator function allows a
comparator to be extended in the same way. The returned comparator will apply the
key function toeach input before applying the original comparator.

```pycon
>>> from operator import eq
>>> parity = lambda x: x % 2
>>> comparator = keyed_comparator(eq, parity)
>>> list(map(comparator, [1, 1, 2, 2], [3, 4, 5, 6]))
[True, False, False, True]
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`Compared`, `Compared`], `Comparison`]

### dol.sig_compare.param_binary_func(param1, param2, \*, name=<built-in function eq>, kind=<built-in function eq>, default=<built-in function eq>, annotation=<built-in function eq>, aggreg=<built-in function all>)

Compare two parameters.

Note that by default, this function is strict, and will return False if
any of the parameters are not equal. This is because the default
aggregation function is `all` and the default comparison functions of the
parameter’s attributes are `eq` (meaning equality, not identity).

But you can change that by passing different comparison functions and/or
aggregation functions.

In fact, the real purpose of this function is to be used as a factory of parameter
binary functions, through parametrizing it with `functools.partial`.

The parameter binary functions themselves are meant to be used to make signature
binary functions.

* **Parameters:**
  * **param1** ([`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)) – first parameter
  * **param2** ([`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)) – second parameter
  * **name** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`Compared`, `Compared`], `Comparison`]) – function to compare names
  * **kind** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`Compared`, `Compared`], `Comparison`]) – function to compare kinds
  * **default** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`Compared`, `Compared`], `Comparison`]) – function to compare defaults
  * **annotation** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`Compared`, `Compared`], `Comparison`]) – function to compare annotations
  * **aggreg** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`Comparison`]], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – function to aggregate results
* **Return type:**
  `Comparison`

```pycon
>>> from inspect import Parameter
>>> param1 = Parameter('x', Parameter.POSITIONAL_OR_KEYWORD)
>>> param2 = Parameter('x', Parameter.POSITIONAL_OR_KEYWORD)
>>> param_binary_func(param1, param2)
True
```

See [https://github.com/i2mint/i2/issues/50#issuecomment-1381686812](https://github.com/i2mint/i2/issues/50#issuecomment-1381686812) for discussion.

### dol.sig_compare.param_comparator(param1, param2, \*, name=<built-in function eq>, kind=<built-in function eq>, default=<built-in function eq>, annotation=<built-in function eq>, aggreg=<built-in function all>)

Compare two parameters.

Note that by default, this function is strict, and will return False if
any of the parameters are not equal. This is because the default
aggregation function is `all` and the default comparison functions of the
parameter’s attributes are `eq` (meaning equality, not identity).

But you can change that by passing different comparison functions and/or
aggregation functions.

In fact, the real purpose of this function is to be used as a factory of parameter
binary functions, through parametrizing it with `functools.partial`.

The parameter binary functions themselves are meant to be used to make signature
binary functions.

* **Parameters:**
  * **param1** ([`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)) – first parameter
  * **param2** ([`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)) – second parameter
  * **name** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`Compared`, `Compared`], `Comparison`]) – function to compare names
  * **kind** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`Compared`, `Compared`], `Comparison`]) – function to compare kinds
  * **default** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`Compared`, `Compared`], `Comparison`]) – function to compare defaults
  * **annotation** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`Compared`, `Compared`], `Comparison`]) – function to compare annotations
  * **aggreg** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`Comparison`]], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – function to aggregate results
* **Return type:**
  `Comparison`

```pycon
>>> from inspect import Parameter
>>> param1 = Parameter('x', Parameter.POSITIONAL_OR_KEYWORD)
>>> param2 = Parameter('x', Parameter.POSITIONAL_OR_KEYWORD)
>>> param_binary_func(param1, param2)
True
```

See [https://github.com/i2mint/i2/issues/50#issuecomment-1381686812](https://github.com/i2mint/i2/issues/50#issuecomment-1381686812) for discussion.

### dol.sig_compare.param_comparison_dict(param1, param2, \*, name=<function return_tuple>, kind=<function return_tuple>, default=<function return_tuple>, annotation=<function return_tuple>, aggreg=<function param_attribute_dict>)

A ParamComparator that returns a dictionary with pairs parameter attributes.

```pycon
>>> param1 = Sig('(a: int = 1)')['a']
>>> param2 = Sig('(a: str = 2)')['a']
>>> param_comparison_dict(param1, param2)
{'name': ('a', 'a'), 'kind': ..., 'default': (1, 2), 'annotation': (<class 'int'>, <class 'str'>)}
```

* **Return type:**
  `Comparison`

### dol.sig_compare.param_differences_dict(param1, param2, \*, name=<built-in function eq>, kind=<built-in function eq>, default=<built-in function eq>, annotation=<built-in function eq>)

Makes a dictionary exibiting the differences between two parameters.

```pycon
>>> param1 = Sig('(a: int = 1)')['a']
>>> param2 = Sig('(a: str = 2)')['a']
>>> param_differences_dict(param1, param2)
{'default': (1, 2), 'annotation': (<class 'int'>, <class 'str'>)}
>>> param_differences_dict(param1, param2, default=lambda x, y: isinstance(x, type(y)))
{'annotation': (<class 'int'>, <class 'str'>)}
```

### dol.sig_compare.permissive_param_comparator(param1, param2, \*, name=<function ignore_any_differences>, kind=<function ignore_any_differences>, default=<function ignore_any_differences>, annotation=<function ignore_any_differences>, aggreg=<built-in function all>)

Permissive version of param_comparator that ignores any differences of parameter
attributes.

It is meant to be used with partial, but with a permissive base, contrary to the
base param_comparator which requires strict equality (`eq`) for all attributes.

* **Return type:**
  `Comparison`

### dol.sig_compare.postprocess(egress)

A decorator that will process the output of the wrapped function with egress
