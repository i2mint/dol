# dol.signatures

Signature calculus: Tools to make it easier to work with function’s signatures.

How to:

> - get names, kinds, defaults, annotations
> - make signatures flexibly
> - merge two or more signatures
> - give a function a specific signature (with a choice of validations)
> - get an equivalent function with a different order of arguments
> - get an equivalent function with a subset of arguments (like partial)
> - get an equivalent function but with variadic `*args` and/or `**kwargs` replaced with
>   non-variadic args (tuple) and kwargs (dict)
> - make an f(a) function in to a f(a, b=None) function with b ignored

Get names, kinds, defaults, annotations:

```pycon
>>> def func(z, a: float=1.0, /, b=2, *, c: int=3):
...     pass
>>> sig = Sig(func)
>>> sig.names
['z', 'a', 'b', 'c']
>>> from inspect import Parameter
>>> assert sig.kinds == {
...     'z': Parameter.POSITIONAL_ONLY,
...     'a': Parameter.POSITIONAL_ONLY,
...     'b': Parameter.POSITIONAL_OR_KEYWORD,
...     'c': Parameter.KEYWORD_ONLY
... }
>>> # Note z is not in there (only defaulted params are included)
>>> sig.defaults
{'a': 1.0, 'b': 2, 'c': 3}
>>> sig.annotations
{'a': <class 'float'>, 'c': <class 'int'>}
```

Make signatures flexibly:

```pycon
>>> Sig(func)
<Sig (z, a: float = 1.0, /, b=2, *, c: int = 3)>
>>> Sig(['a', 'b'])
<Sig (a, b)>
>>> Sig('x y z')
<Sig (x, y, z)>
```

Merge signatures.

```pycon
>>> def foo(x): pass
>>> def bar(y: int, *, z=2): pass  # note the * (keyword only) will be lost!
>>> Sig(foo) + ['a', 'b'] + Sig(bar)
<Sig (x, a, b, y: int, z=2)>
```

Give a function a signature.

```pycon
>>> @Sig('a b c')
... def func(*args, **kwargs):
...     print(args, kwargs)
>>> Sig(func)
<Sig (a, b, c)>
```

**Notes to the reader**

Both in the code and in the docs, we’ll use short hands for parameter (argument) kind.

> - PK = Parameter.POSITIONAL_OR_KEYWORD
> - VP = Parameter.VAR_POSITIONAL
> - VK = Parameter.VAR_KEYWORD
> - PO = Parameter.POSITIONAL_ONLY
> - KO = Parameter.KEYWORD_ONLY

### Functions

| `assure_callable`(obj)                                                                       |                                                                                               |
|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| [`assure_params`](#dol.signatures.assure_params)([obj])                        | Get an interable of Parameter instances from an object.                                       |
| `assure_signature`(obj)                                                                      |                                                                                               |
| `deprecation_of`(func, old_name)                                                             |                                                                                               |
| `ensure_callable`(obj)                                                                       |                                                                                               |
| `ensure_param`(p)                                                                            |                                                                                               |
| [`ensure_params`](#dol.signatures.ensure_params)([obj])                        | Get an interable of Parameter instances from an object.                                       |
| `ensure_signature`(obj)                                                                      |                                                                                               |
| `expand_nested_key`(d, k)                                                                    |                                                                                               |
| [`extract_arguments`](#dol.signatures.extract_arguments)(params, \*[, ...])        | Extract arguments needed to satisfy the params of a callable, dealing with the dirty details. |
| `flatten_if_var_kw`(kvs, var_kw_name)                                                        |                                                                                               |
| `function_caller`(func, args, kwargs)                                                        |                                                                                               |
| [`is_signature_error`](#dol.signatures.is_signature_error)(e)                       | Check if an exception is a signature error                                                    |
| `maybe_first`(items)                                                                         |                                                                                               |
| [`name_of_obj`](#dol.signatures.name_of_obj)(o, \*[, base_name_of_obj, ...]) | Tries to find the (or "a") name for an object, even if `__name__` doesn't exist.              |
| `name_of_var_kw_argument`(sig)                                                               |                                                                                               |
| [`param_for_kind`](#dol.signatures.param_for_kind)([name, kind, with_default])  | Function to easily and flexibly make inspect.Parameter objects for testing.                   |
| `param_has_default_or_is_var_kind`(p)                                                        |                                                                                               |
| `parameter_to_dict`(p)                                                                       |                                                                                               |
| [`sort_params`](#dol.signatures.sort_params)(params)                         |                                                                                               |
| [`validate_signature`](#dol.signatures.validate_signature)(func)                    | Validates the signature of a function.                                                        |

### Classes

| [`MissingArgValFor`](#dol.signatures.MissingArgValFor)(argname)                    | A simple class to wrap an argument name, indicating that it was missing somewhere.                                                                                     |
|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`P`](#dol.signatures.P)                                            |                                                                                                                                                                        |
| [`Param`](#dol.signatures.Param)(name[, kind])                          | A thin wrap of Parameters: Adds shorter aliases to argument kinds and a POSITIONAL_OR_KEYWORD default to the argument kind to make it faster to make Parameter objects |
| [`Sig`](#dol.signatures.Sig)([obj, name, \_\_validate_parameters_\_]) | A subclass of inspect.Signature that has a lot of extra api sugar, such as                                                                                             |

### Exceptions

| [`FuncCallNotMatchingSignature`](#dol.signatures.FuncCallNotMatchingSignature)                 | Raise when the call signature is not valid   |
|-----------------------------------------------------------------------------------------------|----------------------------------------------|
| [`IncompatibleSignatures`](#dol.signatures.IncompatibleSignatures)(\*args[, sig1, sig2]) |                                              |
| [`InvalidSignature`](#dol.signatures.InvalidSignature)                             | Raise when a signature is not valid          |

### *exception* dol.signatures.FuncCallNotMatchingSignature

Bases: [`TypeError`](https://docs.python.org/3/builtins/exceptions.html#TypeError)

Raise when the call signature is not valid

### *exception* dol.signatures.IncompatibleSignatures(\*args, sig1=None, sig2=None, \*\*kwargs)

Bases: [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

#### pformat(indent=1, width=80, depth=None, , compact=False, sort_dicts=True, underscore_numbers=False)

Format a Python object into a pretty-printed representation.

### *exception* dol.signatures.InvalidSignature

Bases: [`SyntaxError`](https://docs.python.org/3/builtins/exceptions.html#SyntaxError), [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

Raise when a signature is not valid

### *class* dol.signatures.MissingArgValFor(argname)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A simple class to wrap an argument name, indicating that it was missing somewhere.

```pycon
>>> MissingArgValFor("argname")
MissingArgValFor("argname")
```

### dol.signatures.P

alias of [`Param`](#dol.signatures.Param)

### *class* dol.signatures.Param(name, kind=\_ParameterKind.POSITIONAL_OR_KEYWORD, , default, annotation)

Bases: [`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)

A thin wrap of Parameters: Adds shorter aliases to argument kinds and
a POSITIONAL_OR_KEYWORD default to the argument kind to make it faster to make
Parameter objects

```pycon
>>> list(map(Param, 'some quick arg params'.split()))
[<Param "some">, <Param "quick">, <Param "arg">, <Param "params">]
>>> from inspect import Signature
>>> P = Param
>>> Signature([P('x', P.PO), P('y', default=42, annotation=int), P('kw', P.KO)])
<Signature (x, /, y: int = 42, *, kw)>
```

### *class* dol.signatures.Sig(obj=None, , name=None, return_annotation, \_\_validate_parameters_\_=True)

Bases: [`Signature`](https://docs.python.org/3/library/inspect.html#inspect.Signature), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

A subclass of inspect.Signature that has a lot of extra api sugar,
such as

- making a signature for a variety of input types (callable,
  iterable of callables, parameter lists, strings, etc.)
- has a dict-like interface
- signature merging (with operator interfaces)
- quick access to signature data
- positional/keyword argument mapping.

### Positional/Keyword argument mapping

In python, arguments can be positional (args) or keyword (kwargs).
… sometimes both, sometimes a single one is imposed.
… and you have variadic versions of both.
… and you can have defaults or not.
… and all these different kinds have a particular order they must be in.
It’s is mess really. The flexibility is nice – but still; a mess.

You only really feel the mess if you try to do some meta-programming with your
functions.
Then, methods like `normalize_kind` can help you out, since you can enforce, and
then assume, some stable interface to your functions.

Two of the base methods for dealing with positional (args) and keyword (kwargs)
inputs are:

> - `map_arguments`: Map some args/kwargs input to a keyword-only
>   expression of the inputs. This is useful if you need to do some processing
>   based on the argument names.
> - `mk_args_and_kwargs`: Translate a fully keyword expression of some
>   inputs into an (args, kwargs) pair that can be used to call the function.
>   (Remember, your function can have constraints, so you may need to do this.

The usual pattern of use of these methods is to use `map_arguments`
to map all the inputs to their corresponding name, do what needs to be done with
that (example, validation, transformation, decoration…) and then map back to an
(args, kwargs) pair than can actually be used to call the function.

Examples of methods and functions using these:
`call_forgivingly`, `tuple_the_args`, `map_arguments_from_variadics`, `extract_args_and_kwargs`,
`source_arguments`, and `source_args_and_kwargs`.

### Making a signature

You can construct a `Sig` object from a callable,

```pycon
>>> def f(w, /, x: float = 1, y=1, *, z: int = 1):
...     ...
>>> Sig(f)
<Sig (w, /, x: float = 1, y=1, *, z: int = 1)>
```

but also from any “ParamsAble” object. Such as…
an iterable of Parameter instances, strings, tuples, or dicts:

```pycon
>>> Sig(
...     [
...         "a",
...         ("b", Parameter.empty, int),
...         ("c", 2),
...         ("d", 1.0, float),
...         dict(name="special", kind=Parameter.KEYWORD_ONLY, default=0),
...     ]
... )
<Sig (a, b: int, c=2, d: float = 1.0, *, special=0)>
>>>
>>> Sig(
...     [
...         "a",
...         "b",
...         dict(name="args", kind=Parameter.VAR_POSITIONAL),
...         dict(name="kwargs", kind=Parameter.VAR_KEYWORD),
...     ]
... )
<Sig (a, b, *args, **kwargs)>
```

The parameters of a signature are like a matrix whose rows are the parameters,
and the 4 columns are their properties: name, kind, default, and annotation
(the two laste ones being optional).
You get a row view when doing `Sig(...).parameters.values()`,
but what if you want a column-view?
Here’s how:

```pycon
>>> def f(w, /, x: float = 1, y=2, *, z: int = 3):
...     ...
>>>
>>> s = Sig(f)
>>> s.kinds
{'w': <_ParameterKind.POSITIONAL_ONLY: 0>,
'x': <_ParameterKind.POSITIONAL_OR_KEYWORD: 1>,
'y': <_ParameterKind.POSITIONAL_OR_KEYWORD: 1>,
'z': <_ParameterKind.KEYWORD_ONLY: 3>}
```

```pycon
>>> s.annotations
{'x': <class 'float'>, 'z': <class 'int'>}
>>> assert (
...     s.annotations == f.__annotations__
... )  # same as what you get in `__annotations__`
>>>
>>> s.defaults
{'x': 1, 'y': 2, 'z': 3}
>>> # Note that it's not the same as you get in __defaults__ though:
>>> assert (
...     s.defaults != f.__defaults__ == (1, 2)
... )  # not 3, since __kwdefaults__ has that!
```

We can sum (i.e. merge) and subtract (i.e. remove arguments) Sig instances.
Also, Sig instance is callable. It has the effect of inserting it’s signature in
the input
(in `__signature__`, but also inserting the resulting `__defaults__` and
`__kwdefaults__`).
One of the intents is to be able to do things like:

```pycon
>>> import inspect
>>> def f(w, /, x: float = 1, y=1, *, z: int = 1):
...     ...
>>> def g(i, w, /, j=2):
...     ...
...
>>>
>>> @Sig.from_objs(f, g, ["a", ("b", 3.14), ("c", 42, int)])
... def some_func(*args, **kwargs):
...     ...
>>> inspect.signature(some_func)
<Sig (w, i, /, a, x: float = 1, y=1, j=2, b=3.14, c: int = 42, *, z: int = 1)>
>>>
>>> sig = Sig(f) + g + ["a", ("b", 3.14), ("c", 42, int)] - "b" - ["a", "z"]
>>> @sig
... def some_func(*args, **kwargs):
...     ...
>>> inspect.signature(some_func)
<Sig (w, i, x: float = 1, y=1, j=2, c: int = 42)>
```

#### add_optional_keywords(kwarg_and_defaults=None, kwarg_annotations=None)

Add optional keyword arguments to a signature.

```pycon
>>> @Sig.add_optional_keywords({"c": 2, "d": 3}, {"c": int})
... def foo(a, *, b=1, **kwargs):
...     return f"{a=}, {b=}, {kwargs=}"
...
```

You can still call the function as before, and like before, any “extra” keyword
arguments will be passed to kwargs:

```pycon
>>> foo(0, d=10)
"a=0, b=1, kwargs={'d': 10}"
```

The difference is that now the signature of `foo` now has `c` and `d`:

```pycon
>>> str(Sig(foo))
'(a, *, c: int = 2, d=3, b=1, **kwargs)'
```

#### add_params(params)

Creates a new instance of Sig after merging the parameters of this signature
with a list of new parameters. The new list of parameters is automatically
sorted based on signature constraints given by kinds and default values.
See Python native signature documentation for more details.

```pycon
>>> s = Sig('(a, /, b, *, c)')
>>> s.add_params([
...     Param('kwargs', VK),
...     dict(name='d', kind=KO),
...     Param('args', VP),
...     'e',
...     Param('f', PO),
... ])
<Sig (a, f, /, b, e, *args, c, d, **kwargs)>
```

#### *property* annotations

annotation, …} dict of annotations of the signature.
What `func.__annotations__` would give you.

* **Type:**
  {arg_name

#### args_and_kwargs_from_kwargs(arguments, , apply_defaults=False, allow_partial=False, allow_excess=False, ignore_kind=False, args_limit=0)

Extract args and kwargs such that `func(*args, **kwargs)` can be called,
where func has instance’s signature.

* **Parameters:**
  * **arguments** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The {param_name: arg_val,…} dict to process
  * **args_limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – 

    How “far” in the params should args (positional arguments)
    be searched for.
    - args_limit==0: Take the minimum number possible of args (positional
      arguments). Only those that are position only or before a var-positional.
    - args_limit is None: Take the maximum number of args (positional arguments).
      The only kwargs (keyword arguments) you should have are keyword-only
      and var-keyword arguments.
    - args_limit positive integer: Take the args_limit first argument names
      (of signature) as args, and the rest as kwargs.
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

```pycon
>>> def foo(w, /, x: float, y=1, *, z: int = 1):
...     return ((w + x) * y) ** z
>>> foo_sig = Sig(foo)
>>> args, kwargs = foo_sig.mk_args_and_kwargs(
...     dict(w=4, x=3, y=2, z=1)
... )
>>> assert (args, kwargs) == ((4,), {"x": 3, "y": 2, "z": 1})
>>> assert foo(*args, **kwargs) == foo(4, 3, 2, z=1) == 14
```

What about variadics?

```pycon
>>> def bar(a, /, b, *args, c=2, **kwargs):
...     pass
>>> Sig(bar).mk_args_and_kwargs(
...     dict(a=1, b=2, args=(3,4), c=5, kwargs=dict(d=6, e=7))
... )
((1, 2, 3, 4), {'c': 5, 'd': 6, 'e': 7})
```

You can also give the arguments in a different order:

```pycon
>>> Sig(bar).mk_args_and_kwargs(
...     dict(args=(3,4), kwargs=dict(d=6, e=7), b=2, c=5, a=1)
... )
((1, 2, 3, 4), {'c': 5, 'd': 6, 'e': 7})
```

The `args_limit` begs explanation.
Consider the signature of `def foo(w, /, x: float, y=1, *, z: int = 1): ...`
for instance. We could call the function with the following (args, kwargs) pairs:

- ((1,), {‘x’: 2, ‘y’: 3, ‘z’: 4})
- ((1, 2), {‘y’: 3, ‘z’: 4})
- ((1, 2, 3), {‘z’: 4})
  The two other combinations (empty args or empty kwargs) are not valid
  because of the / and \* constraints.

But when asked for an (args, kwargs) pair, which of the three valid options
should be returned? This is what the `args_limit` argument controls.

If `args_limit == 0`, the least args (positional arguments) will be returned.
It’s the default.

```pycon
>>> arguments = dict(w=4, x=3, y=2, z=1)
>>> foo_sig.mk_args_and_kwargs(arguments, args_limit=0)
((4,), {'x': 3, 'y': 2, 'z': 1})
```

If `args_limit is None`, the least kwargs (keyword arguments) will be returned.

```pycon
>>> foo_sig.mk_args_and_kwargs(arguments, args_limit=None)
((4, 3, 2), {'z': 1})
```

If `args_limit` is a positive integer, the first `[args_limit]` arguments
will be returned (not checking at all if this is valid!).

```pycon
>>> foo_sig.mk_args_and_kwargs(arguments, args_limit=1)
((4,), {'x': 3, 'y': 2, 'z': 1})
>>> foo_sig.mk_args_and_kwargs(arguments, args_limit=2)
((4, 3), {'y': 2, 'z': 1})
>>> foo_sig.mk_args_and_kwargs(arguments, args_limit=3)
((4, 3, 2), {'z': 1})
```

Note that if you specify `args_limit` to be greater than the maximum of
positional arguments, it behaves as if `args_limit` was `None`:

```pycon
>>> foo_sig.mk_args_and_kwargs(arguments, args_limit=4)
((4, 3, 2), {'z': 1})
```

Note that ‘args_limit’’s behavior is consistent with list behvior in the sense
that:

```pycon
>>> args = (0, 1, 2, 3)
>>> args[:0]
()
>>> args[:None]
(0, 1, 2, 3)
>>> args[2]
2
```

If variable positional arguments are present, `args_limit` is ignored and
all positional arguments are returned as args.

```pycon
>>> Sig(bar).mk_args_and_kwargs(
...     dict(a=1, b=2, args=(3,4), c=5, kwargs=dict(d=6, e=7)),
...     args_limit=1
... )
((1, 2, 3, 4), {'c': 5, 'd': 6, 'e': 7})
```

By default, only the arguments that were given in the `arguments` input will be
returned in the (args, kwargs) output.
If you also want to get those that have defaults (according to signature),
you need to specify it with the `apply_defaults=True` argument.

```pycon
>>> foo_sig.mk_args_and_kwargs(dict(w=4, x=3))
((4,), {'x': 3})
>>> foo_sig.mk_args_and_kwargs(dict(w=4, x=3), apply_defaults=True)
((4,), {'x': 3, 'y': 1, 'z': 1})
```

By default, all required arguments must be given.
Not doing so will lead to a `TypeError`.
If you want to process your arguments anyway, specify `allow_partial=True`.

```pycon
>>> foo_sig.mk_args_and_kwargs(dict(w=4))
Traceback (most recent call last):
  ...
TypeError: missing a required argument: 'x'
>>> foo_sig.mk_args_and_kwargs(dict(w=4), allow_partial=True)
((4,), {})
```

Specifying argument names that are not recognized by the signature will
lead to a `TypeError`.
If you want to avoid this (and just take from the input `kwargs` what ever you
can), specify this with `allow_excess=True`.

```pycon
>>> foo_sig.mk_args_and_kwargs(dict(w=4, x=3, extra='stuff'))
Traceback (most recent call last):
    ...
TypeError: Got unexpected keyword arguments: extra
>>> foo_sig.mk_args_and_kwargs(dict(w=4, x=3, extra='stuff'),
...     allow_excess=True)
((4,), {'x': 3})
```

See `map_arguments` (namely for the description of the arguments).

#### ch_param_attrs(param_attr, \*arg_new_vals, \_allow_reordering=False, \*\*kwargs_new_vals)

Change a specific attribute of the params, returning a modified signature.
This is a convenience method for the modified method when we’re targetting
a fixed param attribute: ‘name’, ‘kind’, ‘default’, or ‘annotation’

Instead of having to do this

```pycon
>>> def foo(a, *b, **c): ...
>>> Sig(foo).modified(a={'name': 'A'}, b={'name': 'B'}, c={'name': 'C'})
<Sig (A, *B, **C)>
```

We can simply do this

```pycon
>>> Sig(foo).ch_param_attrs('name', a='A', b='B', c='C')
<Sig (A, *B, **C)>
```

One quite useful thing you can do with this is to set defaults, or set defaults
where there are none. If you wrap your function with such a modified signature,
you get a “curried” version of your function (called “partial” in python).
(Note that the `functools.wraps` won’t deal with defaults “correctly”, but
wrapping with `Sig` objects takes care of that oversight!)

```pycon
>>> def foo(a, b, c):
...     return a + b * c
>>> special_foo = Sig(foo).ch_param_attrs('default', b=2, c=3)(foo)
>>> Sig(special_foo)
<Sig (a, b=2, c=3)>
>>> special_foo(5)  # should be 5 + 2 * 3 == 11
11
```

#### *property* defaults

A `{name: default,...}` dict of defaults (regardless of kind)

#### extract_args_and_kwargs(\*args, \_ignore_kind=True, \_allow_partial=False, \_allow_excess=True, \_apply_defaults=False, \_args_limit=0, \*\*kwargs)

Source the (args, kwargs) for the signature instance, ignoring excess
arguments.

```pycon
>>> def foo(w, /, x: float, y=2, *, z: int = 1):
...     return w + x * y ** z
>>> args, kwargs = Sig(foo).extract_args_and_kwargs(4, x=3, y=2)
>>> (args, kwargs) == ((4,), {"x": 3, "y": 2})
True
```

The difference with map_arguments_from_variadics is that here the output is
ready to be called by the function whose signature we have, since the
position-only arguments will be returned as args.

```pycon
>>> foo(*args, **kwargs)
10
```

Note that though `w` is a position only argument, you can specify `w=4` as a
keyword argument too (by default):

```pycon
>>> args, kwargs = Sig(foo).extract_args_and_kwargs(w=4, x=3, y=2)
>>> (args, kwargs) == ((4,), {"x": 3, "y": 2})
True
```

If you don’t want to allow that, you can say `_ignore_kind=False`

```pycon
>>> Sig(foo).extract_args_and_kwargs(w=4, x=3, y=2, _ignore_kind=False)
Traceback (most recent call last):
  ...
TypeError:...'w'...
```

You can use `_allow_partial` that will allow you, if
set to `True`, to underspecify the params of a function (in view of being
completed later).

```pycon
>>> Sig(foo).extract_args_and_kwargs(x=3, y=2)
Traceback (most recent call last):
  ...
TypeError:...'w'...
```

But if you specify `_allow_partial=True`…

```pycon
>>> args, kwargs = Sig(foo).extract_args_and_kwargs(
...     x=3, y=2, _allow_partial=True
... )
>>> (args, kwargs) == ((), {"x": 3, "y": 2})
True
```

By default, `_apply_defaults=False`, which will lead to only get those
arguments you input.

```pycon
>>> args, kwargs = Sig(foo).extract_args_and_kwargs(4, x=3, y=2)
>>> (args, kwargs) == ((4,), {"x": 3, "y": 2})
True
```

But if you specify `_apply_defaults=True` non-specified non-require arguments
will be returned with their defaults:

```pycon
>>> args, kwargs = Sig(foo).extract_args_and_kwargs(
...     4, x=3, y=2, _apply_defaults=True
... )
>>> (args, kwargs) == ((4,), {"x": 3, "y": 2, "z": 1})
True
```

#### extract_kwargs(\*args, \_apply_defaults=False, \_allow_partial=False, \_allow_excess=False, \_ignore_kind=False, \*\*kwargs)

Convenience method that calls map_arguments from variadics

```pycon
>>> def foo(w, /, x: float, y="YY", *, z: str = "ZZ"):
...     ...
>>> sig = Sig(foo)
>>> assert (
...     sig.map_arguments_from_variadics(1, 2, 3, z=4)
...     == sig.map_arguments_from_variadics(1, 2, y=3, z=4)
...     == {"w": 1, "x": 2, "y": 3, "z": 4}
... )
```

What about var positional and var keywords?

```pycon
>>> def bar(*args, **kwargs):
...     ...
...
>>> Sig(bar).map_arguments_from_variadics(1, 2, y=3, z=4)
{'args': (1, 2), 'kwargs': {'y': 3, 'z': 4}}
```

Note that though `w` is a position only argument, you can specify `w=11` as
a keyword argument too, using `_ignore_kind=True`:

```pycon
>>> Sig(foo).map_arguments_from_variadics(w=11, x=22, _ignore_kind=True)
{'w': 11, 'x': 22}
```

You can use `_allow_partial` that will allow you, if
set to `True`, to underspecify the params of a function
(in view of being completed later).

```pycon
>>> Sig(foo).map_arguments_from_variadics(x=3, y=2)
Traceback (most recent call last):
  ...
TypeError: missing a required argument: 'w'
```

But if you specify `_allow_partial=True`…

```pycon
>>> Sig(foo).map_arguments_from_variadics(x=3, y=2, _allow_partial=True)
{'x': 3, 'y': 2}
```

By default, `_apply_defaults=False`, which will lead to only get those arguments
you input.

```pycon
>>> Sig(foo).map_arguments_from_variadics(4, x=3, y=2)
{'w': 4, 'x': 3, 'y': 2}
```

But if you specify `_apply_defaults=True` non-specified non-require arguments
will be returned with their defaults:

```pycon
>>> Sig(foo).map_arguments_from_variadics(4, x=3, y=2, _apply_defaults=True)
{'w': 4, 'x': 3, 'y': 2, 'z': 'ZZ'}
```

#### get_names(spec, , conserve_sig_order=True, allow_excess=False)

Return a tuple of names corresponding to the given spec.

* **Parameters:**
  * **spec** – An integer, string, or iterable of intergers and strings
  * **conserve_sig_order** – Whether to order according to the signature
  * **allow_excess** – Whether to allow items in spec that are not in signature

```pycon
>>> sig = Sig('a b c d e')
>>> sig.get_names(0)
('a',)
>>> sig.get_names([0, 2])
('a', 'c')
>>> sig.get_names('b')
('b',)
>>> sig.get_names([0, 'c', -1])
('a', 'c', 'e')
```

See that by default the order of the signature is conserved:

```pycon
>>> sig.get_names('b e d')
('b', 'd', 'e')
```

But you can change that default to conserve the order of the `spec` instead:

```pycon
>>> sig.get_names('b e d', conserve_sig_order=False)
('b', 'e', 'd')
```

By default, you can’t mention names that are not in signature.
To allow this (making `spec` have “extract these” interpretation),
set `allow_excess=True`:

```pycon
>>> sig.get_names(['a', 'c', 'e', 'g', 'h'], allow_excess=True)
('a', 'c', 'e')
```

#### *property* has_var_keyword

Use index_of_var_keyword or var_keyword_name directly when needing that
information as well. This will avoid having to check the kinds list twice.

#### *property* has_var_kinds

Whether the signature has a VAR_POSITIONAL or a VAR_KEYWORD parameter.

```pycon
>>> Sig(lambda x, *, y: None).has_var_kinds
False
>>> Sig(lambda x, *y: None).has_var_kinds
True
>>> Sig(lambda x, **y: None).has_var_kinds
True
```

#### *property* has_var_positional

Use index_of_var_positional or var_keyword_name directly when needing that
information as well. This will avoid having to check the kinds list twice.

#### *property* index_of_var_keyword

The index of a VAR_KEYWORD param kind if any, and None if not.
See also, Sig.index_of_var_positional

```pycon
>>> assert Sig(lambda **kwargs: 0).index_of_var_keyword == 0
>>> assert Sig(lambda a, **kwargs: 0).index_of_var_keyword == 1
>>> assert Sig(lambda a, *args, **kwargs: 0).index_of_var_keyword == 2
```

And if there’s none…

```pycon
>>> assert Sig(lambda a, *args, b=1: 0).index_of_var_keyword is None
```

#### *property* index_of_var_positional

The index of the VAR_POSITIONAL param kind if any, and None if not.
See also, Sig.index_of_var_keyword

```pycon
>>> assert Sig(lambda x, *y, z: 0).index_of_var_positional == 1
>>> assert Sig(lambda x, /, y, **z: 0).index_of_var_positional == None
```

#### *property* inject_into_keyword_variadic

Decorator that uses signature to source the keyword variadic of target function.

See replace_kwargs_using function for more details, including examples.

```pycon
>>> def apple(a, x: int, y=2, *, z=3, **extra_apple_options):
...     return a + x + y + z
>>> @Sig(apple).inject_into_keyword_variadic
... def sauce(a, b, c, **sauce_kwargs):
...     return b * c + apple(a, **sauce_kwargs)
```

The function will works:

```pycon
>>> sauce(1, 2, 3, x=4, z=5)  # func still works? Should be: 1 + 4 + 2 + 5 + 2 * 3
18
```

But the signature now doesn’t have the `**sauce_kwargs`, but more informative
signature elements sourced from `apple`:

```pycon
>>> Sig(sauce)
<Sig (a, b, c, *, x: int, y=2, z=3, **extra_apple_options)>
```

#### is_call_compatible_with(other_sig, , param_comparator=None)

Return True if the signature is compatible with `other_sig`. Meaning that
all valid ways to call the signature are valid for `other_sig`.

#### kwargs_from_args_and_kwargs(args=None, kwargs=None, , apply_defaults=False, allow_partial=False, allow_excess=False, ignore_kind=False)

Map arguments (args and kwargs) to the parameters of function’s signature.

When you need to manage how the arguments of a function are specified,
you need to take care of
multiple cases depending on whether they were specified as positional arguments
(`args`) or keyword arguments (`kwargs`).

The `map_arguments` (and it’s sorta-inverse inverse,
`mk_args_and_kwargs`)
are there to help you manage this.

If you could rely on the the fact that only `kwargs` were given it would
reduce the complexity of your code.
This is why we have the `all_pk_signature` function in `signatures.py`.

We also need to have a means to make a `kwargs` only from the actual `(*args,
**kwargs)` used at runtime.
We have `Signature.bind` (and `bind_partial`) for that.

But these methods will fail if there is extra stuff in the `kwargs`.
Yet sometimes we’d like to have a `dict` that services several functions that
will extract their needs from it.

That’s where  `Sig.map_arguments_from_variadics(*args, **kwargs)` is needed.

* **Parameters:**
  * **args** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)) – The args the function will be called with.
  * **kwargs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The kwargs the function will be called with.
  * **apply_defaults** – (bool) Whether to apply signature defaults to the
    non-specified argument names
  * **allow_partial** – (bool) True iff you want to allow partial signature
    fulfillment.
  * **allow_excess** – (bool) Set to True iff you want to allow extra kwargs
    items to be ignored.
  * **ignore_kind** – (bool) Set to True iff you want to ignore the position and
    keyword only kinds,
    in order to be able to accept args and kwargs in such a way that there can
    be cross-over
    (args that are supposed to be keyword only, and kwargs that are supposed
    to be positional only)
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  An {param_name: arg_val, …} dict

See also the sorta-inverse of this function: mk_args_and_kwargs

```pycon
>>> def foo(w, /, x: float, y="YY", *, z: str = "ZZ"):
...     ...
>>> sig = Sig(foo)
>>> assert (
...     sig.map_arguments((11, 22, "you"), dict(z="zoo"))
...     == sig.map_arguments((11, 22), dict(y="you", z="zoo"))
...     == {"w": 11, "x": 22, "y": "you", "z": "zoo"}
... )
```

By default, `apply_defaults=False`, which will lead to only get those
arguments you input.

```pycon
>>> sig.map_arguments(args=(11,), kwargs={"x": 22})
{'w': 11, 'x': 22}
```

But if you specify `apply_defaults=True` non-specified non-require arguments
will be returned with their defaults:

```pycon
>>> sig.map_arguments(
...     args=(11,), kwargs={"x": 22}, apply_defaults=True
... )
{'w': 11, 'x': 22, 'y': 'YY', 'z': 'ZZ'}
```

By default, `ignore_excess=False`, so specifying kwargs that are not in the
signature will lead to an exception.

```pycon
>>> sig.map_arguments(
...     args=(11,), kwargs={"x": 22, "not_in_sig": -1}
... )
Traceback (most recent call last):
    ...
TypeError: got an unexpected keyword argument 'not_in_sig'
```

Specifying `allow_excess=True` will ignore such excess fields of kwargs.
This is useful when you want to source several functions from a same dict.

```pycon
>>> sig.map_arguments(
...     args=(11,), kwargs={"x": 22, "not_in_sig": -1}, allow_excess=True
... )
{'w': 11, 'x': 22}
```

On the other side of `ignore_excess` you have `allow_partial` that will allow
you, if
set to `True`, to underspecify the params of a function (in view of being
completed later).

```pycon
>>> sig.map_arguments(args=(), kwargs={"x": 22})
Traceback (most recent call last):
...
TypeError: missing a required argument: 'w'
```

But if you specify `allow_partial=True`…

```pycon
>>> sig.map_arguments(
...     args=(), kwargs={"x": 22}, allow_partial=True
... )
{'x': 22}
```

That’s a lot of control (eight combinations total), but not everything is
controllable here:
Position only and keyword only kinds need to be respected:

```pycon
>>> sig.map_arguments(args=(1, 2, 3, 4), kwargs={})
Traceback (most recent call last):
...
TypeError: too many positional arguments
>>> sig.map_arguments(args=(), kwargs=dict(w=1, x=2, y=3, z=4))
Traceback (most recent call last):
...
TypeError:...'w'...
```

But if you want to ignore the kind of parameter, just say so:

```pycon
>>> sig.map_arguments(
...     args=(1, 2, 3, 4), kwargs={}, ignore_kind=True
... )
{'w': 1, 'x': 2, 'y': 3, 'z': 4}
>>> sig.map_arguments(
...     args=(), kwargs=dict(w=1, x=2, y=3, z=4), ignore_kind=True
... )
{'w': 1, 'x': 2, 'y': 3, 'z': 4}
```

#### map_arguments(args=None, kwargs=None, , apply_defaults=False, allow_partial=False, allow_excess=False, ignore_kind=False)

Map arguments (args and kwargs) to the parameters of function’s signature.

When you need to manage how the arguments of a function are specified,
you need to take care of
multiple cases depending on whether they were specified as positional arguments
(`args`) or keyword arguments (`kwargs`).

The `map_arguments` (and it’s sorta-inverse inverse,
`mk_args_and_kwargs`)
are there to help you manage this.

If you could rely on the the fact that only `kwargs` were given it would
reduce the complexity of your code.
This is why we have the `all_pk_signature` function in `signatures.py`.

We also need to have a means to make a `kwargs` only from the actual `(*args,
**kwargs)` used at runtime.
We have `Signature.bind` (and `bind_partial`) for that.

But these methods will fail if there is extra stuff in the `kwargs`.
Yet sometimes we’d like to have a `dict` that services several functions that
will extract their needs from it.

That’s where  `Sig.map_arguments_from_variadics(*args, **kwargs)` is needed.

* **Parameters:**
  * **args** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)) – The args the function will be called with.
  * **kwargs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The kwargs the function will be called with.
  * **apply_defaults** – (bool) Whether to apply signature defaults to the
    non-specified argument names
  * **allow_partial** – (bool) True iff you want to allow partial signature
    fulfillment.
  * **allow_excess** – (bool) Set to True iff you want to allow extra kwargs
    items to be ignored.
  * **ignore_kind** – (bool) Set to True iff you want to ignore the position and
    keyword only kinds,
    in order to be able to accept args and kwargs in such a way that there can
    be cross-over
    (args that are supposed to be keyword only, and kwargs that are supposed
    to be positional only)
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  An {param_name: arg_val, …} dict

See also the sorta-inverse of this function: mk_args_and_kwargs

```pycon
>>> def foo(w, /, x: float, y="YY", *, z: str = "ZZ"):
...     ...
>>> sig = Sig(foo)
>>> assert (
...     sig.map_arguments((11, 22, "you"), dict(z="zoo"))
...     == sig.map_arguments((11, 22), dict(y="you", z="zoo"))
...     == {"w": 11, "x": 22, "y": "you", "z": "zoo"}
... )
```

By default, `apply_defaults=False`, which will lead to only get those
arguments you input.

```pycon
>>> sig.map_arguments(args=(11,), kwargs={"x": 22})
{'w': 11, 'x': 22}
```

But if you specify `apply_defaults=True` non-specified non-require arguments
will be returned with their defaults:

```pycon
>>> sig.map_arguments(
...     args=(11,), kwargs={"x": 22}, apply_defaults=True
... )
{'w': 11, 'x': 22, 'y': 'YY', 'z': 'ZZ'}
```

By default, `ignore_excess=False`, so specifying kwargs that are not in the
signature will lead to an exception.

```pycon
>>> sig.map_arguments(
...     args=(11,), kwargs={"x": 22, "not_in_sig": -1}
... )
Traceback (most recent call last):
    ...
TypeError: got an unexpected keyword argument 'not_in_sig'
```

Specifying `allow_excess=True` will ignore such excess fields of kwargs.
This is useful when you want to source several functions from a same dict.

```pycon
>>> sig.map_arguments(
...     args=(11,), kwargs={"x": 22, "not_in_sig": -1}, allow_excess=True
... )
{'w': 11, 'x': 22}
```

On the other side of `ignore_excess` you have `allow_partial` that will allow
you, if
set to `True`, to underspecify the params of a function (in view of being
completed later).

```pycon
>>> sig.map_arguments(args=(), kwargs={"x": 22})
Traceback (most recent call last):
...
TypeError: missing a required argument: 'w'
```

But if you specify `allow_partial=True`…

```pycon
>>> sig.map_arguments(
...     args=(), kwargs={"x": 22}, allow_partial=True
... )
{'x': 22}
```

That’s a lot of control (eight combinations total), but not everything is
controllable here:
Position only and keyword only kinds need to be respected:

```pycon
>>> sig.map_arguments(args=(1, 2, 3, 4), kwargs={})
Traceback (most recent call last):
...
TypeError: too many positional arguments
>>> sig.map_arguments(args=(), kwargs=dict(w=1, x=2, y=3, z=4))
Traceback (most recent call last):
...
TypeError:...'w'...
```

But if you want to ignore the kind of parameter, just say so:

```pycon
>>> sig.map_arguments(
...     args=(1, 2, 3, 4), kwargs={}, ignore_kind=True
... )
{'w': 1, 'x': 2, 'y': 3, 'z': 4}
>>> sig.map_arguments(
...     args=(), kwargs=dict(w=1, x=2, y=3, z=4), ignore_kind=True
... )
{'w': 1, 'x': 2, 'y': 3, 'z': 4}
```

#### map_arguments_from_variadics(\*args, \_apply_defaults=False, \_allow_partial=False, \_allow_excess=False, \_ignore_kind=False, \*\*kwargs)

Convenience method that calls map_arguments from variadics

```pycon
>>> def foo(w, /, x: float, y="YY", *, z: str = "ZZ"):
...     ...
>>> sig = Sig(foo)
>>> assert (
...     sig.map_arguments_from_variadics(1, 2, 3, z=4)
...     == sig.map_arguments_from_variadics(1, 2, y=3, z=4)
...     == {"w": 1, "x": 2, "y": 3, "z": 4}
... )
```

What about var positional and var keywords?

```pycon
>>> def bar(*args, **kwargs):
...     ...
...
>>> Sig(bar).map_arguments_from_variadics(1, 2, y=3, z=4)
{'args': (1, 2), 'kwargs': {'y': 3, 'z': 4}}
```

Note that though `w` is a position only argument, you can specify `w=11` as
a keyword argument too, using `_ignore_kind=True`:

```pycon
>>> Sig(foo).map_arguments_from_variadics(w=11, x=22, _ignore_kind=True)
{'w': 11, 'x': 22}
```

You can use `_allow_partial` that will allow you, if
set to `True`, to underspecify the params of a function
(in view of being completed later).

```pycon
>>> Sig(foo).map_arguments_from_variadics(x=3, y=2)
Traceback (most recent call last):
  ...
TypeError: missing a required argument: 'w'
```

But if you specify `_allow_partial=True`…

```pycon
>>> Sig(foo).map_arguments_from_variadics(x=3, y=2, _allow_partial=True)
{'x': 3, 'y': 2}
```

By default, `_apply_defaults=False`, which will lead to only get those arguments
you input.

```pycon
>>> Sig(foo).map_arguments_from_variadics(4, x=3, y=2)
{'w': 4, 'x': 3, 'y': 2}
```

But if you specify `_apply_defaults=True` non-specified non-require arguments
will be returned with their defaults:

```pycon
>>> Sig(foo).map_arguments_from_variadics(4, x=3, y=2, _apply_defaults=True)
{'w': 4, 'x': 3, 'y': 2, 'z': 'ZZ'}
```

#### merge_with_sig(sig, ch_to_all_pk=False, , default_conflict_method='strict')

Return a signature obtained by merging self signature with another signature.
Insofar as it can, given the kind precedence rules, the arguments of self will
appear first.

* **Parameters:**
  * **sig** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)], [`Signature`](https://docs.python.org/3/library/inspect.html#inspect.Signature), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)], [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The signature to merge with.
  * **ch_to_all_pk** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to change all kinds of both signatures to PK (
    POSITIONAL_OR_KEYWORD)
* **Returns:**

```pycon
>>> def func(a=None, *, b=1, c=2):
...     ...
...
>>>
>>> s = Sig(func)
>>> s
<Sig (a=None, *, b=1, c=2)>
```

Observe where the new arguments `d` and `e` are placed,
according to whether they have defaults and what their kind is:

```pycon
>>> s.merge_with_sig(["d", "e"])
<Sig (d, e, a=None, *, b=1, c=2)>
>>> s.merge_with_sig(["d", ("e", 4)])
<Sig (d, a=None, e=4, *, b=1, c=2)>
>>> s.merge_with_sig(["d", dict(name="e", kind=KO, default=4)])
<Sig (d, a=None, *, b=1, c=2, e=4)>
>>> s.merge_with_sig(
...     [dict(name="d", kind=KO), dict(name="e", kind=KO, default=4)]
... )
<Sig (a=None, *, d, b=1, c=2, e=4)>
```

If the kind of the params is not important, but order is, you can specify
`ch_to_all_pk=True`:

```pycon
>>> s.merge_with_sig(["d", "e"], ch_to_all_pk=True)
<Sig (d, e, a=None, b=1, c=2)>
>>> s.merge_with_sig([("d", 3), ("e", 4)], ch_to_all_pk=True)
<Sig (a=None, b=1, c=2, d=3, e=4)>
```

#### mk_args_and_kwargs(arguments, , apply_defaults=False, allow_partial=False, allow_excess=False, ignore_kind=False, args_limit=0)

Extract args and kwargs such that `func(*args, **kwargs)` can be called,
where func has instance’s signature.

* **Parameters:**
  * **arguments** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The {param_name: arg_val,…} dict to process
  * **args_limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – 

    How “far” in the params should args (positional arguments)
    be searched for.
    - args_limit==0: Take the minimum number possible of args (positional
      arguments). Only those that are position only or before a var-positional.
    - args_limit is None: Take the maximum number of args (positional arguments).
      The only kwargs (keyword arguments) you should have are keyword-only
      and var-keyword arguments.
    - args_limit positive integer: Take the args_limit first argument names
      (of signature) as args, and the rest as kwargs.
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

```pycon
>>> def foo(w, /, x: float, y=1, *, z: int = 1):
...     return ((w + x) * y) ** z
>>> foo_sig = Sig(foo)
>>> args, kwargs = foo_sig.mk_args_and_kwargs(
...     dict(w=4, x=3, y=2, z=1)
... )
>>> assert (args, kwargs) == ((4,), {"x": 3, "y": 2, "z": 1})
>>> assert foo(*args, **kwargs) == foo(4, 3, 2, z=1) == 14
```

What about variadics?

```pycon
>>> def bar(a, /, b, *args, c=2, **kwargs):
...     pass
>>> Sig(bar).mk_args_and_kwargs(
...     dict(a=1, b=2, args=(3,4), c=5, kwargs=dict(d=6, e=7))
... )
((1, 2, 3, 4), {'c': 5, 'd': 6, 'e': 7})
```

You can also give the arguments in a different order:

```pycon
>>> Sig(bar).mk_args_and_kwargs(
...     dict(args=(3,4), kwargs=dict(d=6, e=7), b=2, c=5, a=1)
... )
((1, 2, 3, 4), {'c': 5, 'd': 6, 'e': 7})
```

The `args_limit` begs explanation.
Consider the signature of `def foo(w, /, x: float, y=1, *, z: int = 1): ...`
for instance. We could call the function with the following (args, kwargs) pairs:

- ((1,), {‘x’: 2, ‘y’: 3, ‘z’: 4})
- ((1, 2), {‘y’: 3, ‘z’: 4})
- ((1, 2, 3), {‘z’: 4})
  The two other combinations (empty args or empty kwargs) are not valid
  because of the / and \* constraints.

But when asked for an (args, kwargs) pair, which of the three valid options
should be returned? This is what the `args_limit` argument controls.

If `args_limit == 0`, the least args (positional arguments) will be returned.
It’s the default.

```pycon
>>> arguments = dict(w=4, x=3, y=2, z=1)
>>> foo_sig.mk_args_and_kwargs(arguments, args_limit=0)
((4,), {'x': 3, 'y': 2, 'z': 1})
```

If `args_limit is None`, the least kwargs (keyword arguments) will be returned.

```pycon
>>> foo_sig.mk_args_and_kwargs(arguments, args_limit=None)
((4, 3, 2), {'z': 1})
```

If `args_limit` is a positive integer, the first `[args_limit]` arguments
will be returned (not checking at all if this is valid!).

```pycon
>>> foo_sig.mk_args_and_kwargs(arguments, args_limit=1)
((4,), {'x': 3, 'y': 2, 'z': 1})
>>> foo_sig.mk_args_and_kwargs(arguments, args_limit=2)
((4, 3), {'y': 2, 'z': 1})
>>> foo_sig.mk_args_and_kwargs(arguments, args_limit=3)
((4, 3, 2), {'z': 1})
```

Note that if you specify `args_limit` to be greater than the maximum of
positional arguments, it behaves as if `args_limit` was `None`:

```pycon
>>> foo_sig.mk_args_and_kwargs(arguments, args_limit=4)
((4, 3, 2), {'z': 1})
```

Note that ‘args_limit’’s behavior is consistent with list behvior in the sense
that:

```pycon
>>> args = (0, 1, 2, 3)
>>> args[:0]
()
>>> args[:None]
(0, 1, 2, 3)
>>> args[2]
2
```

If variable positional arguments are present, `args_limit` is ignored and
all positional arguments are returned as args.

```pycon
>>> Sig(bar).mk_args_and_kwargs(
...     dict(a=1, b=2, args=(3,4), c=5, kwargs=dict(d=6, e=7)),
...     args_limit=1
... )
((1, 2, 3, 4), {'c': 5, 'd': 6, 'e': 7})
```

By default, only the arguments that were given in the `arguments` input will be
returned in the (args, kwargs) output.
If you also want to get those that have defaults (according to signature),
you need to specify it with the `apply_defaults=True` argument.

```pycon
>>> foo_sig.mk_args_and_kwargs(dict(w=4, x=3))
((4,), {'x': 3})
>>> foo_sig.mk_args_and_kwargs(dict(w=4, x=3), apply_defaults=True)
((4,), {'x': 3, 'y': 1, 'z': 1})
```

By default, all required arguments must be given.
Not doing so will lead to a `TypeError`.
If you want to process your arguments anyway, specify `allow_partial=True`.

```pycon
>>> foo_sig.mk_args_and_kwargs(dict(w=4))
Traceback (most recent call last):
  ...
TypeError: missing a required argument: 'x'
>>> foo_sig.mk_args_and_kwargs(dict(w=4), allow_partial=True)
((4,), {})
```

Specifying argument names that are not recognized by the signature will
lead to a `TypeError`.
If you want to avoid this (and just take from the input `kwargs` what ever you
can), specify this with `allow_excess=True`.

```pycon
>>> foo_sig.mk_args_and_kwargs(dict(w=4, x=3, extra='stuff'))
Traceback (most recent call last):
    ...
TypeError: Got unexpected keyword arguments: extra
>>> foo_sig.mk_args_and_kwargs(dict(w=4, x=3, extra='stuff'),
...     allow_excess=True)
((4,), {'x': 3})
```

See `map_arguments` (namely for the description of the arguments).

#### modified(\_allow_reordering=False, \*\*changes_for_name)

Returns a modified (new) signature object.

#### NOTE
This function doesn’t modify the signature, but creates a modified copy
of the signature.

IMPORTANT WARNING: This is an advanced feature. Avoid wrapping a function with
a modified signature, as this may not have the intended effect.

```pycon
>>> def foo(pka, *vpa, koa, **vka): ...
>>> sig = Sig(foo)
>>> sig
<Sig (pka, *vpa, koa, **vka)>
>>> assert sig.kinds['pka'] == PK
```

Let’s make a signature that is the same as sig, except that

> - `poa` is given a PO (POSITIONAL_ONLY) kind insteadk of PK
> - `koa` is given a default of None
> - the signature is given a return_annotation of str
```pycon
>>> new_sig = sig.modified(
...     pka={'kind': PO},
...     koa={'default': None},
...     return_annotation=str
... )
>>> new_sig
<Sig (pka, /, *vpa, koa=None, **vka) -> str>
>>> assert new_sig.kinds['pka'] == PO  # now pos is of the PO kind!
```

Here’s an example of changing signature parameters in bulk.
Here we change all kinds to be the friendly PK kind.

```pycon
>>> sig.modified(**{name: {'kind': PK} for name in sig.names})
<Sig (pka, vpa, koa, vka)>
```

Repetition of the above: This gives you a signature with all PK kinds.
If you wrap a function with it, it will look like it has all PK kinds.
But that doesn’t mean you can actually use thenm as such.
You’ll need to modify (decorate further) your function further to reflect
its new signature.

On the other hand, if you decorate a function with a sig that adds or modifies
defaults, these defaults will actually be used (unlike with `functools.wraps`).

#### *property* n_required

The number of required arguments.
A required argument is one that doesn’t have a default, nor is VAR_POSITIONAL
(`*args`) or VAR_KEYWORD (`**kwargs`).

#### NOTE
Sometimes a minimum number of arguments in VAR_POSITIONAL and
VAR_KEYWORD are in fact required,
but we can’t see this from the signature, so we can’t tell you about that! You
do the math.

```pycon
>>> f = lambda a00, /, a11, a12, *a23, a34, a35=1, a36='two', **a47: None
>>> Sig(f).n_required
4
```

#### names_for_kind(kind)

Get the arg names tuple for a given kind.
Note, if you need to do this several times, or for several kinds, use
`names_of_kind` property (a tuple) instead: It groups all names of kinds once,
and caches the result.

#### pair_with(other_sig)

Get an object that pairs with another signature for comparison, merging, etc.

See `SigPair` for more details.

* **Return type:**
  [`SigPair`](dol.sig_compare.html.md#dol.sig_compare.SigPair)

#### *property* params

Just list(self.parameters.values()), because that’s often what we want.
Why a Sig.params property when we already have a Sig.parameters property?

Well, as much as is boggles my mind, it so happens that the Signature.parameters
is a name->Parameter mapping, but the Signature argument `parameters`,
though baring the same name,
is expected to be a list of Parameter instances.

So Sig.params is there to restore semantic consistence sanity.

#### replace_kwargs_using()

Decorator that replaces the variadic keyword argument of the target function using
the `sig`, the signature of a source function.
It essentially injects the difference between `sig` and the target function’s
signature into the target function’s signature. That is, it replaces the
variadic keyword argument (a.k.a. “kwargs”) with those parameters that are in `sig`
but not in the target function’s signature.

This is meant to be used when a `targ_func` (the function you’ll apply the
decorator to) has a variadict keyword argument that is just used to forward “extra”
arguments to another function, and you want to make sure that the signature of the
`targ_func` is consistent with the `sig` signature.
(Also, you don’t want to copy the signatures around manually.)

In the following, `sauce` (the target function) has a variadic keyword argument,
`sauce_kwargs`, that is used to forward extra arguments to `apple` (the source
function).

```pycon
>>> def apple(a, x: int, y=2, *, z=3, **extra_apple_options):
...     return a + x + y + z
>>> @replace_kwargs_using(apple)
... def sauce(a, b, c, **sauce_kwargs):
...     return b * c + apple(a, **sauce_kwargs)
```

The function will works:

```pycon
>>> sauce(1, 2, 3, x=4, z=5)  # func still works? Should be: 1 + 4 + 2 + 5 + 2 * 3
18
```

But the signature now doesn’t have the `**sauce_kwargs`, but more informative
signature elements sourced from `apple`:

```pycon
>>> Sig(sauce)
<Sig (a, b, c, *, x: int, y=2, z=3, **extra_apple_options)>
```

One thing to note is that the order of the arguments in the signature of `apple`
may change to accomodate for the python parameter order rules
(see [https://docs.python.org/3/reference/compound_stmts.html#function-definitions](https://docs.python.org/3/reference/compound_stmts.html#function-definitions)).
The new order will try to conserve the order of the original arguments of `sauce`
in-so-far as it doesn’t violate the python parameter order rules, though.
See examples below:

```pycon
>>> @Sig.replace_kwargs_using(apple)
... def sauce(a, b=2, c=3, **sauce_kwargs):
...     return b * c + apple(a, **sauce_kwargs)
>>> Sig(sauce)
<Sig (a, b=2, c=3, *, x: int, y=2, z=3, **extra_apple_options)>
```

```pycon
>>> @Sig.replace_kwargs_using(apple)
... def sauce(a=1, b=2, c=3, **sauce_kwargs):
...     return b * c + apple(a, **sauce_kwargs)
>>> Sig(sauce)
<Sig (a=1, b=2, c=3, *, x: int, y=2, z=3, **extra_apple_options)>
```

#### *property* required_names

A tuple of required names, preserving the original signature order.

A required name is that must be given in a function call, that is, the name of a
paramater that doesn’t have a default, and is not a variadic.

That lost one is a frequent gotcha, so oo not fall in that gotcha that easily,
we provide a property that contains what we need.

```pycon
>>> f = lambda a00, /, a11, a12, *a23, a34, a35=1, a36='two', **a47: None
>>> Sig(f).required_names
('a00', 'a11', 'a12', 'a34')
```

#### *classmethod* sig_or_default(obj, default_signature=<Signature (\*no_sig_args, \*\*no_sig_kwargs)>)

Returns a Sig instance, or a default signature if there was a ValueError
trying to construct it.

For example, `time.time` doesn’t have a signature

```pycon
>>> import time
>>> has_signature(time.time)
False
```

But we can tell `Sig` to give it the default one:

```pycon
>>> str(Sig.sig_or_default(time.time))
'(*no_sig_args, **no_sig_kwargs)'
```

That’s the default signature, which should work for most purposes.
You can also specify what the default should be though.

```pycon
>>> fake_signature = Sig(lambda *time_takes_no_arguments: ...)
>>> str(Sig.sig_or_default(time.time, fake_signature))
'(*time_takes_no_arguments)'
```

Careful though. If you assign a signature to a function that is not aligned
with that actually functioning of the function, bad things will happen.
In this case, the actual signature of time is the empty signature:

```pycon
>>> str(Sig.sig_or_default(time.time, Sig(lambda: ...)))
'()'
```

#### *classmethod* sig_or_none(obj)

Returns a Sig instance, or None if there was a ValueError trying to
construct it.
One use case is to be able to tell if an object has a signature or not.

```pycon
>>> robust_has_signature = lambda obj: bool(Sig.sig_or_none(obj))
>>> robust_has_signature(robust_has_signature)  # an easy case
True
>>> robust_has_signature(
...     Sig
... )  # another easy one: This time, a type/class (which is callable, yes)
True
```

But here’s where it get’s interesting. `print`, a builtin, doesn’t have a
signature through inspect.signature.

```pycon
>>> has_signature(print)
False
```

But we do get one with robust_has_signature

```pycon
>>> robust_has_signature(print)
True
```

#### sort_params()

Returns a signature with the parameters sorted by kind and default presence.

#### source_args_and_kwargs(\*args, \_ignore_kind=True, \_allow_partial=False, \_apply_defaults=False, \_args_limit=0, \*\*kwargs)

Source the (args, kwargs) for the signature instance, ignoring excess
arguments.

```pycon
>>> def foo(w, /, x: float, y=2, *, z: int = 1):
...     return w + x * y ** z
>>> args, kwargs = Sig(foo).source_args_and_kwargs(
...     4, x=3, y=2, extra="keywords", are="ignored"
... )
>>> args, kwargs
((4,), {'x': 3, 'y': 2})
```

The difference with source_arguments is that here the output is ready to be
called by the
function whose signature we have, since the position-only arguments will be
returned as
args.

```pycon
>>> foo(*args, **kwargs)
10
```

Note that though `w` is a position only argument, you can specify `w=4` as a
keyword argument too (by default):

```pycon
>>> args, kwargs = Sig(foo).source_args_and_kwargs(
...     w=4, x=3, y=2, extra="keywords", are="ignored"
... )
>>> assert (args, kwargs) == ((4,), {"x": 3, "y": 2})
```

If you don’t want to allow that, you can say `_ignore_kind=False`

```pycon
>>> Sig(foo).source_args_and_kwargs(
...     w=4, x=3, y=2, extra="keywords", are="ignored", _ignore_kind=False
... )
Traceback (most recent call last):
  ...
TypeError: ...'w'...
```

You can use `_allow_partial` that will allow you, if
set to `True`, to underspecify the params of a function (in view of being
completed later).

```pycon
>>> Sig(foo).source_args_and_kwargs(x=3, y=2, extra="keywords", are="ignored")
Traceback (most recent call last):
  ...
TypeError:...'w'...
```

But if you specify `_allow_partial=True`…

```pycon
>>> args, kwargs = Sig(foo).source_args_and_kwargs(
...     x=3, y=2, extra="keywords", are="ignored", _allow_partial=True
... )
>>> (args, kwargs) == ((), {"x": 3, "y": 2})
True
```

By default, `_apply_defaults=False`, which will lead to only get those
arguments you input.

```pycon
>>> args, kwargs = Sig(foo).source_args_and_kwargs(
...     4, x=3, y=2, extra="keywords", are="ignored"
... )
>>> (args, kwargs) == ((4,), {"x": 3, "y": 2})
True
```

But if you specify `_apply_defaults=True` non-specified non-require arguments
will be returned with their defaults:

```pycon
>>> args, kwargs = Sig(foo).source_args_and_kwargs(
...     4, x=3, y=2, extra="keywords", are="ignored", _apply_defaults=True
... )
>>> (args, kwargs) == ((4,), {"x": 3, "y": 2, "z": 1})
True
```

#### source_arguments(\*args, \_apply_defaults=False, \_allow_partial=False, \_ignore_kind=True, \*\*kwargs)

Source the arguments for the signature instance, ignoring excess arguments.

```pycon
>>> def foo(w, /, x: float, y="YY", *, z: str = "ZZ"):
...     ...
>>> Sig(foo).source_arguments(11, x=22, extra="keywords", are="ignored")
{'w': 11, 'x': 22}
```

Note that though `w` is a position only argument, you can specify `w=11` as a
keyword argument too (by default):

```pycon
>>> Sig(foo).source_arguments(w=11, x=22, extra="keywords", are="ignored")
{'w': 11, 'x': 22}
```

If you don’t want to allow that, you can say `_ignore_kind=False`

```pycon
>>> Sig(foo).source_arguments(
...     w=11, x=22, extra="keywords", are="ignored", _ignore_kind=False
... )
Traceback (most recent call last):
  ...
TypeError: ...'w'...
```

You can use `_allow_partial` that will allow you, if
set to `True`, to underspecify the params of a function (in view of being
completed later).

```pycon
>>> Sig(foo).source_arguments(x=3, y=2, extra="keywords", are="ignored")
Traceback (most recent call last):
  ...
TypeError: ...'w'...
```

But if you specify `_allow_partial=True`…

```pycon
>>> Sig(foo).source_arguments(
...     x=3, y=2, extra="keywords", are="ignored", _allow_partial=True
... )
{'x': 3, 'y': 2}
```

By default, `_apply_defaults=False`, which will lead to only get those
arguments you input.

```pycon
>>> Sig(foo).source_arguments(4, x=3, y=2, extra="keywords", are="ignored")
{'w': 4, 'x': 3, 'y': 2}
```

But if you specify `_apply_defaults=True` non-specified non-require arguments
will be returned with their defaults:

```pycon
>>> Sig(foo).source_arguments(
...     4, x=3, y=2, extra="keywords", are="ignored", _apply_defaults=True
... )
{'w': 4, 'x': 3, 'y': 2, 'z': 'ZZ'}
```

#### source_kwargs(\*args, \_apply_defaults=False, \_allow_partial=False, \_ignore_kind=True, \*\*kwargs)

Source the arguments for the signature instance, ignoring excess arguments.

```pycon
>>> def foo(w, /, x: float, y="YY", *, z: str = "ZZ"):
...     ...
>>> Sig(foo).source_arguments(11, x=22, extra="keywords", are="ignored")
{'w': 11, 'x': 22}
```

Note that though `w` is a position only argument, you can specify `w=11` as a
keyword argument too (by default):

```pycon
>>> Sig(foo).source_arguments(w=11, x=22, extra="keywords", are="ignored")
{'w': 11, 'x': 22}
```

If you don’t want to allow that, you can say `_ignore_kind=False`

```pycon
>>> Sig(foo).source_arguments(
...     w=11, x=22, extra="keywords", are="ignored", _ignore_kind=False
... )
Traceback (most recent call last):
  ...
TypeError: ...'w'...
```

You can use `_allow_partial` that will allow you, if
set to `True`, to underspecify the params of a function (in view of being
completed later).

```pycon
>>> Sig(foo).source_arguments(x=3, y=2, extra="keywords", are="ignored")
Traceback (most recent call last):
  ...
TypeError: ...'w'...
```

But if you specify `_allow_partial=True`…

```pycon
>>> Sig(foo).source_arguments(
...     x=3, y=2, extra="keywords", are="ignored", _allow_partial=True
... )
{'x': 3, 'y': 2}
```

By default, `_apply_defaults=False`, which will lead to only get those
arguments you input.

```pycon
>>> Sig(foo).source_arguments(4, x=3, y=2, extra="keywords", are="ignored")
{'w': 4, 'x': 3, 'y': 2}
```

But if you specify `_apply_defaults=True` non-specified non-require arguments
will be returned with their defaults:

```pycon
>>> Sig(foo).source_arguments(
...     4, x=3, y=2, extra="keywords", are="ignored", _apply_defaults=True
... )
{'w': 4, 'x': 3, 'y': 2, 'z': 'ZZ'}
```

#### to_signature_kwargs()

The dict of keyword arguments to make this signature instance.

```pycon
>>> def f(w, /, x: float = 2, y=1, *, z: int = 0) -> float:
...     ...
>>> Sig(f).to_signature_kwargs()
{'parameters':
    [<Parameter "w">,
    <Parameter "x: float = 2">,
    <Parameter "y=1">,
    <Parameter "z: int = 0">],
'return_annotation': <class 'float'>}
```

Note that this does NOT return:

```python
{'parameters': self.parameters,
'return_annotation': self.return_annotation}
```

which would not actually work as keyword arguments of `Signature`.
Yeah, I know. Don’t ask me, ask the authors of `Signature`!

Instead, `parammeters` will be `list(self.parameters.values())`, which does
work.

#### to_simple_signature()

A builtin `inspect.Signature` instance equivalent (i.e. without the extra
properties and methods)

```pycon
>>> def f(w, /, x: float = 2, y=1, *, z: int = 0):
...     ...
>>> Sig(f).to_simple_signature()
<Signature (w, /, x: float = 2, y=1, *, z: int = 0)>
```

#### *property* with_defaults

Sub-signature containing only “not required” (i.e. with defaults) parameters.

```pycon
>>> list(Sig(lambda *args, a, b, x=1, y=1, **kwargs: ...).with_defaults)
['args', 'x', 'y', 'kwargs']
```

#### *property* without_defaults

Sub-signature containing only “required” (i.e. without defaults) parameters.

```pycon
>>> list(Sig(lambda *args, a, b, x=1, y=1, **kwargs: ...).without_defaults)
['a', 'b']
```

#### wrap(func, ignore_incompatible_signatures=True, , copy_function=False)

Gives the input function the signature.

This is similar to the `functools.wraps` function, but parametrized by a
signature
(not a callable). Also, where as both write to the input func’s `__signature__`
attribute, here we also write to

- `__defaults__` and `__kwdefaults__`, extracting these from `__signature__`
  (functools.wraps doesn’t do that at the time of writing this
  (see [https://github.com/python/cpython/pull/21379](https://github.com/python/cpython/pull/21379))).
- `__annotations__` (also extracted from `__signature__`)
- does not write to `__module__`, `__name__`, `__qualname__`, `__doc__`
  (because again, we’re basinig the injecton on a signature, not a function,
  so we have no name, doc, etc…)

#### WARNING
The fact that you’ve modified the signature of your function doesn’t
mean that the decorated function will work as expected (or even work at all).
See below for examples.

```pycon
>>> def f(w, /, x: float = 1, y=2, z: int = 3):
...     return w + x * y ** z
>>> f(0, 1)  # 0 + 1 * 2 ** 3
8
>>> f.__defaults__
(1, 2, 3)
>>> assert 8 == f(0) == f(0, 1) == f(0, 1, 2) == f(0, 1, 2, 3)
```

Now let’s create a very similar function to f, but where:

- w is not position-only
- x annot is int instead of float, and doesn’t have a default
- z’s default changes to 10

```pycon
>>> def g(w, x: int, y=2, z: int = 10):
...     return w + x * y ** z
>>> s = Sig(g)
>>> f = s.wrap(f)
>>> import inspect
>>> inspect.signature(f)  # see that
<Sig (w, x: int, y=2, z: int = 10)>
>>> # But (unlike with functools.wraps) here we get __defaults__ and
__kwdefault__
>>> f.__defaults__  # see that x has no more default & z's default is now 10
(2, 10)
>>> f(
...     0, 1
... )  # see that now we get a different output because using different defaults
1024
```

Remember that you are modifying the signature, not the function itself.
Signature changes in defaults will indeed change the function’s behavior.
But changes in name or kind will only be reflected in the signature, and
misalignment with the wrapped function will lead to unexpected results.

```pycon
>>> def f(w, /, x: float = 1, y=2, *, z: int = 3):
...     return w + x * y ** z
>>> f(0)  # 0 + 1 * 2 ** 3
8
>>> f(0, 1, 2, 3)  # error expected!
Traceback (most recent call last):
  ...
TypeError: f() takes from 1 to 3 positional arguments but 4 were given
```

But if you try to remove the argument kind constraint by just changing the
signature, you’ll fail.

```pycon
>>> def g(w, x: float = 1, y=2, z: int = 3):
...     return w + x * y ** z
>>> f = Sig(g).wrap(f)
>>> f(0)
Traceback (most recent call last):
  ...
TypeError: f() missing 1 required keyword-only argument: 'z'
>>> f(0, 1, 2, 3)
Traceback (most recent call last):
  ...
TypeError: f() takes from 0 to 3 positional arguments but 4 were given
```

### dol.signatures.assure_params(obj=None)

Get an interable of Parameter instances from an object.

* **Parameters:**
  **obj** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)], [`Signature`](https://docs.python.org/3/library/inspect.html#inspect.Signature), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)], [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)])
* **Returns:**

From a callable:

```pycon
>>> def f(w, /, x: float = 1, y=1, *, z: int = 1):
...     ...
>>> ensure_params(f)
[<Parameter "w">, <Parameter "x: float = 1">, <Parameter "y=1">, <Parameter "z: int = 1">]
```

From an iterable of strings, dicts, or tuples

```pycon
>>> ensure_params(
...     [
...         "xyz",
...         (
...             "b",
...             Parameter.empty,
...             int,
...         ),  # if you want an annotation without a default use Parameter.empty
...         (
...             "c",
...             2,
...         ),  # if you just want a default, make it the second element of your tup
...         dict(name="d", kind=Parameter.VAR_KEYWORD),
...     ]
... )  # all kinds are by default PK: Use dict to specify otherwise.
[<Param "xyz">, <Param "b: int">, <Param "c=2">, <Param "**d">]
```

If no input is given, an empty list is returned.

```pycon
>>> ensure_params()  # equivalent to ensure_params(None)
[]
```

### dol.signatures.ensure_params(obj=None)

Get an interable of Parameter instances from an object.

* **Parameters:**
  **obj** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)], [`Signature`](https://docs.python.org/3/library/inspect.html#inspect.Signature), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)], [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)])
* **Returns:**

From a callable:

```pycon
>>> def f(w, /, x: float = 1, y=1, *, z: int = 1):
...     ...
>>> ensure_params(f)
[<Parameter "w">, <Parameter "x: float = 1">, <Parameter "y=1">, <Parameter "z: int = 1">]
```

From an iterable of strings, dicts, or tuples

```pycon
>>> ensure_params(
...     [
...         "xyz",
...         (
...             "b",
...             Parameter.empty,
...             int,
...         ),  # if you want an annotation without a default use Parameter.empty
...         (
...             "c",
...             2,
...         ),  # if you just want a default, make it the second element of your tup
...         dict(name="d", kind=Parameter.VAR_KEYWORD),
...     ]
... )  # all kinds are by default PK: Use dict to specify otherwise.
[<Param "xyz">, <Param "b: int">, <Param "c=2">, <Param "**d">]
```

If no input is given, an empty list is returned.

```pycon
>>> ensure_params()  # equivalent to ensure_params(None)
[]
```

### dol.signatures.extract_arguments(params, , what_to_do_with_remainding='return', include_all_when_var_keywords_in_params=False, assert_no_missing_position_only_args=False, \*\*kwargs)

Extract arguments needed to satisfy the params of a callable, dealing with the
dirty details.

Returns an (param_args, param_kwargs, remaining_kwargs) tuple where

- param_args are the values of kwargs that are PO (POSITION_ONLY) as defined by
  params,
- param_kwargs are those names that are both in params and not in param_args, and
- remaining_kwargs are the remaining.

Intended usage: When you need to call a function `func` that has some
position-only arguments,
but you have a kwargs dict of arguments in your hand. You can’t just to `func(
**kwargs)`.
But you can (now) do

```python
# extract from kwargs what you need for func
args, kwargs, remaining = extract_arguments(kwargs, func)
# ... check if remaining is empty (or not, depending on your paranoia),
# and then call the func:
func(*args, **kwargs)
```

(And if you doing that a lot: Do put it in a decorator!)

#### SEE ALSO
extract_arguments.without_remainding

The most frequent case you’ll encounter is when there’s no POSITION_ONLY args,
your param_args will be empty
and you param_kwargs will contain all the arguments that match params,
in the order of these params.

```pycon
>>> from inspect import signature
>>> def f(a, b, c=None, d=0):
...     ...
...
>>> extract_arguments(f, b=2, a=1, c=3, d=4, extra="stuff")
((), {'a': 1, 'b': 2, 'c': 3, 'd': 4}, {'extra': 'stuff'})
```

But sometimes you do have POSITION_ONLY arguments.
What extract_arguments will do for you is return the value of these as the first
element of
the triple.

```pycon
>>> def f(a, b, c=None, /, d=0):
...     ...
...
>>> extract_arguments(f, b=2, a=1, c=3, d=4, extra="stuff")
((1, 2, 3), {'d': 4}, {'extra': 'stuff'})
```

Note above how we get `(1, 2, 3)`, the order defined by the func’s signature,
instead of `(2, 1, 3)`, the order defined by the kwargs.
So it’s the params (e.g. function signature) that determine the order, not kwargs.
When using to call a function, this is especially crucial if we use POSITION_ONLY
arguments.

See also that the third output, the remaining_kwargs, as `{'extra': 'stuff'}` since
it was not in the params of the function.
Even if you include a VAR_KEYWORD kind of argument in the function, it won’t change
this behavior.

```pycon
>>> def f(a, b, c=None, /, d=0, **kws):
...     ...
...
>>> extract_arguments(f, b=2, a=1, c=3, d=4, extra="stuff")
((1, 2, 3), {'d': 4}, {'extra': 'stuff'})
```

This is because we don’t want to assume that all the kwargs can actually be
included in a call to the function behind the params.
Instead, the user can chose whether to include the remainder by doing a:

```python
param_kwargs.update(remaining_kwargs)
```

et voilà.

That said, we do understand that it may be a common pattern, so we’ll do that
extra step for you
if you specify `include_all_when_var_keywords_in_params=True`.

```pycon
>>> def f(a, b, c=None, /, d=0, **kws):
...     ...
...
>>> extract_arguments(
...     f,
...     b=2,
...     a=1,
...     c=3,
...     d=4,
...     extra="stuff",
...     include_all_when_var_keywords_in_params=True,
... )
((1, 2, 3), {'d': 4, 'extra': 'stuff'}, {})
```

If you’re expecting no remainder you might want to just get the args and kwargs (
not this third
expected-to-be-empty remainder). You have two ways to do that, specifying:

- `what_to_do_with_remainding='ignore'`, which will just return the (args,
  kwargs) pair
- `what_to_do_with_remainding='assert_empty'`, which will do the same, but first
  assert the remainder is empty

We suggest to use `functools.partial` to configure the `argument_argument` you need.

```pycon
>>> from functools import partial
>>> arg_extractor = partial(
...     extract_arguments,
...     what_to_do_with_remainding="assert_empty",
...     include_all_when_var_keywords_in_params=True,
... )
>>> def f(a, b, c=None, /, d=0, **kws):
...     ...
...
>>> arg_extractor(f, b=2, a=1, c=3, d=4, extra="stuff")
((1, 2, 3), {'d': 4, 'extra': 'stuff'})
```

And what happens if the kwargs doesn’t contain all the POSITION_ONLY arguments?

```pycon
>>> def f(a, b, c=None, /, d=0):
...     ...
...
>>> extract_arguments(f, b=2, d="is a kw arg", e="is not an arg at all")
((MissingArgValFor("a"), 2, MissingArgValFor("c")), {'d': 'is a kw arg'}, {'e': 'is not an arg at all'})
```

A few more examples…

Let’s call `extract_arguments` with params being not a function,
but, a Signature instance, a mapping whose values are Parameter instances,
or an iterable of Parameter instances…

```pycon
>>> def func(a, b, /, c=None, *, d=0, **kws):
...     ...
...
>>> sig = Signature.from_callable(func)
>>> param_map = sig.parameters
>>> param_iterable = param_map.values()
>>> kwargs = dict(b=2, a=1, c=3, d=4, extra="stuff")
>>> assert extract_arguments(sig, **kwargs) == extract_arguments(func, **kwargs)
>>> assert extract_arguments(param_map, **kwargs) == extract_arguments(
...     func, **kwargs
... )
>>> assert extract_arguments(param_iterable, **kwargs) == extract_arguments(
...     func, **kwargs
... )
```

Edge case:
No params specified? No problem. You’ll just get empty args and kwargs. Everything
in the remainder

```pycon
>>> extract_arguments(params=(), b=2, a=1, c=3, d=0)
((), {}, {'b': 2, 'a': 1, 'c': 3, 'd': 0})
```

* **Parameters:**
  * **params** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)], [`Signature`](https://docs.python.org/3/library/inspect.html#inspect.Signature), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)], [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Specifies what PO arguments should be extracted.
    Could be a callable, Signature, iterable of Parameters…
  * **what_to_do_with_remainding** – ‘return’ (default): function will return `param_args`, `param_kwargs`,
    `remaining_kwargs`
    ‘ignore’: function will return `param_args`, `param_kwargs`
    ‘assert_empty’: function will assert that `remaining_kwargs` is empty and then
    return `param_args`, `param_kwargs`
  * **include_all_when_var_keywords_in_params** – If True and `params` has a
    VAR_KEYWORD parameter, the remaining kwargs are merged into `param_kwargs`
    (leaving an empty remainder).
  * **assert_no_missing_position_only_args** – If True, assert that no
    position-only argument is missing from `kwargs`.
  * **kwargs** – The kwargs to extract the args from
* **Returns:**
  A (param_args, param_kwargs, remaining_kwargs) tuple.

### dol.signatures.extract_arguments_asserting_no_remainder(params, , what_to_do_with_remainding='assert_empty', include_all_when_var_keywords_in_params=False, assert_no_missing_position_only_args=False, \*\*kwargs)

Extract arguments needed to satisfy the params of a callable, dealing with the
dirty details.

Returns an (param_args, param_kwargs, remaining_kwargs) tuple where

- param_args are the values of kwargs that are PO (POSITION_ONLY) as defined by
  params,
- param_kwargs are those names that are both in params and not in param_args, and
- remaining_kwargs are the remaining.

Intended usage: When you need to call a function `func` that has some
position-only arguments,
but you have a kwargs dict of arguments in your hand. You can’t just to `func(
**kwargs)`.
But you can (now) do

```python
# extract from kwargs what you need for func
args, kwargs, remaining = extract_arguments(kwargs, func)
# ... check if remaining is empty (or not, depending on your paranoia),
# and then call the func:
func(*args, **kwargs)
```

(And if you doing that a lot: Do put it in a decorator!)

#### SEE ALSO
extract_arguments.without_remainding

The most frequent case you’ll encounter is when there’s no POSITION_ONLY args,
your param_args will be empty
and you param_kwargs will contain all the arguments that match params,
in the order of these params.

```pycon
>>> from inspect import signature
>>> def f(a, b, c=None, d=0):
...     ...
...
>>> extract_arguments(f, b=2, a=1, c=3, d=4, extra="stuff")
((), {'a': 1, 'b': 2, 'c': 3, 'd': 4}, {'extra': 'stuff'})
```

But sometimes you do have POSITION_ONLY arguments.
What extract_arguments will do for you is return the value of these as the first
element of
the triple.

```pycon
>>> def f(a, b, c=None, /, d=0):
...     ...
...
>>> extract_arguments(f, b=2, a=1, c=3, d=4, extra="stuff")
((1, 2, 3), {'d': 4}, {'extra': 'stuff'})
```

Note above how we get `(1, 2, 3)`, the order defined by the func’s signature,
instead of `(2, 1, 3)`, the order defined by the kwargs.
So it’s the params (e.g. function signature) that determine the order, not kwargs.
When using to call a function, this is especially crucial if we use POSITION_ONLY
arguments.

See also that the third output, the remaining_kwargs, as `{'extra': 'stuff'}` since
it was not in the params of the function.
Even if you include a VAR_KEYWORD kind of argument in the function, it won’t change
this behavior.

```pycon
>>> def f(a, b, c=None, /, d=0, **kws):
...     ...
...
>>> extract_arguments(f, b=2, a=1, c=3, d=4, extra="stuff")
((1, 2, 3), {'d': 4}, {'extra': 'stuff'})
```

This is because we don’t want to assume that all the kwargs can actually be
included in a call to the function behind the params.
Instead, the user can chose whether to include the remainder by doing a:

```python
param_kwargs.update(remaining_kwargs)
```

et voilà.

That said, we do understand that it may be a common pattern, so we’ll do that
extra step for you
if you specify `include_all_when_var_keywords_in_params=True`.

```pycon
>>> def f(a, b, c=None, /, d=0, **kws):
...     ...
...
>>> extract_arguments(
...     f,
...     b=2,
...     a=1,
...     c=3,
...     d=4,
...     extra="stuff",
...     include_all_when_var_keywords_in_params=True,
... )
((1, 2, 3), {'d': 4, 'extra': 'stuff'}, {})
```

If you’re expecting no remainder you might want to just get the args and kwargs (
not this third
expected-to-be-empty remainder). You have two ways to do that, specifying:

- `what_to_do_with_remainding='ignore'`, which will just return the (args,
  kwargs) pair
- `what_to_do_with_remainding='assert_empty'`, which will do the same, but first
  assert the remainder is empty

We suggest to use `functools.partial` to configure the `argument_argument` you need.

```pycon
>>> from functools import partial
>>> arg_extractor = partial(
...     extract_arguments,
...     what_to_do_with_remainding="assert_empty",
...     include_all_when_var_keywords_in_params=True,
... )
>>> def f(a, b, c=None, /, d=0, **kws):
...     ...
...
>>> arg_extractor(f, b=2, a=1, c=3, d=4, extra="stuff")
((1, 2, 3), {'d': 4, 'extra': 'stuff'})
```

And what happens if the kwargs doesn’t contain all the POSITION_ONLY arguments?

```pycon
>>> def f(a, b, c=None, /, d=0):
...     ...
...
>>> extract_arguments(f, b=2, d="is a kw arg", e="is not an arg at all")
((MissingArgValFor("a"), 2, MissingArgValFor("c")), {'d': 'is a kw arg'}, {'e': 'is not an arg at all'})
```

A few more examples…

Let’s call `extract_arguments` with params being not a function,
but, a Signature instance, a mapping whose values are Parameter instances,
or an iterable of Parameter instances…

```pycon
>>> def func(a, b, /, c=None, *, d=0, **kws):
...     ...
...
>>> sig = Signature.from_callable(func)
>>> param_map = sig.parameters
>>> param_iterable = param_map.values()
>>> kwargs = dict(b=2, a=1, c=3, d=4, extra="stuff")
>>> assert extract_arguments(sig, **kwargs) == extract_arguments(func, **kwargs)
>>> assert extract_arguments(param_map, **kwargs) == extract_arguments(
...     func, **kwargs
... )
>>> assert extract_arguments(param_iterable, **kwargs) == extract_arguments(
...     func, **kwargs
... )
```

Edge case:
No params specified? No problem. You’ll just get empty args and kwargs. Everything
in the remainder

```pycon
>>> extract_arguments(params=(), b=2, a=1, c=3, d=0)
((), {}, {'b': 2, 'a': 1, 'c': 3, 'd': 0})
```

* **Parameters:**
  * **params** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)], [`Signature`](https://docs.python.org/3/library/inspect.html#inspect.Signature), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)], [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Specifies what PO arguments should be extracted.
    Could be a callable, Signature, iterable of Parameters…
  * **what_to_do_with_remainding** – ‘return’ (default): function will return `param_args`, `param_kwargs`,
    `remaining_kwargs`
    ‘ignore’: function will return `param_args`, `param_kwargs`
    ‘assert_empty’: function will assert that `remaining_kwargs` is empty and then
    return `param_args`, `param_kwargs`
  * **include_all_when_var_keywords_in_params** – If True and `params` has a
    VAR_KEYWORD parameter, the remaining kwargs are merged into `param_kwargs`
    (leaving an empty remainder).
  * **assert_no_missing_position_only_args** – If True, assert that no
    position-only argument is missing from `kwargs`.
  * **kwargs** – The kwargs to extract the args from
* **Returns:**
  A (param_args, param_kwargs, remaining_kwargs) tuple.

### dol.signatures.extract_arguments_ignoring_remainder(params, , what_to_do_with_remainding='ignore', include_all_when_var_keywords_in_params=False, assert_no_missing_position_only_args=False, \*\*kwargs)

Extract arguments needed to satisfy the params of a callable, dealing with the
dirty details.

Returns an (param_args, param_kwargs, remaining_kwargs) tuple where

- param_args are the values of kwargs that are PO (POSITION_ONLY) as defined by
  params,
- param_kwargs are those names that are both in params and not in param_args, and
- remaining_kwargs are the remaining.

Intended usage: When you need to call a function `func` that has some
position-only arguments,
but you have a kwargs dict of arguments in your hand. You can’t just to `func(
**kwargs)`.
But you can (now) do

```python
# extract from kwargs what you need for func
args, kwargs, remaining = extract_arguments(kwargs, func)
# ... check if remaining is empty (or not, depending on your paranoia),
# and then call the func:
func(*args, **kwargs)
```

(And if you doing that a lot: Do put it in a decorator!)

#### SEE ALSO
extract_arguments.without_remainding

The most frequent case you’ll encounter is when there’s no POSITION_ONLY args,
your param_args will be empty
and you param_kwargs will contain all the arguments that match params,
in the order of these params.

```pycon
>>> from inspect import signature
>>> def f(a, b, c=None, d=0):
...     ...
...
>>> extract_arguments(f, b=2, a=1, c=3, d=4, extra="stuff")
((), {'a': 1, 'b': 2, 'c': 3, 'd': 4}, {'extra': 'stuff'})
```

But sometimes you do have POSITION_ONLY arguments.
What extract_arguments will do for you is return the value of these as the first
element of
the triple.

```pycon
>>> def f(a, b, c=None, /, d=0):
...     ...
...
>>> extract_arguments(f, b=2, a=1, c=3, d=4, extra="stuff")
((1, 2, 3), {'d': 4}, {'extra': 'stuff'})
```

Note above how we get `(1, 2, 3)`, the order defined by the func’s signature,
instead of `(2, 1, 3)`, the order defined by the kwargs.
So it’s the params (e.g. function signature) that determine the order, not kwargs.
When using to call a function, this is especially crucial if we use POSITION_ONLY
arguments.

See also that the third output, the remaining_kwargs, as `{'extra': 'stuff'}` since
it was not in the params of the function.
Even if you include a VAR_KEYWORD kind of argument in the function, it won’t change
this behavior.

```pycon
>>> def f(a, b, c=None, /, d=0, **kws):
...     ...
...
>>> extract_arguments(f, b=2, a=1, c=3, d=4, extra="stuff")
((1, 2, 3), {'d': 4}, {'extra': 'stuff'})
```

This is because we don’t want to assume that all the kwargs can actually be
included in a call to the function behind the params.
Instead, the user can chose whether to include the remainder by doing a:

```python
param_kwargs.update(remaining_kwargs)
```

et voilà.

That said, we do understand that it may be a common pattern, so we’ll do that
extra step for you
if you specify `include_all_when_var_keywords_in_params=True`.

```pycon
>>> def f(a, b, c=None, /, d=0, **kws):
...     ...
...
>>> extract_arguments(
...     f,
...     b=2,
...     a=1,
...     c=3,
...     d=4,
...     extra="stuff",
...     include_all_when_var_keywords_in_params=True,
... )
((1, 2, 3), {'d': 4, 'extra': 'stuff'}, {})
```

If you’re expecting no remainder you might want to just get the args and kwargs (
not this third
expected-to-be-empty remainder). You have two ways to do that, specifying:

- `what_to_do_with_remainding='ignore'`, which will just return the (args,
  kwargs) pair
- `what_to_do_with_remainding='assert_empty'`, which will do the same, but first
  assert the remainder is empty

We suggest to use `functools.partial` to configure the `argument_argument` you need.

```pycon
>>> from functools import partial
>>> arg_extractor = partial(
...     extract_arguments,
...     what_to_do_with_remainding="assert_empty",
...     include_all_when_var_keywords_in_params=True,
... )
>>> def f(a, b, c=None, /, d=0, **kws):
...     ...
...
>>> arg_extractor(f, b=2, a=1, c=3, d=4, extra="stuff")
((1, 2, 3), {'d': 4, 'extra': 'stuff'})
```

And what happens if the kwargs doesn’t contain all the POSITION_ONLY arguments?

```pycon
>>> def f(a, b, c=None, /, d=0):
...     ...
...
>>> extract_arguments(f, b=2, d="is a kw arg", e="is not an arg at all")
((MissingArgValFor("a"), 2, MissingArgValFor("c")), {'d': 'is a kw arg'}, {'e': 'is not an arg at all'})
```

A few more examples…

Let’s call `extract_arguments` with params being not a function,
but, a Signature instance, a mapping whose values are Parameter instances,
or an iterable of Parameter instances…

```pycon
>>> def func(a, b, /, c=None, *, d=0, **kws):
...     ...
...
>>> sig = Signature.from_callable(func)
>>> param_map = sig.parameters
>>> param_iterable = param_map.values()
>>> kwargs = dict(b=2, a=1, c=3, d=4, extra="stuff")
>>> assert extract_arguments(sig, **kwargs) == extract_arguments(func, **kwargs)
>>> assert extract_arguments(param_map, **kwargs) == extract_arguments(
...     func, **kwargs
... )
>>> assert extract_arguments(param_iterable, **kwargs) == extract_arguments(
...     func, **kwargs
... )
```

Edge case:
No params specified? No problem. You’ll just get empty args and kwargs. Everything
in the remainder

```pycon
>>> extract_arguments(params=(), b=2, a=1, c=3, d=0)
((), {}, {'b': 2, 'a': 1, 'c': 3, 'd': 0})
```

* **Parameters:**
  * **params** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)], [`Signature`](https://docs.python.org/3/library/inspect.html#inspect.Signature), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)], [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Specifies what PO arguments should be extracted.
    Could be a callable, Signature, iterable of Parameters…
  * **what_to_do_with_remainding** – ‘return’ (default): function will return `param_args`, `param_kwargs`,
    `remaining_kwargs`
    ‘ignore’: function will return `param_args`, `param_kwargs`
    ‘assert_empty’: function will assert that `remaining_kwargs` is empty and then
    return `param_args`, `param_kwargs`
  * **include_all_when_var_keywords_in_params** – If True and `params` has a
    VAR_KEYWORD parameter, the remaining kwargs are merged into `param_kwargs`
    (leaving an empty remainder).
  * **assert_no_missing_position_only_args** – If True, assert that no
    position-only argument is missing from `kwargs`.
  * **kwargs** – The kwargs to extract the args from
* **Returns:**
  A (param_args, param_kwargs, remaining_kwargs) tuple.

### dol.signatures.is_signature_error(e)

Check if an exception is a signature error

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### dol.signatures.name_of_obj(o, \*, base_name_of_obj=operator.attrgetter('_\_name_\_'), caught_exceptions=(<class 'AttributeError'>, ), default_factory=<function \_return_none>)

Tries to find the (or “a”) name for an object, even if `__name__` doesn’t exist.

```pycon
>>> name_of_obj(map)
'map'
>>> name_of_obj([1, 2, 3])
'list'
>>> name_of_obj(print)
'print'
>>> name_of_obj(lambda x: x)
'<lambda>'
>>> from functools import partial
>>> name_of_obj(partial(print, sep=","))
'print'
>>> from functools import cached_property
>>> class A:
...     @property
...     def prop(self):
...         return 1.0
...     @cached_property
...     def cached_prop(self):
...         return 2.0
>>> name_of_obj(A.prop)
'prop'
>>> name_of_obj(A.cached_prop)
'cached_prop'
```

Note that `name_of_obj` uses the `__name__` attribute as its base way to get
a name. You can customize this behavior though.
For example, see that:

```pycon
>>> from inspect import Signature
>>> name_of_obj(Signature.replace)
'replace'
```

If you want to get the fully qualified name of an object, you can do:

```pycon
>>> alt = partial(name_of_obj, base_name_of_obj=attrgetter('__qualname__'))
>>> alt(Signature.replace)
'Signature.replace'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### dol.signatures.param_for_kind(name=None, kind='positional_or_keyword', with_default=False, annotation)

Function to easily and flexibly make inspect.Parameter objects for testing.

It’s annoying to have to compose parameters from scratch to testing things.
This tool should help making it less annoying.

```pycon
>>> list(map(param_for_kind, param_kinds))
[<Parameter "POSITIONAL_ONLY">, <Parameter "POSITIONAL_OR_KEYWORD">, <Parameter "VAR_POSITIONAL">, <Parameter "KEYWORD_ONLY">, <Parameter "VAR_KEYWORD">]
>>> param_for_kind.positional_or_keyword()
<Parameter "POSITIONAL_OR_KEYWORD">
>>> param_for_kind.positional_or_keyword("foo")
<Parameter "foo">
>>> param_for_kind.keyword_only()
<Parameter "KEYWORD_ONLY">
>>> param_for_kind.keyword_only("baz", with_default=True)
<Parameter "baz='dflt_keyword_only'">
```

### dol.signatures.sort_params(params)

* **Parameters:**
  **params** – An iterable of `Parameter` instances
* **Returns:**
  A list of these instances sorted so as to obey the `kind` and `default`
  order rules of python signatures.

Note 1: It doesn’t mean that these params constitute a valid signature together,
since it doesn’t verify rules like unicity of names and variadic kinds.

Note 2: Though you can use `sorted` on an iterable of `i2.signatures.Param`
instances, know that even for sorting the three parameters below,
the `sort_params` function is more than twice as fast.

```pycon
>>> from inspect import Parameter
>>> sort_params(
...     [Parameter('a', kind=Parameter.POSITIONAL_OR_KEYWORD, default=1),
...     Parameter('b', kind=Parameter.POSITIONAL_ONLY),
...     Parameter('c', kind=Parameter.POSITIONAL_OR_KEYWORD)]
... )
[<Parameter "b">, <Parameter "c">, <Parameter "a=1">]
```

### dol.signatures.validate_signature(func)

Validates the signature of a function.

```pycon
>>> @validate_signature
... def has_valid_signature(x=Sig.empty, y=2):
...     pass
>>> # all good, no errors raised
>>>
>>> @validate_signature
... def does_no_have_valid_signature(x=2, y=Sig.empty):
...     pass
Traceback (most recent call last):
...
i2.signatures.InvalidSignature: Invalid signature for function <function does_no_have_valid_signature at 0x106a72a70>: non-default argument follows default a
rgument
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)
