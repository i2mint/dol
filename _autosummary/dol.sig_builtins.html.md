# dol.sig_builtins

Signatures for the callables that `inspect.signature` can’t handle.

Many builtins (`print`, `map`, `itertools.product`, …) have no signature that
`inspect` can read. This module holds hand-made ones, and the function `Sig` uses
to get a signature for any callable, falling back to those:

- `_robust_signature_of_callable`: `inspect.signature`, with the fallbacks below
- `sigs_for_builtins`, `sigs_for_builtin_modules`, `sigs_for_type_name`: the
  hand-made signatures, by name (`sigs_for_sigless_builtin_name` merges the first two)
- `resolve_function`, `dict_of_attribute_signatures`: helpers

It depends on nothing else in `dol`. It used to be part of `dol.signatures`, which
still re-exports all of it (i2mint/dol#70).

```pycon
>>> from dol.sig_builtins import _robust_signature_of_callable
>>> _robust_signature_of_callable(isinstance)
<Signature (obj, class_or_tuple, /)>
```

### Functions

| [`dict_of_attribute_signatures`](#dol.sig_builtins.dict_of_attribute_signatures)(cls)   | A function that extracts the signatures of all callable attributes of a class.   |
|--------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`resolve_function`](#dol.sig_builtins.resolve_function)(obj)               | Get the underlying function of a property or cached_property                     |

### dol.sig_builtins.dict_of_attribute_signatures(cls)

A function that extracts the signatures of all callable attributes of a class.

* **Parameters:**
  **cls** ([`type`](https://docs.python.org/3/builtins/functions.html#type)) – The class that holds the the `(name, func)` pairs we want to extract.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Signature`](https://docs.python.org/3/library/inspect.html#inspect.Signature)]
* **Returns:**
  A dict of `(name, signature(func))` pairs extracted from class.

One of the intended applications is to use `dict_of_attribute_signatures` as a
decorator, like so:

```pycon
>>> @dict_of_attribute_signatures
... class names_and_signatures:
...     def foo(x: str, *, y=2) -> tuple: ...
...     def bar(z, /) -> float: ...
>>> names_and_signatures
{'foo': <Signature (x: str, *, y=2) -> tuple>, 'bar': <Signature (z, /) -> float>}
```

### dol.sig_builtins.resolve_function(obj)

Get the underlying function of a property or cached_property

Note that if all conditions fail, the object itself is returned.

The problem this function solves is that sometimes there’s a function behind an
object, but it’s not always easy to get to it. For example, in a class, you might
want to get the source of the code decorated with `@property`, a
`@cached_property`, or a `partial` function.

Consider the following example:

```pycon
>>> from functools import cached_property, partial
>>> class C:
...     @property
...     def prop(self):
...         pass
...     @cached_property
...     def cached_prop(self):
...         pass
...     partial_func = partial(partial)
```

Note that `prop` is not callable, and you can’t get its source.

```pycon
>>> import inspect
>>> callable(C.prop)
False
>>> inspect.getsource(C.prop)
Traceback (most recent call last):
...
TypeError: <property object at 0x...> is not a module, class, method, function, traceback, frame, or code object
```

But if you grab the underlying function, you can get the source:

```pycon
>>> func = resolve_function(C.prop)
>>> callable(func)
True
>>> isinstance(inspect.getsource(func), str)
True
```

Same goes with `cached_property` and `partial`:

```pycon
>>> isinstance(inspect.getsource(resolve_function(C.cached_prop)), str)
True
>>> isinstance(inspect.getsource(resolve_function(C.partial_func)), str)
True
```

* **Return type:**
  [`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[`T`, [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)]
