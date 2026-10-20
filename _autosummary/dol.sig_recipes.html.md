# dol.sig_recipes

Recipes built on `Sig`: make signatures, and change the signature of functions.

- `mk_sig_from_args`, `params_of`, `insert_annotations`: make or edit signatures
- `call_forgivingly`, `call_somewhat_forgivingly`: call a function with the
  arguments it accepts, ignoring the others
- `all_pk_signature`, `ch_func_to_all_pk`, `ch_variadics_to_non_variadic_kind`,
  `kind_forgiving_func`, `normalized_func`: change the kinds of a function’s parameters
- `set_signature_of_func`, `replace_kwargs_using`, `use_interface`: give a function
  another signature
- `sig_to_dataclass`, `has_signature`, `copy_func`, `common_and_diff_argnames`

It used to be part of `dol.signatures`, which still re-exports all of it
(i2mint/dol#70).

```pycon
>>> from dol.sig_recipes import call_forgivingly
>>> def f(a, b=2):
...     return a + b
>>> call_forgivingly(f, 1, b=3, c=4)  # c is not a parameter of f: it is ignored
4
```

### Functions

| [`all_pk_signature`](#dol.sig_recipes.all_pk_signature)(callable_or_signature)            | Changes all (non-variadic) arguments to be of the PK (POSITION_OR_KEYWORD) kind.                                                  |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| [`call_forgivingly`](#dol.sig_recipes.call_forgivingly)(func, \*args, \*\*kwargs)         | Call function on given args and kwargs, but only taking what the function needs (not choking if they're extras variables)         |
| [`call_somewhat_forgivingly`](#dol.sig_recipes.call_somewhat_forgivingly)(func, args, kwargs)      | Call function on given args and kwargs, but with controllable argument leniency.                                                  |
| [`ch_func_to_all_pk`](#dol.sig_recipes.ch_func_to_all_pk)(func)                            | Returns a decorated function where all arguments are of the PK kind.                                                              |
| [`ch_signature_to_all_pk`](#dol.sig_recipes.ch_signature_to_all_pk)(callable_or_signature)      | Changes all (non-variadic) arguments to be of the PK (POSITION_OR_KEYWORD) kind.                                                  |
| [`ch_variadics_to_non_variadic_kind`](#dol.sig_recipes.ch_variadics_to_non_variadic_kind)(func, \*[, ...]) | A decorator that will change a VAR_POSITIONAL (`*args`) argument to a tuple (args) argument of the same name.                     |
| [`common_and_diff_argnames`](#dol.sig_recipes.common_and_diff_argnames)(func1, func2)             | Get list of argument names that are common to two functions, as well as the two lists of names that are different                 |
| `convert_to_PK`(kinds)                                                                              |                                                                                                                                   |
| [`copy_func`](#dol.sig_recipes.copy_func)(f)                                       | Copy a function (not sure it works with all types of callables)                                                                   |
| [`has_signature`](#dol.sig_recipes.has_signature)(obj[, robust])                       | Check if an object has a signature -- i.e. is callable and inspect.signature( obj) returns something.                             |
| [`insert_annotations`](#dol.sig_recipes.insert_annotations)(s, /, \*, ...)                  | Insert annotations in a signature.                                                                                                |
| [`kind_forgiving_func`](#dol.sig_recipes.kind_forgiving_func)(func[, kinds_modifier])        | Wraps the func, changing the argument kinds according to kinds_modifier.                                                          |
| [`mk_sig_from_args`](#dol.sig_recipes.mk_sig_from_args)(\*args_without_default, ...)      | Make a Signature instance by specifying args_without_default and args_with_defaults.                                              |
| `normalized_func`(func)                                                                             |                                                                                                                                   |
| `params_of`(obj)                                                                                    |                                                                                                                                   |
| [`replace_kwargs_using`](#dol.sig_recipes.replace_kwargs_using)(sig)                          | Decorator that replaces the variadic keyword argument of the target function using the `sig`, the signature of a source function. |
| [`set_signature_of_func`](#dol.sig_recipes.set_signature_of_func)(func, parameters, \*, ...)   | Set the signature of a function, with sugar.                                                                                      |
| [`sig_to_dataclass`](#dol.sig_recipes.sig_to_dataclass)(sig, \*[, cls_name, bases, ...])  | Make a `class` (through `make_dataclass`) from the given signature.                                                               |
| [`use_interface`](#dol.sig_recipes.use_interface)(interface_sig)                       | Use interface_sig as (enforced/validated) signature of the decorated function.                                                    |

### dol.sig_recipes.all_pk_signature(callable_or_signature)

Changes all (non-variadic) arguments to be of the PK (POSITION_OR_KEYWORD) kind.

Wrapping a function with the resulting signature doesn’t make that function callable
with PK kinds in itself.
It just gives it a signature without position and keyword ONLY kinds.
It should be used to wrap such a function that actually carries out the
implementation though!

```pycon
>>> def foo(w, /, x: float, y=1, *, z: int = 1, **kwargs):
...     ...
>>> def bar(*args, **kwargs):
...     ...
...
>>> from inspect import signature
>>> new_foo = all_pk_signature(foo)
>>> Sig(new_foo)
<Sig (w, x: float, y=1, z: int = 1, **kwargs)>
>>> all_pk_signature(signature(foo))
<Sig (w, x: float, y=1, z: int = 1, **kwargs)>
```

But note that the variadic arguments `*args` and `**kwargs` remain variadic:

```pycon
>>> all_pk_signature(signature(bar))
<Signature (*args, **kwargs)>
```

It works with `Sig` too (since Sig is a Signature), and maintains it’s other
attributes (like name).

```pycon
>>> sig = all_pk_signature(Sig(bar))
>>> sig
<Sig (*args, **kwargs)>
>>> sig.name
'bar'
```

#### SEE ALSO
`i2.signatures.kind_forgiving_func`

### dol.sig_recipes.call_forgivingly(func, \*args, \*\*kwargs)

Call function on given args and kwargs, but only taking what the function needs
(not choking if they’re extras variables)

#### TIP
If you into trouble because your kwargs has a ‘func’ key,
(which would then clash with the `func` param of call_forgivingly), then
use `_call_forgivingly` instead, specifying args and kwargs as tuple and
dict.

```pycon
>>> def foo(a, b: int = 0, c=None) -> int:
...     return "foo", (a, b, c)
>>> call_forgivingly(
...     foo,  # the function you want to call
...     "input for a",  # meant for a -- the first (and only) argument foo requires
...     c=42,  # skiping b and giving c a non-default value
...     intruder="argument",  # but wait, this argument name doesn't exist! Oh no!
... )  # well, as it happens, nothing bad -- the intruder argument is just ignored
('foo', ('input for a', 0, 42))
```

An example of what happens when variadic kinds are involved:

```pycon
>>> def bar(x, *args1, y=1, **kwargs1):
...     return x, args1, y, kwargs1
>>> call_forgivingly(bar, 1, 2, 3, y=4, z=5)
(1, (2, 3), 4, {'z': 5})
```

### dol.sig_recipes.call_somewhat_forgivingly(func, args, kwargs, enforce_sig=None)

Call function on given args and kwargs, but with controllable argument leniency.
By default, the function will only pick from args and kwargs what matches it’s
signature, ignoring anything else in args and kwargs.

But the real use of `call_somewhat_forgivingly` kicks in when you specify a
`enforce_sig`: A signature (or any object that can be resolved into a signature
through `Sig(enforce_sig)`) that will be used to bind the inputs, thus validating
them against the `enforce_sig` signature (including extra arguments, defaults,
etc.).

`call_somewhat_forgivingly` helps you do this kind of thing systematically.

```pycon
>>> f = lambda a: a * 11
>>> assert call_somewhat_forgivingly(f, (2,), {}) == f(2)
```

In the above, we have no `enforce_sig`. The real use of call_somewhat_forgivingly
is when we ask it to enforce a signature. Let’s do this by specifying a function
(no need for it to do anything: Only the signature is used.

```pycon
>>> g = lambda a, b=None: ...
```

Calling `f` on it’s normal set of inputs (one input in this case) gives you the
same thing as `f`:

```pycon
>>> assert call_somewhat_forgivingly(f, (2,), {}, enforce_sig=g) == f(2)
>>> assert call_somewhat_forgivingly(f, (), {'a': 2}, enforce_sig=g) == f(2)
```

If you call with an extra positional argument, it will just be ignored.

```pycon
>>> assert call_somewhat_forgivingly(f, (2, 'ignored'), {}, enforce_sig=g) == f(2)
```

If you call with a `b` keyword-argument (which matches `g`’s signature,
it will also be ignored.

```pycon
>>> assert call_somewhat_forgivingly(
... f, (2,), {'b': 'ignored'}, enforce_sig=g
... ) == f(2)
>>> assert call_somewhat_forgivingly(
...     f, (), {'a': 2, 'b': 'ignored'}, enforce_sig=g
... ) == f(2)
```

But if you call with three positional arguments (one more than g allows),
or call with a keyword argument that is not in `g`’s signature, it will
raise a `TypeError`:

```pycon
>>> call_somewhat_forgivingly(f,
...     (2, 'ignored', 'does_not_fit_g_signature_anymore'), {}, enforce_sig=g
... )
Traceback (most recent call last):
    ...
TypeError: too many positional arguments
>>> call_somewhat_forgivingly(f,
...     (2,), {'this_argname': 'is not in g'}, enforce_sig=g
... )
Traceback (most recent call last):
    ...
TypeError: got an unexpected keyword argument 'this_argname'
```

### dol.sig_recipes.ch_func_to_all_pk(func)

Returns a decorated function where all arguments are of the PK kind.
(PK: Positional_or_keyword)

* **Parameters:**
  **func** – A callable
* **Returns:**

```pycon
>>> def f(a, /, b, *, c=None, **kwargs):
...     return a + b * c
...
>>> print(Sig(f))
(a, /, b, *, c=None, **kwargs)
>>> ff = ch_func_to_all_pk(f)
>>> print(Sig(ff))
(a, b, c=None, **kwargs)
>>> ff(1, 2, 3)
7
>>>
>>> def g(x, y=1, *args, **kwargs):
...     ...
...
>>> print(Sig(g))
(x, y=1, *args, **kwargs)
>>> gg = ch_func_to_all_pk(g)
>>> print(Sig(gg))
(x, y=1, args=(), **kwargs)
```

### dol.sig_recipes.ch_signature_to_all_pk(callable_or_signature)

Changes all (non-variadic) arguments to be of the PK (POSITION_OR_KEYWORD) kind.

Wrapping a function with the resulting signature doesn’t make that function callable
with PK kinds in itself.
It just gives it a signature without position and keyword ONLY kinds.
It should be used to wrap such a function that actually carries out the
implementation though!

```pycon
>>> def foo(w, /, x: float, y=1, *, z: int = 1, **kwargs):
...     ...
>>> def bar(*args, **kwargs):
...     ...
...
>>> from inspect import signature
>>> new_foo = all_pk_signature(foo)
>>> Sig(new_foo)
<Sig (w, x: float, y=1, z: int = 1, **kwargs)>
>>> all_pk_signature(signature(foo))
<Sig (w, x: float, y=1, z: int = 1, **kwargs)>
```

But note that the variadic arguments `*args` and `**kwargs` remain variadic:

```pycon
>>> all_pk_signature(signature(bar))
<Signature (*args, **kwargs)>
```

It works with `Sig` too (since Sig is a Signature), and maintains it’s other
attributes (like name).

```pycon
>>> sig = all_pk_signature(Sig(bar))
>>> sig
<Sig (*args, **kwargs)>
>>> sig.name
'bar'
```

#### SEE ALSO
`i2.signatures.kind_forgiving_func`

### dol.sig_recipes.ch_variadics_to_non_variadic_kind(func, , ch_variadic_keyword_to_keyword=True)

A decorator that will change a VAR_POSITIONAL (`*args`) argument to a tuple (args)
argument of the same name.

Essentially, given a `func(a, *b, c, **d)` function want to get a
`new_func(a, b=(), c=None, d={})` that has the same functionality
(in fact, calls the original `func` function behind the scenes), but without
where the variadic arguments `*b` and `**d` are replaced with a `b` expecting an
iterable (e.g. tuple/list) and `d` expecting a `dict` to contain the
desired inputs.

Besides this, the decorator tries to be as conservative as possible, making only
the minimum changes needed to meet the goal of getting to a variadic-less
interface. When it doubt, and error will be raised.

```pycon
>>> def foo(a, *args, bar, **kwargs):
...     return f"{a=}, {args=}, {bar=}, {kwargs=}"
>>> assert str(Sig(foo)) == '(a, *args, bar, **kwargs)'
>>> wfoo = ch_variadics_to_non_variadic_kind(foo)
>>> str(Sig(wfoo))
'(a, args=(), *, bar, kwargs={})'
```

And now to do this:

```pycon
>>> foo(1, 2, 3, bar=4, hello="world")
"a=1, args=(2, 3), bar=4, kwargs={'hello': 'world'}"
```

We can do it like this instead:

```pycon
>>> wfoo(1, (2, 3), bar=4, kwargs=dict(hello="world"))
"a=1, args=(2, 3), bar=4, kwargs={'hello': 'world'}"
```

Note, the outputs are the same. It’s just the way we call our function that has
changed.

```pycon
>>> assert wfoo(1, (2, 3), bar=4, kwargs=dict(hello="world")
... ) == foo(1, 2, 3, bar=4, hello="world")
>>> assert wfoo(1, (2, 3), bar=4) == foo(1, 2, 3, bar=4)
>>> assert wfoo(1, (), bar=4) == foo(1, bar=4)
```

Note that if there is not variadic positional arguments, the variadic keyword
will still be a keyword-only kind.

```pycon
>>> @ch_variadics_to_non_variadic_kind
... def func(a, bar=None, **kwargs):
...     return f"{a=}, {bar=}, {kwargs=}"
>>> str(Sig(func))
'(a, bar=None, *, kwargs={})'
>>> assert func(1, bar=4, kwargs=dict(hello="world")
...     ) == "a=1, bar=4, kwargs={'hello': 'world'}"
```

If the function has neither variadic kinds, it will remain untouched.

```pycon
>>> def func(a, /, b, *, c=3):
...     return a + b + c
>>> ch_variadics_to_non_variadic_kind(func) == func
True
```

If you only want the variadic positional to be handled, but leave leave any
VARIADIC_KEYWORD kinds (`**kwargs`) alone, you can do so by setting
`ch_variadic_keyword_to_keyword=False`.
If you’ll need to use `ch_variadics_to_non_variadic_kind` in such a way
repeatedly, we suggest you use `functools.partial` to not have to specify this
configuration repeatedly.

```pycon
>>> from functools import partial
>>> tuple_the_args = partial(ch_variadics_to_non_variadic_kind,
...     ch_variadic_keyword_to_keyword=False
... )
>>> @tuple_the_args
... def foo(a, *args, bar=None, **kwargs):
...     return f"{a=}, {args=}, {bar=}, {kwargs=}"
>>> Sig(foo)
<Sig (a, args=(), *, bar=None, **kwargs)>
>>> foo(1, (2, 3), bar=4, hello="world")
"a=1, args=(2, 3), bar=4, kwargs={'hello': 'world'}"
```

### dol.sig_recipes.common_and_diff_argnames(func1, func2)

Get list of argument names that are common to two functions, as well as the two
lists of names that are different

* **Parameters:**
  * **func1** (`callable`) – First function
  * **func2** (`callable`) – Second function
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  A dict with fields ‘common’, ‘func1_not_func2’, and ‘func2_not_func1’

```pycon
>>> def f(t, h, i, n, k):
...     ...
...
>>> def g(t, w, i, c, e):
...     ...
...
>>> common_and_diff_argnames(f, g)
{'common': ['t', 'i'], 'func1_not_func2': ['h', 'n', 'k'], 'func2_not_func1': ['w', 'c', 'e']}
>>> common_and_diff_argnames(g, f)
{'common': ['t', 'i'], 'func1_not_func2': ['w', 'c', 'e'], 'func2_not_func1': ['h', 'n', 'k']}
```

### dol.sig_recipes.copy_func(f)

Copy a function (not sure it works with all types of callables)

### dol.sig_recipes.has_signature(obj, robust=False)

Check if an object has a signature – i.e. is callable and inspect.signature(
obj) returns something.

This can be used to more easily get signatures in bulk without having to write
try/catches:

```pycon
>>> from functools import partial
>>> len(
...     list(
...         filter(
...             None,
...             map(
...                 partial(has_signature, robust=False),
...                 (Sig, print, map, filter, Sig.wrap),
...             ),
...         )
...     )
... )
2
```

If robust is set to True, `has_signature` will use `Sig` to get the signature,
so will return True in most cases.

### dol.sig_recipes.insert_annotations(s, , , return_annotation, \*\*annotations)

Insert annotations in a signature.
(Note: not really insert but returns a copy of input signature)

```pycon
>>> from inspect import signature
>>> s = signature(lambda a, b, c=1, d="bar": 0)
>>> s
<Signature (a, b, c=1, d='bar')>
>>> ss = insert_annotations(s, b=int, d=str)
>>> ss
<Signature (a, b: int, c=1, d: str = 'bar')>
>>> insert_annotations(s, b=int, d=str, e=list)
Traceback (most recent call last):
...
AssertionError: These argument names weren't found in the signature: {'e'}
```

### dol.sig_recipes.kind_forgiving_func(func, kinds_modifier=<function convert_to_PK>)

Wraps the func, changing the argument kinds according to kinds_modifier.
The default behaviour is to change all kinds to POSITIONAL_OR_KEYWORD kinds.
The original purpose of this function is to remove argument-kind restriction
annoyances when doing functional manipulations such as:

```pycon
>>> from functools import partial
>>> isinstance_of_str = partial(isinstance, class_or_tuple=str)
>>> isinstance_of_str('I am a string')
Traceback (most recent call last):
  ...
TypeError: isinstance() takes no keyword arguments
```

Here, instead, we can just get a kinder version of the function and do what we
want to do:

```pycon
>>> _isinstance = kind_forgiving_func(isinstance)
>>> isinstance_of_str = partial(_isinstance, class_or_tuple=str)
>>> isinstance_of_str('I am a string')
True
>>> isinstance_of_str(42)
False
```

#### SEE ALSO
`i2.signatures.all_pk_signature`

### dol.sig_recipes.mk_sig_from_args(\*args_without_default, \*\*args_with_defaults)

Make a Signature instance by specifying args_without_default and
args_with_defaults.

```pycon
>>> mk_sig_from_args("a", "b", c=1, d="bar")
<Signature (a, b, c=1, d='bar')>
```

### dol.sig_recipes.replace_kwargs_using(sig)

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

### dol.sig_recipes.set_signature_of_func(func, parameters, , return_annotation, \_\_validate_parameters_\_=True)

Set the signature of a function, with sugar.

* **Parameters:**
  * **func** – Function whose signature you want to set
  * **signature** – A list of parameter specifications. This could be an
  * **that** (*inspect.Parameter object* *or* *anything*) – the mk_param function can resolve into an inspect.Parameter object.
  * **return_annotation** – Passed on to inspect.Signature.
  * **\_\_validate_parameters_\_** – Passed on to inspect.Signature.
* **Returns:**
  None (but sets the signature of the input function)

```pycon
>>> import inspect
>>> def foo(*args, **kwargs):
...     pass
...
>>> inspect.signature(foo)
<Signature (*args, **kwargs)>
>>> set_signature_of_func(foo, ["a", "b", "c"])
>>> inspect.signature(foo)
<Signature (a, b, c)>
>>> set_signature_of_func(
...     foo, ["a", ("b", None), ("c", 42, int)]
... )  # specifying defaults and annotations
>>> inspect.signature(foo)
<Signature (a, b=None, c: int = 42)>
>>> set_signature_of_func(
...     foo, ["a", "b", "c"], return_annotation=str
... )  # specifying return annotation
>>> inspect.signature(foo)
<Signature (a, b, c) -> str>
>>> # But you can always specify parameters the "long" way
>>> set_signature_of_func(
...     foo,
...     [inspect.Parameter(name="kws", kind=inspect.Parameter.VAR_KEYWORD)],
...     return_annotation=str,
... )
>>> inspect.signature(foo)
<Signature (**kws) -> str>
```

### dol.sig_recipes.sig_to_dataclass(sig, , cls_name=None, bases=(), module=None, \*\*kwargs)

Make a `class` (through `make_dataclass`) from the given signature.

* **Parameters:**
  * **sig** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Signature`](https://docs.python.org/3/library/inspect.html#inspect.Signature), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)], [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)], [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – A `SignatureAble`, that is, anything that ensure_signature can
    resolve into an `inspect.Signature` object, including a signature object
    itself, but also most callables, a list or params, etc.
  * **cls_name** – The same as `cls_name` of `dataclasses.make_dataclass`
  * **bases** – The same as `bases` of `dataclasses.make_dataclass`
  * **module** – Set to module (usually `__name__` to specify ther module of
    caller) so that the class and instances can be pickle-able.
  * **kwargs** – Passed on to `dataclasses.make_dataclass`
* **Returns:**
  A dataclass

```pycon
>>> def foo(a, /, b : int=2, *, c=3):
...     pass
...
>>> K = sig_to_dataclass(foo, cls_name='K')
>>> str(Sig(K))
'(a, b: int = 2, c=3) -> None'
>>> k = K(1,2,3)
>>> (k.a, k.b, k.c)
(1, 2, 3)
```

Would also work with any of these (and more):

```pycon
>>> K = sig_to_dataclass(Sig(foo), cls_name='K')
>>> K = sig_to_dataclass(Sig(foo).params, cls_name='K')
```

#### NOTE
`cls_name` is not required (we’ll try to figure out a good default for you),
but it’s advised to only use this convenience in extreme mode.
Choosing your own name might make for a safer future if you’re reusing your class.

### dol.sig_recipes.tuple_the_args(func, , ch_variadic_keyword_to_keyword=False)

A decorator that will change a VAR_POSITIONAL (\*args) argument to a tuple (args)
argument of the same name.

### dol.sig_recipes.use_interface(interface_sig)

Use interface_sig as (enforced/validated) signature of the decorated function.
That is, the decorated function will use the original function has the backend,
the function actually doing the work, but with a frontend specified
(in looks and in argument validation) `interface_sig`

consider the situation where are functionality is parametrized by a
function `g` taking two inputs, `a`, and `b`.
Now you want to carry out this functionality using a function `f` that does what
`g` should do, but doesn’t use `a`, and doesn’t even have it in it’s arguments.

The solution to this is to \_adapt_ `f` to the `g` interface:

```python
def my_g(a, b):
    return f(a)
```

and use `my_g`.

```pycon
>>> f = lambda a: a * 11
>>> interface = lambda a, b=None: ...
>>>
>>> new_f = use_interface(interface)(f)
```

See how only the first argument, or `a` keyword argument, is taken into account
in `new_f`:

```pycon
>>> assert new_f(2) == f(2)
>>> assert new_f(2, 3) == f(2)
>>> assert new_f(2, b=3) == f(2)
>>> assert new_f(b=3, a=2) == f(2)
```

But if we add more positional arguments than `interface` allows,
or any keyword arguments that `interface` doesn’t recognize…

```pycon
>>> new_f(1,2,3)
Traceback (most recent call last):
  ...
TypeError: too many positional arguments
>>> new_f(1, c=2)
Traceback (most recent call last):
  ...
TypeError: got an unexpected keyword argument 'c'
```
