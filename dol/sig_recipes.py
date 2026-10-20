"""Recipes built on ``Sig``: make signatures, and change the signature of functions.

- ``mk_sig_from_args``, ``params_of``, ``insert_annotations``: make or edit signatures
- ``call_forgivingly``, ``call_somewhat_forgivingly``: call a function with the
  arguments it accepts, ignoring the others
- ``all_pk_signature``, ``ch_func_to_all_pk``, ``ch_variadics_to_non_variadic_kind``,
  ``kind_forgiving_func``, ``normalized_func``: change the kinds of a function's parameters
- ``set_signature_of_func``, ``replace_kwargs_using``, ``use_interface``: give a function
  another signature
- ``sig_to_dataclass``, ``has_signature``, ``copy_func``, ``common_and_diff_argnames``

It used to be part of ``dol.signatures``, which still re-exports all of it
(i2mint/dol#70).

    >>> from dol.sig_recipes import call_forgivingly
    >>> def f(a, b=2):
    ...     return a + b
    >>> call_forgivingly(f, 1, b=3, c=4)  # c is not a parameter of f: it is ignored
    4
"""

from inspect import Parameter, Signature, signature
from types import FunctionType
from collections.abc import Callable, Mapping
from functools import partial

from dol.signatures import (
    Sig,
    empty,
    wraps,
    update_wrapper,
    ensure_params,
    ensure_signature,
    FuncCallNotMatchingSignature,
    HasParams,
    SignatureAble,
    KO,
    PK,
    PO,
    VP,
    VK,
    var_param_kinds,
)


########################################################################################
# Recipes


def mk_sig_from_args(*args_without_default, **args_with_defaults):
    """Make a Signature instance by specifying args_without_default and
    args_with_defaults.

    >>> mk_sig_from_args("a", "b", c=1, d="bar")
    <Signature (a, b, c=1, d='bar')>
    """
    assert all(isinstance(x, str) for x in args_without_default), (
        "all default-less arguments must be strings"
    )
    return Sig.from_objs(
        *args_without_default, **args_with_defaults
    ).to_simple_signature()


def _remove_variadics_from_sig(sig, ch_variadic_keyword_to_keyword=True):
    """Remove variadics from signature

    >>> def foo(a, *args, bar, **kwargs):
    ...     return f"{a=}, {args=}, {bar=}, {kwargs=}"
    >>> sig = Sig(foo)
    >>> assert str(sig) == '(a, *args, bar, **kwargs)'
    >>> new_sig = _remove_variadics_from_sig(sig)
    >>> str(new_sig)=='(a, args=(), *, bar, kwargs={})'
    True

    Note that if there is not variadic positional arguments, the variadic keyword
    will still be a keyword-only kind.

    >>> def func(a, bar=None, **kwargs):
    ...     return f"{a=}, {bar=}, {kwargs=}"
    >>> nsig = _remove_variadics_from_sig(Sig(func))
    >>> assert str(nsig)=='(a, bar=None, *, kwargs={})'

    If the function has neither variadic kinds, it will remain untouched.

    >>> def func(a, /, b, *, c=3):
    ...     return a + b + c
    >>> sig = _remove_variadics_from_sig(Sig(func))

    >>> assert sig == Sig(func)


    If you only want the variadic positional to be handled, but leave leave any
    VARIADIC_KEYWORD kinds (``**kwargs``) alone, you can do so by setting
    `ch_variadic_keyword_to_keyword=False`.

    >>> def foo(a, *args, bar=None, **kwargs):
    ...     return f"{a=}, {args=}, {bar=}, {kwargs=}"
    >>> assert str(Sig(_remove_variadics_from_sig(Sig(foo))))=='(a, args=(), *, bar=None, kwargs={})'
    """

    idx_of_vp = sig.index_of_var_positional
    var_keyword_argname = sig.var_keyword_name
    result_sig = sig
    if idx_of_vp is not None or var_keyword_argname is not None:
        params = sig.params
        if var_keyword_argname:  # if there's a VAR_KEYWORD argument
            if ch_variadic_keyword_to_keyword:
                i = sig.index_of_var_keyword
                # TODO: Reflect on pros/cons of having mutable {} default here:
                params[i] = params[i].replace(kind=Parameter.KEYWORD_ONLY, default={})

        try:  # TODO: Avoid this try catch. Look in advance for default ordering?
            if idx_of_vp is not None:
                params[idx_of_vp] = params[idx_of_vp].replace(kind=PK, default=())
            result_sig = Signature(params, return_annotation=sig.return_annotation)
        except ValueError:
            if idx_of_vp is not None:
                params[idx_of_vp] = params[idx_of_vp].replace(kind=PK)
            result_sig = Signature(params, return_annotation=sig.return_annotation)

    return result_sig


# TODO: Might want to make func be a positional-only argument, because if kwargs
#  contains a func key, we have a problem. But call_forgivingly is used broadly,
#  so must first test all dependents before making this change.
def call_forgivingly(func, *args, **kwargs):
    """
    Call function on given args and kwargs, but only taking what the function needs
    (not choking if they're extras variables)

    Tip:
        If you into trouble because your kwargs has a 'func' key,
        (which would then clash with the ``func`` param of call_forgivingly), then
        use `_call_forgivingly` instead, specifying args and kwargs as tuple and
        dict.

    >>> def foo(a, b: int = 0, c=None) -> int:
    ...     return "foo", (a, b, c)
    >>> call_forgivingly(
    ...     foo,  # the function you want to call
    ...     "input for a",  # meant for a -- the first (and only) argument foo requires
    ...     c=42,  # skiping b and giving c a non-default value
    ...     intruder="argument",  # but wait, this argument name doesn't exist! Oh no!
    ... )  # well, as it happens, nothing bad -- the intruder argument is just ignored
    ('foo', ('input for a', 0, 42))

    An example of what happens when variadic kinds are involved:

    >>> def bar(x, *args1, y=1, **kwargs1):
    ...     return x, args1, y, kwargs1
    >>> call_forgivingly(bar, 1, 2, 3, y=4, z=5)
    (1, (2, 3), 4, {'z': 5})
    """
    # TODO: Examples that don't yet behave as one might want:
    # >>> def bar(x, y=1, **kwargs1):
    # ...     return x, y, kwargs1
    # >>> call_forgivingly(bar, 1, 2, 3, y=4, z=5)
    # (1, 4, {'z': 5})
    # >>> def bar(x, *args1, y=1):
    # ...     return x, args1, y
    # >>> call_forgivingly(bar, 1, 2, 3, y=4, z=5)
    # (1, (2, 3), {'z': 5})
    return _call_forgivingly(func, args, kwargs)


# TODO: See if there's a more elegant way to do this
def _call_forgivingly(func, args, kwargs):
    """
    Helper for _call_forgivingly.
    """

    sig = Sig(func)
    arguments = sig.map_arguments(args, kwargs, allow_excess=True)
    _args, _kwargs = sig.mk_args_and_kwargs(arguments, args_limit=len(args))
    return func(*_args, **_kwargs)

    # sig = Sig(func)
    # variadic_kinds = {
    #     name: kind for name, kind in sig.kinds.items() if kind in [VP, VK]
    # }
    # if VP in variadic_kinds.values() and VK in variadic_kinds.values():
    #     _args = args
    #     _kwargs = kwargs
    # else:
    #     new_sig = sig - variadic_kinds.keys()
    #     _args, _kwargs = new_sig.source_args_and_kwargs(*args, _ignore_kind=False, **kwargs)
    #     for k, v in _kwargs.items():
    #         if k not in kwargs:
    #             _args = _args + (v,)
    #     _kwargs = {k: v for k, v in _kwargs.items() if k in kwargs}
    #     if VP in variadic_kinds.values():
    #         _args = args
    #     elif VK in variadic_kinds.values():
    #         _kwargs = dict(_kwargs, **kwargs)
    # return func(*_args, **_kwargs)


def call_somewhat_forgivingly(
    func, args, kwargs, enforce_sig: SignatureAble | None = None
):
    """Call function on given args and kwargs, but with controllable argument leniency.
    By default, the function will only pick from args and kwargs what matches it's
    signature, ignoring anything else in args and kwargs.

    But the real use of `call_somewhat_forgivingly` kicks in when you specify a
    `enforce_sig`: A signature (or any object that can be resolved into a signature
    through `Sig(enforce_sig)`) that will be used to bind the inputs, thus validating
    them against the `enforce_sig` signature (including extra arguments, defaults,
    etc.).

    `call_somewhat_forgivingly` helps you do this kind of thing systematically.

    >>> f = lambda a: a * 11
    >>> assert call_somewhat_forgivingly(f, (2,), {}) == f(2)

    In the above, we have no `enforce_sig`. The real use of call_somewhat_forgivingly
    is when we ask it to enforce a signature. Let's do this by specifying a function
    (no need for it to do anything: Only the signature is used.

    >>> g = lambda a, b=None: ...

    Calling `f` on it's normal set of inputs (one input in this case) gives you the
    same thing as `f`:

    >>> assert call_somewhat_forgivingly(f, (2,), {}, enforce_sig=g) == f(2)
    >>> assert call_somewhat_forgivingly(f, (), {'a': 2}, enforce_sig=g) == f(2)

    If you call with an extra positional argument, it will just be ignored.

    >>> assert call_somewhat_forgivingly(f, (2, 'ignored'), {}, enforce_sig=g) == f(2)

    If you call with a `b` keyword-argument (which matches `g`'s signature,
    it will also be ignored.

    >>> assert call_somewhat_forgivingly(
    ... f, (2,), {'b': 'ignored'}, enforce_sig=g
    ... ) == f(2)
    >>> assert call_somewhat_forgivingly(
    ...     f, (), {'a': 2, 'b': 'ignored'}, enforce_sig=g
    ... ) == f(2)

    But if you call with three positional arguments (one more than g allows),
    or call with a keyword argument that is not in `g`'s signature, it will
    raise a `TypeError`:

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
    """
    enforce_sig = Sig(enforce_sig or func)
    # Validate that args and kwargs are compatible with enforce_sig
    enforce_sig.bind(*args, **kwargs)
    return _call_forgivingly(func, args, kwargs)


def convert_to_PK(kinds):
    return {name: PK for name in kinds}


def kind_forgiving_func(func, kinds_modifier=convert_to_PK):
    """Wraps the func, changing the argument kinds according to kinds_modifier.
    The default behaviour is to change all kinds to POSITIONAL_OR_KEYWORD kinds.
    The original purpose of this function is to remove argument-kind restriction
    annoyances when doing functional manipulations such as:

    >>> from functools import partial
    >>> isinstance_of_str = partial(isinstance, class_or_tuple=str)
    >>> isinstance_of_str('I am a string')
    Traceback (most recent call last):
      ...
    TypeError: isinstance() takes no keyword arguments

    Here, instead, we can just get a kinder version of the function and do what we
    want to do:

    >>> _isinstance = kind_forgiving_func(isinstance)
    >>> isinstance_of_str = partial(_isinstance, class_or_tuple=str)
    >>> isinstance_of_str('I am a string')
    True
    >>> isinstance_of_str(42)
    False

    See also:
        ``i2.signatures.all_pk_signature``
    """
    sig = Sig(func)
    kinds_modif = kinds_modifier(sig.kinds)
    _sig = sig.ch_kinds(**kinds_modif)

    @_sig
    @wraps(func)
    def _func(*args, **kwargs):
        _args, _kwargs = sig.extract_args_and_kwargs(
            *args, _allow_excess=False, **kwargs
        )
        return func(*_args, **_kwargs)

    # _func.__signature__ = sig
    return _func


# TODO: Should we protect from misuse with signature compatibility check?
def use_interface(interface_sig):
    """Use interface_sig as (enforced/validated) signature of the decorated function.
    That is, the decorated function will use the original function has the backend,
    the function actually doing the work, but with a frontend specified
    (in looks and in argument validation) `interface_sig`

    consider the situation where are functionality is parametrized by a
    function `g` taking two inputs, `a`, and `b`.
    Now you want to carry out this functionality using a function `f` that does what
    `g` should do, but doesn't use `a`, and doesn't even have it in it's arguments.

    The solution to this is to _adapt_ `f` to the `g` interface:

    .. code-block:: python

        def my_g(a, b):
            return f(a)

    and use `my_g`.

    >>> f = lambda a: a * 11
    >>> interface = lambda a, b=None: ...
    >>>
    >>> new_f = use_interface(interface)(f)

    See how only the first argument, or `a` keyword argument, is taken into account
    in `new_f`:

    >>> assert new_f(2) == f(2)
    >>> assert new_f(2, 3) == f(2)
    >>> assert new_f(2, b=3) == f(2)
    >>> assert new_f(b=3, a=2) == f(2)

    But if we add more positional arguments than `interface` allows,
    or any keyword arguments that `interface` doesn't recognize...

    >>> new_f(1,2,3)
    Traceback (most recent call last):
      ...
    TypeError: too many positional arguments
    >>> new_f(1, c=2)
    Traceback (most recent call last):
      ...
    TypeError: got an unexpected keyword argument 'c'
    """
    interface_sig = Sig(interface_sig)

    def interface_wrapped_decorator(func):
        @interface_sig
        def _func(*args, **kwargs):
            return call_somewhat_forgivingly(
                func, args, kwargs, enforce_sig=interface_sig
            )

        return _func

    return interface_wrapped_decorator


import inspect


def has_signature(obj, robust=False):
    """Check if an object has a signature -- i.e. is callable and inspect.signature(
    obj) returns something.

    This can be used to more easily get signatures in bulk without having to write
    try/catches:

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
    ... )  # doctest: +SKIP
    2

    If robust is set to True, `has_signature` will use `Sig` to get the signature,
    so will return True in most cases.
    """
    if robust:
        return bool(Sig.sig_or_none(obj))
    else:
        try:
            return bool((callable(obj) or None) and signature(obj))
        except ValueError:
            return False


# TODO: Need to define and use this function more carefully.
#   Is the goal to remove positional? Remove variadics? Normalize the signature?
def all_pk_signature(callable_or_signature: Callable | Signature):
    """Changes all (non-variadic) arguments to be of the PK (POSITION_OR_KEYWORD) kind.

    Wrapping a function with the resulting signature doesn't make that function callable
    with PK kinds in itself.
    It just gives it a signature without position and keyword ONLY kinds.
    It should be used to wrap such a function that actually carries out the
    implementation though!

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

    But note that the variadic arguments ``*args`` and ``**kwargs`` remain variadic:

    >>> all_pk_signature(signature(bar))
    <Signature (*args, **kwargs)>

    It works with `Sig` too (since Sig is a Signature), and maintains it's other
    attributes (like name).

    >>> sig = all_pk_signature(Sig(bar))
    >>> sig
    <Sig (*args, **kwargs)>
    >>> sig.name
    'bar'

    See also:
        ``i2.signatures.kind_forgiving_func``
    """

    if isinstance(callable_or_signature, Signature):
        sig = callable_or_signature

        def changed_params():
            for p in sig.parameters.values():
                if p.kind not in var_param_kinds:
                    yield p.replace(kind=PK)
                else:
                    yield p

        new_sig = type(sig)(
            list(changed_params()), return_annotation=sig.return_annotation
        )
        for attrname, attrval in getattr(sig, "__dict__", {}).items():
            setattr(new_sig, attrname, attrval)
        return new_sig
    elif isinstance(callable_or_signature, Callable):
        func = callable_or_signature
        sig = all_pk_signature(Sig(func))
        return sig(func)


# Changed ch_signature_to_all_pk to all_pk_signature because ch_signature_to_all_pk
# was misleading: It doesn't change anything at all, it returns a constructed signature.
# It doesn't change all kinds to PK -- just the non-variadic ones.
ch_signature_to_all_pk = all_pk_signature  # alias for back-compatibility


def normalized_func(func):
    sig = Sig(func)

    def argument_values_tuple(args, kwargs):
        b = sig.bind(*args, **kwargs)
        arg_vals = dict(b.arguments)

        poa, pka, vpa, koa, vka = [], [], (), {}, {}

        for name, val in arg_vals.items():
            kind = sig.kinds[name]
            if kind == PO:
                poa.append(val)
            elif kind == PK:
                pka.append(val)
            elif kind == VP:
                vpa = val  # there can only be one VP!
            elif kind == KO:
                koa.update({name: val})
            elif kind == VK:
                vka = val  # there can only be one VK!
        return poa, pka, vpa, koa, vka

    def _args_and_kwargs(args, kwargs):
        poa, pka, vpa, koa, vka = argument_values_tuple(args, kwargs)

        _args = (*poa, *pka, *vpa)
        _kwargs = {**koa, **vka}

        return _args, _kwargs

    # @sig.modified(**{name: {'kind': PK} for name in sig.names})
    def _func(*args, **kwargs):
        # poa, pka, vpa, koa, vka = argument_values_tuple(args, kwargs)
        # print(poa, pka, vpa, koa, vka)
        _args, _kwargs = _args_and_kwargs(args, kwargs)
        return func(*_args, **_kwargs)

    return _func


def ch_variadics_to_non_variadic_kind(func, *, ch_variadic_keyword_to_keyword=True):
    """A decorator that will change a VAR_POSITIONAL (``*args``) argument to a tuple (args)
    argument of the same name.

    Essentially, given a `func(a, *b, c, **d)` function want to get a
    `new_func(a, b=(), c=None, d={})` that has the same functionality
    (in fact, calls the original `func` function behind the scenes), but without
    where the variadic arguments ``*b`` and ``**d`` are replaced with a `b` expecting an
    iterable (e.g. tuple/list) and `d` expecting a `dict` to contain the
    desired inputs.

    Besides this, the decorator tries to be as conservative as possible, making only
    the minimum changes needed to meet the goal of getting to a variadic-less
    interface. When it doubt, and error will be raised.

    >>> def foo(a, *args, bar, **kwargs):
    ...     return f"{a=}, {args=}, {bar=}, {kwargs=}"
    >>> assert str(Sig(foo)) == '(a, *args, bar, **kwargs)'
    >>> wfoo = ch_variadics_to_non_variadic_kind(foo)
    >>> str(Sig(wfoo))
    '(a, args=(), *, bar, kwargs={})'

    And now to do this:

    >>> foo(1, 2, 3, bar=4, hello="world")
    "a=1, args=(2, 3), bar=4, kwargs={'hello': 'world'}"

    We can do it like this instead:

    >>> wfoo(1, (2, 3), bar=4, kwargs=dict(hello="world"))
    "a=1, args=(2, 3), bar=4, kwargs={'hello': 'world'}"

    Note, the outputs are the same. It's just the way we call our function that has
    changed.

    >>> assert wfoo(1, (2, 3), bar=4, kwargs=dict(hello="world")
    ... ) == foo(1, 2, 3, bar=4, hello="world")
    >>> assert wfoo(1, (2, 3), bar=4) == foo(1, 2, 3, bar=4)
    >>> assert wfoo(1, (), bar=4) == foo(1, bar=4)

    Note that if there is not variadic positional arguments, the variadic keyword
    will still be a keyword-only kind.

    >>> @ch_variadics_to_non_variadic_kind
    ... def func(a, bar=None, **kwargs):
    ...     return f"{a=}, {bar=}, {kwargs=}"
    >>> str(Sig(func))
    '(a, bar=None, *, kwargs={})'
    >>> assert func(1, bar=4, kwargs=dict(hello="world")
    ...     ) == "a=1, bar=4, kwargs={'hello': 'world'}"

    If the function has neither variadic kinds, it will remain untouched.

    >>> def func(a, /, b, *, c=3):
    ...     return a + b + c
    >>> ch_variadics_to_non_variadic_kind(func) == func
    True

    If you only want the variadic positional to be handled, but leave leave any
    VARIADIC_KEYWORD kinds (``**kwargs``) alone, you can do so by setting
    `ch_variadic_keyword_to_keyword=False`.
    If you'll need to use `ch_variadics_to_non_variadic_kind` in such a way
    repeatedly, we suggest you use `functools.partial` to not have to specify this
    configuration repeatedly.

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
    """
    if func is None:
        return partial(
            ch_variadics_to_non_variadic_kind,
            ch_variadic_keyword_to_keyword=ch_variadic_keyword_to_keyword,
        )
    sig = Sig(func)
    idx_of_vp = sig.index_of_var_positional
    var_keyword_argname = sig.var_keyword_name

    if idx_of_vp is not None or var_keyword_argname is not None:
        # If the function has any variadic (position or keyword)...

        @wraps(func)
        def variadic_less_func(*args, **kwargs):
            # extract from kwargs those inputs that need to be expressed positionally
            if ch_variadic_keyword_to_keyword:
                arguments = kwargs
            else:
                arguments = {k: v for k, v in kwargs.items() if k in sig}
                if sig.has_var_keyword:
                    arguments[sig.var_keyword_name] = {
                        k: v for k, v in kwargs.items() if k not in sig
                    }
            _args, _kwargs = sig.mk_args_and_kwargs(arguments, allow_partial=True)
            # print('COUCOU', kwargs, arguments)
            # add these to the existing args
            args = args + _args

            if idx_of_vp is not None:
                # separate the args that are positional, variadic, and after variadic
                a, _vp_args_, args_after_vp = (
                    args[:idx_of_vp],
                    args[idx_of_vp],
                    args[idx_of_vp + 1 :],
                )
                if args_after_vp:
                    raise FuncCallNotMatchingSignature(
                        "There should be only keyword arguments after the Variadic "
                        "args. "
                        f"Function was called with (positional={args}, keywords="
                        f"{_kwargs})"
                    )
            else:
                a, _vp_args_ = args, ()

            # extract from the remaining _kwargs, the dict corresponding to the
            # variadic keywords, if any, since these need to be **-ed later
            _var_keyword_kwargs = _kwargs.pop(var_keyword_argname, {})

            if ch_variadic_keyword_to_keyword:
                # an extra level of extraction is needed in this case
                # _var_keyword_kwargs = _var_keyword_kwargs.pop(var_keyword_argname, {})
                return func(*a, *_vp_args_, **_kwargs, **_var_keyword_kwargs)
            else:
                # call the original function with the unravelled args
                return func(*a, *_vp_args_, **_kwargs, **_var_keyword_kwargs)

        params = sig.params

        if var_keyword_argname:  # if there's a VAR_KEYWORD argument
            if ch_variadic_keyword_to_keyword:
                i = sig.index_of_var_keyword
                # TODO: Reflect on pros/cons of having mutable {} default here:
                params[i] = params[i].replace(kind=Parameter.KEYWORD_ONLY, default={})

        try:  # TODO: Avoid this try catch. Look in advance for default ordering?
            if idx_of_vp is not None:
                params[idx_of_vp] = params[idx_of_vp].replace(kind=PK, default=())
            variadic_less_func.__signature__ = Sig(
                # Note: Changed signature(func) to Sig(func) but don't know if the first
                #  was on purpose.
                params,
                return_annotation=Sig(func).return_annotation,
            )
        except ValueError:
            if idx_of_vp is not None:
                params[idx_of_vp] = params[idx_of_vp].replace(kind=PK)
            variadic_less_func.__signature__ = Sig(
                params, return_annotation=Sig(func).return_annotation
            )

        return variadic_less_func
    else:
        return func


tuple_the_args = partial(
    ch_variadics_to_non_variadic_kind, ch_variadic_keyword_to_keyword=False
)
tuple_the_args.__name__ = "tuple_the_args"
tuple_the_args.__doc__ = """
A decorator that will change a VAR_POSITIONAL (*args) argument to a tuple (args)
argument of the same name.
"""


def ch_func_to_all_pk(func):
    """Returns a decorated function where all arguments are of the PK kind.
    (PK: Positional_or_keyword)

    :param func: A callable
    :return:

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
    """
    # TODO: Make this work (currently raises "non-default argument follows default argument"):
    # >>> def h(x, *y, z):
    # ...     print(f"{x=}, {y=}, {z=}")
    # >>> h(1, 2, 3, z=4)
    # x=1, y=(2, 3), z=4
    # >>> hh = ch_func_to_all_pk(h)
    # >>> hh(1, (2, 3), z=4)
    # x=1, y=(2, 3), z=4

    # _func = tuple_the_args(func)
    # sig = Sig(_func)
    #
    # @wraps(func)
    # def __func(*args, **kwargs):
    #     # b = Sig(_func).bind_partial(*args, **kwargs)
    #     # return _func(*b.args, **b.kwargs)
    #     args, kwargs = Sig(_func).extract_args_and_kwargs(
    #         *args, **kwargs, _ignore_kind=False
    #     )
    #     return _func(*args, **kwargs)
    #
    _func = tuple_the_args(func)
    sig = Sig(_func)

    @wraps(func)
    def __func(*args, **kwargs):
        args, kwargs = Sig(_func).extract_args_and_kwargs(
            *args,
            **kwargs,
            # _ignore_kind=False,
            # _allow_partial=True
        )
        return _func(*args, **kwargs)

    __func.__signature__ = all_pk_signature(sig)
    return __func


def copy_func(f):
    """Copy a function (not sure it works with all types of callables)"""
    g = FunctionType(
        f.__code__,
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    g = update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    if hasattr(f, "__signature__"):
        g.__signature__ = f.__signature__
    return g


# TODO: Similar to other function in this module -- merge.
def params_of(obj: HasParams):
    if isinstance(obj, Signature):
        obj = list(obj.parameters.values())
    elif isinstance(obj, Mapping):
        obj = list(obj.values())
    elif callable(obj):
        obj = list(signature(obj).parameters.values())
    assert all(isinstance(p, Parameter) for p in obj), (
        "obj needs to be a Iterable[Parameter] at this point"
    )
    return obj  # as is


########################################################################################################################
# TODO: Encorporate in Sig
def insert_annotations(s: Signature, /, *, return_annotation=empty, **annotations):
    """Insert annotations in a signature.
    (Note: not really insert but returns a copy of input signature)

    >>> from inspect import signature
    >>> s = signature(lambda a, b, c=1, d="bar": 0)
    >>> s
    <Signature (a, b, c=1, d='bar')>
    >>> ss = insert_annotations(s, b=int, d=str)
    >>> ss
    <Signature (a, b: int, c=1, d: str = 'bar')>
    >>> insert_annotations(s, b=int, d=str, e=list)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    AssertionError: These argument names weren't found in the signature: {'e'}
    """
    assert set(annotations) <= set(s.parameters), (
        f"These argument names weren't found in the signature: "
        f"{set(annotations) - set(s.parameters)}"
    )
    params = dict(s.parameters)
    for name, annotation in annotations.items():
        p = params[name]
        params[name] = Parameter(
            name=name, kind=p.kind, default=p.default, annotation=annotation
        )
    return Signature(params.values(), return_annotation=return_annotation)


def common_and_diff_argnames(func1: callable, func2: callable) -> dict:
    """Get list of argument names that are common to two functions, as well as the two
    lists of names that are different

    Args:
        func1: First function
        func2: Second function

    Returns:
        A dict with fields 'common', 'func1_not_func2', and 'func2_not_func1'

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
    """
    p1 = signature(func1).parameters
    p2 = signature(func2).parameters
    return {
        "common": [x for x in p1 if x in p2],
        "func1_not_func2": [x for x in p1 if x not in p2],
        "func2_not_func1": [x for x in p2 if x not in p1],
    }


dflt_name_for_kind = {
    Parameter.VAR_POSITIONAL: "args",
    Parameter.VAR_KEYWORD: "kwargs",
}

arg_order_for_param_tuple = ("name", "default", "annotation", "kind")


def set_signature_of_func(
    func, parameters, *, return_annotation=empty, __validate_parameters__=True
):
    """Set the signature of a function, with sugar.

    Args:
        func: Function whose signature you want to set
        signature: A list of parameter specifications. This could be an
        inspect.Parameter object or anything that
            the mk_param function can resolve into an inspect.Parameter object.

        return_annotation: Passed on to inspect.Signature.
        __validate_parameters__: Passed on to inspect.Signature.

    Returns:
        None (but sets the signature of the input function)

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
    """
    sig = Sig(
        parameters,
        return_annotation=return_annotation,
        __validate_parameters__=__validate_parameters__,
    )
    func.__signature__ = sig.to_simple_signature()
    # Not returning func so it's clear(er) that the function is transformed in place


# Pattern: (rewiring) wrapper of make_dataclass
# TODO: Is there a clean way for module to be populated by __name__ of caller module?
def sig_to_dataclass(
    sig: SignatureAble, *, cls_name=None, bases=(), module=None, **kwargs
):
    """
    Make a ``class`` (through ``make_dataclass``) from the given signature.

    :param sig: A ``SignatureAble``, that is, anything that ensure_signature can
        resolve into an ``inspect.Signature`` object, including a signature object
        itself, but also most callables, a list or params, etc.
    :param cls_name: The same as ``cls_name`` of ``dataclasses.make_dataclass``
    :param bases: The same as ``bases`` of ``dataclasses.make_dataclass``
    :param module: Set to module (usually ``__name__`` to specify ther module of
        caller) so that the class and instances can be pickle-able.
    :param kwargs: Passed on to ``dataclasses.make_dataclass``
    :return: A dataclass

    >>> def foo(a, /, b : int=2, *, c=3):
    ...     pass
    ...
    >>> K = sig_to_dataclass(foo, cls_name='K')
    >>> str(Sig(K))
    '(a, b: int = 2, c=3) -> None'
    >>> k = K(1,2,3)
    >>> (k.a, k.b, k.c)
    (1, 2, 3)

    Would also work with any of these (and more):

    >>> K = sig_to_dataclass(Sig(foo), cls_name='K')
    >>> K = sig_to_dataclass(Sig(foo).params, cls_name='K')

    Note:
        ``cls_name`` is not required (we'll try to figure out a good default for you),
        but it's advised to only use this convenience in extreme mode.
        Choosing your own name might make for a safer future if you're reusing your class.
    """
    from dataclasses import make_dataclass

    sig = ensure_signature(sig)
    cls_name = cls_name or getattr(sig, "name", "_made_by_sig_to_dataclass")
    params = ensure_params(sig)
    fields = [(p.name, p.annotation, p.default) for p in params]
    cls = make_dataclass(cls_name, fields, bases=bases, **kwargs)
    if module:
        cls.__module__ = module
    return cls


def replace_kwargs_using(sig: SignatureAble):
    """
    Decorator that replaces the variadic keyword argument of the target function using
    the `sig`, the signature of a source function.
    It essentially injects the difference between `sig` and the target function's
    signature into the target function's signature. That is, it replaces the
    variadic keyword argument (a.k.a. "kwargs") with those parameters that are in `sig`
    but not in the target function's signature.

    This is meant to be used when a `targ_func` (the function you'll apply the
    decorator to) has a variadict keyword argument that is just used to forward "extra"
    arguments to another function, and you want to make sure that the signature of the
    `targ_func` is consistent with the `sig` signature.
    (Also, you don't want to copy the signatures around manually.)

    In the following, `sauce` (the target function) has a variadic keyword argument,
    `sauce_kwargs`, that is used to forward extra arguments to `apple` (the source
    function).

    >>> def apple(a, x: int, y=2, *, z=3, **extra_apple_options):
    ...     return a + x + y + z
    >>> @replace_kwargs_using(apple)
    ... def sauce(a, b, c, **sauce_kwargs):
    ...     return b * c + apple(a, **sauce_kwargs)

    The function will works:

    >>> sauce(1, 2, 3, x=4, z=5)  # func still works? Should be: 1 + 4 + 2 + 5 + 2 * 3
    18

    But the signature now doesn't have the `**sauce_kwargs`, but more informative
    signature elements sourced from `apple`:

    >>> Sig(sauce)
    <Sig (a, b, c, *, x: int, y=2, z=3, **extra_apple_options)>

    One thing to note is that the order of the arguments in the signature of `apple`
    may change to accomodate for the python parameter order rules
    (see https://docs.python.org/3/reference/compound_stmts.html#function-definitions).
    The new order will try to conserve the order of the original arguments of `sauce`
    in-so-far as it doesn't violate the python parameter order rules, though.
    See examples below:

    >>> @Sig.replace_kwargs_using(apple)
    ... def sauce(a, b=2, c=3, **sauce_kwargs):
    ...     return b * c + apple(a, **sauce_kwargs)
    >>> Sig(sauce)
    <Sig (a, b=2, c=3, *, x: int, y=2, z=3, **extra_apple_options)>

    >>> @Sig.replace_kwargs_using(apple)
    ... def sauce(a=1, b=2, c=3, **sauce_kwargs):
    ...     return b * c + apple(a, **sauce_kwargs)
    >>> Sig(sauce)
    <Sig (a=1, b=2, c=3, *, x: int, y=2, z=3, **extra_apple_options)>
    """

    def decorator(targ_func):
        targ_func_sig = Sig(targ_func)  # function whose signature we're changing
        if targ_func_sig.has_var_keyword:
            # remove it from the signature of targ_sig (we're replacing it!)
            targ_func_sig = Sig(targ_func)[:-1]
        else:
            # if there is none, we shouldn't be using replace_kwargs_using!
            raise ValueError(
                f"Target function {targ_func} must have a variadict keyword argument"
            )

        src_sig = Sig(sig)  # signature we're using to replace kwargs of targ_func

        # Remove all params of src_sig that are in targ_func_sig
        # This is because if they're used, they will be bound to the non-variadic
        # target arguments, so there's no conflict: the target kind, default,
        # and annotation should be used not the source ones.
        src_sig -= targ_func_sig

        # make all parameters of src_sig keyword-only
        # (they're replacing variadic keywords after all!)
        # All? No -- a variadic keyword in the src_sig should remain so
        names_of_all_params_in_src_sig_that_are_not_variadic_keyword = [
            p.name for p in src_sig.params if p.kind != Parameter.VAR_KEYWORD
        ]
        n = len(names_of_all_params_in_src_sig_that_are_not_variadic_keyword)

        src_sig = src_sig.ch_kinds(
            **dict(
                zip(
                    names_of_all_params_in_src_sig_that_are_not_variadic_keyword,
                    [Parameter.KEYWORD_ONLY] * n,
                )
            )
        )

        new_sig = targ_func_sig.merge_with_sig(src_sig)
        return new_sig(targ_func)

    return decorator


Sig.replace_kwargs_using = replace_kwargs_using
