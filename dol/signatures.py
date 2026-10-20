"""Signature calculus: Tools to make it easier to work with function's signatures.

How to:

    - get names, kinds, defaults, annotations

    - make signatures flexibly

    - merge two or more signatures

    -

    - give a function a specific signature (with a choice of validations)

    - get an equivalent function with a different order of arguments

    - get an equivalent function with a subset of arguments (like partial)

    - get an equivalent function but with variadic ``*args`` and/or ``**kwargs`` replaced with
      non-variadic args (tuple) and kwargs (dict)

    - make an f(a) function in to a f(a, b=None) function with b ignored


Get names, kinds, defaults, annotations:

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

Make signatures flexibly:

>>> Sig(func)
<Sig (z, a: float = 1.0, /, b=2, *, c: int = 3)>
>>> Sig(['a', 'b'])
<Sig (a, b)>
>>> Sig('x y z')
<Sig (x, y, z)>

Merge signatures.

>>> def foo(x): pass
>>> def bar(y: int, *, z=2): pass  # note the * (keyword only) will be lost!
>>> Sig(foo) + ['a', 'b'] + Sig(bar)
<Sig (x, a, b, y: int, z=2)>

Give a function a signature.

>>> @Sig('a b c')
... def func(*args, **kwargs):
...     print(args, kwargs)
>>> Sig(func)
<Sig (a, b, c)>


**Notes to the reader**

Both in the code and in the docs, we'll use short hands for parameter (argument) kind.

    - PK = Parameter.POSITIONAL_OR_KEYWORD

    - VP = Parameter.VAR_POSITIONAL

    - VK = Parameter.VAR_KEYWORD

    - PO = Parameter.POSITIONAL_ONLY

    - KO = Parameter.KEYWORD_ONLY
"""

from inspect import Signature, Parameter, signature, unwrap
import re
import sys
from typing import (
    Union,
    Any,
    Dict,
    Tuple,
    TypeVar,
    Literal,
    Optional,
    get_args,
)
from collections.abc import Callable, Iterable, Iterator, Mapping as MappingType
from typing import KT, VT, T
from types import FunctionType
from collections import defaultdict
from operator import eq, attrgetter

from functools import (
    cached_property,
    update_wrapper,
    partial,
    partialmethod,
    WRAPPER_ASSIGNMENTS,
    wraps as _wraps,
    update_wrapper as _update_wrapper,
)

# Signatures for builtins that ``inspect`` can't read live in the dependency-free
# ``dol.sig_builtins``; re-exported here (i2mint/dol#70).
from dol.sig_builtins import (
    DFLT_SIGNATURE,
    _robust_signature_of_callable,
    resolve_function,
    dict_of_attribute_signatures,
    sigs_for_builtins,
    sigs_for_builtin_modules,
    sigs_for_sigless_builtin_name,
    sigs_for_type_name,
)


def deprecation_of(func, old_name):
    @wraps(func)
    def wrapper(*args, **kwargs):
        from warnings import warn

        warn(
            f"`{old_name}` is deprecated. Use `{func.__module__}.{func.__qualname__}` instead.",
            DeprecationWarning,
        )
        return func(*args, **kwargs)

    return wrapper


# monkey patching WRAPPER_ASSIGNMENTS to get "proper" wrapping (adding defaults and
# kwdefaults

wrapper_assignments = (*WRAPPER_ASSIGNMENTS, "__defaults__", "__kwdefaults__")

update_wrapper = partial(_update_wrapper, assigned=wrapper_assignments)
wraps = partial(_wraps, assigned=wrapper_assignments)

_empty = Parameter.empty
empty = _empty

ParamsType = Iterable[Parameter]
ParamsAble = Union[ParamsType, Signature, MappingType[str, Parameter], Callable, str]
SignatureAble = Union[Signature, ParamsAble]
HasParams = Union[Iterable[Parameter], MappingType[str, Parameter], Signature, Callable]

# short hands for Parameter kinds
PK = Parameter.POSITIONAL_OR_KEYWORD
VP, VK = Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD
PO, KO = Parameter.POSITIONAL_ONLY, Parameter.KEYWORD_ONLY
var_param_kinds = frozenset({VP, VK})
var_param_types = var_param_kinds  # Deprecate: for back-compatibility. Delete in 2021
var_param_kind_dflts_items = tuple({VP: (), VK: {}}.items())

DFLT_DEFAULT_CONFLICT_METHOD = "strict"
SigMergeOptions = Literal[None, "strict", "take_first", "fill_defaults_and_annotations"]

param_attributes = {"name", "kind", "default", "annotation"}


class InvalidSignature(SyntaxError, ValueError):
    """Raise when a signature is not valid"""


class FuncCallNotMatchingSignature(TypeError):
    """Raise when the call signature is not valid"""


class IncompatibleSignatures(ValueError):
    from pprint import pformat

    """Raise when two signatures are not compatible.
    (see https://github.com/i2mint/i2/discussions/76 for more information on signature
    compatibility)"""

    def __init__(self, *args, sig1=None, sig2=None, **kwargs):
        args = list(args or ("",))
        sig_pairs = None
        if sig1 and sig2:
            sig_pairs = SigPair(sig1, sig2)
            # add the signature differences to the error message
            args[0] += (
                f"\n----- Signature differences (not all differences necessarily "
                f"matter in your context): ----- \n{sig_pairs.diff_str()}"
            )
        super().__init__(*args, **kwargs)
        self.sig_pairs = sig_pairs


# TODO: Couldn't make this work. See https://www.python.org/dev/peps/pep-0562/
# deprecated_names = {'assure_callable', 'assure_signature', 'assure_params'}
#
#
# def __getattr__(name):
#     print(name)
#     if name in deprecated_names:
#         from warnings import warn
#         warn(f"{name} is deprecated (see code for new name -- look for aliases)",
#         DeprecationWarning)
#     raise AttributeError(f"module {__name__} has no attribute {name}")


def validate_signature(func: Callable) -> Callable:
    """
    Validates the signature of a function.

    >>> @validate_signature
    ... def has_valid_signature(x=Sig.empty, y=2):
    ...     pass
    >>> # all good, no errors raised
    >>>
    >>> @validate_signature  # doctest: +IGNORE_EXCEPTION_DETAIL
    ... def does_no_have_valid_signature(x=2, y=Sig.empty):
    ...     pass
    Traceback (most recent call last):
    ...
    i2.signatures.InvalidSignature: Invalid signature for function <function does_no_have_valid_signature at 0x106a72a70>: non-default argument follows default a
    rgument
    """
    try:
        Sig(func)  # to get errors if the signature is not valid
    except Exception as e:
        raise InvalidSignature(f"Invalid signature for function {func}: {e}")
    return func  # if all goes well, return the original function


def is_signature_error(e: BaseException) -> bool:
    """Check if an exception is a signature error"""
    return isinstance(InvalidSignature) or (
        isinstance(e, ValueError) and "no signature found" in str(e)
    )


def _param_sort_key(param):
    return (param.kind, param.kind == KO or param.default is not empty)


def sort_params(params):
    """

    :param params: An iterable of `Parameter` instances
    :return: A list of these instances sorted so as to obey the ``kind`` and ``default``
        order rules of python signatures.

    Note 1: It doesn't mean that these params constitute a valid signature together,
    since it doesn't verify rules like unicity of names and variadic kinds.

    Note 2: Though you can use ``sorted`` on an iterable of ``i2.signatures.Param``
    instances, know that even for sorting the three parameters below,
    the ``sort_params`` function is more than twice as fast.

    >>> from inspect import Parameter
    >>> sort_params(
    ...     [Parameter('a', kind=Parameter.POSITIONAL_OR_KEYWORD, default=1),
    ...     Parameter('b', kind=Parameter.POSITIONAL_ONLY),
    ...     Parameter('c', kind=Parameter.POSITIONAL_OR_KEYWORD)]
    ... )
    [<Parameter "b">, <Parameter "c">, <Parameter "a=1">]
    """
    return sorted(params, key=_param_sort_key)


def _return_none(o: object) -> None:
    return None


# (approximately) duplicated from i2.util to keep signatures.py standalone
def name_of_obj(
    o: object,
    *,
    base_name_of_obj: Callable = attrgetter("__name__"),
    caught_exceptions: tuple = (AttributeError,),
    default_factory: Callable = _return_none,
) -> str | None:
    """
    Tries to find the (or "a") name for an object, even if `__name__` doesn't exist.

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

    Note that ``name_of_obj`` uses the ``__name__`` attribute as its base way to get
    a name. You can customize this behavior though.
    For example, see that:

    >>> from inspect import Signature
    >>> name_of_obj(Signature.replace)
    'replace'

    If you want to get the fully qualified name of an object, you can do:

    >>> alt = partial(name_of_obj, base_name_of_obj=attrgetter('__qualname__'))
    >>> alt(Signature.replace)
    'Signature.replace'
    """
    try:
        return base_name_of_obj(o)
    except caught_exceptions:
        kwargs = dict(
            base_name_of_obj=base_name_of_obj,
            caught_exceptions=caught_exceptions,
            default_factory=default_factory,
        )
        if isinstance(o, (cached_property, partial, partialmethod)) and hasattr(
            o, "func"
        ):
            return name_of_obj(o.func, **kwargs)
        elif isinstance(o, property) and hasattr(o, "fget"):
            return name_of_obj(o.fget, **kwargs)
        elif hasattr(o, "__class__"):
            return name_of_obj(type(o), **kwargs)
        elif hasattr(o, "fset"):
            return name_of_obj(o.fset, **kwargs)
        return default_factory(o)


def ensure_callable(obj: SignatureAble):
    if isinstance(obj, Callable):
        return obj
    else:

        def f(*args, **kwargs):
            """Empty function made just to carry a signature"""

        f.__signature__ = ensure_signature(obj)
        return f


assure_callable = ensure_callable  # alias for backcompatibility


def ensure_signature(obj: SignatureAble) -> Signature:
    if isinstance(obj, Signature):
        return obj
    elif isinstance(obj, Callable):
        return _robust_signature_of_callable(obj)
    elif isinstance(obj, Iterable):
        params = ensure_params(obj)
        try:
            return Signature(parameters=params)
        except TypeError:
            raise TypeError(
                f"Don't know how to make that object into a Signature: {obj}"
            )
    elif isinstance(obj, Parameter):
        return Signature(parameters=(obj,))
    elif obj is None:
        return Signature(parameters=())
    # if you get this far...
    raise TypeError(f"Don't know how to make that object into a Signature: {obj}")


assure_signature = ensure_signature  # alias for backcompatibility


def ensure_param(p):
    if isinstance(p, Parameter):
        return p
    elif isinstance(p, dict):
        return Param(**p)
    elif isinstance(p, str):
        return Param(name=p)
    elif isinstance(p, Iterable):
        name, *r = p
        dflt_and_annotation = dict(zip(["default", "annotation"], r))
        return Param(name, PK, **dflt_and_annotation)
    else:
        raise TypeError(f"Don't know how to make {p} into a Parameter object")


def _params_from_mapping(mapping: MappingType):
    def gen():
        for k, v in mapping.items():
            if isinstance(v, MappingType):
                if "name" in v:
                    assert v["name"] == k, (
                        f"In a mapping specification of a params, "
                        f"either the 'name' of the val shouldn't be specified, "
                        f"or it should be the same as the key ({k}): "
                        f"{dict(mapping)}"
                    )
                    yield v
                else:
                    yield dict(name=k, **v)
            else:
                assert isinstance(v, Parameter) and v.name == k, (
                    f"In a mapping specification of a params, "
                    f"either the val should be a Parameter with the same name as the "
                    f"key ({k}), or it should be a mapping with a 'name' key "
                    f"with the same value as the key: {dict(mapping)}"
                )
                yield v

    return list(gen())


def _add_optional_keywords(sig, kwarg_and_defaults, kwarg_annotations=None):
    """
    Enhances a given signature with additional optional keyword-only arguments.

    Args:
        sig (Signature): The original function signature.
        kwarg_and_defaults (dict): A dictionary of keyword arguments and their default values.
        kwarg_annotations (dict, optional): A dictionary of keyword arguments and their type annotations.

    Returns:
        Signature: The enhanced function signature with additional keyword-only arguments.

    >>> from inspect import signature
    >>> def example_func(x, y): pass
    >>> original_sig = signature(example_func)
    >>> enhanced_sig = _add_optional_keywords(
    ...     original_sig, {'z': 3, 'verbose': False}, {'verbose': bool}
    ... )
    >>> str(enhanced_sig)
    '(x, y, *, z=3, verbose: bool = False)'

    Note:

        - Annotations for the additional keywords are optional.
        - All additional keywords are added as keyword-only arguments.
    """
    if isinstance(sig, Signature):
        sig = Sig(sig).merge_with_sig(
            Sig.from_objs(**kwarg_and_defaults), ch_to_all_pk=False
        )
        sig = sig.ch_kinds(**{k: Sig.KEYWORD_ONLY for k in kwarg_and_defaults})

        kwarg_annotations = kwarg_annotations or {}
        assert all(name in kwarg_and_defaults for name in kwarg_annotations), (
            "Some annotations were given for arguments that were not in kwarg_and_defaults:"
            f"\n{kwarg_and_defaults=}\n{kwarg_annotations=}"
        )
        sig = sig.ch_annotations(**kwarg_annotations)
        return sig
    else:
        func = sig  # assume it's a function
        # apply _add_optional_keywords to that function
        sig = Sig(func)
        sig = _add_optional_keywords(sig, kwarg_and_defaults, kwarg_annotations)
        # and inject the new signature into the function
        return sig(func)


def ensure_params(obj: ParamsAble = None):
    """Get an interable of Parameter instances from an object.

    :param obj:
    :return:

    From a callable:

    >>> def f(w, /, x: float = 1, y=1, *, z: int = 1):
    ...     ...
    >>> ensure_params(f)
    [<Parameter "w">, <Parameter "x: float = 1">, <Parameter "y=1">, <Parameter "z: int = 1">]

    From an iterable of strings, dicts, or tuples

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


    If no input is given, an empty list is returned.

    >>> ensure_params()  # equivalent to ensure_params(None)
    []
    """
    # obj = inspect.unwrap(obj, stop=(lambda f: hasattr(f, "__signature__")))

    if obj is None:
        return []
    elif isinstance(obj, Signature):
        return list(obj.parameters.values())
    try:  # to get params from the builtin signature function
        return list(_robust_signature_of_callable(obj).parameters.values())
    except (TypeError, ValueError):
        if isinstance(obj, Iterable):
            if isinstance(obj, str):
                obj = [obj]
            # TODO: Can do better here! See attempt in _params_from_mapping:
            elif isinstance(obj, Mapping):
                obj = _params_from_mapping(obj)
                # obj = list(obj.values())
            else:
                obj = list(obj)
            if len(obj) == 0:
                return obj
            else:
                # TODO: put this in function that has more kind resolution power
                #  e.g. if a KEYWORD_ONLY arg was encountered, all subsequent
                #  have to be unless otherwise specified.
                return [ensure_param(p) for p in obj]
        else:
            if isinstance(obj, Parameter):
                obj = Signature([obj])
            elif isinstance(obj, Callable):
                obj = _robust_signature_of_callable(obj)
            elif obj is None:
                obj = {}
            if isinstance(obj, Signature):
                return list(obj.parameters.values())
        # if nothing above worked, perhaps you have a wrapped object? Try unwrapping until
        # you find a signature...
        if hasattr(obj, "__wrapped__"):
            obj = unwrap(obj, stop=(lambda f: hasattr(f, "__signature__")))
            return ensure_params(obj)
        else:  # if function didn't return at this point, it didn't find a match, so raise
            # a TypeError
            raise TypeError(
                f"Don't know how to make that object into an iterable of inspect.Parameter "
                f"objects: {obj}"
            )


assure_params = ensure_params  # alias for backcompatibility


class MissingArgValFor:
    """A simple class to wrap an argument name, indicating that it was missing somewhere.

    >>> MissingArgValFor("argname")
    MissingArgValFor("argname")
    """

    def __init__(self, argname: str):
        assert isinstance(argname, str)
        self.argname = argname

    def __repr__(self):
        return f'MissingArgValFor("{self.argname}")'


# TODO: Look into the handling of the Parameter.VAR_KEYWORD kind in params
def extract_arguments(
    params: ParamsAble,
    *,
    what_to_do_with_remainding="return",
    include_all_when_var_keywords_in_params=False,
    assert_no_missing_position_only_args=False,
    **kwargs,
):
    """Extract arguments needed to satisfy the params of a callable, dealing with the
    dirty details.

    Returns an (param_args, param_kwargs, remaining_kwargs) tuple where

    - param_args are the values of kwargs that are PO (POSITION_ONLY) as defined by
      params,
    - param_kwargs are those names that are both in params and not in param_args, and
    - remaining_kwargs are the remaining.

    Intended usage: When you need to call a function `func` that has some
    position-only arguments,
    but you have a kwargs dict of arguments in your hand. You can't just to `func(
    **kwargs)`.
    But you can (now) do

    .. code-block:: python

        # extract from kwargs what you need for func
        args, kwargs, remaining = extract_arguments(kwargs, func)
        # ... check if remaining is empty (or not, depending on your paranoia),
        # and then call the func:
        func(*args, **kwargs)

    (And if you doing that a lot: Do put it in a decorator!)

    See Also:
        extract_arguments.without_remainding

    The most frequent case you'll encounter is when there's no POSITION_ONLY args,
    your param_args will be empty
    and you param_kwargs will contain all the arguments that match params,
    in the order of these params.

    >>> from inspect import signature
    >>> def f(a, b, c=None, d=0):
    ...     ...
    ...
    >>> extract_arguments(f, b=2, a=1, c=3, d=4, extra="stuff")
    ((), {'a': 1, 'b': 2, 'c': 3, 'd': 4}, {'extra': 'stuff'})

    But sometimes you do have POSITION_ONLY arguments.
    What extract_arguments will do for you is return the value of these as the first
    element of
    the triple.

    >>> def f(a, b, c=None, /, d=0):
    ...     ...
    ...
    >>> extract_arguments(f, b=2, a=1, c=3, d=4, extra="stuff")
    ((1, 2, 3), {'d': 4}, {'extra': 'stuff'})

    Note above how we get `(1, 2, 3)`, the order defined by the func's signature,
    instead of `(2, 1, 3)`, the order defined by the kwargs.
    So it's the params (e.g. function signature) that determine the order, not kwargs.
    When using to call a function, this is especially crucial if we use POSITION_ONLY
    arguments.

    See also that the third output, the remaining_kwargs, as `{'extra': 'stuff'}` since
    it was not in the params of the function.
    Even if you include a VAR_KEYWORD kind of argument in the function, it won't change
    this behavior.

    >>> def f(a, b, c=None, /, d=0, **kws):
    ...     ...
    ...
    >>> extract_arguments(f, b=2, a=1, c=3, d=4, extra="stuff")
    ((1, 2, 3), {'d': 4}, {'extra': 'stuff'})

    This is because we don't want to assume that all the kwargs can actually be
    included in a call to the function behind the params.
    Instead, the user can chose whether to include the remainder by doing a:

    .. code-block:: python

        param_kwargs.update(remaining_kwargs)

    et voilà.

    That said, we do understand that it may be a common pattern, so we'll do that
    extra step for you
    if you specify `include_all_when_var_keywords_in_params=True`.

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

    If you're expecting no remainder you might want to just get the args and kwargs (
    not this third
    expected-to-be-empty remainder). You have two ways to do that, specifying:

    - `what_to_do_with_remainding='ignore'`, which will just return the (args,
      kwargs) pair
    - `what_to_do_with_remainding='assert_empty'`, which will do the same, but first
      assert the remainder is empty

    We suggest to use `functools.partial` to configure the `argument_argument` you need.

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

    And what happens if the kwargs doesn't contain all the POSITION_ONLY arguments?

    >>> def f(a, b, c=None, /, d=0):
    ...     ...
    ...
    >>> extract_arguments(f, b=2, d="is a kw arg", e="is not an arg at all")
    ((MissingArgValFor("a"), 2, MissingArgValFor("c")), {'d': 'is a kw arg'}, {'e': 'is not an arg at all'})


    A few more examples...

    Let's call `extract_arguments` with params being not a function,
    but, a Signature instance, a mapping whose values are Parameter instances,
    or an iterable of Parameter instances...

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

    Edge case:
    No params specified? No problem. You'll just get empty args and kwargs. Everything
    in the remainder

    >>> extract_arguments(params=(), b=2, a=1, c=3, d=0)
    ((), {}, {'b': 2, 'a': 1, 'c': 3, 'd': 0})

    :param params: Specifies what PO arguments should be extracted.
        Could be a callable, Signature, iterable of Parameters...
    :param what_to_do_with_remainding:
        'return' (default): function will return `param_args`, `param_kwargs`,
        `remaining_kwargs`
        'ignore': function will return `param_args`, `param_kwargs`
        'assert_empty': function will assert that `remaining_kwargs` is empty and then
        return `param_args`, `param_kwargs`
    :param include_all_when_var_keywords_in_params: If True and `params` has a
        VAR_KEYWORD parameter, the remaining kwargs are merged into `param_kwargs`
        (leaving an empty remainder).
    :param assert_no_missing_position_only_args: If True, assert that no
        position-only argument is missing from `kwargs`.
    :param kwargs: The kwargs to extract the args from
    :return: A (param_args, param_kwargs, remaining_kwargs) tuple.
    """

    assert what_to_do_with_remainding in {"return", "ignore", "assert_empty"}
    assert isinstance(include_all_when_var_keywords_in_params, bool)
    assert isinstance(assert_no_missing_position_only_args, bool)

    params = ensure_params(params)
    if not params:
        return (), {}, {k: v for k, v in kwargs.items()}

    params_names = tuple(p.name for p in params)
    names_for_args = [p.name for p in params if p.kind == Parameter.POSITIONAL_ONLY]
    param_kwargs_names = [x for x in params_names if x not in set(names_for_args)]
    remaining_names = [x for x in kwargs if x not in params_names]

    param_args = tuple(kwargs.get(k, MissingArgValFor(k)) for k in names_for_args)
    param_kwargs = {k: kwargs[k] for k in param_kwargs_names if k in kwargs}
    remaining_kwargs = {k: kwargs[k] for k in remaining_names}

    if include_all_when_var_keywords_in_params:
        if (
            next(
                (p.name for p in params if p.kind == Parameter.VAR_KEYWORD),
                None,
            )
            is not None
        ):
            param_kwargs.update(remaining_kwargs)
            remaining_kwargs = {}

    if assert_no_missing_position_only_args:
        missing_argnames = tuple(
            x.argname for x in param_args if isinstance(x, MissingArgValFor)
        )
        assert not missing_argnames, (
            f"There were some missing positional only argnames: {missing_argnames}"
        )

    if what_to_do_with_remainding == "return":
        return param_args, param_kwargs, remaining_kwargs
    elif what_to_do_with_remainding == "ignore":
        return param_args, param_kwargs
    elif what_to_do_with_remainding == "assert_empty":
        assert len(remaining_kwargs) == 0, (
            f"remaining_kwargs not empty: remaining_kwargs={remaining_kwargs}"
        )
        return param_args, param_kwargs


extract_arguments_ignoring_remainder = partial(
    extract_arguments, what_to_do_with_remainding="ignore"
)
extract_arguments_asserting_no_remainder = partial(
    extract_arguments, what_to_do_with_remainding="assert_empty"
)

from collections.abc import Mapping
from typing import Optional
from collections.abc import Iterable


def function_caller(func, args, kwargs):
    return func(*args, **kwargs)


class Param(Parameter):
    """A thin wrap of Parameters: Adds shorter aliases to argument kinds and
    a POSITIONAL_OR_KEYWORD default to the argument kind to make it faster to make
    Parameter objects

    >>> list(map(Param, 'some quick arg params'.split()))
    [<Param "some">, <Param "quick">, <Param "arg">, <Param "params">]
    >>> from inspect import Signature
    >>> P = Param
    >>> Signature([P('x', P.PO), P('y', default=42, annotation=int), P('kw', P.KO)])
    <Signature (x, /, y: int = 42, *, kw)>
    """

    # aliases
    PK = Parameter.POSITIONAL_OR_KEYWORD
    PO = Parameter.POSITIONAL_ONLY
    KO = Parameter.KEYWORD_ONLY
    VP = Parameter.VAR_POSITIONAL
    VK = Parameter.VAR_KEYWORD

    def __init__(self, name, kind=PK, *, default=empty, annotation=empty):
        super().__init__(name, kind, default=default, annotation=annotation)

    def __lt__(self, other) -> bool:
        """Whether the self parameter can be before the other parameter in a signature.

        >>> Param('b') < Param('a', default=1)
        True
        >>> Param('b') > Param('a', default=1)
        False
        >>> Param('b', kind=Param.POSITIONAL_OR_KEYWORD) < Param('a', kind=Param.KEYWORD_ONLY)
        True
        >>> Param('b', kind=Param.POSITIONAL_OR_KEYWORD) > Param('a', kind=Param.KEYWORD_ONLY)
        False

        Note 1: The dual ``>`` operator is also infered.

        Note 2: This means that you can used ``sorted`` on an iterable of Param
        instances, but know that even for sorting the three parameters below,
        the ``sort_params`` function in the ``i2.signatures`` module is more than twice
        as fast.

        >>> sorted(
        ...     [Param('a', default=1),
        ...     Param('b', kind=Param.POSITIONAL_ONLY),
        ...     Param('c')]
        ... )
        [<Param "b">, <Param "c">, <Param "a=1">]
        """
        return (self.kind, self.default is not empty) < (
            other.kind,
            other.default is not empty,
        )


P = Param  # useful shorthand alias


def param_has_default_or_is_var_kind(p: Parameter):
    return p.default is not p.empty or p.kind in var_param_kinds


def parameter_to_dict(p: Parameter) -> dict:
    return dict(name=p.name, kind=p.kind, default=p.default, annotation=p.annotation)


WRAPPER_UPDATES = ("__dict__",)


def _names_of_kind(sig):
    """Compute a tuple containing tuples of names for each kind

    >>> f = lambda a00, /, a11, a12, *a23, a34, a35, a36, **a47: None
    >>> _names_of_kind(Sig(f))
    (('a00',), ('a11', 'a12'), ('a23',), ('a34', 'a35', 'a36'), ('a47',))
    """
    d = defaultdict(list)
    for param in sig.params:
        d[param.kind].append(param.name)
    return tuple(tuple(d[kind]) for kind in range(5))


def maybe_first(items):
    return next(iter(items), None)


def name_of_var_kw_argument(sig):
    var_kw_list = [param.name for param in sig.params if param.kind == VK]
    result = maybe_first(var_kw_list)
    return result


def _map_action_on_cond(kvs, cond, expand):
    for k, v in kvs:
        if cond(
            k
        ):  # make a conditional on (k,v), use type KV, Iterable[KV], expand:KV -> Iterable[KV]
            yield from expand(v[k])  # expand should result in (k,v)
        else:
            yield k, v


def expand_nested_key(d, k):
    for key in d:
        if key == k and isinstance(d[k], dict) and k in d[k]:
            pass

    if k in d and len(d) >= 2:
        return d.items()

    if k in d and isinstance(d[k], dict) and k in d[k]:
        if len(d[k]) == 1:
            return expand_nested_key(d[k], k)
        else:
            return d[k].items()
    else:
        return d.items()


def flatten_if_var_kw(kvs, var_kw_name):
    cond = lambda k: k == var_kw_name
    expand = lambda k: k.items()
    # expand = lambda k: k.values()
    return _map_action_on_cond(kvs, cond, expand)


# TODO: See other signature operating functions below in this module:
#   Do we need them now that we have Sig?
#   Do we want to keep them and have Sig use them?
class Sig(Signature, Mapping):
    """A subclass of inspect.Signature that has a lot of extra api sugar,
    such as

    - making a signature for a variety of input types (callable,
      iterable of callables, parameter lists, strings, etc.)
    - has a dict-like interface
    - signature merging (with operator interfaces)
    - quick access to signature data
    - positional/keyword argument mapping.

    .. rubric:: Positional/Keyword argument mapping

    In python, arguments can be positional (args) or keyword (kwargs).
    ... sometimes both, sometimes a single one is imposed.
    ... and you have variadic versions of both.
    ... and you can have defaults or not.
    ... and all these different kinds have a particular order they must be in.
    It's is mess really. The flexibility is nice -- but still; a mess.

    You only really feel the mess if you try to do some meta-programming with your
    functions.
    Then, methods like `normalize_kind` can help you out, since you can enforce, and
    then assume, some stable interface to your functions.

    Two of the base methods for dealing with positional (args) and keyword (kwargs)
    inputs are:

        - `map_arguments`: Map some args/kwargs input to a keyword-only
          expression of the inputs. This is useful if you need to do some processing
          based on the argument names.
        - `mk_args_and_kwargs`: Translate a fully keyword expression of some
          inputs into an (args, kwargs) pair that can be used to call the function.
          (Remember, your function can have constraints, so you may need to do this.

    The usual pattern of use of these methods is to use `map_arguments`
    to map all the inputs to their corresponding name, do what needs to be done with
    that (example, validation, transformation, decoration...) and then map back to an
    (args, kwargs) pair than can actually be used to call the function.

    Examples of methods and functions using these:
    `call_forgivingly`, `tuple_the_args`, `map_arguments_from_variadics`, `extract_args_and_kwargs`,
    `source_arguments`, and `source_args_and_kwargs`.

    .. rubric:: Making a signature

    You can construct a `Sig` object from a callable,

    >>> def f(w, /, x: float = 1, y=1, *, z: int = 1):
    ...     ...
    >>> Sig(f)
    <Sig (w, /, x: float = 1, y=1, *, z: int = 1)>

    but also from any "ParamsAble" object. Such as...
    an iterable of Parameter instances, strings, tuples, or dicts:

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

    The parameters of a signature are like a matrix whose rows are the parameters,
    and the 4 columns are their properties: name, kind, default, and annotation
    (the two laste ones being optional).
    You get a row view when doing `Sig(...).parameters.values()`,
    but what if you want a column-view?
    Here's how:

    >>> def f(w, /, x: float = 1, y=2, *, z: int = 3):
    ...     ...
    >>>
    >>> s = Sig(f)
    >>> s.kinds  # doctest: +NORMALIZE_WHITESPACE
    {'w': <_ParameterKind.POSITIONAL_ONLY: 0>,
    'x': <_ParameterKind.POSITIONAL_OR_KEYWORD: 1>,
    'y': <_ParameterKind.POSITIONAL_OR_KEYWORD: 1>,
    'z': <_ParameterKind.KEYWORD_ONLY: 3>}

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

    We can sum (i.e. merge) and subtract (i.e. remove arguments) Sig instances.
    Also, Sig instance is callable. It has the effect of inserting it's signature in
    the input
    (in `__signature__`, but also inserting the resulting `__defaults__` and
    `__kwdefaults__`).
    One of the intents is to be able to do things like:

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
    """

    # Adding parameter kinds as class attributes for usage convenience
    POSITIONAL_ONLY = Parameter.POSITIONAL_ONLY
    POSITIONAL_OR_KEYWORD = Parameter.POSITIONAL_OR_KEYWORD
    VAR_POSITIONAL = Parameter.VAR_POSITIONAL
    KEYWORD_ONLY = Parameter.KEYWORD_ONLY
    VAR_KEYWORD = Parameter.VAR_KEYWORD

    def __init__(
        self,
        obj: ParamsAble = None,
        *,
        name=None,
        return_annotation=empty,
        __validate_parameters__=True,
    ):
        """Initialize a Sig instance.

        See Also:
            `ensure_params` to see what kind of objects you can make `Sig`s with.

        :param obj: A ParamsAble object, which could be:

            - a callable,
            - and iterable of Parameter instances
            - an iterable of strings (representing annotation-less, default-less)
              argument names,
            - tuples: (argname, default) or (argname, default, annotation),
            - dicts: ``{'name': REQUIRED,...}`` with optional `kind`, `default` and
              `annotation` fields
            - None (which will produce an argument-less Signature)

        >>> Sig(["a", "b", "c"])
        <Sig (a, b, c)>
        >>> Sig(
        ...     ["a", ("b", None), ("c", 42, int)]
        ... )  # specifying defaults and annotations
        <Sig (a, b=None, c: int = 42)>
        >>> import inspect
        >>> Sig(
        ...     ["a", ("b", inspect._empty, int)]
        ... )  # specifying an annotation without a default
        <Sig (a, b: int)>
        >>> Sig(["a", "b", "c"], return_annotation=str)  # specifying return annotation
        <Sig (a, b, c) -> str>
        >>> Sig('(a: int = 0, b: str = None, c: float = 3.14) -> str')
        <Sig (a: int = 0, b: str = None, c: float = 3.14) -> str>

        But you can always specify parameters the "long" way

        >>> Sig(
        ...     [inspect.Parameter(name="kws", kind=inspect.Parameter.VAR_KEYWORD)],
        ...     return_annotation=str,
        ... )
        <Sig (**kws) -> str>

        And note that:

        >>> Sig()
        <Sig ()>
        >>> Sig(None)
        <Sig ()>
        """
        if isinstance(obj, str):
            if re.match(r"^\(.*\)", obj):
                # This is a string representation of a signature
                # Dynamically create a function with the given signature then generate
                # the Sig object from this function.
                exec_env = dict()
                f_def = f"def f{obj}: pass"
                exec(f_def, exec_env)
                obj = exec_env["f"]
            else:
                obj = obj.split()

        if isinstance(obj, property):
            obj = obj.fget
        elif isinstance(obj, cached_property):
            obj = obj.func

        if (
            not isinstance(obj, Signature)
            and callable(obj)
            and return_annotation is empty
        ):
            return_annotation = _robust_signature_of_callable(obj).return_annotation
        # TODO: Catch errors and enhance error message with more what-to-do-about it
        #  message. For example,
        #  ValueError: wrong parameter order: positional or keyword parameter before
        #  positional-only parameter
        #  --> Here we could tell the user what pair of variables violated the rule
        super().__init__(
            ensure_params(obj),
            return_annotation=return_annotation,
            __validate_parameters__=__validate_parameters__,
        )
        self.names_of_kind = _names_of_kind(self)

        if len(self.names_of_kind[Parameter.VAR_POSITIONAL]) > 1:
            vps = self.names_of_kind[Parameter.VAR_POSITIONAL]
            raise InvalidSignature(f"You can't have several variadic keywords: {vps}")
        if len(self.names_of_kind[Parameter.VAR_KEYWORD]) > 1:
            vks = self.names_of_kind[Parameter.VAR_KEYWORD]
            raise InvalidSignature(f"You can't have several variadic keywords: {vks}")

        self.name = name or name_of_obj(obj)

    # TODO: Add params for more validation (e.g. arg number/name matching?)
    # TODO: Switch to ignore_incompatible_signatures=False when existing code is
    #   changed accordingly.
    def wrap(
        self,
        func: Callable,
        ignore_incompatible_signatures: bool = True,
        *,
        copy_function: bool | Callable = False,
    ):
        """Gives the input function the signature.

        This is similar to the `functools.wraps` function, but parametrized by a
        signature
        (not a callable). Also, where as both write to the input func's `__signature__`
        attribute, here we also write to

        - `__defaults__` and `__kwdefaults__`, extracting these from `__signature__`
          (functools.wraps doesn't do that at the time of writing this
          (see https://github.com/python/cpython/pull/21379)).
        - `__annotations__` (also extracted from `__signature__`)
        - does not write to `__module__`, `__name__`, `__qualname__`, `__doc__`
          (because again, we're basinig the injecton on a signature, not a function,
          so we have no name, doc, etc...)

        WARNING:
            The fact that you've modified the signature of your function doesn't
            mean that the decorated function will work as expected (or even work at all).
            See below for examples.

        >>> def f(w, /, x: float = 1, y=2, z: int = 3):
        ...     return w + x * y ** z
        >>> f(0, 1)  # 0 + 1 * 2 ** 3
        8
        >>> f.__defaults__
        (1, 2, 3)
        >>> assert 8 == f(0) == f(0, 1) == f(0, 1, 2) == f(0, 1, 2, 3)

        Now let's create a very similar function to f, but where:

        - w is not position-only
        - x annot is int instead of float, and doesn't have a default
        - z's default changes to 10

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

        Remember that you are modifying the signature, not the function itself.
        Signature changes in defaults will indeed change the function's behavior.
        But changes in name or kind will only be reflected in the signature, and
        misalignment with the wrapped function will lead to unexpected results.

        >>> def f(w, /, x: float = 1, y=2, *, z: int = 3):
        ...     return w + x * y ** z
        >>> f(0)  # 0 + 1 * 2 ** 3
        8
        >>> f(0, 1, 2, 3)  # error expected!
        Traceback (most recent call last):
          ...
        TypeError: f() takes from 1 to 3 positional arguments but 4 were given

        But if you try to remove the argument kind constraint by just changing the
        signature, you'll fail.

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
        """
        # TODO: Give more explanations why this is.

        # TODO: Should we make copy_function=False the default,
        #  so as to not override decorated function itself by default?
        if copy_function:
            if isinstance(copy_function, bool):
                from i2.util import copy_func as copy_function
            else:
                assert callable(copy_function), (
                    f"copy_function must be a callable. This is not: {copy_function}"
                )
            func = copy_function(func)

        # Analyze self and func signature to validate sanity
        _validate_sanity_of_signature_change(func, self, ignore_incompatible_signatures)

        # Change (mutate!) func, writing a new __signature__, __annotations__,
        # __defaults__ and __kwdefaults__
        func.__signature__ = Sig(
            self.parameters.values(), return_annotation=self.return_annotation
        )
        func.__annotations__ = self.annotations
        func.__defaults__, func.__kwdefaults__ = self._dunder_defaults_and_kwdefaults()

        # special case of functools.partial: need to tell .keywords about kwdefaults
        if isinstance(func, partial):
            # TODO: .args can't be modified -- write test to see if problem.
            #   If it is, consider returning a new partial with updated args & keywords.
            # wrapped_func.args = wrapped_func.args + wrapped_func.__defaults__
            func.keywords.update(func.__kwdefaults__)

        return func

    def __call__(self, func: Callable):
        """Gives the input function the signature.
        Just calls Sig.wrap so see docs of Sig.wrap (which contains examples and
        doctests).
        """
        return self.wrap(func)

    @classmethod
    def sig_or_default(cls, obj, default_signature=DFLT_SIGNATURE):
        """Returns a Sig instance, or a default signature if there was a ValueError
        trying to construct it.

        For example, `time.time` doesn't have a signature

        >>> import time
        >>> has_signature(time.time)
        False

        But we can tell `Sig` to give it the default one:

        >>> str(Sig.sig_or_default(time.time))
        '(*no_sig_args, **no_sig_kwargs)'

        That's the default signature, which should work for most purposes.
        You can also specify what the default should be though.

        >>> fake_signature = Sig(lambda *time_takes_no_arguments: ...)
        >>> str(Sig.sig_or_default(time.time, fake_signature))
        '(*time_takes_no_arguments)'

        Careful though. If you assign a signature to a function that is not aligned
        with that actually functioning of the function, bad things will happen.
        In this case, the actual signature of time is the empty signature:

        >>> str(Sig.sig_or_default(time.time, Sig(lambda: ...)))
        '()'
        """
        try:
            # (try to) return cls(obj) if obj is callable:
            if callable(obj):
                sig = cls(obj)
                # Check if we got our default signature (which means no real signature exists)
                if str(sig) == str(DFLT_SIGNATURE):
                    return Sig(default_signature)
                return sig
            else:
                raise TypeError(f"Object is not callable: {obj}")
        except ValueError:
            # if a ValueError is raised, return the default_signature
            return Sig(default_signature)

    @classmethod
    def sig_or_none(cls, obj):
        """Returns a Sig instance, or None if there was a ValueError trying to
        construct it.
        One use case is to be able to tell if an object has a signature or not.

        >>> robust_has_signature = lambda obj: bool(Sig.sig_or_none(obj))
        >>> robust_has_signature(robust_has_signature)  # an easy case
        True
        >>> robust_has_signature(
        ...     Sig
        ... )  # another easy one: This time, a type/class (which is callable, yes)
        True

        But here's where it get's interesting. `print`, a builtin, doesn't have a
        signature through inspect.signature.

        >>> has_signature(print)  # doctest: +SKIP
        False

        But we do get one with robust_has_signature

        >>> robust_has_signature(print)
        True
        """
        return cls.sig_or_default(obj, default_signature=None)

    def __bool__(self):
        return True

    def _positional_and_keyword_defaults(self):
        """Get ``{name: default, ...}`` dicts of positional and keyword defaults.

        >>> def foo(w, /, x: float, y=1, *, z: int = 1):
        ...     ...
        >>> pos_defaults, kw_defaults = Sig(foo)._positional_and_keyword_defaults()
        >>> pos_defaults
        {'y': 1}
        >>> kw_defaults
        {'z': 1}
        """
        ko_names = self.names_of_kind[KO]
        dflts = self.defaults
        return (
            {name: dflts[name] for name in dflts if name not in ko_names},
            {name: dflts[name] for name in dflts if name in ko_names},
        )

    def _dunder_defaults_and_kwdefaults(self):
        """Get the __defaults__, __kwdefaults__ (i.e. what would be the dunders baring
        these names in a python callable)

        >>> def foo(w, /, x: float, y=1, *, z: int = 1):
        ...     ...
        >>> __defaults__, __kwdefaults__ = Sig(foo)._dunder_defaults_and_kwdefaults()
        >>> __defaults__
        (1,)
        >>> __kwdefaults__
        {'z': 1}
        """

        pos_defaults, kw_defaults = self._positional_and_keyword_defaults()
        return (
            tuple(
                pos_defaults.values()
            ),  # as known as __defaults__ in python callables
            kw_defaults,  # as known as __kwdefaults__ in python callables
        )

    def to_signature_kwargs(self):
        """The dict of keyword arguments to make this signature instance.

        >>> def f(w, /, x: float = 2, y=1, *, z: int = 0) -> float:
        ...     ...
        >>> Sig(f).to_signature_kwargs()  # doctest: +NORMALIZE_WHITESPACE
        {'parameters':
            [<Parameter "w">,
            <Parameter "x: float = 2">,
            <Parameter "y=1">,
            <Parameter "z: int = 0">],
        'return_annotation': <class 'float'>}

        Note that this does NOT return:

        .. code-block:: python

            {'parameters': self.parameters,
            'return_annotation': self.return_annotation}

        which would not actually work as keyword arguments of ``Signature``.
        Yeah, I know. Don't ask me, ask the authors of `Signature`!

        Instead, `parammeters` will be ``list(self.parameters.values())``, which does
        work.
        """
        return {
            "parameters": list(self.parameters.values()),
            "return_annotation": self.return_annotation,
        }

    def to_simple_signature(self):
        """A builtin ``inspect.Signature`` instance equivalent (i.e. without the extra
        properties and methods)

        >>> def f(w, /, x: float = 2, y=1, *, z: int = 0):
        ...     ...
        >>> Sig(f).to_simple_signature()
        <Signature (w, /, x: float = 2, y=1, *, z: int = 0)>
        """
        return Signature(**self.to_signature_kwargs())

    def pair_with(self, other_sig) -> "SigPair":
        """Get an object that pairs with another signature for comparison, merging, etc.

        See `SigPair` for more details.
        """
        return SigPair(self, other_sig)

    def is_call_compatible_with(self, other_sig, *, param_comparator: Callable = None):
        """Return True if the signature is compatible with ``other_sig``. Meaning that
        all valid ways to call the signature are valid for ``other_sig``.
        """
        return is_call_compatible_with(
            self, other_sig, param_comparator=param_comparator
        )

    # TODO: Make these dunders open/close
    # def __le__(self, other_sig):
    #     """The "less than or equal" operator (<=).
    #     Return True if the signature is compatible with ``other_sig``. Meaning that
    #     all valid ways to call the signature are valid for ``other_sig``.
    #     """
    #     return self.is_call_compatible_with(other_sig)

    # def __ge__(self, other_sig):
    #     """The "greater than or equal" operator (>=).
    #     Return True if ``other_sig`` is compatible with the signature. Meaning that
    #     all valid ways to call ``other_sig`` are valid for the signature.
    #     """
    #     return other_sig <= self

    @classmethod
    def from_objs(
        cls,
        *objs,
        default_conflict_method: str = DFLT_DEFAULT_CONFLICT_METHOD,
        return_annotation=empty,
        **name_and_dflts,
    ):
        objs = list(objs)
        for name, default in name_and_dflts.items():
            objs.append([{"name": name, "kind": PK, "default": default}])
        if len(objs) > 0:
            first_obj, *objs = objs
            sig = cls(ensure_params(first_obj))
            for obj in objs:
                sig = sig.merge_with_sig(
                    obj, default_conflict_method=default_conflict_method
                )
                # sig = sig + obj
            return Sig(sig, return_annotation=return_annotation)
        else:  # if no objs are given
            return cls(return_annotation=return_annotation)  # return an empty signature

    @classmethod
    def from_params(cls, params):
        if isinstance(params, Parameter):
            params = (params,)
        return cls(params)

    @property
    def params(self):
        """Just list(self.parameters.values()), because that's often what we want.
        Why a Sig.params property when we already have a Sig.parameters property?

        Well, as much as is boggles my mind, it so happens that the Signature.parameters
        is a name->Parameter mapping, but the Signature argument `parameters`,
        though baring the same name,
        is expected to be a list of Parameter instances.

        So Sig.params is there to restore semantic consistence sanity.
        """
        return list(self.parameters.values())

    @property
    def names(self):
        return list(self.keys())

    @property
    def kinds(self):
        return {p.name: p.kind for p in self.values()}

    @property
    def defaults(self):
        """A ``{name: default,...}`` dict of defaults (regardless of kind)"""
        return {p.name: p.default for p in self.values() if p.default is not p.empty}

    @property
    def _defaults_(self):
        """What the ``__defaults__`` value would be for a func of the same signature"""
        return tuple(
            p.default
            for p in self.values()
            if (p.default is not p.empty and p.kind != KO)
        )

    @property
    def _kwdefaults_(self):
        """What the ``__kwdefaults__`` value would be for a func of the same signature"""
        return {
            p.name: p.default
            for p in self.values()
            if p.default is not p.empty and p.kind == KO
        }

    @property
    def annotations(self):
        """{arg_name: annotation, ...} dict of annotations of the signature.
        What `func.__annotations__` would give you.
        """
        return {
            p.name: p.annotation for p in self.values() if p.annotation is not p.empty
        }

    def detail_names_by_kind(self):
        return (
            self.names_of_kind[PO],
            self.names_of_kind[PK],
            next(iter(self.names_of_kind[VP]), None),
            self.names_of_kind[KO],
            next(iter(self.names_of_kind[VK]), None),
        )

    # TODO: Can be cleaned and generalized (include/exclude, function filter etc.)
    def get_names(self, spec, *, conserve_sig_order=True, allow_excess=False):
        """Return a tuple of names corresponding to the given spec.

        :param spec: An integer, string, or iterable of intergers and strings
        :param conserve_sig_order: Whether to order according to the signature
        :param allow_excess: Whether to allow items in spec that are not in signature

        >>> sig = Sig('a b c d e')
        >>> sig.get_names(0)
        ('a',)
        >>> sig.get_names([0, 2])
        ('a', 'c')
        >>> sig.get_names('b')
        ('b',)
        >>> sig.get_names([0, 'c', -1])
        ('a', 'c', 'e')

        See that by default the order of the signature is conserved:

        >>> sig.get_names('b e d')
        ('b', 'd', 'e')

        But you can change that default to conserve the order of the ``spec`` instead:

        >>> sig.get_names('b e d', conserve_sig_order=False)
        ('b', 'e', 'd')

        By default, you can't mention names that are not in signature.
        To allow this (making ``spec`` have "extract these" interpretation),
        set ``allow_excess=True``:

        >>> sig.get_names(['a', 'c', 'e', 'g', 'h'], allow_excess=True)
        ('a', 'c', 'e')
        """
        if isinstance(spec, str):
            spec = spec.split()
        elif isinstance(spec, int):
            spec = [spec]
        if isinstance(spec, Iterable):

            def find_names():
                names = self.names
                for item in spec:
                    if isinstance(item, int):
                        if item < len(names):
                            yield names[item]
                        elif not allow_excess:
                            raise IndexError(
                                f"There are only {len(names)} names in the signatures,"
                                f"but you asked for the index: {item}"
                            )
                    else:
                        if item in names:
                            yield item
                        elif not allow_excess:
                            raise ValueError(
                                f"No such param name in signatures: {item}"
                            )

            matched_names = tuple(find_names())
            if conserve_sig_order:
                _matched_names = tuple(x for x in self.names if x in matched_names)
                matched_names = _matched_names + tuple(
                    x for x in matched_names if x not in _matched_names
                )
            return matched_names
        else:
            raise TypeError(f"Unknown spec type: {spec}")

    def __iter__(self):
        return iter(self.parameters)

    def __len__(self):
        return len(self.parameters)

    # TODO: Return type inconsistent. When k is a string, returns Parameter,
    #  when an iterable of strings (or 'space separated argument names'),
    #  returns a signature. Could also return a single argument signatures.
    #  Behavior might be confusing. Pros/Cons? See if any current users of getitem,
    #  and switch to single arg signature return (that's consistent, and convenience
    #  of sig[argname] is weak (given sig.params[argname] does it)!)
    def __getitem__(self, k):
        if isinstance(k, int) or isinstance(k, slice):
            # TODO: Could extend slice handing to be able to use names as start and stop
            k = self.names[k]
        if isinstance(k, str):
            names = k.split()  # to handle 'multiple args in a string'
            if len(names) == 1:
                return self.parameters[k]
        else:
            assert isinstance(k, Iterable), f"key should be iterable, was: {k}"
            names = k
        params = [self[name] for name in names]
        return Sig.from_params(params)

    # TODO: Deprecate. Should use names_of_kind directly
    def names_for_kind(self, kind):
        """Get the arg names tuple for a given kind.
        Note, if you need to do this several times, or for several kinds, use
        ``names_of_kind`` property (a tuple) instead: It groups all names of kinds once,
        and caches the result.
        """
        from warnings import warn

        warn("Deprecated", DeprecationWarning)
        return self.names_of_kind[kind]

    # TODO: Consider using names_of_kind in other methods/properties

    @property
    def has_var_kinds(self):
        """Whether the signature has a VAR_POSITIONAL or a VAR_KEYWORD parameter.

        >>> Sig(lambda x, *, y: None).has_var_kinds
        False
        >>> Sig(lambda x, *y: None).has_var_kinds
        True
        >>> Sig(lambda x, **y: None).has_var_kinds
        True
        """
        return bool(self.names_of_kind[VP]) or bool(self.names_of_kind[VK])
        # Old version:
        # return any(p.kind in var_param_kinds for p in self.values())

    @property
    def index_of_var_positional(self):
        """The index of the VAR_POSITIONAL param kind if any, and None if not.
        See also, Sig.index_of_var_keyword

        >>> assert Sig(lambda x, *y, z: 0).index_of_var_positional == 1
        >>> assert Sig(lambda x, /, y, **z: 0).index_of_var_positional == None
        """
        return next((i for i, p in enumerate(self.params) if p.kind == VP), None)

    @property
    def var_positional_name(self):
        idx = self.index_of_var_positional
        if idx is not None:
            return self.names[idx]
        # else returns None

    @property
    def has_var_positional(self):
        """
        Use index_of_var_positional or var_keyword_name directly when needing that
        information as well. This will avoid having to check the kinds list twice.
        """
        return any(p.kind == VP for p in self.values())

    @property
    def index_of_var_keyword(self):
        """The index of a VAR_KEYWORD param kind if any, and None if not.
        See also, Sig.index_of_var_positional

        >>> assert Sig(lambda **kwargs: 0).index_of_var_keyword == 0
        >>> assert Sig(lambda a, **kwargs: 0).index_of_var_keyword == 1
        >>> assert Sig(lambda a, *args, **kwargs: 0).index_of_var_keyword == 2

        And if there's none...

        >>> assert Sig(lambda a, *args, b=1: 0).index_of_var_keyword is None
        """
        last_arg_idx = len(self) - 1
        if last_arg_idx != -1:
            if self.params[last_arg_idx].kind == VK:
                return last_arg_idx
        # else returns None

    @property
    def var_keyword_name(self):
        idx = self.index_of_var_keyword
        if idx is not None:
            return self.names[idx]
        # else returns None

    @property
    def has_var_keyword(self):
        """
        Use index_of_var_keyword or var_keyword_name directly when needing that
        information as well. This will avoid having to check the kinds list twice.
        """
        return any(p.kind == VK for p in self.values())

    @property
    def required_names(self):
        """A tuple of required names, preserving the original signature order.

        A required name is that must be given in a function call, that is, the name of a
        paramater that doesn't have a default, and is not a variadic.

        That lost one is a frequent gotcha, so oo not fall in that gotcha that easily,
        we provide a property that contains what we need.

        >>> f = lambda a00, /, a11, a12, *a23, a34, a35=1, a36='two', **a47: None
        >>> Sig(f).required_names
        ('a00', 'a11', 'a12', 'a34')
        """
        # Note: This is quicker than using self.names_of_kind:
        return tuple(
            p.name
            for p in self.params
            if p.default is empty and p.kind not in var_param_kinds
        )

    @property
    def n_required(self):
        """The number of required arguments.
        A required argument is one that doesn't have a default, nor is VAR_POSITIONAL
        (``*args``) or VAR_KEYWORD (``**kwargs``).

        Note:
            Sometimes a minimum number of arguments in VAR_POSITIONAL and
            VAR_KEYWORD are in fact required,
            but we can't see this from the signature, so we can't tell you about that! You
            do the math.

        >>> f = lambda a00, /, a11, a12, *a23, a34, a35=1, a36='two', **a47: None
        >>> Sig(f).n_required
        4
        """
        return len(self.required_names)

    @property
    def positional_names(self):
        return self.names_of_kind[PO] + self.names_of_kind[PK]

    @property
    def keyword_names(self):
        return self.names_of_kind[PK] + self.names_of_kind[KO]

    def _transform_params(self, changes_for_name: dict):
        for name in self:
            if name in changes_for_name:
                p = changes_for_name[name]
                if isinstance(p, Parameter):
                    p = parameter_to_dict(p)
                yield self[name].replace(**p)
            else:
                # if name is not in params, just use existing param
                yield self[name]

    def modified(self, /, _allow_reordering=False, **changes_for_name):
        """Returns a modified (new) signature object.

        Note:
            This function doesn't modify the signature, but creates a modified copy
            of the signature.

        IMPORTANT WARNING: This is an advanced feature. Avoid wrapping a function with
        a modified signature, as this may not have the intended effect.

        >>> def foo(pka, *vpa, koa, **vka): ...
        >>> sig = Sig(foo)
        >>> sig
        <Sig (pka, *vpa, koa, **vka)>
        >>> assert sig.kinds['pka'] == PK

        Let's make a signature that is the same as sig, except that

            - `poa` is given a PO (POSITIONAL_ONLY) kind insteadk of PK
            - `koa` is given a default of None
            - the signature is given a return_annotation of str

        >>> new_sig = sig.modified(
        ...     pka={'kind': PO},
        ...     koa={'default': None},
        ...     return_annotation=str
        ... )
        >>> new_sig
        <Sig (pka, /, *vpa, koa=None, **vka) -> str>
        >>> assert new_sig.kinds['pka'] == PO  # now pos is of the PO kind!

        Here's an example of changing signature parameters in bulk.
        Here we change all kinds to be the friendly PK kind.

        >>> sig.modified(**{name: {'kind': PK} for name in sig.names})
        <Sig (pka, vpa, koa, vka)>

        Repetition of the above: This gives you a signature with all PK kinds.
        If you wrap a function with it, it will look like it has all PK kinds.
        But that doesn't mean you can actually use thenm as such.
        You'll need to modify (decorate further) your function further to reflect
        its new signature.

        On the other hand, if you decorate a function with a sig that adds or modifies
        defaults, these defaults will actually be used (unlike with `functools.wraps`).
        """
        new_return_annotation = changes_for_name.pop(
            "return_annotation", self.return_annotation
        )

        if _allow_reordering:
            params = sort_params(self._transform_params(changes_for_name))
        else:
            params = list(self._transform_params(changes_for_name))

        return Sig(params, name=self.name, return_annotation=new_return_annotation)

    def sort_params(self):
        """Returns a signature with the parameters sorted by kind and default presence."""
        sorted_params = sort_params(self.params)
        return type(self)(
            sorted_params, name=self.name, return_annotation=self.return_annotation
        )

    def ch_param_attrs(
        self, /, param_attr, *arg_new_vals, _allow_reordering=False, **kwargs_new_vals
    ):
        """Change a specific attribute of the params, returning a modified signature.
        This is a convenience method for the modified method when we're targetting
        a fixed param attribute: 'name', 'kind', 'default', or 'annotation'

        Instead of having to do this

        >>> def foo(a, *b, **c): ...
        >>> Sig(foo).modified(a={'name': 'A'}, b={'name': 'B'}, c={'name': 'C'})
        <Sig (A, *B, **C)>

        We can simply do this

        >>> Sig(foo).ch_param_attrs('name', a='A', b='B', c='C')
        <Sig (A, *B, **C)>

        One quite useful thing you can do with this is to set defaults, or set defaults
        where there are none. If you wrap your function with such a modified signature,
        you get a "curried" version of your function (called "partial" in python).
        (Note that the `functools.wraps` won't deal with defaults "correctly", but
        wrapping with `Sig` objects takes care of that oversight!)

        >>> def foo(a, b, c):
        ...     return a + b * c
        >>> special_foo = Sig(foo).ch_param_attrs('default', b=2, c=3)(foo)
        >>> Sig(special_foo)
        <Sig (a, b=2, c=3)>
        >>> special_foo(5)  # should be 5 + 2 * 3 == 11
        11

        """
        # TODO: Would like to make this work (reordering)
        # Now, if you want to set a default for a but not b and c for example, you'll
        # get complaints (ValueError: non-default argument follows default argument).
        # It's true. But if you're fine with rearranging the argument order,
        # `ch_param_attrs` can take care of that for you.
        # You'll have to tell it explicitly that you wish for this though, because
        # it's conservative.
        #
        # >>> # Note that for time being, Sig.wraps doesn't make a copy of the function
        # >>> #  so we need to redefine foo here@
        # >>> def foo(a, b, c):
        # ...     return a + b * c
        # >>> wrapper = Sig(foo).ch_param_attrs(
        # ... 'default', a=10, _allow_reordering=True
        # ... )
        # >>> another_foo = wrapper(foo)
        # >>> Sig(another_foo)
        # <Sig (b, c, a=10)>
        # >>> another_foo(2, 3)  # should be 10 + (2 * 3) =
        # 16

        if not param_attr in param_attributes:
            raise ValueError(
                f"param_attr needs to be one of: {param_attributes}.",
                f" Was: {param_attr}",
            )
        all_pk_self = self.modified(
            _allow_reordering=True, **{name: {"kind": PK} for name in self.names}
        )
        new_attr_vals = all_pk_self.bind_partial(
            *arg_new_vals, **kwargs_new_vals
        ).arguments
        changes_for_name = {
            name: {param_attr: val} for name, val in new_attr_vals.items()
        }
        return self.modified(_allow_reordering=_allow_reordering, **changes_for_name)

    # Note: Oh, functools, why do you make currying so limited!
    # ch_names = partialmethod(ch_param_attrs, param_attr="name")
    # ch_kinds = partialmethod(ch_param_attrs, param_attr="kind", _allow_reordering=True)
    # ch_defaults = partialmethod(
    #     ch_param_attrs, param_attr="default", _allow_reordering=True
    # )
    # ch_annotations = partialmethod(ch_param_attrs, param_attr="annotation")

    def ch_names(self, /, **changes_for_name):
        argnames_not_in_sig = changes_for_name.keys() - self.keys()
        if argnames_not_in_sig:
            raise ValueError(
                f"argument names not in signature: {', '.join(argnames_not_in_sig)}"
            )
        return self.ch_param_attrs("name", **changes_for_name)

    def ch_kinds(self, /, _allow_reordering=True, **changes_for_name):
        return self.ch_param_attrs(
            "kind", _allow_reordering=_allow_reordering, **changes_for_name
        )

    def ch_kinds_to_position_or_keyword(self):
        return all_pk_signature(self)

    def ch_defaults(self, /, _allow_reordering=True, **changes_for_name):
        return self.ch_param_attrs(
            "default", _allow_reordering=_allow_reordering, **changes_for_name
        )

    def ch_annotations(self, /, **changes_for_name):
        return self.ch_param_attrs("annotation", **changes_for_name)

    def add_optional_keywords(
        self=None, /, kwarg_and_defaults=None, kwarg_annotations=None
    ):
        """Add optional keyword arguments to a signature.

        >>> @Sig.add_optional_keywords({"c": 2, "d": 3}, {"c": int})
        ... def foo(a, *, b=1, **kwargs):
        ...     return f"{a=}, {b=}, {kwargs=}"
        ...

        You can still call the function as before, and like before, any "extra" keyword
        arguments will be passed to kwargs:

        >>> foo(0, d=10)
        "a=0, b=1, kwargs={'d': 10}"

        The difference is that now the signature of `foo` now has `c` and `d`:

        >>> str(Sig(foo))
        '(a, *, c: int = 2, d=3, b=1, **kwargs)'
        """

        # Resolve arguments ( to be able to use this method as a decorator)
        if isinstance(self, dict):
            if kwarg_and_defaults is not None:
                kwarg_annotations = kwarg_and_defaults
                kwarg_and_defaults = None
            if kwarg_and_defaults is None:
                kwarg_and_defaults = self
            self = None

        # If self is None, a factory is returned
        if self is None:
            return partial(
                _add_optional_keywords,
                kwarg_and_defaults=kwarg_and_defaults,
                kwarg_annotations=kwarg_annotations,
            )
        else:  # if not, apply _add_optional_keywords to self
            return _add_optional_keywords(
                self, kwarg_and_defaults, kwarg_annotations=kwarg_annotations
            )

    # TODO: Make default_conflict_method be able to be a callable and get rid of string
    #  mapping complexity in merge_with_sig code
    def merge_with_sig(
        self,
        sig: ParamsAble,
        ch_to_all_pk: bool = False,
        *,
        default_conflict_method: SigMergeOptions = DFLT_DEFAULT_CONFLICT_METHOD,
    ):
        """Return a signature obtained by merging self signature with another signature.
        Insofar as it can, given the kind precedence rules, the arguments of self will
        appear first.

        :param sig: The signature to merge with.
        :param ch_to_all_pk: Whether to change all kinds of both signatures to PK (
            POSITIONAL_OR_KEYWORD)
        :return:

        >>> def func(a=None, *, b=1, c=2):
        ...     ...
        ...
        >>>
        >>> s = Sig(func)
        >>> s
        <Sig (a=None, *, b=1, c=2)>

        Observe where the new arguments ``d`` and ``e`` are placed,
        according to whether they have defaults and what their kind is:

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

        If the kind of the params is not important, but order is, you can specify
        ``ch_to_all_pk=True``:

        >>> s.merge_with_sig(["d", "e"], ch_to_all_pk=True)
        <Sig (d, e, a=None, b=1, c=2)>
        >>> s.merge_with_sig([("d", 3), ("e", 4)], ch_to_all_pk=True)
        <Sig (a=None, b=1, c=2, d=3, e=4)>
        """
        if ch_to_all_pk:
            _self = Sig(all_pk_signature(self))
            _sig = Sig(all_pk_signature(ensure_signature(sig)))
        else:
            _self = self
            _sig = Sig(sig)

        # Validation of the signatures

        _msg = f"\nHappened during an attempt to merge {self} and {sig}"
        errors = {}

        # Check if both signatures have VAR_POSITIONAL parameters
        if _self.has_var_keyword and _sig.has_var_keyword:
            errors["var_positional_conflict"] = (
                f"Can't merge two signatures if they both have a VAR_POSITIONAL parameter: {_msg}"
            )

        # Check if both signatures have VAR_KEYWORD parameters
        if _self.has_var_keyword and _sig.has_var_keyword:
            errors["var_keyword_conflict"] = (
                f"Can't merge two signatures if they both have a VAR_KEYWORD parameter: {_msg}"
            )

        # Check if parameters with the same name have the same kind
        if not all(
            _self[name].kind == _sig[name].kind for name in _self.keys() & _sig.keys()
        ):
            errors["kind_mismatch"] = (
                "During a signature merge, if two names are the same, they must have the "
                f"**same kind**:\n\t{_msg}\n"
                "Tip: If you're trying to merge functions in some way, consider decorating "
                "them with a signature mapping that avoids the argument name clashing"
            )

        # Check if default_conflict_method is a valid SigMergeOption
        if default_conflict_method not in get_args(SigMergeOptions):
            errors["invalid_conflict_method"] = (
                f"default_conflict_method should be one of: {get_args(SigMergeOptions)}"
            )

        if errors:
            # TODO: Raise all errors at once?
            # TODO: Raise custom errors with more info?

            # raise the first error
            error_msg = next(iter(errors.values()))
            raise IncompatibleSignatures(error_msg, sig1=_self, sig2=_sig)

        if default_conflict_method == "take_first":
            _sig = _sig - set(_self.keys() & _sig.keys())
        elif default_conflict_method == "fill_defaults_and_annotations":
            _self = _fill_defaults_and_annotations(_self, _sig)
            _sig = _fill_defaults_and_annotations(_sig, _self)

        if not all(
            _self[name].default == _sig[name].default
            for name in _self.keys() & _sig.keys()
        ):
            # if default_conflict_method == 'take_first':
            #     _sig = _sig - set(_self.keys() & _sig.keys())
            # else:

            error_msg = (
                "During a signature merge, if two names are the same they must have the"
                f"**same default**:\n\t{_msg}\n"
                "Tip: If you're trying to merge functions in some way, consider "
                "decorating "
                "them with a signature mapping that avoids the argument name clashing."
                "You can also set ch_to_all_pk=True to ignore the kind of the "
                'parameters or change the default_conflict_method to "take_first",'
                "or another method that suits your needs."
            )

            raise IncompatibleSignatures(error_msg, sig1=_self, sig2=_sig)

        # assert all(
        #     _self[name].default == _sig[name].default
        #     for name in _self.keys() & _sig.keys()
        # ), (
        #     'During a signature merge, if two names are the same, they must have the '
        #     f'**same default**:\n\t{_msg}\n'
        #     "Tip: If you're trying to merge functions in some way, consider
        #     decorating "
        #     "them a signature mapping that "
        #     'avoids the argument name clashing'
        # )

        params = list(
            self._chain_params_of_signatures(
                _self.without_defaults,
                _sig.without_defaults,
                _self.with_defaults,
                _sig.with_defaults,
            )
        )
        params.sort(key=lambda p: p.kind)
        return self.__class__(params)

    def __add__(self, sig: ParamsAble):
        """Merge two signatures (casting all non-VAR kinds to POSITIONAL_OR_KEYWORD
        before hand)

        Important Notes:

        - The resulting Sig will loose it's return_annotation if it had one.
          This is to avoid making too many assumptions about how the sig sum will be
          used. If a return_annotation is needed (say, for composition, the last
          return_annotation summed), one can subclass Sig and overwrite __add__
        - POSITION_ONLY and KEYWORD_ONLY kinds will be replaced by
          POSITIONAL_OR_KEYWORD kind. This is to simplify the interface and code.
        If the user really wants to maintain those kinds, they can replace them back
        after the fact.

        >>> def f(w, /, x: float = 1, y=1, *, z: int = 1):
        ...     ...
        >>> def h(i, j, w):
        ...     ...  # has a 'w' argument, like f and g
        ...
        >>> def different(a, b: str, c=None):
        ...     ...  # No argument names in common with other functions

        >>> Sig(f) + Sig(different)
        <Sig (w, a, b: str, x: float = 1, y=1, z: int = 1, c=None)>
        >>> Sig(different) + Sig(f)
        <Sig (a, b: str, w, c=None, x: float = 1, y=1, z: int = 1)>

        The order of the first signature will take precedence over the second,
        but default-less arguments have to come before arguments with defaults.
         first, and Note the difference of the orders.

        >>> Sig(f) + Sig(h)
        <Sig (w, i, j, x: float = 1, y=1, z: int = 1)>
        >>> Sig(h) + Sig(f)
        <Sig (i, j, w, x: float = 1, y=1, z: int = 1)>

        The sum of two Sig's takes a safe-or-blow-up-now approach.
        If any of the arguments have different defaults or annotations, summing will
        raise an AssertionError.
        It's up to the user to decorate their input functions to express the default
        they actually desire.

        >>> def ff(w, /, x: float, y=1, *, z: int = 1):
        ...     ...  # just like f, but without the default for x
        >>> Sig(f) + Sig(ff)  # doctest: +IGNORE_EXCEPTION_DETAIL +ELLIPSIS
        Traceback (most recent call last):
        ...
        IncompatibleSignatures: During a signature merge, if two names are the same, they must
        have the **same default**
        ...

        >>> def hh(i, j, w=1):
        ...     ...  # like h, but w has a default
        ...
        >>> Sig(h) + Sig(hh)  # doctest: +IGNORE_EXCEPTION_DETAIL +ELLIPSIS
        Traceback (most recent call last):
        ...
        IncompatibleSignatures: During a signature merge, if two names are the same, they must
        have the **same default**
        ...

        >>> Sig(f) + [
        ...     "w",
        ...     ("y", 1),
        ...     ("d", 1.0, float),
        ...     dict(name="special", kind=Parameter.KEYWORD_ONLY, default=0),
        ... ]
        <Sig (w, x: float = 1, y=1, z: int = 1, d: float = 1.0, special=0)>

        """
        return self.merge_with_sig(sig, ch_to_all_pk=True)

    def __radd__(self, sig: ParamsAble):
        """Adding on the right.
        The raison d'être for this is so that you can start your summing with any
        signature speccifying
         object that Sig will be able to resolve into a signature. Like this:

        >>> ["first_arg", ("second_arg", 42)] + Sig(lambda x, y: x * y)
        <Sig (first_arg, x, y, second_arg=42)>

        Note that the ``second_arg`` doesn't actually end up being the second argument
        because
        it has a default and x and y don't. But if you did this:

        >>> ["first_arg", ("second_arg", 42)] + Sig(lambda x=0, y=1: x * y)
        <Sig (first_arg, second_arg=42, x=0, y=1)>

        you'd get what you expect.

        Of course, we could have just obliged you to say ``Sig(['first_arg',
        ('second_arg', 42)])``
        explicitly and spare ourselves yet another method.
        The reason we made ``__radd__`` is so we can make it handle 0 + Sig(...),
        so that you can
        merge an iterable of signatures like this:

        >>> def f(a, b, c):
        ...     ...
        ...
        >>> def g(c, b, e):
        ...     ...
        ...
        >>> sigs = map(Sig, [f, g])
        >>> sum(sigs)
        <Sig (a, b, c, e)>

        Let's say, for whatever reason (don't ask me), you wanted to make a function
        that contains all the
        arguments of all the functions of ``os.path`` (that don't contain any var arg
        kinds).

        >>> import os.path
        >>> funcs = list(
        ...     filter(
        ...         callable,
        ...         (
        ...             getattr(os.path, a)
        ...             for a in dir(os.path)
        ...             if not a.startswith("_")
        ...         ),
        ...     )
        ... )
        >>> sigs = filter(lambda sig: not sig.has_var_kinds, map(Sig, funcs))
        >>> # Note: Skipping because not stable between python versions
        >>> sum(sigs)  # doctest: +SKIP
        <Sig (path, p, paths, m, filename, s, f1, f2, fp1, fp2, s1, s2, start=None)>
        """
        if sig == 0:  # so that we can do ``sum(iterable_of_sigs)``
            sig = Sig([])
        else:
            sig = Sig(sig)
        return sig.merge_with_sig(self)

    def remove_names(self, names):
        names = {p.name for p in ensure_params(names)}
        new_params = {
            name: p for name, p in self.parameters.items() if name not in names
        }
        return self.__class__(new_params, return_annotation=self.return_annotation)

    def add_params(self, params: Iterable):
        """Creates a new instance of Sig after merging the parameters of this signature
        with a list of new parameters. The new list of parameters is automatically
        sorted based on signature constraints given by kinds and default values.
        See Python native signature documentation for more details.

        >>> s = Sig('(a, /, b, *, c)')
        >>> s.add_params([
        ...     Param('kwargs', VK),
        ...     dict(name='d', kind=KO),
        ...     Param('args', VP),
        ...     'e',
        ...     Param('f', PO),
        ... ])
        <Sig (a, f, /, b, e, *args, c, d, **kwargs)>
        """

        def comparator(param):
            return (param.kind, param.kind == KO or param.default is not empty)

        new_params = self.params + [ensure_param(p) for p in params]
        new_params = sorted(new_params, key=comparator)
        return type(self)(new_params)

    def __sub__(self, sig):
        return self.remove_names(sig)

    @staticmethod
    def _chain_params_of_signatures(*sigs):
        """Yields Parameter instances taken from sigs without repeating the same name
        twice.

        >>> str(list(
        ...     Sig._chain_params_of_signatures(
        ...         Sig(lambda x, *args, y=1: ...),
        ...         Sig(lambda x, y, z, **kwargs: ...),
        ...     )
        ...   )
        ... )
        '[<Parameter "x">, <Parameter "*args">, <Parameter "y=1">, <Parameter "z">, <Parameter "**kwargs">]'
        """
        already_merged_names = set()
        for s in sigs:
            for p in s.parameters.values():
                if p.name not in already_merged_names:
                    yield p
                already_merged_names.add(p.name)

    @property
    def without_defaults(self):
        """Sub-signature containing only "required" (i.e. without defaults) parameters.

        >>> list(Sig(lambda *args, a, b, x=1, y=1, **kwargs: ...).without_defaults)
        ['a', 'b']
        """
        return self.__class__(
            p for p in self.values() if not param_has_default_or_is_var_kind(p)
        )

    @property
    def with_defaults(self):
        """Sub-signature containing only "not required" (i.e. with defaults) parameters.

        >>> list(Sig(lambda *args, a, b, x=1, y=1, **kwargs: ...).with_defaults)
        ['args', 'x', 'y', 'kwargs']
        """
        return self.__class__(
            p for p in self.values() if param_has_default_or_is_var_kind(p)
        )

    def normalize_kind(
        self,
        kind=PK,
        except_kinds=var_param_kinds,
        add_defaults_if_necessary=False,
        argname_to_default=None,
        allow_reordering=False,
    ):
        except_kinds = except_kinds or set()
        if add_defaults_if_necessary:
            if argname_to_default is None:

                def argname_to_default(argname):
                    return None

        def changed_params():
            there_was_a_default = False
            for p in self.parameters.values():
                if p.kind not in except_kinds:
                    if add_defaults_if_necessary:
                        if there_was_a_default and p.default is _empty:
                            p = p.replace(kind=kind, default=argname_to_default(p.name))
                        there_was_a_default = p.default is not _empty
                    else:
                        p = p.replace(kind=kind)
                yield p

        params = list(changed_params())
        try:
            return type(self)(params, return_annotation=self.return_annotation)
        except ValueError as e:
            if allow_reordering:
                return self.sort_params()
            else:
                raise

    def map_arguments(
        self,
        args: tuple = None,
        kwargs: dict = None,
        *,
        apply_defaults=False,
        allow_partial=False,
        allow_excess=False,
        ignore_kind=False,
    ) -> dict:
        """Map arguments (args and kwargs) to the parameters of function's signature.

        When you need to manage how the arguments of a function are specified,
        you need to take care of
        multiple cases depending on whether they were specified as positional arguments
        (`args`) or keyword arguments (`kwargs`).

        The `map_arguments` (and it's sorta-inverse inverse,
        `mk_args_and_kwargs`)
        are there to help you manage this.

        If you could rely on the the fact that only `kwargs` were given it would
        reduce the complexity of your code.
        This is why we have the `all_pk_signature` function in `signatures.py`.

        We also need to have a means to make a `kwargs` only from the actual `(*args,
        **kwargs)` used at runtime.
        We have `Signature.bind` (and `bind_partial`) for that.

        But these methods will fail if there is extra stuff in the `kwargs`.
        Yet sometimes we'd like to have a `dict` that services several functions that
        will extract their needs from it.

        That's where  `Sig.map_arguments_from_variadics(*args, **kwargs)` is needed.

        :param args: The args the function will be called with.
        :param kwargs: The kwargs the function will be called with.
        :param apply_defaults: (bool) Whether to apply signature defaults to the
            non-specified argument names
        :param allow_partial: (bool) True iff you want to allow partial signature
            fulfillment.
        :param allow_excess: (bool) Set to True iff you want to allow extra kwargs
            items to be ignored.
        :param ignore_kind: (bool) Set to True iff you want to ignore the position and
            keyword only kinds,
            in order to be able to accept args and kwargs in such a way that there can
            be cross-over
            (args that are supposed to be keyword only, and kwargs that are supposed
            to be positional only)
        :return: An {param_name: arg_val, ...} dict

        See also the sorta-inverse of this function: mk_args_and_kwargs

        >>> def foo(w, /, x: float, y="YY", *, z: str = "ZZ"):
        ...     ...
        >>> sig = Sig(foo)
        >>> assert (
        ...     sig.map_arguments((11, 22, "you"), dict(z="zoo"))
        ...     == sig.map_arguments((11, 22), dict(y="you", z="zoo"))
        ...     == {"w": 11, "x": 22, "y": "you", "z": "zoo"}
        ... )

        By default, `apply_defaults=False`, which will lead to only get those
        arguments you input.

        >>> sig.map_arguments(args=(11,), kwargs={"x": 22})
        {'w': 11, 'x': 22}

        But if you specify `apply_defaults=True` non-specified non-require arguments
        will be returned with their defaults:

        >>> sig.map_arguments(
        ...     args=(11,), kwargs={"x": 22}, apply_defaults=True
        ... )
        {'w': 11, 'x': 22, 'y': 'YY', 'z': 'ZZ'}

        By default, `ignore_excess=False`, so specifying kwargs that are not in the
        signature will lead to an exception.

        >>> sig.map_arguments(
        ...     args=(11,), kwargs={"x": 22, "not_in_sig": -1}
        ... )
        Traceback (most recent call last):
            ...
        TypeError: got an unexpected keyword argument 'not_in_sig'

        Specifying `allow_excess=True` will ignore such excess fields of kwargs.
        This is useful when you want to source several functions from a same dict.

        >>> sig.map_arguments(
        ...     args=(11,), kwargs={"x": 22, "not_in_sig": -1}, allow_excess=True
        ... )
        {'w': 11, 'x': 22}

        On the other side of `ignore_excess` you have `allow_partial` that will allow
        you, if
        set to `True`, to underspecify the params of a function (in view of being
        completed later).

        >>> sig.map_arguments(args=(), kwargs={"x": 22})
        Traceback (most recent call last):
        ...
        TypeError: missing a required argument: 'w'

        But if you specify `allow_partial=True`...

        >>> sig.map_arguments(
        ...     args=(), kwargs={"x": 22}, allow_partial=True
        ... )
        {'x': 22}

        That's a lot of control (eight combinations total), but not everything is
        controllable here:
        Position only and keyword only kinds need to be respected:

        >>> sig.map_arguments(args=(1, 2, 3, 4), kwargs={})
        Traceback (most recent call last):
        ...
        TypeError: too many positional arguments
        >>> sig.map_arguments(args=(), kwargs=dict(w=1, x=2, y=3, z=4))
        Traceback (most recent call last):
        ...
        TypeError:...'w'...

        But if you want to ignore the kind of parameter, just say so:

        >>> sig.map_arguments(
        ...     args=(1, 2, 3, 4), kwargs={}, ignore_kind=True
        ... )
        {'w': 1, 'x': 2, 'y': 3, 'z': 4}
        >>> sig.map_arguments(
        ...     args=(), kwargs=dict(w=1, x=2, y=3, z=4), ignore_kind=True
        ... )
        {'w': 1, 'x': 2, 'y': 3, 'z': 4}
        """

        def get_var_dflts():
            if self.has_var_positional:
                yield self.var_positional_name, ()
            if self.has_var_keyword:
                yield self.var_keyword_name, {}

        _args = args or ()
        _kwargs = kwargs or {}

        if ignore_kind:
            var_dflts = dict(get_var_dflts())
            sig = self.normalize_kind(kind=KO, except_kinds=None)
            sig = sig.ch_defaults(**var_dflts)
            for arg, p in zip(_args, sig.params):
                if p.name in _kwargs:
                    raise TypeError(f"multiple values for argument '{p.name}'")
                _kwargs[p.name] = arg
            _args = ()
        else:
            sig = self

        if not sig.has_var_positional and allow_excess:
            max_allowed_num_of_posisional_args = sum(
                k <= PK for k in sig.kinds.values()
            )
            _args = _args[:max_allowed_num_of_posisional_args]
        if not sig.has_var_keyword and allow_excess:
            _kwargs = {k: v for k, v in _kwargs.items() if k in sig}

        binder = sig.bind_partial if allow_partial else sig.bind
        b = binder(*_args, **_kwargs)
        if apply_defaults:
            b.apply_defaults()

        return b.arguments

    kwargs_from_args_and_kwargs = deprecation_of(
        map_arguments, "kwargs_from_args_and_kwargs"
    )

    def mk_args_and_kwargs(
        self,
        arguments: dict,
        *,
        apply_defaults=False,
        allow_partial=False,
        allow_excess=False,
        ignore_kind=False,
        args_limit: int | None = 0,
    ) -> tuple[tuple, dict]:
        """Extract args and kwargs such that ``func(*args, **kwargs)`` can be called,
        where func has instance's signature.

        :param arguments: The {param_name: arg_val,...} dict to process
        :param args_limit: How "far" in the params should args (positional arguments)
            be searched for.

            - args_limit==0: Take the minimum number possible of args (positional
              arguments). Only those that are position only or before a var-positional.
            - args_limit is None: Take the maximum number of args (positional arguments).
              The only kwargs (keyword arguments) you should have are keyword-only
              and var-keyword arguments.
            - args_limit positive integer: Take the args_limit first argument names
              (of signature) as args, and the rest as kwargs.

        >>> def foo(w, /, x: float, y=1, *, z: int = 1):
        ...     return ((w + x) * y) ** z
        >>> foo_sig = Sig(foo)
        >>> args, kwargs = foo_sig.mk_args_and_kwargs(
        ...     dict(w=4, x=3, y=2, z=1)
        ... )
        >>> assert (args, kwargs) == ((4,), {"x": 3, "y": 2, "z": 1})
        >>> assert foo(*args, **kwargs) == foo(4, 3, 2, z=1) == 14

        What about variadics?

        >>> def bar(a, /, b, *args, c=2, **kwargs):
        ...     pass
        >>> Sig(bar).mk_args_and_kwargs(
        ...     dict(a=1, b=2, args=(3,4), c=5, kwargs=dict(d=6, e=7))
        ... )
        ((1, 2, 3, 4), {'c': 5, 'd': 6, 'e': 7})

        You can also give the arguments in a different order:

        >>> Sig(bar).mk_args_and_kwargs(
        ...     dict(args=(3,4), kwargs=dict(d=6, e=7), b=2, c=5, a=1)
        ... )
        ((1, 2, 3, 4), {'c': 5, 'd': 6, 'e': 7})

        The `args_limit` begs explanation.
        Consider the signature of `def foo(w, /, x: float, y=1, *, z: int = 1): ...`
        for instance. We could call the function with the following (args, kwargs) pairs:

        - ((1,), {'x': 2, 'y': 3, 'z': 4})
        - ((1, 2), {'y': 3, 'z': 4})
        - ((1, 2, 3), {'z': 4})
          The two other combinations (empty args or empty kwargs) are not valid
          because of the / and * constraints.

        But when asked for an (args, kwargs) pair, which of the three valid options
        should be returned? This is what the `args_limit` argument controls.

        If `args_limit == 0`, the least args (positional arguments) will be returned.
        It's the default.

        >>> arguments = dict(w=4, x=3, y=2, z=1)
        >>> foo_sig.mk_args_and_kwargs(arguments, args_limit=0)
        ((4,), {'x': 3, 'y': 2, 'z': 1})

        If `args_limit is None`, the least kwargs (keyword arguments) will be returned.

        >>> foo_sig.mk_args_and_kwargs(arguments, args_limit=None)
        ((4, 3, 2), {'z': 1})

        If `args_limit` is a positive integer, the first `[args_limit]` arguments
        will be returned (not checking at all if this is valid!).

        >>> foo_sig.mk_args_and_kwargs(arguments, args_limit=1)
        ((4,), {'x': 3, 'y': 2, 'z': 1})
        >>> foo_sig.mk_args_and_kwargs(arguments, args_limit=2)
        ((4, 3), {'y': 2, 'z': 1})
        >>> foo_sig.mk_args_and_kwargs(arguments, args_limit=3)
        ((4, 3, 2), {'z': 1})

        Note that if you specify `args_limit` to be greater than the maximum of
        positional arguments, it behaves as if `args_limit` was `None`:

        >>> foo_sig.mk_args_and_kwargs(arguments, args_limit=4)
        ((4, 3, 2), {'z': 1})

        Note that 'args_limit''s behavior is consistent with list behvior in the sense
        that:

        >>> args = (0, 1, 2, 3)
        >>> args[:0]
        ()
        >>> args[:None]
        (0, 1, 2, 3)
        >>> args[2]
        2

        If variable positional arguments are present, `args_limit` is ignored and
        all positional arguments are returned as args.

        >>> Sig(bar).mk_args_and_kwargs(
        ...     dict(a=1, b=2, args=(3,4), c=5, kwargs=dict(d=6, e=7)),
        ...     args_limit=1
        ... )
        ((1, 2, 3, 4), {'c': 5, 'd': 6, 'e': 7})

        By default, only the arguments that were given in the `arguments` input will be
        returned in the (args, kwargs) output.
        If you also want to get those that have defaults (according to signature),
        you need to specify it with the `apply_defaults=True` argument.

        >>> foo_sig.mk_args_and_kwargs(dict(w=4, x=3))
        ((4,), {'x': 3})
        >>> foo_sig.mk_args_and_kwargs(dict(w=4, x=3), apply_defaults=True)
        ((4,), {'x': 3, 'y': 1, 'z': 1})

        By default, all required arguments must be given.
        Not doing so will lead to a `TypeError`.
        If you want to process your arguments anyway, specify `allow_partial=True`.

        >>> foo_sig.mk_args_and_kwargs(dict(w=4))
        Traceback (most recent call last):
          ...
        TypeError: missing a required argument: 'x'
        >>> foo_sig.mk_args_and_kwargs(dict(w=4), allow_partial=True)
        ((4,), {})

        Specifying argument names that are not recognized by the signature will
        lead to a `TypeError`.
        If you want to avoid this (and just take from the input `kwargs` what ever you
        can), specify this with `allow_excess=True`.

        >>> foo_sig.mk_args_and_kwargs(dict(w=4, x=3, extra='stuff'))
        Traceback (most recent call last):
            ...
        TypeError: Got unexpected keyword arguments: extra
        >>> foo_sig.mk_args_and_kwargs(dict(w=4, x=3, extra='stuff'),
        ...     allow_excess=True)
        ((4,), {'x': 3})

        See `map_arguments` (namely for the description of the arguments).
        """
        arguments = arguments or {}
        extra_arguments = set(arguments) - set(self.names)
        if extra_arguments and not allow_excess:
            raise TypeError(
                f"Got unexpected keyword arguments: {', '.join(extra_arguments)}"
            )
        _arguments = {p: arguments[p] for p in self.names if p in arguments}
        vp_args = _arguments.get(self.var_positional_name, ())
        vk_args = _arguments.get(self.var_keyword_name, {})
        if vp_args:
            # If there are var positional arguments, we ignore the args_limit
            args_limit = None

        pos, pks, kos = (
            self.names_of_kind[PO],
            self.names_of_kind[PK],
            self.names_of_kind[KO],
        )
        names_for_args = pos
        names_for_kwargs = kos
        if args_limit is None:
            # All the PKs go to args, so we have:
            # names_for_args == POs + PKs
            # names_for_kwargs == KOs
            names_for_args += pks
        else:
            # Take the [args_limit] first arguments (of signature) as args. The minimum
            # number of args is the number of POs. The maximum number of args is the
            # number of POs + PKs. The rest are kwargs.
            nb_of_positional_pks = min(max(args_limit - len(pos), 0), len(pks))
            names_for_args += pks[:nb_of_positional_pks]
            names_for_kwargs = pks[nb_of_positional_pks:] + names_for_kwargs

        args = tuple(_arguments[name] for name in names_for_args if name in _arguments)
        kwargs = {
            name: _arguments[name] for name in names_for_kwargs if name in _arguments
        }

        # Note that, at this stage, the variadics arguments are not yet in the args and
        # kwargs variables.
        # We first call map_arguments with the args and kwargs with no variadics to
        # validate that all the explicit arguments are valid and there is no missing
        # required argument.

        # In fact, imagine the following:

        # >>> def foo(a, *args):
        # ...     ...
        # >>> foo_sig = Sig(foo)
        # >>> foo_sig.mk_args_and_kwargs(arguments=dict(args=(1,)))

        # This should fail because `a` is missing in the arguments.
        # But if we included the variadics in the args, the value '1' would have been
        # mapped to `a` by `map_arguments` and the error would not have been caught.
        # Same logic for kwargs.

        __arguments = self.map_arguments(
            args,
            kwargs,
            apply_defaults=apply_defaults,
            allow_partial=allow_partial,
            # allow_excess=allow_excess,
            # ignore_kind=ignore_kind,
        )

        # Let's retrieve the args and kwargs from the output of `map_arguments`, because
        # some extra stuff might have been added (defaults). And let's also add the
        # variadics.
        pos_arguments = {
            name: arg for name, arg in __arguments.items() if name in names_for_args
        }
        kw_arguments = {
            name: arg for name, arg in __arguments.items() if name in names_for_kwargs
        }

        if ignore_kind:
            # If ignore_kind is True, return all arguments as kwargs
            args = ()
            d_vp_args = (
                {self.var_positional_name: vp_args} if self.has_var_positional else {}
            )
            d_vk_args = {self.var_keyword_name: vk_args} if self.has_var_keyword else {}
            kwargs = {**pos_arguments, **d_vp_args, **kw_arguments, **d_vk_args}
        else:
            args = tuple(pos_arguments.values()) + vp_args
            kwargs = dict(kw_arguments, **vk_args)

        return args, kwargs

    args_and_kwargs_from_kwargs = deprecation_of(
        mk_args_and_kwargs, "args_and_kwargs_from_kwargs"
    )

    def map_arguments_from_variadics(
        self,
        *args,
        _apply_defaults=False,
        _allow_partial=False,
        _allow_excess=False,
        _ignore_kind=False,
        **kwargs,
    ):
        """Convenience method that calls map_arguments from variadics

        >>> def foo(w, /, x: float, y="YY", *, z: str = "ZZ"):
        ...     ...
        >>> sig = Sig(foo)
        >>> assert (
        ...     sig.map_arguments_from_variadics(1, 2, 3, z=4)
        ...     == sig.map_arguments_from_variadics(1, 2, y=3, z=4)
        ...     == {"w": 1, "x": 2, "y": 3, "z": 4}
        ... )

        What about var positional and var keywords?

        >>> def bar(*args, **kwargs):
        ...     ...
        ...
        >>> Sig(bar).map_arguments_from_variadics(1, 2, y=3, z=4)
        {'args': (1, 2), 'kwargs': {'y': 3, 'z': 4}}

        Note that though `w` is a position only argument, you can specify `w=11` as
        a keyword argument too, using `_ignore_kind=True`:

        >>> Sig(foo).map_arguments_from_variadics(w=11, x=22, _ignore_kind=True)
        {'w': 11, 'x': 22}

        You can use `_allow_partial` that will allow you, if
        set to `True`, to underspecify the params of a function
        (in view of being completed later).

        >>> Sig(foo).map_arguments_from_variadics(x=3, y=2)
        Traceback (most recent call last):
          ...
        TypeError: missing a required argument: 'w'

        But if you specify `_allow_partial=True`...

        >>> Sig(foo).map_arguments_from_variadics(x=3, y=2, _allow_partial=True)
        {'x': 3, 'y': 2}

        By default, `_apply_defaults=False`, which will lead to only get those arguments
        you input.

        >>> Sig(foo).map_arguments_from_variadics(4, x=3, y=2)
        {'w': 4, 'x': 3, 'y': 2}

        But if you specify `_apply_defaults=True` non-specified non-require arguments
        will be returned with their defaults:

        >>> Sig(foo).map_arguments_from_variadics(4, x=3, y=2, _apply_defaults=True)
        {'w': 4, 'x': 3, 'y': 2, 'z': 'ZZ'}
        """
        return self.map_arguments(
            args,
            kwargs,
            apply_defaults=_apply_defaults,
            allow_partial=_allow_partial,
            allow_excess=_allow_excess,
            ignore_kind=_ignore_kind,
        )

    extract_kwargs = deprecation_of(map_arguments_from_variadics, "extract_kwargs")

    def extract_args_and_kwargs(
        self,
        *args,
        _ignore_kind=True,
        _allow_partial=False,
        _allow_excess=True,
        _apply_defaults=False,
        _args_limit=0,
        **kwargs,
    ):
        """Source the (args, kwargs) for the signature instance, ignoring excess
        arguments.

        >>> def foo(w, /, x: float, y=2, *, z: int = 1):
        ...     return w + x * y ** z
        >>> args, kwargs = Sig(foo).extract_args_and_kwargs(4, x=3, y=2)
        >>> (args, kwargs) == ((4,), {"x": 3, "y": 2})
        True

        The difference with map_arguments_from_variadics is that here the output is
        ready to be called by the function whose signature we have, since the
        position-only arguments will be returned as args.

        >>> foo(*args, **kwargs)
        10

        Note that though `w` is a position only argument, you can specify `w=4` as a
        keyword argument too (by default):

        >>> args, kwargs = Sig(foo).extract_args_and_kwargs(w=4, x=3, y=2)
        >>> (args, kwargs) == ((4,), {"x": 3, "y": 2})
        True

        If you don't want to allow that, you can say `_ignore_kind=False`

        >>> Sig(foo).extract_args_and_kwargs(w=4, x=3, y=2, _ignore_kind=False)
        Traceback (most recent call last):
          ...
        TypeError:...'w'...

        You can use `_allow_partial` that will allow you, if
        set to `True`, to underspecify the params of a function (in view of being
        completed later).

        >>> Sig(foo).extract_args_and_kwargs(x=3, y=2)
        Traceback (most recent call last):
          ...
        TypeError:...'w'...

        But if you specify `_allow_partial=True`...

        >>> args, kwargs = Sig(foo).extract_args_and_kwargs(
        ...     x=3, y=2, _allow_partial=True
        ... )
        >>> (args, kwargs) == ((), {"x": 3, "y": 2})
        True

        By default, `_apply_defaults=False`, which will lead to only get those
        arguments you input.

        >>> args, kwargs = Sig(foo).extract_args_and_kwargs(4, x=3, y=2)
        >>> (args, kwargs) == ((4,), {"x": 3, "y": 2})
        True

        But if you specify `_apply_defaults=True` non-specified non-require arguments
        will be returned with their defaults:

        >>> args, kwargs = Sig(foo).extract_args_and_kwargs(
        ...     4, x=3, y=2, _apply_defaults=True
        ... )
        >>> (args, kwargs) == ((4,), {"x": 3, "y": 2, "z": 1})
        True
        """
        arguments = self.map_arguments(
            args,
            kwargs,
            apply_defaults=_apply_defaults,
            allow_partial=_allow_partial,
            allow_excess=_allow_excess,
            ignore_kind=_ignore_kind,
        )
        return self.mk_args_and_kwargs(
            arguments,
            allow_partial=_allow_partial,
            args_limit=_args_limit,
        )

    def source_arguments(
        self,
        *args,
        _apply_defaults=False,
        _allow_partial=False,
        _ignore_kind=True,
        **kwargs,
    ):
        """Source the arguments for the signature instance, ignoring excess arguments.

        >>> def foo(w, /, x: float, y="YY", *, z: str = "ZZ"):
        ...     ...
        >>> Sig(foo).source_arguments(11, x=22, extra="keywords", are="ignored")
        {'w': 11, 'x': 22}

        Note that though `w` is a position only argument, you can specify `w=11` as a
        keyword argument too (by default):

        >>> Sig(foo).source_arguments(w=11, x=22, extra="keywords", are="ignored")
        {'w': 11, 'x': 22}

        If you don't want to allow that, you can say `_ignore_kind=False`

        >>> Sig(foo).source_arguments(
        ...     w=11, x=22, extra="keywords", are="ignored", _ignore_kind=False
        ... )
        Traceback (most recent call last):
          ...
        TypeError: ...'w'...

        You can use `_allow_partial` that will allow you, if
        set to `True`, to underspecify the params of a function (in view of being
        completed later).

        >>> Sig(foo).source_arguments(x=3, y=2, extra="keywords", are="ignored")
        Traceback (most recent call last):
          ...
        TypeError: ...'w'...

        But if you specify `_allow_partial=True`...

        >>> Sig(foo).source_arguments(
        ...     x=3, y=2, extra="keywords", are="ignored", _allow_partial=True
        ... )
        {'x': 3, 'y': 2}

        By default, `_apply_defaults=False`, which will lead to only get those
        arguments you input.

        >>> Sig(foo).source_arguments(4, x=3, y=2, extra="keywords", are="ignored")
        {'w': 4, 'x': 3, 'y': 2}

        But if you specify `_apply_defaults=True` non-specified non-require arguments
        will be returned with their defaults:

        >>> Sig(foo).source_arguments(
        ...     4, x=3, y=2, extra="keywords", are="ignored", _apply_defaults=True
        ... )
        {'w': 4, 'x': 3, 'y': 2, 'z': 'ZZ'}
        """
        return self.map_arguments(
            args,
            kwargs,
            apply_defaults=_apply_defaults,
            allow_partial=_allow_partial,
            allow_excess=True,
            ignore_kind=_ignore_kind,
        )

    source_kwargs = deprecation_of(source_arguments, "source_kwargs")

    def source_args_and_kwargs(
        self,
        *args,
        _ignore_kind=True,
        _allow_partial=False,
        _apply_defaults=False,
        _args_limit=0,
        **kwargs,
    ):
        """Source the (args, kwargs) for the signature instance, ignoring excess
        arguments.

        >>> def foo(w, /, x: float, y=2, *, z: int = 1):
        ...     return w + x * y ** z
        >>> args, kwargs = Sig(foo).source_args_and_kwargs(
        ...     4, x=3, y=2, extra="keywords", are="ignored"
        ... )
        >>> args, kwargs
        ((4,), {'x': 3, 'y': 2})

        The difference with source_arguments is that here the output is ready to be
        called by the
        function whose signature we have, since the position-only arguments will be
        returned as
        args.

        >>> foo(*args, **kwargs)
        10

        Note that though `w` is a position only argument, you can specify `w=4` as a
        keyword argument too (by default):

        >>> args, kwargs = Sig(foo).source_args_and_kwargs(
        ...     w=4, x=3, y=2, extra="keywords", are="ignored"
        ... )
        >>> assert (args, kwargs) == ((4,), {"x": 3, "y": 2})

        If you don't want to allow that, you can say `_ignore_kind=False`

        >>> Sig(foo).source_args_and_kwargs(
        ...     w=4, x=3, y=2, extra="keywords", are="ignored", _ignore_kind=False
        ... )
        Traceback (most recent call last):
          ...
        TypeError: ...'w'...

        You can use `_allow_partial` that will allow you, if
        set to `True`, to underspecify the params of a function (in view of being
        completed later).

        >>> Sig(foo).source_args_and_kwargs(x=3, y=2, extra="keywords", are="ignored")
        Traceback (most recent call last):
          ...
        TypeError:...'w'...

        But if you specify `_allow_partial=True`...

        >>> args, kwargs = Sig(foo).source_args_and_kwargs(
        ...     x=3, y=2, extra="keywords", are="ignored", _allow_partial=True
        ... )
        >>> (args, kwargs) == ((), {"x": 3, "y": 2})
        True

        By default, `_apply_defaults=False`, which will lead to only get those
        arguments you input.

        >>> args, kwargs = Sig(foo).source_args_and_kwargs(
        ...     4, x=3, y=2, extra="keywords", are="ignored"
        ... )
        >>> (args, kwargs) == ((4,), {"x": 3, "y": 2})
        True

        But if you specify `_apply_defaults=True` non-specified non-require arguments
        will be returned with their defaults:

        >>> args, kwargs = Sig(foo).source_args_and_kwargs(
        ...     4, x=3, y=2, extra="keywords", are="ignored", _apply_defaults=True
        ... )
        >>> (args, kwargs) == ((4,), {"x": 3, "y": 2, "z": 1})
        True
        """
        arguments = self.source_arguments(
            *args,
            _apply_defaults=_apply_defaults,
            _allow_partial=_allow_partial,
            _ignore_kind=_ignore_kind,
            **kwargs,
        )
        return self.mk_args_and_kwargs(
            arguments,
            allow_partial=_allow_partial,
            args_limit=_args_limit,
        )

    @property
    def inject_into_keyword_variadic(self):
        """
        Decorator that uses signature to source the keyword variadic of target function.

        See replace_kwargs_using function for more details, including examples.

        >>> def apple(a, x: int, y=2, *, z=3, **extra_apple_options):
        ...     return a + x + y + z
        >>> @Sig(apple).inject_into_keyword_variadic
        ... def sauce(a, b, c, **sauce_kwargs):
        ...     return b * c + apple(a, **sauce_kwargs)

        The function will works:

        >>> sauce(1, 2, 3, x=4, z=5)  # func still works? Should be: 1 + 4 + 2 + 5 + 2 * 3
        18

        But the signature now doesn't have the `**sauce_kwargs`, but more informative
        signature elements sourced from `apple`:

        >>> Sig(sauce)
        <Sig (a, b, c, *, x: int, y=2, z=3, **extra_apple_options)>
        """
        return replace_kwargs_using(self)


def _fill_defaults_and_annotations(sig1: Sig, sig2: Sig):
    """Return the same signature as ``sig1``, but where empty param properties
    (default or annotation) were filled by the property found in ``sig2`` if it has a
    param of the same name

    >>> _fill_defaults_and_annotations(
    ...    Sig('(a, /, b: str, *, c=3)'), Sig('(a: float, b: int = 2, c=300)')
    ... )
    <Sig (a: float, /, b: str = 2, *, c=3)>
    """

    def filled_properties_of_sig1():
        alt_defaults = sig2.defaults
        alt_annotations = sig2.annotations
        for p in sig1.params:
            yield Parameter(
                p.name,
                p.kind,
                default=(
                    p.default
                    if p.default is not empty
                    else alt_defaults.get(p.name, empty)
                ),
                annotation=(
                    p.annotation
                    if p.annotation is not empty
                    else alt_annotations.get(p.name, empty)
                ),
            )

    return Sig(filled_properties_of_sig1())


def _validate_sanity_of_signature_change(
    func: Callable, new_sig: Sig, ignore_incompatible_signatures: bool = True
):
    func_pos, func_kw = Sig(func)._positional_and_keyword_defaults()
    self_pos, self_kw = new_sig._positional_and_keyword_defaults()
    # print(func_pos, func_kw )
    # print(self_pos, self_kw)

    pos_default_switching_to_kw = set(func_pos) & set(self_kw)
    kw_default_switching_to_pos = set(func_kw) & set(self_pos)

    # print(pos_default_switching_to_kw, kw_default_switching_to_pos)

    if not ignore_incompatible_signatures and (
        pos_default_switching_to_kw or kw_default_switching_to_pos
    ):
        raise IncompatibleSignatures(
            f"Changing both the kind and the default of a param will result to "
            f"unexpected behaviors if the function is not properly wrapped to do so."
            f"If you really want to do this, inject signature using the "
            f"`ignore_incompatible_signatures=True`"
            f"argument in `Sig.wrap(...)`. "
            f"Alternatively, you can use `i2.wrapper` tools to have more control "
            f"over function defaults and signatures."
            f"The function you were wrapping had signature: "
            f"{name_of_obj(func) or ''}{Sig(func)} and "
            f"the signature you wanted to inject was {new_sig.name or ''}{new_sig}",
            sig1=Sig(func),
            sig2=new_sig,
        )


########################################################################################
# Utils


def _signature_differences_str_for_error_msg(sig1, sig2):
    from pprint import pformat

    sig_diff = sig1.pair_with(sig2)

    sig1_name = f"{sig1.name}" if sig1.name else "sig1"
    sig2_name = f"{sig2.name}" if sig2.name else "sig2"

    return (
        "FYI: Here are the raw signature differences for {sig1_name} and {sig2_name} "
        f"(not all need to necessarily be resolved):\n{pformat(sig_diff)}"
    )


############# Tools for testing #########################################################


def param_for_kind(
    name=None,
    kind="positional_or_keyword",
    with_default=False,
    annotation=Parameter.empty,
):
    """Function to easily and flexibly make inspect.Parameter objects for testing.

    It's annoying to have to compose parameters from scratch to testing things.
    This tool should help making it less annoying.

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
    """
    name = name or f"{kind}"
    kind_obj = getattr(Parameter, str(kind).upper())
    kind = str(kind_obj).lower()
    default = (
        f"dflt_{kind}"
        if with_default and kind not in {"var_positional", "var_keyword"}
        else Parameter.empty
    )
    return Parameter(name=name, kind=kind_obj, default=default, annotation=annotation)


param_kinds = list(filter(lambda x: x.upper() == x, Parameter.__dict__))

for kind in param_kinds:
    lower_kind = kind.lower()
    setattr(param_for_kind, lower_kind, partial(param_for_kind, kind=kind))
    setattr(
        param_for_kind,
        "with_default",
        partial(param_for_kind, with_default=True),
    )
    setattr(
        getattr(param_for_kind, lower_kind),
        "with_default",
        partial(param_for_kind, kind=kind, with_default=True),
    )
    setattr(
        getattr(param_for_kind, "with_default"),
        lower_kind,
        partial(param_for_kind, kind=kind, with_default=True),
    )


# --------------------------------------------------------------------------------------
# Back-compatibility re-exports (i2mint/dol#70)
#
# The signature comparison tools moved to ``dol.sig_compare`` and the recipes (call
# forgivingly, change parameter kinds, set signatures, ...) to ``dol.sig_recipes``. Both
# import ``Sig`` from this module, so these imports must stay at the bottom.

from dataclasses import dataclass  # noqa: E402  (importable from here before the split)

from dol.sig_compare import (  # noqa: E402
    Compared,
    Comparison,
    Comparator,
    SignatureComparator,
    ParamComparator,
    CallableComparator,
    ComparisonAggreg,
    CT,
    KeyFunction,
    compare_signatures,
    mk_func_comparator_based_on_signature_comparator,
    _keyed_comparator,
    keyed_comparator,
    _key_function_enabled_operator,
    _key_function_factory,
    param_comparator,
    param_binary_func,
    dflt1_is_empty_or_dflt2_is_not,
    ignore_any_differences,
    permissive_param_comparator,
    dflt1_is_empty_or_dflt2_is_not_param_comparator,
    return_tuple,
    param_attribute_dict,
    param_comparison_dict,
    param_differences_dict,
    defaults_are_the_same_when_not_empty,
    postprocess,
    is_call_compatible_with,
    SigPair,
)
from dol.sig_recipes import (  # noqa: E402
    mk_sig_from_args,
    _remove_variadics_from_sig,
    call_forgivingly,
    _call_forgivingly,
    call_somewhat_forgivingly,
    convert_to_PK,
    kind_forgiving_func,
    use_interface,
    has_signature,
    all_pk_signature,
    ch_signature_to_all_pk,
    normalized_func,
    ch_variadics_to_non_variadic_kind,
    tuple_the_args,
    ch_func_to_all_pk,
    copy_func,
    params_of,
    insert_annotations,
    common_and_diff_argnames,
    dflt_name_for_kind,
    arg_order_for_param_tuple,
    set_signature_of_func,
    sig_to_dataclass,
    replace_kwargs_using,
)
