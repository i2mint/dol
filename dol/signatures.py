"""Signature calculus: Tools to make it easier to work with function's signatures.

How to:

    - get names, kinds, defaults, annotations

    - merge two or more signatures

    - give a function a specific signature (with a choice of validations)

    - get an equivalent function with a different order of arguments

    - get an equivalent function with a subset of arguments (like partial)

    - get an equivalent function but with variadic *args and/or **kwargs replaced with
    non-variadic args (tuple) and kwargs (dict)

- make an f(a) function in to a f(a, b=None) function with b ignored

**Notes to the reader**

Both in the code and in the docs, we'll use short hands for parameter (argument) kind.

    - PK = Parameter.POSITIONAL_OR_KEYWORD

    - VP = Parameter.VAR_POSITIONAL

    - VK = Parameter.VAR_KEYWORD

    - PO = Parameter.POSITIONAL_ONLY

    - KO = Parameter.KEYWORD_ONLY

"""

from inspect import Signature, Parameter, signature, unwrap
from typing import Any, Union, Callable, Iterable, Mapping as MappingType
from types import FunctionType

from functools import (
    update_wrapper,
    partialmethod,
    partial,
    WRAPPER_ASSIGNMENTS,
    wraps as _wraps,
    update_wrapper as _update_wrapper,
)

# monkey patching WRAPPER_ASSIGNMENTS to get "proper" wrapping (adding defaults and
# kwdefaults
wrapper_assignments = (*WRAPPER_ASSIGNMENTS, '__defaults__', '__kwdefaults__')

update_wrapper = partial(_update_wrapper, assigned=wrapper_assignments)
wraps = partial(_wraps, assigned=wrapper_assignments)

_empty = Parameter.empty
empty = _empty

_ParameterKind = type(
    Parameter(name='param_kind', kind=Parameter.POSITIONAL_OR_KEYWORD)
)
ParamsType = Iterable[Parameter]
ParamsAble = Union[ParamsType, MappingType[str, Parameter], Callable]
SignatureAble = Union[Signature, Callable, ParamsType, MappingType[str, Parameter]]
HasParams = Union[Iterable[Parameter], MappingType[str, Parameter], Signature, Callable]

# short hands for Parameter kinds
PK = Parameter.POSITIONAL_OR_KEYWORD
VP, VK = Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD
PO, KO = Parameter.POSITIONAL_ONLY, Parameter.KEYWORD_ONLY
var_param_kinds = frozenset({VP, VK})
var_param_types = var_param_kinds  # Deprecate: for back-compatibility. Delete in 2021

DFLT_DEFAULT_CONFLICT_METHOD = 'strict'
param_attributes = {'name', 'kind', 'default', 'annotation'}


class FuncCallNotMatchingSignature(TypeError):
    """Raise when the call signature is not valid"""


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


def _param_sort_key(param):
    return (param.kind, param.default is not empty)


def sort_params(params):
    return sorted(params, key=_param_sort_key)


def name_of_obj(o: object) -> Union[str, None]:
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
    """
    if hasattr(o, '__name__'):
        return o.__name__
    elif hasattr(o, '__class__'):
        name = name_of_obj(o.__class__)
        if name == 'partial':
            if hasattr(o, 'func'):
                return name_of_obj(o.func)
        return name
    else:
        return None


def ensure_callable(obj: SignatureAble):
    if isinstance(obj, Callable):
        return obj
    else:

        def f(*args, **kwargs):
            """Empty function made just to carry a signature"""

        f.__signature__ = ensure_signature(obj)
        return f


assure_callable = ensure_callable  # alias for backcompatibility


def ensure_signature(obj: SignatureAble):
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
        dflt_and_annotation = dict(zip(['default', 'annotation'], r))
        return Param(name, PK, **dflt_and_annotation)
    else:
        raise TypeError(f"Don't know how to make {p} into a Parameter object")


def _params_from_mapping(mapping: MappingType):
    def gen():
        for k, v in mapping.items():
            if isinstance(v, MappingType):
                if 'name' in v:
                    assert v['name'] == k, (
                        f'In a mapping specification of a params, '
                        f"either the 'name' of the val shouldn't be specified, "
                        f'or it should be the same as the key ({k}): '
                        f'{dict(mapping)}'
                    )
                    yield v
                else:
                    yield dict(name=k, **v)
            else:
                assert isinstance(v, Parameter) and v.name == k
                yield v

    return list(gen())


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
        return list(signature(obj).parameters.values())
    except (TypeError, ValueError):
        if isinstance(obj, Iterable):
            if isinstance(obj, str):
                obj = [obj]
            # TODO: Can do better here! See attempt in _params_from_mapping:
            elif isinstance(obj, Mapping):
                obj = _params_from_mapping(obj)
                # obj = list(obj.values())
            elif isinstance(obj, tuple) and len(obj) in {1, 2, 3}:
                n = len(obj)
                if n == 1:
                    obj = [{'name': obj}]
                elif n == 2:
                    obj = [{'name': obj[0], 'default': obj[1]}]
                elif n == 2:
                    obj = [{'name': obj[0], 'default': obj[1], 'annotation': obj[2]}]
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
        if hasattr(obj, '__wrapped__'):
            obj = unwrap(obj, stop=(lambda f: hasattr(f, '__signature__')))
            return ensure_params(obj)
        else:  # if function didn't return at this point, it didn't find a match, so raise
            # a TypeError
            raise TypeError(
                f"Don't know how to make that object into an iterable of inspect.Parameter "
                f'objects: {obj}'
            )


assure_params = ensure_params  # alias for backcompatibility


class MissingArgValFor(object):
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
    what_to_do_with_remainding='return',
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
    ```
    args, kwargs, remaining = extract_arguments(kwargs, func)  # extract from kwargs
    what you need for func
    # ... check if remaing is empty (or not, depending on your paranoia), and then
    call the func:
    func(*args, **kwargs)
    ```
    (And if you doing that a lot: Do put it in a decorator!)

    See Also: extract_arguments.without_remainding

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
    ```
    param_kwargs.update(remaining_kwargs)
    ```
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
        `what_to_do_with_remainding='ignore'`, which will just return the (args,
        kwargs) pair
        `what_to_do_with_remainding='assert_empty'`, which will do the same, but first
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
    :param include_all_when_var_keywords_in_params=False,
    :param assert_no_missing_position_only_args=False,
    :param kwargs: The kwargs to extract the args from
    :return: A (param_args, param_kwargs, remaining_kwargs) tuple.
    """

    assert what_to_do_with_remainding in {'return', 'ignore', 'assert_empty'}
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
            next((p.name for p in params if p.kind == Parameter.VAR_KEYWORD), None,)
            is not None
        ):
            param_kwargs.update(remaining_kwargs)
            remaining_kwargs = {}

    if assert_no_missing_position_only_args:
        missing_argnames = tuple(
            x.argname for x in param_args if isinstance(x, MissingArgValFor)
        )
        assert (
            not missing_argnames
        ), f'There were some missing positional only argnames: {missing_argnames}'

    if what_to_do_with_remainding == 'return':
        return param_args, param_kwargs, remaining_kwargs
    elif what_to_do_with_remainding == 'ignore':
        return param_args, param_kwargs
    elif what_to_do_with_remainding == 'assert_empty':
        assert (
            len(remaining_kwargs) == 0
        ), f'remaining_kwargs not empty: remaining_kwargs={remaining_kwargs}'
        return param_args, param_kwargs


extract_arguments_ignoring_remainder = partial(
    extract_arguments, what_to_do_with_remainding='ignore'
)
extract_arguments_asserting_no_remainder = partial(
    extract_arguments, what_to_do_with_remainding='assert_empty'
)

from collections.abc import Mapping
from typing import Optional, Iterable
from dataclasses import dataclass


def function_caller(func, args, kwargs):
    return func(*args, **kwargs)


class Command:
    """A class that holds a `(caller, args, kwargs)` triple and allows one to execute
    `caller(*args, **kwargs)`

    :param func: A callable that will be called with (*args, **kwargs) argument.
    :param args: The positional arguments to call the func with.
    :param kwargs: The keyword arguments to call the func with.

    >>> c = Command(print, "hello", "world", sep=", ")
    >>> c()
    hello, world

    What happens (when a command is executed) if some of the arguments are commands
    themselves? Well, the sensible thing happens. These commands are executed.
    You can use this to define, declaratively, some pretty complex instructions, and
    only fetch the data you need and execute everything, once you're ready.

    >>> def show(a, b):
    ...     print(f"Showing this: {a=}, {b=}")
    >>> def take_five():
    ...     return 5
    >>> def double_val(val):
    ...     return val * 2
    >>> command = Command(
    ...     show,
    ...     Command(take_five),
    ...     b=Command(double_val, 'hello'),
    ... )
    >>> command
    Command(show, Command(take_five), b=Command(double_val, 'hello'))
    >>> command()
    Showing this: a=5, b='hellohello'

    Of course, as your use of Command gets more complex, you may want to subclass it
    and include some "validation" and "compilation" in the init.

    The usual way to call a function is to... erm... call it.
    But sometimes you want to do things differently.
    Like validate it, put it on a queue, etc.
    That's where specifying a different _caller will be useful.

    >>> class MyCommand(Command):
    ...     def _caller(self):
    ...         f, a, k = self.func, self.args, self.kwargs
    ...         print(f"Calling {f}(*{a}, **{k}) with result: {f(*a, **k)}")
    ...
    >>> c = MyCommand(print, "hello", "world", sep=", ")
    >>> c()
    hello, world
    Calling <built-in function print>(*('hello', 'world'), **{'sep': ', '}) with result: None

    """

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def curried(cls, func, **kw_defaults):
        """Get an Command maker for a specific function, with defaults and signature!

        >>> def foo(x: str, y: int):
        ...     return x * y
        ...
        >>> foo('hi', 3)
        'hihihi'
        >>>
        >>> foo_command = Command.curried(foo, y=2)
        >>> Sig(foo_command)
        <Sig (x: str, y: int = 2)>
        >>> f = foo_command('hi', y=4)
        >>> f()
        'hihihihi'
        >>> ff = foo_command('hi')
        >>> ff
        Command(foo, 'hi')
        >>> ff()
        'hihi'

        """
        sig = Sig(func)
        sig = sig.ch_defaults(**kw_defaults)

        if kw_defaults:
            func = partial(func, **kw_defaults)

        curried_command_cls = partial(Command, func)
        return sig(curried_command_cls)

    def __repr__(self):
        def to_str(x, quote="'"):
            if isinstance(x, str):
                return quote + x + quote
            else:
                return str(x)

        args_str = ', '.join(to_str(a) for a in self.args)
        kwargs_str = ', '.join(f'{k}={to_str(v)}' for k, v in self.kwargs.items())
        if args_str and kwargs_str:
            sep = ', '
        else:
            sep = ''
        args_kwargs_str = args_str + sep + kwargs_str

        func_name = name_of_obj(self.func)
        if args_kwargs_str:
            return f'{type(self).__name__}({func_name}, {args_kwargs_str})'
        else:
            return f'{type(self).__name__}({func_name})'

    def _caller(self):
        return self.func(*self.args, **self.kwargs)

    def _args_with_executed_commands(self):
        for v in self.args:
            if isinstance(v, Command):
                v = v()  # if a command, execute it
            yield v

    def _kwargs_with_executed_commands(self):
        for k, v in self.kwargs.items():
            if isinstance(v, Command):
                v = v()  # if a command, execute it
            yield k, v

    def _caller(self):
        return self.func(
            *self._args_with_executed_commands(),
            **dict(self._kwargs_with_executed_commands()),
        )

    def __call__(self):
        return self._caller()


def extract_commands(
    funcs: Iterable[Callable],
    *,
    mk_command: Callable[[Callable, tuple, dict], Any] = Command,
    what_to_do_with_remainding='ignore',
    **kwargs,
):
    """

    :param funcs: An iterable of functions
    :param mk_command: The function to make a command object
    :param kwargs: The argname=argval items that the functions should draw from.
    :return:

    >>> def add(a, b: float = 0.0) -> float:
    ...     return a + b
    >>> def mult(x: float, y=1):
    ...     return x * y
    >>> def formula1(w, /, x: float, y=1, *, z: int = 1):
    ...     return ((w + x) * y) ** z
    >>> commands = extract_commands(
    ...     (add, mult, formula1), a=1, b=2, c=3, d=4, e=5, w=6, x=7
    ... )
    >>> for command in commands:
    ...     print(
    ...         f"Calling {command.func.__name__} with "
    ...         f"args={command.args} and kwargs={command.kwargs}"
    ...     )
    ...     print(command())
    ...
    Calling add with args=() and kwargs={'a': 1, 'b': 2}
    3
    Calling mult with args=() and kwargs={'x': 7}
    7
    Calling formula1 with args=(6,) and kwargs={'x': 7}
    13
    """
    extract = partial(
        extract_arguments,
        what_to_do_with_remainding=what_to_do_with_remainding,
        include_all_when_var_keywords_in_params=False,
        assert_no_missing_position_only_args=True,
    )

    if callable(funcs):
        funcs = [funcs]

    for func in funcs:
        func_args, func_kwargs = extract(func, **kwargs)
        yield mk_command(func, *func_args, **func_kwargs)


def commands_dict(
    funcs,
    *,
    mk_command: Callable[[Callable, tuple, dict], Any] = Command,
    what_to_do_with_remainding='ignore',
    **kwargs,
):
    """

    :param funcs:
    :param mk_command:
    :param kwargs:
    :return:

    >>> def add(a, b: float = 0.0) -> float:
    ...     return a + b
    >>> def mult(x: float, y=1):
    ...     return x * y
    >>> def formula1(w, /, x: float, y=1, *, z: int = 1):
    ...     return ((w + x) * y) ** z
    >>> d = commands_dict((add, mult, formula1), a=1, b=2, c=3, d=4, e=5, w=6, x=7)
    >>> d[add]()
    3
    >>> d[mult]()
    7
    >>> d[formula1]()
    13

    """
    if callable(funcs):
        funcs = [funcs]
    it = extract_commands(
        funcs,
        what_to_do_with_remainding=what_to_do_with_remainding,
        mk_command=mk_command,
        **kwargs,
    )
    return dict(zip(funcs, it))


class Param(Parameter):
    # aliases
    PK = Parameter.POSITIONAL_OR_KEYWORD
    PO = Parameter.POSITIONAL_ONLY
    KO = Parameter.KEYWORD_ONLY
    VP = Parameter.VAR_POSITIONAL
    VK = Parameter.VAR_KEYWORD

    # OP = Parameter.POSITIONAL_ONLY
    # OK = Parameter.KEYWORD_ONLY

    def __init__(self, name, kind=PK, *, default=empty, annotation=empty):
        super().__init__(name, kind, default=default, annotation=annotation)

    # Note: Was useful to make Param a mapping, to get (dict(param))
    #  Is not useful anymore, so comment-deprecating
    # def __iter__(self):
    #     yield from ['name', 'kind', 'default', 'annotation']
    #
    # def __getitem__(self, k):
    #     return getattr(self, k)
    #
    # def __len__(self):
    #     return 4


P = Param  # useful shorthand alias


def param_has_default_or_is_var_kind(p: Parameter):
    return p.default != Parameter.empty or p.kind in var_param_kinds


WRAPPER_UPDATES = ('__dict__',)

from typing import Callable


def _robust_signature_of_callable(callable_obj: Callable) -> Signature:
    r"""Get the signature of a Callable, returning a custom made one for those
    builtins that don't have one

    >>> _robust_signature_of_callable(
    ...     _robust_signature_of_callable
    ... )  # has a normal signature
    <Signature (callable_obj: Callable) -> inspect.Signature>
    >>> s = _robust_signature_of_callable(print)  # has one that this module provides
    >>> assert isinstance(s, Signature)
    >>> # Will be: <Signature (*value, sep=' ', end='\n', file=<_io.TextIOWrapper
    name='<stdout>' mode='w' encoding='utf-8'>, flush=False)>
    >>> _robust_signature_of_callable(
    ...     zip
    ... )  # doesn't have one, so will return a blanket one
    <Signature (*no_sig_args, **no_sig_kwargs)>

    """
    try:
        return signature(callable_obj)
    except ValueError:
        # if isinstance(callable_obj, partial):
        #     callable_obj = callable_obj.func
        obj_name = getattr(callable_obj, '__name__', None)
        if obj_name in sigs_for_sigless_builtin_name:
            return sigs_for_sigless_builtin_name[obj_name] or signature(
                lambda *no_sig_args, **no_sig_kwargs: ...
            )
        else:
            raise


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

    # Positional/Keyword argument mapping

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
        - `kwargs_from_args_and_kwargs`: Map some args/kwargs input to a keyword-only
            expression of the inputs. This is useful if you need to do some processing
            based on the argument names.
        - `args_and_kwargs_from_kwargs`: Translate a fully keyword expression of some
            inputs into an (args, kwargs) pair that can be used to call the function.
            (Remember, your function can have constraints, so you may need to do this.

    The usual pattern of use of these methods is to use `kwargs_from_args_and_kwargs`
    to map all the inputs to their corresponding name, do what needs to be done with
    that (example, validation, transformation, decoration...) and then map back to an
    (args, kwargs) pair than can actually be used to call the function.

    Examples of methods and functions using these:
    `call_forgivingly`, `tuple_the_args`, `extract_kwargs`, `extract_args_and_kwargs`,
    `source_kwargs`, and `source_args_and_kwargs`.

    # Making a signature

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
    <Signature (w, i, /, a, x: float = 1, y=1, j=2, b=3.14, c: int = 42, *, z: int = 1)>
    >>>
    >>> sig = Sig(f) + g + ["a", ("b", 3.14), ("c", 42, int)] - "b" - ["a", "z"]
    >>> @sig
    ... def some_func(*args, **kwargs):
    ...     ...
    >>> inspect.signature(some_func)
    <Signature (w, i, x: float = 1, y=1, j=2, c: int = 42)>

    """

    def __init__(
        self,
        obj: ParamsAble = None,
        *,
        name=None,
        return_annotation=empty,
        __validate_parameters__=True,
    ):
        """Initialize a Sig instance.
        See Also: `ensure_params` to see what kind of objects you can make `Sig`s with.

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
        if (
            not isinstance(obj, Signature)
            and callable(obj)
            and return_annotation is empty
        ):
            return_annotation = _robust_signature_of_callable(obj).return_annotation
        super().__init__(
            ensure_params(obj),
            return_annotation=return_annotation,
            __validate_parameters__=__validate_parameters__,
        )
        self.name = name or name_of_obj(obj)

    # TODO: Add params for more validation (e.g. arg number/name matching?)
    def wrap(self, func: Callable, raise_on_error_copying_attrs=False):
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
        <Signature (w, x: int, y=2, z: int = 10)>
        >>> # But (unlike with functools.wraps) here we get __defaults__ and
        __kwdefault__
        >>> f.__defaults__  # see that x has no more default & z's default is now 10
        (2, 10)
        >>> f(
        ...     0, 1
        ... )  # see that now we get a different output because using different defaults
        1024

        TODO: Something goes wrong when using keyword only arguments.
            Note that the same problem occurs with functools.wraps, and even
            boltons.funcutils.wraps.

        >>> def f(w, /, x: float = 1, y=2, *, z: int = 3):
        ...     return w + x * y ** z
        >>> f(0)  # 0 + 1 * 2 ** 3
        8
        >>> f(0, 1, 2, 3)  # error expected!
        Traceback (most recent call last):
          ...
        TypeError: f() takes from 1 to 3 positional arguments but 4 were given
        >>> def g(w, x: int, y=2, *, z: int = 10):
        ...     return w + x * y ** z
        >>> s = Sig(g)
        >>> f = s.wrap(f)
        >>> f.__defaults__
        (2,)
        >>> f.__kwdefaults__
        {'z': 10}
        >>> f(0, 1, 2, 3)  # error not expected! TODO: Make it work!!
        Traceback (most recent call last):
          ...
        TypeError: f() takes from 2 to 3 positional arguments but 4 were given
        """

        # TODO: Would like to make a copy of the function so as to not override
        #  decorated function itself!
        # @wraps(func)
        # def wrapped_func(*args, **kwargs):
        #     return func(*args, **kwargs)
        wrapped_func = func

        wrapped_func.__signature__ = Signature(
            self.parameters.values(), return_annotation=self.return_annotation
        )
        wrapped_func.__annotations__ = self.annotations
        # endow the function with __defaults__ and __kwdefaults__ (not the default of
        # functools.wraps!)
        (
            wrapped_func.__defaults__,
            wrapped_func.__kwdefaults__,
        ) = self._dunder_defaults_and_kwdefaults()
        # "copy" over all other non-dunder attributes (not the default of
        # functools.wraps!)
        for attr in filter(lambda x: not x.startswith('__'), dir(wrapped_func)):
            try:
                setattr(wrapped_func, attr, getattr(wrapped_func, attr))
            except AttributeError as e:
                if raise_on_error_copying_attrs:
                    raise
        return wrapped_func

    def __call__(self, func: Callable):
        """Gives the input function the signature.
        Just calls Sig.wrap so see docs of Sig.wrap (which contains examples and
        doctests).
        """
        return self.wrap(func)

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

        >>> has_signature(print)
        False

        But we do get one with robust_has_signature

        >>> robust_has_signature(print)
        True

        """
        try:
            return (callable(obj) or None) and cls(obj)
        except ValueError:
            return None

    def __bool__(self):
        return True

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
        ko_names = self.names_for_kind(kind=KO)
        dflts = self.defaults
        return (
            tuple(dflts[name] for name in dflts if name not in ko_names),
            # as known as __defaults__ in python callables
            {
                name: dflts[name] for name in dflts if name in ko_names
            },  # as known as __kwdefaults__ in python callables
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
        ```
                {'parameters': self.parameters,
                'return_annotation': self.return_annotation}
        ```
        which would not actually work as keyword arguments of ``Signature``.
        Yeah, I know. Don't ask me, ask the authors of `Signature`!

        Instead, `parammeters` will be ``list(self.parameters.values())``, which does
        work.

        """
        return {
            'parameters': list(self.parameters.values()),
            'return_annotation': self.return_annotation,
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
            objs.append([{'name': name, 'kind': PK, 'default': default}])
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
        return {
            p.name: p.default for p in self.values() if p.default != Parameter.empty
        }

    @property
    def annotations(self):
        """{arg_name: annotation, ...} dict of annotations of the signature.
        What `func.__annotations__` would give you.
        """
        return {
            p.name: p.annotation
            for p in self.values()
            if p.annotation != Parameter.empty
        }

    # def substitute(self, **sub_for_name):
    #     def gen():
    #
    #         for name, substitution in sub_for_name.items():
    #

    def names_for_kind(self, kind):
        return tuple(p.name for p in self.values() if p.kind == kind)

    def __iter__(self):
        return iter(self.parameters)

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, k):
        return self.parameters[k]

    @property
    def has_var_kinds(self):
        return any(p.kind in var_param_kinds for p in self.values())

    @property
    def index_of_var_positional(self):
        """

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
        """

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
    def n_required(self):
        """The number of required arguments.
        A required argument is one that doesn't have a default, nor is VAR_POSITIONAL
        (*args) or VAR_KEYWORD (**kwargs).
        Note: Sometimes a minimum number of arguments in VAR_POSITIONAL and
        VAR_KEYWORD are in fact required,
        but we can't see this from the signature, so we can't tell you about that! You
        do the math.

        # Skipping the actual running of the doctest because some IDEs don't handle
        @property testing well.
        # >>> Sig(lambda x, y, z=None, *args, **kwargs: ...).n_required
        # 2
        """
        return (
            len(self)
            - len(self.defaults)
            - self.has_var_keyword
            - self.has_var_positional
        )

    def _transform_params(self, changes_for_name: dict):
        for name in self:
            if name in changes_for_name:
                p = changes_for_name[name]
                yield self[name].replace(**p)
            else:
                # if name is not in params, just use existing param
                yield self[name]

    def modified(self, _allow_reordering=False, **changes_for_name):
        """Returns a modified (new) signature object

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

        But be warned: This gives you a signature with all PK kinds.
        If you wrap a function with it, it will look like it has all PK kinds.
        But that doesn't mean you can actually use thenm as such.
        You'll need to modify (decorate further) your function further to reflect
        its new signature.

        On the other hand, if you decorate a function with a sig that adds or modifies
        defaults, these defaults will actually be used.

        """
        new_return_annotation = changes_for_name.pop(
            'return_annotation', self.return_annotation
        )

        if _allow_reordering:
            params = sort_params(self._transform_params(changes_for_name))
        else:
            params = list(self._transform_params(changes_for_name))

        return Sig(params, return_annotation=new_return_annotation)

    def ch_param_attrs(
        self, param_attr, *arg_new_vals, _allow_reordering=False, **kwargs_new_vals
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


        # TODO: Would like to make this work:
        # Now, if you want to set a default for a but not b and c for example, you'll
        # get complaints:
        #
        # ```
        # ValueError: non-default argument follows default argument
        # ```
        #
        # will tell you.
        #
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

        """

        if not param_attr in param_attributes:
            raise ValueError(
                f'param_attr needs to be one of: {param_attributes}.',
                f' Was: {param_attr}',
            )
        all_pk_self = self.modified(**{name: {'kind': PK} for name in self.names})
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

    def ch_names(self, **changes_for_name):
        return self.ch_param_attrs('name', **changes_for_name)

    def ch_kinds(self, **changes_for_name):
        return self.ch_param_attrs('kind', _allow_reordering=True, **changes_for_name)

    def ch_defaults(self, **changes_for_name):
        return self.ch_param_attrs(
            'default', _allow_reordering=True, **changes_for_name
        )

    def ch_annotations(self, **changes_for_name):
        return self.ch_param_attrs('annotation', **changes_for_name)

    def merge_with_sig(
        self,
        sig: ParamsAble,
        ch_to_all_pk: bool = False,
        *,
        default_conflict_method: str = DFLT_DEFAULT_CONFLICT_METHOD,
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

        _msg = f'\nHappened during an attempt to merge {self} and {sig}'

        assert not _self.has_var_keyword or not _sig.has_var_keyword, (
            f"Can't merge two signatures if they both have a VAR_POSITIONAL parameter:"
            f'{_msg}'
        )
        assert (
            not _self.has_var_keyword or not _sig.has_var_keyword
        ), "Can't merge two signatures if they both have a VAR_KEYWORD parameter:{_msg}"

        assert all(
            _self[name].kind == _sig[name].kind for name in _self.keys() & _sig.keys()
        ), (
            'During a signature merge, if two names are the same, they must have the '
            f'**same kind**:\n\t{_msg}\n'
            "Tip: If you're trying to merge functions in some way, consider decorating "
            'them with a signature mapping that avoids the argument name clashing'
        )

        assert default_conflict_method in {
            None,
            'strict',
            'take_first',
        }, "default_conflict_method should be in {None, 'strict', 'take_first'}"

        if default_conflict_method == 'take_first':
            _sig = _sig - set(_self.keys() & _sig.keys())

        if not all(
            _self[name].default == _sig[name].default
            for name in _self.keys() & _sig.keys()
        ):
            # if default_conflict_method == 'take_first':
            #     _sig = _sig - set(_self.keys() & _sig.keys())
            # else:
            raise ValueError(
                'During a signature merge, if two names are the same, they must have '
                'the '
                f'**same default**:\n\t{_msg}\n'
                "Tip: If you're trying to merge functions in some way, consider "
                'decorating '
                'them with a signature mapping that avoids the argument name clashing'
            )

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
            used.
            If a return_annotation is needed (say, for composition, the last
            return_annotation
            summed), one can subclass Sig and overwrite __add__
        - POSITION_ONLY and KEYWORD_ONLY kinds will be replaced by
        POSITIONAL_OR_KEYWORD kind.
        This is to simplify the interface and code.
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
        >>> Sig(f) + Sig(ff)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: During a signature merge, if two names are the same, they must
        have the **same default**:
        <BLANKSPACE>
        Happened during an attempt to merge (w, /, x: float = 1, y=1, *, z: int = 1)
        and (w, /, x: float, y=1, *, z: int = 1)
        Tip: If you're trying to merge functions in some way, consider decorating them
        with a signature mapping that avoids the argument name clashing


        >>> def hh(i, j, w=1):
        ...     ...  # like h, but w has a default
        ...
        >>> Sig(h) + Sig(hh)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: During a signature merge, if two names are the same, they must
        have the **same default**:
        <BLANKSPACE>
        Happened during an attempt to merge (i, j, w) and (i, j, w=1)
        Tip: If you're trying to merge functions in some way, consider decorating them
        with a signature mapping that avoids the argument name clashing


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
        >>> sum(sigs)
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
        """

        >>> list(Sig(lambda *args, a, b, x=1, y=1, **kwargs: ...).without_defaults)
        ['a', 'b']
        """
        return self.__class__(
            p for p in self.values() if not param_has_default_or_is_var_kind(p)
        )

    @property
    def with_defaults(self):
        """

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
        if add_defaults_if_necessary:
            if argname_to_default is None:

                def argname_to_default(argname):
                    return None

        def changed_params():
            there_was_a_default = False
            for p in self.parameters.values():
                if p.kind not in except_kinds:
                    # print(p.name)
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
                return type(self)(
                    sort_params(params), return_annotation=self.return_annotation
                )
            else:
                raise

    def kwargs_from_args_and_kwargs(
        self,
        args,
        kwargs,
        *,
        apply_defaults=False,
        allow_partial=False,
        allow_excess=False,
        ignore_kind=False,
    ):
        """Extracts a dict of input argument values for target signature, from args
        and kwargs.

        When you need to manage how the arguments of a function are specified,
        you need to take care of
        multiple cases depending on whether they were specified as positional arguments
        (`args`) or keyword arguments (`kwargs`).

        The `kwargs_from_args_and_kwargs` (and it's sorta-inverse inverse,
        `args_and_kwargs_from_kwargs`)
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

        That's where  `Sig.extract_kwargs(*args, **kwargs)` is needed.
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
        :return: An {argname: argval, ...} dict

        See also the sorta-inverse of this function: args_and_kwargs_from_kwargs

        >>> def foo(w, /, x: float, y="YY", *, z: str = "ZZ"):
        ...     ...
        >>> sig = Sig(foo)
        >>> assert (
        ...     sig.kwargs_from_args_and_kwargs((11, 22, "you"), dict(z="zoo"))
        ...     == sig.kwargs_from_args_and_kwargs((11, 22), dict(y="you", z="zoo"))
        ...     == {"w": 11, "x": 22, "y": "you", "z": "zoo"}
        ... )

        By default, `apply_defaults=False`, which will lead to only get those
        arguments you input.

        >>> sig.kwargs_from_args_and_kwargs(args=(11,), kwargs={"x": 22})
        {'w': 11, 'x': 22}

        But if you specify `apply_defaults=True` non-specified non-require arguments
        will be returned with their defaults:

        >>> sig.kwargs_from_args_and_kwargs(
        ...     args=(11,), kwargs={"x": 22}, apply_defaults=True
        ... )
        {'w': 11, 'x': 22, 'y': 'YY', 'z': 'ZZ'}

        By default, `ignore_excess=False`, so specifying kwargs that are not in the
        signature will lead to an exception.

        >>> sig.kwargs_from_args_and_kwargs(
        ...     args=(11,), kwargs={"x": 22, "not_in_sig": -1}
        ... )
        Traceback (most recent call last):
            ...
        TypeError: Got unexpected keyword arguments: not_in_sig

        Specifying `allow_excess=True` will ignore such excess fields of kwargs.
        This is useful when you want to source several functions from a same dict.

        >>> sig.kwargs_from_args_and_kwargs(
        ...     args=(11,), kwargs={"x": 22, "not_in_sig": -1}, allow_excess=True
        ... )
        {'w': 11, 'x': 22}

        On the other side of `ignore_excess` you have `allow_partial` that will allow
        you, if
        set to `True`, to underspecify the params of a function (in view of being
        completed later).

        >>> sig.kwargs_from_args_and_kwargs(args=(), kwargs={"x": 22})
        Traceback (most recent call last):
          ...
        TypeError: missing a required argument: 'w'

        But if you specify `allow_partial=True`...

        >>> sig.kwargs_from_args_and_kwargs(
        ...     args=(), kwargs={"x": 22}, allow_partial=True
        ... )
        {'x': 22}

        That's a lot of control (eight combinations total), but not everything is
        controllable here:
        Position only and keyword only kinds need to be respected:

        >>> sig.kwargs_from_args_and_kwargs(args=(1, 2, 3, 4), kwargs={})
        Traceback (most recent call last):
          ...
        TypeError: too many positional arguments
        >>> sig.kwargs_from_args_and_kwargs(args=(), kwargs=dict(w=1, x=2, y=3, z=4))
        Traceback (most recent call last):
          ...
        TypeError: 'w' parameter is positional only, but was passed as a keyword

        But if you want to ignore the kind of parameter, just say so:

        >>> sig.kwargs_from_args_and_kwargs(
        ...     args=(1, 2, 3, 4), kwargs={}, ignore_kind=True
        ... )
        {'w': 1, 'x': 2, 'y': 3, 'z': 4}
        >>> sig.kwargs_from_args_and_kwargs(
        ...     args=(), kwargs=dict(w=1, x=2, y=3, z=4), ignore_kind=True
        ... )
        {'w': 1, 'x': 2, 'y': 3, 'z': 4}
        """
        no_var_kw = not self.has_var_keyword

        if ignore_kind:
            sig = self.normalize_kind(
                # except_kinds=frozenset()
            )
        else:
            sig = self

        # no_var_kw = not sig.has_var_keyword
        if no_var_kw:  # has no var keyword kinds
            sig_relevant_kwargs = {
                name: kwargs[name] for name in sig if name in kwargs
            }  # take only what you need
        else:
            sig_relevant_kwargs = kwargs  # take all the kwargs

        binder = sig.bind_partial if allow_partial else sig.bind
        if not self.has_var_positional and allow_excess:
            max_allowed_num_of_posisional_args = sum(
                k <= PK for k in self.kinds.values()
            )
            args = args[:max_allowed_num_of_posisional_args]

        b = binder(*args, **sig_relevant_kwargs)
        if apply_defaults:
            b.apply_defaults()

        if no_var_kw and not allow_excess:  # don't ignore excess kwargs
            excess = kwargs.keys() - b.arguments
            if excess:
                excess_str = ', '.join(excess)
                raise TypeError(f'Got unexpected keyword arguments: {excess_str}')

        return dict(b.arguments)
        # not doing it as dict(b.arguments) because order can be different.
        # return {name: b.arguments[name] for name in self.names if name in b.arguments}

    def args_and_kwargs_from_kwargs(
        self,
        kwargs,
        apply_defaults=False,
        allow_partial=False,
        allow_excess=False,
        ignore_kind=False,
        args_limit: Union[int, None] = 0,
    ):
        """Extract args and kwargs such that func(*args, **kwargs) can be called,
        where func has instance's signature.

        :param kwargs: The {argname: argval,...} dict to process
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
        >>> args, kwargs = foo_sig.args_and_kwargs_from_kwargs(
        ...     dict(w=4, x=3, y=2, z=1)
        ... )
        >>> assert (args, kwargs) == ((4,), {"x": 3, "y": 2, "z": 1})
        >>> assert foo(*args, **kwargs) == foo(4, 3, 2, z=1) == 14

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

        >>> kwargs = dict(w=4, x=3, y=2, z=1)
        >>> foo_sig.args_and_kwargs_from_kwargs(kwargs, args_limit=0)
        ((4,), {'x': 3, 'y': 2, 'z': 1})

        If `args_limit is None`, the least kwargs (keyword arguments) will be returned.

        >>> foo_sig.args_and_kwargs_from_kwargs(kwargs, args_limit=None)
        ((4, 3, 2), {'z': 1})

        If `args_limit` is a positive integer, the first `args_limit` arguments
        will be returned (not checking at all if this is valid!).

        >>> foo_sig.args_and_kwargs_from_kwargs(kwargs, args_limit=1)
        ((4,), {'x': 3, 'y': 2, 'z': 1})
        >>> foo_sig.args_and_kwargs_from_kwargs(kwargs, args_limit=2)
        ((4, 3), {'y': 2, 'z': 1})
        >>> foo_sig.args_and_kwargs_from_kwargs(kwargs, args_limit=3)
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

        By default, only the arguments that were given in the kwargs input will be
        returned in the (args, kwargs) output.
        If you also want to get those that have defaults (according to signature),
        you need to specify it with the `apply_defaults=True` argument.

        >>> foo_sig.args_and_kwargs_from_kwargs(dict(w=4, x=3))
        ((4,), {'x': 3})
        >>> foo_sig.args_and_kwargs_from_kwargs(dict(w=4, x=3), apply_defaults=True)
        ((4,), {'x': 3, 'y': 1, 'z': 1})

        By default, all required arguments must be given.
        Not doing so will lead to a `TypeError`.
        If you want to process your arguments anyway, specify `allow_partial=True`.

        >>> foo_sig.args_and_kwargs_from_kwargs(dict(w=4))
        Traceback (most recent call last):
          ...
        TypeError: missing a required argument: 'x'
        >>> foo_sig.args_and_kwargs_from_kwargs(dict(w=4), allow_partial=True)
        ((4,), {})

        Specifying argument names that are not recognized by the signature will
        lead to a `TypeError`.
        If you want to avoid this (and just take from the input `kwargs` what ever you
        can), specify this with `allow_excess=True`.

        >>> foo_sig.args_and_kwargs_from_kwargs(dict(w=4, x=3, extra='stuff'))
        Traceback (most recent call last):
            ...
        TypeError: Got unexpected keyword arguments: extra
        >>> foo_sig.args_and_kwargs_from_kwargs(dict(w=4, x=3, extra='stuff'),
        ...     allow_excess=True)
        ((4,), {'x': 3})

        An edge case: When a `VAR_POSITIONAL` follows a `POSITION_OR_KEYWORD`...

        >>> Sig(lambda a, *b, c=2: None).args_and_kwargs_from_kwargs(
        ...     {"a": 1, "b": [2, 3], "c": 4}
        ... )
        ((1, [2, 3]), {'c': 4})

        See `kwargs_from_args_and_kwargs` (namely for the description of the arguments.
        """

        if args_limit is None:
            # Take the maximum number of args (positional arguments).
            # The only kwargs (keyword arguments) you should have are keyword-only
            # and var-keyword arguments.
            idx = next((i for i, p in enumerate(self.params) if p.kind > VP), None)
            names_for_args = self.names[:idx]
        elif args_limit == 0:
            # Take the minimum number possible of args (positional arguments)
            # Only those that are position only or before a var-positional.
            vp_idx = self.index_of_var_positional
            if vp_idx is None:
                names_for_args = self.names_for_kind(PO)
            else:
                # When there's a VP present, all arguments before it can only be
                # expressed positionally if the VP argument is non-empty.
                # So, here we just consider all arguments positionally up to the VP arg.
                names_for_args = self.names[: (vp_idx + 1)]
        else:
            names_for_args = self.names[:args_limit]

        args = tuple(kwargs[name] for name in names_for_args if name in kwargs)
        kwargs = {name: kwargs[name] for name in kwargs if name not in names_for_args}

        kwargs = self.kwargs_from_args_and_kwargs(
            args,
            kwargs,
            apply_defaults=apply_defaults,
            allow_partial=allow_partial,
            allow_excess=allow_excess,
            ignore_kind=ignore_kind,
        )
        kwargs = {name: kwargs[name] for name in kwargs if name not in names_for_args}

        return args, kwargs

    def extract_kwargs(
        self,
        *args,
        _ignore_kind=True,
        _allow_partial=False,
        _apply_defaults=False,
        **kwargs,
    ):
        """Convenience method that calls kwargs_from_args_and_kwargs with defaults,
        and ignore_kind=True.

        Strict in the sense that the kwargs cannot contain any arguments that are not
        valid argument names (as per the signature).

        >>> def foo(w, /, x: float, y="YY", *, z: str = "ZZ"):
        ...     ...
        >>> sig = Sig(foo)
        >>> assert (
        ...     sig.extract_kwargs(1, 2, 3, z=4)
        ...     == sig.extract_kwargs(1, 2, y=3, z=4)
        ...     == {"w": 1, "x": 2, "y": 3, "z": 4}
        ... )

        What about var positional and var keywords?

        >>> def bar(*args, **kwargs):
        ...     ...
        ...
        >>> Sig(bar).extract_kwargs(1, 2, y=3, z=4)
        {'args': (1, 2), 'kwargs': {'y': 3, 'z': 4}}

        Note that though `w` is a position only argument, you can specify `w=11` as
        a keyword argument too (by default):

        >>> Sig(foo).extract_kwargs(w=11, x=22)
        {'w': 11, 'x': 22}

        If you don't want to allow that, you can say `_ignore_kind=False`

        >>> Sig(foo).extract_kwargs(w=11, x=22, _ignore_kind=False)
        Traceback (most recent call last):
          ...
        TypeError: 'w' parameter is positional only, but was passed as a keyword

        You can use `_allow_partial` that will allow you, if
        set to `True`, to underspecify the params of a function
        (in view of being completed later).

        >>> Sig(foo).extract_kwargs(x=3, y=2)
        Traceback (most recent call last):
          ...
        TypeError: missing a required argument: 'w'

        But if you specify `_allow_partial=True`...

        >>> Sig(foo).extract_kwargs(x=3, y=2, _allow_partial=True)
        {'x': 3, 'y': 2}

        By default, `_apply_defaults=False`, which will lead to only get those arguments
        you input.

        >>> Sig(foo).extract_kwargs(4, x=3, y=2)
        {'w': 4, 'x': 3, 'y': 2}

        But if you specify `_apply_defaults=True` non-specified non-require arguments
        will be returned with their defaults:

        >>> Sig(foo).extract_kwargs(4, x=3, y=2, _apply_defaults=True)
        {'w': 4, 'x': 3, 'y': 2, 'z': 'ZZ'}
        """
        return self.kwargs_from_args_and_kwargs(
            args,
            kwargs,
            apply_defaults=_apply_defaults,
            allow_partial=_allow_partial,
            allow_excess=False,
            ignore_kind=_ignore_kind,
        )

    def extract_args_and_kwargs(
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
        >>> args, kwargs = Sig(foo).extract_args_and_kwargs(4, x=3, y=2)
        >>> (args, kwargs) == ((4,), {"x": 3, "y": 2})
        True

        The difference with extract_kwargs is that here the output is ready to be
        called by the
        function whose signature we have, since the position-only arguments will be
        returned as
        args.

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
        TypeError: 'w' parameter is positional only, but was passed as a keyword

        You can use `_allow_partial` that will allow you, if
        set to `True`, to underspecify the params of a function (in view of being
        completed later).

        >>> Sig(foo).extract_args_and_kwargs(x=3, y=2)
        Traceback (most recent call last):
          ...
        TypeError: missing a required argument: 'w'

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
        kwargs = self.extract_kwargs(
            *args,
            _ignore_kind=_ignore_kind,
            _allow_partial=_allow_partial,
            _apply_defaults=_apply_defaults,
            **kwargs,
        )
        return self.args_and_kwargs_from_kwargs(
            kwargs,
            allow_partial=_allow_partial,
            apply_defaults=_apply_defaults,
            args_limit=_args_limit,
        )

    def source_kwargs(
        self,
        *args,
        _ignore_kind=True,
        _allow_partial=False,
        _apply_defaults=False,
        **kwargs,
    ):
        """Source the kwargs for the signature instance, ignoring excess arguments.

        >>> def foo(w, /, x: float, y="YY", *, z: str = "ZZ"):
        ...     ...
        >>> Sig(foo).source_kwargs(11, x=22, extra="keywords", are="ignored")
        {'w': 11, 'x': 22}

        Note that though `w` is a position only argument, you can specify `w=11` as a
        keyword argument too (by default):

        >>> Sig(foo).source_kwargs(w=11, x=22, extra="keywords", are="ignored")
        {'w': 11, 'x': 22}

        If you don't want to allow that, you can say `_ignore_kind=False`

        >>> Sig(foo).source_kwargs(
        ...     w=11, x=22, extra="keywords", are="ignored", _ignore_kind=False
        ... )
        Traceback (most recent call last):
          ...
        TypeError: 'w' parameter is positional only, but was passed as a keyword

        You can use `_allow_partial` that will allow you, if
        set to `True`, to underspecify the params of a function (in view of being
        completed later).

        >>> Sig(foo).source_kwargs(x=3, y=2, extra="keywords", are="ignored")
        Traceback (most recent call last):
          ...
        TypeError: missing a required argument: 'w'

        But if you specify `_allow_partial=True`...

        >>> Sig(foo).source_kwargs(
        ...     x=3, y=2, extra="keywords", are="ignored", _allow_partial=True
        ... )
        {'x': 3, 'y': 2}

        By default, `_apply_defaults=False`, which will lead to only get those
        arguments you input.

        >>> Sig(foo).source_kwargs(4, x=3, y=2, extra="keywords", are="ignored")
        {'w': 4, 'x': 3, 'y': 2}

        But if you specify `_apply_defaults=True` non-specified non-require arguments
        will be returned with their defaults:

        >>> Sig(foo).source_kwargs(
        ...     4, x=3, y=2, extra="keywords", are="ignored", _apply_defaults=True
        ... )
        {'w': 4, 'x': 3, 'y': 2, 'z': 'ZZ'}
        """
        return self.kwargs_from_args_and_kwargs(
            args,
            kwargs,
            apply_defaults=_apply_defaults,
            allow_partial=_allow_partial,
            allow_excess=True,
            ignore_kind=_ignore_kind,
        )

    def source_args_and_kwargs(
        self,
        *args,
        _ignore_kind=True,
        _allow_partial=False,
        _apply_defaults=False,
        **kwargs,
    ):
        """Source the (args, kwargs) for the signature instance, ignoring excess
        arguments.

        >>> def foo(w, /, x: float, y=2, *, z: int = 1):
        ...     return w + x * y ** z
        >>> args, kwargs = Sig(foo).source_args_and_kwargs(
        ...     4, x=3, y=2, extra="keywords", are="ignored"
        ... )
        >>> assert (args, kwargs) == ((4,), {"x": 3, "y": 2})
        >>>

        The difference with source_kwargs is that here the output is ready to be
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
        TypeError: 'w' parameter is positional only, but was passed as a keyword

        You can use `_allow_partial` that will allow you, if
        set to `True`, to underspecify the params of a function (in view of being
        completed later).

        >>> Sig(foo).source_args_and_kwargs(x=3, y=2, extra="keywords", are="ignored")
        Traceback (most recent call last):
          ...
        TypeError: missing a required argument: 'w'

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
        kwargs = self.kwargs_from_args_and_kwargs(
            args,
            kwargs,
            allow_excess=True,
            ignore_kind=_ignore_kind,
            allow_partial=_allow_partial,
            apply_defaults=_apply_defaults,
        )
        return self.args_and_kwargs_from_kwargs(
            kwargs,
            allow_excess=True,
            ignore_kind=_ignore_kind,
            allow_partial=_allow_partial,
            apply_defaults=_apply_defaults,
        )


########################################################################################
# Recipes


def mk_sig_from_args(*args_without_default, **args_with_defaults):
    """Make a Signature instance by specifying args_without_default and
    args_with_defaults.

    >>> mk_sig_from_args("a", "b", c=1, d="bar")
    <Signature (a, b, c=1, d='bar')>
    """
    assert all(
        isinstance(x, str) for x in args_without_default
    ), 'all default-less arguments must be strings'
    return Sig.from_objs(
        *args_without_default, **args_with_defaults
    ).to_simple_signature()


def call_forgivingly(func, *args, **kwargs):
    """
    Call function on given args and kwargs, but only taking what the function needs
    (not choking if they're extras variables)

    >>> def foo(a, b: int = 0, c=None) -> int:
    ...     return "foo", (a, b, c)
    >>> call_forgivingly(
    ...     foo,  # the function you want to call
    ...     "input for a",  # meant for a -- the first (and only) argument foo requires
    ...     c=42,  # skiping b and giving c a non-default value
    ...     intruder="argument",  # but wait, this argument name doesn't exist! Oh no!
    ... )  # well, as it happens, nothing bad -- the intruder argument is just ignored
    ('foo', ('input for a', 0, 42))

    """
    args, kwargs = Sig(func).source_args_and_kwargs(*args, **kwargs)
    return func(*args, **kwargs)


def call_somewhat_forgivingly(
    func, args, kwargs, enforce_sig: Optional[SignatureAble] = None
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
    if enforce_sig:
        if enforce_sig is True:
            enforce_sig = Sig(func)  # enforce the func's signature
            # this should be the same constraint level as calling the function itself.
        else:
            enforce_sig = Sig(enforce_sig)
        _kwargs = enforce_sig.bind(*args, **kwargs).arguments
        return call_forgivingly(func, **_kwargs)
    else:
        return call_forgivingly(func, *args, **kwargs)


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
    ```
    def my_g(a, b):
        return f(a)
    ```
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
    ... )
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


def number_of_required_arguments(obj):
    sig = Sig(obj)
    return len(sig) - len(sig.defaults)


# TODO: Need to define and use this function more carefully.
#   Is the goal to remove positional? Remove variadics? Normalize the signature?
def all_pk_signature(callable_or_signature: Signature):
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
    <Signature (w, x: float, y=1, z: int = 1, **kwargs)>

    But note that the variadic arguments *args and **kwargs remain variadic:

    >>> all_pk_signature(signature(bar))
    <Signature (*args, **kwargs)>

    It works with `Sig` too (since Sig is a Signature), and maintains it's other
    attributes (like name).

    >>> sig = all_pk_signature(Sig(bar))
    >>> sig
    <Sig (*args, **kwargs)>
    >>> sig.name
    'bar'

    """

    if isinstance(callable_or_signature, Signature):
        sig = callable_or_signature

        last_kind = -1

        def changed_params():
            for p in sig.parameters.values():
                if p.kind not in var_param_kinds:
                    yield p.replace(kind=PK)
                else:
                    yield p

        new_sig = type(sig)(
            list(changed_params()), return_annotation=sig.return_annotation
        )
        for attrname, attrval in getattr(sig, '__dict__', {}).items():
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
    """A decorator that will change a VAR_POSITIONAL (*args) argument to a tuple (args)
    argument of the same name.

    Essentially, given a `func(a, *b, c, **d)` function want to get a
    `new_func(a, b=(), c=None, d={})` that has the same functionality
    (in fact, calls the original `func` function behind the scenes), but without
    where the variadic arguments *b and **d are replaced with a `b` expecting an
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
    VARIADIC_KEYWORD kinds (**kwargs) alone, you can do so by setting
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
            _args, _kwargs = sig.args_and_kwargs_from_kwargs(kwargs, allow_partial=True)
            # print(sig, kwargs, _args, _kwargs)
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
                        'There should be only keyword arguments after the Variadic '
                        'args. '
                        f'Function was called with (positional={args}, keywords='
                        f'{_kwargs})'
                    )
            else:
                a, _vp_args_ = args, ()

            # extract from the remaining _kwargs, the dict corresponding to the
            # variadic keywords, if any, since these need to be **-ed later
            _var_keyword_kwargs = _kwargs.pop(var_keyword_argname, {})

            if ch_variadic_keyword_to_keyword:
                # an extra level of extraction is needed in this case
                _var_keyword_kwargs = _var_keyword_kwargs.pop(var_keyword_argname, {})
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
            variadic_less_func.__signature__ = Signature(
                params, return_annotation=signature(func).return_annotation
            )
        except ValueError:
            if idx_of_vp is not None:
                params[idx_of_vp] = params[idx_of_vp].replace(kind=PK)
            variadic_less_func.__signature__ = Signature(
                params, return_annotation=signature(func).return_annotation
            )

        return variadic_less_func
    else:
        return func


tuple_the_args = partial(
    ch_variadics_to_non_variadic_kind, ch_variadic_keyword_to_keyword=False
)
tuple_the_args.__name__ = 'tuple_the_args'
tuple_the_args.__doc__ = '''
A decorator that will change a VAR_POSITIONAL (*args) argument to a tuple (args)
argument of the same name.
'''


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

    # >>> def h(x, *y, z):
    # ...     print(f"{x=}, {y=}, {z=}")
    # >>> h(1, 2, 3, z=4)
    # x=1, y=(2, 3), z=4
    # >>> hh = ch_func_to_all_pk(h)
    # >>> hh(1, (2, 3), z=4)
    # x=1, y=(2, 3), z=4
    """

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
    if hasattr(f, '__signature__'):
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
    assert all(
        isinstance(p, Parameter) for p in obj
    ), 'obj needs to be a Iterable[Parameter] at this point'
    return obj  # as is


########################################################################################################################
# TODO: Encorporate in Sig
def insert_annotations(s: Signature, *, return_annotation=empty, **annotations):
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
        f'{set(annotations) - set(s.parameters)}'
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

    Returns: A dict with fields 'common', 'func1_not_func2', and 'func2_not_func1'

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
        'common': [x for x in p1 if x in p2],
        'func1_not_func2': [x for x in p1 if x not in p2],
        'func2_not_func1': [x for x in p2 if x not in p1],
    }


dflt_name_for_kind = {
    Parameter.VAR_POSITIONAL: 'args',
    Parameter.VAR_KEYWORD: 'kwargs',
}

arg_order_for_param_tuple = ('name', 'default', 'annotation', 'kind')


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


########################################################################################################################
# Manual construction of missing signatures
# ############################################################################

import sys

sigs_for_sigless_builtin_name = {
    '__build_class__': None,
    # __build_class__(func, name, /, *bases, [metaclass], **kwds) -> class
    '__import__': None,
    # __import__(name, globals=None, locals=None, fromlist=(), level=0) -> module
    'bool': None,
    # bool(x) -> bool
    'breakpoint': None,
    # breakpoint(*args, **kws)
    'bytearray': None,
    # bytearray(iterable_of_ints) -> bytearray
    # bytearray(string, encoding[, errors]) -> bytearray
    # bytearray(bytes_or_buffer) -> mutable copy of bytes_or_buffer
    # bytearray(int) -> bytes array of size given by the parameter initialized with
    # null bytes
    # bytearray() -> empty bytes array
    'bytes': None,
    # bytes(iterable_of_ints) -> bytes
    # bytes(string, encoding[, errors]) -> bytes
    # bytes(bytes_or_buffer) -> immutable copy of bytes_or_buffer
    # bytes(int) -> bytes object of size given by the parameter initialized with null
    # bytes
    # bytes() -> empty bytes object
    'classmethod': None,
    # classmethod(function) -> method
    'dict': None,
    # dict() -> new empty dictionary
    # dict(mapping) -> new dictionary initialized from a mapping object's
    # dict(iterable) -> new dictionary initialized as if via:
    # dict(**kwargs) -> new dictionary initialized with the name=value pairs
    'dir': None,
    # dir([object]) -> list of strings
    'filter': None,
    # filter(function or None, iterable) --> filter object
    'frozenset': None,
    # frozenset() -> empty frozenset object
    # frozenset(iterable) -> frozenset object
    'getattr': None,
    # getattr(object, name[, default]) -> value
    'int': None,
    # int([x]) -> integer
    # int(x, base=10) -> integer
    'iter': None,
    # iter(iterable) -> iterator
    # iter(callable, sentinel) -> iterator
    'map': signature(lambda func, *iterables: ...),
    # map(func, *iterables) --> map object
    'max': None,
    # max(iterable, *[, default=obj, key=func]) -> value
    # max(arg1, arg2, *args, *[, key=func]) -> value
    'min': None,
    # min(iterable, *[, default=obj, key=func]) -> value
    # min(arg1, arg2, *args, *[, key=func]) -> value
    'next': None,
    # next(iterator[, default])
    'print': signature(
        lambda *value, sep=' ', end='\n', file=sys.stdout, flush=False: ...
    ),
    # print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)
    'range': None,
    # range(stop) -> range object
    # range(start, stop[, step]) -> range object
    'set': None,
    # set() -> new empty set object
    # set(iterable) -> new set object
    'slice': None,
    # slice(stop)
    # slice(start, stop[, step])
    'staticmethod': None,
    # staticmethod(function) -> method
    'str': None,
    # str(object='') -> str
    # str(bytes_or_buffer[, encoding[, errors]]) -> str
    'super': None,
    # super() -> same as super(__class__, <first argument>)
    # super(type) -> unbound super object
    # super(type, obj) -> bound super object; requires isinstance(obj, type)
    # super(type, type2) -> bound super object; requires issubclass(type2, type)
    'type': None,
    # type(object_or_name, bases, dict)
    # type(object) -> the object's type
    # type(name, bases, dict) -> a new type
    'vars': None,
    # vars([object]) -> dictionary
    'zip': None,
    # zip(*iterables) --> A zip object yielding tuples until an input is exhausted.
}

############# Tools for testing
# ########################################################################################
from functools import partial


def param_for_kind(
    name=None,
    kind='positional_or_keyword',
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
    name = name or f'{kind}'
    kind_obj = getattr(Parameter, str(kind).upper())
    kind = str(kind_obj).lower()
    default = (
        f'dflt_{kind}'
        if with_default and kind not in {'var_positional', 'var_keyword'}
        else Parameter.empty
    )
    return Parameter(name=name, kind=kind_obj, default=default, annotation=annotation)


param_kinds = list(filter(lambda x: x.upper() == x, Parameter.__dict__))

for kind in param_kinds:
    lower_kind = kind.lower()
    setattr(param_for_kind, lower_kind, partial(param_for_kind, kind=kind))
    setattr(
        param_for_kind, 'with_default', partial(param_for_kind, with_default=True),
    )
    setattr(
        getattr(param_for_kind, lower_kind),
        'with_default',
        partial(param_for_kind, kind=kind, with_default=True),
    )
    setattr(
        getattr(param_for_kind, 'with_default'),
        lower_kind,
        partial(param_for_kind, kind=kind, with_default=True),
    )
