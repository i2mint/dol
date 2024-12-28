"""General util objects"""

import os
import shutil
import re
import platform
from collections import deque, namedtuple, defaultdict
from warnings import warn

from typing import (
    Any,
    Hashable,
    Callable,
    Iterable,
    Optional,
    Union,
    Mapping,
    Sequence,
    T,
    NewType,
    Tuple,
    TypeVar,
)
from functools import update_wrapper as _update_wrapper
from functools import wraps as _wraps
from functools import partialmethod, partial, WRAPPER_ASSIGNMENTS
from types import MethodType, FunctionType
from inspect import Signature, signature, Parameter, getsource, ismethod


Key = TypeVar("Key")
Key.__doc__ = "The type of the keys used in the interface (outer keys)"
Id = TypeVar("Id")
Id.__doc__ = "The type of the keys used in the backend (inner keys)"
Val = TypeVar("Val")
Val.__doc__ = "The type of the values used in the interface (outer values)"
Data = TypeVar("Data")
Data.__doc__ = "The type of the values used in the backend (inner values)"
Item = Tuple[Key, Val]
KeyIter = Iterable[Key]
ValIter = Iterable[Val]
ItemIter = Iterable[Item]

# monkey patching WRAPPER_ASSIGNMENTS to get "proper" wrapping (adding defaults and kwdefaults
wrapper_assignments = (*WRAPPER_ASSIGNMENTS, "__defaults__", "__kwdefaults__")

update_wrapper = partial(_update_wrapper, assigned=wrapper_assignments)
wraps = partial(_wraps, assigned=wrapper_assignments)

exhaust = partial(deque, maxlen=0)


def safe_compile(path):
    r"""
    Safely compiles a file path into a regex pattern, ensuring compatibility
    across different operating systems (Windows, macOS, Linux).

    This function normalizes the input path to use the correct separators
    for the current platform and escapes any special characters to avoid
    invalid regex patterns.

    Args:
        path (str): The file path to be compiled into a regex pattern.

    Returns:
        re.Pattern: A compiled regular expression object for the given path.

    Examples:
        >>> regex = safe_compile(r"C:\\what\\happens\\if\\you\\escape")
        >>> regex.pattern  # Windows path is escaped properly
        'C:\\\\what\\\\happens\\\\if\\\\you\\\\escape'

        >>> regex = safe_compile("/fun/paths/are/awesome")
        >>> regex.pattern  # Unix path is unmodified
        '/fun/paths/are/awesome'
    """
    # Normalize the path to handle cross-platform differences
    normalized_path = os.path.normpath(path)
    if platform.system() == "Windows":
        # Escape backslashes for Windows paths
        normalized_path = re.escape(normalized_path)
    return re.compile(normalized_path)


# TODO: Make identity_func "identifiable". If we use the following one, we can use == to detect it's use,
# TODO: ... but there may be a way to annotate, register, or type any identity function so it can be detected.
def identity_func(x: T) -> T:
    return x


static_identity_method = staticmethod(identity_func)


def named_partial(func, *args, __name__=None, **keywords):
    """functools.partial, but with a __name__

    >>> f = named_partial(print, sep='\\n')
    >>> f.__name__
    'print'

    >>> f = named_partial(print, sep='\\n', __name__='now_partial_has_a_name')
    >>> f.__name__
    'now_partial_has_a_name'
    """
    f = partial(func, *args, **keywords)
    f.__name__ = __name__ or func.__name__
    return f


def is_classmethod(obj):
    """Checks if an object is a classmethod.

    Args:
        obj: The object to check.

    Returns:
        True if the object is a classmethod, False otherwise.

    Example usage:

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
    """

    return ismethod(obj) and isinstance(obj.__self__, type)


def is_unbound_method(obj):
    """
    Determines if the given object is an unbound method.

    Args:
        obj: The object to check.

    Returns:
        True if obj is an unbound method, False otherwise.

    Examples:
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
    """
    if not isinstance(obj, FunctionType):
        return False
    qualname = getattr(obj, "__qualname__", "")
    # if '<locals>' in qualname:
    #     return False
    return "." in qualname


class staticproperty:
    """A decorator for defining static properties in classes.

    >>> class A:
    ...     @staticproperty
    ...     def foo():
    ...         return 2
    >>> A.foo
    2
    >>> A().foo
    2
    """

    def __init__(self, function):
        self.function = function

    def __get__(self, obj, owner=None):
        return self.function()


def add_as_attribute_of(obj, name=None):
    """Decorator that adds a function as an attribute of a container object ``obj``.

    If no ``name`` is given, the ``__name__`` of the function will be used, with a
    leading underscore removed. This is useful for adding helper functions to main
    "container" functions without polluting the namespace of the module, at least
    from the point of view of imports and tab completion.

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

    In reality, any object that has a ``__name__`` can be added to the attribute of
    ``obj``, but the intention is to add helper functions to main "container" functions.

    """

    def _decorator(f):
        attrname = name or f.__name__
        if attrname.startswith("_"):
            attrname = attrname[1:]  # remove leading underscore
        setattr(obj, attrname, f)
        return f

    return _decorator


def chain_get(d: Mapping, keys, default=None):
    """
    Returns the ``d[key]`` value for the first ``key`` in ``keys`` that is in ``d``, and default if none are found

    Note: Think of ``collections.ChainMap`` where you can look for a single key in a sequence of maps until we find it.
    Here we look for a sequence of keys in a single map, stopping as soon as we find a key that the map has.

    >>> d = {'here': '&', 'there': 'and', 'every': 'where'}
    >>> chain_get(d, ['not there', 'not there either', 'there', 'every'])
    'and'

    Notice how ``'not there'`` and ``'not there either'`` are skipped, ``'there'`` is found and used to retrieve
    the value, and ``'every'`` is not even checked (because ``'there'`` was found).
    If non of the keys are found, ``None`` is returned by default.

    >>> assert chain_get(d, ('none', 'of', 'these')) is None

    You can change this default though:

    >>> chain_get(d, ('none', 'of', 'these'), default='Not Found')
    'Not Found'

    """
    for key in keys:
        if key in d:
            return d[key]
    return default


class LiteralVal:
    """An object to indicate that the value should be considered literally.

    >>> t = LiteralVal(42)
    >>> t.get_val()
    42
    >>> t()
    42

    """

    def __init__(self, val):
        self.val = val

    def get_val(self):
        """Get the value wrapped by LiteralVal instance.

        One might want to use ``literal.get_val()`` instead ``literal()`` to get the
        value a ``LiteralVal`` is wrapping because ``.get_val`` is more explicit.

        That said, with a bit of hesitation, we allow the ``literal()`` form as well
        since it is useful in situations where we need to use a callback function to
        get a value.
        """
        return self.val

    __call__ = get_val

    # def __get__(self, instance, owner):
    #     return self.val


# TODO: The a.big() test is skipped because fails in doctest. It should be fixed.
def decorate_callables(decorator, cls=None):
    """Decorate all (non-underscored) callables in a class with a decorator.

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
    >>> a.big()  # doctest: +SKIP
    'small'

    """
    if cls is None:
        return partial(decorate_callables, decorator)
    for name, attr in vars(cls).items():
        if isinstance(attr, LiteralVal):
            setattr(cls, name, attr.get_val())
        elif not name.startswith("_") and callable(attr):
            setattr(cls, name, decorator(attr))
    return cls


# class LiteralVal:
#     """
#     An object to indicate that the value should be considered literally.

#     >>> t = LiteralVal(42)
#     >>> t.get_val()
#     42
#     >>> t()
#     42

#     >>> class A:
#     ...     @LiteralVal
#     ...     def value(self):
#     ...         return 42
#     >>> a = A()
#     >>> a.value
#     42
#     """

#     def __init__(self, val):
#         if callable(val):
#             self.val = val()
#         else:
#             self.val = val

#     def get_val(self):
#         """Get the value wrapped by LiteralVal instance."""
#         return self.val

#     def __call__(self):
#         return self.get_val()

#     def __get__(self, instance, owner):
#         return self.val

# def decorate_callables(decorator, cls=None):
#     """
#     Decorate all (non-underscored) callables in a class with a decorator.

#     >>> @decorate_callables(property)
#     ... class A:
#     ...     def wet(self):
#     ...         return 'dry'
#     ...     @LiteralVal
#     ...     def big(self):
#     ...         return 'small'
#     >>> a = A()
#     >>> a.wet
#     'dry'
#     >>> a.big
#     'small'
#     """
#     if cls is None:
#         return partial(decorate_callables, decorator)
#     for name, attr in vars(cls).items():
#         if isinstance(attr, LiteralVal):
#             setattr(cls, name, property(attr.get_val))
#         elif not name.startswith('_') and callable(attr):
#             setattr(cls, name, decorator(attr))
#     return cls


def _isinstance(obj, class_or_tuple):
    """The same as the builtin isinstance, but without the position only restriction,
    allowing us to use partial to define filter functions for specific types
    """
    return isinstance(obj, class_or_tuple)


def instance_checker(*types):
    """Makes a filter function that checks the type of an object.

    >>> f = instance_checker(int, float)
    >>> f(1)
    True
    >>> f(1.0)
    True
    >>> f('1.0')
    False
    """
    return partial(_isinstance, class_or_tuple=types)


def not_a_mac_junk_path(path: str):
    """A function that will tell you if the path is not a mac junk path/
    More precisely, doesn't end with '.DS_Store' or have a `__MACOSX` folder somewhere
    on it's way.

    This is usually meant to be used with `filter` or `filt_iter` to "filter in" only
    those actually wanted files (not the junk that mac writes to your filesystem).

    These files annoyingly show up often in zip files, and are usually unwanted.

    See https://apple.stackexchange.com/questions/239578/compress-without-ds-store-and-macosx

    >>> paths = ['A/normal/path', 'A/__MACOSX/path', 'path/ending/in/.DS_Store', 'foo/b']
    >>> list(filter(not_a_mac_junk_path, paths))
    ['A/normal/path', 'foo/b']
    """
    if path.endswith(".DS_Store") or "__MACOSX" in path.split(os.path.sep):
        return False  # This is indeed math junk (so filter out)
    return True  # this is not mac junk (you can keep it)


def inject_method(obj, method_function, method_name=None):
    """
    method_function could be:
        * a function
        * a {method_name: function, ...} dict (for multiple injections)
        * a list of functions or (function, method_name) pairs
    """
    if method_name is None:
        method_name = method_function.__name__
    assert callable(
        method_function
    ), f"method_function (the second argument) is supposed to be a callable!"
    assert isinstance(
        method_name, str
    ), f"method_name (the third argument) is supposed to be a string!"
    if not isinstance(obj, type):
        method_function = MethodType(method_function, obj)
    setattr(obj, method_name, method_function)
    return obj


def _disabled_clear_method(self):
    """The clear method is disabled to make dangerous difficult.
    You don't want to delete your whole DB
    If you really want to delete all your data, you can do so by doing something like this:

    .. code-block:: python

        for k in self:
            del self[k]


    or (in some cases)

    .. code-block:: python

        for k in self:
            try:
                del self[k]
            except KeyError:
                pass

    """
    raise NotImplementedError(f"Instance of {type(self)}: {self.clear.__doc__}")


# to be able to check if clear is disabled (see ensure_clear_method function for example):
_disabled_clear_method.disabled = True


def has_enabled_clear_method(store):
    """Returns True iff obj has a clear method that is enabled (i.e. not disabled)"""
    return hasattr(store, "clear") and (  # has a clear method...
        not hasattr(store.clear, "disabled")  # that doesn't have a disabled attribute
        or not store.clear.disabled
    )  # ... or if it does, than it must not be == True


def _delete_keys_one_by_one(self):
    """clear the entire store (delete all keys)"""
    for k in self:
        del self[k]


def _delete_keys_one_by_one_with_keyerror_supressed(self):
    """clear the entire store (delete all keys), ignoring KeyErrors"""
    for k in self:
        try:
            del self[k]
        except KeyError:
            pass


_delete_keys_one_by_one.disabled = False
_delete_keys_one_by_one_with_keyerror_supressed.disabled = False


# Note: Vendored in i2.multi_objects and lkj.strings
def truncate_string_with_marker(
    s, *, left_limit=15, right_limit=15, middle_marker="..."
):
    """
    Return a string with a limited length.

    If the string is longer than the sum of the left_limit and right_limit,
    the string is truncated and the middle_marker is inserted in the middle.

    If the string is shorter than the sum of the left_limit and right_limit,
    the string is returned as is.

    >>> truncate_string_with_marker('1234567890')
    '1234567890'

    But if the string is longer than the sum of the limits, it is truncated:

    >>> truncate_string_with_marker('1234567890', left_limit=3, right_limit=3)
    '123...890'
    >>> truncate_string_with_marker('1234567890', left_limit=3, right_limit=0)
    '123...'
    >>> truncate_string_with_marker('1234567890', left_limit=0, right_limit=3)
    '...890'

    If you're using a specific parametrization of the function often, you can
    create a partial function with the desired parameters:

    >>> from functools import partial
    >>> truncate_string = partial(truncate_string_with_marker, left_limit=2, right_limit=2, middle_marker='---')
    >>> truncate_string('1234567890')
    '12---90'
    >>> truncate_string('supercalifragilisticexpialidocious')
    'su---us'

    """
    middle_marker_len = len(middle_marker)
    if len(s) <= left_limit + right_limit:
        return s
    elif right_limit == 0:
        return s[:left_limit] + middle_marker
    elif left_limit == 0:
        return middle_marker + s[-right_limit:]
    else:
        return s[:left_limit] + middle_marker + s[-right_limit:]


def signature_string_or_default(func, default="(-no signature-)"):
    try:
        return str(signature(func))
    except ValueError:
        return default


def function_info_string(func: Callable):
    func_name = getattr(func, "__name__", str(func))
    if func_name == "<lambda>":
        return f"a lambda function on {signature(func)}"
    return f"{func_name}{signature_string_or_default(func)}"


# Note: Pipe code is completely independent (with inspect imports signature & Signature)
#  If you only need simple pipelines, use this, or even copy/paste it where needed.
# TODO: Public interface mis-aligned with i2. funcs list here, in i2 it's dict. Align?
#  If we do so, it would be a breaking change since any dependents that expect funcs
#  to be a list of funcs will iterate over a iterable of names instead.
class Pipe:
    """Simple function composition. That is, gives you a callable that implements input -> f_1 -> ... -> f_n -> output.

    >>> def foo(a, b=2):
    ...     return a + b
    >>> f = Pipe(foo, lambda x: print(f"x: {x}"))
    >>> f(3)
    x: 5
    >>> len(f)
    2

    You can name functions, but this would just be for documentation purposes.
    The names are completely ignored.

    >>> g = Pipe(
    ...     add_numbers = lambda x, y: x + y,
    ...     multiply_by_2 = lambda x: x * 2,
    ...     stringify = str
    ... )
    >>> g(2, 3)
    '10'
    >>> len(g)
    3

    Notes:
        - Pipe instances don't have a __name__ etc. So some expectations of normal functions are not met.
        - Pipe instance are pickalable (as long as the functions that compose them are)

    You can specify a single functions:

    >>> Pipe(lambda x: x + 1)(2)
    3

    but

    >>> Pipe()
    Traceback (most recent call last):
      ...
    ValueError: You need to specify at least one function!

    You can specify an instance name and/or doc with the special (reserved) argument
    names ``__name__`` and ``__doc__`` (which therefore can't be used as function names):

    >>> f = Pipe(map, add_it=sum, __name__='map_and_sum', __doc__='Apply func and add')
    >>> f(lambda x: x * 10, [1, 2, 3])
    60
    >>> f.__name__
    'map_and_sum'
    >>> f.__doc__
    'Apply func and add'

    """

    funcs = ()

    def __init__(self, *funcs, **named_funcs):
        named_funcs = self._process_reserved_names(named_funcs)
        funcs = list(funcs) + list(named_funcs.values())
        self.funcs = funcs
        n_funcs = len(funcs)
        if n_funcs == 0:
            raise ValueError("You need to specify at least one function!")

        elif n_funcs == 1:
            other_funcs = ()
            first_func = last_func = funcs[0]
        else:
            first_func, *other_funcs = funcs
            *_, last_func = other_funcs

        self.__signature__ = Pipe._signature_from_first_and_last_func(
            first_func, last_func
        )
        self.first_func, self.other_funcs = first_func, other_funcs

    _reserved_names = ("__name__", "__doc__")

    def _process_reserved_names(self, named_funcs):
        for name in self._reserved_names:
            if (value := named_funcs.pop(name, None)) is not None:
                setattr(self, name, value)
        return named_funcs

    def __call__(self, *args, **kwargs):
        out = self.first_func(*args, **kwargs)
        try:  # first call has no exeption handling, but subsequent calls do
            for i, func in enumerate(self.other_funcs, 1):
                out = func(out)
        except Exception as e:
            raise self._mk_pipe_call_error(e, i, out, args, kwargs) from e
        return out

    def _mk_pipe_call_error(self, error_obj, i, out, args, kwargs):
        msg = f"Error calling function {self._func_info_str(i)}\n"
        out_str = f"{out}"
        msg += f"on input {truncate_string_with_marker(out_str)}\n"
        msg += "which was the output of previous function "
        msg += f"\t{self._func_info_str(i - 1)}\n"
        args_str = ", ".join(map(str, args))
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        msg += f"The error was cause by calling {self} on ({args_str}, {kwargs_str})\n"
        msg += f"Error was: {error_obj}"
        new_error_obj = type(error_obj)(msg)
        new_error_obj.error_context = {
            "Pipe": self,
            "args": args,
            "kwargs": kwargs,
            "func_index": i,
            "func": self.funcs[i],
            "func_input": out,
        }
        return new_error_obj

    def _func_info_str(self, i):
        func = self.funcs[i]
        func_info = function_info_string(func)
        return f"{func_info} (index={i})"

    def __len__(self):
        return len(self.funcs)

    _dflt_signature = Signature.from_callable(lambda *args, **kwargs: None)

    @staticmethod
    def _signature_from_first_and_last_func(first_func, last_func):
        try:
            input_params = signature(first_func).parameters.values()
        except ValueError:  # function doesn't have a signature, so take default
            input_params = Pipe._dflt_signature.parameters.values()
        try:
            return_annotation = signature(last_func).return_annotation
        except ValueError:  # function doesn't have a signature, so take default
            return_annotation = Pipe._dflt_signature.return_annotation
        return Signature(tuple(input_params), return_annotation=return_annotation)


def _flatten_pipe(pipe):
    for func in pipe.funcs:
        if isinstance(func, Pipe):
            yield from _flatten_pipe(func)
        else:
            yield func


def flatten_pipe(pipe):
    """
    Unravel nested Pipes to get a flat 'sequence of functions' version of input.

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
    """
    return Pipe(*_flatten_pipe(pipe))


def partialclass(cls, *args, **kwargs):
    """What partial(cls, *args, **kwargs) does, but returning a class instead of an object.

    :param cls: Class to get the partial of
    :param kwargs: The kwargs to fix

    The raison d'être of partialclass is that it returns a type, so let's have a look at that with
    a useless class.

    >>> from inspect import signature
    >>> class A:
    ...     pass
    >>> assert isinstance(A, type) == isinstance(partialclass(A), type) == True

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

    Note in the last partial that since ``a`` was fixed, you need to specify the keyword ``AA(a=3)``.
    ``AA(3)`` won't work:

    >>> AA(3)  # doctest: +SKIP
    Traceback (most recent call last):
      ...
    TypeError: __init__() got multiple values for argument 'a'

    On the other hand, you can use *args to specify the fixtures:

    >>> AA = partialclass(A, 22)
    >>> assert str(AA()) == 'A(a=22, b=1)'
    >>> assert str(signature(AA)) == '(b=1)'
    >>> assert str(AA(3)) == 'A(a=22, b=3)'


    """
    assert isinstance(cls, type), f"cls should be a type, was a {type(cls)}: {cls}"

    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    copy_attrs(
        PartialClass,
        cls,
        attrs=("__name__", "__qualname__", "__module__", "__doc__"),
    )

    return PartialClass


def copy_attrs(target, source, attrs, raise_error_if_an_attr_is_missing=True):
    """Copy attributes from one object to another.

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

    But if you try to copy something that `B` (the source) doesn't have, copy_attrs will complain:

    >>> copy_attrs(A, B, 'this_is_not_an_attr')
    Traceback (most recent call last):
        ...
    AttributeError: type object 'B' has no attribute 'this_is_not_an_attr'

    If you tell it not to complain, it'll just ignore attributes that are not in source.

    >>> copy_attrs(A, B, ['nothing', 'here', 'exists'], raise_error_if_an_attr_is_missing=False)
    >>> dict_of(A)
    {'x': 1, 'yy': 2, 'zzz': 3}
    """
    if isinstance(attrs, str):
        attrs = (attrs,)
    if raise_error_if_an_attr_is_missing:
        filt = lambda a: True
    else:
        filt = lambda a: hasattr(source, a)
    for a in filter(filt, attrs):
        setattr(target, a, getattr(source, a))


def copy_attrs_from(from_obj, to_obj, attrs):
    from warnings import warn

    warn(f"Deprecated. Use copy_attrs instead.", DeprecationWarning)
    copy_attrs(to_obj, from_obj, attrs)
    return to_obj


def norm_kv_filt(kv_filt: Callable[[Any], bool]):
    """Prepare a boolean function to be used with `filter` when fed an iterable of (k, v) pairs.

    So you have a mapping. Say a dict `d`. Now you want to go through d.items(),
    filtering based on the keys, or the values, or both.

    It's not hard to do, really. If you're using a dict you might use a dict comprehension,
    or in the general case you might do a `filter(lambda kv: my_filt(kv[0], kv[1]), d.items())`
    if you have a `my_filt` that works wiith k and v, etc.

    But thought simple, it can become a bit muddled.
    `norm_kv_filt` simplifies this by allowing you to bring your own filtering boolean function,
    whether it's a key-based, value-based, or key-value-based one, and it will make a
    ready-to-use with `filter` function for you.

    Only thing: Your function needs to call a key `k` and a value `v`.
    But hey, it's alright, if you have a function that calls things differently, just do
    something like

    .. code-block:: python

        new_filt_func = lambda k, v: your_filt_func(..., key=k, ..., value=v, ...)

    and all will be fine.

    :param kv_filt: callable (starting with signature (k), (v), or (k, v)), and returning  a boolean
    :return: A normalized callable.

    >>> d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    >>> list(filter(norm_kv_filt(lambda k: k in {'b', 'd'}), d.items()))
    [('b', 2), ('d', 4)]
    >>> list(filter(norm_kv_filt(lambda v: v > 2), d.items()))
    [('c', 3), ('d', 4)]
    >>> list(filter(norm_kv_filt(lambda k, v: (v > 1) & (k != 'c')), d.items()))
    [('b', 2), ('d', 4)]
    """
    if kv_filt is None:
        return None  # because `filter` works with a callable, or None, so we align

    raise_msg = (
        f"kv_filt should be callable (starting with signature (k), (v), or (k, v)),"
        "and returning  a boolean. What you gave me was {fv_filt}"
    )
    assert callable(kv_filt), raise_msg

    params = list(signature(kv_filt).parameters.values())
    assert len(params), raise_msg
    _kv_filt = kv_filt
    if params[0].name == "v":

        def kv_filt(k, v):
            return _kv_filt(v)

    elif params[0].name == "k":
        if len(params) > 1:
            if params[1].name != "v":
                raise ValueError(raise_msg)
        else:

            def kv_filt(k, v):
                return _kv_filt(k)

    else:
        raise ValueError(raise_msg)

    def __kv_filt(kv_item):
        return kv_filt(*kv_item)

    __kv_filt.__name__ = kv_filt.__name__

    return __kv_filt


var_str_p = re.compile(r"\W|^(?=\d)")

Item = Any


def add_attrs(remember_added_attrs=True, if_attr_exists="raise", **attrs):
    """Make a function that will add attributes to an obj.
    Originally meant to be used as a decorator of a function, to inject

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
    """

    def add_attrs_to_func(obj):
        attrs_added = []
        for attr_name, attr_val in attrs.items():
            if hasattr(obj, attr_name):
                if if_attr_exists == "raise":
                    raise AttributeError(
                        f"Attribute {attr_name} already exists in {obj}"
                    )
                elif if_attr_exists == "warn":
                    warn(f"Attribute {attr_name} already exists in {obj}")
                elif if_attr_exists == "skip":
                    continue
                else:
                    raise ValueError(
                        f"Unknown value for if_attr_exists: {if_attr_exists}"
                    )
            setattr(obj, attr_name, attr_val)
            attrs_added.append(attr_name)

        if remember_added_attrs:
            obj._added_attrs = attrs_added

        return obj

    return add_attrs_to_func


def fullpath(path):
    if path.startswith("~"):
        path = os.path.expanduser(path)
    return os.path.abspath(path)


def attrs_of(obj):
    return set(dir(obj))


def format_invocation(name="", args=(), kwargs=None):
    """Given a name, positional arguments, and keyword arguments, format
    a basic Python-style function call.

    >>> print(format_invocation('func', args=(1, 2), kwargs={'c': 3}))
    func(1, 2, c=3)
    >>> print(format_invocation('a_func', args=(1,)))
    a_func(1)
    >>> print(format_invocation('kw_func', kwargs=[('a', 1), ('b', 2)]))
    kw_func(a=1, b=2)

    """
    kwargs = kwargs or {}
    a_text = ", ".join([repr(a) for a in args])
    if isinstance(kwargs, dict):
        kwarg_items = [(k, kwargs[k]) for k in sorted(kwargs)]
    else:
        kwarg_items = kwargs
    kw_text = ", ".join(["%s=%r" % (k, v) for k, v in kwarg_items])

    all_args_text = a_text
    if all_args_text and kw_text:
        all_args_text += ", "
    all_args_text += kw_text

    return "%s(%s)" % (name, all_args_text)


def groupby(
    items: Iterable[Item],
    key: Callable[[Item], Hashable],
    val: Optional[Callable[[Item], Any]] = None,
    group_factory=list,
) -> dict:
    """Groups items according to group keys updated from those items through the given
    (item_to_)key function.

    Args:
        items: iterable of items
        key: The function that computes a key from an item. Needs to return a hashable.
        val: An optional function that computes a val from an item. If not given, the item itself will be taken.
        group_factory: The function to make new (empty) group objects and accumulate group items.
            group_items = group_factory() will be called to make a new empty group collection
            group_items.append(x) will be called to add x to that collection
            The default is `list`

    Returns: A dict of {group_key: items_in_that_group, ...}

    See Also: regroupby, itertools.groupby, and dol.source.SequenceKvReader

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
    """
    groups = defaultdict(group_factory)
    if val is None:
        for item in items:
            groups[key(item)].append(item)
    else:
        for item in items:
            groups[key(item)].append(val(item))
    return dict(groups)


def regroupby(items, *key_funcs, **named_key_funcs):
    """Recursive groupby. Applies the groupby function recursively, using a sequence of key functions.

    Note: The named_key_funcs argument names don't have any external effect.
        They just give a name to the key function, for code reading clarity purposes.

    See Also: groupby, itertools.groupby, and dol.source.SequenceKvReader

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
    """
    key_funcs = list(key_funcs) + list(named_key_funcs.values())
    assert len(key_funcs) > 0, "You need to have at least one key_func"
    if len(key_funcs) == 1:
        return groupby(items, key=key_funcs[0])
    else:
        key_func, *key_funcs = key_funcs
        groups = groupby(items, key=key_func)
        return {
            group_key: regroupby(group_items, *key_funcs)
            for group_key, group_items in groups.items()
        }


Groups = dict
GroupKey = Hashable
GroupItems = Iterable[Item]
GroupReleaseCond = Union[
    Callable[[GroupKey, GroupItems], bool],
    Callable[[Groups, GroupKey, GroupItems], bool],
]


def igroupby(
    items: Iterable[Item],
    key: Callable[[Item], GroupKey],
    val: Optional[Callable[[Item], Any]] = None,
    group_factory: Callable[[], GroupItems] = list,
    group_release_cond: GroupReleaseCond = lambda k, v: False,
    release_remainding=True,
    append_to_group_items: Callable[[GroupItems, Item], Any] = list.append,
    grouper_mapping=defaultdict,
):
    """The generator version of dol groupby.
    Groups items according to group keys updated from those items through the given (item_to_)key function,
    yielding the groups according to a logic defined by ``group_release_cond``

    Args:
        items: iterable of items
        key: The function that computes a key from an item. Needs to return a hashable.
        val: An optional function that computes a val from an item. If not given, the item itself will be taken.
        group_factory: The function to make new (empty) group objects and accumulate group items.
            group_items = group_collector() will be called to make a new empty group collection
            group_items.append(x) will be called to add x to that collection
            The default is `list`
        group_release_cond: A boolean function that will be applied, at every iteration,
            to the accumulated items of the group that was just updated,
            and determines (if True) if the (group_key, group_items) should be yielded.
            The default is False, which results in
            ``lambda group_key, group_items: False`` being used.
        release_remainding: Once the input items have been consumed, there may still be some
            items in the grouping "cache". ``release_remainding`` is a boolean that indicates whether
            the contents of this cache should be released or not.

    Yields: ``(group_key, items_in_that_group)`` pairs


    The following will group numbers according to their parity (0 for even, 1 for odd),
    releasing a list of numbers collected when that list reaches length 3:

    >>> g = igroupby(items=range(11),
    ...             key=lambda x: x % 2,
    ...             group_release_cond=lambda k, v: len(v) == 3)
    >>> list(g)
    [(0, [0, 2, 4]), (1, [1, 3, 5]), (0, [6, 8, 10]), (1, [7, 9])]

    If we specify ``release_remainding=False`` though, we won't get

    >>> g = igroupby(items=range(11),
    ...             key=lambda x: x % 2,
    ...             group_release_cond=lambda k, v: len(v) == 3,
    ...             release_remainding=False)
    >>> list(g)
    [(0, [0, 2, 4]), (1, [1, 3, 5]), (0, [6, 8, 10])]

    # >>> grps = partial(igroupby, group_release_cond=False, release_remainding=True)


    Below we show that, with the default ``group_release_cond = lambda k, v: False``
    and release_remainding=True`` we have ``dict(igroupby(...)) == groupby(...)``

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

    """
    groups = grouper_mapping(group_factory)

    assert callable(group_release_cond), (
        "group_release_cond should be callable (filter boolean function) or False. "
        f"Was {group_release_cond}"
    )
    n_group_release_cond_args = len(signature(group_release_cond).parameters)
    assert n_group_release_cond_args in {2, 3}, (
        "group_release_cond should take two or three inputs:\n"
        " - (group_key, group_items), or\n"
        " - (groups, group_key, group_items)"
        f"The arguments of the function you gave me are: {signature(group_release_cond)}"
    )

    if val is None:
        _append_to_group_items = append_to_group_items
    else:
        _append_to_group_items = lambda group_items, item: (
            group_items,
            val(item),
        )

    for item in items:
        group_key = key(item)
        group_items = groups[group_key]
        _append_to_group_items(group_items, item)

        if group_release_cond(group_key, group_items):
            yield group_key, group_items
            del groups[group_key]

    if release_remainding:
        for group_key, group_items in groups.items():
            yield group_key, group_items


def ntup(**kwargs):
    return namedtuple("NamedTuple", list(kwargs))(**kwargs)


def str_to_var_str(s: str) -> str:
    """Make a valid python variable string from the input string.
    Left untouched if already valid.

    >>> str_to_var_str('this_is_a_valid_var_name')
    'this_is_a_valid_var_name'
    >>> str_to_var_str('not valid  #)*(&434')
    'not_valid_______434'
    >>> str_to_var_str('99_ballons')
    '_99_ballons'
    """
    return var_str_p.sub("_", s)


def fill_with_dflts(d, dflt_dict=None):
    """
    Fed up with multiline handling of dict arguments?
    Fed up of repeating the if d is None: d = {} lines ad nauseam (because defaults can't be dicts as a default
    because dicts are mutable blah blah, and the python kings don't seem to think a mutable dict is useful enough)?
    Well, my favorite solution would be a built-in handling of the problem of complex/smart defaults,
    that is visible in the code and in the docs. But for now, here's one of the tricks I use.

    Main use is to handle defaults of function arguments. Say you have a function `func(d=None)` and you want
    `d` to be a dict that has at least the keys `foo` and `bar` with default values 7 and 42 respectively.
    Then, in the beginning of your function code you'll say:

        d = fill_with_dflts(d, {'a': 7, 'b': 42})

    See examples to know how to use it.

    ATTENTION: A shallow copy of the dict is made. Know how that affects you (or not).
    ATTENTION: This is not recursive: It won't be filling any nested fields with defaults.

    Args:
        d: The dict you want to "fill"
        dflt_dict: What to fill it with (a {k: v, ...} dict where if k is missing in d, you'll get a new field k, with
            value v.

    Returns:
        a dict with the new key:val entries (if the key was missing in d).

    >>> fill_with_dflts(None)
    {}
    >>> fill_with_dflts(None, {'a': 7, 'b': 42})
    {'a': 7, 'b': 42}
    >>> fill_with_dflts({}, {'a': 7, 'b': 42})
    {'a': 7, 'b': 42}
    >>> fill_with_dflts({'b': 1000}, {'a': 7, 'b': 42})
    {'a': 7, 'b': 1000}
    """
    if d is None:
        d = {}
    if dflt_dict is None:
        dflt_dict = {}
    return dict(dflt_dict, **d)


# Note: Had replaced with cached_property (new in 3.8)
# if not sys.version_info >= (3, 8):
#     from functools import cached_property
# # etc...
# But then I realized that the way cached_property is implemented, pycharm does not see the properties (lint)
# So I'm reverting to lazyprop
# TODO: Keep track of the evolution of functools.cached_property and compare performance.
class lazyprop:
    """
    A descriptor implementation of lazyprop (cached property).
    Made based on David Beazley's "Python Cookbook" book and enhanced with boltons.cacheutils ideas.

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
    """

    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
        self.__isabstractmethod__ = getattr(func, "__isabstractmethod__", False)
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = instance.__dict__[self.func.__name__] = self.func(instance)
            return value

    def __repr__(self):
        cn = self.__class__.__name__
        return "<%s func=%s>" % (cn, self.func)


from functools import lru_cache, wraps
import weakref


@wraps(lru_cache)
def memoized_method(*lru_args, **lru_kwargs):
    def decorator(func):
        @wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # Storing the wrapped method inside the instance since a strong reference to self would not allow it to die.
            self_weak = weakref.ref(self)

            @wraps(func)
            @lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)

        return wrapped_func

    return decorator


class lazyprop_w_sentinel(lazyprop):
    """
    A descriptor implementation of lazyprop (cached property).
    Inserts a `self.func.__name__ + '__cache_active'` attribute

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
    """

    sentinel_prefix = "sentinel_of__"

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = instance.__dict__[self.func.__name__] = self.func(instance)
            setattr(
                instance, self.sentinel_prefix + self.func.__name__, True
            )  # my hack
            return value

    @classmethod
    def cache_is_active(cls, instance, attr):
        return getattr(instance, cls.sentinel_prefix + attr, False)


class Struct:
    def __init__(self, **attr_val_dict):
        for attr, val in attr_val_dict.items():
            setattr(self, attr, val)


class MutableStruct(Struct):
    def extend(self, **attr_val_dict):
        for attr in attr_val_dict.keys():
            if hasattr(self, attr):
                raise AttributeError(
                    f"The attribute {attr} already exists. Delete it if you want to reuse it!"
                )
        for attr, val in attr_val_dict.items():
            setattr(self, attr, val)


def max_common_prefix(a: Sequence, *, default=""):
    """
    Given a list of strings (or other sliceable seq), returns the longest common prefix

    :param a: list-like of strings
    :return: the smallest common prefix of all strings in a

    >>> max_common_prefix(['absolutely', 'abc', 'abba'])
    'ab'
    >>> max_common_prefix(['absolutely', 'not', 'abc', 'abba'])
    ''
    >>> max_common_prefix([[3,2,1], [3,2,0]])
    [3, 2]
    >>> max_common_prefix([[3,2,1], [3,2,0], [1,2,3]])
    []

    If the input is empty, will return default (which defaults to '').

    >>> max_common_prefix([])
    ''

    If you want a different default, you can specify it with the default
    keyword argument.

    >>> from functools import partial
    >>> my_max_common_prefix = partial(max_common_prefix, default=[])
    >>> my_max_common_prefix([])
    []
    """
    if not a:
        return default
    # Note: Try to optimize by using a min_max function to give me both in one pass.
    # The current version is still faster
    s1 = min(a)  # lexicographically minimal
    s2 = max(a)  # lexicographically maximal
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1


class SimpleProperty(object):
    def __get__(self, obj, objtype=None):
        return obj.d

    def __set__(self, obj, value):
        obj.d = value

    def __delete__(self, obj):
        del obj.d


class DelegatedAttribute:
    def __init__(self, delegate_name, attr_name):
        self.attr_name = attr_name
        self.delegate_name = delegate_name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            # return instance.delegate.attr
            return getattr(self.delegate(instance), self.attr_name)

    def __set__(self, instance, value):
        # instance.delegate.attr = value
        setattr(self.delegate(instance), self.attr_name, value)

    def __delete__(self, instance):
        delattr(self.delegate(instance), self.attr_name)

    def delegate(self, instance):
        return getattr(instance, self.delegate_name)

    def __str__(self):
        return ""

    # def __call__(self, instance, *args, **kwargs):
    #     return self.delegate(instance)(*args, **kwargs)


def delegate_as(delegate_cls, to="delegate", include=frozenset(), exclude=frozenset()):
    raise NotImplementedError("Didn't manage to make this work fully")
    # turn include and ignore into sets, if they aren't already
    include = set(include)
    exclude = set(exclude)
    delegate_attrs = set(delegate_cls.__dict__.keys())
    attributes = include | delegate_attrs - exclude

    def inner(cls):
        # create property for storing the delegate
        setattr(cls, to, property())
        # don't bother adding attributes that the class already has
        attrs = attributes - set(cls.__dict__.keys())
        # set all the attributes
        for attr in attrs:
            setattr(cls, attr, DelegatedAttribute(to, attr))
        return cls

    return inner


class HashableMixin:
    def __hash__(self):
        return id(self)


class ImmutableMixin:
    def _immutable(self, *args, **kws):
        raise TypeError("object is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
    pop = _immutable
    popitem = _immutable


# TODO: Lint still considers instances of imdict to be mutable.
#  Probably because it still sees the mutator methods in the class definition.
#  Maybe I should just remove them from the class definition?
# TODO: Generalize to a function that makes any class immutable.
class imdict(ImmutableMixin, dict, HashableMixin):
    """A frozen hashable dict"""


def move_files_of_folder_to_trash(folder):
    trash_dir = os.path.join(
        os.getenv("HOME"), ".Trash"
    )  # works with mac (perhaps linux too?)
    assert os.path.isdir(trash_dir), f"{trash_dir} directory not found"

    for f in os.listdir(folder):
        src = os.path.join(folder, f)
        if os.path.isfile(src):
            dst = os.path.join(trash_dir, f)
            print(f"Moving to trash: {src}")
            shutil.move(src, dst)


class ModuleNotFoundErrorNiceMessage:
    def __init__(self, msg=None):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            if self.msg is not None:
                warn(self.msg)
            else:
                raise ModuleNotFoundError(
                    f"""
It seems you don't have required `{exc_val.name}` package for this Store.
Try installing it by running:

    pip install {exc_val.name}
    
in your terminal.
For more information: https://pypi.org/project/{exc_val.name}
            """
                )


class ModuleNotFoundWarning:
    def __init__(self, msg="It seems you don't have a required package."):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            warn(self.msg)
            #             if exc_val is not None and getattr(exc_val, 'name', None) is not None:
            #                 warn(f"""
            # It seems you don't have required `{exc_val.name}` package for this Store.
            # This is just a warning: The process goes on...
            # (But, hey, if you really need that package, try installing it by running:
            #
            #     pip install {exc_val.name}
            #
            # in your terminal.
            # For more information: https://pypi.org/project/{exc_val.name}, or google around...
            #                 """)
            #             else:
            #                 print("It seems you don't have a required package")
            return True


class ModuleNotFoundIgnore:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            pass
        return True


def num_of_required_args(func):
    """Number or REQUIRED arguments of a function.

    Contrast the behavior below with that of ``num_of_args``, which counts all
    parameters, including the variadics and defaulted ones.

    >>> num_of_required_args(lambda a, b, c: None)
    3
    >>> num_of_required_args(lambda a, b, c=3: None)
    2
    >>> num_of_required_args(lambda a, *args, b, c=1, d=2, **kwargs: None)
    2
    """
    var_param_kinds = {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}
    sig = signature(func)
    return sum(
        1
        for p in sig.parameters.values()
        if p.default is Parameter.empty and p.kind not in var_param_kinds
    )


def num_of_args(func):
    """Number of arguments (parameters) of the function.

    Contrast the behavior below with that of ``num_of_required_args``.

    >>> num_of_args(lambda a, b, c: None)
    3
    >>> num_of_args(lambda a, b, c=3: None)
    3
    >>> num_of_args(lambda a, *args, b, c=1, d=2, **kwargs: None)
    6
    """
    return len(signature(func).parameters)


def single_nest_in_dict(key, value):
    return {key: value}


def nest_in_dict(keys, values):
    return {k: v for k, v in zip(keys, values)}


import io
from typing import Callable, Any, VT, Union, KT
from functools import partial
import tempfile

Buffer = Union[io.BytesIO, io.StringIO]
FileWriter = Callable[[VT, Buffer], Any]
Writer = Union[Callable[[VT, KT], Any], Callable[[KT, VT], Any]]


def _call_writer(
    writer: Writer,
    obj: VT,
    destination: Union[KT, Buffer],
    obj_arg_position_in_writer: int = 0,
):
    """
    Helper function to handle writing to the buffer based on obj_arg_position_in_writer.

    :param writer: A function that writes an object to a file-like object.
    :param obj: The object to write.
    :param destination: The key (e.g. filepath) or file-like object.
    :param obj_arg_position_in_writer: Position of the object argument in writer function (0 or 1).

    :raises ValueError: If obj_arg_position_in_writer is not 0 or 1.
    """
    if obj_arg_position_in_writer == 0:
        writer(obj, destination)
    elif obj_arg_position_in_writer == 1:
        writer(destination, obj)
    else:
        raise ValueError("obj_arg_position_in_writer must be 0 or 1")


def written_bytes(
    file_writer: FileWriter,
    obj: VT = None,
    *,
    obj_arg_position_in_writer: int = 0,
    io_buffer_cls: Buffer = io.BytesIO,
):
    """
    Takes a file writing function that expects an object and a file-like object,
    and returns a function that instead of writing to a file, returns the bytes that
    would have been written.

    This is the write version of the `read_from_bytes` function of the same module.

    Note: If obj is not given, `write_bytes` will return a "bytes writer" function that
    takes obj as the first argument, and uses the file_writer to write the bytes.

    :param file_writer: A function that writes an object to a file-like object.
    :param obj: The object to write.
    :return: The bytes that would have been written to a file.

    Use case: When you have a function that writes to files, and you want to get an
    equivalent function but that gives you what bytes or string WOULD have been written
    to a file, so you can better reuse (to write elsewhere, for example, or because
    you need to pipe those bytes to another function).

    Example usage: Yes, we have json.dumps to get the JSON string, but what if
    (like is often the case) you just have a function that writes to a file-like object,
    like the `json.dump(obj, fp)` function? You can use `written_bytes` to get a
    function that will act as `json.dumps` like so:

    >>> import json
    >>> get_json_bytes = written_bytes(json.dump, io_buffer_cls=io.StringIO)
    >>> get_json_bytes({'a': 1, 'b': 2})
    '{"a": 1, "b": 2}'

    Here's another example with pandas DataFrame.to_parquet:

    >>> import pandas as pd  # doctest: +SKIP
    >>> df = pd.DataFrame({  # doctest: +SKIP
    ...     'column1': [1, 2, 3],
    ...     'column2': ['A', 'B', 'C']
    ... })

    Get a function that converts DataFrame to Parquet bytes

    df_to_parquet_bytes = written_bytes(pd.DataFrame.to_parquet)

    # Get the bytes of the DataFrame in Parquet format
    parquet_bytes = df_to_parquet_bytes(df)
    all(pd.read_parquet(io.BytesIO(parquet_bytes)) == df)


    """
    if obj is None:
        return partial(
            written_bytes,
            file_writer,
            obj_arg_position_in_writer=obj_arg_position_in_writer,
            io_buffer_cls=io_buffer_cls,
        )

    # Create a BytesIO object to act as an in-memory file
    buffer = io_buffer_cls()

    # Use the provided file_writer function to write to the buffer
    _call_writer(file_writer, obj, buffer, obj_arg_position_in_writer)

    # Retrieve the bytes from the buffer
    buffer.seek(0)
    return buffer.read()


def _call_reader(
    reader: Callable,
    buffer: Buffer,
    buffer_arg_position: int = 0,
    buffer_arg_name: str = None,
    *args,
    **kwargs,
):
    """
    Helper function to handle reading from the buffer based on buffer_arg_position or buffer_arg_name.

    :param reader: A function that reads from a file-like object.
    :param buffer: The file-like object to read from.
    :param buffer_arg_position: Position of the file-like object argument in reader function.
    :param buffer_arg_name: Name of the file-like object argument in reader function.
    :raises ValueError: If buffer_arg_position is not valid.
    """
    if buffer_arg_name is not None:
        kwargs[buffer_arg_name] = buffer
        return reader(*args, **kwargs)
    else:
        args = list(args)
        # Ensure the args list is long enough
        while len(args) < buffer_arg_position:
            args.append(None)
        args.insert(buffer_arg_position, buffer)
        return reader(*args, **kwargs)


def read_from_bytes(
    file_reader: Callable,
    obj: bytes = None,
    *,
    buffer_arg_position: int = 0,
    buffer_arg_name: str = None,
    io_buffer_cls: Buffer = io.BytesIO,
    **kwargs,
):
    """
    Takes a file reading function that expects a file-like object,
    and returns a function that instead of reading from a file, reads from bytes.

    This is the read version of the `written_bytes` function of the same module.

    Note: If obj is not given, read_from_bytes will return a "bytes reader" function that
    takes obj as the first argument, and uses the file_reader to read the bytes.

    :param file_reader: A function that reads from a file-like object.
    :param obj: The bytes to read.
    :param buffer_arg_position: The position of the file-like object in file_reader's arguments.
    :param buffer_arg_name: The name of the file-like object argument in file_reader.
    :return: The result of reading from the bytes.

    Example usage:

    Using `json.load` to read a JSON object from bytes:

    >>> import json
    >>> data = {'a': 1, 'b': 2}
    >>> json_bytes = json.dumps(data).encode('utf-8')
    >>> read_json_from_bytes = read_from_bytes(json.load)
    >>> data_loaded = read_json_from_bytes(json_bytes)
    >>> data_loaded == data
    True

    Using `pickle.load` to read an object from bytes:

    >>> import pickle
    >>> obj = {'x': [1, 2, 3], 'y': ('a', 'b')}
    >>> pickle_bytes = pickle.dumps(obj)
    >>> read_pickle_from_bytes = read_from_bytes(pickle.load)
    >>> obj_loaded = read_pickle_from_bytes(pickle_bytes)
    >>> obj_loaded == obj
    True
    """
    if obj is None:
        return partial(
            read_from_bytes,
            file_reader,
            buffer_arg_position=buffer_arg_position,
            buffer_arg_name=buffer_arg_name,
            io_buffer_cls=io_buffer_cls,
            **kwargs,
        )

    buffer = io_buffer_cls(obj)

    return _call_reader(
        file_reader, buffer, buffer_arg_position, buffer_arg_name, **kwargs
    )


def write_to_file(obj: VT, key: KT):
    with open(key, "wb") as f:
        f.write(obj)


def written_key(
    obj: VT = None,
    writer: Writer = write_to_file,
    *,
    key: KT = None,
    obj_arg_position_in_writer: int = 0,
):
    """
    Writes an object to a key and returns the key.
    If key is not given, a temporary file is created and its path is returned.

    :param obj: The object to write.
    :param writer: A function that writes an object to a file.
    :param key: The key (by default, filepath) to write to.
        If None, a temporary file is created.
        If a string with a '*', the '*' is replaced with a unique temporary filename.
    :param obj_arg_position_in_writer: Position of the object argument in writer function (0 or 1).

    :return: The file path where the object was written.

    Example usage:

    Let's make a store and a writer for that store.

    >>> store = dict()
    >>> writer = writer=lambda obj, key: store.__setitem__(key, obj)

    Note the order a writer expects is (obj, key), or we'd just be able to use
    `store.__setitem__` as our writer.

    If we specify a key, the object will be written to that key in the store
    and the key is output.

    >>> written_key(42, writer=writer, key='my_key')
    'my_key'
    >>> store
    {'my_key': 42}

    Often, you'll want to fix your writer (and possibly your key).
    You can do so with `functools.partial`, but for convenience, you can also
    just specify a writer, without an input object, and get a function that
    will write an object to a key.

    >>> write_to_store = written_key(writer=writer, key='another_key')
    >>> write_to_store(99)
    'another_key'
    >>> store
    {'my_key': 42, 'another_key': 99}

    If you don't specify a key, a temporary file is created and the key is output.

    >>> write_to_store = written_key(writer=writer)
    >>> key = write_to_store(43)
    >>> key  # doctest: +SKIP
    '/var/folders/mc/c070wfh51kxd9lft8dl74q1r0000gn/T/tmp8yaczd8b'
    >>> store[key]
    43

    If the key you specify is a string with a '*', the '*' is replaced with a
    unique temporary filename, or the full path of the temporary file if the *
    is at the start.

    >>> write_to_store = written_key(writer=writer, key='*.ext')
    >>> key = write_to_store(44)
    >>> key  # doctest: +ELLIPSIS
    '....ext'
    >>> store[key]
    44

    One useful use case is when you want to pipe the output of one function into
    another function that expects a file path.
    What you need to do then is just pipe your written_key function into that
    function that expects to work with a file path, and it'll be like piping the
    value of your input object into that function (just via a temp file).

    >>> from dol.util import Pipe
    >>> store.clear()
    >>> key_func = lambda key: store.get(key) * 10
    >>> pipe_obj_to_reader = Pipe(written_key(writer=writer), key_func)
    >>> pipe_obj_to_reader(45)
    450
    >>> store  # doctest: +ELLIPSIS
    {...: 45}

    """
    if obj is None:
        return partial(
            written_key,
            writer=writer,
            key=key,
            obj_arg_position_in_writer=obj_arg_position_in_writer,
        )

    if key is None:
        # Create a temporary file
        fd, temp_filepath = tempfile.mkstemp()
        os.close(fd)
        key = temp_filepath
    elif isinstance(key, str) and "*" in key:
        temp_filepath = tempfile.mktemp()
        if key.startswith("*"):
            # Replace * of key with a unique temporary filename
            key = key.replace("*", temp_filepath)
        else:
            # separate directory and filename
            dir_name, base_name = os.path.split(temp_filepath)
            # Replace * of key with a unique temporary filename
            key = key.replace("*", base_name)

    # Write the object to the specified filepath
    _call_writer(writer, obj, key, obj_arg_position_in_writer)

    return key


# TODO: This function should be symmetric, and if so, the code should use recursion
def invertible_maps(
    mapping: Mapping = None, inv_mapping: Mapping = None
) -> Tuple[Mapping, Mapping]:
    """Returns two maps that are inverse of each other.
    Raises an AssertionError iif both maps are None, or if the maps are not inverse of
    each other.

    Get a pair of invertible maps

    >>> invertible_maps({1: 11, 2: 22})
    ({1: 11, 2: 22}, {11: 1, 22: 2})
    >>> invertible_maps(None, {11: 1, 22: 2})
    ({1: 11, 2: 22}, {11: 1, 22: 2})

    You can specify one argument as an iterable (of keys for the mapping) and the
    other as a function (to be applied to the keys to get the inverse mapping).
    The function acts similarly to a `Mapping.__getitem__`, transforming each key to
    its associated value. The iterable defines the keys for the mapping, while the
    function is applied to each key to produce the values.

    >>> invertible_maps([1,2,3], lambda x: x * 10)
    ({10: 1, 20: 2, 30: 3}, {1: 10, 2: 20, 3: 30})
    >>> invertible_maps(lambda x: x * 10, [1,2,3])
    ({1: 10, 2: 20, 3: 30}, {10: 1, 20: 2, 30: 3})

    If two maps are given and invertible, you just get them back

    >>> invertible_maps({1: 11, 2: 22}, {11: 1, 22: 2})
    ({1: 11, 2: 22}, {11: 1, 22: 2})

    Or if they're not invertible

    >>> invertible_maps({1: 11, 2: 22}, {11: 1, 22: 'ha, not what you expected!'})
    Traceback (most recent call last):
      ...
    AssertionError: mapping and inv_mapping are not inverse of each other!

    >>> invertible_maps(None, None)
    Traceback (most recent call last):
      ...
    ValueError: You need to specify one or both maps
    """
    if inv_mapping is None and mapping is None:
        raise ValueError("You need to specify one or both maps")

    # Take care of the case where one is a function and the other is a list
    # Here, we apply the function to the list items to get the mappings
    if callable(mapping):
        assert isinstance(
            inv_mapping, Iterable
        ), f"If one argument is callable, the other one must be an iterable of keys"
        mapping = {k: mapping(k) for k in inv_mapping}
        inv_mapping = {v: k for k, v in mapping.items()}
    elif callable(inv_mapping):
        assert isinstance(
            mapping, Iterable
        ), f"If one argument is callable, the other one must be an iterable of keys"
        inv_mapping = {k: inv_mapping(k) for k in mapping}
        mapping = {v: k for k, v in inv_mapping.items()}

    if inv_mapping is None:
        assert hasattr(mapping, "items")
        inv_mapping = {v: k for k, v in mapping.items()}
        assert len(inv_mapping) == len(
            mapping
        ), "The values of mapping are not unique, so the mapping is not invertible"
    elif mapping is None:
        assert hasattr(inv_mapping, "items")
        mapping = {v: k for k, v in inv_mapping.items()}
        assert len(mapping) == len(
            inv_mapping
        ), "The values of inv_mapping are not unique, so the mapping is not invertible"
    else:
        assert (len(mapping) == len(inv_mapping)) and (
            mapping == {v: k for k, v in inv_mapping.items()}
        ), "mapping and inv_mapping are not inverse of each other!"

    return mapping, inv_mapping
