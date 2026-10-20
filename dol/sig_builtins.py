"""Signatures for the callables that ``inspect.signature`` can't handle.

Many builtins (``print``, ``map``, ``itertools.product``, ...) have no signature that
``inspect`` can read. This module holds hand-made ones, and the function ``Sig`` uses
to get a signature for any callable, falling back to those:

- ``_robust_signature_of_callable``: ``inspect.signature``, with the fallbacks below
- ``sigs_for_builtins``, ``sigs_for_builtin_modules``, ``sigs_for_type_name``: the
  hand-made signatures, by name (``sigs_for_sigless_builtin_name`` merges the first two)
- ``resolve_function``, ``dict_of_attribute_signatures``: helpers

It depends on nothing else in ``dol``. It used to be part of ``dol.signatures``, which
still re-exports all of it (i2mint/dol#70).

    >>> from dol.sig_builtins import _robust_signature_of_callable
    >>> _robust_signature_of_callable(isinstance)
    <Signature (obj, class_or_tuple, /)>
"""

import sys
from inspect import Signature, signature
from typing import Any, KT, VT, T
from collections.abc import Callable, Iterable, Iterator, Mapping as MappingType
from functools import cached_property, partial, partialmethod

# A default signature of (*no_sig_args, **no_sig_kwargs)
DFLT_SIGNATURE = signature(lambda *no_sig_args, **no_sig_kwargs: ...)


#########################################################################################
# Manual construction of missing signatures
# ############################################################################


# TODO: Might want to monkey-patch inspect._signature_from_callable to use
#  sigs_for_sigless_builtin_name
def _robust_signature_of_callable(callable_obj: Callable) -> Signature:
    r"""Get the signature of a Callable, returning a custom made one for those
    builtins that don't have one

    >>> _robust_signature_of_callable(
    ...     _robust_signature_of_callable
    ... )  # has a normal signature
    <Signature (callable_obj: ...Callable) -> inspect.Signature>
    >>> s = _robust_signature_of_callable(print)  # has one that this module provides
    >>> assert isinstance(s, Signature)
    >>> # Will be: <Signature (*value, sep=' ', end='\n', file=<_io.TextIOWrapper
    name='<stdout>' mode='w' encoding='utf-8'>, flush=False)>
    >>> _robust_signature_of_callable(
    ...     slice
    ... )  # doesn't have one, so will return a blanket one
    <Signature (*no_sig_args, **no_sig_kwargs)>
    """
    # First check if we have a custom signature for this type/object
    # This is important for operator instances that might have generic signatures in Python 3.12+
    obj_name = getattr(callable_obj, "__name__", None)
    if obj_name in sigs_for_sigless_builtin_name:
        return sigs_for_sigless_builtin_name[obj_name] or DFLT_SIGNATURE

    type_name = getattr(type(callable_obj), "__name__", None)
    if type_name in sigs_for_type_name:
        return sigs_for_type_name[type_name] or DFLT_SIGNATURE

    # Try to get the signature normally
    try:
        return signature(callable_obj)
    except ValueError:
        # if all attempts fail, return the default signature
        return DFLT_SIGNATURE


def resolve_function(obj: T) -> T | Callable:
    """Get the underlying function of a property or cached_property

    Note that if all conditions fail, the object itself is returned.

    The problem this function solves is that sometimes there's a function behind an
    object, but it's not always easy to get to it. For example, in a class, you might
    want to get the source of the code decorated with ``@property``, a
    ``@cached_property``, or a ``partial`` function.

    Consider the following example:

    >>> from functools import cached_property, partial
    >>> class C:
    ...     @property
    ...     def prop(self):
    ...         pass
    ...     @cached_property
    ...     def cached_prop(self):
    ...         pass
    ...     partial_func = partial(partial)

    Note that ``prop`` is not callable, and you can't get its source.

    >>> import inspect
    >>> callable(C.prop)
    False
    >>> inspect.getsource(C.prop)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    TypeError: <property object at 0x...> is not a module, class, method, function, traceback, frame, or code object

    But if you grab the underlying function, you can get the source:

    >>> func = resolve_function(C.prop)
    >>> callable(func)
    True
    >>> isinstance(inspect.getsource(func), str)
    True

    Same goes with ``cached_property`` and ``partial``:

    >>> isinstance(inspect.getsource(resolve_function(C.cached_prop)), str)
    True
    >>> isinstance(inspect.getsource(resolve_function(C.partial_func)), str)
    True
    """
    if isinstance(obj, cached_property):
        return obj.func
    elif isinstance(obj, property):
        return obj.fget
    elif isinstance(obj, (partial, partialmethod)):
        return obj.func
    elif not callable(obj) and callable(wrapped := getattr(obj, "__wrapped__", None)):
        # If obj is not callable, but has a __wrapped__ attribute that is, return that
        return wrapped
    else:  # if not just return obj
        return obj


def dict_of_attribute_signatures(cls: type) -> dict[str, Signature]:
    """
    A function that extracts the signatures of all callable attributes of a class.

    :param cls: The class that holds the the ``(name, func)`` pairs we want to extract.
    :return: A dict of ``(name, signature(func))`` pairs extracted from class.

    One of the intended applications is to use ``dict_of_attribute_signatures`` as a
    decorator, like so:

    >>> @dict_of_attribute_signatures
    ... class names_and_signatures:
    ...     def foo(x: str, *, y=2) -> tuple: ...
    ...     def bar(z, /) -> float: ...
    >>> names_and_signatures
    {'foo': <Signature (x: str, *, y=2) -> tuple>, 'bar': <Signature (z, /) -> float>}
    """

    def gen():
        object_attr_names = set(vars(object))
        for attr_name, attr_val in vars(cls).items():
            if callable(attr_val):
                if attr_name not in object_attr_names:
                    # if the attr is a callable attribute that's not in all objects...
                    yield attr_name, signature(attr_val)

    return dict(gen())


@dict_of_attribute_signatures
class sigs_for_builtins:
    def __import__(name, globals=None, locals=None, fromlist=(), level=0):
        """__import__(name, globals=None, locals=None, fromlist=(), level=0) -> module"""

    def filter(function, iterable, /):
        """filter(function or None, iterable) --> filter object"""

    def map(func, iterable, /, *iterables):
        """``map(func, *iterables) --> map object``"""

    def print(*value, sep=" ", end="\n", file=sys.stdout, flush=False):
        r"""``print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)``"""

    def zip(*iterables):
        """
        ``zip(*iterables)`` --> A zip object yielding tuples until an input is exhausted.
        """

    def bool(x: Any, /) -> bool: ...

    def bytearray(iterable_of_ints: Iterable[int], /): ...

    def classmethod(function: Callable, /): ...

    def int(x, base=10, /): ...

    def iter(callable: Callable, sentinel=None, /): ...

    def next(iterator: Iterator, default=None, /): ...

    def staticmethod(function: Callable, /): ...

    def str(bytes_or_buffer, encoding=None, errors=None, /): ...

    def super(type_, obj=None, /): ...

    # def type(name, bases=None, dict=None, /):
    #     ...


sigs_for_builtins = dict(
    sigs_for_builtins,
    **{
        "__build_class__": None,
        # __build_class__(func, name, /, *bases, [metaclass], **kwds) -> class
        # "bool": None,
        # bool(x) -> bool
        "breakpoint": None,
        # breakpoint(*args, **kws)
        # "bytearray": None,
        # bytearray(iterable_of_ints) -> bytearray
        # bytearray(string, encoding[, errors]) -> bytearray
        # bytearray(bytes_or_buffer) -> mutable copy of bytes_or_buffer
        # bytearray(int) -> bytes array of size given by the parameter initialized with
        # null bytes
        # bytearray() -> empty bytes array
        "bytes": None,
        # bytes(iterable_of_ints) -> bytes
        # bytes(string, encoding[, errors]) -> bytes
        # bytes(bytes_or_buffer) -> immutable copy of bytes_or_buffer
        # bytes(int) -> bytes object of size given by the parameter initialized with null
        # bytes
        # bytes() -> empty bytes object
        # "classmethod": None,
        # classmethod(function) -> method
        "dict": None,
        # dict() -> new empty dictionary
        # dict(mapping) -> new dictionary initialized from a mapping object's
        # dict(iterable) -> new dictionary initialized as if via:
        # dict(**kwargs) -> new dictionary initialized with the name=value pairs
        "dir": None,
        # dir([object]) -> list of strings
        "frozenset": None,
        # frozenset() -> empty frozenset object
        # frozenset(iterable) -> frozenset object
        "getattr": None,
        # getattr(object, name[, default]) -> value
        # "int": None,
        # int([x]) -> integer
        # int(x, base=10) -> integer
        # "iter": None,
        # iter(iterable) -> iterator
        # iter(callable, sentinel) -> iterator
        "max": None,
        # max(iterable, *[, default=obj, key=func]) -> value
        # max(arg1, arg2, *args, *[, key=func]) -> value
        "min": None,
        # min(iterable, *[, default=obj, key=func]) -> value
        # min(arg1, arg2, *args, *[, key=func]) -> value
        # "next": None,
        # next(iterator[, default])
        "range": None,
        # range(stop) -> range object
        # range(start, stop[, step]) -> range object
        "set": None,
        # set() -> new empty set object
        # set(iterable) -> new set object
        "slice": None,
        # slice(stop)
        # slice(start, stop[, step])
        # "staticmethod": None,
        # staticmethod(function) -> method
        # "str": None,
        # str(object='') -> str
        # str(bytes_or_buffer[, encoding[, errors]]) -> str
        # "super": None,
        # super() -> same as super(__class__, <first argument>)
        # super(type) -> unbound super object
        # super(type, obj) -> bound super object; requires isinstance(obj, type)
        # super(type, type2) -> bound super object; requires issubclass(type2, type)
        # "type": None,
        # type(object_or_name, bases, dict)
        # type(object) -> the object's type
        # type(name, bases, dict) -> a new type
        "vars": None,
        # vars([object]) -> dictionary
    },
)
# # Remove the None-valued elements (No, don't, because we distinguish
# # functions we listed but didn't associate a default signature, with those functions
# # we don't list at all.
# sigs_for_builtins = {
#     k: v for k, v in sigs_for_builtins.items() if v is not None
# }


# TODO: itemgetter, attrgetter and methodcaller use KT as their first argument, but
#  in reality both attrgetter and methodcaller are more restrictive: They need to be
#  valid attributes, therefore valid python identifiers. Any better typing for that?
# TODO: We take care of the MutableMapping dunders below, but some of these dunders
#  are not specific to MutableMapping. The signature used below is somewhat, but not
#  completely, specific to MutableMapping. For example, __contains__ is also defined
#  for the `set` type, but it's input is not called key, nor would the KT annotation
#  be completely correct. The signatures were sometimes made to be more general (such
#  as __setitem__ and __delitem__ returning an Any instead of None), but could be
#  made more (for example, annotating return of __iter__ as Iterator instead of
#  Iterator[KT]). We hope that the fact that all the signatures are positional-only
#  will at least mitigate the problem as far as name differences go.
@dict_of_attribute_signatures
class sigs_for_builtin_modules:
    """
    Below are the signatures, manually created to match those callables of the python
    standard library that don't have signatures (through ``inspect.signature``),
    """

    def __eq__(self, other, /) -> bool:
        """self.__eq__(other) <==> self==other"""

    def __ne__(self, other, /) -> bool:
        """self.__ne__(other) <==> self!=other"""

    def __iter__(self, /) -> Iterator[KT]:
        """self.__iter__() <==> iter(self)"""

    def __getitem__(self, key: KT, /) -> VT:
        """self.__getitem__(key) <==> self[key]"""

    def __len__(self, /) -> int:
        """self.__len__() <==> len(self)"""

    def __contains__(self, key: KT, /) -> bool:
        """self.__contains__(key) <==> key in self"""

    def __setitem__(self, key: KT, value: VT, /) -> Any:
        """self.__setitem__(key, value) <==> self[key] = value"""

    def __delitem__(self, key: KT, /) -> Any:
        """self.__delitem__(key) <==> del self[key]"""

    def itemgetter(item, /, *items) -> Callable[[Iterable[VT]], VT | tuple[VT]]:
        """itemgetter(item, ...) --> itemgetter object,"""

    def attrgetter(attr, /, *attrs) -> Callable[[Iterable[VT]], VT | tuple[VT]]:
        """attrgetter(item, ...) --> attrgetter object,"""

    def methodcaller(
        name: KT, /, *args: Iterable[VT], **kwargs: MappingType[str, Any]
    ) -> Callable[[Any], Any]:
        """methodcaller(name, ...) --> methodcaller object"""

    def partial(func: Callable, *args, **keywords) -> Callable:
        """``partial(func, *args, **keywords)`` - new function with partial application
        of the given arguments and keywords."""

    def partialmethod(func: Callable, *args, **keywords) -> Callable:
        """``functools.partialmethod(func, *args, **keywords)``"""


# Merge sigs_for_builtin_modules and sigs_for_builtins
sigs_for_sigless_builtin_name = dict(sigs_for_builtin_modules, **sigs_for_builtins)


@dict_of_attribute_signatures
class sigs_for_type_name:
    """
    Below are the signatures, manually created to match callable objects that are
    output by builtin functions or are instances of builtin classes, and that have no
    signatures (through ``inspect.signature``),
    """

    def itemgetter(iterable: Iterable[VT], /) -> VT | tuple[VT]: ...

    def attrgetter(iterable: Iterable[VT], /) -> VT | tuple[VT]: ...

    def methodcaller(obj: Any) -> Any: ...
