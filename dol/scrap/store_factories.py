"""Utils to make stores"""

from typing import Mapping, Callable, KT, VT, Iterator, Any, NewType, Collection
import dataclasses
from dataclasses import dataclass
import operator
from functools import partial
from contextlib import suppress
from dol.base import KvReader

Getter = NewType("Getter", Callable[[Mapping, KT], VT])
Lister = NewType("Getter", Callable[[Mapping], Iterator[KT]])
Sizer = NewType("Sizer", Callable[[Mapping], int])
ContainmentChecker = NewType("ContainmentChecker", Callable[[Mapping], bool])
Setter = NewType("Setter", Callable[[Mapping, KT, VT], Any])
Deleter = NewType("Deleter", Callable[[Mapping, KT], Any])

count_by_iteration: Sizer
check_by_iteration: ContainmentChecker
check_by_trying_to_get: ContainmentChecker


def count_by_iteration(collection: Collection) -> int:
    """
    Number of elements in collection of keys.
    Note: This method iterates over all elements of the collection and counts them.
    Therefore it is not efficient, and in most cases should be overridden with a more
    efficient method.
    """
    count = 0
    for _ in iter(collection):
        count += 1
    return count


# TODO: Put KT here because it's the main use, but could be VT.
#  Should probably be T = TypeVar('T')?
def check_by_iteration(collection: Collection[KT], x: KT) -> bool:
    """
    Check if collection of keys contains k.
    Note: Method loops through contents of collection to see if query element exists.
    Therefore it may not be efficient, and in most cases, a method specific to the case should be used.
    :return: True if k is in the collection, and False if not
    """
    for existing_x in iter(collection):
        if existing_x == x:
            return True
    return False


def check_by_trying_to_get(mapping: Mapping, x: KT, false_on_error=(KeyError,)) -> bool:
    """
    Check if mapping contains x.
    Note: This method tries to get x from the mapping, returning ``False`` if it fails.
    Therefore it may not be efficient, and in most cases,
    a method specific to the case should be used.
    :return: True if x is in the mapping, and False if not
    """
    try:
        _ = mapping[x]
        return True
    except false_on_error:
        return False


# def mk_shell_factory(cls):
#     """Make a factory for a shell class"""
#     src_field, *other_fields = cls.__dataclass_fields__.items()
#     return partial(cls, **dict(other_fields))


# def add_shell_factory(cls):
#     """Add a factory for a shell class"""
#     from i2 import FuncFactory
#     cls.factory = FuncFactory(cls)
#     return cls


# TODO: See dol.sources.AttrContainer. Make KvReaderShell subsume it, then refactor
# TODO: Make tools to (1) wrap types and (2) make recursion easy
# @add_shell_factory
@dataclass
class KvReaderShell(KvReader):
    """Wraps an object with a mapping interface

    See below how we can wrap a list with a mapping interface, and use it as a store.

    >>> arr = (1, 2, 3)
    >>> s = KvReaderShell(arr, getter=getattr, lister=dir)

    We defined the ``lister`` to yield the attributes of the list:

    >>> sorted(s)[:2]
    ['__add__', '__class__']

    We defined the ``getter`` to give us attributes:

    >>> callable(s['__add__'])
    True
    >>> s['__add__']((4, 5))
    (1, 2, 3, 4, 5)


    """

    src: Any
    getter: Getter = operator.getitem
    lister: Lister = iter
    with suppress(AttributeError):  # Note: dataclasses.KW_ONLY only >= 3.10
        _ = dataclasses.KW_ONLY
    sizer: Sizer = count_by_iteration
    is_contained: ContainmentChecker = check_by_iteration

    def __getitem__(self, k: KT) -> VT:
        return self.getter(self.src, k)

    def __iter__(self) -> Iterator[KT]:
        return iter(self.lister(self.src))

    def __len__(self):
        return self.sizer(self.src)

    def __contains__(self, k: KT):
        return self.is_contained(self.src, k)

    # @property
    # def factory(self):
    #     return mk_shell_factory(KvReaderShell)


@dataclass
class StoreShell(KvReaderShell):
    """Wraps an object with a mapping interface"""

    setter: Setter = operator.setitem
    deleter: Deleter = operator.delitem

    def __setitem__(self, k: KT, v: VT) -> Any:
        return self.setter(self.src, k, v)

    def __delitem__(self, k: KT) -> Any:
        return self.deleter(self.src, k)
