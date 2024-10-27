"""
Base classes for making stores.
In the language of the collections.abc module, a store is a MutableMapping that is configured to work with a specific
representation of keys, serialization of objects (python values), and persistence of the serialized data.

That is, stores offer the same interface as a dict, but where the actual implementation of writes, reads, and listing
are configurable.

Consider the following example. You're store is meant to store waveforms as wav files on a remote server.
Say waveforms are represented in python as a tuple (wf, sr), where wf is a list of numbers and sr is the sample
rate, an int). The __setitem__ method will specify how to store bytes on a remote server, but you'll need to specify
how to SERIALIZE (wf, sr) to the bytes that constitute that wav file: _data_of_obj specifies that.
You might also want to read those wav files back into a python (wf, sr) tuple. The __getitem__ method will get
you those bytes from the server, but the store will need to know how to DESERIALIZE those bytes back into a python
object: _obj_of_data specifies that

Further, say you're storing these .wav files in /some/folder/on/the/server/, but you don't want the store to use
these as the keys. For one, it's annoying to type and harder to read. But more importantly, it's an irrelevant
implementation detail that shouldn't be exposed. THe _id_of_key and _key_of_id pair are what allow you to
add this key interface layer.

These key converters object serialization methods default to the identity (i.e. they return the input as is).
This means that you don't have to implement these as all, and can choose to implement these concerns within
the storage methods themselves.
"""

from functools import partial, update_wrapper
import copyreg
from collections.abc import Collection as CollectionABC
from collections.abc import Mapping, MutableMapping
from collections.abc import (
    KeysView as BaseKeysView,
    ValuesView as BaseValuesView,
    ItemsView as BaseItemsView,
    Set,
)
from typing import Any, Iterable, Tuple, Callable, Union, Optional

from dol.util import (
    wraps,
    _disabled_clear_method,
    identity_func,
    static_identity_method,
    Key,
    Val,
    Id,
    Data,
    Item,
    KeyIter,
    ValIter,
    ItemIter,
    is_unbound_method,
    is_classmethod,
)
from dol.signatures import Sig


class AttrNames:
    CollectionABC = {"__len__", "__iter__", "__contains__"}
    Mapping = CollectionABC | {
        "keys",
        "get",
        "items",
        "__reversed__",
        "values",
        "__getitem__",
    }
    MutableMapping = Mapping | {
        "setdefault",
        "pop",
        "popitem",
        "clear",
        "update",
        "__delitem__",
        "__setitem__",
    }

    Collection = CollectionABC | {"head"}
    KvReader = (Mapping | {"head"}) - {"__reversed__"}
    KvPersister = (MutableMapping | {"head"}) - {"__reversed__"} - {"clear"}


# TODO: Consider using ContainmentChecker and Sizer attributes which dunders would
#  point to.
class Collection(CollectionABC):
    """The same as collections.abc.Collection, with some modifications:
    - Addition of a ``head``
    """

    def __contains__(self, x) -> bool:
        """
        Check if collection of keys contains k.
        Note: This method loops through all contents of collection to see if query element exists.
        Therefore it may not be efficient, and in most cases, a method specific to the case should be used.
        :return: True if k is in the collection, and False if not
        """
        for existing_x in iter(self):
            if existing_x == x:
                return True
        return False

    def __len__(self) -> int:
        """
        Number of elements in collection of keys.
        Note: This method iterates over all elements of the collection and counts them.
        Therefore it is not efficient, and in most cases should be overridden with a more efficient version.
        :return: The number (int) of elements in the collection of keys.
        """
        # Note: Found that sum(1 for _ in self.__iter__()) was slower for small, slightly faster for big inputs.
        count = 0
        for _ in iter(self):
            count += 1
        return count

    def head(self):
        if hasattr(self, "items"):
            return next(iter(self.items()))
        else:
            return next(iter(self))


# KvCollection = Collection  # alias meant for back-compatibility. Would like to deprecated


# def getitem_based_contains(self, x) -> bool:
#     """
#     Check if collection of keys contains k.
#     Note: This method actually fetches the contents for k, returning False if there's a key error trying to do so
#     Therefore it may not be efficient, and in most cases, a method specific to the case should be used.
#     :return: True if k is in the collection, and False if not
#     """
#
#     try:
#         self.__getitem__(k)
#         return True
#     except KeyError:
#         return False


class MappingViewMixin:
    KeysView: type = BaseKeysView
    ValuesView: type = BaseValuesView
    ItemsView: type = BaseItemsView

    def keys(self) -> KeysView:
        return self.KeysView(self)

    def values(self) -> ValuesView:
        return self.ValuesView(self)

    def items(self) -> ItemsView:
        return self.ItemsView(self)


class KvReader(MappingViewMixin, Collection, Mapping):
    """Acts as a Mapping abc, but with default __len__ (implemented by counting keys)
    and head method to get the first (k, v) item of the store"""

    def head(self):
        """Get the first (key, value) pair"""
        for k, v in self.items():
            return k, v

    def __reversed__(self):
        """The __reversed__ is disabled at the base, but can be re-defined in subclasses.
        Rationale: KvReader is meant to wrap a variety of storage backends or key-value
        perspectives thereof.
        Not all of these would have a natural or intuitive order nor do we want to
        incur the cost of maintaining one systematically.

        If you need a reversed list, here's one way to do it, but note that it
        depends on how self iterates, which is not even assured to be consistent at
        every call:

        .. code-block:: python

            reversed = list(self)[::-1]


        If the keys are comparable, therefore sortable, another natural option would be:

        .. code-block:: python

            reversed = sorted(self)[::-1]

        """
        raise NotImplementedError(__doc__)


Reader = KvReader  # alias for back-compatibility


# TODO: Should we really be using MutableMapping if we're disabling so many of it's methods?
# TODO: Wishful thinking: Define store type so the type is defined by it's methods, not by subclassing.
class KvPersister(KvReader, MutableMapping):
    """Acts as a MutableMapping abc, but disabling the clear and __reversed__ method,
    and computing __len__ by iterating over all keys, and counting them.

    Note that KvPersister is a MutableMapping, and as such, is dict-like.
    But that doesn't mean it's a dict.

    For instance, consider the following code:

    .. code-block:: python

        s = SomeKvPersister()
        s['a']['b'] = 3

    If `s` is a dict, this would have the effect of adding a ('b', 3) item under 'a'.
    But in the general case, this might
    - fail, because the `s['a']` doesn't support sub-scripting (doesn't have a `__getitem__`)
    - or, worse, will pass silently but not actually persist the write as expected (e.g. LocalFileStore)

    Another example: `s.popitem()` will pop a `(k, v)` pair off of the `s` store.
    That is, retrieve the `v` for `k`, delete the entry for `k`, and return a `(k, v)`.
    Note that unlike modern dicts which will return the last item that was stored
     -- that is, LIFO (last-in, first-out) order -- for KvPersisters,
     there's no assurance as to what item will be, since it will depend on the backend storage system
     and/or how the persister was implemented.

    """

    clear = _disabled_clear_method

    # # TODO: Tests and documentation demos needed.
    # def popitem(self):
    #     """pop a (k, v) pair off of the store.
    #     That is, retrieve the v for k, delete the entry for k, and return a (k, v)
    #     Note that unlike modern dicts which will return the last item that was stored
    #      -- that is, LIFO (last-in, first-out) order -- for KvPersisters,
    #      there's no assurance as to what item will be, since it will depend on the backend storage system
    #      and/or how the persister was implemented.
    #     :return:
    #     """
    #     return super(KvPersister, self).popitem()


Persister = KvPersister  # alias for back-compatibility


class NoSuchItem:
    pass


no_such_item = NoSuchItem()

from collections.abc import Set


class DelegatedAttribute:
    def __init__(self, delegate_name, attr_name):
        self.attr_name = attr_name
        self.delegate_name = delegate_name

    def __get__(self, instance, owner):
        if instance is None:
            # return getattr(getattr(owner, self.delegate_name), self.attr_name)
            # return getattr(owner, self.attr_name, None)

            # TODO: Would just return self or self.__wrapped__ here, but
            #   self.__wrapped__ would make it hard to debug and
            #   self would fail with unbound methods (why?)
            #   So doing a check here, but would like to find a better solution.
            wrapped_self = getattr(self, "__wrapped__", None)
            if is_classmethod(wrapped_self) or is_unbound_method(wrapped_self):
                return wrapped_self
            else:
                return self

            # wrapped_self = getattr(self, '__wrapped__', None)
            # if not is_classmethod(wrapped_self):
            #     return self
            # else:
            #     return wrapped_self
        else:
            # i.e. return instance.delegate.attr
            return getattr(getattr(instance, self.delegate_name), self.attr_name)

    def __set__(self, instance, value):
        # instance.delegate.attr = value
        setattr(getattr(instance, self.delegate_name), self.attr_name, value)

    def __delete__(self, instance):
        delattr(getattr(instance, self.delegate_name), self.attr_name)


Decorator = Callable[[Callable], Any]  # TODO: Look up typing protocols


def delegate_to(
    wrapped: type,
    class_trans: Optional[Callable] = None,
    delegation_attr: str = "store",
    include=frozenset(),
    ignore=frozenset(),
) -> Decorator:
    # turn include and ignore into sets, if they aren't already
    if not isinstance(include, Set):
        include = set(include)
    if not isinstance(ignore, Set):
        ignore = set(ignore)
    # delegate_attrs = set(delegate_cls.__dict__)
    delegate_attrs = set(dir(wrapped))
    attributes_of_wrapped = (
        include | delegate_attrs - ignore
    )  # TODO: Look at precedence

    def delegation_decorator(wrapper_cls: type):
        @wraps(wrapper_cls, updated=())
        class Wrap(wrapper_cls):
            # _type_of_wrapped = wrapped
            # _delegation_attr = delegation_attr
            _class_trans = class_trans

            @wraps(wrapper_cls.__init__)
            def __init__(self, *args, **kwargs):
                delegate = wrapped(*args, **kwargs)
                super().__init__(delegate)
                assert isinstance(
                    getattr(self, delegation_attr, None), wrapped
                ), f"The wrapper instance has no (expected) {delegation_attr!r} attribute"

            def __reduce__(self):
                return (
                    # reconstructor
                    wrapped_delegator_reconstruct,
                    # args of reconstructor
                    (wrapper_cls, wrapped, class_trans, delegation_attr),
                    # instance state
                    self.__getstate__(),
                )

        attrs = attributes_of_wrapped - set(
            dir(wrapper_cls)
        )  # don't bother adding attributes that the class already has
        # set all the attributes
        for attr in attrs:
            if attr == "__provides__":  # TODO: Hack. Find better solution.
                # This is because __provides__ happened to be in wrapper_cls but not
                # in wrapped.
                # Happened at some point with `from sqldol import SqlRowsReader``
                continue
            wrapped_attr = getattr(wrapped, attr)
            delegated_attribute = update_wrapper(
                wrapper=DelegatedAttribute(delegation_attr, attr),
                wrapped=wrapped_attr,
            )
            setattr(Wrap, attr, delegated_attribute)

        if class_trans:
            Wrap = class_trans(Wrap)
        return Wrap

    return delegation_decorator


def wrapped_delegator_reconstruct(wrapped_cls, wrapped, class_trans, delegation_attr):
    """"""
    type_ = delegator_wrap(wrapped_cls, wrapped, class_trans, delegation_attr)
    # produce an empty object for pickle to pour the
    # __getstate__ values into, via __setstate__
    return copyreg._reconstructor(type_, object, None)


def delegator_wrap(
    delegator: Callable,
    obj: Union[type, Any],
    class_trans=None,
    delegation_attr: str = "store",
):
    """Wrap a ``obj`` (type or instance) with ``delegator``.

    If obj is not a type, trivially returns ``delegator(obj)``.

    The interesting case of ``delegator_wrap`` is when ``obj`` is a type (a class).
    In this case, ``delegator_wrap`` returns a callable (class or function) that has the
    same signature as obj, but that produces instances that are wrapped by ``delegator``

    :param delegator: An instance wrapper. A Callable (type or function -- with only
        one required input) that will return a wrapped version of it's input instance.
    :param obj: The object (class or instance) to be wrapped.
    :return: A wrapped object

    Let's demo this on a simple Delegator class.

    >>> class Delegator:
    ...     i_think = 'therefore I am delegated'  # this is just to verify that we're in a Delegator
    ...     def __init__(self, wrapped_obj):
    ...         self.wrapped_obj = wrapped_obj
    ...     def __getattr__(self, attr):  # delegation: just forward attributes to wrapped_obj
    ...         return getattr(self.wrapped_obj, attr)
    ...     wrap = classmethod(delegator_wrap)  # this is a useful recipe to have the Delegator carry it's own wrapping method

    The only difference between a wrapped object ``Delegator(obj)`` and the original ``obj`` is
    that the wrapped one has a ``i_think`` attribute.
    The wrapped object should otherwise behave the same (on all but special (dunder) methods).
    So let's test this on dictionaries, using the following test function:

    >>> def test_wrapped_d(wrapped_d, original_d):
    ...     '''A function to test a wrapped dict'''
    ...     assert not hasattr(original_d, 'i_think')  # verify that the unwrapped_d doesn't have an i_think attribute
    ...     assert list(wrapped_d.items()) == list(original_d.items())  # verify that wrapped_d has an items that gives us the same thing as origina_d
    ...     assert hasattr(wrapped_d, 'i_think')  # ... but wrapped_d has a i_think attribute
    ...     assert wrapped_d.i_think == 'therefore I am delegated'  # ... and its what we set it to be

    Let's try delegating a dict INSTANCE first:

    >>> d = {'a': 1, 'b': 2}
    >>> wrapped_d = delegator_wrap(Delegator, d)
    >>> test_wrapped_d(wrapped_d, d)

    If we ask ``delegator_wrap`` to wrap a ``dict`` type, we get a subclass of Delegator
    (NOT dict!) whose instances will have the behavior exhibited above:

    >>> WrappedDict = delegator_wrap(Delegator, dict, delegation_attr='wrapped_obj')
    >>> assert issubclass(WrappedDict, Delegator)
    >>> wrapped_d = WrappedDict(a=1, b=2)

    >>> test_wrapped_d(wrapped_d, wrapped_d.wrapped_obj)

    Now we'll demo/test the ``wrap = classmethod(delegator_wrap)`` trick
    ... with instances

    >>> wrapped_d = Delegator.wrap(d)
    >>> test_wrapped_d(wrapped_d, wrapped_d.wrapped_obj)

    ... with classes

    >>> WrappedDict = Delegator.wrap(dict, delegation_attr='wrapped_obj')
    >>> wrapped_d = WrappedDict(a=1, b=2)

    >>> test_wrapped_d(wrapped_d, wrapped_d.wrapped_obj)
    >>> class A(dict):
    ...     def foo(self, x):
    ...         pass
    >>> hasattr(A, 'foo')
    True
    >>> WrappedA = Delegator.wrap(A)
    >>> hasattr(WrappedA, 'foo')
    True

    """
    if isinstance(obj, type):
        if isinstance(delegator, type):
            type_decorator = delegate_to(
                obj, class_trans=class_trans, delegation_attr=delegation_attr
            )
            wrap = type_decorator(delegator)
            try:  # try to give the wrap the signature of obj (if it has one)
                wrap.__signature__ = Sig(obj)
            except ValueError:
                pass
            return wrap

        else:
            assert isinstance(delegator, Callable)

            @wraps(obj.__init__)
            def wrap(*args, **kwargs):
                wrapped = obj(*args, **kwargs)
                return delegator(wrapped)

            return wrap
    else:
        return delegator(obj)


class Store(KvPersister):
    """
    By store we mean key-value store. This could be files in a filesystem, objects in s3, or a database. Where and
    how the content is stored should be specified, but StoreInterface offers a dict-like interface to this.
    ::
        __getitem__ calls: _id_of_key			                    _obj_of_data
        __setitem__ calls: _id_of_key		        _data_of_obj
        __delitem__ calls: _id_of_key
        __iter__    calls:	            _key_of_id


    >>> # Default store: no key or value conversion #####################################
    >>> from dol import Store
    >>> s = Store()
    >>> s['foo'] = 33
    >>> s['bar'] = 65
    >>> assert list(s.items()) == [('foo', 33), ('bar', 65)]
    >>> assert list(s.store.items()) == [('foo', 33), ('bar', 65)]  # see that the store contains the same thing
    >>>
    >>> #################################################################################
    >>> # Now let's make stores that have a key and value conversion layer ##############
    >>> # input keys will be upper cased, and output keys lower cased ###################
    >>> # input values (assumed int) will be converted to ascii string, and visa versa ##
    >>> #################################################################################
    >>>
    >>> def test_store(s):
    ...     s['foo'] = 33  # write 33 to 'foo'
    ...     assert 'foo' in s  # __contains__ works
    ...     assert 'no_such_key' not in s  # __nin__ works
    ...     s['bar'] = 65  # write 65 to 'bar'
    ...     assert len(s) == 2  # there are indeed two elements
    ...     assert list(s) == ['foo', 'bar']  # these are the keys
    ...     assert list(s.keys()) == ['foo', 'bar']  # the keys() method works!
    ...     assert list(s.values()) == [33, 65]  # the values() method works!
    ...     assert list(s.items()) == [('foo', 33), ('bar', 65)]  # these are the items
    ...     assert list(s.store.items()) == [('FOO', '!'), ('BAR', 'A')]  # but note the internal representation
    ...     assert s.get('foo') == 33  # the get method works
    ...     assert s.get('no_such_key', 'something') == 'something'  # return a default value
    ...     del(s['foo'])  # you can delete an item given its key
    ...     assert len(s) == 1  # see, only one item left!
    ...     assert list(s.items()) == [('bar', 65)]  # here it is
    >>>
    >>> # We can introduce this conversion layer in several ways. Here's a few... ######################
    >>> # by subclassing ###############################################################################
    >>> class MyStore(Store):
    ...     def _id_of_key(self, k):
    ...         return k.upper()
    ...     def _key_of_id(self, _id):
    ...         return _id.lower()
    ...     def _data_of_obj(self, obj):
    ...         return chr(obj)
    ...     def _obj_of_data(self, data):
    ...         return ord(data)
    >>> s = MyStore(store=dict())  # note that you don't need to specify dict(), since it's the default
    >>> test_store(s)
    >>>
    >>> # by assigning functions to converters ##########################################################
    >>> class MyStore(Store):
    ...     def __init__(self, store, _id_of_key, _key_of_id, _data_of_obj, _obj_of_data):
    ...         super().__init__(store)
    ...         self._id_of_key = _id_of_key
    ...         self._key_of_id = _key_of_id
    ...         self._data_of_obj = _data_of_obj
    ...         self._obj_of_data = _obj_of_data
    ...
    >>> s = MyStore(dict(),
    ...             _id_of_key=lambda k: k.upper(),
    ...             _key_of_id=lambda _id: _id.lower(),
    ...             _data_of_obj=lambda obj: chr(obj),
    ...             _obj_of_data=lambda data: ord(data))
    >>> test_store(s)
    >>>
    >>> # using a Mixin class #############################################################################
    >>> class Mixin:
    ...     def _id_of_key(self, k):
    ...         return k.upper()
    ...     def _key_of_id(self, _id):
    ...         return _id.lower()
    ...     def _data_of_obj(self, obj):
    ...         return chr(obj)
    ...     def _obj_of_data(self, data):
    ...         return ord(data)
    ...
    >>> class MyStore(Mixin, Store):  # note that the Mixin must come before Store in the mro
    ...     pass
    ...
    >>> s = MyStore()  # no dict()? No, because default anyway
    >>> test_store(s)
    >>>
    >>> # adding wrapper methods to an already made Store instance #########################################
    >>> s = Store(dict())
    >>> s._id_of_key=lambda k: k.upper()
    >>> s._key_of_id=lambda _id: _id.lower()
    >>> s._data_of_obj=lambda obj: chr(obj)
    >>> s._obj_of_data=lambda data: ord(data)
    >>> test_store(s)

    Note on defining your own "Mapping Views".

    When you do a `.keys()`, a `.values()` or `.items()` you're getting a `MappingView`
    instance; an iterable and sized container that provides some methods to access
    particular aspects of the wrapped mapping.

    If you need to customize the behavior of these instances, you should avoid
    overriding the `keys`, `values` or `items` methods directly, but instead
    override the `KeysView`, `ValuesView` or `ItemsView` classes that they use.

    For more, see: https://github.com/i2mint/dol/wiki/Mapping-Views

    """

    _state_attrs = ["store", "_class_wrapper"]
    # __slots__ = ('_id_of_key', '_key_of_id', '_data_of_obj', '_obj_of_data')

    def __init__(self, store=dict):
        # self._wrapped_methods = set(dir(Store))

        if isinstance(store, type):
            store = store()

        self.store = store

        if hasattr(self.store, "KeysView"):
            self.KeysView = self.store.KeysView

        if hasattr(self.store, "ValuesView"):
            self.ValuesView = self.store.ValuesView

        if hasattr(self.store, "ItemsView"):
            self.ItemsView = self.store.ItemsView

    _id_of_key = static_identity_method
    _key_of_id = static_identity_method
    _data_of_obj = static_identity_method
    _obj_of_data = static_identity_method

    _max_repr_size = None

    _errors_that_trigger_missing = (
        KeyError,
    )  # another option: (KeyError, FileNotFoundError)

    wrap = classmethod(partial(delegator_wrap, delegation_attr="store"))

    def __getattr__(self, attr):
        """Delegate method to wrapped store if not part of wrapper store methods"""
        # Instead of return getattr(self.store, attr), doing the following
        # because self.store had problems with pickling
        return getattr(object.__getattribute__(self, "store"), attr)

    def __dir__(self):
        return list(
            set(dir(self.__class__)).union(self.store.__dir__())
        )  # to forward dir to delegated stream as well

    def __hash__(self):
        return hash(self.store)
        # changed from the following (store.__hash__ was None sometimes (so not callable)
        # return self.store.__hash__()

    # Read ####################################################################

    def __getitem__(self, k: Key) -> Val:
        # essentially: self._obj_of_data(self.store[self._id_of_key(k)])
        _id = self._id_of_key(k)
        try:
            data = self.store[_id]
        except self._errors_that_trigger_missing as error:
            if hasattr(self, "__missing__"):
                data = self.__missing__(k)
            else:
                raise error
        return self._obj_of_data(data)

    def get(self, k: Key, default=None) -> Val:
        try:
            return self[k]
        except KeyError:
            return default

    # Explore ####################################################################
    def __iter__(self) -> KeyIter:
        yield from (self._key_of_id(k) for k in self.store)
        # return map(self._key_of_id, self.store.__iter__())

    def __len__(self) -> int:
        return len(self.store)

    def __contains__(self, k) -> bool:
        return self._id_of_key(k) in self.store

    def head(self) -> Item:
        k = None
        try:
            for k in self:
                return k, self[k]
        except Exception as e:
            from warnings import warn

            if k is None:
                raise
            else:
                msg = f"Couldn't get data for the key {k}. This could be be...\n"
                msg += "... because it's not a store (just a collection, that doesn't have a __getitem__)\n"
                msg += (
                    "... because there's a layer transforming outcoming keys that are not the ones the store actually "
                    "uses? If you didn't wrap the store with the inverse ingoing keys transformation, "
                    "that would happen.\n"
                )
                msg += (
                    "I'll ask the inner-layer what it's head is, but IT MAY NOT REFLECT the reality of your store "
                    "if you have some filtering, caching etc."
                )
                msg += f"The error messages was: \n{e}"
                warn(msg)

            for _id in self.store:
                return self._key_of_id(_id), self._obj_of_data(self.store[_id])
        # NOTE: Old version didn't work when key mapping was asymmetrical
        # for k, v in self.items():
        #     return k, v

    # Write ####################################################################
    def __setitem__(self, k: Key, v: Val):
        return self.store.__setitem__(self._id_of_key(k), self._data_of_obj(v))

    # def update(self, *args, **kwargs):
    #     return self.store.update(*args, **kwargs)

    # Delete ####################################################################
    def __delitem__(self, k: Key):
        return self.store.__delitem__(self._id_of_key(k))

    # def clear(self):
    #     raise NotImplementedError('''
    #     The clear method was overridden to make dangerous difficult.
    #     If you really want to delete all your data, you can do so by doing:
    #         try:
    #             while True:
    #                 self.popitem()
    #         except KeyError:
    #             pass''')

    # Misc ####################################################################
    # TODO: Review this -- must be a better overall solution!
    def __repr__(self):
        x = repr(self.store)
        if isinstance(self._max_repr_size, int):
            half = int(self._max_repr_size)
            if len(x) > self._max_repr_size:
                x = x[:half] + "  ...  " + x[-half:]
        return x
        # return self.store.__repr__()

    def __getstate__(self) -> dict:
        state = {}
        for attr in Store._state_attrs:
            if hasattr(self, attr):
                state[attr] = getattr(self, attr)
        return state

    def __setstate__(self, state: dict):
        for attr in Store._state_attrs:
            if attr in state:
                setattr(self, attr, state[attr])


# Store.register(dict)  # TODO: Would this be a good idea? To make isinstance({}, Store) be True (though missing head())
KvStore = Store  # alias with explict name

########################################################################################################################
# walking in trees

from typing import Callable, KT, VT, Any, TypeVar, Iterator
from collections import deque

PT = TypeVar("PT")  # Path Type
inf = float("infinity")


def val_is_mapping(p: PT, k: KT, v: VT) -> bool:
    return isinstance(v, Mapping)


def asis(p: PT, k: KT, v: VT) -> Any:
    return p, k, v


def tuple_keypath_and_val(p: PT, k: KT, v: VT) -> Tuple[PT, VT]:
    if p == ():  # we're just begining (the root),
        p = (k,)  # so begin the path with the first key.
    else:
        p = (*p, k)  # extend the path (append the new key)
    return p, v


# TODO: More docs and doctests.
#  This one even merits an extensive usage and example tutorial!
def kv_walk(
    v: Mapping,
    leaf_yield: Callable[[PT, KT, VT], Any] = asis,
    walk_filt: Callable[[PT, KT, VT], bool] = val_is_mapping,
    pkv_to_pv: Callable[[PT, KT, VT], Tuple[PT, VT]] = tuple_keypath_and_val,
    *,
    branch_yield: Callable[[PT, KT, VT], Any] = None,
    breadth_first: bool = False,
    p: PT = (),
) -> Iterator[Any]:
    """
    Walks a nested structure of mappings, yielding stuff on the way.

    :param v: A nested structure of mappings
    :param leaf_yield: (pp, k, vv) -> Any, what you want to yield when you encounter
        a leaf node (as define by walk_filt resolving to False)
    :param walk_filt: (p, k, vv) -> (bool) whether to explore the nested structure v further
    :param pkv_to_pv:  (p, k, v) -> (pp, vv)
        where pp is a form of p + k (update of the path with the new node k)
        and vv is the value that will be used by both walk_filt and leaf_yield
    :param p: The path to v (used internally, mainly, to keep track of the path)
    :param breadth_first: Whether to perform breadth-first traversal
        (instead of the default depth-first traversal).
    :param branch_yield: (pp, k, vv) -> Any, optional yield function to yield before
        the recursive walk of a branch. This is useful if you want to yield something
        for every branch, not just the leaves.

    >>> d = {'a': 1, 'b': {'c': 2, 'd': 3}}
    >>> list(kv_walk(d))
    [(('a',), 'a', 1), (('b', 'c'), 'c', 2), (('b', 'd'), 'd', 3)]
    >>> list(kv_walk(d, lambda p, k, v: '.'.join(p)))
    ['a', 'b.c', 'b.d']

    The `walk_filt` argument allows you to control what values the walk encountered
    should be walked through. This also means that this function is what controls
    when to stop the recursive traversal of the tree, and yield an actual "leaf".

    Say we want to get (path, values) items from a nested mapping/store based on
    a ``levels`` argument that determines what the desired values are.
    This can be done as follows:

    >>> def mk_level_walk_filt(levels):
    ...     return lambda p, k, v: len(p) < levels - 1
    ...
    >>> def leveled_map_walk(m, levels):
    ...     yield from kv_walk(
    ...         m,
    ...         leaf_yield=lambda p, k, v: (p, v),
    ...         walk_filt=mk_level_walk_filt(levels)
    ...     )
    >>> m = {
    ...     'a': {'b': {'c': 42}},
    ...     'aa': {'bb': {'cc': 'dragon_con'}}
    ... }
    >>>
    >>> assert (
    ...         list(leveled_map_walk(m, 3))
    ...         == [
    ...             (('a', 'b', 'c'), 42),
    ...             (('aa', 'bb', 'cc'), 'dragon_con')
    ...         ]
    ... )
    >>> assert (
    ...         list(leveled_map_walk(m, 2))
    ...         == [
    ...             (('a', 'b'), {'c': 42}),
    ...             (('aa', 'bb'), {'cc': 'dragon_con'})
    ...         ]
    ... )
    >>>
    >>> assert (
    ...         list(leveled_map_walk(m, 1))
    ...         == [
    ...             (('a',), {'b': {'c': 42}}),
    ...             (('aa',), {'bb': {'cc': 'dragon_con'}})
    ...         ]
    ... )

    Tip: If you want to use ``kv_filt`` to search and extract stuff from a nested
    mapping, you can have your ``leaf_yield`` return a sentinel (say, ``None``) to
    indicate that the value should be skipped, and then filter out the ``None``s from
    your results.

    >>> mm = {
    ...     'a': {'b': {'c': 42}},
    ...     'aa': {'bb': {'cc': 'meaning_of_life'}},
    ...     'aaa': {'bbb': 314},
    ... }
    >>> return_path_if_int_leaf = lambda p, k, v: (p, v) if isinstance(v, int) else None
    >>> list(filter(None, kv_walk(mm, leaf_yield=return_path_if_int_leaf)))
    [(('a', 'b', 'c'), 42), (('aaa', 'bbb'), 314)]

    This "path search" functionality is available as a function in the ``recipes``
    module, as ``search_paths``.

    One last thing. Let's demonstrate the use of `branch_yield` and `breadth_first`.
    Consider the following dictionary:

    >>> d = {'big': {'apple': 1}, 'deal': 3, 'apple': {'pie': 1, 'crumble': 2}}

    Say you wanted to find all the paths that end with 'apple'. You could do:

    >>> from functools import partial
    >>> yield_path_if_ends_with_apple = lambda p, k, v: p if k == 'apple' else None
    >>> walker1 = partial(kv_walk, leaf_yield=yield_path_if_ends_with_apple)
    >>> list(filter(None, walker1(d)))
    [('big', 'apple')]

    It only got `('big', 'apple')` because the `leaf_yield` is only triggered
    for leaf nodes (as defined by the `walk_filt` argument, which defaults to
    `val_is_mapping`). So let's try again, but this time, we'll use `branch_yield`
    to yield the path for every branch (not just the leaves):

    >>> walker2 = partial(walker1, branch_yield=yield_path_if_ends_with_apple)
    >>> list(filter(None, walker2(d)))
    [('big', 'apple'), ('apple',)]

    But this isn't convenient if you'd like your search to finish as soon as you
    find a path ending with `'apple'`. The order here comes from the fact that
    `kv_walk` does a depth-first traversal. If you want to do a breadth-first
    traversal, just say it:

    >>> walker3 = partial(walker2, breadth_first=True)
    >>> list(filter(None, walker3(d)))
    [('apple',), ('big', 'apple')]

    So now, you can get the first apple path by doing:
    >>> next(filter(None, walker3(d)))
    ('apple',)

    """
    if not breadth_first:
        # print(f"1: entered with: v={v}, p={p}")
        for k, vv in v.items():
            # print(f"2: item: k={k}, vv={vv}")
            pp, vv = pkv_to_pv(
                p, k, vv
            )  # update the path with k (and preprocess v if necessary)
            if walk_filt(
                p, k, vv
            ):  # should we recurse? (based on some function of p, k, v)
                # print(f"3: recurse with: pp={pp}, vv={vv}\n")
                if branch_yield:
                    yield branch_yield(pp, k, vv)
                yield from kv_walk(
                    vv,
                    leaf_yield,
                    walk_filt,
                    pkv_to_pv,
                    breadth_first=breadth_first,
                    branch_yield=branch_yield,
                    p=pp,
                )  # recurse
            else:
                # print(f"4: leaf_yield(pp={pp}, k={k}, vv={vv})\n --> {leaf_yield(pp, k, vv)}")
                yield leaf_yield(pp, k, vv)  # yield something computed from p, k, vv
    else:
        queue = deque([(p, v)])

        while queue:
            p, v = queue.popleft()
            for k, vv in v.items():
                pp, vv = pkv_to_pv(p, k, vv)
                if walk_filt(p, k, vv):
                    if branch_yield:
                        yield branch_yield(pp, k, vv)
                    queue.append((pp, vv))
                else:
                    yield leaf_yield(pp, k, vv)


def has_kv_store_interface(o):
    """Check if object has the KvStore interface (that is, has the kv wrapper methods

    Args:
        o: object (class or instance)

    Returns: True if kv has the four key (in/out) and value (in/out) transformation methods

    """
    return (
        hasattr(o, "_id_of_key")
        and hasattr(o, "_key_of_id")
        and hasattr(o, "_data_of_obj")
        and hasattr(o, "_obj_of_data")
    )


from abc import ABCMeta, abstractmethod
from dol.errors import KeyValidationError


def _check_methods(C, *methods):
    """
    Check that all methods listed are in the __dict__ of C, or in the classes of it's mro.
    One trick pony borrowed from collections.abc.
    """
    mro = C.__mro__
    for method in methods:
        for B in mro:
            if method in B.__dict__:
                if B.__dict__[method] is None:
                    return NotImplemented
                break
        else:
            return NotImplemented
    return True


# Note: Not sure I want to do key validation this way. Perhaps better injected in _id_of_key?
class KeyValidationABC(metaclass=ABCMeta):
    """
    An ABC for an object writer.
    Single purpose: store an object under a given key.
    How the object is serialized and or physically stored should be defined in a concrete subclass.
    """

    __slots__ = ()

    @abstractmethod
    def is_valid_key(self, k):
        pass

    def check_key_is_valid(self, k):
        if not self.is_valid_key(k):
            raise KeyValidationError("key is not valid: {}".format(k))

    @classmethod
    def __subclasshook__(cls, C):
        if cls is KeyValidationABC:
            return _check_methods(C, "is_valid_key", "check_key_is_valid")
        return NotImplemented


########################################################################################################################
# Streams


class stream_util:
    def always_true(*args, **kwargs):
        return True

    def do_nothing(*args, **kwargs):
        pass

    def rewind(self, instance):
        instance.seek(0)

    def skip_lines(self, instance, n_lines_to_skip=0):
        instance.seek(0)


class Stream:
    """A layer-able version of the stream interface

        __iter__    calls: _obj_of_data(map)

    >>> from io import StringIO
    >>>
    >>> src = StringIO(
    ... '''a, b, c
    ... 1,2, 3
    ... 4, 5,6
    ... '''
    ... )
    >>>
    >>> from dol.base import Stream
    >>>
    >>> class MyStream(Stream):
    ...     def _obj_of_data(self, line):
    ...         return [x.strip() for x in line.strip().split(',')]
    ...
    >>> stream = MyStream(src)
    >>>
    >>> list(stream)
    [['a', 'b', 'c'], ['1', '2', '3'], ['4', '5', '6']]
    >>> stream.seek(0)  # oh!... but we consumed the stream already, so let's go back to the beginning
    0
    >>> list(stream)
    [['a', 'b', 'c'], ['1', '2', '3'], ['4', '5', '6']]
    >>> stream.seek(0)  # reverse again
    0
    >>> next(stream)
    ['a', 'b', 'c']
    >>> next(stream)
    ['1', '2', '3']

    Let's add a filter! There's two kinds you can use.
    One that is applied to the line before the data is transformed by _obj_of_data,
    and the other that is applied after (to the obj).


    >>> from dol.base import Stream
    >>> from io import StringIO
    >>>
    >>> src = StringIO(
    ...     '''a, b, c
    ... 1,2, 3
    ... 4, 5,6
    ... ''')
    >>> class MyFilteredStream(MyStream):
    ...     def _post_filt(self, obj):
    ...         return str.isnumeric(obj[0])
    >>>
    >>> s = MyFilteredStream(src)
    >>>
    >>> list(s)
    [['1', '2', '3'], ['4', '5', '6']]
    >>> s.seek(0)
    0
    >>> list(s)
    [['1', '2', '3'], ['4', '5', '6']]
    >>> s.seek(0)
    0
    >>> next(s)
    ['1', '2', '3']

    Recipes:

    .. hlist::
        * _pre_iter: involving itertools.islice to skip header lines
        * _pre_iter: involving enumerate to get line indices in stream iterator
        * _pre_iter = functools.partial(map, line_pre_proc_func) to preprocess all lines with line_pre_proc_func
        * _pre_iter: include filter before obj
    """

    def __init__(self, stream):
        self.stream = stream

    wrap = classmethod(partial(delegator_wrap, delegation_attr="stream"))

    # _data_of_obj = static_identity_method  # for write methods
    _pre_iter = static_identity_method
    _obj_of_data = static_identity_method
    _post_filt = stream_util.always_true

    def __iter__(self):
        for line in self._pre_iter(self.stream):
            obj = self._obj_of_data(line)
            if self._post_filt(obj):
                yield obj

        # TODO: See pros and cons of above vs below:
        # yield from filter(self._post_filt,
        #                   map(self._obj_of_data,
        #                       self._pre_iter(self.stream)))

    # _wrapped_methods = {'__iter__'}

    def __next__(self):  # TODO: Pros and cons of having a __next__?
        return next(iter(self))

    def __getattr__(self, attr):
        """Delegate method to wrapped store if not part of wrapper store methods"""
        return getattr(self.stream, attr)
        # if attr in self._wrapped_methods:
        #     return getattr(self, attr)
        # else:
        #     return getattr(self.stream, attr)

    def __enter__(self):
        self.stream.__enter__()
        return self
        # return self._pre_proc(self.stream) # moved to iter to

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.stream.__exit__(
            exc_type, exc_val, exc_tb
        )  # TODO: Should we have a _post_proc? Uses?


########################################################################################################################
