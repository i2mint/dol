# dol.base

Base classes for making stores.
In the language of the collections.abc module, a store is a MutableMapping that is configured to work with a specific
representation of keys, serialization of objects (python values), and persistence of the serialized data.

That is, stores offer the same interface as a dict, but where the actual implementation of writes, reads, and listing
are configurable.

Consider the following example. You’re store is meant to store waveforms as wav files on a remote server.
Say waveforms are represented in python as a tuple (wf, sr), where wf is a list of numbers and sr is the sample
rate, an int). The \_\_setitem_\_ method will specify how to store bytes on a remote server, but you’ll need to specify
how to SERIALIZE (wf, sr) to the bytes that constitute that wav file: \_data_of_obj specifies that.
You might also want to read those wav files back into a python (wf, sr) tuple. The \_\_getitem_\_ method will get
you those bytes from the server, but the store will need to know how to DESERIALIZE those bytes back into a python
object: \_obj_of_data specifies that

Further, say you’re storing these .wav files in /some/folder/on/the/server/, but you don’t want the store to use
these as the keys. For one, it’s annoying to type and harder to read. But more importantly, it’s an irrelevant
implementation detail that shouldn’t be exposed. THe \_id_of_key and \_key_of_id pair are what allow you to
add this key interface layer.

These key converters object serialization methods default to the identity (i.e. they return the input as is).
This means that you don’t have to implement these as all, and can choose to implement these concerns within
the storage methods themselves.

### Functions

| `asis`(p, k, v)                                                                           |                                                                                  |
|-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `delegate_to`(wrapped[, class_trans, ...])                                                |                                                                                  |
| [`delegator_wrap`](#dol.base.delegator_wrap)(delegator, obj[, ...])    | Wrap a `obj` (type or instance) with `delegator`.                                |
| [`has_kv_store_interface`](#dol.base.has_kv_store_interface)(o)                | Check if object has the KvStore interface (that is, has the kv wrapper methods   |
| [`kv_walk`](#dol.base.kv_walk)(v[, leaf_yield, walk_filt, ...]) | Walks a nested structure of mappings, yielding stuff on the way.                 |
| `tuple_keypath_and_val`(p, k, v)                                                          |                                                                                  |
| `val_is_mapping`(p, k, v)                                                                 |                                                                                  |
| `wrapped_delegator_reconstruct`(wrapped_cls, ...)                                         |                                                                                  |
| [`wrapped_self`](#dol.base.wrapped_self)(obj)                        | Return the outermost transform-applying store wrapping `obj`, else `obj` itself. |

### Classes

| `AttrNames`()                                                       |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|---------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`Collection`](#dol.base.Collection)()       | The same as collections.abc.Collection, with some modifications:                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `DelegatedAttribute`(delegate_name, attr_name)                      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| [`KeyValidationABC`](#dol.base.KeyValidationABC)() | An ABC for an object writer.                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| [`KvPersister`](#dol.base.KvPersister)()      | Acts as a MutableMapping abc, but disabling the clear and \_\_reversed_\_ method, and computing \_\_len_\_ by iterating over all keys, and counting them.                                                                                                                                                                                                                                                                                                                               |
| [`KvReader`](#dol.base.KvReader)()         | Acts as a Mapping abc, but with default \_\_len_\_ (implemented by counting keys) and head method to get the first (k, v) item of the store                                                                                                                                                                                                                                                                                                                                             |
| [`KvStore`](#dol.base.KvStore)            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `MappingViewMixin`()                                                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `NoSuchItem`()                                                      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| [`Persister`](#dol.base.Persister)          |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| [`Reader`](#dol.base.Reader)             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| [`Store`](#dol.base.Store)([store])     | By store we mean key-value store. This could be files in a filesystem, objects in s3, or a database. Where and how the content is stored should be specified, but StoreInterface offers a dict-like interface to this. ::     \_\_getitem_\_ calls: \_id_of_key                                       \_obj_of_data     \_\_setitem_\_ calls: \_id_of_key                   \_data_of_obj     \_\_delitem_\_ calls: \_id_of_key     \_\_iter_\_    calls:                  \_key_of_id. |
| [`Stream`](#dol.base.Stream)(stream)     | A layer-able version of the stream interface                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `stream_util`()                                                     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

### *class* dol.base.Collection

Bases: [`Collection`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Collection)

The same as collections.abc.Collection, with some modifications:

- Addition of a `head`

### *class* dol.base.KeyValidationABC

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

An ABC for an object writer.
Single purpose: store an object under a given key.
How the object is serialized and or physically stored should be defined in a concrete subclass.

### *class* dol.base.KvPersister

Bases: [`KvReader`](#dol.base.KvReader), [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

Acts as a MutableMapping abc, but disabling the clear and \_\_reversed_\_ method,
and computing \_\_len_\_ by iterating over all keys, and counting them.

Note that KvPersister is a MutableMapping, and as such, is dict-like.
But that doesn’t mean it’s a dict.

For instance, consider the following code:

```python
s = SomeKvPersister()
s['a']['b'] = 3
```

If `s` is a dict, this would have the effect of adding a (‘b’, 3) item under ‘a’.
But in the general case, this might

- fail, because the `s['a']` doesn’t support sub-scripting (doesn’t have a `__getitem__`)
- or, worse, will pass silently but not actually persist the write as expected (e.g. LocalFileStore)

Another example: `s.popitem()` will pop a `(k, v)` pair off of the `s` store.
That is, retrieve the `v` for `k`, delete the entry for `k`, and return a `(k, v)`.
Note that unlike modern dicts which will return the last item that was stored

> – that is, LIFO (last-in, first-out) order – for KvPersisters,
> there’s no assurance as to what item will be, since it will depend on the backend storage system
> and/or how the persister was implemented.

#### clear()

The clear method is disabled to make dangerous difficult.
You don’t want to delete your whole DB
If you really want to delete all your data, you can do so by doing something like this:

```python
for k in self:
    del self[k]
```

or (in some cases)

```python
for k in self:
    try:
        del self[k]
    except KeyError:
        pass
```

### *class* dol.base.KvReader

Bases: `MappingViewMixin`, [`Collection`](#dol.base.Collection), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

Acts as a Mapping abc, but with default \_\_len_\_ (implemented by counting keys)
and head method to get the first (k, v) item of the store

#### head()

Get the first (key, value) pair

### dol.base.KvStore

alias of [`Store`](#dol.base.Store)

### dol.base.Persister

alias of [`KvPersister`](#dol.base.KvPersister)

### dol.base.Reader

alias of [`KvReader`](#dol.base.KvReader)

### *class* dol.base.Store(store=<class 'dict'>)

Bases: [`KvPersister`](#dol.base.KvPersister)

By store we mean key-value store. This could be files in a filesystem, objects in s3, or a database. Where and
how the content is stored should be specified, but StoreInterface offers a dict-like interface to this.

```default
__getitem__ calls: _id_of_key                                       _obj_of_data
__setitem__ calls: _id_of_key                   _data_of_obj
__delitem__ calls: _id_of_key
__iter__    calls:                  _key_of_id
```

```pycon
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
```

Note on defining your own “Mapping Views”.

When you do a `.keys()`, a `.values()` or `.items()` you’re getting a `MappingView`
instance; an iterable and sized container that provides some methods to access
particular aspects of the wrapped mapping.

If you need to customize the behavior of these instances, you should avoid
overriding the `keys`, `values` or `items` methods directly, but instead
override the `KeysView`, `ValuesView` or `ItemsView` classes that they use.

For more, see: [https://github.com/i2mint/dol/wiki/Mapping-Views](https://github.com/i2mint/dol/wiki/Mapping-Views)

#### get(k, default=None)

* **Return type:**
  [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Val`)

#### head()

Get the first (key, value) pair

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

#### *classmethod* wrap(obj, class_trans=None, , delegation_attr='store')

Wrap a `obj` (type or instance) with `delegator`.

If obj is not a type, trivially returns `delegator(obj)`.

The interesting case of `delegator_wrap` is when `obj` is a type (a class).
In this case, `delegator_wrap` returns a callable (class or function) that has the
same signature as obj, but that produces instances that are wrapped by `delegator`

* **Parameters:**
  * **delegator** – An instance wrapper. A Callable (type or function – with only
    one required input) that will return a wrapped version of it’s input instance.
  * **obj** – The object (class or instance) to be wrapped.
* **Returns:**
  A wrapped object

Let’s demo this on a simple Delegator class.

```pycon
>>> class Delegator:
...     i_think = 'therefore I am delegated'  # this is just to verify that we're in a Delegator
...     def __init__(self, wrapped_obj):
...         self.wrapped_obj = wrapped_obj
...     def __getattr__(self, attr):  # delegation: just forward attributes to wrapped_obj
...         return getattr(self.wrapped_obj, attr)
...     wrap = classmethod(delegator_wrap)  # this is a useful recipe to have the Delegator carry it's own wrapping method
```

The only difference between a wrapped object `Delegator(obj)` and the original `obj` is
that the wrapped one has a `i_think` attribute.
The wrapped object should otherwise behave the same (on all but special (dunder) methods).
So let’s test this on dictionaries, using the following test function:

```pycon
>>> def test_wrapped_d(wrapped_d, original_d):
...     '''A function to test a wrapped dict'''
...     assert not hasattr(original_d, 'i_think')  # verify that the unwrapped_d doesn't have an i_think attribute
...     assert list(wrapped_d.items()) == list(original_d.items())  # verify that wrapped_d has an items that gives us the same thing as origina_d
...     assert hasattr(wrapped_d, 'i_think')  # ... but wrapped_d has a i_think attribute
...     assert wrapped_d.i_think == 'therefore I am delegated'  # ... and its what we set it to be
```

Let’s try delegating a dict INSTANCE first:

```pycon
>>> d = {'a': 1, 'b': 2}
>>> wrapped_d = delegator_wrap(Delegator, d)
>>> test_wrapped_d(wrapped_d, d)
```

If we ask `delegator_wrap` to wrap a `dict` type, we get a subclass of Delegator
(NOT dict!) whose instances will have the behavior exhibited above:

```pycon
>>> WrappedDict = delegator_wrap(Delegator, dict, delegation_attr='wrapped_obj')
>>> assert issubclass(WrappedDict, Delegator)
>>> wrapped_d = WrappedDict(a=1, b=2)
```

```pycon
>>> test_wrapped_d(wrapped_d, wrapped_d.wrapped_obj)
```

Now we’ll demo/test the `wrap = classmethod(delegator_wrap)` trick
… with instances

```pycon
>>> wrapped_d = Delegator.wrap(d)
>>> test_wrapped_d(wrapped_d, wrapped_d.wrapped_obj)
```

… with classes

```pycon
>>> WrappedDict = Delegator.wrap(dict, delegation_attr='wrapped_obj')
>>> wrapped_d = WrappedDict(a=1, b=2)
```

```pycon
>>> test_wrapped_d(wrapped_d, wrapped_d.wrapped_obj)
>>> class A(dict):
...     def foo(self, x):
...         pass
>>> hasattr(A, 'foo')
True
>>> WrappedA = Delegator.wrap(A)
>>> hasattr(WrappedA, 'foo')
True
```

### *class* dol.base.Stream(stream)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

A layer-able version of the stream interface

> \_\_iter_\_    calls: \_obj_of_data(map)
```pycon
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
```

Let’s add a filter! There’s two kinds you can use.
One that is applied to the line before the data is transformed by \_obj_of_data,
and the other that is applied after (to the obj).

```pycon
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
```

Recipes:

#### *classmethod* wrap(obj, class_trans=None, , delegation_attr='stream')

Wrap a `obj` (type or instance) with `delegator`.

If obj is not a type, trivially returns `delegator(obj)`.

The interesting case of `delegator_wrap` is when `obj` is a type (a class).
In this case, `delegator_wrap` returns a callable (class or function) that has the
same signature as obj, but that produces instances that are wrapped by `delegator`

* **Parameters:**
  * **delegator** – An instance wrapper. A Callable (type or function – with only
    one required input) that will return a wrapped version of it’s input instance.
  * **obj** – The object (class or instance) to be wrapped.
* **Returns:**
  A wrapped object

Let’s demo this on a simple Delegator class.

```pycon
>>> class Delegator:
...     i_think = 'therefore I am delegated'  # this is just to verify that we're in a Delegator
...     def __init__(self, wrapped_obj):
...         self.wrapped_obj = wrapped_obj
...     def __getattr__(self, attr):  # delegation: just forward attributes to wrapped_obj
...         return getattr(self.wrapped_obj, attr)
...     wrap = classmethod(delegator_wrap)  # this is a useful recipe to have the Delegator carry it's own wrapping method
```

The only difference between a wrapped object `Delegator(obj)` and the original `obj` is
that the wrapped one has a `i_think` attribute.
The wrapped object should otherwise behave the same (on all but special (dunder) methods).
So let’s test this on dictionaries, using the following test function:

```pycon
>>> def test_wrapped_d(wrapped_d, original_d):
...     '''A function to test a wrapped dict'''
...     assert not hasattr(original_d, 'i_think')  # verify that the unwrapped_d doesn't have an i_think attribute
...     assert list(wrapped_d.items()) == list(original_d.items())  # verify that wrapped_d has an items that gives us the same thing as origina_d
...     assert hasattr(wrapped_d, 'i_think')  # ... but wrapped_d has a i_think attribute
...     assert wrapped_d.i_think == 'therefore I am delegated'  # ... and its what we set it to be
```

Let’s try delegating a dict INSTANCE first:

```pycon
>>> d = {'a': 1, 'b': 2}
>>> wrapped_d = delegator_wrap(Delegator, d)
>>> test_wrapped_d(wrapped_d, d)
```

If we ask `delegator_wrap` to wrap a `dict` type, we get a subclass of Delegator
(NOT dict!) whose instances will have the behavior exhibited above:

```pycon
>>> WrappedDict = delegator_wrap(Delegator, dict, delegation_attr='wrapped_obj')
>>> assert issubclass(WrappedDict, Delegator)
>>> wrapped_d = WrappedDict(a=1, b=2)
```

```pycon
>>> test_wrapped_d(wrapped_d, wrapped_d.wrapped_obj)
```

Now we’ll demo/test the `wrap = classmethod(delegator_wrap)` trick
… with instances

```pycon
>>> wrapped_d = Delegator.wrap(d)
>>> test_wrapped_d(wrapped_d, wrapped_d.wrapped_obj)
```

… with classes

```pycon
>>> WrappedDict = Delegator.wrap(dict, delegation_attr='wrapped_obj')
>>> wrapped_d = WrappedDict(a=1, b=2)
```

```pycon
>>> test_wrapped_d(wrapped_d, wrapped_d.wrapped_obj)
>>> class A(dict):
...     def foo(self, x):
...         pass
>>> hasattr(A, 'foo')
True
>>> WrappedA = Delegator.wrap(A)
>>> hasattr(WrappedA, 'foo')
True
```

### dol.base.delegator_wrap(delegator, obj, class_trans=None, delegation_attr='store')

Wrap a `obj` (type or instance) with `delegator`.

If obj is not a type, trivially returns `delegator(obj)`.

The interesting case of `delegator_wrap` is when `obj` is a type (a class).
In this case, `delegator_wrap` returns a callable (class or function) that has the
same signature as obj, but that produces instances that are wrapped by `delegator`

* **Parameters:**
  * **delegator** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – An instance wrapper. A Callable (type or function – with only
    one required input) that will return a wrapped version of it’s input instance.
  * **obj** ([`type`](https://docs.python.org/3/library/functions.html#type) | [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The object (class or instance) to be wrapped.
* **Returns:**
  A wrapped object

Let’s demo this on a simple Delegator class.

```pycon
>>> class Delegator:
...     i_think = 'therefore I am delegated'  # this is just to verify that we're in a Delegator
...     def __init__(self, wrapped_obj):
...         self.wrapped_obj = wrapped_obj
...     def __getattr__(self, attr):  # delegation: just forward attributes to wrapped_obj
...         return getattr(self.wrapped_obj, attr)
...     wrap = classmethod(delegator_wrap)  # this is a useful recipe to have the Delegator carry it's own wrapping method
```

The only difference between a wrapped object `Delegator(obj)` and the original `obj` is
that the wrapped one has a `i_think` attribute.
The wrapped object should otherwise behave the same (on all but special (dunder) methods).
So let’s test this on dictionaries, using the following test function:

```pycon
>>> def test_wrapped_d(wrapped_d, original_d):
...     '''A function to test a wrapped dict'''
...     assert not hasattr(original_d, 'i_think')  # verify that the unwrapped_d doesn't have an i_think attribute
...     assert list(wrapped_d.items()) == list(original_d.items())  # verify that wrapped_d has an items that gives us the same thing as origina_d
...     assert hasattr(wrapped_d, 'i_think')  # ... but wrapped_d has a i_think attribute
...     assert wrapped_d.i_think == 'therefore I am delegated'  # ... and its what we set it to be
```

Let’s try delegating a dict INSTANCE first:

```pycon
>>> d = {'a': 1, 'b': 2}
>>> wrapped_d = delegator_wrap(Delegator, d)
>>> test_wrapped_d(wrapped_d, d)
```

If we ask `delegator_wrap` to wrap a `dict` type, we get a subclass of Delegator
(NOT dict!) whose instances will have the behavior exhibited above:

```pycon
>>> WrappedDict = delegator_wrap(Delegator, dict, delegation_attr='wrapped_obj')
>>> assert issubclass(WrappedDict, Delegator)
>>> wrapped_d = WrappedDict(a=1, b=2)
```

```pycon
>>> test_wrapped_d(wrapped_d, wrapped_d.wrapped_obj)
```

Now we’ll demo/test the `wrap = classmethod(delegator_wrap)` trick
… with instances

```pycon
>>> wrapped_d = Delegator.wrap(d)
>>> test_wrapped_d(wrapped_d, wrapped_d.wrapped_obj)
```

… with classes

```pycon
>>> WrappedDict = Delegator.wrap(dict, delegation_attr='wrapped_obj')
>>> wrapped_d = WrappedDict(a=1, b=2)
```

```pycon
>>> test_wrapped_d(wrapped_d, wrapped_d.wrapped_obj)
>>> class A(dict):
...     def foo(self, x):
...         pass
>>> hasattr(A, 'foo')
True
>>> WrappedA = Delegator.wrap(A)
>>> hasattr(WrappedA, 'foo')
True
```

### dol.base.has_kv_store_interface(o)

Check if object has the KvStore interface (that is, has the kv wrapper methods

* **Parameters:**
  **o** – object (class or instance)
* **Returns:**
  True if kv has the four key (in/out) and value (in/out) transformation methods

### dol.base.kv_walk(v, leaf_yield=<function asis>, walk_filt=<function val_is_mapping>, pkv_to_pv=<function tuple_keypath_and_val>, \*, branch_yield=None, breadth_first=False, p=())

Walks a nested structure of mappings, yielding stuff on the way.

* **Parameters:**
  * **v** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – A nested structure of mappings
  * **leaf_yield** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`PT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – (pp, k, vv) -> Any, what you want to yield when you encounter
    a leaf node (as define by walk_filt resolving to False)
  * **walk_filt** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`PT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)], [`bool`](https://docs.python.org/3/library/functions.html#bool)]) – (p, k, vv) -> (bool) whether to explore the nested structure v further
  * **pkv_to_pv** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`PT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)], [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`PT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)]]) – (p, k, v) -> (pp, vv)
    where pp is a form of p + k (update of the path with the new node k)
    and vv is the value that will be used by both walk_filt and leaf_yield
  * **p** ([`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`PT`)) – The path to v (used internally, mainly, to keep track of the path)
  * **breadth_first** ([`bool`](https://docs.python.org/3/library/functions.html#bool)) – Whether to perform breadth-first traversal
    (instead of the default depth-first traversal).
  * **branch_yield** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`PT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – (pp, k, vv) -> Any, optional yield function to yield before
    the recursive walk of a branch. This is useful if you want to yield something
    for every branch, not just the leaves.
* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

```pycon
>>> d = {'a': 1, 'b': {'c': 2, 'd': 3}}
>>> list(kv_walk(d))
[(('a',), 'a', 1), (('b', 'c'), 'c', 2), (('b', 'd'), 'd', 3)]
>>> list(kv_walk(d, lambda p, k, v: '.'.join(p)))
['a', 'b.c', 'b.d']
```

The `walk_filt` argument allows you to control what values the walk encountered
should be walked through. This also means that this function is what controls
when to stop the recursive traversal of the tree, and yield an actual “leaf”.

Say we want to get (path, values) items from a nested mapping/store based on
a `levels` argument that determines what the desired values are.
This can be done as follows:

```pycon
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
```

#### TIP
If you want to use `kv_filt` to search and extract stuff from a nested
mapping, you can have your `leaf_yield` return a sentinel (say, `None`) to
indicate that the value should be skipped, and then filter out the 

```
``
```

None\`\`s from
your results.

```pycon
>>> mm = {
...     'a': {'b': {'c': 42}},
...     'aa': {'bb': {'cc': 'meaning_of_life'}},
...     'aaa': {'bbb': 314},
... }
>>> return_path_if_int_leaf = lambda p, k, v: (p, v) if isinstance(v, int) else None
>>> list(filter(None, kv_walk(mm, leaf_yield=return_path_if_int_leaf)))
[(('a', 'b', 'c'), 42), (('aaa', 'bbb'), 314)]
```

This “path search” functionality is available as a function in the `recipes`
module, as `search_paths`.

One last thing. Let’s demonstrate the use of `branch_yield` and `breadth_first`.
Consider the following dictionary:

```pycon
>>> d = {'big': {'apple': 1}, 'deal': 3, 'apple': {'pie': 1, 'crumble': 2}}
```

Say you wanted to find all the paths that end with ‘apple’. You could do:

```pycon
>>> from functools import partial
>>> yield_path_if_ends_with_apple = lambda p, k, v: p if k == 'apple' else None
>>> walker1 = partial(kv_walk, leaf_yield=yield_path_if_ends_with_apple)
>>> list(filter(None, walker1(d)))
[('big', 'apple')]
```

It only got `('big', 'apple')` because the `leaf_yield` is only triggered
for leaf nodes (as defined by the `walk_filt` argument, which defaults to
`val_is_mapping`). So let’s try again, but this time, we’ll use `branch_yield`
to yield the path for every branch (not just the leaves):

```pycon
>>> walker2 = partial(walker1, branch_yield=yield_path_if_ends_with_apple)
>>> list(filter(None, walker2(d)))
[('big', 'apple'), ('apple',)]
```

But this isn’t convenient if you’d like your search to finish as soon as you
find a path ending with `'apple'`. The order here comes from the fact that
`kv_walk` does a depth-first traversal. If you want to do a breadth-first
traversal, just say it:

```pycon
>>> walker3 = partial(walker2, breadth_first=True)
>>> list(filter(None, walker3(d)))
[('apple',), ('big', 'apple')]
```

So now, you can get the first apple path by doing:

```pycon
>>> next(filter(None, walker3(d)))
('apple',)
```

### dol.base.wrapped_self(obj)

Return the outermost transform-applying store wrapping `obj`, else `obj` itself.

Use this inside a method DEFINED ON a class that was wrapped by dol’s delegation
machinery (`wrap_kvs` / `filt_iter` / `cached_keys` / `mk_relative_path_store` /
`Store.wrap` applied to the *class*). There, `self` is bound to the inner, unwrapped
store, so `self[k]` bypasses the transforms (Issue #18). Writing
`wrapped_self(self)[k]` recovers the outer, transform-applying view.

On a direct `Store`/`KvReader` subclass, or a plain object that never went through
the delegation machinery, there is no registered wrapper and `obj` is returned
unchanged – a safe no-op. If the store was wrapped in several layers (e.g. via
`Pipe`), this climbs to the OUTERMOST wrapper.

Limitation: if the *same inner store instance* is wrapped by several live wrappers with
different transforms (e.g. calling `wrap_kvs(shared_instance, ...)` twice), a method
bound to that shared inner cannot know which wrapper it was reached through, so one of
the wrappers is returned (ambiguous but never raw/None). This does not arise for
class-wraps (each construction builds its own inner) nor for `copy.copy` (whose copies
are transform-equivalent, so either answer is correct).

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

```pycon
>>> import math
>>> from dol import wrap_kvs, wrapped_self
>>> sq = wrap_kvs(data_of_obj=lambda x: x * x, obj_of_data=lambda x: math.sqrt(x))
>>> @sq
... class S(dict):
...     def via_self(self, k):
...         return self[k]                 # self is the INNER store: NOT transformed
...     def via_wrapped_self(self, k):
...         return wrapped_self(self)[k]    # outer store: transform applied
>>> s = S()
>>> s['2'] = 2
>>> s['2']                                 # external access is transformed
2.0
>>> s.via_self('2')                        # Issue #18: bypasses the transform
4
>>> s.via_wrapped_self('2')                # wrapped_self recovers the transformed value
2.0
```
