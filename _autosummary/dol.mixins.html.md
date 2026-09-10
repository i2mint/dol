# dol.mixins

Mixins

### Classes

| [`FilteredKeysMixin`](#dol.mixins.FilteredKeysMixin)()            | Filters \_\_iter_\_ and \_\_contains_\_ with (the boolean filter function attribute) \_key_filt.   |
|---------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `GetBasedContainerMixin`()                                                      |                                                                                                    |
| `HashableMixin`()                                                               |                                                                                                    |
| [`IdentityKeysWrapMixin`](#dol.mixins.IdentityKeysWrapMixin)()        | Transparent KeysWrapABC.                                                                           |
| [`IdentityKvWrapMixin`](#dol.mixins.IdentityKvWrapMixin)()          | Transparent Keys and Vals Wrap                                                                     |
| [`IdentityValsWrapMixin`](#dol.mixins.IdentityValsWrapMixin)()        | Transparent ValsWrapABC.                                                                           |
| `IterBasedContainerMixin`()                                                     |                                                                                                    |
| [`IterBasedSizedContainerMixin`](#dol.mixins.IterBasedSizedContainerMixin)() | An ABC that defines                                                                                |
| `IterBasedSizedMixin`()                                                         |                                                                                                    |
| [`OverWritesNotAllowedMixin`](#dol.mixins.OverWritesNotAllowedMixin)()    | Mixin for only allowing a write to a key if they key doesn't already exist.                        |
| [`ReadOnlyMixin`](#dol.mixins.ReadOnlyMixin)()                | Put this as your first parent class to disallow write/delete operations                            |
| [`SimpleJsonMixin`](#dol.mixins.SimpleJsonMixin)()              | simple json serialization.                                                                         |
| [`StringKvWrap`](#dol.mixins.StringKvWrap)()                 |                                                                                                    |

### *class* dol.mixins.FilteredKeysMixin

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Filters \_\_iter_\_ and \_\_contains_\_ with (the boolean filter function attribute) \_key_filt.

### *class* dol.mixins.IdentityKeysWrapMixin

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Transparent KeysWrapABC. Often placed in the mro to satisfy the KeysWrapABC need in a neutral way.
This is useful in cases where the keys the persistence functions work with are the same as those you want to work
with.

### *class* dol.mixins.IdentityKvWrapMixin

Bases: [`IdentityKeysWrapMixin`](#dol.mixins.IdentityKeysWrapMixin), [`IdentityValsWrapMixin`](#dol.mixins.IdentityValsWrapMixin)

Transparent Keys and Vals Wrap

### *class* dol.mixins.IdentityValsWrapMixin

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Transparent ValsWrapABC. Often placed in the mro to satisfy the KeysWrapABC need in a neutral way.
This is useful in cases where the values can be persisted by \_\_setitem_\_ as is (or the serialization is
handled somewhere in the \_\_setitem_\_ method.

### *class* dol.mixins.IterBasedSizedContainerMixin

Bases: `IterBasedSizedMixin`, `IterBasedContainerMixin`

An ABC that defines
: 1. how to iterate over a collection of elements (keys) (_\_iter_\_)
  2. check that a key is contained in the collection (_\_contains_\_), and
  3. how to get the number of elements in the collection

This is exactly what the collections.abc.Collection (from which Keys inherits) does.
The difference here, besides the “Keys” purpose-explicit name, is that Keys offers default

> \_\_len_\_ and \_\_contains_\_  definitions based on what ever \_\_iter_\_ the concrete class defines.

Keys is a collection (i.e. a Sized (has \_\_len_\_), Iterable (has \_\_iter_\_), Container (has \_\_contains_\_).
It’s purpose is to serve as a collection of object identifiers in a key->obj mapping.
The Keys class doesn’t implement \_\_iter_\_ (so needs to be subclassed with a concrete class), but
offers mixin \_\_len_\_ and \_\_contains_\_ methods based on a given \_\_iter_\_ method.
Note that usually \_\_len_\_ and \_\_contains_\_ should be overridden to more, context-dependent, efficient methods.

### *class* dol.mixins.OverWritesNotAllowedMixin

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Mixin for only allowing a write to a key if they key doesn’t already exist.

#### NOTE
Should be before the persister in the MRO.

```pycon
>>> class TestPersister(OverWritesNotAllowedMixin, dict):
...     pass
>>> p = TestPersister()
>>> p['foo'] = 'bar'
>>> #p['foo'] = 'bar2'  # will raise error
>>> p['foo'] = 'this value should not be stored'
Traceback (most recent call last):
  ...
dol.errors.OverWritesNotAllowedError: key foo already exists and cannot be overwritten.
    If you really want to write to that key, delete it before writing
>>> p['foo']  # foo is still bar
'bar'
>>> del p['foo']
>>> p['foo'] = 'this value WILL be stored'
>>> p['foo']
'this value WILL be stored'
```

### *class* dol.mixins.ReadOnlyMixin

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Put this as your first parent class to disallow write/delete operations

### *class* dol.mixins.SimpleJsonMixin

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

simple json serialization.
Useful to store and retrieve

### *class* dol.mixins.StringKvWrap

Bases: [`IdentityKvWrapMixin`](#dol.mixins.IdentityKvWrapMixin)
