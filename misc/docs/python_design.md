# dol: Python Design and Architecture

This document describes the Python-specific implementation of dol's design. For the language-agnostic concepts, see [general_design.md](general_design.md).

---

## Class Hierarchy

```
collections.abc.Collection        (ABC: __iter__, __contains__, __len__)
    │
dol.base.Collection               (adds head(), default __len__/__contains__ via iteration)
    │
dol.base.KvReader  (aka Reader)   (adds __getitem__, keys(), values(), items(); removes __reversed__)
    │
dol.base.KvPersister (aka Persister) (adds __setitem__, __delitem__; disables clear())
    │
dol.base.Store                    (adds 4 transform hooks; wraps an inner store)
```

All classes inherit from `collections.abc` ABCs, so they satisfy `isinstance` checks and abc registration.

### Key design notes on the hierarchy

- `Collection.head()` — returns `next(iter(self.items()))` or `next(iter(self))`. Useful for quick inspection without knowing any key.
- `KvReader.__reversed__` — explicitly raises `NotImplementedError`. Rationale: not all backends have a natural order; forcing the interface to pretend otherwise would be misleading.
- `KvPersister.clear = _disabled_clear_method` — clear is disabled by default, because wiping a persistent store accidentally is catastrophic. Subclasses can re-enable it explicitly.
- `MappingViewMixin` — provides pluggable `KeysView`, `ValuesView`, `ItemsView` classes. Override the *class attribute* (e.g., `MyStore.KeysView = MyKeysView`) to customize view behavior without overriding `.keys()`.

---

## The Store Class: Transform Hooks

`Store` is the central class. It wraps an inner store object (`self.store`) and intercepts reads/writes through 4 hook methods:

```python
class Store(KvPersister):
    _id_of_key = static_identity_method   # outer key → inner key
    _key_of_id = static_identity_method   # inner key → outer key
    _data_of_obj = static_identity_method # outer value → stored data
    _obj_of_data = static_identity_method # stored data → outer value

    def __getitem__(self, k):
        _id = self._id_of_key(k)
        data = self.store[_id]
        return self._obj_of_data(data)

    def __setitem__(self, k, obj):
        _id = self._id_of_key(k)
        data = self._data_of_obj(obj)
        self.store[_id] = data

    def __iter__(self):
        yield from (self._key_of_id(_id) for _id in self.store)
```

The hooks default to identity (no-op), so `Store(dict())` behaves exactly like a dict. You inject transforms by:
1. Subclassing and overriding hook methods
2. Assigning callables directly to hook names on the instance or class
3. Using `wrap_kvs` (the recommended approach for most cases)

**The naming convention** `X_of_Y` means "get X given Y" — identical to mathematical function notation. This is explicit about directionality: `id_of_key` converts a key to an id; `key_of_id` converts an id to a key.

---

## `wrap_kvs`: The Core Transformation Function

Located in `dol/trans.py:1801`. The most important function in the library.

```python
@store_decorator
def wrap_kvs(
    store=None,
    *,
    # Key transforms
    key_of_id=None,       # outgoing: inner_id → outer_key  (for __iter__)
    id_of_key=None,       # incoming: outer_key → inner_id  (for __getitem__, __setitem__, __delitem__)
    # Value transforms
    obj_of_data=None,     # outgoing: stored_data → python_obj  (for __getitem__)
    data_of_obj=None,     # incoming: python_obj → stored_data  (for __setitem__)
    # Key-conditioned value transforms
    preset=None,          # (key, obj) → data  [on write, when value transform depends on key]
    postget=None,         # (key, data) → obj  [on read, when value transform depends on key]
    # Codec shortcuts
    key_codec=None,       # Codec with .encoder (id_of_key) and .decoder (key_of_id)
    value_codec=None,     # Codec with .encoder (data_of_obj) and .decoder (obj_of_data)
    key_encoder=None,     # alias for id_of_key
    key_decoder=None,     # alias for key_of_id
    value_encoder=None,   # alias for data_of_obj
    value_decoder=None,   # alias for obj_of_data
    # Method transforms (advanced)
    outcoming_key_methods=(),
    outcoming_value_methods=(),
    ingoing_key_methods=(),
    ingoing_value_methods=(),
    # Naming
    name=None,
    wrapper=None,         # defaults to Store
):
```

### How `wrap_kvs` works

It creates a new class (or wraps an instance) by applying the given transforms to the appropriate dunder methods. The `@store_decorator` decorator makes it work in 4 modes (see below).

### `obj_of_data` vs `postget`

| Feature | `obj_of_data` | `postget` |
|---------|--------------|-----------|
| Signature | `(data) → obj` | `(key, data) → obj` |
| Knows the key? | No | Yes |
| Use when | Same transform for all values | Transform depends on key (e.g., file extension) |

Same distinction applies to `data_of_obj` vs `preset`.

### Examples

```python
from dol import wrap_kvs
import json, pickle

# 1. Add JSON serialization
JsonStore = wrap_kvs(dict, obj_of_data=json.loads, data_of_obj=json.dumps)

# 2. Add key prefix
PrefixedStore = wrap_kvs(dict,
    id_of_key=lambda k: f"user:{k}",
    key_of_id=lambda _id: _id[len("user:"):],
)

# 3. Extension-based deserialization (key-conditioned)
MultiFormatStore = wrap_kvs(dict,
    postget=lambda k, v: json.loads(v) if k.endswith('.json') else pickle.loads(v),
    preset=lambda k, v: json.dumps(v) if k.endswith('.json') else pickle.dumps(v),
)

# 4. Using codec shortcuts
from dol.trans import ValueCodec
pickle_codec = ValueCodec(encoder=pickle.dumps, decoder=pickle.loads)
PickleStore = wrap_kvs(dict, value_codec=pickle_codec)

# 5. Stacking layers (the "Russian dolls" pattern)
store = dict()
store = wrap_kvs(store, id_of_key=lambda k: k + '.json', key_of_id=lambda _id: _id[:-5])
store = wrap_kvs(store, obj_of_data=json.loads, data_of_obj=json.dumps)
```

---

## `store_decorator`: The Meta-Decorator

Located in `dol/trans.py:130`. Enables writing a class-transforming function once and using it in 4 ways:

```python
@store_decorator
def my_deco(store=None, *, some_param='default'):
    # Transform the store class or instance
    ...
    return transformed_store
```

The 4 usage modes:

```python
# 1. Class decorator (no parens, uses defaults)
@my_deco
class MyStore(dict): ...

# 2. Class decorator factory (with params)
@my_deco(some_param='custom')
class MyStore(dict): ...

# 3. Instance decorator (wraps existing instance in a Store)
s = dict()
s_wrapped = my_deco(s)

# 4. Instance decorator factory
wrap_with_custom = my_deco(some_param='custom')
s_wrapped = wrap_with_custom(s)
```

When decorating an **instance** (modes 3 and 4), `store_decorator` automatically wraps it in `Store` first, so the decorator always receives a class.

### `double_up_as_factory`

A related utility that upgrades a plain decorator to also work as a factory:

```python
@double_up_as_factory
def my_deco(func=None, *, multiplier=2):
    def wrapper(x): return func(x) * multiplier
    return wrapper

# Direct use:      my_deco(f)
# Factory use:     my_deco(multiplier=3)(f)
# As class deco:   @my_deco(multiplier=3)
```

Constraint: first arg must default to `None`; all other args must be keyword-only. This is enforced at decoration time.

---

## Codec Abstraction

Located in `dol/trans.py:3362`. A `Codec` is a dataclass pairing an encoder and decoder:

```python
@dataclass
class Codec(Generic[DecodedType, EncodedType]):
    encoder: Callable[[DecodedType], EncodedType]
    decoder: Callable[[EncodedType], DecodedType]

    def compose_with(self, other): ...   # chain two codecs
    def invert(self): ...                 # swap encoder/decoder
    __add__ = compose_with
    __invert__ = invert
```

**Subclasses** are callable and apply the codec to a store:

```python
class ValueCodec(Codec):
    def __call__(self, obj):
        return wrap_kvs(obj, data_of_obj=self.encoder, obj_of_data=self.decoder)

class KeyCodec(Codec):
    def __call__(self, obj):
        return wrap_kvs(obj, id_of_key=self.encoder, key_of_id=self.decoder)

class KeyValueCodec(Codec):
    def __call__(self, obj):
        return wrap_kvs(obj, preset=self.encoder, postget=self.decoder)
```

Usage:

```python
from dol.trans import ValueCodec
import json

json_codec = ValueCodec(encoder=json.dumps, decoder=json.loads)
MyStore = json_codec(dict)   # wrap dict with json serialization
```

The `kv_codecs.py` module provides ready-made codec factories in two namespaces:

```python
from dol import ValueCodecs, KeyCodecs

# Codec factories
pickle_codec = ValueCodecs.pickle()   # ValueCodec(encoder=pickle.dumps, decoder=pickle.loads)
json_codec   = ValueCodecs.json()
gzip_codec   = ValueCodecs.gzip()
csv_codec    = ValueCodecs.csv()

suffix_codec = KeyCodecs.suffixed('.pkl')  # adds/strips .pkl from keys

# Compose with +
full_codec = ValueCodecs.pickle() + ValueCodecs.gzip()  # pickle then gzip

# Apply to store
MyStore = Pipe(KeyCodecs.suffixed('.pkl'), ValueCodecs.pickle())(dict)
```

---

## `Pipe`: Function Composition

Located in `dol/util.py`. Chains functions left-to-right:

```python
from dol import Pipe

f = Pipe(json.dumps, str.encode, gzip.compress)
# f(obj) == gzip.compress(str.encode(json.dumps(obj)))
```

Codecs support `+` as `Pipe` for composition:

```python
ValueCodecs.str_to_bytes() + ValueCodecs.gzip()
# = ValueCodec where encoder = gzip(str_to_bytes(x)) and decoder = str_from_bytes(gunzip(x))
```

---

## `Sig`: Signature Calculus

Located in `dol/signatures.py`. Rich signature manipulation:

```python
from dol.signatures import Sig

sig = Sig(my_func)
sig.names          # list of parameter names
sig.defaults       # dict of {name: default}
sig.annotations    # dict of {name: type}

# Arithmetic on signatures
new_sig = Sig(f) + ['extra_param'] + Sig(g)  # merge signatures
new_sig = Sig(f) - ['verbose']               # remove parameter

# Apply a signature to a function
@Sig(some_other_func)
def my_func(*args, **kwargs): ...
# my_func now has the signature of some_other_func
```

`Sig` is used internally throughout dol to:
- Compose signatures of transform functions for `wrap_kvs`
- Build the 4-way decorator signature in `store_decorator`
- Generate codec signatures in `kv_codecs.py`

---

## Delegation Pattern

`Store` uses the delegation pattern: it holds a reference to an inner store (`self.store`) and delegates all storage operations to it. Attribute access falls through via `__getattr__`:

```python
def __getattr__(self, attr):
    return getattr(object.__getattribute__(self, "store"), attr)
```

The `DelegatedAttribute` descriptor makes delegation explicit and works with pickling:

```python
class DelegatedAttribute:
    def __get__(self, instance, owner):
        return getattr(getattr(instance, self.delegate_name), self.attr_name)
    def __set__(self, instance, value):
        setattr(getattr(instance, self.delegate_name), self.attr_name, value)
```

`delegator_wrap(delegator, obj)` creates a class/instance that delegates to `obj` via `delegator`. Used by `Store.wrap = classmethod(partial(delegator_wrap, delegation_attr='store'))`.

---

## Caching Patterns (`caching.py`)

### `cache_this` — property/method caching

```python
from dol import cache_this

class MyClass:
    @cache_this
    def expensive_property(self):   # no args → cached_property behavior
        return compute_expensive()

    @cache_this(cache={})           # explicit cache dict
    def expensive_method(self, x, y):
        return compute(x, y)

    @cache_this(cache='my_cache', ignore={'verbose'})
    def parameterized(self, data, mode='fast', verbose=False):
        ...
```

The cache can be any Mapping — including a dol store, enabling persistent or distributed caches.

### `store_cached` — function memoization

```python
from dol import store_cached
import shelve

@store_cached(shelve.open('my_cache'))  # persisted cache
def slow_computation(x, y):
    return ...
```

### `cache_vals` — store-level caching

```python
from dol import cache_vals

# Add an in-memory cache layer in front of a slow store
FastStore = cache_vals(SlowStore, cache=dict)
```

### `WriteBackChainMap`

A ChainMap where writes go to the first (fast) store and reads fall through in order. Useful for layered cache hierarchies.

---

## Composition Stores (`sources.py`)

### `FlatReader` — flatten a store of stores

```python
from dol.sources import FlatReader

outer = {'A': {'x': 1, 'y': 2}, 'B': {'z': 3}}
flat = FlatReader(outer, key_func=lambda outer_k, inner_k: f"{outer_k}/{inner_k}")
list(flat)  # ['A/x', 'A/y', 'B/z']
```

### `FanoutReader` / `FanoutPersister`

Reads/writes broadcast to multiple stores simultaneously.

```python
from dol.sources import FanoutPersister

s = FanoutPersister(local_store, remote_store)
s['key'] = value   # writes to both stores
s['key']           # reads from first store that has the key
```

### `CascadedStores`

Writes go to all stores; reads come from the first store that has the key.

---

## Path Navigation (`paths.py`)

For hierarchical/nested stores:

```python
from dol import path_get, path_set, KeyPath, mk_relative_path_store

d = {'a': {'b': {'c': 42}}}
path_get(d, ('a', 'b', 'c'))   # 42
path_set(d, ('a', 'b', 'd'), 99)

# Convert a path store (full paths as keys) to a relative path store
RelativeStore = mk_relative_path_store(root='/data/users')
s = RelativeStore()
s['john/profile.json']  # reads /data/users/john/profile.json

# KeyTemplate for structured key parsing
from dol.paths import KeyTemplate
kt = KeyTemplate('{user}/{year}/{month}.json')
kt.key_to_dict('john/2024/01.json')  # {'user': 'john', 'year': '2024', 'month': '01'}
kt.dict_to_key({'user': 'john', 'year': '2024', 'month': '01'})  # 'john/2024/01.json'
```

---

## Design Critique and Alternatives

### 1. ABC Inheritance vs. Protocols

**Current approach**: Classes inherit from `collections.abc.Mapping`, `MutableMapping`, etc.

**Pros**:
- `isinstance()` checks work
- Free implementations of derived methods (`get`, `update`, `__eq__`, etc.)
- ABCs document the contract clearly

**Cons**:
- Structural subtyping not supported — you must inherit, not just implement the interface
- Python 3.8+ `typing.Protocol` (structural typing) would allow any class with the right methods to be used without inheritance
- Multiple inheritance from several ABCs creates MRO complexity

**Alternative**: Use `Protocol` for type hints while keeping the ABC base classes for runtime behavior. This is additive (not breaking) and would improve type-checker experience.

### 2. Disabling `clear()` via Assignment

**Current approach**: `KvPersister.clear = _disabled_clear_method`

**Pros**: Clear signal that "this is dangerous"; forces explicit re-enabling

**Cons**:
- Surprising to anyone who calls `dict(store)` — it calls `.update()` and `.clear()`, which would fail
- Violates Liskov Substitution Principle (LSP) — KvPersister claims to be a MutableMapping but breaks one of its methods
- `isinstance(store, MutableMapping)` is True but `.clear()` raises

**Alternative**: Return without error but log a warning, or document why this decision was made in the class docstring (it is, but could be more prominent).

### 3. The `_id_of_key` Naming Convention

**Current approach**: `_id_of_key`, `_key_of_id`, `_data_of_obj`, `_obj_of_data` (from math: `Y_of_X` means `f: X → Y`)

**Pros**: Explicit directionality; clear what goes in and comes out

**Cons**:
- Unfamiliar to most Python developers (unusual naming style)
- `wrap_kvs` uses the *opposite* naming: `key_of_id` (outgoing) vs `id_of_key` (incoming), which is correct but requires mental mapping
- The `_` prefix makes them look like private/internal, but they're the main customization points

**Alternative**: `encode_key`/`decode_key`, `serialize_value`/`deserialize_value` — more conventional names. Or `key_to_id`/`id_to_key` (Python-style, verb-noun).

### 4. `store_decorator`'s 4-Way Usage

**Current approach**: One decorator factory that produces a decorator usable as: class-decorator, class-decorator-factory, instance-decorator, instance-decorator-factory.

**Pros**: Maximum flexibility; no code duplication; same API works in all contexts

**Cons**:
- The 4-way behavior is non-obvious from the signature alone
- Error messages when misused can be cryptic
- Testing all 4 modes for each decorator adds overhead

**Alternative**: Keep the 4-way usage but add type hints that make it clear in IDEs. Or provide separate `as_class_deco` and `as_instance_deco` wrappers.

### 5. Missing: Async Support

**Current approach**: Entirely synchronous.

**Cons**: Cannot be used with `async` backends (asyncio, aiohttp, aiobotocore) without blocking the event loop.

**Alternative**: An `AsyncKvReader` / `AsyncKvPersister` hierarchy with `async def __aiter__`, `async def __agetitem__`, etc. This would be additive and the sync hierarchy could stay as-is.

### 6. No Generic Type Parameters on Classes

**Current approach**: `KvReader` has no type parameters.

**Cons**: Type checkers cannot infer key/value types.

**Alternative**: `KvReader[KT, VT]`, `KvPersister[KT, VT]`, `Store[KT, VT]` — would improve IDE autocompletion and static analysis. Could be done without breaking changes using `Generic[KT, VT]`.

### 7. `wrap_kvs` vs Direct Subclassing

`wrap_kvs` is powerful but creates anonymous classes at runtime, which has implications:
- `type(store).__name__` may not be meaningful
- Pickling can be tricky (though dol handles this via `__reduce__`)
- Debugging stack traces show generic names

For performance-critical code or when pickling is needed, direct subclassing is still more reliable.

---

## Key Idioms Summary

| Idiom | Where | What it does |
|-------|-------|-------------|
| `Y_of_X` naming | `base.py`, `trans.py` | Explicit directionality for transform functions |
| `@store_decorator` | `trans.py` | 4-way usage for class/instance decorators |
| `@double_up_as_factory` | `trans.py` | Decorator works both directly and as factory |
| `Sig` arithmetic | `signatures.py` | Merge/subtract/compose function signatures |
| `Codec` + `__add__` | `trans.py` | Chain encode/decode pairs with `+` operator |
| `Pipe` | `util.py` | Left-to-right function composition |
| `cache_this` | `caching.py` | Pluggable property/method caching |
| `static_identity_method` | `util.py` | No-op hook default (works as static or instance method) |
| `wrap = classmethod(delegator_wrap)` | `base.py` | Class carries its own wrapping method |
