---
name: dol-store-building
description: "Build a dol store: wrap any storage backend (files, S3, DB, dict, an API) behind a uniform dict-like (MutableMapping) interface, adding key and value transforms/serialization. Use when a user wants to give a backend a dict interface, add JSON/pickle/gzip (or custom) serialization to a store, transform or filter keys, compose codecs, cache a slow store, or asks 'how do I use dol to ...'. Covers wrap_kvs (the core), the ValueCodecs/KeyCodecs namespaces, Pipe composition, the ready-made file stores (Files/TextFiles/JsonFiles/PickleFiles), filt_iter, the test-with-dict-then-swap-backend workflow, and self-aware transforms via FirstArgIsMapping. For authoring interactive scaffolds see the /new-store, /add-codec, /explain-store commands; for modifying dol's internals see dol-dev-wrap-kvs."
---

# Building a dol store

dol turns any storage backend into a `dict`-like object: `s[key]` reads, `s[key] = val`
writes, `del s[key]` deletes, `for k in s` / `len(s)` / `k in s` explore. You write your
logic against this uniform interface and swap backends freely. The core move is **wrapping
a backend with key/value transforms**.

## The golden workflow: prototype with `dict`, then swap the backend

Always build and test with a plain `dict` first, then swap in the real backend — the
transform code is identical:
```python
from dol import wrap_kvs
import json

# 1. logic first, dict backend
S = wrap_kvs(dict, obj_of_data=json.loads, data_of_obj=json.dumps)
s = S()
s["x"] = {"a": 1}
assert s["x"] == {"a": 1}

# 2. same transforms, real backend
from dol import Files

s = wrap_kvs(Files("/data"), obj_of_data=json.loads, data_of_obj=json.dumps)
```

## `wrap_kvs` — the core, and its `X_of_Y` naming

`X_of_Y(y) -> x`. Outgoing (read) transforms produce what the user sees; ingoing (write)
transforms produce what the backend stores. They come in inverse pairs:

| kwarg | direction | signature | use |
|---|---|---|---|
| `key_of_id` | read (key out) | `k = key_of_id(_id)` | backend id → user key |
| `id_of_key` | write (key in) | `_id = id_of_key(k)` | user key → backend id |
| `obj_of_data` | read (value out) | `obj = obj_of_data(data)` | deserialize |
| `data_of_obj` | write (value in) | `data = data_of_obj(obj)` | serialize |
| `postget` | read, key-aware | `obj = postget(k, data)` | deserialize depending on key (e.g. by extension) |
| `preset` | write, key-aware | `data = preset(k, obj)` | serialize depending on key |

Use `obj_of_data`/`data_of_obj` when the transform is the same for all values; use
`postget`/`preset` when it depends on the key.

## Prefer ready-made codecs over hand-rolled lambdas

```python
from dol import ValueCodecs, KeyCodecs, Pipe

ValueCodecs.json()  # json.dumps / json.loads
ValueCodecs.pickle()  # pickle
ValueCodecs.gzip()  # compress/decompress
KeyCodecs.suffixed(".json")  # add/strip a key suffix
KeyCodecs.prefixed("ns:")  # add/strip a key prefix

# Compose with + or Pipe (order = application order on the backend side)
MyStore = Pipe(KeyCodecs.suffixed(".pkl"), ValueCodecs.pickle() + ValueCodecs.gzip())(
    dict
)
```

## Ready-made file stores (skip wrap_kvs when one fits)

```python
from dol import Files, TextFiles, JsonFiles, PickleFiles

Files("/data")  # keys=relative paths, values=bytes
TextFiles("/data")  # values=str
JsonFiles("/data")  # values=json-decoded objects
```

## Filtering the key space

```python
from dol import filt_iter

s = filt_iter(store, filt=lambda k: k.endswith(".json"))
# ready-made variants:
filt_iter.suffixes(".json")
filt_iter.prefixes("user/")
filt_iter.regex(r"\d{4}")
```

## Caching a slow store

```python
from dol import cache_vals, cache_this

fast = cache_vals(slow_store)  # in-memory read cache


class C:
    @cache_this(cache="_c")  # cache an expensive property/method
    def expensive(self): ...
```

## Transforms that need the store itself: `FirstArgIsMapping`

Most transforms are pure `f(value)`. When a transform genuinely needs the store instance
(its config, root path, etc.), mark it so dol passes the store as the first arg — instead
of relying on parameter names:
```python
from dol import wrap_kvs, FirstArgIsMapping


def resolve(self, data):  # first arg is the store
    return f"{self.root}/{data}"


s = wrap_kvs(store, obj_of_data=FirstArgIsMapping(resolve))
```
Note: passing a bare `def f(self, data)` (first param named `self`/`store`/`mapping`, ≥2
required params) also works via dol's heuristic — but `FirstArgIsMapping` is explicit and
robust (a plain unary builtin like `bytes.decode` is correctly treated as `f(data)`).

## Read-only vs read-write

Subclass `KvReader` for read-only stores, `KvPersister` for read-write, or just use
`wrap_kvs` on a backend that is/ isn't writable. Note `clear()` is disabled on persisters
by default (guard against wiping a backend); re-enable deliberately if you must.

## Gotchas

- **`bytes.decode` as `obj_of_data`** now works, but prefer `ValueCodecs.str_to_bytes()`
  or `lambda b: b.decode()` for clarity.
- **`clear()` is disabled** on `KvPersister` — deliberate.
- **A wrapped class's own methods see the *unwrapped* store as `self`** — if you write a
  method that does `self[k]` inside a `@wrap_kvs`-decorated class, re-wrap `self` (advanced;
  see the dol-dev-wrap-kvs skill / Issue #18).
- **Compose, don't subclass**, when adding transforms: `wrap_kvs`/codecs/`Pipe` compose
  cleanly; subclassing `Store` is only for custom hook protocols.

## When to reach for the commands

- `/new-store <desc>` — scaffold a new store class/factory interactively.
- `/add-codec <desc>` — add or create a codec.
- `/explain-store <expr>` — trace an existing store's transform pipeline.
