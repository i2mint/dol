# dol — AI Agent Guide

`dol` is a pure-Python (no dependencies) toolkit for wrapping any storage backend (files, S3, databases, dicts) behind a uniform dict-like interface. Version 0.3.38. Python ≥ 3.10.

For a comprehensive agent-readable API reference, see [llms-full.txt](llms-full.txt).
For a quick orientation, see [llms.txt](llms.txt).

---

## Key Files

| File | What's in it |
|------|-------------|
| `dol/base.py` | `Collection`, `KvReader`, `KvPersister`, `Store` — the class hierarchy |
| `dol/trans.py` | `wrap_kvs` (core), `store_decorator`, `filt_iter`, `cached_keys`, `Codec`, `kv_wrap` |
| `dol/kv_codecs.py` | `ValueCodecs`, `KeyCodecs` — ready-made codec namespaces |
| `dol/caching.py` | `cache_this`, `cache_vals`, `store_cached`, `WriteBackChainMap` |
| `dol/paths.py` | `KeyTemplate`, `mk_relative_path_store`, `KeyPath`, `path_get/set/filter` |
| `dol/filesys.py` | `Files`, `TextFiles`, `JsonFiles`, `PickleFiles` — filesystem stores |
| `dol/sources.py` | `FlatReader`, `FanoutReader/Persister`, `CascadedStores` |
| `dol/signatures.py` | `Sig` — signature arithmetic |
| `dol/util.py` | `Pipe`, `lazyprop`, `partialclass`, `groupby` |
| `dol/__init__.py` | Public API — all exports live here |

---

## Core Pattern: Building Stores

The fundamental operation is **wrapping a backend with transforms**:

```python
from dol import wrap_kvs, Files
import json

# Add JSON serialization to a file store
JsonFileStore = wrap_kvs(Files, obj_of_data=json.loads, data_of_obj=json.dumps)

# Or wrap an instance
s = wrap_kvs(dict(), id_of_key=lambda k: k.upper(), key_of_id=str.lower)
```

`wrap_kvs` parameters:
- `key_of_id` / `id_of_key` — outgoing/incoming key transforms
- `obj_of_data` / `data_of_obj` — outgoing/incoming value transforms
- `postget(key, data) → obj` — value transform that knows the key (for reads)
- `preset(key, obj) → data` — value transform that knows the key (for writes)
- `key_codec` / `value_codec` — `Codec` objects (encoder+decoder pair)

---

## Core Conventions

- **`X_of_Y` naming**: `key_of_id` = "give me a key, you give me an id" (outgoing). `id_of_key` = "give me an id, you give me a key" (incoming). Always pairs.
- **KvReader for read-only**: subclass `KvReader` (not `KvPersister`) when writes aren't needed.
- **KvPersister for read-write**: `clear()` is disabled — override only if you're sure.
- **Test with `dict`, deploy with real backend**: `wrap_kvs(dict, ...)` first, then swap `dict` for `Files`, a DB store, etc.
- **Transforms are pure functions**: they should be stateless and not have side effects.

---

## How to Create a New Store

### Option 1: `wrap_kvs` (preferred for most cases)

```python
from dol import wrap_kvs

MyStore = wrap_kvs(dict,
    id_of_key=lambda k: k + '.json',
    key_of_id=lambda _id: _id[:-5],
    obj_of_data=json.loads,
    data_of_obj=json.dumps,
)
```

### Option 2: Subclass `KvReader`/`KvPersister`

```python
from dol.base import KvReader

class MyReader(KvReader):
    def __getitem__(self, k): ...
    def __iter__(self): ...
    def __len__(self): ...  # optional, falls back to iteration count
```

### Option 3: Subclass `Store` (when you need transform hooks)

```python
from dol.base import Store

class MyStore(Store):
    def _id_of_key(self, k): return k.upper()
    def _key_of_id(self, _id): return _id.lower()
    def _data_of_obj(self, obj): return json.dumps(obj)
    def _obj_of_data(self, data): return json.loads(data)
```

---

## Ready-Made Codecs

```python
from dol import ValueCodecs, KeyCodecs, Pipe

# Common value codecs
ValueCodecs.pickle()       # pickle.dumps / pickle.loads
ValueCodecs.json()         # json.dumps / json.loads
ValueCodecs.gzip()         # compress/decompress
ValueCodecs.str_to_bytes() # encode/decode

# Key codecs
KeyCodecs.suffixed('.pkl')  # add/strip suffix
KeyCodecs.prefixed('ns:')   # add/strip prefix

# Chain with Pipe
MyStore = Pipe(KeyCodecs.suffixed('.pkl'), ValueCodecs.pickle())(dict)
```

---

## Store Decorators

Most tools in `trans.py` use `@store_decorator`, making them work 4 ways:

```python
from dol import filt_iter, cached_keys

# As class decorator
@filt_iter(filt=lambda k: k.endswith('.json'))
class MyStore(dict): ...

# As instance wrapper
s = filt_iter(my_store, filt=lambda k: k.endswith('.json'))

# As factory
json_only = filt_iter(filt=lambda k: k.endswith('.json'))
s = json_only(my_store)
```

---

## Caching

```python
from dol import cache_this, cache_vals, store_cached

# Cache a property or method
class MyClass:
    @cache_this
    def expensive(self): return sum(range(1_000_000))

# Cache fetched values from a slow store
fast = cache_vals(slow_store)

# Persist function results across sessions
@store_cached(JsonFiles('/cache'))
def compute(x, y): return slow_computation(x, y)
```

---

## Testing Approach

Always prototype with `dict` as the backend:

```python
# 1. Test logic with dict
s = wrap_kvs(dict(), obj_of_data=json.loads, data_of_obj=json.dumps)
s['key'] = {'a': 1}
assert s['key'] == {'a': 1}

# 2. Swap to real backend
from dol import Files
s = wrap_kvs(Files('/data'), obj_of_data=json.loads, data_of_obj=json.dumps)
```

Run tests: `pytest dol/tests/`

---

## Documentation Index (`misc/docs/`)

| Document | Contents |
|----------|----------|
| [general_design.md](misc/docs/general_design.md) | Language-agnostic design: what dol is, the KV pipeline, layered composition, patterns |
| [python_design.md](misc/docs/python_design.md) | Python architecture: class hierarchy, `wrap_kvs` deep dive, `Codec`/`Sig`/`Pipe`, critique |
| [issues_and_discussions.md](misc/docs/issues_and_discussions.md) | GitHub issues/discussions themes, known limitations, open design questions |
| [frontend_dol_ideas.md](misc/docs/frontend_dol_ideas.md) | `zoddal` design: TypeScript KV interface, adapters, Zod bridge, zod-collection-ui integration |

---

## Known Limitations / Gotchas

- **`wrap_kvs` + `self` inside methods**: When a `wrap_kvs`-decorated class uses `self[k]` in its own methods, `self` is the unwrapped instance. Re-apply the wrapper to `self` if transforms are needed (Issue #18).
- **`clear()` is disabled** on `KvPersister`. Call `ensure_clear_to_kv_store(store)` to re-enable.
- **No async support** in core. Use synchronous wrappers for async backends (thread pool, etc.).
- **`bytes.decode` as `obj_of_data`** causes issues — use `lambda b: b.decode()` instead (Issue #9).
- **Windows paths**: Some path-related code has Unix assumptions. Issues #52, #58 track this.
