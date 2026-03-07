# dol: General Design and Architecture

## What dol Is

`dol` is a toolkit for building **Data Object Layers** — uniform, dict-like interfaces to any storage backend. The core idea: separate *what data operations your domain needs* from *how those operations are implemented in a specific backend*.

This places dol in the family of:
- **Data Access Object (DAO)** — objects that abstract storage operations
- **Repository Pattern** — domain-facing interface to a collection of entities
- **Hexagonal Architecture (Ports & Adapters)** — the "port" is the KV interface; adapters are backend implementations

But dol has a distinctive orientation: it is **middleware**, not domain logic and not backend infrastructure. It provides a common language — the key-value (KV) interface — that both domain code and backend adapters can speak.

---

## The Key Insight: Language-Native Interfaces

Most languages have a first-class "mapping" concept (Python's `dict`, JavaScript's `Map`, Java's `Map<K,V>`). Rather than inventing a new CRUD API, dol maps storage operations onto this native interface:

| Storage operation | KV interface |
|---|---|
| Read item by key | `store[key]` (`__getitem__`) |
| Write item at key | `store[key] = value` (`__setitem__`) |
| Delete item | `del store[key]` (`__delitem__`) |
| List all keys | `for k in store` (`__iter__`) |
| Count items | `len(store)` (`__len__`) |
| Check existence | `key in store` (`__contains__`) |

Code using a `dol` store looks exactly like code using a dict. This means:
- No new API to learn — use the language's built-in mapping idioms
- Tests can use `dict` as a drop-in backend
- Tools that work on dicts (comprehensions, `update`, `copy`, etc.) work on stores

---

## Interface Hierarchy: From Full to Minimal

A full `MutableMapping` (Python's dict-like ABC) includes 14+ methods. dol defines a reduced, pragmatic hierarchy:

```
Collection        ← __iter__, __contains__, __len__, head()
    │
KvReader          ← + __getitem__, keys(), values(), items()    [read-only]
    │
KvPersister       ← + __setitem__, __delitem__                  [read-write]
    │
Store             ← + key/value transform hooks                 [configurable]
```

Key reductions from `MutableMapping`:
- **No `.clear()`** — too destructive for persistent storage; disabled by default
- **No guaranteed order** — backends vary; `__reversed__` raises `NotImplementedError`
- **`__len__` and `__contains__` via iteration** — correct by default; override for efficiency

This hierarchy reflects what storage backends actually provide: list, get, set, delete — not necessarily atomic batch-clear or ordered traversal.

---

## The Middleware Principle

dol occupies the space *between* domain logic and storage backends:

```
Domain Code          dol Layer              Storage Backend
(business logic)     (KV interface)         (files, DB, S3...)
                  ┌──────────────────────┐
  store['user/42'] │  key transform       │  /data/users/00042.json
  store[key]       │──────────────────────│  db.query("SELECT...")
  value = json obj │  value transform     │  bytes on disk
                  └──────────────────────┘
```

The transforms are the core contribution: they let you define the interface your domain wants (clean keys, rich objects) while the backend stores what it needs to store (raw paths, serialized bytes).

---

## The KV Transform Pipeline

Every read and write passes through a pair of transformations:

```
READ:
  key → [id_of_key] → internal_id → backend[id] → raw_data → [obj_of_data] → value

WRITE:
  key → [id_of_key] → internal_id
  value → [data_of_obj] → raw_data → backend[id] = raw_data

ITERATE:
  backend.__iter__() → internal_id → [key_of_id] → key
```

The four transform functions (`id_of_key`, `key_of_id`, `obj_of_data`, `data_of_obj`) default to identity. You only implement what you need.

**Example**: A store of JSON files where keys are relative paths without `.json` extension:

```
key:  "user/42"
  ↓ id_of_key: k → k + ".json"
id:   "user/42.json"
  ↓ backend read
data: b'{"name": "Alice", "age": 30}'
  ↓ obj_of_data: json.loads
obj:  {"name": "Alice", "age": 30}
```

### The `preset`/`postget` Extension

Sometimes the value transform needs to know the key — for example, to choose the right serializer based on file extension. Two additional transforms handle this:

- `preset(key, value) → raw_data` — applied on write, key-aware
- `postget(key, raw_data) → value` — applied on read, key-aware

```python
def postget(k, v):
    if k.endswith('.json'): return json.loads(v)
    if k.endswith('.pkl'):  return pickle.loads(v)
    return v
```

---

## Layered Composition (Russian Dolls)

The name "dol" evokes Russian dolls: layers of wrappers, each adding a transformation. A store is built by stacking layers:

```
  raw_backend      (dict, files, S3, DB...)
       │
  key_transform    (strip prefix, add extension)
       │
  value_transform  (serialize/deserialize)
       │
  filter_layer     (hide internal keys)
       │
  cache_layer      (in-memory cache)
       │
  domain_store     (what domain code sees)
```

Each layer is independent and composable. You can add, remove, or swap layers without touching domain code or the backend.

The primary tool for adding layers is `wrap_kvs` (see [python_design.md](python_design.md) for details):

```python
from dol import wrap_kvs

# Add json serialization to any store
JsonStore = wrap_kvs(dict,
    obj_of_data=json.loads,
    data_of_obj=json.dumps,
)

# Add prefix to all keys
PrefixedStore = wrap_kvs(dict,
    id_of_key=lambda k: f"prefix/{k}",
    key_of_id=lambda id: id[len("prefix/"):],
)

# Stack both layers
store = PrefixedStore()
store = wrap_kvs(store, obj_of_data=json.loads, data_of_obj=json.dumps)
```

---

## Caching as a First-Class Concern

dol treats caching not as an optimization afterthought but as a composable layer:

- **Key caching** — cache iteration results (`cached_keys`)
- **Value caching** — cache fetched values (`cache_vals`, `WriteBackChainMap`)
- **Method caching** — cache expensive property/method results (`cache_this`, `store_cached`)
- **Write-back caching** — reads from fast cache, writes through to slow backend

The cache backend is itself a store — enabling persistent caches, distributed caches, or custom eviction strategies.

---

## Store Composition Patterns

Beyond single-store wrapping, dol supports multi-store composition:

| Pattern | Class | Behavior |
|---------|-------|----------|
| Union view | `FlatReader` | Merge multiple stores into one flat view |
| Fan-out reads | `FanoutReader` | Reads return dict of results from all stores |
| Fan-out writes | `FanoutPersister` | Writes go to all stores simultaneously |
| Cascaded | `CascadedStores` | Writes to all, reads from first available |

---

## Design Benefits

1. **Testability**: Develop with `dict` as backend; swap to real storage when ready
2. **Portability**: Same domain code works with S3, files, MongoDB, SQLite — just change the bottom layer
3. **Incrementalism**: Start with raw backend access; add transform layers progressively
4. **Separation of concerns**: Key format, serialization, filtering, caching are separate layers
5. **Composability**: Layers can be mixed and matched freely
6. **Discoverability**: One interface (`Mapping`) — IDEs and agents know what to expect

---

## What dol Is Not

- **Not an ORM**: dol doesn't map objects to relational tables. It maps Python values to opaque storage cells.
- **Not a query engine**: dol doesn't support complex queries (filter by field, join, aggregate). That's the backend's job. dol provides list+get+set+delete only.
- **Not domain-driven**: dol stores are intentionally domain-agnostic. The domain meaning lives in the code that uses the store, not in the store itself.
- **Not a schema validator**: dol doesn't enforce data schemas. Validation is a layer you add (e.g., via `data_of_obj`).

---

## Relation to Existing Patterns

| Pattern | Similarity | Difference |
|---------|-----------|------------|
| Repository Pattern | Abstracts storage; testable | dol is not domain-driven; uses KV not domain methods |
| DAO | Wraps storage operations | dol focuses on KV interface specifically |
| Active Record | Object knows how to store itself | dol is separate from domain objects |
| Hexagonal | Ports & adapters | dol is specifically the "storage port" |
| Decorator Pattern | Wraps objects adding behavior | dol uses this structurally for transform layers |
| Adapter Pattern | Converts one interface to another | Each dol wrapper is an adapter |
