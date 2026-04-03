# Content-Metadata Bifurcation in dol

_A report on the "split-store problem" as discussed in i2mint/dol and i2mint/i2
issues and discussions._

---

## Introduction

A recurring design challenge in `dol` — and in data-access layers generally — is
what we call **content-metadata bifurcation**: the need to store a primary payload
(content, blob, body) alongside auxiliary descriptive data (metadata, attributes,
annotations) when these two aspects live in different backends, have different
schemas, or evolve at different rates.

In the `dol` ecosystem this surfaces under several names:

| dol / i2 terminology | Standard terminology |
|---|---|
| "key-linked mappings" | Correlated key-value stores; sidecar metadata |
| "Composite Stores" / "Store Meshes" | Composite pattern (GoF); federated store |
| "Fanout" / "StoreFanout" | Write fan-out; broadcast write |
| "Mall" | Named store registry; store catalog |
| "Domain-oriented store" | Repository pattern (DDD); aggregate root persistence |
| `.meta` attribute | Metadata sidecar; attribute store |

The rest of this document catalogues every relevant idea found in the `dol` and
`i2` issue trackers, maps them to standard design-pattern vocabulary, and
hyperlinks to the primary sources.

---

## 1. The Problem Statement

### 1.1 The canonical example

[dol Discussion #35 ("Present and Future of dol")](https://github.com/i2mint/dol/discussions/35)
gives the clearest formulation. A domain object — say a waveform — has both raw
bytes and descriptive info:

```python
# Current (infrastructure-oriented) — the user manually splits:
raw_data_store = S3Store(...)
metadata_store = MongoStore(...)
raw_data_store[key] = wf[bytes]
metadata_store[key] = wf[info]

# Desired (domain-oriented) — the store handles the split:
wf_store = WfStore(...)
wf_store[key] = wf  # dispatches bytes → S3, info → Mongo internally
```

The same discussion illustrates the pattern with a `User` object whose `photo`
(binary blob) and profile fields (structured data) naturally belong in different
storage tiers:

```python
# NOT infrastructure-isolated:
user_store.add_to_db(user_id, user)
photo_store.upload_to_cloud(user.photo)

# Infrastructure-isolated:
user_store[user_id] = user  # store dispatches to DB + cloud
```

In standard design-pattern terms this is the **Repository pattern** from
Domain-Driven Design: a single aggregate-root store that hides the physical
distribution of data across heterogeneous backends.

### 1.2 The generalization: key-linked mappings

[py2store Issue #58 ("Metadata support")](https://github.com/i2mint/py2store/issues/58)
frames the problem more generally:

> The issue of metadata comes up again quite often (or more generally
> **key-linked mappings** — i.e. two or more mappings that are linked in some
> way).

This "key-linked mappings" concept is strictly more general than
content-metadata. Content-metadata assumes one primary store and one auxiliary
store; key-linked mappings allows N co-indexed stores of equal standing
(e.g., waveform + spectrogram + annotations, all keyed by segment ID).

In database terminology this is a **correlated join** across tables sharing a
primary key. In distributed-systems terminology it is a **co-partitioned topic**
(Kafka) or **co-located tables** (CockroachDB).

---

## 2. Approaches Explored

### 2.1 Value-level embedding (rejected)

The first approach attempted, described in
[py2store #58](https://github.com/i2mint/py2store/issues/58), was to embed
metadata directly on the value object:

> I solved the problem by allowing values to have attributes. In
> [`StringWhereYouCanAddAttrs`](https://github.com/otosense/plunk/blob/dc720f5bf19cd5312c1ceaf8277c348a5630a0ab/plunk/tw/potentially_useful_utils.py#L6)
> I extended `str` so that I could add the meta-datas I wanted to directly on
> the values (which were strings).
>
> Doesn't seem like a good way to solve the problem.

This is an instance of the **Decorator pattern** applied at the value level —
enriching a built-in type with extra attributes. It was rejected because:

- It couples metadata to value type (not all types are extensible).
- It breaks serialization round-tripping (the extra attributes are lost when the
  value is stored/loaded).
- It violates separation of concerns: the value should not know about its
  metadata.

### 2.2 The `.meta` sidecar attribute (proposed)

Also from [py2store #58](https://github.com/i2mint/py2store/issues/58):

> A better direction would be to add a `.meta` `MutableMapping` attribute to a
> store, with some mechanism to keep it linked to the store (perhaps through a
> descriptor?). In order to keep things sane, we'd have to make sure that any
> key transformations that are layered on the store end up also being applied to
> the keys of `.meta`.

The key design requirements identified:

1. **Key-transform propagation**: When `wrap_kvs` adds key transforms to the
   primary store, those same transforms must automatically apply to `.meta`'s
   keys. Otherwise `store['foo']` and `store.meta['foo']` might refer to
   different backend keys.

2. **Key-space constraint** (optional): `.meta` could be constrained to only
   have keys that exist in the primary store, with a default (e.g., empty dict)
   for keys without explicit metadata.

3. **Descriptor-based linkage**: The `.meta` attribute could be implemented as a
   Python descriptor so that accessing `store.meta` dynamically constructs a
   view linked to the current store state.

In standard terms this is a **sidecar pattern** — a secondary store that
shadows the primary store's key space. It is analogous to:

- File-system extended attributes (`xattr`)
- S3 object metadata
- Database "shadow tables" for audit/metadata
- Kubernetes sidecar containers

This approach has **not been implemented** as of 2026-04.

### 2.3 Composite Stores / Fan-out (implemented)

[dol Discussion #25 ("Composite Stores")](https://github.com/i2mint/dol/discussions/25)
is the central design discussion. It frames the problem using the **Composite
pattern** (GoF):

> What I, personally mean by it, is "A store that is made of other stores."
> This fits perfectly with the definition of the Composite Pattern [...]:
> *The composite pattern describes a group of objects that are treated the same
> way as a single instance of the same type of object.*

The discussion identifies several sub-patterns:

| Sub-pattern | Standard name | dol implementation |
|---|---|---|
| Store container | Registry / Catalog | `Mall` (py2store) |
| Fan-out write | Broadcast / Scatter | [`FanoutPersister`](https://github.com/i2mint/dol/blob/6f962d25b952ac2542efbc491c902d4290198ad8/dol/sources.py#L294) |
| Fan-out read | Gather / Collect | [`FanoutReader`](https://github.com/i2mint/dol/blob/6f962d25b952ac2542efbc491c902d4290198ad8/dol/sources.py#L122) |
| Layered / chain | Chain of Responsibility | `collections.ChainMap`, [`WriteBackChainMap`](https://github.com/i2mint/dol/blob/6f962d25b952ac2542efbc491c902d4290198ad8/dol/caching.py#L2528) |
| Cascaded cache | Cache-aside with write-through | [`CascadedStores`](https://github.com/i2mint/dol/blob/6f962d25b952ac2542efbc491c902d4290198ad8/dol/sources.py#L452) |
| Flat view | Denormalization / Flattening | [`FlatReader`](https://github.com/i2mint/dol/blob/6f962d25b952ac2542efbc491c902d4290198ad8/dol/sources.py#L79) |
| Multi-source fallback | Failover / Source chain | [`MultiSource`](https://github.com/i2mint/dol/blob/6f962d25b952ac2542efbc491c902d4290198ad8/dol/sources.py#L545) |

#### 2.3.1 FanoutReader

[`FanoutReader`](https://github.com/i2mint/dol/blob/6f962d25b952ac2542efbc491c902d4290198ad8/dol/sources.py#L122)
takes a `{name: store}` mapping. Reading a key returns a dict of values from
all sub-stores:

```python
stores = dict(bytes_store=bytes_store, metadata_store=metadata_store)
reader = FanoutReader(stores)
reader['b']
# → {'bytes_store': b'b', 'metadata_store': {'x': 2}}
```

Keys are the union of all sub-stores' key sets (via `ChainMap`). It supports
`default` values for stores that lack a given key and a
`get_existing_values_only` mode that silently skips missing stores.

This is the **read side** of the content-metadata bifurcation solution: when you
read, you get a composite value aggregated from all backends.

#### 2.3.2 FanoutPersister

[`FanoutPersister`](https://github.com/i2mint/dol/blob/6f962d25b952ac2542efbc491c902d4290198ad8/dol/sources.py#L294)
extends `FanoutReader` with writes:

```python
persister['a'] = dict(bytes_store=b'a', metadata_store=dict(x=1))
```

The value dict is **destructured** and each sub-value is routed to the
corresponding store. Configuration flags control behavior:

- `need_to_set_all_stores`: whether a write must provide values for every
  sub-store (strict mode) or can do partial writes.
- `ignore_non_existing_store_keys`: whether extra keys in the value dict
  (not matching any sub-store name) should raise or be silently dropped.

This is the **write side** of the split-store solution: a single `__setitem__`
dispatches to multiple backends.

#### 2.3.3 CascadedStores

[`CascadedStores`](https://github.com/i2mint/dol/blob/6f962d25b952ac2542efbc491c902d4290198ad8/dol/sources.py#L452)
implements a cascading cache pattern:

- **Writes** propagate to all stores (write-through).
- **Reads** try stores in order; the first hit is returned and back-filled to
  all preceding (faster) stores.

This is a classic **cache hierarchy** (L1 → L2 → L3) applied to the store
abstraction.

### 2.4 The write-interface question (open)

A key comment in
[Discussion #25](https://github.com/i2mint/dol/discussions/25) (by thorwhalen,
2023-09-14) explores the ergonomics of writing to composite stores:

```python
# Option A: tuple-based — explicit but rigid
s[k] = (blob, meta)

# Option B: dict-based — current FanoutPersister approach
s[k] = {'bytes_store': blob, 'metadata_store': meta}

# Option C: functional decomposition — store extracts parts from value
s[k] = v  # with functional parameters that tell s how to extract blob and meta from v

# Option D: append with auto-key — store extracts key, blob, and meta from value
s.append(v)  # where s knows how to get k, meta, and blob from v
```

Further questions raised:

- Can implicit metadata (e.g., timestamps, checksums) be merged with explicit
  metadata?
- Should `s[k] = blob` (without metadata) still work, treating the metadata
  store as optional?

In standard terms this is the question of how to design the **aggregate root's
persistence interface**: should the repository accept a fully-formed aggregate
(Option C/D), or should the caller explicitly destructure it (Option A/B)?

### 2.5 Malls: named store registries

[py2store Issue #45 ("Malls: Building multi-store objects")](https://github.com/i2mint/py2store/issues/45)
introduced the `Mall` pattern — a container that holds multiple named stores,
each with its own configuration:

```python
DFLT_MALL_SPEC = imdict({
    'raw_data': {'func': DfStore},
    'xy_data': {'func': PickleStore},
    'learners': {'func': PickleStore},
    'models': {'func': PickleStore}
})
```

A `Mall` is a **store catalog** — it organizes stores by role but does not
itself act as a single composite store. It is a building block for composite
stores: a `FanoutPersister` can be constructed from a `Mall`'s stores.

[py2store Issue #49 ("Structured View of Messy Data")](https://github.com/i2mint/py2store/issues/49)
extends this with "data-separating malls": a store-of-stores that presents
clean, typed perspectives over heterogeneous raw data.

### 2.6 MultiObj: function-level fan-out (i2)

The `i2` package provides the function-level analog of store fan-out. In
[`i2/multi_object.py`](https://github.com/i2mint/i2/blob/301a775729d74b518f77364b282c8cda5f843b59/i2/multi_object.py):

- [`MultiObj`](https://github.com/i2mint/i2/blob/301a775729d74b518f77364b282c8cda5f843b59/i2/multi_object.py#L263):
  A `Mapping` that holds named objects with attribute access. The base
  container pattern.

- [`FuncFanout`](https://github.com/i2mint/i2/blob/301a775729d74b518f77364b282c8cda5f843b59/i2/multi_object.py#L630):
  Applies multiple functions to the same input, yielding `(name, result)` pairs.
  This is the **Scatter-Gather** pattern at the function level.

- [`ContextFanout`](https://github.com/i2mint/i2/blob/301a775729d74b518f77364b282c8cda5f843b59/i2/multi_object.py#L876):
  Bundles multiple context managers into one composite context manager.

[dol Discussion #25](https://github.com/i2mint/dol/discussions/25) explicitly
references `i2.MultiObject` as prior art for "StoreFanout":

> If it's something like that we want to do with our multiple stores,
> `i2.MultiObject` could be a place to look, and possibly use to make a
> "MultiStore", or "StoreFanout".

---

## 3. Related Patterns and Discussions

### 3.1 Cache-and-sync as a special case

[i2mint Issue #20 ("A fully functional cache-and-sync mechanism")](https://github.com/i2mint/i2mint/issues/20)
discusses coordinating a source store and a target/cache store. It explicitly
connects to the split-store problem:

> Maybe source and/or target can use a store for content and another for
> metadata (see [composite stores](https://github.com/i2mint/dol/discussions/25)).

This frames the cache pattern as a _degenerate case_ of the composite store
pattern: two stores (source + cache) linked by key, with defined read/write
policies.

Related: [dol Issue #56 ("Fast update and synching")](https://github.com/i2mint/dol/issues/56)
discusses the performance challenge of syncing between heterogeneous stores —
the `Mapping.update` interface is too coarse when backends could negotiate
optimized bulk transfers.

### 3.2 Nested store restructuring

Two discussions address how to reshape composite store hierarchies:

- [Discussion #19 ("Permute levels of nested mappings")](https://github.com/i2mint/dol/discussions/19):
  Proposes `flip_levels` / `permute_levels` to restructure
  `dict[user][date]` ↔ `dict[date][user]`. This is an **OLAP pivot** applied
  to nested mappings, and is relevant when composite stores need to be
  queried along different axes.

- [Discussion #20 ("Generalize FlatReader to multiple levels")](https://github.com/i2mint/dol/discussions/20):
  `FlatReader` currently handles two levels; the request is for arbitrary depth.
  This is **denormalization** of a hierarchical store into a flat key space.

### 3.3 Recursive wrapper propagation

[dol Issue #10 ("Recursively applying wrappers")](https://github.com/i2mint/dol/issues/10)
addresses a practical problem with composite stores: when `wrap_kvs` is applied
to the outer store, the transform does not propagate to nested sub-stores
returned as values. A `conditional_data_trans` mechanism is proposed for
recursive wrapping.

This is directly relevant to content-metadata bifurcation because the composite
store's sub-stores often need independent transforms (e.g., JSON codec for
metadata, pickle for content), and those transforms must compose correctly with
the outer store's key transforms.

### 3.4 Routing and declarative dispatch

[i2 Discussion #68 ("Routing: Factories for extendible transformers")](https://github.com/i2mint/i2/discussions/68)
discusses parametrized transformers and declarative routing rules. This is
relevant to composite stores because the core operation — "given a value, route
its parts to the right sub-stores" — is a routing problem.

### 3.5 Representation independence

[i2 Issue #79 ("Stable Roles, Unstable Representations")](https://github.com/i2mint/i2/issues/79)
discusses how resources have stable semantic roles (content, metadata, config)
but unstable representation formats. The solution involves adapters and
**anti-corruption layers** (DDD) — the same architectural motivation behind
separating content from metadata at the store level.

### 3.6 The composite-tree feasibility concern

A key cautionary note from valentin-feron in
[Discussion #35](https://github.com/i2mint/dol/discussions/35):

> @thorwhalen told me today that it will be very hard (or even impossible) to
> get this composite tree since we'll have, at some point, to break the
> interface for some levels of the tree (depending on the underlying data
> infrastructure and/or the data itself).

This highlights a fundamental tension: the Composite pattern promises that the
composite has the same interface as its components, but heterogeneous backends
inevitably have different capabilities (e.g., one supports `__len__`, another
doesn't; one is append-only, another supports random writes). At some level of
the tree, the uniform `Mapping` interface must leak or be restricted.

---

## 4. Design Tensions and Open Questions

### 4.1 Specialization vs. generalization

The `.meta` sidecar proposal (Section 2.2) is specialized for the
content-metadata case: one primary store, one auxiliary store, with the
auxiliary keyed by the primary's key space. It is simple and ergonomic for the
common case.

The `FanoutPersister` approach (Section 2.3.2) is fully general: N co-indexed
stores of equal standing, with explicit routing via dict keys. It handles
content-metadata as a special case but requires the user to name sub-stores in
every write.

The tension: **should dol optimize for the common 2-store case (content +
metadata) with a dedicated API, or rely on the general N-store mechanism?**

### 4.2 Implicit vs. explicit routing

Current `FanoutPersister` uses explicit routing — the user must structure the
value as `{'store_a': ..., 'store_b': ...}`. The alternative (Option C in
Section 2.4) is implicit routing via decomposition functions:

```python
composite = SplitStore(
    stores={'content': s3_store, 'meta': mongo_store},
    split=lambda v: {'content': v.bytes, 'meta': v.info},
    merge=lambda parts: Waveform(bytes=parts['content'], info=parts['meta']),
)
composite[key] = wf  # split is called internally
wf = composite[key]  # merge is called internally
```

This is how `wrap_kvs`'s `preset`/`postget` work for single stores. The open
question is whether to generalize `preset`/`postget` to multi-store scenarios.

### 4.3 Key-transform propagation

When `wrap_kvs` adds key transforms (e.g., path prefix stripping) to a
composite store, those transforms must also apply to sub-stores. Currently,
`FanoutReader`/`FanoutPersister` assume sub-stores share the same key space as
the outer composite, but this is not enforced or automatically propagated.

### 4.4 Consistency and atomicity

When writing to multiple sub-stores, failures can leave the system in an
inconsistent state (content written, metadata not — or vice versa). None of the
current implementations address this. Standard solutions include:

- **Saga pattern**: compensating transactions on failure
- **Outbox pattern**: write to a local log, then asynchronously replicate
- **Two-phase commit**: coordinate all sub-stores (heavyweight)

[Discussion #49](https://github.com/i2mint/dol/discussions/49) on context
managers and transactions is related but does not yet propose a solution for
multi-store atomicity.

---

## 5. Summary of Existing Implementations

| Class | Module | Read | Write | Pattern | Permalink |
|---|---|---|---|---|---|
| `FanoutReader` | `dol/sources.py` | Multi-store gather | — | Scatter-Gather | [L122](https://github.com/i2mint/dol/blob/6f962d25b952ac2542efbc491c902d4290198ad8/dol/sources.py#L122) |
| `FanoutPersister` | `dol/sources.py` | Multi-store gather | Multi-store broadcast | Scatter-Gather | [L294](https://github.com/i2mint/dol/blob/6f962d25b952ac2542efbc491c902d4290198ad8/dol/sources.py#L294) |
| `CascadedStores` | `dol/sources.py` | First-hit + backfill | Write-through | Cache hierarchy | [L452](https://github.com/i2mint/dol/blob/6f962d25b952ac2542efbc491c902d4290198ad8/dol/sources.py#L452) |
| `MultiSource` | `dol/sources.py` | Fallback chain | — | Failover | [L545](https://github.com/i2mint/dol/blob/6f962d25b952ac2542efbc491c902d4290198ad8/dol/sources.py#L545) |
| `FlatReader` | `dol/sources.py` | Denormalized view | — | Flattening | [L79](https://github.com/i2mint/dol/blob/6f962d25b952ac2542efbc491c902d4290198ad8/dol/sources.py#L79) |
| `WriteBackChainMap` | `dol/caching.py` | ChainMap + backfill | First map only | Cache-aside | [L2528](https://github.com/i2mint/dol/blob/6f962d25b952ac2542efbc491c902d4290198ad8/dol/caching.py#L2528) |
| `mk_sourced_store` | `dol/caching.py` | Source fallback | Local cache | Cache-aside | [L2000](https://github.com/i2mint/dol/blob/6f962d25b952ac2542efbc491c902d4290198ad8/dol/caching.py#L2000) |
| `MultiObj` | `i2/multi_object.py` | Named object registry | — | Catalog | [L263](https://github.com/i2mint/i2/blob/301a775729d74b518f77364b282c8cda5f843b59/i2/multi_object.py#L263) |
| `FuncFanout` | `i2/multi_object.py` | — | — | Scatter-Gather (functions) | [L630](https://github.com/i2mint/i2/blob/301a775729d74b518f77364b282c8cda5f843b59/i2/multi_object.py#L630) |

---

## 6. Cross-Reference Index

| Source | Type | URL |
|---|---|---|
| dol Discussion #25: Composite Stores | Discussion | https://github.com/i2mint/dol/discussions/25 |
| dol Discussion #35: Present and Future of dol | Discussion | https://github.com/i2mint/dol/discussions/35 |
| dol Discussion #19: Permute levels | Discussion | https://github.com/i2mint/dol/discussions/19 |
| dol Discussion #20: Generalize FlatReader | Discussion | https://github.com/i2mint/dol/discussions/20 |
| dol Discussion #49: Context managers | Discussion | https://github.com/i2mint/dol/discussions/49 |
| dol Issue #10: Recursive wrappers | Issue | https://github.com/i2mint/dol/issues/10 |
| dol Issue #56: Fast update and synching | Issue | https://github.com/i2mint/dol/issues/56 |
| py2store Issue #58: Metadata support | Issue | https://github.com/i2mint/py2store/issues/58 |
| py2store Issue #45: Malls | Issue | https://github.com/i2mint/py2store/issues/45 |
| py2store Issue #49: Structured view of messy data | Issue | https://github.com/i2mint/py2store/issues/49 |
| i2mint Issue #20: Cache-and-sync | Issue | https://github.com/i2mint/i2mint/issues/20 |
| i2 Issue #79: Stable Roles, Unstable Representations | Issue | https://github.com/i2mint/i2/issues/79 |
| i2 Issue #46: MultiObj single-mapping special case | Issue | https://github.com/i2mint/i2/issues/46 |
| i2 Discussion #68: Routing | Discussion | https://github.com/i2mint/i2/discussions/68 |
