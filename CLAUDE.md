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

MyStore = wrap_kvs(
    dict,
    id_of_key=lambda k: k + ".json",
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
    def _id_of_key(self, k):
        return k.upper()

    def _key_of_id(self, _id):
        return _id.lower()

    def _data_of_obj(self, obj):
        return json.dumps(obj)

    def _obj_of_data(self, data):
        return json.loads(data)
```

---

## Ready-Made Codecs

```python
from dol import ValueCodecs, KeyCodecs, Pipe

# Common value codecs
ValueCodecs.pickle()  # pickle.dumps / pickle.loads
ValueCodecs.json()  # json.dumps / json.loads
ValueCodecs.gzip()  # compress/decompress
ValueCodecs.str_to_bytes()  # encode/decode

# Key codecs
KeyCodecs.suffixed(".pkl")  # add/strip suffix
KeyCodecs.prefixed("ns:")  # add/strip prefix

# Chain with Pipe
MyStore = Pipe(KeyCodecs.suffixed(".pkl"), ValueCodecs.pickle())(dict)
```

---

## Store Decorators

Most tools in `trans.py` use `@store_decorator`, making them work 4 ways:

```python
from dol import filt_iter, cached_keys


# As class decorator
@filt_iter(filt=lambda k: k.endswith(".json"))
class MyStore(dict): ...


# As instance wrapper
s = filt_iter(my_store, filt=lambda k: k.endswith(".json"))

# As factory
json_only = filt_iter(filt=lambda k: k.endswith(".json"))
s = json_only(my_store)
```

---

## Caching

```python
from dol import cache_this, cache_vals, store_cached


# Cache a property or method
class MyClass:
    @cache_this
    def expensive(self):
        return sum(range(1_000_000))


# Cache fetched values from a slow store
fast = cache_vals(slow_store)


# Persist function results across sessions
@store_cached(JsonFiles("/cache"))
def compute(x, y):
    return slow_computation(x, y)
```

---

## Testing Approach

Always prototype with `dict` as the backend:

```python
# 1. Test logic with dict
s = wrap_kvs(dict(), obj_of_data=json.loads, data_of_obj=json.dumps)
s["key"] = {"a": 1}
assert s["key"] == {"a": 1}

# 2. Swap to real backend
from dol import Files

s = wrap_kvs(Files("/data"), obj_of_data=json.loads, data_of_obj=json.dumps)
```

Run tests: a bare `pytest` (from the repo root) runs exactly what CI runs — the
`dol/tests/` unit tests **and** every module doctest, with CI's doctest flags.
Narrow it with `pytest dol/tests/` (unit tests only) or `pytest dol/caching.py`
(one module's doctests). Do not add `NORMALIZE_WHITESPACE` to
`doctest_optionflags`: CI does not pass it, so doctests relying on it would pass
locally and fail in CI.

---

## Documentation Index (`misc/docs/`)

| Document | Contents |
|----------|----------|
| [general_design.md](misc/docs/general_design.md) | Language-agnostic design: what dol is, the KV pipeline, layered composition, patterns |
| [dol_design.md](misc/docs/dol_design.md) | Python architecture: class hierarchy, `wrap_kvs` deep dive, `Codec`/`Sig`/`Pipe`, critique |
| [dol_architecture_map.md](misc/docs/dol_architecture_map.md) | Code-verified structural map: module/dependency graph, public API, class hierarchy, `wrap_kvs`/codec machinery deep dive, ranked tech debt. **Start here for refactors.** |
| [issues_and_discussions.md](misc/docs/issues_and_discussions.md) | GitHub issues/discussions themes, known limitations, open design questions |
| [dol_issues_report.md](misc/docs/dol_issues_report.md) | Prioritized issue triage + wave-by-wave tackle order |
| [dol_issue16_design.md](misc/docs/dol_issue16_design.md) | Issue #16 design: optional key-path write-through / autovivification — opt-in `create_missing`, contextual per-level factory, the `path_set_writeback` boundary engine + persistent-store write-back protocol, scoped plan. Design-only (no code yet). |
| [dol_issue18_design.md](misc/docs/dol_issue18_design.md) | Issue #18 design: `self`-not-wrapped delegation trap — `wrapped_self` (shipped) now, is-a wrapping (deferred, major) later. |
| [dol_issue83_design.md](misc/docs/dol_issue83_design.md) | Issue #83 design study: the **inverse** of #18 — a delegated method *receives* the unmapped key. Two delegation routes (a fix for one is a no-op on the other), a 13-package census (mostly **latent**; 12 claims refuted), and options A–F with verified costs: `wrapped_self` has its own silent hole (degrades with no live strong reference), chain-walking free functions break on non-`Store` layers, and the only form correct *by construction* is routing the capability through `__getitem__` as a sibling store. **§5 is the carry-forward list for a future redesign.** |
| [dol_issue86_design.md](misc/docs/dol_issue86_design.md) | Discussion #86 design study: **Option G** — spec-carried boundary codecs on a flat proxy (wrapt lessons, KT/VT-annotated interface specs, flatten-and-compile codec stacks). Prototype in `dol/_interface_wrap.py` (private, experimental). Headline: is-a does **not** fix #83 for backend-direct method bodies — F and G serve disjoint populations; codec laws (§2.5) are the boundary invariant's fine print; flat-model guarantees are scoped to pure-codec stacks (filters/caches still nest). Open question 0: who owns the wrap_kvs endgame. |
| [dol_issue10_design.md](misc/docs/dol_issue10_design.md) | Issues #10 + #2 (paired) design: recursive wrapping of nested stores (`recursive_wrap`) + a flat `KvReader`/`KvPersister` view (`flat_store`), sharing one `(path,key,value)` descent frontier and reusing the #16 `path_set_writeback` engine. Load-bearing fix: the recursion read-surface and the write-back boundary must be **different** objects (naive `boundary=self` infinite-loops). Model-2 read + write-into-existing in scope; persistent creation deferred to P3. Design-only (no code yet). |
| [frontend_dol_ideas.md](misc/docs/frontend_dol_ideas.md) | `zoddal` design: TypeScript KV interface, adapters, Zod bridge, zod-collection-ui integration |

> A **local-only** ecosystem inventory (gitignored) lives in `misc/data/`: dol's 76
> dependents, their usages (file:line), a pre-PR test-gate order + runner, and the
> `wrap_kvs` blast-radius scan. Regenerate with the scripts there.

---

## Agent Skills & Commands (`.claude/`)

**Dev skills** (`.claude/skills/`, for working *on* dol):
- `dol-dev-wrap-kvs` — the `wrap_kvs`/`store_decorator`/`Store.wrap` machinery: the
  signature-conditioning rule, `FirstArgIsMapping`, the delegation architecture + `self`/
  signature traps (#18/#6), and the mandatory dependents test-gate. Read before touching
  `trans.py`/`base.py`.
- `dol-dev-portability` — Windows/POSIX landmines for path/key code.

**Consumer skills** (`.claude/skills/`, for *using* dol):
- `dol-store-building` — wrap any backend behind a dict interface: `wrap_kvs`, codecs, the
  ready-made file stores, `filt_iter`, caching, and self-aware transforms.

**Commands** (`.claude/commands/`): `/new-store`, `/add-codec`, `/explain-store` — interactive scaffolds.

---

## Known Limitations / Gotchas

- **`wrap_kvs` + `self` inside methods**: When a `wrap_kvs`-decorated class uses `self[k]` in its own methods, `self` is the unwrapped inner store, so transforms are bypassed (Issue #18 — delegation architecture). Blessed fix: `from dol import wrapped_self` and write `wrapped_self(self)[k]` to reach the outer, transform-applying store (climbs to the outermost wrapper for stacked/`Pipe` wraps; a safe no-op on direct `Store` subclasses). The older `sq(self)[k]` re-wrap still works. A structural fix (is-a wrapping) is proposed in `misc/docs/dol_issue18_design.md`.
- **`clear()` is disabled** on `KvPersister`. Call `ensure_clear_to_kv_store(store)` to re-enable.
- **No async support** in core. Use synchronous wrappers for async backends (thread pool, etc.).
- **Transforms wanting the store**: a transform is called `f(self, data)` only if its first param is named `self`/`store`/`mapping` **and** it has ≥2 required params; otherwise `f(data)`. Mark explicitly with `FirstArgIsMapping(f)`. (`bytes.decode` as `obj_of_data` now works — Issue #9 fixed.)
- **Windows paths**: cross-platform fixes landed (Issues #40/#52/#58 resolved, CI green). See the `dol-dev-portability` skill before touching path/key code.
- **Key-path write-through / autovivification (Issue #16)**: by default, writing a deep path whose intermediates are missing raises `KeyError` (unchanged), and — a pre-existing gotcha — writing an *existing* deep path on a **copy-semantics/persistent store** (`Files`, any `wrap_kvs` with a non-identity `obj_of_data`) silently mutates a detached copy and is **lost**. Opt in with `create_missing=True` on `add_path_access`/`KeyPath` (or the `autoviv(store)` alias): missing intermediates are created via a contextual `mk_missing(ctx)` factory and every write is persisted through a single boundary write-back — which also closes the silent-loss. Creation is announced via `warnings.warn` (pass `on_create=None` to silence); unsafe cases raise `PathCreationError`/`PathWritebackError`. Engine lives in the dependency-free leaf `dol/_paths_core.py` (`path_set_writeback`); `path_set` there also fixes the old factory-not-propagated bug. Persistent store-of-stores (nested `Files`) is out of scope → deferred to #10. Full design: `misc/docs/dol_issue16_design.md`.
