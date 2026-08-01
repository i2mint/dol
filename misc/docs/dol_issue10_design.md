# dol Issues #10 + #2 — Design: Recursive Wrapping of Nested Stores & a Flat KvReader/KvPersister View

> **Status:** Design-only (no code yet). Paired design for **#10** (recursively applying
> wrappers / propagating a wrapper's "DNA" to nested values) and **#2** (a `store_decorator`
> that surfaces `kv_walk` paths as a full flat `KvReader`/`KvPersister`). Produced via an
> understand → approaches → judge → synthesize → **adversarial-verify** workflow; the
> load-bearing write-back mechanism below is the *corrected* one — an earlier candidate that
> pinned the write-back boundary to the recursion surface was empirically shown to infinite-loop
> (see §7.4). All prototypes in this doc were run against real `dol` (`0.3.55`).

---

## 1. TL;DR

`dol` wrappers only wrap the **top level**. If a store is nested, the wrap does not carry to
nested values — `add_path_access({'a': {'b': {'c': 42}}})['a']['b', 'c']` raises `KeyError`
(`dol/trans.py:2964-2971`). Two paired issues want to close this:

- **#10** — make it *easy to opt in* to **recursively** presenting nested values with the same
  behavior (propagate the wrapper's "DNA"), general enough to condition on **path/key/value**
  (not just value), and covering both the **read** side (the nested value looks like the nested
  store) and the **write** side (a nested mutation persists, never silently lost).
- **#2** — a `store_decorator` whose **keys ARE the `kv_walk` leaf-paths**: a genuine flat
  `KvReader` (`__iter__`/`__len__`/`__contains__`) and `KvPersister` (flat `__setitem__`/
  `__delitem__`), not just the read-only `path_get` view that `add_path_get`/`flatten` give today.

**Recommended design:** two focused, discoverably-named `@store_decorator` primitives in a new
dependency-free leaf module (`dol/nested.py`), sharing **one `(path, key, value)` descent
frontier** (identical in shape to `kv_walk`'s `walk_filt`) and reusing the **shipped #16 engine**
(`dol/_paths_core.py`) verbatim for all writes:

- **`recursive_wrap(store, *, descend=…, wrapper=…, of_type=…, create_missing=…, …)`** — #10.
- **`flat_store(store, *, descend=…, writable=False, levels=…, cache_keys=…, …)`** — #2.

The **crux** (verified in §7) is that the recursion **read-surface** and the write-back
**boundary** must be **different objects**: a nested value is returned as a lightweight
**boundary-rooted view** holding only `(boundary, prefix)` — never a detached deserialized copy —
and every nested write routes to `path_set_writeback(boundary, prefix + key, v)` against a
**raw-scalar boundary store** (`add_path_access`-shaped: scalar get → raw value, scalar set →
re-serialize). This makes silent write-loss **structurally impossible** *and* avoids the
infinite recursion that sinks the naive "boundary = self" design. Everything is **strictly
opt-in, default-off, stdlib-only**; the risky shared-code changes (a `store_decorator` type
guard, `conditional_data_trans` retirement, error-taxonomy widening, persistent store-of-stores
creation) are **quarantined** out of the correctness-critical first phase.

---

## 2. Problem statement

### 2.1 #10 — the recursion gap

```python
>>> from dol import add_path_access
>>> s = add_path_access({'a': {'b': {'c': 42}}})
>>> s['a', 'b', 'c']        # 42  — flat path at the TOP works
>>> s['a']['b', 'c']        # KeyError — s['a'] is a bare dict, NOT wrapped
```

The issue asks that this be **easy to opt into**, without making it the default (recursion is
undesirable in the general case). The author's own sketch is a **self-referential fixpoint**:

```python
def add_path_access_if_mapping(v):
    if isinstance(v, Mapping):
        return wrap_kvs(
            add_path_access(v), obj_of_data=add_path_access_if_mapping
        )  # re-applies ITSELF
    return v
```

Today's `conditional_data_trans` (`dol/trans.py:2709`, **not exported**) is a *degraded* version
of that sketch: it dropped the self-reference, so it re-wraps only the **immediate** children and
collapses one level down. The issue also flags two open questions we must answer:

1. **Class form is broken.** `S = conditional_data_trans(dict, …); S(d)['a','b','c']` raises
   `KeyError` (root-caused in §6.3). Instance/factory forms work; the class form does not.
2. **Condition generality.** Is a value-only `condition(value)` enough, or do we need
   `condition(key, value)` / `condition(path, key, value)` (to condition on depth/path)?

The broader vision (issue comment): a **store-of-stores** where *each level's behavior* is
customizable by `(path, key, value)` — e.g. `FilesOfZip` that re-wraps a nested-zip value with
`FilesOfZip`, `DirReader` that returns a `DirReader` for a subfolder, per-folder stop rules.

### 2.2 #2 — flat read is not a flat store

`add_path_get`/`KeyPath` give flat **read** access (`path_get`) but override `__getitem__`
**only** — so a wrapped node is *not* a full `KvReader`: `('b', 'c') in s['a']` is `False`,
`__iter__`/`__len__` are not path-aware. `flatten` (`dol/trans.py:3100`) adds a read-only flat
**iter** (via `leveled_paths_walk`) but is effectively **read-only** and has three live defects
(verified §3.2). #2 asks for the missing surface: **keys that ARE the walk-paths**, with working
`__contains__`, `__len__`, and flat **writes/deletes**.

### 2.3 Why the write side is non-trivial: the write-back trap (verified)

Recursive **read** wrapping is easy. Recursive **write** wrapping hits the same trap #16 solved
for flat paths: a `__getitem__` that returns a **fresh deserialized copy** (any `wrap_kvs` with a
non-identity `obj_of_data`, `Files`, a DB) means mutating a nested value is **silently lost**.

```python
# Verified empirically (Understand phase):
backend = {'a': json.dumps({'b': {'c': 42}})}
rec = <recursive wrap of> wrap_kvs(backend, json.loads, json.dumps)
nested = rec['a']; nested['b', 'c'] = 999
rec['a']['b', 'c']            # -> 42   (write SILENTLY LOST — no exception)
backend['a']                 # -> '{"b": {"c": 42}}'  (untouched)
```

This is the load-bearing hazard: a design that enables the nested **read** but not a persisting
nested **write** *enlarges* a silent-corruption footgun. dol's **No-Silent-Failures** rule makes
this non-negotiable: the natural nested write must **persist or raise**, never vanish.

---

## 3. Current machinery (code-verified)

### 3.1 What we build on

| Piece | Location | Role |
|---|---|---|
| `kv_walk(v, leaf_yield, walk_filt, pkv_to_pv, *, branch_yield, breadth_first, p)` | `dol/base.py:903` | Canonical nested-mapping walker. `walk_filt(p,k,v)->bool` decides recursion (default `val_is_mapping`). **Read-only walk.** |
| `add_path_get` | `dol/trans.py:2717` | Read-only flat path access; overrides `__getitem__` only. `_path_type=tuple`. |
| `add_path_access` | `dol/trans.py:2835` | Read+write+del flat path access. **Already routes tuple-key writes through `path_set_writeback`** when `create_missing` (`dol/trans.py:3030`); scalar keys fall through to `super()`. |
| `flatten(store, *, levels, cache_keys)` | `dol/trans.py:3100` | Read-only flat *view*; `__iter__=leveled_paths_walk`, then `add_path_get`. |
| `conditional_data_trans(store, *, condition, data_trans)` | `dol/trans.py:2709` | `Pipe(data_trans, wrap_kvs(store, obj_of_data=_cdt))`. Re-wraps immediate children only; **unexported**; **class-form broken**. |
| **`_paths_core`** (the shipped #16 engine) | `dol/_paths_core.py` | `path_set_writeback` / `path_del_writeback` (descend by **persistence boundary**, single boundary write-back), `_inplace_build` (two-pass splice), `path_set`, `PathContext`, `PathCreationError`/`PathWritebackError`. Dependency-free leaf. |
| `mk_dirs_if_missing(store_cls, *, max_dirs_to_make, verbose)` | `dol/filesys.py:785` | Persistent nested creation for `Files` (the "Model 2 filesystem case"). Prior art for the deferred P3 hook. |
| `instance_checker(*types)` | `dol/util.py:457` | Value-type predicate factory. |

### 3.2 `flatten`'s three live defects (verified — to be fixed under `flat_store`, not `flatten`)

```python
f = flatten({"a": {"b": {"c": 42}}, "x": {"y": 7}}, levels=2)
(
    "a",
    "b",
    "c",
) in f  # AssertionError at dol/trans.py:3211  (assert len(k) < self._levels)
f["a", "b"] = {"c": 100}
f["a", "b"]  # -> {'c': 42}  (write SILENTLY LOST — add_path_get only)
del f[("x", "y")]  # KeyError ('x','y')  (delete hits the wrong key)
```

These are **not** fixed in place (that would perturb `flatten`'s 32-dependent surface); they are
fixed under the **new** `flat_store` name, leaving `flatten` byte-identical (§9).

---

## 4. Design goals & non-goals

**Goals**

1. **Strictly opt-in, default OFF.** Existing behavior byte-identical; the 32 ecosystem
   dependents are untouched by the additive first phase.
2. **Solve BOTH read and write recursion for #10**, including the write-back trap (Model 1),
   with a `(path, key, value)` condition and a value-only auto-lifted default.
3. **Deliver #2's full flat `KvReader`/`KvPersister`** with a correct `k in m ⟺ k in iter(m)`
   membership contract.
4. **No silent data loss, ever.** A nested/flat write persists or raises loudly.
5. **Class-form correct** — the primitives are genuine `@store_decorator`s that work 4 ways
   (class / instance / factory / bare); the #10 class-bug is *designed out*, not patched.
6. **Don't reintroduce the #18 `self`-trap** at nested levels.
7. **SSOT & dependency-free.** Reuse `_paths_core` + `kv_walk`; fork no engine; stdlib only.
8. **Progressive disclosure.** Plain-dict recursion needs **zero** config; per-level control is
   available; the store-of-stores "DNA" vision (`FilesOfZip`/`DirReader`) is a one-liner.

**Non-goals (loudly deferred, never silent)**

- **Persistent store-of-stores *creation*** (auto-`os.makedirs` for a *new* nested `Files`
  sub-store). Reads and writes into **existing** sub-stores are in scope; creating a **new
  persistent** sub-store keeps raising `PathWritebackError` pointing at `mk_dirs_if_missing`
  (the shipped #16 deferral, `dol/_paths_core.py:426-438`). Deferred to **P3** (§10).
- **Transactional/atomic multi-level writes.** The engine is a non-atomic read-modify-write
  (inherited from #16); an optional lock hook exists; distributed atomicity is out of scope.
- **The `#69`/`#70` refactors themselves.** We *coordinate* (land in the consolidated path
  module, reuse the one splicer) but do not perform the dedup/split here (§11).

---

## 5. Prior art & rationale threads

- **Shipped #16** (`misc/docs/dol_issue16_design.md`) — the Model 1/Model 2 distinction and the
  single-boundary write-back protocol this design reuses. #10 **is** the general form of #16's
  deferred "Model 2".
- **`mk_dirs_if_missing`** (`dol/filesys.py:785`) — the persistent-creation prior art the P3
  hook generalizes; its `max_dirs_to_make`/`verbose` knobs align with the engine's
  `max_created`/`on_create`.
- **`FanoutReader`/`CascadedStores`** (`dol/sources.py`) — existing multi-store composition to
  stay consistent with.
- **Discussions #19** (Permute levels of nested mappings), **#20** (Generalize `FlatReader` to
  multiple levels), **#21** (Clean up & centralize path access) — the original rationale threads
  for this territory; cite rather than re-derive.
- **`remap`** (sedimental.org/remap.html) and **`glom`** (glom.readthedocs.io) — *inspiration
  only* (visit/leaf callbacks, a declarative spec object). **Not imported** (dependency-free).

---

## 6. Approaches considered (workflow panel)

Four approaches were generated (diverse angles) and scored by a 3-lens judge panel
(conventions / correctness+blast / UX+does-it-solve-the-issues). Scores 0–10:

| Approach | Conv. | Corr. | UX | Essence |
|---|:--:|:--:|:--:|---|
| **A. Recipe-first** (fix+generalize+export `conditional_data_trans`; `recursive=` flag; finish `flatten`) | **9** | 6 | 6 | Smallest diff, most conventions-native, forks nothing — but *punts* nested-write-back as a documented footgun (rejected: violates No-Silent-Failures). |
| **B. `recursive_wrap` first-class primitive** (+ `flat_store`; `wrapper=` DNA) | 7.5 | 7 | **8** | Only approach that propagates **arbitrary DNA** (`wrapper=`) — the literal essence of #10; best naming; but two internal read paths and a `writeback=True` default divergence. |
| **C. `flat_store` curried** (#2 first; #10 = curried flat view) | 8 | **8** | 7 | Write-back trap impossible **by construction** (bound views); one mental model; but no `wrapper=` (under-serves the FilesOfZip/DirReader vision) and changes what `s['a']` *returns*. |
| **D. `NestedSpec` DNA object** (one spec, two surfaces) | 8.5 | **8.5** | 7 | Cleanest blast-radius + `of_type=` Model-2 consistency; silent-loss impossible; but heaviest concept tax and no `wrapper=`. |

**Synthesis (this design):** name the two surfaces as first-class discoverable primitives
(**B**'s naming + `wrapper=` for arbitrary DNA), give them the **boundary-rooted-view** write
mechanism that makes silent loss impossible (**C**/**D**), unify their descent on **one
`(path,key,value)` frontier** with a `of_type=` consistency builder (**D**), keep the optional
spec object as *sugar* not the entry point (**A**'s progressive disclosure), and quarantine the
high-blast shared-code changes to a gated later phase (**A**'s discipline).

---

## 7. The recommended design

### 7.1 Public API (dict-first)

Two `@store_decorator`s in a new leaf module `dol/nested.py`, re-exported from `dol/__init__.py`.
A small optional **spec** captures the shared "DNA"; the friendly builder means users rarely
construct it directly.

```python
# #10 — recursive_wrap: propagate the wrapper's DNA to nested values
from dol import recursive_wrap

# (A) plain dict — the 90% case, ZERO config (reference semantics: nested writes just persist)
s = recursive_wrap({"a": {"b": {"c": 42}}})
s["a", "b", "c"]  # 42     flat path at the top
s["a"]["b", "c"]  # 42     child re-wrapped (was KeyError in conditional_data_trans)
s["a"]["b"]["c"]  # 42     FIXPOINT — every level carries the DNA
list(s["a"])  # ['b']  nested view iterates IMMEDIATE keys (faithful to #10)
s["a"]["b", "c"] = 99  # persists

# (B) CLASS form — the #10 sub-bug is structurally impossible (genuine type, not a Pipe)
S = recursive_wrap(dict)
assert isinstance(S, type)
S({"a": {"b": {"c": 42}}})["a", "b", "c"]  # 42  (was KeyError)

# (C) Model-1 copy-semantics — THE WRITE-BACK TRAP, closed by construction
import json
from dol import wrap_kvs

s = recursive_wrap(
    wrap_kvs({}, obj_of_data=json.loads, data_of_obj=json.dumps), create_missing=True
)
s["a"] = {"b": {"c": 42}}
s["a"]["b", "c"] = 999
assert s["a", "b", "c"] == 999  # PERSISTS across a fresh re-read (verified §7.4)

# (D) general (path, key, value) condition — per-depth / per-path stop rules
s = recursive_wrap(store, descend=lambda p, k, v: len(p) < 2 and isinstance(v, Mapping))
s = recursive_wrap(
    store, condition=lambda v: isinstance(v, Mapping)
)  # value-only, auto-lifted

# (E) Model-2 store-of-stores — ONE keyword keeps read-descent & write-boundary consistent
s = recursive_wrap(outer_store, of_type=SubStoreType)  # e.g. a Files / DirReader class
s["grp"]["b", "c"] = (
    99  # recurses into the LIVE sub-store, which persists its own write
)

# (F) arbitrary DNA (the store-of-stores vision as a one-liner)
s = recursive_wrap(
    FilesOfZip("a.zip"), wrapper=FilesOfZip, descend=lambda p, k, v: is_zip_bytes(v)
)
```

```python
# #2 — flat_store: a genuine flat KvReader / KvPersister keyed on kv_walk leaf-paths
from dol import flat_store

d = {"a": {"b": {"c": 42}}, "x": {"y": 7}}

m = flat_store(d)  # read-only KvReader
list(m)  # [('a','b','c'), ('x','y')]     keys ARE the kv_walk leaf paths
len(m)  # 2
("a", "b", "c") in m  # True    (leaf; fixes flatten's AssertionError)
("a", "b") in m  # False   (branch — leaf-only membership; k in m ⟺ k in iter(m))

w = flat_store(d, writable=True)  # KvPersister
w["a", "b", "c"] = 99  # writes THROUGH to d['a']['b']['c'] (no literal-tuple key)
del w["x", "y"]  # deletes the nested leaf (fixes flatten's del KeyError)

flat_store(d, levels=2)  # flatten-parity bounded view, but correct
```

`flatten` stays **byte-identical**; it can later be re-expressed internally as
`flat_store(store, levels=N)` read-only (P2, gated).

### 7.2 The shared descent frontier (SSOT)

Both primitives take the **same** predicate — identical in shape to `kv_walk.walk_filt`:

```
descend(path: tuple, key, value) -> bool     # recurse/iterate into this node?
```

Default: `val_is_mapping` (`dol/base.py`), i.e. recurse into `Mapping`s (Model-1 value-nesting).
Progressive disclosure via a friendly builder:

- `condition=<callable>` — a value-only `condition(value)` is **auto-lifted** to
  `descend(p,k,v)=condition(v)` (answers #10's open question: value-only is *not* general enough,
  but it's the ergonomic default).
- `descend=<callable>` — full `(path,key,value)` control (per-depth / per-folder stop rules).
- `levels=N` — frontier at depth N (`flatten` parity via `mk_level_walk_filt`).
- `of_type=SubStoreType` — the **Model-2 convenience**: sets **both** `descend` **and**
  `explore_further` from one `isinstance` test (and, if `wrapper` is unset, `wrapper=SubStoreType`),
  keeping the read-descent frontier and the write-boundary frontier **consistent** (§8, footgun #2).

### 7.3 Read recursion (boundary-rooted views; fixpoint)

`recursive_wrap` layers an **outer recursion surface** over an **inner raw-scalar boundary**:

- **Boundary `B` = `add_path_access(store, …)`** — a genuine `Store` subclass whose **scalar**
  `__getitem__` returns the **raw** (transform-applied but non-recursive) value and whose scalar
  `__setitem__` re-serializes; tuple-path writes already route through `path_set_writeback`
  (`dol/trans.py:3030`). This is the single **persistence boundary**.
- **Outer surface** — scalar `__getitem__(k)`: `raw = B[k]`; if `descend((k,), k, raw)` return a
  **view** rooted at `(B, (k,))` (optionally `wrapper(view)` for Model-2 DNA), else `raw`. Tuple
  keys reduce **over the outer self** (`reduce(lambda s, key: s[key], k, self)`, exactly
  `add_path_get`'s shape at `dol/trans.py:2823`) so a per-depth `descend` is consulted at *every*
  step — `s['a','b','c']` and `s['a']['b']['c']` behave identically.
- **`_PathView(boundary=B, prefix, descend)`** — a `MutableMapping` holding **no data**. Its
  `__getitem__` mirrors the outer logic (scalar → view or raw; tuple → reduce). `__iter__` yields
  the **immediate** keys of the node at `prefix` (so `list(s['a']) == ['b']`, faithful to #10 —
  *not* deep leaf-paths). Recursion compounds to any depth (fixpoint).

### 7.4 Write-back — the corrected mechanism (SSOT: reuse `path_set_writeback` verbatim)

Every `_PathView.__setitem__(k, v)` computes the **absolute** path and calls the shipped engine:

```python
path_set_writeback(self._boundary, self._prefix + normalize(k), v, **spec.write_kwargs)
# __delitem__  ->  path_del_writeback(self._boundary, self._prefix + normalize(k))
```

There is **one** persistence boundary and **one** write engine, so a nested mutation cannot land
on a detached copy.

> **⚠ The load-bearing correction (adversarial finding — do not regress).** An earlier candidate
> pinned `_boundary = self` (the **outer recursion surface**, explicitly "NOT `self.store`").
> Two independent skeptics reproduced that this **infinite-loops**: `path_set_writeback` begins
> its descent with `child = store[head]` (`dol/_paths_core.py:400`). If `store` is the recursion
> surface, `store['a']` returns a **`_PathView`**, not the plain dict the engine needs;
> `_inplace_build` then treats the view as the node and its terminal splice `cur[rest[-1]] = val`
> calls `_PathView.__setitem__` → `path_set_writeback` **again** → non-terminating
> `RecursionError`. **The recursion read-surface and the write-back boundary MUST be different
> objects.** The boundary is the **raw-scalar** store `B` (whose scalar ops never return views),
> i.e. `add_path_access(store)` / `self.store` — *not* the view-returning surface and *not*
> `wrapped_self(self)` (which climbs to the *outermost* view surface). The `#18` `wrapped_self`
> pattern is for a *different* problem (methods calling `self[k]`); the write-back boundary
> deliberately wants the *inner* raw store.

**Why single-boundary write-back is correct for Model 1:** the whole subtree under a top key is
**one serialized value with exactly one persistence boundary** (`B`). `B['a']` returns a detached
(or live) nested dict; `_inplace_build` mutates it in memory; the single `B['a'] = <spliced
branch>` re-serializes the whole branch through the store's own `data_of_obj`. On a plain dict,
`B['a']` is the same object, so it's a harmless re-set. **One uniform mechanism, both models.**

**Verified** (`scratchpad/verify_corrected.py`, run against `dol 0.3.55`):

```
=== #10 recursive_wrap (corrected) ===
A1 s['a','b','c']  = 42     A3 s['a']['b']['c'] = 42 (FIXPOINT)   A4 list(s['a']) = ['b']
=== Model-1 copy-semantics (json) — the write-back trap ===
C2 after s['a']['b','c']=999 -> s['a','b','c'] = 999 (PERSISTS, no RecursionError)
C3 raw backend bytes = {"b": {"c": 999}}     C4 json.loads(raw)['b']['c'] = 999
=== (path,key,value) depth stop rule ===
D1 type s['a'] = _PathView (recursed)   D2 type s['a']['b'] = dict (raw — depth stop honored)
```

### 7.5 #2 — `flat_store` on `kv_walk` + `path_set_writeback`

`flat_store` = a raw-scalar `add_path_access(store)` boundary **plus** a path-aware collection
surface. It overrides all five dunders from the two SSOT engines:

| Dunder | Source | Note |
|---|---|---|
| `__iter__` | `kv_walk(self.store, leaf_yield=lambda p,k,v: p, walk_filt=descend)` | walk **`self.store`**, *not* `self` — `kv_walk(self)` infinite-loops (real `flatten` walks `self.store`, `dol/trans.py:3193`). `levels=N` is sugar over `mk_level_walk_filt`. |
| `__getitem__` | `path_get` reduce (`add_path_get` shape) | flat leaf read |
| `__contains__` | resolve path **and** `not descend(path, path[-1], value)` | **leaf** membership — replaces `flatten`'s `assert len(k) < levels` (`dol/trans.py:3211`). `() -> False`, `m[()]` raises. `k in m ⟺ k in iter(m)`. `branch_paths=True` additionally admits branches (documented policy). |
| `__len__` | count of `kv_walk` leaf-paths (or cached snapshot) | O(N) uncached; O(1) with `cache_keys`. |
| `__setitem__`/`__delitem__` (only `writable=True`) | `path_set_writeback`/`path_del_writeback` on **tuple** keys; scalar keys fall to `super()` | mirrors `add_path_access` — routing scalar boundary re-writes through the engine would infinite-recurse; letting them fall to `super()` fixes it. Fixes `flatten`'s silent-literal-key write and delete-wrong-key defects. |

`cache_keys` is **reimplemented** (not copied — `flatten`'s `cache_keys=True` path crashes today
with `AttributeError: _keys_cache`): the materialized key set is invalidated on every
set/del/autoviv, with explicit invalidation tests.

**Verified** (same script): `list(m) == [('a','b','c'), ('x','y')]`, `len == 2`,
`('a','b','c') in m` True, `('a','b') in m` False, `() in m` False, `set(m) == {k: k in m}` (the
KvReader contract holds), writable writes persist through to `d['a']['b']['c']` with **no** literal
tuple key at top, deletes work, and a copy-semantics flat write persists through serialization.

> **Reduction worth noting:** `flat_store(store)` ≈ `add_path_access(store)` + a
> `kv_walk`-based collection surface (`__iter__`/`__len__`/`__contains__`). `add_path_access`
> already provides correct flat writes/deletes; #2 is *mostly* "add the collection surface".

---

## 8. Edge cases & the two-predicate footgun

| # | Case | Resolution |
|---|---|---|
| 1 | Nested mutation on copy-semantics store | Routed to `path_set_writeback(B, abs_path, v)` — **persists** (§7.4). Never silent. |
| 2 | **Read-descent vs write-boundary mismatch** (Model-2) | `descend(p,k,v)` (read/iterate) and `explore_further(node,path)` (write-boundary) are genuinely different questions; a mismatch could re-serialize a live sub-store through Model-1. **`of_type=` sets both from one `isinstance` test** (blessed path); a detectably-inconsistent hand-spec should **warn**. Detect boundaries via `MutableMapping` membership + the predicate, **never** by type-name (a `wrap_kvs` store reports name `'dict'` yet `isinstance(x, dict)` is `False`, `isinstance(x, MutableMapping)` is `True`). |
| 3 | Nested write on a **read-only** backend (no `__setitem__`) | Raises loudly from the boundary — never swallowed. |
| 4 | Non-mapping blocks a deep path | `PathCreationError` (inherited from `_paths_core` two-pass guard). |
| 5 | Persistent Model-2 **creation** (new nested `Files`) | Raises `PathWritebackError` pointing at `mk_dirs_if_missing` — **loudly deferred** to P3. No `os.makedirs` in P1/P2. |
| 6 | Flat path collides with a **literal tuple key** | Documented policy + the shipped collision guard (`dol/trans.py:3024`). |
| 7 | `del w['x','y']` empties `d['x']` | Leaves an empty branch (`d['x'] == {}`), which then vanishes from `iter`/`len` — documented; pruning is an open question. |

**#18 safety:** the boundary is captured **explicitly** (`self.store` / the inner
`add_path_access`), and spec callables are **closure-captured** (never stashed as bound methods
that would inject `self`). Nested writes go through the boundary, never `self[k]` on the unwrapped
inner store. For **user** store-of-stores DNA classes whose own methods traverse via `self[k]`,
state the rule: use `dol.wrapped_self` (a safe no-op when unwrapped). Plain-`Mapping` data values
are inherently safe.

---

## 9. Backward compatibility & the class-form fix

### 9.1 The #10 class-bug, root-caused and designed out

`conditional_data_trans` returns `Pipe(data_trans, wrap_kvs(store, obj_of_data=_cdt))`
(`dol/trans.py:2708-2713`). For an **instance**, `store_decorator`'s not-a-type branch
(`dol/trans.py:339-346`) builds `func(Store, …)` then instantiates **by reference**, so the
leading `data_trans` survives. For a **class**, the type branch (`dol/trans.py:352`) returns the
`Pipe` **as** the "class"; `S(d)` then runs `wrap_kvs(dict, …)(data_trans(d))` — a `dict`-subclass
**copy-constructs** and discards the top wrapper, so the top-level tuple-path `__getitem__` is
lost (`S(d)['a','b','c'] -> KeyError`) while the one-level `obj_of_data` recursion still fires.

**Fix in this design:** `recursive_wrap` and `flat_store` are plain `@store_decorator` functions
whose inner `func` **always** returns a genuine `Store` subclass via `kv_wrap_persister_cls`
(the exact 4-way pattern of `add_path_access`, `dol/trans.py:2996`). Recursion lives in the
class's own runtime `__getitem__`, **never** in a `Pipe(data_trans, class)`. So class-in →
class-out in every form (verified: `isinstance(recursive_wrap(dict), type)` and
`recursive_wrap(dict)({...})['a','b','c'] == 42`).

### 9.2 What stays byte-identical

- **P1 is purely additive:** `recursive_wrap`, `flat_store`, the spec + builder, `_PathView` are
  **new** symbols in a **new** leaf module. `wrap_kvs`, `Store`/`kv_walk`, `add_path_*`, `flatten`
  defaults, and `_paths_core` are **untouched**. Default-off ⇒ the 32 dependents are unaffected.
- `flatten` stays byte-identical; its bug-fixes land only under `flat_store`.
- `conditional_data_trans` re-pointing (P2) preserves **instance** behavior byte-identically and
  only *adds* a working class form; it is unexported (imported by no code) so external blast ≈ 0.

**Mandatory gate before any merge:** `pip install -e .` the branch; run the full `dol` suite +
doctests (`trans.py`/`base.py`); the `wrap_kvs` blast-radius scan
(`misc/data/wrap_kvs_blast_radius.json`); and `misc/data/run_dependent_tests.py` across the 32
dependents in transitive-importance order, baseline vs modified. Because P1 is default-off, a
green gate is expected; P2 items are gated individually.

---

## 10. Model-2 scope (three-way, deferral loud)

- **IN SCOPE (P1):** Model-1 value-nesting (read via views, write via single-boundary
  writeback); Model-2 **read**-wrapping (`wrapper=`/`of_type=` re-wraps a sub-store value —
  `DirReader`-of-`DirReader`, `FilesOfZip`-of-zip); Model-2 **write into existing** sub-stores
  (`explore_further` descends into the live sub-store, which persists its own write — verified
  working in `_paths_core` today).
- **LOUDLY DEFERRED (P3):** Model-2 **creation** of a **new persistent** sub-store (nested
  `Files`/`os.makedirs`). Still raises `PathWritebackError` → `mk_dirs_if_missing`; the P3 hook
  generalizes its `try-write / except / ensure_dir(max_dirs_to_make, verbose) / retry` into the
  currently-raising branch, aligning with the engine's `may_create`/`max_created`/`on_create`.

---

## 11. Scoped, phased plan

**P1 — pure additions, correctness-critical, default-off (the core deliverable).**
New leaf module `dol/nested.py`: the spec + friendly builder (value-only auto-lift, `of_type=`,
`levels=`), `recursive_wrap`, `flat_store`, `_PathView`. Model-1 recursive read+write via
boundary-rooted views over `path_set_writeback`/`path_del_writeback` (**reused verbatim**); the
full flat `KvReader`/`KvPersister` with `kv_walk`-driven `__iter__`/`__len__`/leaf-`__contains__`
and correct writes; the `(path,key,value)` frontier; class-form correctness
(`kv_wrap_persister_cls`, never a `Pipe`); #18-safety; reference + in-memory Model-2
read/write-into-existing. Re-export from `__init__`. **No shared-code changes.** Run the mandatory
dependents gate (green expected).

**P2 — behavior-adjacent, each a separate gated + revertible commit.**
(a) re-point `conditional_data_trans` to delegate to `recursive_wrap` — instance behavior
byte-identical + a class-form doctest; (b) add the `assert isinstance(r, type)` guard to
`store_decorator`'s class branch (**bundled with (a)** — it would otherwise turn the still-broken
`conditional_data_trans(dict,…)` into a raise); (c) supersede `flatten` via `flat_store` while
keeping `flatten` byte-identical + a deprecation note; (d) add an `add_path_access(recursive=…)`
convenience alias; (e) widen `_paths_core._boundary_write`'s error taxonomy (TypeError-only today,
`dol/_paths_core.py:298-306`) so persistent backends failing with `OSError` surface as
`PathWritebackError`, re-verified against `Files`/`TextFiles`. Full doctests + blast-radius scan +
32-dependent gate **per commit**. Each of (b)/(e) touches shared code and is independently
revertible.

**P3 — the deferred hard parts, coordinated with #69/#70.**
(a) persistent Model-2 **creation** — a side-effecting persistent-boundary factory hook
(generalizing `mk_dirs_if_missing`) wired into the currently-raising branch, guarded by
`may_create`/`max_created`/`on_create`; (b) a **batch mutate-then-writeback** context to amortize
the O(branch) re-serialize on bulk nested edits; (c) an opt-in **snapshot/detached** read mode for
hot read loops (default stays the live view). Land `flat_store`/`recursive_wrap` in the
consolidated path module from **#70**; dedup `path_get`/the two `flatten`s per **#69**.

---

## 12. Pre-work decisions — resolved (for maintainer ratification)

The pre-work alignment surfaced five scoping decisions; recommended answers, all reflected above:

1. **Claim Model 2 now, or defer again?** → **Both, honestly:** read + write-into-existing **in**
   (P1); persistent **creation** loudly deferred (P3). #10 is the general form of #16's Model 2,
   so the design *maps* it fully even where implementation defers.
2. **One doc or two?** → **One unified paired doc** with a shared core (this doc).
3. **Fold in #69?** → **Coordinate, don't fold.** `flat_store` lands in the consolidated module
   (#70); the `path_get`/two-`flatten` dedup (#69) is a separate, later commit.
4. **Mandate `wrapped_self` for #18?** → **State the rule for user DNA classes**, but note the
   write-back **boundary** is the *inner raw* store (`self.store`), which is deliberately **not**
   `wrapped_self(self)` (§7.4). Don't conflate the two.
5. **Class-form support?** → **Mandatory.** Fixed structurally via `kv_wrap_persister_cls`; the
   `store_decorator` guard is a separate, gated P2 hardening.

---

## 13. Open questions (maintainer decision)

1. **Default write-back stance for `recursive_wrap`:** ON (the natural nested write always
   persists — recommended; a silently-lost write violates No-Silent-Failures) vs OFF (matches
   `add_path_access`'s legacy silent-copy default). ON is a deliberate divergence porters must be
   warned about. **Confirm.**
2. **`flat_store.__contains__` policy:** leaf-only (proposed default, satisfies the KvReader
   contract) vs also-branch (`branch_paths=True`). Pick a default.
3. **Spec exposure:** expose the `NestedSpec` dataclass publicly, or only the `nested_spec()`
   builder + kwargs (keep the surface small)?
4. **Live-view aliasing:** the nested value is a *live* boundary-bound view (reflects later
   boundary writes; type/identity differ from a raw dict). Docs + a P3 snapshot mode, or a
   snapshot in P1?
5. **Naming:** `recursive_wrap` + `flat_store` (proposed) vs promoting/renaming
   `conditional_data_trans`; where they land relative to #70's split.
6. **Error-taxonomy scope:** which exception types beyond `TypeError`/`OSError` should
   `_boundary_write` convert to `PathWritebackError` for uniform loudness across arbitrary
   persistent backends?
7. **Empty-branch pruning on delete** (edge #7): prune now-empty auto-created intermediates, or
   leave them (current behavior, matching plain nested mappings)?

---

## 14. Coordination with other issues

- **#16** — reuses its shipped engine (`_paths_core`) verbatim; #10 is the general form of its
  deferred Model 2. The P2 error-taxonomy widening is a change to the #16 leaf.
- **#69/#70** — P3 lands the new primitives in the consolidated path module and dedups the
  `path_get`/two-`flatten` machinery. P1 deliberately adds a *new* module rather than a fourth
  in-place nested-traversal duplicate, to keep the refactor clean.
- **#18/#5** — the boundary-capture rule (§8) keeps recursion #18-safe; `wrapper=`/`of_type=`
  are the "control the wrapper class" ergonomics #5 gestures at.
```
