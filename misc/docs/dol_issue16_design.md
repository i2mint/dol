# dol Issue #16 — Design: Optional Write-Through Autovivification for Key-Paths

## 1. TL;DR

Issue #16 asks that writing through a key-path (`store['a', 'b', 'c'] = v` or `kp['a.b.c'] = v`) **optionally create the missing intermediate levels** — like a `collections.defaultdict`, but with a **contextual, per-level factory** so different depths can produce different container/store types, and working across **heterogeneous, persistent stores** (dicts, `wrap_kvs`-wrapped stores, `Files`), not just in-memory dicts. Today this raises `KeyError` (`dol/trans.py:2980`). The recommended solution adds **strictly opt-in, keyword-only** parameters (`create_missing=False` by default) to the two entry points users already reach for — `add_path_access` (`dol/trans.py:2827`) and `KeyPath` (`dol/paths.py:956`) — backed by **one canonical, store-aware engine** (`path_set_writeback`) plus a single shared in-memory splicer (`_inplace_build`) that also becomes the fixed body of `path_set` (`dol/paths.py:712`). The engine descends by **persistence boundary**, splices missing levels in memory via a contextual `mk_missing(ctx)` factory, and does a **single write-back at the boundary** (`store[head] = node`) — the empirically-verified protocol that persists correctly through copy-semantics/persistent stores and collapses to a harmless re-set on plain dicts — while converting every unsafe situation into a **loud, informative error**, never silent data loss.

---

## 2. Problem statement

### 2.1 The autovivification gap

dol lets you address nested data with a key-path. Reading a deep path is well-supported. **Writing** a deep path when the intermediate levels don't exist is not:

```python
>>> from dol import KeyPath
>>> s = {}
>>> kp = KeyPath('.')(s)
>>> kp['a.b.c'] = 42          # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
    ...
KeyError: 'a'
```

The user wants the intermediate mappings `a` and `a.b` to be created on demand, then the leaf `c` set — the exact behavior a recursive `defaultdict` gives in memory, but generalized so that (a) it is opt-in, (b) each level's container type can be chosen contextually, and (c) it works when the backend is a real, persistent store. The `KeyPath` docstring states the current limitation verbatim (`dol/paths.py:1009-1012`): *"it will not create intermediate nested values for you (as … using `collections.defaultdict`)."* Issue #16 is the request to make that sentence **optionally false**.

### 2.2 Why it's non-trivial: the persistent-store write-back trap

Autoviv is trivial for a plain nested `dict` because `dict.__getitem__` returns a **live reference**: mutating the walked node mutates the store. It is *not* trivial for any store whose `__getitem__` returns a **fresh, detached object** (every `wrap_kvs` store with a non-identity `obj_of_data`, and every persistent backend — `Files`, DB, S3).

The current write path (`add_path_access.__setitem__`, `dol/trans.py:2977-2981`) walks intermediates with `reduce(lambda s, key: s[key], path_head, self)` then mutates the last node in place. On a copy-semantics store this mutates a **detached deserialized copy that is never written back** — and, critically, **raises no error**:

```python
# Verified empirically (Understand phase):
J = wrap_kvs(dict, obj_of_data=json.loads, data_of_obj=json.dumps)(backend)
add_path_access(J)['a', 'b', 'c'] = 999   # existing deep path
# re-read J['a']  ->  {'b': {'c': 42}}    # write SILENTLY LOST, no exception
```

So #16 is not merely a new feature: the walk it must extend **already silently loses writes on every copy-semantics store today, even for existing deep paths**. A correct design must (a) make autoviv opt-in, (b) create missing levels with a contextual factory, and (c) **write each built/mutated branch back to its persistence boundary**, or fail loudly when that is impossible. This is the core engineering difficulty, and it directly implicates dol's paramount **"No Silent Failures"** rule.

---

## 3. Current machinery

| Component | Location | Role | Relevant defect |
|---|---|---|---|
| `path_set(d, key_path, val, *, sep='.', new_mapping=dict)` | `dol/paths.py:712` | In-memory recursive splicer; autoviv **always on**, hardcoded to `new_mapping` **at top level only**. Used by `path_edit` (`dol/paths.py:824`). **Not** on the `KeyPath`/`add_path_access` code path. | **Line-787 bug**: `path_set(d[first_key], remaining_keys, val)` drops both `new_mapping` and `sep`, so every deeper auto-created level silently falls back to `dict`. TODO at `dol/paths.py:709-711` already anticipates a per-level/contextual factory. |
| `path_get` / `_path_get` | `dol/paths.py:405` / `216` | Read-only traversal; `on_error` context-dict handlers; `get_attr_or_item` numeric coercion (read-side only). | None relevant; must stay non-mutating. |
| `KeyPath` (`@dataclass`) | `dol/paths.py:956` | String↔tuple key mapper. `_id_of_key` (`:1025`) splits string→tuple; `__call__` (`:1028`) wraps `add_path_access(store, path_type=...)` then `kv_wrap(self)`. The string-path front door. | Docstring `:1009-1012` is the sentence #16 makes optionally false. |
| `add_path_access(store=None, *, name=None, path_type=tuple)` | `dol/trans.py:2827` | THE write path. `__setitem__` (`:2977`) and `__delitem__` (`:2985`) walk via `reduce(...)` and mutate the penultimate node. `path_type` (`:2811`) is the sole path-vs-literal dispatch switch. | **Line-2980 silent-loss**: mutates a detached copy on copy-semantics stores; no write-back, no error. Missing intermediate raises `KeyError` (correct default). |

**Ecosystem blast radius (76-dependent scan, `misc/data/`):** `KeyPath` → 0 dependents, `add_path_access` → 0, `add_path_get` → 0, `path_set` → 1 (`streamlitfront`, config-tree, default `dict` factory). **Zero existing write-path tests** beyond module doctests. #16 is a near-greenfield, write-side addition; the engineering is correctness, not backward-compat.

---

## 4. Design goals & non-goals

**Goals**
1. **Strictly opt-in, default OFF.** `create_missing=False` everywhere; missing intermediate keeps raising `KeyError`, byte-identically to today.
2. **Contextual per-level factory.** One factory, one signature: `mk_missing(ctx: PathContext) -> MutableMapping`, where `ctx` carries `prev_path`, `key`, `depth`. Plain `dict` autoviv requires *no* factory (progressive disclosure).
3. **Correct write-back through copy-semantics/persistent stores** (`wrap_kvs`+json/pickle, `Files`), verified by read-back from fresh store instances.
4. **No silent data loss, ever.** Every unsafe situation (non-mapping collision, unserializable value, ambiguous key, exceeded guard, unpersisted write-back) is a loud, informative error. Autoviv on **write only**, never on read.
5. **Dependency-free.** No `glom`/`boltons`/`addict`/`i2` imports; stdlib only (a local `_MISSING = object()` sentinel, `warnings`, `collections.namedtuple`, `collections.abc`).
6. **SSOT, not a fork.** One canonical in-memory splicer and one store-aware engine, reused by `path_set`, `path_edit`, and `add_path_access`; fixes the line-787 bug as a byproduct (coordinates with #69).
7. **Observability.** Every fabricated intermediate is announced (defaulting to `warnings.warn`), because a structurally-valid **typo** is otherwise silent.

**Non-goals**
- **Full persistent store-of-stores autoviv** (Model 2 with side-effecting sub-store creation, e.g. nested `Files`/`os.makedirs`) is **out of guaranteed scope** for #16. The `explore_further` hook is provided and works for **reference-semantics (in-memory) sub-store parents**, but auto-materializing *persistent* nested sub-stores is deferred to #10 / `mk_dirs_if_missing`. Attempting it without support yields a loud `PathWritebackError`, never silent junk. (Rationale: adversarial testing showed 2+-level persistent nested-`Files` creation cannot complete safely under a single-engine protocol; see §8.)
- **Read-side changes.** `path_get`/`_path_get` are untouched; no read-side coercion is mirrored onto writes.
- **Transactional/atomic multi-level writes and concurrency control.** The engine is a non-atomic read-modify-write; the concurrency contract is *documented* and an optional lock hook is offered, but distributed atomicity is not in scope (§8).
- **Escape hatch for genuine tuple keys / separator-in-key.** Pre-existing dispatch ambiguity is documented and guarded against silent corruption, but a literal-key marker is out of #16 scope.

---

## 5. Prior art (condensed) + lessons adopted

| Library | Missing-intermediate creation | Contextual factory? | Persistent write-back? | Footgun |
|---|---|---|---|---|
| `collections.defaultdict` | On **read** only; `tree = lambda: defaultdict(tree)` | No — one zero-arg factory | N/A (in-memory, live refs) | Creation on read/membership silently pollutes |
| `glom` `assign(..., missing=dict)` | Explicit opt-in; descend & assign child into parent | No — one uniform callable | Mutates in place; no store notion | Attr/item ambiguity; uniform factory |
| `boltons.iterutils.remap` | Rebuilds bottom-up via `enter(path, key, value)` | **Yes** — `enter` gets full path+key | Agnostic (rebuilds, returns) | Learning curve; not an assignment API |
| `addict.Dict` / `python-box` | Lazy link on attr/key read | No — always same type | In-memory only | **The "avoid" example**: typos silently create |
| `dpath` `new/set` | Type inferred from key (int→list, str→dict) | Partial, by *inference* | In-memory only | Separator/glob collisions; magic inference |

**Lessons adopted:**
1. **Opt-in + typed error** (from `glom`): default `KeyError` preserved; write-through is keyword-only.
2. **Context-carrying factory** (from `remap`'s `enter`): `mk_missing(ctx)` receives `prev_path`, `key`, `depth` — the actual gap every other library leaves open (they max out at a single zero-arg factory). This is exactly the TODO at `dol/paths.py:709-711`.
3. **Fail loud on unsafe write-back** (the `addict` anti-lesson + dol's rule): never a silent no-op.
4. **Autoviv on write only** (avoid `defaultdict`/`addict`'s read-creation footgun).
5. **Explicit factory over inference** (avoid `dpath`'s magic).
6. **Tuple paths as the truth; string+sep is convenience** (`path_type=tuple` sidesteps separator/key collisions).
7. **The write-back-through-persistent-stores problem is dol-specific — no prior art solves it.** dol must add it.

---

## 6. The recommended design

### 6.1 Public API

Three touched entry points (all new args **keyword-only**, all defaulting to today's behavior), one canonical engine, one namedtuple, two exceptions, one optional discoverability alias.

**(a) `add_path_access`** — `dol/trans.py:2827`, currently `(store=None, *, name=None, path_type=tuple)`, gains:

```python
def add_path_access(
    store=None,
    *,
    name=None,
    path_type=tuple,
    create_missing: bool = False,
    mk_missing: 'Callable[[PathContext], MutableMapping] | None' = None,
    explore_further: 'Callable[[Any, tuple], bool] | None' = None,
    may_create: 'Callable[[PathContext], bool] | None' = None,
    on_create: 'Callable[[PathContext], None] | None' = _warn_on_create,
    max_created: 'int | None' = None,
    max_levels: 'int | None' = 20,
    verify_writeback: bool = False,
    writeback_lock=None,
):
```

Passing any of `mk_missing` / `explore_further` / `may_create` / `verify_writeback` **implies `create_missing=True`** (convenience). Idempotency guard retained: apply `add_path_get` only if `_path_type` is absent (so `autoviv(add_path_access(store))` does not double-wrap).

**(b) `KeyPath`** — `dol/paths.py:956` (`@dataclass`), gains matching keyword-only dataclass fields with identical defaults, and forwards them in `__call__` (`dol/paths.py:1028-1030`):

```python
add_path_access(
    store,
    path_type=self._path_type,
    create_missing=self.create_missing,
    mk_missing=self.mk_missing,
    explore_further=self.explore_further,
    may_create=self.may_create,
    on_create=self.on_create,
    max_created=self.max_created,
    max_levels=self.max_levels,
    verify_writeback=self.verify_writeback,
    writeback_lock=self.writeback_lock,
)
```

This is the string-path front door and gives write-through through `KeyPath`'s **own** API — no `Pipe`-ordering footgun.

**(c) `path_set`** — `dol/paths.py:712`, signature preserved for back-compat, gains one arg and is **fixed** to propagate the factory/sep through recursion (line-787 bug):

```python
def path_set(d, key_path, val, *, sep='.', new_mapping=dict,
             mk_missing: 'Callable[[PathContext], MutableMapping] | None' = None):
```

When `mk_missing` is given it supersedes `new_mapping`; otherwise `new_mapping` is adapted internally to `mk_missing=lambda ctx: new_mapping()`, so all existing doctests are byte-identical. `path_set` becomes a **thin wrapper over the shared `_inplace_build`** (see §6.4) — one splicer, not two (#69).

**New canonical engine** (store-aware SSOT), used by `add_path_access.__setitem__`/`__delitem__` on the opt-in path:

```python
def path_set_writeback(store, path, val, *, mk_missing=None, explore_further=None,
                       may_create=None, on_create=_warn_on_create,
                       max_created=None, max_levels=20, verify_writeback=False,
                       writeback_lock=None) -> None:
```

**New public symbols** (exported from `dol/__init__.py`):

```python
PathContext = collections.namedtuple('PathContext', ['prev_path', 'key', 'depth'])

class PathCreationError(KeyError):
    """Blocked/forbidden autoviv: non-mapping collision, may_create veto,
    ambiguous literal-key collision, or guard exceeded.
    Subclasses KeyError so legacy `except KeyError` still catches; only raised on the opt-in path."""

class PathWritebackError(RuntimeError):
    """A boundary write could not be persisted (node is really a sub-store,
    value won't serialize, or verify_writeback mismatch)."""
```

**Discoverability alias** (resolved: ship — §11 Decisions resolved): a thin top-level `autoviv(store, **opts)` ≡ `add_path_access(store, create_missing=True, **opts)`, exported from `dol/__init__.py`.

### 6.2 The contextual per-level factory

**One** factory parameter, **one** name, **one** signature — deliberately avoiding both the `new_mapping`/`intermediate_factory`/`mk_new_store` naming sprawl and any `Sig`-based arity sniffing (the Issue #9 footgun):

```python
mk_missing(ctx: PathContext) -> MutableMapping
```

where `PathContext = namedtuple('PathContext', ['prev_path', 'key', 'depth'])`:
- `prev_path` (tuple): the already-traversed parent path.
- `key`: the missing key about to be created.
- `depth` (int) `== len(prev_path)`: handed directly so "different container per level" is a one-liner.

The factory is **always** called with a single `PathContext` positional — no arity branching. The **default** (when `create_missing=True` and `mk_missing is None`) is an internal `lambda ctx: {}`, so plain-dict autoviv needs no factory. Examples:

```python
# uniform non-dict container
mk_missing = lambda ctx: OrderedDict()

# heterogeneous per-level / per-path (the core #16 ask)
def mk_missing(ctx):
    if ctx.depth == 0:
        return OrderedDict()     # top-level container type
    return {}                    # can branch on ctx.key or ctx.prev_path too
```

Two independent contextual hooks accompany it:
- `explore_further(node, path: tuple) -> bool` (default `None`): returns `True` iff `node` is its **own persistence boundary** (a sub-store that persists its own writes) and should be descended into rather than serialized. **Supported for reference-semantics sub-store parents; persistent nested sub-stores are out of scope (§4, §8).**
- `may_create(ctx: PathContext) -> bool` (default `None` = allow): a whitelist predicate. Returning `False` turns an attempted creation into a loud `PathCreationError` — the only real structural defense against typos (whitelist a fixed config schema).

### 6.3 Core algorithm (descend by persistence boundary)

The engine descends by **persistence boundary**, not blindly by dict level. `store` is a boundary (the outer store, or an `explore_further`-flagged sub-store). Local zero-dep sentinel `_MISSING = object()`.

```python
def path_set_writeback(store, path, val, *, mk_missing=None, explore_further=None,
                       may_create=None, on_create=_warn_on_create,
                       max_created=None, max_levels=20, verify_writeback=False,
                       writeback_lock=None, _prefix=(), _budget=None):
    path = list(path)
    if not path:                                   # CLAIM C fix: explicit, clear error
        raise ValueError("empty key path")
    if mk_missing is None:
        mk_missing = lambda ctx: {}                # trivial default; NO arity sniffing
    if len(_prefix) + len(path) > (max_levels or float('inf')):
        raise PathCreationError(
            f"path too deep (> max_levels={max_levels}) at {_prefix + tuple(path)!r}")
    if _budget is None:
        _budget = [0]                              # count of NEWLY created intermediates (mutable box)

    head, *rest = path
    if not rest:                                   # terminal leaf AT THIS boundary
        _boundary_write(store, head, val, _prefix, verify_writeback, writeback_lock)
        return

    try:
        child = store[head]                        # fresh copy | live ref | sub-store
        created = False
    except KeyError:
        ctx = PathContext(_prefix, head, len(_prefix))
        _authorize_create(ctx, may_create, on_create, _budget, max_created)  # veto/announce/count
        child = mk_missing(ctx)
        created = True

    here = _prefix + (head,)

    if explore_further is not None and explore_further(child, here):
        # child is its OWN boundary -> recurse; it persists its own writes.
        path_set_writeback(child, rest, val, mk_missing=mk_missing,
                           explore_further=explore_further, may_create=may_create,
                           on_create=on_create, max_created=max_created,
                           max_levels=max_levels, verify_writeback=verify_writeback,
                           writeback_lock=writeback_lock, _prefix=here, _budget=_budget)
        if created and _is_reference_semantics_parent(store):
            # ONLY register a newly-created sub-store into an in-memory (reference) parent.
            # NEVER serialize a live sub-store into a persistent parent (adversarial BREAK #2).
            _boundary_write(store, head, child, _prefix, verify_writeback, writeback_lock)
        elif created:
            raise PathWritebackError(
                f"cannot register an auto-created sub-store at {here!r} into a persistent "
                f"parent store. Persistent store-of-stores autoviv is out of scope for #16; "
                f"pre-create the sub-store, or see mk_dirs_if_missing / issue #10.")
        return

    # child is a plain VALUE within this boundary: splice in memory, then ONE write-back.
    _inplace_build(child, rest, val, mk_missing=mk_missing, prefix=here,
                   budget=_budget, may_create=may_create, on_create=on_create,
                   max_created=max_created)
    _boundary_write(store, head, child, _prefix, verify_writeback, writeback_lock)
```

`add_path_access.__setitem__` becomes (replacing `dol/trans.py:2977-2983`) — **note the callables are captured in the generated closure, NOT read off `self` as bound methods** (adversarial BREAK #1, the descriptor trap):

```python
def __setitem__(self, k, v):
    if isinstance(k, self._path_type):
        if not k:
            raise ValueError("empty key path")
        if getattr(self, "_create_missing", False):
            # collision guard (CLAIM A): don't shadow a genuine literal tuple key
            if super(store_cls, self).__contains__(k):
                raise PathCreationError(
                    f"refusing to autoviv path {k!r}: a literal key {k!r} already exists "
                    f"(create_missing is unsafe on stores whose real keys are tuples).")
            path_set_writeback(
                self, list(k), v,
                mk_missing=mk_missing, explore_further=explore_further,   # closure locals
                may_create=may_create, on_create=on_create,
                max_created=max_created, max_levels=max_levels,
                verify_writeback=verify_writeback, writeback_lock=writeback_lock,
            )
        else:                                       # BACKWARD-COMPAT: verbatim legacy walk
            *path_head, last_key = k
            reduce(lambda s, key: s[key], path_head, self)[last_key] = v
    else:
        return super(store_cls, self).__setitem__(k, v)
```

`mk_missing`, `explore_further`, etc. are the **arguments to `add_path_access`**, closed over by the generated method — so `self._mk_missing` (a function on a class = a descriptor = a bound method) is never invoked, eliminating the `TypeError: mk_missing() takes 1 positional argument but 2 were given` crash. Scalars (`create_missing`, `max_levels`, …) may still be stashed on the class for introspection; the *callables* must be closure-captured (or wrapped in `staticmethod`).

**No recursion trap:** the first hop `store[head]` has a **scalar** `head`, so it routes to `super().__getitem__` (backend/codec), never re-entering the path branch; `_inplace_build` mutates a plain returned value, not `self`. The `path_set_writeback(self, ...)` call therefore terminates. Critically, the engine operates on `self`/`super()` and **never uses `wrapped_self`** — `wrapped_self` climbs to the outer `KeyPath` key-split layer and infinite-recurses on the headline `kp['a.b.c'] = v` write; value codecs sit **below** `add_path_access`, so write-back must route **down** via `super()`.

### 6.4 The shared in-memory splicer

`_inplace_build` is the single splice implementation (also the fixed body of `path_set`), with **both** the descent guard **and** the final-target guard (adversarial CLAIM B fix):

```python
def _inplace_build(node, rest, val, *, mk_missing, prefix, budget,
                   may_create=None, on_create=_warn_on_create, max_created=None):
    cur = node
    for i, key in enumerate(rest[:-1]):
        here = prefix + tuple(rest[:i + 1])
        if key not in cur:
            ctx = PathContext(here[:-1], key, len(here) - 1)
            _authorize_create(ctx, may_create, on_create, budget, max_created)
            cur[key] = mk_missing(ctx)
        cur = cur[key]
        if not isinstance(cur, MutableMapping):     # PRE-checked, informative
            raise PathCreationError(
                f"cannot descend into existing non-container at {here!r}: "
                f"found {type(cur).__name__}")
    if not isinstance(cur, MutableMapping):         # CLAIM B fix: guard the LEAF target too
        raise PathCreationError(
            f"cannot set leaf under existing non-container at {prefix!r}: "
            f"found {type(cur).__name__}")
    cur[rest[-1]] = val                             # the leaf write
```

`path_set` (fixed) delegates to this:

```python
def path_set(d, key_path, val, *, sep='.', new_mapping=dict, mk_missing=None):
    if isinstance(key_path, str) and sep is not None:
        key_path = key_path.split(sep)
    key_path = list(key_path)
    if not key_path:
        raise ValueError("empty key path")
    if mk_missing is None:
        mk_missing = lambda ctx: new_mapping()      # adapt zero-arg factory, back-compat
    _inplace_build(d, key_path, val, mk_missing=mk_missing, prefix=(),
                   budget=[0], may_create=None, on_create=None, max_created=None)
```

`_authorize_create` centralizes the veto/announce/count discipline:

```python
def _authorize_create(ctx, may_create, on_create, budget, max_created):
    if may_create is not None and not may_create(ctx):
        raise PathCreationError(
            f"autoviv of {ctx.prev_path + (ctx.key,)!r} vetoed by may_create")
    budget[0] += 1
    if max_created is not None and budget[0] > max_created:
        raise PathCreationError(
            f"exceeded max_created={max_created} newly-created intermediates "
            f"(at {ctx.prev_path + (ctx.key,)!r})")
    if on_create is not None:
        on_create(ctx)                              # default: warnings.warn once per path
```

### 6.5 The loud-failure path

Every unsafe outcome is an exception, never a silent no-op:

| Situation | Error |
|---|---|
| Descend into / set leaf under an existing non-mapping | `PathCreationError` (informative, pre-checked, no partial write) |
| Creation vetoed by `may_create` | `PathCreationError` |
| `max_created` / `max_levels` exceeded | `PathCreationError` |
| Autoviv path collides with a genuine literal tuple key | `PathCreationError` |
| Boundary write-back can't serialize (value / live sub-store) | `PathWritebackError` (with an *accurate* remediation hint) |
| `verify_writeback` re-read differs | `PathWritebackError` (worded truthfully — see §7) |
| Auto-created sub-store into a persistent parent | `PathWritebackError` (deferred to #10) |
| Empty path | `ValueError("empty key path")` |

Default observability: `on_create=_warn_on_create` issues `warnings.warn(f"autoviv created {ctx.prev_path + (ctx.key,)!r}")` (deduped per location by the warnings machinery), so opted-in creation is **never silent** — a typo that fabricates a branch is announced. Bulk tree-builders pass `on_create=None` to silence.

---

## 7. Write-back semantics

### 7.1 The reference-vs-copy rule (verified)

> A path-walk write persists **iff** the walked intermediate node is **aliased to the store's authoritative state**. It is **lost** iff `__getitem__` **materializes a fresh, detached object** each call.

Two aliasing mechanisms persist: (1) **live reference** — the returned object *is* the stored state (plain nested `dict`); (2) **fresh object over shared backing** — a new object each call but a view onto shared external state (e.g. a `Files` sub-view). "Lost" happens when the returned object is a **detached deserialization** with no back-channel (`wrap_kvs`+json over a dict *or* over `Files` — both verified to silently lose today).

Runtime auto-detection (`store[k] is store[k]`) is **unsound for correctness**: a `Files`-of-live-`Files`-views returns `False` identity yet needs no write-back, so `False` cannot be equated with "needs write-back." Therefore the design **always writes back at the boundary**. This is safe: writing back to a live-ref store is a harmless re-set of the same logical value; *not* writing back to a copy store is silent data loss.

### 7.2 The protocol

`_boundary_write` treats **every** store as copy-semantics and always writes the boundary back, converting the pre-existing silent loss into either correct persistence or a loud error:

```python
def _boundary_write(store, key, value, prefix, verify, lock=None):
    ctx_lock = lock if lock is not None else contextlib.nullcontext()
    with ctx_lock:
        try:
            store[key] = value                      # re-serializes on copy/persistent; harmless re-set on dict
        except TypeError as e:                       # e.g. value is a live sub-store the parent can't serialize
            raise PathWritebackError(
                f"could not persist node at {prefix + (key,)!r}: {e}. If this intermediate is "
                f"itself a store, it must be descended into (explore_further) rather than "
                f"serialized; persistent store-of-stores autoviv is out of scope for #16."
            ) from e
        if verify and store[key] != value:
            # NOTE: the value is ALREADY committed at this point (non-atomic).
            raise PathWritebackError(
                f"write-back of {prefix + (key,)!r} PERSISTED but re-read differs "
                f"(lossy/rejecting transform); the store was MUTATED and NOT rolled back.")
```

**Why single-boundary write-back is correct and sufficient for the default (value-nesting / Model 1):** the entire subtree under a top key is **one serialized value with exactly one persistence boundary** (the outer store). Reading it once returns a detached (or live) nested dict whose interior levels are ordinary references; `_inplace_build` mutates them in memory; the single `store[head] = node` re-serializes the whole branch through the store's own `data_of_obj`. Verified on `wrap_kvs(dict, json)` and `wrap_kvs(Files, json)` across fresh store instances re-reading from disk (12/12 checks). This simultaneously delivers autoviv **and** closes the pre-existing silent-loss hole for existing deep paths (overwrite and delete both go through the same read-modify-writeback).

**Rejected alternatives:** (a) blind bottom-up per-level reassign — buys nothing over single-boundary for value-nesting and *breaks* on store-of-stores (serializing a sub-store into its parent → `TypeError`); (b) runtime copy-vs-reference detection — unsound (§7.1); (c) `wrapped_self(self)` — forbidden (§6.3 recursion).

### 7.3 Worked `Files` example (dict first, then Files)

```python
# 1) Prototype with dict (value-nesting via json codec)
>>> import json
>>> from dol import wrap_kvs, add_path_access
>>> backend = {'a': json.dumps({'b': {'c': 42}})}
>>> J = wrap_kvs(backend, obj_of_data=json.loads, data_of_obj=json.dumps)
>>> s = add_path_access(J, create_missing=True)          # opt-in

>>> s['a', 'b', 'c'] = 99                                  # overwrite EXISTING deep path
>>> json.loads(backend['a'])                               # persisted (closes silent-loss)
{'b': {'c': 99}}

>>> s['a', 'b', 'new'] = 7                                 # create ONE missing leaf-level
>>> json.loads(backend['a'])['b']['new']
7

>>> s['fresh', 'x'] = 5                                    # fully-missing top branch
>>> json.loads(backend['fresh'])
{'x': 5}

# 2) Swap to real Files — identical semantics, verified across fresh instances
>>> from dol import Files, wrap_kvs, add_path_access, ValueCodecs, Pipe   # doctest: +SKIP
>>> F = wrap_kvs(Files('/data'), obj_of_data=json.loads, data_of_obj=json.dumps)  # doctest: +SKIP
>>> s = add_path_access(F, create_missing=True)                                   # doctest: +SKIP
>>> s['a', 'b', 'c'] = 99          # writes file 'a' == '{"b": {"c": 99}}'         # doctest: +SKIP
>>> add_path_access(wrap_kvs(Files('/data'), obj_of_data=json.loads))['a', 'b', 'c']  # fresh read-back  # doctest: +SKIP
99
```

String paths via `KeyPath` are identical (it only maps string↔tuple then delegates):

```python
>>> from dol import KeyPath
>>> kp = KeyPath('.', create_missing=True)({})
>>> kp['a.b.c'] = 42
>>> kp['a.b.c']
42
```

---

## 8. Edge cases & decisions

| # | Case | Specified behavior |
|---|---|---|
| 1 | **Empty path `()`** | `ValueError("empty key path")` at engine top **and** in `add_path_access.__setitem__` before dispatch (CLAIM C fix). |
| 2 | **Length-1 path `('solo',)`** | `_boundary_write(store, 'solo', val)` via `super()`; no creation. Identical to today. |
| 3 | **Existing non-mapping intermediate blocks descent** (`{'a': 5}`, write `('a','b')`) | `PathCreationError` — the leaf-target guard now fires for length-1-`rest` too (CLAIM B fix); previously a raw `TypeError` escaped. No partial write. |
| 4 | **Existing non-mapping deeper** (`{'a': {'b': 5}}`, write `('a','b','c')`) | `PathCreationError` from the descent guard. |
| 5 | **Genuine tuple-keyed store** (`{('x','y'): 'real'}`, write `('x','y')`) | `create_missing=True` **collision guard**: `super().__contains__(('x','y'))` is true → `PathCreationError` (CLAIM A fix). Previously this silently produced `{('x','y'):'real', 'x':{'y':'NEW'}}`. Default OFF still raises `KeyError` as today. |
| 6 | **Separator-in-key via `KeyPath`** (`{'a.b': 'real'}`, `kp['a.b'] = v`) | Documented **unsafe combination**: the outer split loses the literal string before the inner guard sees it. Flagged loudly in `KeyPath`/`add_path_access` docstrings and Known Limitations; recommend `path_type=tuple` + explicit tuples when keys may contain the separator. |
| 7 | **Copy-store existing deep path overwrite** | Single-boundary read-modify-writeback persists (closes pre-existing `trans.py:2980` silent-loss). |
| 8 | **Plain nested dict (reference semantics)** | Boundary write-back is a harmless re-set of the same live object; results identical to today. |
| 9 | **Delete of missing intermediate / missing leaf** | `KeyError` (delete **never** vivifies), matching today. |
| 10 | **Delete of existing deep leaf on copy-store under `create_missing`** | Read-modify-writeback persists the deletion (closes delete-side silent-loss). Now-empty intermediates are **left** (matches current observable `{'a': {'b': {}}}`, `dol/paths.py:1006-1007`), not pruned. |
| 11 | **Unserializable value at boundary** | Propagates as `PathWritebackError` (wrapping the codec `TypeError`); never a silent no-op. |
| 12 | **`max_levels` exceeded** | `PathCreationError`; counts total path depth. Default 20. |
| 13 | **`max_created` exceeded** | `PathCreationError`; counts NEWLY-created intermediates per write. Default **`None`** (unbounded) so legitimate multi-level tree-building works; set to a small integer to trip on typos. |
| 14 | **Typo footgun** (`create_missing` ON, `store['databse','host'] = x`) | Not structurally detectable, but **announced**: `on_create` default `warnings.warn` fires per fabricated intermediate (never silent). For hard prevention, pass `may_create` (schema whitelist) → `PathCreationError` on out-of-schema keys, or `max_created`. Default OFF is fully safe (`KeyError`). |
| 15 | **`verify_writeback=True` on accept-but-don't-persist backend** | `PathWritebackError`, **truthfully worded**: "PERSISTED but re-read differs … store was MUTATED and NOT rolled back" (it detects, it cannot undo). |
| 16 | **Model-2 store-of-stores, in-memory reference parent, WITH `explore_further`** | Descend into sub-store (its own `__setitem__` persists); a newly-created sub-store is registered into the reference parent via one `_boundary_write`. |
| 17 | **Model-2 with a PERSISTENT parent (nested `Files`, 2+ levels)** | **Out of scope**: registering an auto-created live sub-store into a persistent parent raises `PathWritebackError` with an accurate hint (adversarial BREAK #2 — the old `if created: _boundary_write` step is dropped for this case). Deferred to #10 / `mk_dirs_if_missing`. |
| 18 | **Model-2 misuse WITHOUT `explore_further`** | `store[head] = <sub-store>` can't serialize → caught → `PathWritebackError` with actionable hint (no longer misdiagnoses by telling the user to set `explore_further` when it is already set). |
| 19 | **Partial failure / atomicity** | A single-threaded value-nesting write **is atomic at the persistence op** (one boundary write; pre-checks raise before any write). Side-effecting (Model-2) factories are **not** transactional: on mid-path failure, already-materialized dirs/stores are **left behind**. The engine tracks newly-created nodes and attaches the created-but-orphaned list to the raised error for caller cleanup. Documented in Known Limitations. |
| 20 | **Concurrency** | The engine is a **non-atomic read-modify-write**. Concurrent writes sharing an ancestor boundary on a copy/persistent store are **last-writer-wins and can silently drop a sibling write** — the one place the feature cannot honor "no silent failures," so it is called out loudly in the docstring + Known Limitations, and optionally guarded by the opt-in `writeback_lock` (a lock/CM wrapping the read-modify-write). Finer-grained sub-stores are advised where concurrent branch updates matter. |
| 21 | **Descriptor trap** (callables stashed on class) | **Avoided**: callables are closure-captured by the generated `__setitem__`/`__delitem__`, not read as bound methods off `self` (adversarial BREAK #1). Regression doctest supplies a real `mk_missing` and a real `explore_further`. |

---

## 9. Backward compatibility & migration

**The default changes nothing — proven empirically (adversarial backward-compat pass).** Applying the OFF-state changes to the real tree (all new keyword-only args added, `getattr(self, '_create_missing', False)` guard with the **verbatim** legacy `reduce`-walk in the `else` branch, KeyPath fields + forwarding, and the `path_set` line-787 fix) yields **identical** results to baseline: 43/43 `paths.py` doctests, 528/528 `trans.py` doctests, 135 tests pass / 2 skip. Specifically:
- `KeyPath('.')({})['a.b.c'] = 1` → `KeyError('a')` (preserved).
- `del s['a.b.c']` still leaves `{'a': {'b': {}}}` (`dol/paths.py:1006-1007` doctest unchanged).
- `streamlitfront`'s 5-level default-`dict` `path_set` usage is byte-identical.
- `store_decorator` tolerates the new keyword-only args (precedent: `add_path_get` already carries `path_type`); appended default-valued `KeyPath` dataclass fields don't disturb construction/ordering.

New args are all keyword-only with today's-behavior defaults; existing positional signatures are unchanged. Ecosystem blast radius is ~zero (§3).

**Two intended behavior changes, both strict bug-fixes with verified zero ecosystem impact, both reachable ONLY on the opt-in path or only affecting non-`dict` factory callers:**
1. **Line-787 fix**: `new_mapping`/`mk_missing` now apply at **every** created level, not just level 1. Only changes output for callers who passed a non-`dict` factory expecting depth > 1 (today silently broken → falls back to `dict`), so any change is a fix. `streamlitfront` uses `dict` → identical; existing `path_set` doctests (incl. the single-level `OrderedDict` one at `dol/paths.py:773`) pass unchanged.
2. **Copy-store silent-loss closed** for writes/deletes to existing deep paths — reachable **solely** via `create_missing=True`, so `create_missing=False` behavior (including the documented pre-existing silent-loss) is unchanged. We do not silently alter the default; we document it loudly and offer the opt-in as the sanctioned fix. *(Maintainer decision §11: whether to make this always-on in a future major.)*

Reads are untouched (`path_get`/`_path_get` unchanged). No new dependencies.

---

## 10. Coordination with #69 / #70 / #10

- **#69 (consolidate `path_get`/`path_set` + duplicate logic):** #16 delivers exactly what #69 wants — **one** canonical in-memory splicer (`_inplace_build`), with `path_set` as a thin wrapper over it, plus the store-aware `path_set_writeback` engine as the SSOT setter. **Do not ship both a recursive `path_set` and an iterative `_inplace_build`** (an earlier draft did — the SSOT claim must be literally true). `path_edit` inherits the line-787 fix and the contextual factory for free. Recommend splitting the pure line-787 bug-fix into its own reviewable PR ahead of the feature (Phase 0, §11).
- **#70 (split oversized `dol/paths.py`, ~2270 LOC):** Create a **required** new leaf module `dol/_paths_core.py` housing `PathContext`, `PathCreationError`, `PathWritebackError`, `_MISSING`, `_authorize_create`, `_warn_on_create`, `_inplace_build`, `path_set`, and `path_set_writeback`. `paths.py` re-exports `path_set`/`PathContext`; `trans.py` imports `path_set_writeback` from it. This **net-shrinks** `paths.py` and pre-stages #70. The leaf is proven cycle-free (references only stdlib + local sentinels; operates on `self`/`super()`, never `wrapped_self`, so it needs no `trans` import). The "function-level deferred import, no new module" alternative **worsens #70 and is recorded as rejected**; a `trans.py`-side deferred import is the only accepted fallback if the module split must slip.
- **#10 (recursive wrapping; `path_type` only affects level 1):** Conceptually adjacent (nested-store traversal) but **not coupled** — #10 is read/wrap-side, #16 is write-side. Persistent store-of-stores autoviv (§4 non-goal, edge case #17) is explicitly deferred to #10 / `mk_dirs_if_missing` (`dol/filesys.py:785`), whose `max_dirs_to_make` guard (`dol/filesys.py:788`) and `verbose` signal (`dol/filesys.py:220`) #16 borrows in spirit (`max_created`, `on_create`).

---

## 11. Scoped implementation plan

**Mandatory before merge of any phase touching `trans.py`/`base.py`/`paths.py`:** load the `dol-dev-wrap-kvs` and `dol-dev-portability` skills, and run the **dependents test-gate** in `misc/data/` (the 76-dependent runner) — especially `streamlitfront` (the sole `path_set` consumer) and the `path_get` consumers (`hubcap`, `ju`, `oa`, `pipoke`, `s3dol`, `yp`).

### Phase 0 — Bug-fix PR (coordinates with #69), no new feature
- **Code:** create `dol/_paths_core.py`; move `path_set` there; introduce `_inplace_build`; make `path_set` a thin wrapper; fix the line-787 propagation (thread `sep`/factory through the shared splicer). Re-export from `paths.py`.
- **Tests (`dol/tests/test_paths.py` — first real write-path suite):** add `test_path_set_factory_propagation` proving a non-`dict` `new_mapping` reaches depth ≥ 2 (currently fails). Combinatorial depth × factory cases.
- **Doctests:** `path_set` doctests unchanged; add one demonstrating per-level `new_mapping` at depth 2.
- **Gate:** full suite + dependents (`streamlitfront`).

### Phase 1 — Opt-in engine + `add_path_access` write-through (Model 1, the core deliverable)
- **Code:** `path_set_writeback`, `_boundary_write`, `_authorize_create`, `_warn_on_create`, `PathContext`, `PathCreationError`, `PathWritebackError`, `_MISSING` in `_paths_core.py`. Extend `add_path_access` with the keyword-only args; **closure-capture** callables in the generated `__setitem__`/`__delitem__`; add the empty-path guard, the literal-collision guard, and the verbatim legacy `else` branch. Mirror `__delitem__` (legacy default; under opt-in, read-modify-writeback the deletion). Export new symbols from `dol/__init__.py`.
- **Tests (`dol/tests/test_path_writethrough.py`, new):**
  - OFF-state parity: `KeyError` on missing intermediate for tuple paths; `del` leaves empty intermediates.
  - Model-1 create + write-back across `dict`, `wrap_kvs(dict, json)`, and `Files` (tempdir) via **fresh-instance read-back**.
  - Contextual `mk_missing` (per-depth container type) **and** `explore_further` on a reference-semantics sub-store parent — the descriptor-trap regression.
  - Every loud-failure edge case (#3, #4, #5, #11, #12, #13, #14 veto, #15, #18).
  - Concurrency doc-example (deterministic interleave) asserting last-writer-wins is documented, not silently "fixed."
- **Doctests:** the §7.3 examples (dict inline; `Files` `+SKIP`).
- **Gate:** full suite + dependents.

### Phase 2 — `KeyPath` forwarding (string-path front door)
- **Code:** add the keyword-only dataclass fields to `KeyPath`; forward all in `__call__` (`dol/paths.py:1028-1030`).
- **Tests:** `KeyPath('.', create_missing=True)` string-path create/write-back; the separator-in-key unsafe combination (edge case #6) asserting the documented behavior; OFF-state `KeyError` parity.
- **Doctests:** update the `KeyPath` docstring — the `dol/paths.py:1009-1012` sentence becomes "by default … ; pass `create_missing=True` to enable it," with a runnable example.
- **Gate:** full suite + dependents.

### Phase 3 — Docs, alias, limitations
- Update `CLAUDE.md` Known Limitations (the silent-loss note becomes "closed under `create_missing`"), `dol/paths.py` module docstring, `llms.txt`/`llms-full.txt`, and `misc/docs/dol_architecture_map.md`.
- Ship the `autoviv` alias (**resolved: yes** — see Decisions resolved below).
- Add the concurrency/atomicity and separator-in-key rows to the public Known Limitations.

### Decisions resolved (maintainer, 2026-07-06)

The four load-bearing API decisions are settled; the implementation should follow these:

1. **Switch = one flag, `create_missing`.** A single `create_missing=True` enables *both*
   autovivification of missing intermediates *and* write-back correctness for existing deep
   paths — simplest mental model, minimal surface. No separate `write_back` flag. Default
   `create_missing=False` remains byte-identical to today (`KeyError` on missing intermediate;
   pre-existing copy-store silent-loss unchanged unless opted in).
2. **Ship the `autoviv` alias.** Export a thin top-level `autoviv(store, **opts)` ≡
   `add_path_access(store, create_missing=True, **opts)` from `dol/__init__.py` — memorable,
   discoverable term-of-art.
3. **Create `dol/_paths_core.py` now (Phase 0).** The leaf module houses `PathContext`, the two
   exceptions, `_MISSING`, `_authorize_create`, `_warn_on_create`, `_inplace_build`, `path_set`,
   and `path_set_writeback`. Kills the `trans → paths` import cycle, net-shrinks `paths.py`, and
   pre-stages the #70 split. (Not a deferred import.)
4. **Default safety posture = loud + unbounded.** When `create_missing=True`: `on_create`
   defaults to `_warn_on_create` (every fabricated intermediate is announced via `warnings.warn`
   — honors "No Silent Failures" for typos), and `max_created` defaults to `None` (legitimate
   deep trees build freely). Hard prevention is opt-in via `may_create` (schema whitelist) or an
   explicit `max_created` integer.

### Open questions still for maintainer decision
1. **Exception taxonomy:** new `PathCreationError(KeyError)` + `PathWritebackError(RuntimeError)` (chosen) vs reusing `path_get`'s `on_error` context-dict style for read/write symmetry.
2. **Always-on silent-loss fix:** should closing the existing-path copy-store silent-loss become always-on (it is a genuine bug, not a relied-upon contract) or stay gated behind the opt-in (chosen, to hold "no default change" verbatim)? Track as a candidate for a future major.
3. **`explore_further` default:** `None` (chosen, safe) vs a convenience heuristic like `isinstance(node, KvPersister)` (re-introduces unsafe runtime store-semantics guessing).
4. **Concurrency hook:** ship `writeback_lock` in v1 (recommended, cheap) and/or an `on_conflict` compare-and-swap callback for backends that support it?
5. **Delete semantics:** leave now-empty auto-created intermediates (chosen, matches current observable behavior) vs prune them.
6. **Persistent store-of-stores (Model 2):** confirm deferral to #10 / `mk_dirs_if_missing` (recommended) vs attempting a bounded persistent-sub-store protocol inside #16.
