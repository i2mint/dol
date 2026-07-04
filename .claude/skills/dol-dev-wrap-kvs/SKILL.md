---
name: dol-dev-wrap-kvs
description: "Understand and safely modify dol's core wrapping machinery — wrap_kvs, store_decorator, Store.wrap, and how transforms are applied. Use when touching dol/trans.py or dol/base.py; when changing how key/value transforms (key_of_id, id_of_key, obj_of_data, data_of_obj, postget, preset) are called; when a transform passed to wrap_kvs behaves unexpectedly (called with/without the store as first arg); when working Issues #9/#12/#18/#6/#5; or when a wrap_kvs change could ripple through the ~32 ecosystem packages that use it. Covers the signature-conditioning rule (name AND arity), the FirstArgIsMapping marker, the delegation (has-a) architecture and its 'self is unwrapped' trap, the subclass-signature-freeze trap, and the mandatory dependents test-gate. For end-user store-building, see the consumer skills; for Windows path issues, see dol-dev-portability."
---

# dol wrapping machinery (wrap_kvs / store_decorator / Store.wrap)

`wrap_kvs` is the heart of dol and its highest-blast-radius surface: **32 of dol's 76
local dependents import it** (`Files` 24, `KvReader` 22, `Store` 12 follow). Any change
here can ripple across the ecosystem, so changes are **breaking-change-class** and gated
by a dependents test run (below). This file maps the machinery and its traps. For the
structural big picture see `misc/docs/dol_architecture_map.md` §5; for the code-verified
issue analysis see `misc/docs/dol_issues_report.md`.

## The one thing to internalize first: dol wraps by **delegation (has-a)**, not inheritance

`wrap_kvs(SomeClass)` does **not** return a subclass of `SomeClass`. It returns a `Store`
subclass (`Wrap`) that holds an *instance* of `SomeClass` in `self.store` and forwards to
it. `Store.wrap = classmethod(partial(delegator_wrap, delegation_attr="store"))`
(`base.py`); the actual subclass is built by `delegate_to` (`base.py`), which:
- makes `class Wrap(Store)` whose `__init__(*args, **kwargs)` builds `delegate = wrapped(*args, **kwargs)` and stores it, and
- copies each *public* attribute/method of the wrapped class onto `Wrap` as a
  `DelegatedAttribute` that forwards to `self.store.<attr>`.

The transform hooks (`_key_of_id`, `_id_of_key`, `_obj_of_data`, `_data_of_obj`) are the
`Store` methods that `Store.__getitem__`/`__iter__`/`__setitem__` call; `wrap_kvs`
overrides them (via `_wrap_outcoming`/`_wrap_ingoing`). Two consequences are the traps
below (#18, #6).

## The signature-conditioning rule (Issue #9/#12) — how a transform is called

A transform can be written two ways and dol must decide which:
```python
obj_of_data=lambda data: ...            # f(data)          — no store
obj_of_data=lambda self, data: ...      # f(self, data)    — wants the store
```
The decision lives in `_has_unbound_self(func)` (`dol/trans.py`) and is used at **four
call sites**: `_wrap_outcoming`, `_wrap_ingoing`, and the `postget`/`preset` blocks in
`_wrap_kvs`. The rule (post-#9 fix) is:

> **wants_self ⟺ first parameter name ∈ `self_names = {"self","store","mapping"}`
> AND the function has ≥ 2 *required* positional params** (`_num_required_positional_params`).

The arity clause is the #9 fix. Without it, unary callables whose first param merely
*happens* to be named `self` — e.g. `bytes.decode`, `str.upper` (method descriptors) —
were mis-called as `f(store, data)` → `TypeError: descriptor 'decode' ... doesn't apply to
a 'Store' object`. **Never reintroduce a name-only check.** When editing, verify with:
```python
_has_unbound_self(bytes.decode)          # -> False   (1 required positional)
_has_unbound_self(lambda self, d: d)     # -> True    (2 required)
```

### The explicit escape hatch: `FirstArgIsMapping`
When the heuristic can't (or shouldn't) infer intent, callers opt in explicitly:
```python
from dol import wrap_kvs, FirstArgIsMapping
wrap_kvs(store, obj_of_data=FirstArgIsMapping(lambda self, data: self.root / data))
```
`FirstArgIsMapping(LiteralVal)` (`trans.py`) marks a transform as wanting the store,
*regardless* of names/arity. All four call sites resolve it through the single helper
**`_resolve_self_convention(trans_func) -> (func, wants_self)`** — use that helper for any
NEW call site; do not re-implement the check. It (a) unwraps the marker and forces
`wants_self=True`, else (b) falls back to `_has_unbound_self`. Note `postget`/`preset`
run their `num_of_args(...) < 2` validation on the **already-unwrapped** function.

This is the intended long-term direction (explicit marker) — the name heuristic is kept
for backward-compat. See `misc/docs/dol_issues_report.md` §Wave-1.

## Trap 1 — `self` inside a wrapped class's methods is the INNER (unwrapped) store (Issue #18)

Because methods are delegated (has-a), a method defined on the wrapped class runs bound to
`self.store`, not the `Wrap` instance:
```python
sq = wrap_kvs(data_of_obj=lambda x: x*x, obj_of_data=lambda x: math.sqrt(x))
@sq
class S(dict):
    def via_self(self, k): return self[k]   # self is the inner dict -> NO transform
s = S(); s['2'] = 2
s['2']            # 2.0  (transformed, via Store.__getitem__)
s.via_self('2')   # 4    (untransformed! self is the inner store)
```
This is **not** the same bug as #9 and the #9 fix does not touch it. Workaround: re-wrap
`self` inside the method (`sq(self)[k]`). A general fix requires rethinking method
delegation — out of scope for conditioning changes; do not conflate.

## Trap 2 — `Store.wrap` freezes a subclass's `__init__` signature (Issue #6)

`delegator_wrap` sets `wrap.__signature__ = Sig(obj)` (`base.py`) so the wrapper advertises
the wrapped class's signature (its own `__init__` is `*args,**kwargs`). But `__signature__`
is a plain class attribute, so **subclasses inherit the frozen signature** and lose their
own params:
```python
@Store.wrap
class A:
    def __init__(self, a=1): ...
class B(A):
    def __init__(self, a=1, b=2): ...
signature(B)        # (a=1)   — B's b=2 is lost; B.__dict__ has no __signature__
```
A fix belongs in the delegation/signature machinery (recompute per-subclass, e.g. via
`__init_subclass__`), independent of the conditioning work.

## `store_decorator`: why wrappers work 4 ways

`wrap_kvs`, `filt_iter`, `cached_keys`, etc. are decorated with `@store_decorator`, which
makes each usable as: class decorator (`@wrap_kvs(...) class S(dict)`), instance wrapper
(`wrap_kvs(a_dict, ...)`), bare-class factory (`wrap_kvs(dict, ...)` → a class), and
parameter-only factory (`wrap_kvs(**opts)` → a decorator). Passing a **type** returns a
class; passing an **instance** returns a wrapped instance. Keep this contract when adding
wrappers, and test all four forms (see `tests/test_trans.py::test_redirect_getattr_to_getitem`).

## Where things live / how to test

- Machinery: `dol/trans.py` (`_has_unbound_self`, `_num_required_positional_params`,
  `_resolve_self_convention`, `FirstArgIsMapping`, `_wrap_outcoming`, `_wrap_ingoing`,
  `_wrap_kvs`, `store_decorator`) and `dol/base.py` (`Store`, `delegator_wrap`,
  `delegate_to`, `DelegatedAttribute`).
- Tests: `dol/tests/test_trans.py`. Doctests in `trans.py` are primary docs — keep runnable.
  (`dol.trans.double_up_as_factory` has a KNOWN pre-existing doctest failure, unrelated.)
- Run: `python -m pytest dol/tests/ -q` and
  `python -m pytest --doctest-modules dol/trans.py -q`.

## MANDATORY: the dependents test-gate (any wrap_kvs/base change)

dol is editably installed across the local ecosystem, so **dependents already import your
working-tree dol** — no reinstall needed. Before landing a wrap_kvs/Store.wrap change:
1. Measure blast radius statically: AST-scan dependents for transforms passed to
   wrap_kvs-family whose first param ∈ self_names (see `misc/data/` scan scripts;
   the reusable pattern found 0 breaking + 12 safe self-convention sites).
2. Run the dependents test-gate **baseline vs modified**: run each key dependent's suite
   with your change, then `git stash` your dol edits, run again (baseline), `git stash pop`;
   a `PASS→FAIL` delta is a real regression (env `FAIL→FAIL` is not). Helpers and the
   gate order live in `misc/data/` (gitignored). Prioritize `config2py → graze → ju →
   tabled → oa` (highest transitive fan-out) plus the self-convention users
   (`py2store, sqldol, tec, unbox, mongodol`).

## Checklist for a PR touching wrap_kvs / Store.wrap

- [ ] All self-vs-data decisions go through `_resolve_self_convention` (no ad-hoc name checks).
- [ ] `_has_unbound_self` keeps BOTH conditions (name AND ≥2 required); `bytes.decode` stays unary.
- [ ] `FirstArgIsMapping` honored in every affected slot (key/value, in/out, postget, preset).
- [ ] All 4 `store_decorator` usage forms still work.
- [ ] Pickling of a wrapped store still round-trips.
- [ ] Blast radius measured + dependents test-gate run (baseline vs modified) → no PASS→FAIL.
- [ ] Didn't conflate #18 (delegation/self) or #6 (subclass signature) with the conditioning fix.
