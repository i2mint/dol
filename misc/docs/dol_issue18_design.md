# dol Issue #18 — Design Proposal: "`self` is unwrapped" in delegation-wrapped stores

> **Purpose:** a decision-ready design for [Issue #18](https://github.com/i2mint/dol/issues/18)
> — why it happens, how far it reaches across the ecosystem, the four fixes considered, and a
> staged recommendation. Companion to
> [dol_issues_report.md](dol_issues_report.md) (tackle order),
> [dol_architecture_map.md](dol_architecture_map.md) §5.4/§5.5 (mechanics), and the
> `dol-dev-wrap-kvs` skill. Full per-package classification (all names) and the raw
> design-workflow output live in the gitignored `misc/data/dol_issue18_blast_radius.json`.
>
> Prepared against dol `master` (post-`0.3.48`). Every verdict below was reproduced or refuted
> by **running code**, not inferred from titles — see the Verification Log (§10).

---

## TL;DR — the recommendation

**Ship Approach B now; commit to Approach C later; reject the rebind family (A, D).**

1. **Now (a `0.3.x` minor):** land **`wrapped_self()`** — a small, *additive*, opt-in helper.
   Inside a delegation-wrapped class's own method you write `wrapped_self(self)[k]` to reach the
   outer, transform-applying store. **Zero default behavior change**, so the 32-dependent
   test-gate stays byte-for-byte green. This blesses (and makes pickle-safe + multi-layer-correct)
   the `sq(self)[k]` workaround the issue author already discovered.
2. **Later (a major version, `0.4`/`1.0`):** land **is-a wrapping** (make `wrap_kvs(Class)` return
   a real subclass of `Class`). This removes the root has-a trap and resolves **#18 and #6
   together**. Land it opt-in first (`mode='isa'`), gate the default flip on a green dependents
   run, co-design with #5. `wrapped_self` degrades to a no-op — shipping it now forecloses nothing.
3. **Reject rebinding delegated methods to the wrapper (Approaches A, D):** *verified* to trade one
   silent bug for a subtler one — under any `Pipe`/stacked wrap it binds `self` to the **innermost**
   layer (so 4 of the 6 fixable sites stay broken and stacked-codec **writes silently corrupt**),
   plus statically-undetectable `super()`/dict-native crashes (§10 EXP6). This violates dol's
   *No Silent Failures* and *solutions-not-patches* conventions.

**Blast radius (26 ecosystem sites surveyed): 0 break · 6 become one-line-fixable latent bugs ·
2 neutral · 18 structurally immune.** The affected sites are real correctness bugs today (e.g.
`xdol` — see §3).

---

## 1. The bug

dol wraps by **delegation (has-a), not inheritance.** `wrap_kvs(SomeClass)` / `Store.wrap(SomeClass)`
returns a `Store` subclass `Wrap` that *holds an instance* of `SomeClass` in `self.store` and
forwards to it. A method **defined on `SomeClass`** is copied onto `Wrap` as a
`DelegatedAttribute` that returns the method **bound to the inner, unwrapped `self.store`**
(`base.py:278`). The transform hooks (`_obj_of_data`/`_data_of_obj`/`_key_of_id`/`_id_of_key`) live
only on the `Store`/`Wrap` layer, which the inner instance never sees. So `self[k]` *inside* such a
method bypasses the transform pipeline and returns **raw** data.

```python
import math
from dol import wrap_kvs

sq = wrap_kvs(data_of_obj=lambda x: x * x, obj_of_data=lambda x: math.sqrt(x))


@sq
class S(dict):
    def via_self(self, k):
        return self[k]  # self is the INNER dict → no transform


s = S()
s["2"] = 2
s["2"]  # 2.0   (transformed, external Store.__getitem__)
s.via_self("2")  # 4     ← BUG: untransformed; self is the inner store
```

**Dispatch path (verified):** `wrap_kvs → _wrap_store → Store.wrap → delegator_wrap(Store, SomeClass)
→ delegate_to` builds `class Wrap(Store)` whose `__init__` constructs `delegate = SomeClass(...)`,
stores it via `super().__init__(delegate)` (→ `Store.__init__`, `base.py:589`), then copies each
public method of `SomeClass` onto `Wrap` as `DelegatedAttribute('store', name)`
(`base.py:334-349`).

**Distinct from #9** (signature conditioning, fixed in PR #72) and **shares its root with #6**
(`delegator_wrap` grafts `wrap.__signature__ = Sig(obj)` at `base.py:451` precisely because `Wrap`
is a generic `*args/**kwargs` shell, not a real subclass — the same has-a design). See §9.

**The bug compounds under nesting.** With two stacked wraps, `self` inside a delegated method is the
*innermost* object and the error accumulates across every layer (§10 EXP2: `9604` vs the correct
`198.0`). This is what sinks the rebind approaches (§4).

---

## 2. Blast radius — 26 ecosystem sites classified

An AST pre-filter found every method on a class that touches the mapping interface on `self`
(`self[...]`, `self.get`, `self.items/keys/values`, `in self`, `for … in self`) across dol's 76
local dependents. Each was then read and classified by whether the class is **delegation-wrapped**
(`self` is the inner store → #18 bites) or a **direct `Store`/`KvReader` subclass** (`self` *is* the
store → immune), and whether the internal access **expects the transformed value** (a fix would
*benefit* it) or the **raw** value (a fix would *break* it).

| Verdict | Count | Meaning |
|---|--:|---|
| **not-affected** | 15 | direct `Store`/`KvReader`/`MutableMapping` subclass, or base-class-is-wrapper but method defined on the wrapper — `self` is the store |
| **false-positive** | 3 | plain `collections.abc` mapping, no dol delegation in the chain |
| **affected-neutral** | 2 | delegation-wrapped but the transform is identity/bijective for this method's use — no observable difference |
| **affected-would-benefit** | 6 | **delegation-wrapped, expects the transformed view → a latent bug a fix would resolve** |
| **affected-would-break** | **0** | *no site relies on the raw/unwrapped behavior* |

**The decisive finding: zero sites break.** Every genuinely affected site is a *latent bug* the fix
would resolve. This mirrors the #9 finding (0 breaking sites across the ecosystem) and means the
directional change is backward-compatible in practice.

### The 6 latent bugs (all in public packages, demonstrated by running code)

| Package | Site | Transform | The actual defect today |
|---|---|---|---|
| `xdol` | `PyFilesReader.is_pkg` | key+value (`Pipe`) | `PyFilesReader(asyncio).is_pkg()` → **`False`** for a real package: `'__init__.py' in self` tests the inner store whose keys are **absolute paths** (relativization bypassed). |
| `xdol` | `PyFilesReader.init_file_contents` | key+value (`Pipe`) | `self.get('__init__.py')` → `None` even for a package, for the same reason. |
| `xdol` | `SetupCfgReader.dependencies_from_all` | filter (`filt_iter`) | A decoy `other.cfg` **leaks** into results (`['SHOULD_NOT_APPEAR', …]`): `self.values()` iterates the **unfiltered** inner store, bypassing `_is_setup_cfg`. |
| `xdol` | `PyprojectReader.dependencies_from_all` | filter (`filt_iter`) | A decoy `*.bak` leaks deps (`['requests>=2.0','DECOY']`): `_is_pyproject_toml` filter bypassed. |
| `unbox` | `ModuleNamesImportedByModule.print_kvs` | key transform | `self.items()` yields inner keys (module *objects*), not the outer key-decoded module-name strings the method prints. |
| `lexis` | `WordnetElement.dict_of_non_empty_values` | filter (`cached_keys`) | `self.items()` over the wrapped view (test-covered — the clearest regression sentinel). |

The 2 neutral sites (`oa.OaMapping._len`, `._contains`) are delegation-adjacent but their key
transform is identity/bijective for count/existence, so behavior is unchanged either way.

---

## 3. The four designs considered

| # | Approach | One-liner | Effort | Judge avg | Verdict |
|---|---|---|--:|--:|---|
| **B** | **blessed re-wrap** | `wrapped_self(self)[k]` helper; zero default change, opt-in | small | **4.0** | **Ship now** |
| **C** | **is-a inheritance/mixin** | `wrap_kvs(Class)` returns a real subclass of `Class` | x-large | 2.33 | **Terminal fix, later, opt-in-first** |
| D | staged opt-in rebind | markers (`@sees_wrapped_self`, `rebind_methods=True`) turn on rebinding | medium | 3.67 | Rejected as primary (see §4) |
| A | rebind (default) | delegated methods bind to the outer `Wrap` | medium | 3.0 | Rejected (see §4) |

(Scores are the mean of three adversarial lenses: backward-compat/blast-radius, correctness/edge-cases,
design-philosophy/ergonomics. No approach was outright killed; the spread reflects risk, not viability.)

---

## 4. Why the rebind family (A, D) is rejected

Both A and D share one engine: make a wrapped-class method run with `self` = the outer `Wrap`.
Attractive in the one-layer toy case, but **three verified defects** disqualify it as the default:

1. **Multi-layer corruption (fatal).** Rebinding binds `self` to the *innermost* `Wrap` that copied
   the raw function. Under any `Pipe`/stacked wrap — dol's headline codec-composition pattern, and
   exactly the shape of **4 of the 6** benefit sites (the `xdol` `Pipe` stacks) — the fix **does not
   land**, and stacked-codec **writes get a partial-transform silent-corruption surface**. It
   "fixes" the sites that need it least and misses the ones that need it most.
2. **Statically-undetectable crashes (verified, §10 EXP6).** A rebound method calling
   `super().__getitem__(k)` compiles `super(SomeClass, self)` — but `self` is now a `Wrap`, not a
   `SomeClass` → `TypeError`. Same for dict-native `dict.__getitem__(self, k)`:
   `descriptor '__getitem__' for 'dict' objects doesn't apply to a 'Wrap' object`. These live in
   method *bodies*; no signature/`getattr_static` vetting can catch them.
3. **Double-transform + state divergence.** A method that already re-does a transform
   (`return self[k].decode()`) now gets the *already-decoded* value → `AttributeError`/silently
   wrong. A rebound method writing `self._cache = …` writes to the `Wrap`, not the inner store →
   silent inconsistency (no `DelegatedAttribute` exists for `__init__`-created attrs).

D is strictly safer than A (default-OFF, explicit markers, closest to #12's `FirstArgIsMapping`
philosophy) but **inherits the same innermost-binding flaw** — its implementation is simply
incorrect for every multi-layer store — and is **class-path-only** (a silent no-op on
instance-wraps, itself a *No Silent Failures* violation). Its marker mechanism is still useful as
the per-class opt-in *vehicle* for enabling C's is-a behavior during transition, which is why it
outranks A.

---

## 5. Recommendation — Approach B (`wrapped_self`), then C

### The mechanism (`wrapped_self`)

A public accessor backed by an `id`-keyed weakref registry populated once in `Store.__init__`:

```python
# user code, inside a delegation-wrapped class's own method:
from dol import wrapped_self


@py_files_wrap  # a Pipe of wrap_kvs/filt_iter/mk_relative_path_store
class PyFilesReader(...):
    def is_pkg(self):
        return "__init__.py" in wrapped_self(self)  # ← outer, relativized+filtered view
```

`wrapped_self(obj)` returns the outer transform-applying `Store` registered for `obj`, or `obj`
unchanged if there is none (a safe no-op on direct subclasses / plain objects). On a `Store`,
`self.wrapped_self` is the property form.

**Why B over the alternatives:**

- **Zero default blast radius** — no method body is altered, nothing is written into any store's
  `__dict__`. It ships in a `0.3.x` minor behind a green dependents gate, exactly how #9/#12 shipped.
- **Sidesteps the rebind showstopper** — `self` is never rebound, so there is no
  innermost-vs-outermost ambiguity, no `super()`/dict-native crash, no double-transform.
- **Covers both class- and instance-wraps** — the single `Store.__init__` registration point fires
  for both (§10 EXP4), unlike A/D.
- **A stepping-stone, not a dead end** — `wrapped_self` becomes a vestigial no-op once C lands, so
  shipping it now forecloses nothing.

### Two verified requirements (must be in the Phase-1 PR)

- **Climb-to-outermost (answers the #1 open question).** The registry maps `id(inner) → wrapper`.
  For the 4 `xdol` `Pipe` stacks, `wrapped_self` must climb to the **outermost** wrapper, not the
  immediate one. **Empirically verified to work** (§10 EXP3): a climb loop resolves the innermost
  raw object → outermost wrapper correctly across a 2-deep stack. A single-layer return would only
  partially fix (or wrongly partially-transform) those sites.
- **`__setstate__` re-registration.** `wrapped_delegator_reconstruct` rebuilds via
  `copyreg._reconstructor` + `__setstate__`, which **bypasses `Store.__init__`** (§10 EXP5) — so an
  unpickled store would not be in the registry and `wrapped_self` would silently revert to the raw
  bug. dol stores are routinely pickled (`mongodol`, `py2store`, `store_cached`), so `__setstate__`
  **must** re-register. Also replace the naive `weakref.finalize(pop, id(inner))` with a guarded
  compare-and-pop (delete only if the stored ref still resolves to the dying wrapper) to close
  `id`-reuse and shared-inner last-wins.

### Honest weaknesses (why it pairs with C)

`wrapped_self` is a **guardrail, not a cure**: the naive `self[k]` stays wrong until an author opts
in, and dol still wraps by delegation. It ergonomizes the correct pattern; it does not auto-repair
existing code. That is exactly why C remains the committed direction.

---

## 6. Staged plan

**Phase 1 — ship B (now, `0.3.x`).** In `dol/base.py`: (a) add a module-global
`_wrapper_backrefs: dict[int, weakref.ref]`, `_register_wrapper_backref(inner, wrapper)`, and
`wrapped_self(obj)` (with the climb loop); (b) one line after `self.store = store` (`base.py:589`)
to register; (c) a `wrapped_self` property on `Store`; (d) `__setstate__` re-registration + guarded
finalizer; (e) gate registration on the wrapped class defining public non-dunder methods (the
`delegate_to` loop already enumerates exactly these) so the common `wrap_kvs(dict, obj_of_data=…)`
and no-method instance-wraps pay zero hot-path cost. **Do not touch** `DelegatedAttribute.__get__`,
the `delegate_to` copy loop, or the `base.py:451` signature graft. Export `wrapped_self` from
`dol/__init__.py`; add a runnable doctest reproducing #18 and its fix; update the `CLAUDE.md`
Known-Limitations #18 entry to make `wrapped_self(self)[k]` the blessed pattern (the `sq(self)[k]`
re-wrap keeps working).

**Phase 2 — fix the 6 sites (follow-on PRs to the dependents, not to dol).** One-line conversions to
`wrapped_self(self)`, each with a regression test — `lexis.WordnetElement.dict_of_non_empty_values`
(the test-covered sentinel), `unbox.ModuleNamesImportedByModule.print_kvs`, and the four `xdol`
sites. The `xdol` tests must build a real dir *with a decoy file* and assert the filter/relativization
is honored — this is the test that would catch a multi-layer resolution regression.

**Phase 3 — land C (later, `0.4`/`1.0`).** Introduce a `_StoreTransformMixin` + is-a subclass builder
behind an opt-in (`mode='isa'` / `isa_wrap`), default stays has-a. Conditionally omit the mixin when
the base already subclasses it (avoids the MRO crash on stacked wraps); drop `Store.__getattr__`'s
delegation on is-a wraps (avoids infinite recursion); ABC-register the mixin under `KvPersister`;
co-design wrapper-class selection with #5. Migrate dol core off distinct-`.store` assumptions
(`Store.__getattr__`/`head`/`sources.py`/`caching.py`). Only after a green dependents run *on the
is-a path*, flip the class-wrap default and drop the `base.py:451` graft — **closing #6**.
Instance-wrap stays has-a permanently (you cannot reclass a live `dict`/`Files`), with `isa_wrap`
as the explicit escape. `wrapped_self` is deprecated, not removed.

---

## 7. Test strategy

The **dependents test-gate** (mandated by `dol-dev-wrap-kvs`) is the primary gate and is decisive
here because B has zero default behavior change: run dol + the 32 `wrap_kvs` dependents' suites with
Phase 1 landed — they **must stay byte-for-byte green**; any red flags a leak (e.g. the registration
line accidentally altering behavior).

- **Pickle round-trip:** an explicit test that constructs a wrapped store, pickles/unpickles it, and
  confirms `wrapped_self` still resolves — validates the `__setstate__` hardening (without it the fix
  silently evaporates across serialization). Use `mongodol`/`py2store`/`config2py` shapes.
- **Neutral sites unchanged:** assert no diff at `oa.OaMapping._len`/`._contains`.
- **Structurally-immune sites inert:** confirm the registration line is a no-op for the ~18
  plain-mapping / direct-subclass sites.
- **Benefit sites (Phase 2):** `lexis` is the sentinel (test-covered); the 4 `xdol` sites get the
  decoy-file regression tests described above.

---

## 8. Interaction with sibling issues (#6, #5, #10) — the leverage argument

- **#6 (subclass `__signature__` freeze) shares #18's root.** Both are symptoms of the has-a design:
  because `Wrap` is a generic `*args/**kwargs` shell (not a real subclass), `delegator_wrap` grafts
  `__signature__ = Sig(obj)` (`base.py:451`), which a user subclass then inherits and never
  recomputes. **Approach C fixes both at once** (a real MRO subclass makes the graft unnecessary) —
  the strongest reason C is the terminal direction. B/D/A do not touch #6.
- **#5 (control the wrapper class)** collides with C's is-a wrapper-selection change (`Store.wrap` and
  #5's `getattr(store, 'wrap')` hook both assume has-a) — so C must be **co-designed with #5**, not
  built in isolation.
- **#10 (recursively wrap nested stores)** overlaps: rebinding + #10's proposed recursive
  `obj_of_data` would double-fire transforms per hop. B avoids this (no rebind). C interacts but is
  tractable.

---

## 9. Open questions for the maintainer

1. **Commit to C as the terminal direction now?** It resolves #18+#6 together at a future major, but
   is coupled to #5 and is x-large. Or keep B as the permanent answer and fix #6 separately?
2. **Naming:** `wrapped_self` vs `outer_store(self)` / `store_self(self)`? A public API name is
   permanent — worth one deliberate bikeshed. (`wrapped_self` reads slightly backwards: it returns
   the *wrapper*, not the wrapped.)
3. **Registry keying:** accept the `id()`-based side registry (required because plain `dict`/C
   backends aren't weakref-able and a weakref in `__dict__` breaks pickling) with the guarded pop +
   `__setstate__` mitigations, or restrict the feature to weakref-able backends?
4. **Ship the off-by-default source lint** (`check_self_access`, AST-based, flags `self[...]` inside
   wrapped-class methods) in dol's own CI only? It has false positives (legitimate raw-self checks
   like `oa._contains`) and false negatives (source-less methods) — advisory value only.

---

## 10. Verification log

Run against dol `master` (post-`0.3.48`); `import dol` used the working tree. Full data in
`misc/data/dol_issue18_blast_radius.json`.

| # | Experiment | Result |
|---|---|---|
| EXP1 | single-layer `via_self` | `s['2']=2.0` but `via_self=4` — bug **confirmed** |
| EXP2 | two-layer stacked wrap | `t['2']=198.0` but `via_self=9604` — **compounds** across layers |
| EXP3 | `wrapped_self` registry + climb | innermost-raw → **outermost** wrapper resolves correctly; `wrapped_self(self)[k]` = `198.0`. **⇒ climb-to-outermost is required** |
| EXP4 | instance-wrap coverage | `wrap_kvs(dict_instance, …)` registers via `Store.__init__` — **B covers instance path**; A/D do not |
| EXP5 | pickle | reconstruct bypasses `Store.__init__` ⇒ **`__setstate__` must re-register** (verified-necessary) |
| EXP6 | rebind crash | `dict.__getitem__(wrapper, …)` → `TypeError: descriptor … doesn't apply to a 'Wrap' object` — **confirms A/D crash hazard** |

> Method note: the blast-radius classification and the four designs were produced by a
> multi-agent design workflow (`wf_1920ccdd-b48`, 34/35 agents; the empirical edge-probe agent
> failed and its work was redone by hand — EXP1–EXP6 above). Every "affected" verdict was
> reproduced by running the dependent's code.
