# dol Issue #83 — Design study: delegation and key mapping

> **Purpose:** the durable record of what a wrapped store gets wrong, how far it reaches, and
> what each candidate fix actually costs — so a future redesign of dol's wrapping machinery
> starts from evidence rather than from first principles again.
>
> Companion to [dol_issue18_design.md](dol_issue18_design.md) (the inverse direction, and the
> is-a recommendation), [dol_issue10_design.md](dol_issue10_design.md) (recursive wrapping), and
> [dol_architecture_map.md](dol_architecture_map.md) §5.4/§5.5 (the mechanics).
>
> Prepared against dol `0.3.58`→`0.3.59`. **Every verdict below was reproduced by running code**,
> and the ones that were refuted are recorded as refuted. Discussion:
> [#86](https://github.com/i2mint/dol/discussions/86). Issues: #83 (this), #18, #82, #10, #6, #5.

---

## TL;DR

1. **Two delegation routes, not one.** `Store.__getattr__` *and* `DelegatedAttribute.__get__`.
   A fix covering one is a silent no-op on the other. Most prior discussion named only the first.
2. **The defect is overwhelmingly latent.** A 13-package census found it bites only when a user
   applies a *key* codec. **12 survey claims were refuted outright.** Do not act on "the family
   is destroying data" — it isn't.
3. **`wrapped_self` — the currently blessed escape — has its own silent failure mode.** It
   degrades to the raw leaf when nothing holds a strong reference to the wrapper, and where the
   leaf owns a prefix the wrong answer is a *plausible string*.
4. **Free functions that walk the chain are not categorically safe either.** They break on a
   non-`Store` layer, where the method form is right.
5. **What is correct by construction: route the capability through `__getitem__`** — expose it as
   a sibling store rather than as a method or an attribute. Needs no key-resolution primitive.
6. **The terminal fix is is-a wrapping** (#18's recommendation), which dissolves 1–5 together.

---

## 1. The mechanism

dol wraps by **delegation (has-a)**: `wrap_kvs(SomeClass)` returns a `Store` subclass holding an
instance of `SomeClass` in `self.store`. The transform hooks
(`_id_of_key`/`_key_of_id`/`_data_of_obj`/`_obj_of_data`) live on the wrapper; the inner instance
never sees them.

Two consequences, in opposite directions:

| | direction | issue |
|---|---|---|
| a method **receives** an unmapped key | outside → in | **#83** |
| a method's `self[k]` **bypasses** the transforms | inside → out | **#18** |

And two routes by which a method reaches the leaf:

| route | when | site |
|---|---|---|
| `Store.__getattr__` | instance-wraps, and `mk_relative_path_store` subclasses | `dol/base.py:742` |
| `DelegatedAttribute.__get__` | class-wraps; `delegate_to` installs one descriptor per attr of `dir(wrapped)` | `dol/base.py:279`, installed `dol/base.py:416-480` |

Both return the method **bound to the leaf**.

```python
from dol import KeyCodecs

class WithUrl(dict):
    def url_for(self, k):
        return f"https://x/{k}"

w = KeyCodecs.prefixed("a/")(WithUrl)({"a/b": 1})
w["b"]  # 1                 correct
w.url_for("b")  # https://x/b       WRONG: the bytes are at a/b
```

Capability detection cannot see it. A `@runtime_checkable` Protocol checks method *presence*;
and since 3.12 `isinstance` uses `getattr_static`, so `isinstance(w, SupportsUrlFor)` is `True`
for a class-wrap and `False` for an instance-wrap. Neither says anything about key correctness.

### Why `mk_relative_path_store` is the most exposed shape

It builds `type(name, (PrefixRelativizationMixin, Store), {})` by hand and puts the leaf in
`.store`. So the leaf's methods are reachable *only* via `Store.__getattr__` — invisible in
`dir(TheClass)`, which is why `Files.is_valid_key` was broken for years without anyone noticing.
Note `dol/paths.py`'s `with_key_validation=True` branch already contained a hand-rolled fix for
exactly this, applied to `_id_of_key` alone.

---

## 2. Scale — the census

13 sibling packages, every claim re-verified by source read plus a runnable repro. Per-package
issues are indexed from #83.

| verdict | packages |
|---|---|
| confirmed-live throughout | `focal` |
| mixed | `azuredol`, `aiofiledol`, `chromadol`, `sqldol`, `mongodol`, `redisdol` |
| latent only | `cosmodol`, `pydrivedol`, `sshdol`, `dynamodol`, `hfdol` |
| refuted | `couchdol` |

**Latent means: correct as shipped, wrong only if a user applies a key codec.** Most of these
packages never wrap their own stores. That distinction is the single most important thing to
carry forward — an earlier draft of the downstream write-up claimed the family was actively
destroying data, and the evidence does not support it.

Latent is still not harmless. The worst cases:

- `cosmodol.CosmosItems.replace` — silently overwrites a *different, real* document, in full.
- `pydrivedol.GDReader.get_url` — returns a URL for the wrong file **and grants anyone/reader
  permission on it**.

Confirmed-live inside dol itself: `Files(d).is_valid_key(k)` returned `False` for a key the store
yields, breaking `all(s.is_valid_key(k) for k in s)`. Fixed in #85.

### The shape that stays clean

`azuredol`'s `ContainerStore` has **essentially no key-taking public methods**. Its rich
per-object surface lives on `BlobHandle`, which binds its blob at *construction*, so a key codec
over the store cannot corrupt it. Residual exposure: two methods in the whole package.

It is not safe because of where its prefix lives. **It is safe because it has almost no seam to
get wrong.** Any future redesign should treat "how many keyed methods does this force adapters to
write?" as a first-class design metric.

---

## 3. The option space, with what each actually costs

### A — `wrapped_self` (shipped; #18's Phase 1)

**Has a silent failure mode.** A delegated bound method holds no reference to the wrapper, so
when nothing else does, the wrapper is freed before the body runs and
`_register_wrapper_backref`'s cleanup **removes the registry entry** — indistinguishable from
"never wrapped":

```python
s = KeyCodecs.prefixed("x/")(BR("logs/"))
s.m_abs("b.txt")  # 'logs/x/b.txt'   correct
KeyCodecs.prefixed("x/")(BR("logs/")).m_abs("b.txt")  # 'logs/b.txt'     WRONG
```

Three aggravating properties:

1. Where the leaf owns a prefix, its own `_id_of_key` still fires, so the wrong answer is a
   **plausible `str`** — a type check cannot catch it. A leaf *without* `_id_of_key` at least
   yields `None`.
2. The predicate is **"no live strong reference"**, not "temporary". A temporary caught in a
   reference cycle silently starts working; `operator.methodcaller('m', k)(obj)` is correct where
   `obj.m(k)` is not. The failure is **intermittent**.
3. Reproduced on CPython **3.10–3.14**. Refcount-driven; `gc.disable()` changes nothing.

**It is detectable, though.** `Store.__init__` probes the leaf with
`hasattr(self.store, "KeysView")`, so a leaf can record that it was ever wrapped and refuse when
`wrapped_self(self) is self` but the flag is set. Verified against instance-wrap, class-wrap and
`Pipe`. A two-line change to `_register_wrapper_backref` (stamp `id(inner)` in a plain set
alongside the weakref) would make this a supported guard rather than a hack — `inner` is provably
alive from registration through the call, so the id cannot have been recycled.

dol's own tests do not cover any of this: every case in `dol/tests/base_test.py` binds the
wrapper to a name.

*Status:* a real guardrail. **Not a correctness mechanism**, and should stop being described as
one.

### B — declarative key-method registration

dol **already has this**: `wrap_kvs(ingoing_key_methods=…, outcoming_key_methods=…)`. It is
untested (the TODO is in `dol/trans.py`) and **verified broken for leaf-defined methods on both
wrap paths** — `getattr(store_cls, name)` at decoration time finds a `DelegatedAttribute` or
nothing, and the generated body calls `super(store_cls, self).<name>()`, which does not consult
`Store.__getattr__`. It fails loudly, at least.

The deeper problem: a method the author forgets to declare fails the same silent way. **The part
that holds is the reflective test, not the registry.** And a signature heuristic is not enough —
"first argument named like a key" misses `delete(name, ...)`, `delete_many(keys)`,
`batch(operations)`, `sync_to(target)`, several of the worst shapes in the census.

### C — free functions that walk the chain

`url_for(store, k)` resolving via `inner_most_key`. Sidesteps A's lifetime problem because the
caller holds the store. **But not categorically safe**, because the walk applies each layer's
`_id_of_key` and that breaks when a layer is not a `Store`:

| chain | wire key | free function | method form |
|---|---|---|---|
| any dol-shipped wrapper | — | correct | correct (if referenced) |
| hand-rolled `__getattr__`-passthrough delegator | `logs/b.txt` | **`logs/logs/b.txt`** | correct |
| same passthrough over a key codec | `logs/x/b.txt` | **`logs/x/x/b.txt`** | correct |
| key codec over a plain `.store`-holding middle layer | `logs/x/b.txt` | **`x/b.txt`** | also wrong |

A passthrough `__getattr__` resolves `_id_of_key` to the *leaf's* bound method, so the walk
applies it twice; a middle layer without `_id_of_key` truncates the walk. Every dol-shipped
wrapper is safe because every `Store` inherits an identity `_id_of_key` (`dol/base.py:729-732`) —
but `dol/base.py:514-518` ships a documented hand-rolled `Delegator` recipe of exactly the
breaking shape.

Note dol's own instance of C, `content_url`, **had the #83 bug** until #85.

*Status:* more reliable than A. Not reliable. Fine for operations that are not keyed.

### D — rebind delegated methods to the wrapper

**Rejected with running-code evidence** in [dol_issue18_design.md](dol_issue18_design.md) §4. The
fatal defect is not blast radius: rebinding binds `self` to the **innermost** `Wrap`, so under a
`Pipe` stack it does not fix the case it exists to fix — 4 of the 6 sites that survey found are
`Pipe` stacks — and stacked-codec writes gain a partial-transform corruption surface. Plus
statically-undetectable crashes (`super(SomeClass, self)` with `self` now a `Wrap`).

*Status:* do not re-propose without new information.

### E — capabilities as parallel Mappings

**Two designs get conflated under this name, and only one works.**

**As a Mapping-valued *attribute* — broken.** A wrapper does not re-wrap it:

```python
KeyCodecs.prefixed("a/")(Leaf)({"a/b": 1}).urls  # {'a/b': ...}  INNER keys
```

The outer store's keys are `['b']`, so `store.urls[k]` `KeyError`s for every key it has. This is
#10's territory, and the `.meta` sidecar proposal in
[dol_content_metadata_bifurcation.md](dol_content_metadata_bifurcation.md) §2.2 names exactly
this as its blocker.

*(There is a working precedent for the propagation half: `Store.__init__` at `dol/base.py:720-727`
copies `KeysView`/`ValuesView`/`ItemsView` up from the leaf and instantiates them with the
**outer** store. Hard-coded to three names, but it is propagation **plus outer binding**, which
is simpler than re-wrapping and may be the right generalization.)*

**As a sibling *store* — correct by construction.**

```python
class BucketHandles(KvReader):  # zero key-taking methods
    def _id_of_key(self, k):
        return self.prefix + k

    def __getitem__(self, k):
        return ObjectHandle(self.bucket, self._id_of_key(k))
```

`__getitem__` is the one thing dol maps correctly at **every** depth, so this needs no
`inner_most_key`, no `wrapped_self`, no guard. Verified correct with no live reference, under
`Pipe`, under `cached_keys`, under a value codec, **and under the non-`Store` passthrough layer
where C is silently wrong.**

Cost: the user must wrap the sibling in parallel with the data store, because re-deriving a codec
chain onto a sibling is #10.

*Status:* the best answer available on 0.3.x. Adopted by `s3dol`
([ADR-0011](https://github.com/i2mint/s3dol/blob/master/misc/docs/decisions/0011-keyed-capability-surface.md)).

### F — is-a wrapping

Make `wrap_kvs(Class)` return a real subclass of `Class`. Then `self` inside a leaf method *is*
the wrapper, the hooks are on the MRO, and **#83 and #18 both disappear** — along with #6, whose
`__signature__` graft exists only because `Wrap` is a generic shell rather than a real subclass.

[dol_issue18_design.md](dol_issue18_design.md) recommends this for 0.4/1.0 with a staged plan
(opt-in `mode='isa'`, default flip gated on a green dependents run). X-large, coupled to #5, and
its §9 still lists the commitment as an open question.

*Status:* the actual fix. Everything above is what you do until it lands.

---

## 4. Guidance for adapter authors on 0.3.x

The criterion that turned out to matter is not *"which is the permanent answer"* but **"which is
correct now AND harmlessly redundant once is-a lands"**.

1. **Have as few key-taking methods as possible — ideally none.** Biggest lever. A capability
   that binds its key at *construction* (a handle) cannot be corrupted by a key codec at all.
2. **Route keyed capabilities through `__getitem__`** — sibling stores (E-as-store).
3. **Free functions for the non-keyed rest** (C), knowing the non-`Store`-layer caveat.
4. **`wrapped_self` only as a guardrail** (A), never as a correctness argument.
5. **Guard with a reflective test** (B's useful half) that fails on any *new* public method
   taking a key, asserted against an explicit inventory rather than a name heuristic.

---

## 5. What a future dol redesign should carry forward

Written for whoever does the "great redesign", when the constraint of not breaking 0.3.x is gone.

1. **The has-a/is-a choice is the root, and it is one decision, not several.** #83, #18 and #6
   are three symptoms of it. Fixing them individually costs more than fixing the cause. Any
   redesign that keeps has-a inherits all three.
2. **A wrapper must be able to express "this method takes a key".** Today the Mapping dunders are
   privileged and everything else is invisible to the transform layer. That asymmetry is the
   whole bug class. Whether the answer is is-a, a declarative spec, or signature introspection,
   the *capability* has to exist.
3. **The two routes must be unified.** `Store.__getattr__` and `DelegatedAttribute.__get__`
   independently reimplement "reach the leaf". Every fix so far has had to be applied twice, and
   the second application is the one people forget.
4. **Key mapping needs an inverse.** `inner_most_key` maps outward→inward; there is no supported
   way to map inward→outward. Anything that *returns* keys (`prefixes`, `walk`, a query, a
   listing) needs it, and today each adapter hand-rolls it.
5. **Resolution helpers must not silently return `None`.** Fixed in #84 for `inner_most`, but the
   pattern (`last_element` over a possibly-empty generator) is worth auditing for elsewhere.
6. **Weakref-based backreferences are the wrong substrate for correctness.** `wrapped_self`'s
   hole is not a bug in its implementation; it is inherent to "recover the wrapper from the leaf
   after the fact". If a redesign needs that relationship, it should be a strong, structural one.
7. **Non-`Store` layers exist in the wild** — dol ships a recipe for one. Any chain-walking
   helper must define its behaviour when a layer does not participate, and should probably refuse
   rather than guess.
8. **Design metric to keep:** how many key-taking methods does the design *force* an adapter to
   write? `azuredol` ≈ 0 and is clean; the packages with the most keyed methods are the ones with
   the most findings.

## 6. What shipped alongside this study

- **#84** — export `inner_most_key`/`unravel_key` from `dol`; `inner_most` raises instead of
  returning `None`; `store_trans_path` recurses with the method it was given (fixing
  `inner_most_val`); two dead bugs in `MakeMissingDirsStoreMixin`'s recovery path.
- **#85** — `content_url` resolves the key through wrapping layers (stopping at the provider so
  its own transform is not double-applied); `mk_relative_path_store` installs key-mapping
  `is_valid_key`/`validate_key`.

Both landed in `0.3.59`. Neither touches `DelegatedAttribute.__get__`, the `delegate_to` copy
loop, or the signature graft — the Phase-1 constraint from
[dol_issue18_design.md](dol_issue18_design.md).

## 7. Verification log

| # | Claim | Result |
|---|---|---|
| 1 | two delegation routes, both leaf-bound | confirmed; class-wrap uses a `DelegatedAttribute` descriptor, `__getattr__` is never hit |
| 2 | `wrapped_self` degrades with no live strong reference | confirmed on CPython 3.10–3.14; `gc.disable()` irrelevant |
| 3 | ...and the wrong answer is a plausible `str` when the leaf owns a prefix | confirmed (`logs/b.txt` for a key at `logs/x/b.txt`) |
| 4 | ...and it is detectable via the `KeysView` probe | confirmed for instance-wrap, class-wrap, `Pipe` |
| 5 | free functions break on a non-`Store` layer | confirmed, 3 shapes, against a wire-key oracle |
| 6 | sibling store correct in all shapes | confirmed incl. no-live-reference, `Pipe`, `cached_keys`, value codec, passthrough |
| 7 | a wrapper does not re-wrap a Mapping-valued attribute | confirmed, class-wrap and instance-wrap |
| 8 | `content_url` returned a URL for the unmapped key | confirmed; fixed in #85 |
| 9 | `Files(d).is_valid_key(k)` False for an existing key | confirmed; fixed in #85 |
| 10 | `wrapped_self` survives pickle/deepcopy | confirmed — it is object lifetime, not serialization, that breaks it |
| 11 | `ingoing_key_methods` broken for leaf-defined methods | confirmed, both wrap paths, fails loudly |
| 12 | 13-package census | 12 survey claims refuted; per-package issues filed |
