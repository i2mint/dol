# The flat model: spec-carried boundary codecs on a flat proxy

> **Status: EXPERIMENTAL** (the roadmap's P0 prototype, now in P1 trials). The
> engine lives in `dol/_interface_wrap.py` — a
> private module, not exported from `dol`. Nothing in `base.py`/`trans.py` changed.
> This doc is the *reference*: how the mechanism works, what it guarantees, and what
> it refuses. The *rationale* — why this design won, the adversarial evidence, and
> the decision record — is [dol_issue86_design.md](dol_issue86_design.md). The
> roadmap for growing it into dol's main wrapping machinery is
> [dol_roadmap.md](dol_roadmap.md). Behavior is pinned by
> `dol/tests/test_interface_wrap.py` (~45 tests). Accurate as of v0.3.63.

## Why this exists (one paragraph)

dol's shipping wrap machinery (`wrap_kvs`, `Store`) composes by **has-a delegation**:
each wrap is an object holding the previous one. That architecture has a two-sided
delegation trap — inside a wrapped method, `self` is the *unwrapped* inner store
(Issue [#18](https://github.com/i2mint/dol/issues/18)), and a delegated non-dunder
method *receives* the outer, unmapped key (Issue
[#83](https://github.com/i2mint/dol/issues/83)) — plus per-layer delegation cost,
broken pickling for stacked wraps, and no inverse key mapping. The flat model
(Option G of discussion [#86](https://github.com/i2mint/dol/discussions/86))
restructures wrapping so the #83 class of bug is impossible *by construction* for
the population it serves, and the other problems dissolve as side effects.

## The model: a wrap is `(leaf, spec, stack)`

One proxy object, however many layers:

| Part | What it is | Where it lives on the proxy |
|---|---|---|
| **leaf** | The raw backend (a `dict`, a `Files`, a boto3-backed store…). Always directly reachable; never a proxy-of-proxy. | `_self_leaf`, exposed as `__wrapped__` |
| **spec** | An interface declaration: per method, *where* the "types of interest" (roles: KT, VT, or any TypeVar name) occur in arguments and returns. | `_self_spec` (an `InterfaceSpec`) |
| **stack** | The flat list of transformer layers. Each layer is a dict `{role: Codec(encoder, decoder)}`. | `_self_stack` (a tuple, innermost-first) |

Method bodies always run against the leaf itself — `self` inside a leaf method **is
the leaf** — so internal `self[k]` / `self.x()` calls stay *below* the boundary and
transforms apply **exactly once**, at the boundary. This is structural, not
careful-coding: the no-double-apply property was verified with a counting encoder
through a spec'd method that internally calls another spec'd method.

### The transformer pair: `Codec`

```python
@dataclass(frozen=True)
class Codec:
    encoder: Callable[[Any], Any]  # outer -> inner (arguments going in)
    decoder: Callable[[Any], Any]  # inner -> outer (results coming out)
    decoded_type: Optional[type] = None  # outer-facing type tag (optional)
    encoded_type: Optional[type] = None  # leafward type tag (optional)
```

`encoder` handles what goes in: keys on `__getitem__`/`__setitem__`/`__delitem__`/
`__contains__`/any spec'd method with a `KT` parameter; values on `__setitem__`.
`decoder` handles what comes out: `__getitem__`'s value, `__iter__`'s keys, any
spec'd return carrying a role. `Iterator[...]`-annotated returns are decoded
**lazily** (via `map`); `Iterable[...]`-annotated *arguments* are **materialized**
to a list (an `Iterable` contract implies the leaf may re-iterate).

Note the vocabulary: **encoder/decoder**, not `id_of_key`/`key_of_id`. The flat
engine adopts the target naming language from day one (see the vocabulary-migration
intent in [dol_roadmap.md](dol_roadmap.md)).

### The stack: extend, never nest

Re-wrapping an already-wrapped object appends a layer to the flat tuple over the
**same leaf** (from `interface_wrap`):

```python
if isinstance(obj, InterfaceProxy):
    leaf = obj._self_leaf  # same leaf — never a proxy-of-proxy
    base_stack = tuple(obj._self_stack)  # copy the existing flat list
    ...
stack = base_stack + ((dict(codecs),) if codecs else ())  # extend, don't nest
```

Wrapping is **copy-not-mutate**: the old proxy stays valid, and an iterator
obtained before a new wrap keeps the pipelines it was compiled with. A 6-deep wrap
is still *one* object: `len(proxy._self_stack) == 6` and `proxy.__wrapped__` is the
raw backend.

**Ordering convention** (the one detail that will bite you if you touch this code):
`_self_stack` is **innermost-first** — index 0 is the first layer applied to the
leaf. Encoding therefore iterates `reversed(stack)` (outermost layer encodes
first, moving inward); decoding iterates `stack` in order (innermost decodes
first, moving outward).

### Flatten-and-compile

At wrap time the stack is compiled **once**:

```python
def _fused_role_funcs(stack, *, direction):
    roles = set()
    for layer in stack:
        roles.update(layer)
    out = {}
    for role in roles:
        if direction == "encode":
            funcs = [layer[role].encoder for layer in reversed(stack) if role in layer]
        else:
            funcs = [layer[role].decoder for layer in stack if role in layer]
        out[role] = _fuse(funcs)  # left-to-right composition into ONE callable
    return out
```

Per role, all encoders fuse into one callable and all decoders into another
(`_fuse` drops identity functions). Then, per spec'd method the leaf actually has,
`_compile_method_plan` bakes a **boundary plan**: encode the role-carrying
arguments → call the *leaf-bound* method → decode the return. Plans live in
`_self_plans`; the generated proxy class dispatches every spec'd method to its
plan.

Plan compilation has three shapes: a positional-only plan for 3.10 builtin slots
with no inspectable signature (`dict.__getitem__`); a fast path when the only
role'd parameter is the first positional; and a general path using
`Signature.bind`. The **spec's signature is the outer contract** — calls bind
against it, so the leaf's own parameter names are irrelevant (`dict` calls its key
`key`; the spec may say `k`).

The "compile for validation" half is `_validate_stack_seams`: where two adjacent
layers carry type tags for the same role, the outer layer's `encoded_type` must be
the inner layer's `decoded_type`, else the wrap refuses loudly. Untagged codecs
stay unchecked (progressive disclosure).

### What the flat list buys

Each of these was a named open problem under the nested model:

- **The inverse key mapping exists by construction** —
  `proxy._decode_role('KT', inner_key)` (the missing primitive of the #83 design
  doc §5.4, needed by anything that *returns* keys: listings, `walk`, prefix
  queries). The nested model has no inverse at all.
- **`inner_most_key` becomes a total fold** — `proxy._encode_role('KT', k)`; no
  `.store`-chain walking, no non-`Store`-layer hazard, because there are no layers
  at runtime.
- **The leaf is a strong structural reference** — `proxy.__wrapped__`, no weakref
  registry.
- **Stacked wraps pickle by construction** — `__reduce__` recompiles from
  `(leaf, spec_source, stack, policy)`; no dynamic class in the payload. (Lambda
  codecs fail loudly; codecs must be module-level/picklable.)
- **One boundary hop at any depth** — measured 3.6× faster than nested `wrap_kvs`
  at depth 6 for `__getitem__`, 3.4× for iteration.

## The spec side

The spec is what lets the boundary cover **all** methods, not just the Mapping
dunders — which is precisely what kills the #83 bug class (a delegated method like
`url_for(k)` receiving the outer, unmapped key).

### Annotated form

```python
from typing import Protocol, TypeVar, Iterator, Iterable

KT, VT = TypeVar("KT"), TypeVar("VT")


class BucketInterface(Protocol[KT, VT]):
    def __getitem__(self, k: KT) -> VT: ...  # Mapping dunders are ordinary
    def __iter__(self) -> Iterator[KT]: ...  # spec entries - no privileged
    def __contains__(self, k: KT) -> bool: ...  # surface
    def url_for(self, k: KT, *, expires_in: int = 3600) -> str: ...
    def delete_many(self, keys: Iterable[KT]) -> None: ...
    def items_page(self) -> Iterator[tuple[KT, VT]]: ...
```

Compilation walks `get_type_hints` + `signature` per method, recording the *paths*
at which role TypeVars occur. Supported shapes: bare, `list/set/frozenset/tuple/
dict[...]`, `Iterable/Iterator[...]` (nesting included), `Optional[...]`,
`*args: KT` (elementwise). Everything else refuses **at wrap time**
(`UnsupportedSpecShape`) — including roles inside `Callable[[KT], ...]`
(contravariant positions) and `**kwargs: VT` (no annotation channel for keyword
names). Roles are matched **by TypeVar name, not identity** — deliberate: identity
matching would silently classify a user's same-named `KT` as "not a key", the
exact silent hole the mechanism exists to kill.

### Dict form (no typing required)

```python
spec = {
    "__getitem__": {0: "KT", "return": "VT"},
    "__setitem__": {0: "KT", 1: "VT"},
}  # int keys = positional index
```

Same compiled algebra, same loudness rules.

### Three refusal layers (loudness)

The failure mode this design targets is **silence-by-omission**, so omission is
loud at every level the mechanism can see:

1. **Undeclared public leaf attributes** — default policy `undeclared='exclude'`:
   the wrap succeeds, and every *use* of an undeclared attribute raises
   `UndeclaredAttributeError` with guidance (refusal at the moment of danger).
   `'raise'` is strict mode (wrap-time refusal naming the attributes);
   `'passthrough'` / `passthrough={...}` forward verbatim, explicitly.
2. **Unannotated parameters in spec'd methods** — `UnderAnnotatedSpecError` at
   compile time (an unannotated key parameter would silently receive outer keys).
3. **Unsupported shapes** — `UnsupportedSpecShape` at compile time: refuse rather
   than guess (properties/classmethods in specs included).

## The proxy carrier

A generated class per `(leaf type, spec, surface)` — cached for source-backed
(Protocol) specs; dict-form specs build a fresh class per wrap (id-keyed caching
was rejected over GC id-reuse collisions). The class namespace
contains **exactly the spec'd methods the leaf actually has** — capability
mirroring (a leaf without `__len__` yields a proxy without `__len__`) — and
nothing else:

- A dunder outside the spec does not exist on the proxy — `proxy | other` on a
  dict-leaf wrap raises `TypeError` instead of silently returning raw inner data
  (today's class-wrap `DelegatedAttribute`s leak `__or__`/`copy`/`fromkeys`).
- Explicit dunder access via `__getattr__` raises plain `AttributeError` — a
  forwarded raw `leaf.__contains__` would answer in the wrong key domain.
- A `__getitem__`-only spec gets a poisoned `__iter__` that raises `TypeError`,
  blocking CPython's legacy sequence protocol from pushing integer keys through
  the key codec.
- When the surface covers `__getitem__` + `__iter__`, an injected `__eq__`
  compares **outer views** (a wrap equals a dict holding its outer items) and
  `__hash__` is `None`. A `__getitem__`-only wrap keeps identity equality.
- Single-underscore attribute access is forwarded to the leaf; proxy-own state
  lives under `_self_*` names (wrapt's lesson).

## The codec laws (the invariant's fine print)

The headline invariant — *a method correct on the bare leaf stays correct under
any codec stack* — holds **iff**:

1. **The key decoder is total and injective on the leaf's actually-occurring
   keys.** dol's own `prefixed`/`suffixed` codecs violate this on out-of-band keys
   (decoding `'z.txt'` under a `'data/'` prefix codec). Filtering first —
   `Pipe(filt_iter.prefixes(p), KeyCodecs.prefixed(p))` — remains the blessed
   guard, and note that composition is a *mixed* stack (the filter is not a codec
   layer).
2. **Encoder and decoder are mutual inverses on both domains.** Under a one-sided
   codec, a method that returns its own key argument returns a *different* key.
   The optional `decoded_type`/`encoded_type` tags are the enforcement hook
   growing toward these laws.
3. **Container-shaped role arguments are copied at the boundary** (encode builds a
   new list/dict/tuple), so a leaf method that *mutates* its argument in place
   loses that side channel — rare, real, documented.

Lazy iterator decoding adds a temporal caveat: a data-dependent decode failure
raises at *consumption* time, arbitrarily far from the call — loud but late.

## Scope limits (read before extrapolating)

- **Pure-codec stacks only.** The flat vocabulary today represents *codecs*: pure,
  invertible, per-role transforms. `filt_iter` (changes the key **set**) and
  `cached_keys` (stateful) are **not** codecs — mixed compositions still nest, and
  the flat-model guarantees (inverse mapping, `__wrapped__` = raw backend, pickle
  uniformity) are scoped to the codec layers. Extending the layer vocabulary is a
  roadmap item, not a given ([dol_roadmap.md](dol_roadmap.md)).
- **Legacy `Store` = opaque leaf.** `interface_wrap` over a legacy `dol` Store
  works but warns: the Store (and its `.store` chain) is treated as an opaque
  leaf; guarantees apply only to the layers above it.
- **Instances only.** Class-wrapping raises `TypeError` (future work). Inherited
  Protocol methods are skipped (TypeVar substitution is future work). No
  `postget`/`preset` (key-aware value transforms) yet.
- **Populations.** Method bodies split three ways, and the flat model serves one:
  - *Leaf-domain bodies* (adapter methods that talk to the backend: `url_for`,
    `replace`, `delete_many`) — **the flat model's population**; boundary
    transformation is the only mechanism in the #83 option space that serves it.
  - *Outer-domain bodies* (extension methods over the wrapped view) — served by
    `wrapped_self` today, is-a wrapping later (#18). The flat model deliberately
    does not touch them.
  - *View-blind bodies* (no key in the signature, semantics still depend on the
    outer view, e.g. a `sync_to`) — no argument/return mechanism can serve these;
    design pressure says have fewer of them.
- **The sibling-store decision rule**: *spec expressibility is the line.* A
  capability whose shape the spec can express may be a method **iff spec'd**; a
  shape the spec refuses (key-space-shifting returns, cross-keyspace values, keys
  embedded in returned records, view-blind operations) must remain a sibling
  store, handle, or free function.

## The six decisions (Q0–Q5, maintainer, 2026-08-10)

Recorded in [dol_issue86_design.md §11](dol_issue86_design.md); implemented in
PR [#89](https://github.com/i2mint/dol/pull/89) where code was called for:

| # | Question | Decision |
|---|---|---|
| Q0 | Who owns the `wrap_kvs` endgame? | **Split synthesis**: codec/instance wrapping compiles to the flat engine; `@wrap_kvs` class-decoration becomes is-a. Each mechanism owns the population it is uniquely correct for. (Design work: P2/P3.) |
| Q1 | Public surface | **Private + facade**: engine stays private; built-in `MappingInterface` spec + `kv_interface_wrap` facade ship. Export reconsidered after P1 adapter trials. |
| Q2 | Loudness default | **`'exclude'`** (was `'raise'`): wrap succeeds, undeclared *use* raises with guidance. |
| Q3 | Typed codecs | **Yes, now**: optional `decoded_type`/`encoded_type` tags + seam validation. Tagging `dol.trans.Codec` is P2 (dependents gate). |
| Q4 | eq/hash | **Outer-view `__eq__`, no `__hash__`** when the surface covers traversal. `Store`'s own eq/hash incoherence queued for 0.4. |
| Q5 | `__class__` transparency | **Off today**; opt-in later, co-designed with #5. |

## Is `Store` obsolete? No — three senses of no

1. **Today**: `Store`/`wrap_kvs` are the only public, shipping mechanism. The flat
   engine is private and used by nothing else in dol.
2. **At the decided endgame** (Q0): `wrap_kvs`'s codec/instance semantics compile
   to the flat engine, but class decoration becomes is-a — `Store`'s role changes
   rather than disappears. A compatibility appendix (what `.store` returns — dol
   core reads it in ≥51 places — the #6 signature graft, the `wrapped_self`
   backref) must be answered before any wiring.
3. **Permanently**: filters and caches are not codecs; unless the layer vocabulary
   is extended (P3 / roadmap), `Store`-style nesting survives for them even in the
   endgame.

## Using it today

```python
from dol._interface_wrap import interface_wrap, kv_interface_wrap, Codec

# The wrap_kvs-shaped facade (built-in Mapping spec):
s = kv_interface_wrap(
    {},
    id_of_key=lambda k: k + ".json",  # facade keeps wrap_kvs vocabulary...
    key_of_id=lambda k: k[:-5],
    data_of_obj=str,
    obj_of_data=int,
)
s["a"] = 1  # leaf now holds {'a.json': '1'}
assert list(s) == ["a"] and s == {"a": 1}

# The general gesture (any spec, any roles):
s = interface_wrap(
    leaf,
    spec=BucketInterface,
    codecs=dict(KT=Codec(encoder=..., decoder=...), VT=Codec(encoder=..., decoder=...)),
    passthrough={"wire"},
)
```

Caveats: `kv_interface_wrap` transforms are **plain unary callables** — no
`wrap_kvs`-style `f(self, x)` signature inference, no `FirstArgIsMapping`. The
Mapping *mixin* methods (`get`, `keys`, `items`, `update`, …) are deliberately not
in the built-in spec — under the default policy they're hidden-and-loud.

## Code map

All in `dol/_interface_wrap.py` (private):

| Identifier | Role |
|---|---|
| `Codec` | frozen dataclass: encoder/decoder + optional type tags |
| `InterfaceSpec` | compiled spec; `.from_annotated(Protocol)`, `.from_dict(...)` |
| `_find_role_sites` / `_transformer_for_path` | locate roles in annotations; build structure-mapping callables |
| `_fuse` / `_fused_role_funcs` | flatten-and-compile the stack into per-role pipelines |
| `_validate_stack_seams` | typed-codec seam validation (Q3) |
| `_compile_method_plan` | bake one boundary plan: encode args → call leaf → decode return |
| `_outer_signature` | the spec's signature is the outer contract |
| `InterfaceProxy` | carrier base: `__wrapped__`, `_encode_role`, `_decode_role`, `__reduce__`, `__getattr__` policy |
| `_build_proxy_class` | generated class per (leaf type, spec, surface); cached for source-backed specs only |
| `interface_wrap` | the entry point: normalize spec/stack, validate, compile, assemble |
| `MappingInterface` / `kv_interface_wrap` | built-in Mapping spec + `wrap_kvs`-shaped facade (Q1) |
| `InterfaceWrapError`, `UnsupportedSpecShape`, `UnderAnnotatedSpecError`, `UndeclaredAttributeError` | the loudness surface |

## History and companions

- Discussion [#90](https://github.com/i2mint/dol/discussions/90) — the complete
  narrative walkthrough of the #86/Option G cycle.
- [dol_issue86_design.md](dol_issue86_design.md) — the design study + decision
  record (adversarial-panel evidence, verification log, migration P0–P3).
- [dol_issue83_design.md](dol_issue83_design.md) — the census + options A–F +
  carry-forward list this design answers.
- [dol_issue18_design.md](dol_issue18_design.md) — the other half (outer-domain
  bodies): `wrapped_self` now, is-a later.
- [dol_roadmap.md](dol_roadmap.md) — where this goes next.
