# dol Discussion #86 — Option G: spec-carried boundary codecs on a flat proxy

> Companion to [dol_issue83_design.md](dol_issue83_design.md) (options A–F) and
> [dol_issue18_design.md](dol_issue18_design.md) (the is-a plan). Responds to the
> maintainer's proposal (relayed in-session, 2026-08-10; summarized in §1 and posted to
> discussion #86): wrapt-inspired object proxies + KT/VT-annotated interface specs +
> flatten-and-compile codec stacks. Prototype: `dol/_interface_wrap.py` (private,
> additive, stdlib-only) + `dol/tests/test_interface_wrap.py`, branch
> `claude/option-g-interface-codecs`. Every claim below marked **[verified]** has
> running-code evidence (§12); this doc survived one adversarial panel round (4 lenses),
> which broke several first-draft claims — the corrections are folded in and flagged.

## TL;DR

The proposal decomposes into three mechanisms. Two survive adversarial review with
stated preconditions; the third inverts into a set of lessons:

1. **A typed interface spec** — per method, *where* the types-of-interest (KT, VT, any
   TypeVar "role") occur in arguments and returns, compiled from annotations (or an
   explicit dict form) at wrap time. This is the "wrapper must be able to express *this
   method takes a key*" capability that #83 §5.2 demands, and it **replaces** the
   verified-broken `ingoing_key_methods`/`outcoming_key_methods`. **Sound, prototyped,
   with three refusal layers making omission loud** (§2.3). Its honest weak edge: spec
   authoring is real work, and loudness degrades one keyword at a time (§7).
2. **A flat codec stack, compiled once** (the "two lists" idea). The structurally
   load-bearing part: re-wrapping *extends a list* instead of *nesting an object*, so —
   **within pure-codec stacks** — the chain-walking family dissolves: `inner_most_key`
   becomes a total fold, the missing **inverse** key mapping (#83 §5.4) exists by
   construction, the leaf is a strong structural reference (#83 §5.6), and the measured
   420–510 ns/layer delegation tax collapses to one boundary hop (**3.6× faster at
   depth 6, [verified]**). The scope limit is real and must be said plainly: `filt_iter`
   (key-*set* change) and `cached_keys` (stateful source) are **not codecs**, so mixed
   compositions still nest (§10).
3. **A wrapt-style transparent proxy carrier — take the lessons, not the goal.**
   Running wrapt 2.3.0 **[verified]**: its proxy has *exactly* dol's #18 hole (inside a
   wrapped method, `self` is the wrapped object — the proxy intercepts only the first
   hop); pickling refuses loudly by default; the `__iter__` mistake is a ten-year
   cautionary tale about eagerly-defined dunders poisoning duck typing; and on modern
   CPython the pure-Python proxy measured *faster* than the C extension (83 vs 445 ns
   attr reads) — a dependency-free carrier costs nothing. Universal transparency is a
   tar pit; the prototype exposes **only the spec'd surface** and makes everything else
   loud (§2.4).

**The headline finding** (new evidence, corrects an overclaim in #86's own option F
verdict): **is-a wrapping does not fix #83 for backend-direct method bodies.**
A method like `cosmodol.replace` or `pydrivedol.get_url` that passes its key straight
to the backend still receives the outer key under is-a — hooks on the MRO don't help a
body that never routes through them **[verified, panel probe p4]**. Boundary
transformation is the only mechanism in the A–G space that serves that population, and
it serves it with a strong invariant: **a method correct on the bare leaf stays correct
under the stack** — *provided the codec laws hold* (§3.1). So Option G is not an
interim stand-in for is-a; the two serve disjoint populations (§8), and "both #83 and
#18 disappear under F" should be retired from the corpus.

## 1. The proposal, restated

The maintainer's proposal (2026-08-10, in-session): (a) study wrapt's object-proxy
design; (b) wrap incoming/outgoing keys and values in *all* methods of an object, not
just the Mapping dunders, with codecs hooked up automatically per a specification —
e.g. a Protocol class annotated with KT/VT; (c) generalize beyond keys/values to any
small set of "types of interest"; (d) accumulate wrapping layers in two lists (encoders,
decoders) and compile them for speed and validation; (e) take special care that a
method calling another method does not double-apply transforms.

Point (e) turned out to be the crux, and the answer is structural rather than careful:
apply codecs **only at the proxy boundary** and keep `self` inside method bodies bound
to the leaf. Then internal `self.x()` calls never cross the boundary, and transforms
apply exactly once **[verified: counting-encoder probe, 1 invocation through a spec'd
method that internally calls another spec'd method]**.

## 2. The mechanism (as prototyped)

A wrap is `(leaf, spec, stack)` — one proxy object, however many layers.

### 2.1 The spec

```python
KT, VT = TypeVar("KT"), TypeVar("VT")


class BucketInterface(Protocol[KT, VT]):
    def __getitem__(self, k: KT) -> VT: ...  # Mapping dunders are ordinary
    def __iter__(self) -> Iterator[KT]: ...  # spec entries — no privileged
    def __contains__(self, k: KT) -> bool: ...  # surface (#83 §5.2)
    def url_for(self, k: KT, *, expires_in: int = 3600) -> str: ...
    def delete_many(self, keys: Iterable[KT]) -> None: ...
    def items_page(self) -> Iterator[tuple[KT, VT]]: ...
```

Compilation walks `get_type_hints` + `signature` per method, recording the *paths* at
which role TypeVars occur. Supported shapes: bare, `list/set/frozenset/tuple/dict[...]`,
`Iterable/Iterator[...]` (lazily mapped, nesting included), `Optional[...]`,
`*args: KT` (elementwise). Everything else **refuses at wrap time**
(`UnsupportedSpecShape`) — including roles inside `Callable[[KT], …]` (contravariant
positions would hand inner keys to outer callbacks) and `**kwargs: VT` (keyword names
as keys have no annotation channel). Roles are matched **by TypeVar name, not
identity** — deliberate: dol itself ships two distinct `KT` objects (`dol.KT` is
`typing.KT`; `dol.caching.KT` is its own) **[verified]**, so identity matching would
silently classify a user's same-named `KT` as "not a key" — the exact silent hole the
mechanism exists to kill. The residual (two *different* roles sharing a name) is the
user's naming responsibility.

The **spec's signature is the outer contract**: calls bind against it, so the leaf's
own parameter names are irrelevant (`dict` calls its key `key`; the spec may say `k`).
A dict form exists for annotation-free use — `{'__getitem__': {0: 'KT', 'return':
'VT'}}`, integer keys = positional index — same compiled algebra, same loudness rules.
(Yes, this is #14's rejected Option-B notation as a *fallback input format*; the
difference from B is everything in §2.3 and §7.)

### 2.2 The stack

`stack` is a tuple of layers, each `{role: Codec(encoder, decoder)}`. Wrapping an
already-wrapped proxy builds a **new** proxy whose stack is a copied-and-extended tuple
over the **same leaf** — never a wrapper-of-wrapper, and never mutation: an in-flight
lazy iterator keeps the fused pipelines it was compiled with **[verified]**. Per role,
encoders fuse outer→inner and decoders inner→outer into single callables; per method, a
plan binds parameters and return paths to them (fast path for the dunder-shaped common
case; `Signature.bind` for the general case).

What this buys, each previously a named open problem — **scoped to pure-codec stacks**:

- `inner_most_key(w, k)` ≡ `w._encode_role('KT', k)` — total, no `.store` walk, no
  non-`Store`-layer hazard, because there *are* no layers at runtime.
- The **inverse mapping** (#83 §5.4 — needed by anything returning keys: `prefixes`,
  `walk`, listings) is `w._decode_role('KT', k)`. It exists by construction; today it
  does not exist at all.
- The leaf is a strong structural reference, `w.__wrapped__` (#83 §5.6) — no weakref
  registry for the innermost direction. **Correction from the panel**: this is *not*
  the #16/#10 write-back boundary. That engine needs the raw-*scalar* surface — leaf
  *plus the value-codec slice* of the stack, below only path/view layers **[verified:
  `path_set_writeback` against the raw leaf raises `PathCreationError`; against the
  value-codec surface it works]**. A flat design can provide that as a *stack slice*
  (a derived proxy over the same leaf with a stack prefix) — future work, and the
  honest statement is that the outermost/innermost directionality clash of #83 §5
  becomes two accessors *plus a slicing operation*, not two accessors.
- Per-op cost is one boundary call at any depth: **[measured]** getitem 675 ns at 6
  layers vs 2426 ns nested (and 309 vs 383 ns at depth 1); iteration 206 µs vs 700 µs
  per 1000 keys at depth 6.

### 2.3 Loudness — three refusal layers

The census's failure mode is silence-by-omission (ADR-0011 D5), so omission is loud at
every level the mechanism can see:

1. **Undeclared public attributes** of the leaf: under the default
   `undeclared='exclude'` (decision 2026-08-10, see §11), the wrap succeeds and
   every *use* of an undeclared attribute raises with guidance — refusal at the
   moment of danger, no habit-forming escape. `'raise'` is the strict mode
   (wrap-time refusal naming the attributes); `'passthrough'` /
   `passthrough={...}` forward verbatim, explicitly.
2. **Unannotated parameters inside spec'd methods** (`UnderAnnotatedSpecError`):
   a spec author who writes `url_for(self, k)` forgot the annotation, and `k` would
   silently receive outer keys — refused at compile time. (Panel-found hole, closed.)
3. **Unsupported shapes** (`UnsupportedSpecShape`): refuse rather than guess (#83
   §5.7) — including properties/classmethods in specs, which would otherwise vanish
   silently.

Two honest caveats the panel established. First, **the raise default is noisy on real
leaves** — a bare `dict` drags 11 MutableMapping publics; `dol.Files` ~20; a boto3
client 122 — so users will reach for `'passthrough'`, and after that, omission is
silent again. The guard therefore needs **two desks**: wrap-time loudness at the
user's desk (this mechanism), plus an adapter-side reflective conformance test at the
author's desk (ADR-0011 D5's shape — unchanged and still recommended). Second,
**parameterless under-declaration survives**: `keys: Iterable` (author forgot `[KT]`)
is indistinguishable from a deliberately role-free iterable. Layer 2 catches the
*unannotated* case; the *under-parameterized* case remains a review-time judgement.

### 2.4 The carrier — and what it refuses to be

A generated class per `(leaf type, spec, surface)`, cached; instances hold
`(leaf, spec, stack, compiled plans)` under `_self_`-prefixed names (wrapt's lesson).
The class namespace contains **exactly the spec'd methods the leaf actually has** —
capability mirroring (a leaf without `__len__` yields a proxy without `__len__`;
contrast `_filt_iter`'s verified `__len__`-resurrection bug), and **nothing else**:

- A dunder outside the spec does not exist on the proxy — `proxy | other` on a
  dict-leaf wrap raises `TypeError` **[verified]** instead of silently returning raw
  inner data the way today's class-wrap `DelegatedAttribute`s do (`__or__`/`copy`/
  `fromkeys` leak, basepy-verified). Loud beats transparent.
- `__eq__`/`__hash__` are **deliberately not defined** in the prototype (object
  identity semantics). The panel is right that no member of {eq, hash,
  len-under-filtering} is derivable from role mapping; each is a policy decision.
  Today's Store is itself incoherent here (eq compares outer views, hash hashes the
  inner store) — the redesign should decide this family *explicitly* rather than
  inherit the incoherence. Open question §11.4.
- `__class__` transparency (isinstance-as-leaf) is **not** implemented: wrapt shows
  it's feasible, but it changes capability detection and interacts with #5 (wrapper-
  class control) — a policy to co-design, not a default to sneak in.
- Pickling: `__reduce__` → `(rebuild, (leaf, spec_source, stack, policy))` — no
  dynamic class in the payload; the rebuild recompiles. **Honest scope [panel-
  corrected]**: this repairs *by construction* the anonymous-intermediate-class
  failure (today's stacked wraps, matrix case g) and keeps instance wraps working;
  it does **not** repair the `Files`/decorator-form failures, whose root cause is the
  *leaf's own class* being shadowed and by-name unreachable — that needs the name-
  shadowing fix in dol itself, and `Files` is route-2 (`mk_relative_path_store`)
  anyway, untouched by any wrap_kvs-side change. Constraints: codecs and spec source
  must be module-level/picklable; lambdas fail loudly **[verified]**.

### 2.5 The codec laws (the invariant's fine print)

"A method correct on the bare leaf stays correct under any codec stack" was **refuted
as stated** by the panel and is hereby restated with its preconditions. It holds iff:

1. **The key decoder is total and injective on the leaf's actually-occurring keys.**
   dol's own `prefixed`/`suffixed` codecs violate this on out-of-band keys (decode
   `'z.txt'` → `''` — ADR-0006's corruption family), and the flat model inherits that
   corruption exactly as today's model does. Filtering first (ADR-0006's
   `Pipe(filt_iter.prefixes(p), KeyCodecs.prefixed(p))`) remains the blessed guard —
   and filters are not codec layers, so this composition is a mixed stack (§10).
2. **Encoder and decoder are mutual inverses on both domains.** dol permits one-sided
   key transforms today (`kv_wrap.outcoming_keys`); under a one-sided codec a method
   that returns its own key argument returns a *different* key. Typed/total codecs
   (the two in-tree TODOs asking for inspectable codec types) graduate from
   nice-to-have to the enforcement hook for these laws — promoted to §11.3.
3. **Container-shaped role arguments are copied at the boundary** (encode = build a
   new list/dict/tuple), so a leaf method that *mutates* its argument in place loses
   that side channel silently. Rare, real, now documented.

Lazy iterator mapping adds a temporal caveat: a data-dependent decode failure raises
at *consumption* time, arbitrarily far from the call — loud but late. An eager/strict
option is cheap future work.

## 3. What the panel verified as HOLDING

- **No-double-apply is structural** — spec'd method calling another spec'd method via
  `self`: encoder fires exactly once (counting probe).
- **The s3dol composition** — a leaf owning its own prefix arithmetic (`url_for`
  routing through `_id_of_key`) under a boundary key codec: byte-identical to ground
  truth, at one and two stacked layers; the unmapped-key counterfactual reproduces
  #83, the rebind counterfactual double-applies.
- **P0 additivity** — nothing in the prototype touches existing modules.
- **Layer metadata half-plumbed** — every Wrap class already carries `_class_trans`
  through `__reduce__`; a layer-list accumulator can ride that channel in P2.
- **G survives is-a landing** (§8) — the one steelman that failed to kill it.

## 4. What it does not cover (complete inventory)

From the s3dol shape census, the shapes the spec vocabulary cannot express — these
**must remain sibling stores, handles, or free functions** (see the decision rule,
§7): key-space-shifting returns (`sub(prefix)`, `mkdir` returning a sub-store, the
trailing-slash `__getitem__` overload) — this is #10's territory; cross-keyspace
values (`S3ClientDol.__setitem__(bucket, mapping-in-another-keyspace)`); keys embedded
in returned records (`info()` → ObjectInfo carrying its key) and in exception payloads
(`S3PartialFailure.succeeded/.failures` — laziness compounds this: the raise site has
no boundary frame); keys nested at data paths (`object_list_pages()` →
`page['Contents'][i]['Key']`); untyped operation batches (`cosmodol.batch`);
view-dependent no-arg operations (`sshdol.sync_to` — no key argument to transform,
yet it violates a filtered outer view's contract); replace-the-write strategies
(multipart uploads); filter/prefix pushdown (#24's fast-op hooks — the spec is a
natural registry for them later, but that is co-design, not coverage). Also not in
the prototype: `postget`/`preset` (key-aware value transforms), `__missing__`
routing (which has *two* paths with different key domains — leaf-native dict
`__missing__` sees inner keys, Store-level routing sees outer keys **[verified]**),
class-wrapping, and inherited-Protocol TypeVar substitution.

## 5. The population map (refined from the draft's dichotomy)

Method bodies split three ways, not two **[panel-corrected]**:

- **Leaf-domain bodies** (adapters: `url_for`, `replace`, `delete_many`, and mixed
  bodies like `sshdol.mkdir` that combine raw-backend calls with `self[...]` — these
  stay coherently leaf-domain): **Option G's population.** Boundary transformation is
  correct for them under the §2.5 laws.
- **Outer-domain bodies** (user extension methods over the wrapped view — every
  `wrapped_self` site in the census: `xdol`, `unbox`, `lexis`; all keyless):
  `wrapped_self` today, is-a later. G deliberately does not touch them.
- **View-blind bodies** (`sync_to`: no keys in the signature, semantics still depend
  on the outer view): **no arg/return mechanism can serve these.** They are the
  strongest surviving argument for #86 §4's rung 1 — have fewer such methods — and
  for sibling stores.

## 6. The rebind family, re-examined honestly

The draft claimed the flat carrier voids two of the three rebind-rejection premises.
The panel refuted both sub-claims **[probes p2, p3]**, and the corrected statement is:
under a flat proxy, rebinding `self` to the proxy makes internal leaf→leaf self-calls
**re-cross the boundary → double-encode** (replacing the innermost-binding defect with
a violation of G's own headline invariant); write-through `__setattr__` *relocates*
state divergence (writes fail on dict/slotted leaves; two proxies alias state through
one leaf; wrapper-domain writes land where leaf methods read them). Reason 2
(`super()`/descriptor TypeErrors in bodies) was never in doubt. **The rejection
stands, on stronger grounds than the draft gave it.** Do not re-propose.

## 7. Placement in the option space — answering B's verdict

G's spec core **is Option B, replaced rather than built on** — exactly what ADR-0011
D5 predicted a real mechanism would be. The #86 verdict on B was "the guard is worth
more than the registry", and G's answer is direct: **in G the guard is the
mechanism** — three refusal layers at wrap/compile time (§2.3) instead of a test the
author must remember to write — and the registry is *derived* (from annotations)
rather than *maintained*. What B's verdict got right and G keeps: the wrap-time guard
serves the wrong desk alone; the adapter-side reflective conformance test stays.

The **sibling-store decision rule** (new, resolves the two-blessed-patterns tension):
*spec expressibility is the line.* A capability whose shape the spec can express
(scalar KT, Iterable[KT], tuple[KT, VT], …) MAY be a method iff spec'd; a shape the
spec refuses (§4's inventory) MUST be a sibling store, handle, or free function. This
preserves #86 §4's rung 1 incentive — every keyed method is a declared, reviewed
liability — while giving the ones that earn their place a correctness mechanism.

During P0–P1 nothing protects users of the *legacy* wrap path: a spec is honored only
by the new engine, so `cosmodol.replace` under today's `KeyCodecs` stays exactly as
broken as the census found it. A cheap P1 mitigation worth considering: teach
`wrap_kvs` to *warn* when key-wrapping a store whose class declares an interface spec.

## 8. Relation to is-a (option F) — and open question 0

New evidence for the F conversation: (a) **is-a does not fix backend-direct keyed
bodies** (§TL;DR) — F's "both #83 and #18 disappear" holds only for bodies that route
through `self[...]`/hooks; (b) **hook-name collision**: a leaf that owns `_id_of_key`
(s3dol's `S3BucketReader`, base.py:201-shape) gets its prefix arithmetic *shadowed* by
a naive is-a hook → `KeyError` **[probe p4]** — a constraint on F's Phase-3 mixin
design that neither prior doc records. So F and G are complements: F serves
outer-domain bodies (#18), G serves leaf-domain bodies (#83), and both leave §5's
third population to design pressure (fewer keyed methods).

This raises **open question 0**, which the staged plans currently answer differently:
*what does `wrap_kvs` compile to at the endgame — an is-a subclass (#18 doc, Phase 3)
or a flat proxy (this doc, P2)?* They contradict; the maintainer should own the call.
A plausible synthesis: `wrap_kvs`'s *codec semantics* compile to the flat engine, while
*class-decorator* usage (the #18 population's home) gets is-a — but that is a proposal,
not a decision.

## 9. Migration (rescoped after the panel)

- **P0 (this branch)**: the private prototype + tests. Additive; no exports; no
  behavior change anywhere.
- **P1**: census-family adapters (`cosmodol`, `pydrivedol`, `sshdol`…) adopt
  `interface_wrap` for spec-expressible capability surfaces; sibling stores per the
  §7 rule. Add the adapter-side conformance helper. Consider the legacy-path warning.
- **P2 — rescoped**: a wrap_kvs facade over the engine is possible **only for
  flat-equivalent configurations** — stacks where no inner layer *observes its
  domain*. A layer-i `wants_self` transform receives the layer-i wrapper today
  **[verified]**; a layer-i `postget` sees intermediate keys; both are observable API
  a fused pipeline never materializes. The facade must detect these and fall back to
  nesting, loudly. Route 2 (`mk_relative_path_store` → `Files`) is **not** wrap_kvs
  and migrates separately or not at all — say so wherever "two routes unified" is
  claimed. A P2 compatibility appendix must answer, before code: does the product
  subclass `Store`; what does `.store` return (≥51 lines in dol core read it); is the
  #6 signature graft kept; does the engine register the `wrapped_self` backref
  (today's blessed #18 fix silently degrades to identity on engine-built wraps
  otherwise **[verified]**).
- **P3**: answer open question 0 with #5 and #10 at the table; extend the layer
  vocabulary (a filter-layer kind for `filt_iter`; `cached_keys` decided separately)
  or permanently scope the flat model to codec stacks.

## 10. Mixed stacks (old/new) — the standing rule

Nothing prevents `KeyCodecs.prefixed('x')(g_proxy)` (legacy Store over a G proxy) or
`interface_wrap(legacy_store, …)` today, and in both directions the flat-model
guarantees silently degrade (`inner_most_key` under-resolves; `__wrapped__` is a
wrapper, not the backend) **[probe p1]**. The prototype's policy: wrapping a legacy
`Store` **warns** and treats it as an opaque leaf; the guarantees are explicitly
scoped to the layers above it. The eventual policy (absorb known layer types? refuse?)
belongs with P2's compatibility appendix.

## 11. Open questions — DECIDED (maintainer, 2026-08-10)

0. **Who owns the wrap_kvs endgame?** → **Split synthesis**: codec/instance wrapping
   compiles to the flat engine; `@wrap_kvs` class-decoration becomes is-a — each
   mechanism serves the population it is uniquely correct for (§8). Consequence: P2
   and the #18 doc's Phase 3 are no longer rivals; the crisp fire-when rule is P2/P3
   design work.
1. **Public surface** → **Private + facade next**: the engine stays private; the
   built-in `MappingInterface` spec and the `wrap_kvs`-shaped `kv_interface_wrap`
   facade ship (done — this decision round), so the simple gesture never regresses.
   Export considered after adapters (P1) validate it. The Mapping *mixin* methods
   (`get`, `keys`, `update`, …) stay out of the built-in spec — each needs its own
   vocabulary decision — and are hidden-loud under the default policy.
2. **Loudness default** → **`'exclude'`**: wrap succeeds; undeclared *use* raises
   with guidance. `'raise'` remains as strict mode. Rationale: wrap-time raise on
   real leaves (dict: 11 publics; boto3: 122) drives users to `'passthrough'`,
   after which omission is silent again.
3. **Typed codecs** → **Yes, now**: `Codec` carries optional
   `decoded_type`/`encoded_type` tags; stack compilation validates adjacent seams
   (outer `encoded_type` ≡ inner `decoded_type`) and refuses loudly on mismatch;
   untagged stays unchecked (done — this decision round). Tagging `dol.trans.Codec`
   is P2 territory (dependents gate).
4. **eq/hash/len** → **Outer-view eq, no hash**: when the spec'd surface covers
   `__getitem__`+`__iter__`, `__eq__` compares outer views (a wrap equals a dict
   holding its outer items) and `__hash__` is None (done — this decision round).
   Store's own eq/hash incoherence is queued for 0.4.
5. **`__class__` transparency** → **Opt-in later, co-designed with #5**. Off today;
   nothing lies by default.

## 12. Verification log

| # | Claim | How verified |
|---|---|---|
| 1 | wrapt proxy has the #18 hole; pickle refuses; pure-Python faster than C ext on 3.12 | wrapt 2.3.0 installed, probes run |
| 2 | today: pickle fails for decorator-form/Files/stacked wraps, one root cause (name shadowing); instance + single class-wrap work | 7-case matrix, exact tracebacks |
| 3 | today: ~420–510 ns/layer getitem; 34–42× dict iteration at 1 layer; 2 delegation objects per instance wrap | timeit + object-graph probes |
| 4 | prototype: no-double-apply on internal self-calls | counting encoder, 1 invocation |
| 5 | prototype: s3dol prefix-owning leaf composes; 2-layer stack byte-identical | faithful simulation + counterfactuals |
| 6 | is-a does not fix backend-direct #83 bodies; leaf `_id_of_key` shadowed by naive is-a hook | panel probe p4 |
| 7 | rebinding on a flat proxy double-encodes internal self-calls; write-through relocates state divergence | panel probes p2, p3 |
| 8 | `__wrapped__` ≠ #16/#10 boundary (raw-scalar surface needed) | real `path_set_writeback` run both ways |
| 9 | boundary invariant fails without codec laws (out-of-band keys, one-sided codecs, arg mutation) | panel probe a, three shapes |
| 10 | flat 3.6×/3.4× faster than nested at depth 6 (getitem/iter) | benchmark, this branch |
| 11 | layer-1 `wants_self` transform receives the layer-1 wrapper (fusion not semantics-preserving) | panel probe A1 |
| 12 | engine wraps don't populate the `wrapped_self` registry | panel probe C2 |
| 13 | prototype hardening: `*keys: KT` elementwise, unannotated-param refusal, None-default rule, no raw-dunder leak, in-flight iterator safety, nested iterators, 3.10 parity | `dol/tests/test_interface_wrap.py` + 3.10 doctest run |
| 14 | independent code review round: 3.10 builtin-slot signature failure (positional plan fallback), dict-form spec pickle/copy crash, explicit-dunder escape (`s.__contains__` answering in the wrong key domain), sequence-protocol iteration leak, fast-path keyword calls, `Iterable`-arg one-shot downgrade, policy-typo silence, id()-keyed class-cache collisions — all fixed with regression tests (39 total) | independent reviewer probes + `dol/tests/test_interface_wrap.py`; accepted follow-ups: fast path does not enforce the outer signature's arity (generic path does); dict-form proxy classes are not cached |

## References

Issues/discussions: #86 (option space) · #83 (census + carry-forward list) · #18
(is-a plan, rebind rejection) · #10, #16 (boundary engine) · #5, #6 · #24 (fast-op
hooks) · s3dol#14, s3dol ADR-0011, ADR-0006. In-tree: `dol/_interface_wrap.py`,
`dol/tests/test_interface_wrap.py`, sibling docs in `misc/docs/`.
