# dol roadmap — the sequencing SSOT

> **Status: living document.** This is the single source of truth for *what happens
> in which order* across dol's redesign program. It supersedes the wave/tackle order
> of [dol_issues_report.md](dol_issues_report.md) §2 (kept as the 2026-07-02 triage
> snapshot — its close-list, dependency map, and verification log remain the
> evidence record). Design content lives in the linked docs; this file only
> sequences it. Last updated 2026-08-21.

## The program in one paragraph

dol's wrapping machinery is migrating from **has-a delegation** (nested wrapper
objects, the #18/#83 delegation traps) to a **two-mechanism endgame** decided on
2026-08-10 (decision Q0, [dol_issue86_design.md §11](dol_issue86_design.md)):
codec/instance wrapping compiles to the **flat boundary-codec engine**
([dol_flat_model.md](dol_flat_model.md)), and `@wrap_kvs` class-decoration becomes
**is-a** ([dol_issue18_design.md](dol_issue18_design.md)). In parallel: the paths /
boundary-write-back engine matures (#16 shipped, #10/#2 designed), and a
fleet-facing program (ledger → vocabulary migration → cruft audit) prepares the
ecosystem for the eventual breaking surface change. Discussion
[#90](https://github.com/i2mint/dol/discussions/90) is the complete narrative
walkthrough of how we got here.

## Track A — the flat engine (Option G)

| Phase | Status | What |
|---|---|---|
| **P0 — prototype** | **done** (v0.3.62–63, PRs [#88](https://github.com/i2mint/dol/pull/88)/[#89](https://github.com/i2mint/dol/pull/89)) | Private `dol/_interface_wrap.py` + ~45 tests; the six decisions Q0–Q5 implemented where code was called for (Q2 `exclude` default, Q3 typed-codec seams, Q4 outer-view eq, Q1 `MappingInterface` + `kv_interface_wrap` facade). |
| **P1 — adapter trials** | **in progress** | Census-family adapters trial `interface_wrap` on spec-expressible surfaces; sibling stores per the spec-expressibility rule. First trial (s3dol `url_for`) posted to [#86](https://github.com/i2mint/dol/discussions/86) on 2026-08-11. Remaining: more adapters, the adapter-side conformance helper, the legacy-path warning idea. Export of the engine is reconsidered only after P1 evidence (decision Q1). |
| **P2 — the wrap_kvs facade** | not started; **gated on the compatibility appendix** | A wrap_kvs facade over the engine for flat-equivalent configurations (falling back to nesting, loudly, where a layer observes its domain). Must answer first: does the product subclass `Store`; what does `.store` return (≥51 lines in dol core read it); is the #6 signature graft kept; does the engine register the `wrapped_self` backref. Also P2: tag `dol.trans.Codec` with the Q3 type tags (dependents test-gate). |
| **P3 — layer vocabulary** | intent recorded → **[#93](https://github.com/i2mint/dol/issues/93)** | Extend the flat stack beyond pure codecs: filter / cache / contextual-codec / interceptor layer kinds, and spec-registered fast-ops ([#24](https://github.com/i2mint/dol/discussions/24), [#56](https://github.com/i2mint/dol/issues/56)). See Track D interplay: the layer-kind design should inherit the new vocabulary from #92. |

## Track B — is-a class decoration

The other half of the Q0 split synthesis: `@wrap_kvs`-as-class-decorator becomes
is-a wrapping, absorbing the #6 (signature freeze) and #5 (wrapper-class control)
concerns. Design: [dol_issue18_design.md](dol_issue18_design.md) Phase 3, with two
constraints discovered since: is-a does **not** fix #83's backend-direct bodies
(disjoint populations — that's Track A's job), and a naive is-a hook shadows a
leaf that owns `_id_of_key` (verified, [dol_issue86_design.md §8](dol_issue86_design.md)).
Not scheduled before P2 lands; the crisp "which mechanism fires when" rule is
P2/P3 design work. Open issues folded in here: [#18](https://github.com/i2mint/dol/issues/18),
[#5](https://github.com/i2mint/dol/issues/5).

## Track C — the paths / boundary-write-back engine

- **#16 — shipped** (v0.3.55): `create_missing`/`autoviv`, the `path_set_writeback`
  boundary engine in `dol/_paths_core.py`. Design record:
  [dol_issue16_design.md](dol_issue16_design.md).
- **#10 + #2 — designed, code pending**: recursive wrapping (`recursive_wrap`) + a
  flat `KvReader`/`KvPersister` view (`flat_store`) sharing one descent frontier
  and reusing the #16 engine. Design: [dol_issue10_design.md](dol_issue10_design.md).
  Persistent store-of-stores creation is deferred inside that design (its P3).

## Track D — the fleet program (intent recorded 2026-08-21)

Maintainer intents, recorded as issues; sequencing is **ledger first** — both
other legs consume it.

1. **[#91 — fleet usage ledger](https://github.com/i2mint/dol/issues/91)**: grow
   the local import-census (85 direct dependents, 2026-08-03 scan) into a
   versioned ledger: full commit pins + baseline test results + restore helper;
   AST-level subclass and kwarg census; zero-usage report; successive scans.
2. **[#92 — vocabulary horizon](https://github.com/i2mint/dol/issues/92)**:
   `{kind}_{encoder|decoder|codec}` replaces the `X_of_Y` language (and
   `kv_wrap`'s `outcoming_*`/`ingoing_*` — three vocabularies to reconcile).
   Strategy: successor surface, not rename-in-place; `wrap_kvs` stays as a
   back-compat facade over it. Execution gated on P1 evidence + P2 facade + #91.
   New surfaces adopt the target vocabulary from day one (the flat engine's
   `Codec(encoder, decoder)` already does).
3. **[#94 — cruft audit](https://github.com/i2mint/dol/issues/94)**: relocate
   fleet-unused, low-value members (→ xdol / tests / recipes). Current scan:
   86/147 `__init__` exports with zero detected fleet usage (61 public >1 year).
   Gated on #91's authoritative census (star-imports + submodule-only compat
   surface). Complementary to [#70](https://github.com/i2mint/dol/issues/70).

## Track E — hygiene and remaining triage

- Open issues not yet claimed by a track: [#69](https://github.com/i2mint/dol/issues/69)
  (path-get/flatten dedup), [#70](https://github.com/i2mint/dol/issues/70) (module
  splitting), [#15](https://github.com/i2mint/dol/issues/15) (AttrContainer tab
  completion), [#2](https://github.com/i2mint/dol/issues/2) (kv_walk docs),
  [#1](https://github.com/i2mint/dol/issues/1) (documentation ideas).
- **Untriaged** (filed after the 2026-07-02 triage): [#80](https://github.com/i2mint/dol/issues/80)
  (record/metadata DataProvider layer — see
  [dol_content_metadata_bifurcation.md](dol_content_metadata_bifurcation.md)),
  [#82](https://github.com/i2mint/dol/issues/82) (prefix-relativization boundary bug).
- Repo cleanup notes: the remote branch refs for merged PRs #88/#89 and the
  2026-07-11 `stash/*` branches (full-tree WIP snapshots predating the whole
  #83/#86 cycle) are stale and can be pruned after a quick skim for salvage.

## Dependency sketch

```
P0 ──► P1 ──► P2 ──► P3 (#93)
              │        ▲
              │   (vocabulary from #92 informs layer-kind naming)
              ▼        │
        Track B (is-a) │
                       │
#91 (ledger) ──► #92 (vocabulary execution) ──► fleet migration
     └─────────► #94 (cruft audit)
Track C (#10/#2 implementation) — independent; reuses the #16 engine
```

## Decided inputs (do not re-litigate; re-open only with new evidence)

Q0 split synthesis · Q1 private + facade · Q2 `exclude` loudness default ·
Q3 typed codecs now · Q4 outer-view eq, no hash · Q5 no `__class__` transparency.
Full record: [dol_issue86_design.md §11](dol_issue86_design.md); narrative:
[#90](https://github.com/i2mint/dol/discussions/90). Also standing: the rebind
family is rejected (twice, on strengthened grounds — §6 of the same doc);
`clear()` stays disabled on `KvPersister`.
