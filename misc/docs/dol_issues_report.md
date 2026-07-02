# dol — Issues Triage & Tackle-Order Report

> **Purpose:** a scannable, prioritized map of dol's open GitHub issues — what is
> already resolved (and should be closed), and in what order to tackle the rest.
> Companion to [issues_and_discussions.md](issues_and_discussions.md) (themes/history),
> [dol_architecture_map.md](dol_architecture_map.md) (code-verified mechanics), and the
> local-only ecosystem inventory (`misc/data/`, gitignored — dependents & the pre-PR
> test gate).
>
> Prepared 2026-07-02 against dol `0.3.46` / HEAD `4a995b8`. Every "verified" verdict
> below was reproduced or refuted by running code, not inferred from titles.

---

## TL;DR

- **Close now (verified resolved): #40, #52, #58.** Three open issues were already fixed by
  merged PRs but never closed. (A fourth, #50, was already closed 2025-10-10.) Closing these
  leaves **14 legacy issues** to work through — plus #67–#70 filed by this study.
- **The single highest-leverage cluster is `wrap_kvs`'s signature conditioning: #9 → #12
  → #18 (+ #5, #6).** One root cause (guessing a transform's calling convention from its
  first *parameter name*) spawns a whole bug class. Fixing it is a **breaking change** and
  `wrap_kvs` is used by **32 of 76** ecosystem dependents, so it must go through the
  dependents test-gate. This is where redesign effort should concentrate first.
- Everything else is correctness fragilities (**#14, #3**), design/feature enhancements
  (**#10, #16, #56, #2**), and small wins (**#13, #15, #1**).

---

## 1. Reconcile first — close/confirm these (verified resolved)

| # | Title | Evidence | Action |
|---|-------|----------|--------|
| **#40** | `re.error: incomplete escape \U` | Root-cause fixed: `naming.py:265–291` now `re.escape`s literal template text and explicitly handles Windows `C:\Users\…` backslash paths (the exact traceback site). Fixed via PR #64/#65. | **Close** (fixed) |
| **#50** | Stacking `cache_this` decorators | Implemented in `caching.py` (stacking-aware `__set_name__` propagation, `:550`, `:692`) + 4 regression tests (`test_caching.py::…test_basic_stacking/test_triple_stacking/…`). Merged PR #57. | Already closed 2025-10-10 — listed for completeness |
| **#52** | Make dol tests windows compatible | Duplicate of #58; superseded by the cross-platform PRs #64/#65. | **Close** (duplicate of #58) |
| **#58** | Make dol tests work on windows | **Windows CI re-run 2026-07-02 → green** (run `28614299483`). 28 failures → 0. Code fixes in `naming.py`, `filesys.py`, `filt_iter`/`filter_regex`. | **Close** (resolved) — see caveat → new issue for auto-trigger |

> **Caveat carried forward:** `.github/workflows/windows_ci.yml` is `workflow_dispatch`
> (manual) only, so nothing guards against Windows regressions on future PRs. Captured as a
> **new issue** (see §4) rather than left implicit in #58.

---

## 2. Tackle order (the 14 that remain)

Waves are ordered by leverage. Within a wave, do items top-to-bottom.

### Wave 1 — The `wrap_kvs` core knot  ⟵ *start here; unblocks the redesign*

This is one architectural problem wearing five hats. See
[dol_architecture_map.md §5.4](dol_architecture_map.md) for the mechanics.

| # | P | Type | Role in the cluster |
|---|---|------|---------------------|
| **#12** | high | enhancement / refactor / breaking | **The fix vehicle.** Proposes an explicit `FirstArgIsMapping` marker so a transform opts *in* to the `(self, data)` convention instead of dol guessing. **The class already exists but is unused** (`trans.py:2113`, `class FirstArgIsMapping(LiteralVal)`, tagged `# TODO: Use this for it's intent!`). Decide the mechanism here. |
| **#9**  | high | bug / breaking | **The flagship symptom.** `wrap_kvs(store, obj_of_data=bytes.decode)` fails while `…=lambda x: x.decode()` works. **Verified root cause:** dol decides `(data)` vs `(self, data)` by whether the transform's *first parameter name* ∈ `{"self","store","mapping"}` (`trans.py:1617`, `_has_unbound_self` `:424`, `_first_param_is_an_instance_param` `:419`). `bytes.decode`'s first param is literally named `self`, so it misfires. **Not** an arg-count check (older docs said so — corrected). Fix by adopting #12. |
| **#18** | high | bug | Same root cause, different face: a `wrap_kvs`-decorated class calling `self[k]` internally hits the *unwrapped* instance (transforms bypassed). **Verified reproducible.** Either fold into the #12 fix or standardize the "re-wrap self" pattern + document. |
| **#6**  | medium | bug / refactor | `Store.wrap` freezes a subclass's `__init__` signature to the base class's (`signature(B) == signature(A)`, dropping B's own params). **Verified reproducible.** Sibling of the same signature-machinery family (`signatures.py` + `Store.wrap`). |
| **#5**  | medium | enhancement / refactor | "Better way to control the wrapper class" — the `wrapper=` argument threaded through every `Store.wrap` caller; proposal to use a hook on the object instead. Do alongside #6. |

**Why first:** `wrap_kvs` (32 pkgs), `Store` (12), `KvReader` (22) top the dependent-usage
table. Any change here is the biggest blast radius in the library → must be paired with the
pre-PR dependents test-gate and a deprecation path. Doing it first means later work builds
on a sound core.

### Wave 2 — Correctness fragilities

| # | P | Type | Note |
|---|---|------|------|
| **#14** | medium | bug | `mk_dirs_if_missing` fragilities: (a) only makes dirs on *write*, so `list()`/`get` on a fresh store raise FS errors instead of empty/`KeyError`; (b) `mk_dirs_if_missing(wrap_kvs(G))` breaks (`FileNotFoundError: ''`) while `wrap_kvs(mk_dirs_if_missing(G))` works — an **ordering fragility tied to Wave 1**. Sequence after #9/#12. |
| **#3**  | low | bug | `FileBytesReader(gettempdir())` — **still fails, but the failure has evolved**: now a `PermissionError` walking macOS-protected temp subdirs (`…/com.apple.appleaccountd/TemporaryItems/`). Reframe as *recursive-walk should skip or gracefully handle unreadable dirs*. Small, self-contained. |

### Wave 3 — Design / feature enhancements

| # | P | Type | Note |
|---|---|------|------|
| **#10** | medium | enhancement | Recursively applying wrappers to nested stores (`conditional_data_trans`); the "store of stores" DNA-propagation problem. Overlaps #2 (both want `kv_walk`-driven traversal). |
| **#16** | medium | enhancement / paths | `KeyPath` write-through / autovivification (`s[1][2][3] = v` when intermediate keys don't exist). Needs contextual per-level factories. |
| **#56** | low | enhancement | Fast `update`/sync between heterogeneous stores (avoid item-by-item copy). Needs a duck-typed "fast-sync" protocol negotiation; clean design open. |
| **#2**  | medium | documentation / enhancement | `kv_walk`: docs, tests, and a `store_decorator` that surfaces walk-paths as a flat `KvReader`/`KvPersister`. Pairs with #10. |

### Wave 4 — Small wins & docs

| # | P | Type | Note |
|---|---|------|------|
| **#13** | low | enhancement / good-first-issue | `confirm_overwrite` preset for `wrap_kvs` (prompt before overwriting a differing value). Self-contained; good onboarding task. |
| **#15** | low | enhancement | `AttrContainer` tab-completion in PyCharm (works in Jupyter). Niche IDE-specific; needs a `__dir__`/dunder investigation. |
| **#1**  | low | documentation | Long-running doc wishlist (`partial`-based custom wrappers; verify postget/obj_of_data order). Largely absorbed by `llms.txt`/`CLAUDE.md`/`misc/docs`; prune to what's still missing. |

---

## 3. The dependency map between issues

```
#12 (marker mechanism)  ──┬──▶ #9  (bytes.decode etc.)
                          ├──▶ #18 (self not wrapped)
                          └──▶ #14 (mk_dirs ordering fragility, partial)
#6 (subclass signature) ──── #5 (wrapper control)      [Store.wrap family]
#10 (recursive wrap) ─────── #2 (kv_walk tooling)      [nested traversal]
#16, #56  standalone
```
Do **#12 before #9/#18**; do **#9/#18 before #14**; **#2 and #10 together**; **#5 with #6**.

---

## 4. New issues to file (from this study + the architecture pass)

Grounded in the code audit — see [dol_architecture_map.md §11](dol_architecture_map.md).
Filed 2026-07-02 as part of this study:

- **[#67](https://github.com/i2mint/dol/issues/67)** — Windows CI is manual-only + not
  regression-guarded; make `windows_ci.yml` run on push/PR (or add a Windows leg to
  `ci.yml`). Carries #58's residual caveat.
- **[#68](https://github.com/i2mint/dol/issues/68)** — `KeyValueCodecs.key_based` /
  `.extension_based` are empty stubs (`kv_codecs.py:580,589`) that silently return `None`;
  the namespace is also unexported — implement or remove.
- **[#69](https://github.com/i2mint/dol/issues/69)** — consolidate the three path-get
  implementations (`path_get`, `_path_get`, `chain_get`) and the two unrelated `flatten`s
  (`trans.flatten` vs `paths.flatten_dict`) — a concrete slice of Discussion #21.
- **[#70](https://github.com/i2mint/dol/issues/70)** — refactor: split oversized modules
  (`trans.py` 3492 LOC, `signatures.py` 5403, `caching.py` 2675) and dedup triplicated
  helpers (`identity`/`identity_func` ×3; `HashableMixin`/`HashableDict`).

> **`FirstArgIsMapping` is dead code** (`trans.py:2113`, tagged `# TODO: Use this for it's
> intent!`) — *not* filed separately; it is the concrete implementation vehicle for **#12**
> and is noted in #12's thread to keep the design decision and its implementation together.

---

## 5. Reference: verification log

| Issue | How verified | Result |
|-------|--------------|--------|
| #40 | Read `naming.py:265–291`; re-ran the escape logic | Fixed |
| #50 | Read `caching.py` stacking code + `test_caching.py` | Fixed (4 tests) |
| #52/#58 | Triggered `windows_ci.yml` on master | Green (run 28614299483) |
| #9  | Ran `wrap_kvs(d, obj_of_data=bytes.decode)` | Fails (TypeError) — reproduced |
| #6  | Ran the `Store.wrap` subclass-signature snippet | Reproduced |
| #3  | Ran `FileBytesReader(gettempdir())` | Fails (PermissionError, evolved) |
| #18 | Documented in CLAUDE.md; mechanism confirmed in `trans.py` | Reproducible |

Backend usage weights (why blast radius matters) come from the local ecosystem scan:
`wrap_kvs` 32 pkgs · `Files` 24 · `KvReader` 22 · `Pipe` 18 · `filt_iter` 15 · `Store` 12.
Full inventory: `misc/data/dol_ecosystem_dependents.md` (gitignored).
