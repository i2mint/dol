# Handoff: dol — #18 downstream + Wave 2 + small wins — 2026-07-05

## State of play
`dol` on `master`, clean, published at **0.3.52** (0.3.53 publishing from the last merge).
Everything below is **merged**; no open dol PRs, no uncommitted work. This session finished
the whole *tractable* roadmap — the correctness bugs and quick wins. What remains is
design-heavy, refactor, or the deferred structural change (see "Next session should").

**Done this session (all merged):**
- **#18 Phase 1** — shipped `wrapped_self()` (dol.base): recovers the outer transform-applying
  store from inside a delegation-wrapped class's own method, so `wrapped_self(self)[k]` works
  where `self[k]` silently bypassed transforms. Design + full analysis in
  `misc/docs/dol_issue18_design.md`. **#18 stays OPEN for Phase 3.**
- **#18 Phase 2** — the 6 real latent bugs fixed downstream via `wrapped_self`:
  `i2mint/xdol#4`, `i2mint/unbox#2`, `thorwhalen/lexis#2` (all merged).
- **#14** (CLOSED) — `mk_dirs_if_missing` fragilities were already resolved on HEAD; added the
  combinatorial regression tests the issue asked for (`dol/tests/test_mk_dirs_if_missing.py`).
- **#3** (CLOSED) — recursive file walk now skips unreadable dirs (`paths_in_dir` catches
  `PermissionError`/`NotADirectoryError`); `FileBytesReader(gettempdir())` works again.
- **#13/#68/#67** (CLOSED) — `confirm_overwrite` preset; removed silent-`None` codec stubs;
  removed the redundant manual `windows_ci.yml` (ci.yml already runs Windows automatically).
- Earlier in the session: merged the waiting #9/#12 fix (closed) and the Wave-1 study docs.

## Open issues (remaining roadmap)
- **#18** (high) — Phase 1+2 done; **Phase 3 = is-a wrapping** (deferred). Resolves #18 **and**
  #5 **and** the #6-family at the root, but it's a **major, breaking change**. Needs a
  maintainer DECISION before any code (see design doc §"open questions").
- **#5** (medium) — "control the wrapper class"; couples with the is-a work.
- **#2, #10, #16** (medium) — design-heavy: `kv_walk` tooling; recursive wrapping of nested
  stores; `KeyPath` write-through. Each wants a design pass first.
- **#69, #70** (low) — refactor: consolidate duplicated `path_get`/`chain_get` + the two
  `flatten`s; split oversized modules + dedup triplicated helpers.
- **#1, #15, #56** (low) — docs wishlist; `AttrContainer` tab-completion; fast update/sync.

## Key decisions this session
- **#18 fix = `wrapped_self` (opt-in helper), NOT method-rebinding.** A 6-lens adversarial
  review + empirical tests proved rebinding `self` to the wrapper silently corrupts writes
  under `Pipe`/stacked wraps (binds to the innermost layer) and crashes on `super()`/dict-native
  ops. `wrapped_self` is zero-default-change and climbs to the outermost wrapper.
- **`wrapped_self` registry is multi-valued + identity-guarded.** The review caught a real bug:
  a shallow `copy.copy` shares the inner store; the single-slot registry let the copy's GC
  evict the original's back-ref (silent revert to raw #18). Fixed with `dict[id -> list[weakref]]`
  + a `getattr(w,'store',None) is cur` guard in the climb. **Don't "simplify" these back.**
- **is-a wrapping is the terminal direction for #18/#6/#5, deliberately deferred** to a major
  version. It's the single highest-leverage next step but needs your go-ahead.

## Next session should
1. **Pick the direction** (this is the fork the session ended on). Read
   `misc/docs/dol_issues_report.md` (tackle order) + `misc/docs/dol_issue18_design.md` (§staged
   plan + open questions). The two strongest candidates:
   - **Decide on #18 Phase 3 (is-a wrapping):** if committing, the first action is an is-a
     **RFC/design** (design-only, no code) co-designed with #5, per the design doc's Phase-3
     plan — highest leverage; resolves #18+#5+#6.
   - **Or a Wave-3 design pass** on #16 (`KeyPath` write-through) or #10 (recursive wrapping).
2. Whatever the pick: it's a **new framed task** → run the pre-work alignment check first, and
   for any core `wrap_kvs`/`base`/`filesys` change, run the **dependents test-gate** before merge.

## Gotchas
- **Dependents test-gate baseline:** 39/56 dependents pass; the **same 17 fail pre-existingly**
  (`hubcap, msword, raglab-bak, sqldol, dn, raglab-app, accompy, dropboxdol, crude, couchdol,
  jy, arangodol, arioso, dynamodol, s3dol, theremin, uf`) — missing DB drivers / cloud creds /
  their own broken tests. **Don't chase these as regressions.** Gate runner + inventory live in
  `misc/data/` (gitignored). The parallel runner used this session was a scratchpad throwaway;
  `misc/data/run_dependent_tests.py` is the durable serial one (or re-derive a parallel one).
- **Editable-install shadow:** dependents import the working-tree dol, so a fix branch is seen
  ecosystem-wide with no reinstall (`dol.__file__` confirms). But **stash carefully** — a bare
  `git stash pop` this session popped one of *your 3 pre-existing dol stashes* and conflicted
  `paths.py`; recovered with `git reset --hard HEAD`. Your 3 stashes are intact; leave them.
- **Pre-existing uncommitted changes on local `unbox` and `lexis` masters** (a docs cleanup;
  `ci.yml` + an examples file) are **yours, not from this session** — left untouched.
- **`dol.trans.double_up_as_factory` doctest fails** under bare `pytest --doctest-modules`
  (needs IGNORE_EXCEPTION_DETAIL) — pre-existing, unrelated; don't chase.
- **CI auto-bump push-back races** on near-simultaneous merges (one run's version-bump push is
  rejected non-ff); it's benign and self-resolves via the next run. Versions ended in sync.
- **`.claude/handoffs/` is still NOT in dol's `.gitignore`** — this file is untracked; don't
  `git add -A` it. Worth adding the ignore line in a future docs change (carried from last handoff).
- **lexis** has an *unrelated* pre-existing bug (`from_name` -> `_from_name` is None) and its
  full `dict_of_non_empty_values` can't materialize in the current env (an nltk-version issue in
  one Synset method) — the #18 fix itself is verified (iteration now works via the outer store).

## Files most touched
- `dol/base.py` — `wrapped_self` + `_wrapper_backrefs` registry (Store.__init__/__setstate__).
- `dol/filesys.py` — `paths_in_dir` PermissionError skip (#3); `mk_dirs_if_missing` (#14, no change).
- `dol/trans.py` — `confirm_overwrite` / `mk_confirm_overwrite_preset` (#13).
- `dol/kv_codecs.py` — removed empty `KeyValueCodecs` stubs (#68).
- `dol/__init__.py` — exports `wrapped_self`, `confirm_overwrite`, `mk_confirm_overwrite_preset`.
- `dol/tests/` — `test_walk_permissions.py`, `test_mk_dirs_if_missing.py`,
  `test_confirm_overwrite.py`, plus `base_test.py` (wrapped_self suite).
- `misc/docs/dol_issue18_design.md` — the #18 design (Phase 1 now / is-a later).
- `misc/data/*` — LOCAL-ONLY (gitignored): dependents inventory, gate runner, blast-radius data.
