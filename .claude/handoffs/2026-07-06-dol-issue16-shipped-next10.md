# Handoff: dol #16 shipped → #10/#2 next — 2026-07-06

## State of play
`dol` on `master`, clean, **local synced to published `0.3.55`** (CI green, PyPI published,
CI-formatted code pulled back). Nothing in flight, nothing uncommitted, no open dol PRs.

**Shipped this session (all on `master`):**
- **#16 — optional key-path write-through / autovivification** (design → full implementation).
  - Opt-in `create_missing=False` on `add_path_access` / `KeyPath` + top-level `autoviv(store)`
    alias. Default OFF is byte-identical to before (missing intermediate → `KeyError`).
  - Contextual per-level factory `mk_missing(ctx)` where `ctx = PathContext(prev_path, key, depth)`.
  - Single **boundary write-back** protocol → persists correctly through copy-semantics /
    persistent stores (`Files`, `wrap_kvs`); also closes the pre-existing copy-store silent-loss
    on existing deep paths. Loud failures only (`PathCreationError`/`PathWritebackError`);
    creation announced via `warnings.warn`.
  - New dependency-free leaf `dol/_paths_core.py` houses the engine (`path_set_writeback`,
    `path_del_writeback`, `_path_set_writeback`, `_inplace_build`, `PathContext`, exceptions).
    `path_set` moved here + its factory-not-propagated bug fixed. Breaks the `trans→paths`
    import cycle; pre-stages the #70 split.
  - 37 tests in `dol/tests/test_path_writethrough.py`. Full suite 172 pass / 2 skip; full
    doctests 472 pass; dependents gate no PASS→FAIL. An adversarial-review pass caught & fixed
    3 bugs my first tests missed (see Key decisions).
  - Design doc `misc/docs/dol_issue16_design.md` (with resolved API decisions §11). Two comments
    on issue #16 (design summary + implementation-landed).

## Open issues (remaining roadmap)
- **#16** — implemented & shipped, but **left OPEN** on purpose for 6 minor design questions
  (design doc §11 "Open questions still for maintainer decision"): exception taxonomy vs
  `path_get`-style `on_error`; make silent-loss fix always-on in a future major; `explore_further`
  default; an `on_conflict` compare-and-swap alongside `writeback_lock`; prune now-empty
  auto-created intermediates on delete; and **persistent store-of-stores ("Model 2") deferred to
  #10**. Close #16 once satisfied, or fold the leftovers into the #10 work.
- **#10** (medium, `wrap_kvs`) — recursively applying wrappers to nested stores. **This is the
  natural next task** (see below).
- **#2** (medium) — `kv_walk`: a `store_decorator` that surfaces walk-paths as flat KvReader/
  KvPersister keys. Pairs with #10.
- **#69, #70** (low, refactor) — #16 already advanced both: `_inplace_build` is the canonical
  splicer (#69) and `_paths_core.py` pre-stages the module split (#70). Remaining: consolidate
  the duplicated `path_get`/`chain_get` + two `flatten`s into `_paths_core` (#69); finish moving
  path code out of the 2270-LOC `paths.py` (#70).
- **#5, #18** — is-a wrapping (deferred, major); `wrapped_self` shipped for #18 Phase 1/2.
- **#1, #15, #56** (low) — docs wishlist; `AttrContainer` tab-completion; fast update/sync.

## Key decisions this session
- **#16 = opt-in kwargs on existing entry points + a canonical engine**, not a standalone
  decorator. Four API decisions resolved WITH the user: single `create_missing` flag (not
  `create_missing`+`write_back`); ship the `autoviv()` alias; create `dol/_paths_core.py` now;
  **loud + unbounded** typo-safety default (`on_create=warnings.warn`, `max_created=None`).
- **The write-back trap is the crux** and is the load-bearing insight for #10 too (see below).
- **Adversarial review earns its keep.** A skeptic agent (running real code) found 3 bugs 32
  hand-written tests missed: (1) raw `TypeError` leaked when an existing scalar blocked a deep
  path — fixed by a **two-pass `_inplace_build`** (guard node → authorize ALL missing levels →
  build off-side → splice atomically), which ALSO fixed (3) partial writes on dict parents;
  (2) `writeback_lock` didn't span the full read-modify-write — fixed by splitting
  `path_set_writeback` into a lock-acquiring entry + a recursive core (`_path_set_writeback`)
  that issues all boundary writes with `lock=None`. Don't regress these — tests guard them.

## Next session should
**Do #10 (with #2) as a fresh, design-first task — it's high-blast-radius, so give it a clean
context.** Concrete first action: run the pre-work alignment check, then a design pass (the same
understand→approaches→judge→adversarial-verify workflow that worked for #16). Start from this
**warm `#16 → #10` bridge** (capture is the whole point of this handoff):

- **#10 IS the general form of #16's deferred "Model 2".** #10 = "wrapping only wraps the top
  level; make it easy to opt into recursively wrapping nested values." The issue itself sketches
  `wrap_kvs(obj_of_data=recursive_wrap)` / `add_path_access_if_mapping` via `conditional_data_trans`.
- **The write-side of recursive wrapping already exists**: `explore_further(node, path) -> bool`
  in `add_path_access`/`path_set_writeback` (`dol/_paths_core.py`). It descends into a sub-store
  that is its own persistence boundary; a newly-created sub-store is registered back into a
  **reference-semantics** parent, and raises `PathWritebackError` for a **persistent** parent
  (the deferred case). #10 should generalize this into a real recursive-wrapping mechanism.
- **The write-back trap governs #10's write side**: a `__getitem__` that returns a fresh
  deserialized copy (any `wrap_kvs` with non-identity `obj_of_data`, `Files`, DB) means mutating
  a nested wrapped value is LOST unless written back to its boundary. Recursive *read* wrapping is
  easy; recursive *write* wrapping must solve write-back (single-boundary re-serialize, per #16 §7).
- **Prior art to reuse**: `dol.filesys.mk_dirs_if_missing` (persistent nested creation — the
  "Model 2" filesystem case), `_paths_core.path_set_writeback` (the boundary protocol),
  `conditional_data_trans` + `instance_checker(Mapping)` (already in `trans.py`), and `remap`
  (sedimental.org) referenced in both #10 and #2.
- **#2 pairs**: once recursive wrapping exists, a `kv_walk`-driven `store_decorator` that exposes
  nested paths as flat KvReader/KvPersister keys (`add_path_access` is read-only-flat today).
- **Mandatory before any merge**: read the `dol-dev-wrap-kvs` skill (this touches
  `conditional_data_trans`/`wrap_kvs` — 32 dependents), and run the dependents test-gate
  (`misc/data/run_dependent_tests.py`), baseline vs modified.

## Gotchas
- **`double_up_as_factory` doctest** fails ONLY when run in isolation (`pytest --doctest-modules
  dol/trans.py::...`); it PASSES in the full-package run (`--doctest-modules dol/`, 472 passed).
  Pre-existing, unrelated, untouched by this session — don't chase.
- **`Files` stores bytes**: a json codec needs `TextFiles` (str-native), not `Files`
  (`json.dumps` → str → `Files` raises `TypeError`→`KeyError`). The design doc §7.3 `+SKIP`
  example was wrong-as-written; fixed to `TextFiles`. Watch for this in #10 test scaffolding.
- **CI auto-bumps + publishes on every push to master** (wads flow): even a trivial one-line
  change triggers a version bump + PyPI publish. `.gitignore` still does NOT list
  `.claude/handoffs/` (carried from prior handoffs) — worth adding, but **bundle it with a
  substantive change** to avoid a solo trivial release. Meanwhile: stage files EXPLICITLY, never
  `git add -A` (this handoff is untracked).
- **Direct push to master** is blocked by the auto-mode classifier unless the user authorizes it
  (they did this session). Expect to ask, or push via a PR.
- **CI reformats your code (black)** on push, then pushes a bump commit — so after a push,
  `git fetch` + `git merge --ff-only origin/master` to pull your reformatted code back, or local
  drifts from published.
- **`_inplace_build` two-pass + `_path_set_writeback` lock split** are the adversarial-fix shapes
  — don't "simplify" them back (they fix 3 real bugs; tests guard them).

## Files most touched
- `dol/_paths_core.py` — NEW: the write-through engine + `path_set` + `PathContext` + exceptions.
- `dol/trans.py` — `add_path_access` opt-in kwargs + closure-captured `__setitem__`/`__delitem__`;
  `autoviv()`; import from `_paths_core`.
- `dol/paths.py` — `path_set` relocated/re-exported; `KeyPath` gained keyword-only opt-in fields
  (KW_ONLY) forwarded in `__call__`.
- `dol/__init__.py` — exports `autoviv`, `path_set_writeback`, `path_del_writeback`, `PathContext`,
  `PathCreationError`, `PathWritebackError`.
- `dol/tests/test_path_writethrough.py` — NEW: 37 tests.
- `misc/docs/dol_issue16_design.md` — the #16 design + resolved decisions.
- `CLAUDE.md` — Known Limitations entry for #16; docs-index rows for the #16/#18 design docs.
