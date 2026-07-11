# Handoff: dol tightening — Wave 1 (wrap_kvs) + study/skills — 2026-07-04

## State of play
Two-part effort on `i2mint/dol`. Repo is on `master`, clean; all work is on two pushed branches / open PRs.

**Part A — analysis & tooling (PR #71, branch `study/dol-tightening`):** ecosystem dependents map, architecture map, issues triage/tackle-order report, issue labeling, and a consumer skill. Docs-only.

**Part B — Wave 1 code fix (PR #72, branch `fix/wrap-kvs-signature-conditioning`):** the `wrap_kvs` signature-conditioning knot **#9 + #12 is FIXED**. CI green. Closes #9 and #12 on merge.

Done this/last session:
- Closed verified-resolved issues **#40, #52, #58** (Windows CI re-run green); confirmed #50 was already closed.
- Labeled all open issues (priority + `wrap_kvs`/`windows`/`caching`/`paths` clusters); filed **#67–#70** (tech debt).
- Fixed #9/#12; validated backward-compatible 6 ways (see Gotchas). 8 regression tests + doctests.
- Assessed #18 and #6 → confirmed **distinct mechanisms** from #9; deferred with root-cause comments.
- Authored skills: `dol-dev-wrap-kvs` (dev, in PR #72) and `dol-store-building` (consumer, in PR #71).

## Open issues / PRs
- i2mint/dol#72 — Fix #9/#12 wrap_kvs conditioning + FirstArgIsMapping — **OPEN, CI green, merge first.**
- i2mint/dol#71 — docs + consumer skill + CLAUDE index (Wave 1 study) — **OPEN, docs-only, rebases cleanly after #72.**
- Wave 1 remainder (open, root-caused, not yet fixed): **#18** (self unwrapped — delegation architecture, priority high), **#6** (Store.wrap freezes subclass `__signature__` — priority medium), **#5** (wrapper-control, with #6).
- Wave 2 (next fixes): **#14** (`mk_dirs_if_missing` fragilities — partly unblocked by the #9 fix), **#3** (recursive-walk PermissionError robustness).
- Also open: #1, #2, #10, #13, #15, #16, #56, and new #67–#70. Full order in `misc/docs/dol_issues_report.md`.

## Key decisions this session
- **#9 fix = name AND arity, not a rewrite.** `_has_unbound_self` now needs first-param-name ∈ {self,store,mapping} **AND ≥2 required positional params**. Chosen because it's backward-compatible (unlike "explicit-marker-only") — proven by 0 blast radius. Long-term clean direction (drop the name heuristic) is a future breaking change.
- **`FirstArgIsMapping` wired in** via one resolver `_resolve_self_convention` (all 4 call sites) — realizes #12; was dead code.
- **Split PRs** (code #72 vs docs #71) — for independent review of a change touching 32 dependents. (User was asked bundle-vs-split; didn't answer in time; split chosen. Reversible if they prefer bundling.)
- **#18/#6 deferred** — different mechanisms (delegation / signature inheritance), riskier, not bundled into the conditioning fix.

## Next session should
1. **First:** check whether the user has reviewed/merged #72; if approved, merge #72 then #71.
2. **Then:** pick the next work item — Wave 2 (**#14** is the highest-value correctness fix and is partly unblocked) OR the `Store.wrap` family (**#6 + #5**) OR **#18**'s delegation rethink. Each open issue has a root-cause comment to start from.
3. Read `misc/docs/dol_issues_report.md` for the full prioritized order before choosing.

## Gotchas
- **#18 "self is unwrapped" is NOT fixed** and is architectural: `Store.wrap` delegates (has-a), so methods on a `@wrap_kvs` class bind to the inner store. Don't assume the #9 fix touched it.
- **Intended-by-design behavior change:** a transform with first param self/store/mapping but `<2` required params (e.g. `def f(self, data=None)`) is now called `f(data)`. Unavoidable (same shape as `bytes.decode`); `FirstArgIsMapping` is the opt-in. Confirmed 0 such sites in the ecosystem.
- **Dependents test-gate is the safety net for any wrap_kvs/base change.** Scripts + gate order live in `misc/data/` (gitignored — names private packages; regenerate with the scan scripts there). Method: run each dependent's suite, `git stash` the dol edit, run again (baseline); a PASS→FAIL delta is a real regression.
- **Editable install shadows site-packages:** the local checkout is imported ahead of the pip-installed `dol` in site-packages, so dependents pick up working-tree changes with no reinstall (verify with `dol.__file__`).
- **Pre-existing, unrelated:** the `dol.trans.double_up_as_factory` doctest fails under bare `pytest --doctest-modules` (needs IGNORE_EXCEPTION_DETAIL) — not from this work; don't chase it.
- **windows_ci.yml is manual (`workflow_dispatch`) only** — #67 tracks making it automatic. Trigger it by hand for path/regex changes.
- **`.claude/handoffs/` is not yet in dol's `.gitignore`** — this file is untracked; don't `git add -A` it. (Could add the ignore line in a future docs PR.)

## Files most touched
- `dol/trans.py` — `_has_unbound_self`, `_num_required_positional_params`, `_resolve_self_convention`, `FirstArgIsMapping`, the 4 call sites (`_wrap_outcoming`/`_wrap_ingoing`/postget/preset). (fix on branch `fix/wrap-kvs-signature-conditioning`)
- `dol/tests/test_trans.py` — 8 new regression tests.
- `dol/__init__.py` — exports `FirstArgIsMapping`.
- `misc/docs/dol_architecture_map.md` (§5.4/§5.5), `dol_issues_report.md`, `issues_and_discussions.md` — study docs (branch `study/dol-tightening`).
- `.claude/skills/dol-dev-wrap-kvs/SKILL.md` (PR #72), `.claude/skills/dol-store-building/SKILL.md` (PR #71), `CLAUDE.md` (index).
- `misc/data/*` — LOCAL-ONLY (gitignored): ecosystem inventory, blast-radius scan, test-gate runner.
