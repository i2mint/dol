# fleet_ledger — who uses dol, how, and at which commit

A versioned record of the local fleet's use of `dol` ([#91](https://github.com/i2mint/dol/issues/91)): per-dependent commit pins with a restore helper, per-dependent test baselines, an AST census (subclassing, call-site keyword arguments), a zero-usage report, and a history of scans. It is the substrate for the vocabulary migration ([#92](https://github.com/i2mint/dol/issues/92)), the cruft audit ([#94](https://github.com/i2mint/dol/issues/94)) and the dependents test gate of any `wrap_kvs`-surface change.

```bash
# from the dol repo root, with the fleet's dev interpreter
python -m misc.fleet_ledger scan                 # ~30s: pin + census every dependent -> one immutable scan
python -m misc.fleet_ledger baseline --jobs 3    # every dependent's suite at the pins
python -m misc.fleet_ledger zero-usage --out ~/zero.md
python -m misc.fleet_ledger report               # headline numbers
python -m misc.fleet_ledger diff BASELINE_A      # newly failing test ids, A vs the latest baseline
python -m misc.fleet_ledger pin check            # where the fleet is vs the latest scan's pins
python -m misc.fleet_ledger pin restore          # dry run; --apply to check out; `pin undo --apply` reverses
python -m misc.fleet_ledger history
python -m misc.fleet_ledger mark-superseded ID "reason"
```

The code is committed and tested with dol's suite (`pytest misc/fleet_ledger`). The tool is not part of the `dol` distribution and has no dependency beyond the standard library, except that the *default* list of dependents and the gate's pytest command come from priv's `dep_graph` (the fleet tooling package). Without priv, pass the dependents yourself: `--repo NAME=PATH` (repeatable), or `repos={name: path}` in Python.

## Where the data lives, and why not here

Everything the tool *produces* is derived from the fleet's source code, and some dependents are private repositories. dol is public, so **no ledger output is ever committed**. Records go to `~/.local/share/dol/fleet_ledger/dol/` (override with `$DOL_FLEET_LEDGER_DIR` or `--root`). Every record is scrubbed of local prefixes (`$PP` → `<PP>`, `$HOME` → `~`) at write time, and the writer **refuses** to persist a string that still looks like an absolute local path, so a leak fails loudly instead of landing on disk. Records store dependent *names*, never paths: a path is resolved again on the machine at hand when it is needed.

## The four asks of #91, and where each lives

| Ask | Module | What it records |
|---|---|---|
| 1. Pin manifest + baselines + restore | `scan.py`, `baseline.py`, `pins.py` | 40-char SHA, branch, upstream, ahead/behind, dirty and untracked counts, commit date; per-suite status, exact command, duration, counts and **failing test ids**; restore/undo |
| 2. AST, not regex | `astscan.py` | imports through every binding form, references (incl. star-imported names), subclassing closed over each repo's class graph, call-site keywords with the kind of each value |
| 3. Zero-usage report | `report.py` | exports split into used / candidates / recent / incidental, the submodule compat surface (imports *and* attribute access, underscored names included), names used that no longer exist, with three generated caveats; it names dependents, so it is kept local like the rest |
| 4. Successive scans | `store.py` | `scans/<utc-id>/`, `schema_version` everywhere, `index.json`, `mark-superseded` (never delete) |

## Layout and schema (`schema_version: 1`)

```
<root>/index.json                   {schema_version, scans: [...], baselines: [...]}
<root>/scans/<id>/meta.json         the scan: when, which dol commit, what `import dol` resolved to
<root>/scans/<id>/surface.json      dol's exports and every submodule's public names at that commit
<root>/scans/<id>/pins.json         {pins: {name: pin}} (small, so restore never loads the census)
<root>/scans/<id>/census.json       {dependents: {name: record}, aggregates: {...}}
<root>/baselines/<id>/baseline.json {scan_id, results: {name: suite result}, package_imported}
<root>/pins/undo-<id>.json          what a `pin restore --apply` changed (+ `.done` once undone)
```

Ids are UTC timestamps (`20261013T081809Z`), so they sort by time. Records are written once and never overwritten; only `index.json` is rewritten, to append an entry or set its `superseded: {reason, by, at}`.

- **index entry** — `id`, `created`, `package_sha` (scans) or `scan_id` and `statuses` (baselines), `n_dependents`/`n`, `note`, `superseded`.
- **meta** — `package`, `package_version`, `package_pin` (the dol commit, as a pin), `import_resolved: {file, from_source}`, `dependents_source` (`priv.dep_graph` or `explicit`), `n_dependents`, `n_found`, `transitive_dependents`, `python`, `note`.
- **pin** — `found`, `git`, `sha`, `branch`, `upstream`, `ahead`, `behind`, `dirty` (tracked), `untracked`, `commit_date`, `subdir` (package directory relative to its git toplevel).
- **census record** — `scanned`, `n_files`, `parse_errors`, and lists of `imports` (`file`, `line`, `module`, `name`, `asname`), `refs` (`file`, `line`, `symbol`, `module`, `via`: `import`|`star`), `calls` (plus `kwargs: [{name, kind}]`, `n_args`, `double_star`, `decorates`: `class`|`function`|null, `bare_decorator`), `subclasses` (`name`, `direct`, `indirect` dol roots), `stars` (`file`, `module`). All `file` values are POSIX paths relative to the dependent.
- **aggregates** — `symbols` (per dol name: `packages`, `import_sites`, `ref_sites`), `calls` (per callee: `sites`, `decorator_sites`, `class_decorator_sites`, `kwargs: {name: {packages, sites, kinds}}`), `subclasses` (per dol base: `classes`, `direct`, `indirect`), `compat_names`, `unknown_names`, `star_imports`. Each has `n_packages`.
- **suite result** — `status` (`PASS`, `FAIL`, `NO_TESTS`, `INTERRUPTED`, `ERROR`, `TIMEOUT`, `DISABLED`, `NOT_FOUND`), `returncode`, `command` (with `<python>` and `<junitxml>` placeholders), `seconds`, `counts: {tests, passed, failures, errors, skipped}`, `failing` (`classname::name` ids), `tail` (last lines, non-passing only), `sha`, `at_pin`, `dirty`.

Bump `SCHEMA_VERSION` in `store.py` when a record's shape changes incompatibly.

## Design choices worth knowing

- **Every dependent is run.** No "has tests" heuristic decides what the gate runs: the v1 heuristic silently dropped real suites (an in-package `tests/`, `*_test.py` naming, doctest-only repos). Pytest's exit code 5 is recorded as `NO_TESTS`.
- **Not fail-fast.** A baseline keeps the whole shape of a suite, so `diff` is a set difference on failing ids. `diff` refuses two baselines whose command for a dependent differs (changing `-rf` to `-rfE` once made three pre-existing collection errors read exactly like regressions).
- **The census is scored against the dol tree you point at** (`--source-root`, default the checkout holding this tool), imported in a subprocess. `meta.import_resolved` says which file that was; the baseline records what `import dol` resolves to for the gate's interpreter, because a gate that measures a stale wheel instead of the candidate is worse than no gate.
- **`pin restore` is conservative.** Dry run by default; refuses repos with tracked modifications; never fetches, never forces; writes its undo record before the first checkout; checks out detached, leaving branches alone.
- **`--jobs 1` by default**, like `priv test-dependents`: some suites share on-disk state.

- **`diff` never reports a test as fixed when it simply did not run.** A dependent whose run did not complete on either side (`INTERRUPTED` by a collection error, `TIMEOUT`, `ERROR`) is reported as `incomparable` when its status changed.
- **One dependent cannot cost the baseline.** A suite whose output quotes a local path has that path redacted (`<abs>`) in its `tail`; an exception while running one dependent is recorded as its `ERROR`.
- **`pin undo` only undoes what `restore` did.** A repo that is no longer detached at the restored pin (a commit made there, a branch switched to) is left alone and reported. A scan taken with `--repo` refuses to be restored or baselined through the real fleet's name resolution.

Known approximations: a name's binding is the latest one before its use in the file, regardless of scope (a function-local `from dol import X` binds `X` file-wide, a later `from other import X` unbinds it); local base classes are matched across a repo by class *name*; usage is counted per dol short name, so a submodule name equal to an `__init__` export counts as one; calls also cover attributes of dol symbols (`Store.wrap(...)`, symbol `Store.wrap`) and `partial(<dol callable>, ...)` (`via_partial`), and a class whose base is such a call is rooted at that callee (`class A(wrap_kvs(dict, ...))` has root `wrap_kvs`); failing ids are JUnit `classname::name`, stable between runs but not always pytest node ids; a dependent's own pytest `addopts` apply, as in its CI. Notebooks are not scanned. On a shallow clone of dol the export dates are meaningless, and the zero-usage report says so.

## Relation to priv

priv's `dep_graph` is the fleet's single source for who depends on dol (rebuilt live at scan time) and for the gate's pytest command (CI's doctest flags plus each repo's `[tool.wads.ci.testing]` exclusions), and the ledger reuses both rather than re-deriving them. What priv does not have, and this adds, is the symbol-level census and per-test baselines; the census is generic in the target package, so it is a candidate to move into priv (see the issue linked from #91).
