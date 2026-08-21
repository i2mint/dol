# dol design documentation — start here

`dol` is a pure-Python, dependency-free toolkit for putting a uniform `dict`-like
(Mapping) face on any storage backend — files, S3, databases, in-memory — by
composing **key transforms**, **value transforms** (codecs), **filters**, and
**caches** over a raw backend. Domain code speaks `store[key]`; the layers decide
what that means physically.

This folder is the design memory of the project: reference docs (how it works),
design studies (why it works that way), decision records, and the roadmap. This
README is the master index — read the two paragraphs below for the big picture,
pick your reading path, then dig in.

## The one thing to understand first: two architectures

**Today (shipping)**: dol wraps by **has-a delegation** — each `wrap_kvs` /
`Store` layer is an object holding the previous one. It works, it's everywhere
(85 local dependent packages), and it has known structural traps: inside a
wrapped method `self` is the *unwrapped* store (#18), a delegated method
*receives* untransformed keys (#83), stacked wraps don't pickle, and there is no
inverse key mapping.

**The decided direction (experimental)**: the **flat model** — a wrap is
`(leaf, spec, stack)`: one proxy, a typed interface spec saying where keys/values
occur in *every* method, and a flat, compiled list of codec layers. Re-wrapping
extends the list; it never nests. The endgame (decision Q0, 2026-08-10) is a
**split synthesis**: codec/instance wrapping compiles to the flat engine;
class-decoration becomes is-a. Both mechanisms exist because they fix *disjoint*
populations of the delegation traps. Narrative walkthrough: discussion
[#90](https://github.com/i2mint/dol/discussions/90).

Where things stand and what's next: **[dol_roadmap.md](dol_roadmap.md)** (the
sequencing SSOT).

## Reading paths

**I want to build stores with dol (consumer).**
Repo-root `README.md` and `llms.txt` → the project `CLAUDE.md` core patterns (or
the `dol-store-building` skill) → [general_design.md](general_design.md) for the
conceptual model. Dip into [dol_design.md](dol_design.md)'s worked examples when
"why does this work" strikes. You should never need the issue docs; the gotchas
that matter (`wrapped_self`, `create_missing`) are in `CLAUDE.md`'s Known
Limitations.

**I'm changing dol itself (maintainer / refactorer).**
[dol_architecture_map.md](dol_architecture_map.md) first (code-verified map) →
the `dol-dev-wrap-kvs` skill before touching `trans.py`/`base.py` →
[dol_roadmap.md](dol_roadmap.md) for what's in flight → then the design-study
chain for your area (see the index). Before any PR touching `wrap_kvs`: the
dependents test-gate (`fleet_ledger baseline`, in local `misc/data/`).

**I'm an agent needing orientation.**
`llms.txt` → project `CLAUDE.md` → stop. Escalate to
[dol_architecture_map.md](dol_architecture_map.md) §2–3 (navigation) and §9
(idioms). Prefer the architecture map over [dol_design.md](dol_design.md) for
citation-grade mechanics — the latter is partially superseded (see its banner).

**I want the design history (how we got here).**
[issues_and_discussions.md](issues_and_discussions.md) →
[dol_issues_report.md](dol_issues_report.md) (2026-07 triage snapshot) → the
study arc: [dol_issue18_design.md](dol_issue18_design.md) →
[dol_issue83_design.md](dol_issue83_design.md) →
[dol_issue86_design.md](dol_issue86_design.md) → discussion
[#90](https://github.com/i2mint/dol/discussions/90). The paths thread runs in
parallel: [dol_issue16_design.md](dol_issue16_design.md) →
[dol_issue10_design.md](dol_issue10_design.md).

## Index

### Reference (how it works)

| Doc | Status | Contents |
|---|---|---|
| [general_design.md](general_design.md) | current (one scoped note) | Language-agnostic architecture: KV interfaces, interface hierarchy, middleware principle, the KV transform pipeline. The timeless layer — note in *Layered Composition* distinguishing semantic layers from runtime nesting. |
| [dol_design.md](dol_design.md) | **partially superseded** — see banner | Python implementation narrative: class hierarchy, `Store` hooks, `wrap_kvs`, codecs, `Sig`, `Pipe`. Still the only home of the `Sig` narrative, the design-critique register, `double_up_as_factory`, and the mid-altitude worked examples. Mechanics claims: prefer the architecture map. |
| [dol_architecture_map.md](dol_architecture_map.md) | current | Code-verified structural map + ranked tech debt. **Start here for refactors.** |
| [dol_flat_model.md](dol_flat_model.md) | current (engine: experimental/private) | **The flat model reference**: `(leaf, spec, stack)`, role codecs, flatten-and-compile, loudness, codec laws, scope limits, the six decisions, code map. |

### Sequencing

| Doc | Status | Contents |
|---|---|---|
| [dol_roadmap.md](dol_roadmap.md) | **living SSOT** | Tracks A–E: flat engine P0–P3, is-a, paths engine, the fleet program (#91 ledger / #92 vocabulary / #94 cruft), hygiene. Supersedes the issues-report waves. |
| [dol_issues_report.md](dol_issues_report.md) | historical snapshot (2026-07-02) | The triage that ordered the program: close-list, dependency map, verification log. Sequencing superseded by the roadmap; evidence still cited. |
| [issues_and_discussions.md](issues_and_discussions.md) | current-ish | Themes from GitHub issues/discussions. |

### Design studies & decision records (why it works that way)

| Doc | Contents |
|---|---|
| [dol_issue18_design.md](dol_issue18_design.md) | The delegation trap (`self` unwrapped inside methods): `wrapped_self` (shipped), is-a plan (Track B), rebind rejection. |
| [dol_issue83_design.md](dol_issue83_design.md) | The inverse trap (methods *receive* unmapped keys): 13-package census, options A–F, the carry-forward list (§5). |
| [dol_issue86_design.md](dol_issue86_design.md) | **Option G decision record**: the flat model's design study, adversarial-panel evidence, codec laws, Q0–Q5 decisions (§11), migration P0–P3. |
| [dol_issue16_design.md](dol_issue16_design.md) | Key-path write-through / autovivification (shipped): the `path_set_writeback` boundary engine. |
| [dol_issue10_design.md](dol_issue10_design.md) | Recursive wrapping + flat store view (designed, code pending); reuses the #16 engine. |
| [dol_content_metadata_bifurcation.md](dol_content_metadata_bifurcation.md) | The content/metadata split-store problem (feeds #80). |

### Operational / peripheral

| Doc | Contents |
|---|---|
| [code-quality-improvements.md](code-quality-improvements.md) | Tech-debt tracker (dead code, coverage gaps). Feeds #94. |
| [CHANGELOG.md](CHANGELOG.md) | Change log. |
| [frontend_dol_ideas.md](frontend_dol_ideas.md) | `zoddal`: the TypeScript/Zod incarnation of the dol idea. |
| [generate_llms_txt_instruction.md](generate_llms_txt_instruction.md) | How the repo's `llms.txt` files are generated. |

> The **fleet usage ledger** ([#91](https://github.com/i2mint/dol/issues/91)) lives
> in gitignored `misc/data/` (see its `README.md`). Each `fleet_ledger scan` is one
> immutable, timestamped record: dol's **89 direct dependents** at full commit pins,
> their imports / AST-verified subclasses / call-site kwargs, a generated zero-usage
> report, and the pre-PR test-gate order. `baseline` captures each dependent's suite
> result — the reference a candidate is judged against — and `pin restore` freezes
> the fleet at a scan. Scans accumulate; figures below are from the 2026-08-21 scan.
