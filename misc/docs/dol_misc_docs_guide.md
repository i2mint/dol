### `dol` — Python Data Object Layer

These documents describe **dol**, a Python library for building uniform `dict`-like (MutableMapping) interfaces to any storage backend. The core idea: map CRUD operations onto Python's native mapping protocol (`__getitem__`, `__setitem__`, `__delitem__`, `__iter__`).

- **`general_design.md`** (~220 lines) — Language-agnostic architecture of dol. Covers the key insight (language-native KV interfaces), the interface hierarchy (`Collection → KvReader → KvPersister → Store`), the middleware principle (dol sits between domain code and storage), and the KV transform pipeline (`key_of_id`, `id_of_key`, `obj_of_data`, `data_of_obj`).

- **`dol_design.md`** (~540 lines) — Python-specific implementation details. Covers the class hierarchy (rooted in `collections.abc`), the `Store` class with its 4 transform hooks, `wrap_kvs` (the core transformation function), key/value codecs (`ValueCodecs`, `KeyCodecs`), path-based stores, caching (`cache_this`), and the `Pipe` composition utility. The reference implementation doc. *(Some line-cites are stale; see `dol_architecture_map.md` for code-verified numbers.)*

- **`dol_architecture_map.md`** (~600 lines) — Code-verified structural/mechanical map of the current source: per-module table + internal dependency graph, the exact public API surface, the class hierarchy, and a deep dive on the `wrap_kvs`/`store_decorator`/codec machinery (including the precise signature-conditioning logic behind Issue #9). Ends with a ranked tech-debt list and "notes for dev-skill authors." **Start here when refactoring or building agent tooling on dol.**

- **`issues_and_discussions.md`** (~270 lines) — Themes from dol's GitHub issues/discussions. Major topics: `wrap_kvs` design tensions (signature-based conditioning, `self` not being the wrapped instance, recursive wrapping), the builtin codec ecosystem, path key handling, store composition patterns, and API ergonomics debates. Kept roughly current (resolved issues flagged).

- **`dol_issues_report.md`** (~150 lines) — Actionable triage: which open issues are already resolved (close them) and a wave-by-wave **tackle order** for the rest, with an inter-issue dependency graph. The "what to work on next" companion to `issues_and_discussions.md`.

- **`dol_content_metadata_bifurcation.md`** (~450 lines) — Design study of the content/metadata split-store problem (a store whose values carry both payload and metadata).

- **`code-quality-improvements.md`** (~230 lines) — Technical debt tracker for dol: dead code, unused parameters, incomplete implementations, test coverage gaps. Operational/maintenance reference.

> A **local-only** ecosystem inventory lives under `misc/data/` (gitignored: it names
> private dependents). It maps dol's 76 local dependents, their dol usages (file:line),
> and a pre-PR test-gate order — regenerate with `misc/data/scan_dol_usages.py`.

