### `dol` — Python Data Object Layer

These documents describe **dol**, a Python library for building uniform `dict`-like (MutableMapping) interfaces to any storage backend. The core idea: map CRUD operations onto Python's native mapping protocol (`__getitem__`, `__setitem__`, `__delitem__`, `__iter__`).

- **`general_design.md`** (~220 lines) — Language-agnostic architecture of dol. Covers the key insight (language-native KV interfaces), the interface hierarchy (`Collection → KvReader → KvPersister → Store`), the middleware principle (dol sits between domain code and storage), and the KV transform pipeline (`key_of_id`, `id_of_key`, `obj_of_data`, `data_of_obj`).

- **`python_design.md`** (~540 lines) — Python-specific implementation details. Covers the class hierarchy (rooted in `collections.abc`), the `Store` class with its 4 transform hooks, `wrap_kvs` (the core transformation function), key/value codecs (`ValueCodecs`, `KeyCodecs`), path-based stores, caching (`cache_this`), and the `Pipe` composition utility. The reference implementation doc.

- **`issues_and_discussions.md`** (~260 lines) — Themes from dol's GitHub issues/discussions. Major topics: `wrap_kvs` design tensions (signature-based conditioning, `self` not being the wrapped instance, recursive wrapping), the builtin codec ecosystem, path key handling, store composition patterns, and API ergonomics debates.

- **`code-quality-improvements.md`** (~230 lines) — Technical debt tracker for dol: dead code, unused parameters, incomplete implementations, test coverage gaps. Operational/maintenance reference.

