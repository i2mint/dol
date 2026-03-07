# dol Coding Conventions

## Interface Choices

- **New read-only stores**: subclass `KvReader` (from `dol.base`), not `Mapping` directly.
- **New read-write stores**: subclass `KvPersister` (from `dol.base`).
- **Avoid subclassing `Store` directly** unless you need the `_id_of_key` / `_key_of_id` / `_data_of_obj` / `_obj_of_data` hook protocol. Prefer `wrap_kvs` instead — it's more composable and doesn't create a new class in the inheritance chain.

## Naming Conventions

- Transform functions use `X_of_Y` naming: `key_of_id` (given `_id`, return `key`), `id_of_key` (given `key`, return `_id`). Follow this in new code.
- Use `_id` for inner/backend keys, `k` or `key` for outer/interface keys.
- Use `data` for serialized/raw backend values, `obj` or `v` or `value` for outer Python objects.

## `wrap_kvs` Usage

- Prefer `wrap_kvs` over subclassing when adding key/value transforms.
- Use `obj_of_data`/`data_of_obj` (no key context) when the transform is the same for all values.
- Use `postget`/`preset` (with key context) only when the transform depends on the key (e.g., file extension).
- **Do not use `bytes.decode` directly** as `obj_of_data` — use `lambda b: b.decode()` instead (signature-inference bug, Issue #9).

## Store Construction

- Always test new stores with `dict()` as backend before using real storage.
- Make stores parametrizable: accept `store=None` (or `store_factory=dict`) as a parameter.
- Return the store from a factory function rather than hardcoding the backend.

## Purity and Dependencies

- Core `dol` has no external dependencies. Keep new core code dependency-free.
- If a new feature needs an external dependency, put it in a separate module with a clear optional import and a helpful error message if the dependency is missing.

## Codecs

- Use `ValueCodecs` and `KeyCodecs` from `dol.kv_codecs` for common encoders/decoders.
- For custom codecs, use `dol.trans.ValueCodec` / `KeyCodec` dataclasses (not ad-hoc lambdas) when the codec needs to be reused or composed.
- Compose codecs with `+` operator: `ValueCodecs.pickle() + ValueCodecs.gzip()`.

## Caching

- Use `@cache_this` for expensive properties/methods, specifying an explicit `cache=` store if persistence across sessions is needed.
- Use `cache_vals(store)` to add an in-memory read cache to a slow store.
- Use `store_cached(cache_store)(func)` to persist function results.

## Testing

- Tests live in `dol/tests/`.
- Doctests in module docstrings are the primary documentation for functions — keep them runnable.
- Use `utils_for_tests.py` for shared test fixtures.

## No Silent Failures

- If a transform function is passed that will fail silently, raise an informative error early (e.g., in `__init__` or at decoration time).
- Prefer explicit errors over silent incorrect behavior.
