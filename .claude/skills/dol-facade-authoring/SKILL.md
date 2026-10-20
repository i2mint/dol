---
name: dol-facade-authoring
description: "Author a dol facade over a new storage system: the backend itself, not codecs on top of one. Use when asked to 'write a dol store for X', start a 'new <x>dol package', 'make X a MutableMapping', build a 'dol facade/backend/adapter for <storage>', or 'wrap this DB/API/file format as a dict'; also when adding a read-only reader for a file format or service to an existing dol package. Covers where the code should live (a module bridged into an existing package's seam vs its own <x>dol package), the layering (connection, raw base store, codecs, one-call factory, malls), the Mapping contract each of the five methods must honour, error translation, why leaf capabilities are free functions or sibling stores and not methods (dol#83), optional backends and system dependencies with check_requirements, and offline testing with a fake source and a tiny generated fixture. For codecs, wrap_kvs, Pipe, filt_iter and caching on an existing backend, see dol-store-building; for dol's wrapping internals, see dol-dev-wrap-kvs."
---

# Authoring a dol facade over a new storage system

A facade here is the code that makes a storage system (a database, a cloud service, an API, a file format) look like a `Mapping` or `MutableMapping`. Codecs and key transforms then go on top with `wrap_kvs`; that part is [dol-store-building](../dol-store-building/SKILL.md) and is not repeated here. This skill is the part underneath: the raw store, its contract, and the package around it.

Worked examples to copy from, in order of usefulness: azuredol ([architecture](https://github.com/i2mint/azuredol/blob/master/misc/docs/architecture.md), [design decisions](https://github.com/i2mint/azuredol/blob/master/misc/docs/design_decisions.md)), s3dol ([ADRs](https://github.com/i2mint/s3dol/tree/master/misc/docs/decisions), especially 0004 on errors and 0011 on capabilities), and `sqldol.access` ([i2mint/sqldol#15](https://github.com/i2mint/sqldol/issues/15)) for a read-only reader with pluggable backends.

## 1. Decide where it lives, after searching

Search first: the fleet may already have it. Look at the i2mint `*dol` repos (`gh search repos dol --owner i2mint`, the local `i/dols/` folder, the package registry), and read the closest one before writing anything. There are dozens (`sqldol`, `odbcdol`, `s3dol`, `azuredol`, `mongodol`, `redisdol`, `sshdol`, `ftpdol`, `dropboxdol`, `pydrivedol`, ...).

Then pick the home:

- **A module in an existing dol package** when the new storage can be bridged into a seam that package already has. `sqldol.access` reads Microsoft Access files: rather than a new package, it offers a plain `AccessReader` Mapping and an `access_engine(path)` that copies the tables into SQLite and returns a SQLAlchemy engine, so every existing sqldol store works on an Access file. One module, one extra, no new concepts. Leave a comment naming the extraction trigger (for `access`: write-back, saved queries, more backends would justify an `mdbdol`).
- **A new `<x>dol` package** (i2mint org, next to its siblings) only when the storage has its own connection, auth, concepts or error taxonomy that no existing package's seam fits.
- **Many providers doing the same job** (several vector DBs, several blob services behind one interface): that is a multi-provider facade; follow the `facade-design` skill for vocabulary, capability honesty and the backend registry, and use this skill for each backend's store.

## 2. Layer it

Every public class belongs to exactly one layer (azuredol's rule):

1. **Connection / client.** Owns the expensive resource (SDK client, DB connection, parsed file). Built lazily (`functools.cached_property`), injectable as an argument so tests can pass a fake or an emulator-backed one. Credential resolution is one function with a documented cascade and an error that lists every source it tried. No module-level client cache.
2. **Raw base store.** Keys are the backend's natural names; values are bytes or the backend's native records. No codecs, no conveniences. Write the read-only class first (`KvReader` subclass), then the persister as a subclass that adds `__setitem__`/`__delitem__` (`class XStore(XReader, KvPersister)`). The reader is a real class, not a store with writes deleted: a read-only credential then never attempts a write.
3. **Codecs and key transforms**, by composition only: `wrap_kvs`, `Pipe`, `ValueCodecs`, `KeyCodecs`, `filt_iter`. Never by subclassing the raw store. See [dol-store-building](../dol-store-building/SKILL.md).
4. **One-call factory** for the common case: `x_store(name, *, client=None, value_codec=..., read_only=False)`, keyword-only options with defaults that work out of the box.

Several collections (tables, containers, buckets) become a **mall**: a store whose keys are collection names and whose values are stores (azuredol `AccountStore` → `ContainerStore`, sqldol `TablesDol`, `AccessReader` table → rows).

## 3. The Mapping contract, method by method

- `__getitem__(k)` raises `KeyError(k)` when the key is absent. Translate the backend's not-found error (`raise KeyError(k) from e`); never return `None`, print, or log and continue. Only not-found becomes `KeyError`: auth, permission, config and transport errors propagate as themselves, or a permission problem reads as a missing key.
- Put that translation in one place (a `translate_x_errors` decorator in `errors.py`), the only code in the raw layer that catches backend exceptions. Classify on what the backend reliably reports (s3dol ADR-0004: HEAD requests carry no error body, so "missing key" and "missing bucket" look the same and must not be guessed between).
- `__iter__` yields keys only, lazily, paginating if needed. Values never come out of `__iter__`.
- `__contains__` and `__len__` must be cheap and agree with `__iter__`. `KvReader`'s defaults scan every key; override them with a native existence check and a native count. If a count is unbounded (a container with billions of blobs), make `__len__` raise `TypeError` with the `sum(1 for _ in store)` workaround, as azuredol does, rather than a scan that occasionally explodes.
- No silent fallbacks: no empty store when the connection failed, no default value for a missing key, no auto-create unless a keyword asks for it.
- When per-key reads are costly, give `values()`/`items()` a bulk implementation (one query or one listing) instead of the inherited per-key loop.
- Destructive operations stay explicit: `__delitem__` on a mall does not cascade into a non-empty collection; a `force=True` free function does.
- Gotcha: `wrap_kvs` over a reader returns a `Store`, which is a `MutableMapping`. Writing to it fails with `AttributeError` at call time, not at wrap time. If read-only must be visible on the wrapped object, say so in the factory's docstring and return type.

## 4. Capabilities: free functions or sibling stores, not methods

dol wraps by delegation. A wrapper maps keys only in the five Mapping methods; any other method is forwarded to the leaf with the outer, unmapped key ([dol#83](https://github.com/i2mint/dol/issues/83)). So `store.url_for(k)`, `store.delete_many(keys)` or `store.sub(prefix)` silently address the wrong object as soon as a user applies a key codec. `wrapped_self` does not fix it reliably (s3dol ADR-0011 D1a).

- **The raw store carries no public method that takes a key**, at any argument position. A reflective test that inventories the public methods of each raw class keeps it that way (s3dol ADR-0011 D5).
- **Per-key capabilities become a sibling store** over the same key space whose `__getitem__` returns the capability (`BucketUrls[k] -> url`, `BucketHandles[k] -> handle`), or a handle object that takes its key at construction (azuredol `BlobHandle`). `sqldol.access.AccessColumns` (table → column names, beside `AccessReader`: table → rows) is a small example. `__getitem__` is the one thing dol maps correctly at every depth.
- **Leaf and bulk operations are free functions in an `ops` module** that take the store first: `query(store)`, `delete_many(store, keys)`. A free function can check whether it was handed a wrapper and refuse unless told (`sqldol.ops.query(store, leaf_coordinates=True)`, raising `WrappedStoreError` otherwise). Do not claim a free function is correct at any wrapper depth.
- Plain attributes for the underlying objects (`client`, `table`, `session`) are fine; document them as leaf escape hatches that speak the leaf's coordinates.

## 5. Optional backends and system dependencies

When the storage can be reached several ways (a pure-Python parser, a command-line tool, an ODBC driver), make the backend a keyword argument, as `sqldol.access` does:

- Each backend is a small class with a `name`, an `install_hint`, an `is_available()` that probes without importing (`importlib.util.find_spec`, `shutil.which`), and the few methods the reader needs. A `BACKENDS` dict lists them in preference order.
- `backend=None` takes the first available one; a name selects one; a callable is accepted as the escape hatch for anything not shipped. An unknown name raises `ValueError` listing the known ones. A known but missing backend raises an `ImportError` subclass whose message carries the install command.
- Python dependencies are extras (`pip install "sqldol[access]"`), imported inside the backend, never at module import. System tools are called as programs and named in the README with their install commands.
- `check_requirements(*, verbose=True)` returns `{name: available}` and prints how to get each missing one.

## 6. Testing without the real system

- Prototype the logic on a `dict` first (the dol convention; see dol-store-building).
- Inject a fake source through the same seam the real backend uses: `AccessReader(path, backend=_DictSource.factory(tables))`, or a fake client passed to the store. The Mapping contract tests (KeyError on miss, keys-only iteration, `len`/`in` consistent with `list`) run everywhere against it.
- Commit a tiny **generated** fixture together with the script that made it (`tests/test_data/tiny_jet4.mdb.gz` and `MkFixture.java` in sqldol). Never commit real or private data, however small: a fixture derived from a private source must not land in a public repo.
- Tests that need a real backend skip when it is not installed (`pytest.mark.skipif(not Source.is_available(), ...)`). Emulators (Azurite, moto) are a middle tier and can differ from production (s3dol ADR-0004). Live-service tests are opt-in through an environment variable and never run in CI by default.

## 7. Minimal skeleton

A read-only reader, its persister and the factory. A fuller version, with error translation, backends, `check_requirements`, an `ops` function and tests, is in [references/skeleton.md](references/skeleton.md).

```python
from collections.abc import Iterator
from functools import cached_property
from dol import KvReader, KvPersister, ValueCodecs
from widget_sdk import (
    WidgetNotFound,
    make_client as _default_client,
)  # the backend's own SDK


class WidgetReader(KvReader):
    """Keys: widget ids (str). Values: raw bytes. Read-only."""

    def __init__(self, client=None, *, namespace: str = "default"):
        self._client = client  # injected (tests pass a fake); built lazily otherwise
        self.namespace = namespace

    @cached_property
    def client(self):
        return self._client if self._client is not None else _default_client()

    def __iter__(self) -> Iterator[str]:
        yield from self.client.list_ids(self.namespace)  # keys only, lazily

    def __contains__(self, k) -> bool:
        return self.client.exists(self.namespace, k)  # one cheap call, not a scan

    def __len__(self) -> int:
        return self.client.count(self.namespace)  # or raise TypeError if unbounded

    def __getitem__(self, k) -> bytes:
        try:
            return self.client.get(self.namespace, k)
        except WidgetNotFound as e:  # not-found only; auth errors propagate
            raise KeyError(k) from e


class WidgetStore(WidgetReader, KvPersister):
    def __setitem__(self, k, v: bytes):
        self.client.put(self.namespace, k, v)

    def __delitem__(self, k):
        try:
            self.client.delete(self.namespace, k)
        except WidgetNotFound as e:
            raise KeyError(k) from e


def widget_store(
    namespace: str = "default",
    *,
    client=None,
    read_only: bool = False,
    value_codec=ValueCodecs.json() + ValueCodecs.str_to_bytes(),
):
    """A ready store: JSON values over the raw bytes store."""
    store = (WidgetReader if read_only else WidgetStore)(client, namespace=namespace)
    return value_codec(store) if value_codec else store
```

Routes: [dol-store-building](../dol-store-building/SKILL.md) (codecs and wrapping on top of the raw store), `python-storage` (Mapping basics, malls), `facade-design` (one interface over many providers), `architecture-first` (deciding the seams before the first commit).
