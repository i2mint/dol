# dol facade skeleton (fuller version)

One file split by the module each part would live in, for an imaginary `widgetdol`. It runs as is: save it as a `.py` file and run `pytest` on it (the backends are imaginary, so only the fake source is exercised). The layering, contract and rules behind each part are in [../SKILL.md](../SKILL.md).

## errors.py: one translation seam

The only code in the raw layer that catches backend exceptions. Catch the backend's specific not-found exception; `LookupError` stands in for it here so the tests below can use a dict.

```python
from functools import wraps


class WidgetError(Exception):
    """Base of everything this package raises that is not a KeyError."""


class MissingBackendError(ImportError):
    """No usable backend (or the one asked for is not installed)."""


def translate_widget_errors(func):
    """The only place in the raw layer that catches backend exceptions."""

    @wraps(func)
    def wrapper(self, k, *args, **kwargs):
        try:
            return func(self, k, *args, **kwargs)
        except LookupError as e:  # stand-in for the backend's not-found error
            raise KeyError(k) from e
        # anything else (auth, permission, transport) propagates unchanged

    return wrapper
```

## Backends: runtime-detected, optional, pluggable

Availability is probed without importing; the import happens only when a backend is chosen. Unknown names raise; missing ones raise with the install command.

```python
import importlib.util
import shutil


class SdkSource:
    name = "sdk"
    install_hint = 'pip install "widgetdol[sdk]"'

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("widget_sdk") is not None  # probe, no import

    def __init__(self, location):
        import widget_sdk  # imported only when chosen

        self._client = widget_sdk.connect(location)

    def list_ids(self): return self._client.list()
    def exists(self, k): return self._client.exists(k)
    def get(self, k): return self._client.get(k)


class CliSource:
    name = "widget-cli"
    install_hint = "brew install widget-cli   (or: apt install widget-cli)"

    @staticmethod
    def is_available() -> bool:
        return shutil.which("widget-cli") is not None

    def __init__(self, location): ...


BACKENDS = {s.name: s for s in (SdkSource, CliSource)}  # preference order


def _resolve_backend(backend):
    if callable(backend):
        return backend  # escape hatch: any backend(location) -> source
    if backend is None:
        for source in BACKENDS.values():
            if source.is_available():
                return source
        hints = "\n".join(f"  - {n}: {s.install_hint}" for n, s in BACKENDS.items())
        raise MissingBackendError(f"No widget backend is available. Install one of:\n{hints}")
    if backend not in BACKENDS:
        raise ValueError(f"Unknown widget backend {backend!r}. Known: {list(BACKENDS)}")
    source = BACKENDS[backend]
    if not source.is_available():
        raise MissingBackendError(f"The {backend!r} backend is not available. {source.install_hint}")
    return source


def check_requirements(*, verbose: bool = True) -> dict:
    available = {n: s.is_available() for n, s in BACKENDS.items()}
    if verbose:
        for n, ok in available.items():
            print(f"{n:12} {'available' if ok else 'missing: ' + BACKENDS[n].install_hint}")
    return available
```

## base.py: the raw reader

Lazy source, keys-only iteration, a native `__contains__`. `__len__` is inherited here (it counts by iterating); override it when the backend can count natively, or make it raise `TypeError` when the count is unbounded. A persister would subclass this and `KvPersister`, as in the SKILL.md skeleton.

```python
from collections.abc import Iterator
from functools import cached_property

from dol import KvReader


class WidgetReader(KvReader):
    def __init__(self, location, *, backend=None):
        self.location = location
        self._backend = backend

    @cached_property
    def source(self):  # lazy: nothing is opened until first use
        return _resolve_backend(self._backend)(self.location)

    def __iter__(self) -> Iterator[str]:
        yield from self.source.list_ids()

    def __contains__(self, k) -> bool:
        return self.source.exists(k)

    @translate_widget_errors
    def __getitem__(self, k) -> bytes:
        return self.source.get(k)
```

## ops.py: a leaf operation as a free function

It takes the store first, so it can see whether it was handed a wrapper and refuse unless the caller says the leaf's coordinates are what they want (the `sqldol.ops.query` pattern, dol#83). `isinstance(store, dol.Store)` is a reliable test here because the function holds the store itself.

```python
from dol import Store


class WrappedStoreError(ValueError):
    """A leaf operation was handed a wrapped store."""


def raw_listing(store, *, leaf_coordinates: bool = False):
    """The backend's own listing, in the leaf's keys (not the caller's)."""
    if isinstance(store, Store) and not leaf_coordinates:
        raise WrappedStoreError(
            f"{type(store).__name__} wraps the reader: its keys are not the leaf's. "
            "Pass leaf_coordinates=True if the leaf's keys are what you want."
        )
    leaf = store
    while isinstance(leaf, Store):
        leaf = leaf.store
    return list(leaf.source.list_ids())
```

## Tests: a fake source through the backend seam

The fake is passed exactly where a real backend would be, so the contract tests run on any machine. Real-backend tests would add `pytest.mark.skipif(not SdkSource.is_available(), reason=...)` and read a tiny generated fixture committed with its generator.

```python
import pytest


class _DictSource:
    def __init__(self, data): self.data = dict(data)

    @classmethod
    def factory(cls, data):
        return lambda location: cls(data)

    def list_ids(self): return list(self.data)
    def exists(self, k): return k in self.data
    def get(self, k): return self.data[k]  # KeyError is a LookupError, like the real one


def test_mapping_contract():
    r = WidgetReader("x", backend=_DictSource.factory({"a": b"1", "b": b"2"}))
    assert list(r) == ["a", "b"] and len(r) == 2
    assert "a" in r and "z" not in r
    assert r["a"] == b"1"
    with pytest.raises(KeyError):
        r["z"]
    assert r.get("z") is None


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Known"):
        WidgetReader("x", backend="nope").source


def test_ops_refuses_wrapped_store():
    from dol import KeyCodecs
    r = WidgetReader("x", backend=_DictSource.factory({"p/a": b"1"}))
    w = KeyCodecs.prefixed("p/")(r)
    assert list(w) == ["a"]
    with pytest.raises(WrappedStoreError):
        raw_listing(w)
    assert raw_listing(w, leaf_coordinates=True) == ["p/a"]
    assert raw_listing(r) == ["p/a"]


def test_check_requirements_lists_every_backend():
    assert set(check_requirements(verbose=False)) == set(BACKENDS)
```
