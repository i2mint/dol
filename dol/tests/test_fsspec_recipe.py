"""The README's "Any fsspec URL is a dol store" recipe (#217).

Skipped when fsspec is not installed: dol itself never depends on it.
"""

import uuid

import pytest

fsspec = pytest.importorskip("fsspec")

from dol import KeyCodecs, Pipe, ValueCodecs


def _json_store(mapper):
    return Pipe(
        ValueCodecs.str_to_bytes(), ValueCodecs.json(), KeyCodecs.suffixed(".json")
    )(mapper)


@pytest.fixture
def mapper():
    return fsspec.get_mapper(f"memory://dol-recipe-{uuid.uuid4().hex}")


def test_round_trip_through_an_fsspec_mapper(mapper):
    store = _json_store(mapper)
    store["config"] = {"a": 1}
    store["sub/b"] = [1, 2]
    assert store["config"] == {"a": 1}
    assert "config" in store
    assert sorted(store) == ["config", "sub/b"]
    assert sorted(mapper) == ["config.json", "sub/b.json"]
    assert mapper["config.json"] == b'{"a": 1}'
    del store["config"]
    assert list(store) == ["sub/b"]


def test_a_missing_key_raises_key_error(mapper):
    store = _json_store(mapper)
    with pytest.raises(KeyError):
        store["nope"]


def test_the_bytes_codec_is_needed_under_json(mapper):
    store = Pipe(ValueCodecs.json(), KeyCodecs.suffixed(".json"))(mapper)
    with pytest.raises(TypeError):
        store["config"] = {"a": 1}
