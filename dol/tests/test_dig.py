"""Tests for dol.dig module (layer introspection utilities)"""

import pytest
from dol.dig import (
    get_first_attr_found,
    recursive_get_attr,
    re_get_attr,
    dig_up,
    store_trans_path,
    print_trans_path,
    last_element,
    inner_most,
    unravel_key,
    inner_most_key,
    next_layer,
    recursive_calls,
    layers,
    trace_getitem,
    not_found,
    no_default,
)
from io import StringIO
import sys


class SimpleStore:
    """A simple test store with nested structure"""

    def __init__(self, data, inner_store=None):
        self.data = data
        self.attr1 = "value1"
        if inner_store:
            self.store = inner_store

    def _id_of_key(self, key):
        return f"id_{key}"

    def _data_of_obj(self, obj):
        return f"data_{obj}"


def test_get_first_attr_found():
    """Test getting first found attribute"""
    store = SimpleStore({})
    store.attr2 = "value2"

    # Should find first existing attribute
    result = get_first_attr_found(store, ["nonexistent", "attr1", "attr2"])
    assert result == "value1"

    # Should find second if first doesn't exist
    result = get_first_attr_found(store, ["nonexistent", "attr2"])
    assert result == "value2"


def test_get_first_attr_found_with_default():
    """Test get_first_attr_found with default value"""
    store = SimpleStore({})

    # Should return default when no attributes found
    result = get_first_attr_found(store, ["x", "y", "z"], default="default_value")
    assert result == "default_value"


def test_get_first_attr_found_no_default():
    """Test get_first_attr_found raises when no default and no attr found"""
    store = SimpleStore({})

    with pytest.raises(AttributeError, match="None of the attributes were found"):
        get_first_attr_found(store, ["x", "y", "z"])


def test_recursive_get_attr():
    """Test recursive attribute lookup"""
    inner_store = SimpleStore({})
    inner_store.deep_attr = "deep_value"

    outer_store = SimpleStore({}, inner_store=inner_store)

    # Should find attribute in current store
    result = recursive_get_attr(outer_store, "attr1")
    assert result == "value1"

    # Should recursively find in inner store
    result = recursive_get_attr(outer_store, "deep_attr")
    assert result == "deep_value"

    # Should return default if not found
    result = recursive_get_attr(outer_store, "nonexistent", default="my_default")
    assert result == "my_default"


def test_re_get_attr_and_dig_up_aliases():
    """Test that re_get_attr and dig_up are aliases for recursive_get_attr"""
    assert re_get_attr is recursive_get_attr
    assert dig_up is recursive_get_attr


def test_store_trans_path():
    """Test store transformation path"""
    inner_store = SimpleStore({})
    outer_store = SimpleStore({}, inner_store=inner_store)

    result = list(store_trans_path(outer_store, "key", "_id_of_key"))
    # Should yield transformed keys at each level
    assert "id_key" in result


def test_print_trans_path(capsys):
    """Test printing transformation path"""
    store = SimpleStore({})

    # Capture stdout
    print_trans_path(store, "test", "_id_of_key")
    captured = capsys.readouterr()
    assert "test" in captured.out
    assert "id_test" in captured.out


def test_print_trans_path_with_type(capsys):
    """Test printing transformation path with type info"""
    store = SimpleStore({})

    print_trans_path(store, "test", "_id_of_key", with_type=True)
    captured = capsys.readouterr()
    assert "<class 'str'>" in captured.out


def test_last_element():
    """Test getting last element from generator"""
    gen = (x for x in [1, 2, 3, 4, 5])
    assert last_element(gen) == 5

    # Empty generator should return None
    gen = (x for x in [])
    assert last_element(gen) is None


def test_inner_most():
    """Test getting innermost transformation"""
    store = SimpleStore({})
    result = inner_most(store, "test", "_id_of_key")
    # Should return the final transformed value
    assert result is not None


def test_next_layer():
    """Test getting next layer of store"""
    inner_store = SimpleStore({})
    outer_store = SimpleStore({}, inner_store=inner_store)

    # Should return inner store
    result = next_layer(outer_store)
    assert result is inner_store

    # Should return not_found if no next layer
    result = next_layer(inner_store)
    assert result is not_found


def test_recursive_calls():
    """Test recursive function calls generator"""
    # Test with simple increment until sentinel
    def increment(x):
        if x >= 5:
            return not_found
        return x + 1

    result = list(recursive_calls(increment, 0))
    assert result == [0, 1, 2, 3, 4, 5]


def test_layers():
    """Test getting all layers of a store"""
    inner_store = SimpleStore({})
    middle_store = SimpleStore({}, inner_store=inner_store)
    outer_store = SimpleStore({}, inner_store=middle_store)

    result = layers(outer_store)
    assert len(result) == 3
    assert result[0] is outer_store
    assert result[1] is middle_store
    assert result[2] is inner_store


def test_trace_getitem():
    """Test tracing getitem operations through layers"""
    from dol.trans import wrap_kvs

    # Create a simple layered store as shown in docstring
    d = {"a.num": "1000", "b.num": "2000"}

    s = wrap_kvs(
        d,
        key_of_id=lambda x: x[: -len(".num")],
        id_of_key=lambda x: x + ".num",
        obj_of_data=lambda x: int(x),
        data_of_obj=lambda x: str(x),
    )

    ss = wrap_kvs(
        s,
        key_of_id=lambda x: x.upper(),
        id_of_key=lambda x: x.lower(),
    )

    # Trace should show transformation through layers
    trace = list(trace_getitem(ss, "A"))
    assert len(trace) > 0

    # Check that trace includes _id_of_key and __getitem__ steps
    methods = [method for _, method, _ in trace]
    assert "_id_of_key" in methods
    assert "__getitem__" in methods


def test_unravel_key():
    """Test key unraveling (specialized store_trans_path)"""
    inner_store = SimpleStore({})
    outer_store = SimpleStore({}, inner_store=inner_store)

    result = list(unravel_key(outer_store, "mykey"))
    # Should show key transformations
    assert len(result) > 0


def test_inner_most_key():
    """Test getting innermost key transformation"""
    store = SimpleStore({})

    result = inner_most_key(store, "test")
    # Should return final key transformation or None
    assert result is None or isinstance(result, str)


# -------------------------------------------------------------------------------------
# Resolution failures must be loud, not None
#
# ``inner_most_key`` used to return ``None`` when no layer of the chain defined
# ``_id_of_key``. Callers use the result as a key or a path, so the ``None`` surfaced far
# from its cause -- as a URL ending in ``/None``, or as
# ``TypeError: expected str, bytes or os.PathLike object, not NoneType`` inside
# ``MakeMissingDirsStoreMixin``.


class _NoKeyMethods:
    """A leaf with no ``_id_of_key`` -- the shape that used to yield a silent ``None``."""

    def __getitem__(self, k):
        return k


def test_inner_most_raises_when_no_layer_defines_the_method():
    with pytest.raises(AttributeError) as excinfo:
        inner_most_key(_NoKeyMethods(), "some_key")
    msg = str(excinfo.value)
    assert "_id_of_key" in msg
    assert "some_key" in msg

    with pytest.raises(AttributeError):
        inner_most_key({}, "some_key")


def test_inner_most_default_opts_out_of_raising():
    assert inner_most_key(_NoKeyMethods(), "k", default=None) is None
    assert inner_most_key({}, "k", default="fallback") == "fallback"
    # default is ignored when resolution succeeds
    assert inner_most_key(SimpleStore({}), "k", default="fallback") == "id_k"


def test_inner_most_still_resolves_through_a_real_wrap():
    from dol import KeyCodecs

    store = KeyCodecs.prefixed("a/")({"a/b": 1})
    assert inner_most_key(store, "b") == "a/b"
    assert list(unravel_key(store, "b")) == ["a/b", "a/b"]


def test_store_trans_path_recurses_with_the_given_method():
    """``inner_most_val`` used to apply ``_data_of_obj`` at the top layer and
    ``_id_of_key`` at every deeper one, because the recursion hardcoded ``unravel_key``."""
    from dol import wrap_kvs
    from dol.dig import inner_most_val

    store = wrap_kvs(
        wrap_kvs({"k": 1}, data_of_obj=lambda v: v * 10), data_of_obj=lambda v: v + 1
    )
    assert inner_most_val(store, 5) == 60  # (5 + 1) * 10, both layers applied


def test_inner_most_key_is_exported_from_dol():
    """s3dol and other adapters need this as public API, not a submodule import."""
    import dol

    assert dol.inner_most_key is inner_most_key
    assert hasattr(dol, "unravel_key")


def test_make_missing_dirs_store_mixin_creates_dirs(tmpdir):
    """Regression: this recovery path raised ``TypeError`` twice over -- once from the
    ``None`` key, once from passing keyword-only ``verbose`` positionally."""
    import os
    from dol.filesys import MakeMissingDirsStoreMixin, FileBytesPersister

    class S(MakeMissingDirsStoreMixin, FileBytesPersister):
        pass

    rootdir = str(tmpdir)
    filepath = os.path.join(rootdir, "deep", "deeper", "f.bin")
    S(rootdir)[filepath] = b"hello"
    with open(filepath, "rb") as fp:
        assert fp.read() == b"hello"
