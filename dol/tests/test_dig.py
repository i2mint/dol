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
