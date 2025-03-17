"""Test the tools module."""

# ------------------------------------------------------------------------------
# cache_method
import pytest
from unittest import mock
import tempfile
import os
from pathlib import Path

from dol import tools


class TestConfirmOverwrite:
    def test_confirm_overwrite_no_existing_key(self):
        # When key doesn't exist, should return value unchanged
        mapping = {}
        result = tools.confirm_overwrite(mapping, "key", "value")
        assert result == "value"

    def test_confirm_overwrite_same_value(self):
        # When key exists with same value, should return value unchanged
        mapping = {"key": "value"}
        result = tools.confirm_overwrite(mapping, "key", "value")
        assert result == "value"

    @mock.patch("builtins.input", return_value="new_value")
    def test_confirm_overwrite_confirmed(self, mock_input):
        # When user confirms overwrite, should return new value
        mapping = {"key": "old_value"}
        result = tools.confirm_overwrite(mapping, "key", "new_value")
        assert result == "new_value"
        mock_input.assert_called_once()

    @mock.patch("builtins.input", return_value="wrong_input")
    @mock.patch("builtins.print")
    def test_confirm_overwrite_rejected(self, mock_print, mock_input):
        # When user doesn't confirm, should return existing value
        mapping = {"key": "old_value"}
        result = tools.confirm_overwrite(mapping, "key", "new_value")
        assert result == "old_value"
        mock_input.assert_called_once()
        mock_print.assert_called_once()


class TestStoreAggregate:
    def test_store_aggregate_with_dict(self):
        content_store = {
            "file1.py": '"""Module docstring."""',
            "file2.py": "def foo(): pass",
        }
        result = tools.store_aggregate(
            content_store=content_store, kv_to_item=lambda k, v: f"{k}: {v}"
        )
        assert "file1.py: " in result
        assert "file2.py: " in result

    def test_store_aggregate_with_filters(self):
        content_store = {
            "file1.py": '"""Module docstring."""',
            "file2.txt": "Plain text",
            "file3.py": "def foo(): pass",
        }
        result = tools.store_aggregate(
            content_store=content_store,
            key_filter=lambda k: k.endswith(".py"),
            aggregator=", ".join,
        )
        assert "file1.py" in result
        assert "file3.py" in result
        assert "file2.txt" not in result

    def test_store_aggregate_with_custom_aggregator(self):
        content_store = {
            "file1": "content1",
            "file2": "content2",
        }
        result = tools.store_aggregate(
            content_store=content_store, aggregator=lambda items: len(list(items))
        )
        assert result == 2

    def test_store_aggregate_with_file_output(self):
        content_store = {
            "file1": "content1",
            "file2": "content2",
        }
        with tempfile.TemporaryDirectory() as tmpdir:

            class TestConvertToNumericalIfPossible:
                def test_convert_integer_string(self):
                    result = tools.convert_to_numerical_if_possible("123")
                    assert result == 123
                    assert isinstance(result, int)

                def test_convert_float_string(self):
                    result = tools.convert_to_numerical_if_possible("123.45")
                    assert result == 123.45
                    assert isinstance(result, float)

                def test_non_numerical_string(self):
                    result = tools.convert_to_numerical_if_possible("hello")
                    assert result == "hello"
                    assert isinstance(result, str)

                def test_empty_string(self):
                    result = tools.convert_to_numerical_if_possible("")
                    assert result == ""
                    assert isinstance(result, str)

                def test_infinity_string(self):
                    result = tools.convert_to_numerical_if_possible("infinity")
                    assert result == float("inf")
                    assert isinstance(result, float)

            class TestAskUserForValueWhenMissing:
                @mock.patch("builtins.input", return_value="user_value")
                def test_ask_user_when_missing_with_input(self, mock_input):
                    from dol.base import Store

                    store = {}
                    wrapped_store = tools.ask_user_for_value_when_missing(Store(store))

                    # Access the key which triggers __missing__
                    value = wrapped_store["missing_key"]

                    # After __missing__ is called, the value should be stored
                    assert store == {"missing_key": "user_value"}
                    assert value == "user_value"
                    mock_input.assert_called_once()

                @mock.patch("builtins.input", return_value="123")
                def test_ask_user_with_preprocessor(self, mock_input):
                    from dol.base import Store

                    store = {}
                    wrapped_store = tools.ask_user_for_value_when_missing(
                        Store(store),
                        value_preprocessor=tools.convert_to_numerical_if_possible,
                    )

                    # Access the key which triggers __missing__
                    value = wrapped_store["missing_key"]

                    # After __missing__ is called, the value should be stored
                    assert store == {"missing_key": 123}
                    assert value == 123
                    mock_input.assert_called_once()

                @mock.patch("builtins.input", return_value="")
                def test_ask_user_with_empty_input(self, mock_input):
                    from dol.base import Store

                    # Set up a mock for __missing__ to verify it gets called
                    store = {}
                    store_obj = Store(store)
                    original_missing = store_obj.__missing__

                    store_obj.__missing__ = mock.MagicMock(side_effect=original_missing)
                    wrapped_store = tools.ask_user_for_value_when_missing(store_obj)

                    # This should raise KeyError since we're returning empty string
                    with pytest.raises(KeyError):
                        wrapped_store["missing_key"]

                    # Verify the original __missing__ was called
                    store_obj.__missing__.assert_called_once()

                @mock.patch("builtins.input", return_value="user_value")
                def test_custom_message(self, mock_input):
                    from dol.base import Store

                    custom_msg = "Custom message for {k}:"
                    store = {}
                    wrapped_store = tools.ask_user_for_value_when_missing(
                        Store(store), on_missing_msg=custom_msg
                    )

                    wrapped_store["missing_key"]

                    # Check that input was called with the custom message
                    mock_input.assert_called_once_with(
                        custom_msg + " Value for missing_key:\n"
                    )

            class TestISliceStoreAdvanced:
                def test_islice_with_step(self):
                    original = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
                    sliceable = tools.iSliceStore(original)

                    # With step=2, should get every other value
                    assert list(sliceable[0:5:2]) == [1, 3, 5]

                def test_islice_with_negative_indices(self):
                    original = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
                    sliceable = tools.iSliceStore(original)

                    # Test with negative indices
                    assert list(sliceable[-3:-1]) == [3, 4]
                    assert list(sliceable[-3:]) == [3, 4, 5]

                def test_islice_single_item_access(self):
                    original = {"a": 1, "b": 2, "c": 3}
                    sliceable = tools.iSliceStore(original)

                    # To get a single item by position
                    item = next(sliceable[1:2])
                    assert item == 2

                def test_islice_out_of_bounds(self):
                    original = {"a": 1, "b": 2, "c": 3}
                    sliceable = tools.iSliceStore(original)

                    # Slicing beyond the end should just stop at the end
                    assert list(sliceable[2:10]) == [3]
                    # Empty slice when start is beyond length
                    assert list(sliceable[10:20]) == []

            class TestForestAdvanced:
                def test_forest_with_custom_filters(self):
                    d = {
                        "apple": {"kind": "fruit", "color": "red", "count": 5},
                        "banana": {"kind": "fruit", "color": "yellow", "count": 3},
                    }

                    # Only include keys that don't start with 'c'
                    forest = tools.Forest(
                        d,
                        is_leaf=lambda k, v: not isinstance(v, dict),
                        get_node_keys=lambda v: [
                            k for k in v.keys() if not k.startswith("c")
                        ],
                        get_src_item=lambda src, k: src[k],
                    )

                    assert list(forest["apple"]) == ["kind", "color"]
                    assert "count" not in list(forest["apple"])

                def test_forest_with_list_source(self):
                    # Test with a list as the source
                    lst = [
                        {"name": "item1", "value": 10},
                        {"name": "item2", "value": 20},
                        {"name": "item3", "value": 30},
                    ]

                    forest = tools.Forest(
                        lst,
                        is_leaf=lambda k, v: not isinstance(v, dict),
                        get_node_keys=lambda v: list(v.keys()),
                        get_src_item=lambda src, k: src[k],
                        forest_type=list,
                    )

                    # Since it's a list, we access by index
                    assert forest[0]["name"] == "item1"
                    assert forest[1]["value"] == 20
                    assert list(forest) == [0, 1, 2]

                def test_forest_with_leaf_transform(self):
                    d = {"a": "1", "b": "2", "c": "3"}

                    # Apply a transformation to leaf values
                    forest = tools.Forest(
                        d,
                        is_leaf=lambda k, v: True,  # all values are leaves
                        get_node_keys=lambda v: list(v.keys()),
                        get_src_item=lambda src, k: src[k],
                        leaf_trans=int,  # Convert string values to integers
                    )

                    assert forest["a"] == 1
                    assert forest["b"] == 2
                    assert isinstance(forest["c"], int)

    def test_islice_store_slicing(self):
        original = {"foo": "bar", "hello": "world", "alice": "bob"}
        sliceable = tools.iSliceStore(original)

        assert list(sliceable[0:2]) == ["bar", "world"]
        assert list(sliceable[-2:]) == ["world", "bob"]
        assert list(sliceable[:-1]) == ["bar", "world"]


class TestForest:
    def test_forest_with_dict(self):
        d = {
            "apple": {
                "kind": "fruit",
                "types": {"granny": {"color": "green"}, "fuji": {"color": "red"}},
            },
            "banana": {"kind": "fruit"},
        }

        forest = tools.Forest(
            d,
            is_leaf=lambda k, v: not isinstance(v, dict),
            get_node_keys=lambda v: list(v.keys()),
            get_src_item=lambda src, k: src[k],
        )

        assert list(forest) == ["apple", "banana"]
        assert forest["apple"]["kind"] == "fruit"
        assert list(forest["apple"]["types"]) == ["granny", "fuji"]
        assert forest["apple"]["types"]["granny"]["color"] == "green"

    def test_forest_to_dict(self):
        d = {
            "apple": {
                "kind": "fruit",
                "types": {
                    "granny": {"color": "green"},
                },
            }
        }

        forest = tools.Forest(
            d,
            is_leaf=lambda k, v: not isinstance(v, dict),
            get_node_keys=lambda v: list(v.keys()),
            get_src_item=lambda src, k: src[k],
        )

        dict_result = forest.to_dict()
        assert dict_result == d
