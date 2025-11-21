"""Tests for dol.errors module"""

import pytest
from dol.errors import (
    items_with_caught_exceptions,
    _assert_condition,
    NotValid,
    KeyValidationError,
    NoSuchKeyError,
    NotAllowed,
    OperationNotAllowed,
    ReadsNotAllowed,
    WritesNotAllowed,
    DeletionsNotAllowed,
    IterationNotAllowed,
    OverWritesNotAllowedError,
    AlreadyExists,
    MethodNameAlreadyExists,
    MethodFuncNotValid,
    SetattrNotAllowed,
)
from collections.abc import Mapping


class OddKeyErrorMapping(Mapping):
    """A test mapping that raises KeyError for odd keys"""

    def __init__(self, n=10):
        self.n = n

    def __iter__(self):
        yield from range(2, self.n)

    def __len__(self):
        return self.n - 2

    def __getitem__(self, k):
        if k % 2 == 0:
            return k * 10
        else:
            raise KeyError(f"Key {k} is odd")


def test_items_with_caught_exceptions_basic():
    """Test basic functionality of items_with_caught_exceptions"""
    test_map = OddKeyErrorMapping(10)
    result = list(items_with_caught_exceptions(test_map))
    # Should only get even keys (2, 4, 6, 8)
    assert result == [(2, 20), (4, 40), (6, 60), (8, 80)]


def test_items_with_caught_exceptions_with_callback():
    """Test items_with_caught_exceptions with a callback"""
    test_map = OddKeyErrorMapping(8)
    caught_keys = []

    def callback(k, e):
        caught_keys.append(k)

    result = list(items_with_caught_exceptions(test_map, callback=callback))
    assert result == [(2, 20), (4, 40), (6, 60)]
    # Odd keys should have been caught: 3, 5, 7
    assert caught_keys == [3, 5, 7]


def test_items_with_caught_exceptions_with_index_callback():
    """Test callback with index parameter"""
    test_map = OddKeyErrorMapping(6)
    caught_indices = []

    def callback(i):
        caught_indices.append(i)

    result = list(items_with_caught_exceptions(test_map, callback=callback))
    assert result == [(2, 20), (4, 40)]
    # Indices where exceptions occurred: 1 (key 3), 3 (key 5)
    assert caught_indices == [1, 3]


def test_items_with_caught_exceptions_yield_callback_output():
    """Test yielding callback output"""
    test_map = OddKeyErrorMapping(6)

    def callback(k):
        return f"error_{k}"

    result = list(
        items_with_caught_exceptions(
            test_map, callback=callback, yield_callback_output=True
        )
    )
    # Should yield both successful items and callback outputs
    assert (2, 20) in result
    assert (4, 40) in result
    assert "error_3" in result
    assert "error_5" in result


def test_items_with_caught_exceptions_specific_exceptions():
    """Test catching specific exception types"""

    class SpecialMapping(Mapping):
        def __iter__(self):
            yield from ["a", "b", "c"]

        def __len__(self):
            return 3

        def __getitem__(self, k):
            if k == "a":
                return "value_a"
            elif k == "b":
                raise KeyError("Key error")
            else:
                raise ValueError("Value error")

    # Only catch KeyError - should catch 'b' but raise exception on 'c'
    with pytest.raises(ValueError, match="Value error"):
        list(
            items_with_caught_exceptions(
                SpecialMapping(), catch_exceptions=(KeyError,)
            )
        )


def test_assert_condition():
    """Test _assert_condition helper function"""
    # Should not raise when condition is True
    _assert_condition(True, "Should not raise")

    # Should raise AssertionError when condition is False
    with pytest.raises(AssertionError, match="Test error"):
        _assert_condition(False, "Test error")

    # Should raise custom error class
    with pytest.raises(ValueError, match="Custom error"):
        _assert_condition(False, "Custom error", ValueError)


# Test exception classes hierarchy
def test_not_valid_exception():
    """Test NotValid exception is both ValueError and TypeError"""
    exc = NotValid("test")
    assert isinstance(exc, ValueError)
    assert isinstance(exc, TypeError)


def test_key_validation_error():
    """Test KeyValidationError inherits from NotValid"""
    exc = KeyValidationError("invalid key")
    assert isinstance(exc, NotValid)
    assert isinstance(exc, ValueError)


def test_no_such_key_error():
    """Test NoSuchKeyError inherits from KeyError"""
    exc = NoSuchKeyError("missing")
    assert isinstance(exc, KeyError)


def test_operation_not_allowed():
    """Test OperationNotAllowed hierarchy"""
    exc = OperationNotAllowed("not allowed")
    assert isinstance(exc, NotAllowed)
    assert isinstance(exc, NotImplementedError)


def test_specific_operation_not_allowed_exceptions():
    """Test specific operation exception types"""
    reads = ReadsNotAllowed("no reads")
    assert isinstance(reads, OperationNotAllowed)

    writes = WritesNotAllowed("no writes")
    assert isinstance(writes, OperationNotAllowed)

    deletes = DeletionsNotAllowed("no deletes")
    assert isinstance(deletes, OperationNotAllowed)

    iteration = IterationNotAllowed("no iteration")
    assert isinstance(iteration, OperationNotAllowed)

    overwrites = OverWritesNotAllowedError("no overwrites")
    assert isinstance(overwrites, OperationNotAllowed)


def test_already_exists():
    """Test AlreadyExists exception"""
    exc = AlreadyExists("exists")
    assert isinstance(exc, ValueError)


def test_method_name_already_exists():
    """Test MethodNameAlreadyExists exception"""
    exc = MethodNameAlreadyExists("method exists")
    assert isinstance(exc, AlreadyExists)


def test_method_func_not_valid():
    """Test MethodFuncNotValid exception"""
    exc = MethodFuncNotValid("invalid method")
    assert isinstance(exc, NotValid)


def test_setattr_not_allowed():
    """Test SetattrNotAllowed exception"""
    exc = SetattrNotAllowed("cannot set")
    assert isinstance(exc, NotAllowed)
