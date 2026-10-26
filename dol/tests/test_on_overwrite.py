"""Tests for ``dol.on_overwrite`` (issue #161)."""

import sys
import warnings

import pytest

from dol import on_overwrite
from dol.errors import OverWritesNotAllowedError


def test_raise_and_same_value_silent():
    d = on_overwrite(dict(a=1), policy="raise")
    d["a"] = 1
    d["b"] = 2
    with pytest.raises(OverWritesNotAllowedError):
        d["a"] = 3
    assert d["a"] == 1


def test_warn_writes_and_warns_once():
    d = on_overwrite(dict(a=1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d["a"] = 1
        assert not w
        d["a"] = 2
    assert len(w) == 1 and d["a"] == 2
    assert w[0].filename == __file__  # points at the caller


def test_overwrite_is_silent():
    d = on_overwrite(dict(a=1), policy="overwrite")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d["a"] = 2
    assert d["a"] == 2


def test_custom_same_and_class_decorator():
    @on_overwrite(policy="raise", same=lambda a, b: a.__name__ == b.__name__)
    class Reg(dict):
        pass

    r = Reg()
    r["f"] = len
    r["f"] = len
    with pytest.raises(OverWritesNotAllowedError):
        r["f"] = print


def test_bad_policy():
    with pytest.raises(ValueError):
        on_overwrite(dict(), policy="nope")


def test_overwrite_policy_returns_store_unwrapped():
    d = dict(a=1)
    assert on_overwrite(d, policy="overwrite") is d


class _AmbiguousBool:
    def __bool__(self):
        raise ValueError("truth value is ambiguous")


class _ArrayLike:
    """numpy-like: ``==`` returns something with an ambiguous truth value."""

    def __eq__(self, other):
        return _AmbiguousBool()

    __hash__ = object.__hash__


class _EqRaises:
    def __eq__(self, other):
        raise TypeError("no comparison")

    __hash__ = object.__hash__


@pytest.mark.parametrize("cls", [_ArrayLike, _EqRaises])
def test_default_same_does_not_crash(cls):
    x = cls()
    d = on_overwrite(dict(a=x), policy="raise")
    d["a"] = x  # identical object: same
    with pytest.raises(OverWritesNotAllowedError):
        d["a"] = cls()  # comparison fails -> treated as different


def test_warning_points_at_setitem_line():
    d = on_overwrite(dict(a=1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lineno = sys._getframe().f_lineno + 1
        d["a"] = 2
    assert (w[0].filename, w[0].lineno) == (__file__, lineno)


def test_factory_form():
    d = on_overwrite(policy="raise")(dict(a=1))
    with pytest.raises(OverWritesNotAllowedError):
        d["a"] = 2


def test_inner_store_mutated():
    inner = dict(a=1)
    d = on_overwrite(inner, policy="raise")
    inner["a"] = 5
    d["a"] = 5  # matches the current inner value: silent
    with pytest.raises(OverWritesNotAllowedError):
        d["a"] = 1


def test_same_argument_order():
    calls = []

    def same(existing, new):
        calls.append((existing, new))
        return False

    d = on_overwrite(dict(a="old"), same=same)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d["a"] = "new"
    assert calls == [("old", "new")]


def test_callable_policy():
    def no_shrink(k, existing, new):
        if new < existing:
            raise ValueError(f"{k}: {new} < {existing}")
        return new

    d = on_overwrite(dict(a=1), policy=no_shrink)
    d["a"] = 3
    with pytest.raises(ValueError, match="a: 2 < 3"):
        d["a"] = 2
    assert d["a"] == 3

    keep = on_overwrite(dict(a=1), policy=lambda k, existing, new: existing)
    keep["a"] = 9
    keep["b"] = 2  # policy only consulted for a differing existing value
    assert dict(keep) == {"a": 1, "b": 2}


def test_update_goes_through_policy():
    d = on_overwrite(dict(a=1), policy="raise")
    d.update(b=2)
    with pytest.raises(OverWritesNotAllowedError):
        d.update(a=3)
    assert dict(d) == {"a": 1, "b": 2}


def test_delete_then_write_bypasses_raise():
    """Documented: unlike ``disallow_overwrites``, deleting first lets a write through."""
    d = on_overwrite(dict(a=1), policy="raise")
    del d["a"]
    d["a"] = 3
    assert d["a"] == 3
