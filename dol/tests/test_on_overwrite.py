"""Tests for ``dol.on_overwrite`` (issue #161)."""

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
