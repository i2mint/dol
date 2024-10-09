"""Test the tools module."""

# ------------------------------------------------------------------------------
# cache_method
import pytest
from functools import cached_property
from dol import cache_this
from dol.tools import CachedProperty, cache_property_method
from typing import Iterable


def test_cache_property_method(capsys):
    """
    The objective of this test is to test the cache_property_method function
    over some edge cases. Namely, what happens if we use try to cache a method
    that is already decorated by a property, cached_property, or cache_this?
    """

    class TestClass:
        def normal_method(self):
            print("normal_method called")
            return 1

        @property
        def property_method(self):
            print("property_method called")
            return 2

        @cached_property
        def cached_property_method(self):
            print("cached_property_method called")
            return 3

        @cache_this
        def cache_this_method(self):
            print("cache_this_method called")
            return 4

    cache_property_method(
        TestClass,
        [
            "normal_method",
            "property_method",
            "cached_property_method",
            "cache_this_method",
        ],
    )

    obj = TestClass()

    # Test normal method
    assert obj.normal_method == 1
    captured = capsys.readouterr()
    assert "normal_method called" in captured.out

    assert obj.normal_method == 1
    captured = capsys.readouterr()
    assert "normal_method called" not in captured.out  # Should not print again

    # Test property method
    assert obj.property_method == 2
    captured = capsys.readouterr()
    assert "property_method called" in captured.out

    assert obj.property_method == 2
    captured = capsys.readouterr()
    assert "property_method called" not in captured.out  # Should not print again

    # Test cached_property method
    assert obj.cached_property_method == 3
    captured = capsys.readouterr()
    assert "cached_property_method called" in captured.out

    assert obj.cached_property_method == 3
    captured = capsys.readouterr()
    assert "cached_property_method called" not in captured.out  # Should not print again

    # Test cache_this method
    assert obj.cache_this_method == 4
    captured = capsys.readouterr()
    assert "cache_this_method called" in captured.out

    assert obj.cache_this_method == 4
    captured = capsys.readouterr()
    assert "cache_this_method called" not in captured.out  # Should not print again
