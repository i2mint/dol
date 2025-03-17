"""Test caching tools"""

import pytest
from functools import partial, cached_property
from collections import UserDict
from typing import Dict, Any

# Import the refactored implementations - adjust the import path as needed
from dol.caching import (
    cache_this,
    ExplicitKey,
    ApplyToMethodName,
    InstanceProp,
    ApplyToInstance,
    add_extension,
    cache_property_method,
)


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


# Utility classes for testing
class LoggedCache(UserDict):
    """Cache that logs get/set operations"""

    def __init__(self, name="cache"):
        super().__init__()
        self.name = name
        self.get_log = []
        self.set_log = []

    def __setitem__(self, key, value):
        self.set_log.append((key, value))
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        self.get_log.append(key)
        return super().__getitem__(key)


class MockValueCodecs:
    """Mock codec for testing"""

    class default:
        @staticmethod
        def pickle(store):
            """Mock function that would normally apply pickle encoding/decoding"""
            return store


# Test class with various key strategy examples
class TestClassWithKeyStrategies:
    def __init__(self):
        self.my_cache = LoggedCache("my_cache")
        self.key_name = "dynamic_key"
        self.compute_count = 0

    # Example using explicit key
    @cache_this(cache="my_cache", key=ExplicitKey("explicit_key"))
    def explicit_key_method(self):
        self.compute_count += 1
        return f"explicit_result_{self.compute_count}"

    # Example using function applied to method name
    @cache_this(
        cache="my_cache", key=ApplyToMethodName(lambda name: f"{name}_processed")
    )
    def method_name_key(self):
        self.compute_count += 1
        return f"method_name_result_{self.compute_count}"

    # Example using instance property
    @cache_this(cache="my_cache", key=InstanceProp("key_name"))
    def instance_prop_key(self):
        self.compute_count += 1
        return f"instance_prop_result_{self.compute_count}"

    # Example using function applied to instance
    @cache_this(
        cache="my_cache",
        key=ApplyToInstance(lambda instance: f"instance_{id(instance)}"),
    )
    def instance_func_key(self):
        self.compute_count += 1
        return f"instance_func_result_{self.compute_count}"

    # Example using string (implicit ExplicitKey)
    @cache_this(cache="my_cache", key="string_key")
    def string_key_method(self):
        self.compute_count += 1
        return f"string_key_result_{self.compute_count}"

    # Example using function with method name (implicit ApplyToMethodName)
    @cache_this(cache="my_cache", key=lambda name: f"{name}_func")
    def simple_func_key(self):
        self.compute_count += 1
        return f"simple_func_result_{self.compute_count}"

    # Example using function with instance (implicit ApplyToInstance)
    @cache_this(cache="my_cache", key=lambda self: f"self_{self.key_name}")
    def implicit_instance_func(self):
        self.compute_count += 1
        return f"implicit_instance_result_{self.compute_count}"

    # Example using a different cache
    def set_external_cache(self, external_cache):
        self.external_cache = external_cache

    @cache_this(cache="external_cache", key="external_key")
    def external_cache_method(self):
        self.compute_count += 1
        return f"external_result_{self.compute_count}"


# Test class for original use case with pickle extension
class PickleCached:
    def __init__(self):
        self._backend_store = LoggedCache("backend")
        self.cache = MockValueCodecs.default.pickle(self._backend_store)
        self.compute_count = 0

    @cache_this(cache="cache", key=ApplyToMethodName(lambda x: f"{x}.pkl"))
    def foo(self):
        self.compute_count += 1
        return f"foo_result_{self.compute_count}"


# Test class for data access pattern
class Dacc:
    def __init__(self):
        self.text_store = LoggedCache("text_store")
        self.json_store = LoggedCache("json_store")
        self.schema_description_key = "schema_description.txt"
        self.pricing_html_key = "pricing.html"
        self.schema_key = "schema.json"
        self.compute_count = 0

    @cache_this(cache="text_store", key=InstanceProp("schema_description_key"))
    def schema_description(self) -> str:
        self.compute_count += 1
        return f"schema_description_{self.compute_count}"

    @cache_this(cache="text_store", key=InstanceProp("pricing_html_key"))
    def pricing_page_html(self) -> str:
        self.compute_count += 1
        return f"pricing_html_{self.compute_count}"

    @cache_this(cache="json_store", key=InstanceProp("schema_key"))
    def schema(self) -> Dict[str, Any]:
        self.compute_count += 1
        return {"version": f"schema_{self.compute_count}"}


# Tests for basic key strategies
class TestKeyStrategies:
    def test_explicit_key(self):
        """Test explicit key strategy"""
        obj = TestClassWithKeyStrategies()

        # First access computes the value
        result1 = obj.explicit_key_method
        assert result1 == "explicit_result_1"
        assert "explicit_key" in obj.my_cache
        assert obj.my_cache["explicit_key"] == "explicit_result_1"

        # Second access uses cached value
        result2 = obj.explicit_key_method
        assert result2 == "explicit_result_1"  # Same result, no recomputation
        assert obj.compute_count == 1  # Only computed once

    def test_method_name_key(self):
        """Test applying function to method name"""
        obj = TestClassWithKeyStrategies()

        # First access computes the value
        result1 = obj.method_name_key
        assert result1 == "method_name_result_1"
        assert "method_name_key_processed" in obj.my_cache
        assert obj.my_cache["method_name_key_processed"] == "method_name_result_1"

        # Second access uses cached value
        result2 = obj.method_name_key
        assert result2 == "method_name_result_1"
        assert obj.compute_count == 1

    def test_instance_prop_key(self):
        """Test using instance property as key"""
        obj = TestClassWithKeyStrategies()

        # First access computes the value
        result1 = obj.instance_prop_key
        assert result1 == "instance_prop_result_1"
        assert "dynamic_key" in obj.my_cache
        assert obj.my_cache["dynamic_key"] == "instance_prop_result_1"

        # Second access uses cached value
        result2 = obj.instance_prop_key
        assert result2 == "instance_prop_result_1"
        assert obj.compute_count == 1

        # Changing the key property forces recomputation with new key
        obj.key_name = "new_dynamic_key"
        result3 = obj.instance_prop_key
        assert result3 == "instance_prop_result_2"
        assert "new_dynamic_key" in obj.my_cache
        assert obj.compute_count == 2

    def test_instance_func_key(self):
        """Test applying function to instance"""
        obj = TestClassWithKeyStrategies()

        # First access computes the value
        result1 = obj.instance_func_key
        expected_key = f"instance_{id(obj)}"
        assert result1 == "instance_func_result_1"
        assert expected_key in obj.my_cache
        assert obj.my_cache[expected_key] == "instance_func_result_1"

        # Second access uses cached value
        result2 = obj.instance_func_key
        assert result2 == "instance_func_result_1"
        assert obj.compute_count == 1

    def test_string_key(self):
        """Test string key (implicit ExplicitKey)"""
        obj = TestClassWithKeyStrategies()

        # First access computes the value
        result1 = obj.string_key_method
        assert result1 == "string_key_result_1"
        assert "string_key" in obj.my_cache
        assert obj.my_cache["string_key"] == "string_key_result_1"

        # Second access uses cached value
        result2 = obj.string_key_method
        assert result2 == "string_key_result_1"
        assert obj.compute_count == 1

    def test_simple_func_key(self):
        """Test function key (implicit ApplyToMethodName)"""
        obj = TestClassWithKeyStrategies()

        # First access computes the value
        result1 = obj.simple_func_key
        assert result1 == "simple_func_result_1"
        assert "simple_func_key_func" in obj.my_cache
        assert obj.my_cache["simple_func_key_func"] == "simple_func_result_1"

        # Second access uses cached value
        result2 = obj.simple_func_key
        assert result2 == "simple_func_result_1"
        assert obj.compute_count == 1

    def test_implicit_instance_func(self):
        """Test function with instance (implicit ApplyToInstance)"""
        obj = TestClassWithKeyStrategies()

        # First access computes the value
        result1 = obj.implicit_instance_func
        assert result1 == "implicit_instance_result_1"
        assert "self_dynamic_key" in obj.my_cache
        assert obj.my_cache["self_dynamic_key"] == "implicit_instance_result_1"

        # Second access uses cached value
        result2 = obj.implicit_instance_func
        assert result2 == "implicit_instance_result_1"
        assert obj.compute_count == 1

        # Changing the key property forces recomputation with new key
        obj.key_name = "new_dynamic_key"
        result3 = obj.implicit_instance_func
        assert result3 == "implicit_instance_result_2"
        assert "self_new_dynamic_key" in obj.my_cache
        assert obj.compute_count == 2

    def test_external_cache(self):
        """Test using an external cache"""
        obj = TestClassWithKeyStrategies()
        external_cache = LoggedCache("external")
        obj.set_external_cache(external_cache)

        # First access computes the value
        result1 = obj.external_cache_method
        assert result1 == "external_result_1"
        assert "external_key" in external_cache
        assert external_cache["external_key"] == "external_result_1"

        # Second access uses cached value
        result2 = obj.external_cache_method
        assert result2 == "external_result_1"
        assert obj.compute_count == 1


# Tests for original pickle extension use case
class TestPickleCached:
    def test_pickle_cache(self):
        """Test the original pickle cache use case"""
        obj = PickleCached()

        # First access computes the value
        result1 = obj.foo
        assert result1 == "foo_result_1"
        assert "foo.pkl" in obj._backend_store
        assert obj._backend_store["foo.pkl"] == "foo_result_1"

        # Second access uses cached value
        result2 = obj.foo
        assert result2 == "foo_result_1"
        assert obj.compute_count == 1


# Tests for data access pattern
class TestDacc:
    def test_dacc_schema_description(self):
        """Test schema description with instance property key"""
        dacc = Dacc()

        # First access computes the value
        result1 = dacc.schema_description
        assert result1 == "schema_description_1"
        assert "schema_description.txt" in dacc.text_store
        assert dacc.text_store["schema_description.txt"] == "schema_description_1"

        # Second access uses cached value
        result2 = dacc.schema_description
        assert result2 == "schema_description_1"
        assert dacc.compute_count == 1

        # Changing the key property forces recomputation with new key
        dacc.schema_description_key = "new_schema_description.txt"
        result3 = dacc.schema_description
        assert result3 == "schema_description_2"
        assert "new_schema_description.txt" in dacc.text_store
        assert dacc.compute_count == 2

    def test_dacc_pricing_html(self):
        """Test pricing HTML with instance property key"""
        dacc = Dacc()

        # First access computes the value
        result1 = dacc.pricing_page_html
        assert result1 == "pricing_html_1"
        assert "pricing.html" in dacc.text_store
        assert dacc.text_store["pricing.html"] == "pricing_html_1"

        # Second access uses cached value
        result2 = dacc.pricing_page_html
        assert result2 == "pricing_html_1"
        assert dacc.compute_count == 1

    def test_dacc_schema(self):
        """Test schema with instance property key"""
        dacc = Dacc()

        # First access computes the value
        result1 = dacc.schema
        assert result1 == {"version": "schema_1"}
        assert "schema.json" in dacc.json_store
        assert dacc.json_store["schema.json"] == {"version": "schema_1"}

        # Second access uses cached value
        result2 = dacc.schema
        assert result2 == {"version": "schema_1"}
        assert dacc.compute_count == 1


# Tests for error cases
class TestErrorCases:
    def test_none_key_not_allowed(self):
        """Test that None keys are not allowed by default"""

        class TestNoneKey:
            def __init__(self):
                self.my_cache = {}

            @cache_this(cache="my_cache", key=lambda self: None)
            def none_key_method(self):
                return 42

        obj = TestNoneKey()
        with pytest.raises(TypeError, match="cannot be None"):
            obj.none_key_method

    def test_missing_cache_attribute(self):
        """Test error when cache attribute is missing"""

        class TestMissingCache:
            @cache_this(cache="nonexistent_cache", key="test_key")
            def test_method(self):
                return 42

        obj = TestMissingCache()
        with pytest.raises(TypeError, match="No attribute named 'nonexistent_cache'"):
            obj.test_method

    def test_invalid_cache_attribute(self):
        """Test error when cache attribute is not a MutableMapping"""

        class TestInvalidCache:
            def __init__(self):
                self.invalid_cache = "not a mapping"

            @cache_this(cache="invalid_cache", key="test_key")
            def test_method(self):
                return 42

        obj = TestInvalidCache()
        with pytest.raises(TypeError, match="is not a MutableMapping"):
            obj.test_method


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
