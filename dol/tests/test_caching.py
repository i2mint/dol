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


# Tests for new CachedMethod functionality
class TestCachedMethodFunctionality:
    """Test the new method caching capabilities added to cache_this"""

    def test_basic_method_caching(self):
        """Test basic method caching with arguments"""

        class DataProcessor:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            @cache_this(cache='cache')
            def process(self, x, y):
                self.call_count += 1
                return x * y

        obj = DataProcessor()

        # First call should compute the result
        result1 = obj.process(2, 3)
        assert result1 == 6
        assert obj.call_count == 1
        assert 'x=2;y=3' in obj.cache
        assert obj.cache['x=2;y=3'] == 6

        # Second call with same args should use cache
        result2 = obj.process(2, 3)
        assert result2 == 6
        assert obj.call_count == 1  # Should not increase

        # Call with different args should compute new result
        result3 = obj.process(4, 5)
        assert result3 == 20
        assert obj.call_count == 2
        assert 'x=4;y=5' in obj.cache
        assert obj.cache['x=4;y=5'] == 20

    def test_method_caching_with_kwargs(self):
        """Test method caching with keyword arguments"""

        class Calculator:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            @cache_this(cache='cache')
            def compute(self, x, y=10, z=None):
                self.call_count += 1
                return x + y + (z or 0)

        obj = Calculator()

        # Test with mixed positional and keyword args
        result1 = obj.compute(1, y=2, z=3)
        assert result1 == 6
        assert obj.call_count == 1
        assert 'x=1;y=2;z=3' in obj.cache

        # Same call should use cache
        result2 = obj.compute(1, y=2, z=3)
        assert result2 == 6
        assert obj.call_count == 1

        # Different kwargs should compute new result
        result3 = obj.compute(1, y=5, z=3)
        assert result3 == 9
        assert obj.call_count == 2
        assert 'x=1;y=5;z=3' in obj.cache

    def test_custom_key_function_for_methods(self):
        """Test method caching with custom key function"""

        class DataProcessor:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            @cache_this(cache='cache', key=lambda self, x, y: f'method__{x},{y}.pkl')
            def multiply(self, x, y):
                self.call_count += 1
                return x * y

        obj = DataProcessor()

        # First call should compute
        result1 = obj.multiply(3, 4)
        assert result1 == 12
        assert obj.call_count == 1
        assert 'method__3,4.pkl' in obj.cache

        # Second call should use cache
        result2 = obj.multiply(3, 4)
        assert result2 == 12
        assert obj.call_count == 1

    def test_auto_detection_property_vs_method(self):
        """Test that cache_this correctly auto-detects properties vs methods"""

        class TestAutoDetect:
            def __init__(self):
                self.cache = {}
                self.property_calls = 0
                self.method_calls = 0

            @cache_this(cache='cache')
            def no_args_property(self):
                """This should be detected as a property"""
                self.property_calls += 1
                return "property_value"

            @cache_this(cache='cache')
            def with_args_method(self, x, y=5):
                """This should be detected as a method"""
                self.method_calls += 1
                return x + y

            @cache_this(cache='cache')
            def with_varargs(self, *args):
                """This should be detected as a method"""
                self.method_calls += 1
                return sum(args)

            @cache_this(cache='cache')
            def with_kwargs(self, **kwargs):
                """This should be detected as a method"""
                self.method_calls += 1
                return len(kwargs)

        obj = TestAutoDetect()

        # Test property detection
        result1 = obj.no_args_property
        assert result1 == "property_value"
        assert obj.property_calls == 1
        assert 'no_args_property' in obj.cache

        # Second access should use cache
        result1_again = obj.no_args_property
        assert result1_again == "property_value"
        assert obj.property_calls == 1

        # Test method detection with regular args
        result2 = obj.with_args_method(10, y=15)
        assert result2 == 25
        assert obj.method_calls == 1
        assert 'x=10;y=15' in obj.cache

        # Test method detection with varargs
        method_func = obj.with_varargs
        result3 = method_func(1, 2, 3, 4)
        assert result3 == 10
        assert obj.method_calls == 2

        # Test method detection with kwargs
        method_func2 = obj.with_kwargs
        result4 = method_func2(a=1, b=2, c=3)
        assert result4 == 3
        assert obj.method_calls == 3

    def test_as_property_override(self):
        """Test the as_property parameter to override auto-detection"""

        class TestOverride:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            @cache_this(cache='cache', as_property=False)
            def property_as_method(self):
                # This has no args but we're forcing it to be treated as a method
                self.call_count += 1
                return "method_result"

        obj = TestOverride()

        # The property forced as method should return a callable
        method_func = obj.property_as_method
        result = method_func()
        assert result == "method_result"
        assert obj.call_count == 1

        # Calling again should use cache
        method_func2 = obj.property_as_method
        result2 = method_func2()
        assert result2 == "method_result"
        assert obj.call_count == 1  # Should not increase due to caching

    def test_method_caching_with_external_cache(self):
        """Test method caching with external cache dictionary"""

        external_cache = {}

        class DataProcessor:
            def __init__(self):
                self.call_count = 0

            @cache_this(cache=external_cache)
            def process(self, x, y):
                self.call_count += 1
                return x * y

        obj = DataProcessor()

        # First call should compute and cache externally
        result1 = obj.process(2, 3)
        assert result1 == 6
        assert obj.call_count == 1
        assert 'x=2;y=3' in external_cache
        assert external_cache['x=2;y=3'] == 6

        # Second call should use external cache
        result2 = obj.process(2, 3)
        assert result2 == 6
        assert obj.call_count == 1

    def test_method_caching_with_pre_cache(self):
        """Test method caching with pre-cache functionality"""

        pre_cache_dict = {}

        class DataProcessor:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            @cache_this(cache='cache', pre_cache=pre_cache_dict)
            def process(self, x, y):
                self.call_count += 1
                return x * y

        obj = DataProcessor()

        # First call should compute and store in both caches
        result1 = obj.process(2, 3)
        assert result1 == 6
        assert obj.call_count == 1
        assert 'x=2;y=3' in obj.cache

        # Second call should use pre-cache
        result2 = obj.process(2, 3)
        assert result2 == 6
        assert obj.call_count == 1

    def test_backward_compatibility_properties(self):
        """Test that existing property caching behavior is unchanged"""

        class TestClass:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            @cache_this(cache='cache')
            def expensive_property(self):
                self.call_count += 1
                return 42

        obj = TestClass()

        # First access should compute the value
        result1 = obj.expensive_property
        assert result1 == 42
        assert obj.call_count == 1
        assert 'expensive_property' in obj.cache

        # Second access should use cached value
        result2 = obj.expensive_property
        assert result2 == 42
        assert obj.call_count == 1  # Should not increase

    def test_method_with_complex_arguments(self):
        """Test method caching with complex argument types"""

        class DataProcessor:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            @cache_this(cache='cache')
            def process_data(self, data_list, multiplier=1, options=None):
                self.call_count += 1
                total = sum(data_list) * multiplier
                if options and options.get('double'):
                    total *= 2
                return total

        obj = DataProcessor()

        # Test with list and dict arguments
        result1 = obj.process_data([1, 2, 3], multiplier=2, options={'double': True})
        assert result1 == 24  # (1+2+3) * 2 * 2 = 24
        assert obj.call_count == 1

        # Same call should use cache
        result2 = obj.process_data([1, 2, 3], multiplier=2, options={'double': True})
        assert result2 == 24
        assert obj.call_count == 1

    def test_method_caching_thread_safety_simulation(self):
        """Test that method caching maintains thread safety patterns"""

        class ThreadSafeProcessor:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            @cache_this(cache='cache')
            def compute(self, x):
                # Simulate computation that might be called from multiple threads
                self.call_count += 1
                return x**2

        obj = ThreadSafeProcessor()

        # Multiple calls with same arguments should only compute once
        results = []
        for _ in range(5):
            results.append(obj.compute(10))

        assert all(r == 100 for r in results)
        assert obj.call_count == 1  # Should only compute once despite multiple calls

    def test_cache_false_behavior_for_methods(self):
        """Test that cache=False works for methods by creating a property"""

        class TestClass:
            def __init__(self):
                self.call_count = 0

            @cache_this(cache=False)
            def no_cache_property(self):
                # Note: when cache=False, methods without args become properties
                self.call_count += 1
                return 42

        obj = TestClass()

        # Each access should compute the result (no caching)
        result1 = obj.no_cache_property
        assert result1 == 42
        assert obj.call_count == 1

        result2 = obj.no_cache_property
        assert result2 == 42
        assert obj.call_count == 2  # Should increase each time

    def test_error_cases_for_methods(self):
        """Test error handling for method caching"""

        class TestErrorCases:
            def __init__(self):
                self.invalid_cache = "not a mapping"

            @cache_this(cache='invalid_cache')
            def method_with_invalid_cache(self, x):
                return x

            @cache_this(cache='nonexistent_cache')
            def method_with_missing_cache(self, x):
                return x

        obj = TestErrorCases()

        # Test invalid cache type
        with pytest.raises(TypeError, match="is not a MutableMapping"):
            obj.method_with_invalid_cache(5)

        # Test missing cache attribute
        with pytest.raises(TypeError, match="No attribute named 'nonexistent_cache'"):
            obj.method_with_missing_cache(5)

    def test_method_caching_integration_with_key_strategies(self):
        """Test that method caching works well with existing key strategies when forced as property"""

        class IntegrationTest:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            # Force a method with args to be treated as property (edge case)
            @cache_this(
                cache='cache', key=ExplicitKey('forced_property'), as_property=True
            )
            def forced_property_method(self):
                # This method will be treated as property despite having the potential for args
                self.call_count += 1
                return "forced_property_result"

        obj = IntegrationTest()

        # Should work as a property
        result1 = obj.forced_property_method
        assert result1 == "forced_property_result"
        assert obj.call_count == 1
        assert 'forced_property' in obj.cache

        # Second access should use cache
        result2 = obj.forced_property_method
        assert result2 == "forced_property_result"
        assert obj.call_count == 1

    def test_comprehensive_requirements_example(self):
        """Test the exact example from the original requirements"""

        class DataProcessor:
            def __init__(self):
                self.cache = {}

            # This should work as before (property)
            @cache_this(cache='cache')
            def expensive_property(self):
                return "compute_once_result"

            # This should now work (method with args)
            @cache_this(cache='cache')
            def process(self, x, y):
                return x * y

            # With custom key function
            @cache_this(cache='cache', key=lambda self, x, y: f'method__{x},{y}.pkl')
            def multiply(self, x, y):
                return x * y

        obj = DataProcessor()

        # Test property caching
        result1 = obj.expensive_property
        assert result1 == "compute_once_result"

        # Test method caching
        result2 = obj.process(3, 4)
        assert result2 == 12

        # Test different args
        result3 = obj.process(5, 6)
        assert result3 == 30

        # Test custom key function
        result4 = obj.multiply(2, 3)
        assert result4 == 6

        # Verify cache contents and keys
        expected_keys = ['expensive_property', 'x=3;y=4', 'x=5;y=6', 'method__2,3.pkl']
        for key in expected_keys:
            assert key in obj.cache


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
