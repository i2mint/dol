from dol.base import Store
import pytest


@pytest.mark.skip(reason="edge case that we will try to address later")
def test_simple_store_wrap_unbound_method_delegation():
    # What does Store.wrap do? It wraps classes or instances in such a way that
    # mapping methods (like __iter__, __getitem__, __setitem__, __delitem__, etc.)
    # are intercepted and transformed, but other methods are not (they stay as they
    # were).

    # This test is about the "stay as they were" part, so let's start with a simple
    # class that has a method that we want to keep untouched.
    class K:
        def pass_through(self):
            return "hi"

    # wrapping an instance
    instance_of_k = K()
    assert instance_of_k.pass_through() == "hi"
    wrapped_instance_of_k = Store.wrap(instance_of_k)
    assert wrapped_instance_of_k.pass_through() == "hi"

    # wrapping a class
    WrappedK = Store.wrap(K)

    instance_of_wrapped_k = WrappedK()
    assert instance_of_wrapped_k.pass_through() == "hi"

    # Everything seems fine, but the problem creeps up when we try to use these methods
    # through an "unbound call".
    # This is when you call the method from a class, feeding an instance.
    # With the original class, this works:
    assert K.pass_through(K()) == "hi"

    # But this gives us an error on the wrapped class
    assert WrappedK.pass_through(WrappedK()) == "hi"  # error
    # or even this:
    assert WrappedK.pass_through(K()) == "hi"  # error


@pytest.mark.skip(reason="edge case that we will try to address later")
def test_store_wrap_unbound_method_delegation():
    """Making sure `dol.base.Store.wrap` doesn't break unbound method calls.

    That is, when you call Klass.method() (where method is a normal, class, or static)

    https://github.com/i2mint/dol/issues/17
    """

    @Store.wrap
    class MyFiles:
        y = 2

        def normal_method(self, x=3):
            return self.y * x

        @classmethod
        def hello(cls):
            print("hello")

        @staticmethod
        def hi():
            print("hi")

    errors = []

    # This works fine!
    instance = MyFiles()
    assert instance.normal_method() == 6

    # But calling the method as a class...
    try:
        MyFiles.normal_method(instance)
    except Exception as e:
        print("method normal_method is broken by wrap_kvs decorator")
        print(f"{type(e).__name__}: {e}")
        errors.append(e)

    try:
        MyFiles.hello()
    except Exception as e:
        print("classmethod hello is broken by wrap_kvs decorator")
        print(f"{type(e).__name__}: {e}")
        errors.append(e)

    try:
        MyFiles.hi()
    except Exception as e:
        print("staticmethod hi is broken by wrap_kvs decorator")
        print(f"{type(e).__name__}: {e}")
        errors.append(e)

    if errors:
        first_error, *_ = errors
        raise first_error
