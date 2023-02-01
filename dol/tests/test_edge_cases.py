from dol.base import Store
import pytest


@pytest.xfail(reason='edge case that we will try to address later')
def test_wrap_kvs_vs_class_and_static_methods():
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
            print('hello')

        @staticmethod
        def hi():
            print('hi')

    errors = []

    # This works fine!
    instance = MyFiles()
    assert instance.normal_method() == 6

    # But calling the method as a class...
    try:
        MyFiles.normal_method(instance)
    except Exception as e:
        print('method normal_method is broken by wrap_kvs decorator')
        print(f'{type(e).__name__}: {e}')
        errors.append(e)

    try:
        MyFiles.hello()
    except Exception as e:
        print('classmethod hello is broken by wrap_kvs decorator')
        print(f'{type(e).__name__}: {e}')
        errors.append(e)

    try:
        MyFiles.hi()
    except Exception as e:
        print('staticmethod hi is broken by wrap_kvs decorator')
        print(f'{type(e).__name__}: {e}')
        errors.append(e)

    if errors:
        first_error, *_ = errors
        raise first_error
