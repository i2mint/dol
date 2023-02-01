from dol import Files, wrap_kvs


def test_wrap_kvs_vs_class_and_static_methods():
    """Adding wrap_kvs breaks methods

    https://github.com/i2mint/dol/issues/17
    """

    def returnx(x):
        return x

    @wrap_kvs(data_of_obj=returnx, obj_of_data=returnx)
    class MyFiles(Files):
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

    try:
        MyFiles.hello()
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
