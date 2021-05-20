import pytest
import pickle

from dol.base import Store
from dol.trans import wrap_kvs, filt_iter, cache_iter

pup = lambda obj: pickle.loads(pickle.dumps(obj))


def add_tag(k):
    return k + '__tag'


def remove_tag(k):
    assert k.endswith('__tag')
    return k[: -len('__tag')]


@pytest.mark.xfail
def test_pickling_w_simple_store():
    store = Store({'a': 1, 'b': 2})
    pickled = pickle.dumps(store)
    unpickled = pickle.loads(pickled)
    assert dict(store) == dict(unpickled)


@pytest.mark.xfail
def test_pickling_with_wrap_kvs_class():
    D = wrap_kvs(key_of_id=add_tag, id_of_key=remove_tag)(dict)
    store = D({'a': 1, 'b': 2})
    pickled = pickle.dumps(store)
    unpickled = pickle.loads(pickled)
    assert dict(store) == dict(unpickled)


@pytest.mark.xfail
def test_pickling_with_wrap_kvs_instance():
    d = {'a': 1, 'b': 2}
    wrapped_d = wrap_kvs(d, key_of_id=add_tag, id_of_key=remove_tag)
    pickled = pickle.dumps(wrapped_d)
    unpickled = pickle.loads(pickled)
    assert dict(unpickled) == dict(wrapped_d)


# @pytest.mark.xfail
# def test_pickling_with_filt_iter_instance():
#     d = {"a": 1, "b": 2, "cc": 3}
#     wrapped_d = filt_iter(d, key_of_id=add_tag, id_of_key=remove_tag)
#     pickled = pickle.dumps(wrapped_d)
#     unpickled = pickle.loads(pickled)
#     assert dict(unpickled) == dict(wrapped_d)
