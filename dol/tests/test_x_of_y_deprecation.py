"""The X_of_Y keywords of wrap_kvs still work in 0.4, warn, and name their replacement (#92)."""

import json
import warnings

import pytest

from dol import wrap_kvs
from dol.trans import X_OF_Y_TO_CODEC_KWARG

OLD = dict(
    key_of_id=str.upper,
    id_of_key=str.lower,
    obj_of_data=json.loads,
    data_of_obj=json.dumps,
)
NEW = {X_OF_Y_TO_CODEC_KWARG[k]: v for k, v in OLD.items()}


def _exercise(s):
    s["a"] = {"x": 1}
    return list(s), s["A"], dict(s.store)


@pytest.mark.parametrize("old_name", sorted(OLD))
def test_each_old_name_warns_with_its_replacement(old_name):
    new_name = X_OF_Y_TO_CODEC_KWARG[old_name]
    with pytest.warns(DeprecationWarning, match=f"use {new_name}= instead of {old_name}="):
        wrap_kvs({}, **{old_name: OLD[old_name]})


def test_old_and_new_spellings_behave_the_same():
    with pytest.warns(DeprecationWarning):
        old = wrap_kvs({}, **OLD)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new = wrap_kvs({}, **NEW)
    assert _exercise(old) == _exercise(new) == (["A"], {"x": 1}, {"a": '{"x": 1}'})


@pytest.mark.parametrize(
    "make",
    [
        lambda kw: wrap_kvs({}, **kw),  # instance
        lambda kw: wrap_kvs(dict, **kw)(),  # class
        lambda kw: wrap_kvs(**kw)(dict)(),  # factory
    ],
    ids=["instance", "class", "factory"],
)
def test_warning_points_at_the_caller(make):
    with pytest.warns(DeprecationWarning) as record:
        make({"obj_of_data": json.loads})
    assert {w.filename for w in record} == {__file__}


def test_new_spelling_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wrap_kvs({}, **NEW)
        wrap_kvs(dict, **NEW)


def test_both_spellings_for_the_same_role_is_an_error():
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError):
        wrap_kvs({}, obj_of_data=json.loads, value_decoder=json.loads)


class _FalsyTransform:
    """A callable whose truth value is False (it has a zero length)."""

    def __len__(self):
        return 0

    def __call__(self, x):
        return x * 10


@pytest.mark.parametrize("name", ["value_decoder", "obj_of_data"])
def test_a_falsy_callable_is_still_applied(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        s = wrap_kvs({"a": 1}, **{name: _FalsyTransform()})
    assert s["a"] == 10


def test_warning_through_pipe_points_at_the_caller():
    from dol import Pipe

    with pytest.warns(DeprecationWarning) as record:
        Pipe(wrap_kvs(obj_of_data=int), wrap_kvs(value_encoder=str))(dict)
    assert {w.filename for w in record} == {__file__}
