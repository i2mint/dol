"""``list(store)`` on a scoped Store walks the backend twice, not three times (#223)."""

from dol.base import Collection, Store
from dol.errors import OutOfScopeKeyError, raises_out_of_scope


class _Counting(Collection):
    """Inner store with the default (scanning) len; counts walks."""

    def __init__(self, d):
        self.d, self.walks = d, 0

    def __iter__(self):
        self.walks += 1
        return iter(self.d)

    def __getitem__(self, k):
        return self.d[k]


class _Refusing(_Counting):
    def __len__(self):
        raise TypeError("remote listing refuses len")


class _Sized(_Counting):
    def __len__(self):
        return len(self.d)


class _Scoped(Store):
    @raises_out_of_scope
    def _key_of_id(self, _id):
        if not _id.startswith("p/"):
            raise OutOfScopeKeyError(_id)
        return _id[2:]


DATA = {"p/a": 1, "p/b": 2, "x": 3}


def test_scoped_list_walks_twice():
    inner = _Counting(DATA)
    assert list(_Scoped(inner)) == ["a", "b"]
    assert inner.walks == 2


def test_scoped_len_is_in_scope_count():
    assert len(_Scoped(_Counting(DATA))) == 2


def test_scoped_refusing_inner_walks_once():
    inner = _Refusing(DATA)
    assert list(_Scoped(inner)) == ["a", "b"]
    assert inner.walks == 1


def test_scoped_sized_inner_len_still_asked():
    inner = _Sized({})
    assert len(_Scoped(inner)) == 0
    assert inner.walks == 0  # an empty sized inner costs no scan


def test_stacked_scoped_layers_walk_twice():
    inner = _Counting(DATA)
    assert list(_Scoped(Store(inner))) == ["a", "b"]
    assert inner.walks == 2


def test_len_refusal_is_preserved():
    import pytest

    with pytest.raises(TypeError):
        len(_Scoped(_Refusing(DATA)))
    with pytest.raises(TypeError):
        len(_Scoped(Store(_Refusing(DATA))))
    with pytest.raises(TypeError):
        len(_Scoped(_Scoped(_Refusing(DATA))))


def test_bool_agrees_with_len():
    assert bool(_Scoped(_Counting(DATA)))
    assert not _Scoped(_Counting({"x": 1}))
    assert len(_Scoped(_Counting({"x": 1}))) == 0
