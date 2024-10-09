"""Ideas for a new store wrapping setup"""

from typing import MutableMapping, Mapping, KT, VT, Iterable, Callable
from dol.util import inject_method


# Note: See dol.base.Store (and complete MappingWrap)
# TODO: Complete with wrap hooks (those of dol.base.Store and more)
class MappingWrap:
    def __init__(self, store: Mapping):
        self.store = store

    # mapping special methods forward to hidden methods

    def __iter__(self) -> Iterable[KT]:
        return self._iter()

    def __getitem__(self, k: KT) -> VT:
        return self._getitem(k)

    def __len__(self, k: KT) -> int:
        return self._len(k)

    def __contains__(self, k: KT) -> bool:
        return self._contains(k)

    def __setitem__(self, k: KT, v: VT):
        return self._setitem(k, v)

    def __delitem__(self, k: KT):
        return self._delitem(k)

    # default mapping hidden methods just forward to store
    # TODO: Add wrapping hooks (_obj_of_data, etc.  including some for filt_iter, etc.)
    def _iter(self) -> Iterable[KT]:
        return iter(self.store)

    def _getitem(self, k: KT) -> VT:
        return self.store[k]

    def _len(self, k: KT) -> int:
        return len(self.store)

    def _contains(self, k: KT) -> bool:
        return k in self.store

    def _setitem(self, k: KT, v: VT):
        return self.store.__setitem__(k, v)

    def _delitem(self, k: KT):
        return self.store.__delitem__(k)

    # util method to inject new

    def _inject_method(self, method_function, method_name=None):
        return inject_method(self, method_function, method_name)


def test_mapping_wrap():
    from dol.scrap.new_store_wrap import MappingWrap

    from dol import TextFiles, filt_iter
    import posixpath

    def filter_when_ending_with_slash(self, k, slash=posixpath.sep):
        if not k.endswith(slash):
            return self.store[k]
        else:
            return filt_iter(self.store, filt=lambda key: key.startswith(k))

    class SlashTriggersFilter(MappingWrap):
        def _getitem(self, k):
            return filter_when_ending_with_slash(self, k)

    from dol.tests.utils_for_tests import mk_test_store_from_keys

    s = mk_test_store_from_keys()
    assert dict(s) == {
        "pluto": "Content of pluto",
        "planets/mercury": "Content of planets/mercury",
        "planets/venus": "Content of planets/venus",
        "planets/earth": "Content of planets/earth",
        "planets/mars": "Content of planets/mars",
        "fruit/apple": "Content of fruit/apple",
        "fruit/banana": "Content of fruit/banana",
        "fruit/cherry": "Content of fruit/cherry",
    }

    ss = SlashTriggersFilter(s)
    assert list(ss) == [
        "pluto",
        "planets/mercury",
        "planets/venus",
        "planets/earth",
        "planets/mars",
        "fruit/apple",
        "fruit/banana",
        "fruit/cherry",
    ]

    sss = ss["planets/"]
    assert dict(sss) == {
        "planets/mercury": "Content of planets/mercury",
        "planets/venus": "Content of planets/venus",
        "planets/earth": "Content of planets/earth",
        "planets/mars": "Content of planets/mars",
    }
