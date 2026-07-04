"""Test trans.py functionality."""

import dol.util
from dol.trans import (
    filt_iter,
    filter_prefixes,
    filter_regex,
    filter_suffixes,
    redirect_getattr_to_getitem,
    wrap_kvs,
    FirstArgIsMapping,
    _has_unbound_self,
    _resolve_self_convention,
)


# ---------------------------------------------------------------------------
# Signature-based conditioning of wrap_kvs transforms (Issues #9, #12, #18)
# ---------------------------------------------------------------------------


def test_wrap_kvs_unary_builtin_transform_issue_9():
    """A unary callable whose first param happens to be named ``self`` must be
    applied as ``f(data)``, not ``f(self, data)`` (Issue #9).

    ``bytes.decode`` is the canonical case: it is effectively unary (one required
    positional), but its first parameter is named ``self``. The old name-only
    heuristic mis-called it as ``bytes.decode(store, data)`` -> TypeError.
    """
    S = wrap_kvs(dict, obj_of_data=bytes.decode)
    s = S({"k": b"hello"})
    assert s["k"] == "hello"

    # str.upper, str.split — same shape, must also be treated as unary
    assert wrap_kvs({"k": "hi"}, obj_of_data=str.upper)["k"] == "HI"

    # The lambda form, which always worked, must keep working
    assert wrap_kvs({"k": b"hi"}, obj_of_data=lambda x: x.decode())["k"] == "hi"


def test_wrap_kvs_self_convention_still_works():
    """Transforms genuinely using the ``(self, data)`` convention (>=2 required
    positional params, first named self/store/mapping) must keep receiving the
    store instance. Must not regress the ~12 real ecosystem usages."""

    def obj_of_data(self, data):
        return f"{getattr(self, 'p', '?')}:{data}"

    S = wrap_kvs(dict, obj_of_data=obj_of_data)
    s = S({"a": "x"})
    s.p = "ns"
    assert s["a"] == "ns:x"

    # first param named 'store' works too
    def key_of_id(store, _id):
        return _id.upper()

    s2 = wrap_kvs({"a": 1, "b": 2}, key_of_id=key_of_id)
    assert sorted(s2) == ["A", "B"]


def test_first_arg_is_mapping_explicit_marker_issue_12():
    """FirstArgIsMapping forces the ``(self, data)`` convention regardless of the
    transform's parameter names, and unwraps to the underlying callable."""

    def needs_store(x, data):  # first param 'x' -> heuristic alone says no-self
        return f"{getattr(x, 'p', '?')}/{data}"

    S = wrap_kvs(dict, obj_of_data=FirstArgIsMapping(needs_store))
    s = S({"a": "v"})
    s.p = "NS"
    assert s["a"] == "NS/v"

    # Marker resolves to (underlying_func, wants_self=True)
    func, wants_self = _resolve_self_convention(FirstArgIsMapping(needs_store))
    assert func is needs_store and wants_self is True


def test_postget_preset_self_conventions():
    """postget/preset honor both the no-self and self conventions, and the
    FirstArgIsMapping marker."""

    def postget_self(self, k, v):
        return f"{k}={v}"

    assert wrap_kvs({"a": 1}, postget=postget_self)["a"] == "a=1"

    def postget_plain(k, v):
        return v * 2

    assert wrap_kvs({"a": 5}, postget=postget_plain)["a"] == 10

    def preset_self(self, k, v):
        return v + 1

    S = wrap_kvs(dict, preset=preset_self)
    s = S()
    s["a"] = 10
    assert dict(s) == {"a": 11}


def test_has_unbound_self_heuristic_units():
    """Unit-level checks of the (name AND >=2 required) heuristic."""
    assert _has_unbound_self(lambda self, data: data) is True
    assert _has_unbound_self(lambda store, data: data) is True
    assert _has_unbound_self(lambda data: data) is False
    assert _has_unbound_self(bytes.decode) is False  # first param 'self', 1 required
    assert _has_unbound_self(str.upper) is False
    # first param self-ish but 2nd is optional -> only 1 required -> no-self
    assert _has_unbound_self(lambda self, data=None: data) is False


def test_filter_regex_is_os_independent():
    """Regex filters must compile as REGEXES, not as path templates.

    Regression for a Windows-only bug: ``filter_regex`` used the path-oriented
    ``safe_compile``, which ``re.escape``s its input on Windows. That turned a
    pattern like ``(\\.json)$`` into a literal matcher, so ``filter_suffixes('.json')``
    rejected every ``*.json`` key on Windows and ``dol.Jsons`` raised
    ``KeyError: 'Key not in store: <key>.json'`` on write.
    """
    real_system = dol.util.platform.system
    try:
        # Simulate every platform, including Windows (the broken one).
        for system in ("Linux", "Darwin", "Windows"):
            dol.util.platform.system = lambda system=system: system
            assert bool(filter_regex(r"(\.json)$")("doc-001.json")) is True
            assert bool(filter_regex(r"(\.json)$")("doc-001.txt")) is False
            assert bool(filter_suffixes(".json")("doc-001.json")) is True
            assert bool(filter_suffixes([".txt", ".doc"])("report.doc")) is True
            assert bool(filter_suffixes(".json")("doc-001.txt")) is False
            assert bool(filter_prefixes("test")("test_image.jpg")) is True
            assert bool(filter_prefixes("test")("report.doc")) is False
    finally:
        dol.util.platform.system = real_system


def test_filt_iter():
    # Demo regex filter on a class
    contains_a = filt_iter.regex(r"a")
    # wrap the dict type with this
    filtered_dict = contains_a(dict)
    # now make a filtered_dict
    d = filtered_dict(apple=1, banana=2, cherry=3)
    # and see that keys not containing "a" are filtered out
    assert dict(d) == {"apple": 1, "banana": 2}

    # With this regex filt_iter, we made two specialized versions:
    # One filtering prefixes, and one filtering suffixes
    is_test = filt_iter.prefixes("test")  # Note, you can also pass a list of prefixes
    d = {"test.txt": 1, "report.doc": 2, "test_image.jpg": 3}
    dd = is_test(d)
    assert dict(dd) == {"test.txt": 1, "test_image.jpg": 3}

    is_text = filt_iter.suffixes([".txt", ".doc", ".pdf"])
    d = {"test.txt": 1, "report.doc": 2, "image.jpg": 3}
    dd = is_text(d)
    assert dict(dd) == {"test.txt": 1, "report.doc": 2}


def test_redirect_getattr_to_getitem():

    # Applying it to a class

    ## ... with the @decorator syntax
    @redirect_getattr_to_getitem
    class MyDict(dict):
        pass

    d1 = MyDict(a=1, b=2)
    assert d1.a == 1
    assert d1.b == 2
    assert list(d1) == ["a", "b"]

    ## ... as a decorator factory
    D = redirect_getattr_to_getitem()(dict)
    d2 = D(a=1, b=2)
    assert d2.a == 1
    assert d2.b == 2
    assert list(d2) == ["a", "b"]

    # Applying it to an instance

    ## ... as a decorator
    backend_d = dict(a=1, b=2)

    d3 = redirect_getattr_to_getitem(backend_d)
    assert d3.a == 1
    assert d3.b == 2
    assert list(d3) == ["a", "b"]

    ## ... as a decorator factory
    d4 = redirect_getattr_to_getitem()(backend_d)
    assert d4.a == 1
    assert d4.b == 2
    assert list(d4) == ["a", "b"]
