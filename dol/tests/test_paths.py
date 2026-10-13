"""Tests for paths.py"""

from dol.paths import KeyTemplate, path_get
import pytest


def test_path_get():
    # NOTE: The following examples test the current default behavior, but that doesn't
    # mean that this default behavior is the best behavior. I've made the choice of
    # aligning (though not completely) with the behavior of glom, which is a great
    # library for getting values from nested data structures (and I recommend to use
    # glom instead of path_get if you need more features than path_get provides).
    # On the other hand, I'm not sure glom's default choices are the best either.
    # I would vote for a more restrictive, but explicit, so predictable behavior by
    # default. That said, it makes path_get more annoying to use out of the box.

    path_get({"a": {"1": {"4": "c"}}}, "a.1.4") == "c"
    # When a path is given as a string, it is split on '.' and each element is used
    # as a key. So 'a.1.4' is equivalent to ['a', '1', '4']
    # Here, each key of the path acts as a key into a Mapping
    path_get({"a": {"1": {"4": "c"}}}, "a.1.4") == "c"
    # But see next, how '1' is actually interpreted as an integer, not a string, since
    # it's indexing into a list, and '4' is interpreted as a string, since it's
    # key-ing into a dict (a Mapping).
    path_get({"a": [7, {"4": "c"}]}, "a.1.4") == "c"
    # 4 would take on the role of an integer index if we replace the {'4': 'c'} Mapping
    # with a list.
    path_get({"a": [7, [0, 1, 2, 3, "c"]]}, "a.1.4") == "c"

    # Using an integer key in a mapping, is also not a problem:
    path_get({"a": [7, {4: "c"}]}, "a.1.4")

    # If you specify an integer key (here, 4) and your mapping (here {"4": "c"})
    # doesn't have one, on the other hand, you'll get a KeyError
    with pytest.raises(KeyError):
        path_get({"a": [7, {"4": "c"}]}, ("a", 1, 4))

    # To get out of such situations, you can either specify a different (functional)
    # `sep` argument, or specify a different `get_value` function. Here we use
    # path_get.chain_of_getters function to create a getter that tries a sequence of
    # getters, in order, until one succeeds.
    getter = path_get.chain_of_getters(
        [getattr, lambda obj, k: obj[k], lambda obj, k: obj[int(k)]]
    )
    path_get({"a": [7, {4: "c"}]}, ("a", 1, "4"), get_value=getter)


def test_string_template_template_construction():
    assert KeyTemplate("{}.ext").template == "{i01_}.ext"
    assert KeyTemplate("{name}.ext").template == "{name}.ext"
    assert KeyTemplate(r"{::\w+}.ext").template == "{i01_}.ext"
    assert KeyTemplate(r"{name::\w+}.ext").template == "{name}.ext"
    assert KeyTemplate(r"{name::\w+}.ext").template == "{name}.ext"
    assert KeyTemplate("{name:0.02f}.ext").template == "{name}.ext"
    assert KeyTemplate(r"{name:0.02f:\w+}.ext").template == "{name}.ext"
    assert KeyTemplate(r"{:0.02f:\w+}.ext").template == "{i01_}.ext"


def test_string_template_regex():
    assert KeyTemplate("{}.ext")._regex.pattern == r"(?P<i01_>.*)\.ext"
    assert KeyTemplate("{name}.ext")._regex.pattern == r"(?P<name>.*)\.ext"
    assert KeyTemplate(r"{::\w+}.ext")._regex.pattern == r"(?P<i01_>\w+)\.ext"
    assert KeyTemplate(r"{name::\w+}.ext")._regex.pattern == r"(?P<name>\w+)\.ext"
    assert KeyTemplate(r"{:0.02f:\w+}.ext")._regex.pattern == r"(?P<i01_>\w+)\.ext"
    assert KeyTemplate(r"{name:0.02f:\w+}.ext")._regex.pattern == r"(?P<name>\w+)\.ext"


def test_string_template_simple():
    from dol.paths import KeyTemplate
    from collections import namedtuple

    st = KeyTemplate(
        r"root/{}/v_{version:03.0f:\d+}.json",
        from_str_funcs={"version": int},
    )

    assert st.str_to_dict("root/life/v_42.json") == {"i01_": "life", "version": 42}
    assert st.dict_to_str({"i01_": "life", "version": 42}) == "root/life/v_042.json"
    assert st.dict_to_tuple({"i01_": "life", "version": 42}) == ("life", 42)
    assert st.tuple_to_dict(("life", 42)) == {"i01_": "life", "version": 42}
    assert st.str_to_tuple("root/life/v_42.json") == ("life", 42)
    assert st.tuple_to_str(("life", 42)) == "root/life/v_042.json"

    assert st.str_to_simple_str("root/life/v_42.json") == "life,042"
    st_clone = st.clone(simple_str_sep="-")
    assert st_clone.str_to_simple_str("root/life/v_42.json") == "life-042"
    assert st_clone.simple_str_to_str("life-42") == "root/life/v_042.json"

    from collections import namedtuple

    VersionedFile = st.dict_to_namedtuple({"i01_": "life", "version": 42})
    assert VersionedFile == namedtuple("VersionedFile", ["i01_", "version"])("life", 42)
    assert st.namedtuple_to_dict(VersionedFile) == {"i01_": "life", "version": 42}


def test_key_validation_wrappers_coexist_with_with_key_validation():
    """``with_key_validation=True`` redefines ``_id_of_key`` to RAISE on an invalid key, so
    the key-mapping wrappers must use the unvalidated mapping -- otherwise ``is_valid_key``
    raises for exactly the input it exists to answer 'no' for."""
    from dol.paths import mk_relative_path_store

    class Leaf(dict):
        _prefix = "/ROOT/"

        def is_valid_key(self, k):
            return k.startswith("/ROOT/") and k.endswith(".txt")

        def validate_key(self, k):
            if not self.is_valid_key(k):
                raise ValueError(k)

    s = mk_relative_path_store(Leaf, with_key_validation=True)()
    assert s.is_valid_key("b.txt") is True
    assert s.is_valid_key("b.json") is False  # a bool, not a raise
    s.validate_key("b.txt")
    try:
        s.validate_key("b.json")
    except ValueError:
        pass  # the LEAF's exception type, not KeyError from the validating _id_of_key
    else:
        raise AssertionError("validate_key should have raised ValueError")


# --- i2mint/dol#82: prefix relativization must not corrupt out-of-scope keys ---------


def _prefix_stores():
    from dol import KeyCodecs, mk_relative_path_store
    from dol.paths import prefixless_view

    base = {"a/b": 1, "a/c": 2, "z": 3, "ab/x": 4}

    def via_mixin():
        s = mk_relative_path_store(dict)(base)
        s._prefix = "a/"
        return s

    yield "KeyCodecs.prefixed", KeyCodecs.prefixed("a/")(dict(base))
    yield "mk_relative_path_store", via_mixin()
    yield "prefixless_view", prefixless_view(dict(base), prefix="a/")


@pytest.mark.parametrize("name, store", list(_prefix_stores()))
def test_prefix_relativization_skips_out_of_scope_ids_in_iteration(name, store):
    # iteration is a view of the in-scope ids: no corrupted keys, no raise
    assert sorted(store) == ["b", "c"]
    assert sorted(store.items()) == [("b", 1), ("c", 2)]
    # in-scope keys behave exactly as before
    assert store["b"] == 1
    store["new"] = 5
    assert store["new"] == 5
    assert "z" not in store and "/x" not in store
    assert "new" in sorted(store)


@pytest.mark.parametrize("name, store", list(_prefix_stores()))
def test_prefix_relativization_key_mapper_raises_on_out_of_scope_ids(name, store):
    from dol.errors import OutOfScopeKeyError

    with pytest.raises(OutOfScopeKeyError, match="does not start with the prefix"):
        store._key_of_id("z")


def test_filter_then_relativize_is_unchanged():
    from dol import KeyCodecs, Pipe, filt_iter

    base = {"a/b": 1, "a/c": 2, "z": 3, "ab/x": 4}
    s = Pipe(filt_iter.prefixes("a/"), KeyCodecs.prefixed("a/"))(dict(base))
    assert sorted(s) == ["b", "c"]


def test_out_of_scope_key_error_is_key_and_value_error():
    from dol.errors import OutOfScopeKeyError

    assert issubclass(OutOfScopeKeyError, KeyError)
    assert issubclass(OutOfScopeKeyError, ValueError)
    assert str(OutOfScopeKeyError("msg")) == "msg"


def test_prefix_reassigned_after_use_relativizes_with_new_prefix():
    from dol import mk_relative_path_store

    s = mk_relative_path_store(dict)()
    s._prefix = "/R/"
    s["a"] = 1
    assert list(s) == ["a"]
    s._prefix = "/RR/"
    s.store.clear()
    s["b"] = 2
    assert list(s) == ["b"]


# --- i2mint/dol#114: the suffix end of the out-of-scope guard ---


@pytest.mark.parametrize(
    "prefix, in_scope_id",
    [("", "foo.txt"), ("a/", "a/foo.txt")],
)
def test_suffix_decoding_skips_out_of_scope_ids_in_iteration(prefix, in_scope_id):
    from dol import KeyCodecs
    from dol.errors import OutOfScopeKeyError

    s = KeyCodecs.affixed(prefix, ".txt")({in_scope_id: 1, prefix + "foo.json": 2})
    assert list(s) == ["foo"]  # no corrupted 'foo.' key
    assert s["foo"] == 1
    with pytest.raises(OutOfScopeKeyError, match="does not end with the suffix"):
        s._key_of_id(prefix + "foo.json")


def test_affix_overlapping_prefix_and_suffix_is_out_of_scope():
    from dol import KeyCodecs
    from dol.errors import OutOfScopeKeyError

    s = KeyCodecs.affixed("ab", "bc")({"abc": 1})  # 'b' would count for both ends
    assert list(s) == []
    with pytest.raises(OutOfScopeKeyError):
        s._key_of_id("abc")


def test_filter_then_strip_suffix_is_unchanged():
    from dol import KeyCodecs, Pipe, filt_iter

    base = {"foo.txt": 1, "bar.txt": 2, ".DS_Store": 3, "README": 4}
    s = Pipe(
        filt_iter(filt=lambda k: k.endswith(".txt")), KeyCodecs.suffixed(".txt")
    )(dict(base))
    assert sorted(s) == ["bar", "foo"]


def test_head_skips_out_of_scope_ids():
    from dol import KeyCodecs

    s = KeyCodecs.prefixed("p/")({"stray": 0, "p/a": 1})
    assert s.head() == ("a", 1)
