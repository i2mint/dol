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

    # Now, if we happen to use an integer key in a mapping, they'll be a problem though:
    with pytest.raises(KeyError):
        path_get({"a": [7, {4: "c"}]}, "a.1.4")
    # If you want to allow for integer keys in mappings as well as in lists, and still
    # maintain the "first try a key as attribute" behavior, you can use the
    # path_get.chain_of_getters function to create a getter that tries a sequence of
    # getters, in order, until one succeeds.
    getter = path_get.chain_of_getters(
        [getattr, lambda obj, k: obj[k], lambda obj, k: obj[int(k)]]
    )
    path_get({"a": [7, {4: "c"}]}, "a.1.4", get_value=getter)


def test_string_template_template_construction():
    assert KeyTemplate("{}.ext").template == "{i01_}.ext"
    assert KeyTemplate("{name}.ext").template == "{name}.ext"
    assert KeyTemplate("{::\w+}.ext").template == "{i01_}.ext"
    assert KeyTemplate("{name::\w+}.ext").template == "{name}.ext"
    assert KeyTemplate("{name::\w+}.ext").template == "{name}.ext"
    assert KeyTemplate("{name:0.02f}.ext").template == "{name}.ext"
    assert KeyTemplate("{name:0.02f:\w+}.ext").template == "{name}.ext"
    assert KeyTemplate("{:0.02f:\w+}.ext").template == "{i01_}.ext"


def test_string_template_regex():
    assert KeyTemplate("{}.ext")._regex.pattern == "(?P<i01_>.*)\\.ext"
    assert KeyTemplate("{name}.ext")._regex.pattern == "(?P<name>.*)\\.ext"
    assert KeyTemplate("{::\w+}.ext")._regex.pattern == "(?P<i01_>\\w+)\\.ext"
    assert KeyTemplate("{name::\w+}.ext")._regex.pattern == "(?P<name>\\w+)\\.ext"
    assert KeyTemplate("{:0.02f:\w+}.ext")._regex.pattern == "(?P<i01_>\\w+)\\.ext"
    assert KeyTemplate("{name:0.02f:\w+}.ext")._regex.pattern == "(?P<name>\\w+)\\.ext"


def test_string_template_simple():
    from dol.paths import KeyTemplate
    from collections import namedtuple

    st = KeyTemplate(
        "root/{}/v_{version:03.0f:\d+}.json",
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
