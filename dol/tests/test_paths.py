"""Tests for paths.py"""

from dol.paths import KeyTemplate


def test_string_template_template_construction():
    assert KeyTemplate('{}.ext').template == '{i01_}.ext'
    assert KeyTemplate('{name}.ext').template == '{name}.ext'
    assert KeyTemplate('{::\w+}.ext').template == '{i01_}.ext'
    assert KeyTemplate('{name::\w+}.ext').template == '{name}.ext'
    assert KeyTemplate('{name::\w+}.ext').template == '{name}.ext'
    assert KeyTemplate('{name:0.02f}.ext').template == '{name}.ext'
    assert KeyTemplate('{name:0.02f:\w+}.ext').template == '{name}.ext'
    assert KeyTemplate('{:0.02f:\w+}.ext').template == '{i01_}.ext'


def test_string_template_regex():
    assert KeyTemplate('{}.ext').regex.pattern == '(?P<i01_>.*)\\.ext'
    assert KeyTemplate('{name}.ext').regex.pattern == '(?P<name>.*)\\.ext'
    assert KeyTemplate('{::\w+}.ext').regex.pattern == '(?P<i01_>\\w+)\\.ext'
    assert KeyTemplate('{name::\w+}.ext').regex.pattern == '(?P<name>\\w+)\\.ext'
    assert KeyTemplate('{:0.02f:\w+}.ext').regex.pattern == '(?P<i01_>\\w+)\\.ext'
    assert KeyTemplate('{name:0.02f:\w+}.ext').regex.pattern == '(?P<name>\\w+)\\.ext'


def test_string_template_simple():
    from dol.paths import KeyTemplate
    from collections import namedtuple

    st = KeyTemplate(
        'root/{}/v_{version:03.0f:\d+}.json', from_str_funcs={'version': int},
    )

    assert st.str_to_dict('root/life/v_42.json') == {'i01_': 'life', 'version': 42}
    assert st.dict_to_str({'i01_': 'life', 'version': 42}) == 'root/life/v_042.json'
    assert st.dict_to_tuple({'i01_': 'life', 'version': 42}) == ('life', 42)
    assert st.tuple_to_dict(('life', 42)) == {'i01_': 'life', 'version': 42}
    assert st.str_to_tuple('root/life/v_42.json') == ('life', 42)
    assert st.tuple_to_str(('life', 42)) == 'root/life/v_042.json'

    assert st.str_to_simple_str('root/life/v_42.json') == 'life,042'
    assert st.str_to_simple_str('root/life/v_42.json', '-') == 'life-042'
    assert st.simple_str_to_str('life-42', '-') == 'root/life/v_042.json'

    from collections import namedtuple

    VersionedFile = st.dict_to_namedtuple({'i01_': 'life', 'version': 42})
    assert VersionedFile == namedtuple('VersionedFile', ['i01_', 'version'])('life', 42)
    assert st.namedtuple_to_dict(VersionedFile) == {'i01_': 'life', 'version': 42}
