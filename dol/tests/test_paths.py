"""Tests for paths.py"""

from dol import StringTemplate


def test_string_template_template_construction():
    assert StringTemplate('{}.ext').template == '{_1}.ext'
    assert StringTemplate('{name}.ext').template == '{name}.ext'
    assert StringTemplate('{::\w+}.ext').template == '{_1}.ext'
    assert StringTemplate('{name::\w+}.ext').template == '{name}.ext'
    assert StringTemplate('{name::\w+}.ext').template == '{name}.ext'
    assert StringTemplate('{name:0.02f}.ext').template == '{name}.ext'
    assert StringTemplate('{name:0.02f:\w+}.ext').template == '{name}.ext'
    assert StringTemplate('{:0.02f:\w+}.ext').template == '{_1}.ext'


def test_string_template_regex():
    assert StringTemplate('{}.ext').regex.pattern == '(?P<_1>.*)\\.ext'
    assert StringTemplate('{name}.ext').regex.pattern == '(?P<name>.*)\\.ext'
    assert StringTemplate('{::\w+}.ext').regex.pattern == '(?P<_1>\\w+)\\.ext'
    assert StringTemplate('{name::\w+}.ext').regex.pattern == '(?P<name>\\w+)\\.ext'
    assert StringTemplate('{:0.02f:\w+}.ext').regex.pattern == '(?P<_1>\\w+)\\.ext'
    assert StringTemplate('{name:0.02f:\w+}.ext').regex.pattern == '(?P<name>\\w+)\\.ext'


def test_string_template_simple():
    from dol.paths import StringTemplate
    from collections import namedtuple

    st = StringTemplate(
        'root/{name}/v_{version}.json', 
        field_patterns={'name': r'\w+', 'version': r'\d+'},
        from_str_funcs={'version': int},
    )

    assert st.str_to_dict('root/Alice/v_30.json') == {'name': 'Alice', 'version': 30}
    assert st.dict_to_str({'name': 'Alice', 'version': 30}) == 'root/Alice/v_30.json'
    assert st.dict_to_tuple({'name': 'Alice', 'version': 30}) == ('Alice', 30)
    assert st.tuple_to_dict(('Alice', 30)) == {'name': 'Alice', 'version': 30}
    assert st.str_to_tuple('root/Alice/v_30.json') == ('Alice', 30)
    assert st.tuple_to_str(('Alice', 30)) == 'root/Alice/v_30.json'

    VersionedFile = st.dict_to_namedtuple({'name': 'Alice', 'version': 30})

    from collections import namedtuple
    assert VersionedFile == namedtuple('VersionedFile', ['name', 'version'])('Alice', 30)
    assert st.namedtuple_to_dict(VersionedFile) == {'name': 'Alice', 'version': 30}

    assert st.str_to_simple_str('root/Alice/v_30.json') == 'Alice,30'
    assert st.str_to_simple_str('root/Alice/v_30.json', '-') == 'Alice-30'
    assert st.simple_str_to_str('Alice-30', '-') == 'root/Alice/v_30.json'