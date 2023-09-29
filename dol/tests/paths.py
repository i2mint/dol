"""Tests for paths.py"""


def test_string_template():
    from dol.paths import StringTemplate
    from collections import namedtuple

    st = StringTemplate('{name} is {age} years old.', {'name': r'\w+', 'age': r'\d+'})

    assert st.str_to_dict('Alice is 30 years old.') == {'name': 'Alice', 'age': '30'}
    assert st.dict_to_str({'name': 'Alice', 'age': '30'}) == 'Alice is 30 years old.'
    assert st.dict_to_tuple({'name': 'Alice', 'age': '30'}) == ('Alice', '30')
    assert st.tuple_to_dict(('Alice', '30')) == {'name': 'Alice', 'age': '30'}
    assert st.str_to_tuple('Alice is 30 years old.') == ('Alice', '30')
    assert st.tuple_to_str(('Alice', '30')) == 'Alice is 30 years old.'

    Person = st.dict_to_namedtuple({'name': 'Alice', 'age': '30'}, 'Person')
    assert Person == namedtuple('Person', ['name', 'age'])('Alice', '30')
    assert st.namedtuple_to_dict(Person) == {'name': 'Alice', 'age': '30'}

    assert st.str_to_simple_str('Alice is 30 years old.', '-') == 'Alice-30'
    assert st.simple_str_to_str('Alice-30', '-') == 'Alice is 30 years old.'
