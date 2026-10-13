"""The census's resolution rules, including the two defects found by hand-checking it.

Both defects were found by reading the generated census against source, not by
tests; they are frozen here so they cannot come back.
"""

import textwrap

from fleet_ledger.astscan import close_class_graph, scan_repo, scan_source

SUBMODULES = {'dol.base', 'dol.trans', 'dol.errors'}
SURFACES = {
    'dol': ['KvReader', 'Store', 'wrap_kvs', 'Files', 'cache_this'],
    'dol.errors': ['NotValid', 'OverWritesNotAllowedError'],
    'dol.base': ['KvReader', 'Store', 'Collection'],
}


def scan(src):
    return scan_source(
        textwrap.dedent(src), submodules=SUBMODULES, module_surfaces=SURFACES
    )


def symbols(rec, key='refs'):
    return sorted({r['symbol'] for r in rec[key]})


def test_every_binding_form_resolves_to_the_dol_name():
    rec = scan(
        """
        import dol
        import dol.trans
        import dol.trans as t
        from dol import Files as F
        from dol import base
        from dol.base import Collection
        dol.wrap_kvs; dol.trans.wrap_kvs; t.wrap_kvs; F; base.KvReader; Collection
        """
    )
    assert symbols(rec) == ['Collection', 'Files', 'KvReader', 'wrap_kvs']
    assert sum(r['symbol'] == 'wrap_kvs' for r in rec['refs']) == 3


def test_from_dol_import_submodule_binds_a_module_not_a_symbol():
    rec = scan('from dol import base\nclass A(base.Store): ...\n')
    assert rec['classes'][0]['target_bases'] == ['Store']
    assert 'base' not in symbols(rec)


def test_non_dol_and_relative_imports_are_ignored():
    rec = scan('from .dol import X\nfrom dolly import Y\nimport doll\nX; Y\n')
    assert rec['imports'] == [] and rec['refs'] == []


def test_call_site_kwargs_and_their_kinds():
    rec = scan(
        """
        from dol import wrap_kvs
        import json
        def f(x): return x
        s = wrap_kvs({}, obj_of_data=json.loads, data_of_obj=f,
                     key_of_id=lambda k: k, id_of_key=str.upper, **extra)
        """
    )
    (call,) = rec['calls']
    kinds = {k['name']: k['kind'] for k in call['kwargs']}
    assert kinds == {
        'obj_of_data': 'attribute',
        'data_of_obj': 'name',
        'key_of_id': 'lambda',
        'id_of_key': 'attribute',
    }
    assert call['n_args'] == 1 and call['double_star'] and call['decorates'] is None


def test_decorators_bare_and_called_class_and_function():
    rec = scan(
        """
        from dol import wrap_kvs, cache_this
        @wrap_kvs(key_of_id=str)
        class A(dict): ...
        class B:
            @cache_this
            def x(self): ...
        """
    )
    by = {c['symbol']: c for c in rec['calls']}
    assert by['wrap_kvs']['decorates'] == 'class' and not by['wrap_kvs']['bare_decorator']
    assert by['cache_this']['decorates'] == 'function' and by['cache_this']['bare_decorator']


def test_star_import_resolves_against_the_starred_submodule_surface():
    """Defect 2: `from dol.errors import *` was scored against `__init__`."""
    rec = scan('from dol.errors import *\nNotValid; Files\n')
    assert symbols(rec) == ['NotValid']
    assert rec['refs'][0]['via'] == 'star' and rec['refs'][0]['module'] == 'dol.errors'


def test_star_import_yields_to_local_bindings():
    rec = scan('from dol import *\ndef Files(): ...\nStore = 1\nFiles; Store; KvReader\n')
    assert symbols(rec) == ['KvReader']


def test_class_rooted_directly_and_through_a_local_base_reports_both():
    """Defect 1: such a class reported only its direct base (a 3x undercount)."""
    rec = scan(
        """
        from dol.base import Collection, KvReader
        class Base(Collection): ...
        class Both(Base, KvReader): ...
        class Deep(Both): ...
        class Plain: ...
        """
    )
    got = {c['name']: (c['direct'], c['indirect']) for c in close_class_graph(rec['classes'])}
    assert got == {
        'Base': (['Collection'], []),
        'Both': (['KvReader'], ['Collection']),
        'Deep': ([], ['Collection', 'KvReader']),
    }


def test_class_graph_survives_cycles_and_unknown_bases():
    cls = [
        dict(file='a', line=1, name='A', target_bases=[], local_bases=['B']),
        dict(file='a', line=2, name='B', target_bases=['Store'], local_bases=['A', 'Nope']),
    ]
    got = {c['name']: c['indirect'] + c['direct'] for c in close_class_graph(cls)}
    assert got == {'A': ['Store'], 'B': ['Store']}


def test_scan_repo_closes_classes_across_files_and_keeps_relative_paths(tmp_path):
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg' / 'a.py').write_text('from dol import Store\nclass A(Store): ...\n')
    (tmp_path / 'pkg' / 'b.py').write_text('from pkg.a import A\nclass B(A): ...\n')
    (tmp_path / 'pkg' / 'bad.py').write_text('def (:\n')
    (tmp_path / '.venv').mkdir()
    (tmp_path / '.venv' / 'x.py').write_text('from dol import Files\n')
    rec = scan_repo(tmp_path, submodules=SUBMODULES)
    assert rec['parse_errors'] == ['pkg/bad.py']
    assert {c['name']: c['indirect'] for c in rec['subclasses']} == {'A': [], 'B': ['Store']}
    assert {r['file'] for r in rec['imports']} == {'pkg/a.py'}


def test_attribute_of_a_symbol_is_a_call_and_a_call_base_roots_the_class():
    rec = scan(
        """
        from dol import Store, wrap_kvs
        from functools import partial
        s = Store.wrap(dict, obj_of_data=str)
        class W(Store.wrap(dict)): ...
        class V(wrap_kvs(dict, key_of_id=str)): ...
        mk = partial(wrap_kvs, postget=f, preset=g)
        """
    )
    calls = {(c['symbol'], c['via_partial']): c for c in rec['calls'] if c['kwargs']}
    assert [k['name'] for k in calls[('Store.wrap', False)]['kwargs']] == ['obj_of_data']
    assert sum(c['symbol'] == 'Store.wrap' for c in rec['calls']) == 2
    assert [k['name'] for k in calls[('wrap_kvs', True)]['kwargs']] == ['postget', 'preset']
    assert calls[('wrap_kvs', True)]['n_args'] == 0
    got = {c['name']: c['direct'] for c in close_class_graph(rec['classes'])}
    assert got == {'W': ['Store'], 'V': ['wrap_kvs']}
    assert 'Store.wrap' not in symbols(rec)  # refs stay plain names


def test_a_later_non_target_binding_overrides_the_target_one():
    rec = scan(
        """
        from dol import Files
        class A(Files): ...
        from mylib import Files
        class B(Files): ...
        Files()
        """
    )
    assert [c['name'] for c in close_class_graph(rec['classes'])] == ['A']
    assert rec['calls'] == []


def test_a_binding_takes_effect_after_its_statement():
    rec = scan('from dol import Store\nclass Store(Store): ...\nStore()\n')
    assert close_class_graph(rec['classes'])[0]['direct'] == ['Store']
    assert rec['calls'] == []  # by then `Store` is the local class


def test_top_level_build_dirs_skipped_but_nested_ones_scanned(tmp_path):
    for d in ('build', 'pkg/build'):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / 'm.py').write_text('from dol import Files\n')
    rec = scan_repo(tmp_path)
    assert [r['file'] for r in rec['imports']] == ['pkg/build/m.py']
