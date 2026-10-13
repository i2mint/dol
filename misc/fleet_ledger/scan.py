"""One scan: pin every dependent, census its use of the target, write an immutable record.

A scan is the unit of the ledger: :func:`run_scan` writes ``scans/<id>/`` once
and never touches it again, so successive scans form a history. The end-to-end
test (``tests/test_ledger.py``) runs it on throwaway git repos.
"""

from __future__ import annotations

import platform
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

from . import astscan, fleet, store, surface as surface_mod

TOOL_NAME = 'fleet_ledger'


def _repo_census(path: Path, *, package: str, surface: dict) -> dict:
    submodules = [m for m in surface['submodules'] if m != package]
    return astscan.scan_repo(
        path,
        package=package,
        submodules=submodules,
        module_surfaces=surface['submodules'],
    )


def aggregate(census: Mapping[str, dict], *, surface: dict) -> dict:
    """Fleet-level counts from per-dependent records.

    - ``symbols``: per target name, the packages and sites that import, reference,
      call or subclass it (the input to the zero-usage report).
    - ``calls``: per callee, call sites, decorator and class-decorator sites, and a
      per-keyword census (packages, sites, value kinds).
    - ``subclasses``: per target base class, packages and classes, direct vs indirect.
    - ``compat_names``: names used from a submodule that ``__init__`` does not export.
    - ``unknown_names``: names imported from the target that no longer exist in it.
    """
    package = surface['package']
    exports = set(surface['exports'])
    module_names = {m: set(ns) for m, ns in surface['submodules'].items()}
    symbols = defaultdict(lambda: dict(packages=set(), import_sites=0, ref_sites=0))
    calls = defaultdict(
        lambda: dict(
            packages=set(),
            sites=0,
            decorator_sites=0,
            class_decorator_sites=0,
            kwargs=defaultdict(lambda: dict(packages=set(), sites=0, kinds=Counter())),
        )
    )
    subclasses = defaultdict(
        lambda: dict(packages=set(), classes=0, direct=0, indirect=0)
    )
    compat = defaultdict(lambda: dict(modules=set(), packages=set()))
    unknown = defaultdict(lambda: dict(modules=set(), packages=set()))
    star_packages = defaultdict(set)

    for pkg, rec in census.items():
        if not rec.get('scanned'):
            continue
        for imp in rec['imports']:
            name, mod = imp['name'], imp['module']
            if name in (None, '*'):
                continue
            if f'{mod}.{name}' in module_names:
                continue  # a submodule import, not a symbol
            symbols[name]['packages'].add(pkg)
            symbols[name]['import_sites'] += 1
            if mod == package:
                if name not in exports:
                    unknown[name]['modules'].add(mod)
                    unknown[name]['packages'].add(pkg)
            elif name not in exports:
                target = compat if name in module_names.get(mod, ()) else unknown
                target[name]['modules'].add(mod)
                target[name]['packages'].add(pkg)
        for ref in rec['refs']:
            symbols[ref['symbol']]['packages'].add(pkg)
            symbols[ref['symbol']]['ref_sites'] += 1
        for c in rec['calls']:
            agg = calls[c['symbol']]
            agg['packages'].add(pkg)
            agg['sites'] += 1
            agg['decorator_sites'] += bool(c['decorates'])
            agg['class_decorator_sites'] += c['decorates'] == 'class'
            for kw in c['kwargs']:
                k = agg['kwargs'][kw['name']]
                k['packages'].add(pkg)
                k['sites'] += 1
                k['kinds'][kw['kind']] += 1
        for cls in rec['subclasses']:
            for root in cls['direct']:
                s = subclasses[root]
                s['packages'].add(pkg)
                s['classes'] += 1
                s['direct'] += 1
            for root in cls['indirect']:
                s = subclasses[root]
                s['packages'].add(pkg)
                s['classes'] += 1
                s['indirect'] += 1
        for st in rec['stars']:
            star_packages[st['module']].add(pkg)

    def finish(d):
        if isinstance(d, (dict, defaultdict)):
            return {k: finish(v) for k, v in sorted(d.items())}
        if isinstance(d, set):
            return sorted(d)
        if isinstance(d, Counter):
            return dict(sorted(d.items()))
        return d

    out = finish(
        dict(
            symbols=symbols,
            calls=calls,
            subclasses=subclasses,
            compat_names=compat,
            unknown_names=unknown,
            star_imports=star_packages,
        )
    )
    for d in (out['symbols'], out['calls'], out['subclasses']):
        for v in d.values():
            v['n_packages'] = len(v['packages'])
    for v in out['calls'].values():
        for k in v['kwargs'].values():
            k['n_packages'] = len(k['packages'])
    return out


def run_scan(
    package: str = 'dol',
    *,
    source_root: str | Path | None = None,
    repos: Mapping[str, str | Path] | None = None,
    root: str | Path | None = None,
    note: str | None = None,
) -> str:
    """Pin and census every dependent; write ``scans/<id>/`` and index it. Returns the id.

    :param source_root: the target's source tree (default: the dol checkout holding
        this tool).
    :param repos: ``{name: path}`` of dependents (default: priv's direct dependents).
    :param root: the ledger directory (default: :func:`store.ledger_root`).
    """
    source_root = Path(source_root or Path(__file__).resolve().parents[2])
    root = Path(root) if root else store.ledger_root(package)
    surf = surface_mod.package_surface(source_root, package=package)
    surf['export_first_seen'] = surface_mod.export_first_seen(
        source_root, surf['exports'], init_relpath=f'{package}/__init__.py'
    )
    targets = fleet.dependents_and_paths(package, repos=repos)
    transitive = None
    if repos is None:
        transitive = fleet.fleet_dependents(package, transitive=True)

    pins, census = {}, {}
    for name, path in sorted(targets.items()):
        if path is None:
            pins[name] = dict(found=False)
            census[name] = dict(scanned=False, reason='not found locally')
            continue
        pins[name] = dict(found=True, **fleet.git_pin(path))
        census[name] = dict(scanned=True, **_repo_census(path, package=package, surface=surf))

    scans_dir = root / 'scans'
    scan_id = store.new_record_id(scans_dir)
    created = store.utc_now().isoformat()
    target_pin = fleet.git_pin(source_root)
    meta = dict(
        schema_version=store.SCHEMA_VERSION,
        kind='scan',
        id=scan_id,
        created=created,
        tool=TOOL_NAME,
        package=package,
        package_version=surf['version'],
        package_pin=target_pin,
        import_resolved=dict(
            file=surf['resolved_file'], from_source=surf['imported_from_source']
        ),
        python=platform.python_version(),
        dependents_source='explicit' if repos is not None else 'priv.dep_graph',
        n_dependents=len(targets),
        n_found=sum(p is not None for p in targets.values()),
        transitive_dependents=transitive,
        note=note,
    )
    d = scans_dir / scan_id
    store.write_json(d / 'surface.json', dict(schema_version=store.SCHEMA_VERSION, **surf))
    store.write_json(
        d / 'pins.json',
        dict(
            schema_version=store.SCHEMA_VERSION,
            scan_id=scan_id,
            package_pin=target_pin,
            pins=pins,
        ),
    )
    store.write_json(
        d / 'census.json',
        dict(
            schema_version=store.SCHEMA_VERSION,
            scan_id=scan_id,
            dependents=census,
            aggregates=aggregate(census, surface=surf),
        ),
    )
    store.write_json(d / 'meta.json', meta)
    store.add_to_index(
        root,
        'scans',
        dict(
            id=scan_id,
            created=created,
            package_sha=target_pin.get('sha'),
            n_dependents=len(targets),
            note=note,
        ),
    )
    return scan_id


def load_scan(root: Path, scan_id: str | None = None) -> dict:
    """``{'meta', 'surface', 'pins', 'census'}`` of a scan (latest by default)."""
    root = Path(root)
    scan_id = scan_id or store.latest_id(root, 'scans')
    d = root / 'scans' / scan_id
    return {
        k: store.read_json(d / f'{k}.json') for k in ('meta', 'surface', 'pins', 'census')
    }
