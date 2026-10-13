"""AST census of how one source tree uses a target package (``dol`` by default).

Stdlib only, and pure: it reads source text and returns plain dicts, so it can be
tested on strings and reused for any package.

What it detects, per file:

- **imports** of the target, through every binding form: ``import dol``,
  ``import dol.trans as t``, ``from dol import X``, ``from dol import X as Y``,
  ``from dol.trans import X``, ``from dol import trans`` (a submodule), and star
  imports (``from dol import *``, ``from dol.errors import *``).
- **references**: any expression that resolves to a target symbol
  (``wrap_kvs``, ``dol.wrap_kvs``, ``t.wrap_kvs``, ``base.KvReader``...). Under a
  star import, a free name counts when it is in the star'd module's own public
  surface and no local binding shadows it; such records carry ``via="star"``.
- **call sites**: every call whose callee resolves to a target symbol, with each
  keyword argument's name and the *kind* of its value (lambda, name, attribute,
  call, constant...), and whether the call is a decorator (and of a class).
- **classes**: each class statement's bases, split into target symbols and local
  names, so :func:`close_class_graph` can find classes that subclass a target
  class through the repo's own hierarchy.

>>> rec = scan_source('''
... from dol import wrap_kvs, KvReader as R
... @wrap_kvs(obj_of_data=lambda x: x)
... class A(R): ...
... class B(A): ...
... ''')
>>> [(c['symbol'], c['decorates'], c['kwargs']) for c in rec['calls']]
[('wrap_kvs', 'class', [{'name': 'obj_of_data', 'kind': 'lambda'}])]
>>> [(c['name'], c['direct'], c['indirect']) for c in close_class_graph(rec['classes'])]
[('A', ['KvReader'], []), ('B', [], ['KvReader'])]
"""

from __future__ import annotations

import ast
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

DFLT_PACKAGE = "dol"

#: Directory names never descended into, at any depth (hidden dirs are skipped too).
DFLT_SKIP_DIRS = frozenset({"node_modules", "__pycache__", "site-packages"})
#: Directory names skipped only at the top of a repo: there they are build output
#: or a virtualenv, while deeper down they can be a real subpackage.
DFLT_TOP_SKIP_DIRS = frozenset({"build", "dist", "env", "venv"})

_VALUE_KINDS = {
    ast.Lambda: "lambda",
    ast.Name: "name",
    ast.Attribute: "attribute",
    ast.Call: "call",
    ast.Constant: "constant",
    ast.Subscript: "subscript",
    ast.Starred: "starred",
}


@dataclass(frozen=True)
class Binding:
    """What a local name is bound to: a target module, or a symbol in one."""

    module: str
    symbol: str | None = None  # None: the binding is the module itself
    via: str = "import"  # 'import' | 'star'


def value_kind(node: ast.AST) -> str:
    """The kind of an argument value, which decides whether a rewrite is mechanical.

    >>> value_kind(ast.parse('lambda x: x', mode='eval').body)
    'lambda'
    >>> value_kind(ast.parse('json.loads', mode='eval').body)
    'attribute'
    >>> value_kind(ast.parse('[1]', mode='eval').body)
    'list'
    """
    return _VALUE_KINDS.get(type(node), type(node).__name__.lower())


def _is_target(module: str | None, package: str) -> bool:
    return bool(module) and (module == package or module.startswith(package + "."))


def _locally_bound_names(tree: ast.AST, package: str) -> set[str]:
    """Every name the module binds itself, anywhere (scope-insensitive, conservative).

    Used only to decide that a free name is *not* coming from a star import.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and _is_target(node.module, package):
                continue  # target imports are bindings, handled separately
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name.split(".")[0])
    return names


def _target_names(node: ast.AST) -> Iterator[str]:
    """Names bound by an assignment target (``a``, ``a, b``, ``[a, *b]``)."""
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            yield from _target_names(elt)
    elif isinstance(node, ast.Starred):
        yield from _target_names(node.value)


def _stmt_binds(node: ast.stmt) -> Iterator[str]:
    """Names a statement itself binds, other than target imports."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield node.name
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            yield from _target_names(t)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
        yield from _target_names(node.target)
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        yield from _target_names(node.target)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            if item.optional_vars is not None:
                yield from _target_names(item.optional_vars)


def _import_bindings(
    tree: ast.AST, *, package: str, submodules: Iterable[str]
) -> tuple[dict[str, list], list[dict], list[str]]:
    """Binding events, import records, and star-imported module names.

    Events are ``{name: [(position, Binding | None), ...]}`` sorted by position,
    where a position is where the binding takes effect (the end of its
    statement) and ``None`` means the name was rebound to something else (a
    non-target import, a def, an assignment).
    """
    submodules = set(submodules)
    events: dict[str, list] = {}
    imports: list[dict] = []
    stars: list[str] = []

    def bind(name, pos, binding):
        events.setdefault(name, []).append((pos, binding))

    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        pos = (node.end_lineno, node.end_col_offset)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_target(alias.name, package):
                    bind(alias.asname or alias.name.split(".")[0], pos, None)
                    continue
                if alias.asname:
                    bind(alias.asname, pos, Binding(alias.name))
                else:  # `import dol.trans` binds `dol`
                    top = alias.name.split(".")[0]
                    bind(top, pos, Binding(top))
                imports.append(
                    dict(
                        line=node.lineno,
                        module=alias.name,
                        name=None,
                        asname=alias.asname,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            target = not node.level and _is_target(node.module, package)
            for alias in node.names:
                if alias.name == "*":
                    if target:
                        stars.append(node.module)
                        imports.append(
                            dict(
                                line=node.lineno,
                                module=node.module,
                                name="*",
                                asname=None,
                            )
                        )
                    continue
                local = alias.asname or alias.name
                if not target:
                    bind(local, pos, None)
                    continue
                imports.append(
                    dict(
                        line=node.lineno,
                        module=node.module,
                        name=alias.name,
                        asname=alias.asname,
                    )
                )
                full = f"{node.module}.{alias.name}"
                if full in submodules:
                    bind(local, pos, Binding(full))
                else:
                    bind(local, pos, Binding(node.module, alias.name))
        else:
            for name in _stmt_binds(node):
                bind(name, pos, None)
    for evs in events.values():
        evs.sort(key=lambda e: e[0])
    return events, imports, stars


def _is_partial(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == "partial") or (
        isinstance(node, ast.Attribute) and node.attr == "partial"
    )


class _Resolver:
    """Resolve an expression to a target :class:`Binding`, at its source position.

    A name resolves to the binding in effect at that point of the file (the latest
    event before it; the first event if it is used before any, as in a function
    body referring to a later import). Scope is ignored: an import inside one
    function binds the name for the whole file.
    """

    def __init__(self, events, *, star_names, submodules):
        self.events = events
        self.star_names = star_names  # {name: Binding(via='star')}
        self.submodules = set(submodules)

    def _name(self, node: ast.Name) -> Binding | None:
        evs = self.events.get(node.id)
        if not evs:
            return self.star_names.get(node.id)
        here = (node.lineno, node.col_offset)
        i = bisect_right([pos for pos, _ in evs], here)
        return evs[i - 1][1] if i else evs[0][1]

    def __call__(self, node: ast.AST) -> Binding | None:
        if isinstance(node, ast.Name):
            return self._name(node)
        if isinstance(node, ast.Attribute):
            base = self(node.value)
            if base is None:
                return None
            if base.symbol is not None:  # an attribute of a target symbol: `Store.wrap`
                return Binding(base.module, f"{base.symbol}.{node.attr}", via=base.via)
            full = f"{base.module}.{node.attr}"
            if full in self.submodules:
                return Binding(full, via=base.via)
            return Binding(base.module, node.attr, via=base.via)
        return None


def _decorator_targets(tree: ast.AST) -> dict[int, str]:
    """``{id(decorator_node): 'class' | 'function'}`` for every decorator expression."""
    targets = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            for d in node.decorator_list:
                targets[id(d)] = kind
    return targets


def scan_source(
    source: str,
    *,
    package: str = DFLT_PACKAGE,
    submodules: Iterable[str] = (),
    module_surfaces: Mapping[str, Iterable[str]] | None = None,
    relpath: str = "<string>",
) -> dict:
    """Census of one module's source text.

    :param package: the target package (``'dol'``).
    :param submodules: dotted names of the target's submodules (``'dol.trans'``...),
        so ``from dol import trans`` is known to bind a module, not a symbol. Leave
        out a submodule whose name the package rebinds to a non-module export.
    :param module_surfaces: ``{module: public names}`` used to resolve star
        imports (the names ``from <module> import *`` would bind).
    :param relpath: the file's repo-relative path, copied into every record.

    Returns ``{'imports', 'refs', 'calls', 'classes', 'stars'}``; each record has
    ``file`` and ``line``. Refs are to plain target names; calls also cover
    attributes of target symbols (``Store.wrap(...)``, symbol ``'Store.wrap'``) and
    ``partial(<target callable>, ...)`` (``via_partial=True``).
    """
    tree = ast.parse(source)
    submodules = set(submodules)
    module_surfaces = module_surfaces or {}
    events, imports, stars = _import_bindings(
        tree, package=package, submodules=submodules
    )
    star_names: dict[str, Binding] = {}
    if stars:
        local = _locally_bound_names(tree, package) | set(events)
        for mod in stars:
            for name in module_surfaces.get(mod, ()):
                if name not in local and name not in star_names:
                    full = f"{mod}.{name}"
                    if full in submodules:
                        star_names[name] = Binding(full, via="star")
                    else:
                        star_names[name] = Binding(mod, name, via="star")
    resolve = _Resolver(events, star_names=star_names, submodules=submodules)
    decorates = _decorator_targets(tree)

    refs, calls, classes = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(
            node.ctx, ast.Load
        ):
            b = resolve(node)
            if b is not None and b.symbol is not None:
                if "." not in b.symbol:
                    refs.append(
                        dict(
                            file=relpath,
                            line=node.lineno,
                            symbol=b.symbol,
                            module=b.module,
                            via=b.via,
                        )
                    )
                if id(node) in decorates:  # bare decorator: `@cache_this`
                    calls.append(
                        _call_record(
                            node,
                            b,
                            relpath=relpath,
                            decorates=decorates[id(node)],
                            bare=True,
                        )
                    )
        if isinstance(node, ast.Call):
            b = resolve(node.func)
            if b is not None and b.symbol is not None:
                calls.append(
                    _call_record(
                        node, b, relpath=relpath, decorates=decorates.get(id(node))
                    )
                )
            elif _is_partial(node.func) and node.args:
                b = resolve(node.args[0])
                if b is not None and b.symbol is not None:
                    calls.append(
                        _call_record(
                            node,
                            b,
                            relpath=relpath,
                            decorates=decorates.get(id(node)),
                            partial=True,
                        )
                    )
        if isinstance(node, ast.ClassDef):
            target_bases, local_bases = [], []
            for base in node.bases:
                if isinstance(
                    base, ast.Call
                ):  # class A(Store.wrap(...)) / A(wrap_kvs(...))
                    b = resolve(base.func)
                    if b is not None and b.symbol is not None:
                        target_bases.append(b.symbol.split(".")[0])
                    continue
                b = resolve(base)
                if b is not None and b.symbol is not None:
                    target_bases.append(b.symbol.split(".")[0])
                elif isinstance(base, ast.Name):
                    local_bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    local_bases.append(base.attr)
            classes.append(
                dict(
                    file=relpath,
                    line=node.lineno,
                    name=node.name,
                    target_bases=target_bases,
                    local_bases=local_bases,
                )
            )
    for rec in imports:
        rec["file"] = relpath
    return dict(imports=imports, refs=refs, calls=calls, classes=classes, stars=stars)


def _call_record(
    node, binding, *, relpath, decorates=None, bare=False, partial=False
) -> dict:
    if bare:
        kwargs, n_args, double_star = [], 0, False
    else:
        kwargs = [
            dict(name=kw.arg, kind=value_kind(kw.value))
            for kw in node.keywords
            if kw.arg is not None
        ]
        n_args = len(node.args) - partial
        double_star = any(kw.arg is None for kw in node.keywords)
    return dict(
        file=relpath,
        line=node.lineno,
        symbol=binding.symbol,
        module=binding.module,
        via=binding.via,
        kwargs=kwargs,
        n_args=n_args,
        double_star=double_star,
        decorates=decorates,
        bare_decorator=bare,
        via_partial=partial,
    )


def close_class_graph(classes: Iterable[dict]) -> list[dict]:
    """Classes that subclass a target symbol, directly or through local bases.

    A class's roots are its own target bases *plus* every root of its local bases
    (looked up by class name within the same repo): a class rooted both directly
    and through a local base reports both. ``indirect`` lists roots reached only
    through a local base.

    >>> cls = [dict(file='a.py', line=1, name='A', target_bases=['Store'], local_bases=[]),
    ...        dict(file='a.py', line=2, name='B', target_bases=['KvReader'], local_bases=['A'])]
    >>> [(c['name'], c['direct'], c['indirect']) for c in close_class_graph(cls)]
    [('A', ['Store'], []), ('B', ['KvReader'], ['Store'])]
    """
    classes = list(classes)
    by_name: dict[str, list[dict]] = {}
    for c in classes:
        by_name.setdefault(c["name"], []).append(c)
    memo: dict[int, frozenset] = {}

    def roots(c, visiting=frozenset()) -> frozenset:
        key = id(c)
        if key in memo:
            return memo[key]
        if key in visiting:
            return frozenset()
        acc = set(c["target_bases"])
        for base in c["local_bases"]:
            for parent in by_name.get(base, ()):
                if parent is not c:
                    acc |= roots(parent, visiting | {key})
        memo[key] = frozenset(acc)
        return memo[key]

    out = []
    for c in classes:
        r = roots(c)
        if r:
            direct = sorted(set(c["target_bases"]))
            out.append(
                dict(
                    file=c["file"],
                    line=c["line"],
                    name=c["name"],
                    direct=direct,
                    indirect=sorted(r - set(direct)),
                )
            )
    return out


def iter_py_files(
    root: Path,
    *,
    skip_dirs: frozenset[str] = DFLT_SKIP_DIRS,
    top_skip_dirs: frozenset[str] = DFLT_TOP_SKIP_DIRS,
) -> Iterator[Path]:
    """Python files under *root*, skipping hidden dirs and the skip sets, sorted."""
    root = Path(root)
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).parts[:-1]
        if any(p.startswith(".") or p in skip_dirs for p in parts):
            continue
        if parts and parts[0] in top_skip_dirs:
            continue
        yield path


def scan_repo(
    root: Path,
    *,
    package: str = DFLT_PACKAGE,
    submodules: Iterable[str] = (),
    module_surfaces: Mapping[str, Iterable[str]] | None = None,
    skip_dirs: frozenset[str] = DFLT_SKIP_DIRS,
) -> dict:
    """Census of a whole source tree: per-file records concatenated, classes closed.

    All ``file`` fields are POSIX paths relative to *root*; nothing absolute is kept.
    """
    root = Path(root)
    submodules = set(submodules)
    acc = dict(imports=[], refs=[], calls=[], classes=[], stars=[])
    parse_errors = []
    n_files = 0
    for path in iter_py_files(root, skip_dirs=skip_dirs):
        relpath = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            rec = scan_source(
                source,
                package=package,
                submodules=submodules,
                module_surfaces=module_surfaces,
                relpath=relpath,
            )
        except (SyntaxError, ValueError):
            parse_errors.append(relpath)
            continue
        n_files += 1
        for k in ("imports", "refs", "calls", "classes"):
            acc[k].extend(rec[k])
        acc["stars"].extend(dict(file=relpath, module=m) for m in rec["stars"])
    subclasses = close_class_graph(acc.pop("classes"))
    return dict(
        n_files=n_files,
        parse_errors=parse_errors,
        imports=acc["imports"],
        refs=acc["refs"],
        calls=acc["calls"],
        stars=acc["stars"],
        subclasses=subclasses,
    )
