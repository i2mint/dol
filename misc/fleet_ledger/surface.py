"""The target package's public surface, read from a given source tree.

The surface is computed in a **subprocess** that imports the package from
*source_root* (put first on ``sys.path``), so the census is scored against the
code you point at, not whatever ``import dol`` happens to resolve to in this
interpreter. The subprocess also reports which file it actually imported, and
the scan records it: a ledger measured against the wrong copy is worse than none.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from . import store

_SURFACE_SCRIPT = r"""
import importlib, json, pkgutil, sys, types
root, pkg_name, skip = sys.argv[1], sys.argv[2], set(sys.argv[3].split(','))
sys.path.insert(0, root)
pkg = importlib.import_module(pkg_name)

def public(mod):
    names = getattr(mod, '__all__', None)
    if names is None:
        names = [n for n in dir(mod) if not n.startswith('_')]
    return sorted(set(names))

def kind(obj):
    if isinstance(obj, types.ModuleType):
        return 'module'
    if isinstance(obj, type):
        return 'class'
    if callable(obj):
        return 'function'
    return 'other'

exports = {}
for name in public(pkg):
    obj = getattr(pkg, name, None)
    k = kind(obj)
    rec = {'kind': k, 'defined_in': getattr(obj, '__module__', None)}
    if k == 'module':
        rec['defined_in'] = obj.__name__
        rec['incidental'] = not obj.__name__.startswith(pkg_name + '.')
    exports[name] = rec

submodules, defined, errors = {}, {}, {}
for info in pkgutil.walk_packages(pkg.__path__, pkg_name + '.'):
    if set(info.name.split('.')) & skip:
        continue
    try:
        mod = importlib.import_module(info.name)
    except Exception as e:  # an optional dependency missing, typically
        errors[info.name] = f'{type(e).__name__}: {e}'[:200]
        continue
    submodules[info.name] = public(mod)
    defined[info.name] = sorted(dir(mod))
submodules[pkg_name] = sorted(exports)
defined[pkg_name] = sorted(dir(pkg))
print(json.dumps({
    'package': pkg_name,
    'version': getattr(pkg, '__version__', None),
    'resolved_file': pkg.__file__,
    'exports': exports,
    'submodules': submodules,
    'defined': defined,
    'import_errors': errors,
}))
"""

#: Submodule name parts never treated as part of the public surface.
DFLT_SKIP_PARTS = ("tests", "scrap")


def package_surface(
    source_root: str | Path,
    *,
    package: str = "dol",
    python: str = sys.executable,
    skip_parts: tuple[str, ...] = DFLT_SKIP_PARTS,
) -> dict:
    """Exports of *package*'s ``__init__`` and every submodule's public names.

    ``submodules`` holds what ``from <module> import *`` would bind; ``defined``
    holds every attribute (underscored included), to tell a compat name that
    exists from one that no longer does.

    ``resolved_file`` is relative to *source_root* when the import came from it,
    and ``imported_from_source`` says whether it did.
    """
    source_root = Path(source_root).resolve()
    proc = subprocess.run(
        [
            python,
            "-I",
            "-c",
            _SURFACE_SCRIPT,
            str(source_root),
            package,
            ",".join(skip_parts),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=source_root,
    )
    if proc.returncode:
        raise RuntimeError(
            f"Could not import {package} from its source tree:\n{proc.stderr[-2000:]}"
        )
    surface = json.loads(proc.stdout)
    surface["version"] = surface["version"] or _pyproject_version(source_root)
    surface["import_errors"] = {
        k: store.redact(store.scrub(v)) for k, v in surface["import_errors"].items()
    }
    resolved = Path(surface["resolved_file"]).resolve()
    try:
        surface["resolved_file"] = resolved.relative_to(source_root).as_posix()
        surface["imported_from_source"] = True
    except ValueError:
        surface["imported_from_source"] = False
    return surface


def _pyproject_version(source_root: Path) -> str | None:
    """``[project].version`` of the source tree (the package may not set ``__version__``)."""
    try:
        import tomllib
    except ImportError:  # Python 3.10
        return None
    p = Path(source_root) / "pyproject.toml"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return tomllib.load(f).get("project", {}).get("version")


def is_shallow(source_root: str | Path) -> bool:
    """Whether the git history is truncated (then first-seen dates are not real)."""
    proc = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "--is-shallow-repository"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() == "true"


def parse_iso(s: str):
    """``datetime.fromisoformat`` that also takes git's ``Z`` suffix on Python 3.10."""
    from datetime import datetime

    return datetime.fromisoformat(s[:-1] + "+00:00" if s.endswith("Z") else s)


def export_first_seen(
    source_root: str | Path, names, *, init_relpath: str = "dol/__init__.py"
) -> dict[str, str]:
    """``{name: ISO date}`` of the first commit whose diff adds *name* to the init file.

    One ``git log -p`` pass over the init file's history, oldest first. Names never
    seen in an added line are left out (exported indirectly, e.g. via ``*``).
    """
    wanted = set(names)
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "log",
            "--reverse",
            "--format=@@COMMIT@@ %cI",
            "-p",
            "--unified=0",
            "--",
            init_relpath,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    seen: dict[str, str] = {}
    date = None
    ident = re.compile(r"[A-Za-z_]\w*")
    for line in proc.stdout.splitlines():
        if line.startswith("@@COMMIT@@ "):
            date = line.split(" ", 1)[1]
        elif line.startswith("+") and not line.startswith("+++"):
            for tok in ident.findall(line):
                if tok in wanted and tok not in seen:
                    seen[tok] = date
    return seen
