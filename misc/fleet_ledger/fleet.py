"""Who the dependents are, where they live, and their git state.

Dependents come from **priv's** ``dep_graph`` (the fleet's single source for the
dependency graph), rebuilt live so a stale saved graph cannot drop a dependent.
That is a seam: every function that needs the fleet takes ``repos=`` — a
``{name: path}`` mapping — and only falls back to priv when it is omitted, so the
ledger works on any set of checkouts, and its tests need no fleet at all.

Records never store a path: a pin names the dependent, and the path is resolved
again, on the machine at hand, when it is needed.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path


def _priv_dep_graph():
    try:
        from priv import dep_graph
    except ImportError as e:  # pragma: no cover - depends on the machine
        raise ImportError(
            'Finding dependents needs `priv` (its dep_graph module), which is not '
            'importable here. Pass the dependents explicitly instead: '
            'repos={name: path, ...} (CLI: --repo NAME=PATH, repeatable).'
        ) from e
    return dep_graph


def fleet_dependents(package: str = 'dol', *, transitive: bool = False) -> list[str]:
    """Names of the local packages depending on *package*, from priv's dep_graph."""
    dg = _priv_dep_graph()
    graph = dg.build_dep_graph(local_only=True)
    fn = dg.transitive_dependents_of if transitive else dg.dependents_of
    return fn(package, graph)


def resolve_repos(names, *, repos: Mapping[str, str | Path] | None = None) -> dict:
    """``{name: Path | None}`` for *names*; ``None`` marks one not found locally."""
    if repos is None:
        repos = _priv_dep_graph().pth_paths_and_names()
    out = {}
    for name in names:
        p = repos.get(name)
        out[name] = Path(p) if p and Path(p).is_dir() else None
    return out


def dependents_and_paths(
    package: str = 'dol', *, repos: Mapping[str, str | Path] | None = None
) -> dict:
    """The census targets: explicit *repos*, or priv's direct dependents resolved."""
    if repos is not None:
        return {n: (Path(p) if Path(p).is_dir() else None) for n, p in repos.items()}
    return resolve_repos(fleet_dependents(package))


# ----------------------------------------------------------------------------- git


def git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ['git', '-C', str(repo), *args], capture_output=True, text=True, check=False
    )
    if check and proc.returncode:
        raise RuntimeError(f'git {" ".join(args)} failed: {proc.stderr.strip()}')
    return proc.stdout.strip() if proc.returncode == 0 else ''


def git_pin(repo: Path) -> dict:
    """Full commit, branch, upstream, ahead/behind, dirtiness and commit date."""
    repo = Path(repo)
    if not git(repo, 'rev-parse', '--is-inside-work-tree', check=False):
        return dict(git=False)
    toplevel = Path(git(repo, 'rev-parse', '--show-toplevel')).resolve()
    sha = git(repo, 'rev-parse', 'HEAD', check=False)
    branch = git(repo, 'symbolic-ref', '--short', '-q', 'HEAD', check=False) or None
    upstream = (
        git(repo, 'rev-parse', '--abbrev-ref', '@{u}', check=False) or None
        if branch
        else None
    )
    ahead = behind = None
    if upstream:
        counts = git(repo, 'rev-list', '--left-right', '--count', 'HEAD...@{u}', check=False)
        if counts:
            ahead, behind = (int(x) for x in counts.split())
    tracked = git(repo, 'status', '--porcelain', '--untracked-files=no', check=False)
    untracked = git(
        repo, 'ls-files', '--others', '--exclude-standard', check=False
    )
    return dict(
        git=True,
        sha=sha or None,
        branch=branch,
        upstream=upstream,
        ahead=ahead,
        behind=behind,
        dirty=len(tracked.splitlines()) if tracked else 0,
        untracked=len(untracked.splitlines()) if untracked else 0,
        commit_date=git(repo, 'show', '-s', '--format=%cI', 'HEAD', check=False) or None,
        subdir=Path(repo).resolve().relative_to(toplevel).as_posix(),
    )
