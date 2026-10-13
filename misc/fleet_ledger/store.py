"""Where the ledger lives, how records are written, and the privacy guard.

The ledger is *data derived from the fleet's source*, and some dependents are
private repos, so it never goes in the (public) dol repo. It lives under
:func:`ledger_root` (``~/.local/share/dol/fleet_ledger/<package>/`` by default,
``$DOL_FLEET_LEDGER_DIR`` to override), laid out as::

    index.json                    every scan and baseline, with supersede marks
    scans/<id>/meta.json          what was scanned, when, against which dol
    scans/<id>/surface.json       the target's public surface at that commit
    scans/<id>/pins.json          per-dependent commit pins (small, for restore)
    scans/<id>/census.json        per-dependent AST census + aggregates
    baselines/<id>/baseline.json  per-dependent suite results
    pins/undo-<id>.json           what `pin restore --apply` changed

Records are immutable: :func:`write_json` refuses to overwrite, and refuses to
persist anything that still looks like an absolute local path after scrubbing.

>>> scrub('see /root/py/proj/i/dol/x.py', replacements={'/root/py/proj': '<PP>'})
'see <PP>/i/dol/x.py'
>>> find_local_paths({'a': ['/home/alice/x']})
['/home/alice/x']
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

#: Bumped whenever a record's shape changes incompatibly.
SCHEMA_VERSION = 1

LEDGER_DIR_ENVVAR = 'DOL_FLEET_LEDGER_DIR'
DFLT_LEDGER_BASE = Path('~/.local/share/dol/fleet_ledger')

#: Patterns of strings that identify a local machine or user.
LOCAL_PATH_PATTERNS = (
    re.compile(r'(?<![\w.])/(?:root|home/[^/\s]+|Users/[^/\s]+)(?=/|\b)'),
    re.compile(r'\b[A-Za-z]:\\(?:Users|Documents and Settings)\\'),
    re.compile(r'/tmp/pytest-of-[^/\s]+'),
)


class LocalPathLeak(ValueError):
    """Raised instead of writing a record that still contains a local path."""


def ledger_root(package: str = 'dol', *, base: str | Path | None = None) -> Path:
    """The directory holding *package*'s ledger (not created here)."""
    base = base or os.environ.get(LEDGER_DIR_ENVVAR) or DFLT_LEDGER_BASE
    return Path(base).expanduser() / package


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def new_record_id(parent: Path, *, now: datetime | None = None) -> str:
    """A fresh, sortable UTC id (``20261013T101500Z``) not yet used under *parent*."""
    stem = (now or utc_now()).strftime('%Y%m%dT%H%M%SZ')
    rid, n = stem, 1
    while (parent / rid).exists():
        n += 1
        rid = f'{stem}-{n}'
    return rid


def default_replacements() -> dict[str, str]:
    """Local prefixes to rewrite before writing, most specific first."""
    reps = {}
    pp = os.environ.get('PP')
    if pp:
        reps[str(Path(pp).expanduser())] = '<PP>'
    reps[str(Path.home())] = '~'
    return reps


def scrub(text: str, *, replacements: Mapping[str, str] | None = None) -> str:
    """Rewrite known local prefixes (longest first) in *text*."""
    reps = default_replacements() if replacements is None else replacements
    for prefix in sorted(reps, key=len, reverse=True):
        text = text.replace(prefix, reps[prefix])
    for pat in LOCAL_PATH_PATTERNS[2:]:
        text = pat.sub('<tmp>', text)
    return text


def scrub_obj(obj, *, replacements: Mapping[str, str] | None = None):
    """:func:`scrub` applied to every string (keys included) in a JSON-able object."""
    reps = default_replacements() if replacements is None else replacements
    if isinstance(obj, str):
        return scrub(obj, replacements=reps)
    if isinstance(obj, Mapping):
        return {
            scrub_obj(k, replacements=reps): scrub_obj(v, replacements=reps)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [scrub_obj(v, replacements=reps) for v in obj]
    return obj


def find_local_paths(obj) -> list[str]:
    """Strings in *obj* that still look like absolute local paths."""
    found = []

    def walk(o):
        if isinstance(o, str):
            if any(p.search(o) for p in LOCAL_PATH_PATTERNS):
                found.append(o)
        elif isinstance(o, Mapping):
            for k, v in o.items():
                walk(k)
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(obj)
    return found


def write_json(path: Path, obj, *, overwrite: bool = False) -> Path:
    """Scrub, check for leaks, and write *obj* as JSON; never overwrites by default."""
    obj = scrub_obj(obj)
    leaks = find_local_paths(obj)
    if leaks:
        raise LocalPathLeak(
            f'Refusing to write {path.name}: {len(leaks)} string(s) still look like '
            f'local paths, e.g. {leaks[0][:120]!r}'
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if overwrite else 'x'
    with open(path, mode, encoding='utf-8') as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write('\n')
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# --------------------------------------------------------------------------- index


def _index_path(root: Path) -> Path:
    return Path(root) / 'index.json'


def read_index(root: Path) -> dict:
    p = _index_path(root)
    if p.exists():
        return read_json(p)
    return dict(schema_version=SCHEMA_VERSION, scans=[], baselines=[])


def add_to_index(root: Path, kind: str, entry: dict) -> None:
    """Append *entry* to ``index[kind]`` (``'scans'`` or ``'baselines'``)."""
    index = read_index(root)
    index.setdefault(kind, []).append(dict(entry, superseded=None))
    write_json(_index_path(root), index, overwrite=True)


def mark_superseded(
    root: Path, record_id: str, *, reason: str, by: str | None = None
) -> dict:
    """Flag a scan or baseline as untrustworthy, keeping it (history is not deleted)."""
    index = read_index(root)
    for kind in ('scans', 'baselines'):
        for entry in index.get(kind, []):
            if entry['id'] == record_id:
                entry['superseded'] = dict(
                    reason=reason, by=by, at=utc_now().isoformat()
                )
                write_json(_index_path(root), index, overwrite=True)
                return entry
    raise KeyError(f'No scan or baseline with id {record_id!r} in {root}')


def latest_id(root: Path, kind: str = 'scans', *, include_superseded=False) -> str:
    """The newest record id of *kind*, skipping superseded ones by default."""
    entries = [
        e
        for e in read_index(root).get(kind, [])
        if include_superseded or not e.get('superseded')
    ]
    if not entries:
        raise LookupError(f'No {kind} recorded under {root}; run `scan` first.')
    return max(e['id'] for e in entries)
