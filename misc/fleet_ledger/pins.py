"""Compare the fleet to a scan's pins, put it back at them, and undo that.

Safety contract of :func:`restore` (all of it is tested):

- dry run unless ``apply=True``;
- refuses a repo with tracked modifications, and never passes ``--force``;
- never fetches: a pinned commit that is not present locally is reported, not got;
- writes its undo record **before** the first checkout, so an interrupted run is
  still undoable with :func:`undo`;
- each repo is checked out *detached* at its pin, leaving its branches untouched.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

from . import fleet, store


def _state(pin: dict, path: Path | None) -> dict:
    if not pin.get("found") or not pin.get("git"):
        return dict(state="unpinned")
    if path is None:
        return dict(state="missing")
    now = fleet.git_pin(path)
    if not now.get("git"):
        return dict(state="missing")
    state = "at-pin" if now["sha"] == pin["sha"] else "moved"
    return dict(
        state=state,
        dirty=now["dirty"],
        branch=now["branch"],
        sha=now["sha"],
        pinned_sha=pin["sha"],
    )


def require_same_source(root: Path, scan_id: str, repos) -> None:
    """Refuse to resolve an explicit-repos scan's names through the real fleet.

    A scan taken with ``repos=`` pins *those* checkouts; resolving its names via
    priv instead could check out the real fleet at foreign commits.
    """
    meta = store.read_json(Path(root) / "scans" / scan_id / "meta.json")
    if repos is None and meta.get("dependents_source") == "explicit":
        raise ValueError(
            f"Scan {scan_id} was taken with explicit repos; pass them again "
            "(repos= / --repo NAME=PATH) rather than resolving names via the fleet."
        )


def _pins_and_paths(root, scan_id, repos):
    root = Path(root)
    scan_id = scan_id or store.latest_id(root, "scans")
    require_same_source(root, scan_id, repos)
    pins = store.read_json(root / "scans" / scan_id / "pins.json")["pins"]
    return scan_id, pins, fleet.resolve_repos(pins, repos=repos)


def check(root, *, scan_id=None, repos: Mapping | None = None) -> dict:
    """``{name: {'state': 'at-pin'|'moved'|'missing'|'unpinned', ...}}``."""
    _, pins, paths = _pins_and_paths(root, scan_id, repos)
    return {name: _state(pin, paths[name]) for name, pin in sorted(pins.items())}


def _toplevel(path: Path) -> Path:
    return Path(fleet.git(path, "rev-parse", "--show-toplevel")).resolve()


def plan_restore(root, *, scan_id=None, repos: Mapping | None = None) -> dict:
    """What :func:`restore` would do per dependent: ``checkout``, ``noop`` or ``skip``."""
    scan_id, pins, paths = _pins_and_paths(root, scan_id, repos)
    plan, seen_toplevels = {}, {}
    for name, pin in sorted(pins.items()):
        st = _state(pin, paths[name])
        if st["state"] in ("unpinned", "missing"):
            plan[name] = dict(action="skip", reason=st["state"])
            continue
        top = _toplevel(paths[name])
        if top in seen_toplevels:
            plan[name] = dict(
                action="skip", reason=f"same repo as {seen_toplevels[top]}"
            )
            continue
        seen_toplevels[top] = name
        if st["state"] == "at-pin":
            plan[name] = dict(action="noop")
        elif st["dirty"]:
            plan[name] = dict(
                action="skip", reason=f"{st['dirty']} modified tracked file(s)"
            )
        elif _missing_commit(paths[name], pin["sha"]):
            plan[name] = dict(
                action="skip",
                reason="pinned commit not present locally (never fetches)",
            )
        else:
            plan[name] = dict(
                action="checkout",
                to=pin["sha"],
                from_sha=st["sha"],
                from_branch=st["branch"],
            )
    return dict(scan_id=scan_id, plan=plan)


def _missing_commit(path: Path, sha: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(path), "cat-file", "-e", f"{sha}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    return proc.returncode != 0


def restore(
    root, *, scan_id=None, repos: Mapping | None = None, apply: bool = False
) -> dict:
    """Detach every clean dependent at its pinned commit. Dry run unless ``apply``."""
    root = Path(root)
    result = plan_restore(root, scan_id=scan_id, repos=repos)
    todo = {n: p for n, p in result["plan"].items() if p["action"] == "checkout"}
    result["applied"] = False
    if not apply or not todo:
        return result
    undo_dir = root / "pins"
    undo_id = store.new_record_id(undo_dir, name_format="undo-{}.json")
    undo_path = undo_dir / f"undo-{undo_id}.json"
    store.write_json(  # BEFORE the first checkout
        undo_path,
        dict(
            schema_version=store.SCHEMA_VERSION,
            scan_id=result["scan_id"],
            created=store.utc_now().isoformat(),
            repos={
                n: dict(sha=p["from_sha"], branch=p["from_branch"], to=p["to"])
                for n, p in todo.items()
            },
        ),
    )
    paths = fleet.resolve_repos(todo, repos=repos)
    done = {}
    for name, p in todo.items():
        fleet.git(paths[name], "checkout", "-q", "--detach", p["to"])
        done[name] = p["to"]
    result.update(applied=True, undo_record=undo_path.name, checked_out=done)
    return result


def _latest_undo(undo_dir: Path) -> Path:
    records = sorted(
        p for p in undo_dir.glob("undo-*.json") if not p.with_suffix(".done").exists()
    )
    if not records:
        raise LookupError(f"No pending undo record in {undo_dir}")
    return records[-1]


def undo(root, *, repos: Mapping | None = None, apply: bool = False) -> dict:
    """Put back what the latest un-undone :func:`restore` changed. Dry run unless ``apply``."""
    undo_dir = Path(root) / "pins"
    rec_path = _latest_undo(undo_dir)
    rec = store.read_json(rec_path)
    paths = fleet.resolve_repos(rec["repos"], repos=repos)
    plan = {}
    for name, prev in sorted(rec["repos"].items()):
        path = paths[name]
        if path is None:
            plan[name] = dict(action="skip", reason="missing")
            continue
        now = fleet.git_pin(path)
        target = prev["branch"] or prev["sha"]
        if now["branch"] == prev["branch"] and now["sha"] == prev["sha"]:
            plan[name] = dict(action="noop")
        elif now["dirty"]:
            plan[name] = dict(
                action="skip", reason=f"{now['dirty']} modified tracked file(s)"
            )
        elif now["branch"] is not None or now["sha"] != prev.get("to"):
            # moved since the restore (a commit made while detached, a branch
            # switched to): leaving would orphan or override that work
            plan[name] = dict(
                action="skip",
                reason="no longer detached at the restored pin; left as is",
            )
        else:
            plan[name] = dict(action="checkout", to=target)
    result = dict(undo_record=rec_path.name, plan=plan, applied=False)
    if not apply:
        return result
    for name, p in plan.items():
        if p["action"] == "checkout":
            args = ["checkout", "-q"]
            if p["to"] == rec["repos"][name]["sha"]:
                args.append("--detach")
            fleet.git(paths[name], *args, p["to"])
    skipped = [n for n, p in plan.items() if p["action"] == "skip"]
    if not skipped:  # fully undone: mark it, so the next undo looks further back
        store.write_json(
            rec_path.with_suffix(".done"), dict(done=store.utc_now().isoformat())
        )
    result.update(applied=True, skipped=skipped)
    return result
