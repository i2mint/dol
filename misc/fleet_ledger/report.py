"""Reports computed from a scan: the zero-usage list, a summary, the scan history.

The zero-usage report is the authoritative input to the cruft audit, and it
carries its caveats *in the generated text*, so they cannot be dropped on the way
to a decision:

1. star imports make usage invisible to name matching; they are resolved against
   the star'd module's surface, but at lower confidence (``via: star``);
2. names imported from dol *submodules* without being in ``__init__`` are a
   compat surface, listed separately, that any cleanup must respect;
3. scope: zero usage *in this local fleet*; dol is on PyPI, so this bounds the
   migration cost we control and does not license a silent removal.

It also splits the zero-usage exports by age (recent ones have not had time to
be adopted, so their non-use is not evidence) and sets aside incidental exports
(a stray ``import os`` in ``__init__`` exporting ``dol.os``).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from . import store
from .surface import parse_iso

DFLT_RECENT_DAYS = 365

CAVEATS = (
    "Star imports: {n_star} dependent(s) star-import from {package} ({star_list}). "
    "Their free names are matched against the star'd module's own public surface "
    "(records marked `via: star`), which is lower-confidence than an explicit import.",
    "Submodule compat surface: {n_compat} name(s) are imported from {package} "
    "submodules without being exported by `__init__` (listed below). Removing or "
    "moving them breaks those imports even though they are not in the export list.",
    'Scope: "zero usage" means zero usage in this local fleet of {n_dep} dependents. '
    "{package} is published on PyPI; this bounds the migration cost we control and "
    "does not license a silent removal.",
)

#: Generated limits of the scan itself, appended after the three caveats.
SCAN_LIMITS = (
    "Not scanned: notebooks (`.ipynb`), hidden directories, and top-level "
    "`build`/`dist`/`env`/`venv` directories of each dependent.",
)
SHALLOW_NOTE = (
    'The {package} history is a shallow clone: "first seen" dates are the '
    "truncation point, not when a name was exported, so the recent/candidate "
    "split below is unreliable."
)
PRIVATE_NAMES_NOTE = (
    "> Generated from the local fleet, some of it private: this page names "
    "dependent packages. Review it before pasting any of it into a public issue."
)


def zero_usage(scan: dict, *, recent_days: int = DFLT_RECENT_DAYS) -> dict:
    """Split the target's exports into used / candidates / recent / incidental."""
    surf, agg = scan["surface"], scan["census"]["aggregates"]
    used = agg["symbols"]
    created = parse_iso(scan["meta"]["created"])
    cutoff = created - timedelta(days=recent_days)
    first_seen = surf.get("export_first_seen", {})
    out = dict(used={}, candidates=[], recent=[], incidental=[], submodules=[])
    for name, rec in sorted(surf["exports"].items()):
        if rec["kind"] == "module":
            key = "incidental" if rec.get("incidental") else "submodules"
            out[key].append(name)
            continue
        if name in used:
            out["used"][name] = used[name]["n_packages"]
            continue
        seen = first_seen.get(name)
        if seen and parse_iso(seen) > cutoff:
            out["recent"].append(dict(name=name, first_seen=seen[:10]))
        else:
            out["candidates"].append(dict(name=name, first_seen=(seen or "")[:10]))
    out["compat_names"] = agg["compat_names"]
    out["unknown_names"] = agg["unknown_names"]
    out["star_imports"] = agg["star_imports"]
    return out


def _table(rows, headers) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return lines


def zero_usage_md(scan: dict, *, recent_days: int = DFLT_RECENT_DAYS) -> str:
    """The zero-usage report as Markdown, caveats first."""
    z = zero_usage(scan, recent_days=recent_days)
    meta = scan["meta"]
    package = meta["package"]
    star_pkgs = sorted({p for ps in z["star_imports"].values() for p in ps})
    fmt = dict(
        package=package,
        n_star=len(star_pkgs),
        star_list=", ".join(star_pkgs) or "none",
        n_compat=len(z["compat_names"]),
        n_dep=meta["n_dependents"],
    )
    n_exports = len(scan["surface"]["exports"])
    n_zero = len(z["candidates"]) + len(z["recent"])
    lines = [
        f"# {package} zero-usage report — scan {meta['id']}",
        "",
        PRIVATE_NAMES_NOTE,
        "",
        f"{package} {meta['package_version']} at "
        f"`{(meta['package_pin'].get('sha') or '?')[:12]}`; "
        f"{meta['n_found']} of {meta['n_dependents']} dependents scanned.",
        "",
        "## Caveats",
        "",
        *[f"{i}. {c.format(**fmt)}" for i, c in enumerate(CAVEATS, 1)],
        *[f"- {c}" for c in SCAN_LIMITS],
        *(
            [f"- **{SHALLOW_NOTE.format(**fmt)}**"]
            if scan["surface"].get("history_shallow")
            else []
        ),
        "",
        "## Summary",
        "",
        f"- `__init__` exports: **{n_exports}** "
        f"({len(z['submodules'])} submodules, {len(z['incidental'])} incidental)",
        f"- used by at least one dependent: **{len(z['used'])}**",
        f"- zero fleet usage: **{n_zero}**, of which **{len(z['candidates'])}** "
        f"exported more than {recent_days} days before the scan (cruft-audit "
        f"candidates) and {len(z['recent'])} more recent",
        f"- incidental exports (a non-{package} module leaked by `__init__`): "
        + (", ".join(f"`{n}`" for n in z["incidental"]) or "none"),
        "",
        f"## Candidates (zero usage, exported > {recent_days} days)",
        "",
        *_table(
            [(f"`{r['name']}`", r["first_seen"] or "?") for r in z["candidates"]],
            ["name", "first seen in `__init__`"],
        ),
        "",
        f"## Recent (zero usage, exported within {recent_days} days — not evidence)",
        "",
        *_table(
            [(f"`{r['name']}`", r["first_seen"]) for r in z["recent"]],
            ["name", "first seen in `__init__`"],
        ),
        "",
        "## Submodule compat surface (imported, not in `__init__`)",
        "",
        *_table(
            [
                (
                    f"`{n}`",
                    ", ".join(f"`{m}`" for m in r["modules"]),
                    len(r["packages"]),
                )
                for n, r in z["compat_names"].items()
            ],
            ["name", "imported from", "packages"],
        ),
    ]
    if z["unknown_names"]:
        lines += [
            "",
            f"## Imported names that no longer exist in {package}",
            "",
            *_table(
                [
                    (
                        f"`{n}`",
                        ", ".join(f"`{m}`" for m in r["modules"]),
                        ", ".join(r["packages"]),
                    )
                    for n, r in z["unknown_names"].items()
                ],
                ["name", "imported from", "packages"],
            ),
        ]
    return "\n".join(lines) + "\n"


def summary_md(scan: dict, *, top: int = 25) -> str:
    """Headline numbers: symbol fan-out, subclassing, call sites, keyword census."""
    meta, agg = scan["meta"], scan["census"]["aggregates"]
    package = meta["package"]
    imp = meta["import_resolved"]
    lines = [
        f"# {package} fleet ledger — scan {meta['id']}",
        "",
        PRIVATE_NAMES_NOTE,
        "",
        f"- {package} {meta['package_version']} at "
        f"`{(meta['package_pin'].get('sha') or '?')[:12]}`, created {meta['created']}",
        f"- dependents: {meta['n_dependents']} direct ({meta['n_found']} found), "
        + (
            f"{len(meta['transitive_dependents'])} transitive"
            if meta.get("transitive_dependents") is not None
            else "transitive not computed"
        ),
        f"- census scored against `{imp['file']}` "
        + (
            "(the source tree)"
            if imp["from_source"]
            else "**(NOT the source tree: the census measured another copy)**"
        ),
        "",
        "## Most-used symbols",
        "",
        *_table(
            [
                (f"`{n}`", r["n_packages"], r["import_sites"], r["ref_sites"])
                for n, r in sorted(
                    agg["symbols"].items(), key=lambda kv: -kv[1]["n_packages"]
                )[:top]
            ],
            ["symbol", "packages", "import sites", "reference sites"],
        ),
        "",
        "## Subclassing",
        "",
        *_table(
            [
                (f"`{n}`", r["n_packages"], r["classes"], r["direct"], r["indirect"])
                for n, r in sorted(
                    agg["subclasses"].items(), key=lambda kv: -kv[1]["n_packages"]
                )
            ],
            ["base", "packages", "classes", "direct", "indirect"],
        ),
        "",
        "## Call sites and keyword census",
        "",
    ]
    for name, r in sorted(agg["calls"].items(), key=lambda kv: -kv[1]["sites"])[:top]:
        lines.append(
            f"- `{name}`: {r['n_packages']} packages, {r['sites']} sites, "
            f"{r['decorator_sites']} as decorator ({r['class_decorator_sites']} of a class)"
        )
        for kw, k in sorted(r["kwargs"].items(), key=lambda kv: -kv[1]["sites"]):
            kinds = ", ".join(f"{kk} {vv}" for kk, vv in k["kinds"].items())
            lines.append(
                f"  - `{kw}`: {k['n_packages']} pkgs / {k['sites']} sites ({kinds})"
            )
    return "\n".join(lines) + "\n"


def history_md(root: Path) -> str:
    """One line per scan and baseline, superseded ones marked."""
    index = store.read_index(Path(root))
    lines = []
    for kind in ("scans", "baselines"):
        lines.append(f"## {kind}")
        for e in index.get(kind, []):
            mark = (
                f" — SUPERSEDED: {e['superseded']['reason']}"
                if e.get("superseded")
                else ""
            )
            extra = e.get("statuses") or (e.get("package_sha") or "")[:12]
            lines.append(
                f"- {e['id']}  n={e.get('n_dependents', e.get('n'))}  {extra}{mark}"
            )
        lines.append("")
    return "\n".join(lines)
