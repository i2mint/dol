"""Command line for the fleet ledger: ``python -m misc.fleet_ledger <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import baseline, pins, report, scan, store


def _repos(pairs):
    if not pairs:
        return None
    out = {}
    for p in pairs:
        name, _, path = p.partition("=")
        if not path:
            raise SystemExit(f"--repo expects NAME=PATH, got {p!r}")
        out[name] = path
    return out


def _print(obj):
    if isinstance(obj, str):
        print(obj, end="" if obj.endswith("\n") else "\n")
    else:
        print(json.dumps(store.scrub_obj(obj), indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fleet_ledger", description=__doc__)
    ap.add_argument("--package", default="dol")
    ap.add_argument(
        "--root",
        help="ledger directory (default: ~/.local/share/dol/fleet_ledger/<package>)",
    )
    ap.add_argument(
        "--repo",
        action="append",
        metavar="NAME=PATH",
        help="explicit dependent (repeatable); default: priv dep_graph",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="pin + census every dependent")
    s.add_argument("--source-root", help="target source tree (default: this checkout)")
    s.add_argument("--note")

    b = sub.add_parser(
        "baseline", help="run every dependent's suite at the latest scan"
    )
    b.add_argument("--scan")
    b.add_argument("--jobs", type=int, default=1)
    b.add_argument("--timeout", type=float, default=baseline.DFLT_TIMEOUT)
    b.add_argument("--only", nargs="*")
    b.add_argument("--note")

    d = sub.add_parser("diff", help="newly failing / fixed tests between two baselines")
    d.add_argument("a")
    d.add_argument("b", nargs="?", help="default: the latest baseline")

    p = sub.add_parser("pin", help="check | restore | undo")
    p.add_argument("action", choices=("check", "restore", "undo"))
    p.add_argument("--scan")
    p.add_argument(
        "--apply", action="store_true", help="actually check out (default: dry run)"
    )

    for name in ("report", "zero-usage"):
        r = sub.add_parser(name)
        r.add_argument("--scan")
        r.add_argument("--out", help="write Markdown here instead of stdout")
    sub.add_parser("history")
    m = sub.add_parser(
        "mark-superseded", help="flag a scan/baseline untrustworthy, keep it"
    )
    m.add_argument("id")
    m.add_argument("reason")
    m.add_argument("--by")

    a = ap.parse_args(argv)
    root = Path(a.root).expanduser() if a.root else store.ledger_root(a.package)
    repos = _repos(a.repo)

    if a.cmd == "scan":
        sid = scan.run_scan(
            a.package, source_root=a.source_root, repos=repos, root=root, note=a.note
        )
        _print(f"scan {sid} written to {store.scrub(str(root / 'scans' / sid))}")
    elif a.cmd == "baseline":

        def progress(name, e):
            print(
                f"  {name:<24} {e['status']:<11} {e.get('seconds', 0)}s",
                file=sys.stderr,
            )

        bid = baseline.run_baseline(
            root,
            package=a.package,
            scan_id=a.scan,
            repos=repos,
            jobs=a.jobs,
            timeout=a.timeout,
            only=a.only,
            note=a.note,
            progress=progress,
        )
        rec = baseline.load_baseline(root, bid)
        _print(f"baseline {bid}: {baseline.summarize(rec['results'])}")
    elif a.cmd == "diff":
        _print(
            baseline.diff(
                baseline.load_baseline(root, a.a), baseline.load_baseline(root, a.b)
            )
        )
    elif a.cmd == "pin":
        if a.action == "check":
            _print(pins.check(root, scan_id=a.scan, repos=repos))
        elif a.action == "restore":
            _print(pins.restore(root, scan_id=a.scan, repos=repos, apply=a.apply))
        else:
            _print(pins.undo(root, repos=repos, apply=a.apply))
    elif a.cmd in ("report", "zero-usage"):
        data = scan.load_scan(root, a.scan)
        md = store.scrub(
            (report.summary_md if a.cmd == "report" else report.zero_usage_md)(data)
        )
        if store.find_local_paths(md):
            raise SystemExit(
                "Refusing to print a report that still contains local paths."
            )
        if a.out:
            out = Path(a.out).expanduser()
            if scan._is_within(out, Path(__file__).resolve().parents[2]):
                raise SystemExit(
                    "Write reports outside the repo: they name private packages."
                )
            out.write_text(md, encoding="utf-8")
            _print(f"wrote {a.out}")
        else:
            _print(md)
    elif a.cmd == "history":
        _print(report.history_md(root))
    elif a.cmd == "mark-superseded":
        _print(store.mark_superseded(root, a.id, reason=a.reason, by=a.by))


if __name__ == "__main__":
    main()
