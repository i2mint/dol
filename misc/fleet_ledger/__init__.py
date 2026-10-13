"""fleet_ledger — a versioned record of who uses dol, how, and at which commit.

A maintainer tool (not part of the ``dol`` distribution) for `i2mint/dol#91`_.
The code is committed; everything it *produces* is derived from the fleet's
source, some of it private, so it is written outside the repo, under
``~/.local/share/dol/fleet_ledger/dol/`` (see :mod:`fleet_ledger.store`).

Run from the dol repo root::

    python -m misc.fleet_ledger scan               # pin + AST census -> one immutable scan
    python -m misc.fleet_ledger baseline           # every dependent's suite at the pins
    python -m misc.fleet_ledger diff A B           # newly failing test ids between baselines
    python -m misc.fleet_ledger pin check          # fleet vs pins; restore/undo (dry run default)
    python -m misc.fleet_ledger zero-usage | report | history | mark-superseded ID REASON

.. _i2mint/dol#91: https://github.com/i2mint/dol/issues/91
"""

from .astscan import scan_source, scan_repo, close_class_graph
from .scan import run_scan, load_scan, aggregate
from .baseline import run_baseline, load_baseline, diff, run_suite
from .pins import check as pin_check, restore as pin_restore, undo as pin_undo
from .report import zero_usage, zero_usage_md, summary_md, history_md
from .store import ledger_root, mark_superseded, SCHEMA_VERSION
