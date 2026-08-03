"""Packaging and test-configuration hygiene checks for the ``dol`` distribution.

``dol`` sits at the bottom of a large dependency tree, so a few "boring"
packaging properties are worth asserting explicitly instead of trusting by
inspection:

- The :pep:`561` ``py.typed`` marker must be present *and shipped*. Without it,
  every ``from dol import ...`` in a downstream package is typed as ``Any``.
- The pytest configuration must point at paths that actually exist and enable
  the same doctest collection CI uses. Otherwise a green local ``pytest`` says
  nothing about a green CI run, on the very package everything depends on.
"""

from pathlib import Path

import pytest

import dol

#: Name of the PEP 561 marker file that tells type checkers the package is typed.
PY_TYPED_MARKER_NAME = "py.typed"

#: Doctest option flags the wads ``run-tests-uv`` CI action passes via ``-o``.
#: Local ``doctest_optionflags`` must agree with these, or local-green does not
#: imply CI-green (notably, CI does *not* pass ``NORMALIZE_WHITESPACE``).
CI_DOCTEST_OPTIONFLAGS = frozenset({"ELLIPSIS", "IGNORE_EXCEPTION_DETAIL"})

#: pytest ``addopts`` entries required for a bare local ``pytest`` to collect
#: what CI collects.
REQUIRED_ADDOPTS = ("--doctest-modules",)

#: Package root directory (works both for editable and installed distributions).
PKG_DIR = Path(dol.__file__).parent


def _load_toml(path: Path) -> dict:
    """Parse a TOML file, skipping the test if no TOML parser is available."""
    try:
        import tomllib  # Python >= 3.11
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 without tomli
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            pytest.skip("No TOML parser available (need Python >= 3.11 or tomli)")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _pytest_ini_options() -> dict:
    """Return ``[tool.pytest.ini_options]``, skipping if there's no source tree.

    Installed (non-editable) distributions have no ``pyproject.toml`` next to
    the package, so the configuration tests are source-tree-only.
    """
    pyproject = PKG_DIR.parent / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("Not running from a source tree (no pyproject.toml)")
    return _load_toml(pyproject).get("tool", {}).get("pytest", {}).get("ini_options", {})


def test_py_typed_marker_is_present():
    """The PEP 561 marker must exist inside the package directory."""
    marker = PKG_DIR / PY_TYPED_MARKER_NAME
    assert marker.is_file(), (
        f"Missing {PY_TYPED_MARKER_NAME} in {PKG_DIR.name}/: without it every "
        "downstream `from dol import ...` is type-checked as Any."
    )


def test_testpaths_all_exist():
    """Every configured ``testpaths`` entry must resolve to a real path.

    A ``testpaths`` pointing at a nonexistent directory makes pytest silently
    fall back to recursive discovery from the cwd, so what a bare ``pytest``
    runs bears no relation to what is configured.
    """
    ini_options = _pytest_ini_options()
    project_root = PKG_DIR.parent
    testpaths = ini_options.get("testpaths", [])
    missing = [p for p in testpaths if not (project_root / p).exists()]
    assert not missing, f"testpaths entries do not exist: {missing}"


def test_addopts_collect_doctests_like_ci():
    """A bare ``pytest`` must collect doctests, as CI does."""
    addopts = _pytest_ini_options().get("addopts", "")
    missing = [opt for opt in REQUIRED_ADDOPTS if opt not in addopts]
    assert not missing, f"pytest addopts is missing {missing} (got {addopts!r})"


def test_doctest_optionflags_match_ci():
    """Local doctest flags must match the flags the CI action passes.

    The wads ``run-tests-uv`` action passes
    ``-o doctest_optionflags='ELLIPSIS IGNORE_EXCEPTION_DETAIL'``, which
    *overrides* the ini value. Configuring anything else locally (e.g. adding
    ``NORMALIZE_WHITESPACE``) means locally-passing doctests can fail in CI.
    """
    flags = frozenset(_pytest_ini_options().get("doctest_optionflags", []))
    assert flags == CI_DOCTEST_OPTIONFLAGS, (
        f"doctest_optionflags {sorted(flags)} != CI's {sorted(CI_DOCTEST_OPTIONFLAGS)}"
    )
