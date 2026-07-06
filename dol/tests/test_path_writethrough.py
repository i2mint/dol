"""Tests for optional key-path write-through / autovivification (issue #16).

Covers the opt-in ``create_missing`` behavior of ``add_path_access`` / ``KeyPath`` /
``autoviv``, the ``path_set_writeback`` / ``path_del_writeback`` engine, the contextual
per-level ``mk_missing`` factory, the loud-failure paths, and the ``path_set``
factory-propagation fix (the historical line-787 bug). The write-back protocol is
verified across an in-memory ``dict``, a copy-semantics ``wrap_kvs`` + json store, and a
real ``TextFiles`` store (via fresh instances re-reading from disk).
"""

import json
import warnings
import tempfile
from collections import OrderedDict

import pytest

from dol import (
    add_path_access,
    autoviv,
    KeyPath,
    wrap_kvs,
    TextFiles,
    path_set,
    path_set_writeback,
    PathContext,
    PathCreationError,
    PathWritebackError,
)


def _json_store(initial=None):
    """A copy-semantics store: ``__getitem__`` returns fresh json.loads copies."""
    backend = dict(initial or {})
    store = wrap_kvs(backend, obj_of_data=json.loads, data_of_obj=json.dumps)
    return backend, store


# --------------------------------------------------------------------------- #
# Backward compatibility: default (OFF) behavior is unchanged                  #
# --------------------------------------------------------------------------- #


def test_off_state_add_path_access_missing_raises():
    s = add_path_access({})
    with pytest.raises(KeyError):
        s["x", "y", "z"] = 1


def test_off_state_keypath_missing_raises():
    s = KeyPath(".")({})
    with pytest.raises(KeyError):
        s["a.b.c"] = 1


def test_off_state_delete_leaves_empty_intermediates():
    # The historical observable behavior (dol/paths.py KeyPath doctest).
    s = KeyPath(".")({"a": {"b": {"c": 42}}})
    del s["a.b.c"]
    assert s.store == {"a": {"b": {}}}


# --------------------------------------------------------------------------- #
# Core: autoviv over an in-memory dict                                          #
# --------------------------------------------------------------------------- #


def test_dict_autoviv_creates_and_sets():
    s = add_path_access({}, create_missing=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s["a", "b", "c"] = 42
    assert s["a", "b", "c"] == 42
    assert s["a"]["b"]["c"] == 42


def test_autoviv_alias_matches_add_path_access():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = autoviv({})
        s["a", "b", "c"] = 42
    assert s["a", "b", "c"] == 42


def test_implied_create_missing_from_mk_missing():
    # Passing a callable option implies create_missing=True.
    s = add_path_access({}, mk_missing=lambda ctx: OrderedDict())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s["a", "b"] = 1
    assert isinstance(s["a"], OrderedDict)


def test_contextual_per_level_factory():
    s = add_path_access(
        {},
        create_missing=True,
        on_create=None,  # silence
        mk_missing=lambda ctx: OrderedDict() if ctx.depth == 0 else {},
    )
    s["x", "y", "z"] = 1
    assert isinstance(s["x"], OrderedDict)
    assert isinstance(s["x"]["y"], dict) and not isinstance(s["x"]["y"], OrderedDict)
    assert s["x", "y", "z"] == 1


# --------------------------------------------------------------------------- #
# The crux: write-back through copy-semantics / persistent stores              #
# --------------------------------------------------------------------------- #


def test_json_store_overwrite_existing_deep_path_persists():
    # This is the pre-existing silent-loss bug, now closed on the opt-in path.
    backend, J = _json_store({"a": json.dumps({"b": {"c": 42}})})
    s = add_path_access(J, create_missing=True)
    s["a", "b", "c"] = 99
    assert json.loads(backend["a"]) == {"b": {"c": 99}}


def test_json_store_create_missing_leaf_level_persists():
    backend, J = _json_store({"a": json.dumps({"b": {"c": 42}})})
    s = add_path_access(J, create_missing=True, on_create=None)
    s["a", "b", "new"] = 7
    assert json.loads(backend["a"]) == {"b": {"c": 42, "new": 7}}


def test_json_store_fully_missing_top_branch_persists():
    backend, J = _json_store()
    s = add_path_access(J, create_missing=True, on_create=None)
    s["fresh", "x"] = 5
    assert json.loads(backend["fresh"]) == {"x": 5}


def test_off_state_json_store_still_silently_loses():
    # Guard the "no default change" contract: OFF behavior (incl. the pre-existing
    # silent-loss) must be preserved verbatim.
    backend, J = _json_store({"a": json.dumps({"b": {"c": 42}})})
    s = add_path_access(J)  # OFF
    s["a", "b", "c"] = 99  # mutates a detached copy -> lost, no error (legacy behavior)
    assert json.loads(backend["a"]) == {"b": {"c": 42}}


def test_files_store_deep_write_persists_to_disk():
    d = tempfile.mkdtemp()
    F = wrap_kvs(TextFiles(d), obj_of_data=json.loads, data_of_obj=json.dumps)
    s = add_path_access(F, create_missing=True, on_create=None)
    s["a", "b", "c"] = 99
    s["a", "b", "new"] = 7
    s["fresh", "x"] = 5
    # Fresh instances re-reading from disk:
    F2 = wrap_kvs(TextFiles(d), obj_of_data=json.loads)
    s2 = add_path_access(F2)
    assert s2["a", "b", "c"] == 99
    assert s2["a", "b", "new"] == 7
    assert s2["fresh", "x"] == 5


# --------------------------------------------------------------------------- #
# KeyPath string-path front door                                               #
# --------------------------------------------------------------------------- #


def test_keypath_string_path_autoviv():
    s = KeyPath(".", create_missing=True, on_create=None)({})
    s["a.b.c"] = 42
    assert s["a.b.c"] == 42
    assert s.store["a"]["b"]["c"] == 42


def test_keypath_forwards_over_json_store():
    backend, J = _json_store()
    s = KeyPath(".", create_missing=True, on_create=None)(J)
    s["a.b.c"] = 1
    assert json.loads(backend["a"]) == {"b": {"c": 1}}


# --------------------------------------------------------------------------- #
# Loud-failure paths (never a silent no-op)                                    #
# --------------------------------------------------------------------------- #


def test_existing_non_mapping_blocks_descent():
    s = add_path_access({"a": 5}, create_missing=True)
    with pytest.raises(PathCreationError):
        s["a", "b"] = 1


def test_existing_non_mapping_blocks_deeper():
    s = add_path_access({"a": {"b": 5}}, create_missing=True)
    with pytest.raises(PathCreationError):
        s["a", "b", "c"] = 1


def test_empty_path_raises_valueerror():
    s = add_path_access({}, create_missing=True)
    with pytest.raises(ValueError):
        s[()] = 1


def test_may_create_veto():
    s = add_path_access(
        {}, create_missing=True, on_create=None, may_create=lambda ctx: ctx.key == "ok"
    )
    with pytest.raises(PathCreationError):
        s["bad", "x"] = 1
    s["ok", "x"] = 1  # whitelisted key is allowed
    assert s["ok", "x"] == 1


def test_max_created_guard():
    s = add_path_access({}, create_missing=True, on_create=None, max_created=1)
    with pytest.raises(PathCreationError):
        s["a", "b", "c"] = 1  # would create 2 intermediates ('a', 'a.b')


def test_max_levels_guard():
    s = add_path_access({}, create_missing=True, on_create=None, max_levels=2)
    with pytest.raises(PathCreationError):
        s["a", "b", "c"] = 1  # depth 3 > max_levels 2


def test_literal_tuple_key_collision_guard():
    # A store whose real key IS a tuple: autoviv must refuse to shadow it.
    s = add_path_access({("x", "y"): "real"}, create_missing=True)
    with pytest.raises(PathCreationError):
        s["x", "y"] = "new"


# --------------------------------------------------------------------------- #
# Delete write-back                                                            #
# --------------------------------------------------------------------------- #


def test_delete_leaf_persists_on_copy_store():
    backend, J = _json_store({"a": json.dumps({"b": {"c": 1, "d": 2}})})
    s = add_path_access(J, create_missing=True)
    del s["a", "b", "c"]
    assert json.loads(backend["a"]) == {"b": {"d": 2}}


def test_delete_missing_leaf_raises():
    backend, J = _json_store({"a": json.dumps({"b": {"c": 1}})})
    s = add_path_access(J, create_missing=True)
    with pytest.raises(KeyError):
        del s["a", "b", "zzz"]


# --------------------------------------------------------------------------- #
# Observability: creation is never silent                                     #
# --------------------------------------------------------------------------- #


def test_creation_emits_warning_by_default():
    s = add_path_access({}, create_missing=True)
    with pytest.warns(UserWarning, match="autoviv created"):
        s["typo", "x"] = 1


def test_on_create_none_silences():
    s = add_path_access({}, create_missing=True, on_create=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        s["a", "b"] = 1
    assert s["a", "b"] == 1


# --------------------------------------------------------------------------- #
# store_decorator: all four usage forms                                       #
# --------------------------------------------------------------------------- #


def test_store_decorator_class_and_factory_forms():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = add_path_access(dict, create_missing=True)  # bare-class factory -> class
        inst = S()
        inst["a", "b"] = 1
        factory = add_path_access(create_missing=True)  # param-only -> decorator
        inst2 = factory({})
        inst2["a", "b"] = 2
    assert inst["a", "b"] == 1 and inst2["a", "b"] == 2


# --------------------------------------------------------------------------- #
# The descriptor trap: callables must be closure-captured, not bound methods  #
# --------------------------------------------------------------------------- #


def test_callables_are_closure_captured_not_bound_methods():
    # If mk_missing / explore_further were stashed on the class and invoked as bound
    # methods, they'd receive `self` as an extra positional arg -> TypeError. Passing
    # real single-arg callables and exercising both proves they are closure-captured.
    seen = []

    def mk_missing(ctx):  # exactly one positional arg
        seen.append(ctx)
        return {}

    def explore_further(node, path):  # exactly two positional args
        return False  # never descend; keep it a pure value-nesting write

    s = add_path_access(
        {}, create_missing=True, on_create=None,
        mk_missing=mk_missing, explore_further=explore_further,
    )
    s["a", "b", "c"] = 1  # would TypeError if mk_missing got (self, ctx)
    assert s["a", "b", "c"] == 1
    assert all(isinstance(ctx, PathContext) for ctx in seen)


# --------------------------------------------------------------------------- #
# explore_further (Model-2) with an in-memory reference-semantics parent       #
# --------------------------------------------------------------------------- #


def test_explore_further_reference_parent_registers_substore():
    # A dict-of-sub-stores: explore_further descends into an existing sub-store, and a
    # newly-created sub-store is registered back into the (reference-semantics) parent.
    inner_a = add_path_access({}, create_missing=True, on_create=None)
    parent = {"a": inner_a}

    def explore_further(node, path):
        return hasattr(node, "_create_missing")  # it's one of our path stores

    root = add_path_access(
        parent, create_missing=True, on_create=None,
        mk_missing=lambda ctx: add_path_access({}, create_missing=True, on_create=None),
        explore_further=explore_further,
    )
    root["a", "k1"] = 1  # descends into existing sub-store inner_a
    assert inner_a["k1"] == 1
    root["b", "k2"] = 2  # creates a new sub-store, registers it into the dict parent
    assert parent["b"]["k2"] == 2


# --------------------------------------------------------------------------- #
# path_set: the line-787 factory-propagation fix (issue #69 coordination)     #
# --------------------------------------------------------------------------- #


def test_path_set_factory_propagates_to_every_level():
    d = {}
    path_set(d, "a.b.c", 42, new_mapping=OrderedDict)
    assert isinstance(d["a"], OrderedDict)
    assert isinstance(d["a"]["b"], OrderedDict)  # regression: used to be a plain dict
    assert d["a"]["b"]["c"] == 42


def test_path_set_contextual_mk_missing():
    d = {}
    path_set(
        d, ("x", "y", "z"), 1,
        mk_missing=lambda ctx: OrderedDict() if ctx.depth == 0 else {},
    )
    assert isinstance(d["x"], OrderedDict)
    assert isinstance(d["x"]["y"], dict) and not isinstance(d["x"]["y"], OrderedDict)
    assert d["x"]["y"]["z"] == 1


def test_path_set_empty_path_raises():
    with pytest.raises(ValueError):
        path_set({}, (), 1)


def test_path_set_backward_compatible_default():
    d = {"a": 1, "b": {"c": 2}}
    path_set(d, ["b", "e"], 42)
    assert d == {"a": 1, "b": {"c": 2, "e": 42}}


# --------------------------------------------------------------------------- #
# Regressions from the adversarial review (findings 1-3)                       #
# --------------------------------------------------------------------------- #


def test_scalar_intermediate_deep_path_raises_path_creation_error():
    # Finding 1: an existing scalar blocking a path of length >= 2 must raise a loud
    # PathCreationError, not a raw TypeError.
    s = add_path_access({"a": 5}, create_missing=True)
    with pytest.raises(PathCreationError):
        s["a", "b", "c"] = 1
    backend, J = _json_store({"a": json.dumps(5)})  # copy-semantics variant
    s2 = add_path_access(J, create_missing=True)
    with pytest.raises(PathCreationError):
        s2["a", "b", "c"] = 1


def test_bad_mk_missing_returning_non_mapping_raises():
    # A misconfigured factory returning a non-mapping is caught loudly, not TypeError.
    s = add_path_access({}, create_missing=True, on_create=None, mk_missing=lambda ctx: 0)
    with pytest.raises(PathCreationError):
        s["a", "b", "c"] = 1


def test_no_partial_write_on_reference_parent_when_max_created_trips():
    # Finding 3: a mid-path guard failure must not leave a spurious intermediate on a
    # reference-semantics (dict) parent.
    backend = {"top": {}}
    s = add_path_access(backend, create_missing=True, on_create=None, max_created=1)
    with pytest.raises(PathCreationError):
        s["top", "a", "b", "c"] = 1
    assert backend == {"top": {}}


def test_no_partial_write_on_may_create_veto():
    backend = {"top": {}}
    s = add_path_access(
        backend, create_missing=True, on_create=None,
        may_create=lambda ctx: ctx.key != "b",  # veto the 2nd missing key
    )
    with pytest.raises(PathCreationError):
        s["top", "a", "b", "c"] = 1
    assert backend == {"top": {}}


class _FlagLock:
    """A context-manager 'lock' that records whether it is currently held."""

    def __init__(self):
        self.held = False

    def __enter__(self):
        self.held = True
        return self

    def __exit__(self, *exc):
        self.held = False
        return False


def test_writeback_lock_spans_read_modify_write():
    # Finding 2: the store read must happen INSIDE the lock (the whole read-modify-write),
    # not just the final write, or a concurrent writer could clobber a sibling update.
    lock = _FlagLock()
    reads_under_lock = []

    class Recording(dict):
        def __getitem__(self, k):
            reads_under_lock.append(lock.held)
            return super().__getitem__(k)

    backend = Recording()
    backend["a"] = {"b": {"c": 1}}
    s = add_path_access(backend, create_missing=True, on_create=None, writeback_lock=lock)
    s["a", "b", "d"] = 2
    assert reads_under_lock and all(reads_under_lock), reads_under_lock
    assert backend["a"] == {"b": {"c": 1, "d": 2}}
