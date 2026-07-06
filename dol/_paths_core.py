"""Dependency-free core of dol's key-path *write* machinery.

This leaf module holds the pieces that both :mod:`dol.paths` and :mod:`dol.trans`
need for setting (and deleting) values through key-paths, including optional
*write-through autovivification* — creating missing intermediate levels on write
(GitHub issue #16). It deliberately imports nothing from ``dol.paths`` or
``dol.trans`` (only the standard library), so ``trans.py`` can use the engine
without recreating the ``trans -> paths`` import cycle. It also pre-stages the
``paths.py`` module split (issue #70).

Two layers live here:

- :func:`_inplace_build` — the single in-memory splicer that walks/creates nested
  levels of a plain ``MutableMapping``. It is also the body of the public
  :func:`path_set` (fixing the historical bug where the ``new_mapping`` factory was
  not propagated past the top level).
- :func:`path_set_writeback` / :func:`path_del_writeback` — the *store-aware* engine
  that descends by **persistence boundary** and does a single write-back at the
  boundary, so a deep write persists correctly through copy-semantics / persistent
  stores (``Files``, ``wrap_kvs``-wrapped stores) instead of silently mutating a
  detached copy. Every unsafe situation raises a loud, informative error
  (:class:`PathCreationError` / :class:`PathWritebackError`), never silent data loss.

The contextual per-level factory has a single signature::

    mk_missing(ctx: PathContext) -> MutableMapping

where ``ctx = PathContext(prev_path, key, depth)`` lets a caller choose a different
container type at each level (the core issue-#16 ask).
"""

import contextlib
import warnings
from collections import namedtuple
from collections.abc import MutableMapping
from typing import Any, Callable, Optional

__all__ = [
    "PathContext",
    "PathCreationError",
    "PathWritebackError",
    "path_set",
    "path_set_writeback",
    "path_del_writeback",
]

# A sentinel distinct from None (which can be a legitimate stored value).
_MISSING = object()

# Context handed to a contextual per-level factory / guard.
#   prev_path: tuple of keys already traversed to reach the parent of `key`
#   key:       the missing key about to be created
#   depth:     len(prev_path) — handed directly so "container per level" is a one-liner
PathContext = namedtuple("PathContext", ["prev_path", "key", "depth"])


class PathCreationError(KeyError):
    """Autovivification was blocked or forbidden.

    Raised when a missing intermediate cannot (or should not) be created: an
    existing non-mapping blocks descent, a ``may_create`` predicate vetoed the key,
    a ``max_created`` / ``max_levels`` guard was exceeded, or an autoviv path would
    shadow a genuine literal key.

    Subclasses :class:`KeyError` so that legacy ``except KeyError`` handlers still
    catch it; it is only ever raised on the opt-in write-through path.
    """


class PathWritebackError(RuntimeError):
    """A boundary write-back could not be persisted.

    Raised when re-serializing an intermediate branch back to its store fails (the
    value or a live sub-store won't serialize), or when ``verify_writeback`` detects
    that a persisted branch reads back differently than it was written.
    """


def _warn_on_create(ctx: PathContext) -> None:
    """Default ``on_create`` hook: announce a fabricated intermediate.

    A structurally-valid *typo* on write would otherwise silently create a bogus
    branch; warning keeps opted-in creation from being silent. Pass
    ``on_create=None`` to silence (e.g. bulk tree building).
    """
    warnings.warn(
        f"autoviv created intermediate {ctx.prev_path + (ctx.key,)!r}",
        stacklevel=3,
    )


def _authorize_create(
    ctx: PathContext,
    may_create: Optional[Callable[[PathContext], bool]],
    on_create: Optional[Callable[[PathContext], None]],
    budget: list,
    max_created: Optional[int],
) -> None:
    """Centralized veto / announce / count discipline for one creation.

    Raises :class:`PathCreationError` if ``may_create`` vetoes the key or the
    ``max_created`` budget is exceeded; otherwise increments the budget and calls
    ``on_create`` (if any).
    """
    if may_create is not None and not may_create(ctx):
        raise PathCreationError(
            f"autoviv of {ctx.prev_path + (ctx.key,)!r} vetoed by may_create"
        )
    budget[0] += 1
    if max_created is not None and budget[0] > max_created:
        raise PathCreationError(
            f"exceeded max_created={max_created} newly-created intermediates "
            f"(at {ctx.prev_path + (ctx.key,)!r})"
        )
    if on_create is not None:
        on_create(ctx)


def _inplace_build(
    node: MutableMapping,
    rest,
    val: Any,
    *,
    mk_missing: Callable[[PathContext], MutableMapping],
    prefix: tuple,
    budget: list,
    may_create: Optional[Callable[[PathContext], bool]] = None,
    on_create: Optional[Callable[[PathContext], None]] = None,
    max_created: Optional[int] = None,
) -> None:
    """Splice ``rest`` into the in-memory mapping ``node``, setting the leaf to ``val``.

    Walks the keys of ``rest``, creating any missing intermediate mapping with
    ``mk_missing(ctx)``. ``prefix`` is the path already traversed to reach ``node``
    (used only to build informative ``PathContext`` / error messages). Both the
    descent path and the final leaf target are guarded: descending into (or setting a
    leaf under) an existing non-mapping raises :class:`PathCreationError` *before* any
    partial write.
    """
    if not isinstance(node, MutableMapping):
        raise PathCreationError(
            f"cannot descend into existing non-mapping at {prefix!r}: "
            f"found {type(node).__name__}"
        )
    cur = node
    n = len(rest)
    # 1) descend through EXISTING mapping levels (no creation, no mutation yet).
    i = 0
    while i < n - 1 and rest[i] in cur:
        nxt = cur[rest[i]]
        here = prefix + tuple(rest[: i + 1])
        if not isinstance(nxt, MutableMapping):
            raise PathCreationError(
                f"cannot descend into existing non-mapping at {here!r}: "
                f"found {type(nxt).__name__}"
            )
        cur = nxt
        i += 1
    # 2) authorize EVERY missing intermediate (indices i..n-2) BEFORE mutating anything,
    #    so a veto / budget-exceed leaves a reference-semantics parent untouched (no
    #    partial write). Once rest[i] is missing, the whole suffix is created fresh.
    contexts = []
    for j in range(i, n - 1):
        here = prefix + tuple(rest[: j + 1])
        ctx = PathContext(here[:-1], rest[j], len(here) - 1)
        _authorize_create(ctx, may_create, on_create, budget, max_created)
        contexts.append(ctx)
    # 3) build the missing suffix off to the side, then splice it in with ONE assignment.
    if contexts:
        chain_root = mk_missing(contexts[0])
        tail = chain_root
        for ctx in contexts[1:]:
            child = mk_missing(ctx)
            tail[ctx.key] = child
            tail = child
        tail[rest[-1]] = val
        cur[rest[i]] = chain_root
    else:
        # no missing intermediates: cur is the (existing) penultimate mapping.
        cur[rest[-1]] = val


def path_set(
    d: MutableMapping,
    key_path,
    val: Any,
    *,
    sep: str = ".",
    new_mapping: Callable[[], MutableMapping] = dict,
    mk_missing: Optional[Callable[[PathContext], MutableMapping]] = None,
) -> None:
    """Set ``val`` at ``key_path`` in the (nested) mapping ``d``, creating levels as needed.

    :param d: The mapping to set the value in (mutated in place).
    :param key_path: The path of keys. If a string, it is split on ``sep``.
    :param val: The value to set.
    :param sep: The separator used to split a string ``key_path`` (default ``"."``).
    :param new_mapping: Zero-argument factory for missing intermediate mappings
        (default ``dict``). Used at **every** created level.
    :param mk_missing: Optional contextual per-level factory
        ``mk_missing(ctx: PathContext) -> MutableMapping``. When given it supersedes
        ``new_mapping`` (so different depths can use different container types).

    >>> d = {'a': 1, 'b': {'c': 2}}
    >>> path_set(d, ['b', 'e'], 42)
    >>> d
    {'a': 1, 'b': {'c': 2, 'e': 42}}

    >>> input_dict = {
    ...   "a": {
    ...     "c": "val of a.c",
    ...     "b": 1,
    ...   },
    ...   "10": 10,
    ...   "b": {
    ...     "B": {
    ...       "AA": 3
    ...     }
    ...   }
    ... }
    >>>
    >>> path_set(input_dict, ('new', 'key', 'path'), 7)
    >>> input_dict  # doctest: +NORMALIZE_WHITESPACE
    {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}},
    'new': {'key': {'path': 7}}}

    You can also use a string as a path, with a separator:

    >>> path_set(input_dict, 'new/key/old/path', 8, sep='/')
    >>> input_dict  # doctest: +NORMALIZE_WHITESPACE
    {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}},
    'new': {'key': {'path': 7, 'old': {'path': 8}}}}

    If you specify a string path and a non-None separator, the separator will be used
    to split the string into a list of keys. The default separator is ``sep='.'``.

    >>> path_set(input_dict, 'new.key', 'new val')
    >>> input_dict  # doctest: +NORMALIZE_WHITESPACE
    {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}},
    'new': {'key': 'new val'}}

    You can also specify a different ``new_mapping`` factory, which will be used to
    create new mappings when a key is missing. The default is ``dict``. Unlike older
    versions, the factory now applies at **every** created level, not just the first:

    >>> from collections import OrderedDict
    >>> d = {}
    >>> path_set(d, 'a.b.c', 42, new_mapping=OrderedDict)
    >>> type(d['a']).__name__, type(d['a']['b']).__name__
    ('OrderedDict', 'OrderedDict')
    >>> d['a']['b']['c']
    42

    For full control, pass a contextual ``mk_missing`` factory (the ``ctx`` carries
    ``prev_path``, ``key`` and ``depth``), so different depths can use different
    container types:

    >>> d = {}
    >>> path_set(d, ('x', 'y', 'z'), 1,
    ...          mk_missing=lambda ctx: OrderedDict() if ctx.depth == 0 else {})
    >>> type(d['x']).__name__, type(d['x']['y']).__name__
    ('OrderedDict', 'dict')
    >>> d['x']['y']['z']
    1
    """
    if isinstance(key_path, str) and sep is not None:
        key_path = key_path.split(sep)
    key_path = list(key_path)
    if not key_path:
        raise ValueError("empty key path")
    if mk_missing is None:
        mk_missing = lambda ctx: new_mapping()
    _inplace_build(
        d,
        key_path,
        val,
        mk_missing=mk_missing,
        prefix=(),
        budget=[0],
        may_create=None,
        on_create=None,
        max_created=None,
    )


def _boundary_write(store, key, value, prefix, verify, lock=None) -> None:
    """Write ``value`` at ``key`` of ``store`` (the persistence boundary), loud on failure.

    On a copy-semantics / persistent store this re-serializes the whole branch through
    the store's own ``data_of_obj``; on a plain in-memory mapping it is a harmless
    re-set of the same object. A serialization ``TypeError`` (e.g. trying to persist a
    live sub-store) becomes an informative :class:`PathWritebackError`. If ``verify``
    is set and the branch reads back differently, raise (the write is already
    committed — non-atomic).
    """
    ctx_lock = lock if lock is not None else contextlib.nullcontext()
    with ctx_lock:
        try:
            store[key] = value
        except TypeError as e:
            raise PathWritebackError(
                f"could not persist node at {prefix + (key,)!r}: {e}. If this "
                f"intermediate is itself a store, it must be descended into "
                f"(explore_further) rather than serialized; persistent "
                f"store-of-stores autoviv is out of scope for issue #16."
            ) from e
        if verify and store[key] != value:
            raise PathWritebackError(
                f"write-back of {prefix + (key,)!r} PERSISTED but re-read differs "
                f"(lossy/rejecting transform); the store was MUTATED and NOT rolled back."
            )


def path_set_writeback(
    store,
    path,
    val: Any,
    *,
    mk_missing: Optional[Callable[[PathContext], MutableMapping]] = None,
    explore_further: Optional[Callable[[Any, tuple], bool]] = None,
    may_create: Optional[Callable[[PathContext], bool]] = None,
    on_create: Optional[Callable[[PathContext], None]] = _warn_on_create,
    max_created: Optional[int] = None,
    max_levels: Optional[int] = 20,
    verify_writeback: bool = False,
    writeback_lock=None,
) -> None:
    """Set ``val`` at ``path`` of ``store``, creating missing levels and writing back.

    Descends by **persistence boundary** (the outer store, plus any sub-store flagged
    by ``explore_further``). For an ordinary value branch it reads the top node once,
    splices the missing suffix in memory via :func:`_inplace_build`, then does a
    single write-back at the boundary — the protocol that persists correctly through
    copy-semantics / persistent stores (see the module docstring and
    ``misc/docs/dol_issue16_design.md`` §7).

    All keyword arguments after ``val`` are keyword-only; ``_prefix`` and ``_budget``
    are internal recursion state. Unsafe situations raise :class:`PathCreationError`
    or :class:`PathWritebackError` — never a silent no-op.
    """
    if mk_missing is None:
        mk_missing = lambda ctx: {}
    lock = writeback_lock if writeback_lock is not None else contextlib.nullcontext()
    # Hold the lock (if any) across the WHOLE read-modify-write so a concurrent writer
    # cannot interleave between our read and our write-back. Inner writes get lock=None
    # (the caller already holds it), avoiding a self-deadlock on a non-reentrant lock.
    with lock:
        _path_set_writeback(
            store,
            list(path),
            val,
            mk_missing=mk_missing,
            explore_further=explore_further,
            may_create=may_create,
            on_create=on_create,
            max_created=max_created,
            max_levels=max_levels,
            verify_writeback=verify_writeback,
            _prefix=(),
            _budget=[0],
        )


def _path_set_writeback(
    store,
    path,
    val,
    *,
    mk_missing,
    explore_further,
    may_create,
    on_create,
    max_created,
    max_levels,
    verify_writeback,
    _prefix,
    _budget,
) -> None:
    """Recursive core of :func:`path_set_writeback` (the caller holds ``writeback_lock``).

    All boundary writes are issued with ``lock=None`` because the public entry already
    holds the lock for the whole read-modify-write.
    """
    path = list(path)
    if not path:
        raise ValueError("empty key path")
    if len(_prefix) + len(path) > (max_levels if max_levels is not None else float("inf")):
        raise PathCreationError(
            f"path too deep (> max_levels={max_levels}) at {_prefix + tuple(path)!r}"
        )

    head, *rest = path
    if not rest:  # terminal leaf at this boundary
        _boundary_write(store, head, val, _prefix, verify_writeback, None)
        return

    try:
        child = store[head]
        created = False
    except KeyError:
        ctx = PathContext(_prefix, head, len(_prefix))
        _authorize_create(ctx, may_create, on_create, _budget, max_created)
        child = mk_missing(ctx)
        created = True

    here = _prefix + (head,)

    if explore_further is not None and explore_further(child, here):
        # child is its OWN persistence boundary: recurse; it persists its own writes.
        _path_set_writeback(
            child,
            rest,
            val,
            mk_missing=mk_missing,
            explore_further=explore_further,
            may_create=may_create,
            on_create=on_create,
            max_created=max_created,
            max_levels=max_levels,
            verify_writeback=verify_writeback,
            _prefix=here,
            _budget=_budget,
        )
        if created:
            # Register the newly-created sub-store into its parent. This works for
            # in-memory (reference-semantics) parents; a persistent/copy parent cannot
            # hold a live sub-store, so surface a loud error (deferred to issue #10).
            try:
                _boundary_write(store, head, child, _prefix, False, None)
            except PathWritebackError as e:
                raise PathWritebackError(
                    f"cannot register an auto-created sub-store at {here!r} into a "
                    f"persistent/copy parent store; persistent store-of-stores autoviv "
                    f"is out of scope for issue #16 (pre-create the sub-store, or see "
                    f"mk_dirs_if_missing / issue #10)."
                ) from e
        return

    # child is a plain value within this boundary: splice in memory, then ONE write-back.
    _inplace_build(
        child,
        rest,
        val,
        mk_missing=mk_missing,
        prefix=here,
        budget=_budget,
        may_create=may_create,
        on_create=on_create,
        max_created=max_created,
    )
    _boundary_write(store, head, child, _prefix, verify_writeback, None)


def path_del_writeback(store, path, *, writeback_lock=None) -> None:
    """Delete the value at ``path`` of ``store``, writing the mutated branch back.

    Delete never vivifies: a missing intermediate or leaf raises :class:`KeyError`
    (matching plain nested-mapping behavior). For a deep path on a copy-semantics /
    persistent store, the top branch is read, the leaf deleted in memory, and the
    branch written back — so the deletion actually persists (closing the delete-side
    silent-loss). Now-empty auto-created intermediates are left in place.
    """
    path = list(path)
    if not path:
        raise ValueError("empty key path")
    head, *rest = path
    lock = writeback_lock if writeback_lock is not None else contextlib.nullcontext()
    with lock:
        if not rest:
            del store[head]
            return
        child = store[head]  # KeyError propagates if the top key is missing
        cur = child
        for key in rest[:-1]:
            cur = cur[key]  # KeyError propagates on a missing intermediate
        del cur[rest[-1]]  # KeyError propagates on a missing leaf
        _boundary_write(store, head, child, (), False, None)
