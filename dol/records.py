"""Records: a queryable CRUD facade for the small rows beside the blobs (#80).

``dol.content`` holds the *content* half of the content–metadata bifurcation (blobs
addressed by :class:`~dol.content.ContentRef`). This module is the *record* half: a
collection facade with CRUD and query, the Python mirror of zodal's ``DataProvider``
(``@zodal/store``), so a Python backend and a zodal frontend speak the same contract.

- :class:`DataProvider` — the protocol: ``get_list(params) -> ListResult``,
  ``get_one``, ``create``, ``update``, ``update_many``, ``delete``, ``delete_many``,
  optional ``upsert`` and ``capabilities()``.
- :class:`ListParams` — structured ``sort`` / ``filter`` / ``search`` /
  ``pagination``. Filters are zodal's ``FilterExpression`` as plain dicts:
  ``{"field": "year", "operator": "gte", "value": 2020}``, combined with
  ``{"and": [...]}``, ``{"or": [...]}``, ``{"not": ...}``.
- :func:`filter_to_function` / :func:`apply_query` — the client-side fallback for any
  backend that cannot push a predicate, a sort or a page down.
- :class:`Capabilities` — what a provider can do and push down, so callers degrade.
- :class:`MappingProvider` — the reference provider over any ``MutableMapping`` of
  ``id -> record`` (a ``dict``, or any dol store of dicts, e.g. JSON files).
- :class:`BifurcatedProvider` — records in a provider, their large fields in a
  ``dol.content`` store, with :class:`~dol.content.ContentRef` s in their place.

Stdlib only, like the rest of dol's core.

>>> people = MappingProvider({})
>>> _ = people.create({"id": "ada", "name": "Ada", "born": 1815})
>>> _ = people.create({"id": "alan", "name": "Alan", "born": 1912})
>>> _ = people.create({"name": "Grace", "born": 1906})       # an id is generated
>>> res = people.get_list(ListParams(
...     filter={"field": "born", "operator": "gt", "value": 1900},
...     sort=[Sort("born", desc=True)]))
>>> [r["name"] for r in res.data], res.total
(['Alan', 'Grace'], 2)
>>> people.update("ada", {"born": 1816})["born"]
1816
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

Record = dict[str, Any]
#: zodal's ``FilterExpression``: a condition dict, or ``and`` / ``or`` / ``not`` of them.
FilterExpression = Mapping[str, Any]

__all__ = [
    "Sort",
    "Pagination",
    "ListParams",
    "ListResult",
    "Capabilities",
    "DataProvider",
    "MappingProvider",
    "BifurcatedProvider",
    "FILTER_OPERATORS",
    "filter_to_function",
    "matches_search",
    "apply_query",
    "RecordNotFound",
    "RecordExists",
]


class RecordNotFound(KeyError):
    """No record has this id."""


class RecordExists(ValueError):
    """A record with this id already exists (``create`` never overwrites)."""


# --------------------------------------------------------------------------- params


@dataclass(frozen=True)
class Sort:
    """Sort by ``field`` (zodal ``SortingState``: ``{id, desc}``)."""

    field: str
    desc: bool = False


@dataclass(frozen=True)
class Pagination:
    """1-based ``page`` of ``page_size`` records."""

    page: int = 1
    page_size: int = 25

    def __post_init__(self):
        if self.page < 1 or self.page_size < 1:
            raise ValueError(f"page and page_size must be >= 1, got {self}")


@dataclass(frozen=True, kw_only=True)
class ListParams:
    """What :meth:`DataProvider.get_list` returns: filtered, searched, sorted, paged."""

    sort: Sequence[Sort] = ()
    filter: FilterExpression | None = None
    search: str | None = None
    pagination: Pagination | None = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> ListParams:
        """From zodal's JSON shape: ``{sort: [{id, desc}], filter, search,
        pagination: {page, pageSize}}`` (snake_case keys accepted too)."""
        d = dict(d or {})
        sort = tuple(
            Sort(s.get("id", s.get("field")), bool(s.get("desc", False)))
            for s in d.get("sort") or ()
        )
        p = d.get("pagination")
        pagination = (
            Pagination(p.get("page", 1), p.get("pageSize", p.get("page_size", 25)))
            if p
            else None
        )
        return cls(
            sort=sort,
            filter=d.get("filter"),
            search=d.get("search"),
            pagination=pagination,
        )


@dataclass(frozen=True)
class ListResult:
    """One page of records and the total number matching (before paging)."""

    data: list[Record]
    total: int


@dataclass(frozen=True, kw_only=True)
class Capabilities:
    """What a provider supports (zodal ``ProviderCapabilities``).

    ``server_*`` say which query parts the backend pushes down (``True``, ``False``,
    or the fields it can do it on); the rest is the caller's to do client-side, e.g.
    with :func:`apply_query`.
    """

    can_create: bool = True
    can_update: bool = True
    can_delete: bool = True
    can_bulk_update: bool = True
    can_bulk_delete: bool = True
    can_upsert: bool = False
    server_sort: bool | Sequence[str] = False
    server_filter: bool | Sequence[str] = False
    server_search: bool = False
    server_pagination: bool = False
    filter_operators: Mapping[str, Sequence[str]] | None = None
    pagination_style: str = "offset"
    realtime: bool = False
    bifurcated: bool = False
    content_fields: Sequence[str] = ()


@runtime_checkable
class DataProvider(Protocol):
    """CRUD + query over a collection of records (zodal ``DataProvider``, sync)."""

    def get_list(self, params: ListParams | None = None) -> ListResult: ...
    def get_one(self, id: str) -> Record: ...
    def create(self, data: Mapping[str, Any]) -> Record: ...
    def update(self, id: str, data: Mapping[str, Any]) -> Record: ...
    def update_many(
        self, ids: Iterable[str], data: Mapping[str, Any]
    ) -> list[Record]: ...
    def delete(self, id: str) -> None: ...
    def delete_many(self, ids: Iterable[str]) -> None: ...


# -------------------------------------------------------------------------- filters


def _lower_str(v):
    return v.lower() if isinstance(v, str) else None


def _str_op(test: Callable[[str, str], bool]):
    def op(field_value, value):
        a, b = _lower_str(field_value), _lower_str(value)
        return a is not None and b is not None and test(a, b)

    return op


def _ordered(test: Callable[[Any, Any], bool]):
    def op(field_value, value):
        try:
            return field_value is not None and test(field_value, value)
        except TypeError:  # incomparable types: no match, as in JS
            return False

    return op


def _same(a, b) -> bool:
    """JS ``===`` for JSON values: ``True`` is not ``1``."""
    return a == b and isinstance(a, bool) == isinstance(b, bool)


def _member(x, values) -> bool:
    return isinstance(values, (list, tuple, set, frozenset)) and any(
        _same(x, v) for v in values
    )


#: zodal's filter operators: ``operator -> (field_value, value) -> bool``.
FILTER_OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": _same,
    "ne": lambda f, v: not _same(f, v),
    "gt": _ordered(lambda f, v: f > v),
    "gte": _ordered(lambda f, v: f >= v),
    "lt": _ordered(lambda f, v: f < v),
    "lte": _ordered(lambda f, v: f <= v),
    "contains": _str_op(lambda f, v: v in f),
    "startsWith": _str_op(lambda f, v: f.startswith(v)),
    "endsWith": _str_op(lambda f, v: f.endswith(v)),
    "in": lambda f, v: _member(f, v),
    "notIn": lambda f, v: not _member(f, v),
    "arrayContains": lambda f, v: isinstance(f, (list, tuple)) and _member(v, f),
    "arrayContainsAny": lambda f, v: (
        isinstance(f, (list, tuple))
        and isinstance(v, (list, tuple, set, frozenset))
        and any(_member(x, f) for x in v)
    ),
    "isNull": lambda f, v: f is None,
    "isNotNull": lambda f, v: f is not None,
}


def filter_to_function(
    filter: FilterExpression,
    *,
    operators: Mapping[str, Callable[[Any, Any], bool]] = FILTER_OPERATORS,
) -> Callable[[Mapping[str, Any]], bool]:
    """A predicate on records from a zodal ``FilterExpression``.

    >>> is_recent = filter_to_function({"and": [
    ...     {"field": "year", "operator": "gte", "value": 2020},
    ...     {"not": {"field": "tags", "operator": "arrayContains", "value": "draft"}}]})
    >>> is_recent({"year": 2021, "tags": []}), is_recent({"year": 2021, "tags": ["draft"]})
    (True, False)

    An unknown operator raises ``ValueError`` (zodal's matches everything; failing
    loud keeps a typo from returning the whole collection).
    """
    if "and" in filter:
        preds = [filter_to_function(f, operators=operators) for f in filter["and"]]
        return lambda r: all(p(r) for p in preds)
    if "or" in filter:
        preds = [filter_to_function(f, operators=operators) for f in filter["or"]]
        return lambda r: any(p(r) for p in preds)
    if "not" in filter:
        pred = filter_to_function(filter["not"], operators=operators)
        return lambda r: not pred(r)
    try:
        name, op_name, value = filter["field"], filter["operator"], filter.get("value")
    except KeyError as e:
        raise ValueError(
            f"a filter condition needs field and operator: {filter!r}"
        ) from e
    try:
        op = operators[op_name]
    except KeyError:
        raise ValueError(
            f"unknown filter operator {op_name!r}; known: {sorted(operators)}"
        ) from None
    return lambda r: op(r.get(name), value)


def matches_search(
    record: Mapping[str, Any], search: str | None, fields: Sequence[str] | None = None
) -> bool:
    """Case-insensitive substring search over ``fields`` (default: every str field)."""
    if not search:
        return True
    needle = search.lower()
    keys = (
        fields
        if fields is not None
        else [k for k, v in record.items() if isinstance(v, str)]
    )
    return any(
        isinstance(record.get(k), str) and needle in record[k].lower() for k in keys
    )


def _sort_key(value):
    # None first, then numbers (NaN after them), strings, bools, others (zodal's
    # compareValues order; strings compare by code point, not localeCompare).
    if value is None:
        return (0, 0)
    if isinstance(value, bool):
        return (3, value)
    if isinstance(value, float) and value != value:  # NaN
        return (1.5, 0)
    if isinstance(value, (int, float)):
        return (1, value)
    if isinstance(value, str):
        return (2, value)
    return (4, 0)  # no meaningful order: keep input order (sorts are stable)


def apply_query(
    records: Iterable[Mapping[str, Any]],
    params: ListParams | None = None,
    *,
    search_fields: Sequence[str] | None = None,
    skip: Iterable[str] = (),
) -> ListResult:
    """Filter, search, sort and page ``records`` client-side (zodal ``applyQuery``).

    ``skip`` names the parts a backend already did (``"filter"``, ``"search"``,
    ``"sort"``, ``"paginate"``).
    """
    params = params or ListParams()
    skip = set(skip)
    out = list(records)
    if params.filter and "filter" not in skip:
        keep = filter_to_function(params.filter)
        out = [r for r in out if keep(r)]
    if params.search and "search" not in skip:
        out = [r for r in out if matches_search(r, params.search, search_fields)]
    if params.sort and "sort" not in skip:
        for s in reversed(params.sort):  # stable sorts, last key first
            out.sort(key=lambda r: _sort_key(r.get(s.field)), reverse=s.desc)
    total = len(out)
    if params.pagination and "paginate" not in skip:
        p = params.pagination
        out = out[(p.page - 1) * p.page_size : p.page * p.page_size]
    return ListResult(out, total)


# ------------------------------------------------------------------------- provider


class MappingProvider:
    """A :class:`DataProvider` over any ``MutableMapping`` of ``id -> record``.

    Queries run client-side (:func:`apply_query`) over the mapping's values, so it
    suits small collections; a backend that can push queries down implements
    :class:`DataProvider` itself. Records come back as copies carrying their id (the
    key) in ``id_field``, so a store keyed by id alone works too. Ids are used as
    given (generated ones are strings).

    :param store: ``id -> record mapping`` (a ``dict``, or a dol store of dicts).
    :param id_field: the record field holding its id (kept equal to the key).
    :param new_id: makes an id for a record created without one.
    :param search_fields: fields ``search`` looks in (default: every str field).
    """

    def __init__(
        self,
        store: MutableMapping[Any, Mapping[str, Any]],
        *,
        id_field: str = "id",
        new_id: Callable[[MutableMapping], Any] | None = None,
        search_fields: Sequence[str] | None = None,
    ):
        self.store = store
        self.id_field = id_field
        self.new_id = new_id or self._next_free_int_id
        self.search_fields = search_fields

    @staticmethod
    def _next_free_int_id(store: Mapping) -> str:
        return next(
            str(i) for i in itertools.count(len(store) + 1) if str(i) not in store
        )

    def _as_record(self, id: Any, value: Any) -> Record:
        if not isinstance(value, Mapping):
            raise TypeError(
                f"the store's value for {id!r} is a {type(value).__name__}, not a "
                "record mapping; wrap the store with a codec that yields dicts"
            )
        return {**copy.deepcopy(dict(value)), self.id_field: id}

    def _record(self, id: Any) -> Record:
        try:
            value = self.store[id]
        except KeyError:
            raise RecordNotFound(id) from None
        return self._as_record(id, value)

    def get_list(self, params: ListParams | Mapping | None = None) -> ListResult:
        if not isinstance(params, ListParams):
            params = ListParams.from_dict(params)
        records = [self._as_record(k, v) for k, v in self.store.items()]
        return apply_query(records, params, search_fields=self.search_fields)

    def get_one(self, id: Any) -> Record:
        return self._record(id)

    def create(self, data: Mapping[str, Any]) -> Record:
        given = data.get(self.id_field)
        id = given if given is not None else self.new_id(self.store)
        if id in self.store:
            raise RecordExists(id)
        record = {**copy.deepcopy(dict(data)), self.id_field: id}
        self.store[id] = record
        return copy.deepcopy(record)

    def update(self, id: Any, data: Mapping[str, Any]) -> Record:
        record = {**self._record(id), **copy.deepcopy(dict(data))}
        new_id = record.get(self.id_field)
        if new_id is None:  # an explicit None keeps the id
            new_id = id
        record[self.id_field] = new_id
        if new_id != id:  # renaming: the key follows the id field
            if new_id in self.store:
                raise RecordExists(new_id)
            self.store[new_id] = record  # write first: a failed write loses nothing
            del self.store[id]
        else:
            self.store[id] = record
        return copy.deepcopy(record)

    def update_many(self, ids: Iterable[Any], data: Mapping[str, Any]) -> list[Record]:
        """Apply ``data`` to every existing record of ``ids`` (missing ones are skipped)."""
        if data.get(self.id_field) is not None:
            raise ValueError(
                f"update_many cannot set {self.id_field!r}: ids are unique"
            )
        return [self.update(id, data) for id in ids if id in self.store]

    def delete(self, id: Any) -> None:
        try:
            del self.store[id]
        except KeyError:
            raise RecordNotFound(id) from None

    def delete_many(self, ids: Iterable[Any]) -> None:
        for id in ids:
            self.store.pop(id, None)

    def upsert(self, data: Mapping[str, Any]) -> Record:
        """Create or replace the record whose id ``data`` carries."""
        id = data.get(self.id_field)
        if id is None:
            raise ValueError(f"upsert needs the record's {self.id_field!r}")
        record = {**copy.deepcopy(dict(data)), self.id_field: id}
        self.store[id] = record
        return copy.deepcopy(record)

    def capabilities(self) -> Capabilities:
        return Capabilities(can_upsert=True)


# ----------------------------------------------------------------- bifurcation


def _default_content_key(item_id: Any, field: str) -> str:
    return f"{item_id}.{field}"


class BifurcatedProvider:
    """Records in a metadata provider, their large fields in a content store.

    The record half of the content-metadata bifurcation: on write, the
    ``content_fields`` of a record go to ``content_store`` (bytes, via
    :func:`dol.content.put_content`) and the record keeps a
    :class:`~dol.content.ContentRef` (its wire-form dict) in their place; on read the
    refs come back, light enough to list, and :meth:`get_content` (or
    ``get_one(id, resolve_content=True)``) fetches the bytes.

    The metadata record is the commit point: content is written before the record
    that refers to it is (for a new record, the record is created first to learn its
    id, then compensated by deletion if a content write fails), and deleted after the
    record is gone (a failed content delete leaves an orphan blob, never a dangling
    ref).

    >>> from dol.content import ContentRef
    >>> clips = BifurcatedProvider(MappingProvider({}), {}, content_fields=["audio"])
    >>> rec = clips.create({"title": "take 1", "audio": b"RIFF..."})
    >>> ContentRef.from_json(rec["audio"]).item_id
    '1.audio'
    >>> clips.get_content(rec["id"], "audio")
    b'RIFF...'
    >>> clips.get_one("1", resolve_content=True)["audio"]
    b'RIFF...'

    :param metadata: the :class:`DataProvider` holding the records.
    :param content_store: a ``MutableMapping`` of bytes (a dict, a dol ``Files``, S3...).
    :param content_fields: the record fields whose values are bytes to offload.
    :param content_key: ``(record id, field) -> content store key``.
    """

    def __init__(
        self,
        metadata: DataProvider,
        content_store: MutableMapping[str, bytes],
        *,
        content_fields: Sequence[str],
        id_field: str = "id",
        content_key: Callable[[Any, str], str] = _default_content_key,
    ):
        if not content_fields:
            raise ValueError("content_fields must name at least one field")
        self.metadata, self.content_store = metadata, content_store
        self.content_fields = tuple(content_fields)
        self.id_field, self.content_key = id_field, content_key

    def _split(self, data: Mapping[str, Any]) -> tuple[dict, dict]:
        meta, content = {}, {}
        for k, v in data.items():  # a None content field stays in the record as None
            (content if k in self.content_fields and v is not None else meta)[k] = v
        return meta, content

    def _put(self, item_id: Any, content: Mapping[str, Any]) -> dict[str, dict]:
        from dol.content import put_content

        refs = {}
        for field_name, value in content.items():
            if value is None:
                continue
            if not isinstance(value, (bytes, bytearray)):
                raise TypeError(
                    f"content field {field_name!r} must be bytes, got {type(value).__name__}"
                )
            key = self.content_key(item_id, field_name)
            refs[field_name] = put_content(
                self.content_store, key, bytes(value), field=field_name
            ).to_json()
        return refs

    def _drop_content(self, record: Mapping[str, Any]) -> None:
        from dol.content import delete_content, is_content_ref

        for field_name in self.content_fields:
            ref = record.get(field_name)
            if is_content_ref(ref):
                try:
                    delete_content(self.content_store, ref)
                except KeyError:
                    pass  # already gone

    # -- DataProvider ---------------------------------------------------------------

    def get_list(self, params: ListParams | Mapping | None = None) -> ListResult:
        return self.metadata.get_list(params)

    def get_one(self, id: Any, *, resolve_content: bool = False) -> Record:
        record = self.metadata.get_one(id)
        if resolve_content:
            for field_name in self.content_fields:
                if record.get(field_name) is not None:
                    record[field_name] = self.get_content(id, field_name)
        return record

    def get_content(self, id: Any, field: str) -> bytes:
        """The bytes of one content field of record ``id``."""
        from dol.content import get_content

        ref = self.metadata.get_one(id).get(field)
        if ref is None:
            raise KeyError(f"record {id!r} has no {field!r} content")
        return get_content(self.content_store, ref)

    def create(self, data: Mapping[str, Any]) -> Record:
        meta, content = self._split(data)
        created = self.metadata.create(meta)
        if not any(v is not None for v in content.values()):
            return created
        id = created[self.id_field]
        try:
            refs = self._put(id, content)
            return self.metadata.update(id, refs)
        except Exception:
            self._drop_content(
                {
                    f: {"_tag": "ContentRef", "itemId": self.content_key(id, f)}
                    for f in content
                }
            )
            try:
                self.metadata.delete(
                    id
                )  # compensate: the record must not outlive a failure
            except Exception:
                pass
            raise

    def update(self, id: Any, data: Mapping[str, Any]) -> Record:
        meta, content = self._split(data)
        self.metadata.get_one(id)  # RecordNotFound before any blob is written
        refs = self._put(id, content)  # content first: the record then commits it
        return self.metadata.update(id, {**meta, **refs})

    def update_many(self, ids: Iterable[Any], data: Mapping[str, Any]) -> list[Record]:
        meta, content = self._split(data)
        if not content:
            return self.metadata.update_many(ids, meta)
        updated = []
        for id in ids:
            try:
                updated.append(self.update(id, data))
            except RecordNotFound:
                continue  # missing ids are skipped, as in update_many
        return updated

    def delete(self, id: Any) -> None:
        record = self.metadata.get_one(id)
        self.metadata.delete(id)  # the commit point goes first: no dangling ref
        self._drop_content(record)

    def delete_many(self, ids: Iterable[Any]) -> None:
        for id in ids:
            try:
                self.delete(id)
            except RecordNotFound:
                pass

    def capabilities(self) -> Capabilities:
        base = getattr(self.metadata, "capabilities", None)
        caps = base() if callable(base) else Capabilities()
        return replace(
            caps, bifurcated=True, content_fields=self.content_fields, can_upsert=False
        )
