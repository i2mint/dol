# dol.records

Records: a queryable CRUD facade for the small rows beside the blobs (#80).

`dol.content` holds the *content* half of the content–metadata bifurcation (blobs
addressed by [`ContentRef`](dol.content.html.md#dol.content.ContentRef)). This module is the *record* half: a
collection facade with CRUD and query, the Python mirror of zodal’s `DataProvider`
(`@zodal/store`), so a Python backend and a zodal frontend speak the same contract.

- [`DataProvider`](#dol.records.DataProvider) — the protocol: `get_list(params) -> ListResult`,
  `get_one`, `create`, `update`, `update_many`, `delete`, `delete_many`,
  optional `upsert` and `capabilities()`.
- [`ListParams`](#dol.records.ListParams) — structured `sort` / `filter` / `search` /
  `pagination`. Filters are zodal’s `FilterExpression` as plain dicts:
  `{"field": "year", "operator": "gte", "value": 2020}`, combined with
  `{"and": [...]}`, `{"or": [...]}`, `{"not": ...}`.
- [`filter_to_function()`](#dol.records.filter_to_function) / [`apply_query()`](#dol.records.apply_query) — the client-side fallback for any
  backend that cannot push a predicate, a sort or a page down.
- [`Capabilities`](#dol.records.Capabilities) — what a provider can do and push down, so callers degrade.
- [`MappingProvider`](#dol.records.MappingProvider) — the reference provider over any `MutableMapping` of
  `id -> record` (a `dict`, or any dol store of dicts, e.g. JSON files).

Stdlib only, like the rest of dol’s core.

```pycon
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
```

### Module Attributes

| [`FILTER_OPERATORS`](#dol.records.FILTER_OPERATORS)   | `operator -> (field_value, value) -> bool`.   |
|---------------------------------------------------------------------|-----------------------------------------------|

### Functions

| [`filter_to_function`](#dol.records.filter_to_function)(filter, \*[, operators])   | A predicate on records from a zodal `FilterExpression`.                     |
|------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`matches_search`](#dol.records.matches_search)(record, search[, fields])      | Case-insensitive substring search over `fields` (default: every str field). |
| [`apply_query`](#dol.records.apply_query)(records[, params, ...])           | Filter, search, sort and page `records` client-side (zodal `applyQuery`).   |

### Classes

| [`Sort`](#dol.records.Sort)(field[, desc])                                | Sort by `field` (zodal `SortingState`: `{id, desc}`).                                                        |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| [`Pagination`](#dol.records.Pagination)([page, page_size])                      | 1-based `page` of `page_size` records.                                                                       |
| [`ListParams`](#dol.records.ListParams)(\*[, sort, filter, search, pagination]) | What `DataProvider.get_list()` returns: filtered, searched, sorted, paged.                                   |
| [`ListResult`](#dol.records.ListResult)(data, total)                            | One page of records and the total number matching (before paging).                                           |
| [`Capabilities`](#dol.records.Capabilities)(\*[, can_create, can_update, ...])    | What a provider supports (zodal `ProviderCapabilities`).                                                     |
| [`DataProvider`](#dol.records.DataProvider)(\*args, \*\*kwargs)                   | CRUD + query over a collection of records (zodal `DataProvider`, sync).                                      |
| [`MappingProvider`](#dol.records.MappingProvider)(store, \*[, id_field, ...])        | A [`DataProvider`](#dol.records.DataProvider) over any `MutableMapping` of `id -> record`. |

### Exceptions

| [`RecordNotFound`](#dol.records.RecordNotFound)   | No record has this id.                                            |
|-------------------------------------------------------------------|-------------------------------------------------------------------|
| [`RecordExists`](#dol.records.RecordExists)     | A record with this id already exists (`create` never overwrites). |

### *class* dol.records.Capabilities(, can_create=True, can_update=True, can_delete=True, can_bulk_update=True, can_bulk_delete=True, can_upsert=False, server_sort=False, server_filter=False, server_search=False, server_pagination=False, filter_operators=None, pagination_style='offset', realtime=False, bifurcated=False, content_fields=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What a provider supports (zodal `ProviderCapabilities`).

`server_*` say which query parts the backend pushes down (`True`, `False`,
or the fields it can do it on); the rest is the caller’s to do client-side, e.g.
with [`apply_query()`](#dol.records.apply_query).

### *class* dol.records.DataProvider(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

CRUD + query over a collection of records (zodal `DataProvider`, sync).

### dol.records.FILTER_OPERATORS *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[Any](https://docs.python.org/3/library/typing.html#typing.Any), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], [bool](https://docs.python.org/3/builtins/functions.html#bool)]]* *= {'arrayContains': <function <lambda>>, 'arrayContainsAny': <function <lambda>>, 'contains': <function \_str_op.<locals>.op>, 'endsWith': <function \_str_op.<locals>.op>, 'eq': <function \_same>, 'gt': <function \_ordered.<locals>.op>, 'gte': <function \_ordered.<locals>.op>, 'in': <function <lambda>>, 'isNotNull': <function <lambda>>, 'isNull': <function <lambda>>, 'lt': <function \_ordered.<locals>.op>, 'lte': <function \_ordered.<locals>.op>, 'ne': <function <lambda>>, 'notIn': <function <lambda>>, 'startsWith': <function \_str_op.<locals>.op>}*

`operator -> (field_value, value) -> bool`.

* **Type:**
  zodal’s filter operators

### *class* dol.records.ListParams(, sort=(), filter=None, search=None, pagination=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What `DataProvider.get_list()` returns: filtered, searched, sorted, paged.

#### *classmethod* from_dict(d)

From zodal’s JSON shape: `{sort: [{id, desc}], filter, search,
pagination: {page, pageSize}}` (snake_case keys accepted too).

* **Return type:**
  [`ListParams`](#dol.records.ListParams)

### *class* dol.records.ListResult(data, total)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One page of records and the total number matching (before paging).

### *class* dol.records.MappingProvider(store, , id_field='id', new_id=None, search_fields=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A [`DataProvider`](#dol.records.DataProvider) over any `MutableMapping` of `id -> record`.

Queries run client-side ([`apply_query()`](#dol.records.apply_query)) over the mapping’s values, so it
suits small collections; a backend that can push queries down implements
[`DataProvider`](#dol.records.DataProvider) itself. Records come back as copies carrying their id (the
key) in `id_field`, so a store keyed by id alone works too. Ids are used as
given (generated ones are strings).

* **Parameters:**
  * **store** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – `id -> record mapping` (a `dict`, or a dol store of dicts).
  * **id_field** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the record field holding its id (kept equal to the key).
  * **new_id** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – makes an id for a record created without one.
  * **search_fields** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – fields `search` looks in (default: every str field).

#### update_many(ids, data)

Apply `data` to every existing record of `ids` (missing ones are skipped).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

#### upsert(data)

Create or replace the record whose id `data` carries.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### *class* dol.records.Pagination(page=1, page_size=25)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

1-based `page` of `page_size` records.

### *exception* dol.records.RecordExists

Bases: [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

A record with this id already exists (`create` never overwrites).

### *exception* dol.records.RecordNotFound

Bases: [`KeyError`](https://docs.python.org/3/builtins/exceptions.html#KeyError)

No record has this id.

### *class* dol.records.Sort(field, desc=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Sort by `field` (zodal `SortingState`: `{id, desc}`).

### dol.records.apply_query(records, params=None, , search_fields=None, skip=())

Filter, search, sort and page `records` client-side (zodal `applyQuery`).

`skip` names the parts a backend already did (`"filter"`, `"search"`,
`"sort"`, `"paginate"`).

* **Return type:**
  [`ListResult`](#dol.records.ListResult)

### dol.records.filter_to_function(filter, \*, operators={'arrayContains': <function <lambda>>, 'arrayContainsAny': <function <lambda>>, 'contains': <function \_str_op.<locals>.op>, 'endsWith': <function \_str_op.<locals>.op>, 'eq': <function \_same>, 'gt': <function \_ordered.<locals>.op>, 'gte': <function \_ordered.<locals>.op>, 'in': <function <lambda>>, 'isNotNull': <function <lambda>>, 'isNull': <function <lambda>>, 'lt': <function \_ordered.<locals>.op>, 'lte': <function \_ordered.<locals>.op>, 'ne': <function <lambda>>, 'notIn': <function <lambda>>, 'startsWith': <function \_str_op.<locals>.op>})

A predicate on records from a zodal `FilterExpression`.

```pycon
>>> is_recent = filter_to_function({"and": [
...     {"field": "year", "operator": "gte", "value": 2020},
...     {"not": {"field": "tags", "operator": "arrayContains", "value": "draft"}}]})
>>> is_recent({"year": 2021, "tags": []}), is_recent({"year": 2021, "tags": ["draft"]})
(True, False)
```

An unknown operator raises `ValueError` (zodal’s matches everything; failing
loud keeps a typo from returning the whole collection).

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]

### dol.records.matches_search(record, search, fields=None)

Case-insensitive substring search over `fields` (default: every str field).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
