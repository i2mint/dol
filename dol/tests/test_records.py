"""dol.records: the record half of the content-metadata bifurcation (#80)."""

import pytest

from dol.records import (
    Capabilities,
    DataProvider,
    ListParams,
    MappingProvider,
    Pagination,
    RecordExists,
    RecordNotFound,
    Sort,
    apply_query,
    filter_to_function,
)

ROWS = [
    {"id": "1", "title": "Dune", "year": 1965, "tags": ["sf"], "rating": 4.5},
    {"id": "2", "title": "Neuromancer", "year": 1984, "tags": ["sf", "cyber"], "rating": None},
    {"id": "3", "title": "Emma", "year": 1815, "tags": [], "rating": 4.0},
    {"id": "4", "title": "dune messiah", "year": 1969, "tags": ["sf"], "rating": 3.5},
]


def _provider():
    return MappingProvider({r["id"]: dict(r) for r in ROWS})


@pytest.mark.parametrize(
    "condition, ids",
    [
        ({"field": "year", "operator": "eq", "value": 1965}, ["1"]),
        ({"field": "year", "operator": "ne", "value": 1965}, ["2", "3", "4"]),
        ({"field": "year", "operator": "lt", "value": 1900}, ["3"]),
        ({"field": "year", "operator": "gte", "value": 1969}, ["2", "4"]),
        ({"field": "title", "operator": "contains", "value": "DUNE"}, ["1", "4"]),
        ({"field": "title", "operator": "startsWith", "value": "neuro"}, ["2"]),
        ({"field": "title", "operator": "endsWith", "value": "MA"}, ["3"]),
        ({"field": "id", "operator": "in", "value": ["1", "3"]}, ["1", "3"]),
        ({"field": "id", "operator": "notIn", "value": ["1", "3"]}, ["2", "4"]),
        ({"field": "tags", "operator": "arrayContains", "value": "cyber"}, ["2"]),
        ({"field": "tags", "operator": "arrayContainsAny", "value": ["x", "sf"]}, ["1", "2", "4"]),
        ({"field": "rating", "operator": "isNull"}, ["2"]),
        ({"field": "rating", "operator": "isNotNull"}, ["1", "3", "4"]),
        ({"field": "rating", "operator": "gt", "value": 4}, ["1"]),  # None never matches
        ({"or": [{"field": "year", "operator": "lt", "value": 1900},
                 {"field": "rating", "operator": "isNull"}]}, ["2", "3"]),
        ({"not": {"field": "tags", "operator": "arrayContains", "value": "sf"}}, ["3"]),
    ],
)
def test_every_zodal_operator(condition, ids):
    keep = filter_to_function(condition)
    assert [r["id"] for r in ROWS if keep(r)] == ids


def test_unknown_operator_and_malformed_condition_fail_loud():
    with pytest.raises(ValueError, match="unknown filter operator"):
        filter_to_function({"field": "year", "operator": "equals", "value": 1})
    with pytest.raises(ValueError, match="field and operator"):
        filter_to_function({"value": 1})


def test_query_search_sort_and_page():
    res = apply_query(
        ROWS,
        ListParams(search="dune", sort=[Sort("year", desc=True)], pagination=Pagination(1, 1)),
    )
    assert res.total == 2 and [r["id"] for r in res.data] == ["4"]
    # None sorts first; ties keep input order (stable)
    assert [r["id"] for r in apply_query(ROWS, ListParams(sort=[Sort("rating")])).data] == ["2", "4", "3", "1"]
    assert apply_query(ROWS, ListParams(pagination=Pagination(9, 2))).data == []
    with pytest.raises(ValueError):
        Pagination(0, 10)


def test_zodal_shaped_params_are_accepted():
    p = ListParams.from_dict(
        {"sort": [{"id": "year", "desc": True}], "pagination": {"page": 2, "pageSize": 2},
         "filter": {"field": "tags", "operator": "arrayContains", "value": "sf"}}
    )
    res = _provider().get_list(p)
    assert res.total == 3 and [r["id"] for r in res.data] == ["1"]
    assert _provider().get_list({"search": "emma"}).total == 1


def test_crud():
    people = _provider()
    assert isinstance(people, DataProvider)
    new = people.create({"title": "Solaris", "year": 1961})
    assert new["id"] == "5" and people.get_one("5")["title"] == "Solaris"
    with pytest.raises(RecordExists):
        people.create({"id": "5", "title": "again"})
    assert people.update("5", {"year": 1962})["year"] == 1962
    assert [r["year"] for r in people.update_many(["1", "4", "nope"], {"year": 2000})] == [2000, 2000]
    people.delete("5")
    with pytest.raises(RecordNotFound):
        people.get_one("5")
    with pytest.raises(RecordNotFound):
        people.delete("5")
    people.delete_many(["1", "nope"])
    assert people.get_list().total == 3
    assert people.upsert({"id": "9", "title": "x"})["id"] == "9"
    assert people.capabilities() == Capabilities(can_upsert=True)


def test_records_come_back_as_copies_and_the_id_follows_renames():
    store = {}
    people = MappingProvider(store)
    rec = people.create({"id": "a", "tags": ["x"]})
    rec["tags"].append("mutated")
    assert store["a"]["tags"] == ["x"]
    people.update("a", {"id": "b"})
    assert list(store) == ["b"] and store["b"]["id"] == "b"
    people.create({"id": "c"})
    with pytest.raises(RecordExists):
        people.update("b", {"id": "c"})


def test_review_edge_cases():
    """Review of #144: renames, bulk ids, non-dict values, id-less values, JS-like eq."""
    store = {"a": {"id": "a", "n": 1}, "b": {"id": "b", "n": 2}}
    people = MappingProvider(store)
    assert people.update("a", {"id": None, "n": 3}) == {"id": "a", "n": 3}
    with pytest.raises(ValueError, match="update_many"):
        people.update_many(["a", "b"], {"id": "z"})
    assert sorted(store) == ["a", "b"]
    with pytest.raises(TypeError, match="not a record mapping"):
        MappingProvider({"x": "text"}).get_list()
    keyed = MappingProvider({"k1": {"name": "Ada"}})  # values without their id
    assert keyed.get_one("k1") == {"name": "Ada", "id": "k1"}
    assert keyed.get_list().data == [{"name": "Ada", "id": "k1"}]
    ints = MappingProvider({})
    assert ints.create({"id": 5})["id"] == 5 and ints.get_one(5)["id"] == 5
    keep = filter_to_function({"field": "flag", "operator": "eq", "value": 1})
    assert keep({"flag": 1}) and not keep({"flag": True})
    has = filter_to_function({"field": "tags", "operator": "arrayContainsAny", "value": {"x"}})
    assert has({"tags": ["x"]}) and not has({"tags": [["x"]]})
    member = filter_to_function({"field": "v", "operator": "in", "value": {1, 2}})
    assert member({"v": 2}) and not member({"v": [1]})
    nan = float("nan")
    rows = [{"v": nan}, {"v": 0.5}, {"v": None}, {"v": 1.0}]
    ordered = [r["v"] for r in apply_query(rows, ListParams(sort=[Sort("v")])).data]
    assert ordered[:3] == [None, 0.5, 1.0] and ordered[3] != ordered[3]
