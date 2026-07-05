"""Tests for the confirm_overwrite wrap_kvs preset (Issue #13)."""

import pytest

from dol import wrap_kvs, confirm_overwrite, mk_confirm_overwrite_preset


def test_confirm_overwrite_no_prompt_for_same_value_or_new_key(monkeypatch):
    # confirm_overwrite uses the builtin input; it must NOT be called in these cases
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("prompted!"))
    d = wrap_kvs(dict(a="apple", b="banana"), preset=confirm_overwrite)
    d["a"] = "apple"  # same value under existing key -> no prompt
    d["c"] = "coconut"  # brand new key -> no prompt
    assert dict(d) == {"a": "apple", "b": "banana", "c": "coconut"}


def test_confirm_overwrite_confirmed(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "alligator")
    store = dict(a="apple")
    d = wrap_kvs(store, preset=confirm_overwrite)
    d["a"] = "alligator"  # differing value; user types the new value -> overwrite
    assert store["a"] == "alligator"


def test_confirm_overwrite_declined(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "")  # anything != new value
    store = dict(a="apple")
    d = wrap_kvs(store, preset=confirm_overwrite)
    d["a"] = "alligator"  # differing value; declined -> keep existing
    assert store["a"] == "apple"


def test_mk_confirm_overwrite_preset_injectable_input():
    typed = []

    def fake_input(prompt):
        typed.append(prompt)
        return "yes-please"

    preset = mk_confirm_overwrite_preset(get_input=fake_input)
    store = dict(k="old")
    d = wrap_kvs(store, preset=preset)
    d["k"] = "yes-please"  # matches what fake_input returns -> overwrite
    assert store["k"] == "yes-please"
    assert typed and "old" in typed[0]  # the prompt mentioned the existing value
