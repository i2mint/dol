"""The #145 shim: names that left dol's top level still resolve, with a DeprecationWarning."""

import importlib
import warnings

import pytest

import dol
from dol._relocated import RELOCATED


@pytest.mark.parametrize("name", sorted(RELOCATED))
def test_relocated_name_warns_and_resolves_to_the_same_object(name):
    defining_module, new_home = RELOCATED[name]
    with pytest.warns(DeprecationWarning, match=f"from {new_home} import {name}"):
        obj = getattr(dol, name)
    assert obj is getattr(importlib.import_module(defining_module), name)


@pytest.mark.parametrize("name", sorted(RELOCATED))
def test_from_import_still_works(name):
    with pytest.warns(DeprecationWarning):
        namespace = {}
        exec(f"from dol import {name}", namespace)
    assert namespace[name] is not None


def test_relocated_names_are_not_in_the_namespace():
    """So ``from dol import *`` and ``dir(dol)`` no longer advertise them."""
    assert not set(RELOCATED) & set(vars(dol))


def test_submodule_imports_do_not_warn():
    """Where dol still defines a name, importing it from that module is silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name, (defining_module, new_home) in RELOCATED.items():
            if new_home.startswith("dol."):
                getattr(importlib.import_module(new_home), name)


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        dol.this_name_does_not_exist
