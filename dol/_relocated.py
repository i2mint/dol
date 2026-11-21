"""Names that left ``dol``'s top-level namespace (issue #145), and how to still reach them.

Accessing one of these as ``dol.<name>`` (or ``from dol import <name>``) still works for
one release, through the module ``__getattr__`` of ``dol/__init__.py``, but emits a
``DeprecationWarning`` that says where to import it from instead:

- **Moved to xdol** (not used anywhere inside dol): import it from ``xdol``.
- **Kept in a dol submodule** (dol itself uses it, and xdol depends on dol, so it cannot
  move): import it from that submodule, e.g. ``from dol.util import partialclass``.

The objects themselves are unchanged: every name resolves to the same object as before.

>>> import warnings
>>> import dol
>>> with warnings.catch_warnings(record=True) as w:
...     warnings.simplefilter("always")
...     f = dol.partialclass
>>> from dol.util import partialclass
>>> f is partialclass
True
>>> print(w[0].category.__name__, '-', w[0].message)  # doctest: +ELLIPSIS
DeprecationWarning - `dol.partialclass` is deprecated ... `from dol.util import partialclass` ...
"""

from importlib import import_module

XDOL = "xdol"

# name -> (dol module that defines it, where users should import it from now)
RELOCATED = {
    # used nowhere inside dol: their public home is now xdol
    "AttributeMutableMapping": ("dol.util", XDOL),
    "ExplicitKeyMap": ("dol.explicit", XDOL),
    "SequenceKvReader": ("dol.sources", XDOL),
    "igroupby": ("dol.util", XDOL),
    "regroupby": ("dol.util", XDOL),
    "ihead": ("dol._relocated", XDOL),
    "kvhead": ("dol._relocated", XDOL),
    "mk_store_from_path_format_store_cls": ("dol.naming", XDOL),
    "redirect_getattr_to_getitem": ("dol.trans", XDOL),
    "trace_getitem": ("dol.dig", XDOL),
    "to_zip_file": ("dol.zipfiledol", XDOL),
    # used inside dol (so they stay), but no longer top-level exports
    "FlatReader": ("dol.sources", "dol.sources"),
    "WriteBackChainMap": ("dol.store_caching", "dol.store_caching"),
    "MappingViewMixin": ("dol.base", "dol.base"),
    "insert_aliases": ("dol.store_addons", "dol.store_addons"),
    "insert_hash_method": ("dol.store_addons", "dol.store_addons"),
    "instance_checker": ("dol.util", "dol.util"),
    "mk_write_cached_store": ("dol.store_caching", "dol.store_caching"),
    "store_cached_with_single_key": ("dol.store_caching", "dol.store_caching"),
    "flush_on_exit": ("dol.store_caching", "dol.store_caching"),
    "partialclass": ("dol.util", "dol.util"),
    "tar_compress": ("dol.zipfiledol", "dol.zipfiledol"),
    "tar_decompress": ("dol.zipfiledol", "dol.zipfiledol"),
    "zip_compress": ("dol.zipfiledol", "dol.zipfiledol"),
}


def deprecation_message(name: str) -> str:
    """The warning text for a relocated top-level name."""
    _, new_home = RELOCATED[name]
    return (
        f"`dol.{name}` is deprecated and will be removed from dol's top-level "
        f"namespace in a future release: use `from {new_home} import {name}` instead "
        "(see https://github.com/i2mint/dol/issues/145)."
    )


def resolve(name: str):
    """Return the object a relocated top-level name refers to (no warning here)."""
    defining_module, _ = RELOCATED[name]
    return getattr(import_module(defining_module), name)


def kvhead(store, n=1):  # frozen copy; xdol.from_dol holds the maintained one
    """Get the first item of a kv store, or a list of the first n items

    >>> kvhead({'a': 1, 'b': 2})
    ('a', 1)
    >>> kvhead({'a': 1, 'b': 2}, 5)
    [('a', 1), ('b', 2)]
    """
    if n == 1:
        for k in store:
            return k, store[k]
    else:
        return [(k, store[k]) for i, k in enumerate(store) if i < n]


def ihead(store, n=1):
    """Get the first item of an iterable, or a list of the first n items

    >>> ihead(iter('abc'))
    'a'
    >>> ihead('abc', 2)
    ['a', 'b']
    """
    if n == 1:
        for item in iter(store):
            return item
    else:
        return [item for i, item in enumerate(store) if i < n]


# Keep pickles and reprs of these two pointing at ``dol`` as before the move
# (``dol.kvhead`` resolves through the shim to this very object).
kvhead.__module__ = ihead.__module__ = "dol"
