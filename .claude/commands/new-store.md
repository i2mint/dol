# Command: new-store

Create a new dol store wrapping a given backend, with key/value transforms.

## Usage

```
/new-store <description of what the store should do>
```

## What this command does

1. Reads `dol/base.py` to understand `KvReader`/`KvPersister`/`Store`
2. Reads `dol/trans.py` to understand `wrap_kvs` and `store_decorator`
3. Asks clarifying questions if needed:
   - What is the backend? (files, S3, DB, dict, custom)
   - What are the keys? (strings, paths, tuples?)
   - What are the values? (bytes, JSON, Python objects?)
   - Read-only or read-write?
   - Any key transformation needed? (prefix, suffix, format conversion)
   - Any value serialization needed? (JSON, pickle, gzip, custom)
4. Generates the store class/factory using `wrap_kvs` (preferred) or subclassing
5. Adds a docstring and a doctest example

## Output pattern

```python
from dol import wrap_kvs, KvReader  # or KvPersister
# + relevant codec imports

def make_<name>_store(root_path, ...):
    """<docstring>

    >>> s = make_<name>_store(...)
    >>> s['key'] = value
    >>> s['key']
    value
    """
    return wrap_kvs(
        <backend>,
        id_of_key=...,
        key_of_id=...,
        obj_of_data=...,
        data_of_obj=...,
    )
```

## Notes

- Always include a working doctest using `dict` as the backend
- Follow the `X_of_Y` naming convention for transforms
- See `CLAUDE.md` and `misc/docs/dol_design.md` for full context
