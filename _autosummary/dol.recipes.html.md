# dol.recipes

Recipes using dol

### Functions

| [`search_paths`](#dol.recipes.search_paths)(pkv_filt, d, \*[, leafs_only, ...])   | Walk a dict, yielding paths to values that pass the `pkv_filt`   |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------|

### dol.recipes.search_paths(pkv_filt, d, , leafs_only=True, breadth_first=False)

Walk a dict, yielding paths to values that pass the `pkv_filt`

* **Parameters:**
  * **pkv_filt** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`PT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)], [`bool`](https://docs.python.org/3/library/functions.html#bool)]) – A function that takes a path, key, and value, and returns
    `True` if the path should be yielded, and `False` otherwise
  * **d** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – The `Mapping` to walk (scan through)
  * **leafs_only** ([`bool`](https://docs.python.org/3/library/functions.html#bool)) – Whether to yield only paths to leafs (default), or to yield
    paths to all values that pass the `pkv_filt`.
  * **breadth_first** ([`bool`](https://docs.python.org/3/library/functions.html#bool)) – Whether to perform breadth-first traversal
    (instead of the default depth-first traversal).
* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`PT`)]
* **Returns:**
  An iterator of paths to values that pass the `pkv_filt`

Example:

```default
>>> d = {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
>>> list(path_filter(lambda p, k, v: v == 2, d))
```

[(‘a’, ‘b’, ‘d’)]

```pycon
>>> mm = {
...     'a': {'b': {'c': 42}},
...     'aa': {'bb': {'cc': 'meaning of life'}},
...     'aaa': {'bbb': 314},
... }
>>> return_path_if_int_leaf = lambda p, k, v: (p, v) if isinstance(v, int) else None
>>> paths = list(path_filter(return_path_if_int_leaf, mm))
>>> paths  # only the paths to the int leaves are returned
[('a', 'b', 'c'), ('aaa', 'bbb')]
```

The `pkv_filt` argument can use path, key, and/or value to define your search
query. For example, let’s extract all the paths that have depth at least 3.

```pycon
>>> paths = list(path_filter(lambda p, k, v: len(p) >= 3, mm))
>>> paths
[('a', 'b', 'c'), ('aa', 'bb', 'cc')]
```

The rationale for `path_filter` yielding matching paths, and not values or keys,
is that if you have the paths, you can than get the keys and values with them,
using `path_get`.

```pycon
>>> from functools import partial, reduce
>>> path_get = lambda m, k: reduce(lambda m, k: m[k], k, m)
>>> extract_paths = lambda m, paths: map(partial(path_get, m), paths)
>>> vals = list(extract_paths(mm, paths))
>>> vals
[42, 'meaning of life']
```

#### NOTE
pkv_filt is first to match the order of the arguments of the
builtin filter function.
