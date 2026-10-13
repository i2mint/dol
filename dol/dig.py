"""Layers introspection: walk the layers of a wrapped store and trace a key through them.

Main entry points:

- ``layers``: the list of nested layers (found through the ``store`` attribute), outermost first
- ``trace_getitem``, ``print_trace_info``: the (layer, method, value) steps of a ``__getitem__``
"""

from functools import partial

# TODO: Make a generator and a index getter for keys and vals (and both).
#  Point is to be able to get views from any level.

not_found = type("NotFound", (), {})()
no_default = type("NoDefault", (), {})()


def get_first_attr_found(store, attrs, default=no_default):
    for attr in attrs:
        a = getattr(store, attr, not_found)
        if a != not_found:
            return a
    if default == no_default:
        raise AttributeError(f"None of the attributes were found: {attrs}")
    else:
        return default


def recursive_get_attr(store, attr, default=None):
    a = getattr(store, attr, None)
    if a is not None:
        return a
    elif hasattr(store, "store"):
        return recursive_get_attr(store.store, attr, default)
    else:
        return default


re_get_attr = recursive_get_attr
dig_up = recursive_get_attr


def store_trans_path(store, arg, method):
    """Yield ``arg`` transformed by ``method`` at each layer, outermost first.

    Walks the ``.store`` chain, applying ``store.<method>`` at every layer that defines it.

    >>> from dol import wrap_kvs
    >>> s = wrap_kvs({'a.txt': 1}, id_of_key=lambda k: k + '.txt')
    >>> list(store_trans_path(s, 'a', '_id_of_key'))
    ['a.txt', 'a.txt']

    Yields nothing when no layer defines ``method`` -- which is why :func:`inner_most`
    raises rather than returning the ``None`` that an empty walk would otherwise produce.

    >>> list(store_trans_path({}, 'a', '_id_of_key'))
    []
    """
    f = getattr(store, method, None)
    if f is not None:
        trans_arg = f(arg)
        yield trans_arg
        if hasattr(store, "store"):
            # NOTE: recurse with the SAME ``method``. This used to hardcode ``unravel_key``
            # (i.e. ``_id_of_key``), so ``unravel_val``/``inner_most_val`` applied
            # ``_data_of_obj`` at the top layer and ``_id_of_key`` at every deeper one.
            yield from store_trans_path(store.store, trans_arg, method)


def print_trans_path(store, arg, method, with_type=False):
    gen = (arg, *store_trans_path(store, arg, method))
    if with_type:
        gen = map(lambda x: f"{type(x)}: {x}", gen)
    else:
        gen = map(str, gen)
    print("\n".join(gen))


def last_element(gen):
    x = None
    for x in gen:
        pass
    return x


def inner_most(store, arg, method, default=no_default):
    """The value of ``arg`` after every layer's ``method`` has been applied.

    >>> from dol import wrap_kvs
    >>> s = wrap_kvs({'a.txt': 1}, id_of_key=lambda k: k + '.txt')
    >>> inner_most(s, 'a', '_id_of_key')
    'a.txt'

    **Raises when no layer defines ``method``**, instead of silently returning ``None``:

    >>> inner_most({}, 'a', '_id_of_key')
    Traceback (most recent call last):
      ...
    AttributeError: No layer of dict defines '_id_of_key', so 'a' cannot be resolved. ...

    A ``None`` here is the worst possible answer: callers use the result as a key or a
    path, so it surfaces far from its cause -- as ``https://.../None``, or as
    ``TypeError: expected str, bytes or os.PathLike object, not NoneType``.

    Pass ``default`` to opt out of raising:

    >>> inner_most({}, 'a', '_id_of_key', default=None) is None
    True
    """
    x = last_element(store_trans_path(store, arg, method))
    if x is None and getattr(store, method, None) is None:
        # Distinguish "no layer supplied the method" (a resolution failure) from "a layer
        # legitimately returned None" (only reachable when the outermost layer HAS the method).
        if default is no_default:
            raise AttributeError(
                f"No layer of {type(store).__name__} defines {method!r}, so {arg!r} "
                f"cannot be resolved. Pass default= if you want a fallback instead of "
                f"this error. (Previously this returned None silently.)"
            )
        return default
    return x


def _layer_filters(layer):
    """The ``filt_iter`` predicates of one layer (all classes of its MRO that recorded some).

    Only a class's *own* record counts: a ``Store`` wrapper also delegates the attributes of
    the store it wraps (a ``DelegatedAttribute``), which would attribute an inner layer's
    predicates to the outer one, in the wrong key coordinates.
    """
    return [
        f
        for cls in type(layer).__mro__
        if isinstance(own := cls.__dict__.get("_filt_iter_filts"), tuple)
        for f in own
    ]


def key_in_scope(store, k) -> bool:
    """Whether the key ``k`` of ``store`` is inside every ``filt_iter`` scope of its layers.

    A side-effect-free probe: nothing is read or written. ``k`` is followed through the layers
    outermost first; each layer's ``filt_iter`` predicates are applied to the key *as it
    enters that layer*, which is what ``store[k] = v`` would do, then ``_id_of_key`` maps it
    for the layer below. A leaf that has an ``is_valid_key`` method (e.g. ``Files``) is asked
    about the innermost key last.

    >>> from dol import filt_iter, KeyCodecs
    >>> s = KeyCodecs.prefixed('data/')(filt_iter(dict, filt=lambda k: k.endswith('.txt')))()
    >>> key_in_scope(s, 'a.txt'), key_in_scope(s, 'a.bin')
    (True, False)

    Here the ``filt_iter`` sits *below* the prefix, so it sees ``'data/a.txt'``, not
    ``'a.txt'``:

    >>> s = KeyCodecs.prefixed('data/')(filt_iter(dict, filt=lambda k: k.startswith('data/a')))()
    >>> key_in_scope(s, 'a.txt'), key_in_scope(s, 'b.txt')
    (True, False)

    The predicate is the layer's own, so a predicate that raises on a key raises here too
    (as ``s[k]`` would). A layer that maps keys but has no ``_id_of_key`` cannot be followed:
    ``TypeError``.
    """
    layer, seen = store, set()
    while True:
        for filt in _layer_filters(layer):
            if not filt(k):
                return False
        inner = getattr(layer, "store", None)
        if inner is None or id(layer) in seen:
            break
        seen.add(id(layer))
        id_of_key = getattr(layer, "_id_of_key", None)
        if not callable(id_of_key):
            raise TypeError(
                f"{type(layer).__name__} wraps a store but defines no _id_of_key, so a key "
                "cannot be followed through it"
            )
        k = id_of_key(k)
        layer = inner
    is_valid_key = getattr(layer, "is_valid_key", None)
    return bool(is_valid_key(k)) if callable(is_valid_key) else True


# TODO: Better change the signature to reflect context (k (key) or v (val) instead of arg)
unravel_key = partial(store_trans_path, method="_id_of_key")
print_key_trans_path = partial(print_trans_path, method="_id_of_key")
inner_most_key = partial(inner_most, method="_id_of_key")

# TODO: inner_most_val doesn't really do what one expects. It just does what inner_most_key does
unravel_val = partial(store_trans_path, method="_data_of_obj")
print_val_trans_path = partial(print_trans_path, method="_data_of_obj")
inner_most_val = partial(inner_most, method="_data_of_obj")

from functools import partial


def next_layer(store, layer_attrs=("store",)):
    for attr in layer_attrs:
        attr_val = getattr(store, attr, not_found)
        if attr_val is not not_found:
            return attr_val
    return not_found


def recursive_calls(func, x, sentinel=not_found):
    while True:
        if x is sentinel:
            break
        else:
            yield x
            x = func(x)


def layers(store, layer_attrs=("store",)):
    _next_layer = partial(next_layer, layer_attrs=layer_attrs)
    return list(recursive_calls(_next_layer, store))


def trace_getitem(store, k, layer_attrs=("store",)):
    """A generator of layered steps to inspect a store.

    :param store: An instance that has the base.Store interface
    :param k: A key
    :param layer_attrs: The attribute names that should be checked to get the next layer.
    :return: A generator of (layer, method, value)

    We start with a small dict:

    >>> d = {'a.num': '1000', 'b.num': '2000'}

    Now let's add layers to it. For example, with wrap_kvs:

    >>> from dol.trans import wrap_kvs

    Say that we want the interface to not see the ``'.num'`` strings, and deal with numerical values, not strings.

    >>> s = wrap_kvs(d,
    ...              key_of_id=lambda x: x[:-len('.num')],
    ...              id_of_key=lambda x: x + '.num',
    ...              obj_of_data=lambda x: int(x),
    ...              data_of_obj=lambda x: str(x)
    ...             )
    >>>

    Oh, and we want the interface to display upper case keys.

    >>> ss = wrap_kvs(s,
    ...              key_of_id=lambda x: x.upper(),
    ...              id_of_key=lambda x: x.lower(),
    ...             )

    And we want the numerical unit to be the kilo (that's 1000):

    >>> sss = wrap_kvs(ss,
    ...                obj_of_data=lambda x: x / 1000,
    ...                data_of_obj=lambda x: x * 1000
    ...               )
    >>>
    >>> dict(sss.items())
    {'A': 1.0, 'B': 2.0}

    Well, if we had bugs, we'd like to inspect the various layers, and how they transform the data.

    .. code-block:: python

        # Here's how to do that:

        # >>> for layer, method, value in trace_getitem(sss, 'A'):
        # ...     print(layer, method, value)
        # ...
        # Traceback (most recent call last):
        #   File "<stdin>", line 1, in <module>
        # NameError: name 'trace_getitem' is not defined

    >>> from dol.dig import trace_getitem
    >>>
    >>> for layer, method, value in trace_getitem(sss, 'A'):
    ...     print(layer, method, value)
    ...
    {'a.num': '1000', 'b.num': '2000'} _id_of_key A
    {'a.num': '1000', 'b.num': '2000'} _id_of_key A
    {'a.num': '1000', 'b.num': '2000'} _id_of_key a
    {'a.num': '1000', 'b.num': '2000'} _id_of_key a
    {'a.num': '1000', 'b.num': '2000'} _id_of_key a.num
    {'a.num': '1000', 'b.num': '2000'} _id_of_key a.num
    {'a.num': '1000', 'b.num': '2000'} __getitem__ 1000
    {'a.num': '1000', 'b.num': '2000'} _obj_of_data 1000
    {'a.num': '1000', 'b.num': '2000'} _obj_of_data 1000
    {'a.num': '1000', 'b.num': '2000'} _obj_of_data 1000
    {'a.num': '1000', 'b.num': '2000'} _obj_of_data 1000
    {'a.num': '1000', 'b.num': '2000'} _obj_of_data 1000
    {'a.num': '1000', 'b.num': '2000'} _obj_of_data 1.0
    """
    _layers = layers(store, layer_attrs)

    layer = None
    for i, layer in enumerate(_layers):
        if hasattr(layer, "_id_of_key"):
            k = layer._id_of_key(k)
            yield (layer, "_id_of_key", k)

    if layer is not None:
        v = layer[k]
        yield (layer, "__getitem__", v)

        for layer in _layers[:i][::-1]:
            if hasattr(layer, "_obj_of_data"):
                v = layer._obj_of_data(v)
                yield (layer, "_obj_of_data", v)


def trace_info(trace, item_func=print):
    for item in trace:
        item_func(item)


def _trace_item_info(item):
    layer, method, value = item
    return f"{layer.__class__.__name__}.{method}: {type(value).__name__}"


def print_trace_info(trace, item_info=_trace_item_info):
    for item in trace:
        print(item_info(item))
