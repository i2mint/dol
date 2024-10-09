"""
Tests for appendable.py
"""

from dol.appendable import Extender, read_add_write


def test_extender():
    store = {"a": "pple"}
    # test normal extend
    a_extender = Extender(store, "a")
    a_extender.extend("sauce")
    assert store == {"a": "pplesauce"}
    # test creation (when key is not in store)
    b_extender = Extender(store, "b")
    b_extender.extend("anana")
    assert store == {"a": "pplesauce", "b": "anana"}
    # you can use the += operator too
    b_extender += " split"
    assert store == {"a": "pplesauce", "b": "anana split"}

    # test append
    # Need to define an append method that makes sense.
    # Here, with strings, we can just call extend.
    b_bis_extender = Extender(
        store, "b", append_method=lambda self, obj: self.extend(obj)
    )
    b_bis_extender.append("s")
    assert store == {"a": "pplesauce", "b": "anana splits"}
    # But if our "extend" values were lists, we'd need to have a different append method,
    # one that puts the single object into a list, so that its sum with the existing list
    # is a list.
    store = {"c": [1, 2, 3]}
    c_extender = Extender(
        store, "c", append_method=lambda self, obj: self.extend([obj])
    )
    c_extender.append(4)
    assert store == {"c": [1, 2, 3, 4]}
    # And if the values were tuples, we'd have to put the single object into a tuple.
    store = {"d": (1, 2, 3)}
    d_extender = Extender(
        store, "d", append_method=lambda self, obj: self.extend((obj,))
    )
    d_extender.append(4)
    assert store == {"d": (1, 2, 3, 4)}

    # Now, the default extend method is `read_add_write`, which retrieves the existing
    # value, sums it to the new value, and writes it back to the store.
    # If the values of your store have a sum defined (i.e. an `__add__` method),
    # **and** that sum method does what you want, then you can use the default
    # `extend_store_value` function.
    # O ye numpy users, beware! The sum of numpy arrays is an elementwise sum,
    # not a concatenation (you'd have to use `np.concatenate` for that).
    try:
        import numpy as np

        store = {"e": np.array([1, 2, 3])}
        e_extender = Extender(store, "e")
        e_extender.extend(np.array([4, 5, 6]))
        assert all(store["e"] == np.array([5, 7, 9]))
        # This is what the `extend_store_value` function is for: you can pass it a function
        # that does what you want.
        store = {"f": np.array([1, 2, 3])}

        def extend_store_value_for_numpy(store, key, iterable):
            store[key] = np.concatenate([store[key], iterable])

        f_extender = Extender(
            store, "f", extend_store_value=extend_store_value_for_numpy
        )
        f_extender.extend(np.array([4, 5, 6]))
        assert all(store["f"] == np.array([1, 2, 3, 4, 5, 6]))
        # WARNING: See that the `extend_store_value`` defined here doesn't accomodate for
        # the case where the key is not in the store. It is the user's responsibility to
        # handle that aspect in the `extend_store_value` they provide.
        # For your convenience, the `read_add_write` that is used as a default has
        # (and which **does** handle the non-existing key case by simply writing the value in
        # the store) has an `add_iterables` argument that can be set to whatever
        # makes sense for your use case.
        from functools import partial

        store = {"g": np.array([1, 2, 3])}
        extend_store_value_for_numpy = partial(
            read_add_write, add_iterables=lambda x, y: np.concatenate([x, y])
        )
        g_extender = Extender(
            store, "g", extend_store_value=extend_store_value_for_numpy
        )
        g_extender.extend(np.array([4, 5, 6]))
        assert all(store["g"] == np.array([1, 2, 3, 4, 5, 6]))
    except (ImportError, ModuleNotFoundError):
        pass
