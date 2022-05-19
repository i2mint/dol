"""
Various tools to add functionality to stores
"""

from itertools import islice
from collections.abc import Mapping


class iSliceStore(Mapping):
    """
    Wraps a store to make a reader that acts as if the store was a list
    (with integer keys, and that can be sliced).
    I say "list", but it should be noted that the behavior is more that of range,
    that outputs an element of the list
    when keying with an integer, but returns an iterable object (a range) if sliced.

    Here, a map object is returned when the sliceable store is sliced.

    >>> s = {'foo': 'bar', 'hello': 'world', 'alice': 'bob'}
    >>> sliceable_s = iSliceStore(s)

    The read-only functionalities of the underlying mapping are still available:

    >>> list(sliceable_s)
    ['foo', 'hello', 'alice']
    >>> 'hello' in sliceable_s
    True
    >>> sliceable_s['hello']
    'world'

    But now you can get slices as well:

    >>> list(sliceable_s[0:2])
    ['bar', 'world']
    >>> list(sliceable_s[-2:])
    ['world', 'bob']
    >>> list(sliceable_s[:-1])
    ['bar', 'world']

    Now, you can't do `sliceable_s[1]` because `1` isn't a valid key.
    But if you really wanted "item number 1", you can do:

    >>> next(sliceable_s[1:2])
    'world'

    Note that `sliceable_s[i:j]` is an iterable that needs to be consumed
    (here, with list) to actually get the data. If you want your data in a different
    format, you can use `dol.trans.wrap_kvs` for that.

    >>> from dol import wrap_kvs
    >>> ss = wrap_kvs(sliceable_s, obj_of_data=list)
    >>> ss[1:3]
    ['world', 'bob']
    >>> sss = wrap_kvs(sliceable_s, obj_of_data=sorted)
    >>> sss[1:3]
    ['bob', 'world']
    """

    def __init__(self, store):
        self.store = store

    def _get_islice(self, k: slice):
        start, stop, step = k.start, k.stop, k.step

        assert (step is None) or (step > 0), "step of slice can't be negative"
        negative_start = start is not None and start < 0
        negative_stop = stop is not None and stop < 0
        if negative_start or negative_stop:
            n = self.__len__()
            if negative_start:
                start = n + start
            if negative_stop:
                stop = n + stop

        return islice(self.store.keys(), start, stop, step)

    def __getitem__(self, k):
        if not isinstance(k, slice):
            return self.store[k]
        else:
            return map(self.store.__getitem__, self._get_islice(k))

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __contains__(self, k):
        return k in self.store


from dol import KvReader
from functools import partial
from typing import Any, Iterable, Union, Callable
from itertools import islice

Src = Any
Key = Any
Val = Any

key_error_flag = type('KeyErrorFlag', (), {})()


def _isinstance(obj, class_or_tuple):
    """Same as builtin isinstance, but without the position only limitation
    that prevents from partializing class_or_tuple"""
    return isinstance(obj, class_or_tuple)


def type_check_if_type(filt):
    if isinstance(filt, type) or isinstance(filt, tuple):
        class_or_tuple = filt
        filt = partial(_isinstance, class_or_tuple=class_or_tuple)
    return filt


def return_input(x):
    return x


# TODO: Try dataclass
# TODO: Generalize forest_type to include mappings too
# @dataclass
class Forest(KvReader):
    """Provides a key-value forest interface to objects.

    A `<tree https://en.wikipedia.org/wiki/Tree_(data_structure)>`_
    is a nested data structure. A tree has a root, which is the parent of children,
    who themselves can be parents of further subtrees, or not; in which case they're
    called leafs.
    For more information, see
    `<wikipedia on trees https://en.wikipedia.org/wiki/Tree_(data_structure)>`_

    Here we allow one to construct a tree view of any python object, using a
    key-value interface to the parent-child relationship.

    A forest is a collection of trees.

    Arguably, a dictionnary might not be the most impactful example to show here, since
    it is naturally a tree (therefore a forest), and naturally key-valued: But it has
    the advantage of being easy to demo with.
    Where Forest would really be useful is when you (1) want to give a consistent
    key-value interface to the many various forms that trees and forest objects come
    in, or even more so when (2) your object's tree/forest structure is not obvious,
    so you need to "extract" that view from it (plus give it a consistent key-value
    interface, so that you can build an ecosystem of tools around it.

    Anyway, here's our dictionary example:

    >>> d = {
    ...     'apple': {
    ...         'kind': 'fruit',
    ...         'types': {
    ...             'granny': {'color': 'green'},
    ...             'fuji': {'color': 'red'}
    ...         },
    ...         'tasty': True
    ...     },
    ...     'acrobat': {
    ...         'kind': 'person',
    ...         'nationality': 'french',
    ...         'brave': True,
    ...     },
    ...     'ball': {
    ...         'kind': 'toy'
    ...     }
    ... }

    Must of the time, you'll want to curry ``Forest`` to make an ``object_to_forest``
    constructor for a given class of objects. In the case of dictionaries as the one
    above, this might look like this:

    >>> from functools import partial
    >>> a_forest = partial(
    ...     Forest,
    ...     is_leaf=lambda k, v: not isinstance(v, dict),
    ...     get_node_keys=lambda v: [vv for vv in iter(v) if not vv.startswith('b')],
    ...     get_src_item=lambda src, k: src[k]
    ... )
    >>>
    >>> f = a_forest(d)
    >>> list(f)
    ['apple', 'acrobat']

    Note that we specified in ``get_node_keys``that we didn't want to include items
    whose keys start with ``b`` as valid children. Therefore we don't have our
    ``'ball'`` in the list above.

    Note below which nodes are themselves ``Forests``, and whic are leafs:

    >>> ff = f['apple']
    >>> isinstance(ff, Forest)
    True
    >>> list(ff)
    ['kind', 'types', 'tasty']
    >>> ff['kind']
    'fruit'
    >>> fff = ff['types']
    >>> isinstance(fff, Forest)
    True
    >>> list(fff)
    ['granny', 'fuji']

    """

    def __init__(
        self,
        src: Src,
        *,
        get_node_keys: Callable[[Src], Iterable[Key]],
        get_src_item: Callable[[Src, Key], bool],
        is_leaf: Callable[[Key, Val], bool],
        forest_type: Union[type, Callable] = list,
        leaf_trans: Callable[[Val], Any] = return_input,
    ):
        """Initialize a ``Forest``

        :param src: The source of the ``Forest``. This could be any object you want.
            The following arguments should know how to handle it.
        :param get_node_keys: How to get the keys of the children of ``src``.
        :param get_src_item: How to get the value of a child of ``src`` from its key
        :param is_leaf: Determines if a ``(k, v)`` pair (child) is a leaf.
        :param forest_type: The type of a forest. Used both to determine if an object
            (must be iterable) is to be considered a forest (i.e. an iterable of sources
            that are roots of trees
        :param leaf_trans:
        """
        self.src = src
        self.get_node_keys = get_node_keys
        self.get_src_item = get_src_item
        self.is_leaf = is_leaf
        self.leaf_trans = leaf_trans
        if isinstance(forest_type, type):
            self.is_forest = isinstance(src, forest_type)
        else:
            self.is_forest = forest_type(src)
        self._forest_maker = partial(
            type(self),
            get_node_keys=get_node_keys,
            get_src_item=get_src_item,
            is_leaf=is_leaf,
            forest_type=forest_type,
            leaf_trans=leaf_trans,
        )

    def is_forest_type(self, obj):
        return isinstance(obj, list)

    def __iter__(self):
        if not self.is_forest:
            yield from self.get_node_keys(self.src)
        else:
            for i, _ in enumerate(self.src):
                yield i

    def __getitem__(self, k):
        if self.is_forest:
            assert isinstance(k, int), (
                f'When the src is a forest, you should key with an '
                f'integer. The key was {k}'
            )
            v = next(
                islice(self.src, k, k + 1), key_error_flag
            )  # TODO: raise KeyError if
            if v is key_error_flag:
                raise KeyError(f'No value for {k=}')
        else:
            v = self.get_src_item(self.src, k)
        if self.is_leaf(k, v):
            return self.leaf_trans(v)
        else:
            return self._forest_maker(v)

    def to_dict(self):
        def gen():
            for k, v in self.items():
                if isinstance(v, Forest):
                    yield k, v.to_dict()
                else:
                    yield k, v

        return dict(gen())

    def __repr__(self):
        return f'{type(self).__name__}({self.src})'
