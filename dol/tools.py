"""
Various tools to add functionality to stores
"""

from dol.trans import store_decorator

NoSuchKey = type("NoSuchKey", (), {})

# ------------ useful trans functions to be used with wrap_kvs etc. ---------------------
# TODO: Consider typing or decorating functions to indicate their role (e.g. id_of_key,
#   key_of_id, data_of_obj, obj_of_data, preset, postget...)


_dflt_confirm_overwrite_user_input_msg = (
    "The key {k} already exists and has value {existing_v}. "
    "If you want to overwrite it with {v}, confirm by typing {v} here: "
)


# TODO: Parametrize user messages (bring to interface)
# role: preset
def confirm_overwrite(
    mapping, k, v, user_input_msg=_dflt_confirm_overwrite_user_input_msg
):
    """A preset function you can use in wrap_kvs to ask the user to confirm if
    they're writing a value in a key that already has a different value under it.

    >>> from dol.trans import wrap_kvs
    >>> d = {'a': 'apple', 'b': 'banana'}
    >>> d = wrap_kvs(d, preset=confirm_overwrite)

    Overwriting ``a`` with the same value it already has is fine (not really an
    over-write):

    >>> d['a'] = 'apple'

    Creating new values is also fine:

    >>> d['c'] = 'coconut'
    >>> assert d == {'a': 'apple', 'b': 'banana', 'c': 'coconut'}

    But if we tried to do ``d['a'] = 'alligator'``, we'll get a user input request:

    .. code-block::

        The key a already exists and has value apple.
        If you want to overwrite it with alligator, confirm by typing alligator here:

    And we'll have to type `alligator` and press RETURN to make the write go through.

    """
    if (existing_v := mapping.get(k, NoSuchKey)) is not NoSuchKey and existing_v != v:
        user_input = input(user_input_msg.format(k=k, v=v, existing_v=existing_v))
        if user_input != v:
            print(f"--> User confirmation failed: I won't overwrite {k}")
            # this will have the effect of rewriting the same value that's there already:
            return existing_v
    return v


# -------------------------------- Aggregate a store -----------------------------------

from typing import (
    Callable,
    Optional,
    KT,
    VT,
    Tuple,
    Mapping,
    Union,
    TypeVar,
    Any,
    Iterable,
)
import os
from functools import partial
from pathlib import Path
from dol.trans import wrap_kvs
from dol.filesys import Files


def decode_as_latin1(b: bytes) -> str:
    return b.decode("latin1")


def markdown_section(k: KT, v: VT) -> str:
    return f"## {k}\n\n{v.strip()}\n\n"


def save_string_to_filepath(filepath: str, string: str):
    filepath = Path(filepath).expanduser().absolute()
    filepath.write_text(string)
    return string


def identity(x):
    return x


Latin1TextFiles = wrap_kvs(Files, value_decoder=decode_as_latin1)

Item = TypeVar("Item")
Aggregate = TypeVar("Aggregate")


def store_aggregate(
    content_store: Union[Mapping[KT, VT], str],  # Path to the folder or dol store
    *,
    kv_to_item: Callable[
        [KT, VT], Item
    ] = markdown_section,  # Function to convert key-value pairs to text
    aggregator: Callable[
        [Iterable[Item]], Aggregate
    ] = "\n\n".join,  # How to aggregate the item's into an aggregate
    egress: Union[
        Callable[[Aggregate], Any], str
    ] = identity,  # function to apply to the aggregate before returning
    key_filter: Optional[Callable[[KT], bool]] = None,  # Filter function for keys
    value_filter: Optional[Callable[[VT], bool]] = None,  # Filter function for values
    kv_filter: Optional[
        Callable[[Tuple[KT, VT]], bool]
    ] = None,  # Filter function for key-value pairs
    local_store_factory: Callable[
        [str], Mapping[KT, VT]
    ] = Latin1TextFiles,  # Factory function for the local store
) -> Any:
    r'''
    Create an aggregate object of a store's (a Mapping of strings) content

    The function is written to be able to aggregate the keys and/or values of a store,
    no matter their type, and concatenate them into an object of arbitrary type.
    That said, the defaults are setup assuming the store's keys and values are text,
    and you want to concatenate them into a single string.
    This is useful, for example, when you have several files in a folder,
    and you want to create a single text/markdown file with all the content therein.

    This function filters content from a given content store, converts the key-value
    pairs to items (usually text), and (if you specify a filepath as the `egress`)
    saves the aggregate (text) before returning it.

    Args:
        content_store (Union[Mapping[KT, VT], str]): Path to the folder or dol store to read from.
        kv_to_item (Callable[[KT, VT], Item]):
            Function to convert key-value pairs to an Item (usually a string).
        aggregator (Callable[[Iterable[Item]], Aggregate]):
            The function that will aggregate the items that `kv_to_item` produces.
            Defaults to '\n\n'.join.
        egress (Union[Callable[[Aggregate], Any], str]):
            The function that will be called on the aggregate before returning it.
            Defaults to identity.
            Note that if you provide a string, the function will save the aggregate
            text to a file, assuming it is indeed text.
        key_filter (Optional[Callable[[KT], bool]]):
            Optional filter for keys. Defaults to None (no filtering).
        value_filter (Optional[Callable[[VT], bool]]):
            Optional filter for values. Defaults to None (no filtering).
        kv_filter (Optional[Callable[[Tuple[KT, VT]], bool]]):
            Optional filter for key-value pairs. Defaults to None (no filtering).
        local_store_factory (Callable[[str], Mapping[KT, VT]]): Factory function for the local store,
            used only if `content_store` is an existing folder path. Defaults to Latin1TextFiles.

    Returns:
        Any: Usually the aggregate object, which is usually the concatenated text.

    Normally, you'd specify your content store by specifying a root folder
    (the function will create a Mapping-view of the contents of the folder for you),
    or make a content store yourself (a Mapping object providing the key-value pairs).

    To provide a small example, we'll take a dict as our content store:

    >>> content_store = {
    ...     'file1.py': '"""Module docstring."""',
    ...     'file2.py': 'def foo(): pass',
    ...     'file3.py': '"""Another docstring."""',
    ...     'file4.md': 'Markdown content here.',
    ...     'file5.py': '"""If I mention file5.py, I will be excluded."""',
    ... }

    Define the filters:

    >>> key_filter = lambda k: k.endswith('.py')  # Only include keys that end with '.py'
    >>> value_filter = lambda v: v.startswith(
    ...     '"""'
    ... )  # Only include values that start with """ (marking a module docstring)
    >>> kv_filter = (
    ...     lambda kv: kv[0] not in kv[1]
    ... )  # Exclude key-value pairs where the value mentions the key

    Call the function with the provided filters and settings

    >>> result = store_aggregate(
    ...     content_store=content_store,  # The content_store dict
    ...     kv_to_item="{} -> {}".format,  # Format key-value pairs as "key -> value"
    ...     key_filter=key_filter,  # Key filter: Include only .py files
    ...     value_filter=value_filter,  # Value filter: Include only values starting with """
    ...     kv_filter=kv_filter,  # KV filter: Exclude if value contains the key
    ...     aggregator=', '.join,
    ...     egress='~/test.md'
    ... )
    >>> result
    'file1.py -> """Module docstring.""", file3.py -> """Another docstring."""'

    Here, you got the string as the result. If you want to save it to a file,
    you can provide the save_filepath argument, and it will save the text to the file,
    and return the save_filepath to you (which )

    Recipe: You can do a lot with the `kv_to_text` argument. For example, if your
    content store doesn't have string keys or values, you can always extract whatever
    information you need from them to produce the text that will represent that item.
    '''
    # Convert content_store to a dol store if it's a directory path
    if isinstance(content_store, str) and os.path.isdir(content_store):
        content_store = local_store_factory(content_store)

    if isinstance(egress, str):
        save_filepath = egress
        if save_filepath.startswith("~"):
            save_filepath = os.path.expanduser(save_filepath)
        # make an egress that will save the string to a file (then return the string)
        egress = partial(save_string_to_filepath, save_filepath)

    # Define default filters if not provided
    key_filter = key_filter or (lambda key: True)
    value_filter = value_filter or (lambda value: True)
    kv_filter = kv_filter or (lambda kv: True)

    def actual_kv_filter(kv):
        k, v = kv
        return key_filter(k) and value_filter(v) and kv_filter(kv)

    # Create the string by applying filters and kv_to_text conversion
    filtered_kv_pairs = filter(actual_kv_filter, content_store.items())
    aggregate = aggregator(kv_to_item(k, v) for k, v in filtered_kv_pairs)

    return egress(aggregate)


# --------------------------------------- Misc ------------------------------------------


from functools import partial
from typing import Any, Iterable, Union, Callable
from itertools import islice

from dol.base import KvReader
from dol.base import Store  # For the ask_user_for_value_when_missing function

_dflt_ask_user_for_value_when_missing_msg = (
    "No such key was found. You can enter a value for it here "
    "or simply hit enter to leave the slot empty"
)


def convert_to_numerical_if_possible(s: str):
    """To be used with ``ask_user_for_value_when_missing`` ``value_preprocessor`` arg

    >>> convert_to_numerical_if_possible("123")
    123
    >>> convert_to_numerical_if_possible("123.4")
    123.4
    >>> convert_to_numerical_if_possible("one")
    'one'

    Border case: The strings "infinity" and "inf" actually convert to a valid float.

    >>> convert_to_numerical_if_possible("infinity")
    inf
    """
    try:
        s = int(s)
    except ValueError:
        try:
            s = float(s)
        except ValueError:
            pass
    return s


@store_decorator
def ask_user_for_value_when_missing(
    store=None,
    *,
    value_preprocessor: Optional[Callable] = None,
    on_missing_msg: str = _dflt_ask_user_for_value_when_missing_msg,
):
    """Wrap a store so if a value is missing when the user asks for it, they will be
    given a chance to enter the value they want to write.

    :param store: The store (instance or class) to wrap
    :param value_preprocessor: Function to transform the user value before trying to
        write it (bearing in mind all user specified values are strings)
    :param on_missing_msg: String that will be displayed to prompt the user to enter a
        value
    :return:
    """

    store = Store.wrap(store)

    def __missing__(self, k):
        user_value = input(on_missing_msg + f" Value for {k}:\n")

        if user_value:
            if value_preprocessor:
                user_value = value_preprocessor(user_value)
            self[k] = user_value
        else:
            super(type(self), self).__missing__(k)

    store.__missing__ = __missing__
    return store


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


Src = Any
Key = Any
Val = Any

key_error_flag = type("KeyErrorFlag", (), {})()


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
                f"When the src is a forest, you should key with an "
                f"integer. The key was {k}"
            )
            v = next(
                islice(self.src, k, k + 1), key_error_flag
            )  # TODO: raise KeyError if
            if v is key_error_flag:
                raise KeyError(f"No value for {k=}")
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
        return f"{type(self).__name__}({self.src})"
