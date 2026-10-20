"""Compare signatures, and check whether calls to one function are valid calls to another.

- ``compare_signatures``, ``param_comparator``, ``keyed_comparator``: comparison tools
- ``param_differences_dict``, ``SigPair``: describe how two signatures differ
- ``is_call_compatible_with``: is every valid call to one signature a valid call to the
  other? (``Sig.is_call_compatible_with`` calls it)

It used to be part of ``dol.signatures``, which still re-exports all of it
(i2mint/dol#70).

    >>> from dol.sig_compare import is_call_compatible_with
    >>> from dol.signatures import Sig
    >>> is_call_compatible_with(Sig('(a, b)'), Sig('(a, b, c=3)'))
    True
    >>> is_call_compatible_with(Sig('(a, b, c=3)'), Sig('(a, b)'))
    False
"""

from inspect import Parameter, Signature
from typing import Any, TypeVar
from collections.abc import Callable, Iterable
from functools import partial
from operator import eq

from dol.signatures import Sig, empty, wraps


########################################################################################
# Signature Comparison and Compatibility #
########################################################################################

Compared = TypeVar("Compared")
Comparison = TypeVar("Comparison")
Comparator = Callable[[Compared, Compared], Comparison]
Comparison.__doc__ = (
    "The return type of a Comparator. Typically a bool, or int, but can be anything."
    'In that sense it is more of a "collation" than I comparison'
)

# TODO: Make function that makes Comparator types according for different kinds of
#  compared types? (e.g. for comparing signatures, for comparing parameters, ...)
#  See HasAttr in https://github.com/i2mint/i2/blob/feb469acdc0bc8268877b400b9af6dda56de6292/i2/itypes.py#L164
#  for inspiration.
SignatureComparator = Callable[[Signature, Signature], Comparison]
ParamComparator = Callable[[Parameter, Parameter], Comparison]
CallableComparator = Callable[[Callable, Callable], Comparison]

ComparisonAggreg = Callable[[Iterable[Comparison]], Any]

CT = TypeVar("CT")  # some other Compared type (used to define KeyFunction
KeyFunction = Callable[[CT], Compared]
# KeyFunction.__doc__ = "Function that transforms one compared type to another"


def compare_signatures(func1, func2, signature_comparator: SignatureComparator = eq):
    return signature_comparator(Sig(func1), Sig(func2))


# TODO: Look into typing: Why does lint complain about this line of code?
def mk_func_comparator_based_on_signature_comparator(
    signature_comparator: SignatureComparator,
) -> CallableComparator:
    return partial(compare_signatures, signature_comparator=signature_comparator)


def _keyed_comparator(
    comparator: Comparator,
    key: KeyFunction,
    x: CT,
    y: CT,
) -> Comparison:
    """Apply a comparator after transforming inputs through a key function.

    >>> from operator import eq
    >>> parity = lambda x: x % 2
    >>> _keyed_comparator(eq, parity, 1, 3)
    True
    >>> _keyed_comparator(eq, parity, 1, 4)
    False
    """
    return comparator(key(x), key(y))


def keyed_comparator(
    comparator: Comparator,
    key: KeyFunction,
) -> Comparator:
    """Create a key-function enabled binary operator.

    In various places in python functionality is extended by allowing a key function.
    For example, the ``sorted`` function allows a key function to be passed, which is
    applied to each element before sorting. The keyed_comparator function allows a
    comparator to be extended in the same way. The returned comparator will apply the
    key function toeach input before applying the original comparator.

    >>> from operator import eq
    >>> parity = lambda x: x % 2
    >>> comparator = keyed_comparator(eq, parity)
    >>> list(map(comparator, [1, 1, 2, 2], [3, 4, 5, 6]))
    [True, False, False, True]
    """
    return partial(_keyed_comparator, comparator, key)


# For back-compatibility:
_key_function_enabled_operator = _keyed_comparator
_key_function_factory = keyed_comparator


# TODO: Show examples of how this can be used to produce precise error messages.
#  The way to do this is to have the attribute binary functions produce some info dicts
#  that can then be aggregated in aggreg to produce a final error message (or even
#  a final error object, which can even be raised) if there is indeed a mismatch at all.
#  Further more, we might want to make a function that will take a parametrized
#  param_binary_func and produce such a error raising function from it, using the
#  specific functions (extracted by Sig) to produce the error message.
def param_comparator(
    param1: Parameter,
    param2: Parameter,
    *,
    name: Comparator = eq,
    kind: Comparator = eq,
    default: Comparator = eq,
    annotation: Comparator = eq,
    aggreg: ComparisonAggreg = all,
) -> Comparison:
    """Compare two parameters.

    Note that by default, this function is strict, and will return False if
    any of the parameters are not equal. This is because the default
    aggregation function is `all` and the default comparison functions of the
    parameter's attributes are `eq` (meaning equality, not identity).

    But you can change that by passing different comparison functions and/or
    aggregation functions.

    In fact, the real purpose of this function is to be used as a factory of parameter
    binary functions, through parametrizing it with `functools.partial`.

    The parameter binary functions themselves are meant to be used to make signature
    binary functions.

    :param param1: first parameter
    :param param2: second parameter
    :param name: function to compare names
    :param kind: function to compare kinds
    :param default: function to compare defaults
    :param annotation: function to compare annotations
    :param aggreg: function to aggregate results

    >>> from inspect import Parameter
    >>> param1 = Parameter('x', Parameter.POSITIONAL_OR_KEYWORD)
    >>> param2 = Parameter('x', Parameter.POSITIONAL_OR_KEYWORD)
    >>> param_binary_func(param1, param2)
    True

    See https://github.com/i2mint/i2/issues/50#issuecomment-1381686812 for discussion.
    """
    return aggreg(
        (
            name(param1.name, param2.name),
            kind(param1.kind, param2.kind),
            default(param1.default, param2.default),
            annotation(param1.annotation, param2.annotation),
        )
    )


param_comparator: ParamComparator
param_binary_func = param_comparator  # back compatibility alias


def dflt1_is_empty_or_dflt2_is_not(dflt1, dflt2):
    """
    Why such a strange default comparison function?

    This is to be used as a default in is_call_compatible_with.

    Consider two functions func1 and func2 with a parameter p with default values
    dflt1 and dflt2 respectively.
    If dflt1 was not empty and dflt2 was, this would mean that func1 could be called
    without specifying p, but func2 couldn't.

    So to avoid this situation, we use dflt1_is_empty_or_dflt2_is_not as the default
    """
    return dflt1 is empty or dflt2 is not empty


# TODO: Implement annotation compatibility
def ignore_any_differences(x, y):
    return True


permissive_param_comparator = partial(
    param_comparator,
    name=ignore_any_differences,
    kind=ignore_any_differences,
    default=ignore_any_differences,
    annotation=ignore_any_differences,
)
permissive_param_comparator.__doc__ = """
Permissive version of param_comparator that ignores any differences of parameter 
attributes.

It is meant to be used with partial, but with a permissive base, contrary to the 
base param_comparator which requires strict equality (`eq`) for all attributes.
"""

dflt1_is_empty_or_dflt2_is_not_param_comparator = partial(
    permissive_param_comparator, default=dflt1_is_empty_or_dflt2_is_not
)


def return_tuple(x, y):
    return x, y


param_attribute_dict: ComparisonAggreg


def param_attribute_dict(name_kind_default_annotation: Iterable[Comparison]) -> dict:
    keys = ["name", "kind", "default", "annotation"]
    return {key: value for key, value in zip(keys, name_kind_default_annotation)}


param_comparison_dict = partial(
    param_comparator,
    name=return_tuple,
    kind=return_tuple,
    default=return_tuple,
    annotation=return_tuple,
    aggreg=param_attribute_dict,
)

param_comparison_dict.__doc__ = """
A ParamComparator that returns a dictionary with pairs parameter attributes.

>>> param1 = Sig('(a: int = 1)')['a']
>>> param2 = Sig('(a: str = 2)')['a']
>>> param_comparison_dict(param1, param2)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
{'name': ('a', 'a'), 'kind': ..., 'default': (1, 2), 'annotation': (<class 'int'>, <class 'str'>)}
"""


def param_differences_dict(
    param1: Parameter,
    param2: Parameter,
    *,
    name: Comparator = eq,
    kind: Comparator = eq,
    default: Comparator = eq,
    annotation: Comparator = eq,
):
    """Makes a dictionary exibiting the differences between two parameters.

    >>> param1 = Sig('(a: int = 1)')['a']
    >>> param2 = Sig('(a: str = 2)')['a']
    >>> param_differences_dict(param1, param2)
    {'default': (1, 2), 'annotation': (<class 'int'>, <class 'str'>)}
    >>> param_differences_dict(param1, param2, default=lambda x, y: isinstance(x, type(y)))
    {'annotation': (<class 'int'>, <class 'str'>)}
    """
    equality_vector = param_comparator(
        param1,
        param2,
        name=name,
        kind=kind,
        default=default,
        annotation=annotation,
        aggreg=tuple,
    )
    comparison_dict = param_comparison_dict(param1, param2)
    return {
        key: comparison_dict[key]
        for key, equal in zip(comparison_dict, equality_vector)
        if not equal
    }


def defaults_are_the_same_when_not_empty(dflt1, dflt2):
    """
    Check if two defaults are the same when they are not empty.

    >>> defaults_are_the_same_when_not_empty(1, 1)
    True
    >>> defaults_are_the_same_when_not_empty(1, 2)
    False
    >>> defaults_are_the_same_when_not_empty(1, None)
    False
    >>> defaults_are_the_same_when_not_empty(1, Parameter.empty)
    True
    """
    return dflt1 is empty or dflt2 is empty or dflt1 == dflt2


def postprocess(egress: Callable):
    """A decorator that will process the output of the wrapped function with egress"""

    # Note: Vendorized version equivalent ones in i2.deco and i2.wrapper
    def postprocessed(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            original_output = func(*args, **kwargs)
            return egress(original_output)

        return wrapped_func

    return postprocessed


# TODO: It seems like param_comparator is really only used to compare parameters on defaults.
#   This may be due to the fact that is_call_compatible_with was developed independently
#   from the other general param_comparator functionality that was developed (see above)
#   The code of is_call_compatible_with should be reviwed and refactored to use general
#   tools.
@postprocess(
    all
)  # see "Use of postprocess" in https://github.com/i2mint/i2/discussions/63#discussioncomment-10394910
def is_call_compatible_with(
    sig1: Sig,
    sig2: Sig,
    *,
    param_comparator: ParamComparator | None = None,
) -> bool:
    """Return True if ``sig1`` is compatible with ``sig2``. Meaning that all valid ways
    to call ``sig1`` are valid for ``sig2``.

    :param sig1: The main signature.
    :param sig2: The signature to be compared with.
    :param param_comparator: The function used to compare two parameters

    >>> is_call_compatible_with(
    ...     Sig('(a, /, b, *, c)'),
    ...     Sig('(a, b, c)')
    ... )
    True
    >>> is_call_compatible_with(
    ...     Sig('()'),
    ...     Sig('(a)')
    ... )
    False
    >>> is_call_compatible_with(
    ...     Sig('()'),
    ...     Sig('(a=0)')
    ... )
    True
    >>> is_call_compatible_with(
    ...     Sig('(a, /, *, c)'),
    ...     Sig('(a, /, b, *, c)')
    ... )
    False
    >>> is_call_compatible_with(
    ...     Sig('(a, /, *, c)'),
    ...     Sig('(a, /, b=0, *, c)')
    ... )
    True
    >>> is_call_compatible_with(
    ...     Sig('(a, /, b)'),
    ...     Sig('(a, /, b, *, c)')
    ... )
    False
    >>> is_call_compatible_with(
    ...     Sig('(a, /, b)'),
    ...     Sig('(a, /, b, *, c=0)')
    ... )
    True
    >>> is_call_compatible_with(
    ...     Sig('(a, /, b, *, c)'),
    ...     Sig('(*args, **kwargs)')
    ... )
    True
    """

    # Note: In case you're tempted to put this default function as an argument default,
    #  don't. Yes, it's preferable in many ways, but makes the "one source of truth"
    #  principle harder to maintain, since this default has to be the same anywhere
    #  the current function is called. Better signature/docs injection functionality
    #  would be warranted.
    #  See https://stackoverflow.com/questions/78874506/how-can-i-avoid-interface-repetition-in-python-function-signatures-and-docstring
    param_comparator = (
        param_comparator or dflt1_is_empty_or_dflt2_is_not_param_comparator
    )

    def validate_variadics():
        return (
            # sig1 can only have a VP if sig2 also has one
            (vp1 is None or vp2 is not None)
            and
            # sig1 can only have a VK if sig2 also has one
            (vk1 is None or vk2 is not None)
        )

    def validate_param_counts():
        # sig1 cannot have more positional params than sig2
        if len(ps1) > len(ps2) and not vp2:
            return False
        # sig1 cannot have keyword params that do not exist in sig2
        if len([n for n in ks1 if n not in ks2]) > 0 and not vk2:
            return False
        return True

    def validate_extra_params():
        # Any extra PO in sig2 must have a default value
        if len(pos1) < len(pos2) and not all(
            sig2.parameters[n].default is not empty for n in pos2[len(pos1) :]
        ):
            return False
        # Any extra PK in sig2 must have its corresponding PO or KO in sig1, or a
        # default value
        for i, n in enumerate(pks2):
            if (
                n not in pks1
                and len(pos1) <= len(pos2) + i
                and n not in kos1
                and sig2.parameters[n].default is empty
            ):
                return False
        # Any extra KO in sig2 must have a default value
        for n in kos2:
            if n not in kos1 and sig2.parameters[n].default == empty:
                return False
        return True

    def validate_param_positions():
        for i, n2 in enumerate(ps2):
            for j, n1 in enumerate(ks1):
                if n1 == n2:
                    if (
                        # It can be a PK in sig1 and a P (PO or PK) in sig2 only if
                        # its position in sig2 is >= to its position in sig1
                        (n1 in pks1 and i < len(pos1) + j)
                        or (
                            n1 in kos1
                            and (
                                # Cannot be a KO in sig1 and a PO in sig2
                                n2 in pos2
                                or
                                # It can be a KO in sig1 and a PK in sig2 only if its
                                # position in sig2 is > than the total number of POs
                                # and PKs in sig1
                                i < len(ps1)
                            )
                        )
                    ):
                        return False
        return True

    def validate_param_compatibility():
        # Every positional param in sig1 must be compatible with its
        # correspondant param in sig2 (at the same index).
        for i in range(len(ps1)):
            if i < len(ps2) and not param_comparator(sig1.params[i], sig2.params[i]):
                return False
        # Every keyword param in sig1 must be compatible with its
        # correspondant param in sig2 (with the same name).
        for n in ks1:
            if n in ks2 and not param_comparator(
                sig1.parameters[n], sig2.parameters[n]
            ):
                return False
        return True

    pos1, pks1, vp1, kos1, vk1 = sig1.detail_names_by_kind()
    ps1 = pos1 + pks1
    ks1 = pks1 + kos1
    pos2, pks2, vp2, kos2, vk2 = sig2.detail_names_by_kind()
    ps2 = pos2 + pks2
    ks2 = pks2 + kos2

    if vp1:
        sig1 -= vp1
    if vk1:
        sig1 -= vk1
    if vp2:
        sig2 -= vp2
    if vk2:
        sig2 -= vk2

    return (
        f()
        for f in [
            validate_variadics,
            validate_param_counts,
            validate_extra_params,
            validate_param_positions,
            validate_param_compatibility,
        ]
    )


from dataclasses import dataclass

from functools import cached_property
from dataclasses import dataclass
from inspect import Parameter


@dataclass
class SigPair:
    """
    Class that operates on a pair of signatures.

    For example, offers methods to compare two signatures in various ways.

    :param sig1: First signature or signature-able object.
    :param sig2: Second signature or signature-able object.

    >>> from pprint import pprint
    >>> def three(a, b: int, c=3): ...
    >>> def little(a, *, b=2, d=4) -> int: ...
    >>> def pigs(a, b) -> int: ...
    >>> sig_pair = SigPair(three, little)
    >>>
    >>> sig_pair.shared_names
    ['a', 'b']
    >>> sig_pair.names_missing_in_sig1
    ['d']
    >>> sig_pair.names_missing_in_sig2
    ['c']
    >>> sig_pair.param_comparison()
    False
    >>> pprint(sig_pair.diff())  # doctest: +NORMALIZE_WHITESPACE
    {'names_missing_in_sig1': ['d'],
    'names_missing_in_sig2': ['c'],
    'param_differences': {'b': {'annotation': (<class 'int'>,
                                                <class 'inspect._empty'>),
                                'default': (<class 'inspect._empty'>, 2),
                                'kind': (<_ParameterKind.POSITIONAL_OR_KEYWORD: 1>,
                                        <_ParameterKind.KEYWORD_ONLY: 3>)}},
    'return_annotation': (<class 'inspect._empty'>, <class 'int'>)}

    Call compatibility says that any arguments leading to a valid call to a function
    having the first signature, will also lead to a valid call to a function having the
    second signature. This is not the case for the signatures of `three` and `little`:

    >>> sig_pair.are_call_compatible()
    False

    But we don't need to have equal signatures to have call compatibility. For example,

    >>> SigPair(three, lambda a, b=2, c=30: None).are_call_compatible()
    True

    Note that call-compatibility is not symmetric. For example, `pigs` is call
    compatible with `three`, since any arguments that are valid for `pigs` are valid
    for `three`:

    >>> SigPair(pigs, three).are_call_compatible()
    True

    But `three` is not call-compatible with `pigs` since `three` requires could include
    a `c` argument, which `pigs` would choke on.

    >>> SigPair(three, pigs).are_call_compatible()
    False
    """

    sig1: Callable | Sig
    sig2: Callable | Sig

    def __post_init__(self):
        self.sig1 = Sig(self.sig1)
        self.sig2 = Sig(self.sig2)

    @cached_property
    def shared_names(self):
        """
        List of names that are common to both signatures, in the order of sig1.

        >>> sig1 = Sig(lambda a, b, c: None)
        >>> sig2 = Sig(lambda b, c, d: None)
        >>> comp = SigPair(sig1, sig2)
        >>> comp.shared_names
        ['b', 'c']
        """
        return [name for name in self.sig1.names if name in self.sig2.names]

    @cached_property
    def names_missing_in_sig2(self):
        """
        List of names that are in the sig1 signature but not in sig2.

        >>> sig1 = Sig(lambda a, b, c: None)
        >>> sig2 = Sig(lambda b, c, d: None)
        >>> comp = SigPair(sig1, sig2)
        >>> comp.names_missing_in_sig2
        ['a']
        """
        return [name for name in self.sig1.names if name not in self.sig2.names]

    @cached_property
    def names_missing_in_sig1(self):
        """
        List of names that are in the sig2 signature but not in sig1.

        >>> sig1 = Sig(lambda a, b, c: None)
        >>> sig2 = Sig(lambda b, c, d: None)
        >>> comp = SigPair(sig1, sig2)
        >>> comp.names_missing_in_sig1
        ['d']
        """
        return [name for name in self.sig2.names if name not in self.sig1.names]

    # TODO: Verify that the doctests are correct!
    def are_call_compatible(self, param_comparator=None) -> bool:
        """
        Check if the signatures are call-compatible.

        Returns True if sig1 can be used to call sig2 or vice versa.

        >>> sig1 = Sig(lambda a, b, c=3: None)
        >>> sig2 = Sig(lambda a, b: None)
        >>> comp = SigPair(sig1, sig2)
        >>> comp.are_call_compatible()
        False

        >>> comp = SigPair(sig2, sig1)
        >>> comp.are_call_compatible()
        True
        """
        return is_call_compatible_with(
            self.sig1, self.sig2, param_comparator=param_comparator
        )

    def param_comparison(self, comparator=param_comparator, aggregation=all) -> bool:
        """
        Compare parameters between the two signatures using the provided comparator function.

        :param comparator: A function to compare two parameters.
        :param aggregation: A function to aggregate the results of the comparisons.
        :return: Boolean result of the aggregated comparisons.

        >>> sig1 = Sig('(a, b: int, c=3)')
        >>> sig2 = Sig('(a, *, b=2, d=4)')
        >>> comp = SigPair(sig1, sig2)
        >>> comp.param_comparison()
        False
        """
        results = [
            comparator(self.sig1.parameters[name], self.sig2.parameters[name])
            for name in self.shared_names
        ]
        return aggregation(results)

    def param_differences(self) -> dict:
        """
        Get a dictionary of parameter differences between the two signatures.

        :return: A dict containing differences for each shared param that has any.

        >>> sig1 = Sig('(a, b: int, c=3)')
        >>> sig2 = Sig('(a, *, b=2, d=4)')
        >>> comp = SigPair(sig1, sig2)
        >>> result = comp.param_differences()
        >>> expected = {
        ...     'b': {
        ...         'kind': (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY),
        ...         'default': (Parameter.empty, 2),
        ...         'annotation': (int, Parameter.empty),
        ...     }
        ... }
        >>> result == expected
        True
        """

        def diff_pairs():
            for name in self.shared_names:
                diff_dict = param_differences_dict(
                    self.sig1.parameters[name], self.sig2.parameters[name]
                )
                if diff_dict:
                    yield name, diff_dict

        return dict(diff_pairs())

    def diff(self) -> dict:
        """
        Get a dictionary of differences between the two signatures.

        >>> from pprint import pprint
        >>> def three(a, b: int, c=3): ...
        >>> def little(a, *, b=2, d=4) -> int: ...
        >>> def pigs(a, b: int = 2) -> int: ...
        >>> pprint(SigPair(three, little).diff())  # doctest: +NORMALIZE_WHITESPACE
        {'names_missing_in_sig1': ['d'],
        'names_missing_in_sig2': ['c'],
        'param_differences': {'b': {'annotation': (<class 'int'>,
                                                    <class 'inspect._empty'>),
                                    'default': (<class 'inspect._empty'>, 2),
                                    'kind': (<_ParameterKind.POSITIONAL_OR_KEYWORD: 1>,
                                            <_ParameterKind.KEYWORD_ONLY: 3>)}},
        'return_annotation': (<class 'inspect._empty'>, <class 'int'>)}
        >>> pprint(SigPair(three, pigs).diff())  # doctest: +NORMALIZE_WHITESPACE
        {'names_missing_in_sig2': ['c'],
        'param_differences': {'b': {'default': (<class 'inspect._empty'>, 2)}},
        'return_annotation': (<class 'inspect._empty'>, <class 'int'>)}
        >>> pprint(SigPair(three, three).diff())
        {}
        """
        d = {
            key: value
            for key, value in {
                "names_missing_in_sig1": self.names_missing_in_sig1,
                "names_missing_in_sig2": self.names_missing_in_sig2,
                "param_differences": self.param_differences(),
            }.items()
            if value
        }
        # add the return_annotation difference, if any
        if self.sig1.return_annotation != self.sig2.return_annotation:
            d["return_annotation"] = (
                self.sig1.return_annotation,
                self.sig2.return_annotation,
            )
        return d

    def diff_str(self) -> str:
        """
        Get a string representation of the differences between the two signatures.
        """
        from pprint import pformat

        return pformat(self.diff())
