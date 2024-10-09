"""
Tools to make Key-Value Codecs (encoder-decoder pairs) from standard library tools.
"""

# ------------------------------------ Codecs ------------------------------------------

from functools import partial
from typing import Callable, Iterable, Any, Optional, KT, VT, Mapping, Union, Dict
from operator import itemgetter

from dol.trans import (
    Codec,
    ValueCodec,
    KeyCodec,
    KeyValueCodec,
    affix_key_codec,
    store_decorator,
)
from dol.paths import KeyTemplate
from dol.signatures import Sig
from dol.util import named_partial, identity_func, single_nest_in_dict, nest_in_dict

# For the codecs:
import csv
import io


@Sig
def _string(string: str): ...


@Sig
def _csv_rw_sig(
    dialect: str = "excel",
    *,
    delimiter: str = ",",
    quotechar: Optional[str] = '"',
    escapechar: Optional[str] = None,
    doublequote: bool = True,
    skipinitialspace: bool = False,
    lineterminator: str = "\r\n",
    quoting=0,
    strict: bool = False,
): ...


@Sig
def _csv_dict_extra_sig(
    fieldnames, restkey=None, restval="", extrasaction="raise", fieldcasts=None
): ...


__csv_rw_sig = _string + _csv_rw_sig
__csv_dict_sig = _string + _csv_rw_sig + _csv_dict_extra_sig


# Note: @(_string + _csv_rw_sig) made (ax)black choke
@__csv_rw_sig
def csv_encode(string, *args, **kwargs):
    with io.StringIO() as buffer:
        writer = csv.writer(buffer, *args, **kwargs)
        writer.writerows(string)
        return buffer.getvalue()


@__csv_rw_sig
def csv_decode(string, *args, **kwargs):
    with io.StringIO(string) as buffer:
        reader = csv.reader(buffer, *args, **kwargs)
        return list(reader)


@__csv_dict_sig
def csv_dict_encode(string, *args, **kwargs):
    r"""Encode a list of dicts into a csv string.

    >>> data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    >>> encoded = csv_dict_encode(data, fieldnames=['a', 'b'])
    >>> encoded
    'a,b\r\n1,2\r\n3,4\r\n'

    """
    _ = kwargs.pop("fieldcasts", None)  # this one is for decoder only
    with io.StringIO() as buffer:
        writer = csv.DictWriter(buffer, *args, **kwargs)
        writer.writeheader()
        writer.writerows(string)
        return buffer.getvalue()


@__csv_dict_sig
def csv_dict_decode(string, *args, **kwargs):
    r"""Decode a csv string into a list of dicts.

    :param string: The csv string to decode
    :param fieldcasts: A function that takes a row and returns a row with the same keys
        but with values cast to the desired type. If a dict, it should be a mapping
        from fieldnames to cast functions. If an iterable, it should be an iterable of
        cast functions, in which case each cast function will be applied to each element
        of the row, element wise.

    >>> data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    >>> encoded = csv_dict_encode(data, fieldnames=['a', 'b'])
    >>> encoded
    'a,b\r\n1,2\r\n3,4\r\n'
    >>> csv_dict_decode(encoded)
    [{'a': '1', 'b': '2'}, {'a': '3', 'b': '4'}]


    See that you don't get back when you started with. The ints aren't ints anymore!
    You can resolve this by using the fieldcasts argument
    (that's our argument -- not present in builtin csv module).
    I should be a function (that transforms a dict to the one you want) or
    list or tuple of the same size as the row (that specifies the cast function for
    each field)


    >>> csv_dict_decode(encoded, fieldnames=['a', 'b'], fieldcasts=[int] * 2)
    [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    >>> csv_dict_decode(encoded, fieldnames=['a', 'b'], fieldcasts={'b': float})
    [{'a': '1', 'b': 2.0}, {'a': '3', 'b': 4.0}]

    """
    fieldcasts = kwargs.pop("fieldcasts", lambda row: row)
    if isinstance(fieldcasts, Iterable):
        if isinstance(fieldcasts, dict):
            cast_dict = dict(fieldcasts)
            cast = lambda k: cast_dict.get(k, lambda x: x)
            fieldcasts = lambda row: {k: cast(k)(v) for k, v in row.items()}
        else:
            _casts = list(fieldcasts)
            # apply each cast function to each element of the row, element wise
            fieldcasts = lambda row: {
                k: cast(v) for cast, (k, v) in zip(_casts, row.items())
            }
    with io.StringIO(string) as buffer:
        reader = csv.DictReader(buffer, *args, **kwargs)
        rows = [row for row in reader]

        def remove_first_row_if_only_header(rows):
            first_row = next(iter(rows), None)
            if first_row is not None and all(k == v for k, v in first_row.items()):
                rows.pop(0)

        remove_first_row_if_only_header(rows)
        return list(map(fieldcasts, rows))


def _xml_tree_encode(element, parser=None):
    # Needed to replace original "text" argument with "element" to be consistent with
    # ET.tostring
    import xml.etree.ElementTree as ET

    return ET.fromstring(text=element, parser=parser)


def _xml_tree_decode(
    element,
    encoding=None,
    method=None,
    *,
    xml_declaration=None,
    default_namespace=None,
    short_empty_elements=True,
):
    import xml.etree.ElementTree as ET

    return ET.tostring(
        element,
        encoding,
        method,
        xml_declaration=xml_declaration,
        default_namespace=default_namespace,
        short_empty_elements=short_empty_elements,
    )


def extract_arguments(func, args, kwargs):
    return Sig(func).map_arguments(
        args, kwargs, allow_partial=True, allow_excess=True, ignore_kind=True
    )


def _var_kinds_less_signature(func):
    sig = Sig(func)
    var_kinds = (
        sig.names_of_kind[Sig.VAR_POSITIONAL] + sig.names_of_kind[Sig.VAR_KEYWORD]
    )
    return sig - var_kinds


def _merge_signatures(encoder, decoder, *, exclude=()):
    return (_var_kinds_less_signature(encoder) - exclude) + (
        _var_kinds_less_signature(decoder) - exclude
    )


def _codec_wrap(cls, encoder: Callable, decoder: Callable, **kwargs):
    return cls(
        encoder=partial(encoder, **extract_arguments(encoder, (), kwargs)),
        decoder=partial(decoder, **extract_arguments(decoder, (), kwargs)),
    )


def codec_wrap(cls, encoder: Callable, decoder: Callable, *, exclude=()):
    _cls_codec_wrap = partial(_codec_wrap, cls)
    factory = partial(_cls_codec_wrap, encoder, decoder)
    # TODO: Review this signature here. Should be keyword-only to match what
    #  _codec_wrap implementation imposses, or _codec_wrap should be made to accpt
    #  positional arguments (when encoder/decoder function are not class methods)
    # See: https://github.com/i2mint/dol/discussions/41#discussioncomment-8015800
    sig = _merge_signatures(encoder, decoder, exclude=exclude)
    # Change all arguments to keyword-only
    # sig = sig.ch_kinds(**{k: Sig.KEYWORD_ONLY for k in sig.names})
    return sig(factory)


# wrappers to manage encoder and decoder arguments and signature
value_wrap = named_partial(codec_wrap, ValueCodec, __name__="value_wrap")
key_wrap = named_partial(codec_wrap, KeyCodec, __name__="key_wrap")
key_value_wrap = named_partial(codec_wrap, KeyValueCodec, __name__="key_value_wrap")


class CodecCollection:
    """The base class for collections of codecs.
    Makes sure that the class cannot be instantiated, but only used as a collection.
    Also provides an _iter_codecs method that iterates over the codec names.
    """

    def __init__(self, *args, **kwargs):
        name = getattr(type(self), "__name__", "")
        raise ValueError(
            f"The {name} class is not meant to be instantiated, "
            "but only act as a collection of codec factories"
        )

    @classmethod
    def _iter_codecs(cls):
        def is_value_codec(attr_val):
            func = getattr(attr_val, "func", None)
            name = getattr(func, "__name__", "")
            return name == "_codec_wrap"

        for attr in dir(cls):
            if not attr.startswith("_"):
                attr_val = getattr(cls, attr, None)
                if is_value_codec(attr_val):
                    yield attr


def _add_default_codecs(cls):
    for codec_name in cls._iter_codecs():
        codec_factory = getattr(cls, codec_name)
        dflt_codec = codec_factory()
        setattr(cls.default, codec_name, dflt_codec)
    return cls


@_add_default_codecs
class ValueCodecs(CodecCollection):
    r"""
    A collection of value codec factories using standard lib tools.

    >>> json_codec = ValueCodecs.json()  # call the json codec factory
    >>> encoder, decoder = json_codec
    >>> encoder({'b': 2})
    '{"b": 2}'
    >>> decoder('{"b": 2}')
    {'b': 2}

    The `json_codec` object is also a `Mapping` value wrapper:

    >>> backend = dict()
    >>> interface = json_codec(backend)
    >>> interface['a'] = {'b': 2}  # we write a dict
    >>> assert backend == {'a': '{"b": 2}'}  # json was written in backend
    >>> interface['a']  # but this json is decoded to a dict when read from interface
    {'b': 2}

    In order not to have to call the codec factory when you just want the default,
    we've made a `default` attribute that contains all the default codecs:

    >>> backend = dict()
    >>> interface = ValueCodecs.default.json(backend)
    >>> interface['a'] = {'b': 2}  # we write a dict
    >>> assert backend == {'a': '{"b": 2}'}  # json was written in backend

    For times when you want to parametrize your code though, know that you can also
    pass arguments to the encoder and decoder when you make your codec.
    For example, to make a json codec that indents the json, you can do:

    >>> json_codec = ValueCodecs.json(indent=2)
    >>> backend = dict()
    >>> interface = json_codec(backend)
    >>> interface['a'] = {'b': 2}  # we write a dict
    >>> print(backend['a'])  # written in backend with indent
    {
      "b": 2
    }


    """

    # TODO: Clean up module import polution?
    # TODO: Import all these in module instead of class
    # TODO: Figure out a way to import these dynamically, only if a particular codec is used
    # TODO: Figure out how to give codecs annotations that can actually be inspected!

    class default:
        """To contain default codecs. Is populated by @_add_default_codecs"""

    import pickle, json, gzip, bz2, base64 as b64, lzma, codecs, io
    from operator import methodcaller
    from dol.zipfiledol import (
        zip_compress,
        zip_decompress,
        tar_compress,
        tar_decompress,
    )

    str_to_bytes: ValueCodec[bytes, bytes] = value_wrap(str.encode, bytes.decode)
    stringio: ValueCodec[str, io.StringIO] = value_wrap(
        io.StringIO, methodcaller("read")
    )
    bytesio: ValueCodec[bytes, io.BytesIO] = value_wrap(
        io.BytesIO, methodcaller("read")
    )

    pickle: ValueCodec[Any, bytes] = value_wrap(pickle.dumps, pickle.loads)
    json: ValueCodec[dict, str] = value_wrap(json.dumps, json.loads)
    csv: ValueCodec[list, str] = value_wrap(csv_encode, csv_decode)
    csv_dict: ValueCodec[list, str] = value_wrap(csv_dict_encode, csv_dict_decode)

    base64: ValueCodec[bytes, bytes] = value_wrap(b64.b64encode, b64.b64decode)
    urlsafe_b64: ValueCodec[bytes, bytes] = value_wrap(
        b64.urlsafe_b64encode, b64.urlsafe_b64decode
    )
    codecs: ValueCodec[str, bytes] = value_wrap(codecs.encode, codecs.decode)

    # Note: Note clear if escaping or unescaping is the encoder or decoder here
    # I have never had the need for stores using it, so will omit for now
    # html: ValueCodec[str, str] = value_wrap(html.unescape, html.escape)

    # Compression
    zipfile: ValueCodec[bytes, bytes] = value_wrap(zip_compress, zip_decompress)
    gzip: ValueCodec[bytes, bytes] = value_wrap(gzip.compress, gzip.decompress)
    bz2: ValueCodec[bytes, bytes] = value_wrap(bz2.compress, bz2.decompress)
    tarfile: ValueCodec[bytes, bytes] = value_wrap(tar_compress, tar_decompress)
    lzma: ValueCodec[bytes, bytes] = value_wrap(
        lzma.compress, lzma.decompress, exclude=("format",)
    )

    import quopri, plistlib

    quopri: ValueCodec[bytes, bytes] = value_wrap(
        quopri.encodestring, quopri.decodestring
    )
    plistlib: ValueCodec[bytes, bytes] = value_wrap(
        plistlib.dumps, plistlib.loads, exclude=("fmt",)
    )

    # Any is really xml.etree.ElementTree.Element, but didn't want to import
    xml_etree: Codec[Any, bytes] = value_wrap(_xml_tree_encode, _xml_tree_decode)

    # TODO: Review value_wrap so it works with non-class functions like below
    #   See: https://github.com/i2mint/dol/discussions/41#discussioncomment-8015800

    single_nested_value: Codec[KT, Dict[KT, VT]]

    def single_nested_value(key):
        """

        >>> d = {
        ...     1: {'en': 'one', 'fr': 'un', 'sp': 'uno'},
        ...     2: {'en': 'two', 'fr': 'deux', 'sp': 'dos'},
        ... }
        >>> en = ValueCodecs.single_nested_value('en')(d)
        >>> en[1]
        'one'
        >>> en[1] = 'ONE'
        >>> d[1]  # note that here d[1] is completely replaced (not updated)
        {'en': 'ONE'}
        """
        return ValueCodec(partial(single_nest_in_dict, key), itemgetter(key))

    tuple_of_dict: Codec[Iterable[VT], Dict[KT, VT]]

    def tuple_of_dict(keys):
        """Get a tuple-view of dict values.

        >>> d = {
        ...     1: {'en': 'one', 'fr': 'un', 'sp': 'uno'},
        ...     2: {'en': 'two', 'fr': 'deux', 'sp': 'dos'},
        ... }
        >>> codec = ValueCodecs.tuple_of_dict(['fr', 'sp'])
        >>> codec.encoder(['deux', 'tre'])
        {'fr': 'deux', 'sp': 'tre'}
        >>> codec.decoder({'en': 'one', 'fr': 'un', 'sp': 'uno'})
        ('un', 'uno')
        >>> frsp = codec(d)
        >>> frsp[2]
        ('deux', 'dos')
        >>> ('deux', 'dos')
        ('deux', 'dos')
        >>> frsp[2] = ('DEUX', 'DOS')
        >>> frsp[2]
        ('DEUX', 'DOS')

        Note that writes completely replace the values in the backend dict,
        it doesn't update them:

        >>> d[2]
        {'fr': 'DEUX', 'sp': 'DOS'}

        See also `dol.KeyTemplate` for more general key-based views.
        """
        return ValueCodec(partial(nest_in_dict, keys), itemgetter(*keys))


from dol.util import invertible_maps


@_add_default_codecs
class KeyCodecs(CodecCollection):
    """
    A collection of key codecs
    """

    def affixed(prefix: str = "", suffix: str = ""):
        return affix_key_codec(prefix=prefix, suffix=suffix)

    def suffixed(suffix: str):
        return affix_key_codec(suffix=suffix)

    def prefixed(prefix: str):
        return affix_key_codec(prefix=prefix)

    def common_prefixed(keys: Iterable[str]):
        from dol.util import max_common_prefix

        prefix = max_common_prefix(keys)
        return KeyCodecs.prefixed(prefix)

    def mapped_keys(
        encoder: Optional[Union[Mapping, Callable]] = None,
        decoder: Optional[Union[Mapping, Callable]] = None,
    ):
        """
        A factory that creates a key codec that uses "explicit" mappings to encode
        and decode keys.

        The encoders and decoders can be an explicit mapping of a function.
        If the encoder is a mapping, the decoder is the inverse of that mapping.
        If given explicitly, this will be asserted.
        If not, the decoder will be computed by swapping the keys and values of the
        encoder and asserting that no values were lost in the process
        (that is, that the mappings are invertible).
        The statements above are true if you swap "encoder" and "decoder".

        >>> km = KeyCodecs.mapped_keys({'a': 1, 'b': 2})
        >>> km.encoder('a')
        1
        >>> km.decoder(1)
        'a'

        If the encoder is a function, the decoder must be an iterable of keys who will
        be used as arguments of the function to get the encoded key, and the decode
        will be the inverse of that mapping.
        The statement above is true if you swap "encoder" and "decoder".

        >>> km = KeyCodecs.mapped_keys(['a', 'b'], str.upper)
        >>> km.encoder('A')
        'a'
        >>> km.decoder('a')
        'A'
        """
        encoder, decoder = invertible_maps(encoder, decoder)
        return KeyCodec(
            encoder=encoder.__getitem__,
            decoder=decoder.__getitem__,
        )


def common_prefix_keys_wrap(s: Mapping):
    """Transforms keys of mapping to omit the longest prefix they have in common"""
    common_prefix_wrap = KeyCodecs.common_prefixed(s)
    return common_prefix_wrap(s)


# TODO: Here, I'd like to decorate with store_decorator, but KeyCodecs.mapped_keys
#  doesn't apply to a Mapping class, only an instance. We have to make the iteration
#  of keys to be inverted have lazy capabilities (only happen when the instance is made)
def add_invertible_key_decoder(store: Mapping, *, decoder: Callable):
    """Add a key decoder to a store (instance)"""
    return KeyCodecs.mapped_keys(store, decoder=decoder)(store)


# --------------------------------- KV Codecs ------------------------------------------


dflt_ext_mapping = {
    ".json": ValueCodecs.json,
    ".csv": ValueCodecs.csv,
    ".csv_dict": ValueCodecs.csv_dict,
    ".pickle": ValueCodecs.pickle,
    ".gz": ValueCodecs.gzip,
    ".bz2": ValueCodecs.bz2,
    ".lzma": ValueCodecs.lzma,
    ".zip": ValueCodecs.zipfile,
    ".tar": ValueCodecs.tarfile,
    ".xml": ValueCodecs.xml_etree,
}


def key_based_codec_factory(key_mapping: dict, key_func: Callable = identity_func):
    """A factory that creates a key codec that uses the key to determine the
    codec to use."""

    def encoder(key):
        return key_mapping[key_func(key)]

    def decoder(key):
        return key_mapping[key_func(key)]

    return ValueCodec(encoder, decoder)


class NotGiven:
    """A singleton to indicate that a value was not given"""

    def __repr__(self):
        return "NotGiven"


from typing import NewType


def key_based_value_trans(
    key_func: Callable[[KT], KT],
    value_trans_mapping,
    default_factory: Callable[[], Callable],
    k=NotGiven,
):
    """A factory that creates a value codec that uses the key to determine the
    codec to use.

    # a key_func that gets the extension of a file path

    >>> import json
    >>> from functools import partial
    >>> key_func = lambda k: os.path.splitext(k)[1]
    >>> value_trans_mapping = {'.json': json.loads, '.txt': bytes.decode}
    >>> default_factory = partial(ValueError, "No codec for this extension")
    >>> trans = key_based_value_trans(
    ...     key_func, value_trans_mapping, default_factory=lambda: identity_func
    ... )


    """
    if k is NotGiven:
        return partial(
            key_based_value_trans, key_func, value_trans_mapping, default_factory
        )
    value_trans_key = key_func(k)
    value_trans = value_trans_mapping.get(value_trans_key, default_factory, None)
    if value_trans is None:
        value_trans = default_factory()
    return value_trans


@_add_default_codecs
class KeyValueCodecs(CodecCollection):
    """
    A collection of key-value codecs that can be used with postget and preset kv_wraps.
    """

    def key_based(
        key_mapping: dict,
        key_func: Callable = identity_func,
        *,
        default: Optional[Callable] = None,
    ):
        """A factory that creates a key-value codec that uses the key to determine the
        value codec to use."""

    def extension_based(
        ext_mapping: dict = dflt_ext_mapping,
        *,
        default: Optional[Callable] = None,
    ):
        """A factory that creates a key-value codec that uses the file extension to
        determine the value codec to use."""
