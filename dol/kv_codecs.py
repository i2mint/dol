"""
Tools to make Key-Value Codecs (encoder-decoder pairs) from standard library tools.
"""
# ------------------------------------ Codecs ------------------------------------------

from functools import partial
from typing import Callable, Iterable, Any, Optional

from dol.trans import Codec, ValueCodec, KeyCodec
from dol.signatures import Sig

# For the codecs:
import csv
import io


@Sig
def _string(string: str):
    ...


@Sig
def _csv_rw_sig(
    dialect: str = 'excel',
    *,
    delimiter: str = ',',
    quotechar: Optional[str] = '"',
    escapechar: Optional[str] = None,
    doublequote: bool = True,
    skipinitialspace: bool = False,
    lineterminator: str = '\r\n',
    quoting=0,
    strict: bool = False,
):
    ...


@Sig
def _csv_dict_extra_sig(
    fieldnames, restkey=None, restval='', extrasaction='raise', fieldcasts=None
):
    ...


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
    _ = kwargs.pop('fieldcasts', None)  # this one is for decoder only
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
    fieldcasts = kwargs.pop('fieldcasts', lambda row: row)
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
    return Sig(func).kwargs_from_args_and_kwargs(
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
    sig = _merge_signatures(encoder, decoder, exclude=exclude)
    return sig(factory)


value_wrap = partial(codec_wrap, ValueCodec)
value_wrap.__name__ = 'value_wrap'
key_wrap = partial(codec_wrap, KeyCodec)
key_wrap.__name__ = 'key_wrap'


class ValueCodecs:
    r"""
    A collection of value codecs using standard lib tools.

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

    def __init__(self, *args, **kwargs):
        raise ValueError(
            'This class is not meant to be instantiated, but only act as a collection '
            'of value codec functions'
        )

    # TODO: Clean up module import polution?
    # TODO: Import all these in module instead of class
    # TODO: Figure out a way to import these dynamically, only if a particular codec is used
    # TODO: Figure out how to give codecs annotations that can actually be inspected!

    @classmethod
    def _iter_codecs(cls):
        def is_value_codec(attr_val):
            func = getattr(attr_val, 'func', None)
            name = getattr(func, '__name__', '')
            return name == '_codec_wrap'

        for attr in dir(cls):
            if not attr.startswith('_'):
                attr_val = getattr(cls, attr, None)
                if is_value_codec(attr_val):
                    yield attr

    class default:
        """To contain default codecs"""

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
        io.StringIO, methodcaller('read')
    )
    bytesio: ValueCodec[bytes, io.BytesIO] = value_wrap(
        io.BytesIO, methodcaller('read')
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
        lzma.compress, lzma.decompress, exclude=('format',)
    )

    import quopri, plistlib

    quopri: ValueCodec[bytes, bytes] = value_wrap(
        quopri.encodestring, quopri.decodestring
    )
    plistlib: ValueCodec[bytes, bytes] = value_wrap(
        plistlib.dumps, plistlib.loads, exclude=('fmt',)
    )

    # Any is really xml.etree.ElementTree.Element, but didn't want to import
    xml_etree: Codec[Any, bytes] = value_wrap(_xml_tree_encode, _xml_tree_decode)


def _add_default_codecs():
    for codec_name in ValueCodecs._iter_codecs():
        codec_factory = getattr(ValueCodecs, codec_name)
        dflt_codec = codec_factory()
        setattr(ValueCodecs.default, codec_name, dflt_codec)


_add_default_codecs()


from dol.paths import KeyTemplate


class KeyCodecs:
    """
    A collection of key codecs
    """

    def __init__(self, *args, **kwargs):
        raise ValueError(
            'This class is not meant to be instantiated, but only act as a collection '
            'of key codec functions'
        )

    def suffixed(suffix: str):
        st = KeyTemplate('{}' + f'{suffix}')
        return KeyCodec(st.simple_str_to_str, st.str_to_simple_str)

    # def suffixed(suffix: str, field_type: FieldTypeNames = 'simple_str'):
    #     st = StringTemplate('{}' + f"{suffix}")
    #     return KeyCodec(st.
