"""The codec types: an encoder/decoder pair, and the store wrappers built on one.

- ``Codec``: an ``encoder``/``decoder`` pair that unpacks as ``(encoder, decoder)``,
  composes with ``+`` and inverts with ``~``
- ``ValueCodec``, ``KeyCodec``, ``KeyValueCodec``: codecs that, called on a store, wrap
  its values, its keys, or its values with the key as context
- ``affix_key_codec``: a key codec that adds (and strips) a prefix and suffix

Import them from ``dol.kv_codecs``, their public home, which also holds the ready-made
codecs (``ValueCodecs``, ``KeyCodecs``). ``dol.trans`` still re-exports them. They are
defined in this dependency-free module so that ``dol.trans`` can import them without
loading the ready-made codecs (i2mint/dol#70).

    >>> from dol.kv_codecs import ValueCodec
    >>> import json
    >>> s = ValueCodec(json.dumps, json.loads)({})
    >>> s['a'] = [1, 2]
    >>> s.store, s['a']
    ({'a': '[1, 2]'}, [1, 2])
"""

from functools import partial
from typing import TypeVar, Generic
from collections.abc import Callable
from dataclasses import dataclass

from dol.errors import (
    out_of_scope_prefix_error,
    out_of_scope_suffix_error,
    raises_out_of_scope,
)
from dol.util import Pipe


EncodedType = TypeVar("EncodedType")
DecodedType = TypeVar("DecodedType")


# TODO: Want a way to specify Encoded type and Decoded type
@dataclass
class Codec(Generic[DecodedType, EncodedType]):
    """An ``encoder``/``decoder`` pair; iterates as ``(encoder, decoder)`` and composes with ``compose_with``."""

    encoder: Callable[[DecodedType], EncodedType]
    decoder: Callable[[EncodedType], DecodedType]

    def __iter__(self):
        return iter((self.encoder, self.decoder))

    def compose_with(self, other):
        cls = type(self)
        return cls(
            encoder=Pipe(self.encoder, other.encoder),
            decoder=Pipe(other.decoder, self.decoder),
        )

    def invert(self):
        """Return a codec that is the inverse of this one.
        That is, encoder and decoder will be swapped."""
        cls = type(self)
        return cls(encoder=self.decoder, decoder=self.encoder)

    # operators
    __add__ = compose_with
    __invert__ = invert


_CodecT = (Generic[DecodedType, EncodedType], Codec[DecodedType, EncodedType])


class ValueCodec(*_CodecT):
    """A ``Codec`` that, called on a store, wraps its values (``data_of_obj``/``obj_of_data``)."""

    def __call__(self, obj):
        from dol.trans import wrap_kvs  # here, since dol.trans imports this module

        return wrap_kvs(obj, data_of_obj=self.encoder, obj_of_data=self.decoder)


class KeyCodec(*_CodecT):
    """A ``Codec`` that, called on a store, wraps its keys (``id_of_key``/``key_of_id``)."""

    def __call__(self, obj):
        from dol.trans import wrap_kvs  # here, since dol.trans imports this module

        return wrap_kvs(obj, id_of_key=self.encoder, key_of_id=self.decoder)


class KeyValueCodec(*_CodecT):
    """A ``Codec`` that, called on a store, wraps values with key context (``preset``/``postget``)."""

    def __call__(self, obj):
        from dol.trans import wrap_kvs  # here, since dol.trans imports this module

        return wrap_kvs(obj, preset=self.encoder, postget=self.decoder)


# Note: An affix is a morpheme that is attached to a word stem to form a new word or
# word form. Affixes include prefixes, suffixes, infixes, and circumfixes.


def _affix_encoder(string: str, prefix: str = "", suffix: str = ""):
    """Affix a prefix and suffix to a string

    >>> _affix_encoder('name', prefix='/folder/', suffix='.txt')
    '/folder/name.txt'
    """
    return f"{prefix}{string}{suffix}"


def _affix_decoder(string: str, prefix: str = "", suffix: str = ""):
    """Remove prefix and suffix from string

    >>> _affix_decoder('/folder/name.txt', prefix='/folder/', suffix='.txt')
    'name'

    A string that doesn't start with ``prefix`` is outside the codec's scope: slicing
    it would silently yield a key addressing a different object (i2mint/dol#82).

    >>> _affix_decoder('/other/name.txt', prefix='/folder/')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    dol.errors.OutOfScopeKeyError: '/other/name.txt' does not start with the prefix '/folder/'...

    The same goes for the suffix end (i2mint/dol#114): ``foo.json`` isn't a ``.txt``
    key, and stripping 4 characters off it would yield ``foo.``, a different object.

    >>> _affix_decoder('foo.json', suffix='.txt')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    dol.errors.OutOfScopeKeyError: 'foo.json' does not end with the suffix '.txt'...
    """
    if prefix and string[: len(prefix)] != prefix:
        raise out_of_scope_prefix_error(string, prefix)
    if suffix:
        # the length check rejects a string where prefix and suffix would overlap
        if not string.endswith(suffix) or len(string) < len(prefix) + len(suffix):
            raise out_of_scope_suffix_error(string, suffix)
    end_idx = -len(suffix) or None  # if suffix is empty, end_idx should be None
    return string[len(prefix) : end_idx]


def affix_key_codec(prefix: str = "", suffix: str = ""):
    """A factory that creates a key codec that affixes a prefix and suffix to the key

    >>> codec = affix_key_codec(prefix='/folder/', suffix='.txt')
    >>> codec.encoder('name')
    '/folder/name.txt'
    >>> codec.decoder('/folder/name.txt')
    'name'
    """
    decoder = partial(_affix_decoder, prefix=prefix, suffix=suffix)
    if prefix or suffix:
        raises_out_of_scope(decoder)
    return KeyCodec(
        encoder=partial(_affix_encoder, prefix=prefix, suffix=suffix), decoder=decoder
    )
