# dol.kv_codecs

Tools to make Key-Value Codecs (encoder-decoder pairs) from standard library tools.

A codec is a store wrapper: `ValueCodecs.json()` encodes values on write and decodes
them on read, `KeyCodecs.suffixed('.json')` adds the suffix on the way in and strips
it on the way out. Codecs compose with `+`.

Main entry points:

- `ValueCodecs`: ready-made value codecs (json, pickle, gzip, csv, str_to_bytes, …)
- `KeyCodecs`: ready-made key codecs (suffixed, prefixed, …)
- `key_based_value_trans`: a value codec chosen from the key
  ```pycon
  >>> from dol.kv_codecs import ValueCodecs, KeyCodecs
  >>> s = ValueCodecs.json()({})
  >>> s['a'] = {'x': 1}
  >>> s.store, s['a']
  ({'a': '{"x": 1}'}, {'x': 1})
  >>> k = KeyCodecs.suffixed('.json')({})
  >>> k['a'] = 1
  >>> k.store, list(k)
  ({'a.json': 1}, ['a'])
  ```

### Functions

| [`add_invertible_key_decoder`](#dol.kv_codecs.add_invertible_key_decoder)(store, \*, decoder)   | Add a key decoder to a store (instance)                                                                 |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| [`codec_wrap`](#dol.kv_codecs.codec_wrap)(cls, encoder, decoder, \*[, exclude]) | Make a `cls` codec factory from an `encoder` and a `decoder`, with the merged signature of both.        |
| [`common_prefix_keys_wrap`](#dol.kv_codecs.common_prefix_keys_wrap)(s)                       | Transforms keys of mapping to omit the longest prefix they have in common                               |
| [`csv_decode`](#dol.kv_codecs.csv_decode)(string[, dialect, delimiter, ...])    | Decode a CSV string into a list of rows (`csv.reader` arguments accepted).                              |
| [`csv_dict_decode`](#dol.kv_codecs.csv_dict_decode)(string, fieldnames[, ...])       | Decode a csv string into a list of dicts.                                                               |
| [`csv_dict_encode`](#dol.kv_codecs.csv_dict_encode)(string, fieldnames[, ...])       | Encode a list of dicts into a csv string.                                                               |
| [`csv_encode`](#dol.kv_codecs.csv_encode)(string[, dialect, delimiter, ...])    | Encode rows (an iterable of iterables) into a CSV string (`csv.writer` arguments accepted).             |
| [`extract_arguments`](#dol.kv_codecs.extract_arguments)(func, args, kwargs)            | Map `args`/`kwargs` to `func`'s parameter names, leniently (partial and excess allowed, kinds ignored). |
| [`key_based_codec_factory`](#dol.kv_codecs.key_based_codec_factory)(key_mapping[, key_func]) | A factory that creates a key codec that uses the key to determine the codec to use.                     |
| [`key_based_value_trans`](#dol.kv_codecs.key_based_value_trans)(key_func, ...[, k])        | A factory that creates a value codec that uses the key to determine the codec to use.                   |

### Classes

| [`CodecCollection`](#dol.kv_codecs.CodecCollection)(\*args, \*\*kwargs)   | The base class for collections of codecs.                                           |
|----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`KeyCodecs`](#dol.kv_codecs.KeyCodecs)(\*args, \*\*kwargs)         | A collection of key codecs                                                          |
| [`KeyValueCodecs`](#dol.kv_codecs.KeyValueCodecs)(\*args, \*\*kwargs)    | A collection of key-value codecs that can be used with postget and preset kv_wraps. |
| [`NotGiven`](#dol.kv_codecs.NotGiven)()                            | A singleton to indicate that a value was not given                                  |
| [`ValueCodecs`](#dol.kv_codecs.ValueCodecs)(\*args, \*\*kwargs)       | A collection of value codec factories using standard lib tools.                     |

### *class* dol.kv_codecs.CodecCollection(\*args, \*\*kwargs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The base class for collections of codecs.
Makes sure that the class cannot be instantiated, but only used as a collection.
Also provides an \_iter_codecs method that iterates over the codec names.

### *class* dol.kv_codecs.KeyCodecs(\*args, \*\*kwargs)

Bases: [`CodecCollection`](#dol.kv_codecs.CodecCollection)

A collection of key codecs

#### mapped_keys(decoder=None)

A factory that creates a key codec that uses “explicit” mappings to encode
and decode keys.

The encoders and decoders can be an explicit mapping of a function.
If the encoder is a mapping, the decoder is the inverse of that mapping.
If given explicitly, this will be asserted.
If not, the decoder will be computed by swapping the keys and values of the
encoder and asserting that no values were lost in the process
(that is, that the mappings are invertible).
The statements above are true if you swap “encoder” and “decoder”.

```pycon
>>> km = KeyCodecs.mapped_keys({'a': 1, 'b': 2})
>>> km.encoder('a')
1
>>> km.decoder(1)
'a'
```

If the encoder is a function, the decoder must be an iterable of keys who will
be used as arguments of the function to get the encoded key, and the decode
will be the inverse of that mapping.
The statement above is true if you swap “encoder” and “decoder”.

```pycon
>>> km = KeyCodecs.mapped_keys(['a', 'b'], str.upper)
>>> km.encoder('A')
'a'
>>> km.decoder('a')
'A'
```

### *class* dol.kv_codecs.KeyValueCodecs(\*args, \*\*kwargs)

Bases: [`CodecCollection`](#dol.kv_codecs.CodecCollection)

A collection of key-value codecs that can be used with postget and preset kv_wraps.

### *class* dol.kv_codecs.NotGiven

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A singleton to indicate that a value was not given

### *class* dol.kv_codecs.ValueCodecs(\*args, \*\*kwargs)

Bases: [`CodecCollection`](#dol.kv_codecs.CodecCollection)

A collection of value codec factories using standard lib tools.

```pycon
>>> json_codec = ValueCodecs.json()  # call the json codec factory
>>> encoder, decoder = json_codec
>>> encoder({'b': 2})
'{"b": 2}'
>>> decoder('{"b": 2}')
{'b': 2}
```

The `json_codec` object is also a `Mapping` value wrapper:

```pycon
>>> backend = dict()
>>> interface = json_codec(backend)
>>> interface['a'] = {'b': 2}  # we write a dict
>>> assert backend == {'a': '{"b": 2}'}  # json was written in backend
>>> interface['a']  # but this json is decoded to a dict when read from interface
{'b': 2}
```

In order not to have to call the codec factory when you just want the default,
we’ve made a `default` attribute that contains all the default codecs:

```pycon
>>> backend = dict()
>>> interface = ValueCodecs.default.json(backend)
>>> interface['a'] = {'b': 2}  # we write a dict
>>> assert backend == {'a': '{"b": 2}'}  # json was written in backend
```

For times when you want to parametrize your code though, know that you can also
pass arguments to the encoder and decoder when you make your codec.
For example, to make a json codec that indents the json, you can do:

```pycon
>>> json_codec = ValueCodecs.json(indent=2)
>>> backend = dict()
>>> interface = json_codec(backend)
>>> interface['a'] = {'b': 2}  # we write a dict
>>> print(backend['a'])  # written in backend with indent
{
  "b": 2
}
```

#### b64 *= <module 'base64' from '/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/base64.py'>*

#### *class* default

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

To contain default codecs. Is populated by @_add_default_codecs

#### io *= <module 'io' (frozen)>*

#### *class* methodcaller(name, , \*args, \*\*kwargs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Return a callable object that calls the given method on its operand.
After f = methodcaller(‘name’), the call f(r) returns r.name().
After g = methodcaller(‘name’, ‘date’, foo=1), the call g(r) returns
r.name(‘date’, foo=1).

#### single_nested_value()

```pycon
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
```

#### tar_compress(file_name='data.bin')

Bytes of an (uncompressed) tar archive holding `data_bytes` as a single file.

```pycon
>>> tar_decompress(tar_compress(b'hello', file_name='x.bin'))
b'hello'
```

#### tar_decompress()

Bytes of the first file found in the tar archive `tar_bytes` (None if none).

#### tuple_of_dict()

Get a tuple-view of dict values.

```pycon
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
```

Note that writes completely replace the values in the backend dict,
it doesn’t update them:

```pycon
>>> d[2]
{'fr': 'DEUX', 'sp': 'DOS'}
```

See also `dol.KeyTemplate` for more general key-based views.

#### zip_compress(filename='some_bytes', , compression=8, allowZip64=True, compresslevel=None, strict_timestamps=True, encoding='utf-8')

Compress input bytes, returning the compressed bytes

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)

```pycon
>>> b = b'x' * 1000 + b'y' * 1000  # 2000 (quite compressible) bytes
>>> len(b)
2000
>>>
>>> zipped_bytes = zip_compress(b)
>>> # Note: Compression details will be system dependent
>>> len(zipped_bytes)
137
>>> unzipped_bytes = zip_decompress(zipped_bytes)
>>> unzipped_bytes == b  # verify that unzipped bytes are the same as the original
True
>>>
>>> from dol.zipfiledol import compression_methods
>>>
>>> zipped_bytes = zip_compress(b, compression=compression_methods['bzip2'])
>>> # Note: Compression details will be system dependent
>>> len(zipped_bytes)
221
>>> unzipped_bytes = zip_decompress(zipped_bytes)
>>> unzipped_bytes == b  # verify that unzipped bytes are the same as the original
True
```

#### zip_decompress(, allowZip64=True, compresslevel=None, strict_timestamps=True)

Decompress input bytes of a single file zip, returning the uncompressed bytes

See `zip_compress` for usage examples.

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)

### dol.kv_codecs.add_invertible_key_decoder(store, , decoder)

Add a key decoder to a store (instance)

### dol.kv_codecs.codec_wrap(cls, encoder, decoder, , exclude=())

Make a `cls` codec factory from an `encoder` and a `decoder`, with the merged signature of both.

### dol.kv_codecs.common_prefix_keys_wrap(s)

Transforms keys of mapping to omit the longest prefix they have in common

### dol.kv_codecs.csv_decode(string, dialect='excel', delimiter=',', quotechar='"', escapechar=None, doublequote=True, skipinitialspace=False, lineterminator='\\\\r\\\\n', quoting=0, strict=False)

Decode a CSV string into a list of rows (`csv.reader` arguments accepted).

### dol.kv_codecs.csv_dict_decode(string, fieldnames, dialect='excel', delimiter=',', quotechar='"', escapechar=None, doublequote=True, skipinitialspace=False, lineterminator='\\\\r\\\\n', quoting=0, strict=False, restkey=None, restval='', extrasaction='raise', fieldcasts=None)

Decode a csv string into a list of dicts.

* **Parameters:**
  * **string** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The csv string to decode
  * **fieldcasts** – A function that takes a row and returns a row with the same keys
    but with values cast to the desired type. If a dict, it should be a mapping
    from fieldnames to cast functions. If an iterable, it should be an iterable of
    cast functions, in which case each cast function will be applied to each element
    of the row, element wise.

```pycon
>>> data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
>>> encoded = csv_dict_encode(data, fieldnames=['a', 'b'])
>>> encoded
'a,b\r\n1,2\r\n3,4\r\n'
>>> csv_dict_decode(encoded)
[{'a': '1', 'b': '2'}, {'a': '3', 'b': '4'}]
```

See that you don’t get back when you started with. The ints aren’t ints anymore!
You can resolve this by using the fieldcasts argument
(that’s our argument – not present in builtin csv module).
I should be a function (that transforms a dict to the one you want) or
list or tuple of the same size as the row (that specifies the cast function for
each field)

```pycon
>>> csv_dict_decode(encoded, fieldnames=['a', 'b'], fieldcasts=[int] * 2)
[{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
>>> csv_dict_decode(encoded, fieldnames=['a', 'b'], fieldcasts={'b': float})
[{'a': '1', 'b': 2.0}, {'a': '3', 'b': 4.0}]
```

### dol.kv_codecs.csv_dict_encode(string, fieldnames, dialect='excel', delimiter=',', quotechar='"', escapechar=None, doublequote=True, skipinitialspace=False, lineterminator='\\\\r\\\\n', quoting=0, strict=False, restkey=None, restval='', extrasaction='raise', fieldcasts=None)

Encode a list of dicts into a csv string.

```pycon
>>> data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
>>> encoded = csv_dict_encode(data, fieldnames=['a', 'b'])
>>> encoded
'a,b\r\n1,2\r\n3,4\r\n'
```

### dol.kv_codecs.csv_encode(string, dialect='excel', delimiter=',', quotechar='"', escapechar=None, doublequote=True, skipinitialspace=False, lineterminator='\\\\r\\\\n', quoting=0, strict=False)

Encode rows (an iterable of iterables) into a CSV string (`csv.writer` arguments accepted).

### dol.kv_codecs.extract_arguments(func, args, kwargs)

Map `args`/`kwargs` to `func`’s parameter names, leniently (partial and excess allowed, kinds ignored).

### dol.kv_codecs.key_based_codec_factory(key_mapping, key_func=<function identity_func>)

A factory that creates a key codec that uses the key to determine the
codec to use.

### dol.kv_codecs.key_based_value_trans(key_func, value_trans_mapping, default_factory, k=<class 'dol.kv_codecs.NotGiven'>)

A factory that creates a value codec that uses the key to determine the
codec to use.

Below, `key_func` gets the extension of a file path:

```pycon
>>> import json
>>> from functools import partial
>>> key_func = lambda k: os.path.splitext(k)[1]
>>> value_trans_mapping = {'.json': json.loads, '.txt': bytes.decode}
>>> default_factory = partial(ValueError, "No codec for this extension")
>>> trans = key_based_value_trans(
...     key_func, value_trans_mapping, default_factory=lambda: identity_func
... )
```

### dol.kv_codecs.key_value_wrap(encoder, decoder, , exclude=())

Make a `cls` codec factory from an `encoder` and a `decoder`, with the merged signature of both.

### dol.kv_codecs.key_wrap(encoder, decoder, , exclude=())

Make a `cls` codec factory from an `encoder` and a `decoder`, with the merged signature of both.

### dol.kv_codecs.value_wrap(encoder, decoder, , exclude=())

Make a `cls` codec factory from an `encoder` and a `decoder`, with the merged signature of both.
