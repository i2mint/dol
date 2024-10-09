"""Test tools.py module."""

from dol.kv_codecs import ValueCodecs
import inspect


def test_kvcodec_user_story_01():
    # See https://github.com/i2mint/dol/discussions/44#discussioncomment-7598805

    # Say you have a source backend that has pickles of some lists-of-lists-of-strings,
    # using the .pkl extension,
    # and you want to copy this data to a target backend, but saving them as gzipped
    # csvs with the csv.gz extension.

    import pickle

    src_backend = {
        "file_1.pkl": pickle.dumps([["A", "B", "C"], ["one", "two", "three"]]),
        "file_2.pkl": pickle.dumps([["apple", "pie"], ["one", "two"], ["hot", "cold"]]),
    }
    targ_backend = dict()

    from dol import ValueCodecs, KeyCodecs, Pipe

    src_wrap = Pipe(KeyCodecs.suffixed(".pkl"), ValueCodecs.pickle())
    targ_wrap = Pipe(
        KeyCodecs.suffixed(".csv.gz"),
        ValueCodecs.csv() + ValueCodecs.str_to_bytes() + ValueCodecs.gzip(),
    )
    src = src_wrap(src_backend)
    targ = targ_wrap(targ_backend)

    targ.update(src)

    # From the point of view of src and targ, you see the same thing:

    assert list(src) == list(targ) == ["file_1", "file_2"]
    assert src["file_1"] == targ["file_1"] == [["A", "B", "C"], ["one", "two", "three"]]

    # But the backend of targ is different:

    src_backend["file_1.pkl"]
    # b'\x80\x04\x95\x19\x00\x00\x00\x00\x00\x00\x00]\x94(]\x94(K\x01K\x02K\x03e]\x94(K\x04K\x05K\x06ee.'
    targ_backend["file_1.csv.gz"]
    # b'\x1f\x8b\x08\x00*YWe\x02\xff3\xd41\xd21\xe6\xe52\xd11\xd51\xe3\xe5\x02\x00)4\x83\x83\x0e\x00\x00\x00'


def _test_codec(codec, obj, encoded=None, decoded=None):
    """Test codec by encoding and decoding obj and comparing to encoded and decoded."""
    if encoded is None:
        # diagnosis mode: Just return the encoded value
        return codec.encoder(obj)
    else:
        if decoded is None:
            decoded = obj
        assert (
            codec.encoder(obj) == encoded
        ), f"Expected {codec.encoder(obj)=} to equal {encoded=}"
        assert (
            codec.decoder(encoded) == decoded
        ), f"Expected {codec.decoder(encoded)=} to equal {decoded=}"


def _test_codec_part(codec, obj, encoded, slice_):
    """Test codec but only testing equality on part of the encoded value.
    This is useful for testing codecs that have a header or footer that is not
    deterministic. For example, gzip has a header that has a timestamp.
    Also, it's useful for when the encoded value is very long and you don't want
    to write it out in the test.
    """
    encoded_actual = codec.encoder(obj)
    assert encoded_actual[slice_] == encoded[slice_]
    assert codec.decoder(encoded_actual) == obj


def test_value_codecs():
    # ----------------- Test codec value wrapper -----------------

    json_codec = ValueCodecs.json()
    # Say you have a backend store with a mapping interface. Say a dict:
    backend = dict()
    # If you call json_codec on this
    interface = json_codec(backend)
    # you'll get json-value a transformation interface
    # That is, when you write a dict (set 'a' to be the dict {'b': 2}):
    interface["a"] = {"b": 2}
    # What's actually written in the backend is the json string:
    assert backend == {"a": '{"b": 2}'}
    # but this json is decoded to a dict when read from interface
    assert interface["a"] == {"b": 2}

    # ----------------- Test encoders and decoders -----------------

    json_codec = ValueCodecs.json()
    # You can get the (encoder, decoder) pair list this:
    encoder, decoder = json_codec
    # Or like this:
    encoder = json_codec.encoder
    decoder = json_codec.decoder
    # See them encode and decode:
    assert encoder({"b": 2}) == '{"b": 2}'
    assert decoder('{"b": 2}') == {"b": 2}

    # And now let's test the many codecs we have:
    # assert sorted(ValueCodecs()) == [
    #     'base64',
    #     'bz2',
    #     'codecs',
    #     'csv',
    #     'csv_dict',
    #     'gzip',
    #     'json',
    #     'lzma',
    #     'pickle',
    #     'tarfile',
    #     'urlsafe_b64',
    #     'xml_etree',
    #     'zipfile',
    # ]

    _test_codec(
        ValueCodecs.pickle(),
        [1, 2, 3],
        b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00]\x94(K\x01K\x02K\x03e.",
    )

    _test_codec(
        ValueCodecs.pickle(protocol=2), [1, 2, 3], b"\x80\x02]q\x00(K\x01K\x02K\x03e."
    )

    assert str(inspect.signature(ValueCodecs.pickle)) == (
        "(obj, data, protocol=None, fix_imports=True, buffer_callback=None, "
        "encoding='ASCII', errors='strict', buffers=())"
    )  # NOTE: May change according to python version. This is 3.8

    _test_codec(
        ValueCodecs.json(),
        {"water": "fire", "earth": "air"},
        '{"water": "fire", "earth": "air"}',
    )

    # test_codec(
    #     ValueCodecs.json(sort_keys=True),
    #     {'water': 'fire', 'earth': 'air'},
    #     '{"earth": "air", "water": "fire"}',  # <-- see how the keys are sorted here!
    #     {'earth': 'air', 'water': 'fire'}
    # )

    _test_codec(ValueCodecs.base64(), b"\xfc\xfd\xfe", b"/P3+")

    _test_codec(ValueCodecs.urlsafe_b64(), b"\xfc\xfd\xfe", b"_P3-")

    _test_codec_part(
        ValueCodecs.gzip(),
        b"hello",
        b"\x1f\x8b\x08\x00t\x85Se\x02\xff\xcbH\xcd\xc9\xc9\x07\x00\x86\xa6\x106\x05\x00\x00\x00",
        slice(10, -8),
    )

    _test_codec(
        ValueCodecs.bz2(),
        b"hello",
        b'BZh91AY&SY\x191e=\x00\x00\x00\x81\x00\x02D\xa0\x00!\x9ah3M\x073\x8b\xb9"\x9c(H\x0c\x98\xb2\x9e\x80',
    )

    _test_codec_part(ValueCodecs.tarfile(), b"hello", b"data.bin", slice(0, 8))

    _test_codec_part(
        ValueCodecs.lzma(),
        b"hello",
        b"\xfd7zXZ",
        slice(0, 4),
    )

    _test_codec_part(
        ValueCodecs.zipfile(),
        b"hello",
        b"PK\x03\x04\x14\x00\x00\x00\x08\x00",
        slice(0, 10),
    )

    _test_codec(ValueCodecs.codecs(), "hello", b"hello")
    # _test_codec(ValueCodecs.plistlib, {'a': 1, 'b': 2}, b'<?xml version="1.0" ...')

    _test_codec(
        ValueCodecs.csv(), [["a", "b", "c"], ["1", "2", "3"]], "a,b,c\r\n1,2,3\r\n"
    )

    _test_codec(
        ValueCodecs.csv(delimiter=";", lineterminator="\n"),
        [["a", "b", "c"], ["1", "2", "3"]],
        "a;b;c\n1;2;3\n",
    )

    _test_codec(
        ValueCodecs.csv_dict(fieldnames=["a", "b"]),
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
        "a,b\r\n1,2\r\n3,4\r\n",
        [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}],
    )

    # See that you don't get back when you started with. The ints aren't ints anymore!
    # You can resolve this by using the fieldcasts argument
    # (that's our argument -- not present in builtin csv module).
    # I should be a function (that transforms a dict to the one you want) or
    # list or tuple of the same size as the row (that specifies the cast function for each field)
    _test_codec(
        ValueCodecs.csv_dict(fieldnames=["a", "b"], fieldcasts=[int] * 2),
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
        "a,b\r\n1,2\r\n3,4\r\n",
    )

    _test_codec(
        ValueCodecs.csv_dict(fieldnames=["a", "b"], fieldcasts={"b": float}),
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
        "a,b\r\n1,2\r\n3,4\r\n",
        [{"a": "1", "b": 2.0}, {"a": "3", "b": 4.0}],
    )

    xml_encoder, xml_decoder = ValueCodecs.xml_etree()
    xml_string = '<html><body><h1 style="color:blue;">Hello, World!</h1></body></html>'
    tree = xml_encoder(xml_string)
    assert tree.tag == "html"
    assert len(tree) == 1
    assert tree[0].tag == "body"
    assert tree[0].attrib == {}
    assert len(tree[0]) == 1
    assert tree[0][0].tag == "h1"
    assert tree[0][0].attrib == {"style": "color:blue;"}

    # Let's change that attribute
    tree[0][0].attrib["style"] = "color:red;"
    # ... and write it back to a string
    new_xml_string = xml_decoder(tree, encoding="unicode")
    assert (
        new_xml_string
        == '<html><body><h1 style="color:red;">Hello, World!</h1></body></html>'
    )
