"""Tests for :mod:`dol.content` — ContentRef + flat content-addressed storage."""

import hashlib

import pytest

from dol.content import (
    ContentRef,
    is_content_ref,
    content_hash,
    guess_mime_type,
    content_url,
    put_content,
    add_content,
    get_content,
    delete_content,
    ContentAddressedStore,
    with_content_addressing,
)


def test_content_ref_wire_roundtrip_matches_zodal_shape():
    ref = ContentRef(
        item_id="abc", field="audio", hash="abc", url="https://x/abc",
        mime_type="audio/mpeg", size=3,
    )
    d = ref.to_json()
    # camelCase keys + _tag discriminator, matching zodal's ContentRef
    assert d == {
        "_tag": "ContentRef", "field": "audio", "itemId": "abc", "hash": "abc",
        "url": "https://x/abc", "mimeType": "audio/mpeg", "size": 3,
    }
    assert ContentRef.from_json(d) == ref


def test_content_ref_drops_empty_fields():
    d = ContentRef("abc").to_json()
    assert d == {"_tag": "ContentRef", "field": "content", "itemId": "abc"}
    assert "hash" not in d and "url" not in d


def test_is_content_ref():
    assert is_content_ref(ContentRef("x"))
    assert is_content_ref({"_tag": "ContentRef", "itemId": "x"})
    assert not is_content_ref({"itemId": "x"})
    assert not is_content_ref("x")
    assert not is_content_ref(None)


def test_content_hash_and_truncation():
    assert content_hash(b"abc") == hashlib.sha256(b"abc").hexdigest()
    assert content_hash(b"abc", length=12) == hashlib.sha256(b"abc").hexdigest()[:12]
    assert content_hash(b"abc", hasher=hashlib.blake2b) == hashlib.blake2b(b"abc").hexdigest()


def test_guess_mime_type():
    assert guess_mime_type("a.mp3") == "audio/mpeg"
    assert guess_mime_type("a.json") == "application/json"
    assert guess_mime_type("noext") is None


def test_put_content_location_addressed():
    s = {}
    ref = put_content(s, "clip1", b"\x00\x01", name="clip1.mp3")
    assert s["clip1"] == b"\x00\x01"
    assert ref.item_id == "clip1" and ref.hash is None  # location-addressed: no hash
    # mimetypes output varies by platform/version — assert the family, not the exact string
    assert ref.mime_type.startswith("audio/") and ref.size == 2
    assert ref.url is None  # never auto-embedded
    assert get_content(s, ref) == b"\x00\x01"


def test_add_content_is_content_addressed_and_idempotent():
    s = {}
    a = add_content(s, b"payload", name="x.bin")
    b = add_content(s, b"payload", name="x.bin")
    assert a == b
    assert a.item_id == a.hash == content_hash(b"payload")
    assert len(s) == 1  # deduplicated
    assert get_content(s, a) == b"payload"


def test_add_content_distinct_payloads_distinct_keys():
    s = {}
    a = add_content(s, b"one")
    b = add_content(s, b"two")
    assert a.item_id != b.item_id and len(s) == 2


def test_get_content_by_ref_dict_or_key():
    s = {}
    ref = add_content(s, b"data")
    assert get_content(s, ref) == b"data"
    assert get_content(s, ref.to_json()) == b"data"
    assert get_content(s, ref.item_id) == b"data"


def test_content_url_resolved_on_demand_not_embedded():
    assert content_url({}, "k1") is None  # plain dict has no url_for

    class Served(dict):
        def url_for(self, key):
            return f"https://cdn/{key}"

    s = Served()
    ref = put_content(s, "k1", b"z")
    assert ref.url is None  # NOT baked into the ref (may be presigned/expiring)
    assert content_url(s, "k1") == "https://cdn/k1"  # resolved on demand
    assert content_url(s, ref) == "https://cdn/k1"  # from a ref too


def test_content_url_prefers_ref_carried_url():
    # A ref that already carries a url wins over the backend (and works with no backend).
    ref = ContentRef("k1", url="https://carried/k1")
    assert content_url({}, ref) == "https://carried/k1"
    assert content_url({}, ref.to_json()) == "https://carried/k1"


def test_delete_content():
    s = {}
    ref = add_content(s, b"gone")
    assert ref.item_id in s
    delete_content(s, ref)
    assert ref.item_id not in s
    delete_content(s, put_content(s, "k2", b"x"))  # by ref again
    assert "k2" not in s


def test_cas_store_mint_read_iter_delete():
    cas = with_content_addressing()
    ref = cas.add(b"hello", name="h.txt")
    key = content_hash(b"hello")
    assert ref.item_id == key
    assert cas[key] == b"hello"
    assert list(cas) == [key] and len(cas) == 1 and key in cas
    del cas[key]
    assert len(cas) == 0


def test_cas_store_rejects_non_hash_key():
    cas = with_content_addressing()
    with pytest.raises(ValueError):
        cas["not-the-hash"] = b"hello"
    # the correct key is accepted
    cas[content_hash(b"hello")] = b"hello"
    assert cas[content_hash(b"hello")] == b"hello"


def test_cas_truncated_length():
    cas = with_content_addressing(length=16)
    assert len(cas.add(b"abc").item_id) == 16


def test_cas_delegates_url_for():
    class Served(dict):
        def url_for(self, key):
            return f"https://cdn/{key}"

    cas = with_content_addressing(Served())
    ref = cas.add(b"payload")
    assert ref.url is None  # on-demand, not embedded
    assert content_url(cas, ref.item_id) == f"https://cdn/{ref.item_id}"
    assert content_url(cas, ref) == f"https://cdn/{ref.item_id}"


def test_backend_injection_dict_vs_class():
    # Same facade code over two backends (dict now; a real dol.Files/s3dol store later).
    backing = {}
    cas = with_content_addressing(backing)
    ref = cas.add(b"shared-bytes")
    assert backing[ref.item_id] == b"shared-bytes"  # writes land in the injected backend


# -------------------------------------------------------------------------------------
# content_url must resolve the key through wrapping layers
#
# It used to do a flat ``getattr(store, 'url_for')(key)``. On a wrapped store that returns
# the method bound to an inner layer, so the backend received the OUTER key and returned a
# URL for a different object than ``store[key]`` reads.


class _Served(dict):
    def url_for(self, key):
        return f"https://cdn/{key}"


def test_content_url_resolves_through_a_key_wrap():
    from dol import KeyCodecs, Pipe, content_url

    # class-wrap: url_for reaches the leaf via a DelegatedAttribute
    wrapped = KeyCodecs.prefixed("a/")(_Served)({"a/k": b"v"})
    assert wrapped["k"] == b"v"
    assert content_url(wrapped, "k") == "https://cdn/a/k"

    # instance-wrap: url_for reaches the leaf via Store.__getattr__
    wrapped = KeyCodecs.prefixed("a/")(_Served({"a/k": b"v"}))
    assert content_url(wrapped, "k") == "https://cdn/a/k"

    # stacked
    stacked = Pipe(KeyCodecs.prefixed("a/"), KeyCodecs.prefixed("b/"))(
        _Served({"a/b/k": b"v"})
    )
    assert content_url(stacked, "k") == "https://cdn/a/b/k"


def test_content_url_unchanged_for_unwrapped_and_keyless_wraps():
    from dol import KeyCodecs, content_url, filt_iter, wrap_kvs

    assert content_url(_Served({"k": b"v"}), "k") == "https://cdn/k"
    assert content_url({}, "k") is None
    # wraps that do not change keys must not change the URL
    assert (
        content_url(wrap_kvs(_Served({"k": b"v"}), obj_of_data=lambda v: v), "k")
        == "https://cdn/k"
    )
    assert (
        content_url(filt_iter(_Served({"k": b"v"}), filt=lambda k: True), "k")
        == "https://cdn/k"
    )


def test_content_url_does_not_double_apply_the_providers_own_transform():
    """A backend whose own ``url_for`` applies ``self._id_of_key`` (e.g. a store that owns
    a prefix) must receive the key in ITS key space, not the fully-resolved one."""
    from dol import KeyCodecs, content_url
    from dol.base import KvReader

    class Prefixed(KvReader):
        def __init__(self, prefix=""):
            self.prefix = prefix

        def _id_of_key(self, k):
            return f"{self.prefix}{k}"

        def _key_of_id(self, i):
            return i[len(self.prefix) :]

        def __iter__(self):
            yield from ()

        def __getitem__(self, k):
            return b"v"

        def url_for(self, k):
            return f"https://s3/{self._id_of_key(k)}"

    assert content_url(Prefixed("logs/"), "f") == "https://s3/logs/f"
    assert content_url(KeyCodecs.prefixed("x/")(Prefixed("logs/")), "f") == (
        "https://s3/logs/x/f"
    )


def test_content_url_terminates_on_pathological_chains():
    """The chain walk must be bounded. A ``MagicMock`` mints a fresh child for every
    ``.store``, and a self-referential store cycles -- both used to hang forever."""
    from unittest.mock import MagicMock

    from dol import content_url

    content_url(MagicMock(), "k")  # must simply return

    class SelfStore:
        @property
        def store(self):
            return self

    assert content_url(SelfStore(), "k") is None


def test_content_url_still_finds_a_dynamically_provided_url_for():
    """``url_for`` supplied via ``__getattr__`` is invisible to a static lookup. Returning
    None there would be a silent wrong answer, so we fall back to plain duck typing."""
    from dol import content_url

    class Dyn(dict):
        def __getattr__(self, name):
            if name == "url_for":
                return lambda key: f"https://dyn/{key}"
            raise AttributeError(name)

    assert content_url(Dyn({"k": 1}), "k") == "https://dyn/k"
