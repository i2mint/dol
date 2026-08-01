"""Content references and content-addressed storage — the flat "blob" layer.

Many apps split their data into two concerns (the *content-metadata bifurcation*;
see ``misc/docs/dol_content_metadata_bifurcation.md``):

- **records / metadata** — small, queryable rows (a ``MutableMapping`` of dicts, a DB);
- **content / blobs** — large bytes (media, documents, renders) that you don't want
  to inline into a record or a query result.

This module is the **content half**: a *flat* bytes store plus a small, serializable
:class:`ContentRef` token that stands in for the bytes inside a record. The store
itself is just a ``MutableMapping[str, bytes]`` — so the backend is **injected**
(``dict`` in tests, :class:`dol.Files` locally, an ``s3dol`` store in the cloud) and
nothing here depends on any of them.

Two addressing modes, mirroring the same convention used by the ``zodal`` TypeScript
stores (so a :class:`ContentRef` serialized here matches ``zodal``'s ``ContentRef`` on
the wire — see :meth:`ContentRef.to_json`):

- **location-addressed** (:func:`put_content`) — the caller supplies the id;
- **content-addressed** (:func:`add_content` / :func:`with_content_addressing`) — the
  id *is* the content hash, which makes writes idempotent and deduplicated (CAS).

**URLs are resolved on demand, never baked in.** ``put_content``/``add_content`` leave
``ContentRef.url`` empty; call :func:`content_url` when you actually need a fetchable
URL. This is deliberate: a backend's ``url_for`` may mint a *presigned, expiring* URL
(e.g. S3), and a :class:`ContentRef` is meant to be *persisted* inside a record — so
freezing an expiring URL into it would be a latent bug. Reads can thus redirect to a
CDN / presigned URL / static route while writes always go to the injected backend.

>>> store = {}
>>> ref = add_content(store, b'hello world', name='greeting.txt')
>>> ref.item_id == content_hash(b'hello world')
True
>>> (ref.size, ref.mime_type, ref.url)
(11, 'text/plain', None)
>>> get_content(store, ref)
b'hello world'
>>> is_content_ref(ref) and is_content_ref(ref.to_json())
True

The wire form is camelCase and drops empty fields, matching ``zodal``'s ``ContentRef``:

>>> ref.to_json()['itemId'] == ref.item_id
True
>>> sorted(ref.to_json())
['_tag', 'field', 'hash', 'itemId', 'mimeType', 'size']
"""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Optional, Protocol, runtime_checkable

from dol.base import KvPersister

#: The ``_tag`` discriminator value carried on the JSON wire form (cross-language parity).
CONTENT_REF_TAG = "ContentRef"

#: A key-minting hash constructor, e.g. ``hashlib.sha256`` — ``bytes -> hash object``.
HashFunc = Callable[[bytes], Any]


@dataclass(frozen=True)
class ContentRef:
    """A small, serializable stand-in for stored content (bytes).

    Held *inside* a record in place of the bytes, so lists/queries stay light. It is
    addressed by ``(item_id, field)`` (a record may have several content fields);
    ``hash`` is populated for content-addressed writes and left ``None`` otherwise.
    ``url`` is an optional directly-fetchable location — normally left empty and
    resolved on demand via :func:`content_url` (see the module docstring).
    """

    item_id: str
    field: str = "content"
    hash: Optional[str] = None
    url: Optional[str] = None
    mime_type: Optional[str] = None
    size: Optional[int] = None

    def to_json(self) -> dict:
        """camelCase wire form matching ``zodal``'s ``ContentRef`` (empty fields dropped)."""
        d = {"_tag": CONTENT_REF_TAG, "field": self.field, "itemId": self.item_id}
        if self.hash is not None:
            d["hash"] = self.hash
        if self.url is not None:
            d["url"] = self.url
        if self.mime_type is not None:
            d["mimeType"] = self.mime_type
        if self.size is not None:
            d["size"] = self.size
        return d

    @classmethod
    def from_json(cls, d: Mapping) -> "ContentRef":
        """Parse a wire-form dict (camelCase) back into a :class:`ContentRef`."""
        return cls(
            item_id=d["itemId"],
            field=d.get("field", "content"),
            hash=d.get("hash"),
            url=d.get("url"),
            mime_type=d.get("mimeType"),
            size=d.get("size"),
        )


def is_content_ref(obj: Any) -> bool:
    """True for a :class:`ContentRef` instance or its wire-form dict (``_tag`` discriminator).

    >>> is_content_ref(ContentRef('id1'))
    True
    >>> is_content_ref({'_tag': 'ContentRef', 'itemId': 'id1'})
    True
    >>> is_content_ref({'itemId': 'id1'}) or is_content_ref('id1')
    False
    """
    if isinstance(obj, ContentRef):
        return True
    return isinstance(obj, Mapping) and obj.get("_tag") == CONTENT_REF_TAG


def content_hash(
    data: bytes, *, hasher: HashFunc = hashlib.sha256, length: Optional[int] = None
) -> str:
    """Hex content hash of ``data`` (sha256 by default), optionally truncated to ``length``.

    Truncation trades key length for a higher collision probability (a 16-hex-char
    prefix is 64 bits) — leave ``length`` unset unless keys must be short and the
    corpus is small.

    >>> content_hash(b'abc') == content_hash(b'abc')
    True
    >>> len(content_hash(b'abc', length=16))
    16
    """
    h = hasher(data).hexdigest()
    return h[:length] if length else h


def guess_mime_type(name: str) -> Optional[str]:
    """Guess a mime type from a filename/key by extension (stdlib ``mimetypes``).

    Results depend on the platform's mime registry, so treat them as best-effort.

    >>> guess_mime_type('a.json')
    'application/json'
    >>> guess_mime_type('no-extension') is None
    True
    """
    return mimetypes.guess_type(name)[0]


@runtime_checkable
class SupportsUrlFor(Protocol):
    """A backend that can hand out a directly-fetchable URL for a stored key.

    The seam that lets **reads** redirect to a CDN / presigned URL / static route while
    **writes** stay on the injected backend. Local file stores typically don't implement
    it (:func:`content_url` then returns ``None``); an ``s3dol`` store implements it with
    a presigned URL — so all S3 knowledge lives in ``s3dol``, never here.
    """

    def url_for(self, key: str) -> Optional[str]: ...


def _key_of(ref_or_key: Any) -> str:
    """The storage key from a :class:`ContentRef`, its wire dict, or a bare key string."""
    if isinstance(ref_or_key, ContentRef):
        return ref_or_key.item_id
    if isinstance(ref_or_key, Mapping) and ref_or_key.get("_tag") == CONTENT_REF_TAG:
        return ref_or_key["itemId"]
    return ref_or_key


def _url_of(ref_or_key: Any) -> Optional[str]:
    """A URL already carried by a ref (if any), else ``None``."""
    if isinstance(ref_or_key, ContentRef):
        return ref_or_key.url
    if isinstance(ref_or_key, Mapping) and ref_or_key.get("_tag") == CONTENT_REF_TAG:
        return ref_or_key.get("url")
    return None


def content_url(store: Any, ref_or_key: Any) -> Optional[str]:
    """A fetchable URL for content, resolved **on demand**.

    Prefers a URL the ref already carries; otherwise asks the backend's ``url_for``
    (the :class:`SupportsUrlFor` seam), returning ``None`` if it has none.

    >>> class Served(dict):
    ...     def url_for(self, key): return f'https://cdn.example/{key}'
    >>> content_url(Served(), 'k1')
    'https://cdn.example/k1'
    >>> content_url({}, 'k1') is None
    True
    >>> content_url({}, ContentRef('k1', url='https://carried/k1'))  # ref carries its own
    'https://carried/k1'
    """
    carried = _url_of(ref_or_key)
    if carried:
        return carried
    url_for = getattr(store, "url_for", None)
    return url_for(_key_of(ref_or_key)) if callable(url_for) else None


def _ref(
    item_id: str,
    data: Optional[bytes],
    *,
    field: str,
    mime_type: Optional[str],
    name: Optional[str],
    hash: Optional[str],
) -> ContentRef:
    if mime_type is None and name is not None:
        mime_type = guess_mime_type(name)
    return ContentRef(
        item_id=item_id,
        field=field,
        hash=hash,
        size=len(data) if data is not None else None,
        mime_type=mime_type,
        url=None,  # resolved on demand via content_url — never baked in (may expire)
    )


def put_content(
    store: MutableMapping,
    item_id: str,
    data: bytes,
    *,
    field: str = "content",
    mime_type: Optional[str] = None,
    name: Optional[str] = None,
) -> ContentRef:
    """Location-addressed write: store ``data`` under a caller-supplied ``item_id``.

    Backend ``store`` is injected. Returns a :class:`ContentRef` (mime guessed from
    ``name`` if not given; ``url`` left empty — resolve via :func:`content_url`).

    >>> s = {}
    >>> ref = put_content(s, 'clip1', b'\\x00\\x01', name='clip1.wav')
    >>> ref.item_id, ref.hash, s['clip1']
    ('clip1', None, b'\\x00\\x01')
    >>> ref.mime_type.startswith('audio/')
    True
    """
    store[item_id] = data
    return _ref(item_id, data, field=field, mime_type=mime_type, name=name, hash=None)


def add_content(
    store: MutableMapping,
    data: bytes,
    *,
    field: str = "content",
    hasher: HashFunc = hashlib.sha256,
    length: Optional[int] = None,
    mime_type: Optional[str] = None,
    name: Optional[str] = None,
) -> ContentRef:
    """Content-addressed write: the key *is* the content hash; idempotent (CAS).

    A second call with identical bytes neither rewrites nor produces a different id, so
    identical content is stored once. Returns a :class:`ContentRef` with ``hash`` set.

    >>> s = {}
    >>> a = add_content(s, b'xyz')
    >>> b = add_content(s, b'xyz')
    >>> a.item_id == b.item_id == a.hash and len(s) == 1
    True
    """
    key = content_hash(data, hasher=hasher, length=length)
    if key not in store:
        store[key] = data
    return _ref(key, data, field=field, mime_type=mime_type, name=name, hash=key)


def get_content(store: Mapping, ref_or_key: Any) -> bytes:
    """Read content bytes by :class:`ContentRef`, wire dict, or bare key.

    >>> s = {}
    >>> ref = add_content(s, b'data')
    >>> get_content(s, ref) == get_content(s, ref.item_id) == b'data'
    True
    """
    return store[_key_of(ref_or_key)]


def delete_content(store: MutableMapping, ref_or_key: Any) -> None:
    """Delete content by :class:`ContentRef`, wire dict, or bare key (``del store[key]``).

    >>> s = {}
    >>> ref = add_content(s, b'gone')
    >>> delete_content(s, ref)
    >>> ref.item_id in s
    False
    """
    del store[_key_of(ref_or_key)]


class ContentAddressedStore(KvPersister):
    """A bytes store whose keys are the content hash of the values (CAS facade).

    Wraps any injected ``MutableMapping`` backend (``dict`` for tests, :class:`dol.Files`
    locally, an ``s3dol`` store in the cloud). Minting is via :meth:`add` (the store
    picks the key); reads/iter/delete delegate to the backend. A direct
    ``store[k] = v`` is allowed only when ``k`` equals the content hash of ``v`` — so the
    CAS invariant can't be silently violated.

    >>> cas = with_content_addressing()   # dict-backed
    >>> ref = cas.add(b'hello', name='h.txt')
    >>> cas[ref.item_id]
    b'hello'
    >>> list(cas) == [content_hash(b'hello')]
    True
    >>> cas.add(b'hello').item_id == ref.item_id   # idempotent / deduplicated
    True
    """

    def __init__(
        self,
        store: Optional[MutableMapping] = None,
        *,
        hasher: HashFunc = hashlib.sha256,
        length: Optional[int] = None,
        field: str = "content",
    ):
        self._store = {} if store is None else store
        self._hasher = hasher
        self._length = length
        self._field = field

    def add(
        self, data: bytes, *, mime_type: Optional[str] = None, name: Optional[str] = None
    ) -> ContentRef:
        """Write ``data`` under its content hash (idempotent); return a :class:`ContentRef`."""
        return add_content(
            self._store,
            data,
            field=self._field,
            hasher=self._hasher,
            length=self._length,
            mime_type=mime_type,
            name=name,
        )

    def __getitem__(self, k):
        return self._store[k]

    def __setitem__(self, k, v):
        expected = content_hash(v, hasher=self._hasher, length=self._length)
        if k != expected:
            raise ValueError(
                f"content-addressed key must equal the content hash: got {k!r}, "
                f"expected {expected!r}. Use `.add(data)` to let the store mint the key."
            )
        self._store[k] = v

    def __delitem__(self, k):
        del self._store[k]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __contains__(self, k):
        return k in self._store

    @property
    def url_for(self):
        """Delegate the ``url_for`` seam to the backend if it has one (else ``None``)."""
        return getattr(self._store, "url_for", None)


def with_content_addressing(
    store: Optional[MutableMapping] = None,
    *,
    hasher: HashFunc = hashlib.sha256,
    length: Optional[int] = None,
    field: str = "content",
) -> ContentAddressedStore:
    """Wrap an injected backend as a :class:`ContentAddressedStore` (``dict`` if ``None``).

    >>> cas = with_content_addressing(length=16)
    >>> len(cas.add(b'abc').item_id)
    16
    """
    return ContentAddressedStore(store, hasher=hasher, length=length, field=field)
