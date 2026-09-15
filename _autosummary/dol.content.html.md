# dol.content

Content references and content-addressed storage — the flat “blob” layer.

Many apps split their data into two concerns (the *content-metadata bifurcation*;
see `misc/docs/dol_content_metadata_bifurcation.md`):

- **records / metadata** — small, queryable rows (a `MutableMapping` of dicts, a DB);
- **content / blobs** — large bytes (media, documents, renders) that you don’t want
  to inline into a record or a query result.

This module is the **content half**: a *flat* bytes store plus a small, serializable
[`ContentRef`](#dol.content.ContentRef) token that stands in for the bytes inside a record. The store
itself is just a `MutableMapping[str, bytes]` — so the backend is **injected**
(`dict` in tests, `dol.Files` locally, an `s3dol` store in the cloud) and
nothing here depends on any of them.

Two addressing modes, mirroring the same convention used by the `zodal` TypeScript
stores (so a [`ContentRef`](#dol.content.ContentRef) serialized here matches `zodal`’s `ContentRef` on
the wire — see [`ContentRef.to_json()`](#dol.content.ContentRef.to_json)):

- **location-addressed** ([`put_content()`](#dol.content.put_content)) — the caller supplies the id;
- **content-addressed** ([`add_content()`](#dol.content.add_content) / [`with_content_addressing()`](#dol.content.with_content_addressing)) — the
  id *is* the content hash, which makes writes idempotent and deduplicated (CAS).

**URLs are resolved on demand, never baked in.** `put_content`/`add_content` leave
`ContentRef.url` empty; call [`content_url()`](#dol.content.content_url) when you actually need a fetchable
URL. This is deliberate: a backend’s `url_for` may mint a *presigned, expiring* URL
(e.g. S3), and a [`ContentRef`](#dol.content.ContentRef) is meant to be *persisted* inside a record — so
freezing an expiring URL into it would be a latent bug. Reads can thus redirect to a
CDN / presigned URL / static route while writes always go to the injected backend.

```pycon
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
```

The wire form is camelCase and drops empty fields, matching `zodal`’s `ContentRef`:

```pycon
>>> ref.to_json()['itemId'] == ref.item_id
True
>>> sorted(ref.to_json())
['_tag', 'field', 'hash', 'itemId', 'mimeType', 'size']
```

### Module Attributes

| [`CONTENT_REF_TAG`](#dol.content.CONTENT_REF_TAG)   | The `_tag` discriminator value carried on the JSON wire form (cross-language parity).   |
|--------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [`HashFunc`](#dol.content.HashFunc)          | A key-minting hash constructor, e.g. `hashlib.sha256` — `bytes -> hash object`.         |

### Functions

| [`add_content`](#dol.content.add_content)(store, data, \*[, field, hasher, ...])   | Content-addressed write: the key *is* the content hash; idempotent (CAS).                                                     |
|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| [`content_hash`](#dol.content.content_hash)(data, \*[, hasher, length])             | Hex content hash of `data` (sha256 by default), optionally truncated to `length`.                                             |
| [`content_url`](#dol.content.content_url)(store, ref_or_key)                       | A fetchable URL for content, resolved **on demand**.                                                                          |
| [`delete_content`](#dol.content.delete_content)(store, ref_or_key)                    | Delete content by [`ContentRef`](#dol.content.ContentRef), wire dict, or bare key (`del store[key]`).     |
| [`get_content`](#dol.content.get_content)(store, ref_or_key)                       | Read content bytes by [`ContentRef`](#dol.content.ContentRef), wire dict, or bare key.                    |
| [`guess_mime_type`](#dol.content.guess_mime_type)(name)                                | Guess a mime type from a filename/key by extension (stdlib `mimetypes`).                                                      |
| [`is_content_ref`](#dol.content.is_content_ref)(obj)                                  | True for a [`ContentRef`](#dol.content.ContentRef) instance or its wire-form dict (`_tag` discriminator). |
| [`put_content`](#dol.content.put_content)(store, item_id, data, \*[, ...])         | Location-addressed write: store `data` under a caller-supplied `item_id`.                                                     |
| [`with_content_addressing`](#dol.content.with_content_addressing)([store, hasher, ...])        | Wrap an injected backend as a [`ContentAddressedStore`](#dol.content.ContentAddressedStore) (`dict` if `None`).      |

### Classes

| [`ContentAddressedStore`](#dol.content.ContentAddressedStore)([store, hasher, ...])   | A bytes store whose keys are the content hash of the values (CAS facade).   |
|------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`ContentRef`](#dol.content.ContentRef)(item_id[, field, hash, url, ...])  | A small, serializable stand-in for stored content (bytes).                  |
| [`SupportsUrlFor`](#dol.content.SupportsUrlFor)(\*args, \*\*kwargs)            | A backend that can hand out a directly-fetchable URL for a stored key.      |

### dol.content.CONTENT_REF_TAG *= 'ContentRef'*

The `_tag` discriminator value carried on the JSON wire form (cross-language parity).

### *class* dol.content.ContentAddressedStore(store=None, \*, hasher=<built-in function openssl_sha256>, length=None, field='content')

Bases: [`KvPersister`](dol.base.html.md#dol.base.KvPersister)

A bytes store whose keys are the content hash of the values (CAS facade).

Wraps any injected `MutableMapping` backend (`dict` for tests, `dol.Files`
locally, an `s3dol` store in the cloud). Minting is via [`add()`](#dol.content.ContentAddressedStore.add) (the store
picks the key); reads/iter/delete delegate to the backend. A direct
`store[k] = v` is allowed only when `k` equals the content hash of `v` — so the
CAS invariant can’t be silently violated.

```pycon
>>> cas = with_content_addressing()   # dict-backed
>>> ref = cas.add(b'hello', name='h.txt')
>>> cas[ref.item_id]
b'hello'
>>> list(cas) == [content_hash(b'hello')]
True
>>> cas.add(b'hello').item_id == ref.item_id   # idempotent / deduplicated
True
```

#### add(data, , mime_type=None, name=None)

Write `data` under its content hash (idempotent); return a [`ContentRef`](#dol.content.ContentRef).

* **Return type:**
  [`ContentRef`](#dol.content.ContentRef)

#### *property* url_for

Delegate the `url_for` seam to the backend if it has one (else `None`).

### *class* dol.content.ContentRef(item_id, field='content', hash=None, url=None, mime_type=None, size=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A small, serializable stand-in for stored content (bytes).

Held *inside* a record in place of the bytes, so lists/queries stay light. It is
addressed by `(item_id, field)` (a record may have several content fields);
`hash` is populated for content-addressed writes and left `None` otherwise.
`url` is an optional directly-fetchable location — normally left empty and
resolved on demand via [`content_url()`](#dol.content.content_url) (see the module docstring).

#### *classmethod* from_json(d)

Parse a wire-form dict (camelCase) back into a [`ContentRef`](#dol.content.ContentRef).

* **Return type:**
  [`ContentRef`](#dol.content.ContentRef)

#### to_json()

camelCase wire form matching `zodal`’s `ContentRef` (empty fields dropped).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### dol.content.HashFunc

A key-minting hash constructor, e.g. `hashlib.sha256` — `bytes -> hash object`.

alias of `Callable`[[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### *class* dol.content.SupportsUrlFor(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A backend that can hand out a directly-fetchable URL for a stored key.

The seam that lets **reads** redirect to a CDN / presigned URL / static route while
**writes** stay on the injected backend. Local file stores typically don’t implement
it ([`content_url()`](#dol.content.content_url) then returns `None`); an `s3dol` store implements it with
a presigned URL — so all S3 knowledge lives in `s3dol`, never here.

### dol.content.add_content(store, data, \*, field='content', hasher=<built-in function openssl_sha256>, length=None, mime_type=None, name=None)

Content-addressed write: the key *is* the content hash; idempotent (CAS).

A second call with identical bytes neither rewrites nor produces a different id, so
identical content is stored once. Returns a [`ContentRef`](#dol.content.ContentRef) with `hash` set.

* **Return type:**
  [`ContentRef`](#dol.content.ContentRef)

```pycon
>>> s = {}
>>> a = add_content(s, b'xyz')
>>> b = add_content(s, b'xyz')
>>> a.item_id == b.item_id == a.hash and len(s) == 1
True
```

### dol.content.content_hash(data, \*, hasher=<built-in function openssl_sha256>, length=None)

Hex content hash of `data` (sha256 by default), optionally truncated to `length`.

Truncation trades key length for a higher collision probability (a 16-hex-char
prefix is 64 bits) — leave `length` unset unless keys must be short and the
corpus is small.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> content_hash(b'abc') == content_hash(b'abc')
True
>>> len(content_hash(b'abc', length=16))
16
```

### dol.content.content_url(store, ref_or_key)

A fetchable URL for content, resolved **on demand**.

Prefers a URL the ref already carries; otherwise asks the backend’s `url_for`
(the [`SupportsUrlFor`](#dol.content.SupportsUrlFor) seam), returning `None` if it has none.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

```pycon
>>> class Served(dict):
...     def url_for(self, key): return f'https://cdn.example/{key}'
>>> content_url(Served(), 'k1')
'https://cdn.example/k1'
>>> content_url({}, 'k1') is None
True
>>> content_url({}, ContentRef('k1', url='https://carried/k1'))  # ref carries its own
'https://carried/k1'
```

The key is resolved **through any wrapping layers**, so a URL addresses the same object
`store[key]` reads. Without this, a key-transforming wrap would hand the backend the
outer key and silently return a URL for a different object:

```pycon
>>> from dol import KeyCodecs
>>> wrapped = KeyCodecs.prefixed('a/')(Served)({'a/k1': b'v'})
>>> wrapped['k1']
b'v'
>>> content_url(wrapped, 'k1')
'https://cdn.example/a/k1'
```

### dol.content.delete_content(store, ref_or_key)

Delete content by [`ContentRef`](#dol.content.ContentRef), wire dict, or bare key (`del store[key]`).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

```pycon
>>> s = {}
>>> ref = add_content(s, b'gone')
>>> delete_content(s, ref)
>>> ref.item_id in s
False
```

### dol.content.get_content(store, ref_or_key)

Read content bytes by [`ContentRef`](#dol.content.ContentRef), wire dict, or bare key.

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)

```pycon
>>> s = {}
>>> ref = add_content(s, b'data')
>>> get_content(s, ref) == get_content(s, ref.item_id) == b'data'
True
```

### dol.content.guess_mime_type(name)

Guess a mime type from a filename/key by extension (stdlib `mimetypes`).

Results depend on the platform’s mime registry, so treat them as best-effort.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

```pycon
>>> guess_mime_type('a.json')
'application/json'
>>> guess_mime_type('no-extension') is None
True
```

### dol.content.is_content_ref(obj)

True for a [`ContentRef`](#dol.content.ContentRef) instance or its wire-form dict (`_tag` discriminator).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

```pycon
>>> is_content_ref(ContentRef('id1'))
True
>>> is_content_ref({'_tag': 'ContentRef', 'itemId': 'id1'})
True
>>> is_content_ref({'itemId': 'id1'}) or is_content_ref('id1')
False
```

### dol.content.put_content(store, item_id, data, , field='content', mime_type=None, name=None)

Location-addressed write: store `data` under a caller-supplied `item_id`.

Backend `store` is injected. Returns a [`ContentRef`](#dol.content.ContentRef) (mime guessed from
`name` if not given; `url` left empty — resolve via [`content_url()`](#dol.content.content_url)).

* **Return type:**
  [`ContentRef`](#dol.content.ContentRef)

```pycon
>>> s = {}
>>> ref = put_content(s, 'clip1', b'\x00\x01', name='clip1.wav')
>>> ref.item_id, ref.hash, s['clip1']
('clip1', None, b'\x00\x01')
>>> ref.mime_type.startswith('audio/')
True
```

### dol.content.with_content_addressing(store=None, \*, hasher=<built-in function openssl_sha256>, length=None, field='content')

Wrap an injected backend as a [`ContentAddressedStore`](#dol.content.ContentAddressedStore) (`dict` if `None`).

* **Return type:**
  [`ContentAddressedStore`](#dol.content.ContentAddressedStore)

```pycon
>>> cas = with_content_addressing(length=16)
>>> len(cas.add(b'abc').item_id)
16
```
