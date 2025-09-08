## a few often useful `dol` tools

Often the first data source or target you work with is local files. `dol.Files` gives you
a `MutableMapping` view over a folder so you can read, write, list and delete files using
simple dict-like operations instead of juggling `os.path` and `open`read/write calls. That
makes higher-level code more independent of the underlying storage, so later you can
swap in S3, a database, or another adapter with minimal changes.

Below are three compact, practical tools you will use frequently when working with local
files: `Files` (and `TextFiles`), `filt_iter`, and `wrap_kvs`.

### Files / TextFiles

- What: `Files(rootdir, ...)` (class or instance) produces a MutableMapping where keys
  are relative paths under `rootdir` (strings like `'path/to/file.txt'`) and values are
  the raw bytes of the file contents.
- Why: Treat folders like dicts. You can iterate, check containment, read and write using
  familiar mapping operations.
- Basic contract:
  - Input: a filesystem directory path (rootdir).
  - Output: mapping where
    - keys -> relative file paths (strings)
    - values -> bytes read/written from/to files
  - Error modes: key validation enforces keys are under the rootdir; invalid keys raise KeyError.

Example:

```python
from dol import Files

files = Files('/path/to/root')  # instance wrapping that folder
list(files)               # list of relative file paths (keys)
files['doc.txt'] = b'hello'  # write bytes
print(files['doc.txt'])      # b'hello'
assert 'doc.txt' in files
del files['doc.txt']         # delete file
```

If you prefer automatic text handling (str instead of bytes), use `TextFiles`
which opens files in text mode by default:

```python
from dol import TextFiles
texts = TextFiles('/path/to/root')
texts['notes.txt'] = 'a string'   # writes text
print(texts['notes.txt'])         # reads str
```

### filt_iter — filter the mapping view

- What: `filt_iter` produces a wrapper (class or instance) that restricts the mapping
  to keys satisfying a filter. The filter can be a callable (k -> bool) or an iterable
  of allowed keys. There are convenient helpers like `filt_iter.suffixes`,
  `filt_iter.prefixes`, and `filt_iter.regex` for common cases.
- Why: Make a focused view (e.g. "only .json files") without copying data. Useful for
  pipelines and for composing with other transformations.

Example — only list and access `.json` files:

```python
from dol import Files
from dol.trans import filt_iter

files = Files('/path/to/root')
json_view = filt_iter.suffixes('.json')(files)
list(json_view)           # only .json keys are shown
obj = json_view['data.json']  # behaves like files['data.json'] (same underlying data)
# writing to a non-matching key raises KeyError
try:
    json_view['other.txt'] = b'no'
except KeyError:
    pass
```

You can also use `filt_iter(filt=callable)` to build arbitrary predicates.

### wrap_kvs — add key/value transformations (codecs, codecs-per-key)

- What: `wrap_kvs` (or its helpers like `kv_wrap`, `KeyCodec`, `ValueCodec`) wraps a
  store so you can transparently transform incoming/outgoing keys and values. Common
  uses are decoding bytes into Python objects on read, and encoding them on write.
- Why: Keep serialization and key-layout concerns orthogonal to business logic.

Contract:
  - Inputs: underlying store (class/instance) and transformation functions.
  - Outputs: mapping that applies the transformations:
    - `value_decoder` / `obj_of_data` converts stored format -> Python value on read
    - `value_encoder` / `data_of_obj` converts Python value -> stored format on write
    - `key_of_id` / `id_of_key` transform keys in/out (e.g. add/remove extensions or prefixes)

Example — treat files as JSON objects instead of bytes:

```python
import json
from dol import Files
from dol.trans import wrap_kvs

files = Files('/path/to/root')
json_store = wrap_kvs(
    files,
    value_decoder=lambda b: json.loads(b.decode()),
    value_encoder=lambda obj: json.dumps(obj, indent=2).encode(),
)

json_store['data.json'] = {'a': 1}        # writes pretty JSON bytes
print(json_store['data.json'])           # reads Python dict
```

You can chain `wrap_kvs` and `filt_iter` to get both decoding and focused views. For
example: restrict to `.json` files and expose them as Python dicts:

```python
json_only = filt_iter.suffixes('.json')(json_store)
for k in json_only:
    print(k, type(json_only[k]))  # keys end with .json and values are dicts
```

Notes, tips and edge-cases
- `Files` keys are relative paths; validation ensures keys are under the rootdir.
- `filt_iter` filters the mapping API surface: non-matching writes/reads raise KeyError.
- `wrap_kvs` accepts several parameter names (value_decoder/value_encoder or
  obj_of_data/data_of_obj) — they are aliases; prefer the clearer names in your code.
- Chain small wrappers: Files -> wrap_kvs (codec) -> filt_iter (view) keeps code
  modular and makes swapping storage implementations easy.

That should give an AI the essentials it needs to read and write local files using
`dol` idioms and to compose filters and codecs when implementing higher-level logic.
