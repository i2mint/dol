# dol.atomic

Atomic file writes: write to a sibling temp file, then `os.replace` it into place.

A crash (or an exception while producing the data) leaves the old content or the new,
never a torn or empty file. Dependency-free; `atomic_write` is the entry point, and
`atomic_delete` its counterpart for removal. A write killed mid-way leaves its temp
file behind; `interrupted_writes` lists them (`TEMP_NAME_PATTERN` is their name).

On Windows, a reader holding the destination open makes `os.replace` and `os.unlink`
fail with `PermissionError` until it closes (Python’s `open` does not share delete).
Both functions therefore retry that briefly, through a [`Retry`](#dol.atomic.Retry) policy (its
`retry` argument): by default Windows’ `PermissionError` only, ~0.6 s in all.
Elsewhere the same error is a real permission problem, so nothing is retried.

`file_lock` is its companion: an atomic replace is only safe against concurrent
appenders and readers when they all hold the same advisory lock. `LockedAppender`
appends under that lock and follows the file when an atomic replace swaps it.

```pycon
>>> import os, tempfile
>>> path = os.path.join(tempfile.mkdtemp(), "f.txt")
>>> atomic_write(path, b"hello")
>>> open(path).read(), os.listdir(os.path.dirname(path))
('hello', ['f.txt'])
>>> atomic_write(path, (c.encode() for c in "ab"))  # chunks are streamed
>>> open(path).read()
'ab'
```

### Module Attributes

| [`SHARING_VIOLATIONS`](#dol.atomic.SHARING_VIOLATIONS)   | Windows' sharing violation (another handle has the file open); nothing elsewhere, where `PermissionError` means a real permission problem.                                                                        |
|-----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`TEMP_NAME_PATTERN`](#dol.atomic.TEMP_NAME_PATTERN)    | Matches (`fullmatch`) every name [`temp_name()`](#dol.atomic.temp_name) makes, and so the temp file a killed [`atomic_write()`](#dol.atomic.atomic_write) leaves behind. |
| [`DFLT_KEEP_HANDLE`](#dol.atomic.DFLT_KEEP_HANDLE)     | Whether a [`LockedAppender`](#dol.atomic.LockedAppender) keeps its file open between appends.                                                                                                    |

### Functions

| [`atomic_delete`](#dol.atomic.atomic_delete)(path, \*[, retry])                  | Remove the file `path`: `retry(os.unlink, path)`, the delete counterpart of [`atomic_write()`](#dol.atomic.atomic_write).                                                                                                                                                                                                                        |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`atomic_write`](#dol.atomic.atomic_write)(path, data, \*[, fsync, perms, ...]) | Write `data` to `path` atomically (temp file in the same folder, then `os.replace`).                                                                                                                                                                                                                                                                                |
| [`file_lock`](#dol.atomic.file_lock)(path, \*[, sidecar, poll_interval])     | Hold an exclusive advisory lock on `path`'s sidecar (see [`FileLock`](#dol.atomic.FileLock)); yields whether it is held.                                                                                                                                                                                                                     |
| [`interrupted_writes`](#dol.atomic.interrupted_writes)(folder, \*[, pattern, ...])    | The temp files writes killed mid-way left in `folder`: sorted absolute paths (`str`; `..` is not collapsed, so each names the file found) of the regular files (never a symlink or folder) whose name fully matches `pattern` (default: [`atomic_write()`](#dol.atomic.atomic_write)'s temps; pass another writer's, e.g. `r"\.blob-.+\.part"`). |
| [`temp_name`](#dol.atomic.temp_name)(name)                                   | The name [`atomic_write()`](#dol.atomic.atomic_write) stages a write to `name` under: a hidden sibling `.<name, clipped to 100 bytes>.<8 random hex>.tmp`, always matched by [`TEMP_NAME_PATTERN`](#dol.atomic.TEMP_NAME_PATTERN).                                                                                                 |
| [`update_json`](#dol.atomic.update_json)(path, change, \*[, default, ...])     | Apply `change` to the JSON document at `path` under its lock; return what `change` returned.                                                                                                                                                                                                                                                                        |

### Classes

| [`FileLock`](#dol.atomic.FileLock)(path, \*[, sidecar, poll_interval])    | A reusable, exclusive, advisory lock on a sidecar file `<path><sidecar>` (a shortened hidden name instead, for names near the 255-byte limit).                                                                                             |
|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`LockedAppender`](#dol.atomic.LockedAppender)(path, \*[, lock, line_end, ...]) | Appends to `path` under a [`FileLock`](#dol.atomic.FileLock), following the file when an [`atomic_write()`](#dol.atomic.atomic_write) (made under the same lock, e.g. a compaction) replaces it. |
| [`Retry`](#dol.atomic.Retry)([on, attempts, backoff_s])                | Call `op(*args)`, retrying on `on` exceptions with doubling backoff.                                                                                                                                                                       |

### Exceptions

| [`InvalidJSONError`](#dol.atomic.InvalidJSONError)   | The file [`update_json()`](#dol.atomic.update_json) was asked to update does not hold valid JSON.   |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|

### dol.atomic.DFLT_KEEP_HANDLE *= True*

Whether a [`LockedAppender`](#dol.atomic.LockedAppender) keeps its file open between appends. Not on
Windows: an open handle there makes the `os.replace` of a compaction fail.

### *class* dol.atomic.FileLock(path, , sidecar='.lock', poll_interval=0.001)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A reusable, exclusive, advisory lock on a sidecar file `<path><sidecar>`
(a shortened hidden name instead, for names near the 255-byte limit).

Use it as `with lock:`; the body runs either way, and `lock.held` says whether
the lock is actually held (it is not when the lock folder is read-only or the
filesystem does not support locking: advisory locks are best effort, and the caller
decides whether that matters).

The handle is kept open between uses (cheaper for many short sections); `close()`
it when done. It is fork-safe: `flock` locks belong to the open file description,
so a forked child that inherited the handle would not exclude its parent; the pid
is remembered and the handle reopened on a mismatch (and a child never releases
its parent’s lock). Threads of one process exclude each other too. Not reentrant:
entering again from the holding thread raises `RuntimeError`, and so does
entering another `FileLock` (or [`file_lock()`](#dol.atomic.file_lock)) of the same path from a
thread that already holds it, which would otherwise wait on itself forever. For
a one-off section, use
[`file_lock()`](#dol.atomic.file_lock).

The lock file is a separate sidecar, not `path` itself, so that
[`atomic_write()`](#dol.atomic.atomic_write) replacing `path` does not swap the file being locked.

```pycon
>>> import tempfile
>>> lock = FileLock(os.path.join(tempfile.mkdtemp(), "data.json"))
>>> with lock:
...     lock.held
True
>>> lock.close()
```

#### close()

Release the kept handle (not from inside the `with` block).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *exception* dol.atomic.InvalidJSONError

Bases: [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

The file [`update_json()`](#dol.atomic.update_json) was asked to update does not hold valid JSON.

### *class* dol.atomic.LockedAppender(path, , lock=None, line_end=None, keep_handle=True)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Appends to `path` under a [`FileLock`](#dol.atomic.FileLock), following the file when an
[`atomic_write()`](#dol.atomic.atomic_write) (made under the same lock, e.g. a compaction) replaces it.

Each append is one unbuffered `O_APPEND` write, so nothing is pending between
appends: a failed append leaves nothing to land later, and an inherited handle
(after a fork) is safe to share. The handle is kept between appends
(`keep_handle`) and reopened when the file at `path` is no longer the one it
holds, or after a failed append.

To replace the file, hold the appender’s lock, so no append lands in the old file
after it was read: `with appender.lock: atomic_write(path, compacted)`. Whether
the lock is actually held there is `appender.lock.held` (see [`FileLock`](#dol.atomic.FileLock)).
`with appender:` does not lock; it only closes on exit. Atomic appends and tail
healing assume the lock is held by every writer: when it cannot be (read-only
folder, filesystem without locks), appends are best effort.

```pycon
>>> import tempfile
>>> path = os.path.join(tempfile.mkdtemp(), "log.jsonl")
>>> log = LockedAppender(path, line_end=b"\n")
>>> log.append(b'{"a": 1}\n')
>>> with log.lock:
...     atomic_write(path, b'{"compacted": true}\n')
>>> log.append(b'{"b": 2}\n')
>>> open(path, "rb").read()
b'{"compacted": true}\n{"b": 2}\n'
>>> log.close()
```

#### append(data)

Append `data` under the lock, to the file now at `path`.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### close()

Close the file and the lock’s handle (both reopen on the next append).
Not from inside `with appender.lock:` (the lock is not reentrant).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* dol.atomic.Retry(on=(), attempts=7, backoff_s=0.01)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Call `op(*args)`, retrying on `on` exceptions with doubling backoff.

`attempts` calls at most, sleeping `backoff_s`, then twice that, and so on
between them (defaults: 7 calls, ~0.6 s of sleep in all; the backoff is not capped,
so keep `attempts` small). The last failure is re-raised. Any other exception
propagates at once. `Retry(attempts=1)` (or `on=()`) calls `op` once.

```pycon
>>> calls = []
>>> def flaky():
...     calls.append(1)
...     if len(calls) < 3:
...         raise PermissionError("in use")
...     return "done"
>>> Retry(on=(PermissionError,), backoff_s=0)(flaky), len(calls)
('done', 3)
```

### dol.atomic.SHARING_VIOLATIONS *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[type](https://docs.python.org/3/builtins/functions.html#type)[[BaseException](https://docs.python.org/3/builtins/exceptions.html#BaseException)], ...]* *= ()*

Windows’ sharing violation (another handle has the
file open); nothing elsewhere, where `PermissionError` means a real permission problem.
Windows reports a sharing violation on `os.replace` with the same error as a genuine
access denial, so there *every* `PermissionError` is retried: a read-only destination
still fails, only ~0.6 s later.

* **Type:**
  What `DFLT_RETRY` retries

### dol.atomic.TEMP_NAME_PATTERN *= re.compile('\\\\..+\\\\.[0-9a-f]{8}\\\\.tmp', re.DOTALL)*

Matches (`fullmatch`) every name [`temp_name()`](#dol.atomic.temp_name) makes, and so the temp file a
killed [`atomic_write()`](#dol.atomic.atomic_write) leaves behind. `DOTALL`: a file name may hold a newline.

### dol.atomic.atomic_delete(path, , retry=Retry(on=(), attempts=7, backoff_s=0.01))

Remove the file `path`: `retry(os.unlink, path)`, the delete counterpart
of [`atomic_write()`](#dol.atomic.atomic_write).

`os.unlink` is already atomic; what this adds is the same retry, so a delete is
not refused on Windows because a reader has the file open. (Windows may then keep
the name “delete pending” until the last handle closes.) A symlink is removed
itself, not its target. A missing file raises `FileNotFoundError` (not retried).

```pycon
>>> import tempfile
>>> path = os.path.join(tempfile.mkdtemp(), "f")
>>> atomic_write(path, b"x")
>>> atomic_delete(path)
>>> os.path.exists(path)
False
```

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### dol.atomic.atomic_write(path, data, , fsync=False, perms=None, retry=Retry(on=(), attempts=7, backoff_s=0.01))

Write `data` to `path` atomically (temp file in the same folder, then `os.replace`).

* **Parameters:**
  * **path** – Target file (`str`, `bytes` or path-like); its parent folder must
    exist. A symlink is followed: its target is what gets replaced.
  * **data** ([`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)]) – Bytes-like, or an iterable of bytes-like chunks (written one at a
    time, so a large payload need not be held in memory). If the iterable raises,
    `path` is untouched. Text must be encoded first.
  * **fsync** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Flush the file (and, on POSIX, the folder) to disk before returning:
    durable across power loss, not just across a process crash. Slower. (On macOS
    `os.fsync` is not `F_FULLFSYNC`: best effort.)
  * **perms** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Mode (e.g. `0o600`) for the file, set at creation so the data is
    never readable more widely, even briefly, and not narrowed by the umask.
    Default: the mode of the file being replaced, else the usual umask mode.
  * **retry** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – How the final `os.replace` is called: `retry(os.replace, tmp,
    path)`. Default `DFLT_RETRY` retries Windows’ sharing violations (a
    reader has `path` open); `Retry(attempts=1)` calls it once.
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

The temp file is a hidden, uniquely named sibling (`.<name>.<random>.tmp`), created
with `O_EXCL`, so concurrent writers never share one and key listings skip it.
On any failure it is removed and the old content is untouched. Atomic on POSIX,
and on Windows within one volume. Hard links to the old file are detached, and its owner is not preserved.

```pycon
>>> import tempfile
>>> d = tempfile.mkdtemp()
>>> atomic_write(os.path.join(d, "a"), [b"x", b"y"], perms=0o600, fsync=True)
>>> open(os.path.join(d, "a"), "rb").read()
b'xy'
>>> def boom():
...     yield b"partial"
...     raise RuntimeError("producer died")
>>> atomic_write(os.path.join(d, "a"), boom())
Traceback (most recent call last):
  ...
RuntimeError: producer died
>>> open(os.path.join(d, "a"), "rb").read(), os.listdir(d)  # old content, no litter
(b'xy', ['a'])
```

### dol.atomic.file_lock(path, , sidecar='.lock', poll_interval=0.001)

Hold an exclusive advisory lock on `path`’s sidecar (see [`FileLock`](#dol.atomic.FileLock));
yields whether it is held.

Cross-process and cross-platform (`flock` on POSIX, `msvcrt.locking` on
Windows). Take it around every append, read and [`atomic_write()`](#dol.atomic.atomic_write) of `path`
so they cannot interleave. The body still runs when locking is unavailable
(read-only folder, filesystem without lock support): the yielded value is then
`False`. A lock file that exists but is not writable is opened read-only.
Nesting it on the same path in one thread raises `RuntimeError` (it would
deadlock); use one section.

```pycon
>>> import tempfile
>>> path = os.path.join(tempfile.mkdtemp(), "data.json")
>>> with file_lock(path) as held:
...     atomic_write(path, b"{}")
>>> held, os.path.exists(path + ".lock")
(True, True)
```

### dol.atomic.interrupted_writes(folder, , pattern=re.compile('\\\\..+\\\\.[0-9a-f]{8}\\\\.tmp', re.DOTALL), recursive=False)

The temp files writes killed mid-way left in `folder`: sorted absolute paths
(`str`; `..` is not collapsed, so each names the file found) of the regular files (never a symlink or folder) whose name fully
matches `pattern` (default: [`atomic_write()`](#dol.atomic.atomic_write)’s temps; pass another writer’s,
e.g. `r"\.blob-.+\.part"`).

Matching is by name only: call it on folders your own stores write, where nothing
else is named like a temp. A listed file may also be a write still in progress in
another process: delete only those older than any write could take
(`os.stat(p).st_mtime`), or while holding the lock every writer takes.

* **Parameters:**
  **recursive** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Also search subfolders (symlinked folders are not followed;
  subfolders that cannot be read, or vanish, are skipped).
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

A missing `folder` holds none (`[]`); any other failure to list it raises
`OSError`.

```pycon
>>> import tempfile
>>> with tempfile.TemporaryDirectory() as d:
...     for name in ("k.mp4", ".k.mp4.ab12cd34.tmp", ".DS_Store", "x.tmp"):
...         open(os.path.join(d, name), "wb").close()
...     [os.path.basename(p) for p in interrupted_writes(d)]
['.k.mp4.ab12cd34.tmp']
```

### dol.atomic.temp_name(name)

The name [`atomic_write()`](#dol.atomic.atomic_write) stages a write to `name` under: a hidden sibling
`.<name, clipped to 100 bytes>.<8 random hex>.tmp`, always matched by
[`TEMP_NAME_PATTERN`](#dol.atomic.TEMP_NAME_PATTERN). Unique per call. `name` is a file name, not a path.

```pycon
>>> bool(TEMP_NAME_PATTERN.fullmatch(temp_name("data.json")))
True
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### dol.atomic.update_json(path, change, \*, default=<class 'dict'>, loads=<function loads>, dumps=<function dumps>, lock=None, warn_unlocked=True, write=<function atomic_write>)

Apply `change` to the JSON document at `path` under its lock; return what
`change` returned.

Under [`file_lock()`](#dol.atomic.file_lock) of `path`: read the document (a `dict` or `list`),
call `change(obj)`, which mutates `obj` in place (and may raise, to write
nothing), then [`atomic_write()`](#dol.atomic.atomic_write) it back, only if `dumps` of it changed. So a
no-op neither creates the file nor moves its mtime, and a file nobody changed
keeps its formatting. What `change` returns is handed back, never written.

The read is strict: only a missing file starts from `default()`. Any other read
error propagates, and invalid JSON raises [`InvalidJSONError`](#dol.atomic.InvalidJSONError) (a
`ValueError`): writing “default plus this change” over a file that could not
be read would destroy what it held.

* **Parameters:**
  * **path** – The JSON file; its folder must exist.
  * **default** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – Factory for the document when `path` does not exist.
  * **dumps** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – The codec. `loads` gets bytes; `dumps` may return str or
    bytes (e.g. `functools.partial(json.dumps, indent=2)`, or orjson’s). Change
    is detected by comparing `dumps` before and after, so it must be
    deterministic.
  * **lock** – 

    The context manager to hold. Default: the [`FileLock`](#dol.atomic.FileLock) every
    [`file_lock()`](#dol.atomic.file_lock) of `path` takes. Any other works; whether it is held is
    > the `held` of what it yields, else the truth of what it yields (`None`,
    > e.g. `nullcontext()`, counts as held). Called while this thread already
    > holds the file’s lock, the default lock raises `RuntimeError`; inside a
    > section that already holds it, pass `lock=contextlib.nullcontext()`.
  * **warn_unlocked** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Warn when the lock is not held (read-only folder,
    filesystem without locks), so the loss of the guarantee is not silent.
  * **write** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – How the new document’s bytes reach `path`: `write(path, data)`,
    called under the lock. Default [`atomic_write()`](#dol.atomic.atomic_write) as is; e.g.
    `functools.partial(atomic_write, fsync=True, perms=0o600)` for a durable,
    owner-only record. It must replace `path` atomically, or a reader can see a
    torn document.

```pycon
>>> import tempfile
>>> path = os.path.join(tempfile.mkdtemp(), "manifest.json")
>>> update_json(path, lambda m: m.setdefault("clips", []).append("a.mp4"))
>>> update_json(path, lambda m: len(m["clips"]))
1
>>> open(path).read()
'{"clips": ["a.mp4"]}'
>>> update_json(path, lambda m: {**m, "lost": True})  # returned, not written
{'clips': ['a.mp4'], 'lost': True}
>>> open(path).read()
'{"clips": ["a.mp4"]}'
```
