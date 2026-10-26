# dol.atomic

Atomic file writes: write to a sibling temp file, then `os.replace` it into place.

A crash (or an exception while producing the data) leaves the old content or the new,
never a torn or empty file. Dependency-free; `atomic_write` is the one entry point.

`file_lock` is its companion: an atomic replace is only safe against concurrent
appenders and readers when they all hold the same advisory lock.

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

### Functions

| [`atomic_write`](#dol.atomic.atomic_write)(path, data, \*[, fsync, perms])   | Write `data` to `path` atomically (temp file in the same folder, then `os.replace`).   |
|-------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| [`file_lock`](#dol.atomic.file_lock)(path, \*[, sidecar, poll_interval])  | Hold an exclusive advisory lock on `<path><sidecar>`; yields whether it is held.       |

### Classes

| [`FileLock`](#dol.atomic.FileLock)(path, \*[, sidecar, poll_interval])   | A reusable, exclusive, advisory lock on a sidecar file `<path><sidecar>`.   |
|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|

### *class* dol.atomic.FileLock(path, , sidecar='.lock', poll_interval=0.001)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A reusable, exclusive, advisory lock on a sidecar file `<path><sidecar>`.

Use it as `with lock:`; the body runs either way, and `lock.held` says whether
the lock is actually held (it is not when the lock folder is read-only or the
filesystem does not support locking: advisory locks are best effort, and the caller
decides whether that matters).

The handle is kept open between uses (cheaper for many short sections); `close()`
it when done. It is fork-safe: `flock` locks belong to the open file description,
so a forked child that inherited the handle would not exclude its parent; the pid
is remembered and the handle reopened on a mismatch (and a child never releases
its parent’s lock). Threads of one process exclude each other too. Not reentrant:
entering again from the holding thread raises `RuntimeError` (a second lock on
the same path from the same thread would deadlock). For a one-off section, use
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

### dol.atomic.atomic_write(path, data, , fsync=False, perms=None)

Write `data` to `path` atomically (temp file in the same folder, then `os.replace`).

* **Parameters:**
  * **path** – Target file; its parent folder must exist. A symlink is followed:
    its target is what gets replaced.
  * **data** ([`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)]) – Bytes-like, or an iterable of bytes-like chunks (written one at a
    time, so a large payload need not be held in memory). If the iterable raises,
    `path` is untouched. Text must be encoded first.
  * **fsync** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Flush the file (and, on POSIX, the folder) to disk before returning:
    durable across power loss, not just across a process crash. Slower. (On macOS
    `os.fsync` is not `F_FULLFSYNC`: best effort.)
  * **perms** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Mode (e.g. `0o600`) for the file, set at creation so the data is
    never readable more widely, even briefly, and not narrowed by the umask.
    Default: the mode of the file being replaced, else the usual umask mode.
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

Hold an exclusive advisory lock on `<path><sidecar>`; yields whether it is held.

Cross-process and cross-platform (`flock` on POSIX, `msvcrt.locking` on
Windows). Take it around every append, read and [`atomic_write()`](#dol.atomic.atomic_write) of `path`
so they cannot interleave. The body still runs when locking is unavailable
(read-only folder, filesystem without lock support): the yielded value is then
`False`. A lock file that exists but is not writable is opened read-only.
Nesting it on the same path in one thread deadlocks; use one section.

```pycon
>>> import tempfile
>>> path = os.path.join(tempfile.mkdtemp(), "data.json")
>>> with file_lock(path) as held:
...     atomic_write(path, b"{}")
>>> held, os.path.exists(path + ".lock")
(True, True)
```
