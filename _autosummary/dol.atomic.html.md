# dol.atomic

Atomic file writes: write to a sibling temp file, then `os.replace` it into place.

A crash (or an exception while producing the data) leaves the old content or the new,
never a torn or empty file. Dependency-free; `atomic_write` is the one entry point.

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
