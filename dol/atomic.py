"""Atomic file writes: write to a sibling temp file, then ``os.replace`` it into place.

A crash (or an exception while producing the data) leaves the old content or the new,
never a torn or empty file. Dependency-free; ``atomic_write`` is the one entry point.

    >>> import os, tempfile
    >>> path = os.path.join(tempfile.mkdtemp(), "f.txt")
    >>> atomic_write(path, b"hello")
    >>> open(path).read(), os.listdir(os.path.dirname(path))
    ('hello', ['f.txt'])
    >>> atomic_write(path, (c.encode() for c in "ab"))  # chunks are streamed
    >>> open(path).read()
    'ab'
"""

import os
import stat
import uuid
from collections.abc import Iterable

_BYTES_LIKE = (bytes, bytearray, memoryview)


def _fsync_dir(dirpath: str) -> None:
    """Best-effort fsync of a directory, so a rename survives power loss (POSIX only)."""
    try:
        fd = os.open(dirpath or ".", os.O_RDONLY)
    except OSError:  # e.g. Windows cannot open directories
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _mode_to_apply(path: str, perms: "int | None") -> "int | None":
    """``perms`` if given, else the mode of the file being replaced (``None`` if new)."""
    if perms is not None:
        return perms
    try:
        return stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        return None


def atomic_write(
    path,
    data: "bytes | Iterable[bytes]",
    *,
    fsync: bool = False,
    perms: "int | None" = None,
) -> None:
    """Write ``data`` to ``path`` atomically (temp file in the same folder, then ``os.replace``).

    :param path: Target file; its parent folder must exist. A symlink is followed:
        its target is what gets replaced.
    :param data: Bytes-like, or an iterable of bytes-like chunks (written one at a
        time, so a large payload need not be held in memory). If the iterable raises,
        ``path`` is untouched. Text must be encoded first.
    :param fsync: Flush the file (and, on POSIX, the folder) to disk before returning:
        durable across power loss, not just across a process crash. Slower. (On macOS
        ``os.fsync`` is not ``F_FULLFSYNC``: best effort.)
    :param perms: Mode (e.g. ``0o600``) for the file, set at creation so the data is
        never readable more widely, even briefly, and not narrowed by the umask.
        Default: the mode of the file being replaced, else the usual umask mode.

    The temp file is a hidden, uniquely named sibling (``.<name>.<random>.tmp``), created
    with ``O_EXCL``, so concurrent writers never share one and key listings skip it.
    On any failure it is removed and the old content is untouched. Atomic on POSIX,
    and on Windows within one volume. Hard links to the old file are detached.

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
    """
    if isinstance(data, str):
        raise TypeError(
            "atomic_write takes bytes (or chunks of bytes); encode text first"
        )
    path = os.path.realpath(os.fspath(path))
    chunks = [data] if isinstance(data, _BYTES_LIKE) else data
    dirname, name = os.path.split(path)
    tmp = os.path.join(dirname, f".{name[:100]}.{uuid.uuid4().hex[:8]}.tmp")
    mode = _mode_to_apply(path, perms)
    try:
        fd = os.open(
            tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666 if mode is None else mode
        )
        with os.fdopen(fd, "wb") as fh:
            if mode is not None and hasattr(os, "fchmod"):
                os.fchmod(fh.fileno(), mode)  # exact, not narrowed by the umask
            for chunk in chunks:
                fh.write(chunk)
            if fsync:
                fh.flush()
                os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if fsync:
        _fsync_dir(dirname)
