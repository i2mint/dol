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


def atomic_write(
    path,
    data: "bytes | Iterable[bytes]",
    *,
    fsync: bool = False,
    perms: "int | None" = None,
    tmp_suffix: str = ".tmp",
    exclusive: bool = True,
) -> None:
    """Write ``data`` to ``path`` atomically (temp file in the same folder, then ``os.replace``).

    :param path: Target file; its parent folder must exist.
    :param data: Bytes-like, or an iterable of bytes-like chunks (written one at a
        time, so a large payload need not be held in memory). If the iterable raises,
        ``path`` is untouched.
    :param fsync: Flush the file (and, on POSIX, the folder) to disk before returning:
        durable across power loss, not just across a process crash. Slower.
    :param perms: Mode (e.g. ``0o600``) for the new file. Default: the usual umask
        permissions. A mode given here is applied exactly, not narrowed by the umask.
    :param tmp_suffix: Suffix of the temp file's name.
    :param exclusive: If true (default), the temp file is a uniquely named sibling
        (``<name>.<random><tmp_suffix>``) created with ``O_EXCL``, so concurrent
        writers never touch each other's temp file. If false, it is the fixed name
        ``<name><tmp_suffix>`` (predictable, but concurrent writers can interleave).

    On any failure the temp file is removed and the old content is untouched.
    Atomic on POSIX, and on Windows within one volume.

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
    path = os.fspath(path)
    chunks = [data] if isinstance(data, _BYTES_LIKE) else data
    tmp = (
        f"{path}.{uuid.uuid4().hex[:8]}{tmp_suffix}"
        if exclusive
        else f"{path}{tmp_suffix}"
    )
    try:
        with open(tmp, "xb" if exclusive else "wb") as fh:
            for chunk in chunks:
                fh.write(chunk)
            if perms is not None:
                os.chmod(tmp, perms)
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
        _fsync_dir(os.path.dirname(path))
