"""Atomic file writes: write to a sibling temp file, then ``os.replace`` it into place.

A crash (or an exception while producing the data) leaves the old content or the new,
never a torn or empty file. Dependency-free; ``atomic_write`` is the one entry point.

``file_lock`` is its companion: an atomic replace is only safe against concurrent
appenders and readers when they all hold the same advisory lock.

    >>> import os, tempfile
    >>> path = os.path.join(tempfile.mkdtemp(), "f.txt")
    >>> atomic_write(path, b"hello")
    >>> open(path).read(), os.listdir(os.path.dirname(path))
    ('hello', ['f.txt'])
    >>> atomic_write(path, (c.encode() for c in "ab"))  # chunks are streamed
    >>> open(path).read()
    'ab'
"""

import errno
import os
import stat
import threading
import time
import uuid
import weakref
from collections.abc import Iterable
from contextlib import contextmanager

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
    and on Windows within one volume. Hard links to the old file are detached, and its owner is not preserved.

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


# ---------------------------------------------------------------------------
# file_lock

DFLT_LOCK_SIDECAR = ".lock"
DFLT_POLL_INTERVAL = 0.001  # seconds between attempts on a contended lock (Windows)

_LOCK_UNSUPPORTED = {  # e.g. Lustre without its flock option, some FUSE/SMB mounts
    getattr(errno, name)
    for name in ("ENOLCK", "ENOTSUP", "EOPNOTSUPP", "ENOSYS")
    if hasattr(errno, name)
}

if os.name == "nt":
    import msvcrt

    def _lock(fh, poll_interval: float) -> None:
        # not LK_LOCK: it sleeps 1s on any contention and raises after ~10s.
        # No timeout: Windows releases a dead process's locks.
        fh.seek(0)
        while True:
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as e:
                if e.errno not in (errno.EACCES, errno.EDEADLOCK):
                    raise
            time.sleep(poll_interval)

    def _unlock(fh) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(fh, poll_interval: float) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)  # excludes other descriptors too

    def _unlock(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _open_lock_file(path: str):
    """``path`` opened for locking: read-only if it exists but is not writable (both
    ``flock`` and ``msvcrt.locking`` work on a read-only handle); ``None`` if the
    folder is read-only, so locking is unavailable. Any other failure (missing folder,
    too many open files...) is a real error and propagates."""
    try:
        return open(path, "a+b")
    except PermissionError:
        try:
            return open(path, "rb")
        except OSError:
            return None
    except OSError as e:
        if e.errno == errno.EROFS:
            return None
        raise


_live_locks = weakref.WeakSet()


def _renew_in_child() -> None:
    # after a fork the child must not use, nor release, what the parent holds
    for lock in list(_live_locks):
        lock._thread_lock = threading.Lock()
        lock._owner = None  # the parent's section; __exit__ in the child is a no-op
        lock.held = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_renew_in_child)


class FileLock:
    """A reusable, exclusive, advisory lock on a sidecar file ``<path><sidecar>``.

    Use it as ``with lock:``; the body runs either way, and ``lock.held`` says whether
    the lock is actually held (it is not when the lock folder is read-only or the
    filesystem does not support locking: advisory locks are best effort, and the caller
    decides whether that matters).

    The handle is kept open between uses (cheaper for many short sections); ``close()``
    it when done. It is fork-safe: ``flock`` locks belong to the open file description,
    so a forked child that inherited the handle would not exclude its parent; the pid
    is remembered and the handle reopened on a mismatch (and a child never releases
    its parent's lock). Threads of one process exclude each other too. Not reentrant:
    entering again from the holding thread raises ``RuntimeError`` (a second lock on
    the same path from the same thread would deadlock). For a one-off section, use
    :func:`file_lock`.

    The lock file is a separate sidecar, not ``path`` itself, so that
    :func:`atomic_write` replacing ``path`` does not swap the file being locked.

    >>> import tempfile
    >>> lock = FileLock(os.path.join(tempfile.mkdtemp(), "data.json"))
    >>> with lock:
    ...     lock.held
    True
    >>> lock.close()
    """

    def __init__(
        self,
        path,
        *,
        sidecar: str = DFLT_LOCK_SIDECAR,
        poll_interval: float = DFLT_POLL_INTERVAL,
    ) -> None:
        """
        :param path: The file being protected; the lock lives next to it.
        :param sidecar: Non-empty suffix of the lock file.
        :param poll_interval: Seconds between attempts on a contended lock (Windows
            only; POSIX blocks in the kernel).
        """
        if not sidecar:
            raise ValueError(
                "sidecar must be non-empty: the lock file cannot be `path`"
            )
        self.path = os.fspath(path) + sidecar
        self.poll_interval = poll_interval
        self.held = False
        self._fh = None
        self._closer = None  # weakref.finalize closing the handle on collection
        self._pid = None  # the process the handle belongs to
        self._owner = None  # thread currently inside the lock
        self._thread_lock = threading.Lock()
        _live_locks.add(self)

    def _ensure_handle(self) -> None:
        if self._pid != os.getpid():
            self._fh = self._closer = None  # inherited over a fork: leave it be
            self._pid = os.getpid()
        if self._fh is None:
            self._fh = _open_lock_file(self.path)
            if self._fh is not None:
                self._closer = weakref.finalize(self, self._fh.close)

    def _try_lock(self) -> bool:
        if self._fh is None:
            return False
        try:
            _lock(self._fh, self.poll_interval)
            return True
        except OSError as e:
            readonly = "+" not in self._fh.mode
            # NFS emulates flock with fcntl locks, refused on a read-only descriptor
            if e.errno in _LOCK_UNSUPPORTED or (readonly and e.errno == errno.EBADF):
                return False
            raise

    def __enter__(self) -> "FileLock":
        if self._owner == threading.get_ident():
            raise RuntimeError(f"FileLock({self.path!r}) is not reentrant")
        self._thread_lock.acquire()
        try:
            self._ensure_handle()
            self.held = self._try_lock()
            self._owner = threading.get_ident()
        except BaseException:
            self._thread_lock.release()
            raise
        return self

    def __exit__(self, *exc) -> None:
        if self._owner != threading.get_ident():
            return  # a forked child leaving the section its parent entered
        try:
            if self.held:
                self.held = False
                _unlock(self._fh)
        finally:
            self._owner = None
            self._thread_lock.release()

    def close(self) -> None:
        """Release the kept handle (not from inside the ``with`` block)."""
        if self._owner == threading.get_ident():
            raise RuntimeError("close() called while holding the lock")
        with self._thread_lock:
            if self._closer is not None:
                self._closer()  # closes the handle, once
            self._fh = self._closer = None


@contextmanager
def file_lock(
    path,
    *,
    sidecar: str = DFLT_LOCK_SIDECAR,
    poll_interval: float = DFLT_POLL_INTERVAL,
):
    """Hold an exclusive advisory lock on ``<path><sidecar>``; yields whether it is held.

    Cross-process and cross-platform (``flock`` on POSIX, ``msvcrt.locking`` on
    Windows). Take it around every append, read and :func:`atomic_write` of ``path``
    so they cannot interleave. The body still runs when locking is unavailable
    (read-only folder, filesystem without lock support): the yielded value is then
    ``False``. A lock file that exists but is not writable is opened read-only.
    Nesting it on the same path in one thread deadlocks; use one section.

    >>> import tempfile
    >>> path = os.path.join(tempfile.mkdtemp(), "data.json")
    >>> with file_lock(path) as held:
    ...     atomic_write(path, b"{}")
    >>> held, os.path.exists(path + ".lock")
    (True, True)
    """
    lock = FileLock(path, sidecar=sidecar, poll_interval=poll_interval)
    try:
        with lock:
            yield lock.held
    finally:
        lock.close()
