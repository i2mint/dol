"""Atomic file writes: write to a sibling temp file, then ``os.replace`` it into place.

A crash (or an exception while producing the data) leaves the old content or the new,
never a torn or empty file. Dependency-free; ``atomic_write`` is the entry point, and
``atomic_delete`` its counterpart for removal. A write killed mid-way leaves its temp
file behind; ``interrupted_writes`` lists them (``TEMP_NAME_PATTERN`` is their name).

On Windows, a reader holding the destination open makes ``os.replace`` and ``os.unlink``
fail with ``PermissionError`` until it closes (Python's ``open`` does not share delete).
Both functions therefore retry that briefly, through a :class:`Retry` policy (its
``retry`` argument): by default Windows' ``PermissionError`` only, ~0.6 s in all.
Elsewhere the same error is a real permission problem, so nothing is retried.

``file_lock`` is its companion: an atomic replace is only safe against concurrent
appenders and readers when they all hold the same advisory lock. ``LockedAppender``
appends under that lock and follows the file when an atomic replace swaps it.

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
import json
import os
import re
import stat
import threading
import time
import uuid
import warnings
import weakref
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass

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


#: What :data:`DFLT_RETRY` retries: Windows' sharing violation (another handle has the
#: file open); nothing elsewhere, where ``PermissionError`` means a real permission problem.
#: Windows reports a sharing violation on ``os.replace`` with the same error as a genuine
#: access denial, so there *every* ``PermissionError`` is retried: a read-only destination
#: still fails, only ~0.6 s later.
SHARING_VIOLATIONS: "tuple[type[BaseException], ...]" = (
    (PermissionError,) if os.name == "nt" else ()
)


@dataclass(frozen=True)
class Retry:
    """Call ``op(*args)``, retrying on ``on`` exceptions with doubling backoff.

    ``attempts`` calls at most, sleeping ``backoff_s``, then twice that, and so on
    between them (defaults: 7 calls, ~0.6 s of sleep in all; the backoff is not capped,
    so keep ``attempts`` small). The last failure is re-raised. Any other exception
    propagates at once. ``Retry(attempts=1)`` (or ``on=()``) calls ``op`` once.

    >>> calls = []
    >>> def flaky():
    ...     calls.append(1)
    ...     if len(calls) < 3:
    ...         raise PermissionError("in use")
    ...     return "done"
    >>> Retry(on=(PermissionError,), backoff_s=0)(flaky), len(calls)
    ('done', 3)
    """

    on: "tuple[type[BaseException], ...]" = SHARING_VIOLATIONS
    attempts: int = 7
    backoff_s: float = 0.01

    def __post_init__(self):
        # checked here, not at the first contended call, where a bad `on` would mask
        # the real error with a TypeError
        on = (self.on,) if isinstance(self.on, type) else tuple(self.on)
        if not all(isinstance(e, type) and issubclass(e, BaseException) for e in on):
            raise TypeError(f"on must be exception classes, got {self.on!r}")
        object.__setattr__(self, "on", on)
        if not isinstance(self.attempts, int) or self.attempts < 1:
            raise ValueError(f"attempts must be an int >= 1, got {self.attempts!r}")
        if not self.backoff_s >= 0:  # also refuses NaN
            raise ValueError(f"backoff_s must be >= 0, got {self.backoff_s!r}")

    def __call__(self, op: Callable, *args):
        for attempt in range(self.attempts):
            try:
                return op(*args)
            except self.on:
                if attempt == self.attempts - 1:
                    raise
                time.sleep(self.backoff_s * 2**attempt)


DFLT_RETRY = Retry()


#: Most bytes of the target's name kept in its temp file's name, leaving room for the
#: random suffix within the usual 255-byte name limit.
_TMP_NAME_PREFIX_BYTES = 100


def _clip_to_bytes(name: str) -> str:
    """``name`` cut, on a character boundary, to at most :data:`_TMP_NAME_PREFIX_BYTES`
    encoded bytes (filesystems limit names in bytes; a character can take several)."""
    name = name[:_TMP_NAME_PREFIX_BYTES]  # a character is at least one byte
    while len(os.fsencode(name)) > _TMP_NAME_PREFIX_BYTES:
        name = name[:-1]
    return name


def temp_name(name: str) -> str:
    """The name :func:`atomic_write` stages a write to ``name`` under: a hidden sibling
    ``.<name, clipped to 100 bytes>.<8 random hex>.tmp``, always matched by
    :data:`TEMP_NAME_PATTERN`. Unique per call. ``name`` is a file name, not a path.

    >>> bool(TEMP_NAME_PATTERN.fullmatch(temp_name("data.json")))
    True
    """
    if not name or os.sep in name or (os.altsep and os.altsep in name):
        raise ValueError(f"temp_name takes a non-empty file name, not {name!r}")
    return f".{_clip_to_bytes(name)}.{uuid.uuid4().hex[:8]}.tmp"


#: Matches (``fullmatch``) every name :func:`temp_name` makes, and so the temp file a
#: killed :func:`atomic_write` leaves behind. ``DOTALL``: a file name may hold a newline.
TEMP_NAME_PATTERN = re.compile(r"\..+\.[0-9a-f]{8}\.tmp", re.DOTALL)


def interrupted_writes(
    folder,
    *,
    pattern: "re.Pattern | str" = TEMP_NAME_PATTERN,
    recursive: bool = False,
) -> "list[str]":
    r"""The temp files writes killed mid-way left in ``folder``: sorted absolute paths
    (``str``) of the regular files (never a symlink or folder) whose name fully
    matches ``pattern`` (default: :func:`atomic_write`'s temps; pass another writer's,
    e.g. ``r"\.blob-.+\.part"``).

    Matching is by name only: call it on folders your own stores write, where nothing
    else is named like a temp. A listed file may also be a write still in progress in
    another process: delete only those older than any write could take
    (``os.stat(p).st_mtime``), or while holding the lock every writer takes.

    :param recursive: Also search subfolders (symlinked folders are not followed;
        subfolders that cannot be read, or vanish, are skipped).

    A missing ``folder`` holds none (``[]``); any other failure to list it raises
    ``OSError``.

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as d:
    ...     for name in ("k.mp4", ".k.mp4.ab12cd34.tmp", ".DS_Store", "x.tmp"):
    ...         open(os.path.join(d, name), "wb").close()
    ...     [os.path.basename(p) for p in interrupted_writes(d)]
    ['.k.mp4.ab12cd34.tmp']
    """
    pattern = re.compile(pattern, re.DOTALL) if isinstance(pattern, str) else pattern

    def scan(dirpath: str, *, top: bool):
        try:
            entries = os.scandir(dirpath)
        except FileNotFoundError:
            return
        except (PermissionError, NotADirectoryError):
            if top:
                raise
            return  # an unreadable subfolder (other errors, e.g. EMFILE, propagate)
        with entries:
            for e in entries:
                if e.is_file(follow_symlinks=False):
                    if pattern.fullmatch(e.name):
                        yield os.path.abspath(e.path)
                elif recursive and e.is_dir(follow_symlinks=False):
                    yield from scan(e.path, top=False)

    return sorted(scan(os.fsdecode(folder), top=True))


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
    retry: Callable = DFLT_RETRY,
) -> None:
    """Write ``data`` to ``path`` atomically (temp file in the same folder, then ``os.replace``).

    :param path: Target file (``str``, ``bytes`` or path-like); its parent folder must
        exist. A symlink is followed: its target is what gets replaced.
    :param data: Bytes-like, or an iterable of bytes-like chunks (written one at a
        time, so a large payload need not be held in memory). If the iterable raises,
        ``path`` is untouched. Text must be encoded first.
    :param fsync: Flush the file (and, on POSIX, the folder) to disk before returning:
        durable across power loss, not just across a process crash. Slower. (On macOS
        ``os.fsync`` is not ``F_FULLFSYNC``: best effort.)
    :param perms: Mode (e.g. ``0o600``) for the file, set at creation so the data is
        never readable more widely, even briefly, and not narrowed by the umask.
        Default: the mode of the file being replaced, else the usual umask mode.
    :param retry: How the final ``os.replace`` is called: ``retry(os.replace, tmp,
        path)``. Default :data:`DFLT_RETRY` retries Windows' sharing violations (a
        reader has ``path`` open); ``Retry(attempts=1)`` calls it once.

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
    path = os.path.realpath(os.fsdecode(path))  # str, so the temp name joins
    chunks = [data] if isinstance(data, _BYTES_LIKE) else data
    dirname, name = os.path.split(path)
    tmp = os.path.join(dirname, temp_name(name))
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
        retry(os.replace, tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if fsync:
        _fsync_dir(dirname)


def atomic_delete(path, *, retry: Callable = DFLT_RETRY) -> None:
    """Remove the file ``path``: ``retry(os.unlink, path)``, the delete counterpart
    of :func:`atomic_write`.

    ``os.unlink`` is already atomic; what this adds is the same retry, so a delete is
    not refused on Windows because a reader has the file open. (Windows may then keep
    the name "delete pending" until the last handle closes.) A symlink is removed
    itself, not its target. A missing file raises ``FileNotFoundError`` (not retried).

    >>> import tempfile
    >>> path = os.path.join(tempfile.mkdtemp(), "f")
    >>> atomic_write(path, b"x")
    >>> atomic_delete(path)
    >>> os.path.exists(path)
    False
    """
    retry(os.unlink, os.fspath(path))


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
        self.path = os.fsdecode(path) + os.fsdecode(sidecar)  # bytes paths too
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


# ---------------------------------------------------------------------------
# LockedAppender

#: Whether a :class:`LockedAppender` keeps its file open between appends. Not on
#: Windows: an open handle there makes the ``os.replace`` of a compaction fail.
DFLT_KEEP_HANDLE = os.name != "nt"

_APPEND_FLAGS = os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)


def _is_same_file(fd: int, path: str) -> bool:
    """Whether the open descriptor ``fd`` is still the file at ``path``."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return False
    fst = os.fstat(fd)
    return (st.st_ino, st.st_dev) == (fst.st_ino, fst.st_dev)


def _missing_end(fd: int, line_end: bytes) -> bytes:
    """What the file ``fd`` lacks to end with ``line_end``: ``b""`` when it is empty or
    ends with it; the rest of it when it ends with a part of it (a torn ``\\r`` of
    ``\\r\\n`` gets just ``\\n``); else all of it."""
    size = os.fstat(fd).st_size
    if not size:
        return b""
    n = min(len(line_end), size)
    os.lseek(fd, size - n, os.SEEK_SET)  # O_APPEND: writes still go to the end
    tail = os.read(fd, n)
    k = next(k for k in range(n, -1, -1) if tail.endswith(line_end[:k]))
    return line_end[k:]


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data).cast("B")  # os.write counts bytes, not items
    while view:
        view = view[os.write(fd, view) :]


class LockedAppender:
    r"""Appends to ``path`` under a :class:`FileLock`, following the file when an
    :func:`atomic_write` (made under the same lock, e.g. a compaction) replaces it.

    Each append is one unbuffered ``O_APPEND`` write, so nothing is pending between
    appends: a failed append leaves nothing to land later, and an inherited handle
    (after a fork) is safe to share. The handle is kept between appends
    (``keep_handle``) and reopened when the file at ``path`` is no longer the one it
    holds, or after a failed append.

    To replace the file, hold the appender's lock, so no append lands in the old file
    after it was read: ``with appender.lock: atomic_write(path, compacted)``. Whether
    the lock is actually held there is ``appender.lock.held`` (see :class:`FileLock`).
    ``with appender:`` does not lock; it only closes on exit. Atomic appends and tail
    healing assume the lock is held by every writer: when it cannot be (read-only
    folder, filesystem without locks), appends are best effort.

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
    """

    def __init__(
        self,
        path,
        *,
        lock: "FileLock | None" = None,
        line_end: "bytes | None" = None,
        keep_handle: bool = DFLT_KEEP_HANDLE,
    ) -> None:
        """
        :param path: The file appended to (created if missing; its folder must exist).
        :param lock: The lock to append under. Default ``FileLock(path)``: the same
            sidecar every :func:`file_lock` of ``path`` uses.
        :param line_end: Record terminator. When given, a file whose last record is
            torn (a crash or a failed write, by any writer) is completed with it
            before the next append, so the torn record cannot swallow that append.
            ``None``: bytes are appended as given.
        :param keep_handle: Keep the file open between appends (fewer opens). Leave it
            off on Windows, where the open handle makes replacing the file fail, and
            on NFS, where attribute caching can hide a replacement.
        """
        self.path = os.fsdecode(os.fspath(path))
        self.lock = FileLock(self.path) if lock is None else lock
        self.line_end = line_end
        self.keep_handle = keep_handle
        self._fd = None
        self._closer = None  # weakref.finalize closing the fd if we are never closed

    def append(self, data: bytes) -> None:
        """Append ``data`` under the lock, to the file now at ``path``."""
        with self.lock:
            try:
                if self._fd is not None and not _is_same_file(self._fd, self.path):
                    self._close_fd()  # replaced since the last append
                if self._fd is None:
                    self._open()
                if self.line_end:
                    _write_all(self._fd, _missing_end(self._fd, self.line_end))
                _write_all(self._fd, data)
            except BaseException:
                self._close_fd()  # the next append reopens and heals the tail
                raise
            if not self.keep_handle:
                self._close_fd()

    def _open(self) -> None:
        access = os.O_RDWR if self.line_end else os.O_WRONLY  # healing reads the tail
        self._fd = os.open(self.path, _APPEND_FLAGS | access, 0o666)
        self._closer = weakref.finalize(self, os.close, self._fd)

    def _close_fd(self) -> None:
        # forgotten first: a failed close cannot leave us holding a dead fd
        closer, self._closer, self._fd = self._closer, None, None
        if closer is not None:
            closer()  # closes the fd, once

    def close(self) -> None:
        """Close the file and the lock's handle (both reopen on the next append).
        Not from inside ``with appender.lock:`` (the lock is not reentrant)."""
        with self.lock:
            self._close_fd()
        self.lock.close()

    def __enter__(self) -> "LockedAppender":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# update_json: a locked read-modify-write of one JSON file


class InvalidJSONError(ValueError):
    """The file :func:`update_json` was asked to update does not hold valid JSON."""


def update_json(
    path,
    change: Callable,
    *,
    default: Callable = dict,
    loads: Callable = json.loads,
    dumps: Callable = json.dumps,
    lock=None,
    warn_unlocked: bool = True,
):
    """Apply ``change`` to the JSON document at ``path`` under its lock; return what
    ``change`` returned.

    Under :func:`file_lock` of ``path``: read the document (a ``dict`` or ``list``),
    call ``change(obj)``, which mutates ``obj`` in place (and may raise, to write
    nothing), then :func:`atomic_write` it back, only if ``dumps`` of it changed. So a
    no-op neither creates the file nor moves its mtime, and a file nobody changed
    keeps its formatting. What ``change`` returns is handed back, never written.

    The read is strict: only a missing file starts from ``default()``. Any other read
    error propagates, and invalid JSON raises :class:`InvalidJSONError` (a
    ``ValueError``): writing "default plus this change" over a file that could not
    be read would destroy what it held.

    :param path: The JSON file; its folder must exist.
    :param default: Factory for the document when ``path`` does not exist.
    :param loads, dumps: The codec. ``loads`` gets bytes; ``dumps`` may return str or
        bytes (e.g. ``functools.partial(json.dumps, indent=2)``, or orjson's). Change
        is detected by comparing ``dumps`` before and after, so it must be
        deterministic.
    :param lock: The context manager to hold. Default: the :class:`FileLock` every
        :func:`file_lock` of ``path`` takes. Any other works; whether it is held is
        the ``held`` of what it yields, else the truth of what it yields (``None``,
        e.g. ``nullcontext()``, counts as held). Do not call this while already holding the same file's
        lock in this thread: the lock is not reentrant, and that deadlocks.
    :param warn_unlocked: Warn when the lock is not held (read-only folder,
        filesystem without locks), so the loss of the guarantee is not silent.

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
    """
    path = os.fsdecode(path)
    if not os.path.basename(path):
        raise ValueError(f"update_json takes a file path, not {path!r}")
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"update_json: the folder of {path!r} does not exist")
    own_lock = lock is None
    if own_lock:
        lock = FileLock(path)
    try:
        with lock as entered:
            held = getattr(entered, "held", entered is None or bool(entered))
            if not held and warn_unlocked:
                warnings.warn(
                    f"update_json: could not lock {path!r}; "
                    "concurrent updates may be lost",
                    stacklevel=2,  # the caller of update_json
                )
            obj = _read_json_strictly(path, default, loads)
            before = dumps(obj)
            result = change(obj)
            after = dumps(obj)
            if after != before:
                atomic_write(
                    path, after.encode("utf-8") if isinstance(after, str) else after
                )
            return result
    finally:
        if own_lock:
            lock.close()


def _read_json_strictly(path: str, default: Callable, loads: Callable):
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except FileNotFoundError:
        obj = default()
    else:
        try:
            obj = loads(data)
        except ValueError as e:
            raise InvalidJSONError(
                f"{path!r} is not valid JSON; nothing was written"
            ) from e
    if not isinstance(obj, (dict, list)):
        raise TypeError(
            f"update_json changes a dict or list in place, so it cannot update a "
            f"{type(obj).__name__} document ({path!r})"
        )
    return obj
