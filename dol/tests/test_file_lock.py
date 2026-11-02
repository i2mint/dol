"""Tests for ``dol.file_lock`` / ``dol.FileLock``."""

import multiprocessing as mp
import os
import stat
import sys
import threading
import time

import pytest

from dol import FileLock, LockedAppender, atomic_write, file_lock

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")


def _increment_n(path, n):
    for _ in range(n):
        with file_lock(path):  # read-modify-write: only correct if truly exclusive
            try:
                count = int(open(path, "rb").read() or 0)
            except FileNotFoundError:
                count = 0
            atomic_write(path, str(count + 1).encode())


def test_creates_sidecar_and_reports_held(tmp_path):
    p = tmp_path / "f"
    with file_lock(p) as held:
        assert held is True
    assert (tmp_path / "f.lock").exists()
    with pytest.raises(ValueError):
        FileLock(p, sidecar="")


def test_bytes_path(tmp_path):
    with file_lock(os.fsencode(tmp_path / "f")) as held:
        assert held is True
    assert (tmp_path / "f.lock").exists()
    assert FileLock(os.fsencode(tmp_path / "g"), sidecar=b".lk").path.endswith("g.lk")


def test_processes_exclude_each_other(tmp_path):
    p = str(tmp_path / "f")
    procs = [mp.Process(target=_increment_n, args=(p, 50)) for _ in range(4)]
    [pr.start() for pr in procs]
    [pr.join(60) for pr in procs]
    assert all(pr.exitcode == 0 for pr in procs)
    assert int(open(p, "rb").read()) == 200


def test_threads_share_one_filelock(tmp_path):
    lock = FileLock(tmp_path / "f")
    inside, overlaps = [0], [0]

    def work():
        for _ in range(20):
            with lock:
                inside[0] += 1
                overlaps[0] += inside[0] > 1
                time.sleep(0.0005)
                inside[0] -= 1

    ts = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    lock.close()
    assert overlaps[0] == 0


def test_lock_released_on_exception(tmp_path):
    lock = FileLock(tmp_path / "f")
    with pytest.raises(ValueError):
        with lock:
            raise ValueError
    with lock:
        assert lock.held


def test_works_with_atomic_write(tmp_path):
    p = tmp_path / "f"
    with file_lock(p):
        atomic_write(p, b"ok")
    assert p.read_bytes() == b"ok"


@posix_only
def test_readonly_lock_file_still_locks(tmp_path):
    p = tmp_path / "f"
    lock_path = tmp_path / "f.lock"
    lock_path.write_bytes(b"")
    lock_path.chmod(stat.S_IRUSR)
    if os.access(lock_path, os.W_OK):  # running as root
        pytest.skip("cannot make a file unwritable for this user")
    with file_lock(p) as held:
        assert held is True


def test_missing_folder_is_an_error_not_a_silent_no_lock(tmp_path):
    with pytest.raises(FileNotFoundError):
        with file_lock(tmp_path / "nodir" / "f"):
            pass


def test_reentry_raises_instead_of_deadlocking(tmp_path):
    lock = FileLock(tmp_path / "f")
    with lock:
        with pytest.raises(RuntimeError, match="reentrant"):
            with lock:
                pass
    lock.close()


@posix_only
def test_fork_child_waits_for_parents_lock(tmp_path):
    lock = FileLock(tmp_path / "f")
    with lock:  # the child inherits this very object, handle included
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                t0 = time.monotonic()
                with lock:
                    os.write(w, b"%.3f" % (time.monotonic() - t0))
            finally:
                os._exit(0)
        time.sleep(0.4)
    waited = float(os.read(r, 16))
    os.waitpid(pid, 0)
    lock.close()
    assert waited >= 0.3  # a shared (inherited) lock would have let it straight in


@posix_only
def test_forked_child_leaving_the_block_keeps_parents_lock(tmp_path):
    import fcntl

    lock = FileLock(tmp_path / "f")
    with lock:
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                lock.__exit__(None, None, None)  # the child leaves the inherited block
                code = 0
            finally:
                os._exit(code)
        assert os.waitpid(pid, 0)[1] == 0  # the child exited cleanly
        with open(lock.path, "rb") as fh:
            with pytest.raises(OSError):
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.close()


def test_close_inside_the_block_raises(tmp_path):
    lock = FileLock(tmp_path / "f")
    with lock:
        with pytest.raises(RuntimeError, match="close"):
            lock.close()
    lock.close()


# ---------------------------------------------------------------------------
# LockedAppender (#165)


@pytest.mark.parametrize("keep_handle", [True, False])
def test_appender_follows_a_replaced_file(tmp_path, keep_handle):
    p = tmp_path / "log"
    with LockedAppender(p, keep_handle=keep_handle) as log:
        log.append(b"a\n")
        with log.lock:
            atomic_write(p, b"compacted\n")
        log.append(b"b\n")
    assert p.read_bytes() == b"compacted\nb\n"


def test_appender_recreates_a_deleted_file(tmp_path):
    p = tmp_path / "log"
    with LockedAppender(p, keep_handle=True) as log:
        log.append(b"a")
        with log.lock:
            os.unlink(p)
        log.append(b"b")
    assert p.read_bytes() == b"b"


def test_appender_keeps_the_handle_only_when_asked(tmp_path):
    p = tmp_path / "log"
    with LockedAppender(p, keep_handle=True) as log:
        log.append(b"a")
        assert log._fd is not None
    assert log._fd is None
    with LockedAppender(p, keep_handle=False) as log:
        log.append(b"b")
        assert log._fd is None
    assert p.read_bytes() == b"ab"


@pytest.mark.parametrize(
    "existing, expected",
    [
        (b"", b"x\n"),
        (b"ok\n", b"ok\nx\n"),
        (b"torn", b"torn\nx\n"),
    ],
)
def test_line_end_heals_a_torn_last_record(tmp_path, existing, expected):
    p = tmp_path / "log"
    p.write_bytes(existing)
    with LockedAppender(p, line_end=b"\n") as log:
        log.append(b"x\n")
    assert p.read_bytes() == expected


@pytest.mark.parametrize(
    "existing, expected",
    [
        (b"x", b"x\r\nr\r\n"),  # shorter than line_end
        (b"ab\r", b"ab\r\nr\r\n"),  # a torn terminator is completed, not doubled
        (b"ab\r\n", b"ab\r\nr\r\n"),
    ],
)
def test_multibyte_line_end(tmp_path, existing, expected):
    p = tmp_path / "log"
    p.write_bytes(existing)
    with LockedAppender(p, line_end=b"\r\n") as log:
        log.append(b"r\r\n")
    assert p.read_bytes() == expected


def test_a_kept_handle_heals_a_tail_torn_by_another_writer(tmp_path):
    p = tmp_path / "log"
    with LockedAppender(p, line_end=b"\n", keep_handle=True) as a:
        a.append(b"a\n")
        with LockedAppender(p) as b:  # no line_end: leaves a torn record
            b.append(b"torn")
        a.append(b"x\n")
    assert p.read_bytes() == b"a\ntorn\nx\n"


@pytest.mark.parametrize("keep_handle", [True, False])
def test_a_failed_write_does_not_wedge_the_appender(tmp_path, monkeypatch, keep_handle):
    import errno

    p = tmp_path / "log"
    real_write = os.write
    fail = [True]

    def write(fd, data):
        if fail[0]:  # a short write, then the disk fills up
            fail[0] = False
            real_write(fd, bytes(data[:2]))
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_write(fd, data)

    with LockedAppender(p, line_end=b"\n", keep_handle=keep_handle) as log:
        log.append(b"a\n")
        monkeypatch.setattr(os, "write", write)
        with pytest.raises(OSError):
            log.append(b"12345\n")
        assert log._fd is None  # dropped: the next append reopens
        log.append(b"next\n")
    assert p.read_bytes() == b"a\n12\nnext\n"  # nothing pending lands later


def test_without_line_end_bytes_are_appended_as_given(tmp_path):
    p = tmp_path / "log"
    p.write_bytes(b"torn")
    with LockedAppender(p) as log:
        log.append(b"x")
    assert p.read_bytes() == b"tornx"


def test_appender_uses_the_injected_lock(tmp_path):
    p = tmp_path / "log"
    lock = FileLock(p, sidecar=".mylock")
    with LockedAppender(p, lock=lock) as log:
        assert log.lock is lock
        log.append(b"a")
    assert (tmp_path / "log.mylock").exists()
    assert not (tmp_path / "log.lock").exists()


def test_append_waits_for_the_lock(tmp_path):
    p = tmp_path / "log"
    log = LockedAppender(p)
    done = []
    with file_lock(p):  # a separate FileLock: another writer, e.g. a compaction
        t = threading.Thread(target=lambda: (log.append(b"a"), done.append(1)))
        t.start()
        time.sleep(0.2)
        assert done == [] and not p.exists()
    t.join(10)
    log.close()
    assert done == [1] and p.read_bytes() == b"a"


def _append_n(path, n, start):
    start.wait()
    with LockedAppender(path, line_end=b"\n") as log:
        for i in range(n):
            log.append(b"%d\n" % i)


def _compact_n(path, n, start):
    start.wait()
    for _ in range(n):
        with file_lock(path):
            lines = open(path, "rb").read().splitlines(keepends=True)
            atomic_write(path, b"".join(lines))  # same content, new file


def test_no_append_is_lost_across_concurrent_compactions(tmp_path):
    p = str(tmp_path / "log")
    open(p, "wb").close()
    start = mp.Barrier(4)  # all start together, so appends and compactions overlap
    procs = [mp.Process(target=_append_n, args=(p, 100, start)) for _ in range(3)]
    procs.append(mp.Process(target=_compact_n, args=(p, 30, start)))
    [pr.start() for pr in procs]
    [pr.join(60) for pr in procs]
    assert all(pr.exitcode == 0 for pr in procs)
    lines = open(p, "rb").read().splitlines()
    assert sorted(lines) == sorted([b"%d" % i for i in range(100)] * 3)
