"""Tests for ``dol.file_lock`` / ``dol.FileLock``."""

import multiprocessing as mp
import os
import stat
import sys
import threading
import time

import pytest

from dol import FileLock, atomic_write, file_lock

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
