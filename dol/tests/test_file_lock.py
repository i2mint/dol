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


def _append_n(path, n):
    for _ in range(n):
        with file_lock(path):
            with open(path, "ab") as fh:
                fh.write(b"x")


def test_creates_sidecar_and_reports_held(tmp_path):
    p = tmp_path / "f"
    with file_lock(p) as held:
        assert held is True
    assert (tmp_path / "f.lock").exists()
    with file_lock(p, sidecar="") as held:  # locks the file itself
        assert held is True


def test_processes_exclude_each_other(tmp_path):
    p = str(tmp_path / "f")
    procs = [mp.Process(target=_append_n, args=(p, 50)) for _ in range(4)]
    [pr.start() for pr in procs]
    [pr.join(60) for pr in procs]
    assert all(pr.exitcode == 0 for pr in procs)
    assert os.path.getsize(p) == 200


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


def test_unopenable_lock_file_does_not_block(tmp_path):
    with file_lock(tmp_path / "nodir" / "f") as held:
        assert held is False


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
    assert waited >= 0.3  # a shared (inherited) lock would have let it straight in
