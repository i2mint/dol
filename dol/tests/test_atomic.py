"""Tests for ``dol.atomic``: ``atomic_write``, ``atomic_delete``, ``Retry``, temp naming."""

import os
import re
import stat
import subprocess
import sys

import pytest

from dol import atomic_delete, atomic_write, interrupted_writes
from dol.atomic import TEMP_NAME_PATTERN, Retry, temp_name

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX mode bits/symlinks"
)


def _mode(p):
    return stat.S_IMODE(os.stat(p).st_mode)


def test_bytes_and_chunks_and_overwrite(tmp_path):
    p = tmp_path / "f"
    atomic_write(p, b"one")
    atomic_write(p, iter([b"t", bytearray(b"w"), memoryview(b"o")]))
    assert p.read_bytes() == b"two"
    assert os.listdir(tmp_path) == ["f"]


def test_bytes_path(tmp_path):
    p = tmp_path / "f"
    atomic_write(os.fsencode(p), b"x")
    assert p.read_bytes() == b"x" and os.listdir(tmp_path) == ["f"]
    atomic_delete(os.fsencode(p))
    assert os.listdir(tmp_path) == []


def test_empty_data(tmp_path):
    atomic_write(tmp_path / "f", iter([]))
    assert (tmp_path / "f").read_bytes() == b""


def test_str_is_refused_clearly(tmp_path):
    with pytest.raises(TypeError, match="encode"):
        atomic_write(tmp_path / "f", "text")
    assert os.listdir(tmp_path) == []


def test_failure_keeps_old_content_and_cleans_tmp(tmp_path):
    p = tmp_path / "f"
    atomic_write(p, b"old")

    def gen():
        yield b"new"
        raise ValueError

    with pytest.raises(ValueError):
        atomic_write(p, gen())
    assert p.read_bytes() == b"old" and os.listdir(tmp_path) == ["f"]


def test_missing_parent_raises_and_leaves_nothing(tmp_path):
    with pytest.raises(OSError):
        atomic_write(tmp_path / "nope" / "f", b"x")
    assert os.listdir(tmp_path) == []


def test_tmp_is_hidden_unique_and_in_same_folder(tmp_path, monkeypatch):
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append(a), real(a, b)))
    atomic_write(tmp_path / "f", b"x")
    atomic_write(tmp_path / "f", b"x")
    assert len(set(seen)) == 2
    for s in seen:
        assert os.path.dirname(s) == os.path.realpath(tmp_path)
        assert os.path.basename(s).startswith(".f.")


def test_long_names_do_not_overflow(tmp_path):
    name = "n" * 240
    atomic_write(tmp_path / name, b"x")
    assert (tmp_path / name).read_bytes() == b"x"


def test_long_non_ascii_names_do_not_overflow(tmp_path, monkeypatch):
    name = "\u6587" * 85  # 85 characters, 255 UTF-8 bytes: a legal name
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append(a), real(a, b)))
    atomic_write(tmp_path / name, b"x")
    assert (tmp_path / name).read_bytes() == b"x"
    tmp_name = os.fsencode(os.path.basename(seen[0]))
    assert len(tmp_name) <= 255  # ext4 refuses longer (ENAMETOOLONG)
    assert tmp_name.decode()  # cut on a character boundary: still valid UTF-8


def test_fsync_only_when_asked(tmp_path, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd)))
    atomic_write(tmp_path / "f", b"x")
    assert calls == []
    atomic_write(tmp_path / "f", b"x", fsync=True)
    assert len(calls) >= 1 and (tmp_path / "f").read_bytes() == b"x"


@posix_only
def test_perms_exact_despite_umask(tmp_path):
    old = os.umask(0o077)
    try:
        atomic_write(tmp_path / "f", b"x", perms=0o644)
    finally:
        os.umask(old)
    assert _mode(tmp_path / "f") == 0o644


@posix_only
def test_perms_set_before_data_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "fchmod", lambda *a: None)  # creation alone must do it
    seen = []

    def gen():
        (tmp,) = [p for p in os.listdir(tmp_path) if p.startswith(".f.")]
        seen.append(_mode(tmp_path / tmp))
        yield b"secret"

    old = os.umask(0)
    try:
        atomic_write(tmp_path / "f", gen(), perms=0o600)
    finally:
        os.umask(old)
    assert seen == [0o600]


@posix_only
def test_existing_mode_preserved(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"old")
    os.chmod(p, 0o600)
    atomic_write(p, b"new")
    assert _mode(p) == 0o600 and p.read_bytes() == b"new"


@posix_only
def test_symlink_is_followed(tmp_path):
    target, link = tmp_path / "target", tmp_path / "link"
    target.write_bytes(b"old")
    link.symlink_to(target)
    atomic_write(link, b"new")
    assert link.is_symlink() and target.read_bytes() == b"new"


# ---------------------------------------------------------------------------
# retry (Windows sharing violations, #167)


def _refuse(n, real, exc=PermissionError):
    """A stand-in for ``real`` that raises ``exc`` on its first ``n`` calls."""
    calls = []

    def op(*args):
        calls.append(args)
        if len(calls) <= n:
            raise exc("file in use")
        return real(*args)

    return op, calls


NO_SLEEP = Retry(on=(PermissionError,), backoff_s=0)


def test_default_retries_only_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr("dol.atomic.time.sleep", lambda s: None)
    op, calls = _refuse(99, os.replace)
    monkeypatch.setattr(os, "replace", op)
    with pytest.raises(PermissionError):
        atomic_write(tmp_path / "f", b"x")  # default retry
    assert len(calls) == (Retry().attempts if os.name == "nt" else 1)
    assert os.listdir(tmp_path) == []


def test_interrupt_during_backoff_keeps_old_and_cleans_tmp(tmp_path, monkeypatch):
    p = tmp_path / "f"
    atomic_write(p, b"old")

    def interrupted(s):
        raise KeyboardInterrupt

    monkeypatch.setattr("dol.atomic.time.sleep", interrupted)
    op, _ = _refuse(99, os.replace)
    monkeypatch.setattr(os, "replace", op)
    with pytest.raises(KeyboardInterrupt):
        atomic_write(p, b"new", retry=Retry(on=(PermissionError,)))
    assert p.read_bytes() == b"old" and os.listdir(tmp_path) == ["f"]


def test_retry_backoff_doubles_and_reraises_the_last(monkeypatch):
    sleeps = []
    monkeypatch.setattr("dol.atomic.time.sleep", sleeps.append)
    op, calls = _refuse(99, lambda: None)
    with pytest.raises(PermissionError):
        Retry(on=(PermissionError,), attempts=4, backoff_s=0.01)(op)
    assert len(calls) == 4 and sleeps == [0.01, 0.02, 0.04]


def test_retry_does_not_catch_other_errors():
    op, calls = _refuse(1, lambda: None, exc=FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        NO_SLEEP(op)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(attempts=0), ValueError),
        (dict(attempts=2.5), ValueError),
        (dict(backoff_s=-1), ValueError),
        (dict(backoff_s=float("nan")), ValueError),
        (dict(on=(PermissionError, "nope")), TypeError),
        (dict(on=(int,)), TypeError),
    ],
)
def test_retry_refuses_bad_config_at_construction(kwargs, error):
    with pytest.raises(error):
        Retry(**kwargs)


def test_retry_on_is_normalized_to_a_tuple():
    assert Retry(on=PermissionError).on == (PermissionError,)
    assert Retry(on=[PermissionError]).on == (PermissionError,)
    hash(Retry(on=[PermissionError]))  # still a hashable value


def test_write_retries_a_refused_replace(tmp_path, monkeypatch):
    p = tmp_path / "f"
    atomic_write(p, b"old")
    op, calls = _refuse(2, os.replace)
    monkeypatch.setattr(os, "replace", op)
    atomic_write(p, b"new", retry=NO_SLEEP)
    assert len(calls) == 3 and p.read_bytes() == b"new"
    assert os.listdir(tmp_path) == ["f"]


def test_write_gives_up_keeps_old_and_cleans_tmp(tmp_path, monkeypatch):
    p = tmp_path / "f"
    atomic_write(p, b"old")
    op, calls = _refuse(99, os.replace)
    monkeypatch.setattr(os, "replace", op)
    with pytest.raises(PermissionError):
        atomic_write(
            p, b"new", retry=Retry(on=(PermissionError,), attempts=3, backoff_s=0)
        )
    assert len(calls) == 3 and p.read_bytes() == b"old"
    assert os.listdir(tmp_path) == ["f"]


def test_single_attempt_calls_replace_once(tmp_path, monkeypatch):
    op, calls = _refuse(1, os.replace)
    monkeypatch.setattr(os, "replace", op)
    with pytest.raises(PermissionError):
        atomic_write(
            tmp_path / "f", b"x", retry=Retry(on=(PermissionError,), attempts=1)
        )
    assert len(calls) == 1 and os.listdir(tmp_path) == []


def test_delete_retries_and_removes(tmp_path, monkeypatch):
    p = tmp_path / "f"
    atomic_write(p, b"x")
    op, calls = _refuse(2, os.unlink)
    monkeypatch.setattr(os, "unlink", op)
    atomic_delete(p, retry=NO_SLEEP)
    assert len(calls) == 3 and not p.exists()


def test_delete_missing_raises_without_retrying(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr("dol.atomic.time.sleep", sleeps.append)
    with pytest.raises(FileNotFoundError):
        atomic_delete(tmp_path / "nope", retry=Retry(on=(PermissionError,)))
    assert sleeps == []


@posix_only
def test_delete_removes_the_link_not_its_target(tmp_path):
    target, link = tmp_path / "target", tmp_path / "link"
    target.write_bytes(b"x")
    link.symlink_to(target)
    atomic_delete(link)
    assert not link.is_symlink() and target.read_bytes() == b"x"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing semantics")
def test_windows_reader_holding_file_open_is_outlasted(tmp_path):
    import threading

    p = tmp_path / "f"
    atomic_write(p, b"old")
    fh = open(p, "rb")  # no FILE_SHARE_DELETE: replace/unlink are refused
    threading.Timer(0.05, fh.close).start()
    atomic_write(p, b"new")
    assert p.read_bytes() == b"new"
    fh = open(p, "rb")
    threading.Timer(0.05, fh.close).start()
    atomic_delete(p)
    assert not p.exists()


# ---------------------------------------------------------------------------
# temp naming and interrupted_writes (#171, #174)


def test_atomic_write_stages_under_temp_name(tmp_path, monkeypatch):
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append(a), real(a, b)))
    names = ["f", "文" * 85, "x.tmp"] + (["a\nb"] if os.name != "nt" else [])
    for name in names:
        atomic_write(tmp_path / name, b"x")
    assert len(seen) == len(names)
    assert all(TEMP_NAME_PATTERN.fullmatch(os.path.basename(a)) for a in seen)


def test_temp_name_is_unique_hidden_and_matched():
    a, b = temp_name("data.json"), temp_name("data.json")
    assert a != b and a.startswith(".data.json.")
    assert TEMP_NAME_PATTERN.fullmatch(temp_name("line\nbreak"))


def test_interrupted_writes_lists_only_regular_temp_files(tmp_path):
    temp = tmp_path / temp_name("k.mp4")
    temp.write_bytes(b"")
    for name in ("k.mp4", ".DS_Store", "x.tmp", ".k.ab12cd3.tmp"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / temp_name("dir")).mkdir()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / temp_name("deep")).write_bytes(b"")  # not directly in folder
    assert interrupted_writes(tmp_path) == [os.path.abspath(temp)]


@posix_only
def test_interrupted_writes_skips_symlinks(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"")
    (tmp_path / temp_name("link")).symlink_to(target)
    assert interrupted_writes(tmp_path) == []


def test_interrupted_writes_paths_are_absolute(tmp_path, monkeypatch):
    (tmp_path / temp_name("f")).write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    (found,) = interrupted_writes(".")
    assert os.path.isabs(found)


def test_interrupted_writes_custom_pattern(tmp_path):
    (tmp_path / ".blob-123.part").write_bytes(b"")
    (tmp_path / temp_name("f")).write_bytes(b"")
    found = interrupted_writes(tmp_path, pattern=re.compile(r"\.blob-.*\.part"))
    assert [os.path.basename(p) for p in found] == [".blob-123.part"]


def test_interrupted_writes_missing_folder_holds_none(tmp_path):
    assert interrupted_writes(tmp_path / "nope") == []


def test_interrupted_writes_unlistable_folder_raises(tmp_path):
    f = tmp_path / "file"
    f.write_bytes(b"")
    with pytest.raises(OSError):
        interrupted_writes(f)  # not a folder


def test_interrupted_writes_bytes_folder_and_str_pattern(tmp_path):
    (tmp_path / ".blob-1.part").write_bytes(b"")
    found = interrupted_writes(os.fsencode(tmp_path), pattern=r"\.blob-.+\.part")
    assert [os.path.basename(p) for p in found] == [".blob-1.part"]


def test_interrupted_writes_recursive(tmp_path):
    top = tmp_path / temp_name("a")
    top.write_bytes(b"")
    (tmp_path / "sub" / "deeper").mkdir(parents=True)
    deep = tmp_path / "sub" / "deeper" / temp_name("b")
    deep.write_bytes(b"")
    assert interrupted_writes(tmp_path) == [os.path.abspath(top)]
    assert interrupted_writes(tmp_path, recursive=True) == sorted(
        [os.path.abspath(top), os.path.abspath(deep)]
    )


@posix_only
def test_interrupted_writes_recursive_skips_symlinked_and_unreadable_dirs(tmp_path):
    root, other = tmp_path / "root", tmp_path / "other"
    root.mkdir()
    other.mkdir()
    (other / temp_name("x")).write_bytes(b"")
    (root / "link").symlink_to(other)
    locked = root / "locked"
    locked.mkdir()
    (locked / temp_name("y")).write_bytes(b"")
    os.chmod(locked, 0)
    try:
        if os.access(locked, os.R_OK):
            pytest.skip("running as root: permissions are not enforced")
        assert interrupted_writes(root, recursive=True) == []
    finally:
        os.chmod(locked, 0o700)


@posix_only
def test_interrupted_writes_unreadable_top_folder_raises(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    os.chmod(locked, 0)
    try:
        if os.access(locked, os.R_OK):
            pytest.skip("running as root: permissions are not enforced")
        with pytest.raises(PermissionError):
            interrupted_writes(locked, recursive=True)
    finally:
        os.chmod(locked, 0o700)


@pytest.mark.parametrize("bad", ["", "a/b", os.path.join("a", "b")])
def test_temp_name_takes_a_file_name(bad):
    with pytest.raises(ValueError):
        temp_name(bad)


def test_interrupted_writes_finds_what_a_killed_write_left(tmp_path):
    target = tmp_path / "big"
    code = (
        "import os, sys\n"
        "from dol import atomic_write\n"
        "def chunks():\n"
        "    yield b'partial'\n"
        "    os._exit(0)  # killed mid-write: no cleanup runs\n"
        "atomic_write(sys.argv[1], chunks())\n"
    )
    subprocess.run([sys.executable, "-c", code, str(target)], check=True)
    assert not target.exists()
    (left,) = interrupted_writes(tmp_path)
    assert os.path.basename(left).startswith(".big.")


@posix_only
def test_interrupted_writes_does_not_collapse_dotdot_through_symlink(tmp_path):
    # link -> elsewhere/deep, so "link/../blobs" is elsewhere/blobs on disk (#197)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "deep").mkdir(parents=True)
    (elsewhere / "blobs").mkdir()
    (tmp_path / "blobs").mkdir()
    (tmp_path / "link").symlink_to(elsewhere / "deep", target_is_directory=True)
    name = temp_name("f")
    (elsewhere / "blobs" / name).write_bytes(b"")
    (found,) = interrupted_writes(str(tmp_path / "link" / ".." / "blobs"))
    assert os.path.samefile(found, elsewhere / "blobs" / name)


def test_interrupted_writes_rejects_an_empty_folder():
    with pytest.raises(ValueError):
        interrupted_writes("")


@posix_only
def test_filelock_symlinks_to_one_file_share_its_lock(tmp_path):
    from dol import FileLock

    target = tmp_path / "data.json"
    target.write_bytes(b"{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    assert FileLock(link).path == FileLock(target).path
    with FileLock(target):
        with pytest.raises(RuntimeError):  # same thread: would deadlock, so refused
            with FileLock(link):
                pass


@posix_only
def test_filelock_follows_a_retargeted_link_and_a_changed_cwd(tmp_path, monkeypatch):
    from dol import FileLock

    (tmp_path / "day1").write_bytes(b"")
    (tmp_path / "day2").write_bytes(b"")
    link = tmp_path / "current"
    link.symlink_to("day1")
    lock = FileLock(link)
    with lock:
        assert lock.path.endswith("day1.lock")
    link.unlink()
    link.symlink_to("day2")
    with lock:
        assert lock.path.endswith("day2.lock")
    lock.close()

    (tmp_path / "p").mkdir()
    (tmp_path / "q").mkdir()
    monkeypatch.chdir(tmp_path / "p")
    rel = FileLock("log.jsonl")
    monkeypatch.chdir(tmp_path / "q")
    with rel:
        assert os.path.samefile(os.path.dirname(rel.path), tmp_path / "q")
    rel.close()


def test_filelock_rejects_a_path_without_a_file_name(tmp_path):
    from dol import FileLock

    for bad in ("", str(tmp_path) + os.sep):
        with pytest.raises(ValueError):
            FileLock(bad)


@posix_only
def test_filelock_reopens_when_its_lock_file_was_deleted_and_recreated(tmp_path):
    import fcntl

    from dol import FileLock

    lock = FileLock(tmp_path / "data.json")
    with lock:
        pass
    os.unlink(lock.path)  # a cleanup sweep removes it; the kept handle is now stale
    with lock:
        # flock locks belong to the open file: a fresh open of the path is refused
        fd = os.open(lock.path, os.O_RDWR)
        try:
            with pytest.raises(OSError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
    lock.close()


def test_filelock_keeps_its_handle_between_sections(tmp_path):
    from dol import FileLock

    lock = FileLock(tmp_path / "data.json")
    with lock:
        fh = lock._fh
    with lock:
        assert lock._fh is fh  # not reopened needlessly (any platform)
    lock.close()


# write_content_addressed (#183)
def test_write_content_addressed_streams_hashes_and_names_by_content(tmp_path):
    import hashlib

    from dol import write_content_addressed

    name = write_content_addressed(tmp_path, iter([b"he", b"llo"]))
    assert name == hashlib.sha256(b"hello").hexdigest()
    assert (tmp_path / name).read_bytes() == b"hello"
    assert os.listdir(tmp_path) == [name]  # no temp left


def test_write_content_addressed_seams(tmp_path):
    import hashlib

    from dol import write_content_addressed

    name = write_content_addressed(
        tmp_path,
        b"x",
        hasher=hashlib.blake2b,
        name_of=lambda h: h.hexdigest()[:16] + ".bin",
        perms=0o600,
    )
    assert name.endswith(".bin") and len(name) == 20
    if os.name != "nt":
        assert os.stat(tmp_path / name).st_mode & 0o777 == 0o600


def test_write_content_addressed_existing_file_is_left_alone(tmp_path):
    from dol import write_content_addressed

    name = write_content_addressed(tmp_path, b"same")
    before = os.stat(tmp_path / name).st_mtime_ns
    assert write_content_addressed(tmp_path, b"same") == name
    assert os.stat(tmp_path / name).st_mtime_ns == before and len(os.listdir(tmp_path)) == 1


def test_write_content_addressed_does_not_trust_a_wrong_thing_under_the_name(tmp_path):
    import hashlib

    from dol import write_content_addressed

    name = hashlib.sha256(b"payload").hexdigest()
    (tmp_path / name).write_bytes(b"")  # truncated by an earlier non-atomic writer
    write_content_addressed(tmp_path, b"payload")
    assert (tmp_path / name).read_bytes() == b"payload"
    (tmp_path / name).unlink()
    (tmp_path / name).mkdir()  # a folder in the way is an error, not a success
    with pytest.raises(OSError):
        write_content_addressed(tmp_path, b"payload")
    assert [p.name for p in tmp_path.iterdir()] == [name]  # and no temp left


@posix_only
def test_write_content_addressed_replaces_a_link_not_its_target(tmp_path):
    import hashlib

    from dol import write_content_addressed

    other = tmp_path / "other"
    other.write_bytes(b"payload")  # same size, different file
    name = hashlib.sha256(b"payload").hexdigest()
    (tmp_path / name).symlink_to(other)
    write_content_addressed(tmp_path, b"payload")
    assert not (tmp_path / name).is_symlink() and other.read_bytes() == b"payload"


@posix_only
def test_write_content_addressed_a_symlinked_folder_is_used_as_given(tmp_path):
    from dol import write_content_addressed

    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    name = write_content_addressed(tmp_path / "link", b"x")
    assert (real / name).read_bytes() == b"x"


def test_write_content_addressed_failure_leaves_nothing(tmp_path):
    from dol import write_content_addressed

    def boom():
        yield b"partial"
        raise RuntimeError("producer died")

    with pytest.raises(RuntimeError):
        write_content_addressed(tmp_path, boom())
    assert os.listdir(tmp_path) == []
    for bad in ("", "..", "a/b", ".x.0123abcd.tmp", 7):
        with pytest.raises(ValueError):
            write_content_addressed(tmp_path, b"x", name_of=lambda h, bad=bad: bad)
    assert os.listdir(tmp_path) == []
    with pytest.raises(TypeError):
        write_content_addressed(tmp_path, "text")


def test_a_killed_write_content_addressed_is_listed_as_interrupted(tmp_path):
    from dol import write_content_addressed

    seen = []

    def killed():
        yield b"x"
        seen.extend(interrupted_writes(tmp_path))  # the temp is there mid-write
        yield b"y"

    write_content_addressed(tmp_path, killed())
    assert len(seen) == 1 and os.path.dirname(seen[0]) == str(tmp_path)


def test_a_temp_name_collision_does_not_delete_the_other_writers_temp(
    tmp_path, monkeypatch
):
    from dol import atomic as atomic_mod
    from dol import write_content_addressed

    other = tmp_path / ".content.deadbeef.tmp"
    other.write_bytes(b"another writer's")
    monkeypatch.setattr(atomic_mod, "temp_name", lambda name: other.name)
    with pytest.raises(FileExistsError):
        write_content_addressed(tmp_path, b"x")
    with pytest.raises(FileExistsError):
        atomic_write(tmp_path / "f", b"x")
    assert other.read_bytes() == b"another writer's"


@posix_only
def test_write_content_addressed_perms_are_exact_and_default_follows_umask(tmp_path):
    from dol import write_content_addressed

    old = os.umask(0o077)
    try:
        exact = write_content_addressed(tmp_path, b"a", perms=0o640)
        default = write_content_addressed(tmp_path, b"b")
    finally:
        os.umask(old)
    assert os.stat(tmp_path / exact).st_mode & 0o777 == 0o640  # not narrowed
    assert os.stat(tmp_path / default).st_mode & 0o777 == 0o600  # 0o666 & ~umask


def test_write_content_addressed_cleans_up_after_a_keyboard_interrupt(tmp_path):
    from dol import write_content_addressed

    def interrupted():
        yield b"x"
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        write_content_addressed(tmp_path, interrupted())
    assert os.listdir(tmp_path) == []
    with pytest.raises(ValueError):
        write_content_addressed(tmp_path, b"x", name_of=lambda h: "a\0b")
    assert os.listdir(tmp_path) == []
