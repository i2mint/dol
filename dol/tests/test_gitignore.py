"""``dol.gitignore_matcher`` reads ``.gitignore`` lines as git does (#198): checked
against ``git check-ignore`` itself over many line shapes, with and without the ``dirs``
seam."""

import shutil
import subprocess

import pytest

from dol import filt_iter, gitignore_matcher

_PATHS = (
    "scene.md", "artifacts/render_reports/r.json", "artifacts/voice_options/jerry.json.lock",
    "artifacts/voice_options/jerry.json", "artifacts/voice_options/sub/x.lock",
    "artifacts/probes/deep/p.png", "artifacts/audio/a.wav", "deep/x.lock",
    "deep/probes/x.png", "a/b/c/d", "artifacts/voice_options/a.json", "x/foo/y",
    "a", "a/z", ".env", "sub/.env", "q/r", "foo", "cache",
)  # fmt: skip
_SHAPES = (
    "*.lock", "probes", "a/*/c", "artifacts/v?ice_options/[ab].json", "/scene.md",
    "[!b].json", "q?r", "foo/", "a/*", ".env", "cache//",
)  # fmt: skip
_DIRS = {"foo", "cache", "a", "a/b", "x/foo"}  # the keys above that are folders


def _git_ignored(tmp_path, lines, paths):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    # bytes, NUL-separated: no newline translation on Windows, no path quoting
    (tmp_path / ".gitignore").write_bytes("".join(f"{x}\n" for x in lines).encode())
    got = subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "core.ignorecase=false"]
        + ["check-ignore", "--no-index", "--stdin", "-z"],
        input="\0".join(paths).encode() + b"\0",
        capture_output=True,
    )
    return {p for p in got.stdout.decode().split("\0") if p}


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_agrees_with_git_check_ignore_taking_keys_as_files(tmp_path):
    files = [p for p in _PATHS if p not in _DIRS]
    ignored = _git_ignored(tmp_path, _SHAPES, files)
    match = gitignore_matcher(_SHAPES)
    assert {p for p in files if match(p)} == ignored


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_agrees_with_git_check_ignore_with_dirs(tmp_path):
    # git learns which keys are folders from the disk
    for p in _PATHS:
        path = tmp_path / p
        if p in _DIRS:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
    ignored = _git_ignored(tmp_path, _SHAPES, _PATHS)
    match = gitignore_matcher(_SHAPES, dirs=_DIRS.__contains__)
    assert {p for p in _PATHS if match(p)} == ignored


def test_dirs_is_asked_once_and_only_when_a_folders_only_line_needs_it():
    asked = []

    def dirs(key):
        asked.append(key)
        return True

    assert gitignore_matcher(["*.lock", "x"], dirs=dirs)("a/b.lock") and asked == []
    match = gitignore_matcher(["c/", "d/"], dirs=dirs)
    assert match("c") and match("e") is False
    assert asked == ["c", "e"]


def test_lines_it_cannot_read_as_git_does_are_refused():
    for line in ("!keep.me", "artifacts/**/x", "# note", "\\*.md", "[^a].txt", "foo ",
                 "[[:alpha:]]x", "", "/"):  # fmt: skip
        with pytest.raises(ValueError, match="cannot use"):
            gitignore_matcher([line])
    with pytest.raises(TypeError, match="not one str"):
        gitignore_matcher("*.lock")
    match = gitignore_matcher(["x"])
    for key in ("", "./a", "a\\b", "a//b", "/a", "a/", "../a"):
        with pytest.raises(ValueError, match="relative POSIX"):
            match(key)


def test_composes_with_filt_iter():
    ignored = gitignore_matcher(["*.lock", "cache/"])
    store = filt_iter(
        {"a.txt": 1, "a.lock": 2, "cache/x": 3, "sub/b.lock": 4},
        filt=lambda k: not ignored(k),
    )
    assert sorted(store) == ["a.txt"]
