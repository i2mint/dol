"""Read ``.gitignore`` lines by git's rules, as a key filter over a store's keys.

A store that copies a project folder (a sync, an export, a push to S3) has to keep
local the same files git keeps out of version control. :func:`gitignore_matcher` turns
``.gitignore`` lines into a ``key -> bool`` predicate that composes with
:func:`~dol.trans.filt_iter`. It reads a deliberate subset of the format and *refuses*
the rest (``ValueError``) rather than misread it: a misread line could copy what
should stay local.

>>> from dol import gitignore_matcher
>>> ignored = gitignore_matcher(["*.lock", "cache/"])
>>> ignored("deep/x.lock"), ignored("cache/blob"), ignored("scene.md")
(True, True, False)
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

#: What a line may hold that this module would not read the way git does; a line with
#: one of these is refused. (``!`` and ``#`` are checked at the start of a line.)
_UNREAD_TOKENS: tuple[str, ...] = ("**", "\\", "[^", "[:", "//")


def _refuse(line: str, why: str = "") -> ValueError:
    return ValueError(
        "a .gitignore line read here cannot use '!', '#', '**', '\\', '[^', '[:', "
        "'//', an unterminated or '/'-holding or reversed [...], surrounding spaces, "
        f"or be empty (got {line!r}){why}"
    )


def _bracket_regex(body: bytes, negated: bool) -> bytes:
    """The regex class for the body of a git ``[...]`` (``-`` first or last is a
    literal; never matches ``/``); ``ValueError`` for a reversed range."""
    items, i = [], 0
    while i < len(body):
        if i + 2 < len(body) and body[i + 1 : i + 2] == b"-":
            lo, hi = body[i], body[i + 2]
            if lo > hi:
                raise ValueError("reversed range")
            items.append(re.escape(bytes([lo])) + b"-" + re.escape(bytes([hi])))
            i += 3
        else:
            items.append(re.escape(bytes([body[i]])))
            i += 1
    return b"[" + (b"^/" if negated else b"") + b"".join(items) + b"]"


def _pattern_regex(pattern: str, line: str) -> "re.Pattern[bytes]":
    """One ``.gitignore`` glob as a regex over UTF-8 bytes, as git matches (``?`` and
    ``[...]`` take a byte; no wildcard takes a ``/``)."""
    raw, out, i = pattern.encode(), [], 0
    while i < len(raw):
        c = raw[i : i + 1]
        if c == b"*":
            out.append(b"[^/]*")
        elif c == b"?":
            out.append(b"[^/]")
        elif c == b"[":
            j = i + 1
            negated = raw[j : j + 1] == b"!"
            j += negated
            start = j
            j += raw[j : j + 1] == b"]"  # a leading ] is a literal
            end = raw.find(b"]", j)
            body = raw[start:end]
            if end < 0 or not body or b"/" in body:
                raise _refuse(line)
            try:
                out.append(_bracket_regex(body, negated))
            except ValueError:
                raise _refuse(line) from None
            i = end
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile(b"".join(out), re.DOTALL)


@dataclass(frozen=True)
class _Line:
    """One ``.gitignore`` line, compiled."""

    regex: "re.Pattern[bytes]"  # of the glob, matched against whole parts
    depth: int  # the ``/`` count of the glob: how many parts it spans, less one
    anchored: bool  # a ``/`` before its end: matches from the root, not at any depth
    folders_only: bool  # a trailing ``/``

    @classmethod
    def parse(cls, line: str) -> "_Line":
        folders_only = line.endswith("/")
        pattern = line[:-1] if folders_only else line
        if (
            line != line.strip()
            or line.startswith(("!", "#"))
            or any(token in line for token in _UNREAD_TOKENS)
            or not pattern.strip("/")  # a line git skips ("", "/") matches nothing
        ):
            raise _refuse(line)
        anchored = "/" in pattern
        pattern = pattern.lstrip("/")
        return cls(
            _pattern_regex(pattern, line), pattern.count("/"), anchored, folders_only
        )

    def covers(self, parts: list[bytes], *, last_is_dir: Callable[[], bool]) -> bool:
        """Whether this line covers the path ``parts`` (UTF-8): it or a folder holding
        it matches (a matched folder covers its contents). ``last_is_dir`` is asked,
        at most once and only when it matters, whether the last part is a folder."""
        for n in range(1, len(parts) + 1):
            if n == len(parts) and self.folders_only and not last_is_dir():
                break  # the key itself is not known to be a folder
            if self.anchored:
                if n - 1 == self.depth and self.regex.fullmatch(b"/".join(parts[:n])):
                    return True
            elif self.regex.fullmatch(parts[n - 1]):
                return True
        return False


def _checked_key(key: str) -> list[bytes]:
    """``key`` split into its UTF-8 parts; ``ValueError`` for what would be misread (not a
    plain relative POSIX path: empty, a backslash, a leading or trailing ``/``, ``//``,
    or a ``.``/``..`` segment)."""
    parts = key.split("/")
    if not key or "\\" in key or any(p in ("", ".", "..") for p in parts):
        raise ValueError(
            f"a relative POSIX path is expected (like 'artifacts/x.png'), got {key!r}"
        )
    return [p.encode() for p in parts]


def gitignore_matcher(
    lines: Iterable[str],
    *,
    dirs: "Callable[[str], bool] | None" = None,
) -> Callable[[str], bool]:
    """``key -> bool``: whether any of ``lines`` (``.gitignore`` lines) covers the
    relative POSIX path ``key``, by git's rules.

    A line with a ``/`` before its end is anchored at the root; one without matches a
    name at any depth; a trailing ``/`` matches folders only; wildcards never cross a
    ``/``; a matched folder covers everything under it; case counts (as git's does
    with ``core.ignorecase=false``). What is not read as git does raises ``ValueError``
    instead of being misread: ``!``, ``#``, ``**``, ``\\``, ``[^``, ``[:``, surrounding
    spaces, empty lines, and a ``key`` that is not a plain relative POSIX path
    (a backslash, ``./``, ``..``: on Windows, give a ``Files`` store's keys as POSIX
    paths first). ``?`` and ``[...]`` take a *byte*, as in git (``?`` does not match
    the one character ``é``), and never ``/``; an unterminated, empty, ``/``-holding
    or reversed ``[...]``, and ``//``, are refused too.

    :param lines: The ``.gitignore`` lines (not one ``str``).
    :param dirs: ``dirs(key)`` says whether ``key`` itself is a folder. A key matcher
        cannot know that, and a trailing-``/`` line matches only folders: so without
        ``dirs`` a key is taken as a file (``cache/`` does not match the key
        ``cache``, only what is under it). Asked at most once per call, and only when
        a folders-only line needs it.

    >>> local = gitignore_matcher(["artifacts/shot_cache/", "notes.txt", "artifacts/v/*.lock"])
    >>> local("artifacts/shot_cache/blobs/x"), local("deep/notes.txt"), local("artifacts/audio/a.wav")
    (True, True, False)
    >>> local("artifacts/v/a.json.lock"), local("artifacts/v/sub/a.lock")
    (True, False)
    >>> gitignore_matcher(["cache/"])("cache"), gitignore_matcher(["a/*/c"])("a/b/c/d")
    (False, True)
    >>> gitignore_matcher(["cache/"], dirs={"cache"}.__contains__)("cache")
    True
    """
    if isinstance(lines, str):  # one line given bare: its characters are not lines
        raise TypeError(f"lines is an iterable of .gitignore lines, not one str: {lines!r}")
    compiled = tuple(_Line.parse(line) for line in lines)

    def covered(key: str) -> bool:
        parts = _checked_key(key)
        answer = []  # ``dirs(key)``, asked once

        def last_is_dir() -> bool:
            if not answer:
                answer.append(bool(dirs(key)) if dirs is not None else False)
            return answer[0]

        return any(line.covers(parts, last_is_dir=last_is_dir) for line in compiled)

    return covered
