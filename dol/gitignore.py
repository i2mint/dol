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

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase

#: What a line may hold that this module would not read the way git does; a line with
#: one of these is refused. (``!`` and ``#`` are checked at the start of a line.)
_UNREAD_TOKENS: tuple[str, ...] = ("**", "\\", "[^", "[:")


@dataclass(frozen=True)
class _Line:
    """One ``.gitignore`` line, compiled."""

    pattern: str
    anchored: bool  # a ``/`` before its end: matches from the root, not at any depth
    folders_only: bool  # a trailing ``/``

    @classmethod
    def parse(cls, line: str) -> "_Line":
        folders_only = line.endswith("/")
        pattern = line[:-1] if folders_only else line  # one ``/``: ``a//`` is anchored
        if (
            line != line.strip()
            or line.startswith(("!", "#"))
            or any(token in line for token in _UNREAD_TOKENS)
            or not pattern.strip("/")  # a line git skips ("", "/") matches nothing
        ):
            raise ValueError(
                "a .gitignore line read here cannot use '!', '#', '**', '\\', '[^', "
                f"'[:', surrounding spaces, or be empty (got {line!r})"
            )
        return cls(pattern.lstrip("/"), "/" in pattern, folders_only)

    def covers(self, parts: list[str], *, last_is_dir: Callable[[], bool]) -> bool:
        """Whether this line covers the path ``parts``: it or a folder holding it
        matches (a matched folder covers its contents). ``last_is_dir`` is asked, at
        most once and only when it matters, whether the last part is a folder."""
        depth = self.pattern.count("/")
        for n in range(1, len(parts) + 1):
            if n == len(parts) and self.folders_only and not last_is_dir():
                break  # the key itself is not known to be a folder
            if self.anchored:
                if n - 1 == depth and fnmatchcase("/".join(parts[:n]), self.pattern):
                    return True
            elif fnmatchcase(parts[n - 1], self.pattern):
                return True
        return False


def _checked_key(key: str) -> list[str]:
    """``key`` split into its parts; ``ValueError`` for what would be misread (not a
    plain relative POSIX path: empty, a backslash, a leading or trailing ``/``, ``//``,
    or a ``.``/``..`` segment)."""
    parts = key.split("/")
    if not key or "\\" in key or any(p in ("", ".", "..") for p in parts):
        raise ValueError(
            f"a relative POSIX path is expected (like 'artifacts/x.png'), got {key!r}"
        )
    return parts


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
    (a backslash, ``./``, ``..``).

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
