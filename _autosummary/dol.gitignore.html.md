# dol.gitignore

Read `.gitignore` lines by git’s rules, as a key filter over a store’s keys.

A store that copies a project folder (a sync, an export, a push to S3) has to keep
local the same files git keeps out of version control. [`gitignore_matcher()`](#dol.gitignore.gitignore_matcher) turns
`.gitignore` lines into a `key -> bool` predicate that composes with
[`filt_iter()`](dol.trans.html.md#dol.trans.filt_iter). It reads a deliberate subset of the format and *refuses*
the rest (`ValueError`) rather than misread it: a misread line could copy what
should stay local.

```pycon
>>> from dol import gitignore_matcher
>>> ignored = gitignore_matcher(["*.lock", "cache/"])
>>> ignored("deep/x.lock"), ignored("cache/blob"), ignored("scene.md")
(True, True, False)
```

### Functions

| [`gitignore_matcher`](#dol.gitignore.gitignore_matcher)(lines, \*[, dirs])   | `key -> bool`: whether any of `lines` (`.gitignore` lines) covers the relative POSIX path `key`, by git's rules.   |
|-----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|

### dol.gitignore.gitignore_matcher(lines, , dirs=None)

`key -> bool`: whether any of `lines` (`.gitignore` lines) covers the
relative POSIX path `key`, by git’s rules.

A line with a `/` before its end is anchored at the root; one without matches a
name at any depth; a trailing `/` matches folders only; wildcards never cross a
`/`; a matched folder covers everything under it; case counts (as git’s does
with `core.ignorecase=false`). What is not read as git does raises `ValueError`
instead of being misread: `!`, `#`, `**`, `\`, `[^`, `[:`, surrounding
spaces, empty lines, and a `key` that is not a plain relative POSIX path
(a backslash, `./`, `..`: on Windows, give a `Files` store’s keys as POSIX
paths first). `?` and `[...]` take a *byte*, as in git (`?` does not match
the one character `é`), and never `/`; an unterminated, empty, `/`-holding
or reversed `[...]`, and `//`, are refused too.

* **Parameters:**
  * **lines** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The `.gitignore` lines (not one `str`).
  * **dirs** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `dirs(key)` says whether `key` itself is a folder. A key matcher
    cannot know that, and a trailing-`/` line matches only folders: so without
    `dirs` a key is taken as a file (`cache/` does not match the key
    `cache`, only what is under it). Asked at most once per call, and only when
    a folders-only line needs it.
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]

```pycon
>>> local = gitignore_matcher(["artifacts/shot_cache/", "notes.txt", "artifacts/v/*.lock"])
>>> local("artifacts/shot_cache/blobs/x"), local("deep/notes.txt"), local("artifacts/audio/a.wav")
(True, True, False)
>>> local("artifacts/v/a.json.lock"), local("artifacts/v/sub/a.lock")
(True, False)
>>> gitignore_matcher(["cache/"])("cache"), gitignore_matcher(["a/*/c"])("a/b/c/d")
(False, True)
>>> gitignore_matcher(["cache/"], dirs={"cache"}.__contains__)("cache")
True
```
