---
name: dol-dev-portability
description: "Keep dol working on Windows as well as Linux/macOS. Use when touching dol's path/key machinery (filesys.py, naming.py, paths.py, util.py), compiling regexes from templates/paths, mapping keys<->filesystem paths, or when a dol test/doctest passes on Linux/macOS but fails on the Windows CI job. Covers dol's specific landmines: safe_compile is path-only (never compile a regex with it), escape template LITERALS not the whole pattern, os.sep consistency in prefix/affix codecs, empty-prefix handling, POSIX-only os calls, and dol's native-separator key convention. For general cross-platform Python principles, see the global `cross-platform-python` skill."
---

# dol portability (Windows + POSIX)

dol is a path/key-mapping library, so it is unusually exposed to OS differences:
nearly every store turns a logical **key** into a filesystem **path** and back, and
several helpers compile **regexes** from path templates. Linux/macOS use `/`;
Windows uses `\` (a regex metacharacter) and lacks some POSIX `os` calls. Green
Linux CI proves nothing — **the Windows CI job is the authority.**

For the general discipline (abstract don't branch; centralize platform branches;
encoding; testing) read the global **`cross-platform-python`** skill. This file is
the dol-specific landmine map.

## The five dol landmines (all have bitten production)

### 1. `safe_compile` is for PATHS, never for regexes
`dol.util.safe_compile(path)` exists to turn a *literal file path* into a regex; it
`re.escape`s its argument **on Windows**. Routing an actual **regex** through it
turns the regex into a literal-string matcher on Windows (it then matches nothing
useful). This silently broke `filter_regex` → `filter_suffixes('.json')` → every
`dol.Jsons` store on Windows (`KeyError: 'Key not in store: <k>.json'`), and the
`KeyTemplate`/`mk_pattern_*` builders.
- **To compile a regex, use `re.compile`.** Never `safe_compile`.
- `safe_compile` stays only for genuine literal-path-to-regex needs; its output is
  intentionally platform-dependent, so don't assert its `.pattern` across OSes.

### 2. Building a regex from a template: escape the LITERALS, not the whole thing
A template can be (or contain) a real path, so its literal text has backslashes on
Windows. Compiling them unescaped raises `re.error: incomplete escape \U`; blanket
`re.escape`-ing the whole pattern corrupts the `(?P<field>...)` capture groups.
- **Pattern:** weave the template with `string.Formatter().parse()`, `re.escape`
  each literal segment, substitute each field's capture group, then `re.compile`.
  See `naming.template_to_pattern` and `paths.KeyTemplate._compile_regex` (both do
  this). The field separator is already escaped in `mk_format_mapping_dict`
  (`"[^" + re.escape(sep) + "]+"`).
- Bonus: escaping literals makes a template `.` match a literal dot, not any char.

### 3. Separators: use `os.sep`/`path_sep` consistently, never a hardcoded `/` or `\`
The centralized constants are `filesys.file_sep` and `paths.path_sep` (both
`os.path.sep`). Build **prefix/suffix/affix key codecs from them**. A hardcoded `/`
suffix won't match a `\`-terminated dir path on Windows, so the affix codec re-adds
a `/` → mixed-separator `...\folder1/` `KeyError` (the `subfolder_stores` bug).

### 4. Empty rootdir/prefix must stay empty
`ensure_slash_suffix("")` must return `""`, not a bare separator. A lone separator
prepended to an *absolute* key yields `\C:\Users\...` — invalid on Windows
(`OSError: [Errno 22]`), silently tolerated as `//...` on POSIX. This bit
`Files("")` in `dol.misc.get_obj`. Guard: `if path and not path.endswith(sep): ...`.

### 5. POSIX-only `os` calls
`os.getuid`/`geteuid`/`fork` and `pwd`/`grp`/`fcntl` don't exist on Windows
(AttributeError). For the current user use `getpass.getuser()` (it bit the per-user
temp dir in `filesys.py`). Feature-detect (`hasattr(os, 'getuid')`) for anything else.

## dol's key↔path layering (where bugs live)

A typical store (`Jsons` = `affix_key_codec('.json')` ∘ `filt_iter.suffixes('.json')`
∘ `JsonFiles`, with `JsonFiles` = json-codec ∘ relative-path-store ∘ file IO) maps a
key through: **affix codec** (add/strip `.json`) → **filt_iter** (regex key filter —
landmine 1) → **relative-path store** (`PrefixRelativizationMixin`: prepend/strip
`rootdir` — landmines 3 & 4) → **`FileSysCollection`** (`is_valid_key` regex from a
path template — landmines 1 & 2) → **file IO**. When a Windows-only `KeyError`/
`re.error`/`OSError 22` appears, walk this stack.

## dol's key-separator convention (and a test gotcha)

`Files`/`Jsons` keys are **relative paths using the native `os.sep`** — so a key is
`subfolder/apple.p` on POSIX but `subfolder\apple.p` on Windows. Therefore:
- **Tests must build expected keys with `os.path.join`**, never a literal
  `"subfolder/apple.p"` or trailing `/`.
- (Design note: canonicalizing keys to `/` internally — converting at the FS edge —
  would remove this whole class; until then, native-sep is the convention.)

## Testing dol for portability

- `python -m pytest dol --doctest-modules -q` — **doctests are the main offenders**;
  they encode platform-specific path/separator output. Make them OS-independent:
  assert *behavior* (matching, `os.path.join` round-trips), not hand-counted
  backslash patterns or literal `/`.
- Reproduce a Windows regex/escape bug on macOS by forcing the platform:
  `monkeypatch.setattr(dol.util.platform, "system", lambda: "Windows")`, then reload
  the module (the broken `re.escape` path runs). Add a loop-over-`("Linux","Darwin",
  "Windows")` regression test (see `tests/test_trans.py::test_filter_regex_is_os_independent`).
- The Windows CI job is non-blocking but authoritative; fetch its log to get the
  exact failure list and iterate.

## Checklist for a PR touching dol path/key code

- [ ] No `safe_compile` on a regex; regexes compiled with `re.compile`.
- [ ] Template→regex escapes literal segments (`re.escape`), never the whole pattern.
- [ ] Prefix/suffix/affix codecs use `os.sep`/`path_sep`, no hardcoded `/` or `\`.
- [ ] Empty prefix stays empty; no bare-separator-on-absolute-key.
- [ ] No POSIX-only `os` call without a guard/`getpass` replacement.
- [ ] Tests/doctests assert behavior or `os.path.join`, not literal separators.
- [ ] Windows CI job is green (not just Linux/macOS).
