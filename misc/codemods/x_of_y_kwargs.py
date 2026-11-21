r"""Rename the deprecated ``X_of_Y`` keywords of ``wrap_kvs`` calls to encoder/decoder names.

A maintainer tool (not part of the ``dol`` distribution) for `i2mint/dol#92`_: dol 0.4
deprecates ``key_of_id=``, ``id_of_key=``, ``obj_of_data=`` and ``data_of_obj=`` on
``wrap_kvs`` in favour of ``key_decoder=``, ``key_encoder=``, ``value_decoder=`` and
``value_encoder=`` (same semantics). This rewrites call sites in code **and** in
doctests (also ``>>>`` examples and fenced ``python`` blocks of Markdown files given
explicitly), with the standard library only.

It works on tokens (``tokenize``), so comments and strings never count: a call is a
``wrap_kvs`` name (also ``dol.wrap_kvs``, ``@wrap_kvs``) followed by ``(``, or
``partial(wrap_kvs, ...)``; only keyword arguments at the call's own nesting level are
renamed, so a nested call's keywords, parameters and assignments are left alone. A
rename whose new name is already passed in the same call is skipped and reported
(it would make a duplicate keyword, a SyntaxError).

Run it by path (``misc`` is a namespace package shared with other repos)::

    python misc/codemods/x_of_y_kwargs.py PATH...           # dry run: list the renames
    python misc/codemods/x_of_y_kwargs.py PATH... --write   # apply them

>>> src = '''s = wrap_kvs(d, obj_of_data=json.loads,  # see wrap_kvs( note
...                  data_of_obj=json.dumps)
... t = _wrap_kvs(d, obj_of_data=f)  # private: untouched
... u = ExplicitKeyMap(key_of_id={'a': 1})  # not a wrap_kvs call: untouched
... def wrap_kvs(store, *, key_of_id=None): ...  # a definition: untouched
... v = wrap_kvs(d, key_codec=mk(key_of_id=1))  # a nested call's keyword: untouched
... '''
>>> new, n = rename_x_of_y_kwargs(src)
>>> print(new)
s = wrap_kvs(d, value_decoder=json.loads,  # see wrap_kvs( note
                 value_encoder=json.dumps)
t = _wrap_kvs(d, obj_of_data=f)  # private: untouched
u = ExplicitKeyMap(key_of_id={'a': 1})  # not a wrap_kvs call: untouched
def wrap_kvs(store, *, key_of_id=None): ...  # a definition: untouched
v = wrap_kvs(d, key_codec=mk(key_of_id=1))  # a nested call's keyword: untouched
<BLANKLINE>
>>> n
2

Doctest lines are handled the same way, prompts included:

>>> doc = '    >>> S = wrap_kvs(dict,\n    ...     key_of_id=str.upper, id_of_key=str.lower)\n'
>>> rename_x_of_y_kwargs(doc)[0].splitlines()[1]
'    ...     key_decoder=str.upper, key_encoder=str.lower)'

.. _i2mint/dol#92: https://github.com/i2mint/dol/issues/92
"""

import io
import re
import sys
import tokenize
from pathlib import Path

from dol.trans import X_OF_Y_TO_CODEC_KWARG

_SKIP_TOKENS = {
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
}
_DOCTEST_LINE = re.compile(r"^(\s*)(>>>|\.\.\.)( |$)")
_FENCE = re.compile(r"^(\s*)```")


def _significant_tokens(source: str):
    toks = tokenize.generate_tokens(io.StringIO(source).readline)
    return [t for t in toks if t.type not in _SKIP_TOKENS]


def _is_call_start(toks, i) -> bool:
    """Whether ``toks[i]`` is the ``(`` opening a ``wrap_kvs(`` or ``partial(wrap_kvs,`` call."""

    def name(j):
        return (
            toks[j].string
            if 0 <= j < len(toks) and toks[j].type == tokenize.NAME
            else None
        )

    def op(j, s):
        return (
            0 <= j < len(toks) and toks[j].type == tokenize.OP and toks[j].string == s
        )

    if not op(i, "("):
        return False
    if name(i - 1) == "wrap_kvs" and name(i - 2) != "def":
        return True
    if name(i - 1) == "partial":  # partial([pkg.]wrap_kvs, ...)
        j = i + 1
        while name(j) and op(j + 1, "."):
            j += 2
        return name(j) == "wrap_kvs" and op(j + 1, ",")
    return False


def _call_edits(toks, open_idx):
    """``(edits, skipped)`` for the call opened at ``toks[open_idx]``.

    ``edits`` are ``(row, col, old, new)`` for each X_of_Y keyword at depth 1;
    ``skipped`` the ones whose new name the call already passes.
    """
    depth, keywords = 0, []
    for j in range(open_idx, len(toks)):
        t = toks[j]
        if t.type == tokenize.OP and t.string in "([{":
            depth += 1
        elif t.type == tokenize.OP and t.string in ")]}":
            depth -= 1
            if depth == 0:
                break
        elif (
            depth == 1
            and t.type == tokenize.NAME
            and toks[j - 1].string in ("(", ",")
            and j + 1 < len(toks)
            and toks[j + 1].string == "="
        ):
            keywords.append(t)
    passed = {t.string for t in keywords}
    edits, skipped = [], []
    for t in keywords:
        new = X_OF_Y_TO_CODEC_KWARG.get(t.string)
        if new is None:
            continue
        edit = (t.start[0], t.start[1], t.string, new)
        (skipped if new in passed else edits).append(edit)
    return edits, skipped


def code_edits(source: str):
    """``(edits, skipped)`` for Python ``source``; ``([], [])`` if it does not tokenize."""
    try:
        toks = _significant_tokens(source)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return [], []
    edits, skipped = [], []
    for i in range(len(toks)):
        if _is_call_start(toks, i):
            e, s = _call_edits(toks, i)
            edits += e
            skipped += s
    return edits, skipped


def _apply(lines, edits, offsets=None):
    """Apply ``(row, col, old, new)`` edits (1-based rows) to ``lines`` in place."""
    for row, col, old, new in sorted(edits, reverse=True):
        idx = row - 1 if offsets is None else offsets[row - 1][0]
        col += 0 if offsets is None else offsets[row - 1][1]
        line = lines[idx]
        assert line[col : col + len(old)] == old, (line, col, old)
        lines[idx] = line[:col] + new + line[col + len(old) :]


def _embedded_blocks(lines):
    """Yield ``[(line index, prefix length), ...]`` for each doctest example or fenced
    ``python`` block in ``lines``: code embedded in docstrings, comments or Markdown."""
    i = 0
    while i < len(lines):
        m = _DOCTEST_LINE.match(lines[i])
        if m and m.group(2) == ">>>":
            block = [(i, len(m.group(0)))]
            i += 1
            while (
                i < len(lines)
                and (c := _DOCTEST_LINE.match(lines[i]))
                and c.group(2) == "..."
            ):
                block.append((i, len(c.group(0))))
                i += 1
            yield block
            continue
        f = _FENCE.match(lines[i])
        if f and lines[i].strip().lower() in ("```python", "```py", "```pycon"):
            start, i = i + 1, i + 1
            while i < len(lines) and not _FENCE.match(lines[i]):
                i += 1
            if not any(_DOCTEST_LINE.match(line) for line in lines[start:i]):
                yield [(k, 0) for k in range(start, i)]
            else:  # a pycon block: its >>> examples are found by the scan above
                i = start
                continue
        i += 1


def rename_x_of_y_kwargs(text: str, *, python=True, report=None) -> tuple:
    """Return ``(new_text, number_of_renames)`` for one source or document text.

    ``python=False`` (a Markdown or text document) only rewrites its embedded code:
    doctest examples and fenced ``python`` blocks. ``report``, if given, is called with ``(row, old, new)`` for each rename skipped
    because the call already passes ``new``.
    """
    lines = text.splitlines(keepends=True)
    edits, skipped = code_edits(text) if python else ([], [])
    _apply(lines, edits)
    n = len(edits)
    for block in _embedded_blocks(lines):
        src = "".join(lines[i][p:] for i, p in block)
        if not src.endswith("\n"):
            src += "\n"
        e, s = code_edits(src)
        _apply(lines, e, offsets=block)
        n += len(e)
        skipped += [(block[r - 1][0] + 1, c, o, nw) for r, c, o, nw in s]
    for row, _, old, new in skipped:
        if report:
            report(row, old, new)
    return "".join(lines), n


def iter_python_files(paths):
    """Every ``.py`` file under the given directories, and every file given explicitly."""
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(
                f
                for f in p.rglob("*.py")
                if not any(
                    part.startswith(".") or part in ("build", "dist", "__pycache__")
                    for part in f.parts
                )
            )
        elif p.is_file():  # an explicit file of any kind (.md, .txt docs too)
            yield p


def main(argv=None):
    """CLI: dry run by default; ``--write`` applies. Prints ``path: n`` per changed file."""
    argv = list(sys.argv[1:] if argv is None else argv)
    write = "--write" in argv
    paths = [a for a in argv if a != "--write"] or ["."]
    total = 0
    for f in iter_python_files(paths):
        text = f.read_text(encoding="utf-8")

        def report(row, old, new, f=f):
            print(f"{f}:{row}: skipped {old}= ({new}= already passed in that call)")

        new, n = rename_x_of_y_kwargs(text, python=f.suffix == ".py", report=report)
        if n:
            total += n
            print(f"{f}: {n}")
            if write:
                f.write_text(new, encoding="utf-8")
    print(f"{'renamed' if write else 'would rename'} {total} keyword(s)")
    return total


if __name__ == "__main__":
    main()
