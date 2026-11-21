r"""Rename the deprecated ``X_of_Y`` keywords of ``wrap_kvs`` calls to encoder/decoder names.

A maintainer tool (not part of the ``dol`` distribution) for `i2mint/dol#92`_: dol 0.4
deprecates ``key_of_id=``, ``id_of_key=``, ``obj_of_data=`` and ``data_of_obj=`` on
``wrap_kvs`` in favour of ``key_decoder=``, ``key_encoder=``, ``value_decoder=`` and
``value_encoder=`` (same semantics). This rewrites call sites, in code **and** in
doctests, without any dependency: it works on the text, finds each ``wrap_kvs(`` call
(also ``dol.wrap_kvs(``, ``@wrap_kvs(``, ``partial(wrap_kvs, ...)``), matches its
closing parenthesis, and renames only keyword arguments inside that span.

Run from the dol repo root::

    python -m misc.codemods.x_of_y_kwargs PATH...           # dry run: list the renames
    python -m misc.codemods.x_of_y_kwargs PATH... --write   # apply them

>>> src = '''s = wrap_kvs(d, obj_of_data=json.loads,
...                  data_of_obj=json.dumps)
... t = _wrap_kvs(d, obj_of_data=f)  # private: untouched
... u = ExplicitKeyMap(key_of_id={'a': 1})  # not a wrap_kvs call: untouched
... def wrap_kvs(store, *, key_of_id=None): ...  # a definition: untouched
... '''
>>> new, n = rename_x_of_y_kwargs(src)
>>> print(new)
s = wrap_kvs(d, value_decoder=json.loads,
                 value_encoder=json.dumps)
t = _wrap_kvs(d, obj_of_data=f)  # private: untouched
u = ExplicitKeyMap(key_of_id={'a': 1})  # not a wrap_kvs call: untouched
def wrap_kvs(store, *, key_of_id=None): ...  # a definition: untouched
<BLANKLINE>
>>> n
2

Doctest lines are handled the same way, prompts included:

>>> doc = '    >>> S = wrap_kvs(dict,\n    ...     key_of_id=str.upper, id_of_key=str.lower)\n'
>>> rename_x_of_y_kwargs(doc)[0].splitlines()[1]
'    ...     key_decoder=str.upper, key_encoder=str.lower)'

A comparison (``key_of_id == x``) or a nested call's own keyword is left alone only if
it is not ``name=``; nested calls inside a ``wrap_kvs(...)`` span are rare, and the dry
run lists every rename for review.

.. _i2mint/dol#92: https://github.com/i2mint/dol/issues/92
"""

import re
import sys
from pathlib import Path

X_OF_Y_TO_CODEC_KWARG = {
    "key_of_id": "key_decoder",
    "id_of_key": "key_encoder",
    "obj_of_data": "value_decoder",
    "data_of_obj": "value_encoder",
}

_CALL_START = re.compile(r"(?<![\w])(?<!def )(?:wrap_kvs\(|partial\(\s*(?:[\w.]+\.)?wrap_kvs\s*,)")
_KWARG = re.compile(
    r"(?<![\w.])(" + "|".join(X_OF_Y_TO_CODEC_KWARG) + r")(\s*=)(?!=)"
)


def _matching_paren_end(text: str, open_idx: int) -> int:
    """Index just past the parenthesis closing the one at ``open_idx`` (or len(text)).

    Skips over string literals (single-line, best effort) so parentheses inside
    quotes do not count.
    """
    depth = 0
    i = open_idx
    quote = None
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote or c == "\n":
                quote = None
        elif c in "'\"":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def call_spans(text: str):
    """Yield ``(start, end)`` of each ``wrap_kvs(...)`` argument span in ``text``."""
    for m in _CALL_START.finditer(text):
        open_idx = text.index("(", m.start())
        yield open_idx, _matching_paren_end(text, open_idx)


def rename_x_of_y_kwargs(text: str) -> tuple[str, int]:
    """Return ``(new_text, number_of_renames)`` for one source text."""
    spans = sorted(set(call_spans(text)))
    out, last, n = [], 0, 0
    for start, end in spans:
        if start < last:  # nested inside an already-rewritten span
            continue
        segment = text[start:end]
        new_segment, k = _KWARG.subn(
            lambda m: X_OF_Y_TO_CODEC_KWARG[m.group(1)] + m.group(2), segment
        )
        out.append(text[last:start])
        out.append(new_segment)
        last, n = end, n + k
    out.append(text[last:])
    return "".join(out), n


def iter_python_files(paths):
    """Every ``.py`` file under the given directories, and every file given explicitly."""
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(
                f for f in p.rglob("*.py") if not any(
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
        new, n = rename_x_of_y_kwargs(text)
        if n:
            total += n
            print(f"{f}: {n}")
            if write:
                f.write_text(new, encoding="utf-8")
    print(f"{'renamed' if write else 'would rename'} {total} keyword(s)")
    return total


if __name__ == "__main__":
    main()
