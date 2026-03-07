# Instruction: Generate `llms.txt` and `llms-full.txt`

Read this codebase and produce two files that help AI agents **use** this package as a tool/dependency (not contribute to it).

## Step 1: Understand the package

- Read `pyproject.toml` (or `setup.cfg`) for: package name, summary, dependencies, entry points.
- Read `__init__.py` to identify the **public API** (i.e. what's in `__all__`, or all non-underscore top-level names).
- Read docstrings and type hints of every public function/class.
- Read doctests and `tests/` for **usage examples**.
- Read `README.md` if present.

## Step 2: Produce `llms.txt`

Follow the spec at https://llmstxt.org. The file must contain, in order:

1. **H1**: Package name.
2. **Blockquote**: One-paragraph summary — what the package does, when to use it, key concepts.
3. **Body notes** (no headings): Important caveats, gotchas, design philosophy — things an agent must know before using the package. Include:
   - Core abstractions (e.g. "all stores are `MutableMapping`").
   - Common patterns and idioms.
   - What this package is **not** (prevent misuse).
4. **`## Core API`**: Links to markdown docs for the most-used functions/classes. If no hosted docs exist, use relative paths like `src/pkg/module.py` or GitHub raw URLs.
5. **`## Examples`**: Links to example scripts, notebooks, or test files that demonstrate typical usage.
6. **`## Optional`**: Links to advanced/niche docs an agent can skip for basic usage.

Keep it **under 4K tokens**. Prefer terse, expert-level language — the reader is an LLM, not a beginner.

## Step 3: Produce `llms-full.txt`

A single markdown file containing **everything an agent needs to use the package**, concatenated in this order:

1. The content of `llms.txt` (as the header/overview).
2. For **each public module**, a section (`## module_name`) containing:
   - Module docstring.
   - For each public function/class: signature, docstring, and **one usage example** (prefer doctests; fall back to test cases).
3. Any additional notes from README that weren't already covered.

Format function signatures as fenced code blocks. Keep docstrings verbatim — don't paraphrase. Strip internal helpers (single-underscore functions) unless they appear in public docstrings or examples.

## Quality checklist

- [ ] An agent reading only `llms.txt` can answer: "What does this package do? Should I use it for X?"
- [ ] An agent reading `llms-full.txt` can write correct code using this package **without** accessing the source.
- [ ] No broken links. No placeholder text.
- [ ] `llms.txt` fits comfortably in a small context window (~4K tokens).
- [ ] `llms-full.txt` includes real, runnable examples for every major public function.
