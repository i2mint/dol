# dol: Issues and Discussions — Themes and Insights

This document summarizes the major themes from GitHub issues and discussions in the [i2mint/dol](https://github.com/i2mint/dol) repository. The emphasis is on **design and architecture** themes, since many issues are dev/design discussions rather than bug reports.

Sources: GitHub issues (as of early 2026) and GitHub discussions.

---

## Theme 1: `wrap_kvs` Design Tensions

The single most recurring topic. `wrap_kvs` is central to dol but its current design has acknowledged problems.

### 1a. Signature-based conditioning (Issue #9, Discussion #34)

**The problem**: `wrap_kvs` uses the *signature* of the transform function to decide how to apply it — specifically, whether the function receives `(data)` or `(self, data)`. This causes divergent behavior for functionally equivalent inputs:

```python
# These should behave identically, but don't:
wrap_kvs(store, obj_of_data=lambda x: bytes.decode(x))   # works
wrap_kvs(store, obj_of_data=bytes.decode)                  # fails!
```

**Root cause**: The code checks whether `obj_of_data` has 1 or 2+ required args, and applies it as `obj_of_data(data)` or `obj_of_data(self, data)` accordingly. This "Postelization" (being liberal in what you accept) leads to bugs.

**Discussion #34** ("Clean way of Postelizing callbacks") proposes a more principled solution: use an explicit marker (e.g., a `Literal` type or wrapper class) to signal "this function takes `self`", instead of inferring it from the signature.

**Proposed fix (Issue #12)**: A `FirstArgIsMapping` literal to mark functions that need the store instance as their first argument, removing the need for signature inspection.

**Status**: Open. The design for fixing this cleanly without breaking changes is actively discussed.

### 1b. `self` not being the wrapped instance (Issue #18)

When a method inside a `wrap_kvs`-decorated class calls `self[key]`, `self` is the unwrapped instance (the inner class), not the outer wrapped class. This means the transform pipeline is bypassed for in-class `self[k]` calls.

**Workaround** shown in issue: re-apply `wrap_kvs` to `self` inside the method, or pass the wrapped instance explicitly.

**Impact**: Affects any class that uses `wrap_kvs` as a class decorator and then uses `self[k]` internally.

### 1c. Recursively applying wrappers (Issue #10)

`wrap_kvs` and all wrappers only apply to the "top level" of a store. If the store contains nested stores (a store of stores), the wrap doesn't propagate to values:

```python
s = add_path_access({'a': {'b': {'c': 42}}})
s['a', 'b', 'c']   # works (top-level wrap applied)
s['a']['b', 'c']   # fails (returned value is plain dict, not wrapped)
```

The issue proposes a `conditional_data_trans` pattern to recursively apply wrapping to values that match a condition (e.g., "if the value is a Mapping, wrap it too"). A prototype is shown:

```python
add_path_access_if_mapping = conditional_data_trans(
    condition=instance_checker(Mapping),
    data_trans=add_path_access,
)
```

**Status**: Partially implemented, but the general mechanism for recursively applying wrappers across levels is still a design open question.

---

## Theme 2: The Builtin Codec Ecosystem

### Discussion #42 / Issue #42: Quick access to builtin codecs

A repeated need: users want ready-to-use codecs for common Python stdlib operations (json, pickle, csv, gzip, base64, etc.) without having to manually construct `wrap_kvs(store, obj_of_data=json.loads, data_of_obj=json.dumps)` every time.

**Resolution**: `dol.kv_codecs.ValueCodecs` and `KeyCodecs` namespaces were added. This became one of the most used parts of the library:

```python
from dol import ValueCodecs, KeyCodecs
store = Pipe(KeyCodecs.suffixed('.pkl'), ValueCodecs.pickle())(dict)
```

### Issue #47: Simpler "affix" key codecs

`KeyCodecs.suffixed()` uses `KeyTemplate` internally, which is overkill for simple prefix/suffix operations. Proposal: use simpler string methods (slice, `startswith`/`endswith`) for these common cases.

**Status**: Open. A minor efficiency/simplicity improvement.

---

## Theme 3: Key Transformation Framework

### Discussion #27: The need for a key transformation framework

The KV abstraction works well for flat key spaces, but real backends often have structured keys (paths, composite keys, namespace prefixes). Multiple discussions converge on the need for a proper key transformation framework:

- **KeyPath**: tuples/strings to represent hierarchical paths
- **KeyTemplate**: parse/format structured key strings (`'{user}/{date}.json'`)
- **Prefix filtering**: show only keys starting with a prefix (subpath filtering)
- **Issue #43**: Request for `KeyTemplate` as the "swiss army knife" of key wrappers

**Discussion #32** (Subpath filtering in path-keyed stores): A common use case is "give me a sub-store for keys starting with X". Related to filesystem navigation patterns.

**Discussion #21**: Cleanup and centralization of path access functionality — multiple overlapping implementations exist (`path_get`, `_path_get`, `KeyPath`, `KeyTemplate`).

**Status**: `KeyTemplate` exists and works. `KeyPath` exists. Subpath filtering exists via `filt_iter`. But no unified "path store API" has been formalized.

---

## Theme 4: Caching and Performance

### Issue #50: Stacking `cache_this` decorators

`cache_this` is powerful, but stacking multiple `@cache_this` decorators on the same method causes problems (cache invalidation, key conflicts). Discussion of how to compose caching correctly.

### Issue #56: Fast `update` and synching

`store.update(other_store)` uses the generic Python MutableMapping implementation: iterate over `other_store`, write each item. For stores with millions of items or remote backends, this is very slow.

The proposal: enable backend-specific fast sync mechanisms. For example, `sshdol` has `ssh_files.sync_to(local_files)` using `rsync`. The challenge: how to expose this via the standard `update` interface when the two sides may know nothing about each other.

**Design tension**: maintaining the clean Mapping interface vs. allowing optimized protocol negotiation between stores. A possible approach: detect if both stores have a shared "fast sync" protocol (duck typing or registration).

**Status**: Open. No general solution. Backend-specific workarounds exist.

---

## Theme 5: Context Managers and Transaction-Like Semantics

### Discussion #49: Context managers in dol

Two recurring patterns where context managers are needed:

1. **Connection lifecycle**: stores that need `connect()`/`disconnect()` (databases, remote APIs). The KV interface hides the connection, but someone has to manage it.

2. **Batching/transactions**: write operations accumulate in a buffer and are sent as a batch when the context exits. Useful for performance and atomicity.

The discussion notes the tension: exposing context managers breaks the "just use it like a dict" simplicity. Solutions proposed:
- `flush_on_exit` decorator (already in `caching.py`)
- Store-level `__enter__`/`__exit__` that batch writes
- Explicit "session" objects that wrap stores

**Status**: `flush_on_exit` exists. No general transaction/batching pattern is standardized.

---

## Theme 6: Composite and Hierarchical Stores

### Discussion #25: Composite Stores

"A store made of other stores" — a recurring architectural need. Variations:
- **Fan-out store**: writes go to all sub-stores, reads come from the first that has the key
- **Layered store**: like ChainMap, reads fall through to next store on miss
- **Segmented store**: different keys go to different backends
- **Nested store**: values are themselves stores

**Status**: `FanoutReader`, `FanoutPersister`, `CascadedStores`, `FlatReader` exist. No general "store mesh" framework.

### Discussion #19: Permute levels of nested mappings

Need to flip/reorder the levels of nested dicts — analogous to a `groupby` for nested structures. Example: a `dict[user][date]` that needs to be accessed as `dict[date][user]`.

### Discussion #20: Generalize `FlatReader` to multiple levels

`FlatReader` currently flattens only two levels. Requested: arbitrary-depth flattening.

---

## Theme 7: Batch/Paging/Chunking Operations

### Discussion #29: Paging tools

Chunked iteration (reading 1000 items at a time), batch writes, streaming reads — these come up constantly with large data stores and remote backends. The Mapping interface doesn't have a natural "paging" concept (`__iter__` always yields all keys).

Proposals:
- `chunked_items(store, chk_size)` utility
- Stores that implement a `_page_` method that `filt_iter` can delegate to
- Discussion of how to push filtering/pagination down to the backend

**Status**: Utility functions exist in the ecosystem, but no standard in `dol` core.

---

## Theme 8: Interface Extension and Customization

### Discussion #34 / Issue #12: When transform functions need access to `self`

The standard `obj_of_data(data)` signature is sufficient for most transforms. But sometimes the transform needs context from the store (e.g., its configuration, its root path). The current approach of inferring this from the signature is problematic.

**Proposed design**: A `FirstArgIsMapping` literal marker, so users explicitly opt in to the `(self, data)` calling convention:

```python
from dol.trans import FirstArgIsMapping

wrap_kvs(store, obj_of_data=FirstArgIsMapping(lambda self, data: self.root / data))
```

### Discussion #24: Include hooks for optimized operations

The idea of "hooks" — special methods on the store that dol's tooling checks before falling back to the generic implementation. Example: if a store has a `_filter_` method, `filt_iter` should use it instead of Python-level iteration. This would enable pushing operations like filtering and sorting down to the backend.

This is analogous to how `__len__` makes `len()` efficient even when `__iter__` would work.

---

## Theme 9: Documentation and Discoverability

### Issue #1: Documentation ideas (open, long-running)

A running wishlist for documentation improvements:
- More examples of `wrap_kvs` combinations
- Step-by-step tutorials for common patterns (migrate data between backends, add serialization layer)
- Better documentation of the `kv_walk` utility

### Issue #22: `kv_walk` docs and recipes

`kv_walk` — recursive iteration over nested stores — is powerful but underdocumented. Multiple use cases need recipes.

### Discussion #51: dol examples and applications in the wild

A collection thread for real-world usages and code snippets.

### Discussion #48: Conversations with AI about dol

A thread recording Q&A with AI chatbots about dol. Interesting as a signal of where the AI knowledge gaps are (and where documentation needs improvement).

### Discussion #55: AI-enhanced assistance

Proposals for improving AI agent assistance with dol — relevant to the `CLAUDE.md` and llms.txt effort.

---

## Theme 10: Cross-Platform and Compatibility

### Issue #58, #52: Windows compatibility

Several tests fail on Windows due to:
- Path separator differences (`/` vs `\`)
- Temp file handling
- Regex patterns with backslashes (Issue #40: `re.error: incomplete escape \U at position 2`)

### Issue #59 (CLOSED): Python 3.12 compatibility

Fixed. dol now works with Python 3.12.

---

## Summary: Open Design Questions

| Question | Location | Status |
|---------|----------|--------|
| How should transform functions signal they need `self`? | Issue #12, Discussion #34 | Open |
| How to support fast `update`/sync between heterogeneous stores? | Issue #56 | Open |
| How to handle context managers / transactions generically? | Discussion #49 | Partial |
| Should wrappers propagate recursively to nested values? | Issue #10 | Partial |
| How to push filtering/pagination to the backend? | Discussion #24, #29 | Open |
| How to unify the path access/key transformation utilities? | Discussion #21, #27 | Partial |

---

## Notable Closed Issues (Design Completions)

| Issue | What Was Done |
|-------|--------------|
| #42 | `ValueCodecs` / `KeyCodecs` namespaces created |
| #43 | `KeyTemplate` implemented in `paths.py` |
| #36 | `appendable.py` module with `Extendible` pattern |
| #7  | `__hash__` added to `Store` |
| #8  | `FlatReader` refactored and stabilized |
| #47 | Simpler affix codecs (partially addressed) |
| #59 | Python 3.12 compatibility fixed |
