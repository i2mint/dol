# Code Quality and Improvement Opportunities

This document tracks potential improvements identified through code analysis, including dead code detection, test coverage gaps, and code smells.

**Last Updated**: 2025-11-21
**Analysis Tools**: vulture (dead code detection), coverage.py (test coverage)

---

## Dead Code Analysis

### High Priority Items (True Dead Code)

#### 1. Unreachable Code After Raise Statement
**File**: `dol/util.py`, lines 1596-1609
**Issue**: The function `delegate_as` raises `NotImplementedError` but has 14+ lines of code after the raise that will never execute.
**Status**: ⚠️ Needs attention
**Recommendation**: Remove unreachable code or convert to comments if implementation is planned

```python
def delegate_as(delegate_cls, to="delegate", include=frozenset(), exclude=frozenset()):
    raise NotImplementedError("Didn't manage to make this work fully")
    # All code after this is unreachable...
```

#### 2. Pass Statement After Raise
**File**: `dol/zipfiledol.py`, line 267
**Issue**: `pass` statement after `raise` is unreachable
**Status**: ⚠️ Needs attention
**Recommendation**: Remove the pass statement

```python
except Exception:
    raise
    pass  # <- Dead code
```

#### 3. Unused Imports
**File**: `dol/util.py`, line 25
**Issue**: `from inspect import getsource` - imported but never used
**Status**: ⚠️ Needs attention
**Recommendation**: Remove import

---

### Medium Priority Items (Incomplete Implementations)

#### 1. Unused Function Parameter: `n_lines_to_skip`
**File**: `dol/base.py`, line 1016
**Issue**: The `skip_lines` method accepts parameter `n_lines_to_skip` but never uses it
**Status**: 🔍 Under review
**Recommendation**: Either implement the skipping logic or remove the parameter

```python
def skip_lines(self, instance, n_lines_to_skip=0):
    instance.seek(0)  # <- Should use n_lines_to_skip?
```

#### 2. Unused Parameter: `ext_mapping`
**File**: `dol/kv_codecs.py`, line 588
**Issue**: Function `extension_based` accepts `ext_mapping` parameter but doesn't use it in the function body
**Status**: 🔍 Under review
**Recommendation**: Complete implementation or document why parameter exists

#### 3. Unused Variable: `key_condition`
**File**: `dol/filesys.py`, line 726
**Status**: 🔍 Under review
**Recommendation**: Investigate whether this was meant to be used in a condition

#### 4. Unused Variable: `key_to_value`
**File**: `dol/paths.py`, line 682
**Status**: 🔍 Under review
**Recommendation**: Investigate intended usage

#### 5. Unused Variable: `disable_deletes`
**File**: `dol/trans.py`, line 551
**Status**: 🔍 Under review
**Recommendation**: Investigate whether this controls deletion behavior

#### 6. Unused Exception Variable
**File**: `dol/zipfiledol.py`, line 874
**Issue**: Exception caught but the variable binding is unused
**Status**: 🔍 Under review

---

### Low Priority Items (False Positives / By Design)

#### 1. Signature Capture Functions
**Files**: `dol/kv_codecs.py` (lines 34-50), `dol/signatures.py` (lines 4442-4473)
**Issue**: Vulture reports function parameters as unused
**Status**: ✅ False positive - by design
**Explanation**: These functions are decorated with `@Sig` and exist solely to capture parameter signatures for later composition. They are never meant to be called.

**Example**:
```python
@Sig
def _csv_rw_sig(
    dialect: str = "excel",      # Reported as "unused"
    delimiter: str = ",",         # But actually used via signature
    # ...
): ...

# Later composed into actual function signatures:
@__csv_rw_sig
def csv_encode(...): ...
```

**Recommendation**: Add documentation comment to clarify the pattern:
```python
# Signature template (not called, used for signature composition via @Sig decorator)
@Sig
def _csv_rw_sig(...): ...
```

#### 2. Type Annotation Imports
**Multiple Files**: Various imports of `Optional`, `Tuple`, `Dict`, `List`, `TypedDict`, etc.
**Issue**: Vulture reports as unused (90% confidence)
**Status**: ✅ False positive
**Explanation**: These are used in type annotations, which static analysis tools sometimes don't detect
**Recommendation**: Keep these imports

---

## Test Coverage Analysis

**Overall Coverage**: 58%
**Test Command**: `python -m coverage run -m pytest && python -m coverage report`

### Files with Low Coverage (<60%)

| File | Coverage | Missing Lines | Priority |
|------|----------|---------------|----------|
| `dol/naming.py` | 31.4% | 293 | High |
| `dol/appendable.py` | 37.8% | 97 | High |
| `dol/util.py` | 41.7% | 359 | High |
| `dol/zipfiledol.py` | 45.5% | 176 | Medium |
| `dol/dig.py` | 45.9% | 46 | Medium |
| `dol/errors.py` | 46.2% | 21 | Low |
| `dol/signatures.py` | 46.2% | 597 | Medium |
| `dol/trans.py` | 50.1% | 394 | Medium |
| `dol/sources.py` | 50.4% | 134 | Medium |
| `dol/explicit.py` | 50.7% | 34 | Low |
| `dol/caching.py` | 56.4% | 281 | Medium |
| `dol/paths.py` | 58.2% | 213 | Medium |

### High Priority Coverage Gaps

#### 1. `dol/naming.py` (31% coverage)
- **Size**: 427 statements, 293 missing
- **Description**: Naming and naming templates functionality
- **Recommendation**: This is a substantial module with very low coverage. Should add comprehensive tests for:
  - Template construction and validation
  - Name generation and parsing
  - Edge cases with special characters
  - Template composition

#### 2. `dol/appendable.py` (38% coverage)
- **Size**: 156 statements, 97 missing
- **Description**: Appendable store functionality
- **Recommendation**: Add tests for:
  - Appending operations
  - Edge cases (empty stores, concurrent appends if supported)
  - Integration with different store backends

#### 3. `dol/util.py` (42% coverage)
- **Size**: 616 statements, 359 missing
- **Description**: General utility functions
- **Recommendation**: This is the largest file with low coverage. Focus on:
  - Most commonly used utility functions
  - Critical path utilities
  - Public API functions

---

## Code Smells and Improvement Opportunities

### 1. Unused Typing Imports Pattern
**Pattern**: Many files import typing constructs that may not be actively used
**Recommendation**: Use `ruff` or `mypy` to automatically detect and remove truly unused typing imports

### 2. Large Files with Low Coverage
**Files**: `dol/util.py` (616 lines), `dol/signatures.py` (1109 lines), `dol/trans.py` (789 lines)
**Recommendation**: Consider refactoring these large modules into smaller, more focused modules that are easier to test and maintain

### 3. Exception Handling Patterns
Several instances of bare `except Exception:` or catching exceptions without using the exception object
**Recommendation**: Review exception handling to ensure proper error context is preserved

---

## Action Items

### Immediate (High Priority)
- [ ] Remove unreachable code in `dol/util.py:1596-1609`
- [ ] Remove pass statement in `dol/zipfiledol.py:267`
- [ ] Remove unused import `getsource` in `dol/util.py:25`
- [ ] Add TODO comments to incomplete implementations (base.py, kv_codecs.py)
- [ ] Add tests for `dol/naming.py` to improve coverage from 31% to >60%
- [ ] Add tests for `dol/appendable.py` to improve coverage from 38% to >60%

### Medium Term
- [ ] Investigate and resolve unused variables in filesys.py, paths.py, trans.py
- [ ] Add documentation comments to signature capture functions
- [ ] Increase coverage for `dol/util.py`, `dol/signatures.py`, `dol/trans.py`
- [ ] Consider refactoring large files into smaller modules

### Long Term
- [ ] Achieve 70%+ overall test coverage
- [ ] Set up automated dead code detection in CI
- [ ] Implement pre-commit hooks for code quality checks
- [ ] Document common patterns (like signature capture) in developer guide

---

## Notes

- This analysis was performed using vulture with 80% confidence threshold and excluded test and scrap directories
- Coverage analysis was run with pytest and coverage.py
- Some "unused" code may be part of the public API and should be retained for backward compatibility
- Signature capture pattern in this codebase is intentional and should not be "fixed"

---

## References

- [Vulture Documentation](https://github.com/jendrikseipp/vulture)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Python Code Quality Tools](https://realpython.com/python-code-quality/)
