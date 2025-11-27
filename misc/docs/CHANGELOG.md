# Changelog

All notable changes to the dol project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

#### Configurable File Deletion System (2025-01-26)

Added comprehensive support for configurable file deletion with cross-platform trash/recycle bin functionality.

**New Module: `dol.trash`**
- Cross-platform trash/recycle bin support for safer file deletion
- Multiple deletion strategies:
  - `default_delete_func`: Safe deletion that tries trash with fallback to `os.remove` (with warning)
  - `permanent_delete`: Direct `os.remove` with no warnings
  - `trash_only`: Only allow trash, error if unavailable
- Platform detection with caching (supports macOS, Windows, Linux)
- Optional `send2trash` library support for better cross-platform compatibility
- Graceful fallback chain: send2trash → platform-specific → os.remove

**Enhanced Classes:**
- `LocalFileDeleteMixin`: Now supports configurable deletion via `_delete_func` attribute
  - Can be configured at class level or instance level
  - Defaults to safe trash with fallback
- `FileBytesPersister`: Refactored to use `LocalFileDeleteMixin`
  - Added `delete_func` parameter to `__init__`
  - Eliminated code duplication
  - All derived classes (`Files`, `TextFiles`, `PickleFiles`, `JsonFiles`, `Jsons`) automatically inherit this functionality
- `AioFileBytesPersister`: Updated async implementation with same configurable deletion pattern

**Benefits:**
- **Safer defaults**: Files moved to trash instead of permanent deletion by default
- **Backwards compatible**: Existing code works without changes
- **Highly configurable**: Users can choose deletion strategy at class or instance level
- **Helpful warnings**: When trash unavailable, provides instructions to suppress warnings or install send2trash

**Usage Examples:**
```python
from dol import Files
from dol.trash import permanent_delete, trash_only

# Default: safe trash with warning on fallback
s = Files('/my/data')
del s['file.txt']  # Moves to trash if available

# Permanent deletion (no warnings)
s = Files('/my/data', delete_func=permanent_delete)
del s['file.txt']  # Permanently deleted

# Trash only (error if unavailable)
s = Files('/my/data', delete_func=trash_only)
del s['file.txt']  # Moves to trash or raises error

# Class-level override
class PermanentFiles(Files):
    _delete_func = permanent_delete

# Custom deletion function
def log_delete(path):
    print(f"Deleting {path}")
    os.remove(path)
s = Files('/my/data', delete_func=log_delete)
```

**Optional Dependencies:**
- `send2trash>=1.8.0`: Recommended for better cross-platform trash support
  - Install with: `pip install send2trash` or `pip install dol[trash]`

### Changed

- `FileBytesPersister` now inherits from `LocalFileDeleteMixin` as first parent class
- Default deletion behavior changed from direct `os.remove` to safe trash with fallback
  - Users who want old behavior can use `permanent_delete` explicitly

### Fixed

- Eliminated code duplication between `LocalFileDeleteMixin` and `FileBytesPersister.__delitem__`
