# dol.trash

Cross-platform file trash/recycle bin functionality for dol.

This module provides configurable file deletion strategies with support for
moving files to trash/recycle bin instead of permanent deletion.

Available deletion strategies:

> - default_delete_func: Safe trash with warning on fallback to os.remove
> - permanent_delete: Direct os.remove (no warnings)
> - trash_only: Error if trash unavailable

Configure deletion behavior when creating file stores by passing the
`delete_func` parameter or setting the `_delete_func` class attribute.

### Functions

| [`default_delete_func`](#dol.trash.default_delete_func)(filepath)   | Try trash, fall back to permanent delete on failure.                               |
|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`get_platform_trash_func`](#dol.trash.get_platform_trash_func)()       | Get platform-specific trash function with caching.                                 |
| [`make_safe_delete_func`](#dol.trash.make_safe_delete_func)([...])    | Create a deletion function that tries trash first, falls back to permanent delete. |
| [`permanent_delete`](#dol.trash.permanent_delete)(filepath)      | Permanently delete a file (no trash, no warnings).                                 |
| [`trash_only`](#dol.trash.trash_only)(filepath)            | Move to trash only - raise error if trash unavailable.                             |

### dol.trash.default_delete_func(filepath)

Try trash, fall back to permanent delete on failure.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### dol.trash.get_platform_trash_func()

Get platform-specific trash function with caching.

Returns None if no trash function is available.

Priority order:

1. send2trash library (if installed)
2. Platform-specific implementation (macOS, Windows, Linux)
3. None (will fall back to os.remove)

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`None`](https://docs.python.org/3/builtins/constants.html#None)]]
* **Returns:**
  Deletion function that moves files to trash, or None if unavailable

### dol.trash.make_safe_delete_func(permanent_delete_func=<built-in function remove>, warn_on_fallback=True)

Create a deletion function that tries trash first, falls back to permanent delete.

* **Parameters:**
  * **permanent_delete_func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Function to use if trash is unavailable
  * **warn_on_fallback** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to warn when falling back to permanent delete
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`None`](https://docs.python.org/3/builtins/constants.html#None)]
* **Returns:**
  A deletion function that tries trash with fallback

### dol.trash.permanent_delete(filepath)

Permanently delete a file (no trash, no warnings).

* **Parameters:**
  **filepath** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to file to delete
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

Use this function when you want permanent deletion without warnings.
Pass it as delete_func parameter: Files(‘/my/data’, delete_func=permanent_delete)

### dol.trash.trash_only(filepath)

Move to trash only - raise error if trash unavailable.

* **Parameters:**
  **filepath** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to file to trash
* **Raises:**
  [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) – If trash functionality not available
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

Use this function when you want to ensure files are only moved to trash,
never permanently deleted. Raises error if trash is unavailable.
Pass it as delete_func parameter: Files(‘/my/data’, delete_func=trash_only)
