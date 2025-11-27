"""Cross-platform file trash/recycle bin functionality for dol.

This module provides configurable file deletion strategies with support for
moving files to trash/recycle bin instead of permanent deletion.

Available deletion strategies:
    - default_delete_func: Safe trash with warning on fallback to os.remove
    - permanent_delete: Direct os.remove (no warnings)
    - trash_only: Error if trash unavailable

Usage:
    Configure deletion behavior when creating file stores by passing the
    delete_func parameter or setting _delete_func class attribute.
"""

import os
import sys
from functools import lru_cache
from typing import Callable, Optional
from warnings import warn

# Deletion function type
DeleteFunc = Callable[[str], None]


@lru_cache(maxsize=1)
def get_platform_trash_func() -> Optional[DeleteFunc]:
    """
    Get platform-specific trash function with caching.

    Returns None if no trash function is available.

    Priority order:
    1. send2trash library (if installed)
    2. Platform-specific implementation (macOS, Windows, Linux)
    3. None (will fall back to os.remove)

    Returns:
        Deletion function that moves files to trash, or None if unavailable
    """
    # Try send2trash first (cross-platform library)
    try:
        from send2trash import send2trash

        return send2trash
    except ImportError:
        pass

    # Platform-specific fallbacks
    if sys.platform == "darwin":  # macOS
        return _macos_trash
    elif sys.platform == "win32":  # Windows
        return _windows_trash
    elif sys.platform.startswith("linux"):  # Linux
        return _linux_trash

    return None


def _macos_trash(filepath: str) -> None:
    """Move file to macOS Trash.

    Args:
        filepath: Absolute path to file to trash

    Raises:
        OSError: If trash directory not found or move fails
    """
    import shutil

    trash_dir = os.path.join(os.path.expanduser("~"), ".Trash")
    if not os.path.isdir(trash_dir):
        raise OSError(f"Trash directory not found: {trash_dir}")

    basename = os.path.basename(filepath)
    dest = os.path.join(trash_dir, basename)

    # Handle name conflicts by appending counter
    counter = 1
    while os.path.exists(dest):
        name, ext = os.path.splitext(basename)
        dest = os.path.join(trash_dir, f"{name}_{counter}{ext}")
        counter += 1

    shutil.move(filepath, dest)


def _windows_trash(filepath: str) -> None:
    """Move file to Windows Recycle Bin.

    Args:
        filepath: Absolute path to file to trash

    Raises:
        ImportError: If pywin32 not installed
        OSError: If recycle operation fails
    """
    try:
        from win32com.shell import shell, shellcon

        result = shell.SHFileOperation(
            (
                0,
                shellcon.FO_DELETE,
                filepath,
                None,
                shellcon.FOF_ALLOWUNDO | shellcon.FOF_NOCONFIRMATION,
                None,
                None,
            )
        )
        if result[0] != 0:
            raise OSError(f"Failed to move to recycle bin: error code {result[0]}")
    except ImportError:
        raise ImportError(
            "Windows trash requires pywin32. Install with: pip install pywin32"
        )


def _linux_trash(filepath: str) -> None:
    """Move file to Linux trash (freedesktop.org spec).

    Args:
        filepath: Absolute path to file to trash

    Raises:
        OSError: If trash operation fails
    """
    import shutil

    trash_dir = os.path.join(
        os.getenv("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
        "Trash",
        "files",
    )
    os.makedirs(trash_dir, exist_ok=True)

    basename = os.path.basename(filepath)
    dest = os.path.join(trash_dir, basename)

    # Handle name conflicts by appending counter
    counter = 1
    while os.path.exists(dest):
        name, ext = os.path.splitext(basename)
        dest = os.path.join(trash_dir, f"{name}_{counter}{ext}")
        counter += 1

    shutil.move(filepath, dest)


def make_safe_delete_func(
    permanent_delete_func: DeleteFunc = os.remove, warn_on_fallback: bool = True
) -> DeleteFunc:
    """
    Create a deletion function that tries trash first, falls back to permanent delete.

    Args:
        permanent_delete_func: Function to use if trash is unavailable
        warn_on_fallback: Whether to warn when falling back to permanent delete

    Returns:
        A deletion function that tries trash with fallback
    """
    trash_func = get_platform_trash_func()

    if trash_func is None:
        if warn_on_fallback:
            warn(
                "Trash functionality not available on this platform. "
                "Using permanent deletion instead.\n"
                "To suppress this warning:\n"
                "  - Install cross-platform trash support: pip install send2trash\n"
                "  - OR use permanent deletion explicitly: "
                "Files(path, delete_func=permanent_delete)",
                UserWarning,
                stacklevel=3,
            )
        return permanent_delete_func

    def safe_delete(filepath: str) -> None:
        """Try trash, fall back to permanent delete on failure."""
        try:
            trash_func(filepath)
        except Exception as e:
            if warn_on_fallback:
                warn(
                    f"Failed to move file to trash ({e}). "
                    "Using permanent deletion instead.\n"
                    "To suppress this warning:\n"
                    "  - Install cross-platform trash support: pip install send2trash\n"
                    "  - OR use permanent deletion explicitly: "
                    "Files(path, delete_func=permanent_delete)",
                    UserWarning,
                    stacklevel=4,
                )
            permanent_delete_func(filepath)

    return safe_delete


# Default deletion function - safe delete with warning on fallback
default_delete_func = make_safe_delete_func(warn_on_fallback=True)


def permanent_delete(filepath: str) -> None:
    """Permanently delete a file (no trash, no warnings).

    Args:
        filepath: Path to file to delete

    Use this function when you want permanent deletion without warnings.
    Pass it as delete_func parameter: Files('/my/data', delete_func=permanent_delete)
    """
    os.remove(filepath)


def trash_only(filepath: str) -> None:
    """Move to trash only - raise error if trash unavailable.

    Args:
        filepath: Path to file to trash

    Raises:
        RuntimeError: If trash functionality not available

    Use this function when you want to ensure files are only moved to trash,
    never permanently deleted. Raises error if trash is unavailable.
    Pass it as delete_func parameter: Files('/my/data', delete_func=trash_only)
    """
    trash_func = get_platform_trash_func()
    if trash_func is None:
        raise RuntimeError(
            "Trash functionality not available. "
            "Install 'send2trash' for cross-platform trash support: "
            "pip install send2trash"
        )
    trash_func(filepath)
