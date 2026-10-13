"""``get_app_folder`` ignores an empty or relative platform variable (XDG spec;
https://github.com/i2mint/dol/issues/120)."""

import os

import pytest

from dol.util import APP_FOLDER_STANDARDS, get_app_folder

posix_only = pytest.mark.skipif(os.name == "nt", reason="XDG semantics are POSIX")


@posix_only
@pytest.mark.parametrize("kind", ["config", "data", "cache", "state", "runtime"])
def test_get_app_folder_ignores_empty_and_relative_env_var(monkeypatch, tmp_path, kind):
    env_var, default = APP_FOLDER_STANDARDS[kind]
    expected_default = os.path.expanduser(default)

    monkeypatch.delenv(env_var, raising=False)
    assert get_app_folder(kind) == expected_default
    for bad in ("", "relative/dir", "."):
        monkeypatch.setenv(env_var, bad)
        assert get_app_folder(kind) == expected_default
    monkeypatch.setenv(env_var, str(tmp_path))
    assert get_app_folder(kind) == str(tmp_path)
    monkeypatch.setenv(env_var, "~/somewhere")  # ``~`` is still expanded
    assert get_app_folder(kind) == os.path.expanduser("~/somewhere")
