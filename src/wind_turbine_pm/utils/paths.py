"""Centralised, machine-independent path handling.

The repository root is discovered by walking upwards from this file until a
directory containing ``pyproject.toml`` is found.  This keeps every path in the
project relative to the checkout and free of hard-coded personal directories.

The root can be overridden with the ``WTPM_PROJECT_ROOT`` environment variable,
which is what the Docker images use.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_ROOT_MARKER = "pyproject.toml"
_ENV_ROOT = "WTPM_PROJECT_ROOT"


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the absolute path of the repository root.

    Returns:
        Absolute :class:`~pathlib.Path` of the project root.

    Raises:
        RuntimeError: If no ``pyproject.toml`` marker can be located and the
            ``WTPM_PROJECT_ROOT`` environment variable is unset.
    """
    override = os.environ.get(_ENV_ROOT)
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(f"{_ENV_ROOT}={override!r} is not an existing directory")
        return root

    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / _ROOT_MARKER).is_file():
            return candidate
    raise RuntimeError(
        f"Could not locate {_ROOT_MARKER} above {here}. Set {_ENV_ROOT} to the repository root."
    )


def resolve(relative: str | Path) -> Path:
    """Resolve a project-relative path to an absolute path.

    Absolute inputs are returned unchanged, which lets configuration files point
    at data outside the repository when needed.

    Args:
        relative: Path relative to the project root, or an absolute path.

    Returns:
        The absolute path.
    """
    path = Path(relative)
    return path if path.is_absolute() else (project_root() / path)


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) as a directory if it does not exist.

    Args:
        path: Directory path, project-relative or absolute.

    Returns:
        The resolved absolute directory path.
    """
    directory = resolve(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent(path: str | Path) -> Path:
    """Ensure the *parent* directory of a file path exists.

    Args:
        path: File path, project-relative or absolute.

    Returns:
        The resolved absolute file path.
    """
    file_path = resolve(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path
