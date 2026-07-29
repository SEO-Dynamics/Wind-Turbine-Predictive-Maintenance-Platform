"""Filesystem helpers for tabular data, JSON documents and model artifacts.

All writes create parent directories and are atomic where practical (write to a
temporary sibling file, then replace) so a crashed run cannot leave a truncated
artifact behind that a later run would silently load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from wind_turbine_pm.logging_config import get_logger
from wind_turbine_pm.utils.paths import ensure_parent, resolve

logger = get_logger(__name__)


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when a required artifact is missing.

    Carries the command that regenerates it, so API and dashboard layers can
    show an actionable message instead of a bare traceback.
    """

    def __init__(self, path: Path | str, hint: str) -> None:
        """Store the missing path and its remediation hint.

        Args:
            path: The artifact path that could not be found.
            hint: Shell command that produces the artifact.
        """
        self.path = Path(path)
        self.hint = hint
        super().__init__(f"Missing artifact: {self.path}. Generate it with: {hint}")


def _atomic_replace(tmp: Path, target: Path) -> None:
    """Atomically move ``tmp`` over ``target``."""
    tmp.replace(target)


def write_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Write ``data`` as UTF-8 JSON.

    Args:
        data: JSON-serialisable object.
        path: Destination path.
        indent: Indentation width.

    Returns:
        The absolute path written.
    """
    target = ensure_parent(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, default=str, ensure_ascii=False)
        handle.write("\n")
    _atomic_replace(tmp, target)
    logger.debug("Wrote JSON", extra={"path": str(target)})
    return target


def read_json(path: str | Path, hint: str = "make pipeline") -> Any:
    """Read a JSON document.

    Args:
        path: Source path.
        hint: Command shown when the file is missing.

    Returns:
        The parsed JSON document.

    Raises:
        ArtifactNotFoundError: If the file does not exist.
    """
    source = resolve(path)
    if not source.is_file():
        raise ArtifactNotFoundError(source, hint)
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a dataframe as Parquet or CSV based on the file suffix.

    Args:
        frame: Dataframe to persist.
        path: Destination path ending in ``.parquet``/``.pq`` or ``.csv``.

    Returns:
        The absolute path written.

    Raises:
        ValueError: If the suffix is not supported.
    """
    target = ensure_parent(path)
    suffix = target.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(target, index=False)
    elif suffix == ".csv":
        frame.to_csv(target, index=False)
    else:
        raise ValueError(f"Unsupported table format {suffix!r} for {target}")
    logger.debug("Wrote table", extra={"path": str(target), "rows": len(frame)})
    return target


def read_table(path: str | Path, hint: str = "make data") -> pd.DataFrame:
    """Read a Parquet or CSV table.

    Args:
        path: Source path.
        hint: Command shown when the file is missing.

    Returns:
        The loaded dataframe.

    Raises:
        ArtifactNotFoundError: If the file does not exist.
        ValueError: If the suffix is not supported.
    """
    source = resolve(path)
    if not source.is_file():
        raise ArtifactNotFoundError(source, hint)
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported table format {suffix!r} for {source}")


def write_pickle(obj: Any, path: str | Path) -> Path:
    """Persist a Python object with joblib.

    Args:
        obj: Object to serialise (typically a fitted sklearn pipeline).
        path: Destination path.

    Returns:
        The absolute path written.
    """
    target = ensure_parent(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    joblib.dump(obj, tmp)
    _atomic_replace(tmp, target)
    logger.debug("Wrote artifact", extra={"path": str(target)})
    return target


def read_pickle(path: str | Path, hint: str = "make train") -> Any:
    """Load a joblib-serialised object.

    Args:
        path: Source path.
        hint: Command shown when the file is missing.

    Returns:
        The deserialised object.

    Raises:
        ArtifactNotFoundError: If the file does not exist.
    """
    source = resolve(path)
    if not source.is_file():
        raise ArtifactNotFoundError(source, hint)
    return joblib.load(source)


def file_exists(path: str | Path) -> bool:
    """Return whether a project-relative or absolute path is an existing file.

    Args:
        path: Path to check.

    Returns:
        ``True`` when the path exists and is a regular file.
    """
    return resolve(path).is_file()
