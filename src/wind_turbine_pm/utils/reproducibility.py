"""Deterministic seeding helpers.

Every stochastic component in the project draws from a generator derived from a
single configured seed, so a given configuration always produces byte-identical
data and model artifacts.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and the ``PYTHONHASHSEED`` environment variable.

    ``PYTHONHASHSEED`` only affects subprocesses started after this call; it is
    set so that scripts spawning workers inherit deterministic hashing.

    Args:
        seed: Non-negative base seed.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))


def derive_seed(base_seed: int, *tokens: str | int) -> int:
    """Derive a stable child seed from a base seed and arbitrary tokens.

    Using a hash of the tokens (rather than ``base_seed + i``) keeps per-turbine
    streams independent even when turbine counts change between runs.

    Args:
        base_seed: The configured base seed.
        *tokens: Identifiers such as a turbine id or a stage name.

    Returns:
        A seed in ``[0, 2**32)``.
    """
    payload = "|".join([str(base_seed), *(str(token) for token in tokens)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def rng_for(base_seed: int, *tokens: str | int) -> np.random.Generator:
    """Return an independent NumPy generator for a named stream.

    Args:
        base_seed: The configured base seed.
        *tokens: Identifiers naming the stream.

    Returns:
        A seeded :class:`numpy.random.Generator`.
    """
    return np.random.default_rng(derive_seed(base_seed, *tokens))
