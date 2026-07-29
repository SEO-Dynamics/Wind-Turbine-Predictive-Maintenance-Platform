#!/usr/bin/env python
"""Run the complete Turbine Health Monitoring pipeline end to end.

Equivalent to running, in order:

    python scripts/generate_synthetic_data.py   (only when data.source is synthetic)
    python scripts/prepare_health_data.py
    python scripts/train_health_model.py

Each stage is invoked in-process, so a failure stops the pipeline immediately
with the originating traceback rather than a generic non-zero exit code.

The generation stage is shared with the Failure Prediction pipeline and is
skipped when the raw dataset already exists, so running both pipelines does not
regenerate (and therefore does not change) the fleet the two modules describe.

Usage:
    python scripts/run_health_pipeline.py [--force] [--skip-figures]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_synthetic_data  # noqa: E402
import prepare_health_data  # noqa: E402
import train_health_model  # noqa: E402
from wind_turbine_pm.data.ingestion import is_synthetic, raw_data_path  # noqa: E402
from wind_turbine_pm.health.config import load_health_config  # noqa: E402
from wind_turbine_pm.logging_config import configure_from_config, get_logger  # noqa: E402

logger = get_logger("scripts.run_health_pipeline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when omitted.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Regenerate data and rebuild the features."
    )
    parser.add_argument("--skip-figures", action="store_true", help="Skip the evaluation figures.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort preparation on any validation or rule-verification problem.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run every health pipeline stage in order.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    cfg = load_health_config()
    configure_from_config(cfg)

    stages: list[tuple[str, Callable[[], int]]] = []
    if is_synthetic(cfg):
        if args.force or not raw_data_path(cfg).is_file():
            stages.append(
                (
                    "generate",
                    lambda: generate_synthetic_data.main(["--force"] if args.force else []),
                )
            )
        else:
            logger.info(
                "Raw dataset already present; reusing it so both modules describe the same fleet",
                extra={"path": str(raw_data_path(cfg))},
            )
    else:
        logger.info("data.source is not synthetic; skipping generation stage")

    prepare_flags = (["--force"] if args.force else []) + (["--strict"] if args.strict else [])
    stages.extend(
        [
            ("prepare-health", lambda: prepare_health_data.main(prepare_flags)),
            (
                "train-health",
                lambda: train_health_model.main(["--skip-figures"] if args.skip_figures else []),
            ),
        ]
    )

    total_start = time.perf_counter()
    for name, stage in stages:
        logger.info("=== Health pipeline stage: %s ===", name)
        start = time.perf_counter()
        code = stage()
        elapsed = time.perf_counter() - start
        if code != 0:
            logger.error("Stage failed", extra={"stage": name, "exit_code": code})
            return code
        logger.info("Stage complete", extra={"stage": name, "seconds": round(elapsed, 1)})

    logger.info(
        "Health pipeline complete",
        extra={"total_seconds": round(time.perf_counter() - total_start, 1)},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
