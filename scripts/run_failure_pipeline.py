#!/usr/bin/env python
"""Run the complete Failure Prediction pipeline end to end.

Equivalent to running, in order:

    python scripts/generate_synthetic_data.py
    python scripts/prepare_data.py
    python scripts/train_failure_model.py
    python scripts/evaluate_failure_model.py

Each stage is invoked in-process, so a failure stops the pipeline immediately
with the originating traceback rather than a generic non-zero exit code.

Usage:
    python scripts/run_failure_pipeline.py [--force] [--skip-shap]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_failure_model  # noqa: E402
import generate_synthetic_data  # noqa: E402
import prepare_data  # noqa: E402
import train_failure_model  # noqa: E402
from wind_turbine_pm.config import load_config  # noqa: E402
from wind_turbine_pm.data.ingestion import is_synthetic  # noqa: E402
from wind_turbine_pm.logging_config import configure_from_config, get_logger  # noqa: E402

logger = get_logger("scripts.run_failure_pipeline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when omitted.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate data and features.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP figures.")
    parser.add_argument(
        "--threshold-method",
        choices=["f2", "cost"],
        default=None,
        help="Override threshold.method.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run every pipeline stage in order.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    cfg = load_config()
    configure_from_config(cfg)

    stages: list[tuple[str, callable]] = []
    if is_synthetic(cfg):
        stages.append(
            ("generate", lambda: generate_synthetic_data.main(["--force"] if args.force else []))
        )
    else:
        logger.info("data.source is not synthetic; skipping generation stage")

    stages.extend(
        [
            ("prepare", lambda: prepare_data.main(["--force"] if args.force else [])),
            (
                "train",
                lambda: train_failure_model.main(
                    ["--threshold-method", args.threshold_method] if args.threshold_method else []
                ),
            ),
            (
                "evaluate",
                lambda: evaluate_failure_model.main(["--skip-shap"] if args.skip_shap else []),
            ),
        ]
    )

    total_start = time.perf_counter()
    for name, stage in stages:
        logger.info("=== Pipeline stage: %s ===", name)
        start = time.perf_counter()
        code = stage()
        elapsed = time.perf_counter() - start
        if code != 0:
            logger.error("Stage failed", extra={"stage": name, "exit_code": code})
            return code
        logger.info("Stage complete", extra={"stage": name, "seconds": round(elapsed, 1)})

    logger.info(
        "Pipeline complete", extra={"total_seconds": round(time.perf_counter() - total_start, 1)}
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
