#!/usr/bin/env python
"""Generate the synthetic SCADA dataset and write it to ``data/raw/``.

Usage:
    python scripts/generate_synthetic_data.py [--force] [--turbines N] [--months N]
"""

from __future__ import annotations

import argparse
import sys

from wind_turbine_pm.config import Config, load_config
from wind_turbine_pm.data.ingestion import is_synthetic, persist_raw_dataset, raw_data_path
from wind_turbine_pm.data.synthetic import generate_scada_dataset
from wind_turbine_pm.data.validation import validate_scada_frame
from wind_turbine_pm.logging_config import configure_from_config, get_logger
from wind_turbine_pm.utils.io import write_json
from wind_turbine_pm.utils.paths import resolve
from wind_turbine_pm.utils.reproducibility import set_global_seed

logger = get_logger("scripts.generate_synthetic_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when omitted.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate even if the file exists.")
    parser.add_argument("--turbines", type=int, default=None, help="Override synthetic.n_turbines.")
    parser.add_argument("--months", type=int, default=None, help="Override synthetic.months.")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    parser.add_argument(
        "--failures-per-year",
        type=float,
        default=None,
        help=(
            "Override synthetic.failures_per_turbine_per_year. Useful for small CI "
            "datasets, where the default rate can leave a split with no positive labels."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the generation step.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    cfg = load_config()
    configure_from_config(cfg)

    if not is_synthetic(cfg):
        logger.error(
            "data.source is not 'synthetic'; nothing to generate. "
            "Point data.path at your SCADA export and run prepare_data.py instead."
        )
        return 1

    overrides = cfg.to_dict()
    if args.turbines is not None:
        overrides["synthetic"]["n_turbines"] = args.turbines
    if args.months is not None:
        overrides["synthetic"]["months"] = args.months
    if args.seed is not None:
        overrides["synthetic"]["seed"] = args.seed
        overrides["random_seed"] = args.seed
    if args.failures_per_year is not None:
        overrides["synthetic"]["failures_per_turbine_per_year"] = args.failures_per_year
    cfg = Config(overrides)

    set_global_seed(int(cfg.get("random_seed", 42)))

    target = raw_data_path(cfg)
    if target.is_file() and not args.force:
        logger.info(
            "Raw dataset already exists; use --force to regenerate", extra={"path": str(target)}
        )
        return 0

    dataset = generate_scada_dataset(cfg)
    raw_path, sample_path = persist_raw_dataset(dataset, cfg)

    report = validate_scada_frame(dataset, cfg)
    report_path = write_json(
        report.to_dict(),
        resolve(str(cfg.require("paths.artifacts_metrics"))) / "raw_validation_report.json",
    )

    logger.info(
        "Generation complete",
        extra={
            "raw": str(raw_path),
            "sample": str(sample_path),
            "validation_report": str(report_path),
            "rows": len(dataset),
            "errors": len(report.errors),
            "warnings": len(report.warnings),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
