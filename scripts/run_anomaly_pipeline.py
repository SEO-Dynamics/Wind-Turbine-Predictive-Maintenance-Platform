#!/usr/bin/env python
"""Run Stage 3 anomaly preparation and training against the shared raw fleet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_synthetic_data  # noqa: E402
import prepare_anomaly_data  # noqa: E402
import train_anomaly_model  # noqa: E402
from wind_turbine_pm.anomaly.config import load_anomaly_config  # noqa: E402
from wind_turbine_pm.data.ingestion import is_synthetic, raw_data_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_anomaly_config()
    if is_synthetic(cfg) and (args.force or not raw_data_path(cfg).is_file()):
        code = generate_synthetic_data.main(["--force"] if args.force else [])
        if code:
            return code
    flags = (["--force"] if args.force else []) + (["--strict"] if args.strict else [])
    code = prepare_anomaly_data.main(flags)
    return code or train_anomaly_model.main([])


if __name__ == "__main__":
    sys.exit(main())
