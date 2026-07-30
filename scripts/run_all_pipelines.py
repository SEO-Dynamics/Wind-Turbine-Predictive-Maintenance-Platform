#!/usr/bin/env python
"""Run failure, health and anomaly pipelines against one shared fleet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_anomaly_pipeline  # noqa: E402
import run_failure_pipeline  # noqa: E402
import run_health_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    failure_flags = ["--force"] if args.force else []
    health_flags = (["--force"] if args.force else []) + (["--strict"] if args.strict else [])
    anomaly_flags = (["--force"] if args.force else []) + (["--strict"] if args.strict else [])
    for stage, flags in (
        (run_failure_pipeline.main, failure_flags),
        (run_health_pipeline.main, health_flags),
        (run_anomaly_pipeline.main, anomaly_flags),
    ):
        code = stage(flags)
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
