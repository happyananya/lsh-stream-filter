#!/usr/bin/env python3
"""
Summarize Phase 2 JSON outputs into a compact table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Phase 2 results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/phase2"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.results_dir.glob("*.json"))
    if not files:
        print(f"No JSON files found under {args.results_dir}")
        return

    rows: list[dict[str, object]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
        rows.append(
            {
                "method": obj.get("method", path.stem),
                "dataset": obj.get("dataset", "unknown"),
                "recall": obj["metrics"]["recall"],
                "qps": obj["metrics"]["qps"],
                "p95_us": obj["metrics"]["p95_us"],
                "path": str(path),
            }
        )

    print("Phase 2 summary")
    print("-" * 86)
    print(f"{'method':<28} {'dataset':<10} {'recall':>10} {'qps':>14} {'p95_us':>12}")
    print("-" * 86)
    for row in rows:
        print(
            f"{row['method']:<28} {row['dataset']:<10} "
            f"{row['recall']:>10.4f} {row['qps']:>14.2f} {row['p95_us']:>12.2f}"
        )
    print("-" * 86)


if __name__ == "__main__":
    main()
