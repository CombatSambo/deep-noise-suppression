from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional


def _read_metric_means(path: Path) -> Dict[str, float]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    means: Dict[str, float] = {}
    for key in rows[0].keys():
        if key == "file":
            continue
        values: List[float] = []
        for row in rows:
            value = row.get(key)
            if value is None or value == "":
                continue
            try:
                values.append(float(value))
            except ValueError:
                continue
        if values:
            means[key] = mean(values)
    return means


def _selected_keys(noisy: Dict[str, float], enhanced: Dict[str, float], keys: Optional[Iterable[str]]) -> List[str]:
    if keys:
        requested = list(keys)
        missing = [k for k in requested if k not in noisy or k not in enhanced]
        if missing:
            raise ValueError(f"Requested metric keys missing from one or both CSVs: {missing}")
        return requested
    return sorted(set(noisy.keys()) & set(enhanced.keys()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare mean metrics between noisy and enhanced CSV files.")
    parser.add_argument("--noisy-csv", type=str, required=True)
    parser.add_argument("--enhanced-csv", type=str, required=True)
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional metric names to print. Defaults to all shared numeric columns.",
    )
    args = parser.parse_args()

    noisy_means = _read_metric_means(Path(args.noisy_csv))
    enhanced_means = _read_metric_means(Path(args.enhanced_csv))
    keys = _selected_keys(noisy_means, enhanced_means, args.metrics)

    print("metric,noisy_mean,enhanced_mean,diff_enhanced_minus_noisy")
    for key in keys:
        noisy = noisy_means[key]
        enhanced = enhanced_means[key]
        print(f"{key},{noisy:.6f},{enhanced:.6f},{enhanced - noisy:+.6f}")


if __name__ == "__main__":
    main()
