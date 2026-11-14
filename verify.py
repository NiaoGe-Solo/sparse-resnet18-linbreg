#!/usr/bin/env python3
"""Validate the scored contract and the supporting reproduction artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def require_finite_number(value: object, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{name}={number} is outside [{lower}, {upper}]")
    return number


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    results = load_json(output_dir / "results/results.json")
    if set(results) != {"test_accuracy", "sparsity"}:
        raise ValueError("results.json must contain exactly test_accuracy and sparsity")
    accuracy = require_finite_number(results["test_accuracy"], "test_accuracy", 0.0, 100.0)
    sparsity = require_finite_number(results["sparsity"], "sparsity", 0.0, 100.0)

    metrics = load_json(output_dir / "results/metrics.json")
    if metrics["thresholded_test_accuracy"] != accuracy:
        raise ValueError("metrics.json and results.json disagree on test_accuracy")
    if not math.isclose(float(metrics["thresholded_sparsity"]), sparsity, abs_tol=1e-4):
        raise ValueError("metrics.json and results.json disagree on sparsity")
    if not 3.0 <= sparsity <= 3.8:
        raise ValueError(f"Reproduction did not hit the paper's 3.4% density target: {sparsity}")

    with (output_dir / "results/history.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("training_history.csv does not contain a completed run")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise ValueError("Training history epochs are not contiguous")

    required_artifacts = [
        "assets/training_curves.svg",
        "assets/paper_comparison.svg",
    ]
    for name in required_artifacts:
        path = output_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    print(
        f"PASS: results.json is valid; test_accuracy={accuracy:.2f}%, "
        f"sparsity={sparsity:.2f}%, epochs={len(rows)}"
    )


if __name__ == "__main__":
    main()
