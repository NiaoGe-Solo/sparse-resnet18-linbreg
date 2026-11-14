#!/usr/bin/env python3
"""Regenerate the vector figures embedded in the README."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def main() -> None:
    with (ROOT / "results/history.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    metrics = json.loads((ROOT / "results/metrics.json").read_text(encoding="utf-8"))
    epochs = [int(row["epoch"]) for row in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [float(row["train_accuracy"]) for row in history], label="Train")
    axes[0].plot(
        epochs, [float(row["validation_accuracy"]) for row in history], label="Validation"
    )
    axes[0].plot(epochs, [float(row["test_accuracy"]) for row in history], label="Test")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy during LinBreg training")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        epochs,
        [float(row["nonzero_weights"]) for row in history],
        color="#b44b32",
    )
    axes[1].axhline(
        metrics["paper_linbreg_thresholded"]["sparsity"],
        color="#4f5966",
        linestyle="--",
        label="Paper thresholded: 3.4%",
    )
    axes[1].axhline(
        metrics["paper_linbreg"]["sparsity"],
        color="#8c6d31",
        linestyle=":",
        label="Paper trained: 5.5%",
    )
    axes[1].set(
        xlabel="Epoch",
        ylabel="Non-zero weights (%)",
        title="Inverse scale-space parameter growth",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(ASSETS / "training_curves.svg", format="svg")
    plt.close(figure)

    labels = ["Paper LinBreg", "This run", "Paper + threshold", "This run + threshold"]
    accuracy = [
        metrics["paper_linbreg"]["test_accuracy"],
        metrics["trained_test_accuracy"],
        metrics["paper_linbreg_thresholded"]["test_accuracy"],
        metrics["thresholded_test_accuracy"],
    ]
    sparsity = [
        metrics["paper_linbreg"]["sparsity"],
        metrics["trained_sparsity"],
        metrics["paper_linbreg_thresholded"]["sparsity"],
        metrics["thresholded_sparsity"],
    ]
    x = range(len(labels))
    colors = ["#8c99a8", "#b44b32", "#8c99a8", "#b44b32"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x, accuracy, color=colors, width=0.62)
    axes[0].set(
        ylabel="Test accuracy (%)",
        title="CIFAR-10 test accuracy",
        ylim=(min(accuracy) - 3.0, max(accuracy) + 1.0),
    )
    axes[1].bar(x, sparsity, color=colors, width=0.62)
    axes[1].set(
        ylabel="Non-zero weights (%)",
        title="Network density",
        ylim=(0.0, max(sparsity) + 1.0),
    )
    for axis, values in zip(axes, (accuracy, sparsity)):
        axis.set_xticks(list(x), labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
        for index, value in enumerate(values):
            axis.text(index, value + 0.15, f"{value:.1f}", ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(ASSETS / "paper_comparison.svg", format="svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
