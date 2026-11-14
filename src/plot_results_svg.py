#!/usr/bin/env python3
"""Generate compact, dependency-free SVG result charts for GitHub."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def polyline(points: list[tuple[float, float]], color: str, width: float = 2.2) -> str:
    encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{encoded}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linejoin="round"/>'
    )


def main() -> None:
    with (ROOT / "results/history.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    metrics = json.loads((ROOT / "results/metrics.json").read_text(encoding="utf-8"))

    def x_epoch(epoch: int) -> float:
        return 70 + epoch / 150 * 390

    def y_accuracy(value: float) -> float:
        return 350 - (value - 20) / 80 * 280

    def y_density(value: float) -> float:
        return 350 - (value - 1) / 4.8 * 280

    train = [(x_epoch(int(row["epoch"])), y_accuracy(float(row["train_accuracy"]))) for row in history]
    validation = [
        (x_epoch(int(row["epoch"])), y_accuracy(float(row["validation_accuracy"])))
        for row in history
    ]
    test = [(x_epoch(int(row["epoch"])), y_accuracy(float(row["test_accuracy"]))) for row in history]
    density = [
        (x_epoch(int(row["epoch"])), y_density(float(row["nonzero_weights"])))
        for row in history
    ]
    paper_thresholded = y_density(3.4)
    paper_trained = y_density(5.5)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 430" role="img" aria-labelledby="title desc">
<title id="title">LinBreg training curves</title>
<desc id="desc">Accuracy and non-zero weight growth over 150 epochs.</desc>
<rect width="1000" height="430" fill="#111722"/>
<g font-family="Arial, sans-serif" fill="#d9e1ec" font-size="14">
<text x="70" y="34" font-size="20" font-weight="bold">Accuracy during LinBreg training</text>
<text x="560" y="34" font-size="20" font-weight="bold">Inverse scale-space parameter growth</text>
<line x1="70" y1="350" x2="460" y2="350" stroke="#64748b"/>
<line x1="70" y1="60" x2="70" y2="350" stroke="#64748b"/>
<line x1="560" y1="350" x2="950" y2="350" stroke="#64748b"/>
<line x1="560" y1="60" x2="560" y2="350" stroke="#64748b"/>
<text x="210" y="400">Epoch</text><text x="18" y="220" transform="rotate(-90 18 220)">Accuracy (%)</text>
<text x="700" y="400">Epoch</text><text x="508" y="220" transform="rotate(-90 508 220)">Non-zero weights (%)</text>
<g stroke="#283447" stroke-width="1">
<line x1="70" y1="277.5" x2="460" y2="277.5"/><line x1="70" y1="205" x2="460" y2="205"/><line x1="70" y1="132.5" x2="460" y2="132.5"/><line x1="70" y1="60" x2="460" y2="60"/>
<line x1="560" y1="292.5" x2="950" y2="292.5"/><line x1="560" y1="234.5" x2="950" y2="234.5"/><line x1="560" y1="176.5" x2="950" y2="176.5"/><line x1="560" y1="118.5" x2="950" y2="118.5"/><line x1="560" y1="60" x2="950" y2="60"/>
</g>
{polyline(train, "#4da3ff")}
{polyline(validation, "#ff9f43")}
{polyline(test, "#4fd18b")}
{polyline(density, "#ff6b57", 2.5)}
<line x1="560" y1="{paper_thresholded:.1f}" x2="950" y2="{paper_thresholded:.1f}" stroke="#9aa8b8" stroke-dasharray="7 5"/>
<line x1="560" y1="{paper_trained:.1f}" x2="950" y2="{paper_trained:.1f}" stroke="#d1aa57" stroke-dasharray="2 5"/>
<rect x="245" y="75" width="195" height="80" rx="5" fill="#182235" opacity=".92"/>
<line x1="260" y1="95" x2="295" y2="95" stroke="#4da3ff" stroke-width="3"/><text x="305" y="100">Train</text>
<line x1="260" y1="116" x2="295" y2="116" stroke="#ff9f43" stroke-width="3"/><text x="305" y="121">Validation</text>
<line x1="260" y1="137" x2="295" y2="137" stroke="#4fd18b" stroke-width="3"/><text x="305" y="142">Test</text>
<line x1="700" y1="286" x2="735" y2="286" stroke="#9aa8b8" stroke-width="2" stroke-dasharray="7 5"/><text x="745" y="291">Paper thresholded: 3.4%</text>
<line x1="700" y1="307" x2="735" y2="307" stroke="#d1aa57" stroke-width="2" stroke-dasharray="2 5"/><text x="745" y="312">Paper trained: 5.5%</text>
</g></svg>"""
    (ASSETS / "training_curves.svg").write_text(svg, encoding="utf-8")

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
    colors = ["#8c99a8", "#d06045", "#8c99a8", "#d06045"]
    xs = [80, 292, 504, 716]
    acc_bars = []
    den_bars = []
    texts = []
    for index, x in enumerate(xs):
        accuracy_height = accuracy[index] / 95 * 260
        density_height = sparsity[index] / 6.5 * 260
        acc_bars.append(f'<rect x="{x}" y="{350-accuracy_height:.1f}" width="120" height="{accuracy_height:.1f}" fill="{colors[index]}"/>')
        den_bars.append(f'<rect x="{x}" y="{350-density_height:.1f}" width="120" height="{density_height:.1f}" fill="{colors[index]}"/>')
        texts.append(f'<text x="{x+60}" y="{342-accuracy_height:.1f}" text-anchor="middle">{accuracy[index]:.1f}</text>')
        texts.append(f'<text x="{x+60}" y="{342-density_height:.1f}" text-anchor="middle">{sparsity[index]:.1f}</text>')
        texts.append(f'<text x="{x+8}" y="374" transform="rotate(18 {x+8} 374)">{labels[index]}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 430" role="img" aria-labelledby="title desc">
<title id="title">Paper comparison</title><desc id="desc">CIFAR-10 accuracy and density compared with the paper.</desc>
<rect width="1000" height="430" fill="#111722"/>
<g font-family="Arial, sans-serif" fill="#d9e1ec" font-size="14">
<text x="80" y="34" font-size="20" font-weight="bold">CIFAR-10 test accuracy</text>
<text x="560" y="34" font-size="20" font-weight="bold">Network density</text>
<line x1="60" y1="350" x2="490" y2="350" stroke="#64748b"/><line x1="540" y1="350" x2="980" y2="350" stroke="#64748b"/>
<text x="20" y="210" transform="rotate(-90 20 210)">Test accuracy (%)</text>
<text x="500" y="210" transform="rotate(-90 500 210)">Non-zero weights (%)</text>
{''.join(acc_bars)}{''.join(den_bars)}{''.join(texts)}
</g></svg>"""
    (ASSETS / "paper_comparison.svg").write_text(svg, encoding="utf-8")

    for old_name in ("paper_comparison.png", "training_curves.png"):
        (ASSETS / old_name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
