#!/usr/bin/env python3
"""Reproduce LinBreg sparse training of ResNet-18 on CIFAR-10.

The implementation follows Bungert et al. (JMLR 2022), "A Bregman
Learning Framework for Sparse Neural Networks":

    v <- v - learning_rate * grad(loss)
    theta <- prox_{delta J}(delta * v)

For J(theta) = lambda * ||theta||_1 and delta = 1, the proximal map is
soft thresholding. Bias and batch-normalization parameters remain dense.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
PAPER_LINBREG_TRAINED = {"test_accuracy": 90.9, "sparsity": 5.5}
PAPER_LINBREG_THRESHOLDED = {"test_accuracy": 90.2, "sparsity": 3.4}
PAPER_DENSE_ADAM = {"test_accuracy": 93.6, "sparsity": 100.0}


@dataclass
class ExperimentConfig:
    seed: int = 2
    epochs: int = 100
    batch_size: int = 128
    test_batch_size: int = 512
    learning_rate: float = 0.1
    min_learning_rate: float = 1e-5
    lamda: float = 0.1
    delta: float = 1.0
    sparse_init: float = 0.01
    validation_fraction: float = 0.05
    target_sparsity: float = 3.4
    scheduler_patience: int = 5
    scheduler_threshold: float = 1.0
    scheduler_factor: float = 0.5
    num_workers: int = 4
    amp: bool = True


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18CIFAR(nn.Module):
    """CIFAR-10 ResNet-18 with 3x3 stem and internal normalization."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1), persistent=False
        )
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (x - self.mean) / self.std
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_cifar_parquet(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read the uoft-cs/cifar10 Parquet mirrors into uint8 tensors."""
    import pyarrow.parquet as pq
    from PIL import Image

    def load(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        table = pq.read_table(path, columns=["img", "label"])
        images = table.column("img").to_pylist()
        labels = table.column("label").to_pylist()
        decoded = np.empty((len(images), 32, 32, 3), dtype=np.uint8)
        for index, value in enumerate(images):
            payload = value["bytes"] if isinstance(value, dict) else value
            with Image.open(__import__("io").BytesIO(payload)) as image:
                decoded[index] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        x = torch.from_numpy(decoded).permute(0, 3, 1, 2).contiguous()
        y = torch.tensor(labels, dtype=torch.long)
        return x, y

    train_x, train_y = load(data_dir / "train.parquet")
    test_x, test_y = load(data_dir / "test.parquet")
    return train_x, train_y, test_x, test_y


def read_cifar_torchvision(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from torchvision.datasets import CIFAR10

    root = data_dir.parent / "torchvision_data"
    train = CIFAR10(root=str(root), train=True, download=True)
    test = CIFAR10(root=str(root), train=False, download=True)
    train_x = torch.from_numpy(train.data).permute(0, 3, 1, 2).contiguous()
    test_x = torch.from_numpy(test.data).permute(0, 3, 1, 2).contiguous()
    return (
        train_x,
        torch.tensor(train.targets, dtype=torch.long),
        test_x,
        torch.tensor(test.targets, dtype=torch.long),
    )


def load_cifar10(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_parquet = data_dir / "train.parquet"
    test_parquet = data_dir / "test.parquet"
    if train_parquet.exists() and test_parquet.exists():
        return read_cifar_parquet(data_dir)
    return read_cifar_torchvision(data_dir)


def make_loaders(
    config: ExperimentConfig, data_dir: Path
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_x, train_y, test_x, test_y = load_cifar10(data_dir)
    generator = torch.Generator().manual_seed(42)
    train_size = int((1.0 - config.validation_fraction) * len(train_y))
    validation_size = len(train_y) - train_size
    train_set, validation_set = torch.utils.data.random_split(
        TensorDataset(train_x, train_y),
        [train_size, validation_size],
        generator=generator,
    )
    test_set = TensorDataset(test_x, test_y)
    common = {
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        **common,
    )
    validation_loader = DataLoader(
        validation_set, batch_size=config.test_batch_size, shuffle=False, **common
    )
    test_loader = DataLoader(
        test_set, batch_size=config.test_batch_size, shuffle=False, **common
    )
    return train_loader, validation_loader, test_loader


def augment_batch(images: torch.Tensor) -> torch.Tensor:
    """Apply the paper repository's pad-4 crop and random horizontal flip."""
    batch_size = images.shape[0]
    padded = F.pad(images, (4, 4, 4, 4), mode="constant", value=0)
    offsets_y = torch.randint(0, 9, (batch_size,), device=images.device)
    offsets_x = torch.randint(0, 9, (batch_size,), device=images.device)
    rows = offsets_y[:, None] + torch.arange(32, device=images.device)[None, :]
    cols = offsets_x[:, None] + torch.arange(32, device=images.device)[None, :]
    cropped = padded[
        torch.arange(batch_size, device=images.device)[:, None, None, None],
        torch.arange(images.shape[1], device=images.device)[None, :, None, None],
        rows[:, None, :, None],
        cols[:, None, None, :],
    ]
    flip = torch.rand(batch_size, device=images.device) < 0.5
    if flip.any():
        cropped[flip] = torch.flip(cropped[flip], dims=[-1])
    return cropped


def weight_modules(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            yield name, module


@torch.no_grad()
def sparse_initialize(model: ResNet18CIFAR, density: float) -> None:
    """Sample layers from Bernoulli(density) and preserve He variance."""
    for _, module in weight_modules(model):
        weight = module.weight
        fan_in = nn.init._calculate_fan_in_and_fan_out(weight)[0]
        variance = 2.0 / (fan_in * density)
        weight.normal_(0.0, math.sqrt(variance))
        mask = torch.bernoulli(torch.full_like(weight, density))
        weight.mul_(mask)
        if module.bias is not None:
            bound = 1.0 / math.sqrt(fan_in)
            module.bias.uniform_(0.0, bound)

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_parameters()


class LinBreg:
    """Minimal LinBreg update for regularized and dense parameter groups."""

    def __init__(
        self,
        regularized: list[nn.Parameter],
        unregularized: list[nn.Parameter],
        learning_rate: float,
        lamda: float,
        delta: float,
    ):
        self.regularized = regularized
        self.unregularized = unregularized
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.delta = delta
        self.regularized_ids = {id(parameter) for parameter in regularized}
        self.subgradients = {
            id(parameter): self._initial_subgradient(parameter)
            for parameter in [*regularized, *unregularized]
        }

    @torch.no_grad()
    def _initial_subgradient(self, parameter: nn.Parameter) -> torch.Tensor:
        value = parameter.detach().clone()
        if id(parameter) in self.regularized_ids:
            value.add_(self.lamda * torch.sign(parameter))
        return value

    @torch.no_grad()
    def step(self) -> None:
        for parameter in self.regularized:
            if parameter.grad is None:
                continue
            value = self.subgradients[id(parameter)]
            value.add_(parameter.grad, alpha=-self.learning_rate)
            scaled = self.delta * value
            parameter.copy_(
                torch.sign(scaled)
                * torch.clamp(torch.abs(scaled) - self.delta * self.lamda, min=0.0)
            )
        for parameter in self.unregularized:
            if parameter.grad is None:
                continue
            value = self.subgradients[id(parameter)]
            value.add_(parameter.grad, alpha=-self.learning_rate)
            parameter.copy_(self.delta * value)


class PlateauScheduler:
    def __init__(
        self,
        learning_rate: float,
        factor: float,
        patience: int,
        min_learning_rate: float,
        threshold: float,
    ):
        self.learning_rate = learning_rate
        self.factor = factor
        self.patience = patience
        self.min_learning_rate = min_learning_rate
        self.threshold = threshold
        self.best = -math.inf
        self.bad_epochs = 0

    def step(self, validation_accuracy: float) -> None:
        if validation_accuracy > self.best + self.threshold:
            self.best = validation_accuracy
            self.bad_epochs = 0
            return
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.learning_rate = max(
                self.learning_rate * self.factor, self.min_learning_rate
            )
            self.bad_epochs = 0


def train_epoch(
    model: nn.Module,
    optimizer: LinBreg,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in loader:
        images = augment_batch(images.to(device, non_blocking=True)).float().div_(255.0)
        labels = labels.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled and device.type == "cuda",
        ):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.numel()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.numel()
    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True).float().div_(255.0)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled and device.type == "cuda",
        ):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.numel()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.numel()
    return total_loss / total_examples, 100.0 * total_correct / total_examples


@torch.no_grad()
def nonzero_weight_percent(model: nn.Module) -> float:
    nonzero = 0
    total = 0
    for _, module in weight_modules(model):
        nonzero += torch.count_nonzero(module.weight).item()
        total += module.weight.numel()
    return 100.0 * nonzero / total


@torch.no_grad()
def global_magnitude_threshold(model: nn.Module, target_percent: float) -> float:
    values = torch.cat([module.weight.detach().abs().flatten() for _, module in weight_modules(model)])
    keep = max(1, min(values.numel(), int(round(values.numel() * target_percent / 100.0))))
    threshold = torch.topk(values, keep, largest=True, sorted=False).values.min().item()
    for _, module in weight_modules(model):
        module.weight.mul_((module.weight.abs() >= threshold).to(module.weight.dtype))
    return threshold


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: LinBreg,
    scheduler: PlateauScheduler,
    config: ExperimentConfig,
    epoch: int,
    best_accuracy: float,
    best_state: dict[str, torch.Tensor],
    history: list[dict[str, float]],
) -> None:
    parameters = [
        *optimizer.regularized,
        *optimizer.unregularized,
    ]
    subgradient_values = [
        optimizer.subgradients[id(parameter)].detach().cpu() for parameter in parameters
    ]
    torch.save(
        {
            "model": clone_state(model),
            "subgradient_values": subgradient_values,
            "config": asdict(config),
            "scheduler": {
                "learning_rate": scheduler.learning_rate,
                "best": scheduler.best,
                "bad_epochs": scheduler.bad_epochs,
            },
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "best_state": best_state,
            "history": history,
        },
        path,
    )


def load_optimizer_state(
    optimizer: LinBreg, checkpoint: dict, device: torch.device
) -> None:
    parameters = [*optimizer.regularized, *optimizer.unregularized]
    values = checkpoint["subgradient_values"]
    if len(parameters) != len(values):
        raise RuntimeError("Checkpoint parameter layout does not match the model.")
    for parameter, value in zip(parameters, values):
        optimizer.subgradients[id(parameter)] = value.to(device)


def write_results(output_dir: Path, test_accuracy: float, sparsity: float) -> None:
    payload = {
        "test_accuracy": round(float(test_accuracy), 4),
        "sparsity": round(float(sparsity), 4),
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def write_history_csv(output_dir: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def write_figures(output_dir: Path, history: list[dict[str, float]], final: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["train_accuracy"] for row in history], label="Train")
    axes[0].plot(epochs, [row["validation_accuracy"] for row in history], label="Validation")
    axes[0].plot(epochs, [row["test_accuracy"] for row in history], label="Test")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy during LinBreg training")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [row["nonzero_weights"] for row in history], color="#b44b32")
    axes[1].axhline(
        PAPER_LINBREG_THRESHOLDED["sparsity"],
        color="#4f5966",
        linestyle="--",
        label="Paper thresholded: 3.4%",
    )
    axes[1].axhline(
        PAPER_LINBREG_TRAINED["sparsity"],
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
    figure.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(figure)

    labels = [
        "Paper LinBreg",
        "Reproduced LinBreg",
        "Paper + threshold",
        "Reproduced + threshold",
    ]
    accuracy = [
        PAPER_LINBREG_TRAINED["test_accuracy"],
        final["trained_test_accuracy"],
        PAPER_LINBREG_THRESHOLDED["test_accuracy"],
        final["thresholded_test_accuracy"],
    ]
    sparsity = [
        PAPER_LINBREG_TRAINED["sparsity"],
        final["trained_sparsity"],
        PAPER_LINBREG_THRESHOLDED["sparsity"],
        final["thresholded_sparsity"],
    ]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    paper = ["#8c99a8", "#b44b32", "#8c99a8", "#b44b32"]
    axes[0].bar(x, accuracy, color=paper, width=0.62)
    axes[0].set(
        ylabel="Test accuracy (%)",
        title="CIFAR-10 test accuracy",
        ylim=(min(accuracy) - 3.0, max(accuracy) + 1.0),
    )
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    for index, value in enumerate(accuracy):
        axes[0].text(index, value + 0.15, f"{value:.1f}", ha="center", va="bottom")
    axes[1].bar(x, sparsity, color=paper, width=0.62)
    axes[1].set(
        ylabel="Non-zero weights (%)",
        title="Network density",
        ylim=(0.0, max(sparsity) + 1.0),
    )
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    for index, value in enumerate(sparsity):
        axes[1].text(index, value + 0.15, f"{value:.1f}", ha="center", va="bottom")
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output_dir / "paper_comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("work/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--lamda", type=float, default=0.1)
    parser.add_argument("--sparse-init", type=float, default=0.01)
    parser.add_argument("--target-sparsity", type=float, default=3.4)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-threshold", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use two tiny batches per epoch.")
    args = parser.parse_args()

    config = ExperimentConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lamda=args.lamda,
        sparse_init=args.sparse_init,
        target_sparsity=args.target_sparsity,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        scheduler_threshold=args.scheduler_threshold,
        num_workers=args.workers,
        amp=not args.no_amp,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(json.dumps(asdict(config), indent=2))

    train_loader, validation_loader, test_loader = make_loaders(config, args.data_dir)
    if args.smoke:
        train_loader = DataLoader(
            torch.utils.data.Subset(train_loader.dataset, list(range(config.batch_size * 2))),
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        validation_loader = DataLoader(
            torch.utils.data.Subset(
                validation_loader.dataset, list(range(min(512, len(validation_loader.dataset))))
            ),
            batch_size=config.test_batch_size,
        )
        test_loader = DataLoader(
            torch.utils.data.Subset(test_loader.dataset, list(range(512))),
            batch_size=config.test_batch_size,
        )

    model = ResNet18CIFAR().to(device)
    sparse_initialize(model, config.sparse_init)
    regularized = [module.weight for _, module in weight_modules(model)]
    weight_parameter_ids = {id(parameter) for parameter in regularized}
    unregularized = [
        parameter for parameter in model.parameters() if id(parameter) not in weight_parameter_ids
    ]
    optimizer = LinBreg(
        regularized,
        unregularized,
        learning_rate=config.learning_rate,
        lamda=config.lamda,
        delta=config.delta,
    )
    scheduler = PlateauScheduler(
        learning_rate=config.learning_rate,
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_learning_rate=config.min_learning_rate,
        threshold=config.scheduler_threshold,
    )
    history: list[dict[str, float]] = []
    start_epoch = 0
    best_accuracy = -1.0
    best_state = clone_state(model)
    checkpoint_path = args.output_dir / "checkpoint.pt"

    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        load_optimizer_state(optimizer, checkpoint, device)
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        best_accuracy = checkpoint["best_accuracy"]
        best_state = checkpoint["best_state"]
        scheduler.learning_rate = checkpoint["scheduler"]["learning_rate"]
        scheduler.best = checkpoint["scheduler"]["best"]
        scheduler.bad_epochs = checkpoint["scheduler"]["bad_epochs"]
        optimizer.learning_rate = scheduler.learning_rate
        print(f"Resumed from epoch {start_epoch}.")

    print(f"Initial non-zero weights: {nonzero_weight_percent(model):.3f}%")
    start_time = time.time()
    for epoch in range(start_epoch, config.epochs):
        train_loss, train_accuracy = train_epoch(
            model, optimizer, train_loader, device, config.amp
        )
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, device, config.amp
        )
        test_loss, test_accuracy = evaluate(model, test_loader, device, config.amp)
        nonzero = nonzero_weight_percent(model)
        learning_rate = scheduler.learning_rate
        scheduler.step(validation_accuracy)
        optimizer.learning_rate = scheduler.learning_rate
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": 100.0 * train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "nonzero_weights": nonzero,
            "learning_rate": learning_rate,
        }
        history.append(row)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = clone_state(model)
        write_history_csv(args.output_dir, history)
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            config,
            epoch,
            best_accuracy,
            best_state,
            history,
        )
        elapsed = (time.time() - start_time) / 60.0
        print(
            f"Epoch {epoch + 1:03d}/{config.epochs:03d} "
            f"train={row['train_accuracy']:.2f}% "
            f"val={validation_accuracy:.2f}% "
            f"test={test_accuracy:.2f}% "
            f"nnz={nonzero:.2f}% "
            f"lr={learning_rate:.5g} "
            f"elapsed={elapsed:.1f}m",
            flush=True,
        )
        if args.smoke and epoch + 1 >= 2:
            break

    model.load_state_dict(best_state)
    trained_loss, trained_accuracy = evaluate(model, test_loader, device, config.amp)
    trained_sparsity = nonzero_weight_percent(model)
    threshold = global_magnitude_threshold(model, config.target_sparsity)
    thresholded_loss, thresholded_accuracy = evaluate(model, test_loader, device, config.amp)
    thresholded_sparsity = nonzero_weight_percent(model)
    write_results(args.output_dir, thresholded_accuracy, thresholded_sparsity)
    metrics = {
        "config": asdict(config),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "best_validation_accuracy": best_accuracy,
        "trained_test_accuracy": trained_accuracy,
        "trained_test_loss": trained_loss,
        "trained_sparsity": trained_sparsity,
        "threshold": threshold,
        "thresholded_test_accuracy": thresholded_accuracy,
        "thresholded_test_loss": thresholded_loss,
        "thresholded_sparsity": thresholded_sparsity,
        "paper_linbreg": PAPER_LINBREG_TRAINED,
        "paper_linbreg_thresholded": PAPER_LINBREG_THRESHOLDED,
        "paper_dense_adam": PAPER_DENSE_ADAM,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    write_figures(args.output_dir, history, metrics)
    print(
        f"Final: test={thresholded_accuracy:.2f}% "
        f"nnz={thresholded_sparsity:.2f}% "
        f"(trained: {trained_accuracy:.2f}%, {trained_sparsity:.2f}%)"
    )


if __name__ == "__main__":
    main()
