"""Model architectures per the paper's specifications.

  - LeNetMNIST : modified LeNet (MNIST / synthetic)
  - TanhLeNet  : conv 16@8x8 s2 -> conv 32@4x4 s2 -> maxpool 4x4 s1 -> FC 32
                 -> FC 10, tanh activations (Fashion-MNIST)
  - AlexNetS   : 2x conv 64@5x5 (maxpool 3x3 s2 after each) -> FC 384 -> FC 192
                 -> FC 10 (CIFAR-10)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LeNetMNIST(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class TanhLeNet(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 8, stride=2, padding=1), nn.Tanh(),
            nn.Conv2d(16, 32, 4, stride=2), nn.Tanh(),
            nn.MaxPool2d(4, stride=1),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, in_ch, 28, 28)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 32), nn.Tanh(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class AlexNetS(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, in_ch, 32, 32)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 384), nn.ReLU(),
            nn.Linear(384, 192), nn.ReLU(), nn.Linear(192, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(dataset: str, input_shape, num_classes: int) -> nn.Module:
    d = dataset.lower()
    c = input_shape[0]
    if d in ("mnist", "synthetic"):
        return LeNetMNIST(c, num_classes)
    if d in ("fmnist", "fashionmnist", "fashion-mnist"):
        return TanhLeNet(c, num_classes)
    if d == "cifar10":
        return AlexNetS(c, num_classes)
    raise ValueError(f"no model registered for {dataset}")
