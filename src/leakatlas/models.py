"""Target / reference architectures and the shared training loop.

No BatchNorm anywhere: Opacus DP-SGD needs per-sample gradients, which BatchNorm breaks.
"""

import random

import numpy as np
import torch
from torch import nn


class AdultMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class FmnistCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_model(dataset: str, in_dim: int, num_classes: int) -> nn.Module:
    return AdultMLP(in_dim, num_classes) if dataset == "adult" else FmnistCNN(num_classes)


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
