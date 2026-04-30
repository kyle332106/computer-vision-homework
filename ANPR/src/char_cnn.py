"""Собственная CNN для классификации символов UA номеров.

Архитектура — прямое переиспользование паттернов из ДЗ13 (baseline)
и ДЗ14 (improved: BatchNorm + Dropout + 2×Conv блоки) в PyTorch.

Вход:  32×32 grayscale crop символа (1 канал)
Выход: softmax над 24 классами ['0'..'9','A','B','C','E','H','I','K','M','O','P','T','X','Y','Z']
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


CHAR_CLASSES = list("0123456789ABCEHIKMOPTXYZ")
NUM_CLASSES = len(CHAR_CLASSES)  # 24


class CharCNN(nn.Module):
    """ДЗ14-style: Conv→BN→ReLU→Conv→BN→ReLU→MaxPool→Dropout × 2 → Dense → Dropout → softmax."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_conv: float = 0.25, dropout_fc: float = 0.5):
        super().__init__()
        # Блок 1
        self.conv1a = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout2d(dropout_conv)

        # Блок 2
        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout2d(dropout_conv)

        # Классификатор — после двух maxpool 32→16→8, каналов 64
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.drop_fc = nn.Dropout(dropout_fc)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = F.max_pool2d(x, 2)
        x = self.drop1(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = F.max_pool2d(x, 2)
        x = self.drop2(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        return self.fc2(x)


def class_idx(ch: str) -> int:
    return CHAR_CLASSES.index(ch.upper())


def idx_class(i: int) -> str:
    return CHAR_CLASSES[i]
