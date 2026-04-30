"""CRNN + CTC для распознавания произвольного текста на номерных знаках.

Зачем отдельно от char_cnn.py:
  char_cnn.py — per-slot классификатор (как ДЗ14) под фиксированный формат.
  CRNN читает произвольную строку любой длины — это нужно для именных
  номеров и нестандартных форматов (дипломатические, спецтехника, тюнинг).

Архитектура — классический Shi et al. 2015 (https://arxiv.org/abs/1507.05717):
  CNN-backbone (в стиле ДЗ14: Conv→BN→ReLU блоки) → BiLSTM×2 → linear → CTC.

Вход:  (B, 1, 32, W)   grayscale, H=32 фиксировано, W свободный
Выход: (T, B, C)        log_softmax по классам, T ≈ W/4
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Charset — покрывает все типы UA-номеров + именные + иностранные
# ---------------------------------------------------------------------------
# 0 зарезервирован под CTC-blank, символы начинаются с индекса 1.

DIGITS = "0123456789"
LATIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CYRILLIC_UA = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
EXTRA = "- "  # тире и пробел для именных/многословных

CHARSET = DIGITS + LATIN + CYRILLIC_UA + EXTRA
BLANK_IDX = 0
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARSET)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1   # +1 под blank


def encode(text: str) -> list[int]:
    """Строка → список индексов. Неизвестные символы пропускаются."""
    return [CHAR_TO_IDX[c] for c in text.upper() if c in CHAR_TO_IDX]


def decode_greedy(log_probs: torch.Tensor) -> list[str]:
    """CTC greedy decode: argmax по классам → collapse повторов → удалить blank.

    log_probs: (T, B, C). Возвращает список длиной B.
    """
    preds = log_probs.argmax(dim=2).transpose(0, 1).cpu().numpy()  # (B, T)
    out: list[str] = []
    for seq in preds:
        chars: list[str] = []
        prev = -1
        for idx in seq:
            idx = int(idx)
            if idx != prev and idx != BLANK_IDX:
                chars.append(IDX_TO_CHAR.get(idx, ""))
            prev = idx
        out.append("".join(chars))
    return out


# ---------------------------------------------------------------------------
# Архитектура
# ---------------------------------------------------------------------------

def _conv_bn_relu(in_ch: int, out_ch: int, k: int = 3, p: int = 1) -> nn.Sequential:
    """Блок из ДЗ14: Conv→BN→ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=p),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CRNN(nn.Module):
    """CRNN для plate-OCR. Вход 32×W grayscale, выход (T, B, num_classes)."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        rnn_hidden: int = 256,
        in_channels: int = 1,
    ):
        super().__init__()

        # CNN-backbone. Высота 32 → 1, ширина W → W/4.
        self.cnn = nn.Sequential(
            _conv_bn_relu(in_channels, 64),
            nn.MaxPool2d(2, 2),                   # 32→16, W→W/2
            _conv_bn_relu(64, 128),
            nn.MaxPool2d(2, 2),                   # 16→8, W/2→W/4
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),         # 8→4, W/4 без изменений
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d((2, 1), (2, 1)),         # 4→2, W/4 без изменений
            nn.Conv2d(512, 512, (2, 2)),          # 2→1, W/4→W/4-1
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Рекуррентная часть: 2× BiLSTM
        self.rnn = nn.Sequential(
            BidirLSTM(512, rnn_hidden, rnn_hidden),
            BidirLSTM(rnn_hidden, rnn_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(x)                       # (B, 512, 1, T)
        assert feats.size(2) == 1, f"ожидался H=1, получили {feats.size(2)}"
        feats = feats.squeeze(2).permute(2, 0, 1)  # (T, B, 512)
        out = self.rnn(feats)                     # (T, B, num_classes)
        return F.log_softmax(out, dim=2)


class BidirLSTM(nn.Module):
    """BiLSTM с линейной проекцией в заданное число каналов."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hidden, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rec, _ = self.rnn(x)             # (T, B, 2H)
        T, B, H = rec.size()
        return self.fc(rec.view(T * B, H)).view(T, B, -1)


# ---------------------------------------------------------------------------
# Вспомогательные утилиты препроцессинга под CRNN (fixed height=32)
# ---------------------------------------------------------------------------

def preprocess_for_crnn(crop_rgb, target_h: int = 32, min_w: int = 128, max_w: int = 256):
    """RGB crop → grayscale tensor (1, 1, 32, W).

    Ресайз по высоте c сохранением aspect ratio, затем pad справа до min_w.
    Pad гарантирует ≥ 31 timestep после backbone (T = W/4 - 1), чего достаточно
    для CTC-декодирования строки 8-10 символов даже на почти квадратных кропах.
    """
    import cv2
    import numpy as np

    if crop_rgb.ndim == 3:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop_rgb

    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return None
    aspect_w = max(8, int(round(w * target_h / h)))
    aspect_w = min(aspect_w, max_w)
    resized = cv2.resize(gray, (aspect_w, target_h), interpolation=cv2.INTER_CUBIC)

    if aspect_w < min_w:
        # Pad справа средним значением фона (обычно белый край номера или околобелый)
        pad_value = int(np.median(resized[:, -2:]))
        canvas = np.full((target_h, min_w), pad_value, dtype=resized.dtype)
        canvas[:, :aspect_w] = resized
        resized = canvas

    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    tensor = (tensor - 0.5) / 0.5                 # [-1, 1]
    return tensor.unsqueeze(0).unsqueeze(0)        # (1, 1, H, W)


def ctc_collate(batch: Iterable[tuple]):
    """DataLoader collate: список (img_tensor, text) → батч с паддингом по ширине."""
    imgs, texts = zip(*batch)
    max_w = max(img.size(-1) for img in imgs)
    padded = torch.zeros(len(imgs), 1, imgs[0].size(-2), max_w)
    for i, img in enumerate(imgs):
        padded[i, :, :, : img.size(-1)] = img
    targets = [torch.tensor(encode(t), dtype=torch.long) for t in texts]
    target_lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_flat = torch.cat(targets) if targets else torch.empty(0, dtype=torch.long)
    return padded, targets_flat, target_lens, list(texts)
