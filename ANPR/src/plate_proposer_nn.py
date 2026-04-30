"""Plate-proposer как nn.Module: классические фильтры → conv-слои на GPU.

Идея взята из lesson_14/LearnSobelCNN.ipynb курса — любой классический оператор
изображения можно реализовать как свёртку и перенести на GPU. Наш CPU-вариант
(`plate_finder.py`) делает sobel → otsu → morph_close → contours — всё это
можно переформулировать как:

  Input (B, 3, H, W)
  → conv_edge  (3×3, Sobel-X init, LEARNABLE)          — character strokes
  → abs + pow(0.7) — compress dynamic range
  → conv_fuse  (1×15, horizontal uniform init, LEARNABLE) — "morph close"
  → sigmoid(shift)                                      — soft threshold (заменяет Otsu)
  → heatmap plate-likeness

Contours — единственный оставшийся CPU-шаг, но работает уже на маленькой
heatmap (после downsample), а не на full-frame. Суммарный speedup 3-5× на GPU.

Начальные веса — классический Sobel (lesson_4) + равномерное ядро для закрытия.
Опционально можно дообучить на bbox'ах YOLO как heatmap-целях (MSE loss) —
это полностью совпадает с lesson_14 рабочим процессом LearnSobelCNN.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlateProposerCNN(nn.Module):
    """Plate-heatmap предиктор по мотивам lesson_14/LearnSobelCNN.

    Стек фильтров курса, переформулированных как свёртки + небольшой CNN-blend:
      Conv3×3 (8 каналов, Sobel-семейство init)  ← ДЗ4 / lesson_14
      ReLU + Conv3×3 (16, learn. feature blend)
      ReLU + Conv1×11 (16, horizontal fuse)       ← аналог морф-closing
      ReLU + Conv1×1  (1, aggregation to heatmap)
      → logits (BCE/focal loss снаружи), sigmoid при inference

    ~1800 params, всё в одном forward на GPU. Sobel-X/Y используются как init,
    остальные каналы инициализируются Xavier — дают capacity для обучения
    под domain (парковочные сцены, угловые ракурсы).
    """

    def __init__(self, downsample: int = 2):
        super().__init__()
        self.downsample = downsample

        # ----- Layer 1: 1→8 edge filters (Sobel-X/Y + variations), 3×3 -----
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=True)
        self._init_edge_filters(self.conv1)

        # ----- Layer 2: 8→16 mixing, 3×3 -----
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=True)

        # ----- Layer 3: 16→16 horizontal fuse, 1×11 (морф-closing аналог) -----
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(1, 11), padding=(0, 5), bias=True)

        # ----- Layer 4: 16→1 heatmap aggregation, 1×1 -----
        self.conv_out = nn.Conv2d(16, 1, 1, bias=True)
        # init bias_out так, чтобы сигмоид начинал около 0.03 (редкий класс)
        with torch.no_grad():
            self.conv_out.bias.fill_(-3.0)

    @staticmethod
    def _init_edge_filters(conv: nn.Conv2d) -> None:
        """Инициализируем первые 2 out-канала Sobel-X/Y, остальные — Xavier."""
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]]) / 4.0
        sobel_y = sobel_x.t().contiguous()
        with torch.no_grad():
            nn.init.xavier_uniform_(conv.weight)
            conv.weight[0, 0] = sobel_x
            conv.weight[1, 0] = sobel_y
            if conv.bias is not None:
                conv.bias.zero_()

    def forward_logits(self, rgb: torch.Tensor) -> torch.Tensor:
        """Input: (B, 3, H, W) uint8 или float [0,1]. Output: (B, 1, H', W') LOGITS."""
        if rgb.dtype == torch.uint8:
            x = rgb.float() / 255.0
        else:
            x = rgb
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if self.downsample > 1:
            x = F.avg_pool2d(x, self.downsample)

        # Grayscale (ДЗ2 weights)
        gw = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        gray = (x * gw).sum(dim=1, keepdim=True)

        f = F.relu(self.conv1(gray))
        f = F.relu(self.conv2(f))
        f = F.relu(self.conv3(f))
        logits = self.conv_out(f)
        return logits

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(rgb))


def heatmap_to_boxes(
    heatmap: np.ndarray,
    downsample: int,
    thr: float = 0.4,
    min_aspect: float = 1.8,
    max_aspect: float = 6.5,
    min_h: int = 10,
    min_w: int = 30,
    max_w_frac: float = 0.5,
    max_h_frac: float = 0.3,
    orig_hw: tuple[int, int] | None = None,
) -> list[tuple[int, int, int, int, float]]:
    """Heatmap (H', W') float → bbox-кандидаты в координатах оригинального кадра."""
    binary = (heatmap > thr).astype(np.uint8) * 255
    # Лёгкий morph-open убирает одиночные peak-шумы
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H_full, W_full = orig_hw if orig_hw is not None else (heatmap.shape[0] * downsample,
                                                          heatmap.shape[1] * downsample)
    boxes: list[tuple[int, int, int, int, float]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # back to full-res coords
        x *= downsample; y *= downsample; w *= downsample; h *= downsample
        if h < min_h or w < min_w:
            continue
        if w > W_full * max_w_frac or h > H_full * max_h_frac:
            continue
        aspect = w / max(h, 1)
        if not (min_aspect <= aspect <= max_aspect):
            continue
        # score = средняя интенсивность heatmap внутри contour
        y0, x0 = y // downsample, x // downsample
        y1, x1 = (y + h) // downsample, (x + w) // downsample
        score = float(heatmap[y0:y1, x0:x1].mean())
        boxes.append((x, y, x + w, y + h, score))
    return boxes


@torch.no_grad()
def propose(
    model: PlateProposerCNN,
    frame_rgb: np.ndarray,
    device: str | torch.device,
    thr: float = 0.4,
) -> list[tuple[int, int, int, int, float]]:
    """Полный пайплайн: numpy RGB frame → list of bbox candidates. GPU forward."""
    H, W = frame_rgb.shape[:2]
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).to(device).float() / 255.0
    heatmap = model(tensor)[0, 0].cpu().numpy()
    return heatmap_to_boxes(heatmap, downsample=model.downsample,
                            thr=thr, orig_hw=(H, W))
