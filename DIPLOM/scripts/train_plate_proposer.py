"""Обучение PlateProposerCNN по методу из lesson_14/LearnSobelCNN.ipynb.

Идея: цель — heatmap, в которой вокруг центра каждого plate-bbox'а стоит
Gaussian-блоб. Обучаем MSE между предсказанной heatmap и таргетом. Начальные
веса — Sobel-X + uniform fuse, так что уже работает до обучения, но после
обучения становится в разы точнее и даёт меньше false positives.

Использование:
    python scripts/train_plate_proposer.py --epochs 6 --batch 8
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io_utils import imread
from src.plate_proposer_nn import PlateProposerCNN


class PlateHeatmapDataset(Dataset):
    """Читает изображения + YOLO-labels из yolo_plates/, рендерит Gaussian-target heatmap."""

    def __init__(self, img_dir: Path, lbl_dir: Path, size: int = 1024, ds: int = 2, sigma_frac: float = 0.25):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.size = size
        self.ds = ds
        self.sigma_frac = sigma_frac
        self.files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

    def __len__(self):
        return len(self.files)

    def _load_labels(self, stem: str, w: int, h: int):
        lbl = self.lbl_dir / f"{stem}.txt"
        if not lbl.exists():
            return []
        out = []
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            out.append((cx * w, cy * h, bw * w, bh * h))
        return out

    def __getitem__(self, i):
        img = imread(self.files[i])
        if img is None:
            return torch.zeros(3, self.size, self.size), torch.zeros(1, self.size // self.ds, self.size // self.ds)
        H, W = img.shape[:2]
        boxes = self._load_labels(self.files[i].stem, W, H)

        # Letterbox до self.size
        scale = self.size / max(H, W)
        new_h, new_w = int(H * scale), int(W * scale)
        resized = cv2.resize(img, (new_w, new_h))
        pad_top = (self.size - new_h) // 2
        pad_left = (self.size - new_w) // 2
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        # Target heatmap: Гауссиана вокруг каждого plate-центра
        ht_h, ht_w = self.size // self.ds, self.size // self.ds
        target = np.zeros((ht_h, ht_w), dtype=np.float32)
        for (cx, cy, bw, bh) in boxes:
            # масштабируем bbox в координаты heatmap
            cx_h = (cx * scale + pad_left) / self.ds
            cy_h = (cy * scale + pad_top) / self.ds
            bw_h = bw * scale / self.ds
            bh_h = bh * scale / self.ds
            sigma_x = max(1.5, bw_h * self.sigma_frac)
            sigma_y = max(1.0, bh_h * self.sigma_frac)
            # rasterize Gaussian в ограниченном окне
            x0 = max(0, int(cx_h - 3 * sigma_x))
            x1 = min(ht_w, int(cx_h + 3 * sigma_x))
            y0 = max(0, int(cy_h - 3 * sigma_y))
            y1 = min(ht_h, int(cy_h + 3 * sigma_y))
            if x1 <= x0 or y1 <= y0:
                continue
            ys, xs = np.mgrid[y0:y1, x0:x1]
            g = np.exp(-(((xs - cx_h) / sigma_x) ** 2 + ((ys - cy_h) / sigma_y) ** 2) / 2)
            target[y0:y1, x0:x1] = np.maximum(target[y0:y1, x0:x1], g)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return rgb_t, torch.from_numpy(target).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    yolo_root = Path("D:/Новая папка/alpr_data/yolo_plates")

    train_ds = PlateHeatmapDataset(
        yolo_root / "images/train", yolo_root / "labels/train",
        size=args.size,
    )
    val_ds = PlateHeatmapDataset(
        yolo_root / "images/val", yolo_root / "labels/val",
        size=args.size,
    )
    print(f"train={len(train_ds)}  val={len(val_ds)}  size={args.size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PlateProposerCNN(downsample=2).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters())}  device={device}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Focal loss вручную — стабильнее BCE+pos_weight при сильном imbalance
    # (редкие plate-пиксели vs много фоновых). Даунвейтит "лёгкие" негативы.
    def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
        bce_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(gamma) * bce_raw
        return loss.mean()

    best_val = float("inf")
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0; n = 0
        for imgs, targets in tqdm(train_dl, desc=f"ep {epoch:02d}", leave=False):
            imgs = imgs.to(device); targets = targets.to(device)
            logits = model.forward_logits(imgs)
            loss = focal_loss(logits, targets)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * imgs.size(0); n += imgs.size(0)
        sched.step()
        tr_loss = tot / max(n, 1)

        model.eval()
        tot = 0.0; n = 0
        ap_sum = 0.0
        with torch.no_grad():
            for imgs, targets in val_dl:
                imgs = imgs.to(device); targets = targets.to(device)
                logits = model.forward_logits(imgs)
                loss = focal_loss(logits, targets)
                if torch.isfinite(loss):
                    tot += loss.item() * imgs.size(0); n += imgs.size(0)
                # peak correlation: насколько heatmap совпадает с target
                heatmap = torch.sigmoid(logits)
                ap_sum += float((heatmap * targets).sum().item())
        val_loss = tot / max(n, 1)
        print(f"ep {epoch:02d}  train_focal={tr_loss:.5f}  val_focal={val_loss:.5f}  heat·target={ap_sum:.1f}")

        if val_loss < best_val:
            best_val = val_loss
            out = models_dir / "plate_proposer.pt"
            torch.save({"model_state_dict": model.state_dict(),
                        "downsample": model.downsample}, out)
            print(f"  ↳ saved best → {out}")

    print(f"\nDone. best_val_mse={best_val:.4f}")


if __name__ == "__main__":
    main()
