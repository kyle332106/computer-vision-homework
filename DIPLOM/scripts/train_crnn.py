"""Обучение CRNN+CTC на synthetic_ocr/ + zenodo_ocr/ + ocr_crops/ — CLI-версия 02b_train_crnn_ocr.ipynb.

Использование:
    python scripts/train_crnn.py                    # обучение с нуля
    python scripts/train_crnn.py --resume --lr 1e-4 --real-boost 10 --epochs 15    # fine-tune
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.crnn import CHARSET, CRNN, NUM_CLASSES, ctc_collate, decode_greedy, preprocess_for_crnn
from src.io_utils import imread


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlateOCRDataset(Dataset):
    def __init__(self, sources, train=False, min_len=2, max_len=10):
        self.samples = []
        for root, csv_path in sources:
            if not csv_path.exists():
                print(f"  [skip] {csv_path} не найден")
                continue
            with open(csv_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    text = row["text"].strip().upper()
                    if not (min_len <= len(text) <= max_len):
                        continue
                    if any(c not in CHARSET for c in text):
                        continue
                    self.samples.append((root / row["file"], text))
        self.train = train
        print(f"  собрано: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, text = self.samples[i]
        img = imread(path)
        if img is None:
            return torch.zeros(1, 32, 64), text
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.train and random.random() < 0.3:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-20, 20)
            rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        tensor = preprocess_for_crnn(rgb, target_h=32, max_w=256)
        return tensor.squeeze(0), text


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    total_cer = 0.0
    total_chars = 0
    exact = 0
    n = 0
    for imgs, _, _, texts in dl:
        imgs = imgs.to(DEVICE)
        preds = decode_greedy(model(imgs))
        for p, t in zip(preds, texts):
            total_cer += levenshtein(p, t)
            total_chars += max(1, len(t))
            exact += int(p == t)
            n += 1
    return total_cer / max(total_chars, 1), exact / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="грузить существующие веса models/crnn_ocr.pt")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--real-boost", type=int, default=1,
                    help="сколько раз продублировать real train (oversample против synthetic-overfit)")
    ap.add_argument("--scene-boost", type=int, default=4,
                    help="сколько раз продублировать scene-specific RTSP train")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    synth = root / "data" / "synthetic_ocr"
    zenodo = root / "data" / "zenodo_ocr"
    real = root / "data" / "ocr_crops"
    scene = root / "data" / "scene_ocr"

    print("train (synthetic):")
    synth_train = PlateOCRDataset(
        [(synth / "images/train", synth / "labels_train.csv")], train=True,
    )
    print("train (zenodo UA plate font):")
    zenodo_train = PlateOCRDataset(
        [(zenodo / "images/train", zenodo / "labels_train.csv")], train=True,
    )
    print("train (real AUTO.RIA):")
    real_train = PlateOCRDataset(
        [(real / "train", real / "labels_train.csv")], train=True,
    )
    print("train (scene RTSP):")
    scene_train = PlateOCRDataset(
        [(scene / "images/train", scene / "labels_train.csv")], train=True,
    )
    print("val:")
    val_ds = PlateOCRDataset(
        [(synth / "images/val", synth / "labels_val.csv"),
         (zenodo / "images/val", zenodo / "labels_val.csv"),
         (real / "val", real / "labels_val.csv"),
         (scene / "images/val", scene / "labels_val.csv")],
        train=False,
    )

    train_parts: list[Dataset] = []
    if len(synth_train):
        train_parts.append(synth_train)
    if len(zenodo_train):
        train_parts.append(zenodo_train)
    if len(real_train):
        # oversample real для компенсации дисбаланса
        for _ in range(max(1, args.real_boost)):
            train_parts.append(real_train)
    if len(scene_train):
        for _ in range(max(1, args.scene_boost)):
            train_parts.append(scene_train)
    if not train_parts:
        sys.exit("train-данные отсутствуют — запустите generate_synthetic_ocr.py/prepare_*")
    train_ds = ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]
    print(f"\nitems: train={len(train_ds)} (real_boost×{args.real_boost}, scene_boost×{args.scene_boost})  val={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, collate_fn=ctc_collate)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=ctc_collate)

    model = CRNN(num_classes=NUM_CLASSES).to(DEVICE)
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    if args.resume:
        ckpt_path = models_dir / "crnn_ocr.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state)
            print(f"resumed from {ckpt_path}")
        else:
            print(f"WARN: --resume указан, но {ckpt_path} не найден")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    print(f"\nCRNN: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params | device={DEVICE} | lr={args.lr} | epochs={args.epochs}")
    best_cer = 1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        pbar = tqdm(train_dl, desc=f"ep {epoch:02d}", leave=False)
        for imgs, targets, target_lens, _ in pbar:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            target_lens = target_lens.to(DEVICE)
            log_probs = model(imgs)
            T = log_probs.size(0)
            B = log_probs.size(1)
            input_lens = torch.full((B,), T, dtype=torch.long, device=DEVICE)
            loss = ctc(log_probs, targets, input_lens, target_lens)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * B
            n += B
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()

        val_cer, val_exact = evaluate(model, val_dl)
        tr_loss = total / max(n, 1)
        print(f"ep {epoch:02d}  train_loss={tr_loss:.3f}  val_CER={val_cer:.3f}  val_exact={val_exact:.3f}")

        if val_cer < best_cer:
            best_cer = val_cer
            out = models_dir / "crnn_ocr.pt"
            torch.save({"model_state_dict": model.state_dict(), "charset": CHARSET}, out)
            print(f"  ↳ saved best → {out}")

    print(f"\nDone. best_val_CER = {best_cer:.3f}")


if __name__ == "__main__":
    main()
