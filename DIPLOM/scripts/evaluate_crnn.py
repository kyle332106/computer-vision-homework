"""Оценка CRNN на synthetic val и real AUTO.RIA val с breakdown по kind/region_name.

Использование:
    python scripts/evaluate_crnn.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.crnn import CHARSET, CRNN, NUM_CLASSES, decode_greedy, preprocess_for_crnn
from src.io_utils import imread


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


def load_samples(root: Path, csv_path: Path):
    if not csv_path.exists():
        return []
    out = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row["text"].strip().upper()
            if not (2 <= len(text) <= 10):
                continue
            if any(c not in CHARSET for c in text):
                continue
            out.append({
                "path": root / row["file"],
                "text": text,
                "kind": row.get("kind", "unknown") or "unknown",
            })
    return out


@torch.no_grad()
def run(model, samples, device, title):
    model.eval()
    groups = defaultdict(lambda: {"n": 0, "cer_sum": 0.0, "chars": 0, "exact": 0})
    overall = {"n": 0, "cer_sum": 0.0, "chars": 0, "exact": 0}

    for s in tqdm(samples, desc=title):
        img = imread(s["path"])
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = preprocess_for_crnn(rgb)
        if tensor is None:
            continue
        tensor = tensor.to(device)
        log_probs = model(tensor)
        pred = decode_greedy(log_probs)[0]

        gt = s["text"]
        dist = levenshtein(pred, gt)
        exact = int(pred == gt)

        for bucket in (groups[s["kind"]], overall):
            bucket["n"] += 1
            bucket["cer_sum"] += dist
            bucket["chars"] += max(1, len(gt))
            bucket["exact"] += exact

    def _metrics(b):
        return {
            "n": b["n"],
            "mean_CER": round(b["cer_sum"] / max(b["chars"], 1), 4),
            "exact_pct": round(100 * b["exact"] / max(b["n"], 1), 2),
        }

    return {
        "overall": _metrics(overall),
        "by_kind": {k: _metrics(v) for k, v in groups.items()},
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = root / "models" / "crnn_ocr.pt"
    if not ckpt_path.exists():
        sys.exit(f"не найдены веса: {ckpt_path}")

    model = CRNN(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"CRNN loaded: {ckpt_path}, device={device}")

    synth = root / "data" / "synthetic_ocr"
    real = root / "data" / "ocr_crops"
    results = {}
    synth_samples = load_samples(synth / "images/val", synth / "labels_val.csv")
    real_samples = load_samples(real / "val", real / "labels_val.csv")
    if synth_samples:
        results["synthetic_val"] = run(model, synth_samples, device, "synthetic")
    if real_samples:
        results["real_val"] = run(model, real_samples, device, "real")

    out_dir = root / "runs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "crnn_summary.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Итого ===")
    for split, r in results.items():
        print(f"\n[{split}]  overall: n={r['overall']['n']}  CER={r['overall']['mean_CER']}  exact={r['overall']['exact_pct']}%")
        for k, m in sorted(r["by_kind"].items(), key=lambda x: -x[1]["n"]):
            print(f"  {k:<22}  n={m['n']:>5}  CER={m['mean_CER']:.3f}  exact={m['exact_pct']:>5.1f}%")
    print(f"\nJSON → {out}")


if __name__ == "__main__":
    main()
