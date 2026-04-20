"""ALPR pipeline: YOLO26 детекция → препроцессинг (ДЗ2,3,6,7,8) → OCR → трекинг (ДЗ10).

OCR-бэкенды:
  • "crnn"    — собственная CRNN+CTC (src/crnn.py). Читает любой текст
                любой длины — поддерживает именные/нестандартные номера.
  • "easyocr" — baseline. Используется, если CRNN-веса не обучены.

postprocess может работать в двух режимах:
  • strict_format=True  — стрипает всё кроме [A-Z0-9] (для строгого UA-формата)
  • strict_format=False — возвращает текст как есть (для именных/иностранных)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from . import plate_finder, preprocess
from .plate_color import classify_plate_color_info
from .tracker import TrackedPlate, has_tracker, make_tracker


UA_ALLOWED = "ABCEHIKMOPTX0123456789АВСЕНІКМОРТХ"
UA_CYRILLIC_TO_LATIN = str.maketrans("АВСЕНІКМОРТХ", "ABCEHIKMOPTX")
PLATE_CYR_TO_LATIN = str.maketrans({
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "К": "K",
    "М": "M", "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y",
    "Ё": "E", "Ї": "I",
    "Ь": "", "Ъ": "",
    "Г": "G", "Ґ": "G", "Д": "D", "Ж": "ZH", "З": "Z", "И": "I", "Й": "I",
    "Л": "L", "П": "P", "Ф": "F", "Ц": "TS", "Ч": "CH", "Ш": "SH", "Щ": "SH",
    "Ы": "Y", "Э": "E", "Ю": "YU", "Я": "YA",
})
UA_STD_LETTERS = set("ABCEHIKMOPTX")
LETTER_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "B": "8",
    "T": "7",
}
DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "4": "A",
    "7": "T",
    "8": "B",
}

OcrBackend = Literal["crnn", "easyocr"]


@dataclass
class PipelineConfig:
    detector_weights: str = "models/yolo26_plate.pt"
    fallback_weights: str = "yolo26n.pt"
    conf: float = 0.2
    iou: float = 0.45
    imgsz: int = 1280            # 1280+ для high-res камер; для стандартных фото 640 хватит
    device: str | int = 0
    use_sliced_yolo: bool = True
    sliced_min_frame_side: int = 1800
    sliced_iou: float = 0.30
    ignore_top_band_frac: float = 0.08

    # Classical plate-finder (ДЗ2/3/4/6/7/8) в дополнение к YOLO — помогает на сценах,
    # где YOLO плохо обобщает (далёкие/повёрнутые plate на общем плане).
    # Реализация: "cpu" — numpy/OpenCV, "nn" — nn.Module на GPU (Sobel как обучаемая
    # свёртка, как в lesson_14/LearnSobelCNN). NN-вариант ~18× быстрее.
    use_classical_finder: bool = True
    classical_backend: str = "cpu"        # "cpu" (стабильнее) | "nn" (18× быстрее, но нужен train)
    classical_nn_thr: float = 0.65        # порог sigmoid(heatmap) для NN-варианта
    classical_min_ocr_conf: float = 0.8
    classical_min_len: int = 5
    final_min_aspect: float = 1.6
    final_max_aspect: float = 8.0
    dedupe_iou: float = 0.35

    # OCR
    ocr_backend: OcrBackend = "crnn"
    crnn_weights: str = "models/crnn_ocr.pt"
    ocr_languages: tuple[str, ...] = ("en",)
    strict_format: bool = False
    ua_format_aware: bool = True
    min_letters_in_text: int = 2
    min_digits_in_text: int = 2
    track_iou_match: float = 0.30

    # Препроцессинг-флаги (управление из Streamlit)
    use_gray_world: bool = True
    use_clahe: bool = True            # lesson_2: восстановление контраста
    use_auto_gamma: bool = True       # lesson_2: адаптация под освещённость
    use_unsharp: bool = True
    use_denoise: bool = False         # lesson_3: bilateral (против ISO-шума)
    use_rectify: bool = True
    use_otsu: bool = False

    # Трекинг
    use_tracker: bool = True
    tracker_kind: str = "csrt"
    detect_every_n: int = 5


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    conf: float
    crop: np.ndarray
    rectified: np.ndarray | None = None
    text: str = ""
    raw_text: str = ""
    normalized_text: str = ""
    source: str = "yolo"        # "yolo" | "cv" | "both"
    ocr_conf: float = 0.0
    plate_color: str = "неизвестно"
    plate_color_conf: float = 0.0
    plate_color_context: str = ""
    plate_format: str = "unknown"


class ALPRPipeline:
    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self._detector = None
        self._reader = None                 # EasyOCR
        self._crnn = None                   # CRNN nn.Module
        self._crnn_device = None
        self._proposer_nn = None            # GPU plate-proposer (lesson_14 style)
        self._tracked: list[TrackedPlate] = []
        self._frame_idx = 0

    @property
    def proposer_nn(self):
        if self._proposer_nn is None:
            import torch
            from .plate_proposer_nn import PlateProposerCNN
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = PlateProposerCNN().to(device).eval()
            # Опционально — подгрузить обученные веса, если файл есть
            from pathlib import Path as _P
            ckpt_path = _P("models/plate_proposer.pt")
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device)
                state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                try:
                    model.load_state_dict(state)
                except Exception:
                    pass   # несовместимые веса — используем Sobel-init
            self._proposer_nn = model
        return self._proposer_nn

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------
    @property
    def detector(self):
        if self._detector is None:
            from ultralytics import YOLO
            weights = self.cfg.detector_weights
            if not Path(weights).exists():
                weights = self.cfg.fallback_weights
            self._detector = YOLO(weights)
        return self._detector

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(list(self.cfg.ocr_languages), gpu=True)
        return self._reader

    @property
    def crnn(self):
        if self._crnn is None:
            import torch
            from .crnn import CRNN
            if not Path(self.cfg.crnn_weights).exists():
                raise FileNotFoundError(
                    f"CRNN веса не найдены: {self.cfg.crnn_weights}. "
                    "Обучите в notebooks/02b_train_crnn_ocr.ipynb или переключите "
                    "pipeline.cfg.ocr_backend='easyocr'."
                )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CRNN().to(device)
            state = torch.load(self.cfg.crnn_weights, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            model.eval()
            self._crnn = model
            self._crnn_device = device
        return self._crnn

    # ------------------------------------------------------------------
    # Stage 1: детекция
    # ------------------------------------------------------------------
    def detect(self, frame_rgb: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        # YOLO accepts only multiples of 32 for imgsz
        imgsz = max(320, (self.cfg.imgsz // 32) * 32)
        h, w = frame_rgb.shape[:2]

        def _predict_tile(tile_rgb: np.ndarray, local_imgsz: int, local_conf: float):
            res = self.detector.predict(
                tile_rgb,
                conf=local_conf,
                iou=self.cfg.iou,
                imgsz=max(320, (local_imgsz // 32) * 32),
                device=self.cfg.device,
                verbose=False,
            )[0]
            out = []
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                out.append((x1, y1, x2, y2, conf))
            return out

        if not self.cfg.use_sliced_yolo or max(h, w) < self.cfg.sliced_min_frame_side:
            return _predict_tile(frame_rgb, imgsz, self.cfg.conf)

        # Full-frame + tiled high-res inference. Это поднимает recall на парковочных
        # сценах, где мелкие номера теряются при обычном resize всего кадра.
        raw_boxes: list[tuple[int, int, int, int, float]] = []
        raw_boxes.extend(_predict_tile(frame_rgb, imgsz, self.cfg.conf))
        tile_cfgs = [
            (2, 2, 0.25, min(imgsz, 1280), min(self.cfg.conf, 0.08)),
            (3, 2, 0.25, min(max(imgsz, 1280), 1280), min(self.cfg.conf, 0.06)),
            (5, 3, 0.25, 960, min(self.cfg.conf, 0.03)),
            (6, 3, 0.20, 960, min(self.cfg.conf, 0.02)),
        ]
        for tiles_x, tiles_y, overlap, tile_imgsz, tile_conf in tile_cfgs:
            tile_w = int(w * (1 + overlap * (tiles_x - 1)) / tiles_x)
            tile_h = int(h * (1 + overlap * (tiles_y - 1)) / tiles_y)
            stride_x = max(1, int(tile_w * (1 - overlap)))
            stride_y = max(1, int(tile_h * (1 - overlap)))
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    x0 = min(tx * stride_x, max(0, w - tile_w))
                    y0 = min(ty * stride_y, max(0, h - tile_h))
                    tile = frame_rgb[y0:y0 + tile_h, x0:x0 + tile_w]
                    for x1, y1, x2, y2, conf in _predict_tile(tile, tile_imgsz, tile_conf):
                        raw_boxes.append((x0 + x1, y0 + y1, x0 + x2, y0 + y2, conf))
        return self._dedupe_yolo_boxes(raw_boxes, iou_thr=self.cfg.sliced_iou)

    def _dedupe_yolo_boxes(
        self,
        boxes: list[tuple[int, int, int, int, float]],
        iou_thr: float,
    ) -> list[tuple[int, int, int, int, float]]:
        kept: list[tuple[int, int, int, int, float]] = []
        for box in sorted(boxes, key=lambda b: -b[4]):
            if any(self._iou(box[:4], prev[:4]) >= iou_thr for prev in kept):
                continue
            kept.append(box)
        return kept

    # ------------------------------------------------------------------
    # Stage 2: препроцессинг (ДЗ2,3,6,7,8)
    # ------------------------------------------------------------------
    def preprocess_crop(self, crop_rgb: np.ndarray) -> np.ndarray:
        return preprocess.prep_for_ocr(
            crop_rgb,
            use_gray_world=self.cfg.use_gray_world,
            use_clahe=self.cfg.use_clahe,
            use_auto_gamma=self.cfg.use_auto_gamma,
            use_unsharp=self.cfg.use_unsharp,
            use_denoise=self.cfg.use_denoise,
            use_rectify=self.cfg.use_rectify,
            use_otsu=self.cfg.use_otsu,
        )

    def preprocess_color_crop(self, crop_rgb: np.ndarray) -> np.ndarray:
        return preprocess.prep_for_color(
            crop_rgb,
            use_rectify=self.cfg.use_rectify,
        )

    # ------------------------------------------------------------------
    # Stage 3: OCR — диспетчер по backend'у
    # ------------------------------------------------------------------
    def recognize(self, img_rgb: np.ndarray) -> tuple[str, str]:
        """Возвращает (raw_text, postprocessed_text)."""
        if self.cfg.ocr_backend == "crnn":
            raw = self._recognize_crnn(img_rgb)
        else:
            raw = self._recognize_easyocr(img_rgb)
        return raw, self.postprocess(raw)

    def _recognize_easyocr(self, img_rgb: np.ndarray) -> str:
        # allowlist только в strict-режиме — иначе именные не прочитаются
        allowlist = UA_ALLOWED if self.cfg.strict_format else None
        results = self.reader.readtext(img_rgb, allowlist=allowlist, detail=1)
        if not results:
            return ""
        results.sort(key=lambda r: r[0][0][0])
        return "".join(r[1] for r in results)

    def _recognize_crnn(self, img_rgb: np.ndarray) -> str:
        text, _ = self._recognize_crnn_with_conf(img_rgb)
        return text

    def _recognize_crnn_with_conf(self, img_rgb: np.ndarray) -> tuple[str, float]:
        """CRNN + CTC avg-max-prob как confidence (для фильтрации CV-кандидатов)."""
        import torch
        from .crnn import preprocess_for_crnn, decode_greedy

        tensor = preprocess_for_crnn(img_rgb)
        if tensor is None:
            return "", 0.0
        model = self.crnn
        tensor = tensor.to(self._crnn_device)
        with torch.no_grad():
            log_probs = model(tensor)
            avg_conf = float(log_probs.exp().max(dim=2).values.mean().item())
        text = decode_greedy(log_probs)[0]
        return text, avg_conf

    def postprocess(self, text: str) -> str:
        text = text.upper()
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        exact = self._cleanup_plate_text(text)
        normalized = self._normalize_raw_plate_text(text)
        if self.cfg.ua_format_aware and self._should_force_ua_format(exact):
            ua_text, ua_score = self._best_ua_plate_candidate(normalized)
            if ua_text and ua_score >= 5.0:
                return ua_text
        if not self.cfg.strict_format:
            return exact
        return re.sub(r"[^A-Z0-9]", "", normalized)

    def _cleanup_plate_text(self, text: str) -> str:
        text = text.upper()
        text = text.translate(PLATE_CYR_TO_LATIN)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^A-Z0-9 -]", "", text)
        text = re.sub(r"\s*-\s*", "-", text)
        return text

    def _normalize_raw_plate_text(self, text: str) -> str:
        text = text.upper().translate(PLATE_CYR_TO_LATIN).translate(UA_CYRILLIC_TO_LATIN)
        return re.sub(r"[^A-Z0-9]", "", text)

    def _looks_like_plate_text(self, text: str) -> bool:
        text = self._cleanup_plate_text(text)
        compact = re.sub(r"[\s-]+", "", text)
        if len(compact) < self.cfg.classical_min_len:
            return False
        letters = sum(ch.isalpha() for ch in compact)
        digits = sum(ch.isdigit() for ch in compact)
        # standard / foreign mixed
        if letters >= self.cfg.min_letters_in_text and digits >= self.cfg.min_digits_in_text:
            return True
        # именные / vanity
        if letters >= 4 and digits == 0:
            return True
        # цифровые/короткие спецформаты
        if digits >= 4 and letters == 0:
            return True
        return len(compact) >= 5 and (letters + digits) == len(compact)

    def _infer_plate_format(self, exact_text: str, color_name: str) -> str:
        compact = re.sub(r"[\s-]+", "", self._cleanup_plate_text(exact_text))
        letters = sum(ch.isalpha() for ch in compact)
        digits = sum(ch.isdigit() for ch in compact)
        if color_name == "красный":
            return "diplomatic"
        if color_name == "синий":
            return "special"
        if color_name == "зелёный":
            return "military"
        if color_name == "жёлтый":
            return "taxi"
        if "-" in exact_text or " " in exact_text or (letters >= 4 and digits == 0):
            return "named_or_custom"
        if letters >= 2 and digits >= 2:
            return "mixed_alnum"
        if digits >= 4 and letters == 0:
            return "numeric"
        return "unknown"

    def _should_force_ua_format(self, exact_text: str) -> bool:
        compact = re.sub(r"[\s-]+", "", self._cleanup_plate_text(exact_text))
        if "-" in exact_text or " " in exact_text:
            return False
        if len(compact) != 8:
            return False
        letters = sum(ch.isalpha() for ch in compact)
        digits = sum(ch.isdigit() for ch in compact)
        return letters >= 4 and digits >= 4

    def _coerce_char(self, ch: str, expect_letter: bool) -> tuple[str | None, float]:
        if expect_letter:
            if ch in UA_STD_LETTERS:
                return ch, 0.0
            mapped = DIGIT_TO_LETTER.get(ch)
            if mapped in UA_STD_LETTERS:
                return mapped, 0.8
            return None, 999.0
        if ch.isdigit():
            return ch, 0.0
        mapped = LETTER_TO_DIGIT.get(ch)
        if mapped is not None:
            return mapped, 0.8
        return None, 999.0

    def _score_std_pattern(self, candidate: str) -> tuple[str, float]:
        if len(candidate) != 8:
            return "", -1e9
        out = []
        penalty = 0.0
        for i, ch in enumerate(candidate):
            coerced, cost = self._coerce_char(ch, expect_letter=(i in (0, 1, 6, 7)))
            if coerced is None:
                return "", -1e9
            out.append(coerced)
            penalty += cost
        fixed = "".join(out)
        if fixed[0] not in UA_STD_LETTERS or fixed[1] not in UA_STD_LETTERS:
            return "", -1e9
        if fixed[6] not in UA_STD_LETTERS or fixed[7] not in UA_STD_LETTERS:
            return "", -1e9
        score = 10.0 - penalty
        return fixed, score

    def _best_ua_plate_candidate(self, text: str) -> tuple[str, float]:
        cleaned = self._normalize_raw_plate_text(text)
        if not cleaned:
            return "", -1e9
        variants = {cleaned}
        if len(cleaned) > 8:
            for i in range(len(cleaned)):
                variants.add(cleaned[:i] + cleaned[i + 1:])
        best_text, best_score = "", -1e9
        for cand in variants:
            fixed, score = self._score_std_pattern(cand)
            if fixed and score > best_score:
                delete_penalty = 0.4 * max(0, len(cleaned) - len(cand))
                best_text, best_score = fixed, score - delete_penalty
        return best_text, best_score

    # ------------------------------------------------------------------
    # End-to-end: один кадр. YOLO + классический plate-finder (ДЗ2/3/4/6/7/8)
    # объединяются, каждый candidate прогоняется через OCR.
    # ------------------------------------------------------------------
    def process_frame(self, frame_rgb: np.ndarray) -> list[Detection]:
        yolo_boxes = self.detect(frame_rgb)
        cv_boxes: list[tuple[int, int, int, int, float]] = []
        if self.cfg.use_classical_finder:
            if self.cfg.classical_backend == "nn":
                from .plate_proposer_nn import propose
                cv_boxes = propose(self.proposer_nn, frame_rgb,
                                   device=next(self.proposer_nn.parameters()).device,
                                   thr=self.cfg.classical_nn_thr)
            else:
                cv_boxes = plate_finder.find_candidates(frame_rgb)
        merged = plate_finder.merge_with_yolo(yolo_boxes, cv_boxes)

        out: list[Detection] = []
        for x1, y1, x2, y2, conf, src in merged:
            if y2 <= int(frame_rgb.shape[0] * self.cfg.ignore_top_band_frac):
                continue
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            aspect = w / h
            if not (self.cfg.final_min_aspect <= aspect <= self.cfg.final_max_aspect):
                continue
            crop = frame_rgb[max(0, y1):y2, max(0, x1):x2].copy()
            if crop.size == 0:
                continue
            color_crop = self.preprocess_color_crop(crop)
            rectified = self.preprocess_crop(crop)
            color_info, color_conf = classify_plate_color_info(color_crop)
            if self.cfg.ocr_backend == "crnn":
                raw, ocr_conf = self._recognize_crnn_with_conf(rectified)
            else:
                raw = self._recognize_easyocr(rectified)
                ocr_conf = 1.0 if raw else 0.0
            text = self.postprocess(raw)
            normalized_text = self._normalize_raw_plate_text(text)
            plate_format = self._infer_plate_format(text, color_info.name)

            # Для CV-кандидатов применяем доп. фильтр: минимальная длина и OCR-confidence.
            # YOLO-детекции доверяем без этого фильтра.
            if src == "cv":
                if not self._looks_like_plate_text(text):
                    continue
                if ocr_conf < self.cfg.classical_min_ocr_conf:
                    continue
                if plate_format not in {"mixed_alnum", "named_or_custom", "diplomatic", "special", "military", "taxi", "numeric"}:
                    continue
            elif not self._looks_like_plate_text(text):
                continue

            out.append(Detection(
                bbox=(x1, y1, x2, y2), conf=conf,
                crop=crop, rectified=rectified,
                raw_text=raw, text=text, normalized_text=normalized_text,
                source=src, ocr_conf=ocr_conf,
                plate_color=color_info.name,
                plate_color_conf=color_conf,
                plate_color_context=color_info.context,
                plate_format=plate_format,
            ))
        return self._dedupe_detections(out)

    def _iou(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        xa = max(a[0], b[0]); ya = max(a[1], b[1])
        xb = min(a[2], b[2]); yb = min(a[3], b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        ua = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        ub = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        union = ua + ub - inter
        return inter / union if union > 0 else 0.0

    def _det_rank(self, d: Detection) -> tuple[float, float, float]:
        source_bonus = {"both": 2.0, "yolo": 1.0, "cv": 0.0}.get(d.source, 0.0)
        return (source_bonus, d.ocr_conf, d.conf)

    def _bbox_xyxy_to_xywh(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        return (x1, y1, x2 - x1, y2 - y1)

    def _merge_votes(self, prev: TrackedPlate | None, det: Detection) -> dict[str, float]:
        votes = dict(prev.text_votes) if prev is not None else {}
        weight = 1.0 + det.ocr_conf + det.conf
        weight += {"both": 0.5, "yolo": 0.2, "cv": 0.0}.get(det.source, 0.0)
        if det.text:
            votes[det.text] = votes.get(det.text, 0.0) + weight
        return votes

    def _best_vote_text(self, votes: dict[str, float], fallback: str) -> str:
        if not votes:
            return fallback
        return max(votes.items(), key=lambda kv: kv[1])[0]

    def _merge_color_votes(self, prev: TrackedPlate | None, det: Detection) -> dict[str, float]:
        votes = dict(prev.color_votes) if prev is not None else {}
        if det.plate_color:
            weight = 0.5 + det.plate_color_conf
            if det.plate_color != "неизвестно":
                weight += 0.5
            votes[det.plate_color] = votes.get(det.plate_color, 0.0) + weight
        return votes

    def _best_vote_color(self, votes: dict[str, float], fallback: str) -> str:
        if not votes:
            return fallback
        best_color, _ = max(votes.items(), key=lambda kv: kv[1])
        if best_color == "неизвестно" and fallback and fallback != "неизвестно":
            return fallback
        return best_color

    def _match_prev_track(self, det: Detection, prev_tracks: list[TrackedPlate]) -> TrackedPlate | None:
        prev_match = None
        prev_iou = 0.0
        for tp in prev_tracks:
            xyxy_prev = (tp.bbox[0], tp.bbox[1], tp.bbox[0] + tp.bbox[2], tp.bbox[1] + tp.bbox[3])
            cur_iou = self._iou(det.bbox, xyxy_prev)
            if cur_iou > prev_iou and cur_iou >= self.cfg.track_iou_match:
                prev_iou = cur_iou
                prev_match = tp
        return prev_match

    def _apply_temporal_votes(
        self,
        detections: list[Detection],
        prev_tracks: list[TrackedPlate],
        frame_rgb: np.ndarray | None = None,
    ) -> list[TrackedPlate]:
        new_tracks: list[TrackedPlate] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            bbox_xywh = (x1, y1, x2 - x1, y2 - y1)
            prev_match = self._match_prev_track(det, prev_tracks)
            text_votes = self._merge_votes(prev_match, det)
            color_votes = self._merge_color_votes(prev_match, det)
            det.text = self._best_vote_text(text_votes, det.text)
            det.plate_color = self._best_vote_color(color_votes, det.plate_color)

            tracker = None
            if frame_rgb is not None and self.cfg.use_tracker:
                tracker = make_tracker(self.cfg.tracker_kind)
                tracker.init(frame_rgb, bbox_xywh)

            new_tracks.append(TrackedPlate(
                bbox=bbox_xywh,
                text=det.text,
                plate_color=det.plate_color,
                tracker=tracker,
                text_votes=text_votes,
                color_votes=color_votes,
            ))
        return new_tracks

    def _dedupe_detections(self, detections: list[Detection]) -> list[Detection]:
        kept: list[Detection] = []
        for det in sorted(detections, key=self._det_rank, reverse=True):
            if any(self._iou(det.bbox, prev.bbox) >= self.cfg.dedupe_iou for prev in kept):
                continue
            kept.append(det)
        return kept

    # ------------------------------------------------------------------
    # Real-time: детекция раз в N кадров, между — трекер (ДЗ10)
    # ------------------------------------------------------------------
    def process_stream_frame(self, frame_rgb: np.ndarray) -> list[Detection]:
        if self.cfg.use_tracker and not has_tracker(self.cfg.tracker_kind):
            detections = self.process_frame(frame_rgb)
            self._tracked = self._apply_temporal_votes(detections, list(self._tracked))
            return detections
        self._frame_idx += 1
        should_detect = (
            not self.cfg.use_tracker
            or not self._tracked
            or self._frame_idx % self.cfg.detect_every_n == 0
        )

        if should_detect:
            detections = self.process_frame(frame_rgb)
            self._tracked = self._apply_temporal_votes(detections, list(self._tracked), frame_rgb=frame_rgb)
            return detections

        results: list[Detection] = []
        still_alive: list[TrackedPlate] = []
        for tp in self._tracked:
            ok, bbox = tp.tracker.update(frame_rgb)
            if not ok:
                continue
            x, y, w, h = [int(v) for v in bbox]
            crop = frame_rgb[max(0, y):y + h, max(0, x):x + w].copy()
            if crop.size == 0:
                continue
            tp.bbox = (x, y, w, h)
            tp.age += 1
            still_alive.append(tp)
            results.append(Detection(
                bbox=(x, y, x + w, y + h), conf=0.0,
                crop=crop, rectified=None,
                raw_text=tp.text, text=tp.text,
                normalized_text=self._normalize_raw_plate_text(tp.text),
                plate_color=tp.plate_color,
            ))
        self._tracked = still_alive
        return results

    def reset(self) -> None:
        self._tracked = []
        self._frame_idx = 0


# ---------------------------------------------------------------------------
# Утилита рисования
# ---------------------------------------------------------------------------

SOURCE_COLORS = {
    "yolo": (0, 255, 0),       # зелёный — YOLO
    "cv":   (0, 150, 255),     # оранжевый — classical (фильтры курса)
    "both": (0, 255, 255),     # жёлтый — оба согласны
}


def draw_detections(frame_rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame_rgb.copy()
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        color = SOURCE_COLORS.get(d.source, (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = d.text or "?"
        if getattr(d, "plate_color", "") and d.plate_color != "неизвестно":
            label = f"{label} · {d.plate_color}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return out
