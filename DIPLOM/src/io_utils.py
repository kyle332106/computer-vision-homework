"""Unicode-safe обёртки вокруг cv2.imread/imwrite.

На Windows cv2.imread не умеет открывать пути с не-ASCII символами
(например, "Новая папка") — вместо этого падает с WARN: can't open/read.
Пользуемся np.fromfile + cv2.imdecode.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, flags)
    except Exception:
        return None


def imwrite(path: str | Path, img: np.ndarray) -> bool:
    try:
        ext = Path(path).suffix or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False
