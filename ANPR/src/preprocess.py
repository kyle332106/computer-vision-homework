"""Classical CV preprocessing for license plate recognition.

Каждая функция — прямое переиспользование кода из соответствующего ДЗ.
Вход/выход — RGB изображения np.uint8 (H, W, 3), кроме явных случаев.

Для восстановления качества plate-кропов в сложных условиях (пересвет,
недоэкспонирование, шум ночью) используется цепочка lesson_2/3:
  gray_world → CLAHE → auto-gamma → unsharp → bilateral denoise → rectify → otsu
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# ДЗ2: Color balancing
# ---------------------------------------------------------------------------

def gray_world(img: np.ndarray) -> np.ndarray:
    mean_r = img[:, :, 0].mean()
    mean_g = img[:, :, 1].mean()
    mean_b = img[:, :, 2].mean()

    means = np.array([mean_r, mean_g, mean_b])
    brightest = means.argmax()

    kr = mean_g / mean_r
    kg = 1.0
    kb = mean_g / mean_b

    if brightest == 0:
        kr = 1.0
        kg = mean_r / mean_g
        kb = mean_r / mean_b
    elif brightest == 2:
        kb = 1.0
        kr = mean_b / mean_r
        kg = mean_b / mean_g

    balanced = img.astype(np.float32)
    balanced[:, :, 0] *= kr
    balanced[:, :, 1] *= kg
    balanced[:, :, 2] *= kb
    return np.clip(balanced, 0, 255).astype(np.uint8)


def scale_by_max(img: np.ndarray) -> np.ndarray:
    max_r = img[:, :, 0].max()
    max_g = img[:, :, 1].max()
    max_b = img[:, :, 2].max()

    balanced = img.astype(np.float32)
    balanced[:, :, 0] *= 255.0 / max(max_r, 1)
    balanced[:, :, 1] *= 255.0 / max(max_g, 1)
    balanced[:, :, 2] *= 255.0 / max(max_b, 1)
    return np.clip(balanced, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# lesson_2: Histogram equalization (CLAHE) и auto-gamma
# Восстанавливают контраст в тенях/пересвете, критично для plate-кропов
# снятых против солнца, в сумерках или при жёстком свете фар.
# ---------------------------------------------------------------------------

def clahe(img: np.ndarray, clip: float = 3.0, tile: int = 8) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization на L-канале LAB.

    Работает локально — улучшает контраст в тёмных и светлых областях
    одновременно, в отличие от глобального equalize. Клип ограничивает
    усиление шума в плоских регионах.
    """
    if img.ndim == 2:
        eq = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        return eq.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    eq = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[:, :, 0] = eq.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def auto_gamma(img: np.ndarray, target_mean: float = 0.5) -> np.ndarray:
    """Автоматическая γ-коррекция: сдвигает среднюю яркость к target_mean ∈ (0,1).

    γ < 1 осветляет тени (тёмный кадр), γ > 1 сдерживает пересвет.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    mean = float(gray.mean()) / 255.0 + 1e-6
    gamma = float(np.log(target_mean) / np.log(mean))
    gamma = float(np.clip(gamma, 0.35, 2.5))       # не ломаем сверхтёмные/яркие кропы
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


# ---------------------------------------------------------------------------
# lesson_3: денойз — bilateral сохраняет края символов
# ---------------------------------------------------------------------------

def bilateral_denoise(img: np.ndarray, d: int = 5, sigma_color: float = 50, sigma_space: float = 50) -> np.ndarray:
    """Bilateral filter — убирает шум, но края символов остаются резкими."""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


# ---------------------------------------------------------------------------
# ДЗ3: Unsharp masking
# ---------------------------------------------------------------------------

def unsharp_mask(img: np.ndarray, sigma: float = 1.5, amount: float = 1.0) -> np.ndarray:
    unsharp = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    diff = img.astype(np.float32) - unsharp.astype(np.float32)
    sharpened = img.astype(np.float32) + diff * amount
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# ДЗ6: Harris corner detector (для поиска 4 углов таблички)
# ---------------------------------------------------------------------------

def harris_response(gray: np.ndarray, block_size: int = 2, ksize: int = 3, k: float = 0.04) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    cornerness = cv2.cornerHarris(g, blockSize=block_size, ksize=ksize, k=k)
    cornerness[cornerness < 0] = 0
    return cornerness


def find_plate_corners(crop_rgb: np.ndarray) -> np.ndarray | None:
    """Найти 4 угла номерной таблички внутри crop'а (детекции YOLO).
    Возвращает массив формы (4, 2) в порядке TL, TR, BR, BL, либо None.
    """
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Ищем самый большой контур, аппроксимируем 4-угольником.
    biggest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, 0.02 * peri, True)
    if len(approx) != 4:
        return None

    pts = approx.reshape(4, 2).astype(np.float32)
    return _order_corners(pts)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Отсортировать 4 точки в порядке TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL — мин сумма
    rect[2] = pts[np.argmax(s)]  # BR — макс сумма
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR — мин разница
    rect[3] = pts[np.argmax(diff)]  # BL — макс разница
    return rect


# ---------------------------------------------------------------------------
# ДЗ7: Homography-based rectification
# ---------------------------------------------------------------------------

def rectify_plate(img: np.ndarray, corners: np.ndarray, out_size: tuple[int, int] = (260, 80)) -> np.ndarray:
    """Выровнять номерную табличку по 4 углам → прямоугольник out_size (W, H)."""
    w, h = out_size
    src = corners.astype(np.float32)  # TL, TR, BR, BL
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h))


# ---------------------------------------------------------------------------
# ДЗ8: Otsu binarization
# ---------------------------------------------------------------------------

def otsu_binarize(img: np.ndarray) -> np.ndarray:
    """Бинаризация по Оцу. Вход — RGB или grayscale, выход — uint8 {0, 255}."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binarized = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized


# ---------------------------------------------------------------------------
# Композиция для OCR
# ---------------------------------------------------------------------------

def prep_for_ocr(
    crop_rgb: np.ndarray,
    *,
    use_gray_world: bool = True,
    use_clahe: bool = True,
    use_auto_gamma: bool = True,
    use_unsharp: bool = True,
    use_denoise: bool = False,
    use_rectify: bool = True,
    use_otsu: bool = False,
) -> np.ndarray:
    """Собрать цепочку препроцессинга под OCR. Каждый шаг — из конкретного ДЗ.

    Порядок:
      1. Rectify (ДЗ6 + ДЗ7)       — геометрическое выравнивание таблички
      2. Gray-world (ДЗ2)          — нормализация цветового баланса
      3. CLAHE (lesson 2)          — восстановление контраста в тенях/пересвете
      4. Auto-gamma (lesson 2)     — компенсация недо/переэкспозиции
      5. Bilateral denoise (les.3) — убрать шум, не ломая границы символов
      6. Unsharp (ДЗ3)             — подчеркнуть границы символов
      7. Otsu (ДЗ8)                — бинаризация (опционально)

    Такой порядок лучше соответствует учебным материалам: сначала приводим
    номер к канонической геометрии, затем восстанавливаем качество уже на
    фронтально выровненном кропе, и только после этого распознаём символы.
    """
    img = crop_rgb.copy()
    if use_rectify:
        # Углы ищем на слегка усиленной копии: так устойчивее на слабом контрасте,
        # но сам warp делаем по исходному RGB без артефактов усиления.
        probe = unsharp_mask(img, sigma=1.0, amount=0.6)
        corners = find_plate_corners(probe)
        if corners is None:
            corners = find_plate_corners(gray_world(probe))
        if corners is not None:
            img = rectify_plate(img, corners)
    if use_gray_world:
        img = gray_world(img)
    if use_clahe:
        img = clahe(img)
    if use_auto_gamma:
        img = auto_gamma(img)
    if use_denoise:
        img = bilateral_denoise(img)
    if use_unsharp:
        img = unsharp_mask(img, sigma=1.5, amount=1.0)
    if use_otsu:
        img = otsu_binarize(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def prep_for_color(
    crop_rgb: np.ndarray,
    *,
    use_rectify: bool = True,
) -> np.ndarray:
    """Подготовка кропа для определения цвета фона номерного знака.

    В отличие от OCR-пайплайна, здесь нельзя применять gray-world / CLAHE /
    gamma / unsharp, потому что они искажают восприятие цвета. Нужна только
    геометрия: по возможности выровнять табличку, но сохранить исходные RGB.
    """
    if crop_rgb is None or crop_rgb.size == 0:
        return crop_rgb

    img = crop_rgb.copy()
    if not use_rectify:
        return img

    # Углы ищем на слегка усиленной версии, но warp делаем по исходному RGB,
    # чтобы не ломать цвет фона.
    probe = unsharp_mask(img, sigma=1.0, amount=0.6)
    corners = find_plate_corners(probe)
    if corners is not None:
        img = rectify_plate(img, corners)
    return img
