"""Streamlit-демо ALPR-системы для защиты диплома.

Запуск:
    streamlit run app.py

Три режима:
  1. Загрузка фото
  2. Живое видео с вебкамеры (cv2.VideoCapture — надёжнее webrtc на Windows)
  3. Загрузка видеофайла

Сайдбар управляет:
  • confidence threshold детектора
  • переключатели шагов препроцессинга из ДЗ (gray-world, unsharp, rectify, otsu)
  • выбор трекера (kcf / csrt) и частоты детекции
"""

from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import streamlit as st

from src.pipeline import ALPRPipeline, PipelineConfig, draw_detections


st.set_page_config(page_title="ALPR Diploma — License Plate Recognition", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙ Параметры")

st.sidebar.subheader("Детектор YOLO26")
conf = st.sidebar.slider("YOLO confidence", 0.05, 0.9, 0.2, 0.05)
imgsz = st.sidebar.select_slider(
    "YOLO imgsz (больше → мелкие plate детектятся, медленнее)",
    options=[640, 960, 1280, 1536, 1920, 2048], value=1280,
)
min_plate_w = st.sidebar.slider("Min plate width, px", 20, 160, 40, 2)
min_plate_h = st.sidebar.slider("Min plate height, px", 8, 60, 12, 1)
min_plate_area = st.sidebar.slider("Min plate area, px²", 200, 5000, 600, 50)

st.sidebar.subheader("Classical plate-finder (ДЗ2/3/4/6/7/8)")
use_classical = st.sidebar.checkbox(
    "Включить классический поиск", value=True,
    help="Sobel+Otsu+morphology + Harris+Homography fills gaps YOLO",
)
classical_backend = st.sidebar.radio(
    "Backend",
    ["cpu", "nn"],
    index=0,
    help="cpu — OpenCV/numpy (стабильно); nn — nn.Module на GPU по мотивам "
         "lesson_14/LearnSobelCNN (Sobel как обучаемая свёртка). NN без обучения "
         "даёт много false-positives — нужен train_plate_proposer.py.",
)
cls_min_ocr = st.sidebar.slider("Classical: min OCR-confidence", 0.5, 0.99, 0.8, 0.01)

st.sidebar.subheader("OCR")
ocr_backend = st.sidebar.radio(
    "Бэкенд",
    ["crnn", "easyocr"],
    index=0,
    help="CRNN+CTC — наша сеть, читает произвольный текст. EasyOCR — baseline.",
)
strict_format = st.sidebar.checkbox(
    "Strict UA-формат (стрипать всё кроме A-Z0-9)",
    value=False,
    help="Выключено — читает именные/нестандартные номера как есть.",
)

st.sidebar.subheader("Препроцессинг plate-кропа (ДЗ2-8)")
use_gray_world = st.sidebar.checkbox("Gray-world color balance (ДЗ2)", value=True)
use_clahe = st.sidebar.checkbox("CLAHE (lesson 2) — тени/пересвет", value=True)
use_auto_gamma = st.sidebar.checkbox("Auto-gamma (lesson 2)", value=True)
use_unsharp = st.sidebar.checkbox("Unsharp mask (ДЗ3)", value=True)
use_denoise = st.sidebar.checkbox("Bilateral denoise (lesson 3)", value=False)
use_rectify = st.sidebar.checkbox("Rectify (Harris+Homography, ДЗ6-7)", value=True)
use_otsu = st.sidebar.checkbox("Otsu binarize (ДЗ8)", value=False)

st.sidebar.subheader("Трекинг (ДЗ10)")
use_tracker = st.sidebar.checkbox("Использовать трекер между детекциями", value=True)
tracker_kind = st.sidebar.radio(
    "Тип трекера",
    ["bytetrack", "sort", "csrt", "kcf"],
    index=0,
    help="ByteTrack/SORT — tracking-by-detection с Kalman+IoU matching; CSRT/KCF — OpenCV-трекеры между детекциями.",
)
detect_every_n = st.sidebar.slider(
    "YOLO раз в N кадров",
    1, 15, 5,
    help="Используется для CSRT/KCF. ByteTrack/SORT обновляются по детекциям каждого кадра.",
)
temporal_stabilization = st.sidebar.checkbox(
    "Стабилизировать OCR по времени",
    value=True,
    help="Копит OCR-гипотезы одного трека и выбирает устойчивую строку через weighted/char-wise voting.",
)
temporal_max_missing = st.sidebar.slider(
    "Память трека без детекции, detect-шагов",
    0, 8, 3,
)


@st.cache_resource
def load_pipeline(
    conf: float, imgsz: int, backend: str, strict: bool,
    min_w: int, min_h: int, min_area: int,
    use_classical: bool, cls_backend: str, cls_min_ocr: float,
    use_gw: bool, use_clahe_: bool, use_gamma: bool, use_sharp: bool,
    use_denoise_: bool, use_rect: bool, use_otsu: bool,
    use_trk: bool, tk: str, n: int, temporal: bool, max_missing: int,
) -> ALPRPipeline:
    cfg = PipelineConfig(
        conf=conf, imgsz=imgsz,
        final_min_w=min_w, final_min_h=min_h, final_min_area=min_area,
        ocr_backend=backend, strict_format=strict,
        use_classical_finder=use_classical, classical_backend=cls_backend,
        classical_min_ocr_conf=cls_min_ocr,
        use_gray_world=use_gw, use_clahe=use_clahe_, use_auto_gamma=use_gamma,
        use_unsharp=use_sharp, use_denoise=use_denoise_,
        use_rectify=use_rect, use_otsu=use_otsu,
        use_tracker=use_trk, tracker_kind=tk, detect_every_n=n,
        temporal_stabilization=temporal,
        temporal_max_missing_detections=max_missing,
    )
    return ALPRPipeline(cfg)


pipeline = load_pipeline(
    conf, imgsz, ocr_backend, strict_format, min_plate_w, min_plate_h, min_plate_area,
    use_classical, classical_backend, cls_min_ocr,
    use_gray_world, use_clahe, use_auto_gamma, use_unsharp,
    use_denoise, use_rectify, use_otsu,
    use_tracker, tracker_kind, detect_every_n,
    temporal_stabilization, temporal_max_missing,
)
pipeline.cfg.conf = conf
pipeline.cfg.imgsz = imgsz
pipeline.cfg.final_min_w = min_plate_w
pipeline.cfg.final_min_h = min_plate_h
pipeline.cfg.final_min_area = min_plate_area
pipeline.cfg.ocr_backend = ocr_backend
pipeline.cfg.strict_format = strict_format
pipeline.cfg.use_classical_finder = use_classical
pipeline.cfg.classical_backend = classical_backend
pipeline.cfg.classical_min_ocr_conf = cls_min_ocr
pipeline.cfg.use_gray_world = use_gray_world
pipeline.cfg.use_clahe = use_clahe
pipeline.cfg.use_auto_gamma = use_auto_gamma
pipeline.cfg.use_unsharp = use_unsharp
pipeline.cfg.use_denoise = use_denoise
pipeline.cfg.use_rectify = use_rectify
pipeline.cfg.use_otsu = use_otsu
pipeline.cfg.use_tracker = use_tracker
pipeline.cfg.tracker_kind = tracker_kind
pipeline.cfg.detect_every_n = detect_every_n
pipeline.cfg.temporal_stabilization = temporal_stabilization
pipeline.cfg.temporal_max_missing_detections = temporal_max_missing


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
st.title("🚗 ALPR — распознавание автомобильных номерных знаков")
st.caption(
    "Дипломный проект. YOLO26n fine-tune на AUTO.RIA + **classical plate-finder "
    "(Sobel+морфология+Otsu, ДЗ4/8)** → препроцессинг ДЗ2/3/6/7/8 → CRNN+CTC или EasyOCR. "
    "Цвета боксов: 🟢 YOLO, 🟡 YOLO+CV agree, 🟠 только classical."
)

tab_photo, tab_webcam, tab_video, tab_rtsp = st.tabs(["📷 Фото", "🎥 Вебкамера", "🎬 Видео", "📡 RTSP-камера"])


# -----------------------------------------------------------------------
# Tab: Photo
# -----------------------------------------------------------------------
with tab_photo:
    up = st.file_uploader("Загрузите фото с автомобилем", type=["jpg", "jpeg", "png"])
    if up is not None:
        file_bytes = np.frombuffer(up.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        pipeline.reset()
        detections = pipeline.process_frame(img_rgb)
        t1 = time.perf_counter()

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(draw_detections(img_rgb, detections), caption=f"Обработано за {t1-t0:.2f} с", use_column_width=True)
        with col2:
            st.markdown("### Распознано")
            if not detections:
                st.warning("Номера не найдены.")
            for i, d in enumerate(detections, 1):
                track = f" · track: **#{d.track_id}**" if getattr(d, "track_id", 0) else ""
                st.markdown(
                    f"**#{i}** `{d.text or '???'}`{track} · цвет: **{d.plate_color}** · формат: **{getattr(d, 'plate_format', 'unknown')}** "
                    f"(det={d.conf:.2f}, ocr={d.ocr_conf:.2f})"
                )
                if getattr(d, "plate_color_context", ""):
                    st.caption(f"Контекст: {d.plate_color_context}")
                if d.rectified is not None:
                    st.image(
                        d.rectified,
                        caption=(
                            f"После препроцессинга (raw: {d.raw_text}, exact: {d.text}, "
                            f"normalized: {getattr(d, 'normalized_text', '')}, "
                            f"color: {d.plate_color}, format: {getattr(d, 'plate_format', 'unknown')}, "
                            f"context: {getattr(d, 'plate_color_context', '')})"
                        ),
                    )


# -----------------------------------------------------------------------
# Tab: Webcam
# -----------------------------------------------------------------------
with tab_webcam:
    st.markdown("Нажмите **Start**, чтобы открыть вебкамеру. Нажмите **Stop** для остановки.")
    col1, col2 = st.columns(2)
    start = col1.button("▶ Start", key="start_cam")
    stop = col2.button("⏹ Stop", key="stop_cam")
    frame_slot = st.empty()
    fps_slot = st.empty()

    if start:
        st.session_state["cam_running"] = True
        pipeline.reset()
    if stop:
        st.session_state["cam_running"] = False

    if st.session_state.get("cam_running"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error("Не удалось открыть камеру (CAP_DSHOW).")
            st.session_state["cam_running"] = False
        else:
            last_t = time.perf_counter()
            while st.session_state.get("cam_running"):
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                detections = pipeline.process_stream_frame(frame_rgb)
                out = draw_detections(frame_rgb, detections)
                now = time.perf_counter()
                fps = 1.0 / (now - last_t + 1e-6); last_t = now
                frame_slot.image(out, channels="RGB")
                fps_slot.metric("FPS", f"{fps:.1f}")
            cap.release()


# -----------------------------------------------------------------------
# Tab: RTSP stream (IP-камера)
# -----------------------------------------------------------------------
with tab_rtsp:
    st.markdown("Укажите RTSP URL (например `rtsp://user:pass@host:port/path`). Креды остаются в рамках сессии и не сохраняются.")
    rtsp_url = st.text_input("RTSP URL", value="", type="password", key="rtsp_url",
                             help="URL вводится как пароль — не отобразится в истории браузера.")
    col1, col2 = st.columns(2)
    start_rtsp = col1.button("▶ Start RTSP", key="start_rtsp")
    stop_rtsp = col2.button("⏹ Stop RTSP", key="stop_rtsp")
    rtsp_slot = st.empty()
    rtsp_fps = st.empty()
    rtsp_info = st.empty()

    if start_rtsp and rtsp_url.strip():
        st.session_state["rtsp_running"] = True
        st.session_state["rtsp_url_val"] = rtsp_url.strip()
        pipeline.reset()
    if stop_rtsp:
        st.session_state["rtsp_running"] = False

    if st.session_state.get("rtsp_running"):
        url = st.session_state.get("rtsp_url_val", "")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(url)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            st.error("Не удалось открыть RTSP-поток. Проверьте URL, креды и сетевой доступ.")
            st.session_state["rtsp_running"] = False
        else:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            rtsp_info.info(f"Поток открыт: {w}×{h}")
            last_t = time.perf_counter()
            fail_streak = 0
            while st.session_state.get("rtsp_running"):
                ok, frame_bgr = cap.read()
                if not ok:
                    fail_streak += 1
                    if fail_streak > 20:
                        st.warning("Потеряна связь с камерой.")
                        break
                    continue
                fail_streak = 0
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                detections = pipeline.process_stream_frame(frame_rgb)
                out = draw_detections(frame_rgb, detections)
                now = time.perf_counter()
                fps = 1.0 / (now - last_t + 1e-6); last_t = now
                rtsp_slot.image(out, channels="RGB")
                rtsp_fps.metric("FPS", f"{fps:.1f}")
            cap.release()


# -----------------------------------------------------------------------
# Tab: Video file
# -----------------------------------------------------------------------
with tab_video:
    vup = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov", "mkv"])
    if vup is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(vup.name).suffix) as tmp:
            tmp.write(vup.read()); tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        placeholder = st.empty()
        progress = st.progress(0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        pipeline.reset()
        idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections = pipeline.process_stream_frame(frame_rgb)
            out = draw_detections(frame_rgb, detections)
            placeholder.image(out, channels="RGB")
            idx += 1
            progress.progress(min(1.0, idx / total))
        cap.release()
