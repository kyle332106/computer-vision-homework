"""Multi-RTSP low-latency runner (no queue, latest-frame only).

Цель: максимальная скорость и минимальная задержка для множества RTSP потоков.

Ключевой принцип:
- БЕЗ ОЧЕРЕДИ: каждый reader-поток всегда перезаписывает только ПОСЛЕДНИЙ кадр.
- processor-поток берёт только самый свежий кадр и пропускает устаревшие.

Запуск (Windows PowerShell):
    python scripts/multi_rtsp_fast.py \
      --url "rtsp://user:pass@cam1/path" \
      --url "rtsp://user:pass@cam2/path" \
      --mode detect \
      --target-fps 10

Режимы:
- detect: только детекция (самый быстрый, рекомендован для 10+ FPS)
- full: полный ALPR (det + OCR), обычно медленнее и зависит от GPU
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import ALPRPipeline, Detection, PipelineConfig


def open_stream(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(url)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def safe_stream_name(url: str, idx: int) -> str:
    tail = url.split("@")[-1] if "@" in url else url
    return f"cam{idx}:{tail}"


@dataclass
class StreamStats:
    read_frames: int = 0
    processed_frames: int = 0
    detected_items: int = 0
    skipped_frames: int = 0
    read_fails: int = 0
    read_fps: float = 0.0
    proc_fps: float = 0.0


@dataclass
class StreamState:
    name: str
    url: str
    mode: str
    cfg: PipelineConfig
    target_fps: float
    auto_tune: bool
    report_every: float
    max_read_fails: int

    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)

    latest_bgr: Optional[np.ndarray] = None
    latest_frame_id: int = 0

    stats: StreamStats = field(default_factory=StreamStats)

    read_thread: Optional[threading.Thread] = None
    proc_thread: Optional[threading.Thread] = None

    pipeline: Optional[ALPRPipeline] = None


class StreamWorker:
    def __init__(self, state: StreamState):
        self.state = state
        self._last_processed_id = 0
        self._last_report_t = time.perf_counter()
        self._last_report_read = 0
        self._last_report_proc = 0

    def start(self) -> None:
        self.state.pipeline = ALPRPipeline(self.state.cfg)
        _ = self.state.pipeline.detector

        self.state.read_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.state.proc_thread = threading.Thread(target=self._processor_loop, daemon=True)
        self.state.read_thread.start()
        self.state.proc_thread.start()

    def stop(self) -> None:
        self.state.stop_event.set()

    def join(self, timeout: float = 2.0) -> None:
        if self.state.read_thread:
            self.state.read_thread.join(timeout=timeout)
        if self.state.proc_thread:
            self.state.proc_thread.join(timeout=timeout)

    def _reader_loop(self) -> None:
        cap = open_stream(self.state.url)
        if not cap.isOpened():
            print(f"[{self.state.name}] ERROR: cannot open RTSP stream")
            self.state.stop_event.set()
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        print(f"[{self.state.name}] stream opened: {w}x{h} @ {fps:.1f}")

        try:
            while not self.state.stop_event.is_set():
                ok, frame_bgr = cap.read()
                if not ok:
                    self.state.stats.read_fails += 1
                    if self.state.stats.read_fails >= self.state.max_read_fails:
                        print(f"[{self.state.name}] ERROR: too many read fails, stopping")
                        self.state.stop_event.set()
                        break
                    continue

                self.state.stats.read_fails = 0
                self.state.stats.read_frames += 1

                # No queue: overwrite latest frame atomically.
                with self.state.lock:
                    self.state.latest_bgr = frame_bgr
                    self.state.latest_frame_id += 1
        finally:
            cap.release()

    def _processor_loop(self) -> None:
        while not self.state.stop_event.is_set():
            with self.state.lock:
                frame_id = self.state.latest_frame_id
                frame_bgr = None if self.state.latest_bgr is None else self.state.latest_bgr.copy()

            if frame_bgr is None or frame_id == self._last_processed_id:
                time.sleep(0.001)
                self._maybe_report()
                continue

            if self._last_processed_id > 0 and frame_id > self._last_processed_id + 1:
                self.state.stats.skipped_frames += (frame_id - self._last_processed_id - 1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if self.state.mode == "detect":
                boxes = self.state.pipeline.detect(frame_rgb)  # type: ignore[union-attr]
                self.state.stats.detected_items += len(boxes)
            else:
                dets: list[Detection] = self.state.pipeline.process_stream_frame(frame_rgb)  # type: ignore[union-attr]
                self.state.stats.detected_items += len(dets)

            self.state.stats.processed_frames += 1
            self._last_processed_id = frame_id

            self._maybe_report()

    def _maybe_report(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_report_t
        if dt < self.state.report_every:
            return

        r_now = self.state.stats.read_frames
        p_now = self.state.stats.processed_frames
        read_fps = (r_now - self._last_report_read) / dt
        proc_fps = (p_now - self._last_report_proc) / dt
        self.state.stats.read_fps = read_fps
        self.state.stats.proc_fps = proc_fps

        msg = (
            f"[{self.state.name}] read={read_fps:.1f} FPS  proc={proc_fps:.1f} FPS  "
            f"processed={self.state.stats.processed_frames}  skipped={self.state.stats.skipped_frames}"
        )

        if self.state.mode == "full" and self.state.auto_tune and proc_fps < self.state.target_fps:
            old_every = self.state.pipeline.cfg.detect_every_n  # type: ignore[union-attr]
            old_imgsz = self.state.pipeline.cfg.imgsz  # type: ignore[union-attr]

            if self.state.pipeline.cfg.detect_every_n < 12:  # type: ignore[union-attr]
                self.state.pipeline.cfg.detect_every_n += 1  # type: ignore[union-attr]
            elif self.state.pipeline.cfg.imgsz > 512:  # type: ignore[union-attr]
                self.state.pipeline.cfg.imgsz = max(512, self.state.pipeline.cfg.imgsz - 128)  # type: ignore[union-attr]

            new_every = self.state.pipeline.cfg.detect_every_n  # type: ignore[union-attr]
            new_imgsz = self.state.pipeline.cfg.imgsz  # type: ignore[union-attr]
            if (new_every, new_imgsz) != (old_every, old_imgsz):
                msg += f"  autotune: detect_every_n {old_every}->{new_every}, imgsz {old_imgsz}->{new_imgsz}"

        print(msg)

        self._last_report_t = now
        self._last_report_read = r_now
        self._last_report_proc = p_now


def load_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = []
    if args.url:
        urls.extend(args.url)
    if args.urls_file:
        with open(args.urls_file, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    urls.append(s)
    return urls


def build_cfg(args: argparse.Namespace) -> PipelineConfig:
    # Fast defaults focused on low latency.
    return PipelineConfig(
        detector_weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        use_classical_finder=False,
        use_tracker=(args.mode == "full"),
        detect_every_n=args.detect_every_n,
        ocr_backend="crnn",
        final_min_area=args.min_area,
        min_plate_distance=120,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", default=[], help="RTSP URL (repeat flag for many streams)")
    ap.add_argument("--urls-file", default=None, help="txt file with RTSP URLs (one per line)")

    ap.add_argument("--mode", choices=["detect", "full"], default="detect",
                    help="detect=fast detector-only, full=det+ocr")
    ap.add_argument("--weights", default="models/yolo26_plate_combined.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--detect-every-n", type=int, default=4,
                    help="used in full mode with tracker")
    ap.add_argument("--min-area", type=int, default=700)

    ap.add_argument("--target-fps", type=float, default=10.0,
                    help="target processing FPS per stream")
    ap.add_argument("--auto-tune", action="store_true",
                    help="in full mode, auto-tune detect_every_n/imgsz if FPS drops")
    ap.add_argument("--report-every", type=float, default=2.0)
    ap.add_argument("--max-read-fails", type=int, default=30)
    ap.add_argument("--seconds", type=int, default=0,
                    help="0 = run until Ctrl+C")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    urls = load_urls(args)
    if not urls:
        raise SystemExit("Provide at least one stream: --url <rtsp://...> or --urls-file <file.txt>")

    workers: list[StreamWorker] = []
    for i, url in enumerate(urls, 1):
        state = StreamState(
            name=safe_stream_name(url, i),
            url=url,
            mode=args.mode,
            cfg=build_cfg(args),
            target_fps=args.target_fps,
            auto_tune=args.auto_tune,
            report_every=args.report_every,
            max_read_fails=args.max_read_fails,
        )
        worker = StreamWorker(state)
        worker.start()
        workers.append(worker)

    print(
        f"started {len(workers)} streams | mode={args.mode} | target_fps={args.target_fps:.1f} | "
        "frame policy=no-queue(latest-only)"
    )

    t0 = time.perf_counter()
    try:
        while True:
            if args.seconds > 0 and (time.perf_counter() - t0) >= args.seconds:
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        for w in workers:
            w.stop()
        for w in workers:
            w.join()

    print("\n=== SUMMARY ===")
    for w in workers:
        s = w.state.stats
        print(
            f"{w.state.name}: read={s.read_frames}, processed={s.processed_frames}, "
            f"detected={s.detected_items}, skipped={s.skipped_frames}, "
            f"proc_fps_last={s.proc_fps:.1f}"
        )


if __name__ == "__main__":
    main()
