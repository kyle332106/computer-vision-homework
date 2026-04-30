"""Source manager — универсальный интерфейс для различных источников видео/изображений.

Поддерживаемые источники:
  - Статические изображения (jpg, png, bmp)
  - Видеофайлы (mp4, avi, mov, mkv)
  - RTSP потоки (IP-камеры)
  - Вебкамеры (local webcam)
  - Скриншоты рабочего стола (desktop capture)

Использование:
    source = create_source("image", path="photo.jpg")
    source = create_source("video", path="video.mp4")
    source = create_source("rtsp", url="rtsp://user:pass@host/path")
    source = create_source("webcam", device_id=0)
    
    for frame in source:
        # Обработка кадра
        pass
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class SourceConfig:
    """Конфигурация источника."""
    source_type: str  # "image", "video", "rtsp", "webcam", "desktop"
    path: Optional[str] = None  # Для image/video
    url: Optional[str] = None  # Для RTSP
    device_id: int = 0  # Для webcam
    fps_limit: Optional[int] = None  # Ограничение FPS
    max_frames: Optional[int] = None  # Макс кадров (для видео)
    resize_ratio: float = 1.0  # Масштабирование (0.5 = 50%)


class VideoSource(ABC):
    """Абстрактный класс для источников видео."""

    def __init__(self, config: SourceConfig):
        self.config = config
        self.frame_count = 0
        self.total_frames = 0

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        """Итератор по кадрам (RGB)."""
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """Информация об источнике (resolution, fps, etc)."""
        pass

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Масштабировать кадр если нужно."""
        if self.config.resize_ratio != 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.config.resize_ratio)
            new_h = int(h * self.config.resize_ratio)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return frame


class ImageSource(VideoSource):
    """Источник: статическое изображение."""

    def __init__(self, config: SourceConfig):
        super().__init__(config)
        if not config.path:
            raise ValueError("path требуется для image source")
        self.image_path = Path(config.path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Изображение не найдено: {self.image_path}")

    def __iter__(self) -> Iterator[np.ndarray]:
        img_bgr = cv2.imread(str(self.image_path))
        if img_bgr is None:
            raise ValueError(f"Не удалось прочитать изображение: {self.image_path}")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = self.resize_frame(img_rgb)
        
        self.frame_count = 1
        yield img_rgb

    def get_info(self) -> dict:
        img = cv2.imread(str(self.image_path))
        h, w = img.shape[:2]
        return {
            "type": "image",
            "path": str(self.image_path),
            "resolution": (w, h),
            "format": self.image_path.suffix
        }


class VideoFileSource(VideoSource):
    """Источник: видеофайл."""

    def __init__(self, config: SourceConfig):
        super().__init__(config)
        if not config.path:
            raise ValueError("path требуется для video source")
        self.video_path = Path(config.path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Видео не найдено: {self.video_path}")

    def __iter__(self) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {self.video_path}")

        try:
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_count = 0

            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = self.resize_frame(frame_rgb)
                
                self.frame_count += 1
                yield frame_rgb

                if self.config.max_frames and self.frame_count >= self.config.max_frames:
                    break
        finally:
            cap.release()

    def get_info(self) -> dict:
        cap = cv2.VideoCapture(str(self.video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        return {
            "type": "video",
            "path": str(self.video_path),
            "resolution": (w, h),
            "fps": fps,
            "total_frames": total,
            "duration_sec": total / fps if fps > 0 else 0
        }


class RTSPSource(VideoSource):
    """Источник: RTSP поток (IP-камера)."""

    def __init__(self, config: SourceConfig):
        super().__init__(config)
        if not config.url:
            raise ValueError("url требуется для rtsp source")
        self.rtsp_url = config.url

    def __iter__(self) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            # Fallback без FFMPEG
            cap = cv2.VideoCapture(self.rtsp_url)
        
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Минимизировать задержку
        except Exception:
            pass

        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть RTSP поток: {self.rtsp_url}")

        try:
            fail_streak = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    fail_streak += 1
                    if fail_streak > 20:
                        break
                    continue
                
                fail_streak = 0
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = self.resize_frame(frame_rgb)
                
                self.frame_count += 1
                yield frame_rgb

                if self.config.max_frames and self.frame_count >= self.config.max_frames:
                    break
        finally:
            cap.release()

    def get_info(self) -> dict:
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.rtsp_url)
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cap.release()
        
        return {
            "type": "rtsp",
            "url": self.rtsp_url.split('@')[-1] if '@' in self.rtsp_url else self.rtsp_url,
            "resolution": (w, h),
            "fps": fps
        }


class WebcamSource(VideoSource):
    """Источник: Вебкамера."""

    def __init__(self, config: SourceConfig):
        super().__init__(config)
        self.device_id = config.device_id

    def __iter__(self) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.device_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть веб-камеру #{self.device_id}")

        try:
            self.frame_count = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = self.resize_frame(frame_rgb)
                
                self.frame_count += 1
                yield frame_rgb

                if self.config.max_frames and self.frame_count >= self.config.max_frames:
                    break
        finally:
            cap.release()

    def get_info(self) -> dict:
        cap = cv2.VideoCapture(self.device_id)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        return {
            "type": "webcam",
            "device_id": self.device_id,
            "resolution": (w, h),
            "fps": fps
        }


class ScreenshotSource(VideoSource):
    """Источник: Скриншоты рабочего стола (требует pyautogui или PIL)."""

    def __iter__(self) -> Iterator[np.ndarray]:
        try:
            from PIL import ImageGrab
        except ImportError:
            raise ImportError("pip install pillow для screenshot source")

        try:
            self.frame_count = 0
            while True:
                screenshot = ImageGrab.grab()
                frame_rgb = np.array(screenshot)
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
                frame_rgb = self.resize_frame(frame_rgb)
                
                self.frame_count += 1
                yield frame_rgb

                if self.config.max_frames and self.frame_count >= self.config.max_frames:
                    break
        except Exception as e:
            raise RuntimeError(f"Ошибка при захвате скриншота: {e}")

    def get_info(self) -> dict:
        try:
            from PIL import ImageGrab
            screenshot = ImageGrab.grab()
            w, h = screenshot.size
        except Exception:
            w, h = 1920, 1080  # Fallback
        
        return {
            "type": "screenshot",
            "resolution": (w, h)
        }


def create_source(source_type: str, **kwargs) -> VideoSource:
    """Фабрика для создания источников.
    
    Args:
        source_type: Тип источника ("image", "video", "rtsp", "webcam", "screenshot")
        **kwargs: Параметры конфигурации
    
    Returns:
        VideoSource субкласс
    
    Examples:
        create_source("image", path="photo.jpg")
        create_source("video", path="video.mp4", max_frames=100)
        create_source("rtsp", url="rtsp://user:pass@host/path")
        create_source("webcam", device_id=0)
        create_source("screenshot", max_frames=50)
    """
    config = SourceConfig(source_type=source_type, **kwargs)
    
    sources = {
        "image": ImageSource,
        "photo": ImageSource,
        "jpg": ImageSource,
        "png": ImageSource,
        "video": VideoFileSource,
        "mp4": VideoFileSource,
        "avi": VideoFileSource,
        "mov": VideoFileSource,
        "rtsp": RTSPSource,
        "webcam": WebcamSource,
        "camera": WebcamSource,
        "screenshot": ScreenshotSource,
        "screen": ScreenshotSource,
    }
    
    source_class = sources.get(source_type.lower())
    if not source_class:
        raise ValueError(f"Неизвестный тип источника: {source_type}. "
                        f"Доступные: {', '.join(sources.keys())}")
    
    return source_class(config)


if __name__ == "__main__":
    # Примеры использования
    import sys
    
    if len(sys.argv) > 1:
        source_arg = sys.argv[1]
        
        # Автоматическое определение типа по расширению
        if source_arg.startswith("rtsp://"):
            source = create_source("rtsp", url=source_arg, max_frames=5)
        elif source_arg.isdigit():
            source = create_source("webcam", device_id=int(source_arg))
        elif source_arg == "screenshot":
            source = create_source("screenshot", max_frames=10)
        else:
            # Определить по расширению
            path = Path(source_arg)
            if path.suffix.lower() in [".jpg", ".png", ".bmp"]:
                source = create_source("image", path=source_arg)
            else:
                source = create_source("video", path=source_arg, max_frames=5)
        
        print(f"Источник: {source.get_info()}")
        print(f"Обработка 5 кадров...")
        
        for i, frame in enumerate(source):
            print(f"  Кадр {i+1}: {frame.shape}")
