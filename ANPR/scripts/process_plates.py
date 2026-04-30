#!/usr/bin/env python
"""CLI инструмент для обработки номеров пластин из различных источников.

Использование:
    python process_plates.py image photo.jpg
    python process_plates.py video video.mp4 --max-frames 100
    python process_plates.py rtsp "rtsp://user:pass@host/path" --max-frames 50
    python process_plates.py webcam 0 --max-frames 20
    python process_plates.py screenshot --max-frames 30
    
    # С сохранением результатов
    python process_plates.py video video.mp4 --output results/ --save-crops --save-csv

Examples:
    # Обработать фото с сохранением
    python process_plates.py image car.jpg --output ./results/
    
    # Обработать видео (100 кадров, сохранить CSV)
    python process_plates.py video traffic.mp4 --max-frames 100 --save-csv
    
    # Live вебкамера (5 минут, 30 FPS)
    python process_plates.py webcam --max-frames 9000 --output ./live_results/
    
    # RTSP камера (1 минута)
    python process_plates.py rtsp "rtsp://admin:qwer@192.168.1.100:554/stream" --max-frames 3000 --resize 0.5
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.pipeline import ALPRPipeline, PipelineConfig
from src.source_manager import create_source


class PlateProcessor:
    """Обработчик для обнаружения и распознания номеров."""

    def __init__(self, output_dir: Optional[Path] = None, save_crops: bool = False):
        self.output_dir = output_dir
        self.save_crops = save_crops
        self.results = []
        self.frame_idx = 0
        
        # Инициализация pipeline
        self.pipeline = ALPRPipeline(PipelineConfig(
            conf=0.25,
            imgsz=1536,
            use_classical_finder=False,
            final_min_area=800,
            min_plate_distance=150,
            use_tracker=False,
        ))
        
        if self.output_dir:
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.save_crops:
                (self.output_dir / "crops").mkdir(exist_ok=True)
            (self.output_dir / "frames").mkdir(exist_ok=True)

    def process_frame(self, frame_rgb: np.ndarray) -> dict:
        """Обработать один кадр.
        
        Args:
            frame_rgb: Кадр в формате RGB
            
        Returns:
            Словарь с результатами
        """
        self.frame_idx += 1
        
        # Конвертировать RGB → BGR для pipeline
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Обработка
        detections = self.pipeline(frame_bgr)
        
        # Сохранение результатов
        result = {
            "frame_idx": self.frame_idx,
            "timestamp": time.time(),
            "num_plates": len(detections),
            "plates": []
        }
        
        for det in detections:
            plate_info = {
                "text": det.text,
                "confidence": det.confidence,
                "bbox": det.bbox,
                "color": det.color if hasattr(det, 'color') else None,
            }
            result["plates"].append(plate_info)
            
            # Сохранить crop изображения
            if self.save_crops and det.bbox:
                x1, y1, x2, y2 = map(int, det.bbox)
                crop = frame_bgr[y1:y2, x1:x2]
                crop_path = self.output_dir / "crops" / f"frame{self.frame_idx:05d}_{det.text}.jpg"
                cv2.imwrite(str(crop_path), crop)
        
        # Сохранить аннотированный кадр
        if self.output_dir:
            annotated = self.pipeline.draw_detections(frame_bgr, detections)
            frame_path = self.output_dir / "frames" / f"frame{self.frame_idx:05d}.jpg"
            cv2.imwrite(str(frame_path), annotated)
        
        self.results.append(result)
        return result

    def save_csv(self, filename: str = "plates.csv"):
        """Сохранить результаты в CSV."""
        if not self.output_dir:
            return
        
        csv_path = self.output_dir / filename
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Timestamp", "Plate Text", "Confidence", "Color", "BBox"])
            
            for result in self.results:
                for plate in result.get("plates", []):
                    writer.writerow([
                        result["frame_idx"],
                        result["timestamp"],
                        plate["text"],
                        f"{plate['confidence']:.2f}",
                        plate["color"],
                        str(plate["bbox"])
                    ])
        
        print(f"✅ CSV сохранен: {csv_path}")

    def save_summary(self, filename: str = "summary.txt"):
        """Сохранить итоговую статистику."""
        if not self.output_dir:
            return
        
        summary_path = self.output_dir / filename
        total_plates = sum(r["num_plates"] for r in self.results)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("PLATE DETECTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total frames: {len(self.results)}\n")
            f.write(f"Total plates detected: {total_plates}\n")
            f.write(f"Average plates per frame: {total_plates / len(self.results):.2f}\n")
            f.write(f"Processing time: {time.time():.1f}s\n\n")
            
            f.write("DETECTED PLATES:\n")
            f.write("-" * 60 + "\n")
            
            plates_dict = {}
            for result in self.results:
                for plate in result.get("plates", []):
                    text = plate["text"]
                    if text not in plates_dict:
                        plates_dict[text] = 0
                    plates_dict[text] += 1
            
            for text, count in sorted(plates_dict.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {text:15} × {count:3} раз\n")
        
        print(f"✅ Summary сохранен: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Обработка номеров пластин из различных источников",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Позиционные аргументы
    parser.add_argument("source_type", 
                       choices=["image", "video", "rtsp", "webcam", "screenshot"],
                       help="Тип источника")
    
    parser.add_argument("source_arg", nargs="?",
                       help="Аргумент источника (путь/URL/device_id)")
    
    # Опциональные аргументы
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Максимум кадров для обработки")
    
    parser.add_argument("--output", type=str, default=None,
                       help="Папка для сохранения результатов")
    
    parser.add_argument("--save-crops", action="store_true",
                       help="Сохранять обрезанные номера")
    
    parser.add_argument("--save-csv", action="store_true",
                       help="Сохранять результаты в CSV")
    
    parser.add_argument("--resize", type=float, default=1.0,
                       help="Масштабирование кадров (0.5 = 50%)")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Подробный вывод")
    
    args = parser.parse_args()
    
    # Проверка аргументов
    if args.source_type != "screenshot" and not args.source_arg:
        parser.error(f"source_arg требуется для {args.source_type}")
    
    # Создание источника
    print(f"🔧 Инициализация источника: {args.source_type}")
    try:
        if args.source_type == "image":
            source = create_source("image", path=args.source_arg, resize_ratio=args.resize)
        elif args.source_type == "video":
            source = create_source("video", path=args.source_arg, 
                                 max_frames=args.max_frames, resize_ratio=args.resize)
        elif args.source_type == "rtsp":
            source = create_source("rtsp", url=args.source_arg, 
                                 max_frames=args.max_frames, resize_ratio=args.resize)
        elif args.source_type == "webcam":
            device_id = int(args.source_arg) if args.source_arg else 0
            source = create_source("webcam", device_id=device_id,
                                 max_frames=args.max_frames, resize_ratio=args.resize)
        elif args.source_type == "screenshot":
            source = create_source("screenshot", 
                                 max_frames=args.max_frames, resize_ratio=args.resize)
        
        info = source.get_info()
        print(f"✅ Источник готов:")
        for key, val in info.items():
            print(f"   {key}: {val}")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Создание обработчика
    processor = PlateProcessor(output_dir=args.output, save_crops=args.save_crops)
    print(f"\n🚀 Начало обработки...")
    
    try:
        for frame_idx, frame_rgb in enumerate(source, 1):
            result = processor.process_frame(frame_rgb)
            
            if args.verbose or frame_idx % 10 == 0:
                num_plates = result["num_plates"]
                plates_str = ", ".join(p["text"] for p in result["plates"]) or "нет"
                print(f"  Кадр {frame_idx}: {num_plates} номер(ов) - {plates_str}")
            
            if args.max_frames and frame_idx >= args.max_frames:
                break
    
    except KeyboardInterrupt:
        print("\n⏹️  Обработка остановлена пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка обработки: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Итоги
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"  Обработано кадров: {processor.frame_idx}")
    total_plates = sum(r["num_plates"] for r in processor.results)
    print(f"  Всего номеров обнаружено: {total_plates}")
    
    if total_plates > 0:
        plates_dict = {}
        for result in processor.results:
            for plate in result.get("plates", []):
                text = plate["text"]
                if text not in plates_dict:
                    plates_dict[text] = 0
                plates_dict[text] += 1
        
        print(f"\n  Уникальные номера:")
        for text, count in sorted(plates_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"    {text}: {count}x")
    
    # Сохранение результатов
    if args.output:
        print(f"\n💾 Сохранение результатов в: {args.output}")
        if args.save_csv:
            processor.save_csv()
        processor.save_summary()
        print(f"✅ Готово!")


if __name__ == "__main__":
    main()
