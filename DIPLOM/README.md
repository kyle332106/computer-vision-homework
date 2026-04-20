# Дипломная работа: ALPR — распознавание номерных знаков (любой формат, любой шрифт)

Система детекции + распознавания автомобильных номеров в реальном времени.
Читает **произвольный текст** на табличке: стандартные UA, дипломатические,
спецтехнику, такси, военные, **именные/vanity**, иностранные.

## Стек

| Компонент | Решение | Источник / связь с курсом |
|-----------|---------|----------|
| Детекция | YOLO26n (Ultralytics), fine-tune | lesson 18 (YOLO), ДЗ13–15 (CNN) |
| Color balance | Gray-world | ДЗ2 |
| Sharpening | Unsharp masking | ДЗ3 |
| Corner detection | Harris | ДЗ6 |
| Rectification | Homography | ДЗ7 |
| Binarization | Otsu | ДЗ8 |
| **OCR (основной)** | **CRNN+CTC** (своя сеть) — variable-length, любой формат | ДЗ13/14/15 (CNN-backbone + training) |
| OCR baseline | EasyOCR `['uk','en']` | — |
| Tracking | KCF vs CSRT | ДЗ10 |
| EDA | — | ДЗ12 |
| CharCNN per-slot (архив) | Классификатор на 24 класса | ДЗ14 (см. 02c_archive) |

**Почему CRNN+CTC, а не CharCNN:** CharCNN читает фиксированную длину/формат.
CRNN+CTC читает строку произвольной длины — обязательное условие для
именных номеров и нестандартных форматов.

## Поддерживаемые форматы номеров

Генератор синтетики (`scripts/generate_synthetic_ocr.py`) покрывает:

| Тип | Пример | Фон / текст | Доля в синтетике |
|---|---|---|---|
| Стандартный UA | `АВ1234СР` | белый / чёрный | 35% |
| Именной / vanity | `SANDY`, `BOSS-2024` | белый / чёрный | 25% |
| Иностранные | `ABC-123-DE`, `123AB456` | белый / чёрный | 15% |
| Дипломатический | `001CD123` | красный / белый | 8% |
| Спецтехника | `С1234АВ` | синий / белый | 7% |
| Военный | `ВТ12345` | зелёный / белый | 5% |
| Такси | `АВ1234ТХ` | жёлтый / чёрный | 5% |

Рендер идёт 15+ системными шрифтами (Arial, Calibri, Consolas, Courier, Times,
Verdana, Impact, Tahoma, Franklin, Georgia, Trebuchet…) + любыми `.ttf/.otf`
из `data/fonts/`. Это и обеспечивает независимость от шрифта.

## Железо / окружение

- Python 3.13
- PyTorch 2.6 + CUDA 12.4
- NVIDIA RTX 3050 Laptop (6 GB VRAM)

## Структура

```
DIPLOM/
├── config.yaml                     # пути, параметры OCR/CRNN, флаги
├── requirements.txt
├── app.py                          # Streamlit-демо (3 вкладки, CRNN/EasyOCR переключатель)
├── src/
│   ├── preprocess.py               # Gray-world, Unsharp, Harris, Homography, Otsu
│   ├── pipeline.py                 # ALPRPipeline (детекция → препроц → OCR → трекинг)
│   ├── tracker.py                  # KCF vs CSRT
│   ├── crnn.py                     # ⭐ CRNN + CTC (основной OCR)
│   └── char_cnn.py                 # per-slot CNN (ДЗ14-baseline, архив)
├── scripts/
│   ├── convert_to_yolo.py          # AUTO.RIA VIA JSON → YOLO + ocr_{split}.csv
│   ├── prepare_ocr_crops.py        # реальные кропы + GT-текст из ocr_*.csv
│   └── generate_synthetic_ocr.py   # ⭐ все форматы + шрифты + аугментации
├── notebooks/
│   ├── 01_dataset_analysis.ipynb           # EDA
│   ├── 02_train_detector.ipynb             # fine-tune YOLO26n
│   ├── 02b_train_crnn_ocr.ipynb            # ⭐ обучение CRNN+CTC
│   ├── 02c_train_char_cnn_archive.ipynb    # (архив) CharCNN baseline
│   ├── 03_ocr_baseline.ipynb               # CRNN vs EasyOCR + ablation ДЗ
│   └── 04_evaluation.ipynb                 # финальный отчёт
├── data/
│   ├── synthetic_ocr/              # сгенерированная синтетика (в .gitignore)
│   ├── ocr_crops/                  # реальные кропы из AUTO.RIA
│   ├── raw/                        # исходники (опц.)
│   └── yolo/                       # ссылка на D:/Новая папка/alpr_data/yolo_plates
└── models/                         # сохранённые веса (в .gitignore)
    ├── yolo26n.pt                  # baseline
    ├── yolo26_plate.pt             # fine-tuned (после 02_)
    └── crnn_ocr.pt                 # обученная CRNN (после 02b_)
```

## Датасеты

Датасеты лежат в `D:/Новая папка/alpr_data/` — пути в [config.yaml](config.yaml).

1. **[AUTO.RIA Numberplate Dataset](https://nomeroff.net.ua/datasets/autoriaNumberplateDataset-2026-01-13.zip)** (~10.6 GB, VIA-формат) — реальные UA-номера для детекции. При наличии текстовых меток в VIA JSON (`description`/`label`) используется и для CRNN.
2. **Synthetic Ukrainian LP** ([Zenodo](https://zenodo.org/records/13342103), 236 MB) — 10 000 синтетических с character-level YOLO — для детектора.
3. **`data/synthetic_ocr/`** (генерим сами) — 20 000+ разнообразных табличек с разметкой по строке. Главный источник для CRNN под «любой формат/шрифт».

## Запуск с нуля

```bash
# 1. venv + зависимости
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. Сконвертировать AUTO.RIA в YOLO + OCR-разметку
python scripts/convert_to_yolo.py \
    --source D:/Новая папка/alpr_data/autoriaNumberplateDataset-2026-01-13 \
    --dest   D:/Новая папка/alpr_data/yolo_plates \
    --subset 4000

# 3. (опц., если VIA JSON содержит текст) Нарезать реальные OCR-кропы
python scripts/prepare_ocr_crops.py \
    --yolo-root   D:/Новая папка/alpr_data/yolo_plates \
    --images-root D:/Новая папка/alpr_data/autoriaNumberplateDataset-2026-01-13 \
    --out         data/ocr_crops

# 4. Сгенерировать синтетику (все форматы + шрифты)
python scripts/generate_synthetic_ocr.py --out data/synthetic_ocr --n 20000

# 5. Прогнать ноутбуки по порядку
jupyter lab notebooks/
#   01_dataset_analysis.ipynb        — EDA
#   02_train_detector.ipynb          — fine-tune YOLO26n
#   02b_train_crnn_ocr.ipynb         — обучение CRNN+CTC на synthetic_ocr (+real)
#   03_ocr_baseline.ipynb            — CRNN vs EasyOCR + ablation
#   04_evaluation.ipynb              — финальный отчёт

# 6. Streamlit-демо
streamlit run app.py
```

## RTSP (IP-камеры)

Поддерживается через вкладку **📡 RTSP-камера** в Streamlit-демо и отдельный скрипт для быстрой проверки:

```bash
# проверить доступность потока + сохранить первый кадр
python scripts/test_rtsp.py "rtsp://user:pass@host:port/path" --save-frame probe.jpg

# прогнать 30 кадров через ALPR end-to-end
python scripts/test_rtsp.py "rtsp://..." --frames 30
```

Бэкенд — OpenCV/FFmpeg с `CAP_PROP_BUFFERSIZE=1` для снижения задержки.
На тестовой камере (2592×1520 @ 50fps) достигнуто ~12 FPS end-to-end без трекера.

## Что демонстрирует диплом

1. **Fine-tuning современной архитектуры** — YOLO26n (lesson 18).
2. **Классические CV-техники из курса** как препроцессинг — каждая из ДЗ2/3/6/7/8 попадает в итоговый пайплайн и валидируется в ablation.
3. **Собственная нейросеть** — CRNN+CTC:
   - Backbone по канонам ДЗ14 (Conv→BN→ReLU блоки, MaxPool)
   - Sequence-часть (BiLSTM×2) — необходимое расширение под переменную длину
   - CTC-loss — alignment-free обучение
4. **Сравнение трекеров KCF vs CSRT** как в ДЗ10 — выбор оптимального для real-time.
5. **Streamlit-демо** с переключателем CRNN↔EasyOCR, флагом strict_format и живой вебкамерой.

## Метрики (актуальные)

### Детектор YOLO26n (val AUTO.RIA, 335 изображений, 355 bbox)

| Метрика | Target | Actual |
|---------|--------|--------|
| mAP50 | ≥ 0.85 | **0.9748** ✓ |
| mAP50-95 | — | 0.8486 |
| Precision | — | 0.977 |
| Recall | — | 0.941 |

### CRNN+CTC OCR

**Synthetic val (2000 семплов, overall):** CER = **0.034**, exact-match = **84.6%** ✓

| Формат | n | mean CER | exact-match |
|---|---|---|---|
| taxi (жёлтый) | 100 | 0.000 | 100.0% |
| diplomat (красный) | 145 | 0.001 | 99.3% |
| standard UA | 715 | 0.001 | 99.2% |
| military (зелёный) | 106 | 0.003 | 98.1% |
| foreign | 289 | 0.003 | 98.3% |
| spec (синий) | 140 | 0.005 | 97.9% |
| **named / vanity** | 505 | 0.176 | 42.4% |

> Named — произвольный текст со случайным charset и длиной 2–8; 42% exact с CER 0.176 означает, что в среднем промахивается 1 символ из 5–8 — ожидаемый результат для действительно произвольных строк.

### CRNN на real AUTO.RIA (85/15 merge+resplit, 136 val семплов)

| Метрика | Значение |
|---|---|
| mean CER | **0.237** |
| exact-match (strict) | **47.79%** |

**Breakdown по формату:**

| region_name | n | mean CER | exact-match |
|---|---|---|---|
| unknown (без метки) | 41 | 0.076 | 85.4% |
| eu-ua-2004 | 11 | 0.115 | 54.5% |
| xx-unknown | 4 | 0.250 | 50.0% |
| su (советский) | 13 | 0.275 | 46.1% |
| eu-ua-2015 (новый UA) | 60 | 0.310 | 26.7% |
| eu-ua-1995, eu | 6 | 0.77 | 0% |

**Что сработало (3 фактора):**
1. Rectify (Harris+Homography, ДЗ6+7) в `prepare_ocr_crops.py --rectify` — кропы стали aspect-ratio-unified
2. Padded preprocessing в CRNN (`min_w=128`) — достаточно timesteps для 8-символьной строки
3. Честный 85/15 merge+resplit из объединённого пула AUTO.RIA train∪val (до этого val был из другого распределения)

**Pipeline обучения CRNN (4 прохода):**

| Проход | Конфигурация | Best val CER |
|---|---|---|
| 1 | 20 эпох, synthetic_ocr + real, LR 3e-4 | 0.125 |
| 2 | +Zenodo +real×10, 15 эпох fine-tune, LR 1e-4 | 0.086 |
| 3 | +padded preprocess min_w=128, 10 эпох fine-tune | 0.082 |
| 4 | +rectified real +85/15 merge-split +real×15, 15 эпох fine-tune | **0.036** |

### FPS (real-time на RTX 3050)
Измеряется в `app.py` в live-режиме. Target ≥ 15 FPS.
