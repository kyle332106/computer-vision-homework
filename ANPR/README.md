# ALPR Diploma Project

Система распознавания автомобильных номерных знаков в реальном времени для
парковочной/дорожной камеры. Проект объединяет классические методы Computer
Vision из курса и современные нейросетевые модели для детекции, выравнивания,
улучшения и OCR номерных знаков.

Проект подготовлен как `4. YOUR OWN PROJECT` для финального задания по курсу
Computer Vision.

## Motivation

Задача проекта: построить end-to-end ALPR-систему, которая умеет работать на
реальной сцене с IP-камеры, где:

- номера маленькие относительно всего кадра;
- номера сняты под углом;
- в одном кадре присутствует несколько автомобилей;
- форматы номерных знаков различаются;
- встречаются иностранные, нестандартные и именные таблички;
- качество изображения ухудшается из-за расстояния, освещения и компрессии RTSP.

Цель системы не только в том, чтобы найти номер, но и:

- геометрически выровнять его;
- улучшить качество кропа;
- прочитать символы в правильном порядке;
- определить тип/цвет фона номерной плашки в терминах предметной области.

## Introduction

В проекте комбинируются два класса подходов:

- **Classical CV**
  - color balance (Gray-world)
  - sharpening (Unsharp Mask)
  - corner detection (Harris)
  - homography-based rectification
  - thresholding (Otsu)
  - morphology-based plate proposals
- **Deep Learning**
  - YOLO26n для детекции номерных знаков
  - CRNN + CTC для OCR переменной длины

Почему выбран именно такой стек:

- detector должен работать на wide-shot сцене, где номера занимают малую часть кадра;
- OCR должен читать не фиксированный формат, а строку произвольной длины;
- classical preprocessing улучшает качество plate-crop до подачи в OCR;
- RTSP-demo требует практического компромисса между recall, OCR quality и FPS.

## Description

### Pipeline

Итоговый пайплайн:

1. YOLO26n detector
2. Sliced high-resolution inference для крупных RTSP-кадров
3. Classical candidate recovery на основе фильтров курса
4. Dedupe / merge кандидатов
5. Harris + Homography rectification
6. Quality restoration:
   - Gray-world
   - CLAHE
   - Auto-gamma
   - Bilateral denoise
   - Unsharp mask
7. OCR via CRNN+CTC
8. Postprocess:
   - латинизация символов
   - сохранение порядка символов
   - определение plate format
   - определение цвета/типа фона плашки

### Связь с курсом

| Компонент | Реализация | Связь с курсом |
|---|---|---|
| Детекция | YOLO26n fine-tune | lesson 18, ДЗ13–15 |
| Color balance | Gray-world | ДЗ2 |
| Contrast recovery | CLAHE, gamma | lesson 2 |
| Sharpening | Unsharp mask | ДЗ3 |
| Denoising | Bilateral filter | lesson 3 |
| Corners | Harris | ДЗ6 |
| Rectification | Homography | ДЗ7 |
| Thresholding | Otsu | ДЗ8 |
| Tracking | KCF / CSRT API | ДЗ10 |
| OCR | CRNN + CTC | ДЗ13–15 |
| EDA | dataset analysis | ДЗ12 |

### Почему CRNN+CTC, а не fixed-format classifier

Fixed-slot classifier удобен только для жёсткого шаблона номера.

В этом проекте OCR должен читать:

- обычные номера;
- номера разных стран;
- нестандартные строки;
- именные таблички;
- plate strings разной длины.

Поэтому основным OCR выбран **CRNN + CTC**.

## Supported Plate Types

Система проектируется под mixed-domain scenario, а не только под один UA-шаблон.

Синтетический OCR-генератор покрывает:

| Тип | Пример | Фон |
|---|---|---|
| Standard | `AB1234CE` | белый |
| Foreign | `ABC-123-DE` | белый |
| Named / vanity | `SANDY`, `BOSS-2024` | белый |
| Diplomatic | `001CD123` | красный |
| Special | `C1234AB` | синий |
| Military | `BT12345` | зелёный |
| Taxi | `AB1234TX` | жёлтый |

Система интерпретирует цвет не как произвольный RGB-оттенок, а как допустимый
тип фона номерной плашки:

- `белый`
- `жёлтый`
- `красный`
- `синий`
- `зелёный`
- `неизвестно`

## Repository Structure

```text
DIPLOM/
├── app.py
├── config.yaml
├── requirements.txt
├── README.md
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_train_detector.ipynb
│   ├── 02b_train_crnn_ocr.ipynb
│   ├── 02c_train_char_cnn_archive.ipynb
│   ├── 03_ocr_baseline.ipynb
│   └── 04_evaluation.ipynb
├── scripts/
│   ├── convert_to_yolo.py
│   ├── prepare_ocr_crops.py
│   ├── generate_synthetic_ocr.py
│   ├── prepare_zenodo_ocr.py
│   ├── prepare_combined_detector.py
│   ├── train_detector.py
│   ├── finetune_detector.py
│   ├── train_crnn.py
│   ├── evaluate_crnn.py
│   ├── collect_scene_ocr.py
│   ├── process_plates.py
│   ├── multi_rtsp_fast.py
│   └── test_rtsp.py
├── src/
│   ├── preprocess.py
│   ├── pipeline.py
│   ├── tracker.py
│   ├── plate_color.py
│   ├── plate_finder.py
│   ├── plate_proposer_nn.py
│   ├── crnn.py
│   ├── char_cnn.py
│   ├── source_manager.py
│   └── io_utils.py
└── train_log.txt
```

## Datasets

Основные источники данных:

1. **AUTO.RIA Numberplate Dataset**
   - real detection dataset
   - VIA annotations
   - используется для детектора и real OCR crops
2. **Zenodo synthetic UA**
   - synthetic plate-font dataset
   - используется для OCR fine-tune
3. **Self-generated synthetic OCR**
   - `data/synthetic_ocr`
   - форматы, цвета и шрифты
4. **Scene-specific RTSP OCR dataset**
   - `data/scene_ocr`
   - автоматически собранные и отфильтрованные plate crops с камеры

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Quick Start

Если веса уже обучены и лежат в `models/`, самый короткий путь получить результат:

### Поддерживаемые входные источники

- фото (upload в Streamlit)
- видеофайл (upload в Streamlit)
- RTSP-ссылка
- веб-камера

### 1. Streamlit demo

```bash
cd DIPLOM
streamlit run app.py
```

Что делать дальше:

1. открыть вкладку `RTSP-камера`
2. вставить RTSP URL
3. нажать `Start RTSP`
4. увидеть bbox, текст номера, цвет плашки и FPS

### 2. CLI demo на RTSP

```bash
cd DIPLOM
python scripts/test_rtsp.py "rtsp://user:pass@host/path" ^
  --frames 30 ^
  --imgsz 1536 ^
  --conf 0.25 ^
  --weights models/yolo26_plate_combined.pt ^
  --classical-backend cpu ^
  --classical-ocr-thr 0.8
```

### 3. CLI demo на одном кадре

```bash
cd DIPLOM
python scripts/test_rtsp.py "rtsp://user:pass@host/path" --save-frame runs/eval/rtsp_probe.jpg
```

### 4. Multi-RTSP ultra-fast (10+ FPS target, no queue)

Для нескольких RTSP потоков с минимальной задержкой:

```bash
cd DIPLOM
python scripts/multi_rtsp_fast.py ^
  --url "rtsp://user:pass@cam1/path" ^
  --url "rtsp://user:pass@cam2/path" ^
  --mode detect ^
  --imgsz 640 ^
  --conf 0.35 ^
  --target-fps 10
```

Принцип работы этого режима:

- без очереди кадров (latest-frame only);
- старые кадры отбрасываются;
- каждый поток обрабатывается в своём thread;
- `--mode detect` — самый быстрый режим для цели 10+ FPS.

## Expected Output

После запуска пользователь должен получить:

- изображение или поток с найденными номерными знаками;
- текст номера в правильном порядке;
- цвет/тип номерной плашки человеческим названием:
  - `белый`
  - `жёлтый`
  - `красный`
  - `синий`
  - `зелёный`
  - `неизвестно`
- для Streamlit:
  - bbox на кадре
  - plate text
  - plate color
  - format
  - FPS
- для CLI:
  - строки вида `frame N  plates=K  [TEXT [цвет], ...]`

Ожидаемые артефакты после полного обучения:

- `models/yolo26_plate_combined.pt`
- `models/crnn_ocr.pt`
- `runs/detect/...` с логами обучения
- `runs/eval/...` с проверочными изображениями и JSON-отчётами

## Reproducibility

### Detector

```bash
python scripts/convert_to_yolo.py \
  --source D:/Новая папка/alpr_data/autoriaNumberplateDataset-2026-01-13 \
  --dest   D:/Новая папка/alpr_data/yolo_plates \
  --subset 4000

python scripts/prepare_combined_detector.py
python scripts/train_detector.py
python scripts/finetune_detector.py --epochs 5 --batch 12 --imgsz 640 --workers 0 --name ua_plates_combined_v2
```

### OCR

```bash
python scripts/generate_synthetic_ocr.py --out data/synthetic_ocr --n 20000
python scripts/prepare_zenodo_ocr.py
python scripts/prepare_ocr_crops.py --rectify

python scripts/train_crnn.py
python scripts/train_crnn.py --resume --lr 1e-4 --real-boost 10 --epochs 15
python scripts/train_crnn.py --resume --lr 5e-5 --epochs 6 --real-boost 10 --scene-boost 30
```

### RTSP scene-specific OCR collection

```bash
python scripts/collect_scene_ocr.py "rtsp://user:pass@host/path" \
  --frames 180 \
  --imgsz 1536 \
  --conf 0.08 \
  --weights models/yolo26_plate_combined.pt \
  --out data/scene_ocr
```

### Demo

```bash
streamlit run app.py
```

CLI quick check:

```bash
python scripts/test_rtsp.py "rtsp://user:pass@host/path" --frames 30
```

## Demo

Демо-сценарий для защиты:

1. открыть Streamlit или `test_rtsp.py`
2. показать кадр с несколькими автомобилями
3. показать, что detector видит несколько номеров в wide-shot сцене
4. показать итоговый текст номера
5. показать цвет/тип фона номерной плашки

Практический RTSP-scenario в текущем проекте:

- камера: `2592×1520`
- production-like режим: `imgsz=1536`, `conf=0.25`
- detector использует sliced inference

## Real Run Examples

Ниже приведены реальные фрагменты запусков, полученные в текущем проекте.

### Example 1. Detector training result

Combined fine-tune detector:

```text
=== Метрики (combined val) ===
mAP50    : 0.9851
mAP50-95 : 0.7057
Precision: 0.9783
Recall   : 0.9462
```

Это означает, что detector уверенно находит номерные знаки и после fine-tune
лучше обобщается на parking-angle scene.

### Example 2. OCR training result

Synthetic OCR validation:

```text
CER         : 0.034
Exact-match : 84.6%
```

Real AUTO.RIA validation:

```text
mean CER    : 0.237
exact-match : 47.79%
```

Scene-specific OCR fine-tune заметно улучшил стабильность строк именно на
текущей RTSP-камере.

### Example 3. RTSP run with stable recognized plates

Реальный прогон:

```text
[rtsp] resolution=2592x1520  declared_fps=50.0
[alpr] OCR backend = crnn  imgsz=1536  conf=0.25
  frame   0  plates=3  [AA5104EK [белый], KA1959BI [белый], AA2162TX [белый]]
```

В этом режиме стабильно читаются реальные plate strings:

- `AA5104EK`
- `KA1959BI`
- `AA2162TX`

После повышения порога confidence до 0.25 ложные срабатывания существенно снижены.

## Results

### Detector

AUTO.RIA validation:

| Metric | Value |
|---|---|
| mAP50 | **0.9748** |
| mAP50-95 | **0.8486** |
| Precision | **0.977** |
| Recall | **0.941** |

Combined fine-tune (`AUTO.RIA + parking-angle dataset`):

| Metric | Value |
|---|---|
| mAP50 | **0.9851** |
| mAP50-95 | **0.7057** |
| Precision | **0.9783** |
| Recall | **0.9462** |

### OCR

Synthetic validation:

| Metric | Value |
|---|---|
| CER | **0.034** |
| Exact-match | **84.6%** |

Real AUTO.RIA validation:

| Metric | Value |
|---|---|
| mean CER | **0.237** |
| exact-match | **47.79%** |

Scene-specific fine-tune:

- scene dataset: 3 stable RTSP groups
- corrected scene labels:
  - `AA5104EK`
  - `KA1959BI`
  - `AA2162TX`
- after scene fine-tune these strings stabilize significantly better than before

### Current RTSP Behavior

На чистом RTSP-потоке без внутренних наложений система сейчас выходит на:

- **3 plate detections** в кадре в production-like режиме
- стабильные correct plates:
  - `AA5104EK`
  - `KA1959BI`
  - `AA2162TX`
- дополнительные мелкие plate-like detections существенно снижены порогом `conf=0.25`

### Strengths

- end-to-end система работает на реальной RTSP-камере;
- использованы и classical CV, и deep learning;
- OCR не ограничен одним фиксированным шаблоном;
- есть scene-specific adaptation;
- цвет интерпретируется как plate type, а не как raw RGB code.

### Weaknesses

- дальние номера всё ещё дают OCR-ошибки на 1 символ;
- high-recall mode заметно снижает FPS;
- часть scene labels требует ручной проверки, если используется pseudo-labeling;
- нестандартные и очень маленькие plate crops остаются сложными.

## Conclusions

Проект показывает, что practical ALPR для реальной wide-shot сцены нельзя
решить только одной моделью detector или только OCR.

Лучший результат дал комбинированный подход:

- fine-tuned detector
- sliced inference на high-resolution кадре
- геометрическое выравнивание номера
- восстановление качества plate crop
- CRNN+CTC OCR
- scene-specific fine-tune
- ByteTrack/SORT-style tracking-by-detection: Kalman prediction + IoU/Hungarian
  matching + low-confidence recovery
- temporal OCR stabilization: weighted voting по stable `track_id` + char-wise consensus

Это соответствует задачам курса: применить classical CV и DL-компоненты в
одной рабочей системе и оценить их вклад на реальных данных.

## Future Work

- отдельный OCR fine-tune на дальних plate crops
- beam search decoding вместо greedy CTC
- more foreign/custom plate formats in OCR training
- small super-resolution module for tiny plate crops

## Project Status

Проект находится в активной доработке.

Текущий приоритет:

1. улучшение OCR на дальних номерах;
2. temporal stabilization текста между кадрами;
3. финальная упаковка demo под защиту.
