# Demo Artifacts

Эта папка содержит готовые материалы для демонстрации результата проекта:

- скриншоты работы системы;
- численные результаты обучения;
- примеры реального RTSP-запуска;
- логи, которые можно приложить к отчёту или презентации.

## Structure

### `screenshots/`

- `01_rtsp_clean.jpg`
  - чистый кадр RTSP без внутренних отрисовок
- `02_rtsp_merged.jpg`
  - результат работы пайплайна на RTSP-кадре
- `03_crop_raw.png`
  - пример исходного crop номерного знака
- `04_crop_preprocessed.png`
  - тот же crop после выравнивания и улучшения качества
- `05_detector_training_curves.png`
  - графики обучения детектора
- `06_detector_val_pred.jpg`
  - пример предсказаний детектора на validation batch

### `results/`

- `crnn_summary.json`
  - сводка OCR-оценки
- `detector_results.csv`
  - метрики обучения/fine-tune детектора
- `train_log.txt`
  - лог обучения и экспериментов
- `rtsp_demo_output.txt`
  - реальный пример вывода CLI при прогоне RTSP

## Recommended Demo Order

Для защиты удобно показывать материалы в таком порядке:

1. `screenshots/01_rtsp_clean.jpg`
   - показать исходную сложную сцену
2. `screenshots/02_rtsp_merged.jpg`
   - показать итоговые detections
3. `screenshots/03_crop_raw.png` → `04_crop_preprocessed.png`
   - показать эффект выравнивания и восстановления качества
4. `results/rtsp_demo_output.txt`
   - показать реальный текстовый результат запуска
5. `screenshots/05_detector_training_curves.png`
   - показать, что обучение действительно проводилось
6. `results/detector_results.csv` и `results/crnn_summary.json`
   - показать метрики

## Key Real Result

Актуальный пример RTSP-вывода:

```text
[rtsp] resolution=2592x1520  declared_fps=50.0
[alpr] OCR backend = crnn  imgsz=1536  conf=0.08  weights=models/yolo26_plate_combined.pt  classical=cpu
  frame   0  plates=5  [KA9537EC [белый], AI6524OA [белый], AA5104EK [белый], KA1959BI [белый], A504E [белый]]
  frame   1  plates=5  [KA9537EC [белый], AI6524OA [белый], AA5104EK [белый], KA1959BI [белый], A504E [белый]]
```

Стабильно читаемые номера в текущей сцене:

- `KA9537EC`
- `AI6524OA`
- `AA5104EK`
- `KA1959BI`

Пятый дальний номер пока остаётся самым сложным для OCR.
