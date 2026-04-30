# Changelog

## [1.1.0] — 2026-04-20

### Added
- `scripts/multi_rtsp_fast.py` — no-queue multi-RTSP runner with per-stream reader/processor threads, auto-tune support
- `src/source_manager.py` — source abstraction: RTSPSource, VideoSource, WebcamSource, create_source() factory
- `PRESENTATION_5MIN.md` — 5-minute diploma presentation script
- `demo_artifacts/DEMO_CHECKLIST.md` — demo run checklist
- `demo_artifacts/rtsp_demo_output.txt` — latest RTSP run output
- `scripts/capture_crop_examples.py`, `scripts/capture_demo_screenshots.py` — screenshot capture utilities
- `scripts/diagnose_detections.py`, `scripts/diagnose_ocr_errors.py` — diagnostics scripts
- `scripts/process_plates.py` — plate batch processor

### Changed
- `README.md` — updated with conf=0.25 examples, multi-RTSP quick start, supported sources section, corrected plate numbers (AA5104EK, KA1959BI, AA2162TX)
- `demo_artifacts/README.md` — synced to latest RTSP run results
- `demo_artifacts/results/rtsp_demo_output.txt` — updated to latest actual run (conf=0.25, 3 plates)
- `demo_artifacts/results/train_log.txt` — updated training log
- `demo_artifacts/screenshots/` — updated demo screenshots
- `src/pipeline.py` — pipeline improvements

## [1.0.0] — initial

- ALPR diploma project baseline: YOLO26n detector + CRNN+CTC OCR, Streamlit UI, RTSP support
