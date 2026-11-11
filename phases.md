# Phases (8-Day Plan)

Note: 9 project modules are compressed into 8 days by using pretrained backbones for detection/classification instead of training from scratch, and by treating collection-centre data as static/mock. Revisit this trade-off if timeline extends.

## Day 1 — Setup + Data Collection (Module 1)
- Initialize repo structure (this zip), venv, requirements.txt.
- Download/select a public waste-classification dataset (6–7 categories).
- Build `data/raw` ingestion + SQLite table for metadata (filename, category, upload date).
- Deliverable: raw dataset in `data/raw`, metadata schema working.

## Day 2 — Preprocessing (Module 2)
- Resize, normalize, augment (flip/rotate/brightness), remove corrupt/noisy images.
- Split into train/val/test.
- Deliverable: clean dataset in `data/processed`, documented in a notebook.

## Day 3 — Classical ML Baseline (Module 3)
- Extract simple features (color histograms/HOG) or flattened pixels.
- Train Logistic Regression, Decision Tree, Random Forest, XGBoost.
- Compare Accuracy/Precision/Recall/F1.
- Deliverable: `models/ml/*.pkl` + comparison table in `reports/`.

## Day 4 — Deep Learning Classification (Module 4)
- Fine-tune a pretrained MobileNetV2 or EfficientNet on the processed dataset (transfer learning, not from scratch).
- Optionally train a small CNN baseline for comparison.
- Deliverable: best DL model saved to `models/dl/`, accuracy ≥ 75% target.

## Day 5 — Object Detection / Computer Vision (Module 5)
- Integrate pretrained YOLOv8 for waste object detection/bounding box.
- Wire detection → crop → classification pipeline.
- Deliverable: `src/detection/` working end-to-end with sample images.

## Day 6 — Recommendation + LLM Assistant (Modules 6 & 7)
- Build rule-based recyclability engine (`src/recommendation/`).
- Build static collection-centre lookup table.
- Add LLM call for disposal guidance text generation, with rule-based fallback.
- Deliverable: full pipeline from image → category → guidance → centre suggestion.

## Day 7 — Streamlit UI (Module 9)
- Build multipage app: Upload & Classify, Model Comparison, Collection Centres, Analytics.
- Wire all backend modules into the UI, add error handling per `rules.md`.
- Deliverable: working `streamlit run app.py` demo.

## Day 8 — Evaluation, Polish, Documentation (Module 8 + Deliverables)
- Compute final ML/CV metrics (Accuracy, F1, mAP/IoU if detection was fine-tuned, confusion matrix).
- Write evaluation report (`reports/evaluation_report.md`).
- Record demo video, finalize README, package source code zip.
- Deliverable: final submission — source code, trained models, report, demo video.

## Buffer Note
Days are tight with zero slack. If any day overruns, cut Day 5 fine-tuning (keep YOLO fully pretrained, no fine-tune) or drop the "Camera-based live capture" / "Analytics dashboard" Could-have items from `prd.md` first.
