# Project Memory — AI Waste Segregation & Circular Economy System

## Project Overview

Streamlit app that classifies waste images into 7 categories, detects objects, recommends disposal guidance, and suggests nearby recycling centres. Built with Python, PyTorch, YOLOv8, scikit-learn/XGBoost, and Streamlit.

**7 Target Categories:** organic, plastic, paper, glass, metal, electronic, hazardous

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit (multipage) |
| DL | PyTorch + torchvision (classification, transfer learning) |
| Detection | Ultralytics YOLOv8 (pretrained) |
| Classical ML | scikit-learn, XGBoost |
| Image ops | OpenCV, Pillow, numpy |
| Metadata | SQLite |
| LLM (optional) | Groq API with fallback to rule-based text |

---

## Project Documents

| File | Purpose |
|------|---------|
| `prd.md` | Product requirements — features, success criteria, out-of-scope |
| `architecture.md` | Tech stack, high-level flow, folder structure, data contracts |
| `design.md` | Streamlit theme, color palette, layout principles, UI copy tone |
| `phases.md` | 8-day compressed build plan |
| `rules.md` | Coding conventions, error handling, library boundaries |

---

## Build Summary

### Day 1 — Setup + Data Collection
- Downloaded & merged 2 Kaggle datasets (23,172 images across 7 categories)
- Dataset sources: `salam035/waste-datasets`, `akshat103/e-waste-image-dataset`
- Files: `config/settings.py`, `src/preprocessing/download_datasets.py`

### Day 2 — Preprocessing
- Validated images (0 corrupt, 0 near-duplicates found)
- Created SQLite metadata DB (`waste_metadata.db`)
- Resized to 256x256, applied augmentation (flip, rotate, brightness), stratified train/val/test split
- Final split: 48,642 train / 3,479 val / 3,479 test (55,600 total with augmented copies)
- Files: `validate.py`, `metadata.py`, `preprocess.py`

### Day 3 — Model Training (Classification)
- **ML baseline:** RGB histogram + Logistic Regression, Decision Tree, Random Forest, XGBoost — best: XGBoost (53.35%)
- **DL models:** EfficientNet-B0 (85.94%), MobileNetV2 (84.88%), CNN from scratch (73.73%)
- **YOLOv8n-cls:** 92.9% test accuracy, 9.6 ms/image — selected as primary classifier
- Files: `ml_baseline.py`, `dl_train.py`, `dl_train_cnn.py`, `dl_train_mobilenet.py`, `dl_compare.py`, `dl_evaluate.py`

### Day 4-5 — Object Detection
- YOLOv8n, Faster R-CNN, Mask R-CNN (all COCO-pretrained)
- YOLOv8n selected for production (fastest, adequate quality)
- Detection + classification pipeline in `pipeline.py`
- Files: `detect.py`, `detect_frcnn.py`, `detect_maskrcnn.py`, `detector_compare.py`, `pipeline.py`

### Day 6 — Recommendation + LLM Assistant
- Static guidance rules for all 7 categories (disposal method, hazard level, recyclability)
- 7 demo collection centres across Indian cities
- Groq LLM integration (optional, falls back to rule-based text)
- Files: `rules.py`, `centres.py`, `llm_guidance.py`

### Day 7 — Streamlit UI
- 4-page multipage app: Home, Upload & Classify, Model Comparison, Analytics
- SQLite disposal history logging
- Animated SVG confidence gauge, PDF report download, camera input, interactive Plotly charts
- Files: `app.py`, `1_Upload_and_Classify.py`, `2_Model_Comparison.py`, `3_Analytics.py`, `history_db.py`

### Day 8 — Evaluation
- Confusion matrix heatmaps (ML + DL) — Plotly interactive
- Grad-CAM explainability for DL, feature importance for ML
- LLM response quality evaluation (5 test cases, pass/fail)
- Files: `confusion_matrix_viz.py`, `xai.py`, `llm_eval.py`

---

## Model Performance Summary

| Rank | Model | Test Accuracy | Type |
|------|-------|---------------|------|
| 1 | YOLOv8n-cls | **92.9%** | YOLO classification |
| 2 | EfficientNet-B0 | 85.94% | Transfer learning |
| 3 | MobileNetV2 | 84.88% | Transfer learning |
| 4 | CNN from scratch | 73.73% | From scratch |
| 5 | XGBoost | 53.35% | Classical ML |

### Detector Comparison

| Detector | Detection Rate | Avg Confidence | Avg Time (ms) |
|----------|---------------|----------------|---------------|
| YOLOv8n | 0.86 | 0.453 | 31.4 |
| Faster R-CNN | 1.00 | 0.446 | 402.4 |
| Mask R-CNN | 1.00 | 0.540 | 529.6 |

---

## Key Design Decisions

1. **YOLOv8n-cls as primary classifier** — 92.9% accuracy, 9-10x faster than torchvision models
2. **Class weighting over manual balancing** — mild 2.2:1 imbalance handled via PyTorch `CrossEntropyLoss(weight=...)`
3. **LLM is optional+swappable** — Groq call wrapped in try/except; missing key falls back to rule-based text
4. **Detection + Classification** — YOLO detects object, separate classifier categorizes waste type
5. **Low-confidence threshold (60%)** — shows "Uncertain" warning if confidence below threshold
6. **SQLite for history** — lightweight local storage, no external database required
7. **Detection quality accepted as-is** — all detectors are COCO-pretrained, not waste-fine-tuned
8. **Pre-saved augmentation** — augmented copies generated at preprocessing time, not on-the-fly

---

## Key Commands

```bash
# Run scripts directly (no PYTHONPATH needed)
python3 src/preprocessing/validate.py
python3 src/preprocessing/metadata.py
python3 src/preprocessing/preprocess.py
python3 src/classification/ml_baseline.py
python3 src/classification/dl_train.py
python3 src/classification/dl_evaluate.py
python3 src/classification/dl_train_cnn.py
python3 src/classification/dl_train_mobilenet.py
python3 src/classification/dl_compare.py
python3 src/detection/detect.py
python3 src/detection/detector_compare.py
python3 src/evaluation/llm_eval.py
python3 src/evaluation/confusion_matrix_viz.py
python3 src/evaluation/xai.py

# Pipeline test
python3 -c "from src.detection.pipeline import classify_waste_image; import glob; r = classify_waste_image(glob.glob('data/raw/plastic/*')[0]); print(r['category'], r['confidence'])"

# Run the app
streamlit run app.py

# Module quick tests
python3 -c "from src.recommendation.rules import get_guidance; print(get_guidance('plastic'))"
python3 -c "from src.recommendation.centres import nearby_centres; print(nearby_centres('plastic'))"
python3 -c "from src.assistant.llm_guidance import generate_explanation; print(generate_explanation('plastic', {'recyclable': True, 'disposal_method': 'Rinse and place in recycling bin.', 'hazard_level': 'low'}))"
```

---

## Repository Structure

```
waste-segregation-app/
├── memory.md                       # Project memory & progress
├── prd.md                          # Product requirements
├── architecture.md                 # Architecture & tech stack
├── design.md                       # UI design guide
├── phases.md                       # Build plan
├── rules.md                        # Coding rules & boundaries
├── app.py                          # Streamlit entry point
├── config/
│   └── settings.py                 # Paths, categories, image size, split ratios
├── data/
│   ├── raw/                        # Original dataset (7 category folders)
│   ├── processed/                  # Resized + augmented splits
│   └── waste_metadata.db           # SQLite metadata
├── models/
│   ├── ml/                         # XGBoost model (.pkl)
│   └── dl/                         # DL model weights (.pt)
├── src/
│   ├── preprocessing/              # Download, validate, metadata, preprocess
│   ├── classification/             # ML & DL training and inference
│   ├── detection/                  # YOLOv8 / Faster R-CNN / Mask R-CNN
│   ├── recommendation/             # Disposal rules + collection centres
│   ├── assistant/                  # LLM guidance with fallback
│   ├── evaluation/                 # Confusion matrix, Grad-CAM, LLM eval
│   └── utils/                      # SQLite history DB
├── pages/
│   ├── 1_Upload_and_Classify.py    # Main classification page
│   ├── 2_Model_Comparison.py       # Model performance comparison
│   └── 3_Analytics.py              # Disposal history dashboard
├── runs/                           # YOLO training outputs
├── reports/                        # Evaluation artifacts & sample outputs
└── docs/                           # Presentation & project PDF
```

---

## Dataset Summary

| Category | Count |
|----------|-------|
| organic | 4,114 |
| plastic | 3,436 |
| paper | 4,515 |
| glass | 2,011 |
| metal | 2,039 |
| electronic | 3,000 |
| hazardous | 4,057 |
| **Total** | **23,172** |

Stratified split: train 70% / val 15% / test 15%.

---

## Disclaimer

This is a demo system. Always verify disposal guidance with your local municipal waste management authority.
