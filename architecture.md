# Architecture

## 1. Tech Stack
- **Frontend/App:** Streamlit (multipage)
- **Language:** Python 3.10+
- **Classical ML:** scikit-learn, XGBoost
- **Deep Learning:** PyTorch or TensorFlow/Keras (pick ONE, do not mix) — recommend PyTorch + `timm` for MobileNetV2/EfficientNet
- **Object Detection:** Ultralytics YOLOv8 (pretrained, fine-tuned if time allows)
- **Computer Vision utils:** OpenCV, Pillow
- **LLM layer:** any single provider via API (OpenAI/Anthropic) with LangChain optional — must be optional/swappable
- **Storage:** SQLite (local) for metadata/disposal history; no external DB needed
- **Data handling:** pandas, numpy
- **Explainability:** Grad-CAM (`pytorch-grad-cam`) or simple saliency
- **Packaging:** requirements.txt, no Docker required for 8-day scope

## 2. High-Level Flow
```
Image Upload/Camera
      ↓
Preprocessing (resize, normalize)
      ↓
Object Detection (YOLO) → bounding box + crop
      ↓
Classification (DL model) → waste category + confidence
      ↓
Recyclability Rule Engine → recyclable / non-recyclable / hazardous
      ↓
LLM Prompt (category + rules) → disposal guidance text
      ↓
Collection Centre Lookup (static DB by category + city)
      ↓
Streamlit Dashboard (results + analytics)
```

## 3. Folder & File Structure
```
waste-segregation-app/
├── app.py                        # Streamlit entry point
├── prd.md / architecture.md / rules.md / phases.md / design.md
├── requirements.txt
├── .env.example                  # LLM API key placeholder
├── config/
│   └── settings.py               # paths, category labels, thresholds
├── data/
│   ├── raw/                      # original downloaded dataset
│   ├── processed/                # resized/augmented dataset
│   └── sample_images/            # demo images for testing
├── models/
│   ├── ml/                       # saved sklearn/xgboost models (.pkl)
│   ├── dl/                       # saved CNN/MobileNet/EfficientNet weights
│   └── detection/                # YOLO weights
├── src/
│   ├── preprocessing/            # resize, augment, clean scripts
│   ├── classification/           # ML + DL training & inference code
│   ├── detection/                # YOLO inference wrapper
│   ├── recommendation/           # recyclability rules + centre lookup
│   ├── assistant/                # LLM prompt templates + call wrapper
│   └── utils/                    # shared helpers (image io, metrics, logging)
├── pages/                        # Streamlit multipage UI
│   ├── 1_Upload_&_Classify.py
│   ├── 2_Model_Comparison.py
│   ├── 3_Collection_Centres.py
│   └── 4_Analytics.py
├── notebooks/                    # EDA + experiment notebooks
├── reports/                      # evaluation report, metrics output
├── assets/
│   ├── images/                   # UI static images/logos
│   └── icons/                    # category icons
├── tests/                        # unit tests
└── docs/                         # extra documentation, demo video link
```

## 4. Module → Folder Mapping
- Module 1–2 (Data Collection/Preprocessing) → `data/`, `src/preprocessing/`
- Module 3 (Classical ML) → `src/classification/` (ml part), `models/ml/`
- Module 4–5 (DL & CV) → `src/classification/` (dl part), `src/detection/`, `models/dl/`, `models/detection/`
- Module 6 (Circular Economy Recommendation) → `src/recommendation/`
- Module 7 (LLM Assistant) → `src/assistant/`
- Module 8 (Evaluation) → `reports/`, `tests/`
- Module 9 (UI) → `app.py`, `pages/`

## 5. Data Flow Contracts (keep interfaces simple)
- `preprocessing.preprocess(image) -> np.ndarray`
- `detection.detect(image) -> {bbox, cropped_image}`
- `classification.predict(cropped_image) -> {label, confidence}`
- `recommendation.get_guidance(label) -> {recyclable: bool, rule_text: str}`
- `assistant.explain(label, rule_text) -> str` (LLM call, fallback to `rule_text` if no API key)
- `recommendation.nearby_centres(label, city) -> list[dict]`

## 6. Session State (Streamlit)
Keep model objects (DL model, YOLO model) loaded once via `st.cache_resource`. Keep dataset/results via `st.session_state`, not global variables.
