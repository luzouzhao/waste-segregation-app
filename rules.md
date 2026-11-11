# Rules & Boundaries

## 1. Libraries — Use
- Streamlit for all UI (no Flask/FastAPI wrapper needed for this scope).
- PyTorch + `timm` for DL models (or TensorFlow — pick one on day 1 and never mix frameworks in the same repo).
- Ultralytics YOLOv8 for detection (pretrained weights; fine-tune only if Day 5+ has slack).
- scikit-learn + XGBoost for classical ML comparison.
- OpenCV/Pillow for image ops.
- SQLite for any persistence (no Postgres/Mongo — adds ops overhead with no benefit at this scale).
- `python-dotenv` for API key management.

## 2. Libraries — Avoid
- No Docker, Kubernetes, or cloud deployment tooling — out of scope for 8 days.
- No Flask/Django alongside Streamlit — one framework only.
- No LangChain agents/tool-calling chains — a single direct LLM API call with a fixed prompt template is enough; agent frameworks add debugging overhead with no scope benefit here.
- No custom-built object detection architecture from scratch — always start from pretrained weights.
- No paid/proprietary datasets — use public waste-classification datasets only.
- No `pickle` for anything except trained sklearn models (never unpickle untrusted files).

## 3. Error Handling
- Every model inference call must be wrapped in try/except with a user-facing Streamlit error message (`st.error`), never a raw traceback shown to the user.
- If the LLM API key is missing or the call fails, fall back to the rule-based disposal text — the app must never hard-crash on missing API keys.
- Validate uploaded file type/size before processing (reject non-image files, cap size e.g. 10MB).
- Log errors to a local file (`logs/app.log`), don't just print to console.

## 4. AI/Model Boundaries
- Do not claim medical/safety-critical accuracy — hazardous waste guidance must include a disclaimer: "verify with local municipal rules."
- Classification confidence below a threshold (e.g. 60%) must be shown as "uncertain" in the UI, not presented as a confident answer.
- LLM output is for explanation/guidance text only — it must never be the source of the classification label (label always comes from the CV/ML model, not the LLM).
- Do not fabricate collection-centre data as if it were live/verified — clearly label it as a static/demo directory unless a real API is integrated.

## 5. Coding Conventions
- One responsibility per file inside `src/` (no god-files).
- All model paths, thresholds, category labels centralized in `config/settings.py` — no hardcoded magic strings scattered across files.
- Functions get type hints and docstrings; no bare `except:` blocks.
- Notebooks are for experimentation only — final logic must be ported into `src/`, not left only in a notebook.

## 6. Scope Discipline
- If a feature isn't in `prd.md` Must/Should list, it does not get built before Day 7.
- Any new dependency added mid-project must be added to `requirements.txt` immediately, not left implicit.
