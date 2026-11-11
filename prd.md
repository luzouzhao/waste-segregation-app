# Product Requirements Document (PRD)

## 1. Project Name
AI-Powered Waste Segregation & Circular Economy System

## 2. Problem Statement
Manual waste sorting is slow, inconsistent, and error-prone. This leads to low recycling rates, higher landfill usage, pollution, and higher waste-processing costs. There is no easy way for citizens/municipal workers to identify a waste item, know how to dispose of it, and find the nearest collection/recycling centre.

## 3. Goal
Build an AI assistant that takes a waste image, classifies the waste type, detects the object in the image, explains recyclability, recommends disposal guidance, and suggests nearby recycling/collection centres — delivered as a Streamlit web app.

## 4. Target Users
- Citizens who want to dispose of waste correctly.
- Municipal/sanitation workers doing bulk sorting.
- Recycling facility staff verifying incoming waste.
- (Secondary) Project evaluators/interns reviewing the pipeline as a learning deliverable.

## 5. Core Waste Categories
Organic, Plastic, Paper, Glass, Metal, Electronic (E-waste), Hazardous.

## 6. Features (MVP scope for 8 days)
| # | Feature | Priority |
|---|---------|----------|
| 1 | Image upload (single image) | Must |
| 2 | Waste classification (7 categories) via DL model | Must |
| 3 | Object detection — bounding box on the waste item | Must |
| 4 | Recyclability verdict (Recyclable / Non-recyclable / Hazardous) | Must |
| 5 | Disposal guidance text (rule-based + LLM-generated explanation) | Must |
| 6 | Nearby collection centre suggestion (mock/static DB, location input) | Should |
| 7 | Confidence score + explainability (Grad-CAM or simple heatmap) | Should |
| 8 | Classical ML model comparison view (accuracy table) | Should |
| 9 | Camera-based live capture | Could (stretch, if time allows) |
| 10 | Waste segregation analytics dashboard (counts by category) | Could |

## 7. Out of Scope (for this 8-day cycle)
- Training custom object detectors from scratch (use pretrained YOLO/pretrained CNN backbones instead).
- Real GPS-based recommendation engine (use a static/mock centre database keyed by city/pincode).
- Multi-user auth, production deployment, mobile app.

## 8. Success Criteria
- User can upload an image and get category + recyclability + disposal advice in under 5 seconds.
- Classification accuracy ≥ 75% on the held-out test split of the chosen dataset.
- End-to-end demo runs without crashing on 10 sample images spanning all 7 categories.

## 9. Non-Functional Requirements
- Runs locally via `streamlit run app.py` with no external paid API required except optionally one LLM API key (kept swappable/optional).
- Must degrade gracefully if LLM API key is missing (fallback to rule-based disposal text).

## 10. Assumptions
- Dataset: a public waste-classification image dataset (e.g., TrashNet-style, 6–7 classes) will be used/downloaded, not collected fresh.
- Collection-centre data is mocked/static for the demo, not a live API.
