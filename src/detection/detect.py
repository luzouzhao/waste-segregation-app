"""
Day 4-5 — YOLOv8 object detection.
Loads pretrained YOLOv8n, runs inference on a waste image,
returns the best detection (largest / most central foreground object),
cropped region, and outputs sample visualizations for sanity checks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO

import config.settings as s


_DETECTION_MODEL: YOLO | None = None


def _get_model() -> YOLO:
    global _DETECTION_MODEL
    if _DETECTION_MODEL is None:
        ckpt = s.PROJECT_ROOT / "yolov8n.pt"
        # 优先用随仓库提交的权重，避免云端运行时联网下载
        _DETECTION_MODEL = YOLO(str(ckpt) if ckpt.exists() else "yolov8n.pt")
    return _DETECTION_MODEL


def _best_detection(
    results, conf_threshold: float, img_shape: tuple[int, int]
) -> tuple[int, int, int, int, float] | None:
    h, w = img_shape
    cx, cy = w / 2, h / 2
    best = None
    best_score = -1.0

    for box in results[0].boxes:
        conf = box.conf.item()
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area = (x2 - x1) * (y2 - y1)
        box_cx = (x1 + x2) / 2
        box_cy = (y1 + y2) / 2
        dist = np.sqrt((box_cx - cx) ** 2 + (box_cy - cy) ** 2)
        score = area / (dist + 1)
        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2, conf)

    return best


def detect_object(
    image_path: str | Path, conf_threshold: float = 0.25,
    det_model: YOLO | None = None,
) -> dict:
    """Run YOLOv8 detection and return the best object's bbox, confidence, and crop.
    
    Args:
        image_path: Path to the input image.
        conf_threshold: Detection confidence threshold.
        det_model: Optional pre-loaded YOLO detection model. If None, loads
                   the default model (may be expensive on each call).
    
    Returns dict with keys:
      bbox (list[int] | None)  — [x1, y1, x2, y2] or None
      confidence (float)       — detection confidence (0.0 if none)
      cropped_image (np.ndarray) — BGR crop region, or full image on fallback
      full_image (np.ndarray)  — original BGR image
    """
    model = det_model if det_model is not None else _get_model()
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = img_bgr.shape[:2]

    results = model(img_bgr, verbose=False)
    det = _best_detection(results, conf_threshold, (h, w))

    if det is not None:
        x1, y1, x2, y2, conf = det
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img_bgr
        bbox = [x1, y1, x2, y2]
    else:
        bbox = None
        conf = 0.0
        crop = img_bgr

    return {
        "bbox": bbox,
        "confidence": conf,
        "cropped_image": crop,
        "full_image": img_bgr,
    }


def save_samples(num_samples: int = 5) -> None:
    """Run detection on sample images from data/raw/ and save annotated outputs."""
    model = _get_model()
    sample_dir = s.PROJECT_ROOT / "reports" / "detection_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_files: list[Path] = []
    for cat_dir in sorted(s.RAW_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        files = sorted(cat_dir.glob("*"))
        if files:
            image_files.append(files[0])
        if len(image_files) >= num_samples:
            break

    for img_path in image_files[:num_samples]:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        results = model(img_bgr, verbose=False)
        det = _best_detection(results, 0.25, img_bgr.shape[:2])
        vis = img_bgr.copy()
        if det is not None:
            x1, y1, x2, y2, conf = det
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{conf:.2f}"
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out_path = sample_dir / f"detect_{img_path.parent.name}_{img_path.name}"
        cv2.imwrite(str(out_path), vis)

    print(f"Sample detection outputs saved to {sample_dir}")


if __name__ == "__main__":
    save_samples()
