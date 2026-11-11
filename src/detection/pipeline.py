"""
图像分类推理管线。

分类阶段：整图直接送入 YOLOv8n-cls，并对原图 / 水平翻转 / 1.15 倍 / 1.3 倍中心缩放
四个视图的概率取平均（TTA）。

注意：
1. 不再用 YOLOv8 检测框裁剪后再分类。COCO 预训练检测器只认识杯子、瓶子等常见物体，
   裁出的常是局部区域，实测会把准确率从 ~95% 打到 ~53%（玻璃最容易被判成电子/有害垃圾）。
   检测框仅用于前端可视化。
2. Ultralytics 的 numpy 输入按 BGR 处理，必须直接传 OpenCV 读到的 BGR 图，不能再转 RGB。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO

import config.settings as s
from src.detection.detect import detect_object

_CLASSIFIER: YOLO | None = None
_CLASS_NAMES: list[str] = sorted(s.TARGET_CATEGORIES)

_TTA_SCALES = (1.0, 1.15, 1.3)


def load_classifier() -> YOLO:
    global _CLASSIFIER
    if _CLASSIFIER is not None:
        return _CLASSIFIER
    ckpt = s.PROJECT_ROOT / "runs" / "classify" / "train-2" / "weights" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"YOLOv8n-cls checkpoint not found at {ckpt}. "
            "Run: yolo classify train data=data/processed model=yolov8n-cls.pt epochs=15 imgsz=256"
        )
    _CLASSIFIER = YOLO(str(ckpt))
    return _CLASSIFIER


def _center_zoom(img: np.ndarray, scale: float) -> np.ndarray:
    """按比例裁剪中心区域，scale 越大裁得越小（相当于放大）。"""
    h, w = img.shape[:2]
    nh, nw = int(h / scale), int(w / scale)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    return img[y0:y0 + nh, x0:x0 + nw]


def _yolo_probs(img_bgr: np.ndarray, model: YOLO) -> np.ndarray:
    probs = model(img_bgr, verbose=False)[0].probs
    return np.array([probs.data[i].item() for i in range(len(_CLASS_NAMES))], dtype=np.float32)


def _classify_image(
    img_bgr: np.ndarray, classifier: YOLO | None = None, topk: int = 3
) -> tuple[str, float, list[tuple[str, float]]]:
    model = classifier if classifier is not None else load_classifier()
    views = [img_bgr] + [cv2.flip(img_bgr, 1)] + [
        _center_zoom(img_bgr, sc) for sc in _TTA_SCALES if sc > 1.0
    ]
    probs = np.mean([_yolo_probs(v, model) for v in views], axis=0)

    idx = int(probs.argmax())
    order = np.argsort(probs)[::-1][:topk]
    top3 = [(_CLASS_NAMES[i], float(probs[i])) for i in order]
    return _CLASS_NAMES[idx], float(probs[idx]), top3


def classify_waste_image(
    image_path: str | Path,
    det_conf_threshold: float = 0.25,
    classifier: YOLO | None = None,
    det_model: YOLO | None = None,
) -> dict:
    """Detect and classify a waste image end-to-end using YOLOv8n-cls.

    Args:
        image_path: Path to the input image.
        det_conf_threshold: Detection confidence threshold (仅用于可视化框).
        classifier: Optional pre-loaded YOLO classifier. If None, loads
                    the default classifier (which may be expensive on each call).
        det_model: Optional pre-loaded YOLO detection model. If None, loads
                   the default detector.

    Returns dict with keys:
      category (str)           — predicted waste category
      confidence (float)       — softmax confidence
      top3 (list[tuple])       — top-3 (category, prob)
      bbox (list[int] | None)  — detection bounding box [x1, y1, x2, y2]（仅供参考，不参与分类）
      detection_confidence (float) — YOLOv8 detection confidence
    """
    det_result = detect_object(image_path, conf_threshold=det_conf_threshold, det_model=det_model)

    # 分类使用完整原图，检测裁剪仅用于展示
    category, confidence, top3 = _classify_image(
        det_result["full_image"], classifier=classifier
    )

    return {
        "category": category,
        "confidence": confidence,
        "top3": top3,
        "bbox": det_result["bbox"],
        "detection_confidence": det_result["confidence"],
    }
