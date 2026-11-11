"""
Day 5 (Extended) — Mask R-CNN object detection + segmentation.
Loads pretrained Mask R-CNN (torchvision, COCO weights), runs inference,
returns best detection with same interface as detect.py plus a binary mask.
COCO-pretrained, NOT waste-fine-tuned — detection quality will vary.
"""

import ssl
import sys
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

import config.settings as s

_DETECTION_MODEL: torch.nn.Module | None = None
_DEVICE: torch.device | None = None


def _get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        if torch.cuda.is_available():
            _DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available():
            _DEVICE = torch.device("mps")
        else:
            _DEVICE = torch.device("cpu")
    return _DEVICE


def _get_model() -> torch.nn.Module:
    global _DETECTION_MODEL
    if _DETECTION_MODEL is None:
        weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        _DETECTION_MODEL = maskrcnn_resnet50_fpn(weights=weights)
        _DETECTION_MODEL.eval()
        _DETECTION_MODEL.to(_get_device())
    return _DETECTION_MODEL


def _best_detection(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    masks: torch.Tensor,
    conf_threshold: float,
    img_shape: tuple[int, int],
) -> tuple[int, int, int, int, float, np.ndarray | None] | None:
    h, w = img_shape
    cx, cy = w / 2, h / 2
    best = None
    best_score = -1.0

    for i in range(len(boxes)):
        conf = scores[i].item()
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, boxes[i].tolist())
        area = (x2 - x1) * (y2 - y1)
        box_cx = (x1 + x2) / 2
        box_cy = (y1 + y2) / 2
        dist = np.sqrt((box_cx - cx) ** 2 + (box_cy - cy) ** 2)
        score = area / (dist + 1)
        if score > best_score:
            best_score = score
            mask_np = masks[i, 0].cpu().numpy() if masks is not None else None
            mask_bin = (mask_np > 0.5).astype(np.uint8) * 255 if mask_np is not None else None
            best = (x1, y1, x2, y2, conf, mask_bin)

    return best


@torch.no_grad()
def detect_object(
    image_path: str | Path, conf_threshold: float = 0.25
) -> dict:
    """Run Mask R-CNN detection and return the best object's bbox, confidence, crop, and mask.

    Returns dict with keys:
      bbox (list[int] | None)  — [x1, y1, x2, y2] or None
      confidence (float)       — detection confidence (0.0 if none)
      cropped_image (np.ndarray) — BGR crop region, or full image on fallback
      full_image (np.ndarray)  — original BGR image
      mask (np.ndarray | None) — binary segmentation mask (HxW uint8) or None
    """
    model = _get_model()
    device = _get_device()

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = img_bgr.shape[:2]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255).to(device)
    img_tensor = img_tensor.unsqueeze(0)

    output = model(img_tensor)[0]
    boxes = output["boxes"]
    scores = output["scores"]
    masks = output.get("masks")

    det = _best_detection(boxes, scores, masks, conf_threshold, (h, w))

    if det is not None:
        x1, y1, x2, y2, conf, mask = det
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img_bgr
        bbox = [x1, y1, x2, y2]
    else:
        bbox = None
        conf = 0.0
        crop = img_bgr
        mask = None

    return {
        "bbox": bbox,
        "confidence": conf,
        "cropped_image": crop,
        "full_image": img_bgr,
        "mask": mask,
    }


@torch.no_grad()
def save_samples(num_samples: int = 5) -> None:
    """Run detection on sample images from data/raw/ and save annotated outputs with mask overlay."""
    model = _get_model()
    device = _get_device()
    sample_dir = s.PROJECT_ROOT / "reports" / "detection_samples_maskrcnn"
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
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255).to(device)
        img_tensor = img_tensor.unsqueeze(0)
        output = model(img_tensor)[0]
        det = _best_detection(
            output["boxes"], output["scores"], output.get("masks"), 0.25, (h, w),
        )
        vis = img_bgr.copy()
        if det is not None:
            x1, y1, x2, y2, conf, mask = det
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{conf:.2f}"
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if mask is not None:
                mask_resized = cv2.resize(mask, (w, h))
                overlay = vis.copy()
                overlay[mask_resized > 0] = (0, 255, 0)
                vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        out_path = sample_dir / f"maskrcnn_{img_path.parent.name}_{img_path.name}"
        cv2.imwrite(str(out_path), vis)

    print(f"Mask R-CNN sample outputs saved to {sample_dir}")


if __name__ == "__main__":
    save_samples()
