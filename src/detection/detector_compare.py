"""
Day 5 (Extended) — Multi-detector comparison.
Runs YOLOv8n, Faster R-CNN, and Mask R-CNN on sample images,
reports detection rate, average confidence, and pseudo-GT mAP@0.5.
All detectors are COCO-pretrained, NOT waste-fine-tuned.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou

import config.settings as s
from src.detection import detect as yolo_detector
from src.detection import detect_frcnn as frcnn_detector
from src.detection import detect_maskrcnn as maskrcnn_detector


def _load_sample_images(num_per_category: int = 1) -> list[Path]:
    images: list[Path] = []
    for cat_dir in sorted(s.RAW_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        files = sorted(cat_dir.glob("*"))
        for f in files[:num_per_category]:
            images.append(f)
    return images


def _compute_map(
    pred_boxes: list,
    pred_scores: list,
    gt_boxes: list,
    iou_threshold: float = 0.5,
) -> float:
    if not pred_boxes or not gt_boxes:
        return 0.0
    pred_t = torch.tensor(pred_boxes, dtype=torch.float)
    gt_t = torch.tensor(gt_boxes, dtype=torch.float)
    ious = box_iou(pred_t, gt_t)
    max_ious = ious.max(dim=1).values
    tp = (max_ious >= iou_threshold).sum().item()
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _run_detector(detector_module, image_paths: list[Path], conf_thresh: float = 0.25) -> dict:
    detections = []
    total_time = 0.0
    for p in image_paths:
        start = time.perf_counter()
        result = detector_module.detect_object(str(p), conf_threshold=conf_thresh)
        elapsed = time.perf_counter() - start
        total_time += elapsed
        detections.append(result)
    return {
        "detections": detections,
        "total_time": total_time,
    }


def main() -> None:
    image_paths = _load_sample_images(num_per_category=1)
    print(f"Loaded {len(image_paths)} sample images (one per category)")

    conf_thresh = 0.25
    detectors = [
        ("YOLOv8n", yolo_detector),
        ("Faster R-CNN", frcnn_detector),
        ("Mask R-CNN", maskrcnn_detector),
    ]

    results = {}
    for name, module in detectors:
        print(f"  Running {name}...")
        data = _run_detector(module, image_paths, conf_thresh)
        dets = data["detections"]
        detections_fired = sum(1 for d in dets if d["bbox"] is not None)
        detection_rate = detections_fired / len(dets)
        valid_confs = [d["confidence"] for d in dets if d["confidence"] > 0]
        avg_conf = sum(valid_confs) / len(valid_confs) if valid_confs else 0.0
        avg_time = data["total_time"] / len(dets)
        results[name] = {
            "detections": dets,
            "detection_rate": detection_rate,
            "avg_confidence": avg_conf,
            "avg_time_ms": avg_time * 1000,
        }
        print(f"    Detection rate: {detection_rate:.2f}  Avg conf: {avg_conf:.3f}  Avg time: {avg_time*1000:.1f} ms")

    yolo_dets = results["YOLOv8n"]["detections"]
    yolo_gt_boxes = []
    for d in yolo_dets:
        if d["bbox"] is not None and d["confidence"] >= 0.5:
            yolo_gt_boxes.append(d["bbox"])

    report_data = []
    for name in ["Faster R-CNN", "Mask R-CNN"]:
        r = results[name]
        pred_boxes = [d["bbox"] for d in r["detections"] if d["bbox"] is not None]
        pred_scores = [d["confidence"] for d in r["detections"] if d["confidence"] > 0]
        m = _compute_map(pred_boxes, pred_scores, yolo_gt_boxes)
        m_str = f"{m:.4f}" if yolo_gt_boxes else "N/A (no pseudo-GT)"
        report_data.append((name, r["detection_rate"], r["avg_confidence"], r["avg_time_ms"], m_str))

    report_dir = Path(__file__).resolve().parent.parent.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "detector_comparison.md"

    lines = ["# Detector Comparison Report\n\n"]
    lines.append(f"**Sample images:** {len(image_paths)} (one per category)\n")
    lines.append(f"**Confidence threshold:** {conf_thresh}\n")
    lines.append(f"**Pseudo ground truth:** YOLOv8n detections at confidence ≥ 0.5\n")
    lines.append("*All detectors are COCO-pretrained, NOT waste-fine-tuned.*\n\n")
    lines.append("## Summary\n\n")
    lines.append("| Detector | Detection Rate | Avg Confidence | Avg Time (ms) | mAP@0.5 (pseudo-GT) |\n")
    lines.append("|----------|---------------|----------------|---------------|---------------------|\n")

    for name, dr, ac, t, m in [
        ("YOLOv8n", results["YOLOv8n"]["detection_rate"], results["YOLOv8n"]["avg_confidence"], results["YOLOv8n"]["avg_time_ms"], "— (pseudo-GT source)"),
    ] + report_data:
        lines.append(f"| {name} | {dr:.2f} | {ac:.3f} | {t:.1f} | {m} |\n")

    lines.append("\n## Per-Image Breakdown\n\n")
    lines.append("| Image | Category | YOLO bbox | FRCNN bbox | MRCNN bbox |\n")
    lines.append("|-------|----------|-----------|------------|------------|\n")
    for i, p in enumerate(image_paths):
        cat = p.parent.name
        yb = "yes" if results["YOLOv8n"]["detections"][i]["bbox"] is not None else "no"
        fb = "yes" if results["Faster R-CNN"]["detections"][i]["bbox"] is not None else "no"
        mb = "yes" if results["Mask R-CNN"]["detections"][i]["bbox"] is not None else "no"
        lines.append(f"| {p.name} | {cat} | {yb} | {fb} | {mb} |\n")

    report_path.write_text("".join(lines))
    print(f"\nDetector comparison report written to {report_path}")


if __name__ == "__main__":
    main()
