"""
Day 3 (Extended) — Multi-architecture DL comparison.
Loads CNN, MobileNetV2, and EfficientNet-B0 checkpoints, evaluates each
on the same test set, and writes a comparison report to
reports/dl_model_comparison.md with accuracy, per-class F1, and inference time.
"""

import ssl
import sys
import time
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0, mobilenet_v2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config.settings as s
from src.classification.dl_train_cnn import SimpleCNN
from src.preprocessing.preprocess import get_eval_transforms


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.tolist())
    return all_preds, all_labels


@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 10,
) -> float:
    model.eval()
    it = iter(loader)
    inputs, _ = next(it)
    inputs = inputs.to(device)
    model(inputs)

    count = 0
    start = time.perf_counter()
    for batch_idx, (inputs, _) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        inputs = inputs.to(device)
        model(inputs)
        count += inputs.size(0)
    elapsed = time.perf_counter() - start
    return (elapsed / count) * 1000


def _build_cnn(num_classes: int) -> nn.Module:
    return SimpleCNN(num_classes=num_classes)


def _build_mobilenet(num_classes: int) -> nn.Module:
    model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_efficientnet(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def _load_checkpoint(model: nn.Module, path: Path, tag: str) -> nn.Module:
    if not path.exists():
        print(f"[{tag}] Checkpoint not found at {path}. Skipping.")
        return None
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    print(f"[{tag}] Loaded checkpoint from {path}")
    return model


_MODEL_SPECS: list[tuple[str, str, Path]] = [
    ("CNN (from scratch)", "cnn", s.MODELS_DIR / "dl" / "cnn_best_model.pt"),
    ("MobileNetV2 (fine-tuned)", "mobilenet", s.MODELS_DIR / "dl" / "mobilenet_best_model.pt"),
    ("EfficientNet-B0 (fine-tuned)", "efficientnet", s.MODELS_DIR / "dl" / "best_model.pt"),
]


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    test_dataset = ImageFolder(
        str(s.PROCESSED_DIR / "test"),
        transform=get_eval_transforms(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4,
    )
    class_names = test_dataset.classes
    n_classes = len(class_names)
    print(f"Test samples: {len(test_dataset)}  Classes: {class_names}")

    builders = {
        "cnn": _build_cnn,
        "mobilenet": _build_mobilenet,
        "efficientnet": _build_efficientnet,
    }

    results: list[dict] = []

    for display_name, tag, ckpt_path in _MODEL_SPECS:
        print(f"\n--- {display_name} ---")
        model = builders[tag](n_classes)
        model = _load_checkpoint(model, ckpt_path, tag)
        if model is None:
            continue
        model = model.to(device)
        model.eval()

        y_pred, y_true = evaluate(model, test_loader, device)
        acc = accuracy_score(y_true, y_pred)
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(n_classes))

        inference_ms = measure_inference_time(model, test_loader, device)

        results.append({
            "name": display_name,
            "tag": tag,
            "accuracy": acc,
            "per_class_f1": per_class_f1,
            "inference_ms": inference_ms,
        })

        print(f"  Accuracy: {acc:.4f}  Inference: {inference_ms:.2f} ms/img")

    if not results:
        print("No checkpoints were loaded. Nothing to compare.")
        return

    report_dir = Path(__file__).resolve().parent.parent.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "dl_model_comparison.md"

    lines: list[str] = []
    lines.append("# DL Model Comparison Report\n\n")
    lines.append(f"**Test set size:** {len(test_dataset)}\n\n")

    lines.append("## Summary\n\n")
    hdr = "| Model | Accuracy | Per-class F1 (macro) | Inference (ms/img) |\n"
    sep = "|-------|----------|---------------------|--------------------|\n"
    lines.append(hdr)
    lines.append(sep)
    for r in results:
        macro_f1 = sum(r["per_class_f1"]) / len(r["per_class_f1"])
        lines.append(
            f"| {r['name']} | {r['accuracy']:.4f} | {macro_f1:.4f} | {r['inference_ms']:.2f} |\n"
        )

    lines.append("\n---\n")

    for r in results:
        lines.append(f"\n## {r['name']}\n\n")
        lines.append("| Class | F1-Score |\n")
        lines.append("|-------|----------|\n")
        for i, name in enumerate(class_names):
            lines.append(f"| {name} | {r['per_class_f1'][i]:.4f} |\n")

    report_path.write_text("".join(lines))
    print(f"\nComparison report written to {report_path}")


if __name__ == "__main__":
    main()
