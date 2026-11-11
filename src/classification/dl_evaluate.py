"""
Day 3 — DL evaluation.
Loads the best checkpoint from models/dl/best_model.pt,
evaluates on the held-out test set,
writes accuracy, per-class F1, and confusion matrix to reports/dl_evaluation.md.
"""

import ssl
import sys
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config.settings as s
from src.preprocessing.preprocess import get_eval_transforms


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[str]]:
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

    model = efficientnet_b0(weights=None, num_classes=n_classes)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(model.classifier[1].in_features, n_classes),
    )

    checkpoint_path = s.MODELS_DIR / "dl" / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}. Train first.")
        sys.exit(1)

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    print(f"Loaded checkpoint from {checkpoint_path}")

    y_pred, y_true = evaluate(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))

    report_dir = Path(__file__).resolve().parent.parent.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "dl_evaluation.md"

    lines = ["# DL Evaluation Report\n\n"]
    lines.append(f"**Model:** EfficientNet-B0 (fine-tuned)\n\n")
    lines.append(f"**Test set size:** {len(test_dataset)}\n\n")
    lines.append(f"## Overall Accuracy\n\n**{acc:.4f}**\n\n")
    lines.append("## Per-Class F1-Score\n\n")
    lines.append("| Class | F1-Score |\n")
    lines.append("|-------|----------|\n")

    for i, name in enumerate(class_names):
        lines.append(f"| {name} | {per_class_f1[i]:.4f} |\n")

    lines.append("\n## Confusion Matrix\n\n")
    lines.append("| ↓True \\ Pred→ | " + " | ".join(class_names) + " |\n")
    lines.append("|---" * (n_classes + 1) + "|\n")
    for i, name in enumerate(class_names):
        row = " | ".join(f"{cm[i, j]}" for j in range(n_classes))
        lines.append(f"| **{name}** | {row} |\n")

    report_path.write_text("".join(lines))
    print(f"Evaluation report written to {report_path}")
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
