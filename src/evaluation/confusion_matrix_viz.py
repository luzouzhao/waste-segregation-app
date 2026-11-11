"""
Module 8 — Confusion Matrix Visualisation.

Generates and saves styled confusion matrix heatmaps for:
  1. Best ML model  (models/ml/best_model.pkl)  evaluated on the test split.
  2. Best DL model  (models/dl/best_model.pt)   evaluated on the test split.

Output images are written to reports/confusion_matrix_ml.png
and reports/confusion_matrix_dl.png so they can be embedded in the
Streamlit UI or any report document.
"""

import pickle
import ssl
import sys
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts & Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config.settings as s


# ── helpers ──────────────────────────────────────────────────────────────────

CLASS_NAMES: list[str] = s.TARGET_CATEGORIES  # 7 classes

REPORT_DIR = Path(__file__).resolve().parent.parent.parent / "reports"

CATEGORY_COLORS: dict[str, str] = {
    "organic":   "#6D4C29",
    "plastic":   "#1976D2",
    "paper":     "#8D6E63",
    "glass":     "#00897B",
    "metal":     "#78909C",
    "electronic":"#5E35B1",
    "hazardous": "#C62828",
}


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: Path,
) -> None:
    """Render a styled confusion-matrix heatmap and save to *save_path*."""
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalise by row (true class) to get recall-style proportions
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1          # avoid division by zero
    cm_norm = cm_norm / row_sums

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Greens", vmin=0, vmax=1)

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall proportion", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    # Cell annotations — raw count + normalised value
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val_norm = cm_norm[i, j]
            val_raw  = int(cm[i, j])
            color = "white" if val_norm > thresh else "#1a1a1a"
            ax.text(
                j, i,
                f"{val_raw}\n({val_norm:.0%})",
                ha="center", va="center",
                fontsize=8, color=color, fontweight="bold" if i == j else "normal",
            )

    ax.set_xlabel("Predicted label", fontsize=12, labelpad=10)
    ax.set_ylabel("True label", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)

    # Diagonal border highlight
    for i in range(n):
        rect = mpatches.FancyBboxPatch(
            (i - 0.48, i - 0.48), 0.96, 0.96,
            boxstyle="round,pad=0.02",
            linewidth=1.8, edgecolor="#2E7D32", facecolor="none",
        )
        ax.add_patch(rect)

    fig.tight_layout()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved → {save_path}")


# ── ML confusion matrix ───────────────────────────────────────────────────────

def _color_histogram(image_path: str) -> np.ndarray:
    """32-bin RGB histogram (96-dim) — same as ml_baseline.py."""
    img = np.array(Image.open(image_path).convert("RGB"))
    hists: list[np.ndarray] = []
    for ch in range(3):
        hist, _ = np.histogram(img[:, :, ch], bins=32, range=(0, 256))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist /= total
        hists.append(hist)
    return np.concatenate(hists)


def generate_ml_confusion_matrix(
    save_path: Path | None = None,
) -> tuple[np.ndarray, Path]:
    """Load best ML model, run inference on test split, save confusion matrix PNG.
    
    Returns:
        (confusion_matrix_array, save_path)
    """
    save_path = save_path or (REPORT_DIR / "confusion_matrix_ml.png")

    model_path = s.MODELS_DIR / "ml" / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"ML model not found at {model_path}. Train first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    label_to_idx = {cat: i for i, cat in enumerate(CLASS_NAMES)}
    test_dir = s.PROCESSED_DIR / "test"

    paths: list[str] = []
    labels: list[int] = []
    for category in CLASS_NAMES:
        cat_dir = test_dir / category
        if not cat_dir.exists():
            continue
        for img_f in sorted(cat_dir.iterdir()):
            if img_f.suffix.lower() in s.VALID_EXTENSIONS:
                paths.append(str(img_f))
                labels.append(label_to_idx[category])

    if not paths:
        raise RuntimeError("No test images found. Run preprocess.py first.")

    print(f"Extracting features for {len(paths)} test images ...")
    X = np.array([_color_histogram(p) for p in paths])
    y_true = np.array(labels)
    y_pred = model.predict(X)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    model_name = type(model).__name__
    _plot_confusion_matrix(
        cm, CLASS_NAMES,
        title=f"ML Confusion Matrix — {model_name}",
        save_path=save_path,
    )
    np.save(save_path.with_suffix(".npy"), cm)
    return cm, save_path


# ── DL confusion matrix ───────────────────────────────────────────────────────

def generate_dl_confusion_matrix(
    save_path: Path | None = None,
) -> tuple[np.ndarray, Path]:
    """Load best DL model (EfficientNet-B0), run inference on test split, save PNG.
    
    Returns:
        (confusion_matrix_array, save_path)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.models import efficientnet_b0

    save_path = save_path or (REPORT_DIR / "confusion_matrix_dl.png")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    from src.preprocessing.preprocess import get_eval_transforms

    test_dataset = ImageFolder(
        str(s.PROCESSED_DIR / "test"),
        transform=get_eval_transforms(),
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    class_names = test_dataset.classes
    n_classes = len(class_names)

    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, n_classes),
    )

    ckpt_path = s.MODELS_DIR / "dl" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"DL checkpoint not found at {ckpt_path}. Train first.")

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for inputs, lbls in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(lbls.tolist())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))
    _plot_confusion_matrix(
        cm, class_names,
        title="DL Confusion Matrix — EfficientNet-B0",
        save_path=save_path,
    )
    np.save(save_path.with_suffix(".npy"), cm)
    return cm, save_path


# ── YOLOv8n-cls (deployed model) confusion matrix ────────────────────────────

def generate_yolo_confusion_matrix(
    save_path: Path | None = None,
) -> tuple[np.ndarray, Path]:
    """Load the deployed YOLOv8n-cls classifier and run the production pipeline
    (full image + 4-view TTA) over the whole test split, then save the matrix.

    Returns:
        (confusion_matrix_array, save_path)
    """
    import cv2

    from src.detection.pipeline import _classify_image, load_classifier

    save_path = save_path or (REPORT_DIR / "confusion_matrix_yolo.png")

    classifier = load_classifier()
    test_dir = s.PROCESSED_DIR / "test"

    y_true: list[int] = []
    y_pred: list[int] = []
    for idx, category in enumerate(CLASS_NAMES):
        cat_dir = test_dir / category
        if not cat_dir.exists():
            continue
        files = sorted(
            f for f in cat_dir.iterdir() if f.suffix.lower() in s.VALID_EXTENSIONS
        )
        print(f"  {category}: {len(files)} images")
        for img_f in files:
            img = cv2.imread(str(img_f))
            if img is None:
                continue
            pred, _, _ = _classify_image(img, classifier=classifier)
            y_true.append(idx)
            y_pred.append(CLASS_NAMES.index(pred))

    if not y_true:
        raise RuntimeError("No test images found. Run preprocess.py first.")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    _plot_confusion_matrix(
        cm, CLASS_NAMES,
        title="YOLOv8n-cls Confusion Matrix — 4-view TTA",
        save_path=save_path,
    )
    print(f"  accuracy: {np.trace(cm) / cm.sum():.4f}")
    np.save(save_path.with_suffix(".npy"), cm)
    return cm, save_path


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Generating ML confusion matrix ===")
    try:
        p = generate_ml_confusion_matrix()
        print(f"  OK  {p}")
    except Exception as e:
        print(f"  FAILED  ML confusion matrix failed: {e}")

    print("\n=== Generating DL confusion matrix ===")
    try:
        p = generate_dl_confusion_matrix()
        print(f"  OK  {p}")
    except Exception as e:
        print(f"  FAILED  DL confusion matrix failed: {e}")

    print("\n=== Generating YOLOv8n-cls confusion matrix ===")
    try:
        p = generate_yolo_confusion_matrix()
        print(f"  OK  {p}")
    except Exception as e:
        print(f"  FAILED  YOLO confusion matrix failed: {e}")


if __name__ == "__main__":
    main()
