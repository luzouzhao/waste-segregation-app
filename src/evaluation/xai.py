"""
Module 8 — Explainable AI (XAI).

Implements two complementary explanation techniques:

  1. Grad-CAM (Gradient-weighted Class Activation Mapping)
     — for any PyTorch CNN / EfficientNet / MobileNetV2 model.
     Produces a heatmap overlaid on the original image showing WHICH
     spatial regions most influenced the predicted class.

  2. ML Feature Importance
     — for tree-based sklearn models (Random Forest / XGBoost / Decision Tree).
     Produces a bar chart showing which RGB histogram bins drove predictions.

Public API
----------
  gradcam_explain(image_path, model, target_layer, class_idx=None) -> np.ndarray
      Returns an BGR image (uint8) with the Grad-CAM overlay.

  explain_image_gradcam(image_path, save_path=None) -> Path
      Full pipeline: loads EfficientNet-B0 checkpoint, runs Grad-CAM,
      saves output PNG. Returns save_path.

  explain_ml_feature_importance(save_path=None) -> Path
      Loads best ML model, plots feature importance bar chart.
      Works for RF/XGB (built-in importance) and LR (coefficient magnitude).
"""

import sys
import ssl
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    import torch

ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config.settings as s

REPORT_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
CLASS_NAMES: list[str] = s.TARGET_CATEGORIES


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Grad-CAM core
# ═══════════════════════════════════════════════════════════════════════════════

class _GradCAMHook:
    """Registers forward/backward hooks on *layer* to capture activations & grads."""

    def __init__(self, layer) -> None:
        self.activations: "torch.Tensor | None" = None
        self.gradients: "torch.Tensor | None" = None
        self._fwd = layer.register_forward_hook(self._save_activation)
        self._bwd = layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        self._fwd.remove()
        self._bwd.remove()


def gradcam_explain(
    image: "np.ndarray",              # RGB uint8 (H×W×3)
    model: "torch.nn.Module",
    target_layer: "torch.nn.Module",
    class_idx: int | None = None,
    alpha: float = 0.55,
) -> np.ndarray:
    """Run Grad-CAM and return an RGB overlay image (same size as *image*).

    Args:
        image:        Original image as RGB numpy array (H×W×3, uint8).
        model:        PyTorch model in eval mode. Must be able to process
                      a single image through get_eval_transforms().
        target_layer: The convolutional layer to hook (e.g. last conv block).
        class_idx:    Class to explain. If None, uses the argmax prediction.
        alpha:        Blend weight of the heatmap overlay (0 = original only).

    Returns:
        RGB overlay image (uint8, same size as *image*).
    """
    import torch
    from src.preprocessing.preprocess import get_eval_transforms

    transform = get_eval_transforms()

    # Preprocess
    pil_img = Image.fromarray(image)
    tensor = transform(pil_img).unsqueeze(0)   # (1, 3, H, W)

    # Determine device from model
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    hook = _GradCAMHook(target_layer)
    model.eval()

    # Forward
    output = model(tensor)                      # (1, num_classes)
    if class_idx is None:
        class_idx = int(output.argmax(dim=1).item())

    # Backward for the chosen class
    model.zero_grad()
    score = output[0, class_idx]
    score.backward()

    hook.remove()

    activations = hook.activations[0]          # (C, H', W')
    gradients   = hook.gradients[0]            # (C, H', W')

    # Grad-CAM weights: global-average-pool the gradients
    weights = gradients.mean(dim=(1, 2))       # (C,)
    cam = (weights[:, None, None] * activations).sum(dim=0)  # (H', W')
    cam = torch.relu(cam)

    # Normalise to [0, 1]
    cam_np = cam.cpu().numpy()
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()

    # Resize to original image size
    h, w = image.shape[:2]
    pil_cam = Image.fromarray(cam_np).resize((w, h), resample=Image.Resampling.BILINEAR)
    cam_resized = np.array(pil_cam)

    # Convert to colour heatmap (JET)
    import matplotlib.cm as cm_mpl
    jet = cm_mpl.get_cmap("jet")
    # jet returns (R,G,B,A) floats in [0, 1]
    heatmap_rgba = jet(cam_resized)
    heatmap_rgb = np.uint8(255 * heatmap_rgba[:, :, :3])

    # Blend with original
    overlay = (alpha * heatmap_rgb + (1 - alpha) * image).astype(np.uint8)
    return overlay, class_idx, cam_resized


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Full Grad-CAM pipeline (EfficientNet-B0)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_efficientnet(device: "torch.device") -> "torch.nn.Module":
    import torch.nn as nn
    from torchvision.models import efficientnet_b0

    n_classes = len(CLASS_NAMES)
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, n_classes),
    )

    ckpt_path = s.MODELS_DIR / "dl" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"DL checkpoint not found at {ckpt_path}. Train EfficientNet first."
        )

    import torch
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def explain_image_gradcam(
    image_path: str | Path,
    save_path: Path | None = None,
    class_idx: int | None = None,
) -> Path:
    """Run Grad-CAM on *image_path* using the EfficientNet-B0 checkpoint.

    Saves a 3-panel figure (original | heatmap | overlay) to *save_path*.

    Args:
        image_path: Path to the input waste image.
        save_path:  Where to save the output PNG.
                    Defaults to reports/gradcam_<stem>.png.
        class_idx:  Class index to explain. None = argmax (predicted class).

    Returns:
        Path to the saved PNG.
    """
    import torch

    image_path = Path(image_path)
    save_path = save_path or (
        REPORT_DIR / f"gradcam_{image_path.stem}.png"
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = _load_efficientnet(device)

    # Target layer = last convolutional block of EfficientNet-B0 features
    target_layer = model.features[-1]

    # Load image as RGB
    pil_img = Image.open(image_path).convert("RGB")
    img_rgb = np.array(pil_img)

    overlay, pred_class_idx, cam = gradcam_explain(
        img_rgb, model, target_layer, class_idx=class_idx
    )

    pred_label = CLASS_NAMES[pred_class_idx]

    # ── 3-panel figure ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Grad-CAM Explanation  |  Predicted: {pred_label.upper()}",
        fontsize=14, fontweight="bold", y=1.02,
    )

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=11)
    axes[0].axis("off")

    cam_disp = axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Activation Heatmap", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(cam_disp, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay  (class: {pred_label})", fontsize=11)
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grad-CAM saved → {save_path}")
    return save_path


def gradcam_to_image(
    image_path: str | Path,
    class_idx: int | None = None,
) -> tuple[np.ndarray, str]:
    """Return the Grad-CAM overlay as an RGB numpy array (for Streamlit display).

    Returns:
        (overlay_rgb, predicted_label)
    """
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = _load_efficientnet(device)
    target_layer = model.features[-1]

    try:
        pil_img = Image.open(image_path).convert("RGB")
        img_rgb = np.array(pil_img)
    except Exception as e:
        raise ValueError(f"Cannot read image {image_path}: {e}")

    overlay, pred_idx, _cam = gradcam_explain(img_rgb, model, target_layer, class_idx)
    return overlay, CLASS_NAMES[pred_idx]


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  ML Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════

def explain_ml_feature_importance(
    save_path: Path | None = None,
    top_n: int = 20,
) -> tuple[list[str], list[float], Path]:
    """Generate a feature importance bar chart for the best ML model.

    Works for:
      - RandomForestClassifier / DecisionTreeClassifier → feature_importances_
      - XGBClassifier                                   → feature_importances_
      - LogisticRegression                              → mean |coef_| across classes

    The 96-dim feature vector consists of 32-bin histograms for R, G, B channels
    respectively (bins 0-31=R, 32-63=G, 64-95=B).

    Args:
        save_path: Where to save the PNG. Defaults to reports/ml_feature_importance.png.
        top_n:     Show only the top-N most important features.

    Returns:
        (top_names, top_values, save_path)
    """
    import pickle

    save_path = save_path or (REPORT_DIR / "ml_feature_importance.png")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = s.MODELS_DIR / "ml" / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"ML model not found at {model_path}. Run ml_baseline.py first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Feature names: R_bin0 … R_bin31, G_bin0 … G_bin31, B_bin0 … B_bin31
    channels = ["R", "G", "B"]
    feature_names = [f"{ch}_bin{b}" for ch in channels for b in range(32)]

    model_name = type(model).__name__

    # Extract importances
    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.array(model.coef_)).mean(axis=0)
    else:
        raise RuntimeError(
            f"Cannot extract feature importances from {model_name}."
        )

    # Select top-N
    top_idx = np.argsort(importances)[-top_n:][::-1]
    top_names = [feature_names[i] for i in top_idx]
    top_vals  = importances[top_idx]

    # Assign colour by channel
    channel_colors = {"R": "#E53935", "G": "#43A047", "B": "#1E88E5"}
    bar_colors = [channel_colors[n.split("_")[0]] for n in top_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(top_n), top_vals[::-1], color=bar_colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(
        f"ML Feature Importance — {model_name}\n"
        f"(Top {top_n} of 96 RGB histogram bins)",
        fontsize=13, fontweight="bold",
    )

    # Legend
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, fc=channel_colors["R"]),
        plt.Rectangle((0, 0), 1, 1, fc=channel_colors["G"]),
        plt.Rectangle((0, 0), 1, 1, fc=channel_colors["B"]),
    ]
    ax.legend(legend_patches, ["Red channel", "Green channel", "Blue channel"],
              loc="lower right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    np.save(save_path.with_suffix(".npy"), np.array(top_vals))
    np.save(save_path.with_suffix(".names.npy"), np.array(top_names))
    print(f"ML feature importance saved → {save_path}")
    return top_names, top_vals, save_path


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Entry point — generate all static XAI assets
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=== ML Feature Importance ===")
    try:
        p = explain_ml_feature_importance()
        print(f"  OK  {p}")
    except Exception as exc:
        print(f"  FAILED  {exc}")

    print("\n=== Grad-CAM (sample images) ===")
    sample_dir = s.RAW_DIR
    ran_once = False
    for cat in CLASS_NAMES:
        cat_dir = sample_dir / cat
        if not cat_dir.exists():
            continue
        imgs = sorted(cat_dir.glob("*.jpg"))
        if not imgs:
            imgs = sorted(cat_dir.glob("*.png"))
        if not imgs:
            continue
        img_path = imgs[0]
        try:
            out = REPORT_DIR / f"gradcam_{cat}.png"
            explain_image_gradcam(img_path, save_path=out)
            ran_once = True
            print(f"  OK  {out}")
        except Exception as exc:
            print(f"  FAILED  {cat}: {exc}")

    if not ran_once:
        print("  WARNING  No sample images found for Grad-CAM.")


if __name__ == "__main__":
    main()
