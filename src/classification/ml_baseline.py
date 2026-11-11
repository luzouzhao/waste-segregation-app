"""
Day 3 — Classical ML baseline.
Extracts RGB color histograms (32 bins per channel → 96-dim feature vector),
trains Logistic Regression / Decision Tree / Random Forest / XGBoost,
saves best model (by val accuracy) to models/ml/best_model.pkl,
writes comparison table to reports/ml_comparison.md.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config.settings as s


def _load_image_paths(split: str) -> tuple[list[str], list[str]]:
    split_dir = s.PROCESSED_DIR / split
    paths: list[str] = []
    labels: list[str] = []
    for category in s.TARGET_CATEGORIES:
        cat_dir = split_dir / category
        if not cat_dir.exists():
            continue
        for f in sorted(cat_dir.iterdir()):
            if f.suffix.lower() in s.VALID_EXTENSIONS:
                paths.append(str(f))
                labels.append(category)
    return paths, labels


def _color_histogram(image_path: str) -> np.ndarray:
    img = np.array(Image.open(image_path).convert("RGB"))
    hists: list[np.ndarray] = []
    for channel in range(3):
        hist, _ = np.histogram(img[:, :, channel], bins=32, range=(0, 256))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist /= total
        hists.append(hist)
    return np.concatenate(hists)


def extract_features(paths: list[str]) -> np.ndarray:
    return np.array([_color_histogram(p) for p in paths])


def main() -> None:
    train_paths, train_labels = _load_image_paths("train")
    val_paths, val_labels = _load_image_paths("val")
    print(f"Loaded {len(train_paths)} train / {len(val_paths)} val samples")

    print("Extracting colour histogram features ...")
    X_train = extract_features(train_paths)
    X_val = extract_features(val_paths)

    label_to_idx = {cat: i for i, cat in enumerate(s.TARGET_CATEGORIES)}
    y_train = np.array([label_to_idx[lbl] for lbl in train_labels])
    y_val = np.array([label_to_idx[lbl] for lbl in val_labels])

    models = {
        "Logistic Regression": LogisticRegression(
            solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=s.SEED,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=s.SEED,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=s.SEED, n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            objective="multi:softprob", num_class=len(s.TARGET_CATEGORIES),
            eval_metric="mlogloss", random_state=s.SEED, n_jobs=-1,
        ),
    }

    results: list[dict[str, float]] = []
    best_model = None
    best_acc = 0.0

    for name, model in models.items():
        print(f"Training {name} ...")
        if isinstance(model, xgb.XGBClassifier):
            counts = np.bincount(y_train)
            sample_weight = counts.sum() / (len(counts) * counts.astype(float))
            model.fit(X_train, y_train, sample_weight=sample_weight[y_train])
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average="macro", zero_division=0,
        )
        results.append({"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
        print(f"  → Val Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    save_dir = s.MODELS_DIR / "ml"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best model saved to {save_dir / 'best_model.pkl'} (val_acc={best_acc:.4f})")

    report_dir = Path(__file__).resolve().parent.parent.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ML Model Comparison\n\n",
        "| Model | Accuracy | Precision | Recall | F1-Score |\n",
        "|-------|----------|-----------|--------|----------|\n",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['accuracy']:.4f} | {r['precision']:.4f} "
            f"| {r['recall']:.4f} | {r['f1']:.4f} |\n"
        )
    lines.append(f"\n*Best model saved: `{best_model.__class__.__name__}` "
                  f"(val accuracy = {best_acc:.4f})*\n")
    report_path = report_dir / "ml_comparison.md"
    report_path.write_text("".join(lines))
    print(f"Comparison table written to {report_path}")


if __name__ == "__main__":
    main()
