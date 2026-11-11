"""
Day 3 (Extended) — CNN from scratch baseline.
Simple 4-block convolutional network trained for 7-class waste classification.
Same training loop pattern as dl_train.py: class-weighted CrossEntropyLoss,
best checkpoint (by val accuracy) saved to models/dl/cnn_best_model.pt,
per-epoch metrics logged to reports/cnn_training_log.csv.
"""

import csv
import ssl
import sys
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config.settings as s
from src.preprocessing.preprocess import get_train_transforms, get_eval_transforms


class SimpleCNN(nn.Module):
    """Custom CNN from scratch with 4 convolutional blocks and a fully-connected head."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def compute_class_weights(dataset: ImageFolder) -> torch.Tensor:
    n_per_class = [0] * len(dataset.classes)
    for _, cls_idx in dataset.samples:
        n_per_class[cls_idx] += 1
    n_total = sum(n_per_class)
    n_classes = len(n_per_class)
    weights = [n_total / (n_classes * c) for c in n_per_class]
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_dataset = ImageFolder(
        str(s.PROCESSED_DIR / "train"),
        transform=get_train_transforms(),
    )
    val_dataset = ImageFolder(
        str(s.PROCESSED_DIR / "val"),
        transform=get_eval_transforms(),
    )
    n_classes = len(train_dataset.classes)
    print(f"Classes: {train_dataset.classes}")

    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
    )

    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"Class weights: {class_weights}")

    model = SimpleCNN(num_classes=n_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    num_epochs = 15
    best_val_acc = 0.0

    log_dir = Path(__file__).resolve().parent.parent.parent / "reports"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "cnn_training_log.csv"

    save_dir = s.MODELS_DIR / "dl"
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device,
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            writer.writerow([
                epoch,
                f"{train_loss:.4f}",
                f"{train_acc:.4f}",
                f"{val_loss:.4f}",
                f"{val_acc:.4f}",
            ])
            f.flush()

            print(
                f"Epoch {epoch:2d}/{num_epochs}  "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_dir / "cnn_best_model.pt")
                print(f"  \u2192 New best checkpoint saved (val_acc={val_acc:.4f})")

    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
