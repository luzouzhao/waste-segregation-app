"""
Day 2 — Preprocessing pipeline.
Resizes/normalises images, applies augmentation (flip, rotate, brightness/contrast)
for training, and produces a stratified train/val/test split saved to data/processed/.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config.settings as s


class PreprocessingPipeline:
    def __init__(self, img_size: int = s.IMG_SIZE, augment_copies: int = s.AUGMENT_COPIES):
        self.img_size = img_size
        self.augment_copies = augment_copies
        self.resize = T.Resize((img_size, img_size), antialias=True)
        self.augment = T.Compose([
            T.Resize((img_size, img_size), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ])

    def _load_and_convert(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def _save_image(self, img: Image.Image, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path, quality=95, subsampling=0)

    def process_split(self, paths: list[Path], split_dir: Path, apply_aug: bool = False) -> int:
        n = 0
        for img_path in paths:
            img = self._load_and_convert(img_path)
            resized = self.resize(img)
            resized_np = np.array(resized)
            resized_pil = Image.fromarray(resized_np)
            base_path = split_dir / img_path.parent.name / img_path.name
            self._save_image(resized_pil, base_path)
            n += 1

            if apply_aug:
                for i in range(self.augment_copies):
                    aug_img = self.augment(img)
                    aug_np = np.array(aug_img)
                    aug_pil = Image.fromarray(aug_np)
                    stem = img_path.stem
                    suffix = img_path.suffix
                    aug_name = f"{stem}_aug_{i}{suffix}"
                    aug_path = split_dir / img_path.parent.name / aug_name
                    self._save_image(aug_pil, aug_path)
                    n += 1
        return n

    def run(self) -> dict[str, int]:
        paths_by_category: dict[str, list[Path]] = {}
        for category in s.TARGET_CATEGORIES:
            cat_dir = s.RAW_DIR / category
            if not cat_dir.exists():
                continue
            images = sorted(
                p for p in cat_dir.iterdir()
                if p.suffix.lower() in s.VALID_EXTENSIONS and not p.is_dir()
            )
            if images:
                paths_by_category[category] = images

        train_paths: list[Path] = []
        val_paths: list[Path] = []
        test_paths: list[Path] = []

        for category, images in paths_by_category.items():
            labels = [category] * len(images)
            tr, te = train_test_split(
                images, test_size=s.TEST_SPLIT, random_state=s.SEED,
                stratify=labels,
            )
            tr_labels = [category] * len(tr)
            tr, va = train_test_split(
                tr, test_size=s.VAL_SPLIT / (s.TRAIN_SPLIT + s.VAL_SPLIT),
                random_state=s.SEED, stratify=tr_labels,
            )
            train_paths.extend(tr)
            val_paths.extend(va)
            test_paths.extend(te)

        train_dir = s.PROCESSED_DIR / "train"
        val_dir = s.PROCESSED_DIR / "val"
        test_dir = s.PROCESSED_DIR / "test"

        for d in [train_dir, val_dir, test_dir]:
            for cat in s.TARGET_CATEGORIES:
                (d / cat).mkdir(parents=True, exist_ok=True)

        counts: dict[str, int] = {}
        counts["train"] = self.process_split(train_paths, train_dir, apply_aug=True)
        counts["val"] = self.process_split(val_paths, val_dir, apply_aug=False)
        counts["test"] = self.process_split(test_paths, test_dir, apply_aug=False)

        split_manifest = {
            "train": [(p.parent.name, p.name) for p in sorted(train_paths)],
            "val": [(p.parent.name, p.name) for p in sorted(val_paths)],
            "test": [(p.parent.name, p.name) for p in sorted(test_paths)],
        }
        manifest_path = s.PROCESSED_DIR / "split_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(split_manifest, f, indent=2)

        return counts


def get_train_transforms() -> T.Compose:
    return T.Compose([
        T.Resize((s.IMG_SIZE, s.IMG_SIZE), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms() -> T.Compose:
    return T.Compose([
        T.Resize((s.IMG_SIZE, s.IMG_SIZE), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    counts = pipeline.run()
    print("Preprocessing complete:")
    for split, n in counts.items():
        print(f"  {split}: {n} images")
