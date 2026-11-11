"""
Day 1 - Data Collection.
Pulls 2 Kaggle datasets and merges into data/raw/<category>/ for our 7 classes.

Setup: pip install kagglehub  |  put kaggle.json in ~/.kaggle/
Run:   python src/preprocessing/download_datasets.py
"""

import shutil
from pathlib import Path
import kagglehub

RAW_DIR = Path("data/raw")

# dataset 1: salam035/waste-datasets  ->  Dataset_ThirdStage/Dataset/<folder>
TRICASCADE_MAP = {
    "A_Foods": "organic",
    "C_Cardboard": "paper",
    "D_Newspaper": "paper",
    "E_Paper Cups": "paper",
    "F_Papers": "paper",
    "G_Brown Glass": "glass",
    "I_Green Glass": "glass",
    "J_White Glass": "glass",
    "K_Beverage Cans": "metal",
    "M_Metal Containers": "metal",
    "N_Plastic Bag": "plastic",
    "O_Plastic Bottle": "plastic",
    "Q_Plastic Containers": "plastic",
    "R_Plastic Cups": "plastic",
    "S_Tetra Pak": "plastic",
    "Z_H_Thermometer": "hazardous",
    "Z_I_Cigarette Butt": "hazardous",
    "Z_J_Pesticidebottle": "hazardous",
    "Z_K_Spray cans": "hazardous",
    "Y_Medicine and Medicine Strip": "hazardous",
    "Z_A_A_Syringe": "hazardous",
}
TRICASCADE_ID = "salam035/waste-datasets"
TRICASCADE_SUBPATH = Path("Dataset_ThirdStage/Dataset")

# dataset 2: akshat103/e-waste-image-dataset -> modified-dataset/train/<folder> (all -> electronic)
EWASTE_ID = "akshat103/e-waste-image-dataset"
EWASTE_SPLITS = ["train", "test", "val"]  # pooled here; we re-split ourselves in Day 2 preprocessing
EWASTE_FOLDERS = [
    "Battery", "Keyboard", "Microwave", "Mobile", "Mouse",
    "PCB", "Player", "Printer", "Television", "Washing Machine",
]

TARGET_CATEGORIES = ["organic", "plastic", "paper", "glass", "metal", "electronic", "hazardous"]


def copy_images(src_folder: Path, target_cat: str, prefix: str):
    if not src_folder.exists():
        print(f"  [skip] not found: {src_folder}")
        return
    target = RAW_DIR / target_cat
    target.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in src_folder.glob("*"):
        if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
            shutil.copy2(img, target / f"{prefix}_{img.name}")
            n += 1
    print(f"  [ok] {src_folder.name} -> {target_cat} ({n} images)")


def main():
    for cat in TARGET_CATEGORIES:
        (RAW_DIR / cat).mkdir(parents=True, exist_ok=True)

    print(f"Downloading {TRICASCADE_ID} ...")
    base = Path(kagglehub.dataset_download(TRICASCADE_ID)) / TRICASCADE_SUBPATH
    for folder, cat in TRICASCADE_MAP.items():
        copy_images(base / folder, cat, folder.replace(" ", "_"))

    print(f"Downloading {EWASTE_ID} ...")
    base = Path(kagglehub.dataset_download(EWASTE_ID)) / "modified-dataset"
    for split in EWASTE_SPLITS:
        for folder in EWASTE_FOLDERS:
            copy_images(base / split / folder, "electronic", f"{split}_{folder.replace(' ', '_')}")

    print("\nImage counts per category:")
    for cat in TARGET_CATEGORIES:
        print(f"  {cat}: {len(list((RAW_DIR / cat).glob('*')))}")


if __name__ == "__main__":
    main()
