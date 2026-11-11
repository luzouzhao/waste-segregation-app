from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DB_PATH = PROJECT_ROOT / "data" / "waste_metadata.db"

TARGET_CATEGORIES: list[str] = [
    "organic", "plastic", "paper", "glass",
    "metal", "electronic", "hazardous",
]

IMG_SIZE: int = 256
VALID_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}
SEED: int = 42

TRAIN_SPLIT: float = 0.7
VAL_SPLIT: float = 0.15
TEST_SPLIT: float = 0.15

AUGMENT_COPIES: int = 2
DUPLICATE_HAMMING_THRESHOLD: int = 8
