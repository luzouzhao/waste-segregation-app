"""
Day 2 — Data validation.
Scans data/raw/<category>/, drops corrupt / unreadable / near-duplicate images,
and writes a validation report to reports/validation_report.txt.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config.settings as s


ValidationReport = dict[str, dict[str, int | list[str]]]


def _average_hash(image: Image.Image, hash_size: int = 8) -> int:
    gray = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.array(gray, dtype=np.float64)
    avg = pixels.mean()
    bits = (pixels > avg).flatten()
    return sum(b << i for i, b in enumerate(bits))


def _hamming_distance(h1: int, h2: int) -> int:
    return (h1 ^ h2).bit_count()


def validate_image(path: Path) -> Optional[str]:
    if path.suffix.lower() not in s.VALID_EXTENSIONS:
        return "invalid_extension"
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception:
        return "corrupt_header"
    try:
        with Image.open(path) as img:
            img.load()
    except Exception:
        return "unreadable_pixels"
    if path.stat().st_size == 0:
        return "zero_bytes"
    return None


def find_duplicates(paths: list[Path]) -> list[list[Path]]:
    groups: dict[int, list[Path]] = {}
    for p in paths:
        size = p.stat().st_size
        groups.setdefault(size, []).append(p)

    dup_groups: list[list[Path]] = []
    for file_size, group in groups.items():
        if len(group) < 2:
            continue
        hashes: dict[int, list[Path]] = {}
        for p in group:
            try:
                with Image.open(p) as img:
                    h = _average_hash(img)
            except Exception:
                continue
            hashes.setdefault(h, []).append(p)

        for h, same_hash_group in hashes.items():
            if len(same_hash_group) < 2:
                continue
            seen: list[Path] = []
            for p in same_hash_group:
                match_found = False
                for existing in seen:
                    d = _hamming_distance(
                        _average_hash(Image.open(p)),
                        _average_hash(Image.open(existing)),
                    )
                    if d < s.DUPLICATE_HAMMING_THRESHOLD:
                        match_found = True
                        break
                if not match_found:
                    seen.append(p)
            if len(seen) > 1:
                dup_groups.append(seen)

    return dup_groups


def run_validation() -> ValidationReport:
    report: ValidationReport = {}
    total_removed = 0
    removal_log: list[tuple[str, str, str]] = []

    for category in s.TARGET_CATEGORIES:
        cat_dir = s.RAW_DIR / category
        if not cat_dir.exists():
            report[category] = {"initial": 0, "removed": 0, "final": 0, "errors": []}
            continue

        all_images = sorted(cat_dir.iterdir())
        initial = len(all_images)
        removed = 0
        errors: list[str] = []

        corrupt: list[Path] = []
        good: list[Path] = []
        for img_path in all_images:
            err = validate_image(img_path)
            if err is not None:
                corrupt.append(img_path)
                errors.append(f"{img_path.name}: {err}")
            else:
                good.append(img_path)

        for p in corrupt:
            removal_log.append((category, p.name, "corrupt"))
            p.unlink()
            removed += 1

        dup_groups = find_duplicates(good)
        for group in dup_groups:
            for p in group[1:]:
                removal_log.append((category, p.name, "duplicate"))
                p.unlink()
                removed += 1

        dup_removed = sum(len(g) - 1 for g in dup_groups)
        final = len(good) - dup_removed
        total_removed += removed
        report[category] = {"initial": initial, "removed": removed, "final": final, "errors": errors}

    report["_meta"] = {
        "timestamp": datetime.now().isoformat(),
        "total_removed": total_removed,
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_report(report, removal_log)
    return report


def _write_report(report: ValidationReport, removal_log: list[tuple[str, str, str]]) -> None:
    txt_path = Path("reports/validation_report.txt")
    csv_path = Path("reports/validation_removals.csv")
    meta = report.pop("_meta", {})

    with open(txt_path, "w") as f:
        f.write(f"Validation Report — {meta.get('timestamp', 'unknown')}\n")
        f.write(f"{'='*60}\n\n")
        for cat, info in report.items():
            f.write(f"Category: {cat}\n")
            f.write(f"  Initial : {info['initial']}\n")
            f.write(f"  Removed : {info['removed']}\n")
            f.write(f"  Final   : {info['final']}\n")
            if info["errors"]:
                f.write("  Errors:\n")
                for err in info["errors"]:
                    f.write(f"    - {err}\n")
            f.write("\n")
        f.write(f"Total removed: {meta.get('total_removed', 0)}\n")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "filename", "reason"])
        writer.writerows(removal_log)


if __name__ == "__main__":
    report = run_validation()
    meta = report.pop("_meta", {})
    print(f"Validation complete. Removed {meta.get('total_removed', 0)} files.")
