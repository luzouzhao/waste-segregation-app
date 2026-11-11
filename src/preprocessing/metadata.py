"""
Day 2 — SQLite metadata store.
Creates the waste_metadata table and populates it from data/raw/<category>/.
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config.settings as s


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS waste_metadata (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    file_size   INTEGER NOT NULL,
    source_dataset TEXT,
    ingested_at TEXT    NOT NULL,
    split       TEXT
);
CREATE INDEX IF NOT EXISTS idx_wm_category ON waste_metadata(category);
CREATE INDEX IF NOT EXISTS idx_wm_split ON waste_metadata(split);
"""


def _detect_source(filename: str) -> str:
    if filename.startswith("train_") or filename.startswith("test_") or filename.startswith("val_"):
        return "akshat103/e-waste-image-dataset"
    return "salam035/waste-datasets"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(s.DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_table(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def populate_from_raw(conn: sqlite3.Connection) -> dict[str, int]:
    now = datetime.now().isoformat()
    counts: dict[str, int] = {}
    insert_sql = """
        INSERT INTO waste_metadata (filename, category, width, height, file_size, source_dataset, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    cursor = conn.execute("SELECT COUNT(*) FROM waste_metadata")
    if cursor.fetchone()[0] > 0:
        print("Metadata table already populated. Truncating and re-populating.")
        conn.execute("DELETE FROM waste_metadata")

    for category in s.TARGET_CATEGORIES:
        cat_dir = s.RAW_DIR / category
        if not cat_dir.exists():
            counts[category] = 0
            continue

        n = 0
        for img_path in sorted(cat_dir.iterdir()):
            if img_path.suffix.lower() not in s.VALID_EXTENSIONS:
                continue
            if img_path.is_dir():
                continue
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception:
                continue

            file_size = img_path.stat().st_size
            source = _detect_source(img_path.name)

            conn.execute(insert_sql, (
                img_path.name, category, width, height,
                file_size, source, now,
            ))
            n += 1

        counts[category] = n

    conn.commit()
    return counts


def update_split_column(conn: sqlite3.Connection) -> dict[str, int]:
    manifest_path = s.PROCESSED_DIR / "split_manifest.json"
    if not manifest_path.exists():
        print("split_manifest.json not found — skipping split column update.")
        return {}

    with open(manifest_path) as f:
        manifest = json.load(f)

    counts: dict[str, int] = {}
    for split_name, entries in manifest.items():
        for category, filename in entries:
            conn.execute(
                "UPDATE waste_metadata SET split = ? WHERE filename = ? AND category = ?",
                (split_name, filename, category),
            )
        conn.commit()
        counts[split_name] = conn.execute(
            "SELECT COUNT(*) FROM waste_metadata WHERE split = ?", (split_name,)
        ).fetchone()[0]

    return counts


def report_summary(conn: Optional[sqlite3.Connection] = None) -> dict:
    close = conn is None
    if conn is None:
        conn = get_connection()
    cursor = conn.execute(
        "SELECT category, COUNT(*), AVG(width), AVG(height), AVG(file_size) "
        "FROM waste_metadata GROUP BY category ORDER BY category"
    )
    rows = cursor.fetchall()
    if close:
        conn.close()
    return {row[0]: {"count": row[1], "avg_width": row[2], "avg_height": row[3], "avg_size_bytes": row[4]} for row in rows}


if __name__ == "__main__":
    conn = get_connection()
    create_table(conn)
    counts = populate_from_raw(conn)
    print("Metadata populated:")
    for cat, n in counts.items():
        print(f"  {cat}: {n}")

    split_counts = update_split_column(conn)
    if split_counts:
        print("\nSplit column updated:")
        for split_name, n in split_counts.items():
            print(f"  {split_name}: {n}")
    conn.close()
