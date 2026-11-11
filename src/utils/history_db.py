"""
Day 7 — Disposal history database.
Manages a SQLite table for logging every classification event:
timestamp, image_filename, category, confidence, recyclable, hazard_level, city.
"""

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config.settings as s

_INIT_DONE: bool = False

# 云端容器时区是 UTC，用 localtime 记录会比北京时间慢 8 小时，这里固定按东八区写入。
_BEIJING_TZ = timezone(timedelta(hours=8))


def _now_beijing() -> str:
    """返回东八区当前时间字符串（YYYY-MM-DD HH:MM:SS）。"""
    return datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _get_connection() -> sqlite3.Connection:
    global _INIT_DONE
    # 全新克隆的仓库里没有 data/ 目录（已被 .gitignore 排除），先确保目录存在。
    s.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(s.DB_PATH))
    conn.row_factory = sqlite3.Row
    # 关闭 rollback journal：该文件写入在部分环境（沙箱/虚拟文件系统）会被静默丢弃，
    # 导致 commit 返回成功但数据实际未落盘。
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    if not _INIT_DONE:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS disposal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT (datetime('now', '+8 hours')),
                image_filename TEXT,
                category TEXT NOT NULL,
                confidence REAL,
                recyclable INTEGER,
                hazard_level TEXT,
                city TEXT DEFAULT ''
            )
        """)
        conn.commit()
        _INIT_DONE = True
    return conn


def log_disposal(result: dict, city: str = "") -> None:
    """Write one disposal event to the history table.

    Args:
        result: Dict from classify_waste_image() with keys
                category, confidence, bbox, detection_confidence.
        city: Optional city string provided by the user.
    """
    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO disposal_history
            (timestamp, image_filename, category, confidence, recyclable, hazard_level, city)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _now_beijing(),
            result.get("image_filename", ""),
            result.get("category", ""),
            result.get("confidence", 0.0),
            1 if result.get("recyclable", False) else 0,
            result.get("hazard_level", ""),
            city or "",
        ),
    )
    conn.commit()
    conn.close()


def clear_history() -> int:
    """清空全部处置记录并重置自增编号。

    Returns:
        被删除的记录条数。
    """
    conn = _get_connection()
    cur = conn.execute("DELETE FROM disposal_history")
    deleted = cur.rowcount
    conn.execute("DELETE FROM sqlite_sequence WHERE name = 'disposal_history'")
    conn.commit()
    conn.close()
    return deleted


def get_history(limit: int | None = 50) -> list[dict]:
    """Return the most recent disposal history entries.

    Args:
        limit: Maximum number of rows to return (latest first).
            Pass None to return the full history.

    Returns:
        List of dicts with keys: id, timestamp, image_filename, category,
        confidence, recyclable, hazard_level, city.
    """
    sql = "SELECT * FROM disposal_history ORDER BY id DESC"
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    conn = _get_connection()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
