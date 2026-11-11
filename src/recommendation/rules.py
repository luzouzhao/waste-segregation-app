"""
Day 6 — Rule-based recyclability & disposal guidance.
Maps each of the 7 waste categories to a static guidance dict with
recyclability, disposal method, and hazard level.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config.settings as s

CATEGORY_LABELS: dict[str, str] = {
    "organic": "厨余垃圾",
    "plastic": "塑料",
    "paper": "纸类",
    "glass": "玻璃",
    "metal": "金属",
    "electronic": "电子垃圾",
    "hazardous": "有害垃圾",
}

HAZARD_LABELS: dict[str, str] = {
    "low": "低",
    "moderate": "中",
    "high": "高",
}

GUIDANCE: dict[str, dict] = {
    "organic": {
        "recyclable": True,
        "disposal_method": "投入家庭堆肥桶或厨余垃圾（湿垃圾）收集桶。请去除塑料袋、金属等非有机物杂质。除非当地允许，否则不要堆肥肉类、奶制品或含油剩菜。",
        "hazard_level": "low",
    },
    "plastic": {
        "recyclable": True,
        "disposal_method": "冲洗干净并晾干后投入可回收物桶（请核对当地接受哪些塑料树脂类型）。塑料袋、保鲜膜等软塑料通常需单独送到支持回收的门店。",
        "hazard_level": "low",
    },
    "paper": {
        "recyclable": True,
        "disposal_method": "投入干燥的纸类/纸板回收桶。尽量去除胶带、订书钉和塑料开窗。被油浸透的纸（如披萨盒）不要投入可回收物，应作为厨余垃圾处理。",
        "hazard_level": "low",
    },
    "glass": {
        "recyclable": True,
        "disposal_method": "冲洗干净并取下瓶盖后投入玻璃回收桶。不要与陶瓷、耐热玻璃（Pyrex）或窗玻璃混投，它们的熔点不同。破碎玻璃请小心处理，避免划伤。",
        "hazard_level": "moderate",
    },
    "metal": {
        "recyclable": True,
        "disposal_method": "将食品容器冲洗干净后投入金属回收桶。易拉罐、马口铁罐和铝箔一般都可回收。气雾罐必须完全排空后才能投放。",
        "hazard_level": "low",
    },
    "electronic": {
        "recyclable": True,
        "disposal_method": "不要投入普通垃圾桶或可回收桶。请送至电子垃圾回收点、商家回收计划或市政电子垃圾回收活动。投放前尽量取出电池。",
        "hazard_level": "high",
    },
    "hazardous": {
        "recyclable": False,
        "disposal_method": "不要投入普通垃圾桶或可回收物。请送至指定的有害垃圾收集点。尽量保留原包装并保持标签完整。具体处置要求请向当地市政部门确认。",
        "hazard_level": "high",
    },
}


def get_category_label(category: str) -> str:
    """Return the Chinese display name for a waste category."""
    return CATEGORY_LABELS.get(category, category)


def get_guidance(category: str) -> dict:
    """Return disposal guidance for a given waste category.

    Args:
        category: One of the 7 recognised waste categories
                  (organic, plastic, paper, glass, metal, electronic, hazardous).

    Returns:
        Dict with keys: recyclable (bool), disposal_method (str),
                        hazard_level (str).

    Raises:
        ValueError: If category is not in TARGET_CATEGORIES.
    """
    if category not in s.TARGET_CATEGORIES:
        valid = ", ".join(s.TARGET_CATEGORIES)
        raise ValueError(f"Unknown category '{category}'. Valid: {valid}")
    return dict(GUIDANCE[category])
