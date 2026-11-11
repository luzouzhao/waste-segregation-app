"""
Day 6 — LLM-based disposal explanation generator.
Optionally uses Groq (llama-3.1-8b-instant) to produce a short plain-language
explanation for end users. Falls back to rule-based text if the API key is
missing or the API call raises any exception.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

load_dotenv()


def _format_fallback(category: str, rule_guidance: dict) -> str:
    """Produce a plain-language Chinese sentence from the rule guidance dict."""
    from src.recommendation.rules import get_category_label

    label = get_category_label(category)
    recyclable_str = (
        "属于可回收物"
        if rule_guidance.get("recyclable", False)
        else "不可回收，需要单独处理"
    )
    parts = [
        f"该物品被识别为「{label}」，{recyclable_str}。",
        rule_guidance.get("disposal_method", ""),
    ]
    return " ".join(p for p in parts if p)


def generate_explanation(category: str, rule_guidance: dict) -> str:
    """Generate a 2-3 sentence disposal explanation using Groq or fallback.

    Checks for GROQ_API_KEY in the environment (loaded from .env). If the key
    is present, calls Groq's llama-3.1-8b-instant with a fixed prompt template.
    On any failure — missing key, network error, API error, empty response —
    falls back to a rule-based formatted sentence.

    Args:
        category: Waste category label (e.g. "plastic").
        rule_guidance: Dict with keys recyclable (bool),
                       disposal_method (str), hazard_level (str).

    Returns:
        A short plain-language explanation string.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _format_fallback(category, rule_guidance)

    try:
        from groq import Groq

        from src.recommendation.rules import (
            HAZARD_LABELS,
            get_category_label,
        )

        client = Groq(api_key=api_key)
        prompt = (
            f"你是一名帮助市民处理垃圾的智能助手。"
            f"该物品被分类为「{get_category_label(category)}」。"
            f"是否可回收：{'是' if rule_guidance.get('recyclable', False) else '否'}。"
            f"处置方式：{rule_guidance.get('disposal_method', '')} "
            f"危害等级：{HAZARD_LABELS.get(rule_guidance.get('hazard_level', 'low'), '低')}。"
            f"请用 2-3 句通俗易懂的中文说明用户应如何处置该物品，"
            f"以及这样做对环境的意义。"
        )
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        text = response.choices[0].message.content.strip()
        if text:
            return text
    except Exception:
        pass

    return _format_fallback(category, rule_guidance)
