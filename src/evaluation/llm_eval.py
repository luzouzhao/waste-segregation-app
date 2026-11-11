"""
Day 8 — AI Assistant evaluation.
Tests generate_explanation() on fixed prompts with and without an API key.
Checks that output is non-empty and contains expected keywords.
Reports pass/fail per test case.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.assistant.llm_guidance import generate_explanation
from src.recommendation.rules import get_guidance


TEST_CASES = [
    {
        "category": "plastic",
        "must_contain": ["plastic", "recycl", "bin"],
    },
    {
        "category": "hazardous",
        "must_contain": ["hazardous", "collection", "dispos"],
    },
    {
        "category": "electronic",
        "must_contain": ["electronic", "waste", "collection"],
    },
    {
        "category": "organic",
        "must_contain": ["organic", "compost"],
    },
    {
        "category": "glass",
        "must_contain": ["glass", "recycl", "bin"],
    },
]


def _check_keywords(text: str, keywords: list[str]) -> list[str]:
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() not in text_lower]


def main() -> None:
    report_dir = Path(__file__).resolve().parent.parent.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "llm_evaluation.md"

    lines = ["# AI Assistant Evaluation Report\n\n"]
    lines.append("## Test Setup\n\n")
    lines.append(f"- {len(TEST_CASES)} test cases across waste categories\n")
    lines.append("- Each test checks output is non-empty and contains expected keywords\n")
    lines.append("- Tests run twice: with API key (if available) and without (fallback)\n\n")

    api_key = os.environ.get("GROQ_API_KEY")
    modes = [
        ("with API key" if api_key else "without API key (fallback)", True),
        ("without API key (fallback)", False),
    ]

    if api_key:
        lines.append(f"GROQ_API_KEY found — testing LLM mode + fallback mode.\n\n")

    all_pass = True
    for mode_label, use_key in modes:
        if not use_key:
            if api_key:
                del os.environ["GROQ_API_KEY"]
        elif not api_key:
            lines.append(f"### Mode: {mode_label}\n\n")
            lines.append("Skipped — no API key set.\n\n")
            continue

        lines.append(f"### Mode: {mode_label}\n\n")
        lines.append("| # | Category | Output Empty? | Missing Keywords | Result |\n")
        lines.append("|---|----------|--------------|-----------------|--------|\n")

        for i, tc in enumerate(TEST_CASES, 1):
            cat = tc["category"]
            guidance = get_guidance(cat)
            try:
                result = generate_explanation(cat, guidance)
            except Exception as e:
                result = ""
                print(f"  Test {i} crashed: {e}")

            empty = len(result.strip()) == 0
            missing = _check_keywords(result, tc["must_contain"]) if not empty else tc["must_contain"]
            passed = not empty and len(missing) == 0

            lines.append(
                f"| {i} | {cat} | {'yes' if empty else 'no'} | "
                f"{', '.join(missing) if missing else 'none'} | "
                f"{'✅ PASS' if passed else '❌ FAIL'} |\n"
            )
            if not passed:
                all_pass = False

    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    lines.append(f"\n## Summary\n\n")
    lines.append(f"**Overall result:** {'✅ ALL PASS' if all_pass else '❌ SOME TESTS FAILED'}\n")

    report_path.write_text("".join(lines))
    print(f"AI assistant evaluation report written to {report_path}")
    print(f"Result: {'✅ ALL PASS' if all_pass else '❌ SOME TESTS FAILED'}")


if __name__ == "__main__":
    main()
