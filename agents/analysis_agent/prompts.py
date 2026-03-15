from __future__ import annotations

from .contracts import CATEGORY_INTENT_OPTIONS


def build_issue_classification_system_prompt() -> str:
    sections = [
        "You label support dialogues for a support worker.",
        "Choose exactly one category and exactly one intent.",
        "The intent must belong to the chosen category.",
        "Use only the allowed options below.",
        "",
        "Allowed categories and intents:",
    ]
    for category, intents in CATEGORY_INTENT_OPTIONS.items():
        sections.append(f"- {category}")
        for intent in sorted(intents):
            sections.append(f"  - {intent}")
    sections.extend(
        [
            "",
            "Important:",
            "- Base the label on the actual user issue in the dialogue, not on random phrases.",
            "- Prefer the most specific valid intent.",
            "- If the case does not clearly fit a specific option, choose category 'general' and intent 'general_support_request'.",
            "- Return only the structured category and intent fields.",
        ]
    )
    return "\n".join(sections)


def build_issue_classification_user_prompt(
    *,
    active_issue: str,
    latest_user_message: str,
    dialogue_summary: str,
) -> str:
    return "\n".join(
        [
            f"Primary user issue: {active_issue or 'Unknown'}",
            f"Latest user message: {latest_user_message or 'Unknown'}",
            f"Dialogue summary: {dialogue_summary or 'Unknown'}",
        ]
    )
