from __future__ import annotations

from typing import Any

from .contracts import Tone, WorkerInstruction


TONE_GUIDANCE = {
    Tone.FORMAL: "Use a formal, clear, and professional support tone.",
    Tone.FRIENDLY_AND_CALM: "Use a friendly, calm, and reassuring support tone.",
    Tone.SHORT_AND_NEUTRAL: "Use a short, neutral, and efficient support tone.",
}


BASE_RULES = """You are the support agent in a supervised customer-support system.
You are the only agent that talks to the customer.

Operational rules:
- Use tools when facts depend on account, billing, access, lesson, or support history data.
- Use policy files for rules and exception boundaries.
- Use reference files for product and how-to guidance.
- Ask a clarifying question if the available information is not enough.
- Never invent missing facts.
- Never mention CSV files, markdown files, policies, tools, or internal sources in customer-facing replies.
- Keep replies simple and understandable.
- Do not mention escalation, the analysis agent, or the support worker unless a support-worker instruction explicitly requires customer-facing wording.
- The analysis agent decides escalation. You do not decide it.
"""


def _format_identifiers(identifiers: dict[str, str]) -> str:
    if not identifiers:
        return "- none"
    return "\n".join(f"- {key}: {value}" for key, value in sorted(identifiers.items()))


def _format_worker_instruction(instruction: WorkerInstruction | None) -> str:
    if instruction is None:
        return "No support-worker instruction for this turn."

    constraints = instruction.constraints or ["none"]
    notes = instruction.notes or ["none"]
    return "\n".join(
        [
            "Support-worker instruction for the next reply only:",
            f"- task: {instruction.task}",
            f"- constraints: {', '.join(constraints)}",
            f"- answer_format: {instruction.answer_format or 'default conversational reply'}",
            f"- tone override: {instruction.tone.value if instruction.tone else 'use default tone'}",
            f"- review_before_send: {instruction.review_before_send}",
            f"- regenerate: {instruction.regenerate}",
            f"- notes: {', '.join(notes)}",
            "- Follow this instruction for the next generated reply only.",
            "- Write the reply in customer-ready language even if it is being routed back for worker review.",
        ]
    )


def build_system_prompt(
    *,
    domain_context: str,
    tooling_overview: str,
    state: dict[str, Any],
) -> str:
    instruction = None
    if state.get("open_worker_instruction"):
        instruction = WorkerInstruction.model_validate(state["open_worker_instruction"])

    resolved_tone = instruction.tone if instruction and instruction.tone else Tone.FORMAL
    previous_draft = state.get("previous_draft_message")
    identifiers = dict(state.get("known_user_identifiers", {}))
    display_name = state.get("user_display_name")
    if display_name and "display_name" not in identifiers:
        identifiers["display_name"] = display_name

    sections = [
        BASE_RULES,
        domain_context.strip(),
        TONE_GUIDANCE[resolved_tone],
        f"Active user issue: {state.get('active_user_issue') or 'unknown'}",
        "Known user identifiers:\n" + _format_identifiers(identifiers),
        _format_worker_instruction(instruction),
        "Available retrieval scope:\n" + tooling_overview.strip(),
    ]

    if instruction and instruction.regenerate and previous_draft:
        sections.append("Previous draft to replace:\n" + previous_draft)

    return "\n\n".join(sections)

