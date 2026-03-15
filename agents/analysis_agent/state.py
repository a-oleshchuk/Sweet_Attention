from __future__ import annotations

from typing import Any, TypedDict


class AnalysisAgentState(TypedDict, total=False):
    conversation_id: str
    domain_key: str
    full_dialogue_snapshot: list[dict[str, Any]]
    support_agent_state: dict[str, Any]
    last_analysis_result: dict[str, Any] | None
    escalation_history: list[dict[str, Any]]
    worker_decisions: list[dict[str, Any]]
    conversation_status: str
    incoming_full_dialogue_snapshot: list[dict[str, Any]] | None
    incoming_support_agent_state: dict[str, Any] | None
    incoming_worker_decision: dict[str, Any] | None
    worker_decision_applied: bool
    current_reasons: list[dict[str, Any]]
    current_reason_events: list[dict[str, Any]]
    needs_escalation: bool
    paused: bool
    dialogue_category: str
    dialogue_intent: str
    dialogue_summary: str
    possible_next_steps: list[str]
    recommended_tone: str
    suggested_constraints: list[str]
    suggested_worker_instruction: dict[str, Any] | None
    worker_package: dict[str, Any] | None
    verification_evidence: list[dict[str, Any]]
    last_user_message: str | None
    last_support_message: str | None
    last_user_index: int | None
    last_support_index: int | None
