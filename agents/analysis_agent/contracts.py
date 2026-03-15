from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.support_agent.contracts import Tone, ToolResultRecord, WorkerInstruction


class AnalysisConversationStatus(str, Enum):
    NORMAL = "normal"
    WATCH = "watch"
    ESCALATED = "escalated"
    WAITING_FOR_WORKER = "waiting_for_worker"
    WORKER_REVIEWING = "worker_reviewing"
    RESOLVED = "resolved"


class DialogueMessage(BaseModel):
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SupportAgentSnapshot(BaseModel):
    model_config = ConfigDict(extra="allow")

    active_user_issue: str | None = None
    known_user_identifiers: dict[str, str] = Field(default_factory=dict)
    last_tool_results: list[ToolResultRecord] = Field(default_factory=list)
    open_worker_instruction: dict[str, Any] | None = None
    pending_review: bool = False
    draft_message: str | None = None
    assistant_message: str | None = None
    message_history: list[dict[str, Any]] = Field(default_factory=list)


class WorkerDecision(BaseModel):
    action: str
    notes: list[str] = Field(default_factory=list)
    mark_resolved: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("notes", mode="before")
    @classmethod
    def _normalize_notes(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]


class AnalysisReason(BaseModel):
    code: str
    description: str
    evidence: list[str] = Field(default_factory=list)


class WorkerEscalationPackage(BaseModel):
    full_dialogue: list[DialogueMessage]
    dialogue_summary: str
    reason_of_escalation: list[AnalysisReason]
    possible_next_steps: list[str]
    recommended_tone: Tone
    suggested_constraints: list[str]
    suggested_worker_instruction: WorkerInstruction


class AnalysisTurnInput(BaseModel):
    conversation_id: str
    full_dialogue_snapshot: list[DialogueMessage]
    support_agent_state: SupportAgentSnapshot | dict[str, Any]
    worker_decision_update: WorkerDecision | None = None
    domain_key: str = "ai_school"

    @field_validator("support_agent_state", mode="before")
    @classmethod
    def _normalize_support_state(cls, value: Any) -> SupportAgentSnapshot:
        if isinstance(value, SupportAgentSnapshot):
            return value
        return SupportAgentSnapshot.model_validate(value or {})


class AnalysisTurnOutput(BaseModel):
    conversation_id: str
    conversation_status: AnalysisConversationStatus
    needs_escalation: bool
    paused: bool
    reasons: list[AnalysisReason] = Field(default_factory=list)
    dialogue_summary: str
    possible_next_steps: list[str] = Field(default_factory=list)
    recommended_tone: Tone
    suggested_constraints: list[str] = Field(default_factory=list)
    suggested_worker_instruction: WorkerInstruction | None = None
    worker_package: WorkerEscalationPackage | None = None

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "AnalysisTurnOutput":
        return cls(
            conversation_id=state["conversation_id"],
            conversation_status=state["conversation_status"],
            needs_escalation=bool(state.get("needs_escalation")),
            paused=bool(state.get("paused")),
            reasons=[AnalysisReason.model_validate(item) for item in state.get("current_reasons", [])],
            dialogue_summary=state.get("dialogue_summary", ""),
            possible_next_steps=list(state.get("possible_next_steps", [])),
            recommended_tone=Tone(state.get("recommended_tone", Tone.FORMAL.value)),
            suggested_constraints=list(state.get("suggested_constraints", [])),
            suggested_worker_instruction=WorkerInstruction.model_validate(state["suggested_worker_instruction"])
            if state.get("suggested_worker_instruction")
            else None,
            worker_package=WorkerEscalationPackage.model_validate(state["worker_package"])
            if state.get("worker_package")
            else None,
        )

