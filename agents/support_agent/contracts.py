from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Tone(str, Enum):
    FRIENDLY_AND_CALM = "friendly_and_calm"
    FORMAL = "formal"
    SHORT_AND_NEUTRAL = "short_and_neutral"


class WorkerInstruction(BaseModel):
    task: str
    constraints: list[str] = Field(default_factory=list)
    answer_format: str | None = None
    tone: Tone | None = None
    review_before_send: bool = False
    regenerate: bool = False
    notes: list[str] = Field(default_factory=list)

    @field_validator("constraints", "notes", mode="before")
    @classmethod
    def _normalize_string_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]


class UserMetadata(BaseModel):
    display_name: str
    known_identifiers: dict[str, str] = Field(default_factory=dict)


class ToolResultRecord(BaseModel):
    tool_name: str
    query: str | None = None
    result_count: int | None = None
    content: str


class MessageHistoryEntry(BaseModel):
    role: Literal["user", "assistant", "assistant_draft", "tool", "support_worker_instruction"]
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SupportTurnInput(BaseModel):
    conversation_id: str
    user_metadata: UserMetadata
    user_message: str | None = None
    worker_instruction: WorkerInstruction | None = None
    domain_key: str = "ai_school"

    @model_validator(mode="after")
    def _require_turn_driver(self) -> "SupportTurnInput":
        if not self.user_message and not self.worker_instruction:
            raise ValueError("A turn requires either user_message or worker_instruction.")
        return self


class SupportTurnOutput(BaseModel):
    conversation_id: str
    send_target: Literal["user", "support_worker"]
    assistant_message: str
    pending_review: bool
    draft_message: str | None = None
    active_user_issue: str | None = None
    known_user_identifiers: dict[str, str] = Field(default_factory=dict)
    last_tool_results: list[ToolResultRecord] = Field(default_factory=list)
    message_history: list[MessageHistoryEntry] = Field(default_factory=list)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "SupportTurnOutput":
        return cls(
            conversation_id=state["conversation_id"],
            send_target=state["send_target"],
            assistant_message=state["assistant_message"],
            pending_review=bool(state.get("pending_review")),
            draft_message=state.get("draft_message"),
            active_user_issue=state.get("active_user_issue"),
            known_user_identifiers=state.get("known_user_identifiers", {}),
            last_tool_results=[ToolResultRecord.model_validate(item) for item in state.get("last_tool_results", [])],
            message_history=[MessageHistoryEntry.model_validate(item) for item in state.get("message_history", [])],
        )

