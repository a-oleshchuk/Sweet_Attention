from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Priority = Literal["urgent", "medium", "low"]


class AnalysisTurnInput(BaseModel):
    conversation_id: str
    domain_key: str = "ai_school"
    refresh: bool = False
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    user_profile: dict[str, Any] = Field(default_factory=dict)


class DetectionSignals(BaseModel):
    user_sentiment: Literal["positive", "neutral", "negative"] = "neutral"
    frustration_detected: bool = False
    repetition_detected: bool = False
    unresolved_problem: bool = False
    support_agent_confusion: bool = False
    support_agent_missed_step: bool = False
    support_agent_possible_error: bool = False
    escalation_needed: bool = False


class SupportValidationFinding(BaseModel):
    issue: str
    severity: Literal["info", "warning", "error"] = "info"


class AnalysisResult(BaseModel):
    conversation_id: str
    domain_key: str
    issue_type: str | None = None
    priority: Priority = "medium"
    summary: str
    user_profile: dict[str, Any] = Field(default_factory=dict)
    detected_issues: list[str] = Field(default_factory=list)
    detection_signals: DetectionSignals = Field(default_factory=DetectionSignals)
    support_agent_findings: list[SupportValidationFinding] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    analysis_metadata: dict[str, Any] = Field(default_factory=dict)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "domain_key": self.domain_key,
            "issue_type": self.issue_type,
            "priority": self.priority,
            "summary": self.summary,
            "user_profile": self.user_profile,
            "detected_issues": self.detected_issues,
            "detection_signals": self.detection_signals.model_dump(),
            "support_agent_findings": [item.model_dump() for item in self.support_agent_findings],
            "next_steps": self.next_steps,
            "analysis_metadata": self.analysis_metadata,
            "conversation_history": self.conversation_history,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "AnalysisResult":
        return cls(
            conversation_id=record["conversation_id"],
            domain_key=record.get("domain_key", "ai_school"),
            issue_type=record.get("issue_type"),
            priority=record.get("priority", "medium"),
            summary=record.get("summary", ""),
            user_profile=record.get("user_profile", {}),
            detected_issues=record.get("detected_issues", []),
            detection_signals=DetectionSignals.model_validate(record.get("detection_signals", {})),
            support_agent_findings=[
                SupportValidationFinding.model_validate(item)
                for item in record.get("support_agent_findings", [])
            ],
            next_steps=record.get("next_steps", []),
            analysis_metadata=record.get("analysis_metadata", {}),
            conversation_history=record.get("conversation_history", []),
        )
