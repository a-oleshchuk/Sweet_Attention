from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from agents.support_agent.contracts import Tone


class DefaultInstructionPolicy(BaseModel):
    review_before_send: bool = True
    regenerate: bool = True
    answer_format: str = "short clear reply to user"


class ThresholdPolicy(BaseModel):
    escalate_after_support_replies_on_same_issue: int = 3
    escalate_after_repeated_clarifying_questions: int = 5
    escalate_after_total_turns_without_resolution: int = 12
    watch_after_support_replies_on_same_issue: int = 2
    watch_after_repeated_clarifying_questions: int = 3
    watch_after_total_turns_without_resolution: int = 8
    confusing_message_word_count: int = 220


class PatternPolicy(BaseModel):
    user_resolution_phrases: list[str] = Field(default_factory=list)
    explicit_dissatisfaction_phrases: list[str] = Field(default_factory=list)
    repeated_complaint_markers: list[str] = Field(default_factory=list)
    clarifying_question_markers: list[str] = Field(default_factory=list)
    factual_claim_markers: list[str] = Field(default_factory=list)
    policy_guidance_markers: list[str] = Field(default_factory=list)
    sensitive_data_patterns: list[str] = Field(default_factory=list)
    internal_detail_patterns: list[str] = Field(default_factory=list)
    rude_patterns: list[str] = Field(default_factory=list)
    blaming_patterns: list[str] = Field(default_factory=list)
    unsupported_action_patterns: list[str] = Field(default_factory=list)
    account_specific_issue_markers: list[str] = Field(default_factory=list)


class AnalysisPolicy(BaseModel):
    require_worker_confirmation_on_resolution: bool = False
    default_output_tone: Tone = Tone.FORMAL
    default_instruction: DefaultInstructionPolicy = Field(default_factory=DefaultInstructionPolicy)
    thresholds: ThresholdPolicy = Field(default_factory=ThresholdPolicy)
    patterns: PatternPolicy = Field(default_factory=PatternPolicy)


def load_analysis_policy(*, project_root: Path, config_path: Path | None = None) -> AnalysisPolicy:
    resolved_path = config_path or project_root / "analysis_rules.yaml"
    raw_payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    return AnalysisPolicy.model_validate(raw_payload.get("analysis", {}))
