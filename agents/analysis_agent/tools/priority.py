from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool


def build_priority_tool():
    @tool
    def priority_evaluation_tool(detection_signals: dict[str, Any]) -> str:
        """Estimate case priority from detection signals."""
        priority = "low"
        rationale: list[str] = []
        if detection_signals.get("escalation_needed") or detection_signals.get("support_agent_possible_error"):
            priority = "urgent"
            rationale.append("escalation or agent error flagged")
        elif detection_signals.get("unresolved_problem") or detection_signals.get("frustration_detected"):
            priority = "medium"
            rationale.append("unresolved problem or user frustration detected")

        if detection_signals.get("repetition_detected") and priority == "low":
            priority = "medium"
            rationale.append("user repeating requests")

        if detection_signals.get("user_sentiment") == "negative" and priority != "urgent":
            priority = "medium"
            rationale.append("negative sentiment")

        payload = {"priority": priority, "rationale": rationale}
        return json.dumps(payload)

    return priority_evaluation_tool
