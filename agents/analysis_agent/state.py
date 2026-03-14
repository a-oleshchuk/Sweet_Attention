from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AnalysisAgentState(TypedDict, total=False):
    conversation_id: str
    domain_key: str
    conversation_history: list[dict[str, Any]]
    issue_type: str | None
    detection_signals: dict[str, Any]
    support_agent_findings: list[dict[str, Any]]
    priority: str | None
    summary: str | None
    next_steps: list[str]
    user_profile: dict[str, Any]
    analysis_metadata: dict[str, Any]
    last_tool_results: list[dict[str, Any]]
    turn_messages: Annotated[list[AnyMessage], add_messages]
    formatted_history: str
