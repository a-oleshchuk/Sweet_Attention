from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


SendTarget = Literal["user", "support_worker"]


class SupportAgentState(TypedDict, total=False):
    conversation_id: str
    domain_key: str
    user_display_name: str | None
    known_user_identifiers: dict[str, str]
    active_user_issue: str | None
    last_tool_results: list[dict[str, Any]]
    open_worker_instruction: dict[str, Any] | None
    pending_review: bool
    draft_message: str | None
    previous_draft_message: str | None
    assistant_message: str | None
    send_target: SendTarget | None
    message_history: list[dict[str, Any]]
    conversation_messages: Annotated[list[AnyMessage], add_messages]
    turn_messages: Annotated[list[AnyMessage], add_messages]
    incoming_user_message: str | None
    incoming_worker_instruction: dict[str, Any] | None
    incoming_user_metadata: dict[str, Any] | None

