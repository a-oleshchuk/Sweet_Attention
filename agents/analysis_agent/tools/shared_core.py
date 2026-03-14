from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from agents.support_agent.repositories import CsvRepository, KnowledgeRepository
from agents.support_agent.tools import build_support_tools

from ..repositories import SupportConversationRepository


ISSUE_KEYWORDS = {
    "billing": ["invoice", "charge", "payment", "bill", "refund", "card", "charged"],
    "subscription": ["cancel", "upgrade", "downgrade", "subscription", "plan", "pause"],
    "access": ["login", "password", "access", "reset", "portal", "signin", "sign in"],
    "lessons": ["lesson", "class", "session", "reschedule", "recording", "attendance"],
    "technical": ["bug", "error", "issue", "crash", "not working", "loading"],
}


def _detect_issue_type(text: str) -> str:
    lower = text.lower()
    for label, keywords in ISSUE_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            return label
    return "general"


def build_shared_core_tools(
    *,
    csv_repository: CsvRepository,
    knowledge_repository: KnowledgeRepository,
    conversation_repository: SupportConversationRepository,
    data_search_limit: int,
    knowledge_search_limit: int,
):
    support_tools = build_support_tools(
        csv_repository=csv_repository,
        knowledge_repository=knowledge_repository,
        data_search_limit=data_search_limit,
        knowledge_search_limit=knowledge_search_limit,
    )

    @tool
    def load_conversation_history(conversation_id: str, limit: int | None = None) -> str:
        """Load the full conversation history and user identifiers for a given conversation_id."""
        messages, profile = conversation_repository.load(conversation_id)
        limited = messages[-limit:] if limit else messages
        payload = {
            "conversation_id": conversation_id,
            "messages": limited,
            "user_profile": profile,
            "message_count": len(limited),
        }
        return json.dumps(payload)

    @tool
    def issue_classifier(conversation_text: str) -> str:
        """Classify the issue type based on conversation text."""
        issue_type = _detect_issue_type(conversation_text)
        payload: dict[str, Any] = {
            "issue_type": issue_type,
            "detected_keywords": [
                keyword
                for keyword_list in ISSUE_KEYWORDS.values()
                for keyword in keyword_list
                if keyword in conversation_text.lower()
            ],
        }
        return json.dumps(payload)

    return support_tools + [load_conversation_history, issue_classifier]
