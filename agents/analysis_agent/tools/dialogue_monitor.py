from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


MONITOR_PROMPT = ChatPromptTemplate.from_template(
    """
    You are monitoring a support conversation.
    Use the transcript below to produce monitoring signals.

    Conversation:
    {conversation}

    Latest support reply:
    {last_support_message}

    Return JSON with keys:
    - user_sentiment: positive | neutral | negative
    - frustration_detected: bool
    - repetition_detected: bool
    - unresolved_problem: bool
    - support_agent_confusion: bool
    - support_agent_missed_step: bool
    - support_agent_possible_error: bool
    - escalation_needed: bool
    """
)


def build_dialogue_monitor_tool(model: ChatOpenAI):
    @tool
    def dialogue_monitor_tool(conversation: str, last_support_message: str | None = None) -> str:
        """Monitor a conversation transcript and emit quality and risk signals."""
        message = MONITOR_PROMPT.format_messages(
            conversation=conversation,
            last_support_message=last_support_message or "",
        )
        response = model.invoke(message)
        content = response.content if hasattr(response, "content") else str(response)
        try:
            payload = json.loads(content)
        except Exception:
            payload = {
                "user_sentiment": "neutral",
                "frustration_detected": "frustrated" in content.lower() or "angry" in content.lower(),
                "repetition_detected": False,
                "unresolved_problem": True,
                "support_agent_confusion": "not sure" in content.lower(),
                "support_agent_missed_step": "missed" in content.lower(),
                "support_agent_possible_error": "incorrect" in content.lower() or "wrong" in content.lower(),
                "escalation_needed": "escalate" in content.lower(),
            }
        return json.dumps(payload)

    return dialogue_monitor_tool
