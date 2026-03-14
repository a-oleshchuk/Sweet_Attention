from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ..prompts import build_next_steps_prompt


def build_next_steps_tool(model: ChatOpenAI):
    prompt = build_next_steps_prompt()

    @tool
    def next_steps_recommendation_tool(summary: str, priority: str, detected_issues: list[str]) -> str:  # type: ignore
        """Suggest next steps for the support manager."""
        context = f"Priority: {priority}\nDetected issues: {detected_issues}\nSummary: {summary}"
        response = model.invoke([SystemMessage(content=prompt), SystemMessage(content=context)])
        content = response.content if hasattr(response, "content") else str(response)
        try:
            payload = json.loads(content)
        except Exception:
            payload = [
                "Review and send a personal follow-up to the user",
                "Ask clarifying questions",
                "Verify account data before next reply",
            ]
        if isinstance(payload, dict):
            payload = list(payload.values())
        return json.dumps(payload)

    return next_steps_recommendation_tool
