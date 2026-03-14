from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ..prompts import build_summary_prompt


def build_summary_tool(model: ChatOpenAI, *, domain_context: str):
    @tool
    def summary_tool(
        conversation_history: str,
        issue_type: str | None,
        detected_issues: list[str],
        user_profile: dict[str, Any] | None = None,
    ) -> str:  # type: ignore
        """Generate a structured summary for the support manager."""
        prompt = build_summary_prompt(domain_context=domain_context, user_profile=user_profile or {})
        messages = [SystemMessage(content=prompt)]
        summary_input = (
            f"Issue type: {issue_type}\nDetected issues: {detected_issues}\nUser profile: {user_profile}\n"
            f"Conversation history:\n{conversation_history}"
        )
        response = model.invoke(messages + [SystemMessage(content=summary_input)])
        content = response.content if hasattr(response, "content") else str(response)
        try:
            payload = json.loads(content)
        except Exception:
            payload = {
                "case_summary": content.strip(),
                "user_problem": issue_type or "unspecified",
                "steps_attempted": "Not available",
                "current_status": "Needs review",
                "why_attention": "Auto-generated summary fallback",
            }
        return json.dumps(payload)

    return summary_tool
