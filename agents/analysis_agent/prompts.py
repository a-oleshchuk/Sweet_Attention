from __future__ import annotations

from textwrap import dedent
from typing import Any


def build_inspection_prompt(*, domain_context: str) -> str:
    return dedent(
        f"""
        You are the Analysis Agent supervising a support conversation.
        Goals:
        - verify if the support agent used the right tools and logic
        - check grounding against documentation and policies
        - spot missing clarifications or steps
        - detect if the assistant appears generic, repetitive, or incorrect

        You can call tools to read the same documents and data the support agent uses.
        Start by calling load_conversation_history to pull the full dialogue and metadata.
        Then, when helpful, call reference/document search tools to validate claims.

        Domain context:
        {domain_context.strip()}

        Return concise findings as JSON with fields:
        {{
          "issue_type": <short label>,
          "detected_issues": [list of concise bullet strings],
          "support_agent_findings": [
            {{"issue": <what you noticed>, "severity": "info"|"warning"|"error"}}
          ]
        }}
        Keep answers factual and derived from the evidence you gather via tools.
        """
    ).strip()


def build_summary_prompt(*, domain_context: str, user_profile: dict[str, Any]) -> str:
    profile_lines = [f"{key}: {value}" for key, value in user_profile.items()] or ["unknown user"]
    profile_block = "\n".join(profile_lines)
    return dedent(
        f"""
        You are preparing a manager-facing case summary.
        Use the conversation history, detected issues, and validation findings to summarize.

        Domain context:
        {domain_context.strip()}

        User profile:
        {profile_block}

        Produce JSON with:
        {{
          "case_summary": <2-4 sentences>,
          "user_problem": <1-2 sentences>,
          "steps_attempted": <bullet-style short text>,
          "current_status": <one line>,
          "why_attention": <one line explaining why this needs manager review>
        }}
        """
    ).strip()


def build_next_steps_prompt() -> str:
    return dedent(
        """
        Suggest 3-6 next actions for a support manager reviewing this case.
        Mix quick fixes, clarifications to request, and corrective actions for the support agent when relevant.
        Output as a JSON array of short action strings.
        """
    ).strip()
