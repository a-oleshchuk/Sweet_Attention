from __future__ import annotations

from typing import Any, List

from langchain_openai import ChatOpenAI

from agents.support_agent.repositories import CsvRepository, KnowledgeRepository

from ..repositories import SupportConversationRepository
from .dialogue_monitor import build_dialogue_monitor_tool
from .next_steps import build_next_steps_tool
from .priority import build_priority_tool
from .shared_core import build_shared_core_tools
from .summary import build_summary_tool


def build_analysis_tools(
    *,
    csv_repository: CsvRepository,
    knowledge_repository: KnowledgeRepository,
    conversation_repository: SupportConversationRepository,
    data_search_limit: int,
    knowledge_search_limit: int,
    analysis_model: ChatOpenAI,
    domain_context: str,
) -> List[Any]:
    shared_tools = build_shared_core_tools(
        csv_repository=csv_repository,
        knowledge_repository=knowledge_repository,
        conversation_repository=conversation_repository,
        data_search_limit=data_search_limit,
        knowledge_search_limit=knowledge_search_limit,
    )

    monitor_tool = build_dialogue_monitor_tool(analysis_model)
    priority_tool = build_priority_tool()
    summary_tool = build_summary_tool(analysis_model, domain_context=domain_context)
    next_steps_tool = build_next_steps_tool(analysis_model)

    return shared_tools + [monitor_tool, priority_tool, summary_tool, next_steps_tool]
