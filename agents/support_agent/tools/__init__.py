from __future__ import annotations

from ..repositories import CsvRepository, KnowledgeRepository
from .csv_search import build_query_data_records_tool
from .knowledge import build_read_policy_files_tool, build_read_reference_files_tool


def build_support_tools(
    *,
    csv_repository: CsvRepository,
    knowledge_repository: KnowledgeRepository,
    data_search_limit: int,
    knowledge_search_limit: int,
):
    return [
        build_query_data_records_tool(csv_repository, data_search_limit),
        build_read_reference_files_tool(knowledge_repository, knowledge_search_limit),
        build_read_policy_files_tool(knowledge_repository, knowledge_search_limit),
    ]
