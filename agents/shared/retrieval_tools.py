from __future__ import annotations

import json

from langchain_core.tools import tool

from .repositories import CsvRepository, KnowledgeRepository


def build_query_data_records_tool(repository: CsvRepository, default_limit: int):
    @tool
    def query_data_records(query: str, tables: list[str] | None = None, limit: int = default_limit) -> str:
        """Search CSV-backed operational records using a natural-language query."""
        results = repository.search(query, tables=tables, limit=limit)
        payload = {
            "query": query,
            "tables": tables or repository.available_tables(),
            "result_count": len(results),
            "results": [{"table": item.table, "score": item.score, "row": item.row} for item in results],
        }
        return json.dumps(payload)

    return query_data_records


def build_read_reference_files_tool(repository: KnowledgeRepository, default_limit: int):
    @tool
    def read_reference_files(query: str, limit: int = default_limit) -> str:
        """Search reference docs and how-to guides and return whole-file content for the best matches."""
        results = repository.search(query, categories=["docs", "how_to"], limit=limit)
        payload = {
            "query": query,
            "categories": ["docs", "how_to"],
            "result_count": len(results),
            "results": [
                {
                    "file_id": item.file_id,
                    "category": item.category,
                    "score": item.score,
                    "content": item.content,
                }
                for item in results
            ],
        }
        return json.dumps(payload)

    return read_reference_files


def build_read_pattern_files_tool(repository: KnowledgeRepository, default_limit: int):
    @tool
    def read_pattern_files(query: str, limit: int = default_limit) -> str:
        """Search common-case support patterns and return whole-file content for the best matches."""
        results = repository.search(query, categories=["patterns"], limit=limit)
        payload = {
            "query": query,
            "categories": ["patterns"],
            "result_count": len(results),
            "results": [
                {
                    "file_id": item.file_id,
                    "category": item.category,
                    "score": item.score,
                    "content": item.content,
                }
                for item in results
            ],
        }
        return json.dumps(payload)

    return read_pattern_files


def build_read_policy_files_tool(repository: KnowledgeRepository, default_limit: int):
    @tool
    def read_policy_files(query: str, limit: int = default_limit) -> str:
        """Search policy files and return whole-file content for the best matches."""
        results = repository.search(query, categories=["policies"], limit=limit)
        payload = {
            "query": query,
            "categories": ["policies"],
            "result_count": len(results),
            "results": [
                {
                    "file_id": item.file_id,
                    "category": item.category,
                    "score": item.score,
                    "content": item.content,
                }
                for item in results
            ],
        }
        return json.dumps(payload)

    return read_policy_files


def build_retrieval_tools(
    *,
    csv_repository: CsvRepository,
    knowledge_repository: KnowledgeRepository,
    data_search_limit: int,
    knowledge_search_limit: int,
):
    return [
        build_query_data_records_tool(csv_repository, data_search_limit),
        build_read_pattern_files_tool(knowledge_repository, knowledge_search_limit),
        build_read_reference_files_tool(knowledge_repository, knowledge_search_limit),
        build_read_policy_files_tool(knowledge_repository, knowledge_search_limit),
    ]
