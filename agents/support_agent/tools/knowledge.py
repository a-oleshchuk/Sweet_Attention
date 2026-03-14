from __future__ import annotations

import json

from langchain_core.tools import tool

from ..repositories import KnowledgeRepository


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

