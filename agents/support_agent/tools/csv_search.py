from __future__ import annotations

import json

from langchain_core.tools import tool

from ..repositories import CsvRepository


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

