from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "please",
    "show",
    "the",
    "to",
    "with",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9_@.+-]+", _normalize(text)) if token not in STOPWORDS]


def _score_text(query: str, text: str, *, title: str = "") -> int:
    normalized_query = _normalize(query)
    haystack = _normalize(f"{title} {text}")
    tokens = _tokenize(query)
    if not normalized_query:
        return 0

    score = 0
    if normalized_query in haystack:
        score += max(6, len(tokens) + 4)

    for token in tokens:
        if token in haystack:
            score += 2 if len(token) > 3 else 1

    return score


@dataclass(frozen=True)
class CsvSearchHit:
    table: str
    score: int
    row: dict[str, str]


@dataclass(frozen=True)
class KnowledgeHit:
    file_id: str
    category: str
    path: Path
    score: int
    content: str


class CsvRepository:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self._tables = self._load_tables()

    def _load_tables(self) -> dict[str, list[dict[str, str]]]:
        tables: dict[str, list[dict[str, str]]] = {}
        for csv_path in sorted(self.data_root.rglob("*.csv")):
            table_name = ".".join(csv_path.relative_to(self.data_root).with_suffix("").parts)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                tables[table_name] = list(csv.DictReader(handle))
        return tables

    def available_tables(self) -> list[str]:
        return sorted(self._tables)

    def _resolve_tables(self, requested: list[str] | None) -> list[str]:
        if not requested:
            return self.available_tables()

        resolved: list[str] = []
        for item in requested:
            normalized = item.lower().replace("/", ".").replace("\\", ".")
            matches = [
                table_name
                for table_name in self._tables
                if table_name == normalized
                or table_name.endswith(f".{normalized}")
                or table_name.split(".")[-1] == normalized
            ]
            for match in matches:
                if match not in resolved:
                    resolved.append(match)
        return resolved

    def search(self, query: str, *, tables: list[str] | None = None, limit: int = 5) -> list[CsvSearchHit]:
        hits: list[CsvSearchHit] = []
        normalized_query = _normalize(query)
        tokens = _tokenize(query)
        selected_tables = self._resolve_tables(tables)

        for table_name in selected_tables:
            for row in self._tables.get(table_name, []):
                row_text = " ".join(f"{key} {value}" for key, value in row.items())
                score = _score_text(query, row_text, title=table_name)

                for value in row.values():
                    normalized_value = _normalize(str(value))
                    if normalized_value == normalized_query:
                        score += 8
                    elif normalized_query and normalized_query in normalized_value:
                        score += 4
                    elif any(token == normalized_value for token in tokens):
                        score += 3

                if score > 0:
                    hits.append(CsvSearchHit(table=table_name, score=score, row=row))

        hits.sort(key=lambda item: (-item.score, item.table))
        return hits[:limit]


class KnowledgeRepository:
    def __init__(self, knowledge_root: Path):
        self.knowledge_root = knowledge_root
        self._files = self._load_files()

    def _load_files(self) -> dict[str, KnowledgeHit]:
        files: dict[str, KnowledgeHit] = {}
        for md_path in sorted(self.knowledge_root.rglob("*.md")):
            if md_path.name.lower() == "readme.md":
                continue
            relative = md_path.relative_to(self.knowledge_root)
            category = relative.parts[0]
            file_id = ".".join(relative.with_suffix("").parts)
            content = md_path.read_text(encoding="utf-8")
            files[file_id] = KnowledgeHit(
                file_id=file_id,
                category=category,
                path=md_path,
                score=0,
                content=content,
            )
        return files

    def available_files(self, *, categories: list[str] | None = None) -> list[str]:
        selected = self._resolve_categories(categories)
        return sorted(file_id for file_id, hit in self._files.items() if hit.category in selected)

    def _resolve_categories(self, categories: list[str] | None) -> set[str]:
        if not categories:
            return {hit.category for hit in self._files.values()}
        return {category.lower() for category in categories}

    def search(self, query: str, *, categories: list[str] | None = None, limit: int = 2) -> list[KnowledgeHit]:
        hits: list[KnowledgeHit] = []
        selected = self._resolve_categories(categories)

        for hit in self._files.values():
            if hit.category not in selected:
                continue
            score = _score_text(query, hit.content, title=hit.file_id)
            if score > 0:
                hits.append(
                    KnowledgeHit(
                        file_id=hit.file_id,
                        category=hit.category,
                        path=hit.path,
                        score=score,
                        content=hit.content,
                    )
                )

        hits.sort(key=lambda item: (-item.score, item.file_id))
        return hits[:limit]
