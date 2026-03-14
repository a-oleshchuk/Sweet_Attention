from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DomainConfig:
    key: str
    display_name: str
    data_root: Path
    knowledge_root: Path
    system_context: str


def get_domain_config(domain_key: str, project_root: Path) -> DomainConfig:
    from .ai_school import build_ai_school_config

    if domain_key == "ai_school":
        return build_ai_school_config(project_root)
    raise ValueError(f"Unsupported domain: {domain_key}")
