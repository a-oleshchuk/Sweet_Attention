from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AnalysisAgentSettings(BaseModel):
    project_root: Path = PROJECT_ROOT
    domain_config_path: Path = PROJECT_ROOT / "domains.yaml"
    default_domain_key: str = "ai_school"
    openai_model: str = Field(default="gpt-4.1")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    checkpoint_path: Path = PROJECT_ROOT / ".langgraph" / "analysis_agent.sqlite"
    support_checkpoint_path: Path = PROJECT_ROOT / ".langgraph" / "support_agent.sqlite"
    analysis_results_path: Path = PROJECT_ROOT / ".langgraph" / "analysis_results.sqlite"
    max_history_messages: int = Field(default=30, ge=1, le=200)
    data_search_limit: int = Field(default=5, ge=1, le=20)
    knowledge_search_limit: int = Field(default=2, ge=1, le=5)

    @classmethod
    def from_env(cls) -> "AnalysisAgentSettings":
        checkpoint = os.getenv("ANALYSIS_AGENT_CHECKPOINTER")
        support_checkpoint = os.getenv("SUPPORT_AGENT_CHECKPOINTER")
        domain_config = os.getenv("ANALYSIS_AGENT_DOMAIN_CONFIG")
        return cls(
            domain_config_path=Path(domain_config) if domain_config else PROJECT_ROOT / "domains.yaml",
            openai_model=os.getenv("ANALYSIS_AGENT_MODEL", "gpt-4.1"),
            temperature=float(os.getenv("ANALYSIS_AGENT_TEMPERATURE", "0.0")),
            checkpoint_path=Path(checkpoint) if checkpoint else PROJECT_ROOT / ".langgraph" / "analysis_agent.sqlite",
            support_checkpoint_path=Path(support_checkpoint)
            if support_checkpoint
            else PROJECT_ROOT / ".langgraph" / "support_agent.sqlite",
            analysis_results_path=Path(os.getenv("ANALYSIS_RESULTS_DB", PROJECT_ROOT / ".langgraph" / "analysis_results.sqlite")),
            data_search_limit=int(os.getenv("ANALYSIS_DATA_SEARCH_LIMIT", "5")),
            knowledge_search_limit=int(os.getenv("ANALYSIS_KNOWLEDGE_SEARCH_LIMIT", "2")),
        )

    def ensure_runtime_dirs(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.support_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.analysis_results_path.parent.mkdir(parents=True, exist_ok=True)
