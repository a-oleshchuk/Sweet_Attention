from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from agents.shared import load_project_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AnalysisAgentSettings(BaseModel):
    project_root: Path = PROJECT_ROOT
    domain_config_path: Path = PROJECT_ROOT / "domains.yaml"
    analysis_policy_path: Path = PROJECT_ROOT / "analysis_rules.yaml"
    default_domain_key: str = "ai_school"
    checkpoint_path: Path = PROJECT_ROOT / ".langgraph" / "analysis_agent.sqlite"
    data_search_limit: int = Field(default=3, ge=1, le=20)
    knowledge_search_limit: int = Field(default=1, ge=1, le=5)

    @classmethod
    def from_env(cls) -> "AnalysisAgentSettings":
        load_project_env(PROJECT_ROOT)
        checkpoint = os.getenv("ANALYSIS_AGENT_CHECKPOINTER")
        domain_config = os.getenv("ANALYSIS_AGENT_DOMAIN_CONFIG")
        policy_path = os.getenv("ANALYSIS_AGENT_RULES_CONFIG")
        return cls(
            domain_config_path=Path(domain_config) if domain_config else PROJECT_ROOT / "domains.yaml",
            analysis_policy_path=Path(policy_path) if policy_path else PROJECT_ROOT / "analysis_rules.yaml",
            checkpoint_path=Path(checkpoint) if checkpoint else PROJECT_ROOT / ".langgraph" / "analysis_agent.sqlite",
        )

    def ensure_runtime_dirs(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
