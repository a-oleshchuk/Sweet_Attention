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
    llm_classification_enabled: bool = False
    openai_model: str = Field(default="gpt-4.1-mini")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)

    @classmethod
    def from_env(cls) -> "AnalysisAgentSettings":
        load_project_env(PROJECT_ROOT)
        checkpoint = os.getenv("ANALYSIS_AGENT_CHECKPOINTER")
        domain_config = os.getenv("ANALYSIS_AGENT_DOMAIN_CONFIG")
        policy_path = os.getenv("ANALYSIS_AGENT_RULES_CONFIG")
        classification_enabled = os.getenv("ANALYSIS_AGENT_LLM_CLASSIFICATION_ENABLED", "1")
        return cls(
            domain_config_path=Path(domain_config) if domain_config else PROJECT_ROOT / "domains.yaml",
            analysis_policy_path=Path(policy_path) if policy_path else PROJECT_ROOT / "analysis_rules.yaml",
            checkpoint_path=Path(checkpoint) if checkpoint else PROJECT_ROOT / ".langgraph" / "analysis_agent.sqlite",
            llm_classification_enabled=classification_enabled.lower() not in {"0", "false", "no"},
            openai_model=os.getenv("ANALYSIS_AGENT_MODEL", "gpt-4.1-mini"),
            temperature=float(os.getenv("ANALYSIS_AGENT_TEMPERATURE", "0.0")),
        )

    def ensure_runtime_dirs(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
