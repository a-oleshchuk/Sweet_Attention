from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from agents.shared import load_project_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SupportAgentSettings(BaseModel):
    project_root: Path = PROJECT_ROOT
    domain_config_path: Path = PROJECT_ROOT / "domains.yaml"
    default_domain_key: str = "ai_school"
    openai_model: str = Field(default="gpt-4.1-mini")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    checkpoint_path: Path = PROJECT_ROOT / ".langgraph" / "support_agent.sqlite"
    data_search_limit: int = Field(default=5, ge=1, le=20)
    knowledge_search_limit: int = Field(default=2, ge=1, le=5)

    @classmethod
    def from_env(cls) -> "SupportAgentSettings":
        load_project_env(PROJECT_ROOT)
        checkpoint = os.getenv("SUPPORT_AGENT_CHECKPOINTER")
        domain_config = os.getenv("SUPPORT_AGENT_DOMAIN_CONFIG")
        return cls(
            domain_config_path=Path(domain_config) if domain_config else PROJECT_ROOT / "domains.yaml",
            openai_model=os.getenv("SUPPORT_AGENT_MODEL", "gpt-4.1-mini"),
            temperature=float(os.getenv("SUPPORT_AGENT_TEMPERATURE", "0.1")),
            checkpoint_path=Path(checkpoint) if checkpoint else PROJECT_ROOT / ".langgraph" / "support_agent.sqlite",
        )

    def ensure_runtime_dirs(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
