from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DomainConfig(BaseModel):
    key: str
    display_name: str
    data_root: Path
    knowledge_root: Path
    system_context: str


class DomainConfigRegistry(BaseModel):
    domains: dict[str, DomainConfig] = Field(default_factory=dict)

    def get(self, domain_key: str) -> DomainConfig:
        try:
            return self.domains[domain_key]
        except KeyError as exc:
            raise ValueError(f"Unsupported domain: {domain_key}") from exc


def load_domain_registry(*, project_root: Path, config_path: Path | None = None) -> DomainConfigRegistry:
    resolved_path = config_path or project_root / "domains.yaml"
    raw_payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    raw_domains = raw_payload.get("domains", {})

    domains: dict[str, DomainConfig] = {}
    for key, payload in raw_domains.items():
        domains[key] = DomainConfig(
            key=key,
            display_name=payload["display_name"],
            data_root=(project_root / payload["data_root"]).resolve(),
            knowledge_root=(project_root / payload["knowledge_root"]).resolve(),
            system_context=payload["system_context"].strip(),
        )

    return DomainConfigRegistry(domains=domains)
