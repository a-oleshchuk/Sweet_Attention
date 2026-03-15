from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_project_env(project_root: Path) -> None:
    candidates: list[Path] = []
    for root in reversed([project_root, *project_root.parents]):
        env_path = root / ".env"
        if env_path.is_file():
            candidates.append(env_path)

    for env_path in candidates:
        load_dotenv(env_path, override=False)
