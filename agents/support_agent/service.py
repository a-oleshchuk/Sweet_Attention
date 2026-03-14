from __future__ import annotations

from types import TracebackType
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from agents.shared.domain_config import DomainConfig, load_domain_registry

from .config import SupportAgentSettings
from .contracts import SupportTurnInput, SupportTurnOutput
from .graph import build_support_agent_graph


class SupportAgentService:
    def __init__(
        self,
        *,
        settings: SupportAgentSettings,
        domain_config: DomainConfig,
        graph: Any,
        checkpoint_manager: Any | None = None,
    ):
        self.settings = settings
        self.domain_config = domain_config
        self.graph = graph
        self._checkpoint_manager = checkpoint_manager

    @classmethod
    def from_env(cls, settings: SupportAgentSettings | None = None, model: Any | None = None) -> "SupportAgentService":
        resolved_settings = settings or SupportAgentSettings.from_env()
        resolved_settings.ensure_runtime_dirs()
        registry = load_domain_registry(
            project_root=resolved_settings.project_root,
            config_path=resolved_settings.domain_config_path,
        )
        domain_config = registry.get(resolved_settings.default_domain_key)
        checkpoint_manager = SqliteSaver.from_conn_string(str(resolved_settings.checkpoint_path))
        checkpointer = checkpoint_manager.__enter__()
        graph = build_support_agent_graph(
            settings=resolved_settings,
            domain_config=domain_config,
            model=model,
            checkpointer=checkpointer,
        )
        return cls(
            settings=resolved_settings,
            domain_config=domain_config,
            graph=graph,
            checkpoint_manager=checkpoint_manager,
        )

    def close(self) -> None:
        if self._checkpoint_manager is not None:
            self._checkpoint_manager.__exit__(None, None, None)
            self._checkpoint_manager = None

    def __enter__(self) -> "SupportAgentService":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def invoke(self, turn: SupportTurnInput) -> SupportTurnOutput:
        if turn.domain_key != self.domain_config.key:
            raise ValueError(
                f"Service is configured for domain '{self.domain_config.key}', got '{turn.domain_key}'."
            )

        state_input = {
            "conversation_id": turn.conversation_id,
            "domain_key": turn.domain_key,
            "incoming_user_message": turn.user_message,
            "incoming_worker_instruction": turn.worker_instruction.model_dump() if turn.worker_instruction else None,
            "incoming_user_metadata": turn.user_metadata.model_dump(),
        }
        result = self.graph.invoke(
            state_input,
            config={"configurable": {"thread_id": turn.conversation_id}},
        )
        return SupportTurnOutput.from_state(result)
