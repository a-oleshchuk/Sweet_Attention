from __future__ import annotations

from types import TracebackType
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from agents.shared.domain_config import DomainConfig, load_domain_registry
from .config import AnalysisAgentSettings
from .contracts import AnalysisResult, AnalysisTurnInput
from .graph import build_analysis_agent_graph
from .repositories import AnalysisResultRepository, SupportConversationRepository


class AnalysisAgentService:
    def __init__(
        self,
        *,
        settings: AnalysisAgentSettings,
        domain_config: DomainConfig,
        graph: Any,
        support_checkpointer: Any,
        analysis_repository: AnalysisResultRepository,
        conversation_repository: SupportConversationRepository,
        checkpoint_manager: Any | None = None,
        support_checkpoint_manager: Any | None = None,
    ):
        self.settings = settings
        self.domain_config = domain_config
        self.graph = graph
        self.support_checkpointer = support_checkpointer
        self.analysis_repository = analysis_repository
        self.conversation_repository = conversation_repository
        self._checkpoint_manager = checkpoint_manager
        self._support_checkpoint_manager = support_checkpoint_manager

    @classmethod
    def from_env(
        cls,
        settings: AnalysisAgentSettings | None = None,
        model: Any | None = None,
        support_checkpointer: Any | None = None,
    ) -> "AnalysisAgentService":
        resolved_settings = settings or AnalysisAgentSettings.from_env()
        resolved_settings.ensure_runtime_dirs()
        registry = load_domain_registry(
            project_root=resolved_settings.project_root,
            config_path=resolved_settings.domain_config_path,
        )
        domain_config = registry.get(resolved_settings.default_domain_key)
        checkpoint_manager = SqliteSaver.from_conn_string(str(resolved_settings.checkpoint_path))
        checkpointer = checkpoint_manager.__enter__()

        support_manager = support_checkpointer
        support_cp_manager = None
        if support_manager is None:
            support_cp_manager = SqliteSaver.from_conn_string(str(resolved_settings.support_checkpoint_path))
            support_manager = support_cp_manager.__enter__()

        analysis_repository = AnalysisResultRepository(str(resolved_settings.analysis_results_path))
        conversation_repository = SupportConversationRepository(
            checkpointer=support_manager,
            max_messages=resolved_settings.max_history_messages,
        )
        graph = build_analysis_agent_graph(
            settings=resolved_settings,
            domain_config=domain_config,
            model=model,
            checkpointer=checkpointer,
            support_checkpointer=support_manager,
            analysis_repository=analysis_repository,
            conversation_repository=conversation_repository,
        )
        return cls(
            settings=resolved_settings,
            domain_config=domain_config,
            graph=graph,
            support_checkpointer=support_manager,
            analysis_repository=analysis_repository,
            conversation_repository=conversation_repository,
            checkpoint_manager=checkpoint_manager,
            support_checkpoint_manager=support_cp_manager,
        )

    def close(self) -> None:
        if self._checkpoint_manager is not None:
            self._checkpoint_manager.__exit__(None, None, None)
            self._checkpoint_manager = None
        if self._support_checkpoint_manager is not None:
            self._support_checkpoint_manager.__exit__(None, None, None)
            self._support_checkpoint_manager = None

    def __enter__(self) -> "AnalysisAgentService":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def invoke(self, turn: AnalysisTurnInput) -> AnalysisResult:
        state_input = {
            "conversation_id": turn.conversation_id,
            "domain_key": turn.domain_key,
            "conversation_history": turn.conversation_history,
            "user_profile": turn.user_profile,
        }
        try:
            self.graph.invoke(
                state_input,
                config={
                    "configurable": {"thread_id": turn.conversation_id},
                    "tags": ["analysis-agent"],
                },
            )
            result = self.analysis_repository.get(turn.conversation_id)
            if result is None:
                raise RuntimeError("Analysis run did not produce a persisted result.")
            return result
        finally:
            self.conversation_repository.clear_override(turn.conversation_id)
