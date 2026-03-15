from __future__ import annotations

from types import TracebackType
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from agents.shared import load_analysis_policy, load_domain_registry

from .config import AnalysisAgentSettings
from .contracts import AnalysisTurnInput, AnalysisTurnOutput
from .graph import build_analysis_agent_graph


class AnalysisAgentService:
    def __init__(
        self,
        *,
        settings: AnalysisAgentSettings,
        graph: Any,
        checkpoint_manager: Any | None = None,
    ):
        self.settings = settings
        self.graph = graph
        self._checkpoint_manager = checkpoint_manager

    @classmethod
    def from_env(cls, settings: AnalysisAgentSettings | None = None) -> "AnalysisAgentService":
        resolved_settings = settings or AnalysisAgentSettings.from_env()
        resolved_settings.ensure_runtime_dirs()
        domain_registry = load_domain_registry(
            project_root=resolved_settings.project_root,
            config_path=resolved_settings.domain_config_path,
        )
        policy = load_analysis_policy(
            project_root=resolved_settings.project_root,
            config_path=resolved_settings.analysis_policy_path,
        )
        domain_config = domain_registry.get(resolved_settings.default_domain_key)
        checkpoint_manager = SqliteSaver.from_conn_string(str(resolved_settings.checkpoint_path))
        checkpointer = checkpoint_manager.__enter__()
        graph = build_analysis_agent_graph(
            domain_config=domain_config,
            policy=policy,
            data_search_limit=resolved_settings.data_search_limit,
            knowledge_search_limit=resolved_settings.knowledge_search_limit,
            checkpointer=checkpointer,
        )
        return cls(
            settings=resolved_settings,
            graph=graph,
            checkpoint_manager=checkpoint_manager,
        )

    def close(self) -> None:
        if self._checkpoint_manager is not None:
            self._checkpoint_manager.__exit__(None, None, None)
            self._checkpoint_manager = None

    def __enter__(self) -> "AnalysisAgentService":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def invoke(self, turn: AnalysisTurnInput) -> AnalysisTurnOutput:
        state_input = {
            "conversation_id": turn.conversation_id,
            "domain_key": turn.domain_key,
            "incoming_full_dialogue_snapshot": [message.model_dump() for message in turn.full_dialogue_snapshot],
            "incoming_support_agent_state": turn.support_agent_state.model_dump(),
            "incoming_worker_decision": turn.worker_decision_update.model_dump() if turn.worker_decision_update else None,
        }
        result = self.graph.invoke(
            state_input,
            config={"configurable": {"thread_id": turn.conversation_id}},
        )
        return AnalysisTurnOutput.from_state(result)

