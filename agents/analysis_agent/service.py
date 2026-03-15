from __future__ import annotations

from types import TracebackType
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.shared import load_analysis_policy, load_domain_registry

from .config import AnalysisAgentSettings
from .contracts import AnalysisTurnInput, AnalysisTurnOutput, DialogueLabelChoice, DialogueMessage, SupportAgentSnapshot
from .graph import build_analysis_agent_graph
from .prompts import build_issue_classification_system_prompt, build_issue_classification_user_prompt


def _build_issue_classifier(settings: AnalysisAgentSettings) -> Any | None:
    if not settings.llm_classification_enabled:
        return None

    model = ChatOpenAI(model=settings.openai_model, temperature=settings.temperature)
    structured_model = model.with_structured_output(DialogueLabelChoice)
    system_prompt = build_issue_classification_system_prompt()

    def classify(
        dialogue: list[DialogueMessage],
        snapshot: SupportAgentSnapshot,
        active_issue: str,
        dialogue_summary: str,
    ) -> tuple[str, str]:
        latest_user_message = next(
            (message.content for message in reversed(dialogue) if message.role == "user" and message.content.strip()),
            active_issue,
        )
        try:
            response = structured_model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=build_issue_classification_user_prompt(
                            active_issue=active_issue,
                            latest_user_message=latest_user_message or snapshot.active_user_issue or active_issue,
                            dialogue_summary=dialogue_summary,
                        )
                    ),
                ]
            )
        except Exception:
            return "general", "general_support_request"
        return response.category, response.intent

    return classify


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
        issue_classifier = _build_issue_classifier(resolved_settings)
        graph = build_analysis_agent_graph(
            domain_config=domain_config,
            policy=policy,
            data_search_limit=resolved_settings.data_search_limit,
            knowledge_search_limit=resolved_settings.knowledge_search_limit,
            issue_classifier=issue_classifier,
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
