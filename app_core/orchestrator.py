from __future__ import annotations

from agents.analysis_agent.contracts import AnalysisTurnInput, DialogueMessage, SupportAgentSnapshot, WorkerDecision
from agents.analysis_agent.service import AnalysisAgentService
from agents.support_agent.contracts import SupportTurnInput, UserMetadata, WorkerInstruction
from agents.support_agent.service import SupportAgentService

from .store import ConversationRecord, ConversationStore


class AppOrchestrator:
    def __init__(self, store: ConversationStore):
        self.store = store

    def send_user_message(self, conversation_id: str, message: str) -> ConversationRecord:
        record = self.store.get(conversation_id)
        record.messages.append(DialogueMessage(role="user", content=message))

        with SupportAgentService.from_env() as support_service:
            support_output = support_service.invoke(
                SupportTurnInput(
                    conversation_id=record.conversation_id,
                    domain_key=record.domain_key,
                    user_message=message,
                    user_metadata=UserMetadata(display_name=record.user_display_name),
                )
            )

        if support_output.send_target == "user":
            record.messages.append(DialogueMessage(role="assistant", content=support_output.assistant_message))
        else:
            record.pending_review = support_output.pending_review
            record.pending_draft = support_output.draft_message or support_output.assistant_message

        record.support_state = support_output.model_dump(mode="json")

        with AnalysisAgentService.from_env() as analysis_service:
            analysis_output = analysis_service.invoke(
                AnalysisTurnInput(
                    conversation_id=record.conversation_id,
                    domain_key=record.domain_key,
                    full_dialogue_snapshot=record.messages,
                    support_agent_state=SupportAgentSnapshot.model_validate(record.support_state),
                )
            )

        record.analysis_result = analysis_output.model_dump(mode="json")
        record.worker_package = (
            analysis_output.worker_package.model_dump(mode="json") if analysis_output.worker_package else None
        )
        if analysis_output.needs_escalation:
            record.pending_review = False
            record.pending_draft = None
        return self.store.save(record)

    def run_worker_instruction(self, conversation_id: str, instruction: WorkerInstruction) -> ConversationRecord:
        record = self.store.get(conversation_id)
        opened_decision = WorkerDecision(action="opened", notes=["Support worker started review."])

        with AnalysisAgentService.from_env() as analysis_service:
            analysis_output = analysis_service.invoke(
                AnalysisTurnInput(
                    conversation_id=record.conversation_id,
                    domain_key=record.domain_key,
                    full_dialogue_snapshot=record.messages,
                    support_agent_state=SupportAgentSnapshot.model_validate(record.support_state or {}),
                    worker_decision_update=opened_decision,
                )
            )
            record.analysis_result = analysis_output.model_dump(mode="json")

        record.worker_decisions.append(opened_decision.model_dump(mode="json"))
        record.worker_instruction_history.append(instruction.model_dump(mode="json"))

        with SupportAgentService.from_env() as support_service:
            support_output = support_service.invoke(
                SupportTurnInput(
                    conversation_id=record.conversation_id,
                    domain_key=record.domain_key,
                    worker_instruction=instruction,
                    user_metadata=UserMetadata(display_name=record.user_display_name),
                )
            )

        record.support_state = support_output.model_dump(mode="json")

        if support_output.pending_review:
            record.pending_review = True
            record.pending_draft = support_output.draft_message or support_output.assistant_message
            return self.store.save(record)

        record.messages.append(DialogueMessage(role="assistant", content=support_output.assistant_message))
        record.pending_review = False
        record.pending_draft = None
        return self._resume_monitoring(
            record=record,
            note="Support worker sent a revised reply directly to the user.",
        )

    def approve_pending_draft(self, conversation_id: str, draft_text: str | None = None) -> ConversationRecord:
        record = self.store.get(conversation_id)
        if not record.pending_draft:
            return record

        approved_draft = (draft_text or record.pending_draft).strip()
        if not approved_draft:
            return record

        record.pending_draft = approved_draft
        record.messages.append(DialogueMessage(role="assistant", content=approved_draft))
        record.pending_review = False
        record.pending_draft = None
        return self._resume_monitoring(
            record=record,
            note="Support worker approved a support-agent draft and sent it to the user.",
        )

    def mark_resolved(self, conversation_id: str, notes: list[str] | None = None) -> ConversationRecord:
        record = self.store.get(conversation_id)
        resolved_decision = WorkerDecision(
            action="resolved",
            mark_resolved=True,
            notes=notes or ["Support worker marked the conversation as resolved."],
        )

        with AnalysisAgentService.from_env() as analysis_service:
            analysis_output = analysis_service.invoke(
                AnalysisTurnInput(
                    conversation_id=record.conversation_id,
                    domain_key=record.domain_key,
                    full_dialogue_snapshot=record.messages,
                    support_agent_state=SupportAgentSnapshot.model_validate(record.support_state or {}),
                    worker_decision_update=resolved_decision,
                )
            )

        record.analysis_result = analysis_output.model_dump(mode="json")
        record.worker_package = None
        record.pending_review = False
        record.pending_draft = None
        record.worker_decisions.append(resolved_decision.model_dump(mode="json"))
        return self.store.save(record)

    def _resume_monitoring(self, *, record: ConversationRecord, note: str) -> ConversationRecord:
        resume_decision = WorkerDecision(action="resume_monitoring", notes=[note])
        with AnalysisAgentService.from_env() as analysis_service:
            analysis_output = analysis_service.invoke(
                AnalysisTurnInput(
                    conversation_id=record.conversation_id,
                    domain_key=record.domain_key,
                    full_dialogue_snapshot=record.messages,
                    support_agent_state=SupportAgentSnapshot.model_validate(record.support_state or {}),
                    worker_decision_update=resume_decision,
                )
            )

        record.analysis_result = analysis_output.model_dump(mode="json")
        record.worker_package = None
        record.worker_decisions.append(resume_decision.model_dump(mode="json"))
        return self.store.save(record)
