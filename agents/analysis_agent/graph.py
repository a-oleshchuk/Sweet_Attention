from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.shared import AnalysisPolicy, CsvRepository, DomainConfig, KnowledgeRepository

from .contracts import AnalysisConversationStatus, AnalysisReason, DialogueMessage, SupportAgentSnapshot, WorkerDecision
from .rules import (
    build_dialogue_summary,
    build_suggested_instruction,
    build_worker_package,
    derive_constraints,
    derive_next_steps,
    derive_recommended_tone,
    evaluate_rules,
    summarize_codes,
)
from .state import AnalysisAgentState


def build_analysis_agent_graph(
    *,
    domain_config: DomainConfig,
    policy: AnalysisPolicy,
    data_search_limit: int,
    knowledge_search_limit: int,
    checkpointer: Any | None = None,
):
    csv_repository = CsvRepository(domain_config.data_root)
    knowledge_repository = KnowledgeRepository(domain_config.knowledge_root)

    def prepare_turn(state: AnalysisAgentState) -> dict[str, Any]:
        full_dialogue_snapshot = list(state.get("incoming_full_dialogue_snapshot") or state.get("full_dialogue_snapshot") or [])
        support_agent_state = dict(state.get("incoming_support_agent_state") or state.get("support_agent_state") or {})
        incoming_worker_decision = dict(state.get("incoming_worker_decision") or {})
        worker_decisions = list(state.get("worker_decisions", []))
        conversation_status = state.get("conversation_status", AnalysisConversationStatus.NORMAL.value)

        if incoming_worker_decision:
            worker_decisions.append(incoming_worker_decision)
            worker_decision = WorkerDecision.model_validate(incoming_worker_decision)
            if worker_decision.mark_resolved or worker_decision.action.lower() == AnalysisConversationStatus.RESOLVED.value:
                conversation_status = AnalysisConversationStatus.RESOLVED.value
            elif worker_decision.action.lower() == "resume_monitoring":
                conversation_status = AnalysisConversationStatus.NORMAL.value
            else:
                conversation_status = AnalysisConversationStatus.WORKER_REVIEWING.value

        return {
            "conversation_id": state["conversation_id"],
            "domain_key": state["domain_key"],
            "full_dialogue_snapshot": full_dialogue_snapshot,
            "support_agent_state": support_agent_state,
            "worker_decisions": worker_decisions,
            "conversation_status": conversation_status,
            "worker_decision_applied": bool(incoming_worker_decision),
            "current_reasons": [],
            "needs_escalation": False,
            "paused": False,
            "dialogue_summary": "",
            "possible_next_steps": [],
            "recommended_tone": policy.default_output_tone.value,
            "suggested_constraints": [],
            "suggested_worker_instruction": None,
            "worker_package": None,
            "verification_evidence": [],
            "last_user_message": None,
            "last_support_message": None,
            "incoming_full_dialogue_snapshot": None,
            "incoming_support_agent_state": None,
            "incoming_worker_decision": None,
        }

    def route_after_prepare(state: AnalysisAgentState) -> str:
        if state["conversation_status"] == AnalysisConversationStatus.RESOLVED.value and state.get("worker_decision_applied"):
            return "finalize_worker_update"
        if state["conversation_status"] == AnalysisConversationStatus.WORKER_REVIEWING.value and state.get("worker_decision_applied"):
            return "finalize_worker_update"
        if state["conversation_status"] in {
            AnalysisConversationStatus.WAITING_FOR_WORKER.value,
            AnalysisConversationStatus.WORKER_REVIEWING.value,
        }:
            return "finalize_paused"
        return "evaluate"

    def evaluate(state: AnalysisAgentState) -> dict[str, Any]:
        dialogue = [DialogueMessage.model_validate(item) for item in state.get("full_dialogue_snapshot", [])]
        support_snapshot = SupportAgentSnapshot.model_validate(state.get("support_agent_state", {}))
        result = evaluate_rules(
            dialogue=dialogue,
            snapshot=support_snapshot,
            policy=policy,
            csv_repository=csv_repository,
            knowledge_repository=knowledge_repository,
            data_limit=data_search_limit,
            knowledge_limit=knowledge_search_limit,
        )
        reasons = [AnalysisReason.model_validate(item) for item in result["current_reasons"]]
        reason_codes = summarize_codes(result["current_reasons"])
        conversation_status = AnalysisConversationStatus(result["conversation_status"])
        recommended_tone = derive_recommended_tone(reason_codes, policy.default_output_tone)
        constraints = derive_constraints(reason_codes)
        next_steps = derive_next_steps(reason_codes, result["needs_escalation"])
        summary = build_dialogue_summary(
            snapshot=support_snapshot,
            latest_user_message=result["last_user_message"],
            latest_support_message=result["last_support_message"],
            status=conversation_status,
            reasons=reasons,
        )

        suggested_instruction = None
        worker_package = None
        if result["needs_escalation"]:
            suggested_instruction = build_suggested_instruction(
                policy=policy,
                recommended_tone=recommended_tone,
                suggested_constraints=constraints,
                possible_next_steps=next_steps,
                reason_codes=reason_codes,
            )
            worker_package = build_worker_package(
                dialogue=dialogue,
                dialogue_summary=summary,
                reasons=reasons,
                possible_next_steps=next_steps,
                recommended_tone=recommended_tone,
                suggested_constraints=constraints,
                suggested_instruction=suggested_instruction,
            )

        return {
            "current_reasons": [reason.model_dump() for reason in reasons],
            "needs_escalation": result["needs_escalation"],
            "conversation_status": conversation_status.value,
            "dialogue_summary": summary,
            "possible_next_steps": next_steps,
            "recommended_tone": recommended_tone.value,
            "suggested_constraints": constraints,
            "suggested_worker_instruction": suggested_instruction.model_dump() if suggested_instruction else None,
            "worker_package": worker_package,
            "verification_evidence": result["verification_evidence"],
            "last_user_message": result["last_user_message"],
            "last_support_message": result["last_support_message"],
        }

    def finalize_paused(state: AnalysisAgentState) -> dict[str, Any]:
        previous = dict(state.get("last_analysis_result") or {})
        previous.setdefault("conversation_status", state["conversation_status"])
        previous["paused"] = True
        return previous

    def finalize_worker_update(state: AnalysisAgentState) -> dict[str, Any]:
        last_result = dict(state.get("last_analysis_result") or {})
        dialogue = [DialogueMessage.model_validate(item) for item in state.get("full_dialogue_snapshot", [])]
        latest_user = dialogue[-1].content if dialogue else ""
        latest_worker_decision = WorkerDecision.model_validate(state["worker_decisions"][-1]) if state.get("worker_decisions") else None
        if state["conversation_status"] == AnalysisConversationStatus.RESOLVED.value:
            summary = last_result.get("dialogue_summary") or "The support worker marked the conversation as resolved."
            return {
                "current_reasons": [],
                "needs_escalation": False,
                "paused": False,
                "conversation_status": AnalysisConversationStatus.RESOLVED.value,
                "dialogue_summary": summary,
                "possible_next_steps": ["No further analysis action is required."],
                "recommended_tone": last_result.get("recommended_tone", policy.default_output_tone.value),
                "suggested_constraints": [],
                "suggested_worker_instruction": None,
                "worker_package": None,
                "verification_evidence": last_result.get("verification_evidence", []),
                "last_user_message": latest_user,
                "last_support_message": last_result.get("last_support_message"),
            }

        if (
            state["conversation_status"] == AnalysisConversationStatus.NORMAL.value
            and latest_worker_decision
            and latest_worker_decision.action.lower() == "resume_monitoring"
        ):
            return {
                "current_reasons": [],
                "needs_escalation": False,
                "paused": False,
                "conversation_status": AnalysisConversationStatus.NORMAL.value,
                "dialogue_summary": "Worker review is complete. Continue monitoring the next user turn.",
                "possible_next_steps": ["Resume normal monitoring on the next user message."],
                "recommended_tone": last_result.get("recommended_tone", policy.default_output_tone.value),
                "suggested_constraints": [],
                "suggested_worker_instruction": None,
                "worker_package": None,
                "verification_evidence": last_result.get("verification_evidence", []),
                "last_user_message": latest_user,
                "last_support_message": last_result.get("last_support_message"),
            }

        return {
            "current_reasons": last_result.get("current_reasons", []),
            "needs_escalation": last_result.get("needs_escalation", False),
            "paused": True,
            "conversation_status": AnalysisConversationStatus.WORKER_REVIEWING.value,
            "dialogue_summary": last_result.get("dialogue_summary", "The support worker is reviewing the escalated conversation."),
            "possible_next_steps": last_result.get("possible_next_steps", []),
            "recommended_tone": last_result.get("recommended_tone", policy.default_output_tone.value),
            "suggested_constraints": last_result.get("suggested_constraints", []),
            "suggested_worker_instruction": last_result.get("suggested_worker_instruction"),
            "worker_package": last_result.get("worker_package"),
            "verification_evidence": last_result.get("verification_evidence", []),
            "last_user_message": latest_user,
            "last_support_message": last_result.get("last_support_message"),
        }

    def finalize_turn(state: AnalysisAgentState) -> dict[str, Any]:
        current_result = {
            "conversation_id": state["conversation_id"],
            "conversation_status": state["conversation_status"],
            "needs_escalation": state.get("needs_escalation", False),
            "paused": state.get("paused", False),
            "current_reasons": list(state.get("current_reasons", [])),
            "dialogue_summary": state.get("dialogue_summary", ""),
            "possible_next_steps": list(state.get("possible_next_steps", [])),
            "recommended_tone": state.get("recommended_tone", policy.default_output_tone.value),
            "suggested_constraints": list(state.get("suggested_constraints", [])),
            "suggested_worker_instruction": state.get("suggested_worker_instruction"),
            "worker_package": state.get("worker_package"),
            "verification_evidence": list(state.get("verification_evidence", [])),
            "last_user_message": state.get("last_user_message"),
            "last_support_message": state.get("last_support_message"),
        }

        escalation_history = list(state.get("escalation_history", []))
        if state.get("needs_escalation") and state["conversation_status"] == AnalysisConversationStatus.WAITING_FOR_WORKER.value:
            escalation_history.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "status": AnalysisConversationStatus.ESCALATED.value,
                    "reasons": list(state.get("current_reasons", [])),
                }
            )

        return {
            "last_analysis_result": current_result,
            "escalation_history": escalation_history,
        }

    graph = StateGraph(AnalysisAgentState)
    graph.add_node("prepare_turn", prepare_turn)
    graph.add_node("evaluate", evaluate)
    graph.add_node("finalize_paused", finalize_paused)
    graph.add_node("finalize_worker_update", finalize_worker_update)
    graph.add_node("finalize_turn", finalize_turn)
    graph.add_edge(START, "prepare_turn")
    graph.add_conditional_edges(
        "prepare_turn",
        route_after_prepare,
        {
            "evaluate": "evaluate",
            "finalize_paused": "finalize_paused",
            "finalize_worker_update": "finalize_worker_update",
        },
    )
    graph.add_edge("evaluate", "finalize_turn")
    graph.add_edge("finalize_paused", "finalize_turn")
    graph.add_edge("finalize_worker_update", "finalize_turn")
    graph.add_edge("finalize_turn", END)
    return graph.compile(checkpointer=checkpointer)
