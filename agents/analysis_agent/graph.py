from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agents.shared.domain_config import DomainConfig
from agents.support_agent.repositories import CsvRepository, KnowledgeRepository

from .config import AnalysisAgentSettings
from .contracts import AnalysisResult, DetectionSignals, SupportValidationFinding
from .prompts import build_inspection_prompt
from .repositories import AnalysisResultRepository, SupportConversationRepository, UserProfileRepository
from .state import AnalysisAgentState
from .tools import build_analysis_tools


def _message_text(message: AIMessage | HumanMessage | ToolMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(json.dumps(item))
        return "\n".join(parts)
    return str(content)


def _format_history(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for entry in messages:
        role = entry.get("role", "unknown")
        content = entry.get("content", "")
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def _latest_support_reply(messages: list[dict[str, Any]]) -> str | None:
    for entry in reversed(messages):
        if entry.get("role") in {"assistant", "assistant_draft"}:
            return entry.get("content")
    return None


def build_analysis_agent_graph(
    *,
    settings: AnalysisAgentSettings,
    domain_config: DomainConfig,
    model: Any | None = None,
    checkpointer: Any | None = None,
    support_checkpointer: Any | None = None,
    analysis_repository: AnalysisResultRepository,
    conversation_repository: SupportConversationRepository | None = None,
):
    csv_repository = CsvRepository(domain_config.data_root)
    knowledge_repository = KnowledgeRepository(domain_config.knowledge_root)
    conversation_repository = conversation_repository or SupportConversationRepository(
        checkpointer=support_checkpointer,
        max_messages=settings.max_history_messages,
    )
    user_profile_repository = UserProfileRepository(csv_repository)

    llm = model or ChatOpenAI(model=settings.openai_model, temperature=settings.temperature)
    tools = build_analysis_tools(
        csv_repository=csv_repository,
        knowledge_repository=knowledge_repository,
        conversation_repository=conversation_repository,
        data_search_limit=settings.data_search_limit,
        knowledge_search_limit=settings.knowledge_search_limit,
        analysis_model=llm,
        domain_context=domain_config.system_context,
    )
    tool_lookup = {tool.name: tool for tool in tools}
    bound_model = llm.bind_tools(tools)
    tool_node = ToolNode(tools, messages_key="turn_messages")

    def prepare_context(state: AnalysisAgentState) -> dict[str, Any]:
        provided_history = list(state.get("conversation_history", []))
        provided_profile = dict(state.get("user_profile", {}))
        if provided_history:
            conversation_repository.set_override(
                state["conversation_id"],
                provided_history,
                provided_profile,
            )
            history, profile = conversation_repository.load(state["conversation_id"])
        else:
            history, profile = conversation_repository.load(state["conversation_id"])
        state_history = history[-settings.max_history_messages :]
        formatted_history = _format_history(state_history)
        system_prompt = build_inspection_prompt(domain_context=domain_config.system_context)
        turn_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    "Conversation ID: "
                    f"{state['conversation_id']}. Call load_conversation_history to inspect messages,"
                    " then validate the support agent."
                ),
                id=f"analysis-user-{uuid4()}",
            ),
        ]
        return {
            "conversation_history": state_history,
            "user_profile": profile,
            "analysis_metadata": {
                "history_source": "input" if provided_history else "checkpoint",
            },
            "turn_messages": turn_messages,
            "last_tool_results": [],
            "formatted_history": formatted_history,
        }

    def inspection_reasoning(state: AnalysisAgentState) -> dict[str, Any]:
        response = bound_model.invoke(state.get("turn_messages", []))
        turn_messages = list(state.get("turn_messages", [])) + [response]
        return {"turn_messages": turn_messages}

    def route_after_inspection(state: AnalysisAgentState) -> str:
        last_message = state.get("turn_messages", [])[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "dialogue_monitor"

    def capture_tool_results(state: AnalysisAgentState) -> dict[str, Any]:
        history = list(state.get("last_tool_results", []))
        tool_messages: list[ToolMessage] = []
        for message in reversed(state.get("turn_messages", [])):
            if isinstance(message, ToolMessage):
                tool_messages.append(message)
                continue
            if tool_messages:
                break

        tool_messages.reverse()
        for message in tool_messages:
            content = _message_text(message)
            record: dict[str, Any] = {
                "tool_name": message.name or "tool",
                "content": content,
            }
            try:
                record.update(json.loads(content))
            except Exception:
                record["raw"] = content
            history.append(record)
        turn_messages = list(state.get("turn_messages", []))
        turn_messages.append(AIMessage(content="Tool results captured."))
        return {
            "last_tool_results": history,
            "turn_messages": turn_messages,
        }

    def dialogue_monitor(state: AnalysisAgentState) -> dict[str, Any]:
        conversation_text = state.get("formatted_history") or _format_history(state.get("conversation_history", []))
        last_support_message = _latest_support_reply(state.get("conversation_history", []))
        monitor_tool = tool_lookup.get("dialogue_monitor_tool")
        if monitor_tool is None:
            return {}
        raw = monitor_tool.invoke({"conversation": conversation_text, "last_support_message": last_support_message})
        try:
            signals = json.loads(raw)
        except Exception:
            signals = {}
        return {"detection_signals": signals}

    def priority_node(state: AnalysisAgentState) -> dict[str, Any]:
        priority_tool = tool_lookup.get("priority_evaluation_tool")
        detection = state.get("detection_signals", {})
        if priority_tool is None:
            return {}
        raw = priority_tool.invoke({"detection_signals": detection})
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"priority": "medium"}
        return {"priority": payload.get("priority", "medium")}

    def parse_inspection(state: AnalysisAgentState) -> dict[str, Any]:
        issue_type = None
        detected_issues: list[str] = []
        findings: list[dict[str, Any]] = []
        for message in reversed(state.get("turn_messages", [])):
            if isinstance(message, AIMessage) and not message.tool_calls:
                try:
                    payload = json.loads(_message_text(message))
                    issue_type = payload.get("issue_type")
                    detected_issues = list(payload.get("detected_issues", []))
                    findings = list(payload.get("support_agent_findings", []))
                except Exception:
                    continue
                break
        return {
            "issue_type": issue_type,
            "detected_issues": detected_issues,
            "support_agent_findings": findings,
        }

    def summary_node(state: AnalysisAgentState) -> dict[str, Any]:
        summary_tool = tool_lookup.get("summary_tool")
        if summary_tool is None:
            return {}
        conversation_text = state.get("formatted_history") or _format_history(state.get("conversation_history", []))
        raw = summary_tool.invoke(
            {
                "conversation_history": conversation_text,
                "issue_type": state.get("issue_type"),
                "detected_issues": state.get("detected_issues", []),
                "user_profile": state.get("user_profile", {}),
            }
        )
        try:
            payload = json.loads(raw)
            summary_text = payload.get("case_summary") or json.dumps(payload)
            analysis_metadata = {"summary_payload": payload}
        except Exception:
            summary_text = str(raw)
            analysis_metadata = {}
        return {"summary": summary_text, "analysis_metadata": analysis_metadata}

    def next_steps_node(state: AnalysisAgentState) -> dict[str, Any]:
        tool = tool_lookup.get("next_steps_recommendation_tool")
        if tool is None:
            return {}
        raw = tool.invoke(
            {
                "summary": state.get("summary", ""),
                "priority": state.get("priority", "medium"),
                "detected_issues": state.get("detected_issues", []),
            }
        )
        try:
            steps = json.loads(raw)
        except Exception:
            steps = ["Review case with engineering", "Send personal follow-up to user"]
        if isinstance(steps, dict):
            steps = list(steps.values())
        return {"next_steps": steps}

    def save_results(state: AnalysisAgentState) -> dict[str, Any]:
        detection = DetectionSignals.model_validate(state.get("detection_signals", {}))
        findings = [SupportValidationFinding.model_validate(item) for item in state.get("support_agent_findings", [])]
        analysis_result = AnalysisResult(
            conversation_id=state["conversation_id"],
            domain_key=state["domain_key"],
            issue_type=state.get("issue_type"),
            priority=state.get("priority", "medium"),
            summary=state.get("summary", ""),
            user_profile=user_profile_repository.find_profile(
                state.get("user_profile", {}).get("display_name"),
                state.get("user_profile", {}),
            ),
            detected_issues=state.get("detected_issues", []),
            detection_signals=detection,
            support_agent_findings=findings,
            next_steps=state.get("next_steps", []),
            analysis_metadata=state.get("analysis_metadata", {}),
            conversation_history=state.get("conversation_history", []),
        )
        analysis_repository.upsert(analysis_result)
        return {"analysis_metadata": {**state.get("analysis_metadata", {}), "saved": True}}

    graph = StateGraph(AnalysisAgentState)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("inspection_reasoning", inspection_reasoning)
    graph.add_node("tools", tool_node)
    graph.add_node("capture_tool_results", capture_tool_results)
    graph.add_node("dialogue_monitor", dialogue_monitor)
    graph.add_node("priority", priority_node)
    graph.add_node("parse_inspection", parse_inspection)
    graph.add_node("summary", summary_node)
    graph.add_node("next_steps", next_steps_node)
    graph.add_node("save_results", save_results)

    graph.add_edge(START, "prepare_context")
    graph.add_edge("prepare_context", "inspection_reasoning")
    graph.add_conditional_edges(
        "inspection_reasoning",
        route_after_inspection,
        {
            "tools": "tools",
            "dialogue_monitor": "dialogue_monitor",
        },
    )
    graph.add_edge("tools", "capture_tool_results")
    graph.add_edge("capture_tool_results", "inspection_reasoning")
    graph.add_edge("dialogue_monitor", "priority")
    graph.add_edge("priority", "parse_inspection")
    graph.add_edge("parse_inspection", "summary")
    graph.add_edge("summary", "next_steps")
    graph.add_edge("next_steps", "save_results")
    graph.add_edge("save_results", END)

    return graph.compile(checkpointer=checkpointer)
