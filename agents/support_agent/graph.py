from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agents.shared import CsvRepository, DomainConfig, KnowledgeRepository, build_retrieval_tools
from agents.shared.user_reply_safety import sanitize_user_reply

from .config import SupportAgentSettings
from .contracts import MessageHistoryEntry, WorkerInstruction
from .prompts import build_system_prompt
from .state import SupportAgentState


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


def _clear_messages(messages: list[Any]) -> list[RemoveMessage]:
    removals: list[RemoveMessage] = []
    for message in messages:
        message_id = getattr(message, "id", None)
        if message_id:
            removals.append(RemoveMessage(id=message_id))
    return removals


def _serialize_instruction(instruction: dict[str, Any]) -> str:
    return json.dumps(instruction, sort_keys=True)


def _tooling_overview(csv_repository: CsvRepository, knowledge_repository: KnowledgeRepository) -> str:
    sections = [
        "Operational tables:",
        "\n".join(f"- {table_name}" for table_name in csv_repository.available_tables()),
        "Pattern files:",
        "\n".join(f"- {file_id}" for file_id in knowledge_repository.available_files(categories=["patterns"])),
        "Reference files:",
        "\n".join(
            f"- {file_id}"
            for file_id in knowledge_repository.available_files(categories=["docs", "how_to"])
        ),
        "Policy files:",
        "\n".join(f"- {file_id}" for file_id in knowledge_repository.available_files(categories=["policies"])),
    ]
    return "\n".join(sections)


def build_support_agent_graph(
    *,
    settings: SupportAgentSettings,
    domain_config: DomainConfig,
    model: Any | None = None,
    checkpointer: Any | None = None,
):
    csv_repository = CsvRepository(domain_config.data_root)
    knowledge_repository = KnowledgeRepository(domain_config.knowledge_root)
    tools = build_retrieval_tools(
        csv_repository=csv_repository,
        knowledge_repository=knowledge_repository,
        data_search_limit=settings.data_search_limit,
        knowledge_search_limit=settings.knowledge_search_limit,
    )
    bound_model = (model or ChatOpenAI(model=settings.openai_model, temperature=settings.temperature)).bind_tools(tools)
    tooling_overview = _tooling_overview(csv_repository, knowledge_repository)
    tool_node = ToolNode(tools, messages_key="turn_messages")

    def prepare_turn(state: SupportAgentState) -> dict[str, Any]:
        history = list(state.get("message_history", []))
        known_identifiers = dict(state.get("known_user_identifiers", {}))
        incoming_metadata = dict(state.get("incoming_user_metadata", {}) or {})
        display_name = incoming_metadata.get("display_name") or state.get("user_display_name")

        if display_name:
            known_identifiers["display_name"] = display_name
        for key, value in dict(incoming_metadata.get("known_identifiers", {})).items():
            known_identifiers[key] = str(value)

        turn_messages = _clear_messages(list(state.get("turn_messages", [])))
        turn_messages.extend(list(state.get("conversation_messages", [])))

        conversation_updates: list[Any] = []
        incoming_user_message = state.get("incoming_user_message")
        if incoming_user_message:
            user_message = HumanMessage(content=incoming_user_message, id=f"user-{uuid4()}")
            turn_messages.append(user_message)
            conversation_updates.append(user_message)
            history.append(
                MessageHistoryEntry(role="user", content=incoming_user_message).model_dump()
            )

        incoming_instruction = dict(state.get("incoming_worker_instruction", {}) or {})
        if incoming_instruction:
            history.append(
                MessageHistoryEntry(
                    role="support_worker_instruction",
                    content=_serialize_instruction(incoming_instruction),
                ).model_dump()
            )

        active_user_issue = state.get("active_user_issue")
        if incoming_instruction and not incoming_user_message:
            for entry in reversed(history):
                if entry.get("role") == "user" and entry.get("content"):
                    active_user_issue = str(entry["content"])
                    break
        if incoming_user_message:
            active_user_issue = incoming_user_message

        # Preserve prior lookup context when the worker asks for a revised reply
        # on the same issue. A worker instruction is not itself the user issue.
        last_tool_results = (
            list(state.get("last_tool_results", []))
            if incoming_instruction and not incoming_user_message
            else []
        )

        previous_draft = state.get("draft_message") or state.get("previous_draft_message")

        return {
            "conversation_id": state["conversation_id"],
            "domain_key": state["domain_key"],
            "user_display_name": display_name,
            "known_user_identifiers": known_identifiers,
            "active_user_issue": active_user_issue,
            "open_worker_instruction": incoming_instruction or None,
            "pending_review": False,
            "draft_message": None,
            "assistant_message": None,
            "send_target": None,
            "last_tool_results": last_tool_results,
            "previous_draft_message": previous_draft,
            "message_history": history,
            "conversation_messages": conversation_updates,
            "turn_messages": turn_messages,
        }

    def reasoning(state: SupportAgentState) -> dict[str, Any]:
        prompt = build_system_prompt(
            domain_context=domain_config.system_context,
            tooling_overview=tooling_overview,
            state=state,
        )
        response = bound_model.invoke([SystemMessage(content=prompt), *state.get("turn_messages", [])])
        return {"turn_messages": [response]}

    def capture_tool_results(state: SupportAgentState) -> dict[str, Any]:
        history = list(state.get("message_history", []))
        tool_messages: list[ToolMessage] = []
        for message in reversed(state.get("turn_messages", [])):
            if isinstance(message, ToolMessage):
                tool_messages.append(message)
                continue
            if tool_messages:
                break

        tool_messages.reverse()
        records: list[dict[str, Any]] = []
        for message in tool_messages:
            content = _message_text(message)
            record: dict[str, Any] = {
                "tool_name": message.name or "tool",
                "content": content,
            }
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                payload = None

            if isinstance(payload, dict):
                record["query"] = payload.get("query")
                record["result_count"] = payload.get("result_count")

            records.append(record)
            history.append(
                MessageHistoryEntry(
                    role="tool",
                    content=content,
                    metadata={"tool_name": message.name or "tool"},
                ).model_dump()
            )

        return {
            "last_tool_results": records,
            "message_history": history,
        }

    def finalize_turn(state: SupportAgentState) -> dict[str, Any]:
        last_ai_message: AIMessage | None = None
        for message in reversed(state.get("turn_messages", [])):
            if isinstance(message, AIMessage) and not message.tool_calls:
                last_ai_message = message
                break

        if last_ai_message is None:
            raise ValueError("No final assistant message was produced for this turn.")

        instruction = None
        if state.get("open_worker_instruction"):
            instruction = WorkerInstruction.model_validate(state["open_worker_instruction"])
        review_before_send = bool(instruction.review_before_send) if instruction else False
        send_target = "support_worker" if review_before_send else "user"
        assistant_text = _message_text(last_ai_message)
        if send_target == "user":
            assistant_text = sanitize_user_reply(assistant_text)
        history = list(state.get("message_history", []))
        history.append(
            MessageHistoryEntry(
                role="assistant_draft" if review_before_send else "assistant",
                content=assistant_text,
                metadata={"send_target": send_target},
            ).model_dump()
        )

        updates: dict[str, Any] = {
            "assistant_message": assistant_text,
            "draft_message": assistant_text if review_before_send else None,
            "pending_review": review_before_send,
            "send_target": send_target,
            "open_worker_instruction": None,
            "incoming_user_message": None,
            "incoming_worker_instruction": None,
            "incoming_user_metadata": None,
            "message_history": history,
            "turn_messages": _clear_messages(list(state.get("turn_messages", []))),
        }

        if review_before_send:
            updates["previous_draft_message"] = assistant_text
        else:
            updates["conversation_messages"] = [last_ai_message]

        return updates

    def route_after_reasoning(state: SupportAgentState) -> str:
        last_message = state.get("turn_messages", [])[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "finalize_turn"

    graph = StateGraph(SupportAgentState)
    graph.add_node("prepare_turn", prepare_turn)
    graph.add_node("reasoning", reasoning)
    graph.add_node("tools", tool_node)
    graph.add_node("capture_tool_results", capture_tool_results)
    graph.add_node("finalize_turn", finalize_turn)
    graph.add_edge(START, "prepare_turn")
    graph.add_edge("prepare_turn", "reasoning")
    graph.add_conditional_edges(
        "reasoning",
        route_after_reasoning,
        {
            "tools": "tools",
            "finalize_turn": "finalize_turn",
        },
    )
    graph.add_edge("tools", "capture_tool_results")
    graph.add_edge("capture_tool_results", "reasoning")
    graph.add_edge("finalize_turn", END)
    return graph.compile(checkpointer=checkpointer)
