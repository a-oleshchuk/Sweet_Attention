from __future__ import annotations

import json
import os
from typing import List
from uuid import uuid4

import streamlit as st

from agents.analysis_agent.config import AnalysisAgentSettings
from agents.analysis_agent.contracts import AnalysisResult, AnalysisTurnInput
from agents.analysis_agent.repositories import AnalysisResultRepository
from agents.analysis_agent.service import AnalysisAgentService
from agents.shared.domain_config import load_domain_registry
from agents.support_agent.config import SupportAgentSettings
from agents.support_agent.contracts import SupportTurnInput, UserMetadata, WorkerInstruction
from agents.support_agent.repositories import CsvRepository
from agents.support_agent.service import SupportAgentService

st.set_page_config(page_title="Sweet Attention Support", layout="wide")

PRIORITY_COLORS = {
    "urgent": "#e74c3c",
    "medium": "#f1c40f",
    "low": "#2ecc71",
}


def _priority_badge(priority: str) -> str:
    color = PRIORITY_COLORS.get(priority, "#95a5a6")
    return f"<span style='background:{color}; color:#111; padding:4px 8px; border-radius:6px; font-weight:600'>{priority.upper()}</span>"


def _priority_icon(priority: str) -> str:
    return {"urgent": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪️")


def load_analysis_repo() -> AnalysisResultRepository:
    settings = AnalysisAgentSettings.from_env()
    settings.ensure_runtime_dirs()
    return AnalysisResultRepository(str(settings.analysis_results_path))


def load_default_chat_user() -> dict[str, str]:
    settings = SupportAgentSettings.from_env()
    registry = load_domain_registry(
        project_root=settings.project_root,
        config_path=settings.domain_config_path,
    )
    domain_config = registry.get(settings.default_domain_key)
    csv_repository = CsvRepository(domain_config.data_root)
    hits = csv_repository.search("Jane Miller", tables=["core.guardians"], limit=1)
    if not hits:
        return {
            "display_name": "Jane Miller",
        }

    row = hits[0].row
    return {
        "display_name": row.get("full_name", "Jane Miller"),
        "guardian_id": row.get("guardian_id", ""),
        "account_id": row.get("account_id", ""),
        "email": row.get("email", ""),
        "phone": row.get("phone", ""),
        "city": row.get("city", ""),
        "country": row.get("country", ""),
    }


def build_analysis_context(assistant_message: str | None = None) -> tuple[list[dict[str, str]], dict[str, str]]:
    history = [
        {
            "role": str(item.get("role", "")),
            "content": str(item.get("content", "")),
        }
        for item in st.session_state.get("conversation_history", [])
        if item.get("content") != "Опрацьовуємо ваше запитання..."
    ]
    if assistant_message:
        history.append({"role": "assistant", "content": assistant_message})
    user_profile = {
        "display_name": str(st.session_state.get("user_name", "")),
        **{
            str(key): str(value)
            for key, value in dict(st.session_state.get("known_user_identifiers", {})).items()
            if value
        },
    }
    return history, user_profile


def run_support_turn(conversation_id: str, user_name: str, message: str) -> tuple[SupportTurnInput, AnalysisResult | None]:
    worker_instruction = None
    pending_instruction = st.session_state.pop("pending_worker_instruction", None)
    known_identifiers = dict(st.session_state.get("known_user_identifiers", {}))
    if pending_instruction:
        worker_instruction = WorkerInstruction(task=pending_instruction)

    turn = SupportTurnInput(
        conversation_id=conversation_id,
        user_message=message,
        user_metadata=UserMetadata(
            display_name=user_name,
            known_identifiers=known_identifiers,
        ),
        worker_instruction=worker_instruction,
    )
    with SupportAgentService.from_env() as support_service:
        support_result = support_service.invoke(turn)

    st.session_state["last_support_result"] = support_result
    st.session_state["known_user_identifiers"] = dict(support_result.known_user_identifiers)

    analysis_result: AnalysisResult | None = None
    try:
        analysis_history, analysis_user_profile = build_analysis_context(support_result.assistant_message)
        with AnalysisAgentService.from_env() as analysis_service:
            analysis_result = analysis_service.invoke(
                AnalysisTurnInput(
                    conversation_id=conversation_id,
                    domain_key=turn.domain_key,
                    conversation_history=analysis_history,
                    user_profile=analysis_user_profile,
                )
            )
        st.session_state["last_analysis_result"] = analysis_result
    except Exception as exc:  # pragma: no cover - UI feedback only
        st.session_state["analysis_error"] = str(exc)

    return turn, analysis_result


def render_history(history: List[dict]) -> None:
    for item in history:
        role = item.get("role", "unknown")
        content = item.get("content", "")
        st.markdown(f"**{role}**: {content}")


def render_chat_history(history: List[dict]) -> None:
    chat_roles = {"user", "assistant", "assistant_draft", "manager"}
    for item in history:  # show full thread; user can scroll the page
        role = item.get("role")
        if role not in chat_roles:
            continue
        content = item.get("content", "")
        streamlit_role = "assistant" if role.startswith("assistant") else ("user" if role == "user" else "assistant")
        with st.chat_message(streamlit_role):
            st.write(content)


def render_chat_tab():
    st.header("Chat")
    # Initialize fresh chat state per session
    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = f"chat-{uuid4().hex[:8]}"
    default_chat_user = load_default_chat_user()
    if "user_name" not in st.session_state:
        st.session_state["user_name"] = default_chat_user.get("display_name", "Jane Miller")
    if "known_user_identifiers" not in st.session_state:
        st.session_state["known_user_identifiers"] = {
            key: value
            for key, value in default_chat_user.items()
            if key != "display_name" and value
        }
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = [
            {"role": "assistant", "content": "Чим я можу вам допомогти?"}
        ]
    if "manager_takeover" not in st.session_state:
        st.session_state["manager_takeover"] = False
    if "pending_support_message" not in st.session_state:
        st.session_state["pending_support_message"] = None

    # One-click reset to start a brand-new chat (no previous messages)
    if st.button("Почати новий чат"):
        st.session_state["conversation_id"] = f"chat-{uuid4().hex[:8]}"
        st.session_state["conversation_history"] = [
            {"role": "assistant", "content": "Чим я можу вам допомогти?"}
        ]
        st.session_state["user_name"] = default_chat_user.get("display_name", "Jane Miller")
        st.session_state["known_user_identifiers"] = {
            key: value
            for key, value in default_chat_user.items()
            if key != "display_name" and value
        }
        st.session_state["last_support_result"] = None
        st.session_state["last_analysis_result"] = None
        st.session_state["pending_worker_instruction"] = None
        st.session_state["pending_support_message"] = None
        st.session_state["manager_takeover"] = False
        st.rerun()

    conversation_id = st.session_state["conversation_id"]
    user_name = st.session_state["user_name"]
    manager_takeover = st.session_state.get("manager_takeover", False)
    pending_support_message = st.session_state.get("pending_support_message")

    history = st.session_state.get("conversation_history") or []
    render_chat_history(history)

    if pending_support_message:
        try:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            st.session_state["conversation_id"] = conversation_id
            st.session_state["user_name"] = user_name
            run_support_turn(conversation_id, user_name, pending_support_message)
            result = st.session_state.get("last_support_result")
            if result:
                updated_history = list(st.session_state.get("conversation_history", []))
                if updated_history and updated_history[-1].get("content") == "Опрацьовуємо ваше запитання...":
                    updated_history[-1] = {"role": "assistant", "content": result.assistant_message}
                else:
                    updated_history.append({"role": "assistant", "content": result.assistant_message})
                st.session_state["conversation_history"] = updated_history
        except Exception as exc:
            updated_history = list(st.session_state.get("conversation_history", []))
            if updated_history and updated_history[-1].get("content") == "Опрацьовуємо ваше запитання...":
                updated_history[-1] = {"role": "assistant", "content": f"Сталася помилка: {exc}"}
            else:
                updated_history.append({"role": "assistant", "content": f"Сталася помилка: {exc}"})
            st.session_state["conversation_history"] = updated_history
        finally:
            st.session_state["pending_support_message"] = None
        st.rerun()

    chat_input = st.chat_input("Напишіть повідомлення")
    if chat_input:
        updated_history = list(st.session_state.get("conversation_history", []))
        updated_history.append({"role": "user", "content": chat_input})
        updated_history.append({"role": "assistant", "content": "Опрацьовуємо ваше запитання..."})
        st.session_state["conversation_history"] = updated_history
        st.session_state["pending_support_message"] = chat_input
        st.rerun()


def render_support_tab():
    st.header("Support")
    repo = load_analysis_repo()
    results = repo.list_all()

    if not results:
        st.info("No conversations have been analyzed yet.")
        return

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Conversations")
        display_items = []
        for res in results:
            icon = _priority_icon(res.priority)
            intent = res.issue_type or "unknown"
            display_items.append((f"{icon} {res.conversation_id} — {intent}", res.conversation_id, res.priority))

        display_items.sort(key=lambda item: {"urgent": 0, "medium": 1, "low": 2}.get(item[2], 3))
        labels = [label for label, _, _ in display_items]
        choices = [cid for _, cid, _ in display_items]
        selected = st.radio(
            "Виберіть розмову",
            options=choices,
            format_func=lambda cid: labels[choices.index(cid)],
            label_visibility="visible",
        )

    with right:
        selected_result = next((res for res in results if res.conversation_id == selected), results[0])
        st.subheader(f"Conversation {selected_result.conversation_id}")
        st.markdown(_priority_badge(selected_result.priority), unsafe_allow_html=True)
        st.write("Issue type:", selected_result.issue_type)
        st.write("Summary:")
        st.write(selected_result.summary)
        st.write("Recommended next steps:")
        suggested = selected_result.next_steps or []
        chosen_steps = st.multiselect("Select suggested steps", options=suggested, default=suggested[:2] if len(suggested) >= 2 else suggested)
        custom_steps_text = st.text_area("Add or replace instructions", value="", placeholder="Write custom instructions for the support agent or your own reply")
        action = st.radio(
            "Action",
            options=["manager_continue", "send_to_agent"],
            format_func=lambda v: "Manager will reply next" if v == "manager_continue" else "Send instructions to support agent",
        )

        if st.button("Apply decision"):
            combined_instructions = []
            combined_instructions.extend(chosen_steps)
            if custom_steps_text.strip():
                combined_instructions.append(custom_steps_text.strip())

            if action == "manager_continue":
                st.session_state["manager_takeover"] = True
                st.session_state["pending_worker_instruction"] = None
                st.success("Manager will reply next in Chat tab.")
            else:
                st.session_state["manager_takeover"] = False
                st.session_state["pending_worker_instruction"] = "\n".join(combined_instructions)
                st.success("Instructions saved. Support Agent will use them on the next reply.")

        st.write("Detected issues:")
        st.json(selected_result.detected_issues)
        st.write("Full dialogue:")
        render_history(selected_result.conversation_history)


chat_tab, support_tab = st.tabs(["Chat", "Support"])
with chat_tab:
    render_chat_tab()
with support_tab:
    render_support_tab()
