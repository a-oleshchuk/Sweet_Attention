from __future__ import annotations

import streamlit as st

from app_core.orchestrator import AppOrchestrator
from app_core.store import ConversationRecord


def _select_conversation(conversations: list[ConversationRecord]) -> ConversationRecord | None:
    if not conversations:
        return None

    current = st.session_state.get("selected_conversation_id", conversations[0].conversation_id)
    if current not in {record.conversation_id for record in conversations}:
        current = conversations[0].conversation_id

    st.markdown("## Dialogues")
    with st.container(key="conversation-list"):
        for record in conversations:
            label = (
                f"{record.conversation_id}\n"
                f"{record.user_display_name}\n"
                f"{record.status.replace('_', ' ')}"
            )
            button_type = "primary" if record.conversation_id == current else "secondary"
            if st.button(
                label,
                key=f"conversation-button-{record.conversation_id}",
                use_container_width=True,
                type=button_type,
            ):
                st.session_state["selected_conversation_id"] = record.conversation_id
                st.rerun()

    selected_id = st.session_state.get("selected_conversation_id", current)
    return next(record for record in conversations if record.conversation_id == selected_id)


def _create_conversation(orchestrator: AppOrchestrator) -> None:
    with st.container(key="dialogues-create"):
        st.markdown("## New dialogue")
        with st.form("create-conversation", clear_on_submit=True, enter_to_submit=False):
            user_name = st.text_input("User name")
            submitted = st.form_submit_button("Create")
        if submitted and user_name.strip():
            created = orchestrator.store.create(user_display_name=user_name.strip())
            st.session_state["selected_conversation_id"] = created.conversation_id
            st.rerun()


def render_dialogues_panel(
    orchestrator: AppOrchestrator,
    conversations: list[ConversationRecord],
) -> ConversationRecord | None:
    with st.container(key="dialogues-shell"):
        with st.container(key="dialogues-panel"):
            _create_conversation(orchestrator)
            return _select_conversation(conversations)
