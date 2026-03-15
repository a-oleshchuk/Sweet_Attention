from __future__ import annotations

import streamlit as st

from app_core.orchestrator import AppOrchestrator
from app_core.store import ConversationRecord


def render_user_view(orchestrator: AppOrchestrator, record: ConversationRecord) -> None:
    for message in record.messages:
        avatar = "assistant" if message.role == "assistant" else "user"
        with st.chat_message(avatar):
            st.write(message.content)

    blocked = record.pending_review or record.status in {"waiting_for_worker", "worker_reviewing"}
    if record.pending_review:
        st.info("Support worker review is in progress. The draft has not been sent to the user yet.")
    elif record.status in {"waiting_for_worker", "worker_reviewing"}:
        st.info("This conversation is paused while the support worker reviews the escalation.")

    prompt = st.chat_input("Send a user message", disabled=blocked)
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.spinner("Support agent is responding..."):
            updated = orchestrator.send_user_message(record.conversation_id, prompt)
        st.session_state["selected_conversation_id"] = updated.conversation_id
        st.rerun()
