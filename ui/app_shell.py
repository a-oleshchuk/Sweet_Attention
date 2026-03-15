from __future__ import annotations

import streamlit as st

from app_core.orchestrator import AppOrchestrator
from app_core.store import ConversationRecord

from .dialogues_panel import render_dialogues_panel
from .support_worker_view import render_support_worker_view
from .user_view import render_user_view


LEFT_RATIO = 1.05
RIGHT_RATIO = 3.45


def render_app_shell(orchestrator: AppOrchestrator, conversations: list[ConversationRecord]) -> None:
    left_col, right_col = st.columns([LEFT_RATIO, RIGHT_RATIO], gap="medium")

    with left_col:
        record = render_dialogues_panel(orchestrator, conversations)

    if record is None:
        st.info("Create a conversation in the dialogues panel to start.")
        return

    with right_col:
        st.markdown("<div class='main-top-spacer'></div>", unsafe_allow_html=True)
        st.title("Support Desk")
        user_tab, worker_tab = st.tabs(["User view", "Support worker view"])
        with user_tab:
            render_user_view(orchestrator, record)
        with worker_tab:
            render_support_worker_view(orchestrator, record)
