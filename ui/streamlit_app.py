from __future__ import annotations

from time import sleep
from typing import Iterable

import streamlit as st
import streamlit.components.v1 as components

from agents.support_agent.contracts import Tone, WorkerInstruction
from app_core.orchestrator import AppOrchestrator
from app_core.store import ConversationRecord, ConversationStore


STATUS_COLORS = {
    "normal": ("#2f855a", "#e6ffef"),
    "watch": ("#b7791f", "#fff6dd"),
    "escalated": ("#b83280", "#ffe6f4"),
    "waiting_for_worker": ("#c05621", "#fff1e8"),
    "worker_reviewing": ("#2b6cb0", "#e9f3ff"),
    "resolved": ("#276749", "#e6ffef"),
}


def configure_page() -> None:
    st.set_page_config(
        page_title="Support Desk Supervisor",
        page_icon="SD",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        :root {
          --bg: #f6f1e8;
          --panel: #fffdf8;
          --ink: #1f2933;
          --muted: #5a6775;
          --line: #e2d8c7;
          --accent: #9f5f3d;
        }
        .stApp {
          background:
            radial-gradient(circle at top right, rgba(222, 183, 127, 0.18), transparent 28%),
            linear-gradient(180deg, #f8f4ed 0%, #f1e8dc 100%);
          color: var(--ink);
        }
        /* Make page split read as two real parts: dark left rail + light main area */
        [data-testid="stHorizontalBlock"] > div:first-child {
          background: linear-gradient(180deg, #262a37 0%, #1d2230 100%);
          min-height: 100vh;
          padding: 0;
        }
        [data-testid="stHorizontalBlock"] > div:first-child > div {
          height: 100%;
        }
        section[data-testid="stSidebar"] {
          display: none !important;
        }
        .st-key-dialogues-shell {
          background: #1f2534;
          border: none;
          border-radius: 0;
          padding: calc(52px + 1rem) 1rem 2rem;
          box-shadow: none;
          min-height: calc(100vh + 8rem);
          margin: 0;
        }
        .st-key-dialogues-shell,
        .st-key-dialogues-shell > div,
        .st-key-dialogues-shell * {
          color: #eef2fb !important;
        }
        .st-key-dialogues-shell .stTextInput input,
        .st-key-dialogues-shell .stTextArea textarea {
          background: #ffffff !important;
          color: var(--ink) !important;
        }
        .st-key-dialogues-shell .stButton > button,
        .st-key-dialogues-shell .stFormSubmitButton > button {
          background: #ffffff !important;
          color: var(--ink) !important;
          border: 1px solid #d8deeb !important;
        }
        .st-key-dialogues-shell .stFormSubmitButton > button:disabled,
        .st-key-dialogues-shell .stButton > button:disabled {
          color: #111111 !important;
          -webkit-text-fill-color: #111111 !important;
          opacity: 1 !important;
        }
        .st-key-dialogues-shell .stMarkdown h1,
        .st-key-dialogues-shell .stMarkdown h2,
        .st-key-dialogues-shell .stMarkdown h3,
        .st-key-dialogues-shell .stMarkdown p {
          color: #eef2fb !important;
        }
        .st-key-dialogues-panel {
          background: #1f2534;
          border: none;
          border-radius: 0;
          padding: 0;
          box-shadow: none;
        }
        .st-key-dialogues-panel > div {
          display: flex;
          flex-direction: column;
          align-items: center;
          width: 100%;
        }
        .st-key-dialogues-create {
          background: #1f2534;
          border: 1px solid #39415a;
          border-radius: 18px;
          padding: 0.85rem 0.95rem;
          margin-bottom: 1rem;
        }
        .st-key-conversation-list .stButton > button {
          min-height: 5.5rem;
          height: auto !important;
          white-space: pre-line !important;
          text-align: left !important;
          justify-content: flex-start !important;
          align-items: flex-start !important;
          line-height: 1.35 !important;
          padding: 0.95rem 1rem !important;
          border-radius: 18px !important;
        }
        .st-key-conversation-list .stButton > button[kind="primary"] {
          background: #ffffff !important;
          border-color: #d6ddea !important;
        }
        .st-key-conversation-list .stButton > button p {
          font-size: 0.98rem !important;
          margin: 0 !important;
          text-align: left !important;
        }
        .st-key-conversation-list .stButton > button > div {
          text-align: left !important;
          width: 100% !important;
        }
        .st-key-dialogues-shell .st-key-conversation-list .stButton > button p,
        .st-key-dialogues-shell .st-key-conversation-list .stButton > button div,
        .st-key-dialogues-shell .st-key-conversation-list .stButton > button span {
          color: var(--ink) !important;
        }
        .st-key-dialogues-create,
        .st-key-conversation-list {
          max-width: 390px;
          width: min(390px, 100%);
          margin-left: auto;
          margin-right: auto;
        }
        header[data-testid="stHeader"] {
          display: none !important;
        }
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
          display: none !important;
        }
        [data-testid="stTabs"] [role="tablist"] {
          gap: 1rem;
          border-bottom: 1px solid var(--line);
          margin-bottom: 1.25rem;
        }
        [data-testid="stTabs"] [role="tab"] {
          font-size: 1.2rem !important;
          font-weight: 700 !important;
          padding: 0.6rem 1rem 0.7rem !important;
          border-radius: 14px 14px 0 0;
          background: rgba(255, 253, 248, 0.72);
          border: 1px solid transparent;
          border-bottom: none;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="false"] {
          color: var(--muted) !important;
          border-color: #eadbc7;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
          background: #ffffff;
          border-color: var(--line);
          color: var(--ink) !important;
        }
        .stApp,
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp span,
        .stApp div,
        .stMarkdown,
        .stCaption,
        .stText,
        .stChatMessage [data-testid="stMarkdownContainer"] *,
        .stChatMessage [data-testid="stChatMessageContent"] * {
          color: var(--ink) !important;
        }
        .stApp a {
          color: #0f62fe !important;
        }
        .stButton > button,
        .stFormSubmitButton > button,
        .stDownloadButton > button,
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-primary"] {
          background: #ffffff !important;
          color: var(--ink) !important;
          border: 1px solid var(--line) !important;
          box-shadow: 0 4px 14px rgba(71, 55, 39, 0.08);
        }
        .stButton > button:hover,
        .stFormSubmitButton > button:hover,
        .stDownloadButton > button:hover {
          background: #fbf8f2 !important;
          border-color: #ccb79a !important;
        }
        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div,
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] input {
          background: #ffffff !important;
          color: var(--ink) !important;
          border-color: var(--line) !important;
        }
        .stTextArea textarea:disabled,
        .stTextInput input:disabled {
          color: var(--ink) !important;
          -webkit-text-fill-color: var(--ink) !important;
          opacity: 1 !important;
          background: #fffdfa !important;
        }
        [data-testid="stChatInput"] {
          background: var(--panel) !important;
          border: 1px solid var(--line) !important;
          border-radius: 18px !important;
          padding: 0.45rem !important;
          box-shadow: 0 10px 30px rgba(71, 55, 39, 0.06);
        }
        [data-testid="stChatInput"] > div {
          background: transparent !important;
        }
        [data-testid="stChatInput"] button {
          background: #ffffff !important;
          color: var(--ink) !important;
          border: 1px solid var(--line) !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
          background: #ffffff;
          border: 1px solid var(--line);
          border-radius: 18px;
          box-shadow: 0 10px 30px rgba(71, 55, 39, 0.06);
        }
        .st-key-overview-card,
        .st-key-analysis-card,
        .st-key-worker-card,
        .st-key-dialogue-card {
          background: #ffffff;
          border: 1px solid var(--line);
          border-radius: 22px;
          padding: 1.15rem 1.25rem;
          box-shadow: 0 10px 30px rgba(71, 55, 39, 0.06);
        }
        .st-key-overview-status-card,
        .st-key-overview-user-card,
        .st-key-overview-summary-card,
        .st-key-worker-steps-card,
        .st-key-worker-tone-card,
        .st-key-worker-constraints-card,
        .st-key-worker-form-card,
        .st-key-dialogue-inner-card {
          background: #fffdfa;
          border: 1px solid #eee1cf;
          border-radius: 18px;
          padding: 1rem 1.1rem;
        }
        .st-key-overview-user-card p {
          font-size: 0.95rem !important;
        }
        .st-key-overview-user-card .stCaption {
          font-size: 0.9rem !important;
        }
        .st-key-dialogue-card [data-testid="stExpander"] {
          background: #fffdfa !important;
          border: 1px solid #eee1cf !important;
          border-radius: 18px !important;
          overflow: hidden;
        }
        .st-key-dialogue-card details {
          background: #fffdfa !important;
        }
        .st-key-dialogue-card summary {
          background: #fffdfa !important;
          color: var(--ink) !important;
          font-size: 1.85rem !important;
          font-weight: 700 !important;
          padding: 1rem 1.1rem !important;
        }
        .st-key-dialogue-card [data-testid="stExpanderDetails"] {
          background: #ffffff !important;
          border-top: 1px solid #eee1cf;
          padding: 0.35rem 1.1rem 1rem !important;
        }
        .draft-fade-out {
          animation: draftFadeOut 240ms ease forwards;
          transform-origin: top;
        }
        @keyframes draftFadeOut {
          from { opacity: 1; transform: translateY(0); max-height: 420px; }
          to { opacity: 0; transform: translateY(-8px); max-height: 0; margin: 0; }
        }
        .st-key-reason-card-0,
        .st-key-reason-card-1,
        .st-key-reason-card-2,
        .st-key-reason-card-3,
        .st-key-reason-card-4,
        .st-key-reason-card-5,
        .st-key-reason-card-6,
        .st-key-reason-card-7,
        .st-key-reason-card-8,
        .st-key-reason-card-9 {
          background: #fffdfa;
          border: 1px solid #eee1cf;
          border-radius: 18px;
          padding: 1rem 1.1rem;
        }
        .info-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 1rem;
        }
        .analysis-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 1rem;
        }
        .info-card-body {
          display: flex;
          flex-direction: column;
          gap: 0.8rem;
        }
        .analysis-card {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 18px;
          padding: 1rem 1.1rem;
          box-shadow: 0 10px 30px rgba(71, 55, 39, 0.06);
          height: 100%;
        }
        .analysis-item {
          border: 1px solid #eee1cf;
          background: #fffcf6;
          border-radius: 14px;
          padding: 0.9rem 1rem;
          margin-bottom: 0.8rem;
        }
        .analysis-code {
          font-size: 0.9rem;
          font-weight: 800;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          color: #7a4b2f;
          margin-bottom: 0.5rem;
        }
        .analysis-evidence {
          margin-top: 0.75rem;
          padding-top: 0.75rem;
          border-top: 1px solid #eee1cf;
          color: var(--muted);
          white-space: pre-wrap;
          overflow-wrap: anywhere;
        }
        .summary-text {
          white-space: pre-wrap;
          overflow-wrap: anywhere;
          line-height: 1.5;
        }
        @media (max-width: 900px) {
          .info-grid,
          .analysis-grid {
            grid-template-columns: 1fr;
          }
        }
        .block-container {
          padding-top: 0;
          padding-bottom: 2rem;
        }
        .main-top-spacer {
          height: calc(52px + 1rem);
        }
        .status-pill {
          display: inline-block;
          padding: 0.3rem 0.7rem;
          border-radius: 999px;
          font-size: 0.85rem;
          font-weight: 700;
          letter-spacing: 0.02em;
          border: 1px solid transparent;
        }
        .info-card {
          background: #ffffff;
          border: 1px solid var(--line);
          border-radius: 18px;
          padding: 1rem 1.1rem;
          box-shadow: 0 10px 30px rgba(71, 55, 39, 0.06);
        }
        .section-title {
          font-size: 0.95rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: var(--muted);
          margin-bottom: 0.7rem;
        }
        /* Keep left panel headings/labels readable over dark background. */
        .st-key-dialogues-shell .stMarkdown h1,
        .st-key-dialogues-shell .stMarkdown h2,
        .st-key-dialogues-shell .stMarkdown h3,
        .st-key-dialogues-shell .stMarkdown p,
        .st-key-dialogues-shell .stCaption,
        .st-key-dialogues-shell label,
        .st-key-dialogues-shell [data-testid="stMarkdownContainer"] * {
          color: #eef2fb !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_store() -> ConversationStore:
    return ConversationStore()


def get_orchestrator() -> AppOrchestrator:
    return AppOrchestrator(get_store())


def render_split_background(left_ratio: float, right_ratio: float) -> None:
    left_pct = (left_ratio / (left_ratio + right_ratio)) * 100
    st.markdown(
        f"""
        <style>
        .stApp {{
          background: linear-gradient(
            90deg,
            #1f2534 0%,
            #1f2534 {left_pct:.4f}%,
            #f8f4ed {left_pct:.4f}%,
            #f1e8dc 100%
          ) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_line() -> None:
    components.html(
        """
        <script>
        const parentDoc = window.parent.document;
        let bar = parentDoc.getElementById("support-desk-top-line");
        if (!bar) {
          bar = parentDoc.createElement("div");
          bar.id = "support-desk-top-line";
          Object.assign(bar.style, {
            position: "fixed",
            top: "0",
            left: "0",
            right: "0",
            height: "52px",
            background: "#1f2534",
            zIndex: "9999",
            transform: "translateY(0)",
            transition: "transform 180ms ease"
          });
          parentDoc.body.appendChild(bar);
        }

        const updateBar = () => {
          const scrollTop = window.parent.scrollY || parentDoc.documentElement.scrollTop || 0;
          bar.style.transform = scrollTop > 10 ? "translateY(-120%)" : "translateY(0)";
        };

        if (!window.parent.__supportDeskTopLineBound) {
          window.parent.addEventListener("scroll", updateBar, { passive: true });
          window.parent.__supportDeskTopLineBound = true;
        }

        updateBar();
        </script>
        """,
        height=0,
        width=0,
    )


def split_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def tone_label(tone: str) -> str:
    labels = {
        Tone.FORMAL.value: "formal",
        Tone.FRIENDLY_AND_CALM.value: "friendly_and_calm",
        Tone.SHORT_AND_NEUTRAL.value: "short_and_neutral",
    }
    return labels[tone]


def render_status(status: str) -> str:
    fg, bg = STATUS_COLORS.get(status, ("#334155", "#e2e8f0"))
    label = status.replace("_", " ")
    return (
        f"<span class='status-pill' style='color:{fg};background:{bg};border-color:{fg}33;'>"
        f"{label}</span>"
    )


def render_list(items: Iterable[str], empty_text: str) -> None:
    values = [item for item in items if item]
    if not values:
        st.caption(empty_text)
        return
    for item in values:
        st.markdown(f"- {item}")


def instruction_defaults(record: ConversationRecord) -> dict:
    if record.worker_instruction_history:
        defaults = dict(record.worker_instruction_history[-1])
    else:
        defaults = {
            "task": "",
            "constraints": [],
            "answer_format": "short clear reply to user",
            "tone": Tone.FORMAL.value,
            "review_before_send": True,
            "regenerate": True,
            "notes": [],
        }

    suggested_instruction = None
    if record.worker_package and record.worker_package.get("suggested_worker_instruction"):
        suggested_instruction = record.worker_package["suggested_worker_instruction"]
    elif (record.analysis_result or {}).get("suggested_worker_instruction"):
        suggested_instruction = record.analysis_result["suggested_worker_instruction"]

    if suggested_instruction:
        defaults["constraints"] = list(
            suggested_instruction.get(
                "constraints",
                (record.analysis_result or {}).get("suggested_constraints", defaults.get("constraints", [])),
            )
        )
        defaults["tone"] = suggested_instruction.get(
            "tone",
            (record.analysis_result or {}).get("recommended_tone", defaults.get("tone", Tone.FORMAL.value)),
        )
        defaults["review_before_send"] = bool(
            suggested_instruction.get("review_before_send", defaults.get("review_before_send", True))
        )
        defaults["regenerate"] = bool(suggested_instruction.get("regenerate", defaults.get("regenerate", True)))

    defaults["task"] = ""
    return defaults


def select_conversation(conversations: list[ConversationRecord]) -> ConversationRecord | None:
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


def sidebar_create_conversation(orchestrator: AppOrchestrator) -> None:
    with st.container(key="dialogues-create"):
        st.markdown("## New conversation")
        with st.form("create-conversation", clear_on_submit=True):
            user_name = st.text_input("User name")
            submitted = st.form_submit_button("Create")
        if submitted and user_name.strip():
            created = orchestrator.store.create(user_display_name=user_name.strip())
            st.session_state["selected_conversation_id"] = created.conversation_id
            st.rerun()


def render_overview(record: ConversationRecord) -> None:
    with st.container(key="overview-card"):
        st.markdown("#### Conversation overview")
        col1, col2 = st.columns(2)
        with col1:
            with st.container(key="overview-user-card"):
                st.caption("User")
                st.write(record.user_display_name)
                st.caption(record.conversation_id)
        with col2:
            with st.container(key="overview-status-card"):
                st.caption("Status")
                st.markdown(render_status(record.status), unsafe_allow_html=True)
        with st.container(key="overview-summary-card"):
            st.caption("Summary")
            st.write(record.last_summary or "No analysis summary yet.")


def render_reason_card(reason: dict, index: int) -> None:
    evidence = [item for item in reason.get("evidence", []) if item]
    with st.container(key=f"reason-card-{index}"):
        st.markdown(f"##### {reason.get('code', f'Reason {index + 1}').replace('_', ' ')}")
        st.write(reason.get("description") or "No description.")
        if evidence:
            st.caption("Evidence")
            for item in evidence:
                st.write(item)


def render_next_steps_checklist(record: ConversationRecord, steps: list[str], key_prefix: str) -> None:
    if not steps:
        st.caption("No next steps.")
        return

    for index, step in enumerate(steps):
        st.checkbox(step, value=True, key=f"{record.conversation_id}-{key_prefix}-step-{index}")


def render_user_tab(orchestrator: AppOrchestrator, record: ConversationRecord) -> None:
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
        with st.spinner("Support agent is responding..."):
            updated = orchestrator.send_user_message(record.conversation_id, prompt)
        st.session_state["selected_conversation_id"] = updated.conversation_id
        st.rerun()


def render_analysis_package(record: ConversationRecord) -> None:
    analysis_result = record.analysis_result or {}
    reasons = analysis_result.get("reasons", [])
    with st.container(key="analysis-card"):
        st.markdown("#### Escalation reasons")
        if reasons:
            for index, reason in enumerate(reasons):
                render_reason_card(reason, index)
        else:
            st.caption("No active escalation reasons.")


def render_worker_form(orchestrator: AppOrchestrator, record: ConversationRecord) -> None:
    defaults = instruction_defaults(record)
    tone_options = [tone.value for tone in Tone]
    default_tone = defaults.get("tone", Tone.FORMAL.value)
    tone_index = tone_options.index(default_tone) if default_tone in tone_options else 0

    with st.container(key="worker-card"):
        st.markdown("#### Instructions")
        worker_package = record.worker_package or {}
        suggested_steps = (record.worker_package or {}).get(
            "possible_next_steps",
            (record.analysis_result or {}).get("possible_next_steps", []),
        )
        recommended_tone = worker_package.get("recommended_tone", (record.analysis_result or {}).get("recommended_tone", Tone.FORMAL.value))
        suggested_constraints = worker_package.get(
            "suggested_constraints",
            (record.analysis_result or {}).get("suggested_constraints", []),
        )

        col1, col2, col3 = st.columns([1.2, 1.3, 1.5])
        with col1:
            with st.container(key="worker-steps-card"):
                st.markdown("##### Next steps")
                render_next_steps_checklist(record, suggested_steps, "worker")
        with col2:
            with st.container(key="worker-constraints-card"):
                st.markdown("##### Suggested constraints")
                render_list(suggested_constraints, "No constraints.")
        with col3:
            with st.container(key="worker-tone-card"):
                st.markdown("##### Recommended tone")
                st.write(recommended_tone)

        with st.container(key="worker-form-card"):
            with st.form(f"worker-instruction-{record.conversation_id}"):
                task = st.text_area("Task", value=defaults.get("task", ""), height=120)
                constraints = st.text_area(
                    "Constraints",
                    value="\n".join(defaults.get("constraints", [])),
                    height=130,
                )
                tone = st.selectbox("Tone", options=tone_options, index=tone_index, format_func=tone_label)
                review_before_send = st.checkbox(
                    "Review generated draft before sending to user",
                    value=bool(defaults.get("review_before_send", True)),
                )
                regenerate = st.checkbox(
                    "Treat this as a regeneration of the next reply",
                    value=bool(defaults.get("regenerate", True)),
                )
                action_col1, action_col2, _ = st.columns([1, 1, 3.5])
                with action_col1:
                    submitted = st.form_submit_button("Run support agent", use_container_width=True)
                with action_col2:
                    mark_resolved_clicked = st.form_submit_button("Mark resolved", use_container_width=True)

    if submitted:
        instruction = WorkerInstruction(
            task=task.strip(),
            constraints=split_lines(constraints),
            answer_format=defaults.get("answer_format", "short clear reply to user"),
            tone=Tone(tone),
            review_before_send=review_before_send,
            regenerate=regenerate,
            notes=defaults.get("notes", []),
        )
        with st.spinner("Generating the next support-agent response..."):
            updated = orchestrator.run_worker_instruction(record.conversation_id, instruction)
        st.session_state["selected_conversation_id"] = updated.conversation_id
        st.rerun()

    if mark_resolved_clicked:
        updated = orchestrator.mark_resolved(record.conversation_id)
        st.session_state["selected_conversation_id"] = updated.conversation_id
        st.rerun()


def render_draft_controls(orchestrator: AppOrchestrator, record: ConversationRecord) -> None:
    if not record.pending_draft:
        return

    draft_key = f"{record.conversation_id}-editable-draft"
    draft_source_key = f"{record.conversation_id}-editable-draft-source"
    if st.session_state.get(draft_source_key) != record.pending_draft:
        st.session_state[draft_key] = record.pending_draft
        st.session_state[draft_source_key] = record.pending_draft

    section = st.container()
    with section:
        st.markdown("### Draft overview")
        edited_draft = st.text_area("Draft", key=draft_key, height=220)
        col1, col2, _ = st.columns([1, 1, 3.5])
        with col1:
            send_clicked = st.button("Send draft to user", use_container_width=True)
        with col2:
            resolved_clicked = st.button("Mark resolved", use_container_width=True)

    if send_clicked:
        section.markdown(
            """
            <div class="draft-fade-out" style="height: 22rem; border-radius: 18px; background: rgba(255,253,248,0.92);"></div>
            """,
            unsafe_allow_html=True,
        )
        updated = orchestrator.approve_pending_draft(record.conversation_id, edited_draft)
        st.session_state["selected_conversation_id"] = updated.conversation_id
        st.session_state.pop(draft_key, None)
        st.session_state.pop(draft_source_key, None)
        sleep(0.24)
        st.rerun()

    if resolved_clicked:
        updated = orchestrator.mark_resolved(record.conversation_id)
        st.session_state["selected_conversation_id"] = updated.conversation_id
        st.session_state.pop(draft_key, None)
        st.session_state.pop(draft_source_key, None)
        st.rerun()


def render_worker_tab(orchestrator: AppOrchestrator, record: ConversationRecord) -> None:
    render_overview(record)
    render_analysis_package(record)

    with st.container(key="dialogue-card"):
        with st.expander("Full dialogue", expanded=False):
            for message in record.messages:
                st.markdown(f"**{message.role}**")
                st.write(message.content)

    render_worker_form(orchestrator, record)
    render_draft_controls(orchestrator, record)


def main() -> None:
    configure_page()
    render_top_line()

    store = get_store()
    orchestrator = get_orchestrator()

    conversations = store.list()

    left_ratio = 1.05
    right_ratio = 3.45
    render_split_background(left_ratio, right_ratio)
    left_col, right_col = st.columns([left_ratio, right_ratio], gap="medium")

    with left_col:
        with st.container(key="dialogues-shell"):
            with st.container(key="dialogues-panel"):
                sidebar_create_conversation(orchestrator)
                record = select_conversation(conversations)

    if record is None:
        st.info("Create a conversation in the dialogues panel to start.")
        return

    with right_col:
        st.markdown("<div class='main-top-spacer'></div>", unsafe_allow_html=True)
        st.title("Support Desk")
        user_tab, worker_tab = st.tabs(["User view", "Support worker view"])
        with user_tab:
            render_user_tab(orchestrator, record)
        with worker_tab:
            render_worker_tab(orchestrator, record)


if __name__ == "__main__":
    main()
