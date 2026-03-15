from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from app_core.orchestrator import AppOrchestrator
from app_core.store import ConversationStore

from .app_shell import LEFT_RATIO, RIGHT_RATIO, render_app_shell


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
        section[data-testid="stSidebar"] {
          display: none !important;
        }
        .st-key-dialogues-shell {
          background: transparent;
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
          color: var(--ink) !important;
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
          background: transparent;
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
          background: #fffdfa;
          border: 1px solid #eee1cf;
          border-radius: 18px;
          padding: 0.7rem 0.8rem;
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
        .st-key-dialogues-create {
          max-width: 340px;
          width: min(340px, 100%);
          margin-left: auto;
          margin-right: auto;
        }
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
        /* Keep left panel headings/labels consistent with light background. */
        .st-key-dialogues-shell .stMarkdown h1,
        .st-key-dialogues-shell .stMarkdown h2,
        .st-key-dialogues-shell .stMarkdown h3,
        .st-key-dialogues-shell .stMarkdown p,
        .st-key-dialogues-shell .stCaption,
        .st-key-dialogues-shell label,
        .st-key-dialogues-shell [data-testid="stMarkdownContainer"] * {
          color: var(--ink) !important;
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
    st.markdown(
        """
        <style>
        .stApp {
          background:
            radial-gradient(circle at top right, rgba(222, 183, 127, 0.18), transparent 28%),
            linear-gradient(180deg, #f8f4ed 0%, #f1e8dc 100%) !important;
        }
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


def main() -> None:
    configure_page()
    render_top_line()

    store = get_store()
    orchestrator = get_orchestrator()

    conversations = store.list()
    render_split_background(LEFT_RATIO, RIGHT_RATIO)
    render_app_shell(orchestrator, conversations)


if __name__ == "__main__":
    main()
