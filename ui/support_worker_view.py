from __future__ import annotations

from time import sleep
from typing import Iterable

import streamlit as st

from agents.support_agent.contracts import Tone, WorkerInstruction
from app_core.orchestrator import AppOrchestrator
from app_core.store import ConversationRecord


STATUS_COLORS = {
    "normal": ("#2f855a", "#e6ffef"),
    "watch": ("#b7791f", "#fff6dd"),
    "escalated": ("#b83280", "#ffe6f4"),
    "waiting_for_worker": ("#c05621", "#fff1e8"),
    "worker_reviewing": ("#2b6cb0", "#e9f3ff"),
    "resolved": ("#276749", "#e6ffef"),
}


def _split_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _tone_label(tone: str) -> str:
    labels = {
        Tone.FORMAL.value: "formal",
        Tone.FRIENDLY_AND_CALM.value: "friendly_and_calm",
        Tone.SHORT_AND_NEUTRAL.value: "short_and_neutral",
    }
    return labels[tone]


def _render_status(status: str) -> str:
    fg, bg = STATUS_COLORS.get(status, ("#334155", "#e2e8f0"))
    label = status.replace("_", " ")
    return (
        f"<span class='status-pill' style='color:{fg};background:{bg};border-color:{fg}33;'>"
        f"{label}</span>"
    )


def _render_list(items: Iterable[str], empty_text: str) -> None:
    values = [item for item in items if item]
    if not values:
        st.caption(empty_text)
        return
    for item in values:
        st.markdown(f"- {item}")


def _instruction_defaults(record: ConversationRecord) -> dict:
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


def _resolved_label_defaults(record: ConversationRecord) -> dict[str, str]:
    return {
        "status": record.resolved_label_status or record.status,
        "category": record.resolved_label_category or record.dialogue_category,
        "intent": record.resolved_label_intent or record.dialogue_intent,
        "summary": record.resolved_label_summary or record.dialogue_summary_text,
    }


def _render_overview(record: ConversationRecord) -> None:
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
                st.markdown(_render_status(record.status), unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            with st.container(key="overview-category-card"):
                st.caption("Category")
                st.write(record.dialogue_category or "Not set")
        with col4:
            with st.container(key="overview-intent-card"):
                st.caption("Intent")
                st.write(record.dialogue_intent or "Not set")
        with st.container(key="overview-summary-card"):
            st.caption("Summary")
            st.write(record.last_summary or "No analysis summary yet.")


def _render_reason_card(reason: dict, index: int) -> None:
    evidence = [item for item in reason.get("evidence", []) if item]
    with st.container(key=f"reason-card-{index}"):
        st.markdown(f"##### {reason.get('code', f'Reason {index + 1}').replace('_', ' ')}")
        st.write(reason.get("description") or "No description.")
        if evidence:
            st.caption("Evidence")
            for item in evidence:
                st.write(item)


def _render_next_steps_checklist(record: ConversationRecord, steps: list[str], key_prefix: str) -> None:
    if not steps:
        st.caption("No next steps.")
        return

    for index, step in enumerate(steps):
        st.checkbox(step, value=True, key=f"{record.conversation_id}-{key_prefix}-step-{index}")


def _render_analysis_package(record: ConversationRecord) -> None:
    analysis_result = record.analysis_result or {}
    reasons = analysis_result.get("reasons", [])
    with st.container(key="analysis-card"):
        st.markdown("#### Escalation reasons")
        if reasons:
            for index, reason in enumerate(reasons):
                _render_reason_card(reason, index)
        else:
            st.caption("No active escalation reasons.")


def _render_worker_form(orchestrator: AppOrchestrator, record: ConversationRecord) -> None:
    defaults = _instruction_defaults(record)
    tone_options = [tone.value for tone in Tone]
    default_tone = defaults.get("tone", Tone.FORMAL.value)
    tone_index = tone_options.index(default_tone) if default_tone in tone_options else 0
    submitted = False
    mark_resolved_clicked = False

    with st.container(key="worker-card"):
        st.markdown("#### Instructions")
        worker_package = record.worker_package or {}
        suggested_steps = (record.worker_package or {}).get(
            "possible_next_steps",
            (record.analysis_result or {}).get("possible_next_steps", []),
        )
        recommended_tone = worker_package.get(
            "recommended_tone", (record.analysis_result or {}).get("recommended_tone", Tone.FORMAL.value)
        )
        suggested_constraints = worker_package.get(
            "suggested_constraints",
            (record.analysis_result or {}).get("suggested_constraints", []),
        )

        col1, col2, col3 = st.columns([1.2, 1.3, 1.5])
        with col1:
            with st.container(key="worker-steps-card"):
                st.markdown("##### Next steps")
                _render_next_steps_checklist(record, suggested_steps, "worker")
        with col2:
            with st.container(key="worker-constraints-card"):
                st.markdown("##### Suggested constraints")
                _render_list(suggested_constraints, "No constraints.")
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
                tone = st.selectbox("Tone", options=tone_options, index=tone_index, format_func=_tone_label)
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
            constraints=_split_lines(constraints),
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


def _render_draft_controls(orchestrator: AppOrchestrator, record: ConversationRecord) -> None:
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


def _render_resolved_labeling_block(orchestrator: AppOrchestrator, record: ConversationRecord) -> None:
    if record.status != "resolved":
        return

    defaults = _resolved_label_defaults(record)
    with st.container(key="resolved-labeling-card"):
        st.markdown("#### Labeling")
        st.caption("Review the LLM labels and update them if needed before submitting.")
        with st.form(f"resolved-labeling-{record.conversation_id}"):
            status = st.text_input("Status", value=str(defaults["status"]))
            category = st.text_input("Category", value=str(defaults["category"]))
            intent = st.text_input("Intent", value=str(defaults["intent"]))
            summary = st.text_area("Summary", value=str(defaults["summary"]), height=180)
            submit_col, _ = st.columns([1, 3.5])
            with submit_col:
                submitted = st.form_submit_button("Submit", use_container_width=True)

    if submitted:
        updated = orchestrator.save_resolved_labeling(
            record.conversation_id,
            status=status,
            category=category,
            intent=intent,
            summary=summary,
        )
        st.session_state["selected_conversation_id"] = updated.conversation_id
        st.rerun()


def render_support_worker_view(orchestrator: AppOrchestrator, record: ConversationRecord) -> None:
    _render_overview(record)
    _render_analysis_package(record)

    with st.container(key="dialogue-card"):
        with st.expander("Full dialogue", expanded=False):
            for message in record.messages:
                st.markdown(f"**{message.role}**")
                st.write(message.content)

    _render_worker_form(orchestrator, record)
    _render_draft_controls(orchestrator, record)
    _render_resolved_labeling_block(orchestrator, record)
