from __future__ import annotations

import re
from collections import Counter
from typing import Any

from agents.shared import AnalysisPolicy, CsvRepository, KnowledgeRepository
from agents.shared.repositories import normalize_text
from agents.support_agent.contracts import Tone, WorkerInstruction

from .contracts import AnalysisConversationStatus, AnalysisReason, DialogueMessage, SupportAgentSnapshot

_POSITIVE_ACKNOWLEDGEMENT_MARKERS = (
    "i see it",
    "got it",
    "that helps",
    "makes sense",
    "understood",
    "thank you",
    "thanks",
)

_REPEATED_COMPLAINT_CONTEXT_MARKERS = (
    "can't",
    "cannot",
    "cant",
    "unable",
    "blocked",
    "not working",
    "not fixed",
    "not resolved",
    "still no",
    "still can't",
    "still cannot",
    "still waiting",
    "issue",
    "problem",
    "wrong",
    "error",
    "failed",
    "fail",
    "access",
)

_SENSITIVE_REQUEST_PATTERNS = (
    r"\b(?:please\s+)?(?:share|provide|send|tell me|give me|reply with|type|enter|confirm)\b[^.!?\n]{{0,60}}\b{term}\b",
    r"\b(?:what(?:'s| is)\s+your|may i have|can you share|could you share|can you provide|could you provide|please confirm)\b[^.!?\n]{{0,60}}\b{term}\b",
)

_SAFE_SENSITIVE_CONTEXT_MARKERS = (
    "do not ask for",
    "don't ask for",
    "did not ask for",
    "didn't ask for",
    "will not ask for",
    "won't ask for",
    "never ask for",
    "do not share",
    "don't share",
    "do not send",
    "don't send",
    "do not provide",
    "don't provide",
    "do not include",
    "don't include",
    "without sharing",
    "without sending",
)

_POLICY_TOPIC_MARKERS = (
    "policy",
    "eligible",
    "eligibility",
    "refund",
    "refundable",
    "refunds",
    "reimbursement",
    "cancel",
    "cancellation",
    "subscription",
    "reschedule",
    "class credit",
    "trial",
    "teacher change",
    "attendance policy",
)


def _contains_any(text: str, phrases: list[str]) -> list[str]:
    normalized = normalize_text(text)
    return [phrase for phrase in phrases if normalize_text(phrase) in normalized]


def _latest_message(dialogue: list[DialogueMessage], role: str) -> DialogueMessage | None:
    for message in reversed(dialogue):
        if message.role == role:
            return message
    return None


def _visible_messages(dialogue: list[DialogueMessage]) -> list[DialogueMessage]:
    return [message for message in dialogue if message.role in {"user", "assistant"}]


def _assistant_messages(dialogue: list[DialogueMessage]) -> list[DialogueMessage]:
    return [message for message in dialogue if message.role == "assistant"]


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _sentence_chunks(text: str) -> list[str]:
    return [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+|\n+", normalize_text(text)) if chunk.strip()]


def _is_clarifying_question(text: str, policy: AnalysisPolicy) -> bool:
    normalized = normalize_text(text)
    return bool(_contains_any(normalized, policy.patterns.clarifying_question_markers))


def _has_factual_claim(text: str, policy: AnalysisPolicy) -> bool:
    normalized = normalize_text(text)
    if _contains_any(normalized, policy.patterns.factual_claim_markers):
        return True
    if re.search(r"\b(g\d{4}|s\d{4}|sub\d{4}|inv\d{4})\b", normalized):
        return True
    return bool(re.search(r"\b\d{4}-\d{2}-\d{2}\b", normalized))


def _has_policy_guidance(text: str, policy: AnalysisPolicy) -> bool:
    normalized = normalize_text(text)
    markers = _contains_any(normalized, policy.patterns.policy_guidance_markers)
    if not markers:
        return False
    strong_markers = {"eligible for", "allowed to", "not allowed"}
    if strong_markers & set(markers):
        return True
    return any(topic in normalized for topic in _POLICY_TOPIC_MARKERS)


def _is_account_specific_issue(text: str, policy: AnalysisPolicy) -> bool:
    return bool(_contains_any(text, policy.patterns.account_specific_issue_markers))


def _is_positive_acknowledgement(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized or "?" in normalized:
        return False
    return any(marker in normalized for marker in _POSITIVE_ACKNOWLEDGEMENT_MARKERS)


def _repeated_complaint_evidence(text: str, policy: AnalysisPolicy) -> list[str]:
    normalized = normalize_text(text)
    if _is_positive_acknowledgement(normalized):
        return []
    markers = _contains_any(normalized, policy.patterns.repeated_complaint_markers)
    if not markers:
        return []
    if not any(marker in normalized for marker in _REPEATED_COMPLAINT_CONTEXT_MARKERS):
        return []
    return markers


def _sensitive_request_evidence(text: str, policy: AnalysisPolicy) -> list[str]:
    evidence: list[str] = []
    for sentence in _sentence_chunks(text):
        if any(marker in sentence for marker in _SAFE_SENSITIVE_CONTEXT_MARKERS):
            continue
        for term in policy.patterns.sensitive_data_patterns:
            if term not in sentence:
                continue
            if any(re.search(pattern.format(term=re.escape(term)), sentence) for pattern in _SENSITIVE_REQUEST_PATTERNS):
                evidence.append(term)
    return list(dict.fromkeys(evidence))


def _is_confusing_message(text: str, threshold: int) -> bool:
    word_count = _word_count(text)
    if word_count < threshold:
        return False
    sentence_count = max(len(re.findall(r"[.!?]", text)), 1)
    average_sentence_length = word_count / sentence_count
    structured = "\n-" in text or "\n1." in text or text.count("\n") >= 3
    if structured and average_sentence_length < 30:
        return False
    return average_sentence_length >= 28


def _support_tool_evidence(snapshot: SupportAgentSnapshot) -> tuple[bool, list[dict[str, Any]], list[str]]:
    evidence_records: list[dict[str, Any]] = []
    queries: list[str] = []
    has_positive_evidence = False

    for item in snapshot.last_tool_results:
        record = item.model_dump()
        evidence_records.append(record)
        if item.query:
            queries.append(item.query)
        if item.result_count and item.result_count > 0:
            has_positive_evidence = True

    return has_positive_evidence, evidence_records, queries


def _independent_queries(
    latest_user_message: str,
    latest_support_message: str,
    snapshot: SupportAgentSnapshot,
    prior_queries: list[str],
) -> list[str]:
    issue = snapshot.active_user_issue or latest_user_message
    identifiers = " ".join(snapshot.known_user_identifiers.values())
    queries = [query for query in prior_queries if query]
    seen = {normalize_text(item) for item in queries}
    for candidate in [issue, latest_user_message, f"{identifiers} {issue}".strip(), latest_support_message]:
        normalized = normalize_text(candidate or "")
        if normalized and normalized not in seen:
            queries.append(candidate)
            seen.add(normalized)
    return queries[:3]


def _independent_verification(
    *,
    csv_repository: CsvRepository,
    knowledge_repository: KnowledgeRepository,
    latest_user_message: str,
    latest_support_message: str,
    snapshot: SupportAgentSnapshot,
    prior_queries: list[str],
    data_limit: int,
    knowledge_limit: int,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for query in _independent_queries(latest_user_message, latest_support_message, snapshot, prior_queries):
        data_hits = csv_repository.search(query, limit=data_limit)
        if data_hits:
            evidence.append(
                {
                    "source": "data",
                    "query": query,
                    "result_count": len(data_hits),
                    "results": [{"table": item.table, "row": item.row} for item in data_hits],
                }
            )
        policy_hits = knowledge_repository.search(query, categories=["policies"], limit=knowledge_limit)
        if policy_hits:
            evidence.append(
                {
                    "source": "policies",
                    "query": query,
                    "result_count": len(policy_hits),
                    "results": [{"file_id": item.file_id, "content": item.content} for item in policy_hits],
                }
            )
        reference_hits = knowledge_repository.search(query, categories=["docs", "how_to"], limit=knowledge_limit)
        if reference_hits:
            evidence.append(
                {
                    "source": "reference",
                    "query": query,
                    "result_count": len(reference_hits),
                    "results": [{"file_id": item.file_id, "content": item.content} for item in reference_hits],
                }
            )
        if evidence:
            break
    return evidence


def evaluate_rules(
    *,
    dialogue: list[DialogueMessage],
    snapshot: SupportAgentSnapshot,
    policy: AnalysisPolicy,
    csv_repository: CsvRepository,
    knowledge_repository: KnowledgeRepository,
    data_limit: int,
    knowledge_limit: int,
) -> dict[str, Any]:
    reasons: list[AnalysisReason] = []
    visible_messages = _visible_messages(dialogue)
    support_messages = _assistant_messages(dialogue)
    user_messages = [message for message in visible_messages if message.role == "user"]
    latest_user = _latest_message(dialogue, "user")
    latest_support = _latest_message(dialogue, "assistant")
    latest_user_message = latest_user.content if latest_user else ""
    latest_support_message = latest_support.content if latest_support else ""
    support_reply_count = len(support_messages)
    user_turn_count = len(user_messages)
    clarifying_question_count = sum(1 for message in support_messages if _is_clarifying_question(message.content, policy))
    total_turn_count = len(visible_messages)
    explicit_resolution = _contains_any(latest_user_message, policy.patterns.user_resolution_phrases) or _is_positive_acknowledgement(latest_user_message)
    factual_claim = _has_factual_claim(latest_support_message, policy)
    policy_guidance = _has_policy_guidance(latest_support_message, policy)
    support_has_evidence, support_evidence_records, support_queries = _support_tool_evidence(snapshot)
    verification_evidence = list(support_evidence_records)
    independent_evidence: list[dict[str, Any]] = []

    if explicit_resolution:
        if policy.require_worker_confirmation_on_resolution:
            reasons.append(
                AnalysisReason(
                    code="resolution_requires_worker_confirmation",
                    description="The user confirmed the issue is solved, but worker confirmation is required by policy.",
                    evidence=[latest_user_message],
                )
            )
            status = AnalysisConversationStatus.WAITING_FOR_WORKER
            needs_escalation = True
        else:
            status = AnalysisConversationStatus.RESOLVED
            needs_escalation = False
        return {
            "current_reasons": [reason.model_dump() for reason in reasons],
            "needs_escalation": needs_escalation,
            "conversation_status": status.value,
            "verification_evidence": verification_evidence,
            "last_user_message": latest_user_message,
            "last_support_message": latest_support_message,
        }

    explicit_dissatisfaction = _contains_any(latest_user_message, policy.patterns.explicit_dissatisfaction_phrases)
    repeated_complaints = _repeated_complaint_evidence(latest_user_message, policy)
    internal_details = _contains_any(latest_support_message, policy.patterns.internal_detail_patterns)
    sensitive_request = _sensitive_request_evidence(latest_support_message, policy)
    rude_tone = _contains_any(latest_support_message, policy.patterns.rude_patterns)
    blaming_tone = _contains_any(latest_support_message, policy.patterns.blaming_patterns)
    unsupported_action = _contains_any(latest_support_message, policy.patterns.unsupported_action_patterns)

    if explicit_dissatisfaction:
        reasons.append(
            AnalysisReason(
                code="explicit_user_dissatisfaction",
                description="The user explicitly stated dissatisfaction with the support experience.",
                evidence=[latest_user_message],
            )
        )
    elif repeated_complaints and support_reply_count >= 2:
        reasons.append(
            AnalysisReason(
                code="repeated_user_dissatisfaction",
                description="The user repeated the same complaint after multiple support replies.",
                evidence=[latest_user_message],
            )
        )

    if (
        clarifying_question_count >= policy.thresholds.escalate_after_repeated_clarifying_questions
        and user_turn_count >= 2
        and not support_has_evidence
    ):
        reasons.append(
            AnalysisReason(
                code="too_many_clarifying_questions",
                description="The support agent asked too many clarifying questions without resolving the issue.",
                evidence=[f"clarifying_question_count={clarifying_question_count}"],
            )
        )

    if (
        total_turn_count >= policy.thresholds.escalate_after_total_turns_without_resolution
        and user_turn_count >= 3
        and support_reply_count >= 3
        and not support_has_evidence
        and (clarifying_question_count >= policy.thresholds.watch_after_repeated_clarifying_questions or bool(repeated_complaints) or bool(explicit_dissatisfaction))
    ):
        reasons.append(
            AnalysisReason(
                code="too_many_total_turns",
                description="The conversation exceeded the maximum turn count without resolution.",
                evidence=[f"total_turn_count={total_turn_count}"],
            )
        )

    if internal_details:
        reasons.append(
            AnalysisReason(
                code="unsafe_internal_details",
                description="The support reply exposed internal or system-level details.",
                evidence=internal_details,
            )
        )

    if sensitive_request:
        reasons.append(
            AnalysisReason(
                code="unsafe_sensitive_data_request",
                description="The support reply asked for sensitive data that should not be requested in chat.",
                evidence=sensitive_request,
            )
        )

    if rude_tone:
        reasons.append(
            AnalysisReason(
                code="bad_tone_rude",
                description="The support reply used rude wording.",
                evidence=rude_tone,
            )
        )

    if blaming_tone:
        reasons.append(
            AnalysisReason(
                code="bad_tone_blaming",
                description="The support reply blamed the user.",
                evidence=blaming_tone,
            )
        )

    if _is_confusing_message(latest_support_message, policy.thresholds.confusing_message_word_count):
        reasons.append(
            AnalysisReason(
                code="bad_tone_confusing",
                description="The support reply is too long and likely confusing.",
                evidence=[f"word_count={_word_count(latest_support_message)}"],
            )
        )

    if unsupported_action:
        reasons.append(
            AnalysisReason(
                code="unsupported_action_claim",
                description="The support reply claimed that an action was completed even though the agent is read-only.",
                evidence=unsupported_action,
            )
        )

    if factual_claim and not support_has_evidence:
        reasons.append(
            AnalysisReason(
                code="factual_claim_without_tool_evidence",
                description="The support reply made factual claims without support-tool evidence.",
                evidence=[latest_support_message],
            )
        )

    if factual_claim or policy_guidance:
        if not support_has_evidence:
            independent_evidence = _independent_verification(
                csv_repository=csv_repository,
                knowledge_repository=knowledge_repository,
                latest_user_message=latest_user_message,
                latest_support_message=latest_support_message,
                snapshot=snapshot,
                prior_queries=support_queries,
                data_limit=data_limit,
                knowledge_limit=knowledge_limit,
            )
            verification_evidence.extend(independent_evidence)

    if factual_claim and not support_has_evidence and not independent_evidence:
        reasons.append(
            AnalysisReason(
                code="independent_verification_failed",
                description="Independent verification did not find evidence supporting the factual reply.",
                evidence=[latest_support_message],
            )
        )

    if policy_guidance and not any(item.get("source") == "policies" for item in verification_evidence):
        reasons.append(
            AnalysisReason(
                code="unsupported_policy_guidance",
                description="The support reply gave policy guidance without policy verification.",
                evidence=[latest_support_message],
            )
        )

    account_specific_issue = _is_account_specific_issue(snapshot.active_user_issue or latest_user_message, policy)
    if (
        account_specific_issue
        and not support_has_evidence
        and not _is_clarifying_question(latest_support_message, policy)
        and not factual_claim
        and not unsupported_action
        and not policy_guidance
    ):
        reasons.append(
            AnalysisReason(
                code="missing_clarification_or_lookup",
                description="The support reply handled an account-specific issue without a lookup or a focused clarifying question.",
                evidence=[snapshot.active_user_issue or latest_user_message],
            )
        )

    deduped: list[AnalysisReason] = []
    seen = set()
    for reason in reasons:
        if reason.code not in seen:
            seen.add(reason.code)
            deduped.append(reason)
    reasons = deduped

    if reasons:
        status = AnalysisConversationStatus.WAITING_FOR_WORKER
        needs_escalation = True
    else:
        watch_signals = [
            clarifying_question_count >= policy.thresholds.watch_after_repeated_clarifying_questions and user_turn_count >= 2 and not support_has_evidence,
            total_turn_count >= policy.thresholds.watch_after_total_turns_without_resolution and user_turn_count >= 2 and support_reply_count >= 2,
            bool(repeated_complaints),
        ]
        status = AnalysisConversationStatus.WATCH if any(watch_signals) else AnalysisConversationStatus.NORMAL
        needs_escalation = False

    return {
        "current_reasons": [reason.model_dump() for reason in reasons],
        "needs_escalation": needs_escalation,
        "conversation_status": status.value,
        "watch_recommended": any(watch_signals) if not reasons else False,
        "verification_evidence": verification_evidence,
        "last_user_message": latest_user_message,
        "last_support_message": latest_support_message,
    }


def derive_recommended_tone(reason_codes: list[str], default_tone: Tone) -> Tone:
    code_set = set(reason_codes)
    if {"explicit_user_dissatisfaction", "repeated_user_dissatisfaction", "bad_tone_rude", "bad_tone_blaming"} & code_set:
        return Tone.FRIENDLY_AND_CALM
    if {"too_many_clarifying_questions", "too_many_total_turns"} & code_set:
        return Tone.SHORT_AND_NEUTRAL
    if code_set:
        return Tone.FORMAL
    return default_tone


def derive_constraints(reason_codes: list[str]) -> list[str]:
    constraints: list[str] = []
    code_set = set(reason_codes)

    if {"explicit_user_dissatisfaction", "repeated_user_dissatisfaction"} & code_set:
        constraints.append("Acknowledge the frustration once and focus on the next concrete step.")
    if {"too_many_clarifying_questions", "too_many_total_turns"} & code_set:
        constraints.append("Do not repeat earlier guidance; keep the reply concise and move the case forward.")
    if {"factual_claim_without_tool_evidence", "independent_verification_failed"} & code_set:
        constraints.append("Verify the factual answer against data before responding.")
    if {"unsupported_policy_guidance"} & code_set:
        constraints.append("Confirm policy guidance before telling the user what is allowed.")
    if {"unsafe_internal_details"} & code_set:
        constraints.append("Do not mention internal systems, files, or agent workflow.")
    if {"unsafe_sensitive_data_request"} & code_set:
        constraints.append("Do not ask for passwords, card details, CVV, SSN, or other sensitive credentials.")
    if {"unsupported_action_claim"} & code_set:
        constraints.append("Do not claim that actions were completed when no action was actually performed.")
    if {"missing_clarification_or_lookup"} & code_set:
        constraints.append("Ask one focused clarifying question if a lookup still cannot be performed safely.")
    if not constraints:
        constraints.append("Keep the reply simple, understandable, and focused on resolution.")
    return constraints


def derive_next_steps(reason_codes: list[str], needs_escalation: bool) -> list[str]:
    steps: list[str] = []
    code_set = set(reason_codes)

    if not needs_escalation:
        return ["Continue monitoring the conversation on the next user turn."]

    if {"explicit_user_dissatisfaction", "repeated_user_dissatisfaction"} & code_set:
        steps.append("Send a revised reply that acknowledges the issue and corrects the support path.")
    if {"too_many_clarifying_questions", "too_many_total_turns"} & code_set:
        steps.append("Avoid another repetitive loop; either resolve the issue or ask one final focused question.")
    if {"factual_claim_without_tool_evidence", "independent_verification_failed"} & code_set:
        steps.append("Re-check the relevant data before sending another factual answer.")
    if {"unsupported_policy_guidance"} & code_set:
        steps.append("Verify the applicable policy before giving further instructions.")
    if {"unsafe_internal_details", "unsafe_sensitive_data_request", "unsupported_action_claim"} & code_set:
        steps.append("Replace the unsafe wording with a compliant customer-facing reply.")
    if {"missing_clarification_or_lookup"} & code_set:
        steps.append("Refocus the next reply on the user's exact issue and collect only the minimum missing detail.")
    if {"resolution_requires_worker_confirmation"} & code_set:
        steps.append("Review the resolved conversation and confirm whether it can be closed.")

    return list(dict.fromkeys(steps)) or ["Review the conversation and provide new instructions to the support agent."]


def build_dialogue_summary(
    *,
    snapshot: SupportAgentSnapshot,
    latest_user_message: str,
    latest_support_message: str,
    status: AnalysisConversationStatus,
    reasons: list[AnalysisReason],
) -> str:
    active_issue = snapshot.active_user_issue or latest_user_message or "Unknown issue"
    parts = [f"Active issue: {active_issue}."]
    if latest_user_message:
        parts.append(f"Latest user message: {latest_user_message}")
    if latest_support_message:
        parts.append(f"Latest support reply: {latest_support_message}")
    if reasons:
        parts.append(
            "Escalation reasons: "
            + "; ".join(f"{reason.code} ({reason.description})" for reason in reasons)
            + "."
        )
    else:
        parts.append(f"No escalation trigger matched. Current status: {status.value}.")
    return " ".join(parts)


def build_suggested_instruction(
    *,
    policy: AnalysisPolicy,
    recommended_tone: Tone,
    suggested_constraints: list[str],
    possible_next_steps: list[str],
    reason_codes: list[str],
) -> WorkerInstruction:
    code_set = set(reason_codes)

    if "resolution_requires_worker_confirmation" in code_set:
        task = "Review the resolved dialogue and decide whether the case can be closed or needs one final reply."
    elif {"factual_claim_without_tool_evidence", "independent_verification_failed", "unsupported_policy_guidance"} & code_set:
        task = "Have the support agent send a revised reply based only on verified data and policy."
    elif {"explicit_user_dissatisfaction", "repeated_user_dissatisfaction", "bad_tone_rude", "bad_tone_blaming"} & code_set:
        task = "Have the support agent send a calmer revised reply that corrects the tone and moves the issue forward."
    elif {"too_many_clarifying_questions", "too_many_total_turns"} & code_set:
        task = "Have the support agent send one concise reply that avoids repetition and moves toward resolution."
    else:
        task = "Have the support agent send a corrected reply."

    return WorkerInstruction(
        task=task,
        constraints=suggested_constraints,
        answer_format=policy.default_instruction.answer_format,
        tone=recommended_tone,
        review_before_send=policy.default_instruction.review_before_send,
        regenerate=policy.default_instruction.regenerate,
        notes=possible_next_steps,
    )


def build_worker_package(
    *,
    dialogue: list[DialogueMessage],
    dialogue_summary: str,
    reasons: list[AnalysisReason],
    possible_next_steps: list[str],
    recommended_tone: Tone,
    suggested_constraints: list[str],
    suggested_instruction: WorkerInstruction,
) -> dict[str, Any]:
    return {
        "full_dialogue": [message.model_dump() for message in dialogue],
        "dialogue_summary": dialogue_summary,
        "reason_of_escalation": [reason.model_dump() for reason in reasons],
        "possible_next_steps": possible_next_steps,
        "recommended_tone": recommended_tone.value,
        "suggested_constraints": suggested_constraints,
        "suggested_worker_instruction": suggested_instruction.model_dump(),
    }
def build_reason_counter(reasons: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(reason["code"] for reason in reasons))
