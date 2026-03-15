from __future__ import annotations

import re
from collections import Counter
from typing import Any

from agents.shared import AnalysisPolicy, CsvRepository, KnowledgeRepository
from agents.shared.repositories import normalize_text, tokenize_text
from agents.support_agent.contracts import Tone, WorkerInstruction

from .contracts import AnalysisConversationStatus, AnalysisReason, DialogueMessage, SupportAgentSnapshot


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


def _is_clarifying_question(text: str, policy: AnalysisPolicy) -> bool:
    normalized = normalize_text(text)
    return "?" in normalized and bool(_contains_any(normalized, policy.patterns.clarifying_question_markers))


def _has_factual_claim(text: str, policy: AnalysisPolicy) -> bool:
    normalized = normalize_text(text)
    if _contains_any(normalized, policy.patterns.factual_claim_markers):
        return True
    if re.search(r"\b(g\d{4}|s\d{4}|sub\d{4}|inv\d{4})\b", normalized):
        return True
    return bool(re.search(r"\b\d{4}-\d{2}-\d{2}\b", normalized))


def _has_policy_guidance(text: str, policy: AnalysisPolicy) -> bool:
    return bool(_contains_any(text, policy.patterns.policy_guidance_markers))


def _is_account_specific_issue(text: str, policy: AnalysisPolicy) -> bool:
    return bool(_contains_any(text, policy.patterns.account_specific_issue_markers))


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


def _latest_support_mentions_keywords(latest_user_message: str, latest_support_message: str) -> bool:
    user_tokens = [token for token in tokenize_text(latest_user_message) if len(token) > 3]
    if not user_tokens:
        return True
    support_text = normalize_text(latest_support_message)
    return any(token in support_text for token in user_tokens[:5])


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
    latest_user = _latest_message(dialogue, "user")
    latest_support = _latest_message(dialogue, "assistant")
    latest_user_message = latest_user.content if latest_user else ""
    latest_support_message = latest_support.content if latest_support else ""
    support_reply_count = len(support_messages)
    clarifying_question_count = sum(1 for message in support_messages if _is_clarifying_question(message.content, policy))
    total_turn_count = len(visible_messages)
    explicit_resolution = _contains_any(latest_user_message, policy.patterns.user_resolution_phrases)
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
    repeated_complaints = _contains_any(latest_user_message, policy.patterns.repeated_complaint_markers)
    internal_details = _contains_any(latest_support_message, policy.patterns.internal_detail_patterns)
    sensitive_request = _contains_any(latest_support_message, policy.patterns.sensitive_data_patterns)
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

    if support_reply_count >= policy.thresholds.escalate_after_support_replies_on_same_issue:
        reasons.append(
            AnalysisReason(
                code="too_many_support_replies",
                description="The same issue has already received too many support replies.",
                evidence=[f"support_reply_count={support_reply_count}"],
            )
        )

    if clarifying_question_count >= policy.thresholds.escalate_after_repeated_clarifying_questions:
        reasons.append(
            AnalysisReason(
                code="too_many_clarifying_questions",
                description="The support agent asked too many clarifying questions without resolving the issue.",
                evidence=[f"clarifying_question_count={clarifying_question_count}"],
            )
        )

    if total_turn_count >= policy.thresholds.escalate_after_total_turns_without_resolution:
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

    if _word_count(latest_support_message) >= policy.thresholds.confusing_message_word_count:
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
    if account_specific_issue and not support_has_evidence and not _is_clarifying_question(latest_support_message, policy):
        reasons.append(
            AnalysisReason(
                code="missing_clarification_or_lookup",
                description="The support reply handled an account-specific issue without a lookup or a focused clarifying question.",
                evidence=[snapshot.active_user_issue or latest_user_message],
            )
        )

    if latest_user_message and latest_support_message and not _is_clarifying_question(latest_support_message, policy):
        if not _latest_support_mentions_keywords(latest_user_message, latest_support_message):
            reasons.append(
                AnalysisReason(
                    code="reply_did_not_address_user_issue",
                    description="The support reply does not appear to address the user's stated issue directly.",
                    evidence=[latest_user_message, latest_support_message],
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
            support_reply_count >= policy.thresholds.watch_after_support_replies_on_same_issue,
            clarifying_question_count >= policy.thresholds.watch_after_repeated_clarifying_questions,
            total_turn_count >= policy.thresholds.watch_after_total_turns_without_resolution,
            bool(repeated_complaints),
        ]
        status = AnalysisConversationStatus.WATCH if any(watch_signals) else AnalysisConversationStatus.NORMAL
        needs_escalation = False

    return {
        "current_reasons": [reason.model_dump() for reason in reasons],
        "needs_escalation": needs_escalation,
        "conversation_status": status.value,
        "verification_evidence": verification_evidence,
        "last_user_message": latest_user_message,
        "last_support_message": latest_support_message,
    }


def derive_recommended_tone(reason_codes: list[str], default_tone: Tone) -> Tone:
    code_set = set(reason_codes)
    if {"explicit_user_dissatisfaction", "repeated_user_dissatisfaction", "bad_tone_rude", "bad_tone_blaming"} & code_set:
        return Tone.FRIENDLY_AND_CALM
    if {"too_many_support_replies", "too_many_clarifying_questions", "too_many_total_turns"} & code_set:
        return Tone.SHORT_AND_NEUTRAL
    if code_set:
        return Tone.FORMAL
    return default_tone


def derive_constraints(reason_codes: list[str]) -> list[str]:
    constraints: list[str] = []
    code_set = set(reason_codes)

    if {"explicit_user_dissatisfaction", "repeated_user_dissatisfaction"} & code_set:
        constraints.append("Acknowledge the frustration once and focus on the next concrete step.")
    if {"too_many_support_replies", "too_many_clarifying_questions", "too_many_total_turns"} & code_set:
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
    if {"reply_did_not_address_user_issue"} & code_set:
        constraints.append("Address the user's actual request directly in the next reply.")

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
    if {"too_many_support_replies", "too_many_clarifying_questions", "too_many_total_turns"} & code_set:
        steps.append("Avoid another repetitive loop; either resolve the issue or ask one final focused question.")
    if {"factual_claim_without_tool_evidence", "independent_verification_failed"} & code_set:
        steps.append("Re-check the relevant data before sending another factual answer.")
    if {"unsupported_policy_guidance"} & code_set:
        steps.append("Verify the applicable policy before giving further instructions.")
    if {"unsafe_internal_details", "unsafe_sensitive_data_request", "unsupported_action_claim"} & code_set:
        steps.append("Replace the unsafe wording with a compliant customer-facing reply.")
    if {"missing_clarification_or_lookup", "reply_did_not_address_user_issue"} & code_set:
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
    elif {"too_many_support_replies", "too_many_clarifying_questions", "too_many_total_turns"} & code_set:
        task = "Have the support agent send one concise reply that avoids repetition and moves toward resolution."
    else:
        task = "Have the support agent send a corrected reply that addresses the user's issue directly."

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


def summarize_codes(reasons: list[dict[str, Any]]) -> list[str]:
    return [str(reason["code"]) for reason in reasons]


def build_reason_counter(reasons: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(reason["code"] for reason in reasons))
