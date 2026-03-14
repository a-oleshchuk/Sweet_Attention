from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Tuple

from langgraph.checkpoint.base import BaseCheckpointSaver

from agents.support_agent.repositories import CsvRepository

from .contracts import AnalysisResult


def _extract_state(checkpoint: Any) -> dict[str, Any]:
    if checkpoint is None:
        return {}
    if isinstance(checkpoint, dict):
        if "state" in checkpoint:
            return checkpoint.get("state", {}) or {}
        inner = checkpoint.get("checkpoint") or {}
        if isinstance(inner, dict):
            if "state" in inner:
                return inner.get("state", {}) or {}
            return inner
    candidate = getattr(checkpoint, "state", None)
    if candidate:
        return candidate
    inner = getattr(checkpoint, "checkpoint", None)
    if inner is None:
        return {}
    if isinstance(inner, dict):
        if "state" in inner:
            return inner.get("state", {}) or {}
        return inner
    state_attr = getattr(inner, "state", None)
    return state_attr or {}


class SupportConversationRepository:
    def __init__(self, *, checkpointer: BaseCheckpointSaver | None, max_messages: int = 50):
        self.checkpointer = checkpointer
        self.max_messages = max_messages
        self._overrides: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}

    def set_override(
        self,
        conversation_id: str,
        messages: list[dict[str, Any]],
        profile: dict[str, Any],
    ) -> None:
        self._overrides[conversation_id] = (list(messages), dict(profile))

    def clear_override(self, conversation_id: str) -> None:
        self._overrides.pop(conversation_id, None)

    def load(self, conversation_id: str) -> Tuple[list[dict[str, Any]], dict[str, Any]]:
        if conversation_id in self._overrides:
            messages, profile = self._overrides[conversation_id]
            return messages[-self.max_messages :], dict(profile)
        # If checkpoint API isn't available, fall back to empty history to avoid hard failures.
        if self.checkpointer is None or not hasattr(self.checkpointer, "get_latest_checkpoint"):
            return [], {}

        checkpoint = self.checkpointer.get_latest_checkpoint({"configurable": {"thread_id": conversation_id}})
        state = _extract_state(checkpoint)
        history = list(state.get("message_history", []))
        profile = {}
        if state.get("user_display_name"):
            profile["display_name"] = state.get("user_display_name")
        for key, value in dict(state.get("known_user_identifiers", {})).items():
            if value:
                profile[key] = value
        return history[-self.max_messages :], profile


class UserProfileRepository:
    def __init__(self, csv_repository: CsvRepository):
        self.csv_repository = csv_repository

    def find_profile(self, display_name: str | None, known_identifiers: dict[str, str] | None = None) -> dict[str, Any]:
        identifiers = known_identifiers or {}
        query_parts = [display_name] if display_name else []
        query_parts.extend(identifiers.values())
        query = " ".join(query_parts).strip() or (display_name or "")
        results = self.csv_repository.search(query, tables=["core.guardians"], limit=1)
        if not results:
            return {key: value for key, value in identifiers.items() if value}

        row = results[0].row
        profile = {
            "user_id": row.get("guardian_id"),
            "user_name": row.get("full_name"),
            "account_id": row.get("account_id"),
            "subscription_level": row.get("account_status"),
            "email": row.get("email"),
            "phone": row.get("phone"),
            "customer_tier": row.get("payment_risk_flag"),
            "preferred_language": row.get("preferred_language"),
        }
        for key, value in identifiers.items():
            profile.setdefault(key, value)
        return {key: value for key, value in profile.items() if value}


class AnalysisResultRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure()

    def _ensure(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_results (
                conversation_id TEXT PRIMARY KEY,
                domain_key TEXT,
                issue_type TEXT,
                priority TEXT,
                summary TEXT,
                user_profile TEXT,
                detected_issues TEXT,
                detection_signals TEXT,
                support_agent_findings TEXT,
                next_steps TEXT,
                analysis_metadata TEXT,
                conversation_history TEXT,
                updated_at TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def upsert(self, result: AnalysisResult) -> None:
        payload = result.to_record()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO analysis_results (
                conversation_id, domain_key, issue_type, priority, summary, user_profile,
                detected_issues, detection_signals, support_agent_findings, next_steps,
                analysis_metadata, conversation_history, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                domain_key=excluded.domain_key,
                issue_type=excluded.issue_type,
                priority=excluded.priority,
                summary=excluded.summary,
                user_profile=excluded.user_profile,
                detected_issues=excluded.detected_issues,
                detection_signals=excluded.detection_signals,
                support_agent_findings=excluded.support_agent_findings,
                next_steps=excluded.next_steps,
                analysis_metadata=excluded.analysis_metadata,
                conversation_history=excluded.conversation_history,
                updated_at=excluded.updated_at
            """,
            (
                payload["conversation_id"],
                payload["domain_key"],
                payload.get("issue_type"),
                payload.get("priority"),
                payload.get("summary"),
                json.dumps(payload.get("user_profile", {})),
                json.dumps(payload.get("detected_issues", [])),
                json.dumps(payload.get("detection_signals", {})),
                json.dumps(payload.get("support_agent_findings", [])),
                json.dumps(payload.get("next_steps", [])),
                json.dumps(payload.get("analysis_metadata", {})),
                json.dumps(payload.get("conversation_history", [])),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    def get(self, conversation_id: str) -> AnalysisResult | None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """
            SELECT conversation_id, domain_key, issue_type, priority, summary, user_profile,
                   detected_issues, detection_signals, support_agent_findings, next_steps,
                   analysis_metadata, conversation_history
            FROM analysis_results WHERE conversation_id = ?
            """,
            (conversation_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return self._row_to_result(row)

    def list_all(self) -> list[AnalysisResult]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """
            SELECT conversation_id, domain_key, issue_type, priority, summary, user_profile,
                   detected_issues, detection_signals, support_agent_findings, next_steps,
                   analysis_metadata, conversation_history
            FROM analysis_results
            ORDER BY
                CASE priority WHEN 'urgent' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                updated_at DESC
            """
        )
        rows = cursor.fetchall()
        conn.close()
        results = [self._row_to_result(row) for row in rows]
        return [result for result in results if result.conversation_history]

    def _row_to_result(self, row: tuple[Any, ...]) -> AnalysisResult:
        (
            conversation_id,
            domain_key,
            issue_type,
            priority,
            summary,
            user_profile,
            detected_issues,
            detection_signals,
            support_agent_findings,
            next_steps,
            analysis_metadata,
            conversation_history,
        ) = row
        record = {
            "conversation_id": conversation_id,
            "domain_key": domain_key,
            "issue_type": issue_type,
            "priority": priority,
            "summary": summary,
            "user_profile": json.loads(user_profile or "{}"),
            "detected_issues": json.loads(detected_issues or "[]"),
            "detection_signals": json.loads(detection_signals or "{}"),
            "support_agent_findings": json.loads(support_agent_findings or "[]"),
            "next_steps": json.loads(next_steps or "[]"),
            "analysis_metadata": json.loads(analysis_metadata or "{}"),
            "conversation_history": json.loads(conversation_history or "[]"),
        }
        return AnalysisResult.from_record(record)
