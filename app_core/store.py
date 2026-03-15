from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re
from uuid import uuid4

from pydantic import BaseModel, Field

from agents.analysis_agent.contracts import DialogueMessage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_STATE_ROOT = PROJECT_ROOT / ".app_state" / "conversations"


class ConversationRecord(BaseModel):
    conversation_id: str
    domain_key: str = "ai_school"
    user_display_name: str
    created_at: str
    updated_at: str
    messages: list[DialogueMessage] = Field(default_factory=list)
    support_state: dict = Field(default_factory=dict)
    analysis_result: dict = Field(default_factory=dict)
    worker_package: dict | None = None
    pending_review: bool = False
    pending_draft: str | None = None
    resolved_labeling: dict = Field(default_factory=dict)
    worker_instruction_history: list[dict] = Field(default_factory=list)
    worker_decisions: list[dict] = Field(default_factory=list)

    @property
    def status(self) -> str:
        return str(self.analysis_result.get("conversation_status", "normal"))

    @property
    def last_summary(self) -> str:
        return str(self.analysis_result.get("dialogue_summary", ""))

    @property
    def dialogue_summary_text(self) -> str:
        summary = self.last_summary
        match = re.search(r"^Summary:\s*(.+)$", summary, flags=re.MULTILINE)
        return match.group(1).strip() if match else summary

    @property
    def dialogue_category(self) -> str:
        return str(self.analysis_result.get("dialogue_category", ""))

    @property
    def dialogue_intent(self) -> str:
        return str(self.analysis_result.get("dialogue_intent", ""))

    @property
    def resolved_label_category(self) -> str:
        return str(self.resolved_labeling.get("category", ""))

    @property
    def resolved_label_intent(self) -> str:
        return str(self.resolved_labeling.get("intent", ""))

    @property
    def resolved_label_summary(self) -> str:
        return str(self.resolved_labeling.get("summary", ""))

    @property
    def resolved_label_status(self) -> str:
        return str(self.resolved_labeling.get("status", ""))


class ConversationStore:
    def __init__(self, root: Path = APP_STATE_ROOT):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, conversation_id: str) -> Path:
        return self.root / f"{conversation_id}.json"

    def create(self, *, user_display_name: str, domain_key: str = "ai_school") -> ConversationRecord:
        now = datetime.now(UTC).isoformat()
        record = ConversationRecord(
            conversation_id=f"conv-{uuid4().hex[:8]}",
            domain_key=domain_key,
            user_display_name=user_display_name,
            created_at=now,
            updated_at=now,
        )
        self.save(record)
        return record

    def save(self, record: ConversationRecord) -> ConversationRecord:
        record.updated_at = datetime.now(UTC).isoformat()
        self._path(record.conversation_id).write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return record

    def get(self, conversation_id: str) -> ConversationRecord:
        path = self._path(conversation_id)
        if not path.exists():
            raise FileNotFoundError(f"Conversation not found: {conversation_id}")
        return ConversationRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[ConversationRecord]:
        records: list[ConversationRecord] = []
        for path in sorted(self.root.glob("*.json")):
            records.append(ConversationRecord.model_validate_json(path.read_text(encoding="utf-8")))
        records.sort(key=lambda item: item.updated_at, reverse=True)
        return records
