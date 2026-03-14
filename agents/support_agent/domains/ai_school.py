from __future__ import annotations

from pathlib import Path

from . import DomainConfig


AI_SCHOOL_CONTEXT = """You support AstraLearn, an online AI school for parents and guardians.
Common topics include billing, invoices, receipts, subscriptions, lessons, attendance,
portal access, password resets, and course enrollment.

The chat metadata includes the user's display name. Use that information when it helps
you look up the correct mock records.

Knowledge sources are split into:
- reference docs
- how-to guides
- policies

Operational records are split into:
- core account data
- billing data
- learning data
- support history
"""


def build_ai_school_config(project_root: Path) -> DomainConfig:
    return DomainConfig(
        key="ai_school",
        display_name="AstraLearn AI School",
        data_root=project_root / "ai_school" / "data",
        knowledge_root=project_root / "ai_school" / "knowledge",
        system_context=AI_SCHOOL_CONTEXT,
    )
