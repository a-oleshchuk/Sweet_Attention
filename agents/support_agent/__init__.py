"""Support-agent package."""

from .contracts import SupportTurnInput, SupportTurnOutput, Tone, UserMetadata, WorkerInstruction

__all__ = [
    "SupportAgentService",
    "SupportTurnInput",
    "SupportTurnOutput",
    "Tone",
    "UserMetadata",
    "WorkerInstruction",
]


def __getattr__(name: str):
    if name == "SupportAgentService":
        from .service import SupportAgentService

        return SupportAgentService
    raise AttributeError(name)
