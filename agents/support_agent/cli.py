from __future__ import annotations

import argparse
import json

from .contracts import SupportTurnInput, UserMetadata, WorkerInstruction
from .service import SupportAgentService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one support-agent turn.")
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--user-name", required=True)
    parser.add_argument("--message")
    parser.add_argument("--domain-key", default="ai_school")
    parser.add_argument(
        "--worker-instruction-json",
        help="JSON object matching the WorkerInstruction schema.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    worker_instruction = None
    if args.worker_instruction_json:
        worker_instruction = WorkerInstruction.model_validate(json.loads(args.worker_instruction_json))

    turn = SupportTurnInput(
        conversation_id=args.conversation_id,
        domain_key=args.domain_key,
        user_message=args.message,
        worker_instruction=worker_instruction,
        user_metadata=UserMetadata(display_name=args.user_name),
    )

    with SupportAgentService.from_env() as service:
        result = service.invoke(turn)

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
