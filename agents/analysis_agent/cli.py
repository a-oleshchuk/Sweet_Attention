from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import AnalysisTurnInput, DialogueMessage, SupportAgentSnapshot, WorkerDecision
from .service import AnalysisAgentService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one analysis-agent turn.")
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--dialogue-json", required=True, help="Path to a JSON file containing a list of dialogue messages.")
    parser.add_argument("--support-state-json", required=True, help="Path to a JSON file containing support-agent state.")
    parser.add_argument("--worker-decision-json", help="Optional path to a JSON file containing a worker decision update.")
    parser.add_argument("--domain-key", default="ai_school")
    return parser


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dialogue = [DialogueMessage.model_validate(item) for item in _load_json(args.dialogue_json)]
    support_state = SupportAgentSnapshot.model_validate(_load_json(args.support_state_json))
    worker_decision = None
    if args.worker_decision_json:
        worker_decision = WorkerDecision.model_validate(_load_json(args.worker_decision_json))

    turn = AnalysisTurnInput(
        conversation_id=args.conversation_id,
        full_dialogue_snapshot=dialogue,
        support_agent_state=support_state,
        worker_decision_update=worker_decision,
        domain_key=args.domain_key,
    )

    with AnalysisAgentService.from_env() as service:
        result = service.invoke(turn)

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

