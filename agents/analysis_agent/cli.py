from __future__ import annotations

import argparse

from .contracts import AnalysisTurnInput
from .service import AnalysisAgentService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one analysis-agent turn.")
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--domain-key", default="ai_school")
    parser.add_argument("--refresh", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    turn = AnalysisTurnInput(
        conversation_id=args.conversation_id,
        domain_key=args.domain_key,
        refresh=args.refresh,
    )

    with AnalysisAgentService.from_env() as service:
        result = service.invoke(turn)

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
