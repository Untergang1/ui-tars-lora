#!/usr/bin/env python3
"""Pin the UI-TARS grounding contract from the vendored Agent-S revision."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_S_DIR = PROJECT_ROOT / "third_party" / "Agent-S"
SOURCE = AGENT_S_DIR / "gui_agents" / "s3" / "agents" / "grounding.py"
OUTPUT = PROJECT_ROOT / "configs" / "grounding_contract.json"
EXPECTED_PROMPT = 'prompt = f"Query:{ref_expr}\\nOutput only the coordinate of one point in your response.\\n"'


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"missing Agent-S grounding source: {SOURCE}")
    source_text = SOURCE.read_text(encoding="utf-8")
    if EXPECTED_PROMPT not in source_text:
        raise SystemExit("Agent-S grounding prompt changed; review before creating training labels.")
    if 'return [int(numericals[0]), int(numericals[1])]' not in source_text:
        raise SystemExit("Agent-S coordinate parser changed; review before creating training labels.")
    commit = subprocess.check_output(
        ["git", "-C", str(AGENT_S_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
    contract = {
        "source": "Agent-S gui_agents/s3/agents/grounding.py",
        "source_commit": commit,
        "prompt_template": "Query:{description}\nOutput only the coordinate of one point in your response.\n",
        "coordinate_space": {"width": 1920, "height": 1080},
        "assistant_response_template": "({x}, {y})",
        "parser": "first two decimal integer substrings in the model response",
    }
    OUTPUT.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote pinned grounding contract to {OUTPUT}")


if __name__ == "__main__":
    main()
