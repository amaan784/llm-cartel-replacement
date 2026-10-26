"""Run the organizer's branch_game.py with a larger output-token cap for the agents.

    python src/branch_long.py --max_tokens 8192 <branch_game.py arguments...>

branch_game.py builds every agent with ChatProvider's default max_tokens (4096) and has no option to change it.
qwen3-14b reasons for a median of ~2,950 and up to ~4,060 output tokens, so at 4096 many answers are cut off
before the price line and re-sampled (and 9 cut-offs in a row abort a branch). This wrapper only raises that
default, then runs branch_game.main() unchanged. Run it with the harness environment from the llm-cartel folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

CARTEL = Path(__file__).resolve().parents[1] / "vendor" / "collusion_hackathon" / "llm-cartel"


def main() -> None:
    argv = sys.argv[1:]
    max_tokens = 4096
    if "--max_tokens" in argv:
        i = argv.index("--max_tokens")
        max_tokens = int(argv[i + 1])
        del argv[i : i + 2]
    sys.path.insert(0, str(CARTEL))
    from harness import providers

    d = list(providers.ChatProvider.__init__.__defaults__)  # (temperature, max_tokens, reasoning)
    d[1] = max_tokens
    providers.ChatProvider.__init__.__defaults__ = tuple(d)
    print(f"agent max_tokens = {max_tokens} (branch_game default raised by src/branch_long.py)", flush=True)
    import branch_game
    from run_game import AgentCallError
    from harness.providers import FatalProviderError

    sys.argv = [str(CARTEL / "branch_game.py")] + argv
    try:
        branch_game.main()
    except (FatalProviderError, AgentCallError) as e:
        sys.exit(f"\n❌ {e}")


if __name__ == "__main__":
    main()
