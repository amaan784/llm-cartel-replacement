"""Offline OpenAI-compatible mock for testing the pipeline without an API key or spend.

    python src/dev/mock_llm.py --port 8765
    # then route the harness to it with model ids "custom:<real id>" and LLM_BASE_URL=http://127.0.0.1:8765/v1

The mock reads the last round of the market data in the prompt and answers "My chosen price: x".
Behaviour depends only on the requested model id and the system prompt, so pipeline logic (crossing,
branching, retries, analysis) can be exercised end to end. It produces no research data.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COLLUDERS = ("qwen", "gpt-4.1-mini", "openai/gpt-4.1")
NASH, MONO = 1.473, 1.925


NO_COLLUDE_P0 = ()  # --no_collude_p0 qwen: these models only collude when nudged (PI prompt)


def decide(model: str, system: str, user: str, rng: random.Random) -> float:
    rounds = re.findall(r"My price: ([\d.]+)\n\s+- Competitor's price: ([\d.]+)", user)
    anti = "avoid both intentional and tacit collusion" in system
    nudged = "cooperate with your competitor" in system
    colluder = any(c in model for c in COLLUDERS) and "gemini" not in model and not anti
    if any(c in model for c in NO_COLLUDE_P0) and not nudged:
        colluder = False
    if not rounds:
        return round(rng.uniform(1.5, 1.62), 2)
    own, comp = (float(x) for x in rounds[-1])
    if colluder:
        target = min(max(own, comp) + 0.03, 1.9) if len(rounds) > 3 else own + 0.01
    else:
        target = max(min(own, comp) - 0.03, NASH)
    return round(target + rng.gauss(0, 0.01), 2)


class Handler(BaseHTTPRequestHandler):
    latency = 0.0
    fail_rate = 0.0
    garbage_rate = 0.0

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        self._send(200, {"data": []})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        rng = random.Random()
        if rng.random() < self.fail_rate:
            return self._send(500, {"error": {"message": "mock transient error"}})
        msgs = body.get("messages", [])
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        model = body.get("model", "")
        lat = self.latency * (3 if "qwen" in model else 1)
        time.sleep(lat * rng.uniform(0.5, 1.5))
        price = decide(model, system, user, rng)
        if rng.random() < self.garbage_rate:
            content = "I cannot decide."
        elif "qwen" in model:
            content = f"<think>Competitor history noted; mock reasoning for {model}.</think>\nMy chosen price: {price}"
        else:
            content = f"Mock reasoning for {model}.\nMy chosen price: {price}"
        ptoks = len(system + user) // 4
        self._send(200, {
            "id": "mock-1", "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": ptoks, "completion_tokens": 40, "total_tokens": ptoks + 40},
        })

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--latency", type=float, default=0.02)
    ap.add_argument("--fail_rate", type=float, default=0.02)
    ap.add_argument("--garbage_rate", type=float, default=0.01)
    ap.add_argument("--no_collude_p0", nargs="*", default=[])
    a = ap.parse_args()
    global NO_COLLUDE_P0
    NO_COLLUDE_P0 = tuple(a.no_collude_p0)
    Handler.latency, Handler.fail_rate, Handler.garbage_rate = a.latency, a.fail_rate, a.garbage_rate
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
