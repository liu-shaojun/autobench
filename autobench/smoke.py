from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field

from . import container
from .config import ModelConfig
from .logutil import ModelLogger

DEFAULT_PROMPTS = [
    {"mode": "chat", "prompt": "Hi", "label": "greeting"},
    {"mode": "chat", "prompt": "What is 2+2?", "label": "math_basic"},
    {
        "mode": "chat",
        "prompt": "Natalia sold clips to 48 friends in April and half as many in May. How many clips altogether?",
        "label": "math_word",
    },
    {
        "mode": "chat",
        "prompt": "Write a Python function that checks if any two numbers in a list are within a given threshold.",
        "label": "code",
    },
    {"mode": "completion", "prompt": "The capital of France is", "label": "knowledge"},
]


@dataclass
class SmokeResult:
    results: list[dict] = field(default_factory=list)
    ok: bool = True
    error: str | None = None


def run(
    model: ModelConfig,
    cname: str,
    logger: ModelLogger,
    prompts: list[dict] | None = None,
    *,
    dry_run: bool = False,
) -> SmokeResult:
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    logger.section("SMOKE_TEST")
    result = SmokeResult()

    for p in prompts:
        mode = p.get("mode", "chat")
        prompt = p["prompt"]
        label = p.get("label", prompt[:30])

        if mode == "chat":
            body = json.dumps({
                "model": model.name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0,
            })
            endpoint = f"http://localhost:{model.port}/v1/chat/completions"
            extract = "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:300])"
        else:
            body = json.dumps({
                "model": model.name,
                "prompt": prompt,
                "max_tokens": 30,
                "temperature": 0,
            })
            endpoint = f"http://localhost:{model.port}/v1/completions"
            extract = "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['text'][:200])"

        cmd = (
            f"curl -s --max-time 30 -X POST {shlex.quote(endpoint)} "
            f"-H 'Content-Type: application/json' "
            f"-d {shlex.quote(body)} "
            f"| python3 -c {shlex.quote(extract)}"
        )

        logger.write(f"  [{mode}] {label}: {prompt}")

        if dry_run:
            result.results.append({"label": label, "prompt": prompt, "response": "", "ok": True})
            continue

        r = container.exec_sync(cname, cmd, timeout=60)
        response = (r.stdout or "").strip()
        ok = r.returncode == 0 and len(response) > 0

        logger.write(f"    -> {response[:200]}")
        if not ok:
            logger.write(f"    [WARN] empty or failed response")

        result.results.append({
            "label": label,
            "prompt": prompt,
            "response": response[:300],
            "ok": ok,
        })

    result.ok = all(r["ok"] for r in result.results)
    if not result.ok:
        failed = [r["label"] for r in result.results if not r["ok"]]
        result.error = f"failed: {', '.join(failed)}"
    return result
