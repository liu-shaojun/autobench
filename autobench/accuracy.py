from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass

from . import container
from .config import ModelConfig
from .logutil import ModelLogger


GSM8K_URL = "http://xin-dev.sh.intel.com/share/gsm8k_eval.py"
GSM8K_SCRIPT = "/tmp/gsm8k_eval.py"
DEFAULT_NUM_QUESTIONS = 100


@dataclass
class AccuracyResult:
    accuracy: float | None
    num_questions: int
    raw_output: str
    ok: bool
    error: str | None = None


_ACC_PATTERNS = [
    re.compile(r"accuracy[^\d]*([0-9]+\.?[0-9]*)\s*%", re.IGNORECASE),
    re.compile(r"accuracy[^\d]*([0-1]\.[0-9]+)", re.IGNORECASE),
    re.compile(r"acc[^\d]*([0-9]+\.?[0-9]*)\s*%", re.IGNORECASE),
]


def parse_accuracy(text: str) -> float | None:
    for pat in _ACC_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                v = float(m.group(1))
                # Normalize: if looks like a percent (>1) keep as percent, else convert
                return v if v > 1.0 else v * 100.0
            except ValueError:
                continue
    return None


def run(
    model: ModelConfig,
    cname: str,
    logger: ModelLogger,
    *,
    num_questions: int = DEFAULT_NUM_QUESTIONS,
    timeout_sec: int = 1800,
    dry_run: bool = False,
) -> AccuracyResult:
    logger.section(f"GSM8K num_questions={num_questions}")
    if not dry_run:
        dl_cmd = f"curl --noproxy '*' -sf -o {GSM8K_SCRIPT} {shlex.quote(GSM8K_URL)}"
        logger.write(f"[gsm8k] downloading script: {dl_cmd}")
        container.exec_sync(cname, dl_cmd)

    cmd = (
        f"python3 {shlex.quote(GSM8K_SCRIPT)} "
        f"--port {model.port} --num-questions {num_questions}"
    )
    logger.write(f"(in container {cname}) {cmd}")
    if dry_run:
        return AccuracyResult(accuracy=None, num_questions=num_questions, raw_output="", ok=True)

    try:
        rc, output = container.exec_stream_capture(cname, cmd, logger.path, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        logger.write("[gsm8k] TIMEOUT")
        return AccuracyResult(
            accuracy=None, num_questions=num_questions, raw_output="",
            ok=False, error=f"timeout after {timeout_sec}s",
        )

    if rc != 0:
        return AccuracyResult(
            accuracy=None, num_questions=num_questions, raw_output=output,
            ok=False, error=f"exit={rc}",
        )

    acc = parse_accuracy(output)
    return AccuracyResult(
        accuracy=acc,
        num_questions=num_questions,
        raw_output=output,
        ok=True,
        error=None if acc is not None else "could not parse accuracy",
    )
