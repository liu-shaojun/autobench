from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass, field

from . import container
from .config import ModelConfig
from .logutil import ModelLogger


@dataclass
class LmEvalTaskResult:
    task: str
    limit: int
    metrics: dict[str, float] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None


@dataclass
class LmEvalResult:
    tasks: list[LmEvalTaskResult] = field(default_factory=list)
    raw_output: str = ""
    ok: bool = True
    error: str | None = None


# Parse lm-eval table rows like:
# |arc_challenge|      1|none  |     0|acc     |↑  |  0.4|±  |0.2449|
_TABLE_ROW = re.compile(
    r"\|\s*(\S.*?)\s*\|[^|]*\|[^|]*\|[^|]*\|\s*(\S+)\s*\|[^|]*\|\s*([0-9.]+)\s*\|"
)


def parse_lm_eval_output(text: str) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    current_task = ""
    for line in text.splitlines():
        m = _TABLE_ROW.search(line)
        if not m:
            continue
        task_name = m.group(1).strip()
        metric = m.group(2).strip()
        value = m.group(3).strip()
        if task_name and task_name.lower() not in ("tasks", "-"):
            current_task = task_name
        if current_task and metric and value:
            try:
                results.setdefault(current_task, {})[metric] = float(value)
            except ValueError:
                continue
    return results


def _proxy_env(model: ModelConfig) -> str:
    parts = []
    if model.http_proxy:
        parts.append(f"http_proxy={shlex.quote(model.http_proxy)}")
    if model.https_proxy:
        parts.append(f"https_proxy={shlex.quote(model.https_proxy)}")
    if model.no_proxy:
        parts.append(f"no_proxy={shlex.quote(model.no_proxy)}")
    return "export " + " ".join(parts) + "; " if parts else ""


def run(
    model: ModelConfig,
    cname: str,
    logger: ModelLogger,
    tasks: list[dict],
    *,
    timeout_sec: int = 3600,
    dry_run: bool = False,
) -> LmEvalResult:
    result = LmEvalResult()
    proxy = _proxy_env(model)

    if not dry_run:
        logger.write("[lm_eval] ensuring lm-eval + tenacity installed...")
        install_cmd = (
            f"{proxy}"
            f"pip show lm-eval >/dev/null 2>&1 || "
            f"pip install --proxy {shlex.quote(model.http_proxy or '')} "
            f"'lm-eval==0.4.4' tenacity 2>&1 | tail -5"
        )
        install_r = container.exec_sync(cname, install_cmd, timeout=300)
        logger.write((install_r.stdout or "").strip())

    for task_cfg in tasks:
        task_name = task_cfg["name"]
        limit = task_cfg.get("limit", 500)
        num_fewshot = task_cfg.get("num_fewshot", 0)
        num_concurrent = task_cfg.get("num_concurrent", 4)

        cmd = (
            f"{proxy}"
            f"lm_eval"
            f" --model local-completions"
            f" --model_args model={shlex.quote(model.name)},"
            f"tokenizer={shlex.quote(model.model_path)},"
            f"base_url=http://localhost:{model.port}/v1/completions,"
            f"num_concurrent={num_concurrent},max_retries=2,tokenized_requests=False"
            f" --tasks {shlex.quote(task_name)}"
            f" --limit {limit}"
            f" --num_fewshot {num_fewshot}"
        )

        logger.section(f"LM_EVAL task={task_name} limit={limit}")
        logger.write(f"(in container {cname}) {cmd}")

        if dry_run:
            result.tasks.append(LmEvalTaskResult(task=task_name, limit=limit, ok=True))
            continue

        try:
            rc, output = container.exec_stream_capture(
                cname, cmd, logger.path, timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            logger.write(f"[lm_eval] {task_name} TIMEOUT")
            result.tasks.append(LmEvalTaskResult(
                task=task_name, limit=limit, ok=False,
                error=f"timeout after {timeout_sec}s",
            ))
            continue

        if rc != 0:
            result.tasks.append(LmEvalTaskResult(
                task=task_name, limit=limit, ok=False,
                error=f"exit={rc}",
            ))
            continue

        parsed = parse_lm_eval_output(output)
        # Match by exact name first, then try suffix match (e.g. mmlu_high_school_... -> high_school_...)
        metrics = parsed.get(task_name, {})
        if not metrics:
            for key, val in parsed.items():
                if task_name.endswith(key) or key.endswith(task_name):
                    metrics = val
                    break
        result.tasks.append(LmEvalTaskResult(
            task=task_name, limit=limit, metrics=metrics,
            ok=len(metrics) > 0,
            error=None if metrics else "could not parse metrics",
        ))

    result.ok = all(t.ok for t in result.tasks)
    return result
