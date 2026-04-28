from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass, field

from . import container
from .config import ModelConfig, PerfConfig
from .logutil import ModelLogger


@dataclass
class PerfResult:
    concurrency: int
    input_len: int
    output_len: int
    num_prompts: int
    metrics: dict[str, float] = field(default_factory=dict)
    raw_output: str = ""
    ok: bool = True
    error: str | None = None


# Patterns to match "vllm bench serve" summary lines, e.g. "Mean TTFT (ms): 123.45"
_METRIC_PATTERNS: dict[str, re.Pattern] = {
    "successful_requests":  re.compile(r"Successful requests:\s*([0-9.]+)"),
    "benchmark_duration_s": re.compile(r"Benchmark duration \(s\):\s*([0-9.]+)"),
    "total_input_tokens":   re.compile(r"Total input tokens:\s*([0-9.]+)"),
    "total_generated_tokens": re.compile(r"Total generated tokens:\s*([0-9.]+)"),
    "request_throughput":   re.compile(r"Request throughput \(req/s\):\s*([0-9.]+)"),
    "output_throughput":    re.compile(r"Output token throughput \(tok/s\):\s*([0-9.]+)"),
    "total_throughput":     re.compile(r"Total [Tt]oken throughput \(tok/s\):\s*([0-9.]+)"),
    "ttft_mean_ms":         re.compile(r"Mean TTFT \(ms\):\s*([0-9.]+)"),
    "tpot_mean_ms":         re.compile(r"Mean TPOT \(ms\):\s*([0-9.]+)"),
    "itl_mean_ms":          re.compile(r"Mean ITL \(ms\):\s*([0-9.]+)"),
}


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, pat in _METRIC_PATTERNS.items():
        m = pat.search(text)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                continue
    return out


def build_bench_cmd(model: ModelConfig, concurrency: int, input_len: int, output_len: int,
                    num_prompts: int, request_rate: str) -> str:
    parts = [
        "vllm", "bench", "serve",
        f"--model={model.model_path}",
        f"--served-model-name={model.name}",
        "--dataset-name=random",
        f"--random-input-len={input_len}",
        f"--random-output-len={output_len}",
        "--ignore-eos",
        f"--num-prompt={num_prompts}",
        "--trust_remote_code",
        f"--request-rate={request_rate}",
        "--backend=vllm",
        f"--port={model.port}",
        f"--max-concurrency={concurrency}",
    ]
    return " ".join(shlex.quote(p) for p in parts)


def run(
    model: ModelConfig,
    cname: str,
    logger: ModelLogger,
    perf_cfg: PerfConfig,
    concurrency: int,
    input_len: int,
    output_len: int,
    *,
    dry_run: bool = False,
) -> PerfResult:
    num_prompts = max(1, concurrency * perf_cfg.num_prompts_multiplier)
    cmd = build_bench_cmd(model, concurrency, input_len, output_len, num_prompts, perf_cfg.request_rate)
    timeout_sec = perf_cfg.timeout_sec

    logger.section(f"PERF c={concurrency} in={input_len} out={output_len} n={num_prompts}")
    logger.write(f"(in container {cname}) {cmd}")

    if dry_run:
        return PerfResult(concurrency=concurrency, input_len=input_len, output_len=output_len,
                          num_prompts=num_prompts, ok=True)

    try:
        rc, output = container.exec_stream_capture(cname, cmd, logger.path, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        logger.write("[perf] TIMEOUT")
        return PerfResult(
            concurrency=concurrency, input_len=input_len, output_len=output_len,
            num_prompts=num_prompts, ok=False, error=f"timeout after {timeout_sec}s",
        )

    if rc != 0:
        return PerfResult(
            concurrency=concurrency, input_len=input_len, output_len=output_len,
            num_prompts=num_prompts, raw_output=output, ok=False,
            error=f"exit={r.returncode}",
        )

    metrics = parse_metrics(output)
    ok = len(metrics) > 0
    return PerfResult(
        concurrency=concurrency, input_len=input_len, output_len=output_len,
        num_prompts=num_prompts, metrics=metrics, raw_output=output, ok=ok,
        error=None if ok else "could not parse metrics",
    )
