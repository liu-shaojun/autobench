#!/usr/bin/env python3
"""Parse autobench log files and generate summary.csv from the raw log data."""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


PERF_SECTION = re.compile(
    r"=====.*PERF c=(\d+) in=(\d+) out=(\d+) n=(\d+)"
)

METRIC_PATTERNS = {
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

GSM8K_ACC = re.compile(r"[Aa]ccuracy:\s*([0-9.]+)")

LM_EVAL_SECTION = re.compile(r"=====.*LM_EVAL task=(\S+) limit=(\d+)")
LM_EVAL_ROW = re.compile(
    r"\|\s*(\S.*?)\s*\|[^|]*\|[^|]*\|[^|]*\|\s*(\S+)\s*\|[^|]*\|\s*([0-9.]+)\s*\|"
)

SMOKE_OK = re.compile(r"=====.*SMOKE_TEST")

SERVER_CMD = re.compile(r"--tensor-parallel-size=(\d+)")
QUANT_RE = re.compile(r"--quantization=(\S+)")
MODEL_RE = re.compile(r"--served-model-name=(\S+)")


def parse_log(log_path: Path) -> dict:
    text = log_path.read_text()
    lines = text.splitlines()

    result: dict = {
        "model_label": log_path.stem,
        "model_name": "",
        "tp": "",
        "quantization": "",
        "smoke_ok": "",
        "gsm8k_accuracy": "",
        "lm_eval": {},
        "perf": [],
    }

    # Extract model info from SERVER_START_CMD
    for line in lines:
        m = SERVER_CMD.search(line)
        if m:
            result["tp"] = m.group(1)
        m = QUANT_RE.search(line)
        if m:
            result["quantization"] = m.group(1)
        m = MODEL_RE.search(line)
        if m:
            result["model_name"] = m.group(1)
            break

    # Parse smoke
    for line in lines:
        if SMOKE_OK.search(line):
            result["smoke_ok"] = "yes"
            break

    # Parse gsm8k
    for line in lines:
        m = GSM8K_ACC.search(line)
        if m:
            result["gsm8k_accuracy"] = m.group(1)
            break

    # Parse lm_eval results
    current_lm_task = ""
    original_lm_task = ""
    for line in lines:
        m = LM_EVAL_SECTION.search(line)
        if m:
            original_lm_task = m.group(1)
            current_lm_task = original_lm_task
            continue
        if current_lm_task:
            m = LM_EVAL_ROW.search(line)
            if m:
                metric = m.group(2).strip()
                value = m.group(3).strip()
                if metric in ("acc", "acc_norm"):
                    result["lm_eval"].setdefault(original_lm_task, {})[metric] = value

    # Parse perf sections
    i = 0
    while i < len(lines):
        m = PERF_SECTION.search(lines[i])
        if m:
            c, inp, out, n = m.group(1), m.group(2), m.group(3), m.group(4)
            # Skip warmup (in=128 out=16)
            if inp == "128" and out == "16":
                i += 1
                continue
            # Collect the benchmark result block (stop at next section header)
            metrics = {}
            for j in range(i + 1, min(i + 80, len(lines))):
                if re.match(r"\s*=====\s*\[", lines[j]):
                    break
                for key, pat in METRIC_PATTERNS.items():
                    pm = pat.search(lines[j])
                    if pm:
                        metrics[key] = pm.group(1)
            if metrics:
                result["perf"].append({
                    "concurrency": c,
                    "input_len": inp,
                    "output_len": out,
                    "num_prompts": n,
                    **metrics,
                })
        i += 1

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m autobench.parse_logs <results_dir>")
        print("  e.g. python3 -m autobench.parse_logs results/20260428_085829")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    logs_dir = results_dir / "logs"
    if not logs_dir.is_dir():
        print(f"No logs directory found at {logs_dir}")
        sys.exit(1)

    log_files = sorted(logs_dir.glob("*.log"))
    log_files = [f for f in log_files if f.name != "all.log"]

    if not log_files:
        print("No log files found")
        sys.exit(1)

    # Collect all lm_eval task names
    all_results = [parse_log(f) for f in log_files]
    lm_tasks: list[str] = []
    seen: set[str] = set()
    for r in all_results:
        for t in r["lm_eval"]:
            if t not in seen:
                lm_tasks.append(t)
                seen.add(t)

    metric_cols = [
        "successful_requests", "benchmark_duration_s",
        "total_input_tokens", "total_generated_tokens",
        "request_throughput", "output_throughput", "total_throughput",
        "ttft_mean_ms", "tpot_mean_ms", "itl_mean_ms",
    ]

    cols = [
        "model", "tp", "smoke_ok", "gsm8k_accuracy",
        *lm_tasks,
        "concurrency", "input_len", "output_len", "num_prompts",
        *metric_cols,
    ]

    out_path = results_dir / "summary.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_results:
            base = {
                "model": r["model_label"],
                "tp": r["tp"],
                "smoke_ok": r["smoke_ok"],
                "gsm8k_accuracy": r["gsm8k_accuracy"],
            }
            for t in lm_tasks:
                metrics = r["lm_eval"].get(t, {})
                base[t] = metrics.get("acc_norm", metrics.get("acc", ""))

            if not r["perf"]:
                row = {c: "" for c in cols}
                row.update(base)
                w.writerow(row)
            else:
                for p in r["perf"]:
                    row = dict(base)
                    for c in ["concurrency", "input_len", "output_len", "num_prompts"] + metric_cols:
                        row[c] = p.get(c, "")
                    w.writerow(row)

    print(f"Wrote {out_path}")
    print(f"  {len(all_results)} models, {sum(len(r['perf']) for r in all_results)} perf entries")


if __name__ == "__main__":
    main()
