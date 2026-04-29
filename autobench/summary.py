from __future__ import annotations

import csv
import threading
from pathlib import Path

from .state import ModelState, RunState


_LOCK = threading.Lock()


METRIC_COLS = [
    "successful_requests", "benchmark_duration_s",
    "total_input_tokens", "total_generated_tokens",
    "request_throughput", "output_throughput", "total_throughput",
    "ttft_mean_ms", "tpot_mean_ms", "itl_mean_ms",
]


BASE_COLS = [
    "model", "tp", "smoke_ok", "gsm8k_accuracy", "gsm8k_ok",
]

PERF_COLS = [
    "concurrency", "input_len", "output_len",
    "status", "error",
    *METRIC_COLS,
]


def _lm_eval_cols(state: RunState) -> list[str]:
    """Return lm_eval task names from config for consistent CSV columns."""
    return state.lm_eval_task_names


def _lm_eval_values(m: ModelState, lm_cols: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for task in lm_cols:
        metrics = m.lm_eval_results.get(task, {})
        if "error" in metrics:
            out[task] = "fail"
        elif metrics:
            acc = metrics.get("acc_norm", metrics.get("acc", 0))
            out[task] = f"{acc:.4f}"
        else:
            out[task] = ""
    return out


def write(state: RunState, path: Path) -> None:
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        lm_cols = _lm_eval_cols(state)
        cols = BASE_COLS + lm_cols + PERF_COLS
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for name, m in state.models.items():
                base = {
                    "model": name,
                    "tp": m.tp,
                    "smoke_ok": {"ok": "yes", "fail": "no", "disabled": "skip"}.get(m.smoke_status, ""),
                    "gsm8k_accuracy": f"{m.accuracy:.2f}%" if m.accuracy is not None else "",
                    "gsm8k_ok": "skip" if m.accuracy_error == "disabled" else ("yes" if m.accuracy_ok else "no"),
                }
                base.update(_lm_eval_values(m, lm_cols))
                if not m.perf_entries:
                    row = {c: "" for c in cols}
                    row.update(base)
                    row["status"] = m.stage
                    row["error"] = m.error or ""
                    w.writerow(row)
                    continue
                for e in m.perf_entries:
                    row = dict(base)
                    row["concurrency"] = e.concurrency
                    row["input_len"] = e.input_len
                    row["output_len"] = e.output_len
                    row["status"] = e.status
                    row["error"] = e.error or ""
                    for c in METRIC_COLS:
                        row[c] = f"{e.metrics[c]:.4f}" if c in e.metrics else ""
                    w.writerow(row)
