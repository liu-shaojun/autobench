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


COLS = [
    "model", "gsm8k_accuracy", "gsm8k_ok",
    "concurrency", "input_len", "output_len",
    "status", "error",
    *METRIC_COLS,
]


def write(state: RunState, path: Path) -> None:
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            for name, m in state.models.items():
                if not m.perf_entries:
                    # Emit one row so failed/no-perf models still show up
                    w.writerow(_row_for_model_no_perf(m))
                    continue
                for e in m.perf_entries:
                    row = {
                        "model": name,
                        "gsm8k_accuracy": f"{m.accuracy:.2f}" if m.accuracy is not None else "",
                        "gsm8k_ok": "yes" if m.accuracy_ok else "no",
                        "concurrency": e.concurrency,
                        "input_len": e.input_len,
                        "output_len": e.output_len,
                        "status": e.status,
                        "error": e.error or "",
                    }
                    for c in METRIC_COLS:
                        row[c] = f"{e.metrics[c]:.4f}" if c in e.metrics else ""
                    w.writerow(row)


def _row_for_model_no_perf(m: ModelState) -> dict:
    row = {c: "" for c in COLS}
    row["model"] = m.name
    row["gsm8k_accuracy"] = f"{m.accuracy:.2f}" if m.accuracy is not None else ""
    row["gsm8k_ok"] = "yes" if m.accuracy_ok else "no"
    row["status"] = m.stage
    row["error"] = m.error or ""
    return row
