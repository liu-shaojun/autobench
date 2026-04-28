from __future__ import annotations

import datetime
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PerfEntry:
    concurrency: int
    input_len: int
    output_len: int
    status: str = "pending"   # pending, running, ok, fail, skipped
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelState:
    name: str
    stage: str = "pending"    # pending, container_up, server_starting, server_ready,
                              # accuracy, perf, done, failed
    container_status: str = "-"
    server_status: str = "-"
    accuracy: float | None = None
    accuracy_ok: bool = False
    accuracy_error: str | None = None
    error: str | None = None
    perf_entries: list[PerfEntry] = field(default_factory=list)

    def perf_counts(self) -> tuple[int, int, int]:
        """returns (done, total, fail)"""
        total = len(self.perf_entries)
        done = sum(1 for e in self.perf_entries if e.status in {"ok", "fail", "skipped"})
        fail = sum(1 for e in self.perf_entries if e.status == "fail")
        return done, total, fail


class RunState:
    def __init__(self, run_id: str, results_dir: Path):
        self.run_id = run_id
        self.results_dir = results_dir
        self.status_path = results_dir / "status.json"
        self.started_at = datetime.datetime.now().isoformat(timespec="seconds")
        self.models: dict[str, ModelState] = {}
        self._lock = threading.Lock()
        self._observers: list = []

    # ---- observer ----
    def subscribe(self, fn) -> None:
        self._observers.append(fn)

    def _notify(self) -> None:
        for fn in list(self._observers):
            try:
                fn(self)
            except Exception:
                pass

    # ---- mutation ----
    def init_model(self, name: str, combos: list[tuple[int, int, int]]) -> None:
        with self._lock:
            m = ModelState(name=name)
            m.perf_entries = [PerfEntry(c, i, o) for (c, i, o) in combos]
            self.models[name] = m
        self._flush()
        self._notify()

    def set_stage(self, name: str, stage: str, **extra: Any) -> None:
        with self._lock:
            m = self.models[name]
            m.stage = stage
            for k, v in extra.items():
                setattr(m, k, v)
        self._flush()
        self._notify()

    def set_accuracy(self, name: str, *, accuracy: float | None, ok: bool, error: str | None) -> None:
        with self._lock:
            m = self.models[name]
            m.accuracy = accuracy
            m.accuracy_ok = ok
            m.accuracy_error = error
        self._flush()
        self._notify()

    def set_perf(self, name: str, c: int, i: int, o: int, *, status: str,
                 metrics: dict[str, float] | None = None, error: str | None = None) -> None:
        with self._lock:
            m = self.models[name]
            for e in m.perf_entries:
                if e.concurrency == c and e.input_len == i and e.output_len == o:
                    e.status = status
                    if metrics:
                        e.metrics = metrics
                    e.error = error
                    break
        self._flush()
        self._notify()

    def fail_model(self, name: str, error: str) -> None:
        with self._lock:
            m = self.models[name]
            m.stage = "failed"
            m.error = error
        self._flush()
        self._notify()

    # ---- persistence ----
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "models": {
                name: {
                    "stage": m.stage,
                    "container_status": m.container_status,
                    "server_status": m.server_status,
                    "accuracy": m.accuracy,
                    "accuracy_ok": m.accuracy_ok,
                    "accuracy_error": m.accuracy_error,
                    "error": m.error,
                    "perf": [
                        {
                            "concurrency": e.concurrency,
                            "input_len": e.input_len,
                            "output_len": e.output_len,
                            "status": e.status,
                            "error": e.error,
                            "metrics": e.metrics,
                        }
                        for e in m.perf_entries
                    ],
                }
                for name, m in self.models.items()
            },
        }

    def _flush(self) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(self.status_path)
