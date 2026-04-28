from __future__ import annotations

import datetime
import threading
from pathlib import Path


class ModelLogger:
    """Append-only per-model log with section markers."""

    def __init__(self, path: Path, mirror: Path | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mirror = mirror
        if mirror:
            mirror.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _now(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write_both(self, text: str) -> None:
        with self.path.open("a") as f:
            f.write(text)
            f.flush()
        if self._mirror:
            with self._mirror.open("a") as f:
                f.write(text)
                f.flush()

    def section(self, title: str) -> None:
        line = f"\n===== [{self._now()}] {title} =====\n"
        with self._lock:
            self._write_both(line)

    def write(self, text: str) -> None:
        with self._lock:
            if not text.endswith("\n"):
                text += "\n"
            self._write_both(text)
