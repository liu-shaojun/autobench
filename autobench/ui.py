from __future__ import annotations

from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.table import Table

from .state import ModelState, RunState


_STAGE_STYLE = {
    "pending": "dim",
    "container_up": "cyan",
    "server_starting": "yellow",
    "server_ready": "yellow",
    "accuracy": "yellow",
    "warmup": "yellow",
    "perf": "magenta",
    "done": "green",
    "failed": "red",
}


def _fmt_stage(stage: str) -> str:
    style = _STAGE_STYLE.get(stage, "white")
    return f"[{style}]{stage}[/{style}]"


def _fmt_accuracy(m: ModelState) -> str:
    if m.accuracy is not None:
        return f"[green]{m.accuracy:.1f}%[/green]"
    if m.accuracy_error:
        return f"[red]fail[/red]"
    return "-"


def _fmt_perf(m: ModelState) -> str:
    done, total, fail = m.perf_counts()
    if total == 0:
        return "-"
    bar_w = 10
    filled = int(bar_w * done / total) if total else 0
    bar = "█" * filled + "░" * (bar_w - filled)
    color = "green" if fail == 0 else "yellow"
    return f"[{color}]{bar}[/{color}] {done}/{total}" + (f" ([red]{fail} fail[/red])" if fail else "")


def render(state: RunState) -> Table:
    table = Table(title=f"autobench run {state.run_id}")
    table.add_column("Model", overflow="fold")
    table.add_column("Stage")
    table.add_column("GSM8K", justify="right")
    table.add_column("Perf", justify="left")
    table.add_column("Error", overflow="ellipsis", max_width=40, no_wrap=True)

    for name, m in state.models.items():
        err = ""
        if m.stage == "failed" and m.error:
            first_line = m.error.split("\n")[0][:60]
            err = f"[red]{first_line}[/red]"
        table.add_row(
            name,
            _fmt_stage(m.stage),
            _fmt_accuracy(m),
            _fmt_perf(m),
            err,
        )
    return table


@contextmanager
def live_display(state: RunState):
    console = Console()
    with Live(render(state), console=console, refresh_per_second=4, transient=False) as live:
        def _on_update(s: RunState) -> None:
            live.update(render(s))
        state.subscribe(_on_update)
        try:
            yield
        finally:
            live.update(render(state))
