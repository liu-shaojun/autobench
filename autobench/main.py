from __future__ import annotations

import argparse
import atexit
import datetime
import os
import signal
import sys
from pathlib import Path

from . import config as config_mod
from . import container as container_mod
from . import runner as runner_mod
from . import summary as summary_mod
from . import ui as ui_mod
from .state import RunState


HERE = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = HERE / "configs" / "config.yaml"
DEFAULT_RESULTS = HERE / "results"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="autobench",
        description="vLLM multi-model benchmark runner. Edit configs/config.yaml, then just run.",
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help=f"path to config.yaml (default: {DEFAULT_CONFIG})")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS),
                   help=f"where to put results (default: {DEFAULT_RESULTS})")
    p.add_argument("--run-id", help="custom run id (default: timestamp)")
    p.add_argument("--no-ui", action="store_true", help="disable rich Live UI (plain logs)")
    p.add_argument("--dry-run", action="store_true",
                   help="print commands, don't touch docker")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"[autobench] config file not found: {config_path}", file=sys.stderr)
        return 2

    cfg = config_mod.load(config_path)
    if not cfg.models:
        print("[autobench] no models defined in config", file=sys.stderr)
        return 2

    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    state = RunState(run_id=run_id, results_dir=results_dir)
    for m in cfg.models:
        tp = int(m.server_args.get("tensor-parallel-size", 1))
        state.init_model(m.label, m.perf.combinations() if m.perf.enabled else [], tp=tp)

    def _shutdown_handler(signum, frame):
        print(f"\n[autobench] received signal {signum}, cleaning up...")
        summary_mod.write(state, results_dir / "summary.csv")
        container_mod.cleanup_all()
        print(f"[autobench] wrote summary, cleaned containers. exiting.")
        os._exit(130)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    # Clean up any leftover autobench containers from previous runs
    stale = container_mod.cleanup_all(dry_run=args.dry_run)
    if stale:
        print(f"[autobench] cleaned up {len(stale)} leftover container(s): {stale}")

    print(f"[autobench] run_id={run_id}")
    print(f"[autobench] results={results_dir}")
    print(f"[autobench] models: {[m.label for m in cfg.models]}")
    for m in cfg.models:
        print(f"  - {m.label}  path={m.model_path}")
        if m.smoke.enabled:
            print(f"    smoke: 5 prompts")
        if m.gsm8k.enabled:
            print(f"    gsm8k: {m.gsm8k.num_questions} questions")
        if m.lm_eval.enabled:
            task_names = [t["name"] for t in m.lm_eval.tasks]
            print(f"    lm_eval: {task_names}")
        if m.perf.enabled:
            p = m.perf
            print(f"    perf: concurrency={p.concurrency}  input_len={p.input_len}  output_len={p.output_len}")

    def _drive() -> None:
        for m in cfg.models:
            runner_mod.run_model(m, state, run_id, results_dir, dry_run=args.dry_run)
        summary_mod.write(state, results_dir / "summary.csv")

    if args.no_ui or args.dry_run:
        _drive()
    else:
        with ui_mod.live_display(state):
            _drive()

    print(f"[autobench] done. summary: {results_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
