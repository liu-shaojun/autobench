from __future__ import annotations

import threading
import traceback
from pathlib import Path

from . import accuracy as accuracy_mod
from . import container as container_mod
from . import lm_eval as lm_eval_mod
from . import perf as perf_mod
from . import server as server_mod
from . import smoke as smoke_mod
from . import summary as summary_mod
from .config import ModelConfig
from .logutil import ModelLogger
from .state import RunState


def run_model(
    model: ModelConfig,
    state: RunState,
    run_id: str,
    results_dir: Path,
    *,
    dry_run: bool = False,
) -> None:
    cname = container_mod.container_name(model, run_id)
    log_path = results_dir / "logs" / f"{model.label.replace('/', '_')}.log"
    all_log = results_dir / "logs" / "all.log"
    logger = ModelLogger(log_path, mirror=all_log)
    tail_stop = threading.Event()
    tail_thread = None

    # 1. container up
    state.set_stage(model.label, "container_up", container_status="starting")
    try:
        logger.section(f"CONTAINER_UP name={cname}")
        logger.write("docker run: " + str(container_mod.build_run_cmd(model, cname)))
        container_mod.up(model, cname, dry_run=dry_run)
        state.set_stage(model.label, "container_up", container_status="up")
    except Exception as e:
        logger.section("CONTAINER_UP_FAIL")
        logger.write(traceback.format_exc())
        state.fail_model(model.label, f"container up: {e}")
        return

    try:
        # 2. server start
        state.set_stage(model.label, "server_starting", server_status="starting")
        try:
            server_mod.start(model, cname, logger, dry_run=dry_run)
            if not dry_run:
                tail_thread = server_mod._tail_thread(cname, logger, tail_stop)
                server_mod.wait_ready(model, cname, logger)
            state.set_stage(model.label, "server_ready", server_status="ready")
        except Exception as e:
            logger.section("SERVER_START_FAIL")
            logger.write(traceback.format_exc())
            state.set_stage(model.label, "failed", server_status="fail", error=f"server: {e}")
            return

        # 3. smoke test
        if model.smoke.enabled:
            state.set_stage(model.label, "smoke")
            try:
                smoke_result = smoke_mod.run(model, cname, logger, dry_run=dry_run)
                state.set_smoke(model.label, smoke_result)
            except Exception as e:
                logger.section("SMOKE_EXCEPTION")
                logger.write(traceback.format_exc())

        # 4. accuracy (gsm8k) — per-model switch
        if not model.gsm8k.enabled:
            state.set_accuracy(model.label, accuracy=None, ok=True, error="disabled")
        else:
            state.set_stage(model.label, "accuracy")
            try:
                acc = accuracy_mod.run(
                    model, cname, logger,
                    num_questions=model.gsm8k.num_questions,
                    timeout_sec=model.gsm8k.timeout_sec,
                    dry_run=dry_run,
                )
                state.set_accuracy(
                    model.label,
                    accuracy=acc.accuracy,
                    ok=acc.ok,
                    error=acc.error,
                )
            except Exception as e:
                logger.section("GSM8K_EXCEPTION")
                logger.write(traceback.format_exc())
                state.set_accuracy(model.label, accuracy=None, ok=False, error=str(e))

        # 4. lm_eval — per-model switch
        if model.lm_eval.enabled:
            state.set_stage(model.label, "lm_eval")
            try:
                lm_result = lm_eval_mod.run(
                    model, cname, logger,
                    tasks=model.lm_eval.tasks,
                    timeout_sec=model.lm_eval.timeout_sec,
                    dry_run=dry_run,
                )
                state.set_lm_eval(model.label, lm_result)
            except Exception as e:
                logger.section("LM_EVAL_EXCEPTION")
                logger.write(traceback.format_exc())

        # 5. perf matrix — per-model switch
        if model.perf.enabled:
            state.set_stage(model.label, "warmup")
            logger.section("WARMUP")
            try:
                perf_mod.run(
                    model, cname, logger, model.perf,
                    concurrency=1, input_len=128, output_len=16,
                    dry_run=dry_run,
                )
            except Exception:
                logger.write("[warmup] failed, continuing anyway")
                logger.write(traceback.format_exc())

            state.set_stage(model.label, "perf")
            combos = model.perf.combinations()
            for (c, i, o) in combos:
                state.set_perf(model.label, c, i, o, status="running")
                try:
                    pr = perf_mod.run(
                        model, cname, logger, model.perf, c, i, o,
                        dry_run=dry_run,
                    )
                    if pr.ok:
                        state.set_perf(model.label, c, i, o, status="ok", metrics=pr.metrics)
                    else:
                        state.set_perf(model.label, c, i, o, status="fail", error=pr.error)
                except Exception as e:
                    logger.section(f"PERF_EXCEPTION c={c} in={i} out={o}")
                    logger.write(traceback.format_exc())
                    state.set_perf(model.label, c, i, o, status="fail", error=str(e))

                # short-circuit if server died
                if not dry_run and not server_mod.is_alive(cname, model.port):
                    logger.section("SERVER_DIED_DURING_PERF")
                    for e in state.models[model.label].perf_entries:
                        if e.status == "pending":
                            state.set_perf(
                                model.label,
                                e.concurrency, e.input_len, e.output_len,
                                status="skipped", error="server died",
                            )
                    break

        if state.models[model.label].stage != "failed":
            state.set_stage(model.label, "done")

        summary_mod.write(state, results_dir / "summary.csv")

    finally:
        try:
            if not dry_run:
                server_mod.stop(cname, logger)
        except Exception:
            logger.write(traceback.format_exc())

        tail_stop.set()
        if tail_thread is not None:
            tail_thread.join(timeout=5)

        try:
            logger.section("CONTAINER_DOWN")
            container_mod.down(cname, dry_run=dry_run)
        except Exception:
            logger.write(traceback.format_exc())
