from __future__ import annotations

import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import requests

from . import container

_NO_PROXY_SESSION = requests.Session()
_NO_PROXY_SESSION.trust_env = False
from .config import ModelConfig
from .logutil import ModelLogger


class ServerNotReady(Exception):
    pass


class ServerCrashed(Exception):
    pass


SERVER_LOG_IN_CONTAINER = "/tmp/autobench_vllm_server.log"
SERVER_PID_FILE = "/tmp/autobench_vllm_server.pid"
READY_MARKER = "Application startup complete"


def _format_arg(key: str, val: Any) -> list[str]:
    if isinstance(val, bool):
        return [f"--{key}"] if val else []
    return [f"--{key}={val}"]


def build_server_cmd(model: ModelConfig) -> str:
    parts: list[str] = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        f"--model={model.model_path}",
        f"--served-model-name={model.name}",
        f"--port={model.port}",
    ]
    skip = {"model", "served-model-name", "port"}
    for k, v in model.server_args.items():
        if k in skip:
            continue
        parts.extend(_format_arg(k, v))
    return " ".join(shlex.quote(p) for p in parts)


def build_env_exports(model: ModelConfig) -> str:
    env_kv: dict[str, str] = {}
    if model.ze_affinity_mask:
        env_kv["ZE_AFFINITY_MASK"] = model.ze_affinity_mask
    env_kv.update({k: str(v) for k, v in model.server_env.items()})
    if not env_kv:
        return ""
    return "export " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env_kv.items()) + "; "


def start(
    model: ModelConfig,
    cname: str,
    logger: ModelLogger,
    *,
    dry_run: bool = False,
) -> str | None:
    """Launch vLLM server inside container, detached. Returns the shell command."""
    inner = build_server_cmd(model)
    env_exports = build_env_exports(model)
    server_cmd = f"{inner} > {SERVER_LOG_IN_CONTAINER} 2>&1"
    full = f"{env_exports}{server_cmd}"
    logger.section("SERVER_START_CMD")
    logger.write(f"(in container {cname}) {full}")
    if dry_run:
        return full
    # Make sure log file and pid file are clean before start
    container.exec_sync(cname, f": > {SERVER_LOG_IN_CONTAINER}")
    container.exec_sync(cname, f"rm -f {SERVER_PID_FILE}")
    # Wrap: export envs, record PID, then exec the server
    wrapped = f"bash -c '{env_exports}echo $$ > {SERVER_PID_FILE}; exec {server_cmd}'"
    container.exec_detached(cname, wrapped)
    return full


def _tail_thread(cname: str, logger: ModelLogger, stop_evt: threading.Event) -> threading.Thread:
    """Background thread that streams container-side server log into the model log."""
    def _run() -> None:
        logger.section("SERVER_STDOUT")
        proc = subprocess.Popen(
            ["docker", "exec", cname, "bash", "-lc",
             f"touch {SERVER_LOG_IN_CONTAINER}; tail -n +1 -F {SERVER_LOG_IN_CONTAINER}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if stop_evt.is_set():
                    break
                logger.write(line.rstrip("\n"))
        finally:
            try:
                proc.terminate()
            except Exception:
                pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def wait_ready(
    model: ModelConfig,
    cname: str,
    logger: ModelLogger,
    *,
    on_tick=None,
) -> None:
    """Block until server is ready, or raise ServerNotReady after startup_timeout_sec."""
    deadline = time.time() + model.startup_timeout_sec
    log_marker_found = False
    url = f"http://127.0.0.1:{model.port}/v1/models"

    while time.time() < deadline:
        if on_tick:
            on_tick(int(deadline - time.time()))

        if not log_marker_found:
            r = container.exec_sync(
                cname,
                f"grep -q {shlex.quote(READY_MARKER)} {SERVER_LOG_IN_CONTAINER}",
            )
            if r.returncode == 0:
                log_marker_found = True
                logger.section("SERVER_LOG_MARKER_FOUND")

        if log_marker_found:
            try:
                resp = _NO_PROXY_SESSION.get(url, timeout=5)
                if resp.status_code == 200:
                    logger.section("SERVER_READY")
                    return
            except requests.RequestException:
                pass

        # Check the server process hasn't died early (use recorded PID to avoid self-match)
        r = container.exec_sync(
            cname,
            f"test -f {SERVER_PID_FILE} && kill -0 $(cat {SERVER_PID_FILE}) 2>/dev/null "
            f"&& echo alive || echo dead",
        )
        if r.stdout.strip() == "dead":
            # Grab last lines of log to include in the error message
            log_r = container.exec_sync(cname, f"tail -20 {SERVER_LOG_IN_CONTAINER}")
            tail_text = (log_r.stdout or "").strip()
            logger.section("SERVER_DIED_DURING_STARTUP")
            logger.write(tail_text)
            raise ServerNotReady(f"vLLM process exited before ready. Last log:\n{tail_text}")

        time.sleep(5)

    logger.section("SERVER_READY_TIMEOUT")
    raise ServerNotReady(f"Server not ready after {model.startup_timeout_sec}s")


def is_alive(cname: str, port: int) -> bool:
    if not container.is_running(cname):
        return False
    try:
        r = _NO_PROXY_SESSION.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def stop(cname: str, logger: ModelLogger) -> None:
    logger.section("SERVER_STOP")
    kill_cmd = (
        f"if [ -f {SERVER_PID_FILE} ]; then "
        f"  PID=$(cat {SERVER_PID_FILE}); "
        f"  kill -TERM $PID 2>/dev/null; sleep 3; "
        f"  kill -KILL $PID 2>/dev/null; "
        f"  rm -f {SERVER_PID_FILE}; "
        f"fi"
    )
    container.exec_sync(cname, kill_cmd)
