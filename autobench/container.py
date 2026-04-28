from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from .config import ModelConfig


def _run(cmd: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
    )


def container_name(model: ModelConfig, run_id: str) -> str:
    safe = model.label.replace("/", "_").replace(":", "_")
    return f"autobench-{safe}-{run_id}"


def build_run_cmd(model: ModelConfig, name: str) -> list[str]:
    cmd = [
        "docker", "run", "-td",
        "--privileged",
        "--net=host",
        "--device=/dev/dri",
        f"--name={name}",
        "-v", f"{model.host_model_dir}:{model.container_model_dir}",
        "-v", f"{model.host_work_dir}:{model.container_work_dir}",
        "-v", f"{model.host_tmp_dir}:{model.container_tmp_dir}",
        "-e", f"no_proxy={model.no_proxy}",
    ]
    if model.http_proxy:
        cmd += ["-e", f"http_proxy={model.http_proxy}"]
    if model.https_proxy:
        cmd += ["-e", f"https_proxy={model.https_proxy}"]
    cmd += [
        f"--shm-size={model.shm_size}",
        "--entrypoint", "/bin/bash",
        model.image,
    ]
    return cmd


def up(model: ModelConfig, name: str, *, dry_run: bool = False) -> str:
    cmd = build_run_cmd(model, name)
    if dry_run:
        return " ".join(shlex.quote(c) for c in cmd)
    _run(["docker", "rm", "-f", name], check=False)
    _run(cmd)
    return name


def exec_detached(name: str, inner_cmd: str) -> None:
    """Run a shell command inside the container, detached."""
    subprocess.run(
        ["docker", "exec", "-d", name, "bash", "-lc", inner_cmd],
        check=True,
    )


def exec_sync(
    name: str,
    inner_cmd: str,
    *,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a shell command inside the container synchronously, capture output."""
    envs: list[str] = []
    if env:
        for k, v in env.items():
            envs += ["-e", f"{k}={v}"]
    return subprocess.run(
        ["docker", "exec", *envs, name, "bash", "-lc", inner_cmd],
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def exec_stream(
    name: str,
    inner_cmd: str,
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> int:
    """Run a shell command inside the container, streaming stdout+stderr into log_path."""
    envs: list[str] = []
    if env:
        for k, v in env.items():
            envs += ["-e", f"{k}={v}"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as f:
        proc = subprocess.Popen(
            ["docker", "exec", *envs, name, "bash", "-lc", inner_cmd],
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise


def exec_stream_capture(
    name: str,
    inner_cmd: str,
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> tuple[int, str]:
    """Run command in container, stream each line to log file in real-time, and return (rc, full_output)."""
    envs: list[str] = []
    if env:
        for k, v in env.items():
            envs += ["-e", f"{k}={v}"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["docker", "exec", *envs, name, "bash", "-lc", inner_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    try:
        assert proc.stdout is not None
        with log_path.open("a") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                lines.append(line)
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise
    return rc, "".join(lines)


def is_running(name: str) -> bool:
    r = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        text=True, capture_output=True,
    )
    return r.returncode == 0 and r.stdout.strip() == "true"


def down(name: str, *, dry_run: bool = False) -> None:
    if dry_run:
        return
    _run(["docker", "stop", "-t", "10", name], check=False)
    _run(["docker", "rm", "-f", name], check=False)


def cleanup_all(*, dry_run: bool = False) -> list[str]:
    """Stop and remove all containers with the autobench- prefix."""
    r = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}", "--filter", "name=autobench-"],
        text=True, capture_output=True,
    )
    names = [n.strip() for n in r.stdout.splitlines() if n.strip()]
    if not dry_run:
        for n in names:
            _run(["docker", "rm", "-f", n], check=False)
    return names
