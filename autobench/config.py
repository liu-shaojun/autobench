from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GSM8KConfig:
    enabled: bool
    num_questions: int
    timeout_sec: int


@dataclass
class PerfConfig:
    enabled: bool
    concurrency: list[int]
    input_len: list[int]
    output_len: list[int]
    num_prompts_multiplier: int
    request_rate: str
    timeout_sec: int

    def combinations(self) -> list[tuple[int, int, int]]:
        return [
            (c, i, o)
            for c in self.concurrency
            for i in self.input_len
            for o in self.output_len
        ]


@dataclass
class ModelConfig:
    name: str
    label: str                      # unique display key (e.g. "Qwen3.5-27B_sym_int4")
    image: str
    host_model_dir: str
    container_model_dir: str
    host_work_dir: str
    container_work_dir: str
    host_tmp_dir: str
    container_tmp_dir: str
    shm_size: str
    http_proxy: str
    https_proxy: str
    no_proxy: str
    port: int
    startup_timeout_sec: int
    ze_affinity_mask: str | None
    server_env: dict[str, str]
    server_args: dict[str, Any]
    model_path: str                 # absolute path inside container
    gsm8k: GSM8KConfig
    perf: PerfConfig


@dataclass
class RunConfig:
    models: list[ModelConfig] = field(default_factory=list)


def _deep_merge(base: dict, override: dict) -> dict:
    if not isinstance(base, dict):
        return copy.deepcopy(override)
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _build_gsm8k(d: dict) -> GSM8KConfig:
    return GSM8KConfig(
        enabled=bool(d.get("enabled", True)),
        num_questions=int(d.get("num_questions", 100)),
        timeout_sec=int(d.get("timeout_sec", 1800)),
    )


def _build_perf(d: dict) -> PerfConfig:
    return PerfConfig(
        enabled=bool(d.get("enabled", True)),
        concurrency=list(d.get("concurrency", [1])),
        input_len=list(d.get("input_len", [1024])),
        output_len=list(d.get("output_len", [1024])),
        num_prompts_multiplier=int(d.get("num_prompts_multiplier", 1)),
        request_rate=str(d.get("request_rate", "inf")),
        timeout_sec=int(d.get("timeout_sec", 3600)),
    )


def load(config_path: Path) -> RunConfig:
    doc = yaml.safe_load(config_path.read_text())

    docker_cfg = doc.get("docker", {}) or {}
    server_defaults = doc.get("server_defaults", {}) or {}
    tests_defaults = doc.get("tests", {}) or {}

    models: list[ModelConfig] = []
    for entry in doc.get("models", []) or []:
        name = entry["name"]

        # merge server config (defaults <- per-model overrides)
        server_merged = _deep_merge(server_defaults, entry.get("server", {}) or {})

        # merge tests (defaults <- per-model overrides)
        tests_merged = _deep_merge(tests_defaults, entry.get("tests", {}) or {})
        gsm8k = _build_gsm8k(tests_merged.get("gsm8k", {}) or {})
        perf = _build_perf(tests_merged.get("perf", {}) or {})

        container_model_dir = docker_cfg.get("container_model_dir", "/llm/models")
        model_path = entry.get("model_path") or f"{container_model_dir}/{name}"

        # Build unique label: use explicit label, or auto-generate from name + quantization
        label = entry.get("label")
        if not label:
            quant = (server_merged.get("args") or {}).get("quantization")
            label = f"{name}_{quant}" if quant else name

        models.append(ModelConfig(
            name=name,
            label=label,
            image=docker_cfg["image"],
            host_model_dir=docker_cfg["host_model_dir"],
            container_model_dir=container_model_dir,
            host_work_dir=docker_cfg.get("host_work_dir", ""),
            container_work_dir=docker_cfg.get("container_work_dir", "/llm/shaojun"),
            host_tmp_dir=docker_cfg.get("host_tmp_dir", "/tmp"),
            container_tmp_dir=docker_cfg.get("container_tmp_dir", "/tmp"),
            shm_size=docker_cfg.get("shm_size", "32g"),
            http_proxy=docker_cfg.get("http_proxy", ""),
            https_proxy=docker_cfg.get("https_proxy", ""),
            no_proxy=docker_cfg.get("no_proxy", "localhost,127.0.0.1"),
            port=int(server_merged.get("port", 9005)),
            startup_timeout_sec=int(server_merged.get("startup_timeout_sec", 1200)),
            ze_affinity_mask=entry.get("ze_affinity_mask"),
            server_env=dict(server_merged.get("env", {}) or {}),
            server_args=dict(server_merged.get("args", {}) or {}),
            model_path=model_path,
            gsm8k=gsm8k,
            perf=perf,
        ))

    return RunConfig(models=models)
