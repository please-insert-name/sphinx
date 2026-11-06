"""Konfiguration: config.json im Skill-Ordner, überschreibbar per Umgebung/CLI.

Ollama-Host (Priorität): --ollama-host > PSEUDO_OLLAMA_HOST > config.llm.host
                         > OLLAMA_HOST > http://localhost:11434
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = SKILL_DIR / "config.json"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    cfg = json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    if overrides:
        cfg = _merge(cfg, overrides)
    host = (os.environ.get("PSEUDO_OLLAMA_HOST") or cfg["llm"].get("host")
            or os.environ.get("OLLAMA_HOST") or "http://localhost:11434")
    if overrides and overrides.get("llm", {}).get("host"):
        host = overrides["llm"]["host"]
    cfg["llm"]["host"] = host
    return cfg
