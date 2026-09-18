"""Filesystem layout. One place so nothing hardcodes a path twice."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Runtime state: server registry, logs, the opencode manifest.
RUN_DIR = Path(os.environ.get("DEKHO_RUN_DIR", Path.home() / ".cache" / "dekho-local-inference"))
SERVERS_DIR = RUN_DIR / "servers"
LOG_DIR = RUN_DIR / "logs"
MANIFEST = Path(os.environ.get("DEKHO_MANIFEST", RUN_DIR / "manifest.json"))

# Repo data.
CATALOG = REPO_ROOT / "llmctl" / "catalog.json"
APPLE_DIR = REPO_ROOT / "apple-m-series"

ENV_NAME = os.environ.get("DEKHO_ENV_NAME", "dekho-apple-local-llm")


def ensure_dirs() -> None:
    for d in (RUN_DIR, SERVERS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def env_python() -> Path:
    """The interpreter that has mlx-lm, used to spawn servers.

    Prefer the one running us; fall back to the named conda env.
    """
    import sys

    here = Path(sys.executable)
    try:
        import mlx_lm  # noqa: F401

        return here
    except ImportError:
        pass
    base = os.environ.get("CONDA_PREFIX_1") or os.environ.get("CONDA_PREFIX")
    if base:
        cand = Path(base).parent / "envs" / ENV_NAME / "bin" / "python"
        if cand.exists():
            return cand
    cand = Path.home() / "miniforge3" / "envs" / ENV_NAME / "bin" / "python"
    if cand.exists():
        return cand
    return here
