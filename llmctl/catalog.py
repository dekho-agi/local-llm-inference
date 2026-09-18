"""Model catalog: what exists, what is cached, what it costs.

Two sources, merged:
  - catalog.json in this package — curated reference: every model we have
    evaluated, including ones not downloaded and ones served by other runtimes
    (image, VLM, audio). This is the repo's model reference.
  - the local HuggingFace cache — ground truth for what is actually available
    offline, plus real on-disk size, context window and tool parser read from
    each model's own files.
"""

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

from .paths import CATALOG


@dataclass
class Model:
    repo_id: str
    label: str = ""
    model_class: str = (
        "text"  # text | image | image-edit | vlm | omni | stt | tts | music | video | embed
    )
    runtime: str = "mlx-lm"  # mlx-lm | mlx-vlm | mflux | mlx-audio | mlx-gen
    role: str = ""  # daily-driver | fast | heavy | chat-only | reference
    params: str = ""
    size_gb: float | None = None  # catalog estimate
    disk_gb: float | None = None  # measured, if cached
    context: int | None = None
    context_native: int | None = None
    output: int = 32768
    tool_parser: str | None = None
    arch: str | None = None
    kv_bytes_per_token: int | None = None
    cached: bool = False
    incomplete: bool = False  # download in progress / interrupted
    notes: str = ""
    profiles: list[str] = field(default_factory=list)

    @property
    def servable(self) -> bool:
        """Can `llmctl start` serve this? Only complete mlx-lm text models."""
        return self.runtime == "mlx-lm" and self.cached and not self.incomplete

    @property
    def ready(self) -> bool:
        """Present and fully downloaded — safe to offer in a picker."""
        return self.cached and not self.incomplete

    @property
    def agentic(self) -> bool:
        return bool(self.tool_parser)

    def kv_gb(self, ctx: int | None = None) -> float | None:
        n = ctx or self.context
        if not (self.kv_bytes_per_token and n):
            return None
        return round(self.kv_bytes_per_token * n / 1e9, 2)

    def total_gb(self, ctx: int | None = None) -> float | None:
        w = self.disk_gb or self.size_gb
        kv = self.kv_gb(ctx)
        if w is None:
            return None
        return round(w + (kv or 0), 2)


def _read_catalog() -> dict:
    if not CATALOG.exists():
        return {"models": {}}
    try:
        return json.loads(CATALOG.read_text())
    except json.JSONDecodeError:
        return {"models": {}}


def kv_bytes_per_token(cfg: dict) -> int | None:
    """Full-attention KV per token. Hybrid models pay it on only some layers."""
    layers = cfg.get("num_hidden_layers")
    kvh = cfg.get("num_key_value_heads")
    if not (layers and kvh):
        return None
    hd = cfg.get("head_dim")
    if not hd:
        hidden, heads = cfg.get("hidden_size"), cfg.get("num_attention_heads")
        if not (hidden and heads):
            return None
        hd = hidden // heads
    interval = cfg.get("full_attention_interval")
    full = layers // interval if interval else layers
    return 2 * kvh * hd * 2 * full


def _chat_template(snap: str) -> str | None:
    j = glob.glob(snap + "chat_template.jinja")
    if j:
        try:
            return Path(j[0]).read_text()
        except OSError:
            return None
    tc = glob.glob(snap + "tokenizer_config.json")
    if tc:
        try:
            return json.loads(Path(tc[0]).read_text()).get("chat_template")
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _infer_parser(tpl: str | None) -> str | None:
    if not isinstance(tpl, str):
        return None
    try:
        from mlx_lm.tokenizer_utils import _infer_tool_parser
    except Exception:
        return None
    return _infer_tool_parser(tpl)


def scan_cache() -> dict[str, Model]:
    """Models present in the local HF cache, with facts from their own files."""
    out: dict[str, Model] = {}
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return out
    try:
        info = scan_cache_dir()
    except Exception:
        return out

    for repo in info.repos:
        if repo.repo_type != "model":
            continue
        snaps = sorted(glob.glob(str(repo.repo_path) + "/snapshots/*/"))
        if not snaps:
            continue
        snap = snaps[-1]
        m = Model(repo_id=repo.repo_id, cached=True, disk_gb=round(repo.size_on_disk / 1e9, 1))
        cfgs = glob.glob(snap + "config.json")
        if cfgs:
            try:
                cfg = json.loads(Path(cfgs[0]).read_text())
            except (OSError, json.JSONDecodeError):
                cfg = {}
            native = cfg.get("max_position_embeddings")
            m.context_native = native
            m.context = native
            m.arch = (cfg.get("architectures") or [None])[0]
            m.kv_bytes_per_token = kv_bytes_per_token(cfg)
            if not native:
                # no context window: not an mlx-lm text model (vision tower,
                # ASR, diffusion, ...). Leave runtime for the catalog to say.
                m.runtime = "other"
        else:
            m.runtime = "other"
        m.tool_parser = _infer_parser(_chat_template(snap))
        out[repo.repo_id] = m
    return out


def load(include_uncached: bool = True) -> list[Model]:
    """The merged view: curated catalog + what is on disk."""
    cat = _read_catalog().get("models", {})
    cached = scan_cache()
    merged: dict[str, Model] = {}

    for repo_id, meta in cat.items():
        m = cached.get(repo_id) or Model(repo_id=repo_id)
        m.label = meta.get("label") or m.label or repo_id.split("/")[-1]
        m.model_class = meta.get("class", m.model_class)
        # a cached model's own files win on runtime only when the catalog is silent
        m.runtime = meta.get("runtime") or (m.runtime if m.runtime != "other" else "other")
        m.role = meta.get("role", "")
        m.params = meta.get("params", "")
        m.size_gb = meta.get("size_gb")
        m.output = meta.get("output", m.output)
        m.notes = meta.get("notes", "")
        m.profiles = meta.get("profiles", [])
        merged[repo_id] = m

    for repo_id, m in cached.items():
        if repo_id not in merged:
            m.label = m.label or repo_id.split("/")[-1]
            if m.runtime == "mlx-lm" or m.context_native:
                m.model_class = "text"
            merged[repo_id] = m

    models = list(merged.values())
    if not include_uncached:
        models = [m for m in models if m.cached]
    models.sort(key=lambda m: (not m.cached, m.model_class, -(m.disk_gb or m.size_gb or 0)))
    return models


def servable() -> list[Model]:
    return [m for m in load(include_uncached=False) if m.servable]


def find(query: str) -> Model | None:
    """Resolve a repo id, or a unique case-insensitive substring of one."""
    models = load()
    for m in models:
        if m.repo_id == query:
            return m
    q = query.lower()
    hits = [m for m in models if q in m.repo_id.lower()]
    if len(hits) == 1:
        return hits[0]
    return None
