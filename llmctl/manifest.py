"""The manifest the opencode plugin reads.

Rewritten on every start/stop so it can never describe a server that is gone.
Written atomically so the plugin never reads a partial file.

Shape (v2 — supports several concurrent servers):

  { "version": 2,
    "servers": [ {"port":8000,"endpoint":...,"model":...,"context":...} ],
    "models":  { "<repo id>": {label, context, output, tool_call, ...} } }

Each server becomes its own opencode provider, because a provider has exactly
one baseURL. The lowest-numbered port keeps the stable id
"dekho-local-inference" so an existing model selection keeps working.
"""

import json
from datetime import UTC, datetime

from . import catalog, servers
from .paths import MANIFEST, ensure_dirs

PRIMARY_ID = "dekho-local-inference"


def build() -> dict:
    # Only LLM servers belong here. The generative server (`llmctl gen serve`)
    # speaks the same OpenAI paths but its models are TTS/STT/embedding —
    # useless to opencode, and it would otherwise claim the primary provider id
    # whenever it held the lower port.
    live = sorted(
        (s for s in servers.list_servers() if not s.is_gen), key=lambda s: s.port
    )
    models = {}

    for m in catalog.load(include_uncached=False):
        if m.runtime != "mlx-lm" or not m.context_native:
            continue
        models[m.repo_id] = {
            "label": m.label or m.repo_id.split("/")[-1],
            "context": m.context or m.context_native,
            "context_native": m.context_native,
            "output": m.output,
            "tool_call": bool(m.tool_parser),
            "tool_parser": m.tool_parser,
            "arch": m.arch,
            "disk_gb": m.disk_gb,
            "kv_bytes_per_token": m.kv_bytes_per_token,
            "notes": m.notes or None,
        }

    server_entries = []
    for i, s in enumerate(live):
        # Clamp the advertised window to each model's own trained window.
        ctx = s.context
        server_entries.append(
            {
                "provider_id": PRIMARY_ID if i == 0 else f"{PRIMARY_ID}-{s.port}",
                "provider_name": (
                    "Dekho Local Inference (MLX)" if i == 0 else f"Dekho Local :{s.port}"
                ),
                "endpoint": s.endpoint,
                "port": s.port,
                "pid": s.pid,
                "model": s.model,
                "context": ctx,
                "profile": s.profile,
                "started_at": s.started_at,
            }
        )

    return {
        "version": 2,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "servers": server_entries,
        "models": dict(sorted(models.items())),
        # kept for the v1 plugin so an un-upgraded install still works
        "provider_id": PRIMARY_ID,
        "provider_name": "Dekho Local Inference (MLX)",
        "endpoint": server_entries[0]["endpoint"] if server_entries else None,
        "preloaded": server_entries[0]["model"] if server_entries else None,
        "max_context_cap": server_entries[0]["context"] if server_entries else None,
    }


def write() -> dict:
    ensure_dirs()
    data = build()
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(MANIFEST)
    return data
