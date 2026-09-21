# AGENTS.md

Instructions for coding agents in this repo ([agents.md](https://agents.md)).

## Entry point

Use `llmctl`, not `mlx_lm.server` directly — it owns port allocation, the
server registry and the opencode manifest. A server started outside it is not
discovered.

```bash
./llmctl.sh host          # ceiling and profile
./llmctl.sh models        # what fits, what can tool-call
./llmctl.sh verify        # shard integrity
./llmctl.sh validate      # load and run models, write markdown
./llmctl.sh monitor       # GPU and real memory
```

## Rules

1. **Never run Apple-silicon inference in Docker.** No Metal passthrough on
   macOS; a container is CPU-only. The `rtx-*` Compose pattern does not
   transfer.
2. **Verify model facts against the source, not memory.** Sizes from the HF
   API (`?blobs=true`, sum `siblings[].size`); parameter counts and context
   from the model's `config.json`; CLI flags from the installed package's
   argparse. `--max-kv-size` does not exist. `hf download` can exit 0 on an
   incomplete download.
3. **Tool-call support is a hard gate and fails silently.** Run
   `apple-m-series/m5-128gb/check-tool-parser.py <repo-id>` before downloading.
4. **Never commit secrets.** The HF token is resolved at runtime by
   `apple-m-series/common/hf-token.sh` from `~/.dekho/`. Never echo it.
5. **The ceiling is the GPU working set, not RAM** — 115.4 GB of 137.4 GB
   here, and plan against ~100 GB. mlx-lm raises the wired limit itself; do
   not suggest `sudo sysctl`.
6. **Reuse what is installed.** mlx-lm brings typer, rich, huggingface_hub and
   mlx; llmctl uses only those. Prefer mlx-lm's own helpers (import
   `_infer_tool_parser`, call `scan_cache_dir`) over reimplementing.
7. **Generative models do not go through opencode.** It speaks
   `/v1/chat/completions` with tool calls — wrong shape for diffusion and
   audio. They use `llmctl gen`, in a separate conda env.
8. **Read a runtime's `--help`, then run it.** Flag names differ between
   packages and between subcommands of the same package, and several ship one
   console script per model family.

## Conventions

- Port `:8000/v1` on every target.
- Shared logic in `apple-m-series/common/`; target dirs hold defaults and
  `models.txt`.
- Shell: `bash`, `set -euo pipefail`, shellcheck-clean at warning level.
- Python: stdlib only for anything run before the conda env exists.
- Git: feature branches, `type(scope): summary`. Tests must not import `mlx`.

## Verified machine facts

Cached so sessions need not re-probe. Re-derive with `llmctl host`.

| | m5-128gb | m2-16gb |
|---|---|---|
| Chip | Apple M5 Max, 18 CPU / 40 GPU | Apple M2, 8 CPU / 10 GPU |
| Memory | 128 GB (115.4 GB working set) | 16 GB |
| Practical model budget | ~100 GB | ~10 GB |
| Default model | `Qwen3-Coder-Next-4bit` preloads small; server hot-swaps | `Qwen2.5-3B-Instruct-4bit` |

Numbers and limits: [docs/hardware.md](docs/hardware.md). Model status:
[docs/models.md](docs/models.md).

## Tests

```bash
pip install -e ".[test]" && pytest
```

When changing KV arithmetic, port allocation, the registry or the manifest
shape, add or update a test — those are what make `llmctl` trustworthy.

## Skills

`agents/skills/setup-local-inference/` — setting up a machine from scratch.
