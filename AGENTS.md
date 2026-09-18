# AGENTS.md

Instructions for coding agents working in this repository. Written to the
[agents.md](https://agents.md) convention — tool-agnostic, not specific to any
one assistant.

## What this repo is

Self-contained local LLM inference stacks, organised **by hardware target**.
Each target directory is independent and exposes an OpenAI-compatible API on
`:8000/v1`.

```
llmctl/            the toolkit — host detection, catalog, servers, cache
llmctl.sh          launcher (resolves the conda env interpreter)
opencode/plugins/  the opencode model-discovery plugin
apple-m-series/    Apple silicon profiles — native MLX (NOT Docker)
  common/          environment.yml, serve.sh (foreground), hf-token.sh
  m2-16gb/         MacBook Air M2, 16 GB     — notes, models.txt, shims
  m5-128gb/        MacBook Pro M5 Max, 128 GB — notes, model docs, shims
rtx-3080-16gb/     NVIDIA Ampere 16 GB    — Docker Compose + vLLM
rtx-pro-6000/      NVIDIA Blackwell 96 GB — Docker Compose + vLLM
                   (Helm/k8s deployment lives in gagandaroach/dlan, not here)
agents/skills/     setup and operations skills (see below)
```

**Use `llmctl` rather than driving mlx_lm.server by hand.** It owns port
allocation, the server registry, and the opencode manifest; a server started
outside it is not auto-discovered. The per-target `start.sh`/`stop.sh`/etc are
shims that call it.

## Hard rules

1. **Never run Apple-silicon inference in Docker.** Docker Desktop on macOS has
   no Metal passthrough, so a container is CPU-only and 5–10x slower. The
   `rtx-*` Compose pattern does not transfer to `apple-m-series/`.
2. **Verify model facts against the source, not memory.** Model names, sizes,
   context windows and flags change fast. Sizes come from the HuggingFace API
   (`/api/models/<repo>?blobs=true`, sum `siblings[].size`); parameter counts
   and context windows come from the model's own `config.json`; CLI flags come
   from the installed package's argparse. Several plausible-sounding flags do
   not exist — `--max-kv-size` is one.
3. **Tool-call support is a hard gate, not a nice-to-have.** A model is only
   usable agentically if mlx-lm has a tool parser matching its chat template.
   Run `apple-m-series/m5-128gb/check-tool-parser.py <repo-id>` **before**
   downloading. `gpt-oss-120b` has no parser and fails *silently* — no error,
   `finish_reason=stop`, raw syntax in message content.
4. **Never commit secrets.** The HuggingFace token lives at
   `~/.dekho/hugging-face-token.txt` and is resolved at runtime by
   `apple-m-series/common/hf-token.sh`. Never echo its value, never inline it.
5. **Memory ceiling is the GPU working set, not total RAM.** On the M5 Max that
   is 115.4 GB of 137.4 GB. `mlx-lm` raises the wired limit itself; do not tell
   users to run `sudo sysctl iogpu.wired_limit_mb`.
6. **Reuse what is already installed.** mlx-lm brings typer, rich,
   huggingface_hub and mlx; llmctl is built only on those, and adding a
   dependency needs a real reason. Prefer mlx-lm's own helpers over
   reimplementing (e.g. import `_infer_tool_parser`; call `scan_cache_dir`).
7. **Don't edit tracked config to change behaviour.** Pass flags to `llmctl`,
   or add an option to `llmctl/cli.py`.

## Conventions

- Port `:8000/v1` for every target, on every platform.
- Shared logic goes in `apple-m-series/common/`; target directories hold only
  defaults, `models.txt`, and a `CLAUDE-NOTES.md` of probed machine facts.
- Shell: `bash`, `set -euo pipefail`, `bash -n` clean.
- Python: standard library only for anything a user runs before the conda env
  exists; `mlx-lm`'s own helpers over reimplementation (e.g. import
  `_infer_tool_parser` rather than hardcoding a parser list).
- Git: feature branches, `type(scope): summary` commit subjects. Do not commit
  to `main` directly.

## Tests

```bash
pip install -e ".[test]" && pytest
```

Tests must not import `mlx` — CI runs on Linux. Keep GPU/weight-dependent
checks in the integration scripts instead. When you change KV arithmetic,
port allocation, the registry, or the manifest shape, add or update a test:
those are what make `llmctl` trustworthy rather than merely convenient.

## Before you claim something works

`apple-m-series/m5-128gb/` carries the verification tools. Use them; do not
assert behaviour you have not run.

```bash
./llmctl.sh host                             # ceiling and profile
./llmctl.sh models --ctx 262144              # what fits, what can tool-call
./llmctl.sh verify                           # shard integrity + tool parser
./llmctl.sh monitor                          # GPU + real Metal memory
./llmctl.sh start --offline                  # prove the offline path
python apple-m-series/m5-128gb/check-tool-parser.py <repo-id>   # before downloading
python apple-m-series/m5-128gb/smoke-test.py <repo-id>          # real tool call
```

## Skills

Task-scoped procedures live in `agents/skills/<name>/SKILL.md`.

| Skill | Use when |
|---|---|
| `setup-local-inference` | Setting up a Mac from scratch for local inference + opencode, or explaining how the running system works |

Generative models (image, VLM, speech, music, video) are **not** served through
opencode — they use `llmctl gen`, backed by a separate conda env. When adding a
runner, read the runtime's `--help` and then actually run it: the flag names
differ between packages and even between subcommands of the same package.

## Key documents

| File | Contents |
|---|---|
| `apple-m-series/README.md` | Runtime, env vars, troubleshooting |
| `apple-m-series/m5-128gb/MODEL-COMPARISON.md` | Model quality, sizes, benchmarks with sources |
| `apple-m-series/m5-128gb/CONTEXT.md` | Context sizing: what it costs, how to evaluate it |
| `apple-m-series/m5-128gb/GENERATIVE-MODELS.md` | Image gen, image editing, VLM, omni, STT/TTS, music, video, embeddings |
| `apple-m-series/m*/CLAUDE-NOTES.md` | Probed per-machine facts — read before re-probing |
