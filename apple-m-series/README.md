# apple-m-series

Local inference on Apple silicon via **MLX**, exposing the repo-standard
OpenAI-compatible endpoint at `:8000/v1`.

Unlike the `rtx-*` targets, these are **not** Docker Compose stacks: Docker
Desktop on macOS runs a Linux VM with no Metal passthrough, so a containerized
runtime would be CPU-only and 5–10× slower. Everything here runs natively.

```
common/          shared bones — conda env, serve.sh, pull.sh, verify.sh,
                 manifest writer, context calculator, opencode plugin
m2-16gb/         MacBook Air M2, 16 GB   → small models, ~8B @ 4-bit ceiling
m5-128gb/        MacBook Pro M5 Max, 128 GB → agentic coding kit for opencode
```

Each target directory carries only what differs: default model, memory limits,
its `models.txt`, and a `CLAUDE-NOTES.md` with the probed machine facts.

**Choosing a model offline:** see
[`m5-128gb/MODEL-COMPARISON.md`](m5-128gb/MODEL-COMPARISON.md) — size, weights,
quantization, mode, and published benchmark scores with sources, captured while
online so it's readable on a plane.

**Context sizing:** [`m5-128gb/CONTEXT.md`](m5-128gb/CONTEXT.md) — what a context
window actually costs, measured, and how to evaluate one against this machine's
115.4 GB ceiling. Counterintuitive headline: the 80B model is **4x cheaper per
token of context** than the 30B, because only 12 of its 48 layers hold a KV
cache.

**Setting up a new machine:** the
[`setup-local-inference`](../agents/skills/setup-local-inference/SKILL.md) skill,
plus [`AGENTS.md`](../AGENTS.md) at the repo root.

---

## Quick start

```bash
cd m5-128gb          # or m2-16gb
./pull.sh            # download models in models.txt (do this while online)
./verify.sh          # confirm the weights are intact
./start.sh           # background, returns when the endpoint answers
./stop.sh            # when you're done
```

| Script | |
|---|---|
| `./start.sh` | Detaches the server, records a pidfile, tees a log, and **blocks until `/v1/models` answers** — so a failed load surfaces here, not later in opencode. Idempotent; refuses if another process holds the port. |
| `./stop.sh` | SIGTERM, escalates to SIGKILL, also catches foreground `./serve.sh` instances, then confirms the port is free. Idempotent. |
| `./serve.sh` | Foreground equivalent. Use it when you want the log in front of you. |
| `./monitor.sh` | GPU utilization and real memory (`-w` to watch). |

pidfile and log live in `~/.cache/dekho-local-inference/` (`DEKHO_RUN_DIR` to
move them).

First run creates the shared conda env `dekho-apple-local-llm` from
`common/environment.yml`.

Override any default without editing tracked files:

```bash
cp .env.example .env     # then edit
# or, one-off:
MODEL=mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit ./serve.sh
```

---

## One server, every model

`mlx_lm.server` loads whichever repo id arrives in a request's `model` field and
reloads on change, so a single server on `:8000` backs every model you've pulled.
`--model` (i.e. `MODEL`) only chooses what gets preloaded at boot.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Expect a pause on the first request for a model while its weights load.

---

## Using it from opencode

Install the discovery plugin once:

```bash
./common/install-opencode.sh
```

After that, whatever you launch with `./serve.sh` appears in opencode's
`/models` picker under **Dekho Local Inference (MLX)** — no config editing per
model.

```bash
opencode models dekho-local-inference    # verify without entering the TUI
opencode                                 # then press /models
```

### How discovery works

opencode needs two things per model that models.dev cannot supply for a local
repo id: the context window (it drives compaction) and whether the model can
tool-call at all. So the plugin merges two sources:

| Source | Supplies |
|---|---|
| `GET <endpoint>/v1/models` | what the server can reach **right now** |
| `~/.cache/dekho-local-inference/manifest.json`, written by `serve.sh` | context window, generation ceiling, and the tool parser — resolved with mlx-lm's own `_infer_tool_parser`, not guessed |

Models that are cached but not currently served still appear, tagged
`[offline]`; chat-only models are tagged `[no tools]`. A model hand-pinned in
your own `opencode.json` always wins over discovery. If the server is down the
picker still lists the manifest's models; if both sources are missing the
plugin registers nothing rather than guessing, and never throws — a throwing
plugin blocks opencode from starting.

`m5-128gb/opencode.fallback.json` is the static alternative if you would rather
not use a plugin.

---

## Verifying a model is agent-ready

Tool calling is the thing that decides whether opencode can actually drive a
model, and it depends on the model's chat template matching one of mlx-lm's
tool parsers (`mlx_lm/tool_parsers/`). Not every model has one — **gpt-oss-120b
does not**, so it cannot be driven agentically here at all.

Check *before* spending a download:

```bash
conda run -n dekho-apple-local-llm python m5-128gb/check-tool-parser.py
conda run -n dekho-apple-local-llm python m5-128gb/check-tool-parser.py <repo-id>
```

It fetches only the chat template (a few KB) and runs mlx-lm's own
`_infer_tool_parser` against it. `parser=None` means unusable for agentic work.
 `m5-128gb/smoke-test.py` checks the four things that matter: `/models`, a
plain completion, a **native `tool_calls` object**, and the tool-result round
trip an agent loop repeats.

```bash
cd m5-128gb
./serve.sh                       # in one shell
# in another (stdlib only, but needs a python — reuse the serve env):
conda run -n dekho-apple-local-llm python smoke-test.py
conda run -n dekho-apple-local-llm python smoke-test.py mlx-community/GLM-4.7-Flash-4bit
```

A model that fails step 3 is fine for chat but unusable as an opencode agent.

---

## Seeing what is actually running

`htop` and `ps` cannot show MLX memory: weights live in Metal buffers that
macOS does not count in RSS. The server reports ~16 GB RSS while holding 27 GB
of buffers.

```bash
./monitor.sh          # one snapshot
./monitor.sh -w       # refresh every 2s
```

It reads GPU utilization and in-use memory from `ioreg` (no sudo), Metal buffer
residency and footprint peak from `vmmap`, the servable model list from the
running server, and swap activity from `vm_stat`. If `pageouts` climbs you are
over the GPU working-set ceiling; if `footprint peak` sits far above current
`Metal buffers`, the server has been hot-swapping models and a restart will
recover throughput (measured ~3x on the 30B).

---

## Environment variables

Read by `common/serve.sh`, overridable per target via `.env`:

| Var | Default | Notes |
|---|---|---|
| `MODEL` | per target | Model preloaded at boot. Small by default — the server hot-swaps, so preloading the big one only slows startup |
| `PORT` | `8000` | Repo convention |
| `HOST` | `127.0.0.1` | Loopback only |
| `MAX_TOKENS` | `32768` (m5) / `8192` (m2) | mlx-lm's own default is **512**, which truncates agent turns |
| `MAX_CONTEXT` | `262144` (m5) / `16384` (m2) | Context **advertised to opencode**, clamped per model to its trained window. There is no server-side context flag — see [`m5-128gb/CONTEXT.md`](m5-128gb/CONTEXT.md) |
| `PROMPT_CACHE_BYTES` | `24G` (m5) / `2G` (m2) | Caps the reusable prompt-cache pool; accepts `24G` |
| `PROMPT_CACHE_SIZE` | `10` | Distinct cached conversations retained |
| `KV_BITS` | unset | KV quantization (4/8). **Disables batching** — off by default |
| `EXTRA_ARGS` | unset | Passed straight through to `mlx_lm.server` |

---

## HuggingFace token

`common/hf-token.sh` is sourced by both `serve.sh` and `pull.sh`. It resolves a
token in this order and never echoes the value:

1. an existing `$HF_TOKEN`
2. `$HF_TOKEN_FILE`
3. `~/.dekho/hugging-face-token.txt`

Every model in these targets is public, so a token is not strictly required —
but unauthenticated hub requests are rate-limited and noticeably slower, which
matters when you're pulling 125 GB.

---

## Staying usable offline

Weights are resolved through the HuggingFace hub, so pull everything **before**
you lose connectivity, then verify it — a truncated shard should be a download
away, not a dead end at 30,000 feet.

```bash
./pull.sh                        # download models.txt
./verify.sh                      # shard integrity + tool parser per model
HF_HUB_OFFLINE=1 ./serve.sh      # prove it boots with no network
conda run -n dekho-apple-local-llm python smoke-test.py   # prove tools work
```

`verify.sh` compares each model's tensor bytes against its
`model.safetensors.index.json` (accounting for safetensors headers), reports
orphaned `.incomplete` blobs, and names the mlx-lm tool parser it resolves to.

---

## Troubleshooting

**Model not found / connection error while offline**
The repo id was never cached. `conda run -n dekho-apple-local-llm hf cache scan`
to see what you actually have.

**opencode's tool calls don't work against a model**
Two distinct failures. If the model's chat template declares no tool support,
mlx-lm rejects the request outright — *"Received tools but model does not
support tool calling"*; pick a model marked `"tool_call": true` in
`opencode.json`. If the template *does* advertise tools but mlx-lm has no
matching parser (gpt-oss's harmony format), you get something worse: no error,
`finish_reason=stop`, and the raw tool syntax leaking into message content.
`check-tool-parser.py` catches this case up front.
If instead the server logs *"Failed to parse tool call … likely truncated
mid-generation"*, the call was cut off by the token ceiling — raise `MAX_TOKENS`.
`smoke-test.py` distinguishes the two.

**Machine starts swapping**
`mlx-lm` raises the GPU wired limit to the GPU's recommended working set
automatically — 115.4 GB on the M5 Max, not the full 128 GB. Weights plus prompt
cache must fit *that*, not total RAM. Lower `PROMPT_CACHE_BYTES` or pick a
smaller model.

**Unsupported architecture on load**
`mlx-lm` must know the arch. Check the model's `config.json` `architectures`
against `mlx_lm/models/`; bump `mlx-lm` in `common/environment.yml` if it's newer.
