# MacBook Pro M5 Max 128 GB — Notes for Claude

Cached context so future sessions don't need to re-probe the machine.
Captured: 2026-09-17. Purpose of this target: **offline agentic coding via
opencode** when Claude Code is unreachable (flights, no connectivity).

---

## Target machine (probed, not assumed)

| | |
|---|---|
| Model | MacBook Pro `Mac17,7` |
| Chip | Apple M5 Max |
| CPU | 18 cores (6 "Super" + 12 performance) |
| GPU | 40 cores, Metal 4 |
| Unified memory | **128 GB** |
| Disk free | ~1.8 TB |
| macOS | 26.5.1 (build 25F80) |
| User dir | `$HOME` |
| Already installed | `opencode` 1.18.30, brew, miniforge conda, nvm node v24 |
| Not installed | ollama, llama.cpp, LM Studio, mlx-lm (this stack installs mlx-lm) |

Re-probe only if hardware may have changed:
```bash
system_profiler SPHardwareDataType SPDisplaysDataType | grep -E "Chip|Memory|Cores|Metal"
sw_vers; df -h /
```

---

## Memory budget (128 GB unified)

- `mlx-lm` calls `maybe_set_recommended_wired_limit()` on startup, so it raises
  the GPU wired limit to `max_recommended_working_set_size` itself. **No manual
  `sudo sysctl iogpu.wired_limit_mb` needed.**
- Probed on this machine (`mx.device_info()`):
  - `memory_size` = 137.4 GB
  - `max_recommended_working_set_size` = **115.4 GB**  ← the real ceiling
  - `architecture` = `applegpu_g17s`
- So **plan for ~115 GB of weights + KV cache**, not 128 GB.
- Comfortable target: weights ≤ 65 GB, leaving room for prompt cache and apps.

MoE models are the right shape here: total params set the memory cost, but only
the *active* params set the speed. A 80B-A3B model costs 45 GB of RAM and
generates at roughly 3B-model speed.

---

## Runtime choice: mlx-lm, served headless

Same reasoning as the M2 target — **Docker cannot reach Metal on macOS**, so the
Docker Compose pattern from `rtx-*` does not transfer. Native `mlx_lm.server`
is used instead, exposing the repo-standard `:8000/v1`.

Two properties of `mlx_lm.server` (verified in 0.31.3 source) that shape this stack:

1. **It hot-swaps models per request.** `ModelProvider.load()` keys on the
   resolved model path and reloads when the requested path differs, so the
   `model` field in each request selects the model. `--model` only picks what
   gets preloaded at boot. One server therefore backs every entry in
   opencode's `/models` picker — no per-model server, no port juggling.
2. **It parses tool calls properly — but only for templates it has a parser for.**
   `mlx_lm/tool_parsers/` contains exactly 11: `qwen3_coder`, `glm47`,
   `mistral`, `minimax_m2`, `kimi_k2`, `gemma4`, `function_gemma`, `pythonic`,
   `json_tools`, `longcat`. `_infer_tool_parser()` string-matches the template
   to pick one. **No harmony/gpt-oss parser exists**, and the failure is silent
   — raw syntax in `content`, `finish_reason=stop`, no error. Always run
   `check-tool-parser.py` before downloading a new model. `ToolCallFormatter` + `tokenizer.tool_parser`
   turn the model's raw tool syntax into OpenAI `tool_calls`, and it rejects
   `tools` outright for models whose template lacks tool support. This is what
   makes agentic opencode use work rather than dumping raw XML into the chat.

### Why the small model is the boot default

`MODEL` in `serve.sh` is `Qwen3-Coder-30B-A3B-Instruct-4bit`, not the daily
driver. Because the server hot-swaps per request, preloading the 45 GB model
only delays `./serve.sh` becoming ready — selecting Qwen3-Coder-Next in
opencode loads it on demand anyway. Preload small, pick big.

### Flags that matter (real flags in 0.31.3 — verified against `server.py`)

| Flag | Why |
|---|---|
| `--max-tokens` | Default is **512**, which silently truncates agent turns. Set to 32768. |
| `--prompt-cache-bytes` | Caps the reusable KV prompt-cache pool. Accepts `24G`. Prefix reuse across agent turns is the biggest latency win. |
| `--prompt-cache-size` | Number of distinct cached conversations to retain (default 10). |
| `--kv-bits` | KV quantization (4/8). **Disables batching — requests serialize.** Off by default; only for extreme contexts. |
| `--decode-concurrency` / `--prompt-concurrency` | Batching widths; defaults (32/8) are fine single-user. |

There is **no** `--max-kv-size` flag. Use `--prompt-cache-bytes`.

---

## Model set (sizes verified against the HF API, 2026-09-17)

> For response-quality comparisons with published benchmark numbers and sources,
> see [`MODEL-COMPARISON.md`](MODEL-COMPARISON.md) in this directory.

All are `mlx-community` MLX-format repos and all architectures are supported by
`mlx-lm` 0.31.3 (`qwen3_next`, `qwen3_moe`, `glm4_moe`, `glm4_moe_lite`,
`gpt_oss`, `mistral3`).

### The kit

| Model | Size | Arch / shape | Role |
|---|---|---|---|
| `Qwen3-Coder-Next-4bit` | **44.9 GB** | Qwen3Next, 80B total / ~3B active, hybrid (linear + full) attention | **Daily driver.** Purpose-built agentic coder. Hybrid attention keeps long agent contexts cheap. |
| `Qwen3-Coder-30B-A3B-Instruct-4bit` | **17.2 GB** | Qwen3MoE, 30B / 3B active | Fast, battery-friendly, loads in seconds. Same job, less quality. **Also the boot preload default** — see below. |
| `gpt-oss-120b-MXFP4-Q8` | **63.4 GB** | GptOss MoE, ~5B active | ⚠️ **Chat only.** mlx-lm 0.31.3 has no harmony tool parser, so it cannot tool-call — verified by smoke test, which it fails at step 3. Use `GLM-4.5-Air-4bit` for this slot instead. |

### Worth having

| Model | Size | Note |
|---|---|---|
| `GLM-4.7-Flash-4bit` | 16.9 GB | `Glm4MoeLite`. Newest small MoE in the set. |
| `Devstral-Small-2-24B-Instruct-2512-4bit` | 15.1 GB | Dense agentic coder from Mistral; predictable, no MoE routing variance. |
| `GLM-4.5-Air-4bit` | 60.2 GB | 106B A12B. More active params than gpt-oss-120b → slower but sturdier. |
| `Qwen3.8-27B-4bit` | 16.1 GB | Dense generalist, not coder-tuned. |
| `Qwen3-VL-4B-Instruct-4bit` | ~3 GB | Vision — reading screenshots/diagrams offline. |

### Deliberately excluded

| Model | Size | Why not |
|---|---|---|
| `MiniMax-M2.1-4bit` / `M2.7-4bit` | 128.7 GB | Exceeds total RAM, over the 115.4 GB working set. |
| `Qwen3.8-Flash-Next-4bit` | 111.5 GB | Fits the 115.4 GB ceiling on paper, but leaves nothing for KV cache — and `Qwen4ExpForConditionalGeneration` is not in `mlx-lm` 0.31.3's model list anyway. |
| `GLM-4.7-4bit` | 198.6 GB | Way over. (`GLM-4.7-Flash` is the usable sibling.) |
| `GLM-5.2-mxfp4` | 395.1 GB | Way over. |
| `Qwen3-Coder-480B-A35B-Instruct-4bit` | 270.1 GB | Way over. |
| `Kimi-K2.5` | 657.6 GB | Way over. |

Rule for re-evaluating later: **4-bit MLX weights ≈ 0.56 GB per billion total
params.** Anything above ~150B total params is out at 4-bit on this machine.

---

## opencode wiring

`opencode.json` in this directory defines an `mlx` provider via
`@ai-sdk/openai-compatible` pointed at `http://127.0.0.1:8000/v1`. Model keys
must be the **exact HF repo ids**, because that string is what the server
resolves and loads.

`limit.context` / `limit.output` are what let opencode track remaining context,
so they are set per model rather than left to models.dev (which has no entry for
local repos).

Install it globally:
```bash
cp opencode.json ~/.config/opencode/opencode.json
```

---

## Measured on this machine (2026-09-17)

`Qwen3-Coder-30B-A3B-Instruct-4bit`, server warm, weights already resident:

| | |
|---|---|
| Decode (fresh server, 30B) | **95–114 tok/s** |
| Decode (after model swaps) | **36–38 tok/s** — ~3x degradation, restart to recover |
| Decode (Qwen3-Coder-Next, warm) | ~24 tok/s |
| Cold model load | ~3 s to first token after server start |
| Tool call latency | 0.8 s to emit a `tool_calls` object |
| Tool-result turn | 0.4 s |

`smoke-test.py` passed all four checks **with `HF_HUB_OFFLINE=1`**, which is the
real proof the flight case works: server boots, tool calls parse natively, and
the tool-result round trip completes with no network.

**Hot-swapping costs throughput.** Alternating models in one server drops the
30B from 95–114 tok/s to 36–38. `vmmap` shows current footprint 27.2 GB against
a 62.1 GB peak — retained/fragmented Metal buffers. Restart `./serve.sh` with
the model you intend to use for a long session. `./monitor.sh` surfaces the
gap between current and peak footprint.

**RSS is meaningless for MLX.** Weights live in Metal buffers that macOS does
not count in RSS: the server reported 16.5 GB RSS while `vmmap` showed
27.2 GB of `IOAccelerator (graphics)`. Use `./monitor.sh` (reads `ioreg` for
GPU utilization and `vmmap` for buffer residency), not `htop`/`ps`.

Download throughput from the hub varies a lot with the xet backend — observed
2 MB/s during chunk verification phases and 70 MB/s steady-state on the same
session. A slow window is not a stall; check for `.incomplete` files growing
rather than trusting a short `du` sample.

---

## HuggingFace token

Read token lives at `~/.dekho/hugging-face-token.txt` (37 chars). Resolved by
`common/hf-token.sh`, which both `serve.sh` and `pull.sh` source. Order:
`$HF_TOKEN` → `$HF_TOKEN_FILE` → the `~/.dekho` path. The models are all public,
but unauthenticated hub requests are rate-limited and slower.

---

## Offline checklist

Models must be in `~/.cache/huggingface` **before** losing connectivity —
`mlx_lm.server` resolves repo ids through the hub and a cold repo id fails
without network. Run `./pull.sh` while online, then verify:

```bash
conda run -n dekho-apple-local-llm hf cache scan
HF_HUB_OFFLINE=1 ./serve.sh     # proves it boots with no network
```

---

## User profile notes

- Prefers terse answers with tradeoffs over long explanations.
- Repo layout is hardware-target-first; `apple-m-series/` is split per machine
  (`m2-16gb/`, `m5-128gb/`) over shared `common/` bones.
