---
name: setup-local-inference
description: Set up a Mac for local LLM inference with MLX and opencode, end to end — detect the machine, create the env, pull and verify models, install the opencode discovery plugin, and explain how the pieces fit. Also use to explain or troubleshoot an existing setup, or to add a model. Covers Apple silicon only; the rtx-* targets are Docker Compose and out of scope.
---

# setup-local-inference

Gets a Mac from nothing to "opencode drives a local model with no network", and
explains the system well enough that the user can operate it alone.

## When to invoke

- "set up local inference", "set up local models", "get opencode working offline"
- Preparing a machine for offline work (a flight, no connectivity)
- Adding a model to an existing setup
- Explaining or debugging a setup that already exists

Apple silicon only. For NVIDIA, use the `rtx-*` Compose stacks instead.

## Ground rules

Read `AGENTS.md` first — the hard rules there apply, especially: never Docker on
macOS, verify model facts against the source, and tool-call support is a gate.

Two failure modes make this worth doing carefully:

1. **A model that cannot tool-call is useless to opencode**, and the failure is
   silent. Always gate on `check-tool-parser.py`.
2. **Weights must be cached before connectivity is lost.** A repo id resolves
   through the HuggingFace hub; a cold id fails offline no matter how much disk
   is free.

---

## Step 1 — Probe, don't assume

```bash
system_profiler SPHardwareDataType | grep -E "Chip|Memory|Model Name"
sw_vers; df -h /
command -v opencode conda brew
```

Then check `apple-m-series/m*/CLAUDE-NOTES.md` — if a target already describes
this machine, use it and skip re-probing.

Pick the target by unified memory:

| Memory | Target | Ceiling |
|---|---|---|
| ≤ 24 GB | `m2-16gb` | ~8B @ 4-bit |
| ≥ 64 GB | `m5-128gb` | see below |

Neither fits? Copy the closest target directory, adjust its defaults and
`models.txt`, and write a new `CLAUDE-NOTES.md` with what you probed.

The real memory ceiling is the **GPU working set**, not total RAM:

```bash
conda run -n dekho-apple-local-llm python -c \
  "import mlx.core as mx; d=mx.device_info(); \
   print(d['max_recommended_working_set_size']/1e9, 'GB of', d['memory_size']/1e9)"
```

On the M5 Max that is 115.4 of 137.4 GB. `mlx-lm` raises the wired limit itself
— never instruct the user to run `sudo sysctl iogpu.wired_limit_mb`.

## Step 2 — Environment

```bash
cd apple-m-series
conda env create -f common/environment.yml   # creates dekho-apple-local-llm
```

`serve.sh` and `pull.sh` create it automatically on first run, so this is only
worth doing up front to surface install errors early.

Needs `mlx-lm >= 0.31.3` for the current architectures (`qwen3_next`,
`glm4_moe_lite`, `gpt_oss`, `qwen3_5`, `gemma4`). If a model fails to load with
an unknown-architecture error, check its `config.json` `architectures` against
the installed `mlx_lm/models/` and bump the pin.

## Step 3 — Choose models, and gate on tool calling

Do this **before** downloading — it costs a few KB instead of tens of GB:

```bash
cd m5-128gb
conda run -n dekho-apple-local-llm python check-tool-parser.py            # shortlist
conda run -n dekho-apple-local-llm python check-tool-parser.py <repo-id>  # a candidate
```

`parser=None` means opencode cannot drive it. Known result: **gpt-oss-120b has
no parser** — mlx-lm ships none for its harmony format — so it is chat-only and
is commented out of `models.txt`.

Recommend from `MODEL-COMPARISON.md`, which has sizes, published benchmarks and
sources. Default kit:

| Model | Size | Role |
|---|---|---|
| `Qwen3-Coder-Next-4bit` | 44.9 GB | Daily driver. 80B/3B active, 256k ctx, best tool-call robustness |
| `Qwen3-Coder-30B-A3B-Instruct-4bit` | 17.2 GB | Fast, battery, ~103 tok/s |

Do not recommend on benchmark score alone. Check `CONTEXT.md` too — at long
context the 80B is **4x cheaper per token** than the 30B, because only 12 of its
48 layers hold a KV cache.

## Step 4 — Pull and verify

```bash
./pull.sh              # models.txt, resumable; token auto-resolved
./verify.sh            # shard integrity + tool parser per model
```

Tell the user the download is the long pole and that throughput swings widely
(2–70 MB/s observed on one session with the xet backend). A slow window is not
a stall — check that `.incomplete` files are growing rather than trusting a
short `du` sample.

`verify.sh` compares tensor bytes against each `model.safetensors.index.json`,
**accounting for safetensors headers** — comparing raw file size to the index's
`total_size` gives a false mismatch.

## Step 5 — Serve

```bash
./start.sh                                   # background, waits for readiness
MODEL=<repo-id> MAX_CONTEXT=131072 ./start.sh
./stop.sh
./serve.sh                                   # foreground alternative
```

Prefer `start.sh` for anything the user will actually work in: it detaches, so
closing the terminal does not kill the server, and it blocks until
`/v1/models` answers so a failed weight load is reported immediately instead of
surfacing as a confusing opencode error.

Explain what the user is looking at:

- One server backs **every** cached model. `mlx_lm.server` loads whichever repo
  id arrives in the request's `model` field and reloads on change, so `MODEL`
  only picks what is preloaded at boot. The small model is preloaded on purpose
  — preloading the 45 GB one just slows startup.
- `MAX_TOKENS` defaults to 32768 here because **mlx-lm's own default is 512**,
  which truncates agent turns mid-tool-call.
- `MAX_CONTEXT` sets the window *advertised* to opencode, clamped per model to
  its trained window. There is no server-side context flag at all — see
  `CONTEXT.md`.
- On start, `serve.sh` writes a manifest describing what is servable now, which
  is what the opencode plugin reads.

## Step 6 — Wire up opencode

```bash
../common/install-opencode.sh
```

Symlinks the discovery plugin into `~/.config/opencode/plugins/`. After that,
whatever is launched shows up in `/models` under **Dekho Local Inference (MLX)**
with no config editing.

Verify without entering the TUI:

```bash
opencode models dekho-local-inference
```

Explain the design: opencode needs each model's context window and whether it
can tool-call. models.dev has no entries for local repo ids, so the plugin
merges the **live server** (`/v1/models`, what is reachable now) with the
**manifest** (context, generation ceiling, tool parser resolved by mlx-lm
itself). Models the server isn't currently serving still appear, tagged
`[offline]`; chat-only ones are tagged `[no tools]`.

## Step 7 — Prove it, especially the offline path

```bash
conda run -n dekho-apple-local-llm python smoke-test.py <repo-id>
```

Four checks: `/models`, a plain completion, a **native `tool_calls` object**,
and the tool-result round trip an agent loop repeats. Step 3 is the one that
matters — it is what separates a usable agent model from a chat model.

Then prove the case the whole setup exists for:

```bash
pkill -f mlx_lm.server
HF_HUB_OFFLINE=1 ./serve.sh
opencode run --model dekho-local-inference/<repo-id> "make a trivial edit to a scratch file"
```

Do not report success until a real tool call has executed with
`HF_HUB_OFFLINE=1`. Everything else can pass while the actual goal fails.

---

## Adding a model later

1. `check-tool-parser.py <repo-id>` — stop here if `None`.
2. Add it to `models.txt` with its size and parser in a comment.
3. Optionally add a label to `common/models-meta.json`.
4. `./pull.sh <repo-id>` then `./verify.sh`.
5. `python ../common/context-calc.py --model <repo-id>` to confirm it fits.
6. Restart `./serve.sh` so the manifest is rewritten; it appears in `/models`.

## Troubleshooting map

| Symptom | Cause | Fix |
|---|---|---|
| Model not in `/models` | Manifest stale or server down | Restart `./serve.sh`; check `~/.cache/dekho-local-inference/manifest.json` |
| Raw `<\|channel\|>` or XML in replies | No mlx-lm tool parser for that model | `check-tool-parser.py`; pick a model with a parser |
| Agent turn truncates mid-tool-call | `MAX_TOKENS` too low | Raise it; mlx-lm's default 512 is the trap |
| "Received tools but model does not support tool calling" | Template declares no tools | Different model |
| Machine starts swapping (`vm_stat` Pageouts climbing) | Over the 115.4 GB working set | Lower `MAX_CONTEXT` / `PROMPT_CACHE_BYTES`, or smaller model |
| Fails offline, worked online | Repo id never cached | `./verify.sh`, re-`pull.sh` while online |
| Unknown architecture on load | mlx-lm too old | Bump `mlx-lm` in `common/environment.yml` |
| First request very slow | Cold weight load + prefill | Expected: ~2,300–3,700 tok/s prefill, so 100k ctx ≈ 30–45 s |

## What to tell the user at the end

- Which model to pick when, and why (point at `MODEL-COMPARISON.md`).
- That one server serves all models and switching is just the `/models` picker.
- That `./verify.sh` is the pre-flight check to run **before** losing network.
- The numbers that matter on their machine: working-set ceiling, weights + KV at
  their chosen context, and measured tok/s.
