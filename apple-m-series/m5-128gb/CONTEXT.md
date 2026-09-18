# Context size — how it works, how to deploy it, how to evaluate it

Measured on this machine (M5 Max, 128 GB) on 2026-09-17. Numbers here come from
loading the models and reading the cache objects, not from arithmetic alone.

---

## There is no "context size" flag

`mlx_lm.server` has **no context-length option**. The full flag list is
`--model --port --host --max-tokens --prompt-cache-bytes --prompt-cache-size
--kv-bits --kv-group-size --quantized-kv-start --decode-concurrency
--prompt-concurrency --prefill-step-size --draft-model --num-draft-tokens
--pipeline --temp --top-p --top-k --min-p --chat-template --chat-template-args
--adapter-path --trust-remote-code --use-default-chat-template
--allowed-origins --log-level`. Nothing sets a context window.

That is because three different things get called "context", and only one of
them is a server setting:

| What | Set by | Effect if you get it wrong |
|---|---|---|
| **Trained window** — `max_position_embeddings` in the model's `config.json` | the model, immutable | Exceed it and quality degrades silently; the server will not refuse |
| **Advertised window** — `limit.context` seen by opencode | `MAX_CONTEXT` in `serve.sh` → manifest → plugin | Too high: opencode overfills and quality falls off. Too low: it compacts earlier than needed |
| **Affordable window** — what KV cache memory you have | the model's architecture + your RAM | Too high: swapping, then a crawl |

`MAX_CONTEXT` sets the *advertised* number, clamped per model to that model's
own trained window. It is what drives opencode's compaction, which is the only
place the number actually changes behaviour.

### On 333k

Qwen3-Coder-Next's trained window is **262,144** and its config has
`rope_scaling: null`, so there is no scaling factor to stretch it. Asking for
333k would not error — mlx-lm would extrapolate RoPE past where the model was
trained and degrade in ways that are hard to notice from the outside. So the
default here is **262,144**: the real ceiling, used in full. Reaching 333k
honestly would need a YaRN/RoPE-scaled conversion of the weights, which is a
different model, not a flag.

---

## What a context window costs

KV cache is the cost, and it is **linear in tokens**:

```
bytes/token = 2 (K and V) × kv_heads × head_dim × 2 (bf16) × layers_holding_KV
```

The last term is where the models differ enormously. Measured:

| Model | Layers holding KV | B/token | KV at 262k | + weights | Total |
|---|---|---|---|---|---|
| `Qwen3-Coder-Next-4bit` | **12 of 48** (hybrid) | **24,576** | **6.44 GB** | 44.9 GB | **51.3 GB** |
| `Qwen3-Coder-30B-A3B-Instruct-4bit` | 48 of 48 | 98,304 | 25.77 GB | 17.2 GB | 43.0 GB |
| `gpt-oss-120b-MXFP4-Q8` | 36 of 36 | 73,728 | — (native 131k) | 63.4 GB | 73.1 GB @ 131k |

### The counterintuitive result

**The 80B model is four times cheaper per token of context than the 30B.**

Qwen3-Coder-Next is a hybrid: its card describes the layout as
`12 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`. Only every
4th layer is full attention; the other 36 hold a **fixed-size recurrent state**
— measured at 77.3 MB total, and constant no matter how long the context gets.
Verified by inspecting the cache objects:

```
cache layer types: {'ArraysCache': 36, 'KVCache': 12}
keys shape (1, 2, 2048, 256)  dtype bfloat16   →  2,048 B/layer/token × 12
```

So at full 262k context the 80B model needs 51.3 GB and the 30B needs 43.0 GB —
the gap between them narrows from 27.7 GB to 8.3 GB, and the 30B's KV cache
(25.8 GB) ends up larger than its own weights (17.2 GB).

**Practical consequence:** long-context agentic work is the case where
Qwen3-Coder-Next is *most* worth its size. Reach for the 30B for short tasks and
battery life, not to save memory on long ones.

### Allocation overshoot — budget 2x

mlx-lm's `KVCache` grows in steps rather than exactly, so allocation during
growth runs ahead of the steady-state figure. Measured while prefilling:

| Prefilled tokens | Steady-state KV | Actually allocated |
|---|---|---|
| 4,096 | 0.10 GB | 0.28 GB |
| 16,384 | 0.40 GB | 0.88 GB |
| 65,536 | 1.61 GB | 3.30 GB |

About **2x**. Budget for that: the 51.3 GB figure for Qwen3-Coder-Next at 262k
can transiently approach ~58 GB. Still comfortable against 115.4 GB.

### The prompt cache multiplies it

`PROMPT_CACHE_BYTES` (default `24G` here) caps a pool of *retained*
conversations — `--prompt-cache-size` of them, 10 by default. Prefix reuse
across agent turns is the single biggest latency win available, and this is
what pays for it. But each retained conversation holds its own KV, so the
budget is `min(PROMPT_CACHE_BYTES, size × per-conversation KV)`. At 24 GB and
6.44 GB per full-length conversation, that is roughly **3 full-context
conversations** retained. Lower it if you would rather spend the memory on a
bigger model.

---

## The ceiling is 115.4 GB, not 128 GB

```
mx.device_info()["max_recommended_working_set_size"] = 115.4 GB
mx.device_info()["memory_size"]                      = 137.4 GB
```

`mlx-lm` calls `maybe_set_recommended_wired_limit()` at startup and raises the
GPU wired limit to that recommended value itself, so **no `sudo sysctl
iogpu.wired_limit_mb` is needed**. Weights plus KV plus prompt cache must fit
under 115.4 GB.

### But plan against ~100 GiB, not 115.4

115.4 GB is where allocation stops succeeding; it is not where performance
stops. Approaching it means swapping, and the penalty is not gradual. From
`GENERATIVE-MODELS.md`, same prompt/seed/latent on a 128 GB Mac:

| Precision | Peak | Time |
|---|---|---|
| bf16 | 118.55 GiB | **730.96 s** |
| INT8 | 67.63 GiB | **70.54 s** |

**10.4x slower** — that is paging, not quantization cost. Below the ceiling,
higher precision measures as free or better; above it you fall off a cliff.

So the operating rule for this machine is **keep peak under ~100 GiB**, and
"use the 128 GB" means "use up to ~100 GiB at the highest bit depth that
fits". `llmctl models` flags `tight` above 80% of budget for this reason.

---

## How to evaluate a context size

```bash
# every cached model against the real ceiling, across a grid of context sizes
conda run -n dekho-apple-local-llm python ../common/context-calc.py

# a single question
conda run -n dekho-apple-local-llm python ../common/context-calc.py \
  --model mlx-community/Qwen3-Coder-Next-4bit --ctx 262144

# stop trusting arithmetic: load it and read the real cache objects
conda run -n dekho-apple-local-llm python ../common/context-calc.py \
  --measure mlx-community/Qwen3-Coder-Next-4bit
```

`context-calc.py` derives everything from each model's own `config.json`, so it
stays correct for hybrid architectures, and flags `OVER native` when a context
exceeds the trained window and `DOES NOT FIT` / `tight` against 115.4 GB.

### Deploying a specific context

```bash
MAX_CONTEXT=131072 ./serve.sh     # advertise 128k instead of 256k
```

`serve.sh` writes the value into the manifest, the opencode plugin reads it,
and `/models` reflects it — no opencode config editing. Per model it is clamped
to that model's trained window, which is why gpt-oss shows 131,072 even when
`MAX_CONTEXT=262144`.

### Hot-swapping degrades throughput ~3x until restart

Measured 2026-09-17. One server, repeatedly alternating between
Qwen3-Coder-Next and the 30B:

| Server state | 30B decode | Qwen3-Coder-Next decode |
|---|---|---|
| Fresh, one model ever loaded | **95–114 tok/s** | — |
| After several model swaps | **36–38 tok/s** | 24 tok/s |

Same model, same prompt, 3x slower. The `vmmap` footprint tells the story:
current 27.2 GB but **peak 62.1 GB** — the allocator retains freed Metal
buffers across swaps, and the prompt cache keeps its own pools, so later
allocations come from a fragmented heap.

Also note the *first* request after a swap pays the full weight load: 89 tokens
took 109.8 s (0.8 tok/s) on the swap itself, then 3.7 s (24.3 tok/s) warm.

**Why, precisely:** `ModelProvider._load()` does free the old model first
(`reset()` + `gc.collect()` + `mx.clear_cache()`), so two sets of weights are
not held at once. But the `LRUPromptCache` is constructed once at server
startup, independently of the `ModelProvider`, and **nothing clears it on a
model swap** — so KV caches built for the previous model stay resident until
LRU eviction, bounded only by `--prompt-cache-size` (10 entries) and
`--prompt-cache-bytes` (24 GB here). That retained pool is what the 62.1 GB
peak is made of: ~44.8 GB of weights plus ~17 GB of stale prompt caches.

No correctness risk — `fetch_nearest_cache(model, tokens)` keys entries by
model, so a cache built for one model is never served to another. It is purely
memory retention. Lowering `PROMPT_CACHE_BYTES` bounds it directly.

**Operational consequence:** hot-swapping is for convenience — trying models,
short comparisons. For a long session on one model, **restart `./serve.sh` with
that model as `MODEL`** and leave it alone. `./monitor.sh` shows the signal:
if `footprint peak` is far above `Metal buffers`, the server has been swapping
and a restart will win back throughput.

### Watch it in practice

Prefill is the cost you feel, not decode. Measured on Qwen3-Coder-Next:
**~2,300–3,700 tok/s prefill**, so a cold 100k-token context takes roughly
30–45 s before the first token. Decode on a freshly started server is ~95–114
tok/s for the 30B; see the hot-swap note above for why it can be much lower. This is why
`--prompt-cache-bytes` matters so much: the second turn on the same
conversation skips that prefill entirely.

To watch memory live while a long session runs:

```bash
sudo powermetrics --samplers gpu_power -i 1000 -n 5   # GPU residency
vm_stat 5                                              # swap activity
```

If `Pageouts` starts climbing, you are over the ceiling — reduce
`MAX_CONTEXT`, lower `PROMPT_CACHE_BYTES`, or pick a smaller model.

---

## Quick reference

| Goal | Setting |
|---|---|
| Max useful context, daily driver | `MODEL=…Qwen3-Coder-Next-4bit MAX_CONTEXT=262144` (the default) |
| Long context on the cheap | Still Qwen3-Coder-Next — it is 4x cheaper per token than the 30B |
| Save memory for something else | `MAX_CONTEXT=65536 PROMPT_CACHE_BYTES=8G` |
| Halve KV at long context | `KV_BITS=8` — but it **disables batching**, serialising requests |
| Short tasks, battery | `MODEL=…Qwen3-Coder-30B-A3B-Instruct-4bit MAX_CONTEXT=65536` |
