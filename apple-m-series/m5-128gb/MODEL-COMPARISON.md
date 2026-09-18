# Model comparison — which one to use offline

Captured **2026-09-17 while online**, because the point of this file is to be
readable on a plane with no network. Every number here is copied from the
upstream model card or technical report, with the source named. Nothing is
from memory.

Read the **caveats** section before trusting any single number.

---

## ⚠️ Verified locally: tool-call support is the first filter

Benchmark scores are irrelevant if mlx-lm cannot parse the model's tool calls.
Support depends on the chat template matching a parser in
`mlx_lm/tool_parsers/`, and **mlx-lm 0.31.3 ships 11 parsers: none for
gpt-oss's harmony format.** Checked with `check-tool-parser.py`:

| Model | mlx-lm parser | Agentic in opencode? |
|---|---|---|
| `Qwen3-Coder-Next-4bit` | `qwen3_coder` | ✅ smoke-tested, passing |
| `Qwen3-Coder-30B-A3B-Instruct-4bit` | `qwen3_coder` | ✅ smoke-tested, passing |
| `GLM-4.7-Flash-4bit` | `glm47` | ✅ parser present |
| `GLM-4.5-Air-4bit` | `glm47` | ✅ parser present |
| `Devstral-Small-2-24B-Instruct-2512-4bit` | `mistral` | ✅ parser present |
| `Qwen3.8-27B-4bit` | `qwen3_coder` | ✅ parser present |
| `Qwen3.6-35B-A3B-4bit` | `qwen3_coder` | ✅ parser present |
| `gemma-4-26b-a4b-it-4bit` | `gemma4` | ✅ parser present |
| **`gpt-oss-120b-MXFP4-Q8`** | **none** | ❌ **chat only** |

**gpt-oss-120b fails in the worst way — silently.** It returns
`finish_reason=stop` with no error and leaks its raw syntax into `content`:

```
<|channel|>analysis<|message|>We need to read src/main.py. Use function read_file.
<|end|><|start|>assistant<|channel|>commentary to=functions.read_file
<|constrain|>json<|message|>{"path": "src/main.py"}
```

Its reasoning channel markers also leak into plain completions, so it is not
even clean for chat without post-processing. It is marked `"tool_call": false`
in `opencode.json` and commented out of `models.txt`. The 63.4 GB is currently
dead weight — **`GLM-4.5-Air-4bit` (60.2 GB, `glm47` parser) is the drop-in
replacement for the "different family, heavy second opinion" slot.**

Run `check-tool-parser.py` before any future download.

---

## The kit at a glance

| Model (MLX repo) | Total params | Active | Weights on disk | Quant | Native ctx | Mode |
|---|---|---|---|---|---|---|
| `mlx-community/Qwen3-Coder-Next-4bit` | **80 B** | **3 B** | 44.8 GB | 4-bit affine, g64 | 262,144 | non-thinking only |
| `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` | **30.5 B** | **3.3 B** | 17.2 GB | 4-bit affine, g64 | 262,144 | non-thinking |
| `mlx-community/gpt-oss-120b-MXFP4-Q8` | **117 B** | **5.1 B** | 63.4 GB | MXFP4 MoE (native) | 131,072 | reasoning: low/med/high |

Also on the shortlist but not pulled (see `models.txt` to enable):

| Model | Total | Active | Disk | Native ctx | Mode |
|---|---|---|---|---|---|
| `GLM-4.7-Flash-4bit` | ~30 B (MoE-Lite) | ~3 B | 16.9 GB | 131,072 | thinking / preserved-thinking |
| `Devstral-Small-2-24B-Instruct-2512-4bit` | 24 B **dense** | 24 B | 15.1 GB | 131,072 | non-thinking |
| `GLM-4.5-Air-4bit` | 106 B | 12 B | 60.2 GB | 131,072 | thinking |

Parameter counts for `Qwen3-Coder-Next` were **verified locally** against the
downloaded safetensors headers (9.97 B `uint32` elements × 8 packed 4-bit values
= 79.72 B), not just read off the card. The card says "80B total, 3B activated,
79B non-embedding" — consistent.

---

## Response quality

### Qwen3-Coder-Next (80B-A3B) — the daily driver

From the [Qwen3-Coder-Next Technical Report](https://arxiv.org/html/2603.00729v1).
`Size` is written as vendor shorthand: `80A3` = 80 B total, 3 B active.

**SWE-bench Verified** (scores differ by agent scaffold — this is the point):

| Model | Size | SWE-Agent | mini-SWE-Agent | OpenHands |
|---|---|---|---|---|
| Claude-Opus-4.5 | — | 78.2 | 77.8 | 79.0 |
| Claude-Sonnet-4.5 | — | 76.0 | 68.4 | 74.6 |
| MiniMax-M2.1 | 230A10 | 74.8 | 70.4 | 71.0 |
| GLM-4.7 | 358A32 | 74.2 | 70.4 | 70.6 |
| Kimi-K2.5 | 1000A32 | 73.2 | 70.8 | — |
| DeepSeek-V3.2 | 671A37 | 70.2 | 67.2 | 72.6 |
| **Qwen3-Coder-Next** | **80A3** | **70.6** | **71.1** | **71.3** |

**SWE-bench Multilingual / Pro:**

| Model | Size | Multilingual | Pro (SWE-Agent) | Pro (mini-SWE) |
|---|---|---|---|---|
| Claude-Opus-4.5 | — | 71.7 | 51.6 | 50.2 |
| MiniMax-M2.1 | 230A10 | 66.2 | 40.8 | 39.1 |
| GLM-4.7 | 358A32 | 63.7 | 45.1 | 39.4 |
| Kimi-K2.5 | 1000A32 | 63.7 | 47.3 | 42.8 |
| DeepSeek-V3.2 | 671A37 | 62.3 | 46.0 | 32.4 |
| **Qwen3-Coder-Next** | **80A3** | **62.8** | **42.7** | **38.7** |

**Terminal-Bench 2.0** — its relative weak spot:

| Model | Size | Terminus2-xml | Terminus2-json |
|---|---|---|---|
| Claude-Opus-4.5 | — | 58.4 | 57.3 |
| Claude-Sonnet-4.5 | — | 51.7 | 51.7 |
| GLM-4.7 | 358A32 | 44.9 | 37.1 |
| Kimi-K2.5 | 1000A32 | 38.8 | 49.4 |
| DeepSeek-V3.2 | 671A37 | 34.8 | 39.3 |
| **Qwen3-Coder-Next** | **80A3** | **34.2** | **36.2** |

**Static code benchmarks:**

| | EvalPlus | MultiPL-E | CRUXEval | LiveCodeBench |
|---|---|---|---|---|
| Qwen3-Coder-480B-A35B | 86.66 | 88.00 | 92.13 | 44.93 |
| Qwen3-Next (general) | 89.00 | 89.00 | 94.81 | 51.79 |
| **Qwen3-Coder-Next** | 86.56 | 88.23 | **95.88** | **58.93** |

**The headline result for our use case:** the report measures **92.7% average
tool-call accuracy across five community IDE/CLI environments**, versus a
49.3–93.7% spread for competitors. Robustness to *someone else's* tool schema is
exactly what opencode depends on, and it's the strongest argument for this model
over a higher-SWE-bench alternative.

### Qwen3-Coder-30B-A3B-Instruct — the fast one

Its card points to a blog for numbers rather than stating them. Third-party
reproductions put **SWE-bench Verified around 50–52** (reported: 51.6 with
OpenHands, 100 turns) — see the
[HF discussion](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/discussions/30)
and [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-coder-30b-a3b-instruct).
Treat as **~51**, less firmly sourced than the others here.

So: roughly **20 points of SWE-bench below Qwen3-Coder-Next** for 1/3 the
memory. That gap is the entire reason to prefer the 80B when plugged in.

### gpt-oss-120b — the second opinion

From the [gpt-oss model card](https://arxiv.org/html/2508.10925v1): 117 B total /
5.1 B active, **SWE-bench Verified ≈ 62.4**, τ-bench Retail ≈ 67.8.
Configurable reasoning effort via `"Reasoning: low|medium|high"` in the system
prompt.

**Note: it cannot tool-call under mlx-lm** (see the section at the top), so the
following is of academic interest here until mlx-lm adds a harmony parser.

**Why it was chosen originally:** it is the only model here whose **published
evals were run at the same quantization we execute**.
Its card states the MoE weights were *post-trained* in MXFP4 and "all evals were
performed with the same MXFP4 quantization." For every other model in this file,
the published number describes bf16 weights we are not running (see caveats).
It's also a genuinely different family and training pipeline, which is what makes
it useful when the Qwen models are both stuck on the same problem.

### GLM-4.7-Flash — not pulled, strong on paper

From the [GLM-4.7-Flash card](https://huggingface.co/zai-org/GLM-4.7-Flash):

| Benchmark | GLM-4.7-Flash | Qwen3-30B-A3B-Thinking-2507 | GPT-OSS-20B |
|---|---|---|---|
| SWE-bench Verified | **59.2** | 22.0 | 34.0 |
| τ²-Bench | **79.5** | 49.0 | 47.7 |
| AIME 25 | 91.6 | 85.0 | 91.7 |
| GPQA | 75.2 | 73.4 | 71.5 |
| LCB v6 | 64.0 | 66.0 | 61.0 |
| HLE | 14.4 | 9.8 | 10.9 |
| BrowseComp | 42.8 | 2.29 | 28.3 |

At 16.9 GB with a claimed 59.2 SWE-bench, this is the most interesting
*unpulled* model in the set — it would beat the 30B-A3B at the same size. Its
τ²-Bench 79.5 is the best tool-use figure anywhere in this document. Caveat:
multi-turn agentic scores assume **Preserved Thinking mode**, which mlx-lm does
not implement, so expect to land below the published number.

### Devstral-Small-2-24B — not pulled, dense

From the [Devstral card](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512),
whose table is the most useful cross-vendor anchor available:

| Model | Size (B) | SWE Bench Verified | SWE Bench Multilingual | Terminal Bench 2 |
|---|---|---|---|---|
| GPT 5.1 Codex Max | — | 77.9% | — | 60.4% |
| Claude Sonnet 4.5 | — | 77.2% | 68.0% | 42.8% |
| Gemini 3 Pro | — | 76.2% | — | 54.2% |
| DeepSeek v3.2 | 671 | 73.1% | 70.2% | 46.4% |
| **Devstral 2** | 123 | 72.2% | 61.3% | 32.6% |
| Kimi K2 Thinking | 1000 | 71.3% | 61.1% | 35.7% |
| Qwen 3 Coder Plus | 480 | 69.6% | 54.7% | 25.4% |
| MiniMax M2 | 230 | 69.4% | 56.5% | 30.0% |
| **Devstral Small 2** | **24** | **68.0%** | 55.7% | 22.5% |
| GLM 4.6 | 355 | 68.0% | — | 24.6% |

68.0% from a **24 B dense** model at 15.1 GB is the standout efficiency claim in
this whole document — nominally beating GLM-4.6 at 355 B. If the Qwen models
disappoint in practice, this is the first thing to try. Being dense, it also has
no MoE routing variance, so its behaviour is more predictable turn to turn.

---

## Caveats — read before trusting the tables

1. **Cross-vendor numbers are not comparable.** Each vendor evaluates with its
   own scaffold, turn limit, and prompt. Qwen3-Coder-Next's own report shows the
   same model scoring 70.6 / 71.1 / 71.3 on *one* benchmark purely by changing
   the agent harness. Differences under ~3 points across vendors are noise.
2. **All scores are vendor self-reported** except the 30B-A3B figure, which is
   third-party reproduction.
3. **We run 4-bit; the benchmarks are not.** Published numbers for the Qwen,
   GLM, and Devstral models describe bf16 weights. Our 4-bit conversions will be
   measurably worse — 4-bit affine typically costs a few points on coding tasks.
   `gpt-oss-120b` is the exception: its evals were run at its native MXFP4.
4. **Terminal-Bench 2.0 is the metric that best predicts agentic CLI work**, and
   every local model here is weak on it (22–36) versus frontier models (42–60).
   Expect to babysit long autonomous runs offline.
5. **Benchmarks don't measure the thing that breaks first.** Tool-call schema
   adherence does, which is why Qwen3-Coder-Next's 92.7% cross-environment
   figure matters more here than two points of SWE-bench.

---

## Decision guide

| Situation | Use | Why |
|---|---|---|
| Plugged in, real agentic work | `Qwen3-Coder-Next-4bit` | Best tool-call robustness (92.7%), ~71 SWE-bench, 256k ctx. 44.8 GB is affordable at 115.4 GB working set. |
| On battery, or quick edits | `Qwen3-Coder-30B-A3B-Instruct-4bit` | ~103 tok/s measured, 17.2 GB, seconds to load. Costs ~20 SWE-bench points. |
| Qwen models both stuck | `GLM-4.5-Air-4bit` (not yet pulled) | Different family, 106B/12B, `glm47` parser verified. Replaces gpt-oss, which cannot tool-call. |
| Plain chat / reasoning only, no tools | `gpt-oss-120b-MXFP4-Q8` | Already on disk; reasoning-effort knob. **Leaks channel markers** — needs post-processing. |
| Need max quality per GB | try `Devstral-Small-2-24B` | Claims 68.0% at 24 B dense, 15.1 GB. |
| Long multi-turn tool loops | try `GLM-4.7-Flash` | τ²-Bench 79.5 — but mlx-lm lacks Preserved Thinking, so discount it. |

**Measured locally** (M5 Max, server warm) — see `CLAUDE-NOTES.md`:
`Qwen3-Coder-30B-A3B-Instruct-4bit` decodes at **~103 tok/s**, emits a tool call
in 0.8 s, and passed all four `smoke-test.py` checks with `HF_HUB_OFFLINE=1`.

Fill this in for the others as you use them — a local measurement beats every
table above for deciding what to actually run.

---

## Sources

- [Qwen3-Coder-Next Technical Report (arXiv 2603.00729)](https://arxiv.org/html/2603.00729v1)
- [Qwen/Qwen3-Coder-Next model card](https://huggingface.co/Qwen/Qwen3-Coder-Next)
- [Qwen/Qwen3-Coder-30B-A3B-Instruct model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) · [SWE-bench repro discussion](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/discussions/30)
- [gpt-oss-120b & gpt-oss-20b Model Card (arXiv 2508.10925)](https://arxiv.org/html/2508.10925v1) · [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
- [zai-org/GLM-4.7-Flash model card](https://huggingface.co/zai-org/GLM-4.7-Flash)
- [mistralai/Devstral-Small-2-24B-Instruct-2512 model card](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512)
- [zai-org/GLM-4.5-Air model card](https://huggingface.co/zai-org/GLM-4.5-Air)
