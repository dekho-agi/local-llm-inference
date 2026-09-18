# Generative models beyond text — what actually runs on this machine

**Captured 2026-09-17.** Every size in this file was summed from the HuggingFace
API (`?blobs=true`, `siblings[].size`), not read off a model card. Every
"MLX-native" claim was checked against the package's actual source tree
(GitHub contents API), not against memory. Package versions came from PyPI on
the capture date. Where a claim could not be verified it says so.

Text-only coding LLMs are covered in `MODEL-COMPARISON.md` — not repeated here.

---

## Summary

The Apple-silicon generative stack matured a lot in 2026 and the core of it is
three packages, not ten: **`mflux`** for images, **`mlx-vlm`** for everything
that reads (vision, audio-in, omni, embeddings, reranking) and **`mlx-audio`**
for everything that listens or speaks. All three are MLX-native and all three
shipped releases within the last 72 hours — `mflux` 0.19.2 landed on the capture
date itself. Video needs a fourth, less settled family (`mlx-video`,
`ltx-2-mlx`, `mlx-gen`), none of which is properly on PyPI. Between them, every
class in this document has a working native path.

Three things worth knowing before the tables:

- **The omni model you named is real and it runs.** `omni-nemotron-30b` is
  `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning`, MLX-native via `mlx-vlm` at
  19.65 GB — but it **outputs text only**. For a model that talks back, use
  Qwen3-Omni (38.78 GB at 8-bit) or, much cheaper, MiniCPM-o-4.5 (6.16 GB,
  Apache-2.0). You no longer need to chain STT → LLM → TTS.
- **Local video generation now works**, which contradicts what you'd expect.
  A 4-second 768×512 LTX-2.5 clip is ~50 s; a 5-second 832×480 Wan 2.2 clip is
  12–23 min. Anything at 720p or over ~8 s is hours. Iterable, not deliverable.
- **Instruction image editing is the strongest class here.** Qwen-Image-Edit-2511
  is Apache-2.0 and within noise of the leaderboard leader.

The 115.4 GB ceiling binds in only three places: `Qwen3-VL-235B` at 4-bit
(133 GB — will not load), MiniMax-H3 video unquantized (125 GiB — the runtime
refuses it), and Wan 2.2 at 1280×704 (103.7 GiB peak, ~12 GB of margin). Most
picks are under 25 GB. What binds everywhere else is *time*, and the two
measurements published on this exact machine make that concrete:
HunyuanImage-3.0 editing at **3.5 min/edit and 69.5 GiB peak**, and Wan 2.2
TI2V-5B at **23 min for 5 seconds of 832×480**.

### What to install first

Only `mlx` + `mlx-lm` are in `dekho-apple-local-llm` today. In priority order:

1. **`mlx-audio`** (`pip install mlx-audio misaki`) — 2.9 GB of weights buys the
   single most useful thing here: `parakeet-tdt-0.6b-v3` transcription (2.51 GB)
   plus `Kokoro-82M-bf16` speech (0.39 GB). Also covers music.
2. **`mlx-vlm`** (`pip install mlx-vlm`) — one package and one
   OpenAI-compatible server for Qwen3-VL, the omni models, embeddings **and**
   reranking. Start with `Qwen3-VL-30B-A3B-Instruct-8bit` (33.53 GB),
   `GLM-OCR-8bit` (1.59 GB), and for RAG
   `Qwen3-Embedding-0.6B-8bit` + `Qwen3-Reranker-0.6B-4bit` (1.0 GB together).
3. **`mflux`** (`uv tool install --upgrade mflux`) — as an isolated `uv` tool,
   **not** into the conda env. `FLUX.2-Klein-4B-6bit` (6.59 GB) to iterate;
   Qwen-Image-Edit-2511 at 8-bit (~35 GB) as the local "Nano Banana".
4. Optional: **Draw Things** as a non-MLX hedge — the only runtime with
   published M5-Max-specific Metal work, and the most stable route to video.
5. Only if you want video: `mlx-gen` (its own env) plus
   `Wan2.2-TI2V-5B-mlx-q8` (19.58 GB).

Smallest useful starting set: **~45 GB** of weights gets you transcription,
speech, a strong VLM, OCR, RAG embeddings + reranking, and fast image editing.

**If you don't care about small and want the quality ceiling instead, skip to
[Big-iron picks](#big-iron-picks--using-all-1154-gb)** — the list above is
tuned for battery and load time, and leaves ~100 GB of this machine idle.

Skip `diffusionkit` and `lightning-whisper-mlx` — both abandoned. Prefer
`mlx-vlm` over `mlx-embeddings` for embeddings (the latter has open
silent-corruption bugs on this exact stack). Never `pip install mlx-video` —
that name belongs to a different package. And do not install `mlx-gen` next to
`mflux`. See *don't bother*.

### Memory budget

`mx.device_info()["max_recommended_working_set_size"]` = **115.4 GB**, not
128 GB. For diffusion models the weights figure is *not* the peak: mflux holds
a transformer plus a text encoder plus a VAE, and latents at 2048×2048 are not
free. Budget roughly weights + 15–25% for image models, and use
`--low-ram` / `--mlx-cache-limit-gb` when it gets tight. For VLMs and omni
models the vision/audio towers add a few GB on top of the weights, and long
video or hour-long audio inputs blow up the KV cache far more than the weights do.

Video is the one class where you must plan peaks explicitly rather than reading
the weight size. Two counter-intuitive facts: LTX-2.5's peak is **set by its
text encoder, not the geometry** — a 256×256×9 clip and a 512×288×121 clip peak
within 0.01 GB of each other — and evicting the DiT around the encode phase
drops peak from 62.40 GB to 40.66 GB with **bit-identical output**. Meanwhile
Wan 2.2 at 1280×704 peaks at **103.7 GiB**, which fits but leaves ~12 GB of
margin, and MiniMax-H3 unquantized needs 125 GiB and will be refused. Quantized
*storage* also no longer implies lower runtime memory for Wan A14B.

---

## Big-iron picks — using all 115.4 GB

The per-class picks further down optimise for weights-per-quality, which leaves
~100 GB of this machine idle. This section does the opposite: **for each class,
the highest-quality variant that actually fits**, budgeting ~100 GB for weights
plus activations and leaving ~15 GB for the OS and working memory.

The main lever is **quantization**: `mflux`'s `--quantize` defaults to `None`,
which means omitting `-q` runs **full bf16** (verified at
`src/mflux/cli/parser/parsers.py:142`). On a 36–64 GB Mac you have no choice but
4-bit or 6-bit; here you can run the same model at bf16, which is a bigger
quality win than switching to a larger model.

### ⚠️ But there is a cliff at ~100 GiB, and it costs 10×

Below the working-set ceiling, **higher precision is free or better than free** —
three independent measurements agree:

| Project | Machine | Finding |
|---|---|---|
| `flux-2-swift-mlx` | M2 Ultra 96 GB, FLUX.2-dev 32 B, 1024², 28 steps | bf16 **1758.6 s** vs qint8 1842.5 s vs int4 1779.6 s — **quantized is slightly *slower***. Their note: *"The real benefit is memory savings, not speed."* |
| `hymlx` | M5 Max 128 GB, HunyuanImage-3.0 | 9.7–11.0 s/step across a **2× range in bit depth**. *"Precision buys memory, not speed. This path is compute-bound."* (14 TFLOPS measured against a ~19–20 TFLOPS 8-bit ceiling.) |
| `mdream` | HiDream-O1-Image | 6-bit is **slower than bf16 at every resolution**; quantization *"is not recommended… the memory it saves is only worth having on a machine that cannot hold bf16."* |

**Cross the ceiling and that reverses violently.** The single most useful
measurement I found, from `cosmos3-quant-mlx-cuda` on an **M4 Max 128 GB**, same
prompt / seed / serialised latent / 1024² / 4 steps:

| Variant | Weights | **Peak** | Wall clock |
|---|---|---|---|
| **bf16** | 119.21 GiB | **118.55 GiB** | **730.96 s** |
| **INT8 g64** | 64.77 GiB | **67.63 GiB** | **70.54 s** |
| INT4 g64 | 35.70 GiB | 38.97 GiB | 75.43 s |

**10.4× slower** — and that is not a quantization penalty, it is what happens
when peak exceeds `max_recommended_working_set_size` and the system starts
swapping. 118.55 GiB is *above* this machine's 115.4 GiB.

**The operating rule for this machine: keep peak under ~100 GiB.** Between
~100 and 115.4 GiB you are gambling; above it you lose an order of magnitude.
So "use all 115.4 GB" really means **"use up to about 100, at the highest bit
depth that fits under it"** — not "fill it."

Activation overhead is real and must be budgeted: HunyuanImage measures
**19.8 GiB** on top of weights (69.5 peak − 49.7 weights). That is the number to
add when you see a weights figure below.

### What to run when you have 115 GB

| Class | Best-that-fits | Weights | Quant | Latency / peak (measured where noted) |
|---|---|---|---|---|
| Image gen | `mflux-community/krea-2-turbo-mflux-bf16` | **34.20 GB** | bf16 | not published. **T2I Elo 1017.99** — highest open-weights, ⚠️ mapping inferred |
| Image gen, alt | `mflux-community/ideogram-4-mflux-q8` | **26 GB** | 8-bit | **34 s, 28.7 GB peak** (M5 Pro 64 GB). Elo 1011.97 |
| Image gen, dark horse | `JuliaML/Cosmos3-Super-Text2Image-4Step-INT8-G64-BF16` | **69.54 GB** | INT8 g64 | **70.5 s, 67.63 GiB peak** (M4 Max). Elo 969.96, **commercial licence** |
| Image **editing** | HunyuanImage-3.0-Instruct via `hymlx`, published mixed-4/8 | **52.45 GiB** | experts 4-bit / rest 8-bit | **4.2 min, 69.5 GiB peak** (M5 Max 128 GB). **Editing Elo 1025.53 — #1 open** |
| Image editing, spend more | same, converted to **6-bit g64** | **70.8 GiB** | 6-bit g64 | ~90.6 GiB peak *(estimated)*. Correlation +0.9980 |
| Image editing, Apache-2.0 | `mflux-community/qwen-image-edit-2511-mflux-bf16` | **56.62 GB** | bf16 | not published. Elo 1023.51 |
| VLM | `mlx-community/Qwen3.5-122B-A10B-5bit` | **84.88 GB** | 5-bit | not published. Apache-2.0, 30.5 GB headroom, **KV only 24 KiB/token** |
| VLM, largest comfortable | `mlx-community/Qwen3.5-122B-A10B-6bit` | **100.15 GB** | 6-bit | **100.12 GB measured resident.** ~15 GB headroom |
| Omni (speech in **and** out) | `mlx-community/Qwen3-Omni-30B-A3B-Instruct-bf16` | **70.53 GB** | bf16 | not published |
| Video | LTX-2.5 fully bf16, DiT evicted | — | bf16 | **40.66 GB peak**, bit-identical to the 62.40 GB path |
| Video + synced audio | `ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit` | **70.08 GB** | 8-bit | 11–17 min @960×544 (M5 Max 128 GB) |
| STT | — | — | — | **going bigger buys nothing.** Stay on Parakeet, 2.51 GB. |
| TTS | — | — | — | **going bigger buys control, not quality.** |

**Read the image rows carefully — the biggest model is the wrong answer for
generation.** HunyuanImage-3.0-Instruct is **#1 open-weights at editing
(1025.53)** but only **11th at text-to-image (964.80)** on Artificial Analysis,
*below* Ideogram 4.0 Quality (1011.97, 9 B, 26.96 GB, native mflux) and Krea 2
Turbo (1017.99, 34.20 GB, native mflux). Spending 70–101 GiB and 2.7 min/image
on an 80 B MoE to *generate* is a bad trade against a 27 GB model that scores
50 Elo higher and finishes in 34 seconds. Spend the memory on **editing**,
where Hunyuan genuinely leads and nothing open beats it.

⚠️ The Krea mapping is **inferred, not confirmed** — Artificial Analysis lists
three Krea 2 entries but never sets an open-weights URL for them, and Krea
released exactly two checkpoints (Turbo distilled, Raw base). Medium
Turbo↔Turbo is the only self-consistent reading. Verify against Krea's release
notes before betting on it; Ideogram 4 at 1011.97 is the confirmed
second-best and is also native in mflux.

Total if you wanted the image-editing and omni picks resident at once:
56.62 + 70.53 = **127 GB — does not fit.** Pick one workload at a time; these
are not co-resident budgets.

### Image generation and editing — bf16 everything

The official **`mflux-community`** org publishes a bf16 rung for every model,
so there is no conversion step. Verified sizes:

| Model repo id | Size | Quant | Runtime | MLX-native? | Verdict at 115.4 GB |
|---|---|---|---|---|---|
| `mflux-community/qwen-image-edit-2511-mflux-bf16` | **56.62 GB** | bf16 | `mflux-generate-qwen-edit` | ✅ mflux | **Best Apache-2.0 editor.** ~59 GB spare. |
| `mflux-community/qwen-image-2512-mflux-bf16` | **55.27 GB** | bf16 | `mflux-generate-qwen` | ✅ mflux | **Best Apache-2.0 generator.** ~60 GB spare. |
| `mflux-community/boogu-image-turbo-mflux-bf16` | 35.90 GB | bf16 | `mflux-generate-boogu` | ✅ mflux | Comfortable. T2I only — Boogu *Edit* is not in mflux. |
| `mflux-community/krea-2-turbo-mflux-bf16` | **34.20 GB** | bf16 | `mflux-generate-krea2` | ✅ mflux | **Probably the #1 open-weights T2I model — Elo 1017.99** (⚠️ mapping inferred). `krea-2-raw` is the same size, Elo 1010.62. |
| `mflux-community/flux-1-dev-kontext-mflux-bf16` | 33.75 GB | bf16 | `mflux-generate-kontext` | ✅ mflux | Comfortable. Kontext at full precision. **Non-commercial.** |
| `mflux-community/flux-1-dev-mflux-bf16` | 33.75 GB | bf16 | `mflux-generate` | ✅ mflux | Comfortable. **Legacy** family — mflux may deprecate it. |
| `mflux-community/flux2-klein-9b-kv-mflux-bf16` | 33.47 GB | bf16 | `mflux-generate-flux2-edit` | ✅ mflux | Comfortable. **Multi-reference editing, ~2.4× KV speedup.** Non-commercial. |
| `mflux-community/flux2-klein-9b-mflux-bf16` | 33.47 GB | bf16 | `mflux-generate-flux2` | ✅ mflux | Comfortable. Non-commercial. |
| `mflux-community/ideogram-4-mflux-q8` | **26 GB** | 8-bit | `mflux-generate-ideogram4` | ✅ mflux | **T2I Elo 1011.97.** Measured **34 s / 28.7 GB peak** (M5 Pro 64 GB). ⚠️ Prefer q8 over the `-bf16` repo: Ideogram ships **fp8 only**, so "bf16" is an upcast with no quality gain, and q8 removes a dequantization step from every forward. |
| `mflux-community/fibo-mflux-bf16` / `fibo-edit-mflux-bf16` | 25.50 / 24.12 GB | bf16 | `mflux-generate-fibo{,-edit}` | ✅ mflux | Comfortable. JSON prompts. |
| `mflux-community/ernie-image-turbo-mflux-bf16` | 23.11 GB | bf16 | `mflux-generate-ernie-image-turbo` | ✅ mflux | Comfortable. |
| `mflux-community/z-image-turbo-mflux-bf16` | 20.53 GB | bf16 | `mflux-generate-z-image-turbo` | ✅ mflux | Comfortable. 6 B, 9 steps, Apache-2.0. |

**Is bf16 worth it over 8-bit?** Probably, but the argument is "the memory is
free," not a measured delta. mflux's own Qwen README warns *"6-bit or below can
degrade the image a lot more compared to Flux,"* which says the Qwen transformer
is quantization-sensitive; bf16 is the only rung with no loss at all; and the
19 GB between 37.47 and 56.62 GB is memory you are not otherwise using. But
**there is no published bf16-vs-8-bit fidelity measurement for any mflux image
model, and no measured peak-RAM figure for Qwen-Image 20 B at either
precision.** Every `mflux-community` model card is **empty** — `raw/main/README.md`
returns "Entry not found" (tracking issue **#676**) — and no mflux issue or
release note carries a Qwen RAM number.

Two things that *are* known and change the arithmetic:

- Since **0.18.0 mflux evicts the text encoder after encoding** and clears the
  MLX cache between seeds, specifically "to prevent OOM on large models such as
  FLUX.2 Klein 9B". So peak is **max(text-encoder phase, transformer phase)**,
  not the sum. For `qwen-image-edit-2511-mflux-bf16` that means roughly
  **38 GiB** of transformer resident, not 52.7 GiB.
- The **VAE decode dominates the overhead**, not the transformer. On klein-4B q4
  at 1024², issue #407 measured total peak 14,032 MB → **7,178 MB** with 512 px
  VAE tiles → 6,326 MB at 256 px, with the stage breakdown showing VAE decode as
  ~100% of the overhead (transformer 1,550 MB, text encoding 1,106 MB) **at no
  speed cost**. So `--vae-tiling` is close to a free win. Note mflux still has
  **no attention slicing** (open issue #718).

⚠️ **One caveat on mflux's Qwen implementation specifically.** The maintainer
wrote in issue #338 (2026-01-28): *"I'm starting to doubt my implementation of
Qwen Image in general (it was done rather quickly this summer with limited
resources). I tried one LoRA yesterday with the 2509 edit model and it produced
absolute garbage results, and nothing compared to Flux2 Klein."* The same thread
measured `Qwen-Image-2512-8bit` + 4-step Lightning LoRA at **11.90 s/it in
mflux vs 4.83 s/it in ComfyUI Torch-MPS — ~2.4× slower**. That is eight months
old and `Qwen-Image-2512` has since become a first-class entry, but it is the
last public word I found. If Qwen output disappoints, try FLUX.2-klein-9B bf16
(33.47 GB) before assuming the model is the problem.

### The one thing 128 GB genuinely unlocks: HunyuanImage-3.0

This is the only entry in this document that is *impossible* on a 36–64 GB Mac
and comfortable here. **80 B total / 64 experts / 13 B active** per Tencent's
card (hymlx counts 84.9 B from the tensor inventory).

**Use it for editing, not generation.** On Artificial Analysis it is
**#1 open-weights at editing — Elo 1025.53**, ahead of Qwen-Image-Edit-2511
(1023.51) and FLUX.2-klein-9B (1013.15). But at **text-to-image it is 11th,
Elo 964.80** — below Ideogram 4.0 Quality (1011.97), Krea 2 Turbo (1017.99),
Qwen-Image-Max-2512 (998.33) and even HiDream-O1-Image (979.75), all of which
are smaller and far faster. Note also that these Elos are **FLUX.2-dev-relative**
(dev is pinned at exactly 1000.00 with zero CI), so they are not comparable to
LMArena-style absolute numbers.

Runtime is **`KaedeTai/hymlx`** (MIT, MLX-native, not mflux, not on PyPI):

```sh
pip install mlx numpy pillow safetensors transformers huggingface_hub torch
git clone https://github.com/KaedeTai/hymlx && cd hymlx

# text to image — the model writes its own <think>/<recaption> stage first
python3 tools/dream.py "a portrait photograph of an elderly fisherman" --steps 14

# reference-image editing — the text stage sees the reference image too
python3 tools/dream.py "make the hamster wear a tiny red wizard hat"         --image hamster.png --steps 14

# skip the CoT stage: ~45 s faster
python3 tools/dream.py "..." --no-cot --steps 14

# reuse a CoT and drop CFG after step 6: ~20% faster
python3 tools/dream.py "..." --cot-file fox_cot.txt --cfg-steps 6

# change precision in place — no re-download
python3 tools/requantize.py ...
```

**Measured by the author on an M5 Max / 128 GB, 1024², CFG on, 14 steps:**

| Task | Sampling | Total | Peak |
|---|---|---|---|
| text to image | 161 s | **2.7 min** | 69.5 GiB |
| text to image, reusing a CoT | 161 s | **2.0 min** | 69.5 GiB |
| reference-image editing | 193 s | **4.2 min** | 69.5 GiB |
| reference-image editing, `--no-cot` | 194 s | **3.5 min** | 69.5 GiB |

⚠️ The *model card* says editing is 3.5 min flat; the *repo README* shows 3.5 min
is the `--no-cot` figure and full CoT editing is **4.2 min**. The CoT stage
landed in the final commit, after the card was written. The edit's text stage
costs 44 s because its prefix is 6,368 tokens (4,096 VAE + 1,024 ViT for the
reference image) against 1,235 for plain text-to-image. CoT-on gives better
instruction adherence but drifts further from the source image — an
obedience-vs-fidelity trade, not a bug.
| chain-of-thought alone (1,100 tokens) | — | 100 s | — |

**Spend the spare memory on precision — but there is a catch.** The released
checkpoint is mixed 4/8-bit at 49.7 GiB. The author's measured precision table
(σ=1.0 clean-latent correlation against a fully-8-bit reference) says which
rung to want:

| Configuration | Weights | s/step | Correlation vs 8-bit |
|---|---|---|---|
| 8-bit g64 | **81.3 GiB** | 11.0 | baseline |
| **6-bit g64** | **70.8 GiB** | 10.4 | **+0.9980** |
| experts 4-bit, rest 8-bit *(released)* | 49.7 GiB | 11.0 | +0.9907 |
| 4-bit g64 | 48.8 GiB | 10.7 | +0.9650 |
| 4-bit g32 | 58.0 GiB | 9.7 | +0.9622 |

🚩 **You cannot get there from the published repo.** `tools/requantize.py`
only converts *downward* — its metadata records `"derived_from": "<N>-bit"`. To
obtain 6-bit or 8-bit you must download the **168.66 GB bf16 upstream** and
convert:

```sh
hf download tencent/HunyuanImage-3.0-Instruct        # 168.66 GB, ungated
python3 tools/convert.py --bits 6 --group-size 64 --out ~/models/hymlx-6bit
```

`convert.py` reads from a **hardcoded cache glob**
(`~/.cache/huggingface/hub/models--tencent--HunyuanImage-3.0-Instruct/snapshots/*`),
so download to the cache, **not** `--local-dir`. It processes one layer at a
time, so conversion peak is only ~10 GiB, and `--prune-source` deletes consumed
shards as it goes. Your 1.6 TB of disk is ample; budget the download time.

**If you do convert, target 6-bit g64 (70.8 GiB weights).** Adding the measured
19.8 GiB activation overhead puts peak at roughly **90.6 GiB — an estimate, not
a measurement** — which clears the ~100 GiB rule. 8-bit at 81.3 GiB would land
near **101 GiB**, which is exactly the zone the Cosmos3 data says to avoid. So
6-bit is the ceiling worth running, and it is +0.9980 against 8-bit anyway.

Two further findings worth knowing: **group size is not the lever, bit depth
is** (g32 costs 9 GiB more than g64 and measures *worse*), and **attention is
the sensitive part, not the experts** — the 64 experts are 91.7% of the
parameters, and dropping only them to 4-bit lifts correlation from 0.965 to
0.991 for 1 GiB. Also, requantizing 8-bit→6-bit costs **3.2% more error** than
quantizing from bf16 directly (theory says √(17/16) = 3.1%; eight real tensors
all measured 3.2%) — another reason to convert from the bf16 source.

**Honestly, start with the published mixed-4/8 at 52.45 GiB.** It is
known-good, +0.9907, needs no 168 GB download, and peaks at a measured
69.5 GiB. The jump to 6-bit buys 0.0073 of correlation for a 168 GB download
and 21 GiB more peak.

**Now the reasons not to.** Be clear-eyed about this one:

- **`hymlx` has 0 GitHub stars and the weights repo has 67 downloads.** It is a
  single-author project, last pushed 2026-09-01. Every number above is the
  author's own, unreproduced. The verification methodology described (each
  stage diffed against the official PyTorch implementation on CPU, tolerance
  set to the reference's own measured bf16 error per module) is unusually
  rigorous for a project this small — but it is still one person's claim.
- **Resolution is not adjustable.** The `ResolutionGroup` has 37 entries, all
  between 0.85 and 1.05 MP. Asking for 512² or 2048² both give 1024². Only
  aspect ratio moves. If you need 2 MP output, this model cannot do it.
- **The licence excludes the EU, the UK and South Korea — and it binds the
  images, not just the weights.** Verbatim from §3(c): *"You must not use,
  reproduce, modify, distribute, or display the Tencent Hunyuan Works, **Output
  or results** of the Tencent Hunyuan Works outside the Territory."* §1(l)
  defines Territory as *"the worldwide territory, excluding the territory of the
  European Union, United Kingdom and South Korea."* A separate licence is
  required above 100 M MAU. The hymlx code is MIT; the weights and their outputs
  are not.
- **The official Python is not vendored** — tokenizer, image processor and
  sequence assembly are downloaded from Tencent's HF repo at run time, so you
  acquire them under Tencent's licence along with the weights.
- **It is slow because of tokens, not size.** At 1024² Hunyuan produces 4,096
  image tokens against HiDream-O1's 1,024, and runs 2 forwards per step to
  HiDream's 1. On the same machine, same day: **173 s vs 9.9 s per image —
  17.5×.** For editing the gap narrows to 1.8×, and Hunyuan can edit at 1 MP
  where HiDream collapses below ~1.7 MP.

So: the highest-quality local editor available on this machine is also the
slowest, the least proven, the most licence-restricted, and fixed at 1 MP. Use
`qwen-image-edit-2511-mflux-bf16` (56.62 GB, Apache-2.0, arbitrary resolution,
1,023 Elo) as the default and reach for Hunyuan when the extra 5 Elo points
actually matter.

### Two non-mflux image options worth knowing

Both are outside mflux, both are MLX-native, and both matter because **mflux's
absolute ceiling is ~57 GB of weights** — Qwen-Image-Edit-2511 bf16 at
52.73 GiB is the largest thing in the whole family, and everything else is
≤ 33.5 GiB. You cannot spend 100 GB inside mflux.

**NVIDIA Cosmos3-Super-Text2Image-4Step — the dark horse.** ~64 B dense,
**OpenMDW-1.1 (commercially permissive**, unlike Hunyuan and FLUX), T2I Elo
969.96. Runtime is `gtrg55/cosmos3-quant-mlx-cuda` (MLX transformer + a
Torch/MPS VAE bridge). The INT8 conversion
`JuliaML/Cosmos3-Super-Text2Image-4Step-INT8-G64-BF16` is **69.54 GB** and its
measured numbers on an M4 Max 128 GB are the best speed/memory pair in this
whole section: **67.63 GiB peak, 70.5 s** at 1024². That is Hunyuan's memory
footprint at **2.3× the speed**, with a licence you can actually ship under.
Reproducibility is SHA-256-pinned on prompt, latent and each output PNG, with
PSNR 28.77 dB for INT8 against bf16.

Caveats, and they are not small: **text-to-image only, no editing**; only the
4-step distilled transformer has an MLX quant (the higher-scoring 983.21 and
992.22 variants do not); validated on **one machine, one prompt**; **all three
MLX repos have 0 downloads**; and it is fundamentally a Physical-AI world model,
so aesthetic quality is unproven despite the Elo. An evening's experiment, not
a plan.

**HiDream-O1-Image via `KaedeTai/mdream` — the actual speed/quality sweet
spot.** 35.25 GB, **MIT**, ungated, and **T2I Elo 979.75 — above
HunyuanImage-3.0-Instruct's 964.80**. Same author as hymlx. It is not a latent
DiT: it is Qwen3-VL-8B with 32×32 pixel-patch shims run as a flow model in pixel
space, no VAE and no frozen text encoder, which is why 1024² is 1,024 image
tokens against Hunyuan's 4,096. Measured: **8.0 s** per 768×1024 image at 28
steps, and 46.3 dB PSNR against ComfyUI fp32 — 16.2 dB tighter than the
reference's own bf16-vs-fp32 envelope.

Its author explicitly **recommends against quantizing it**: 6-bit is visually
indistinguishable but *slower than bf16 at every resolution*, so no quantized
checkpoint is published on purpose. Hard constraints found by testing against
ComfyUI as a control: use `base` not `dev` for anything with skin (`dev` gives
crazed faces); cfg > 1 when editing (cfg 1.0 returns pure noise); and
**editing needs ≥ ~1.7 MP** or it collapses — 768×1024 is noise, 1152×1536 is
clean. That last one is the mirror image of Hunyuan's 1 MP lock: Hunyuan edits
*only* at 1 MP, HiDream edits *only above* 1.7 MP.

### VLM — Qwen3.5-122B-A10B at 5-bit, and the reason is the KV cache

The obvious answers here are all wrong, and the reason is generational: the
Qwen3-VL-235B / InternVL-78B era is from **October 2025**. Two newer natively
multimodal generations exist in `mlx-community` with full quant ladders, and the
older one already **beats Qwen3-VL-235B on every published vision benchmark
while being smaller**:

| Benchmark | Qwen3.5-122B-A10B | Qwen3-VL-235B-A22B |
|---|---|---|
| MMMU | **83.9** | 80.6 |
| MMMU-Pro | **76.9** | 69.3 |
| MathVista (mini) | **87.4** | 85.8 |
| MMStar | **82.9** | 78.7 |
| OCRBench | **92.1** | 87.5 |
| VideoMME (w/ subs) | **87.3** | 83.8 |

| Model repo id | Size | Quant | MLX-native? | Verdict at 115.4 GB |
|---|---|---|---|---|
| **`mlx-community/Qwen3.5-122B-A10B-5bit`** | **84.88 GB** | **5-bit** | ✅ `qwen3_5_moe` | ✅ **The pick.** 30.5 GB headroom, Apache-2.0, 122 B/10 B active. |
| `mlx-community/Qwen3.5-122B-A10B-6bit` | **100.15 GB** | 6-bit | ✅ | ✅ Largest comfortable fit — **measured 100.12 GB resident** (LM 99.22 + vision tower 0.90). ~15 GB headroom. |
| `mlx-community/Qwen3.5-122B-A10B-4bit` | 69.62 GB | 4-bit | ✅ | Comfortable, 45 GB spare. 5,078 downloads — the well-trodden rung. |
| `Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP` | 92.51 GB | oQ3 | ✅ `qwen4_exp` | Fits. ~180 B, 512 experts, Aug 2026 — newest generation. ⚠️ no head-to-head vs the 122 B. |
| `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` | 73.52 GB | 4-bit, expert-pruned | ✅ `qwen4_exp` | Comfortable. **20,862 downloads** — the most-used large MLX VLM. |
| `mlx-community/GLM-4.6V-6bit` | 88.59 GB | 6-bit | ✅ `glm4v_moe` | Fits — but **184 KiB/token** of KV. See below. MIT. |
| `mlx-community/Llama-4-Scout-17B-16E-Instruct-6bit` | 88.30 GB | 6-bit | ✅ `llama4` (real 17.7 KB `vision.py`) | Fits — but **superseded**; see below. |
| `mlx-community/Qwen2.5-VL-72B-Instruct-8bit` | 78.02 GB | 8-bit | ✅ `qwen2_5_vl` | Fits. 72 B dense, but Feb 2025. |
| `mlx-community/InternVL3-38B-bf16` | 76.79 GB | bf16 | ✅ `internvl_chat` | Fits. Full precision, different family. |
| `mlx-community/Qwen3-VL-32B-Instruct-bf16` | 66.73 GB | bf16 | ✅ `qwen3_vl` | Comfortable. Dense 32 B at full precision. |
| `mlx-community/Qwen3-VL-235B-A22B-Instruct-3bit` | 104.02 GB | 3-bit | ✅ `qwen3_vl_moe` | ⚠️ Technically fits, now the **worst** ~100 GB option. See below. |
| `mlx-community/GLM-4.5V-8bit` | **115.29 GB** | 8-bit | ✅ | ❌ Equals the ceiling exactly. Unrunnable. |
| `mlx-community/Llama-4-Scout-...-8bit` | **115.46 GB** | 8-bit | ✅ | ❌ Over the ceiling. |
| `mlx-community/Qwen3-VL-235B-A22B-Instruct-4bit` | **133.41 GB** | 4-bit | ✅ | ❌ Well over. Will not load. |

**The decisive factor is KV cache, not weights.** Qwen3.5-122B is a *hybrid*
attention model (`full_attention_interval: 4` — verified in its config, so only
12 of 48 layers are full attention). That changes the per-token cost by ~8×:

| Model | Full-attn layers | KV per token | @32k | @128k |
|---|---|---|---|---|
| **Qwen3.5-122B-A10B** | **12 of 48** | **24.0 KiB** | **0.75 GiB** | **3.00 GiB** |
| Qwen3-VL-235B-A22B | 94 of 94 | 188.0 KiB | 5.88 GiB | 23.50 GiB |
| GLM-4.6V | 46 of 46 | 184.0 KiB | 5.75 GiB | 23.00 GiB |

So the 235 B at 3-bit is 104.02 GB of weights **plus 5.88 GiB of KV at only 32 k**
— about 110.3 GB before the vision tower, before deepstack's three extra
feature maps, before MoE routing buffers. A 1280×1280 image is 1,600 visual
tokens; a multi-image or video prompt at 8–16 k visual tokens pushes you over.
**It will run short single-image prompts and thrash on long ones.** The 122 B at
5-bit uses 19 GB less and costs 3 GiB of KV at *128 k*.

**Three further reasons to skip the 235 B at 3-bit:**

1. **Uniform 3-bit on a MoE VLM is expensive.** The MODE paper
   (arXiv 2606.17118) reports that an *ILP-optimised per-expert* bit allocation
   limits loss to *"within 2.9% at W3A16"* — and that 2.9% is the **floor**.
   `mlx-community/-3bit` is uniform, `group_size: 64`, with no such
   optimisation, so its loss is larger.
2. MODE also names the failure mode that matters here: *"the numerical dominance
   of vision tokens causes expert selection frequency to be dominated by vision
   tokens, masking experts that are critical to the text modality."* Naive
   MoE-VLM quantization damages the **text** pathway first.
3. **There is no better 235 B quant.** Across all authors the complete MLX set
   is 3-bit (104.02), 4-bit (133.41), an abliterated 3-bit (119.07), and
   LibraxisAI's nvfp4/mxfp4/mxfp8. **No 2-bit, no DWQ, no OptiQ, no mixed.**
   The sub-3-bit 235 B work is all GGUF, which mlx-vlm cannot load.

**`--kv-bits` is the lever if you insist on an all-full-attention model.**
mlx-vlm 0.7.1 supports KV quantization with TurboQuant, and its README reports
*"up to 3.6× at 8-bit and 6.4× at 4-bit"* for all-full-attention models, with
8-bit measured as slightly **faster** than none (52.6 vs 50.3 gen tok/s on
gemma-4-26b-a4b at 20 k). That takes the 235 B's 5.88 GiB at 32 k to ~1.6 GiB.
⚠️ But KV quantization is **incompatible with continuous QSA batching** on
Qwen3.8-Flash-Next — requesting both raises an explicit error.

**Llama-4 vision is real but dead.** `mlx_vlm/models/llama4/vision.py` is
17,669 bytes — a full ViT, not a stub — and Scout conversions exist. But every
Llama-4 MLX conversion is dated **April–May 2025** and nothing has been
reconverted in 16 months; peak downloads are 1,048, roughly 20× below the
Qwen3.8-Flash-Next conversions. Scout 6-bit costs 88.30 GB for a 17 B-active
model against `Qwen3.5-122B-A10B-5bit` at 84.88 GB. **Don't.**

⚠️ **Broken or absent, so you don't chase them:**
`mlx-community/GLM-4.6V-8bit` is 1 file and 0 bytes;
`mlx-community/InternVL3-78B-8bit` returns HTTP 401 (does not exist);
**`InternVL3_5-241B-A28B` has no MLX conversion anywhere** (MLX InternVL stops at
`InternVL3-78B-4bit`, April 2025); **Pixtral-Large and Mistral-Large have no MLX
vision** — `mlx_vlm/models/mistral4/` contains only `__init__.py` and
`language.py`, i.e. text only; `hunyuan_vl`, `zaya1_vl` and `youtu_vl` have
mlx-vlm directories but **zero MLX weights on the Hub**; and there is no GLM-5
VLM in MLX. Also over the ceiling: `Step-3.7-Flash-8bit` 209.28 GB,
`Step-3.5-Flash-bf16` 393.92 GB, `MiniMax-M3-4bit` 241.50 GB,
`Llama-4-Maverick-...-4bit` 225.92 GB.

**Reality check on "fits".** A widely-cited write-up on Apple unified memory
puts it bluntly: *"A 128GB machine does not give a 128GB model home; it gives
roughly 90GB of comfortable resident room once everything else is accounted
for."* And the failure mode is a cliff, not a slope — throughput *"collapses
from 7–8 tok/s down to <0.5 tok/s as the kernel thrashes swap."* mlx-gen's own
docs record a process *"killed by the OS at a footprint of at least 92.7 GiB…
on a 128 GiB machine at recommended settings, because another application held
7.5 GiB."* Your 115.4 GB is already a raised `iogpu.wired_limit_mb`, so you have
more room than default — but that is the shape of the risk above ~100 GB.

### Omni — Qwen3-Omni bf16 is the ceiling, and it's the right call

| Model repo id | Size | Quant | Audio out? | Verdict at 115.4 GB |
|---|---|---|---|---|
| `mlx-community/Qwen3-Omni-30B-A3B-Instruct-bf16` | **70.53 GB** | bf16 | ✅ | **The pick.** ~45 GB spare. Largest omni with speech out that mlx-vlm implements. |
| `mlx-community/Qwen3-Omni-30B-A3B-Instruct-8bit` | 38.78 GB | 8-bit | ✅ | Comfortable. The everyday choice. |
| `mlx-community/Nemotron-3-Nano-Omni-...-bf16` | 66.05 GB | bf16 | ❌ text only | Fits. Best *analyst* — 256 k ctx, 1 h audio, 2 min video. |
| `mlx-community/MiniCPM-o-4_5-5bit` | 7.19 GB | 5-bit | ✅ | **Tops out here** — no larger MiniCPM-o conversion exists. |

**bf16 is worth it here, and this is now the best-evidenced call in the
document — the vocoder really is quantized.** Measured per-component by reading
each shard's safetensors header:

| Component | bf16 build | 8-bit build |
|---|---|---|
| thinker (the LM) | 63.438 GB | 34.814 GB |
| **talker** | **6.649 GB** | **3.665 GB** ← quantized |
| **code2wav** (vocoder) | **0.432 GB** | **0.283 GB** ← quantized |
| total | **70.520 GB** | 38.762 GB |

`model.safetensors.index.json` confirms it: the 8-bit build's talker carries
246 `.scales`/`.biases` pairs and code2wav 61 — i.e. both are genuinely
quantized, not passed through. The root cause is in the library.
`mlx_vlm/utils.py:670` lists the modules that escape quantization:

```python
multimodal_modules = (
    "vision_model", "vision_tower", "vl_connector", "sam_model",
    "audio_model", "audio_tower", "code_predictor", "img_projector",
    "multi_modal_projector", "patch_merge_mlp",
)
```

**`talker` and `code2wav` are not in that list**, and
`qwen3_omni_moe.Model.quant_predicate` just delegates to the thinker's. So
*every* quantized mlx-community Qwen3-Omni build has a lossy vocoder. This
matches what NVIDIA does upstream — in their FP8 checkpoint the *"vision
encoder, audio encoder, Talker, and Code2Wav retain BF16 precision"*, and only
the Thinker is quantized.

**The elegant fix, if you ever want the memory back.** The whole audio-output
path is only **7.08 GB at bf16**. So an 8-bit thinker with a bf16 talker and
code2wav would be ~41.9 GB — **3.1 GB more than the stock 8-bit build, 28.6 GB
less than bf16, with bit-identical speech.** `--quant-predicate` can't express
it (all seven built-in recipes route through `skip_multimodal_module`), so it
needs a short script calling `mlx_vlm.convert.convert()` with a custom
predicate that returns `False` for paths starting `talker.` / `code2wav.`.
Since 70.53 GB fits with 45 GB to spare, **just run bf16** — but that is the
right shape if you need the headroom.

⚠️ **No listening test exists.** The bf16 recommendation rests on the
*structural* fact that the vocoder is quantized plus NVIDIA's upstream choice —
not on a measured MOS or WER. The one nearby measurement (Qwen3-TTS at
4/6/8-bit/bf16) reports throughput and memory only and explicitly contains no
audio-quality metric.

Nothing larger exists. Searched `pipeline_tag=any-to-any` and
`audio-text-to-text` with `filter=mlx`: **`Qwen3-Omni-Flash` has zero MLX
results, `Qwen3.5-Omni` does not exist**, there is no 70 B+ omni and no
`GLM-4-Voice` in MLX. `Ming-omni-tts-16.8B-A3B-bf16` (35.82 GB) is **TTS-only**
with no VLM directory, and `mlx-community/Ming-omni-tts-16.8B-A3B-8bit` is
another 1-file/0-byte empty repo. `mlx-community/Step-Audio-2-token2wav`
(0.78 GB, 0 downloads) is an **orphaned vocoder shard** — there is no
`step_audio` module in either mlx-vlm or mlx-audio. You are memory-rich and
model-poor in this class: the ceiling option uses 61% of your budget.

Exhaustively, **only three families in mlx-vlm 0.7.1 can emit audio** —
`qwen3_omni_moe` (`talker.py` + `code2wav.py`), `minicpmo` (`tts.py` +
`vocoder.py`) and `nemotron_voicechat` (`tts.py` + `streaming.py`). Everything
else with an `audio.py` — `gemma4`, `gemma3n`, `phi4mm`, `qwen2_audio`,
`nemotron_h_nano_omni`, `inkling` — is input-only.

### Video — LTX-2.5 at full bf16

Video is the class where "what fits" must be read as **peak**, not weights, and
where the peak is counter-intuitive.

| Option | Peak | Quant | Verdict at 115.4 GB |
|---|---|---|---|
| **LTX-2.5, default (encoder co-resident with DiT)** | **62.40 GB** | bf16 | **The best-quality video option that fits.** ~53 GB spare. |
| LTX-2.5, DiT evicted around encode | **40.66 GB** | bf16 | Same output, **bit-identical**, −34.8% peak. Use this. |
| LTX-2.5, int8 text encoder | 52.18 GB | bf16 DiT + int8 TE | Only needed on 24–32 GB machines. Don't bother here. |
| `Sawfwair/MiniMax-H3-FastH3-VSA-DataFree-MLX-BF16` | **82.32 GB** disk | bf16 | Fits. **Largest self-contained H3 pack.** ⚠️ Loader and resident footprint undocumented. |
| `Sawfwair/MiniMax-H3-FL2VA-MLX-BF16` | **77.09 GB** disk | bf16 | Fits. ⚠️ Same caveat. |
| `ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit` | 70.08 GB disk | 8-bit | Fits, ~45 GB spare. **Video + synchronised stereo audio.** 11–17 min @960×544, 88.2 GiB process peak. |
| `PipeNetwork/minimax-h3-mlx` bf16 (transformer 66.28 GB) | **~102 GB resident** | bf16 | ⚠️ **The true largest MLX video model.** Two structural savings (drop `adaln_proj` after precompute, load only Qwen3-VL-32B layers 0–49) take the pipeline from 144 GB to ~102 GB. But **8.8 min/step** at 5 s/1344×768 on an M3 Ultra — MiniMax never released its sparse attention, so attention is dense and quadratic. |
| `mlx-community/LTX-2-dev-bf16` | 93.33 GB disk | bf16 | Fits but superseded by 2.5. |
| `mlx-community/LongCat-Video-bf16` | 45.75 GB disk | bf16 | Comfortable. T2V + I2V + continuation. |
| `Anes1032/Wan2.2-I2V-A14B-mlx-q8` | 42.68 GB disk | 8-bit | Comfortable, but **slow** — A14B is two 14 B experts, not MoE. |
| `rickylin20260522/Wan2.2-T2V-A14B-mlx` (bf16) | 69.02 GB disk | bf16 | ❌ **Non-contender.** Peaks at only **41.48 GiB**, and its own benchmark says bf16 gives *"no quality win in this seed"* over q8 (11:03 vs 11:17 at 512×320×81f). Don't spend the download. |
| `PocketAiHub/HunyuanVideo-1.5-Distilled-MLX` | 21.08 GB (Q8) | 8-bit | Trivial — but **gated, 0 downloads**, and needs its own bundled `pocketai-video` CLI. No general MLX runtime implements Hunyuan video. |
| MiniMax-H3 unquantized | **125 GiB** | bf16 | ❌ 97% of the machine. `mlx-gen` **refuses the load**. |
| `mlx-community/ltx-2.5-mlx` (the full tree) | 110.08 GB **disk** | bf16 | ⚠️ Not a single model — dev **and** distilled 22 B DiTs (37.99 GB each) plus a bf16 Gemma-4-12B encoder (23.81 GB). Don't pull it blind. |

**Run LTX-2.5 fully bf16 and evict the DiT around the encode phase.** Peak
40.66 GB with output the project measured as *bit-identical* to the naive
62.40 GB path, and wall-clock *"within run-to-run noise"* — on by default in
the Swift consumer. So you get full precision and still use less than half the
ceiling. **Spend the 50 GB of headroom on resolution and frame count, not on
precision.**

⚠️ **Runtime correction:** the consumer for `mlx-community/ltx-2.5-mlx` is
**`xocialize/ltx-2-mlx` (branch `ltx-2.5`)** plus `xocialize/ltx-2-mlx-swift`,
**not** `dgrauet/ltx-2-mlx`. The dgrauet repo is more active and better starred
but its "Pre-converted Weights" table lists **LTX-2.3 only** (bf16 ~42 GB /
int8 ~21 GB / int4 ~12 GB), its default `--model` is `dgrauet/ltx-2.3-mlx-q8`,
and its `## LTX-2.5` heading has an empty body. The 62.40 / 40.66 / 52.18 GB
figures appear only on the mlx-community model card.

**bf16 costs about 1.35× q8 in wall clock, not a multiple.** On an M5 Max
128 GB, a 5-second generation measured **44.9 s at bf16 (42.0 GB peak)** vs
33.3 s at q8 (24.0 GB) — ⚠️ from a social-media snippet I could not fetch
directly, so treat as indicative. The mechanism is documented though: *"the DiT
is compute-bound at these token counts, so the wider weights cost a few percent
of generation time, not a multiple"* and *"quantization here buys FOOTPRINT,
not speed."* For the quality side, the 8-bit DiT reads **cosine 0.99824**
against bf16, while int4 was **rejected** at 0.996728 — *"27× the baseline
angular error. It is not published because it should not be used."*

If you only ever run `--distilled`, you can delete
`transformer-dev.safetensors` (37.99 GB) and take the tree to 72.09 GB on disk.
Keep both for `--two-stage`. There is also
**`xocialize/ltx-2.5-granules`** — the same bytes re-laid-out per block for SSD
streaming, bf16 at 34.58 GiB, output *"bit-identical to the resident path
(memcmp-gated, with a poisoned-slot negative control)"* and *"no wall-clock
penalty at real generation sizes."* The reason the numbers look odd: LTX-2.5's peak is
**encoder-bound, not geometry-bound** — a 256×256×9 clip and a 512×288×121 clip
peak within 0.01 GB of each other, because the floor is the bf16 Gemma-4-12B
text encoder at ~24.4 GB resident. The int4 text encoder was measured and
**rejected** (cosine 0.996728 against a 0.999879 bf16 floor); several
third-party MLX packs ship it at 4-bit with no quality data.

Two cautions carried over from section 8: `wan2.2-ti2v-5b` q8 at 1280×704 peaks
at **103.7 GiB** — inside the ceiling with ~12 GB of margin, so don't run
anything alongside it — and quantized *storage* no longer implies lower runtime
memory for Wan A14B (it runs at BF16-class ~33 GiB peak since a 2026-06-12 fix).

### STT and TTS — going bigger is pointless, and sometimes actively worse

**The 115.4 GB ceiling unlocks nothing in these classes, and it is not close.**
The largest MLX ASR model in existence is 16.67 GB (14% of the ceiling) and is
**measurably worse** than the 2.51 GB model. The largest MLX TTS model is
35.82 GB (31% of the ceiling) and has **never appeared on any quality
leaderboard.** Worse, the curve is non-monotonic — sometimes inverted.

**STT, from the live Open ASR Leaderboard** (the Gradio UI isn't scrapeable but
the backing datasets are — `hf-audio/open-asr-leaderboard-results`,
`english_short_latest.csv`, pulled 2026-09-14). English short-form, average WER
over 8 sets:

| Model | Params | Avg WER | RTFx | vs parakeet-v3 |
|---|---|---|---|---|
| `Qwen/Qwen3-ASR-1.7B-hf` (MLX ✅) | 2.0 B | **4.311** | 820 | **−0.55 (−11.3% rel)** |
| `nvidia/canary-qwen-2.5b` | 2.5 B | 4.428 | 867 | −0.43 (**no MLX conversion**) |
| `nvidia/parakeet-tdt-0.6b-v2` | 0.6 B | 4.703 | 6,025 | −0.16 |
| `ibm-granite/granite-speech-5.0-470m-turboctc` | **0.47 B** | 4.779 | **12,762** | −0.08 |
| **`nvidia/parakeet-tdt-0.6b-v3`** ← the baseline | **0.6 B** | **4.859** | **6,076** | — |
| `mistralai/Voxtral-Small-24B-2507` | **24 B** | 4.994 | 101 | **+0.13 worse** |
| `bosonai/higgs-audio-v3-8b-stt-v2` | 8.9 B | 5.043 | 137 | **+0.18 worse** |
| **`microsoft/VibeVoice-ASR-HF`** (= the 16.67 GB MLX repo) | **8 B** | **5.575** | 219 | **+0.72 worse** |
| `openai/whisper-large-v3` | 2 B | 5.780 | 470 | +0.92 worse |
| `facebook/omniASR-LLM-7B-v2` | **7.8 B** | 6.403 | 129 | **+1.54 worse** |

And the number that ends the argument — **long-form** (earnings21/22, tedlium,
CORAAL), where Parakeet v3 is the best open model on the board:

| Model | Long-form avg | RTFx |
|---|---|---|
| **`parakeet-tdt-0.6b-v3`** | **10.72** | 1,003 |
| `whisper-large-v3-turbo` | 11.01 | 148 |
| **`parakeet-tdt-1.1b`** | **15.81 — +5.09 worse** | 894 |

**The 1.1 B Parakeet is 47% worse than the 0.6 B on long-form.** Spending
4.28 GB instead of 2.51 GB actively hurts you. On multilingual (13 EU sets),
parakeet-v3 at **4.814 beats Qwen3-ASR-1.7B at 5.115** — so the one model that
wins on English loses on everything else. Note also the residual 4.3 → 3.6 WER
gap at the top of the board belongs entirely to **proprietary APIs** (Zoom
Scribe v2 Pro, Azure, ElevenLabs). There is no open model to buy with memory.

**TTS, from the live TTS Arena V2 Elo** (pulled from its API, 41 systems):

| Rank | System | Elo | ± | Open? |
|---|---|---|---|---|
| 1 | Aurora | 1580 | 30 | no |
| 3 | Inworld TTS MAX | 1558 | 20 | no |
| 27 | Eleven v3 | 1502 | 30 | no |
| **28** | **Chatterbox** (2.71 GB in MLX) | **1479** | 19 | yes |
| **30** | **Kokoro v1.0** (0.39 GB in MLX) | **1477** | 25 | **yes** |

**Kokoro at 0.39 GB and Chatterbox at 2.71 GB are a statistical tie** — 2 Elo
apart with overlapping CIs — and Kokoro is the top-ranked open entry, ~100 Elo
behind the best proprietary system. Critically, **VibeVoice, MOSS-TTS,
IndexTTS-2, Qwen3-TTS, Higgs-Audio, VoxCPM2, Ming-omni, Breeze-TTS-2 and
kugelaudio are all absent from the arena** — there is no human-preference
evidence for any of them at any size.

The objective side explains why. From the IndexTTS 2.5 technical report
(arXiv 2601.03888), WER of an ASR system transcribing synthesized speech:

| System | test-en WER |
|---|---|
| IndexTTS 2 | **1.521%** |
| **Ground truth (real human recordings)** | **1.897%** |
| IndexTTS 2.5 (0.8 B) | 1.889% |
| CosyVoice 3 | 2.020% |

**Intelligibility is saturated past human.** The big models transcribe *better
than the original recordings*. There is no naturalness headroom left for
parameters to claim. What separates them from Kokoro is **speaker similarity** —
Kokoro cannot clone a voice at all. That is a capability difference, not a
quality one.

⚠️ The TTSDS benchmark is dead, so don't go looking: the Space's backing
`results.csv` was last modified 2025-04-25 and contains only 2024-era systems
(no Kokoro), and `ttsdsbenchmark.com` no longer resolves.

**The recommended speech stack is 5.6 GB — 4.9% of the ceiling:**

| Role | Repo | Size | Why |
|---|---|---|---|
| Primary ASR | `mlx-community/parakeet-tdt-0.6b-v3` | **2.51 GB** | 4.859 avg WER, RTFx 6,076, **best long-form of any open model (10.72)**. Keep it. |
| Optional ASR | `mlx-community/Qwen3-ASR-1.7B-8bit` | 2.47 GB | −11.3% relative on English short-form, 7.4× slower, worse multilingual |
| Primary TTS | `mlx-community/Kokoro-82M-bf16` | **0.39 GB** | Top open model on the only live Elo board, 54 voices, 8 languages |
| Optional TTS | `mlx-community/chatterbox-multilingual-v3` | 2.71 GB | Ties Kokoro on Elo **and** clones voices in 23 languages — the one upgrade with verified justification, and it is a *capability* upgrade |

**Do not download** VibeVoice-ASR (16.67 GB, measurably worse),
`parakeet-tdt-1.1b` (worse on long-form), `MiMo-V2.5-ASR-bf16` (16.05 GB,
unmeasured), Ming-omni-16.8B (35.82 GB, unmeasured) or MOSS-TTS-8B (10.48 GB,
unmeasured) expecting a quality win. If you want a big TTS model, want it for a
job Kokoro cannot do at all — **long-form multi-speaker dialogue** (VibeVoice,
MOSS-TTSD) or **music and event synthesis** (Ming-omni).

Also worth knowing: **NVIDIA Canary has no `mlx-community` conversion** despite
`mlx_audio/stt/models/canary` existing in code, and
`mlx-community/canary-1b-v2`, `canary-qwen-2.5b` and the bare
`Qwen3-ASR-1.7B` all return HTTP 401 — they do not exist.

**Music and embeddings, for completeness.** `MiniMax-Music3-bf16` at 28.52 GB
is a defensible big-iron pick — the project warns low-bit quantization *"may
alter or omit more requested words"*, so lyric fidelity does track precision —
but mxfp8 at 13.87 GB is its own recommendation and there is nothing at 60 GB
to reach for. `majentik/harrier-oss-v1-27b-MLX-8bit` at **28.73 GB** is the #1
open multilingual embedder and a genuine 128 GB unlock, but for most corpora a
0.65 GB embedder plus a reranker beats a 28.73 GB embedder alone, and domain
fine-tuning beats both (~15 points vs ~4).

### Summary: where the memory actually pays

| Genuinely unlocked by 128 GB | Pointless to scale up |
|---|---|
| **HunyuanImage-3.0** (80 B MoE, 52.45 GiB) — **#1 open-weights *editor*** (Elo 1025.53), impossible under ~70 GB | **STT** — 2.51 GB is the plateau |
| **Cosmos3-Super INT8** (69.54 GB) — 64 B dense, 70.5 s, **commercial licence** | **Image *generation* at 80 B** — Hunyuan is 11th at T2I (964.80); a 26 GB Ideogram 4 scores 1011.97 in 34 s |
| **Qwen3.5-122B-A10B at 5–6 bit** (84.88–100.15 GB) — beats the 235 B on every vision benchmark, at 1/8 the KV cost | **VLM at 3-bit** — buy bit depth and newer architecture, not parameters |
| **Qwen3-Omni bf16** (70.53 GB) — the *only* way to get an unquantized vocoder | **Wan 2.2 A14B bf16** — peaks at 41 GiB and its own benchmark shows no quality win over q8 |
| **bf16 image models** (Qwen-Image / Qwen-Image-Edit at ~56 GB) — no quantization loss at all | **TTS** — size buys control, not fidelity |
| **LTX-2.5 fully bf16** (40.66–62.40 GB peak) | **Video >8 s or >720p** — hours, regardless of memory |
| **MiniMax-H3 8-bit** (70.08 GB) — video with synced audio | **Music** — mxfp8 at 13.87 GB is already the recommendation |
| **harrier-oss-v1-27b** (28.73 GB) — #1 open multilingual embedder | **Anything peaking above ~100 GiB** — Cosmos3 bf16 at 118.55 GiB peak ran **10.4× slower** than INT8 at 67.63 GiB |

---

---

## 1. Image generation

> Best-that-fits: **`mflux-community/krea-2-turbo-mflux-bf16`, 34.20 GB**
> (T2I Elo 1017.99) or `ideogram-4-mflux-q8` at 26 GB (1011.97, **34 s measured**).
> Do **not** use the 80 B HunyuanImage for generation — it ranks 11th at T2I.
> See **[Big-iron picks](#big-iron-picks--using-all-1154-gb)**.

`mflux` is the whole story. Version **0.19.2, uploaded 2026-09-17** (the
capture date), 2.3k stars, 17 model families in-tree, MLX-native. It pins
`mlx>=0.32.0,<0.33.0` — which matches the installed 0.32.2 exactly.

**The repo moved: `filipstrand/mflux` → `mflux-community/mflux`.** PyPI
metadata still points at the old URL. Old links redirect.

Before trusting any table in this file, run **`mflux-capabilities`** — it emits
a machine-readable JSON/YAML dump of exactly which flags each CLI honours,
ignores, or rejects, generated live from the installed parsers. That is ground
truth; docs drift (see the 2509/2511 trap in the next section).

| Model repo id | Size | Quantization | Runtime | MLX-native? | Memory verdict |
|---|---|---|---|---|---|
| `mlx-community/FLUX.2-Klein-4B-4bit` | **4.62 GB** | 4-bit | `mflux-generate-flux2` | ✅ mflux | Trivial. 4 steps. Best size/quality ratio here. |
| `mlx-community/FLUX.2-Klein-4B-6bit` | 6.59 GB | 6-bit | `mflux-generate-flux2` | ✅ mflux | Trivial. Use this over 4-bit — 2 GB is free at 115.4 GB. |
| `mlx-community/FLUX.2-klein-4B-bf16` | 23.74 GB | bf16 | `mflux-generate-flux2` | ✅ mflux | Comfortable. Reference quality for the 4B. |
| `mlx-community/flux2-klein-4b-8bit` | 8.57 GB | 8-bit | `mflux-generate-flux2` | ✅ mflux | Trivial. Apache-2.0, 4 steps. The sweet spot. |
| `mlx-community/flux2-klein-9b-8bit` | 17.87 GB | 8-bit | `mflux-generate-flux2` | ✅ mflux | Comfortable. Pre-quantized 9B. **Non-commercial.** |
| `mlx-community/FLUX.2-klein-9B` | 52.88 GB | bf16 | `mflux-generate-flux2` | ✅ mflux | Fits, ~62 GB spare. **Non-commercial licence.** |
| `mlx-community/Qwen-Image-2512-4bit` | 25.91 GB | 4-bit | `mflux-generate-qwen` | ✅ mflux | Comfortable. Most-downloaded MLX image model. |
| `mlx-community/Qwen-Image-2512-8bit` | 36.12 GB | 8-bit | `mflux-generate-qwen` | ✅ mflux | Comfortable. 20 B model — slow but best prompt adherence. |
| `mflux-community/qwen-image-2512-mflux-bf16` | **55.27 GB** | **bf16** | `mflux-generate-qwen` | ✅ mflux | Fits, ~60 GB spare. **Highest-quality Apache-2.0 generator.** See *Big-iron picks*. |
| `mflux-community/krea-2-turbo-mflux-bf16` | 34.20 GB | bf16 | `mflux-generate-krea2` | ✅ mflux | Comfortable. 12 B turbo. `-q8` is 22.19 GB. |
| `mflux-community/ideogram-4-mflux-bf16` | 26.96 GB | bf16 | `mflux-generate-ideogram4` | ✅ mflux | Comfortable. Typography-focused; the upstream repo is gated, this one is not. |
| `mlx-community/Z-Image-Turbo-bf16` | 20.54 GB | bf16 | `mflux-generate-z-image-turbo` | ✅ mflux | Comfortable. 6 B, 9 steps, Apache-2.0. |
| `mlx-community/Lens-Turbo-3.8B-4bit` | 2.35 GB | 4-bit | `mflux-generate-lens` | ✅ mflux | Trivial — but pulls a **20 B GPT-OSS text encoder** separately. |
| `Runpod/FLUX.2-klein-4B-mflux-4bit` | 4.62 GB | 4-bit | `mflux-generate-flux2` | ✅ mflux | Trivial. **76,607 downloads — the most-used MLX FLUX.2 repo.** |
| `mlx-community/Qwen-Image-Flash-8bit` | 29.53 GB | 8-bit | see note | ⚠️ tagged `mlx-swift` | Comfortable, but the card points at mlx-swift, not mflux. Unverified from Python. |
| `mlx-community/SeedVR2-3B-mlx-int8` | 4.72 GB | int8 | `mflux-upscale-seedvr2` | ✅ mflux | Trivial. Upscaler, not a generator. |

**Where to get weights.** There is now an official **`mflux-community` HF org**
with 200 repos: every mflux model at `q3 / q4 / q5 / q6 / q8 / bf16`, one
quantization per repo, in `mflux-save` format (`manifest.json` +
`transformer/N.safetensors` + `text_encoder/N.safetensors`), loadable directly
with `--model <repo-id>`. Prefer it over `mlx-community` for mflux work — it is
the only source that publishes **bf16**, and it covers families that have no
`mlx-community` conversion at all: Krea 2 (turbo and raw), Ideogram 4,
ERNIE-Image (base and turbo), FIBO (+ lite, edit, edit-rmbg), Boogu Turbo,
FLUX.1-dev (+ Kontext, Fill, Depth, Redux, ControlNet, CatVTON), FLUX.1-schnell,
FLUX.1-Krea-dev, and the Z-Image ControlNet.

Two notes on it: the repos carry **no model card** (so they look empty on the
Hub and their HF `tags` are blank — that is normal, not broken), and
`fibo-edit-1-5-base-mflux-{q6,q8}` exists there even though mflux `main` has no
`fibo-edit-1.5` entry in `model_config.py` — load it with
`--model <repo> --base-model fibo-edit`.

**FLUX.2 family, for context.** The open line is `klein`; `pro`/`flex`/`max`
are API-only. Parameter counts and weight sizes:

| Variant | Open | Licence | Params | Weights | mflux? |
|---|---|---|---|---|---|
| klein 4B (4-step distilled) | ✅ | **Apache-2.0**, ungated | 3.88 B | 15.97 GB | ✅ default |
| klein 9B (4-step distilled) | ✅ | FLUX non-commercial | 9.08 B | 34.7 GB | ✅ |
| klein-9b-kv (KV-cache edit) | ✅ | FLUX non-commercial | 9.08 B | 34.7 GB | ✅ |
| klein base 4B / base 9B | ✅ | Apache / NCL | 3.88 / 9.08 B | 15.97 / 34.7 GB | ✅ (training) |
| **dev** | ✅ | FLUX non-commercial | **32.2 B** | **112.8 GB** | ❌ not in mflux — but see below |
| pro / flex / max | ❌ API | proprietary | — | — | ❌ |

FLUX.2 dropped T5 and CLIP entirely: klein-4B's text encoder is **Qwen3-4B**,
klein-9B's is **Qwen3-8B**, dev's is a **Mistral-3 24B VLM**. Component sums for
dev: transformer 64.45 GB + text encoder 48.02 GB + VAE 0.34 GB = **112.81 GB**.

**Correction: FLUX.2-dev does run on MLX, and the memory fits.** mflux does not
implement it — verified in `model_config.py`, which has only the five
`flux2-klein-*` entries; tracking issue **#707** names the two gaps (no
`ModelConfig.flux2_dev()`, and no all-MLX Mistral3 text encoder). But
**`VincentGourbin/flux-2-swift-mlx`** (Swift / mlx-swift, MIT, 40 stars, pushed
2026-09-12) generates end to end, and because the text encoder is staged
separately you never hold 112.8 GB at once. Measured on an **M2 Ultra 96 GB**,
1024², 28 steps: **peak 71,930 MiB (~70.2 GiB), 1758.6 s**.

So it fits comfortably. **It is still not worth running**: 29 minutes per image,
Elo exactly 1000 on both boards — *below* Krea 2, Ideogram 4, Qwen-Image-Edit-2511
and klein-9B — under a non-commercial licence. Skip it for quality reasons, not
memory ones.

**Picks.** `FLUX.2-Klein-4B-6bit` for iteration speed — 6.6 GB and 4 steps
means you can loop on prompts in near real time, and it's Apache-2.0.
`Qwen-Image-2512-8bit` when you need the prompt actually followed, or Chinese
text rendered in-image. `Z-Image-Turbo` for Apache-2.0 at 9 steps.

**Tradeoffs.**
- FLUX.1 is not just "legacy" informally — its own mflux README says *"We do
  not plan to add new features or expand support for FLUX.1 beyond maintenance
  fixes, and FLUX.1 may be deprecated in future releases."* Don't start there.
- **Licensing trap:** only `klein-4B` and `klein-base-4B` are Apache-2.0.
  `klein-9B`, `klein-9b-kv`, `klein-base-9B`, `dev` and **all of FLUX.1** are
  non-commercial. Qwen-Image, Z-Image and Boogu are Apache-2.0.
- **Quantization levels are `{3,4,5,6,8}`** (no 2-bit). Qwen's own mflux README
  warns *"6-bit or below can degrade the image a lot more compared to Flux"* —
  so quantize Flux aggressively and Qwen conservatively.
- `mflux-generate-lens` pulls a **20 B GPT-OSS text encoder** on top of the
  3.8 B transformer. The 2.35 GB figure is the transformer only.
- **LoRA flags were removed from `mflux-generate-boogu`, `-fibo` and
  `-fibo-edit` in 0.19.0** — they previously accepted and silently discarded
  them. `mflux-generate-lens` has none either.
- Memory controls beyond `--low-ram`: `--mlx-cache-limit-gb`, `--vae-tiling`,
  `--vae-tile-size`.

**Do not install `mlx-gen` and `mflux` in the same environment.** `mlx-gen`
(0.37.0) is a fork of mflux that still uses the `mflux` import namespace and
installs **colliding `mflux-*` console scripts**. It adds some things mflux
lacks (`mflux-generate-wan` for Wan 2.2 video, masked inpaint) and is missing
others (boogu, krea2, ideogram4, lens, fibo-edit, seedvr2). One maintainer.
If you want it, give it its own env.

---

## 2. Image editing — the local "Nano Banana"

> Largest-that-fits: **`mflux-community/qwen-image-edit-2511-mflux-bf16`, 56.62 GB**
> (Apache-2.0, any resolution), or **HunyuanImage-3.0-Instruct** at 6-bit
> (52.45 GiB, **#1 open-weights editing Elo 1025.53**, but 1 MP only, 4.2 min/edit,
> and the licence restricts the *output images* by territory).
> See **[Big-iron picks](#the-one-thing-128-gb-genuinely-unlocks-hunyuanimage-30)**.

This is the class that changed most in 2026. The answer is
**`mflux-generate-qwen-edit` with `Qwen-Image-Edit-2511`** — MLX-native,
Apache-2.0, instruction-driven, multi-reference, LoRA-capable. `FLUX.1 Kontext`
— the thing everyone points at — is now the *third* choice here: legacy family,
non-commercial licence.

### ⚠️ Read this before running `mflux-generate-qwen-edit`

**With no `--model`, it downloads 2509, not 2511.** The prose in
`src/mflux/models/qwen/README.md` says *"The model uses
`Qwen/Qwen-Image-Edit-2511`"*, but `model_config.py:568` sets
`model_name="Qwen/Qwen-Image-Edit-2509"`, and the `qwen-edit-2511` /
`qwen-image-edit-2511` names are **aliases onto that same 2509 config**. A
third README warning still cites the *"~58 GB Qwen-Image-Edit-2509"* download.
No open issue tracks it. To actually get 2511 you must point `--model` at a
2511 checkpoint. The cleanest source is the **official `mflux-community` HF
org**, which publishes a pre-converted `mflux-save` ladder
(`q3/q4/q5/q6/q8/bf16`) for every mflux model, one quantization per repo:

```sh
# 8-bit, 37.47 GB — the everyday 2511 editor
mflux-generate-qwen-edit \
  --model mflux-community/qwen-image-edit-2511-mflux-q8 --base-model qwen \
  --image-paths input.jpg --prompt "..." --steps 30 --guidance 2.5

# bf16, 56.62 GB — the best-quality 2511 editor. See "Big-iron picks".
mflux-generate-qwen-edit \
  --model mflux-community/qwen-image-edit-2511-mflux-bf16 --base-model qwen \
  --image-paths input.jpg --prompt "..." --steps 30 --guidance 2.5
```

`fcreait/Qwen-Image-Edit-mflux` also works and is the most-downloaded community
option, but it holds q3–q8 in **one 151.62 GB repo**, so you must fetch a single
subfolder. The `mflux-community` repos are one-quant-per-repo and need no
subfolder juggling.

Also: the two `mlx-community/qwen-image-edit-*-8bit` model cards print a
**wrong command** (`mflux-generate-controlnet --model ...`). Copy-paste error.
The entry point is `mflux-generate-qwen-edit`.

| Model repo id | Size | Quantization | Runtime | MLX-native? | Memory verdict |
|---|---|---|---|---|---|
| `mflux-community/qwen-image-edit-2511-mflux-bf16` | **56.62 GB** | **bf16** | `mflux-generate-qwen-edit` | ✅ mflux | Fits, ~59 GB spare. **Best-quality Apache-2.0 editor.** See *Big-iron picks*. |
| `mflux-community/qwen-image-edit-2511-mflux-q8` | **37.47 GB** | 8-bit | `mflux-generate-qwen-edit` | ✅ mflux | Comfortable. **The everyday pick.** Official mflux org, genuinely 2511. |
| `mflux-community/qwen-image-edit-2511-mflux-q6` | 32.37 GB | 6-bit | `mflux-generate-qwen-edit` | ✅ mflux | Comfortable — but mflux warns Qwen degrades hard at ≤6-bit. Prefer q8. |
| `fcreait/Qwen-Image-Edit-mflux` → `Qwen-Image-Edit-2511-q8` | ~35 GB (repo total **151.62 GB** — holds q3–q8 side by side, so fetch the **one subfolder**) | 8-bit | `mflux-generate-qwen-edit` | ✅ mflux | Comfortable. Most-downloaded community option (5,047). Ships the VisionTransformer weights editing needs. |
| `mlx-community/qwen-image-edit-2511-8bit` | 37.47 GB | 8-bit | `mflux-generate-qwen-edit` | ✅ mflux | Comfortable. Same checkpoint, wrong command on the card. |
| `mlx-community/qwen-image-edit-2509-8bit` | 37.47 GB | 8-bit | `mflux-generate-qwen-edit` | ✅ mflux | Comfortable. Older checkpoint; this is what the default pulls. |
| `KaedeTai/HunyuanImage-3.0-mlx-mixed4-8-hymlx` | **56.3 GB** | mixed 4/8-bit | `KaedeTai/hymlx` | ✅ MLX (third-party) | Fits — **69.5 GiB peak during editing**, benchmarked on *this exact machine*. Tops the open-weights editing leaderboard. Licence **excludes EU/UK/South Korea**. |
| `mlx-community/SenseNova-U1.5-8B-MoT-8bit` | 19.9 GB | 8-bit | `xocialize/sensenova-u1-swift` | ⚠️ MLX-**Swift**, alpha | Comfortable — 22.9 GB peak, **~7.4 s at 1024²**. **Apache-2.0.** Fastest real editor here. |
| `mlx-community/FLUX.2-Klein-4B-4bit` | 4.62 GB | 4-bit | `mflux-generate-flux2-edit` | ✅ mflux | Trivial. Fastest mflux edit loop. Apache-2.0. |
| `mlx-community/flux2-klein-4b-8bit` | 8.57 GB | 8-bit | `mflux-generate-flux2-edit` | ✅ mflux | Trivial. Take this over the 4-bit — Apache-2.0 and still 4 steps. |
| `mlx-community/flux2-klein-9b-8bit` | 17.87 GB | 8-bit | `mflux-generate-flux2-edit` | ✅ mflux | Comfortable. Pre-quantized 9B, saves the 34.7 GB pull. |
| `black-forest-labs/FLUX.2-klein-9b-kv` | 34.7 GB upstream | bf16 (use `-q`) | `mflux-generate-flux2-edit --model flux2-klein-9b-kv` | ✅ mflux | Fits. **~2.4× faster on multi-reference edits** (KV cache auto-enables with `--image-paths`). Non-commercial. |
| `mlx-community/FLUX.2-klein-9B` | 52.88 GB | bf16 | `mflux-generate-flux2-edit` | ✅ mflux | Fits. Non-commercial. Quantize with `-q 8`. |
| `akx/FLUX.1-Kontext-dev-mflux-4bit` | **9.61 GB** | 4-bit | `mflux-generate-kontext` | ✅ mflux | Trivial. The **only** mflux-format Kontext conversion anywhere. Non-commercial. |
| `black-forest-labs/FLUX.1-Kontext-dev` | 33.8 GB distinct | bf16 (use `-q`) | `mflux-generate-kontext` | ✅ mflux | Comfortable. 11.9 B. **Non-commercial**, gated, legacy family. |
| `mlx-community/HiDream-O1-Image-Dev-mlx-bf16` | **17.65 GB** | bf16 | tagged `mlx-vlm`; card points at `mrbizarro/phosphene` | ⚠️ third-party MLX | Comfortable. 8.8 B, **MIT**. Card says **editing requires bf16** — the q6/q8 are T2I-only. |
| `briaai/Fibo-Edit` / `Fibo-Edit-RMBG` | 8.3 B upstream | use `-q` | `mflux-generate-fibo-edit` | ✅ mflux | Comfortable. JSON-prompt editing, `--mask-path`, transparent-PNG cutout mode. Custom `bria-fibo-edit` licence. |
| `mlx-community/Boogu-Image-0.1-Edit-{4bit,bf16}` | 8.24 / 20.92 GB | 4-bit / bf16 | ❌ **not mflux** | ⚠️ alpha third-party only | See the warning below. |
| `mlx-community/TeleStyleV2-Qwen-Image-Edit-2511-bf16` | not sized | bf16 | presumed mflux | ⚠️ unverified | Community style fine-tune of 2511. Not checked. |

**Where quality actually sits** (Artificial Analysis image-editing Elo,
open weights): HunyuanImage 3.0 Instruct **1028** > Qwen-Image-Edit 2511
**1023** > FLUX.2 klein 9B **1013** > FLUX.2 dev **1000**. The top two are
within noise of each other, and Qwen is a quarter the size and Apache-2.0 —
which is why it's the pick and Hunyuan is the "because I have 128 GB" option.

**Boogu Edit does not run under mflux.** Worth stating plainly because the
model name and the `mflux-generate-boogu` command make it look like it should:
`mflux-generate-boogu` covers **Turbo (text-to-image) only**.
`src/mflux/models/boogu/variants/` contains only `txt2img/`, with no `edit/`,
and the main README's Boogu row omits the "has edit capabilities" note the
FLUX.2/FIBO/Qwen/FLUX.1 rows carry. The `mlx-community/Boogu-Image-0.1-Edit-*`
weights load via `xocialize/boogu-image-mlx`, which is **1 star, 3 commits all
on one day, no licence file, not on PyPI**, and whose own README says the Edit
variant is still "in progress" — contradicting the confident snippet on the
model card. Boogu is OmniGen2/Lumina/NextDiT lineage: ~10.3 B DiT +
Qwen3-VL-8B conditioner + FLUX.1 VAE. Skip it.

Beyond whole-image instruction editing, mflux also ships the surrounding
toolkit, all MLX-native:

| Task | Command | Notes |
|---|---|---|
| Inpaint / fill | `mflux-generate-fill` | Pulls `FLUX.1-Fill-dev`, **33.92 GB** extra |
| Depth-conditioned | `mflux-generate-depth` | Pulls `FLUX.1-Depth-dev`, **33.92 GB** extra |
| Depth map only | `mflux-save-depth` | Apple Depth Pro, 1.9 GB |
| Style/subject transfer | `mflux-generate-redux` | `FLUX.1-Redux-dev`, 1.1 GB extra |
| ControlNet (Canny) | `mflux-generate-controlnet` | |
| In-context edit | `mflux-generate-in-context-edit` | |
| Virtual try-on | `mflux-generate-in-context-catvton` | |
| Upscale | `mflux-upscale-seedvr2`, `mflux-upscale-controlnet` | SeedVR2 is the better one |
| JSON-prompt edit | `mflux-generate-fibo-edit` | FIBO, plus `mflux-refine-fibo` / `mflux-inspire-fibo` VLM helpers |
| LoRA training | `mflux-train` | Supports FLUX.1, FLUX.2 (incl. **edit training**), Z-Image |

**What this does not match.** Gemini's image editing is a single model doing
conversational, multi-turn, world-knowledge-grounded edits. Qwen-Image-Edit is
one-shot instruction editing: you get the edit, not the conversation. mflux's
Kontext docs describe *sequential* editing (feed the output back in) as the
workaround, and it works, but you are managing the loop yourself. FLUX.2 is the
closer analogue architecturally — every variant takes up to **10 reference
images** at up to 4 MP and edits natively — but its open tier is the small
klein models.

**No MLX path at all, despite being well-known:** **Step1X-Edit** and
**Step1X-Edit-v1p2** (12.4 B, Apache-2.0), **OmniGen2**, **BAGEL-7B-MoT**,
**HiDream-E1 / E1-1** (both abandoned upstream since Jul 2025),
**LongCat-Image-Edit**, **FLUX.2-dev**, FireRed-Image-Edit, JoyAI-Image-Edit.
Don't go looking — Qwen-Image-Edit-2511 supersedes all of them on the
leaderboard anyway.

**There is no Qwen-Image-Edit-2512.** The Qwen image line is exactly seven
repos, all Apache-2.0, all 20.43 B: `Qwen-Image` → `Qwen-Image-Edit` →
`-Edit-2509` → `-Edit-2511` + `Qwen-Image-Layered` → `Qwen-Image-2512`
(**text-to-image only**). 2511, from Dec 2025, is still the newest editor. Any
claim of a "Qwen-Image 2.0" in Feb 2026 is not on HuggingFace — treat as false.

**There is no Kontext v2.** BFL's own docs now say *"For new projects, we
recommend FLUX.2."*

### The non-MLX image runtimes, honestly assessed

**Draw Things — install it as a hedge.** It is a native Mac/iOS app (App Store
v26.0914.0; community repo GPL-3.0, pushed 2026-09-17) and it is **not MLX**:
it runs its own hand-written Metal via `liuliu/ccv` + `s4nnc`, with CoreML/ANE
as an optional secondary path. That independence is exactly why it's worth
having. It is also **the only runtime with published M5-Max-specific
optimisation work**: a vendor engineering post on custom Metal Flash Attention
plus online Int8 attention quantization reports **1.61–1.87× over FP16** for
fused Int8 matmul, *"around 110 TFLOPs on M5 Max"*, and **3.3× over M4 Max**,
explicitly exploiting the M5 Neural Accelerators. Nothing in the MLX world
publishes anything comparable for this chip.

Its model zoo carries Qwen-Image-Edit **1.0 / 2509 / 2511**, FLUX.2 klein 4B,
Kontext, Z-Image, Krea 2, Ideogram 4, ERNIE-Image, HiDream I1/O1, **LTX-2/2.3,
Wan 2.1/2.2**, SeedVR2 and Cosmos 2.5, with q4p/q5p/q6p/q8p/i8x/SVDQuant
quantization and on-device LoRA training. It is scriptable — `draw-things-cli`
(`brew install --HEAD drawthingsai/draw-things/draw-things-cli`) has `generate`,
`models list/ensure/import` and `train lora`, plus a gRPC server binary. Both
were made macOS-native in v26.0910.1. *Unverified:* whether the shipping app
exposes FLUX/Qwen-architecture ControlNets — the synced zoo only has SD1.x/SDXL
ones.

**ComfyUI on macOS — last resort, and there is a specific reason.** v0.36.0
(2026-09-15), repo now `Comfy-Org/ComfyUI`. The models are in-tree
(`comfy/ldm/qwen_image/`, `comfy/ldm/flux/`) and there are official blueprints
for Qwen 2509/2511 and FLUX.2. But:

- 🔴 **Issue #14837 corrupts output silently, and only on big-memory Macs.**
  MPS uses 32-bit indexing, so once `batch*heads*seq_q*seq_k` approaches 2³¹
  the attention result is wrong with **no crash and no warning**. The existing
  memory-pressure chunking doesn't catch it because it exists to avoid OOM —
  and *"on a machine with a lot of unified memory it just computes the whole
  oversized matrix in one shot."* PR #14838 is **unmerged**. This is a
  128 GB-specific footgun.
- **fp8 never works.** `supports_fp8_compute()` returns False off NVIDIA;
  issue #16107 is a `TORCH_CHECK`, so `PYTORCH_ENABLE_MPS_FALLBACK=1` does not
  help. That rules out the standard fp8-scaled FLUX.2 and Qwen distributions —
  you must source bf16 or GGUF.
- **No offload on MPS by design:** `model_management.py:594` forces
  `VRAMState.SHARED`, taking the HIGH_VRAM path and skipping all
  partial-offload.
- Int8 weight-only quant is unavailable (`torch._int_mm` unimplemented), and
  there is **no fast attention kernel** — `sageattention`/`flash_attn` are
  CUDA-only, so you must pass `--use-pytorch-cross-attention` manually.
- 86 open issues match `MPS in:title`.
- **MLX inside ComfyUI is effectively dead:** `raysers/Mflux-ComfyUI` is 18
  months stale, `thoddnn/ComfyUI-MLX` is deleted, and there is no official MLX
  backend.

ComfyUI is still the only working route to **Boogu-Edit** (via
`Comfy-Org/Boogu-Image`, bf16 only). That's the one reason to reach for it.

---

## 3. VLM / image understanding

> Best-that-fits: **`mlx-community/Qwen3.5-122B-A10B-5bit`, 84.88 GB** — it beats
> Qwen3-VL-235B on every vision benchmark at 19 GB less, and its hybrid attention
> costs **24 KiB/token of KV against the 235 B's 188**. See
> **[Big-iron picks](#vlm--qwen35-122b-a10b-at-5-bit-and-the-reason-is-the-kv-cache)**.

`mlx-vlm` **0.7.1 (2026-09-14)** is MLX-native and has **240 entries** in
`mlx_vlm/models/`. Qwen3-VL is fully implemented (`qwen3_vl`, `qwen3_vl_moe`,
`qwen3_vl_embedding`), as are `glm_ocr`, `internvl_chat`, `gemma4`,
`paddleocr_vl`, `deepseekocr`, `dots_ocr`, `molmo2`, `moondream3` and many more.

| Model repo id | Size | Quantization | Runtime | MLX-native? | Memory verdict |
|---|---|---|---|---|---|
| `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit` | **18.27 GB** | 4-bit | `mlx-vlm` | ✅ `qwen3_vl_moe` | Comfortable. **The pick** — 30 B/3 B active, so fast. |
| `mlx-community/Qwen3-VL-30B-A3B-Instruct-8bit` | 33.53 GB | 8-bit | `mlx-vlm` | ✅ | Comfortable. Take this over 4-bit; vision heads are quant-sensitive. |
| `mlx-community/Qwen3-VL-30B-A3B-Thinking-4bit` | 18.27 GB | 4-bit | `mlx-vlm` | ✅ | Comfortable. For charts/diagrams needing reasoning. |
| `mlx-community/Qwen3-VL-32B-Instruct-4bit` | 19.64 GB | 4-bit | `mlx-vlm` | ✅ `qwen3_vl` | Comfortable. **Dense** 32 B — slower than the MoE, more predictable. |
| `mlx-community/Qwen3-VL-8B-Instruct-8bit` | 9.87 GB | 8-bit | `mlx-vlm` | ✅ | Trivial. Good battery option. |
| `mlx-community/Qwen3-VL-4B-Instruct-4bit` | 3.11 GB | 4-bit | `mlx-vlm` | ✅ | Trivial. Most-downloaded MLX VLM (21.7k). |
| `mlx-community/GLM-OCR-4bit` | **1.25 GB** | 4-bit | `mlx-vlm` | ✅ `glm_ocr` | Trivial. 56.4k downloads. Best GB-per-usefulness in this doc. |
| `mlx-community/GLM-OCR-8bit` | 1.59 GB | 8-bit | `mlx-vlm` | ✅ | Trivial. Just take the 8-bit. |
| `mlx-community/PaddleOCR-VL-1.5-4bit` | 0.72 GB | 4-bit | `mlx-vlm` | ✅ `paddleocr_vl` | Trivial. Pure OCR, no chat. |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 15.37 GB | 4-bit | `mlx-vlm` | ✅ `gemma4` | Comfortable. Vision ✅, **audio ❌**. |
| `mlx-community/gemma-4-31b-it-4bit` | 18.44 GB | 4-bit | `mlx-vlm` | ✅ | Comfortable. Dense, vision ✅, audio ❌. |
| `mlx-community/gemma-4-e4b-it-4bit` | 5.18 GB | 4-bit | `mlx-vlm` | ✅ | Trivial. Only Gemma 4 tier with **vision + audio**. |
| `mlx-community/InternVL3_5-30B-A3B-4bit` | 17.81 GB | 4-bit | `mlx-vlm` | ✅ `internvl_chat` | Comfortable. Different family for second opinions. |
| `mlx-community/GLM-4.6V-6bit` | **88.59 GB** | 6-bit | `mlx-vlm` | ✅ `glm4v_moe` | Fits, ~27 GB spare. **MIT.** The big-iron VLM pick. |
| `mlx-community/GLM-4.6V-4bit` | 61.88 GB | 4-bit | `mlx-vlm` | ✅ `glm4v_moe` | Comfortable. MIT. |
| `mlx-community/GLM-4.6V-Flash-8bit` | 11.79 GB | 8-bit | `mlx-vlm` | ✅ `glm4v` | Trivial. The small GLM-4.6V — 95.6k downloads on the lmstudio mirror. |
| `mlx-community/Qwen2.5-VL-72B-Instruct-8bit` | 78.02 GB | 8-bit | `mlx-vlm` | ✅ `qwen2_5_vl` | Fits, ~37 GB spare. 72 B dense, but an older generation. |
| `mlx-community/InternVL3-78B-4bit` | 52.08 GB | 4-bit | `mlx-vlm` | ✅ `internvl_chat` | Comfortable. Largest dense VLM with real headroom. |
| `mlx-community/Qwen3-VL-235B-A22B-Instruct-3bit` | 104.02 GB | 3-bit | `mlx-vlm` | ✅ | ⚠️ **Fits with ~11 GB headroom.** 3-bit vision is lossy. Don't. |
| `mlx-community/Qwen3-VL-235B-A22B-Instruct-4bit` | 133.41 GB | 4-bit | `mlx-vlm` | ✅ | ❌ **Exceeds the 115.4 GB ceiling.** Will not load. |

The Gemma 4 vision/audio matrix is from the in-tree
`mlx_vlm/models/gemma4/README.md`, which is explicit that only the 2B/4B tiers
do audio — the 26B-A4B and 31B are vision-only. That is the opposite of what
you'd assume from the parameter count.

**Devstral vision:** the Devstral Small 2 in `MODEL-COMPARISON.md` is
text-only; `mlx-vlm` does carry `mistral3`/`mistral4` for the Mistral vision
line, but that is not the Devstral coding model. No vision path for Devstral.

**Pick:** `Qwen3-VL-30B-A3B-Instruct-8bit` (33.53 GB) as the general VLM, plus
`GLM-OCR-8bit` (1.59 GB) resident permanently because it costs nothing.

---

## 4. Omni / any-to-any

> Largest-that-fits: **`Qwen3-Omni-30B-A3B-Instruct-bf16`, 70.53 GB** — and bf16 is
> the best-justified upgrade in this document, because 4-bit wrecks the vocoder.
> See **[Big-iron picks](#omni--qwen3-omni-bf16-is-the-ceiling-and-its-the-right-call)**.

**The model you named exists, and it runs.** `omni-nemotron-30b` is
`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning` — 31 B total, ~3 B active,
Mamba2-Transformer hybrid MoE. `mlx-vlm` implements it as
`mlx_vlm/models/nemotron_h_nano_omni`, and `mlx-community` has it at 4-bit for
19.65 GB. **But its output is text only.** Per NVIDIA's own card: "Modalities
(in): Video, Audio, Image, Text / Modality (out): Text". It is a
listener-and-watcher, not a speaker.

For **actual any-to-any including speech out**, the answer is Qwen3-Omni, and
it is the only one: a GitHub code search confirms `talker.py` and
`code2wav.py` exist in exactly one directory in mlx-vlm —
`mlx_vlm/models/qwen3_omni_moe`. The mlx-vlm README additionally documents
`--output-modality audio` with MiniCPM-o-4_5.

| Model repo id | Size | Quantization | Runtime | MLX-native? | Memory verdict |
|---|---|---|---|---|---|
| `mlx-community/MiniCPM-o-4_5-4bit` | **6.16 GB** | 4-bit | `mlx-vlm` | ✅ `minicpmo` (`tts.py` + `vocoder.py`) | Trivial. **Start here.** Apache-2.0, audio in + **cloned-voice audio out** + vision. |
| `mlx-community/MiniCPM-o-4_5-5bit` | 7.19 GB | 5-bit | `mlx-vlm` | ✅ | Trivial. |
| `mlx-community/Qwen3-Omni-30B-A3B-Instruct-8bit` | **38.78 GB** | 8-bit | `mlx-vlm` | ✅ `qwen3_omni_moe` (thinker **+ talker + code2wav**) | Comfortable. Highest ceiling. Audio+video+text in, audio+text out. |
| `mlx-community/Qwen3-Omni-30B-A3B-Instruct-bf16` | 70.53 GB | bf16 | `mlx-vlm` | ✅ | Fits, ~45 GB spare. Use if voice fidelity is the point. |
| `mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit` | 21.84 GB | 4-bit affine g64 | `mlx-vlm` | ✅ | Comfortable — but see the 4-bit warning below. |
| `mlx-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-mxfp4` | **18.64 GB** | MXFP4 | `mlx-vlm` | ✅ `nemotron_h_nano_omni` | Comfortable. Audio/video/image **in**, text **out only**. |
| `mlx-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit` | 19.65 GB | 4-bit | `mlx-vlm` | ✅ | Comfortable. |
| `mlx-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-8bit` | 35.79 GB | 8-bit | `mlx-vlm` | ✅ | Comfortable. The one to actually run. |
| `mlx-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-bf16` | 66.05 GB | bf16 | `mlx-vlm` | ✅ | Fits, ~49 GB spare. Little reason over 8-bit. |
| `mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit` | 20.67 GB | 4-bit | `mlx-vlm` | ✅ | Comfortable. Non-reasoning variant. |
| `mlx-community/NemotronLabs-VoiceChat-11B-4bit` | 9.18 GB | 4-bit | `mlx-vlm` `/v1/realtime` + `mlx-audio` STS | ✅ `nemotron_voicechat` | Trivial. **Full-duplex** voice with function calling. |
| `mlx-community/LFM2.5-Audio-1.5B-8bit` | 2.44 GB | 8-bit | `mlx-audio` STS | ✅ `lfm_audio` | Trivial. Tiny speech-to-speech. |

**What each gets you.**
- **MiniCPM-o-4.5** — 9 B (Qwen3-8B + Whisper-medium in + SigLip2 + CosyVoice2
  out). **Apache-2.0**, the only permissive licence among the true omnis, and
  the only model wired into mlx-vlm's *ergonomic* audio-output API. 24 kHz
  speech out with voice cloning, real-time full-duplex, 30+ languages. At
  6.16 GB this is the cheapest way to get a talking model on this machine by a
  factor of six.
- **Qwen3-Omni-30B-A3B-Instruct** — 119 text languages, 19 speech-input, 10
  speech-output. Thinker–Talker MoE, 128 experts / 8 active, 65,536 ctx, three
  built-in speakers (`chelsie`, `ethan`, `aiden`). `disable_talker()` saves
  ~10 GB if you only need text out. Note that *upstream* recommends vLLM and
  says audio-out there is still pending — mlx-vlm has the full
  talker → code2wav path in-tree, a rare case of the Mac path being ahead of
  the CUDA path.
- **Nemotron-3-Nano-Omni** — the better *analyst*: **256 k context**, mp4 up to
  2 minutes (1 fps/128 frames at 1080p), **audio up to one hour**, plus
  OCR/chart/GUI strengths and word-level timestamps. A
  Nemotron-3-Nano-30B-A3B backbone with a C-RADIOv4-H vision encoder and a
  **Parakeet** speech encoder, which is why its ASR is good. **English only.**
  Licence is the NVIDIA Open Model Agreement — commercial use permitted.
- **NemotronLabs-VoiceChat-11B** — if what you actually want is a voice
  assistant rather than an omni model, this is the answer: 9.18 GB,
  full-duplex, streaming transcription, function calling, over
  `mlx_vlm.server` at `/v1/realtime` (WebSocket).

### Two gotchas that will cost you an hour each

1. **`--output-modality audio` does not work with Qwen3-Omni.** The model class
   implements `generate(..., return_audio=True, speaker="Ethan")` but does
   *not* implement `generate_audio` / `supports_audio_generation`, so
   `is_audio_generation_model()` returns False and both the CLI flag and the
   `/v1/audio/speech` endpoint reject it. Speech out requires calling
   `model.generate(..., return_audio=True)` from Python. Only **MiniCPM-o**
   gets the ergonomic path.
2. **Don't use Qwen3-Omni at 4-bit for voice.** Its quantization config is
   `{group_size: 64, bits: 4, mode: "affine"}` applied flat across the whole
   model **including the code2wav vocoder** — no per-module exclusion. Text is
   fine; the synthesised audio is not. Use 8-bit (38.78 GB) or bf16.

**Verdict:** you do **not** need to chain STT → LLM → TTS any more. You did in
2025; you don't in 2026. Start with `MiniCPM-o-4_5-4bit` (6.16 GB) because it
is cheap, permissive and ergonomic. Move to
`Qwen3-Omni-30B-A3B-Instruct-8bit` (38.78 GB) when you need the ceiling. Use
`Nemotron-3-Nano-Omni-...-mxfp4` (18.64 GB) when the job is understanding
hour-long audio, 2-minute video and 256 k of context rather than talking back.

That said: a `parakeet` → LLM → `kokoro` chain is still **faster and far more
controllable** than a 30 B omni. Omni is for when the model needs to *hear
tone*, not just words.

---

## 5. Speech-to-text

> **Do not scale this class up.** 2.51 GB is the plateau — see
> **[STT and TTS — going bigger is pointless](#stt-and-tts--going-bigger-is-pointless-and-sometimes-actively-worse)**.

`mlx-audio` **0.5.4 (2026-09-14)** implements **29 STT families**. The claim
about Parakeet checks out and then some: `mlx-community/parakeet-tdt-0.6b-v3`
is the **#1 most-downloaded model in the entire `mlx-community` org** at
**1,813,064 downloads** — ahead of `Llama-3.1-8B-Instruct-4bit`. v2 is #2.

| Model repo id | Size | Quantization | Runtime | MLX-native? | Memory verdict |
|---|---|---|---|---|---|
| `mlx-community/parakeet-tdt-0.6b-v3` | **2.51 GB** | fp32/bf16 | `parakeet-mlx` or `mlx-audio` | ✅ `parakeet` | Trivial. **The pick.** 25 EU languages. |
| `mlx-community/parakeet-tdt-0.6b-v2` | 2.47 GB | fp32/bf16 | `parakeet-mlx` or `mlx-audio` | ✅ | Trivial. English-only, slightly better on English. |
| `mlx-community/whisper-large-v3-turbo` | 1.61 GB | fp16 | `mlx-whisper` | ✅ | Trivial. 99+ languages — Parakeet's coverage gap. |
| `mlx-community/whisper-large-v3-turbo-asr-fp16` | 1.62 GB | fp16 | `mlx-audio` | ✅ `whisper` | Trivial. Same weights, mlx-audio packaging. |
| `mlx-community/whisper-large-v3-turbo-4bit` | 0.46 GB | 4-bit | `mlx-whisper` | ✅ | Trivial. Only if you're desperate for speed. |
| `mlx-community/Qwen3-ASR-0.6B-8bit` | 1.01 GB | 8-bit | `mlx-audio` | ✅ `qwen3_asr` | Trivial. #8 model in the whole org (222k downloads). |
| `mlx-community/Qwen3-ASR-1.7B-8bit` | 2.47 GB | 8-bit | `mlx-audio` | ✅ | Trivial. The better Qwen3-ASR tier. |
| `mlx-community/nemotron-3.5-asr-streaming-0.6b` | 1.28 GB | fp32 | `mlx-audio` | ✅ `nemotron_asr` | Trivial. **Cache-aware streaming**, 40 locales. |
| `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` | 3.15 GB | 4-bit | `mlx-audio` | ✅ `voxtral_realtime` | Trivial. Mistral streaming STT. |
| `mlx-community/VibeVoice-ASR-bf16` | 16.66 GB | bf16 | `mlx-audio` | ✅ `vibevoice_asr` | Comfortable. **Diarization + timestamps + hotword biasing**, 50+ languages. |
| `mlx-community/diar_sortformer_4spk-v1-fp32` | 0.49 GB | fp32 | `mlx-audio` VAD | ✅ `sortformer` | Trivial. **Speaker diarization**, up to 4 speakers. |
| `mlx-community/silero-vad` | ~2 MB | fp32 | `mlx-audio` VAD | ✅ `silero_vad` | Free. Use it to chunk long files. |

Also implemented in-tree but with no size verified here: `canary` (NVIDIA,
ASR + translation), `moonshine`, `mms` (1000+ languages), `granite_speech`,
`granite_speech5_ctc`, `moss_transcribe_diarize`, `qwen2_audio`,
`qwen3_forced_aligner` (word-level alignment), `mega_asr`, `sensevoice`,
`fireredasr2`, `glmasr`, `cohere_asr`, `phonon`, `lasr_ctc`, `fun_asr_nano`,
`nemo`, `wav2vec`, `higgs_audio_3`.

**Pick:** `parakeet-tdt-0.6b-v3` (2.51 GB) as the default,
`whisper-large-v3-turbo` (1.61 GB) alongside it for languages Parakeet doesn't
cover. Together 4.1 GB — just keep both. Add
`Qwen3-ASR-1.7B-8bit` (2.47 GB) for hard or accented audio, and
`VibeVoice-ASR-bf16` (16.66 GB) only when you actually need speaker labels.

**Tradeoffs.**
- Parakeet is a FastConformer-TDT: much faster than Whisper, and it has no
  hallucination-on-silence failure mode. Its coverage is 25 European languages
  against Whisper's 99+. That is the whole trade.
- **`parakeet-mlx` cannot diarize.** It gives you streaming
  (`transcribe_stream`), three levels of timestamp (result → sentence → token),
  beam decoding on TDT, chunking (`--chunk-duration 120 --overlap-duration 15`)
  and `txt/srt/vtt/json` output — but there is no speaker model in the package.
  For "who said what", use `mlx-audio` with `sortformer` or `VibeVoice-ASR`.
  It also needs `ffmpeg` on PATH.
- **Whisper is now the legacy option.** `whisper-large-v3-turbo` dates from
  October 2024 and OpenAI has shipped nothing since. Parakeet v3, Qwen3-ASR and
  `nemotron-3.5-asr-streaming-0.6b` (2026-09-10, the newest thing in this
  table) have all moved past it. Keep Whisper for the 99-language tail, not as
  your default.
- Use `parakeet-mlx` if you want the focused CLI; use `mlx-audio` if you want
  one package for all 29 STT families.

---

## 6. Text-to-speech / voice

> Bigger TTS buys **controllability, not fidelity** — see
> **[STT and TTS — going bigger is pointless](#stt-and-tts--going-bigger-is-pointless-and-sometimes-actively-worse)**.

`mlx-audio` implements **43 TTS families**. This is the most crowded class in
the document and the one where the README undersells itself.

| Model repo id | Size | Quantization | Runtime | MLX-native? | Memory verdict |
|---|---|---|---|---|---|
| `mlx-community/Kokoro-82M-bf16` | **0.39 GB** | bf16 | `mlx-audio` | ✅ `kokoro` | Free. **Default pick.** 54 voices, 8 languages. |
| `mlx-community/Kokoro-82M-8bit` | 0.35 GB | 8-bit | `mlx-audio` | ✅ | Free. No reason — just run bf16. |
| `mlx-community/index-tts2-mlx` | 8.08 GB | bf16 | `mlx-audio` | ⚠️ `indextts` in tree, **absent from the README table** | Trivial. Emotion control + duration control + zero-shot cloning. 23.7k downloads. |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | 4.52 GB | bf16 | `mlx-audio` | ✅ `qwen3_tts` | Trivial. **Voice design from a text description.** |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | 3.08 GB | 8-bit | `mlx-audio` | ✅ | Trivial. Voice cloning tier. |
| `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` | 1.99 GB | 8-bit | `mlx-audio` | ✅ | Trivial. The README's default example model. |
| `mlx-community/VoxCPM2-8bit` | 3.23 GB | 8-bit | `mlx-audio` | ✅ `voxcpm2` | Trivial. **48 kHz** output, 30 languages, tokenizer-free. |
| `mlx-community/chatterbox-multilingual-v3` | 2.71 GB | fp16 | `mlx-audio` | ✅ `chatterbox` | Trivial. 23 languages, expressive. |
| `mlx-community/csm-1b` | 6.22 GB | bf16 | `mlx-audio` | ✅ `sesame` | Trivial. Sesame CSM. English only. |
| `mlx-community/OmniVoice-8bit` | 1.45 GB | 8-bit | `mlx-audio` | ✅ `omnivoice` | Trivial. **646+ languages**, zero-shot cloning. |
| `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` | 2.54 GB | 4-bit | `mlx-audio` | ✅ `voxtral_tts` | Trivial. Mistral, 20 voices, 9 languages. |
| `mlx-community/Dia-1.6B-fp16` | 3.22 GB | fp16 | `mlx-audio` | ✅ `dia` | Trivial. Dialogue-focused. |
| `mlx-community/Spark-TTS-0.5B-bf16` | 2.92 GB | bf16 | `mlx-audio` | ✅ `spark` | Trivial. **CC-BY-NC-SA** — non-commercial. |
| `mlx-community/Breeze-TTS-2-mlx` | 7.63 GB | bf16 | `mlx-audio` | ✅ `breeze_tts` | Trivial. Voice clone + design + direction. |
| `mlx-community/kitten-tts-nano-0.8` | **0.06 GB** | fp32 | `mlx-audio` | ✅ `kitten_tts` | Free. 60 MB. Smallest thing here that speaks. |
| `mlx-community/LongCat-AudioDiT-1B-bf16` | 5.68 GB | bf16 | `mlx-audio` | ✅ `longcat_audiodit` | Trivial. Diffusion TTS in waveform latent space. |
| `mlx-community/MOSS-TTS-Nano-100M` | 0.29 GB | bf16 | `mlx-audio` | ✅ `moss_tts_nano` | Free. 20 languages, cloning. |

Further in-tree families not sized here: `higgs_audio_v3` (4 B, 100
languages), `kugelaudio` (7 B AR+diffusion, 24 European languages),
`rumik_oss` (3 B, 22 Indic languages), `zonos2`, `melotts`, `outetts`,
`bark`, `soprano`, `arktts`, `pocket_tts`, `vibevoice`, `moss_tts` (8 B,
31 languages), `bailingmm` / Ming-Omni TTS, `fish_qwen3_omni`,
`echo_tts`, `dramabox`, `irodori_tts`, `confucius4`, `tada`,
`chatterbox_turbo`, `voxcpm`, `llama`, `dense`.

**Picks.** `Kokoro-82M-bf16` at 0.39 GB is the default — it is fast, good
enough, and effectively free. Step up to **`index-tts2-mlx`** (8.08 GB) when
you need emotional control and duration control, or
**`Qwen3-TTS-...-VoiceDesign`** (4.52 GB) when you want to *describe* a voice
rather than clone one. `VoxCPM2` if 48 kHz output matters.

**Caveat on IndexTTS-2:** `index-tts2-mlx` is the 4th most-downloaded TTS repo
in the org and `mlx_audio/tts/models/indextts` exists in the source tree, but
IndexTTS is **not listed in the mlx-audio README's supported-model table** and
there is no `index-tts2-mlx` package on PyPI. Treat it as present but
undocumented — expect to read the module to drive it.

**XTTS-v2 / Coqui: no MLX path.** There is no `xtts` module anywhere in
mlx-audio. See *don't bother*.

---

## 7. Music / audio generation

> Largest-that-fits: `MiniMax-Music3-bf16`, **28.52 GB** — and precision does
> track lyric fidelity here, though mxfp8 (13.87 GB) is the project's own
> recommendation. See **[Big-iron picks](#stt-and-tts--going-bigger-is-pointless-and-sometimes-actively-worse)**.

The thinnest class. `mlx_audio/music/models/` contains **exactly one**
architecture: `minimax_music3`. Everything else in the audio-generation space
is either TTS or understanding.

| Model repo id | Size | Quantization | Runtime | MLX-native? | Memory verdict |
|---|---|---|---|---|---|
| `mlx-community/MiniMax-Music3-mxfp8` | **13.87 GB** | MXFP8 | `mlx_audio.music.generate` | ✅ `minimax_music3` | Trivial. **The pick** — most-downloaded, and the one the MLX README recommends. |
| `mlx-community/MiniMax-Music3-8bit` | 14.18 GB | 8-bit | same | ✅ | Trivial. Equivalent to mxfp8. |
| `mlx-community/MiniMax-Music3-bf16` | 28.52 GB | bf16 | same | ✅ | Comfortable. Reference quality, 2× the 8-bit. |
| `mlx-community/MiniMax-Music3-6bit` | 11.70 GB | 6-bit | same | ✅ | Trivial. |
| `mlx-community/MiniMax-Music3-4bit` | 9.21 GB | 4-bit | same | ✅ | Trivial. Lyric fidelity starts to slip. |
| `mlx-community/MiniMax-Music3-nvfp4` | 9.21 GB | NVFP4 | same | ✅ | Trivial. |
| `mlx-community/MiniMax-Music3-mxfp4` | **8.90 GB** | MXFP4 | same | ✅ | Trivial — but ⚠️ see the lyric warning. |
| `mlx-community/MOSS-Music-8B-Thinking-8bit` | 10.24 GB | 8-bit | `mlx-audio` **STT** | ✅ `moss_music` | Trivial. **Understanding + lyrics ASR, not generation.** |
| `mlx-community/ACE-Step1.5-MLX` | not sized | bf16 | its card says `mlx_audio.tts` | ❌ **no `ace_step` module anywhere in mlx-audio 0.5.4** | Will not load. See *don't bother*. |
| `mlx-community/stable-audio-3-small-music` | 3.49 GB | bf16 | card claims `library_name: mlx-audio` | ❌ **no `stable_audio_3` module in mlx-audio 0.5.4** | Will not load. Also **gated**. |

MiniMax Music 3 is the only working local music generator on this machine.
Upstream is `MiniMaxAI/MiniMax-Music3`: 8 B global LLM + 0.6 B local LLM +
2.4 B flow-matching + 123 M flow-VAE decoder. **Lyrics in, full song out** —
vocals, arrangement and all, with `[verse]` / `[chorus]` / `[instrumental]`
section tags. Notably the upstream card says inference requires CUDA and 24 GB+
of VRAM; the MLX port makes that irrelevant.

```bash
python -m mlx_audio.music.generate \
  --model mlx-community/MiniMax-Music3-mxfp8 \
  --caption "<genre, BPM, key, instrumentation, vocal gender/timbre>" \
  --lyrics "[verse] ... [chorus] ..." \
  --duration 30 --steps 30 --seed 7 --output song.wav
```

**Tradeoffs, quoting the MLX README rather than guessing.**
- **Lyrics are required.** There is no instrumental-only mode via an empty
  lyric field; use the `[instrumental]` tag instead.
- **"MXFP4 may alter or omit more requested words than BF16 or MXFP8. Prefer
  MXFP8 when lyric fidelity matters."** Saving 5 GB costs you words. Run mxfp8.
- Caption controls are **"probabilistic rather than strict"** — genre and BPM
  are suggestions, not constraints. The AR stage can also terminate early.
- **Conflicting docs, unresolved:** upstream says 32 kHz stereo output and up to
  5 minutes; mlx-audio's README says 44.1 kHz and the implementation caps at
  360 s. Treat **300 s as the safe duration cap** and verify the sample rate on
  your first output. No Apple-silicon speed or peak-RAM figures are published.
- Licence is Creative Commons; the exact variant was not verified.

Worth knowing: `mlx-audio` also has a whole **speech-to-speech** section that
is adjacent to audio generation — `sam_audio` (text-guided source separation:
"extract the vocals"), `dialogue_sidon` (two-speaker separation),
`mel_roformer`, `mossformer2_se` and `deepfilternet` (denoising),
`moshi`, `mimo_audio`. If your actual goal is audio *processing* rather than
composition, that is where to look.

**MusicGen, YuE, ACE-Step, Magenta RT: no MLX module.** Not in
`mlx_audio/music/models/`. See *don't bother*.

---

## 8. Video generation

> Largest-that-fits: **LTX-2.5 fully bf16**, 40.66 GB peak with the DiT evicted
> (bit-identical output) or 62.40 GB naive. See
> **[Video — LTX-2.5 at full bf16](#video--ltx-25-at-full-bf16)**.

**This is the section I got wrong on the first pass, so it is the one to read
carefully.** Local video generation on Apple silicon went from "no" to
"conditionally yes" during 2026. There are now **three independent MLX-native
video stacks**, `mlx-community` hosts real LTX-2.5 and LongCat conversions, and
there are published wall-clock numbers measured on an **M5 Max / 128 GB** —
this exact machine.

**The honest verdict is bimodal, not "too slow".** A 4-second 768×512 clip is
under a minute. A 5-second 832×480 clip is 12–23 minutes. Anything at 720p,
1080p, or over ~8 seconds is 1–9 hours. It is iterable at small geometry and
useless for deliverables at large geometry.

### Runtimes

| Runtime | Status | Covers | MLX-native? |
|---|---|---|---|
| **`Blaizzy/mlx-video`** ★301, pushed 2026-05-13 | GitHub only | Wan 2.1 1.3B/14B, Wan 2.2 T2V-A14B / I2V-A14B / TI2V-5B, LTX-2/2.3; 4-bit & 8-bit, Wan2.2-Lightning 4-step LoRA, VAE tiling | ✅ |
| **`dgrauet/ltx-2-mlx`** ★112, pushed 2026-09-17 | GitHub only, most active | LTX-2.3 **and LTX-2.5**; `--low-ram` block streaming, `--tile-frames`/`--tile-spatial` | ✅ |
| **`lpalbou/mlx-gen`** 0.37.0 (PyPI) | The only pip-installable path | Wan2.2 T2V/I2V/V2V, VACE, MiniMax-H3, Bernini-R, SeedVR2 | ✅ |
| `ml-explore/mlx-examples/video/wan2.1` | **Frozen** — one commit, 2026-04-06 | Wan 2.1 only | ✅ but unoptimised |
| `ddalcu/mlx-serve` ★1371 (Zig) | Active | LTX-2.5, LTX-2.3, MiniMax-H3 via `POST /v1/video/generations` | ✅ |
| `mrbizarro/Phosphene` ★225 | Active | LTX-2.5 + MiniMax-H3, joint A/V, LoRA training | ✅ |
| `james-see/ltx-video-mac` ★412 | Active | LTX-2/2.3/2.5, MiniMax-H3; SwiftUI app | ✅ |
| **Draw Things** | App Store, weekly | LTX-2/2.3, Wan 2.1/2.2, HunyuanVideo, SkyReels, MiniMax-H3, LongCat-Avatar | ❌ own Metal engine |
| `mflux` 0.19.2 | ❌ **No video generation.** Its "image & video" description means **SeedVR2**, a video *upscaler* used here on stills. No video family in `src/mflux/models/`, no video command in `[project.scripts]`. | | |
| ComfyUI on MPS | ⚠️ **Avoid for video** — see below | | ❌ |

⚠️ **`pip install mlx-video` gets the wrong package.** PyPI `mlx-video` 0.1.0
is `AmiraniLabs/mlx-video`, *"video loading and preprocessing utilities"* — not
generation. Blaizzy's declares the same name but is unpublished. Use
`pip install git+https://github.com/Blaizzy/mlx-video.git`.

### Weights

| Model repo id | Size | Quantization | Runtime | MLX-native? | Memory verdict |
|---|---|---|---|---|---|
| `mlx-community/ltx-2.5-mlx-ditq8` | **20.60 GB** | int8 DiT | `ltx-2-mlx` | ✅ | Comfortable. Mix-and-match with an encoder repo. |
| `mlx-community/ltx-2.5-mlx-q8` | 23.89 GB | int8 **text encoder**, bf16 DiT | `ltx-2-mlx` | ✅ | Comfortable. **Not** a quantized DiT — read the card. |
| `mlx-community/ltx-2.5-mlx` | **110.08 GB** | bf16 | `ltx-2-mlx` | ✅ | ⚠️ Whole component tree: dev **and** distilled 22 B DiTs (37.99 GB each) + a bf16 Gemma-4-12B encoder (23.81 GB). Don't pull blind. |
| `mlx-community/LTX-2-dev-bf16` | 93.33 GB | bf16 | `mlx-video` / `ltx-2-mlx` | ✅ | Fits. Superseded by 2.5. |
| `mlx-community/LongCat-Video-q8` | 33.62 GB | 8-bit | `mlx-video` | ✅ | Comfortable. T2V + I2V + video continuation. |
| `mlx-community/LongCat-Video-bf16` | 45.75 GB | bf16 | `mlx-video` | ✅ | Comfortable. |
| `Anes1032/Wan2.2-TI2V-5B-mlx-q8` | **19.58 GB** | 8-bit | `mlx-video` / `mlx-gen` | ✅ | Trivial. **The practical pick** — best speed/size. |
| `Anes1032/Wan2.2-I2V-A14B-mlx-q8` | 42.68 GB | 8-bit | `mlx-video` / `mlx-gen` | ✅ | Comfortable, but slow — A14B is **two** 14 B experts, not MoE. |
| `mlx-community/Phantom-Wan-1.3B` | 15.58 GB | bf16 | `mlx-video` | ✅ | Trivial. Subject-to-video. |
| `ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit` | 41.11 GB | 4-bit | `mlx-serve` / `mlx-gen` | ✅ | Comfortable. **Video + synchronised stereo audio in one pass.** |
| `ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit` | 70.08 GB | 8-bit | `mlx-serve` / `mlx-gen` | ✅ | Fits, ~45 GB spare. |
| `ddalcu/LTX-2.5-MLX-Serve-8bit` | 64.13 GB | 8-bit | `mlx-serve` | ✅ | Fits. |

Licences to check before use: LTX-2.x is `ltx-2-community-license-agreement`;
**MiniMax-H3's community licence reportedly excludes local deployment in the
US/EU/UK/South Korea** — verify on the card before building on it.

### Measured on an M5 Max / 128 GB

From `mlx-gen`'s own docs (`--quantize 8`, 124 frames, MLX 0.31):

| Route | Canvas | Steps | Wall clock | Peak |
|---|---|---|---|---|
| `wan2.2-ti2v-5b` q8, 121 f @24fps, silent | 832×480 | 50 | **23 min** | **25 GB** |
| Wan2.2 TI2V-5B, 101 f @20fps | 832×480 | 25 | **12 min** | — |
| Wan2.2 T2V-A14B, 101 f | 480×240 | 25 | **30 min** | — |
| Wan2.2 TI2V-5B, 101 f | 1280×704 | 25 | **35 min** | — |
| `minimax-h3-turbo-544p` (with audio) | 960×544 | 8 | **11–17 min** | **80 GB** |
| `minimax-h3` base | 960×544 | 50 | **62 min** | 79 GB |
| `minimax-h3-turbo` | 1344×768 | 8 | **34 min** | 85 GB |
| VACE 1.3B bf16, 17 f | 448×256 | 16 | **239 s** | 11–12 GiB |
| VACE 1.3B **flag-free defaults**, 81 f | 832×480 | 30 | **1 h 56 min** | 31.7 GiB |

LTX-2.5 via `ltx-2-mlx` (Apple silicon, 128 GB — exact chip not stated in the
repo, so treat as M3/M4/M5-class):

| Config | Wall clock | Peak |
|---|---|---|
| **768×512, 97 frames (4.0 s @24fps), distilled 8+3 steps** | **49.4 s** | **40.76 GB** |
| same + 3 generated keyframe slots | 54.7 s | 40.76 GB |
| DFR temporal rounds=1 / rounds=2 | 138.9 s / 177.8 s | 44.37 GB |

And the MLX-vs-PyTorch comparison that justifies staying on MLX — same LTX-2.3
work, M4 Max 128 GB, 576×1024, 121 frames (5 s), 8+3 steps:

| Backend | Wall clock | Peak |
|---|---|---|
| **MLX fp16** | **152.5 s** | **34.2 GB** |
| PyTorch MPS bf16 | 264.0 s | **>60 GB** |

**MLX is ~1.7× faster at roughly half the peak memory.** That is the single
strongest argument for the MLX path over ComfyUI/diffusers on this hardware.

Draw Things publishes its own M5 Max ledger (LongCat-Avatar 1.5, 448×320,
8 steps, M5 Max/48 GB): 200 frames (8.0 s) at 24 steps = **421.8 s** (i8x) or
**550.1 s** (q8p); 93 frames at 8 steps = **134.7 s**. A community datapoint has
LTX-2.3 22B distilled, default settings, 5-second video at **M5 Max 121 s vs
M3 Ultra 206 s**.

### Three traps specific to a 128 GB machine

1. **MiniMax-H3 unquantized is 125 GiB resident — 97% of the machine.**
   `mlx-gen` *refuses the load* rather than let the OS kill it. `--quantize 8`
   is mandatory. First load reads ~133 GB of shards (~4.5 min); `mlxgen prepare`
   writes a 75 GB package to skip that. Budget ~140 GB of disk.
2. **`wan2.2-ti2v-5b` q8 at 1280×704 / 17 f / 20 steps peaks at 103.7 GiB.**
   That is inside the 115.4 GB ceiling with only ~12 GB of margin. Don't run it
   alongside anything else.
3. **q8 storage no longer buys runtime memory for Wan A14B.** Since a
   2026-06-12 runtime-precision fix it runs at BF16-class memory (~33 GiB peak)
   regardless. And `ltx-2-mlx` exposes `LTX2_GEMMA_EVAL_EVERY` /
   `LTX2_DIT_EVAL_EVERY` specifically to keep Metal command buffers under the
   **~10 s macOS GPU watchdog** — on a 128 GB machine that never crashes, set
   them to `0` for full throughput.

Also useful: LTX-2.5's peak is **encoder-bound, not geometry-bound**. A
256×256×9 clip and a 512×288×121 clip peak within 0.01 GB of each other,
because the floor is the bf16 Gemma-4-12B text encoder (~24.4 GB resident).
Evicting the DiT around the encode phase drops peak **62.40 → 40.66 GB with
bit-identical output**. The int4 text encoder was measured and **rejected**
(cosine 0.996728 against a 0.999879 bf16 floor) — several third-party MLX packs
ship it at 4-bit with no quality data.

### What does not work

- **Mochi-1** — 133.5 GB, never left "preview", **zero MLX ports**. Dead end.
- **HunyuanVideo** — the only MLX port was abandoned 2024-12-17.
  HunyuanVideo-1.5's sole MLX repo has **0 downloads**. Use Draw Things, which
  does support it.
- **CogVideoX** — upstream last touched 2024-11-23. Only `dgrauet` community
  conversions.
- **Wan open weights stop at 2.2.** Wan 2.5 / 2.6 / 2.7 / 3.0 are API-only.
  Several SEO sites (`wan27.org`, `atlascloud.ai`, `localaimaster.com`) claim
  "Wan 3.0 open-weight Apache-2.0, April 2026" — **false**, HuggingFace has
  nothing. Don't trust them.
- **LTX-2.x claims no macOS support upstream.** The old `Lightricks/LTX-Video`
  README explicitly supported MPS; `Lightricks/LTX-2` does not carry that
  forward. On Mac, LTX-2.x is **MLX-only in practice**.
- **ComfyUI on MPS for video — a correctness hazard, not just slow.** Open
  issues document *silent* failure on this exact chip: **#15804** (M5 Max,
  macOS 26.6.2 — **19 of 26 T2V runs produced solid black video** with bf16
  LTX-2.x), **#15793** (Wan 2.1/2.2 progressive corruption, deterministic and
  silicon-generation dependent), **#15818** (LTX-2.5 all-black unless
  `--use-split-cross-attention`), **#15010** (Wan 2.2 vertical banding on
  M4 Max), **#2044** (Conv3D unsupported on MPS, open since 2023). Plus
  **#14838**, the unmerged fix for attention corruption above 2³¹ elements —
  its cited trigger is a *61-frame 832×640* generation. For calibration, a
  ComfyUI+MPS Wan 2.2 I2V A14B run at 832×480 / 33 frames (~2 s) on an
  M1 Max took **82 minutes**.

### Recommended video stack

1. **`Anes1032/Wan2.2-TI2V-5B-mlx-q8` (19.58 GB) via `mlx-gen`** — 832×480,
   5 s, ~12–23 min at 25 GB peak. Published numbers on this chip.
2. **`dgrauet/ltx-2-mlx` with LTX-2.5** for the fast loop — 768×512, 4 s in
   ~50 s. Pull `ltx-2.5-mlx-ditq8` + `ltx-2.5-mlx-q8`, not the 110 GB tree.
3. **Draw Things** for stability — its own Metal engine makes it immune to
   every MPS bug above, and it has two shipped M5-Max-specific fixes.
4. **MiniMax-H3 4-bit** only when you specifically need synchronised audio, and
   check the licence territory restriction first.
5. **Do not use ComfyUI on MPS for video.**

---

## 9. Embeddings and reranking

> Largest-that-fits: `majentik/harrier-oss-v1-27b-MLX-8bit`, **28.73 GB** — the
> #1 open multilingual embedder, and a genuine 128 GB unlock. But a 0.65 GB
> embedder plus a reranker usually beats it. See **[Big-iron picks](#stt-and-tts--going-bigger-is-pointless-and-sometimes-actively-worse)**.

MLX-native, cheap, and the useful part of this class fits in **about 1 GB**.
Use **`mlx_vlm.server`**, not `mlx-embeddings` — that recommendation is the
main finding here and the reason is below.

`mlx-vlm` 0.7.1 has a full, working — but **completely undocumented** —
embedding and reranking surface. Verified in source: `mlx_vlm/reranker.py`,
`embedding_loader.py`, `server/embeddings.py` (`@app.post("/v1/embeddings")`),
`server/reranking.py` (`@app.post("/v1/rerank")`), and the
`--embedding-model` / `--reranker-model` CLI flags. Its reranker registry:

```python
_RERANKER_KINDS = {
    "bert":        SEQUENCE_CLASSIFIER,
    "modernbert":  SEQUENCE_CLASSIFIER,
    "qwen3":       GENERATIVE_TEXT,
    "qwen3_vl":    GENERATIVE_VL,
    "xlm_roberta": SEQUENCE_CLASSIFIER,
}
```

with proper `sigmoid(logit_yes − logit_no)` scoring on the generative path.
Neither `docs/usage.md` nor `docs/cli_reference.md` mentions any of it, so
expect to read source. It does **not** depend on `mlx-embeddings` — these are
its own implementations.

| Model repo id | Size | Quantization | Runtime | MLX-native? | Memory verdict |
|---|---|---|---|---|---|
| `mlx-community/Qwen3-Embedding-0.6B-8bit` | **0.65 GB** | 8-bit | `mlx_vlm.server` | ✅ `qwen3_embedding` | Free. **The pick.** MTEB(eng,v2) 70.47 at **95% zero-shot**. 32k ctx. |
| `majentik/harrier-oss-v1-0.6b-MLX-8bit` | **0.65 GB** | 8-bit | `mlx_vlm.server` | ✅ (`model_type: qwen3`) | Free. **+4.7 multilingual points** over Qwen3-0.6B for identical size/cost. MIT, 32k ctx. Community quant. |
| `mlx-community/Nemotron-3-Embed-1B-BF16-8bit` | 1.23 GB | 8-bit | **`mlx-vlm` only** | ✅ `ministral3_embedding` | Free. Best retrieval-per-param — RTEB(eng) above Qwen3-Embedding-8B. Licence OpenMDW-1.1, read it. |
| `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` | 2.28 GB | 4-bit DWQ | `mlx_vlm.server` | ✅ | Trivial. DWQ recovers most of the 4-bit loss. |
| `mlx-community/Qwen3-Embedding-8B-4bit-DWQ` | 4.27 GB | 4-bit DWQ | `mlx_vlm.server` | ✅ | Trivial. Top of the Qwen3 line. |
| `majentik/harrier-oss-v1-27b-MLX-8bit` | **28.73 GB** | 8-bit | `mlx_vlm.server` | ✅ (`gemma3_text`) | Comfortable. **The #1 open multilingual embedder in existence** (74.27) — and you can actually run it. MIT. |
| `mlx-community/nomicai-modernbert-embed-base-bf16` | 0.30 GB | bf16 | `mlx_vlm.server` | ✅ `modernbert` | Free, 27.8k downloads, 8192 ctx — but see the bug warning. Not on MTEB(eng,v2) at all. |
| `mlx-community/embeddinggemma-300m-8bit` | 0.37 GB | 8-bit | `mlx_vlm.server` | ✅ `gemma3_embedding` | Free. 69.67 — but only **2048 ctx**. |
| `mlx-community/all-MiniLM-L6-v2-4bit` | **0.01 GB** | 4-bit | `mlx_vlm.server` | ✅ `bert` | Free. 10 MB, 92.5k downloads. 512 ctx, 65.14. Weak but instant. |
| `mlx-community/Qwen3-VL-Embedding-2B-8bit` | 2.66 GB | 8-bit | `mlx-vlm` | ✅ `qwen3_vl_embedding` | Trivial. **Multimodal** — images and text in one space. |
| `mlx-community/Qwen3-Reranker-0.6B-4bit` | **0.35 GB** | 4-bit | `mlx_vlm.server --reranker-model` | ✅ generative | Free. **The reranker pick** — NDCG@10 0.5940. |
| `mlx-community/Qwen3-Reranker-4B-mxfp8` | 4.16 GB | MXFP8 | same | ✅ | Trivial. **NDCG@10 0.6367** — the best MLX-native reranker. |
| `BAAI/bge-reranker-v2-m3` | 568 M upstream | convert yourself | `mlx_vlm.server` | ✅ `xlm_roberta` | Free. 17.8M downloads and the most-deployed reranker alive, but **now the weakest modern option** (0.5526). No v3 is coming. |
| `Alibaba-NLP/gte-reranker-modernbert-base` | 150 M upstream | convert yourself | `mlx_vlm.server` | ✅ `modernbert` | Free. 0.5843. **No `mlx-community` conversion exists** — convert it. |
| `mlx-community/Qwen3-VL-Reranker-2B-8bit` | 2.66 GB | 8-bit | `mlx-vlm` | ✅ | Trivial. Reranks **images** as documents. |
| `mlx-community/jina-reranker-v3-4bit-mxfp4` | 0.33 GB | MXFP4 | ⚠️ | ⚠️ base arch `JinaForRanking` is in **neither** package's registry | Free if it loads. **Loadability unverified.** |
| `mlx-community/mxbai-rerank-large-v2` | 3.10 GB | bf16 | ⚠️ | ⚠️ base is `Qwen2ForCausalLM` — generative, **no native scoring API** | 0.6115 on paper, but you hand-roll the scoring. |

**Pick for local RAG:** `Qwen3-Embedding-0.6B-8bit` (0.65 GB) +
`Qwen3-Reranker-0.6B-4bit` (0.35 GB) = **1.0 GB total**, both served from one
`mlx_vlm.server`. If your corpus is multilingual, swap the embedder for
`harrier-oss-v1-0.6b` at the same size — it drops straight in, since its
`model_type` is `qwen3`. If retrieval quality is measurably the bottleneck, go
to `Qwen3-Reranker-4B-mxfp8` (0.6367) before touching the embedder.

### ⚠️ Why not `mlx-embeddings`

It works, but on **this exact stack** it has three open, zero-comment,
unfixed bugs that *silently corrupt* embeddings — and one of them hits the
model that looked like the obvious pick:

- **#80** — ModernBERT **mixed-length batches return all-NaN vectors for every
  padded item** when the longest item tokenizes to an exact multiple of 32 at
  ≥128 tokens. Filed against `nomic-ai/modernbert-embed-base`. *"The response
  is otherwise normal, so downstream code gets silently corrupt embeddings."*
- **#74** — the per-item `generate()` loop hits an **unrecoverable Metal OOM at
  ~3,800 sequential calls**. Reported on **mlx 0.32.2, macOS 26.6.2, Apple M5**
  — this machine's exact configuration.
- **#72** — `gemma3_text.py` casts the additive attention mask to
  `embed_tokens.weight.dtype`, which is **packed uint32 in quantized
  checkpoints**, destroying the `-inf` entries, so **pad tokens leak into
  attention on every padded batch**. fp16 checkpoints work only by accident.

It is also **4 months stale** (last push 2026-05-13), **GPLv3** (flag that for
commercial use), and architecturally narrower than mlx-vlm: its
`xlm_roberta.py` has no sequence-classification head (so **BGE-reranker-v2-m3
does not work there**), `qwen3.py` has no reranker class (so **Qwen3-Reranker
text does not work there**), and `llama_bidirec.py` has no classification head.
All three *do* work via mlx-vlm.

If you keep `nomicai-modernbert-embed-base`, serve it through `mlx-vlm`.

### Two traps

- **`mlx-lm` 0.31.3 has no embedding support at all.** Confirmed negative:
  `mlx_lm/server.py` has zero `embed` references and exposes only
  `/v1/completions` and `/v1/chat/completions`. Yet
  `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`, `Qwen3-Reranker-0.6B-4bit` and
  `mxbai-rerank-large-v2` all carry `library_name: mlx` and tell you to
  `pip install mlx-lm`. They are plain causal-LM conversions with **no pooling
  or scoring API** — you hand-roll it. The `Qwen3-Reranker-0.6B-4bit` card at
  least publishes the correct recipe:
  `softmax([logit("no"), logit("yes")])[1]` at the last position.
- **Read MTEB's `zeroShotPct` column before believing a score.**
  `QZhou-Embedding` posts 75.97 at **48%** zero-shot; `Qwen3-Embedding-8B`
  posts 75.23 at **95%**. Much of that gap is contamination, not capability.
  Filter to `zeroShotPct >= 90`. Note also that `MTEB(eng, v2)` and
  `MTEB(Multilingual, v2)` are now `displayOnLeaderboard: false` — still
  populated, no longer the front door — and **RTEB's private column was removed
  in Jan 2026** over a conflict of interest, so every RTEB row returns
  `meanPrivate: null`. There is no "MTEB v3".

**Worth knowing:** domain fine-tuning dwarfs generational progress.
`litillabs/litil-embed-0.6b` scores **73.30** on MTEB(Law) where general 0.6 B
models score 56–58 — **~15 points from domain tuning against ~4 from a model
generation.** If your RAG corpus is narrow, fine-tuning a 0.6 B embedder beats
any model swap in this table.

**Dead lines — don't start from these.** BAAI, Snowflake, Nomic, mixedbread and
intfloat have each shipped **zero** text-embedding models in 2026. There is no
Nomic v3, no Arctic v3, no BGE v3. One exception worth keeping:
`snowflake-arctic-embed-l-v2.0` is still #1 sub-1B on MTEB(Law) and
MTEB(Medical) at 100% zero-shot.

**Newer and under-noticed:** `voyageai/voyage-4-nano` — a frontier vendor
open-sourcing a **346 M, Apache-2.0** embedder with 2048d MRL and 32k ctx. And
`perplexity-ai/pplx-embed-v1-{0.6b,4b}` — MIT, 32k, MRL, and **no instruction
prefix required**, which is genuinely valuable for pipeline stability. Neither
is on MTEB yet; neither has an `mlx-community` conversion.

---

## Install

The env today has only `mlx 0.32.2`, `mlx-metal`, `mlx-lm 0.31.3` and
`numpy 2.5.3` on Python 3.12.14. Every package below declares
`requires-python >=3.10`, so 3.12 is fine. Install **only the class you need** —
these are independent.

```sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dekho-apple-local-llm
```

### Images and image editing — `mflux`

Install as an **isolated `uv` tool**, not into the conda env. mflux pins
`mlx>=0.32.0,<0.33.0` and drags in a CPU-side torch; keeping it separate stops
it from fighting `mlx-lm`.

```sh
# if uv is not present: brew install uv
uv tool install --upgrade mflux --with hf_transfer
mflux-completions          # zsh completions
mflux-capabilities         # ground truth: which flags each command honours
```

Then, in rough order of usefulness:

```sh
# fastest edit + generate loop, Apache-2.0, 4 steps
mflux-generate-flux2 --model flux2-klein-4b --prompt "..." -q 6
mflux-generate-flux2-edit --model flux2-klein-4b --image-paths in.jpg --prompt "..." -q 6

# the "Nano Banana" pick — note the explicit 2511 weights
hf download fcreait/Qwen-Image-Edit-mflux Qwen-Image-Edit-2511-q8 --local-dir ./qwen-edit-q8
mflux-generate-qwen-edit --model ./qwen-edit-q8/Qwen-Image-Edit-2511-q8 \
  --base-model qwen --image-paths in.jpg --prompt "..." --steps 30 --guidance 2.5
```

Do **not** also `pip install mlx-gen` — it installs colliding `mflux-*`
scripts. If you want its Wan video / masked-inpaint extras, give it a separate
env.

### Audio — STT, TTS, music — `mlx-audio`

One package covers 29 STT, 43 TTS and the single music architecture.

```sh
pip install mlx-audio
pip install misaki                  # required for Kokoro text processing
pip install "misaki[ja]" "misaki[zh]"   # only if you need JA / ZH Kokoro
brew install ffmpeg                 # needed by the STT CLIs

# transcribe
python -m mlx_audio.stt.generate --model mlx-community/parakeet-tdt-0.6b-v3 --audio in.wav
# speak
mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 --text "Hello" --voice af_heart --play
# compose
python -m mlx_audio.music.generate --model mlx-community/MiniMax-Music3-mxfp8 \
  --caption "..." --lyrics "[verse] ..." --duration 30 --steps 30 --output song.wav
```

For a focused transcription CLI with word timestamps and SRT/VTT output:

```sh
pip install parakeet-mlx           # 0.5.2; needs ffmpeg
parakeet-mlx in.wav --output-format srt --highlight-words \
  --chunk-duration 120 --overlap-duration 15
```

### Vision, omni, embeddings, reranking — `mlx-vlm`

One install and one server covers VLM + omni + embeddings + rerank + (via
`mlx-audio`) TTS/STT endpoints.

```sh
pip install mlx-vlm

# one-shot
mlx_vlm.generate --model mlx-community/Qwen3-VL-30B-A3B-Instruct-8bit \
  --image page.png --prompt "Extract the table as CSV" --max-tokens 800

# omni with speech out (MiniCPM-o is the ergonomic path)
mlx_vlm.generate --model mlx-community/MiniCPM-o-4_5-4bit --output-modality audio \
  --prompt "Say hello." --ref-audio voice.wav --output speech.wav

# everything behind one OpenAI-compatible server
mlx_vlm.server --port 8080 \
  --model mlx-community/Qwen3-VL-30B-A3B-Instruct-8bit \
  --embedding-model mlx-community/Qwen3-Embedding-0.6B-8bit \
  --reranker-model mlx-community/Qwen3-Reranker-0.6B-4bit \
  --stt-model mlx-community/parakeet-tdt-0.6b-v3 \
  --tts-model mlx-community/Kokoro-82M-bf16
```

Endpoints: `/v1/chat/completions` (image + audio + text), `/v1/responses`,
`/v1/embeddings`, `/v1/rerank`, `/v1/audio/speech`,
`/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/realtime`
(WebSocket full-duplex), `/health`, `/metrics`, `/unload`.

### Embeddings only, as a library

Prefer the `mlx_vlm.server` route above. If you do want the library, note that
`mlx-embeddings` is GPLv3, four months stale, and has three open
silent-corruption bugs — one filed on mlx 0.32.2 / Apple M5:

```sh
pip install mlx-embeddings
# repo main is ahead of the 0.1.0 release:
pip install "git+https://github.com/Blaizzy/mlx-embeddings"
```

### Video — `mlx-gen` or `ltx-2-mlx`, each in its own env

Neither is safely installable next to `mflux`, and the good LTX runtime is not
on PyPI at all. Give each its own environment.

```sh
# Wan 2.2 / MiniMax-H3 — the only pip-installable path.
# NOTE: installs colliding mflux-* scripts. Separate env, always.
python3 -m venv ~/.venvs/mlxgen && ~/.venvs/mlxgen/bin/pip install mlx-gen
~/.venvs/mlxgen/bin/mflux-generate-wan --help

# LTX-2.5 — most active runtime, GitHub only
git clone -b ltx-2.5 https://github.com/dgrauet/ltx-2-mlx && cd ltx-2-mlx

# Wan + LTX-2.3 via Blaizzy's package.
# NOTE: `pip install mlx-video` is a DIFFERENT package. Use the git URL.
pip install git+https://github.com/Blaizzy/mlx-video.git
```

On a 128 GB machine, set `LTX2_GEMMA_EVAL_EVERY=0` and `LTX2_DIT_EVAL_EVERY=0`
for full throughput — those knobs exist to stay under the ~10 s macOS GPU
watchdog on smaller machines.

### Optional non-MLX hedge — Draw Things

Worth having as a second backend with genuinely different performance
characteristics, immune to every MPS bug in this document, and the most stable
route to local video.

```sh
brew install --HEAD drawthingsai/draw-things/draw-things-cli
draw-things-cli models list
```

---

## Does not work / don't bother

| Thing | Why not |
|---|---|
| **`diffusionkit`** | GitHub repo is **archived**; last PyPI release **2024-12-06**. Fatally, it pins **`mlx==0.17.3`** while mflux needs ≥0.32.0 — they cannot share an environment, and MLX 0.17 predates most of the Metal kernels this chip has. Only ever did SD3 + FLUX.1-schnell/dev. |
| **`lightning-whisper-mlx`** | Last release **2024-04-02**, last commit 2024-05-08. Abandoned for ~2.5 years. Use `mlx-whisper` or `mlx-audio`. |
| **`mlx-examples/stable_diffusion`** | Directory still exists, but the SD code was last touched **2024-11-21** (SDXL-turbo and SD-2.1 only) and the `flux/` dir 2025-03-25. Educational reference; mflux is its productionised descendant. |
| **`mlx-image`** | Real package (0.1.10) but it is timm/torchvision **classifiers** — ResNet, ViT. Not generation. Easy to grab by mistake. |
| `mlx-diffusers`, `qwen-image-mlx`, `mlx-flux`, `mlx-stable-diffusion`, `mlx-imagegen`, `boogu-image-mlx`, `mflux-comfyui`, `index-tts2-mlx` | **All 404 on PyPI.** They do not exist. `index-tts2-mlx` is an HF *model* run by `mlx-audio`, not a package. |
| **`mlx-gen` alongside `mflux`** | It is an mflux fork that vendors the same `mflux` module name and installs the same `mflux-*` console scripts. Guaranteed collision. Separate env or skip. |
| **`pip install mlx-video`** | Gets the **wrong package** — PyPI `mlx-video` is AmiraniLabs' video *loading/preprocessing* utilities, not generation. Blaizzy's generation package declares the same name but is unpublished. Use `pip install git+https://github.com/Blaizzy/mlx-video.git`. |
| **Video at 720p+, or clips over ~8 s** | Runs, but 1–9 **hours**. Apple's own unoptimised `mlx-examples` Wan path is 75 min (1.3 B) to 3.2 h (14 B) at 50 steps. Stay at 480p-class geometry and distilled/turbo step counts. |
| **Mochi-1, HunyuanVideo, CogVideoX on MLX** | Mochi-1 is 133.5 GB, never left "preview", and has **zero** MLX ports. HunyuanVideo's only MLX port was abandoned 2024-12-17; HunyuanVideo-1.5's sole MLX repo has **0 downloads**. CogVideoX upstream last moved 2024-11-23. Use Draw Things for HunyuanVideo. |
| **Wan 2.5 / 2.6 / 2.7 / 3.0** | **No open weights** — Alibaba kept 2.5+ API-only. Open Wan stops at **2.2**. Sites like `wan27.org`, `atlascloud.ai` and `localaimaster.com` claim "Wan 3.0 open-weight Apache-2.0, April 2026"; HuggingFace has nothing. SEO spam. |
| **`mlx-examples/video/wan2.1`** | Frozen — added in one commit 2026-04-06 and untouched since, with 180 open issues on the repo. Unoptimised: ~90 s/step (1.3 B) to ~250 s/step (14 B I2V) at 81 frames on an M4 Max. |
| **ComfyUI on MPS for video specifically** | Not just slow — **silently wrong**. #15804: on **M5 Max / macOS 26.6.2, 19 of 26 T2V runs produced solid black video** with bf16 LTX-2.x. #15793: deterministic Wan corruption that varies by silicon generation. #15818, #15010, #2044 (Conv3D unsupported since 2023). Use Draw Things or an MLX runtime. |
| **ComfyUI on MPS at 128 GB** | Runs, but: **#14837** silently corrupts attention output *specifically* on large-unified-memory Macs (unmerged), fp8 is hard-blocked, `torch._int_mm` is unimplemented, offload is disabled by design (`VRAMState.SHARED`), and there is no fast attention kernel. 86 open MPS issues. Last resort only. |
| **FLUX.2-dev** | 32.2 B, **112.8 GB** of weights. Would leave ~3 GB under the 115.4 GB ceiling for activations, and mflux does not implement it regardless. |
| **`Qwen3-VL-235B-A22B-Instruct-4bit`** | **133.41 GB > 115.4 GB.** Will not load. The 3-bit (104.02 GB) technically fits but leaves ~11 GB for the vision tower and KV cache, at 3-bit vision quality. Use the 30B-A3B. |
| **Boogu-Image-0.1-Edit under mflux** | `mflux-generate-boogu` is **text-to-image only** — `variants/` has no `edit/`. The third-party MLX loader is a 3-commit, unlicensed, self-described in-progress repo. ComfyUI-MPS bf16 is the only working route. |
| **Qwen3-Omni at 4-bit for speech** | Flat `affine` 4-bit is applied across the whole model **including the code2wav vocoder**, with no per-module exclusion. Text is fine, synthesised audio is not. Use 8-bit. |
| **Qwen3-Omni via `--output-modality audio`** | The class lacks `generate_audio`/`supports_audio_generation`, so the CLI flag and `/v1/audio/speech` reject it. Call `model.generate(..., return_audio=True, speaker="Ethan")` from Python. MiniCPM-o is the one with the ergonomic path. |
| **Nemotron-3-Nano-Omni for speech synthesis** | It physically cannot. No talker, no TTS, no vocoder in the implementation — grep confirms. Text out only. |
| **`ACE-Step1.5-MLX`, `stable-audio-3-small-music`** | Weights are published and tagged `mlx-audio`, but **there is no `ace_step` or `stable_audio_3` module in mlx-audio 0.5.4**. They will not load. Stable Audio is also gated. |
| **MusicGen, YuE, Magenta RT** | Zero `mlx-community` repos, no MLX module. No Mac path. MiniMax Music 3 is the only local music generator. |
| **XTTS-v2 / Coqui** | Coqui Inc. shut down **January 2024**. No `xtts` module in mlx-audio, no `mlx-community/XTTS*` repo. The `idiap/coqui-ai-TTS` fork works but is CPU-only in practice on Apple silicon (MPS is buggy). **IndexTTS-2 — which itself credits XTTSv2 — supersedes it and is MLX-native.** |
| **Step1X-Edit, OmniGen2, BAGEL-7B-MoT, HiDream-E1** | No MLX implementation at all. Qwen-Image-Edit-2511 beats all of them on the editing leaderboard anyway. |
| **Whisper as your default STT** | `large-v3-turbo` is from **October 2024** and OpenAI has shipped nothing since. Parakeet v3, Qwen3-ASR and `nemotron-3.5-asr-streaming-0.6b` have all moved past it. Keep it for the 99-language tail only. |
| **`parakeet-mlx` for diarization** | It has no speaker model. Timestamps and streaming yes; "who said what" no. Use `mlx-audio` with `sortformer` or `VibeVoice-ASR`. |
| **`mlx-embeddings` for production RAG** | Three open, unfixed, zero-comment bugs that **silently corrupt** output: **#80** all-NaN vectors for padded items in mixed-length ModernBERT batches, **#74** unrecoverable Metal OOM at ~3,800 sequential calls (filed on **mlx 0.32.2 / Apple M5** — this stack), **#72** pad tokens leaking into attention on quantized Gemma checkpoints. Also GPLv3, 4 months stale, and missing the xlm-roberta and qwen3 reranker heads. Use `mlx_vlm.server`. |
| **`mlx-lm` for embeddings** | It has none. `mlx_lm/server.py` has zero `embed` references and exposes only `/v1/completions` and `/v1/chat/completions` — despite several `mlx-community` embedding cards telling you to `pip install mlx-lm`. Those are plain causal-LM conversions with no pooling or scoring API. |
| **`BAAI/bge-reranker-v2-m3` as a default** | Works fine via mlx-vlm's `xlm_roberta` path and has 17.8M downloads, but at NDCG@10 **0.5526** it is now the weakest modern option — `Qwen3-Reranker-0.6B` scores 0.5940 at a similar size. There is no bge-reranker-v3. |
| **MTEB scores at face value** | Check the `zeroShotPct` column. `QZhou-Embedding` posts 75.97 at **48%** zero-shot vs `Qwen3-Embedding-8B` at 75.23 and **95%**. Filter to ≥90. RTEB's private column was removed in Jan 2026 over a conflict of interest. |
| **`gpt-oss-120b`-style tool-call surprises** | Not applicable here, but the same lesson holds: check the implementation, not the model card. Two cards in this document print commands that do not exist. |

### Claims that did not survive checking

- **"Qwen-Image 2.0, February 2026"** — not on HuggingFace. The Qwen image line
  ends at `Qwen-Image-2512`. False.
- **"Qwen-Image-Edit-2512"** — does not exist. 2512 is the text-to-image base;
  2511 is the newest editor.
- **"Kontext v2"** — does not exist. BFL points new projects at FLUX.2.
- **"FLUX 3 Dev weights"** — FLUX 3 was announced 2026-07-23 and FLUX 3 Video
  went GA 2026-08-04, but **no open weights have been released** and none are in
  the BFL HuggingFace org. Announcement only.
- **mflux "video models"** — the repo description says image *and video*, but
  the only video-adjacent thing in-tree is the SeedVR2 upscaler. mflux does not
  generate video.
- **"Wan 3.0 is open-weight Apache-2.0 as of April 2026"** — false, and
  actively pushed by several SEO domains. Open Wan weights stop at 2.2.
- **"LTX-2 supports macOS"** — the *old* `Lightricks/LTX-Video` README did
  explicitly support MPS; `Lightricks/LTX-2` does not carry that forward. On
  Mac, LTX-2.x is MLX-only in practice.
- **My own first-pass conclusion that video generation has no MLX path** — wrong.
  Three MLX video stacks exist and `mlx-community` hosts LTX-2.5 and LongCat
  conversions. Corrected in section 8.
- **mflux docs saying `qwen-image-edit` = 2511** — the code says 2509.

---

## Unverified — flagged rather than guessed

- Whether **Gemma 4 can emit audio**. Audio *input* is confirmed
  (`mlx_vlm/models/gemma4/audio.py`, `audio_config`, audio token ids). Output is
  assumed text-only, like gemma-3n, but not proven.
- **Orpheus** (`mlx-community/orpheus-3b-0.1-ft-bf16`, 7.8k downloads) and
  **Maya1** (`mlx-community/maya1-4bit`) have weights but **no matching
  directory** in mlx-audio. Orpheus may load through the generic `llama`
  handler. Untested.
- **MiniMax Music 3's true output sample rate** (upstream says 32 kHz, mlx-audio
  says 44.1 kHz) and **max duration** (300 s vs 360 s). The docs disagree. Also
  the exact Creative Commons variant of its licence.
- **MiniCPM-o-4.5's release date** — the card cites an older MiniCPM-V paper.
  The MLX conversion landed 2026-03-05.
- **`mlx-community/Qwen-Image-Flash-*`** is tagged `mlx-swift`, not `mflux`. Not
  driven from Python here.
- **`mlx-community/TeleStyleV2-Qwen-Image-Edit-2511-bf16`** — assumed mflux
  format by analogy. Not checked.
- **FLUX.2 [max] release date** — no primary BFL source.
- **Whether Draw Things exposes FLUX/Qwen-architecture ControlNets** — the
  synced public zoo has only SD1.x/SDXL ones.
- **Any concrete report of Qwen-Image-Edit-2511 or FLUX.2 on ComfyUI+MPS at
  ~128 GB**, success or failure. No such issue exists either way.
- **No Apple-silicon speed or peak-RAM figures are published for MiniMax
  Music 3.**
- **The exact chip behind the LTX-2.5 numbers** (49.4 s / 40.76 GB). The repo
  says only "Apple Silicon, 128 GB unified" — treat as M3/M4/M5-class, not
  necessarily M5 Max.
- **The M5 Max vs M3 Ultra LTX-2.3 figure** (121 s vs 206 s) came from a
  search-index snippet of an x.com post, not the page — x.com was unfetchable.
- **No ComfyUI wall-clock video timing exists publicly for any M5-generation
  chip** — only failure reports.
- **MiniMax-H3's licence territory restriction** (reportedly excludes local
  deployment in US/EU/UK/South Korea). Read the card before building on it.
- **Whether `mlx-community/jina-reranker-v3-4bit-mxfp4` loads at all** — its
  base architecture `JinaForRanking` is in neither package's registry.
- **Whether Wan2GP or InvokeAI 6.14's Wan path work on Apple silicon** —
  Wan2GP's install docs have zero mac/mps/metal references.
- The MTEB leaderboard backend API is **undocumented and unversioned**, and its
  licence metadata has at least one confirmed error. Verify licences on model
  cards, not the leaderboard.

New in the big-iron section:

- **The ~100 GiB cliff is inferred from one model on one machine.** The Cosmos3
  bf16-vs-INT8 10.4× gap was measured on an **M4 Max 128 GB**, not an M5 Max,
  with one prompt. The mechanism (exceeding
  `max_recommended_working_set_size` → swap) is sound and the 118.55 GiB peak is
  genuinely above this machine's 115.4 GiB, but the exact location of the knee
  on *this* chip is not measured.
- **HunyuanImage 6-bit peak (~90.6 GiB) and 8-bit peak (~101.1 GiB) are my
  arithmetic**, extrapolated from the measured 19.8 GiB activation overhead at
  mixed-4/8. Only the mixed-4/8 peak (69.5 GiB) is measured.
- **The Krea 2 → `krea/Krea-2-Turbo` mapping is inferred.** Artificial Analysis
  lists three Krea 2 entries and sets no open-weights URL for any of them. If
  the mapping is wrong, the #1 open-weights T2I slot changes.
- **Artificial Analysis Elos are FLUX.2-dev-relative** — dev is pinned at
  exactly 1000.00 with zero CI in every group. They are not comparable to
  absolute or LMArena-style numbers, and CIs are ±8–10, so gaps under ~18 points
  are not resolvable.
- **Cosmos3 on MLX is essentially unvalidated**: all three MLX repos have **0
  downloads**, one machine, one prompt, and it is a Physical-AI world model
  being judged on aesthetics.
- **No published latency figures** for Qwen3.5-122B-A10B, GLM-4.6V,
  Qwen3-Omni bf16, LTX-2.5 bf16 or any mflux bf16 row — verified sizes,
  unverified speed. No measured tok/s exists for *any* ~100 GB MLX VLM on Apple
  silicon at any quantization.
- **No head-to-head between Qwen3.5-122B-A10B and Qwen3.8-Flash-Next** (~180 B,
  the newer generation). MLX download counts strongly favour Flash-Next, but
  that is popularity, not measured quality. I picked the 122 B because its
  benchmarks against the 235 B are published and its KV cost is known.
- **No listening test for Qwen3-Omni bf16 vs 8-bit.** The recommendation rests
  on the verified structural fact that the talker and code2wav are quantized in
  every quantized build, plus NVIDIA's upstream choice to keep them BF16 — not
  on a measured MOS or WER.
- **The LTX-2.5 bf16-vs-q8 wall clock (44.9 s vs 33.3 s)** came from a
  social-media snippet that could not be fetched directly (HTTP 402). The
  mechanism is documented; the numbers are indicative.
- **Wan 2.2 T2V-A14B above 480×240 is genuinely unpublished anywhere.** The
  runtime accepts 832×480 and 1280×720 but prints no timing. A 4–5 hour
  estimate for 832×480×101f/25 steps is extrapolation from the TI2V-5B pair.
- **Which runtime consumes the `Sawfwair/MiniMax-H3-*-BF16` packs** (77.09 and
  82.32 GB), and their resident footprint. Undocumented.
- **`PocketAiHub/HunyuanVideo-1.5-Distilled-MLX` Q4/BF16 revision sizes** — the
  repo is gated and returns 401 for those refs. Only the Q8 default (21.08 GB)
  is API-verified.
- **No quality data exists at all** for VibeVoice TTS, MOSS-TTS, Qwen3-TTS,
  Higgs v2/v3, VoxCPM2, Ming-omni, Breeze-TTS-2, LongCat-AudioDiT, OmniVoice,
  kugelaudio or fish-audio-s2-pro. Not "I couldn't find it" — none is published.
- Whether `mlx-community/jina-reranker-v3-4bit-mxfp4` loads (unchanged), and
  whether an MLX `Voxtral-Small-24B` exists (it is the only open model that
  clearly beats Parakeet v3 on multilingual, and I found no conversion).

- **No published bf16-vs-8-bit fidelity measurement exists for any mflux image
  model.** The argument for bf16 is "the memory is free and quantization can
  only lose information," not a measured delta. The only rigorous
  quantization-fidelity data in the image space is hymlx's own HunyuanImage
  table and the `ltx-2.5-mlx-q8` card's encoder analysis.
- **Every HunyuanImage-3.0 number is the author's own, unreproduced.** `hymlx`
  has **0 GitHub stars** and the weights repo has **67 downloads**. The stated
  verification method is unusually rigorous for a project that size, but it is
  one person's claim and nobody has independently confirmed the 2.7 min / 3.5 min
  / 69.5 GiB figures or the precision correlations.
- **Whether "precision buys memory, not speed" generalises beyond
  HunyuanImage.** hymlx measures 9.7–11.0 s/step across a 2× bit-depth range
  and concludes its path is compute-bound. I have assumed this likely holds for
  the other MLX diffusion models at these sizes, but that is inference, not
  measurement — if it is wrong, bf16 costs time as well as memory.
- **No latency figures are published for any mflux model at bf16**, on this
  chip or any other. The image-generation and image-editing big-iron rows have
  verified sizes and no verified speed.
- **Whether `Qwen3-VL-235B-A22B-Instruct-3bit` (104.02 GB) actually runs** at
  115.4 GB with a real image prompt. I recommend against it on headroom
  grounds, but I did not find a report of anyone trying.
- **`mlx-community/GLM-4.6V-8bit` appears to be an empty repo** (1 file, ~0
  bytes). Treated as broken; not confirmed with the uploader.
- **GLM-4.6V quality relative to Qwen3-VL** — I picked GLM-4.6V-6bit on size,
  licence and bit depth, not on benchmark evidence. I found no head-to-head.

Measurements in this document that came from third-party model cards or project
docs rather than from us: HunyuanImage-3.0 (69.5 GiB peak, 3.5 min/edit),
SenseNova-U1.5 (22.9 GB peak, ~7.4 s), the `mlx-gen` M5 Max video table, the
LTX-2.5 timings, and Draw Things' LongCat ledger.

**Nothing in this document was benchmarked locally.** `MODEL-COMPARISON.md`
has measured numbers for the text models; this file has none. Fill them in as
you go — one local measurement beats every table here. The video numbers in
particular vary enormously with geometry and step count, so treat them as
order-of-magnitude guidance rather than specifications.

---

## Sources

Package metadata (PyPI JSON API, fetched 2026-09-17):
`mflux` · `mlx` · `mlx-lm` · `mlx-vlm` · `mlx-audio` · `parakeet-mlx` ·
`mlx-whisper` · `mlx-embeddings` · `mlx-image` · `mlx-gen` · `diffusionkit` ·
`lightning-whisper-mlx` — via `https://pypi.org/pypi/<pkg>/json`

Model existence and sizes — HuggingFace API, `?blobs=true`, summing
`siblings[].size`:
- `https://huggingface.co/api/models?author=mlx-community&search=<term>&sort=downloads&direction=-1`
- `https://huggingface.co/api/models/<repo>?blobs=true`
- `https://huggingface.co/api/models?author=mlx-community&sort=downloads&direction=-1&limit=10` (the top-10 that confirms Parakeet is #1)

Implementation ground truth — GitHub contents and code-search APIs:
- [`Blaizzy/mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) — `mlx_vlm/models/` (240 entries), `mlx_vlm/models/{qwen3_omni_moe,nemotron_h_nano_omni,minicpmo,gemma4}/`
- [`Blaizzy/mlx-audio`](https://github.com/Blaizzy/mlx-audio) — `mlx_audio/{tts,stt,music,codec,sts,vad,lid}/models/`, `registry.py`
- [`Blaizzy/mlx-embeddings`](https://github.com/Blaizzy/mlx-embeddings) — `mlx_embeddings/models/`; open issues #72, #74, #80 (the silent-corruption bugs)
- `mlx_vlm/reranker.py`, `mlx_vlm/embedding_loader.py`, `mlx_vlm/server/{embeddings,reranking}.py` — the undocumented embedding/rerank surface
- [`Blaizzy/mlx-video`](https://github.com/Blaizzy/mlx-video) — `mlx_video/models/{wan_2,ltx_2}/`
- [`dgrauet/ltx-2-mlx`](https://github.com/dgrauet/ltx-2-mlx) — LTX-2.5 runtime; `docs/claims/C1-multishot-report.json` for the measured timings
- [`lpalbou/mlx-gen`](https://github.com/lpalbou/mlx-gen) — `docs/wan-video.md`, the M5 Max video profiles
- [`ddalcu/mlx-serve`](https://github.com/ddalcu/mlx-serve) · [`mrbizarro/Phosphene`](https://github.com/mrbizarro/Phosphene) · [`james-see/ltx-video-mac`](https://github.com/james-see/ltx-video-mac)
- [`mflux-community/mflux`](https://github.com/mflux-community/mflux) — `pyproject.toml` (`[project.scripts]`), `src/mflux/models/`, `src/mflux/models/common/config/model_config.py`, `src/mflux/cli/defaults/defaults.py`, and the per-model READMEs for `flux`, `flux2`, `qwen`, `z_image`, `boogu`, `common`
- [`ml-explore/mlx-lm`](https://github.com/ml-explore/mlx-lm) — `mlx_lm/` listing
- [`ml-explore/mlx-examples`](https://github.com/ml-explore/mlx-examples) — top-level listing (`stable_diffusion`, `flux`, `video`)
- [`Comfy-Org/ComfyUI`](https://github.com/Comfy-Org/ComfyUI) — issues #2044, #12416, #13200, #14837, #14838, #15010, #15029, #15512, #15793, #15804, #15818, #16107, #16110, #16284; `comfy/model_management.py`
- [`drawthingsai/draw-things-community`](https://github.com/drawthingsai/draw-things-community) — release notes (the two M5-Max-specific fixes) and the in-repo LongCat-Avatar M5 Max timing ledger

Model cards and upstream documentation:
- [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) — the modality table
- [`Qwen/Qwen3-Omni-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) — Thinker–Talker, speech-output languages, `disable_talker()`
- [`openbmb/MiniCPM-o-4_5`](https://huggingface.co/openbmb/MiniCPM-o-4_5)
- [`Qwen/Qwen-Image-Edit-2511`](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) · [`Qwen/Qwen-Image-2512`](https://huggingface.co/Qwen/Qwen-Image-2512)
- [`fcreait/Qwen-Image-Edit-mflux`](https://huggingface.co/fcreait/Qwen-Image-Edit-mflux) — the 2511 mflux checkpoints
- [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (Apache-2.0) · [`FLUX.2-klein-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) · [`FLUX.1-Kontext-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- [`MiniMaxAI/MiniMax-Music3`](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
- [`KaedeTai/hymlx`](https://github.com/KaedeTai/hymlx) (MIT, 0 stars, pushed 2026-09-01) — install/usage commands, the measured precision table (8/6/mixed/4-bit correlations), the M5 Max 128 GB speed table, the fixed-1 MP `ResolutionGroup` limitation, and the Tencent licence's EU/UK/South Korea exclusion
- [`KaedeTai/HunyuanImage-3.0-mlx-mixed4-8-hymlx`](https://huggingface.co/KaedeTai/HunyuanImage-3.0-mlx-mixed4-8-hymlx) — 56.32 GB verified (52.5 GiB), 67 downloads
- [`mflux-community`](https://huggingface.co/mflux-community) — 200 repos, the official `q3/q4/q5/q6/q8/bf16` mflux-save ladder; bf16 sizes verified per repo
- `src/mflux/cli/parser/parsers.py:142` — `--quantize` default is `None`, i.e. bf16
- `src/mflux/models/common/config/model_config.py` — `"qwen-image-edit"` → `model_name="Qwen/Qwen-Image-Edit-2509"` with `qwen-image-edit-2511` as an alias onto it (verified independently)
- [`mlx-community/SenseNova-U1.5-8B-MoT-8bit`](https://huggingface.co/mlx-community/SenseNova-U1.5-8B-MoT-8bit)
- [`Tongyi-MAI/Z-Image-Turbo`](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [`mlx-community/ltx-2.5-mlx-q8`](https://huggingface.co/mlx-community/ltx-2.5-mlx-q8) · [`ltx-2.5-mlx`](https://huggingface.co/mlx-community/ltx-2.5-mlx) — the encoder-bound memory analysis and the rejected int4 encoder
- [`Anes1032/Wan2.2-TI2V-5B-mlx-q8`](https://huggingface.co/Anes1032/Wan2.2-TI2V-5B-mlx-q8) · [`Wan-AI`](https://huggingface.co/Wan-AI) org listing (open weights stop at 2.2)
- [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) · [`ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit`](https://huggingface.co/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit)
- [`Lightricks/LTX-2`](https://huggingface.co/Lightricks/LTX-2) — and the absence of any macOS claim in it
- [`mlx-community/Qwen3-Reranker-0.6B-4bit`](https://huggingface.co/mlx-community/Qwen3-Reranker-0.6B-4bit) — the `softmax([logit("no"), logit("yes")])[1]` scoring recipe
- [`mlx-community/nomicai-modernbert-embed-base-bf16`](https://huggingface.co/mlx-community/nomicai-modernbert-embed-base-bf16) · [`majentik/harrier-oss-v1-0.6b-MLX-8bit`](https://huggingface.co/majentik/harrier-oss-v1-0.6b-MLX-8bit) · [`mlx-community/Nemotron-3-Embed-1B-BF16-8bit`](https://huggingface.co/mlx-community/Nemotron-3-Embed-1B-BF16-8bit)
- MTEB leaderboard backend (undocumented): `https://mteb-leaderboard-backend.hf.space/v1/benchmarks/MTEB%28eng%2C%20v2%29/scores`

Vendor announcements:
- [BFL — FLUX.2](https://bfl.ai/blog/flux-2) (2025-11-25) · [FLUX.2 klein](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence) (2026-01-15) · [FLUX.1 Kontext dev](https://bfl.ai/announcements/flux-1-kontext-dev) (2025-06-26) · [FLUX 3](https://bfl.ai/blog/flux-3) (2026-07-23, weights not released)
- [Draw Things — Metal Quantized Attention, M5 Max Int8](https://releases.drawthings.ai/p/metal-quantized-attention-pulling) (2026-04-01) — the ~110 TFLOPs / 3.3×-over-M4-Max figures
