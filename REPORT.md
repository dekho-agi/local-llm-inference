# Running AI models locally on a MacBook Pro (M-series, 128 GB)

A practical report for the team. Everything here was run on one machine —
MacBook Pro, Apple M5 Max, 128 GB unified memory, macOS 26.5 — and every
number is measured, not estimated.

**The short version:** a 128 GB MacBook runs a genuinely useful local stack.
Coding agents, image generation and editing, vision, speech both directions,
music and video all work offline. The main surprises are that the usable
memory ceiling is lower than you'd think, and that "the model downloaded" is
not the same as "the model works".

---

## 1. Start here (about 15 minutes, plus download time)

```bash
git clone https://github.com/dekho-agi/local-llm-inference
cd local-llm-inference
conda env create -f apple-m-series/common/environment.yml

./llmctl.sh host          # confirms what your machine can hold
./llmctl.sh pull          # pick a model from a list
./llmctl.sh start         # serve it
./llmctl.sh opencode --install
opencode                  # /models → Dekho Local Inference
```

That is the whole path to a local coding agent. `llmctl` detects your
hardware, picks a matching profile, and registers whatever you launch with
opencode automatically — there is no per-model configuration to write.

If you only ever run one command from this report, make it
`./llmctl.sh host`. It tells you the real ceiling for your specific machine.

---

## 2. What your machine can actually hold

| | |
|---|---|
| Installed | 128 GB |
| GPU working-set ceiling | **115.4 GB** |
| Plan against | **~100 GB** |

Crossing it swaps, and not gradually — same prompt and seed, bf16 at
118.55 GiB peak took **730.96 s** against INT8 at 67.63 GiB in **70.54 s**.
10.4x, from paging. Below the ceiling, higher precision is free or better.

Context cost is architectural, not size-based: Qwen3-Coder-Next (80B) is **4x
cheaper per context token** than the 30B, because only 12 of its 48 layers
hold a KV cache. For long agent sessions the bigger model is the cheaper one.

Run `./llmctl.sh host` for your machine. Full detail:
[docs/hardware.md](docs/hardware.md).

## 3. What works

All verified by running the model and checking the output is real — a PNG
whose header parses, a WAV that opens, an actual transcription. Never "the
command exited zero".

### Coding agents

| Model | Size | Speed | Notes |
|---|---|---|---|
| `Qwen3-Coder-Next-4bit` | 44.9 GB | — | **Default.** 80B/3B active, 256k context, best tool-call reliability |
| `Qwen3-Coder-30B-A3B` | 17.2 GB | 95–114 tok/s | Battery-friendly, loads in seconds |
| `Qwen2.5-Coder-7B` | 4.3 GB | fast | For smaller machines, or to leave room for image work |

Verified end to end: opencode read a file, called the edit tool, and wrote
correct working code in ~30 s — with networking disabled.

### Vision

| Model | Size | Time | Good for |
|---|---|---|---|
| `GLM-OCR-8bit` | 1.6 GB | 3 s | Documents, screenshots, receipts |
| `Qwen3-VL-30B-A3B-8bit` | 33.5 GB | 6 s | General image understanding |
| `Qwen3.5-122B-A10B-5bit` | **84.9 GB** | 16 s | Best quality. Yes, an 85 GB model runs fine |

### Image generation and editing

| Model | Size | Time | Peak |
|---|---|---|---|
| `FLUX.2-Klein-4B-6bit` | 6.6 GB | **6–11 s** | 13.3 GB |
| `krea-2-turbo-mflux-bf16` | 34.2 GB | 44 s | 41.7 GB |
| `qwen-image-edit-2511-bf16` | 56.6 GB | 86 s–7 min | 58.5 GB |

Image editing is the standout: given a photo and "change the bicycle from red
to bright yellow, keep everything else identical", it recoloured only the
frame and left the saddle, bottle cage, tyre walls and background intact.
That is the locally-runnable equivalent of the hosted editing tools.

### Speech, music, video, embeddings

| Task | Model | Size | Result |
|---|---|---|---|
| Text → speech | `Kokoro-82M` | 0.39 GB | 3.2 s of 24 kHz audio in ~3 s |
| Speech → text | `parakeet-tdt-0.6b-v3` | 2.5 GB | Transcribed Kokoro's output verbatim |
| Music | `MiniMax-Music3` | 13.9 GB | 21 s of 44.1 kHz stereo in 58 s |
| Video | `Wan2.2-TI2V-5B` | 19.6 GB | 3.4 s at 1280×704 in ~19 min |
| Embeddings | `Qwen3-Embedding-0.6B` | 0.7 GB | 1024-dim vectors |

**Do not scale up speech models.** `parakeet` at 2.5 GB beats every larger MLX
ASR model on word error rate — the 1.1B version is 47% *worse* on long-form
audio. Kokoro at 0.39 GB ties models 7× its size. Bigger buys voice cloning,
not quality.

---

## 4. What does not work

Worth knowing before you spend an afternoon on it.

| Thing | Why |
|---|---|
| **`gpt-oss-120b` via opencode** | The model tool-calls correctly, and `llmctl patch-mlx-lm` makes mlx-lm deliver the calls. opencode still rejects it because harmony channel markers leak into `content` — a second mlx-lm gap (upstream #1867). Usable over the raw API. |
| **Speech output from omni models** | Qwen3-Omni ships the audio weights (417 talker tensors), but mlx-vlm 0.7.1 has no `generate_audio`. Text and vision work. Use Kokoro for speech. |
| **`index-tts2-mlx`** | Ships `config.yaml`; mlx-audio requires `config.json`. Use `mlx-community/IndexTTS-2-MLX` instead. |
| **Image generation over the unified server** | `/images/generations` only accepts canonical `black-forest-labs/*` ids, not quantized repos. Use the CLI path. |
| **`mlx-gen`** | Pins `mlx<0.32.0` and will downgrade your working install. Do not put it in the same environment. |
| **`mlx-video` from PyPI** | Wrong package — video I/O helpers, no generation. Install from git. |

---

## 5. Five things that will save you hours

**1. A model that downloaded is not a model that works.** `hf download` exited
zero on a download that was 28 GB short, and a failed pull leaves the
directory, snapshot and `config.json` behind so everything *looks* fine until
load time. Always run `./llmctl.sh verify`.

**2. Tool calling is a hard gate for agent work.** A model either has a chat
template mlx-lm can parse tool calls from, or it cannot drive opencode at all —
and the failure is silent. Check before downloading 60 GB:

```bash
python apple-m-series/m5-128gb/check-tool-parser.py <repo-id>
```

**3. Don't switch models inside one server.** Hot-swapping costs about **3×
throughput** until restart (95–114 tok/s dropped to 36–38). Stop and restart
with the model you want, or run two servers on different ports.

**4. `htop` cannot see MLX memory.** Weights live in Metal buffers macOS does
not count in RSS — a server shows 16 GB RSS while holding 42 GB. Use
`./llmctl.sh monitor`, which reads `ioreg` and `vmmap`.

**5. Download serially.** Six parallel pulls saturated the link and four of
them failed. One at a time with retries is faster in wall-clock terms.

---

## 6. What to download first

Disk is cheap; your time and bandwidth are not. Suggested order:

| Priority | Models | Size | Gets you |
|---|---|---|---|
| 1 | `Qwen3-Coder-30B-A3B` | 17 GB | A working coding agent, fast |
| 2 | `FLUX.2-Klein-4B` + `Kokoro` + `parakeet` | 10 GB | Images and speech both ways |
| 3 | `Qwen3-Coder-Next` | 45 GB | The better coding model for long sessions |
| 4 | `GLM-OCR` or `Qwen3-VL-30B` | 2–34 GB | Reading screenshots and documents |
| 5 | `qwen-image-edit-2511` | 57 GB | Instruction-guided image editing |
| 6 | `Qwen3.5-122B-5bit` | 85 GB | Best vision quality, if you have the disk |

The full set used here is about 480 GB. Steps 1–2 are ~27 GB and already
useful.

---

## 7. Using it day to day

```bash
./llmctl.sh ps                # what's running
./llmctl.sh monitor -w        # GPU and real memory, live
./llmctl.sh models            # what's cached and whether it fits
./llmctl.sh stop              # pick one, or --all
```

Per-task recipes: [README](README.md#using-each-model). Per-model status and
known failures: [docs/models.md](docs/models.md). Test results:
[VALIDATION.md](VALIDATION.md).

## 8. Honest limitations

- Everything here is measured on **one machine**, an M5 Max with 128 GB. A
  64 GB machine will not run the 85 GB models, and `llmctl host` will tell you
  so rather than letting you find out at load time.
- Generative models do **not** go through opencode. It speaks
  `/v1/chat/completions` with tool calls, which is the wrong shape for
  diffusion and audio, so those use `llmctl gen` instead.
- Video is slow enough (~19 min for 3.4 s) that it is a batch job, not
  something to iterate on interactively.
- The model landscape moves fast. Sizes and names here were verified against
  the HuggingFace API in September 2026; re-check before trusting them.

---

*Generated from work in [dekho-agi/local-llm-inference](https://github.com/dekho-agi/local-llm-inference).
Reproduce any of it with `./llmctl.sh validate`.*
