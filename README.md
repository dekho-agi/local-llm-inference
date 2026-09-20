# local-llm-inference

A simple toolkit for **local LLM inference**, built to plug into
[opencode](https://opencode.ai) so you can keep working when a hosted model
isn't available — on a plane, on a locked-down network, or just off the clock.

One command detects your hardware, picks a model that fits, serves it on an
OpenAI-compatible endpoint, and registers it in opencode automatically.

```bash
./llmctl.sh host                # what machine is this, and what fits
./llmctl.sh pull                # download a model (pick from a list)
./llmctl.sh start               # serve it (pick from a list)
./llmctl.sh opencode --install  # one-time: opencode discovers your models
opencode                        # /models → Dekho Local Inference
./llmctl.sh stop                # done
```

---

## What this repo is

Inference configs organised **by hardware target**, plus a toolkit that picks
the right one for the machine you're on.

| Target | Hardware | Runtime |
|---|---|---|
| `apple-m-series/m5-128gb/` | MacBook Pro M5 Max, 128 GB | native MLX |
| `apple-m-series/m2-16gb/` | MacBook Air M2, 16 GB | native MLX |
| `rtx-pro-6000/` | RTX Pro 6000 Blackwell, 96 GB | Docker + vLLM |
| `rtx-3080-16gb/` | RTX 3080 / any Ampere 16 GB | Docker + vLLM |

Apple targets run **natively, never in Docker** — Docker Desktop on macOS has
no Metal passthrough, so a container is CPU-only and 5–10× slower. The NVIDIA
targets are Docker because Linux + nvidia-container-toolkit gives real GPU
passthrough.

Adding a machine means adding a profile directory; `llmctl` maps memory size to
a profile automatically, so `./llmctl.sh host` should be right on a new box
without configuration.

---

## llmctl

Everything is driven by one CLI at the repo root. It uses only libraries mlx-lm
already installs — typer, rich, huggingface_hub, mlx — so there is nothing
extra to set up.

| Command | |
|---|---|
| `host` | Detected chip, cores, memory, **GPU working-set ceiling**, and the matching profile |
| `models` | What's cached, what it costs at context, whether it can tool-call, whether it fits. `--all` includes the catalog; `--class image` filters |
| `pull [model]` | Download into the local cache. No argument ⇒ numbered picker |
| `start [model]` | Serve it. No argument ⇒ numbered picker. Auto-assigns a port |
| `ps` | Running servers: port, model, context, uptime, health |
| `stop` | No argument ⇒ numbered picker, `0` for all. Or `--port N` / `--all` |
| `monitor` | GPU utilization and **real** memory use. `-w` to watch |
| `verify` | Shard integrity per model, plus its tool parser. Run before you lose network |
| `cache` | Cache usage by model; `--prune` to delete one interactively |
| `opencode` | Integration status; `--install` to install the discovery plugin |

### Running several models at once

`start` takes the first free port from 8000, so a big model and a small one can
run side by side:

```bash
./llmctl.sh start Qwen3-Coder-Next     # :8000
./llmctl.sh start Qwen3-Coder-30B      # :8001
./llmctl.sh ps
```

```
port  model                               ctx    pid   up  health  endpoint
8000  Qwen3-Coder-Next-4bit              256k  38326  68s  ok      http://127.0.0.1:8000/v1
8001  Qwen3-Coder-30B-A3B-Instruct-4bit  256k  38366  58s  ok      http://127.0.0.1:8001/v1
```

Each server is tracked in `~/.cache/dekho-local-inference/servers/`, rebuilt
from disk on every read, so a crashed server never lingers as a phantom entry.

**Prefer separate servers over switching models in one.** A single
`mlx_lm.server` *can* hot-swap, but swapping costs ~3× throughput until
restart — see [CONTEXT.md](apple-m-series/m5-128gb/CONTEXT.md).

---

## opencode integration

`./llmctl.sh opencode --install` symlinks a plugin into
`~/.config/opencode/plugins/`. After that, model discovery is automatic:
launch anything and it appears in `/models`.

It registers **one provider per running server** (a provider has exactly one
baseURL), merging two sources:

| Source | Supplies |
|---|---|
| `GET <endpoint>/models` | what each server can reach right now |
| `manifest.json`, rewritten by `llmctl` on every start/stop | context window, generation ceiling, and the tool parser |

The manifest is necessary because these are local repo ids that models.dev has
no entry for, so opencode would otherwise know neither the context window
(which drives compaction) nor whether a model can tool-call at all.

Models are tagged in the picker: `[offline]` if cached but not currently
served, `[no tools]` if mlx-lm has no tool parser for them, `[swaps in]` if
picking them would force that server to swap models. A model hand-pinned in
your own `opencode.json` always wins over discovery.

### Tool calling is a hard gate

opencode drives models through tool calls, and mlx-lm can only parse them for
chat templates it has a parser for. `llmctl models` shows a `tools` column;
anything `no` is chat-only. **`gpt-oss-120b` is the notable failure** — mlx-lm
ships no parser for its harmony format, and it fails *silently*: no error, and
raw `<|channel|>commentary to=functions.…` syntax in the message content.

---

## Models

`llmctl models --all` lists the catalog. Reference documents:

| Document | Contents |
|---|---|
| [MODEL-COMPARISON.md](apple-m-series/m5-128gb/MODEL-COMPARISON.md) | Coding models: sizes, quantization, published benchmarks **with sources** |
| [CONTEXT.md](apple-m-series/m5-128gb/CONTEXT.md) | What a context window costs, measured, and how to evaluate one |
| [GENERATIVE-MODELS.md](apple-m-series/m5-128gb/GENERATIVE-MODELS.md) | Image, image-editing, VLM, omni, STT, TTS, music, video, embeddings — including which larger variants are worth the memory and which are not |

Current coding picks on the 128 GB profile:

| Model | Size | Role |
|---|---|---|
| `Qwen3-Coder-Next-4bit` | 44.9 GB | Daily driver — 80B/3B active, 256k context |
| `Qwen3-Coder-30B-A3B-Instruct-4bit` | 17.2 GB | Fast, battery-friendly |

A counterintuitive one worth knowing: **the 80B is 4× cheaper per token of
context than the 30B**, because only 12 of its 48 layers hold a KV cache. Long
agentic sessions are where the bigger model is *most* worth its size.

### Generative models: `llmctl gen`

opencode speaks `/v1/chat/completions` with tool calls — the right shape for a
coding agent, the wrong shape for diffusion and audio. So generative models get
their own path, deliberately: **opencode for coding work, `llmctl gen` for
everything else.**

```bash
./llmctl.sh gen setup            # one-time: creates the generative env
./llmctl.sh gen list             # classes, runtime readiness, models cached
./llmctl.sh gen doctor           # what's installed and what isn't

./llmctl.sh gen run image "a red bicycle against a white wall" -o bike.png
./llmctl.sh gen run vlm "transcribe this" --input scan.png
./llmctl.sh gen run tts "hello there" -o ./speech
./llmctl.sh gen run stt --input speech/audio_000.wav -o transcript
./llmctl.sh gen run music "warm acoustic guitar, instrumental" -o track.wav
```

`--dry-run` prints the exact command without running it, which is the fastest
way to see what a runtime is being asked to do.

### One server for every generative role

`mlx_vlm.server` serves all of them at once over an OpenAI-compatible API, so
models load once instead of per call:

```bash
./llmctl.sh gen serve --auto          # picks a cached model per role
./llmctl.sh gen serve --tts mlx-community/Kokoro-82M-bf16 --embed Qwen3-Embedding
./llmctl.sh ps                        # shows it as kind=gen with its roles
```

| Endpoint | Verified |
|---|---|
| `POST /v1/embeddings` | 1024-dim vectors |
| `POST /v1/rerank` | needs `--rerank` |
| `POST /audio/speech` | pass `response_format: "wav"` — mp3 needs ffmpeg |
| `POST /audio/transcriptions` | multipart `file=@audio.wav`; round-tripped TTS output back to text |
| `POST /v1/chat/completions` | works with the VLM role |
| `POST /images/generations` | **does not work** with quantized repos — only accepts canonical `black-forest-labs/*` ids. Use `llmctl gen run image` (mflux). |

The generative server is deliberately **not** advertised to opencode: it speaks
the same paths, but its models are TTS/STT/embedding, which are useless as chat
models. `llmctl ps` shows both kinds side by side.

The runtimes live in a **separate conda env** (`dekho-apple-gen`) so a
dependency conflict there cannot break the mlx-lm serving env your coding
workflow depends on.

Each runtime names its flags differently — mlx-audio's TTS wants `--text` and
`--output_path`, its STT wants `--audio` and `--output-path`, mlx-vlm wants
`--image`, mflux wants `--image-paths` — and mflux ships a console script per
model family. `llmctl/gen.py` records the real names (read from each `--help`,
then confirmed by running them) and the catalog records which script a model
needs, so you don't have to remember any of it.

Measured on the M5 Max: FLUX.2-Klein-4B renders 1024x1024 in **11 s** at
13.34 GB peak; MiniMax-Music3 produced 21 s of 44.1 kHz stereo in 47 s; Kokoro
TTS and parakeet STT round-trip a sentence back verbatim.

---

## Setup on a new machine

```bash
conda env create -f apple-m-series/common/environment.yml
./llmctl.sh host
./llmctl.sh pull
./llmctl.sh opencode --install
```

Agents should read [AGENTS.md](AGENTS.md) and the
[`setup-local-inference`](agents/skills/setup-local-inference/SKILL.md) skill,
which walks the whole path and refuses to claim success until a real tool call
has run offline.

A HuggingFace token is picked up from `~/.dekho/hugging-face-token.txt` (or
`$HF_TOKEN`) and never printed. Every model here is public, so it's optional —
but unauthenticated hub requests are rate-limited.

---

## Before you lose connectivity

See [FLIGHT.md](FLIGHT.md) for the runbook — every step in it was run on the
machine with `HF_HUB_OFFLINE=1`.

```bash
./llmctl.sh verify     # every shard complete, tool parser per model
./llmctl.sh start --offline   # proves it boots with HF_HUB_OFFLINE=1
```

Weights resolve through the HuggingFace hub, so a repo id that was never
cached fails offline no matter how much disk you have. `verify` is the check
that catches it while you can still fix it.

---

## NVIDIA targets

Unchanged Docker Compose stacks, each exposing `:8000/v1`:

```bash
cd rtx-pro-6000/nemotron-120b
cp .env.example .env          # set HF_TOKEN
docker compose up -d
```

`llmctl` is Apple-silicon only for now; the NVIDIA stacks are driven by
Compose. The Kubernetes/Helm path used to live here and has moved to
[`gagandaroach/dlan`](https://github.com/gagandaroach/dlan).

---

## How to use each model

Every command here has been run on this machine. Timings and peak memory are
measured, not estimated — see [VALIDATION.md](VALIDATION.md) for the full
per-model results.

### Coding — launch opencode against a local LLM

```bash
./llmctl.sh start                                  # numbered picker
./llmctl.sh start Qwen3-Coder-Next                 # or name it
opencode                                           # /models → Dekho Local Inference
```

That's the whole flow: `start` writes the manifest, the plugin registers the
provider, and opencode picks it up. To drive it without the TUI:

```bash
opencode run --model dekho-local-inference/mlx-community/Qwen3-Coder-Next-4bit \
  "implement parse_kv in parse.py using the edit tool"
```

| Model | Size | Use it when |
|---|---|---|
| `Qwen3-Coder-Next-4bit` | 44.9 GB | Default. 80B/3B active, 256k context, best tool-call robustness |
| `Qwen3-Coder-30B-A3B-Instruct-4bit` | 17.2 GB | On battery, or quick edits — 95–114 tok/s |
| `GLM-4.7-Flash-4bit` | 16.9 GB | Same size class, different family for a second opinion |
| `Devstral-Small-2-24B-Instruct-2512-4bit` | 15.1 GB | Dense, so no MoE routing variance |
| `GLM-4.5-Air-4bit` | 60.2 GB | Heavier alternative, 106B/12B active |
| `Qwen2.5-Coder-7B-Instruct-4bit` | 4.3 GB | Small machines, or leaving memory for generative work |
| `gpt-oss-120b-MXFP4-Q8` | 63.4 GB | **Chat only — cannot tool-call**, so not usable as an agent |

### Run several at once

Each gets its own port, and both kinds coexist:

```bash
./llmctl.sh start Qwen3-Coder-Next        # :8000
./llmctl.sh start Qwen3-Coder-30B         # :8001
./llmctl.sh gen serve --auto              # :8002, all generative roles
./llmctl.sh ps
```

Prefer this over switching models inside one server: hot-swapping costs ~3x
throughput until restart.

### Generate an image

```bash
./llmctl.sh gen run image "a red bicycle against a white wall, photograph" \
  -o bike.png                                      # FLUX.2-Klein-4B, ~11 s

./llmctl.sh gen run image "a lighthouse at dawn, long exposure" \
  -m krea-2-turbo -o lighthouse.png --steps 8      # higher quality, ~46 s
```

| Model | Size | Time | Peak |
|---|---|---|---|
| `FLUX.2-Klein-4B-6bit` | 6.6 GB | 11 s (4 steps) | 13.3 GB |
| `krea-2-turbo-mflux-bf16` | 34.2 GB | 46 s (8 steps) | 41.7 GB |
| `ideogram-4-mflux-q8` | 26.0 GB | — | licence-gated on HF |

### Edit an image (instruction-guided)

```bash
./llmctl.sh gen run image-edit \
  "change the bicycle from red to bright yellow, keep everything else identical" \
  -i bike.png -o yellow.png                        # ~7 min, 58.5 GB peak
```

This is the locally-runnable equivalent of the hosted image-editing tools. It
preserved the saddle, bottle cage, tyre walls and background while recolouring
only the frame.

### Read an image (vision / OCR)

```bash
./llmctl.sh gen run vlm "Transcribe all text in this image." -i scan.png
./llmctl.sh gen run vlm "Describe this in one sentence." -i photo.png -m Qwen3-VL
```

| Model | Size | Good for |
|---|---|---|
| `GLM-OCR-8bit` | 1.6 GB | Documents, screenshots, receipts |
| `Qwen3-VL-30B-A3B-Instruct-8bit` | 33.5 GB | General description and reasoning about images |
| `Qwen3.5-122B-A10B-5bit` | 84.9 GB | Best quality; hybrid attention keeps KV at 24 KB/token |

### Speech

```bash
./llmctl.sh gen run tts "The quick brown fox jumps over the lazy dog." -o ./speech
./llmctl.sh gen run stt -i speech/audio_000.wav -o transcript
cat transcript.txt
```

Kokoro (0.39 GB) and parakeet (2.5 GB) are both at the quality ceiling for
their tasks — larger ASR models measure *worse*, so there is nothing to gain by
scaling up here.

Over HTTP, via the unified server:

```bash
./llmctl.sh gen serve --auto
curl localhost:8002/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Kokoro-82M-bf16","input":"hello","response_format":"wav"}' \
  -o hello.wav                                     # mp3 needs ffmpeg; wav does not
curl localhost:8002/audio/transcriptions -F file=@hello.wav \
  -F model=mlx-community/parakeet-tdt-0.6b-v3
```

### Generate music

```bash
./llmctl.sh gen run music "warm acoustic guitar, slow fingerpicking, instrumental" \
  -o track.wav --steps 20                          # ~47 s → 21 s of 44.1 kHz stereo
```

`--lyrics` is required by the runtime and defaults to `[instrumental]`; pass
your own for vocals.

### Generate video

```bash
./llmctl.sh gen run video "a red balloon rising into a blue sky" -o clip.mp4
```

**Plug in first.** Measured: 3.4 s of 1280×704 in ~19 minutes. The default
resolution is what costs the time — pass `--width 832 --height 480` to cut it
down substantially.

### Embeddings and reranking

```bash
./llmctl.sh gen serve --embed Qwen3-Embedding --rerank Qwen3-Reranker
curl localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3-Embedding-0.6B-8bit","input":["hello","world"]}'
```

Returns 1024-dim vectors. Avoid `mlx-embeddings` with ModernBERT — an open bug
returns silent all-NaN vectors on mixed-length batches.

### Things that do not work, and why

| Attempt | Outcome |
|---|---|
| `gpt-oss-120b` as an opencode agent | mlx-lm has no parser for its harmony tool format, and it fails **silently** — raw syntax in the message content |
| Speech out from `Qwen3-Omni` | The weights ship the talker, but mlx-vlm 0.7.1 defines no `generate_audio`. Use Kokoro |
| `index-tts2-mlx` | Ships `config.yaml`; mlx-audio's loader requires `config.json` |
| `/images/generations` on the unified server | Only accepts canonical `black-forest-labs/*` ids, not quantized repos. Use `gen run image` |
| `ideogram-4-mflux-q8` | Licence-gated on HuggingFace; accept it on the model page first |

---

## Development

```bash
pip install -e ".[test]"
pytest                      # 59 tests, no GPU or model weights needed
ruff check llmctl tests
ruff format llmctl tests
```

Tests deliberately avoid importing `mlx`, which is Apple-silicon only, so they
run on Linux CI. What they cover is the bookkeeping that decides whether the
tool tells you the truth: KV-cache arithmetic anchored to measured values,
port allocation, registry pruning of dead servers, the manifest shape opencode
depends on, `vmmap` size parsing, and catalog invariants. Anything needing
Metal or real weights is an integration check you run on the machine
(`llmctl verify`, `smoke-test.py`).

CI runs tests on Python 3.11/3.12/3.13, ruff, shellcheck, an ES-module parse of
the opencode plugin, and a scan for credentials or personal email addresses in
the tree.

---

## Seeing what's actually running

`htop` and `ps` cannot show MLX memory — weights live in Metal buffers that
macOS doesn't count in RSS, so a server reports 16 GB RSS while holding 42 GB.

```bash
./llmctl.sh monitor -w
```

Reads GPU utilization from `ioreg`, Metal buffer residency and footprint peak
from `vmmap`, and pageouts from `vm_stat`. If pageouts climb you're over the
GPU working-set ceiling (115.4 GB on an M5 Max, **not** the full 128 GB).
