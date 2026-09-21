# local-llm-inference

Local LLM and generative model inference on Apple silicon and NVIDIA, wired
into [opencode](https://opencode.ai).

```bash
conda env create -f apple-m-series/common/environment.yml
./llmctl.sh host                  # what this machine can hold
./llmctl.sh pull                  # download a model (picker)
./llmctl.sh start                 # serve it (picker, auto port)
./llmctl.sh opencode --install    # once
opencode                          # /models → Dekho Local Inference
```

## What works

All verified by loading the model and checking the output is real — a PNG whose
header parses, a WAV that opens, an actual transcription. Not "the command
exited zero". Full per-model status in [docs/models.md](docs/models.md), test
results in [VALIDATION.md](VALIDATION.md).

| Task | Model | Size | Measured |
|---|---|---|---|
| Coding agent | `Qwen3-Coder-Next-4bit` | 44.9 GB | Drove an offline opencode edit, correct code in ~30 s |
| Coding, fast | `Qwen3-Coder-30B-A3B` | 17.2 GB | 95–114 tok/s |
| Vision | `Qwen3.5-122B-A10B-5bit` | 84.9 GB | 16 s — an 85 GB model runs fine |
| Vision, small | `GLM-OCR-8bit` | 1.6 GB | 3 s, reads text from images |
| Image gen | `FLUX.2-Klein-4B` | 6.6 GB | 1024² in 6–11 s, 13.3 GB peak |
| Image gen, better | `krea-2-turbo-bf16` | 34.2 GB | 1024² in 44 s, 41.7 GB peak |
| Image edit | `qwen-image-edit-2511-bf16` | 56.6 GB | 86 s–7 min, 58.5 GB peak |
| Speech both ways | `Kokoro` + `parakeet` | 2.9 GB | Round-trips a sentence verbatim |
| Music | `MiniMax-Music3` | 13.9 GB | 21 s of 44.1 kHz stereo in 58 s |
| Video | `Wan2.2-TI2V-5B` | 19.6 GB | 3.4 s at 1280×704 in ~19 min |
| Embeddings | `Qwen3-Embedding-0.6B` | 0.7 GB | 1024-dim |

Image editing is the standout: given a photo and "change the bicycle from red
to bright yellow, keep everything else identical", it recoloured only the frame
and left the saddle, bottle cage, tyre walls and background intact.

**Don't scale up speech models.** `parakeet` at 2.5 GB beats every larger MLX
ASR model on word error rate — the 1.1B is 47% *worse* on long-form. Bigger
buys voice cloning, not quality.

## What doesn't work

| | |
|---|---|
| `gpt-oss-120b` via opencode | Tool calling works after `llmctl patch-mlx-lm`; opencode still rejects it because harmony channel markers leak into `content`. Raw API is fine |
| Speech out from omni models | Qwen3-Omni ships the talker weights but mlx-vlm defines no `generate_audio`. Use Kokoro |
| `/images/generations` on the gen server | Only accepts canonical `black-forest-labs/*` ids. Use `gen run image` |
| `ideogram-4` | Licence-gated on HuggingFace; accept it on the model page |
| `mlx-gen`, PyPI `mlx-video` | One downgrades mlx; the other is an unrelated package |

## Memory reality

| | |
|---|---|
| Installed | 128 GB |
| GPU working-set ceiling | **115.4 GB** |
| Plan against | **~100 GB** |

Crossing it swaps, and not gradually — same prompt and seed, bf16 at
118.55 GiB peak took **730.96 s** against INT8 at 67.63 GiB in **70.54 s**.
10.4×, from paging.

Context cost is architectural, not size-based: Qwen3-Coder-Next (80B) is **4×
cheaper per context token** than the 30B, because only 12 of its 48 layers hold
a KV cache. For long agent sessions the bigger model is the cheaper one.

`./llmctl.sh host` reports your machine. Detail:
[docs/hardware.md](docs/hardware.md).

## Five things that save hours

1. **A downloaded model is not a working model.** `hf download` exited 0 on a
   download 28 GB short, and a failed pull leaves the directory and
   `config.json` behind. Run `./llmctl.sh verify`.
2. **Tool calling is a hard gate and fails silently.** Check before spending a
   download: `python apple-m-series/m5-128gb/check-tool-parser.py <repo-id>`.
3. **Don't switch models inside one server.** ~3× throughput penalty until
   restart (95–114 tok/s → 36–38). Restart, or use two ports.
4. **`htop` can't see MLX memory.** Weights live in Metal buffers macOS
   excludes from RSS — 16 GB RSS while holding 42 GB. Use
   `./llmctl.sh monitor`.
5. **Download serially.** Six parallel pulls saturated the link and four
   failed.

## What to download first

| Step | Models | Size | Gets you |
|---|---|---|---|
| 1 | `Qwen3-Coder-30B-A3B` | 17 GB | A working coding agent |
| 2 | `FLUX.2-Klein-4B` + `Kokoro` + `parakeet` | 10 GB | Images and speech |
| 3 | `Qwen3-Coder-Next` | 45 GB | The better coding model for long sessions |
| 4 | `GLM-OCR` or `Qwen3-VL-30B` | 2–34 GB | Reading screenshots and documents |
| 5 | `qwen-image-edit-2511` | 57 GB | Instruction-guided image editing |
| 6 | `Qwen3.5-122B-5bit` | 85 GB | Best vision quality |

Steps 1–2 are ~27 GB and already useful. The full set here is ~480 GB.


## Layout

```
llmctl/            the CLI: host detection, catalog, servers, cache, validate
llmctl.sh          launcher (resolves the conda env)
opencode/plugins/  opencode model-discovery plugin
apple-m-series/    MLX profiles — m5-128gb, m2-16gb, shared common/
rtx-3080-16gb/     Docker Compose + vLLM
rtx-pro-6000/      Docker Compose + vLLM
docs/              hardware limits, model reference
```

Apple targets run natively: Docker Desktop on macOS has no Metal passthrough,
so a container is CPU-only. NVIDIA targets use Compose, where passthrough is
real. `llmctl` is Apple-silicon only.

## Commands

| | |
|---|---|
| `host` | Chip, cores, memory, GPU working-set ceiling, matching profile |
| `models` | Cached state, cost at context, tool-call support, fit. `--all`, `--class`, `--ctx` |
| `pull [model]` | Download. No argument ⇒ picker |
| `start [model]` | Serve. No argument ⇒ picker. Auto-assigns a port from 8000. Honours `MODEL` |
| `ps` | Running servers, both kinds, with health |
| `stop` | Picker (`0` = all), or `--port` / `--all` |
| `monitor` | GPU utilization and real memory. `-w` to watch |
| `verify` | Shard integrity and tool parser per model |
| `validate` | Load and run each model; writes markdown. `--class`, `--out` |
| `cache` | Usage by model; `--prune` |
| `opencode` | Integration status; `--install` |
| `gen` | Generative models — `list`, `doctor`, `setup`, `run`, `serve` |
| `patch-mlx-lm` | Install the harmony tool parser (gpt-oss). `--undo` |

## Several models at once

```bash
./llmctl.sh start Qwen3-Coder-Next      # :8000
./llmctl.sh start Qwen3-Coder-30B       # :8001
./llmctl.sh gen serve --auto            # :8002, all generative roles
./llmctl.sh ps
```

Prefer this to switching models inside one server — hot-swapping costs ~3x
throughput until restart. Each server is tracked in
`~/.cache/dekho-local-inference/servers/`, rebuilt from disk on read, so a
killed server never lingers.

## opencode

`./llmctl.sh opencode --install` symlinks a plugin that registers one provider
per running LLM server. It merges the live `/v1/models` list with a manifest
`llmctl` rewrites on every start/stop, which supplies the context window and
tool-call verdict — models.dev has no entries for local repo ids.

Models are tagged `[offline]`, `[no tools]` or `[swaps in]` in the picker. A
model hand-pinned in your own `opencode.json` wins over discovery. The
generative server is deliberately excluded: its models are TTS/STT/embedding.

Tool calling is a hard gate, and the failure is silent. Check before spending
a download:

```bash
python apple-m-series/m5-128gb/check-tool-parser.py <repo-id>
```

## Using each model

### Coding

```bash
./llmctl.sh start Qwen3-Coder-Next
opencode
opencode run --model dekho-local-inference/mlx-community/Qwen3-Coder-Next-4bit \
  "implement parse_kv in parse.py using the edit tool"
```

### Images

```bash
./llmctl.sh gen run image "a red bicycle against a white wall" -o bike.png
./llmctl.sh gen run image "a lighthouse at dawn" -m krea-2-turbo -o l.png --steps 8
./llmctl.sh gen run image-edit "make the bicycle yellow" -i bike.png -o y.png
```

### Vision, speech, music, video

```bash
./llmctl.sh gen run vlm "Transcribe all text." -i scan.png
./llmctl.sh gen run tts "hello there" -o ./speech
./llmctl.sh gen run stt -i speech/audio_000.wav -o transcript
./llmctl.sh gen run music "warm acoustic guitar, instrumental" -o t.wav --steps 20
./llmctl.sh gen run video "a red balloon rising" -o clip.mp4   # ~19 min, plug in
```

`--dry-run` prints the command without running it. `--model/-m`, `--steps`,
`--seed`, `--lyrics` where the runtime supports them.

### One endpoint for generative roles

```bash
./llmctl.sh gen serve --auto
curl localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3-Embedding-0.6B-8bit","input":["a","b"]}'
curl localhost:8000/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Kokoro-82M-bf16","input":"hi","response_format":"wav"}' -o hi.wav
curl localhost:8000/audio/transcriptions -F file=@hi.wav \
  -F model=mlx-community/parakeet-tdt-0.6b-v3
```

Models load once instead of per call. `/images/generations` does **not** work
with quantized repos — use `gen run image`.

Generative runtimes live in a separate conda env (`dekho-apple-gen`) so a
dependency conflict cannot break the serving env. `./llmctl.sh gen setup`.

## Offline

```bash
./llmctl.sh verify              # before losing network
./llmctl.sh start --offline     # sets HF_HUB_OFFLINE=1
```

Weights resolve through the HuggingFace hub, so an uncached repo id fails
offline regardless of disk. Validated cold: wiped state, `verify` passed on all
cached models, the 80B served at 51.3 GB of a 105.4 GB budget, opencode
registered automatically, and an agentic edit produced correct code in ~30 s.

A HuggingFace token is read from `$HF_TOKEN`, `$HF_TOKEN_FILE`, then
`~/.dekho/hugging-face-token.txt`, and never printed.

## Reference

| | |
|---|---|
| [docs/hardware.md](docs/hardware.md) | Memory ceiling, KV cache, context, measured throughput |
| [docs/models.md](docs/models.md) | Every model: size, runtime, status, known failures |
| [VALIDATION.md](VALIDATION.md) | Generated per-model test results |
| [AGENTS.md](AGENTS.md) | Rules for agents working in this repo |

## NVIDIA

```bash
cd rtx-pro-6000/nemotron-120b && cp .env.example .env && docker compose up -d
```

Helm/k8s moved to [gagandaroach/dlan](https://github.com/gagandaroach/dlan).

## Development

```bash
pip install -e ".[test]"
pytest && ruff check llmctl tests && ruff format --check llmctl tests
```

81 tests, no GPU required — they avoid importing `mlx` so CI runs on Linux.
They cover KV arithmetic anchored to measured values, port allocation, registry
pruning, the opencode manifest contract, `vmmap` parsing, and catalog
invariants. GPU-dependent checks live in `llmctl validate` and `smoke-test.py`.

CI: pytest on 3.11–3.13, ruff, shellcheck, an ES-module parse of the plugin,
and a credential/PII scan.
