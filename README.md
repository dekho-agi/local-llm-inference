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

New to this, or sharing with a colleague: [REPORT.md](REPORT.md).

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
| [REPORT.md](REPORT.md) | Team-facing write-up |
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
