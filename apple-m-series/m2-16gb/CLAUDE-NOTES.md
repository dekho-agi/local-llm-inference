# MacBook Air M2 16 GB — Notes for Claude

Cached context so future sessions don't need to re-probe the machine.
Captured: 2026-04-20. Split out of the old flat `apple-m-series/` on 2026-09-17
when the M5 Max target was added; shared bones now live in `../common/`.

---

## Target machine

| | |
|---|---|
| Model | MacBook Air |
| Chip | Apple M2 |
| CPU | 8 cores (4 performance + 4 efficiency) |
| GPU | 10 cores, Metal 4 |
| Unified memory | **16 GB** |
| Disk free | ~500 GB (of 926 GB) |
| macOS | 26.1 (build 25B78) |
| Shell | zsh |
| Primary user dir | `$HOME` |

Useful re-probe commands (only run if hardware may have changed):
```bash
system_profiler SPHardwareDataType SPDisplaysDataType | grep -E "Chip|Memory|Cores|Metal"
sw_vers
df -h /
```

---

## Memory budget (16 GB unified)

- macOS + browser + apps reserve ~4–6 GB.
- **Practical ceiling for model + KV cache: ~10–11 GB.**
- Unified memory = GPU and CPU share the same pool; no separate VRAM.
- Heuristic for GGUF footprint:
  - Q4_K_M ≈ 0.55 GB per B params
  - Q5_K_M ≈ 0.70 GB per B params
  - Q8_0 ≈ 1.1 GB per B params
- Add ~1–2 GB for KV cache at 8k ctx, more at larger contexts.

**Rule of thumb:** stay at or below 8B @ Q4 for comfortable daily use. 12–14B Q4 works if you close everything else.

---

## Runtime options (all expose OpenAI-compatible `/v1`)

| Runtime | Install | Notes |
|---|---|---|
| Ollama | `brew install ollama` | Easiest. Endpoint `http://localhost:11434/v1`. Metal out of the box. Not used here — see below. |
| LM Studio | dmg from lmstudio.ai | GUI + OpenAI server. llama.cpp under the hood. |
| llama.cpp `llama-server` | `brew install llama.cpp` | Lower level, max control, best raw tok/s for GGUF. |
| **MLX-LM** (chosen) | `pip install mlx-lm` → `mlx_lm.server` | Apple's framework. Fastest on M-series for MLX-quantized models. Smaller model catalog than GGUF. |

### Important: Ollama in Docker on macOS cannot use Metal
Docker Desktop on Mac runs Linux VMs with no GPU passthrough. Run Ollama **natively** (brew / launchd) on this machine, not in Docker. Docker-Ollama works on the RTX box because Linux + nvidia-container-toolkit; it does NOT work here.

### Nothing installed as of 2026-04-20
`which ollama lm-studio llama-server mlx_lm.server` → all not found on *that*
machine. The stack now installs mlx-lm into the conda env
`dekho-apple-local-llm` on first `./serve.sh`.

---

## Recommended models for 16 GB M2

### Primary picks (comfortable headroom)

| Model | Size (Q4) | Ollama tag | Best for |
|---|---|---|---|
| Llama 3.2 3B | ~2.0 GB | `llama3.2:3b` | Fast chat, low-latency UI |
| Qwen 2.5 7B Instruct | ~4.7 GB | `qwen2.5:7b` | General workhorse |
| Qwen 2.5 Coder 7B | ~4.7 GB | `qwen2.5-coder:7b` | Best small coder |
| Llama 3.1 8B | ~4.9 GB | `llama3.1:8b` | Strong tool-calling, baseline |
| Gemma 3 4B | ~3.0 GB | `gemma3:4b` | Multimodal (vision) small |
| Phi-4-mini 3.8B | ~2.5 GB | `phi4-mini` | Reasoning at tiny size |
| nomic-embed-text | ~275 MB | `nomic-embed-text` | Embeddings for RAG |
| mxbai-embed-large | ~670 MB | `mxbai-embed-large` | Higher-quality embeddings |

### Stretch (tight — close other apps)

| Model | Size (Q4) | Ollama tag | Notes |
|---|---|---|---|
| Mistral Nemo 12B | ~7.1 GB | `mistral-nemo` | 128k context |
| Gemma 2 9B | ~5.4 GB | `gemma2:9b` | |
| Phi-4 14B | ~8.5 GB | `phi4` | Slow prompt processing on M2 |

### Avoid on this machine
Llama 3.3 70B, Qwen 2.5 32B/72B, DeepSeek-V3, Mixtral 8x22B — will swap to disk, tok/s drops to single digits.

---

## Repo context (relevant bits)

- Repo: `local-llm-inference` (was `dekho-agi/dev-ai-inference`)
- Structure: hardware-target-first. Each subdir is a self-contained Docker Compose stack exposing OpenAI-compatible API.
- `rtx-pro-6000/nemotron-120b/` — vLLM + Nemotron-3-Super-120B NVFP4 on RTX Pro 6000 Blackwell (96 GB). Port `:8000`.
- `apple-m-series/` — now split into `m2-16gb/` and `m5-128gb/` over shared `common/`.
- Git: main branch, clean. Recent commits show the repo was restructured to be GPU-first and the Ollama health-check was fixed to use `ollama list`.

### Resolved: native MLX, not Docker, not Ollama
Docker Compose cannot be used the way `rtx-*` does — Metal isn't reachable from
Docker on macOS. The decision taken (2026-09-17) was **native `mlx_lm.server`**
driven by `./serve.sh`, with shared bones in `../common/`:

- `../common/serve.sh`  — the runtime; targets only supply defaults
- `../common/pull.sh`   — pre-download `models.txt` for offline use
- `../common/hf-token.sh` — resolves the HF token from `~/.dekho/`

MLX over Ollama because MLX is the fastest path on Apple silicon and keeps one
runtime across both Apple targets. Port stays at the repo-standard `:8000/v1`.

---

## Endpoints / conventions (follow repo pattern)

- Port convention in this repo: `:8000/v1` for the OpenAI-compatible endpoint. `mlx_lm.server` is told `--port 8000` directly, so no proxy is needed (this was a concern only for Ollama, whose default is `:11434`).
- Health check pattern: `ollama list` (not curl) — see commit `b73346e`.

---

## Quick test curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## User profile notes (relevant to this machine)

- Prefers terse answers with tradeoffs over long explanations.
- Repo already follows a GPU-first, per-hardware-subdir layout; new Mac stack should feel consistent with that.
