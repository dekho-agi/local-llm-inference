# Flight runbook

Everything here was run on this machine with `HF_HUB_OFFLINE=1`. No step is
aspirational.

---

## Before you lose network

```bash
cd ~/repos/local-llm-inference
./llmctl.sh verify                 # every shard complete, tool parser per model
./llmctl.sh gen doctor             # generative runtimes importable
```

`verify` is the one that matters. A repo id that was never cached fails offline
no matter how much disk you have, and there is no fixing it at altitude.

---

## Coding (the main event)

```bash
./llmctl.sh start                  # numbered picker
./llmctl.sh opencode               # confirm the provider is registered
opencode                           # then /models → Dekho Local Inference
```

Pick **one** model and stay on it. Hot-swapping inside a server costs ~3x
throughput until restart; `./llmctl.sh stop && ./llmctl.sh start <model>` is
the cheaper move.

| Situation | Model | Why |
|---|---|---|
| Plugged in, real work | `Qwen3-Coder-Next-4bit` | 80B/3B active, 256k context, best tool-call robustness. 51 GB at full context. |
| On battery, quick edits | `Qwen3-Coder-30B-A3B-Instruct-4bit` | 95–114 tok/s on a fresh server, loads in seconds |

Long context is *cheaper* on the 80B, not dearer: only 12 of its 48 layers hold
a KV cache, so it costs 24.6 KB/token against the 30B's 98 KB.

**`gpt-oss-120b` cannot tool-call** — mlx-lm has no parser for its harmony
format and it fails silently. It is marked chat-only and will not work as an
opencode agent.

---

## Generative work

```bash
./llmctl.sh gen list               # what's available
./llmctl.sh gen serve --auto       # one endpoint for tts/stt/embed/vlm
```

One-shot runs, all verified on this machine:

```bash
./llmctl.sh gen run image "a red bicycle against a white wall" -o bike.png
./llmctl.sh gen run image-edit "make the bicycle yellow" -i bike.png -o yellow.png
./llmctl.sh gen run vlm "transcribe this" -i scan.png
./llmctl.sh gen run tts "hello there" -o ./speech
./llmctl.sh gen run stt -i speech/audio_000.wav -o transcript
./llmctl.sh gen run music "warm acoustic guitar, instrumental" -o track.wav
./llmctl.sh gen run video "a red balloon rising" -o clip.mp4     # ~19 min
```

**Speech out:** use `gen run tts` (Kokoro). Qwen3-Omni ships the audio weights
but mlx-vlm 0.7.1 has no `generate_audio` for it, so it only emits text.

`--dry-run` shows the exact command first. Add `--steps`/`--seed` where the
runtime supports them.

### Measured on this M5 Max

| Class | Model | Time | Peak memory |
|---|---|---|---|
| image | FLUX.2-Klein-4B | **11 s** (4 steps) | 13.3 GB |
| image | Krea 2 Turbo bf16 | 46 s (8 steps) | 41.7 GB |
| image-edit | Qwen-Image-Edit-2511 bf16 | 7m02s (20 steps) | 58.5 GB |
| tts | Kokoro | seconds | small |
| stt | parakeet | seconds | small |
| music | MiniMax-Music3 | 47 s → 21 s of 44.1 kHz stereo | — |
| video | Wan2.2-TI2V-5B | **18m55s** → 3.4 s at 1280×704 | — |

Video defaults to 1280×704; pass `--width`/`--height` to cut the time.

---

## Watching it

```bash
./llmctl.sh ps                     # both server kinds, health, uptime
./llmctl.sh monitor -w             # GPU utilization and real memory
```

`htop` is useless here: MLX weights live in Metal buffers macOS does not count
in RSS, so a server shows ~16 GB RSS while holding 42 GB. `monitor` reads
`ioreg` and `vmmap` instead.

**Memory rule: plan against ~100 GiB, not the 115.4 GiB ceiling.** 115.4 is
where allocation stops succeeding, not where performance stops — one measured
case ran 10.4x slower at 118 GiB peak than at 67 GiB, because it was paging. If
`monitor` shows pageouts climbing, drop `MAX_CONTEXT`, lower the prompt cache,
or use a smaller model.

---

## When something breaks

| Symptom | Fix |
|---|---|
| Model missing from `/models` | `./llmctl.sh stop && ./llmctl.sh start` — the manifest is rewritten on start |
| Raw `<\|channel\|>` or XML in replies | That model has no mlx-lm tool parser. Use one `llmctl models` marks `tools=yes` |
| Agent turn truncates mid-tool-call | `MAX_TOKENS` too low; mlx-lm's own default is 512 |
| Throughput dropped ~3x | The server has been hot-swapping. Restart it with the model you want |
| Machine swapping | Over the working set. Smaller model or smaller context |
| `gen run` fails on flags | `--dry-run` to see the command; each runtime names flags differently |
| TTS 400s on mp3 | `response_format: "wav"`, or `brew install ffmpeg` |
| Image gen over HTTP fails | Expected — `/images/generations` rejects quantized repos. Use `gen run image` |

Logs: `~/.cache/dekho-local-inference/logs/<port>.log`

---

## What is not going to work offline

- Anything not already in `~/.cache/huggingface`. `verify` tells you what you have.
- `ideogram-4-mflux-q8` unless you accepted its licence while online (it is gated).
- `/images/generations` on the unified server, for any quantized repo.
- `llmctl pull`, obviously.
