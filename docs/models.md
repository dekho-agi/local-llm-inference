# Model reference

Sizes verified against the HuggingFace API. Status from `llmctl validate` —
see [../VALIDATION.md](../VALIDATION.md) for the current run.

`llmctl models --all` prints this live, including what is cached.

## Text / coding — `llmctl start`

| Model | Size | Params | Ctx | Tool calls | Notes |
|---|---|---|---|---|---|
| `Qwen3-Coder-Next-4bit` | 44.9 GB | 80B-A3B | 256k | yes | Default. 92.7% tool-call accuracy across 5 IDE/CLI scaffolds; SWE-bench 70.6–71.3. 24.6 KB/token KV |
| `Qwen3-Coder-30B-A3B-Instruct-4bit` | 17.2 GB | 30B-A3B | 256k | yes | 95–114 tok/s. SWE-bench ~51. 98 KB/token KV |
| `GLM-4.7-Flash-4bit` | 16.9 GB | ~30B MoE | 128k | yes | Claims SWE-bench 59.2, τ²-bench 79.5. Multi-turn scores assume Preserved Thinking, which mlx-lm lacks |
| `Devstral-Small-2-24B-Instruct-2512-4bit` | 15.1 GB | 24B dense | 128k | yes | Claims 68.0% SWE-bench at 24B dense. No MoE routing variance |
| `GLM-4.5-Air-4bit` | 60.2 GB | 106B-A12B | 128k | yes | Different family for a second opinion |
| `Qwen2.5-Coder-7B-Instruct-4bit` | 4.3 GB | 7B dense | 32k | yes | Small machines |
| `Qwen2.5-3B-Instruct-4bit` | 1.8 GB | 3B dense | 32k | yes | 16 GB machines |
| `gpt-oss-120b-MXFP4-Q8` | 63.4 GB | 117B-A5.1B | 128k | after patch | See below. SWE-bench 62.4 — and uniquely, its published evals ran at this quantization |

Cross-vendor benchmark numbers are not comparable: Qwen3-Coder-Next's own
report has it at 70.6/71.1/71.3 on one benchmark by changing only the agent
scaffold. Differences under ~3 points are noise. All of these run 4-bit while
the published numbers describe bf16.

### gpt-oss

Tool calling works after `llmctl patch-mlx-lm`, verified
(`finish_reason=tool_calls`). Not yet usable via opencode: harmony channel
markers leak into `content`. Both causes are documented in
`llmctl/tool_parsers/harmony.py`; upstream ml-explore/mlx-lm#1867.

## Generative — `llmctl gen`

| Class | Model | Size | Runtime | Measured |
|---|---|---|---|---|
| image | `FLUX.2-Klein-4B-6bit` | 6.6 GB | mflux | 1024², 4 steps, 6–11 s, 13.3 GB peak |
| image | `krea-2-turbo-mflux-bf16` | 34.2 GB | mflux | 1024², 8 steps, 44 s, 41.7 GB peak |
| image | `ideogram-4-mflux-q8` | 26.0 GB | mflux | Licence-gated on HF; accept it first |
| image-edit | `qwen-image-edit-2511-mflux-bf16` | 56.6 GB | mflux | 86 s–7 min, 58.5 GB peak |
| vlm | `GLM-OCR-8bit` | 1.6 GB | mlx-vlm | 3 s. Documents, screenshots |
| vlm | `Qwen3-VL-30B-A3B-Instruct-8bit` | 33.5 GB | mlx-vlm | 6 s |
| vlm | `Qwen3.5-122B-A10B-5bit` | 84.9 GB | mlx-vlm | 16 s. Hybrid attention → 24 KB/token, vs 188 for Qwen3-VL-235B |
| omni | `MiniCPM-o-4_5-4bit` | 6.2 GB | mlx-vlm | 12 s, text+vision |
| omni | `Qwen3-Omni-30B-A3B-Instruct-bf16` | 70.5 GB | mlx-vlm | Text only — see below |
| stt | `parakeet-tdt-0.6b-v3` | 2.5 GB | mlx-audio | Verbatim transcription in ~3 s |
| tts | `Kokoro-82M-bf16` | 0.4 GB | mlx-audio | 3.2 s of 24 kHz in ~3 s |
| tts | `IndexTTS-2-MLX` | 4.3 GB | mlx-audio | Voice cloning |
| music | `MiniMax-Music3-mxfp8` | 13.9 GB | mlx-audio | 21.2 s of 44.1 kHz stereo in 58 s |
| video | `Wan2.2-TI2V-5B-mlx-q8` | 19.6 GB | mlx-video | 3.4 s at 1280×704 in 19 min |
| video | `ltx-2.5-mlx-q8` | 23.9 GB | mlx-video | |
| embed | `Qwen3-Embedding-0.6B-8bit` | 0.7 GB | mlx-vlm | 1024-dim |

### Where bigger is worse

- `parakeet-tdt-1.1b` is **+5.09 WER worse** than the 0.6B on long-form
  (15.81 vs 10.72). VibeVoice-ASR (16.7 GB) is +0.72 worse.
- Kokoro (0.4 GB) ties Chatterbox (2.7 GB) on TTS Arena V2. IndexTTS-2
  transcribes at 1.521% WER against 1.897% for human recordings —
  intelligibility is saturated, so size buys cloning, not quality.
- Wan 2.2 A14B's own benchmark reports no quality win for bf16 over q8.
- HunyuanImage-3.0 is #1 at *editing* (Elo 1025.53) but 11th at
  text-to-image (964.80), below a 26 GB Ideogram 4 (1011.97, 34 s).

### Too big for 128 GB

`MiniMax-M2.1-4bit` 128.7 GB · `GLM-4.7-4bit` 198.6 GB · `GLM-5.2-mxfp4`
395.1 GB · `Qwen3-Coder-480B-4bit` 270.1 GB · `Kimi-K2.5` 657.6 GB.
`Qwen3.8-Flash-Next-4bit` (111.5 GB) fits the ceiling but leaves nothing for
KV, and its `Qwen4Exp` arch is not in mlx-lm.

Rule: 4-bit MLX weights ≈ 0.56 GB per billion total params.

## Known failures

| Thing | Cause |
|---|---|
| gpt-oss via opencode | Harmony channel markers leak into `content`. mlx-lm's `_infer_thinking` knows `<think>` and a `<\|channel>` spelling that differs from gpt-oss's `<\|channel\|>`. Upstream #1867 |
| Qwen3-Omni speech out | Weights ship the talker (417 tensors) and the module exposes `.talker`, but mlx-vlm 0.7.1 defines no `generate_audio`, which its capability gate requires. Use Kokoro |
| `index-tts2-mlx` | Ships `config.yaml`; mlx-audio requires `config.json`. Use `IndexTTS-2-MLX` |
| `/images/generations` on the gen server | Only accepts canonical `black-forest-labs/*` ids. Use `llmctl gen run image` |
| `mlx-gen` | Pins `mlx<0.32.0`, downgrades mlx-metal to 0.31.2. Keep out of the shared env |
| PyPI `mlx-video` | Unrelated video-I/O package. Install from `git+https://github.com/Blaizzy/mlx-video.git` |
| `/audio/speech` mp3 | Needs ffmpeg. Pass `response_format: "wav"` |

## Runtime quirks

- mflux and mlx-video ship a console script **per model family**
  (`mflux-generate-krea2`, `mlx_video.wan_2.generate`). The catalog records
  which via an `entry` field.
- Flag names differ per runtime: TTS `--text`/`--output_path` (underscores),
  STT `--audio`/`--output-path` (hyphens), mlx-vlm `--image`, mflux
  `--image-paths`. mlx-video's `--model-dir` wants a snapshot *directory*.
- Kokoro needs `misaki`, `num2words`, `phonemizer`, `espeakng_loader`.
  mlx-audio declares none of them and its error names only `misaki` whichever
  is missing. `misaki[en]` pins a spacy that fails to build.
- `hf download` can exit 0 on an incomplete download, and a failed pull leaves
  the directory and `config.json` behind. Run `llmctl verify`.
