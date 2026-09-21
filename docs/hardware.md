# Hardware limits

Measured on MacBook Pro, Apple M5 Max, 128 GB, macOS 26.5. Re-derive with
`llmctl host`.

## Memory ceiling

| | |
|---|---|
| Installed | 128 GB |
| `max_recommended_working_set_size` | 115.4 GB |
| Plan against | **~100 GB** |

mlx-lm raises the wired limit to the working set itself. No `sudo sysctl
iogpu.wired_limit_mb`.

Crossing the ceiling swaps, and the penalty is not gradual. Same prompt/seed:

| Precision | Peak | Time |
|---|---|---|
| bf16 | 118.55 GiB | 730.96 s |
| INT8 | 67.63 GiB | 70.54 s |

10.4x, from paging. Below the ceiling, higher precision is free or better.

## KV cache

```
bytes/token = 2 (K,V) x kv_heads x head_dim x 2 (bf16) x layers_holding_KV
```

The last term is architectural, and dominates:

| Model | KV layers | B/token | KV @ 256k |
|---|---|---|---|
| Qwen3-Coder-Next (80B) | 12 of 48 | 24,576 | 6.4 GB |
| Qwen3-Coder-30B | 48 of 48 | 98,304 | 25.8 GB |
| gpt-oss-120b | 36 of 36 | 73,728 | 9.7 GB @ 131k |

The 80B costs 4x less per context token than the 30B: it is hybrid, and its
other 36 layers hold a fixed 77 MB recurrent state. At full context the 30B's
cache exceeds its own weights. **Long sessions favour the larger model.**

`llmctl models --ctx N` computes fit. Allocation grows in steps, so transient
peak runs ~2x steady state.

## Context is not a server setting

`mlx_lm.server` has no context flag. Three different things get called context:

| | Set by | Wrong value costs |
|---|---|---|
| Trained window (`max_position_embeddings`) | the model | silent quality loss past it |
| Advertised window (`limit.context`) | `MAX_CONTEXT` → manifest → opencode | too high: overfill; too low: early compaction |
| Affordable window | architecture + RAM | swapping |

Qwen3-Coder-Next is 262,144 with `rope_scaling: null` — no factor to stretch
it. Higher would need a YaRN-scaled conversion, i.e. a different model.

## Measured throughput

| | |
|---|---|
| 30B decode, fresh server | 95–114 tok/s |
| 30B decode, after model swaps | 36–38 tok/s |
| Qwen3-Coder-Next decode | ~24 tok/s |
| Prefill | 2,300–3,700 tok/s (100k ctx ≈ 30–45 s) |

**Hot-swapping costs ~3x until restart.** `vmmap` showed 27.2 GB resident
against a 62.1 GB peak — retained buffers plus a prompt cache that
`LRUPromptCache` never clears on model change. Run one model per server, or
one server per model.

## Reading memory

`ps`/`htop` cannot see it: weights live in Metal buffers macOS excludes from
RSS (16.5 GB RSS vs 27.2 GB actual). `llmctl monitor` reads `ioreg` for GPU
utilization and `vmmap` for buffer residency.
