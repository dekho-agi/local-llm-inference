#!/usr/bin/env bash
# Shared MLX serve bones for all Apple-silicon targets.
#
# Callers (m2-16gb/serve.sh, m5-128gb/serve.sh) set defaults and exec this.
# Contract: TARGET_DIR, MODEL, PORT, and optionally MAX_TOKENS, MAX_CONTEXT,
# PROMPT_CACHE_BYTES, PROMPT_CACHE_SIZE, KV_BITS, EXTRA_ARGS.
#
# mlx_lm.server hot-swaps models: it loads whatever repo id arrives in the
# request's "model" field, so one server backs every model in opencode's
# picker. --model only sets the default that gets preloaded at boot.
set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-dekho-apple-local-llm}"

# shellcheck source=./hf-token.sh
source "$COMMON_DIR/hf-token.sh"

# Target-local overrides, loaded before defaults are resolved.
if [[ -n "${TARGET_DIR:-}" && -f "$TARGET_DIR/.env" ]]; then
  set -a; source "$TARGET_DIR/.env"; set +a
fi

MODEL="${MODEL:?MODEL must be set by the calling target}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

if ! conda env list | grep -q "^${ENV_NAME} "; then
  echo "First run: creating conda environment '${ENV_NAME}'..."
  conda env create -f "$COMMON_DIR/environment.yml"
fi

# Resolve the env's interpreter and exec it directly rather than going through
# `conda run`, which forks a wrapper: that left two PIDs, so a SIGTERM to the
# parent could orphan the server holding :8000. One process, clean signals.
ENV_PY="$(conda info --base)/envs/${ENV_NAME}/bin/python"
[[ -x "$ENV_PY" ]] || { echo "interpreter not found: $ENV_PY" >&2; exit 1; }

# Note: mlx_lm.server has NO context-length flag. The model's own
# max_position_embeddings is the hard ceiling and MAX_CONTEXT only sets what we
# advertise to opencode (which is what drives its compaction). See docs/hardware.md.
args=(--model "$MODEL" --port "$PORT" --host "$HOST")

# mlx_lm.server defaults --max-tokens to 512, which truncates agent turns.
args+=(--max-tokens "${MAX_TOKENS:-32768}")

# Bound the reusable prompt-cache pool. Prefix reuse across agent turns is
# the single biggest latency win in opencode; this caps what it may hold.
[[ -n "${PROMPT_CACHE_BYTES:-}" ]] && args+=(--prompt-cache-bytes "$PROMPT_CACHE_BYTES")
[[ -n "${PROMPT_CACHE_SIZE:-}"  ]] && args+=(--prompt-cache-size "$PROMPT_CACHE_SIZE")

# KV quantization. Note: --kv-bits disables batching (requests serialize),
# so it is off by default and only worth it for very long contexts.
[[ -n "${KV_BITS:-}" ]] && args+=(--kv-bits "$KV_BITS")

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # EXTRA_ARGS is a flag string, so word-splitting is the point — do it
  # explicitly rather than relying on an unquoted expansion.
  read -r -a _extra <<< "$EXTRA_ARGS"
  args+=("${_extra[@]}")
fi

# Note: this foreground path does NOT write the opencode manifest, so a server
# started this way is not auto-discovered. Use `./llmctl.sh start` for that.

echo "model    : $MODEL   (server hot-swaps per request)"
echo "endpoint : http://$HOST:$PORT/v1"
echo "args     : ${args[*]}"
echo "Press Ctrl+C to stop."
echo ""

exec "$ENV_PY" -m mlx_lm.server "${args[@]}"
