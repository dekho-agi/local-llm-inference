#!/usr/bin/env bash
# MacBook Pro M5 Max / 128 GB unified memory.
set -euo pipefail
TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TARGET_DIR
export MODEL="${MODEL:-mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit}"
export PORT="${PORT:-8000}"
export MAX_TOKENS="${MAX_TOKENS:-32768}"
# Context advertised to opencode. 262144 is Qwen3-Coder-Next's native ceiling;
# it is clamped per model to that model's own max_position_embeddings.
export MAX_CONTEXT="${MAX_CONTEXT:-262144}"
# Room for several cached agent conversations without crowding the weights.
export PROMPT_CACHE_BYTES="${PROMPT_CACHE_BYTES:-24G}"
exec "$TARGET_DIR/../common/serve.sh"
