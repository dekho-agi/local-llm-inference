#!/usr/bin/env bash
# MacBook Air M2 / 16 GB unified memory.
set -euo pipefail
TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TARGET_DIR
export MODEL="${MODEL:-mlx-community/Qwen2.5-3B-Instruct-4bit}"
export PORT="${PORT:-8000}"
export MAX_TOKENS="${MAX_TOKENS:-8192}"
export MAX_CONTEXT="${MAX_CONTEXT:-16384}"
export PROMPT_CACHE_BYTES="${PROMPT_CACHE_BYTES:-2G}"
exec "$TARGET_DIR/../common/serve.sh"
