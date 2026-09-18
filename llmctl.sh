#!/usr/bin/env bash
# llmctl launcher: resolves the env interpreter so this works from any shell,
# with or without the conda env activated.
set -euo pipefail
ENV_NAME="${DEKHO_ENV_NAME:-dekho-apple-local-llm}"

PY=""
for c in "$(conda info --base 2>/dev/null)/envs/$ENV_NAME/bin/python" \
         "$HOME/miniforge3/envs/$ENV_NAME/bin/python" \
         "$HOME/miniconda3/envs/$ENV_NAME/bin/python"; do
  [[ -x "$c" ]] && { PY="$c"; break; }
done
if [[ -z "$PY" ]]; then
  echo "conda env '$ENV_NAME' not found." >&2
  echo "create it with: conda env create -f apple-m-series/common/environment.yml" >&2
  exit 1
fi
exec "$PY" -m llmctl "$@"
