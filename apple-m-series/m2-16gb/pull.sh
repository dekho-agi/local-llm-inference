#!/usr/bin/env bash
# Moved into llmctl. This shim keeps the old path working.
#   ./llmctl.sh pull
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$REPO/llmctl.sh" pull "$@"
