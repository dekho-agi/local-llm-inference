#!/usr/bin/env bash
# Deprecated: use `./llmctl.sh opencode --install` from the repo root.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "note: this script has moved into llmctl." >&2
exec "$REPO/llmctl.sh" opencode --install
