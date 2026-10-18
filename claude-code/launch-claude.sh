#!/usr/bin/env bash
# Run Claude Code against a self-hosted model.
#
#   launch-claude.sh                          interactive session
#   launch-claude.sh -p "<task>" [flags...]   headless worker; every argument
#                                             goes to `claude`
#
# Claude Code speaks the Anthropic Messages API (POST <base>/v1/messages), so
# the endpoint must serve it: a gateway that translates (Bifrost's /anthropic
# route, LiteLLM) or a server that serves it natively. An OpenAI-only
# /v1/chat/completions server is not enough.
#
# Site settings live outside the repo, in ~/.dekho/claude-code.env (override
# with DEKHO_CLAUDE_ENV), one KEY=value per line:
#
#   CLAUDE_LOCAL_BASE_URL=http://gateway.example.lan/anthropic   (required)
#   CLAUDE_LOCAL_MODEL=provider/model-name                       (required)
#   CLAUDE_LOCAL_KEY_FILE=~/.config/gateway/key    key file, if the gateway wants one
#   CLAUDE_LOCAL_KEY_HEADER=x-bf-vk                send the key in this header;
#                                                  unset sends it as a Bearer token
#   CLAUDE_LOCAL_1M=1                              plan for a 1M window (default 1)
#
# The same names in the environment override the file. The key is read from
# its file and passed to claude in the environment, never in argv: argv is
# world-readable through /proc and `ps`, a process's environment is not.
#
# Claude Code does not know a self-hosted model, so it would assume a 200k
# window. CLAUDE_LOCAL_1M appends `[1m]` to the model name, which makes it plan
# for 1M; Claude Code strips the suffix before sending. Size this to what the
# server actually holds. Claude Code still prints an "unrecognized_model" line
# and a cost figure that means nothing here.
#
# Like `claude --dangerously-skip-permissions`, the session runs every tool
# without asking. Point it only at a directory you would let it change.
set -euo pipefail

ENV_FILE=${DEKHO_CLAUDE_ENV:-$HOME/.dekho/claude-code.env}

# Read KEY=value lines without executing the file; the environment wins.
if [[ -r "$ENV_FILE" ]]; then
  while IFS='=' read -r k v; do
    [[ $k =~ ^CLAUDE_LOCAL_[A-Z0-9_]+$ ]] || continue
    [[ -z "${!k:-}" ]] && printf -v "$k" '%s' "$v"
  done < "$ENV_FILE"
fi

BASE_URL=${CLAUDE_LOCAL_BASE_URL:-}
MODEL=${CLAUDE_LOCAL_MODEL:-}
KEY_FILE=${CLAUDE_LOCAL_KEY_FILE:-}
KEY_FILE=${KEY_FILE/#\~/$HOME}
KEY_HEADER=${CLAUDE_LOCAL_KEY_HEADER:-}
ONE_M=${CLAUDE_LOCAL_1M:-1}

if [[ -z "$BASE_URL" || -z "$MODEL" ]]; then
  echo "!! set CLAUDE_LOCAL_BASE_URL and CLAUDE_LOCAL_MODEL in $ENV_FILE" >&2
  echo "   (see the header of $0)" >&2
  exit 1
fi
command -v claude >/dev/null || { echo "!! claude not on PATH" >&2; exit 1; }

KEY=""
if [[ -n "$KEY_FILE" ]]; then
  [[ -r "$KEY_FILE" ]] || { echo "!! key file not readable: $KEY_FILE" >&2; exit 1; }
  KEY=$(tr -d '[:space:]' < "$KEY_FILE")
  [[ -n "$KEY" ]] || { echo "!! key file is empty: $KEY_FILE" >&2; exit 1; }
fi

if [[ "$ONE_M" == 1 ]]; then
  case $MODEL in *'[1m]') ;; *) MODEL="$MODEL[1m]" ;; esac
fi

# An Anthropic key or a claude.ai login would otherwise win over the token.
unset ANTHROPIC_API_KEY ANTHROPIC_CUSTOM_HEADERS
export ANTHROPIC_BASE_URL=$BASE_URL
export ANTHROPIC_MODEL=$MODEL
export ANTHROPIC_SMALL_FAST_MODEL=$MODEL
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
if [[ -n "$KEY" && -n "$KEY_HEADER" ]]; then
  export ANTHROPIC_AUTH_TOKEN=unused
  export ANTHROPIC_CUSTOM_HEADERS="$KEY_HEADER: $KEY"
else
  export ANTHROPIC_AUTH_TOKEN=${KEY:-unused}
fi
unset KEY

exec claude --dangerously-skip-permissions "$@"
