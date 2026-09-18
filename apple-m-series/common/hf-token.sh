#!/usr/bin/env bash
# Resolve a HuggingFace token without printing it.
#
# Precedence: existing $HF_TOKEN  >  $HF_TOKEN_FILE  >  ~/.dekho/hugging-face-token.txt
# Unauthenticated hub requests are rate-limited and slower, so this is worth
# having even though every model in these targets is public.
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.dekho/hugging-face-token.txt}"

if [[ -z "${HF_TOKEN:-}" && -r "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  echo "hf auth  : token loaded (${#HF_TOKEN} chars) from ${HF_TOKEN_FILE/#$HOME/~}"
else
  echo "hf auth  : none — unauthenticated hub requests are rate-limited."
  echo "           put a read token in ~/.dekho/hugging-face-token.txt or set HF_TOKEN."
fi
