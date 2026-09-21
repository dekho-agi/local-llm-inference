# Tool-call parser for OpenAI's harmony format (gpt-oss).
#
# mlx-lm ships parsers for qwen3_coder, glm47, mistral, kimi_k2, minimax_m2,
# gemma4, pythonic, json_tools, longcat and laguna — but none for harmony, so
# gpt-oss is reported as `has_tool_calling == False` and its perfectly
# well-formed tool calls are returned as raw text in `content`.
#
# The model is not at fault. Given a tools array it emits:
#
#   <|channel|>analysis<|message|>reasoning here<|end|>
#   <|start|>assistant<|channel|>commentary to=functions.read_file
#   <|constrain|>json<|message|>{"path": "src/main.py"}<|call|>
#
# That is a complete call: a recipient (`to=functions.NAME`), a declared
# argument encoding (`<|constrain|>json`) and a JSON payload. This translates
# it into the {name, arguments} shape mlx-lm's server expects.
#
# Installed by `llmctl patch-mlx-lm`, which copies this into
# mlx_lm/tool_parsers/ and sets "tool_parser_type": "harmony" in the model's
# tokenizer_config.json — mlx-lm reads that key from the model itself, so
# selection needs no changes to mlx-lm's own source.

import json
import re
from typing import Any

# `to=` may carry a namespace (functions.foo) or a bare name, and the trailing
# space before <|constrain|> is optional in practice.
_RECIPIENT = re.compile(
    r"to=(?:functions\.)?(?P<name>[A-Za-z0-9_.\-]+)\s*"
    r"(?:<\|constrain\|>\s*(?P<fmt>[A-Za-z0-9_\-]+)\s*)?"
    r"<\|message\|>(?P<args>.*?)"
    r"(?=<\|call\|>|<\|end\|>|<\|start\|>|$)",
    re.DOTALL,
)

# Where a call begins in the token stream, and what closes it.
#
# These MUST be special tokens. The server tokenizes them and looks for that
# id sequence in the output, and ordinary text does not tokenize the same way
# standalone as it does in context: "to=functions." alone is
# [935, 28, 44580, 13] but in a real generation it is [316, 28, 44580, 7211, …]
# — a different leading token and a different merge — so the subsequence never
# matches and the tool state is never entered. `<|start|>` is a single id
# (200006) and, unlike `<|channel|>`, it sits *before* the `to=functions.NAME`
# recipient, so the accumulated tool text still contains the function name.
tool_call_start = "<|start|>"
tool_call_end = "<|call|>"


def _coerce(raw: str, name: str, tools: Any | None) -> dict:
    """Arguments are JSON in every harmony variant seen; fall back gracefully."""
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Trailing prose or a truncated generation: recover the outermost
        # JSON object rather than discarding the whole call.
        start, depth = raw.find("{"), 0
        if start < 0:
            raise ValueError(f"harmony: no JSON arguments for {name!r}") from None
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(raw[start : i + 1])
                        break
                    except json.JSONDecodeError:
                        continue
        else:
            raise ValueError(f"harmony: unparseable arguments for {name!r}") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"harmony: arguments for {name!r} are not an object")
    return parsed


def parse_tool_call(model_output: str, tools: Any | None = None) -> dict:
    """Return {"name", "arguments"} for the first call in `model_output`.

    Raises ValueError when no call is present, matching the other parsers —
    the server treats that as "the model chose to answer in text".
    """
    match = _RECIPIENT.search(model_output)
    if not match:
        raise ValueError("No function provided.")
    name = match.group("name")
    return {"name": name, "arguments": _coerce(match.group("args"), name, tools)}


def parse_all_tool_calls(model_output: str, tools: Any | None = None) -> list[dict]:
    """Every call in the output, for callers that support parallel calls."""
    out = []
    for m in _RECIPIENT.finditer(model_output):
        name = m.group("name")
        try:
            out.append({"name": name, "arguments": _coerce(m.group("args"), name, tools)})
        except ValueError:
            continue
    return out


# STATUS on mlx-lm 0.31.3
# ----------------------
# TOOL CALLING WORKS with `llmctl patch-mlx-lm`. Verified:
#
#   finish_reason: tool_calls
#   tool_calls[0]: read_file({"path": "src/main.py"})
#
# Two things were needed. This parser, and a two-line fix to server.py
# (see server_flush.patch.py): gpt-oss ends a tool call with <|call|>, which
# its generation_config.json also declares as an EOS token — and mlx-lm honours
# that — so <|call|> matches both the tool-exit edge and a stop edge. One trie
# per state means one match, the stop wins, the state becomes None, and the
# accumulated tool text was dropped because the flush only fired on the
# transition to "normal".
#
# Note also that marker matching is at the TOKEN level: "to=functions." alone
# is [935, 28, 44580, 13] but in a real generation [316, 28, 44580, 7211, …],
# so only special tokens match reliably. Hence <|start|> (200006).
#
# STILL BROKEN: opencode rejects the responses, because the harmony *channel*
# markers leak into message.content:
#
#   "You have passed a message containing <|channel|> tags in the content
#    field. Instead ... pass analysis messages in the 'thinking' field, and
#    final messages in the 'content' field."
#
# That is a separate root cause. mlx-lm's _infer_thinking() only recognises
# <think>/</think>, <longcat_think>, and a `<|channel>`/`<channel|>` pair whose
# spelling differs from gpt-oss's `<|channel|>` — so analysis is never routed
# to reasoning_text. There is no config override; it is called unconditionally.
# Fixing it properly needs a channel-aware state machine, which is what
# upstream ml-explore/mlx-lm#1867 implements.
#
# So today: gpt-oss tool-calls correctly over the raw API, and is not yet
# usable through opencode. Every other coding model in the catalog is.
