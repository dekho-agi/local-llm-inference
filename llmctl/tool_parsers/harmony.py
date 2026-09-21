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
# This parser is correct and unit-tested, and the tokenizer picks it up:
# has_tool_calling becomes True and tool_parser(...) returns the right
# {name, arguments}. It still does not reach the client, for a reason in
# mlx-lm's server rather than here.
#
# server.py collects generated text by state:
#
#     if   gen.state == "reasoning": reasoning_text += gen.text
#     elif gen.state == "tool":      tool_text      += gen.text
#     elif gen.state == "normal":
#         if prev_state == "tool": tool_calls.append(tool_text)   # flush
#
# A tool region is therefore only flushed by transitioning back to "normal".
# The stop transitions target state None, which matches no branch — so the
# accumulated tool_text is dropped and the end-of-loop flush
# (`if prev_state == "tool"`) also fails because prev_state is None.
#
# gpt-oss ends a tool call with EOS (<|return|>) and does not emit <|call|>
# under this template, so the region is always closed by a stop. Setting
# tool_call_end to <|return|> does not help: server.py builds
# transitions["tool"] = [(te, "normal")] and then appends the stop edges, and
# both carry the same token sequence, so the Aho-Corasick trie keeps one of
# them — the stop. SequenceStateMachine's own docstring shows tool_end mapping
# to None, which is consistent with the stop winning.
#
# So: the MODEL supports tool calling and emits well-formed calls; mlx-lm
# cannot deliver them when the call is terminated by EOS. Fixing it needs
# either an upstream change (flush pending tool_text when the state goes to
# None) or a translating proxy in front of the server.
#
# Every other coding model in the catalog has a working parser, so this is
# gpt-oss-specific.
