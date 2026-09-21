"""Fix: mlx-lm drops tool text when a tool call ends on an EOS token.

Applied by `llmctl patch-mlx-lm`. Idempotent, reversible, and narrow.

## The bug

server.py accumulates generated text per state and flushes a tool region only
when the state transitions back to "normal":

    elif gen.state == "normal":
        if prev_state == "tool":
            tool_calls.append(tool_text)          # the only flush in the loop
    ...
    if prev_state == "tool" and tool_text:        # end-of-loop flush
        tool_calls.append(tool_text)

_make_state_machine builds, for the tool state:

    transitions["tool"] = [(tool_call_end_tokens, "normal")]
    transitions["tool"].extend(common_stops)      # one edge per EOS token

gpt-oss ends a tool call with `<|call|>` (200012), which its
generation_config.json also declares as an EOS token — and mlx-lm honours that,
so `<|call|>` appears on BOTH edges. One Aho-Corasick trie per state means one
match per token, and the stop edge wins: the state becomes None.

None matches neither the "tool" nor the "normal" branch, so the accumulated
tool_text is silently discarded, and `prev_state` is None by the end so the
end-of-loop flush does not fire either. The model's tool call is correct and
already parsed-able; it simply never reaches the response.

This affects any model whose tool-call terminator is also an EOS token.

## The fix

Flush pending tool_text regardless of which state ended the region. Two edits:

  - in the loop, treat a stop (state None) with pending tool_text as a flush
  - at the end, drop the `prev_state == "tool"` precondition

Upstream: ml-explore/mlx-lm#1867 addresses harmony channel parsing; #613
reported the tool-call symptom and was closed unfixed as "non-essential".
"""

MARKER = "# llmctl: flush tool_text on stop-state exit"

IN_LOOP_OLD = """                elif gen.state == "normal":
                    if prev_state == "tool":
                        tool_calls.append(tool_text)
                        tool_text = ""
                        made_tool_call = True
                    text += gen.text"""

IN_LOOP_NEW = """                elif gen.state == "normal":
                    if prev_state == "tool":
                        tool_calls.append(tool_text)
                        tool_text = ""
                        made_tool_call = True
                    text += gen.text
                elif gen.state is None and tool_text:
                    # llmctl: flush tool_text on stop-state exit
                    # A tool terminator that is also an EOS token (gpt-oss's
                    # <|call|>) matches the stop edge, so the region closes
                    # into state None and the text would otherwise be dropped.
                    tool_calls.append(tool_text)
                    tool_text = ""
                    made_tool_call = True"""

END_OLD = """            if prev_state == "tool" and tool_text:
                tool_calls.append(tool_text)
                made_tool_call = True"""

END_NEW = """            if tool_text:
                # llmctl: flush tool_text on stop-state exit
                # prev_state is None when the region closed on a stop token,
                # so keying the flush on prev_state == "tool" missed it.
                tool_calls.append(tool_text)
                made_tool_call = True"""

EDITS = [(IN_LOOP_OLD, IN_LOOP_NEW), (END_OLD, END_NEW)]
