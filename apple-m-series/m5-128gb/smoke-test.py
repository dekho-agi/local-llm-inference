"""
Smoke test for the local MLX server — the checks that decide whether opencode
will actually work agentically against it.

Unlike the vLLM/Nemotron path (see ../../weather_agent_demo.py, which parses
XML tool calls out of the text), mlx_lm.server runs the model's tool parser
itself and returns real OpenAI `tool_calls` objects. This verifies that, plus
the multi-turn tool-result round trip that every agent loop depends on.

Stdlib only — runs in the serve env with no extra dependencies.

  python smoke-test.py                                  # default model
  python smoke-test.py mlx-community/GLM-4.7-Flash-4bit  # a specific one
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "mlx-community/Qwen3-Coder-Next-4bit"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to repo root"}
                },
                "required": ["path"],
            },
        },
    }
]


# ── transport ─────────────────────────────────────────────────────────────────
def post(path: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(model: str, messages: list, **kw) -> dict:
    return post("/chat/completions", {"model": model, "messages": messages, **kw})


# ── checks ────────────────────────────────────────────────────────────────────
def check_models(model: str) -> bool:
    """The server lists what it can serve; opencode's picker needs the id to match."""
    with urllib.request.urlopen(BASE_URL + "/models", timeout=30) as r:
        ids = [m["id"] for m in json.loads(r.read()).get("data", [])]
    print(f"  server reports {len(ids)} cached model(s)")
    if model in ids:
        print(f"  ✓ {model} is present")
        return True
    print(f"  ! {model} not in /models — it will be fetched from the hub on first use")
    return True


def check_plain(model: str) -> bool:
    t = time.time()
    r = chat(model, [{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=16)
    txt = r["choices"][0]["message"]["content"].strip()
    print(f"  reply {txt!r} in {time.time() - t:.1f}s (load + generate)")
    return bool(txt)


def check_tool_call(model: str) -> dict | None:
    """The critical one: a native tool_calls object, not XML in the text."""
    msgs = [
        {"role": "system", "content": "You are a coding agent. Use the provided tools."},
        {"role": "user", "content": "Read the file src/main.py and tell me what it does."},
    ]
    t = time.time()
    r = chat(model, msgs, tools=TOOLS, max_tokens=512)
    msg = r["choices"][0]["message"]
    calls = msg.get("tool_calls")
    finish = r["choices"][0].get("finish_reason")

    if not calls:
        print(f"  ✗ no tool_calls (finish_reason={finish})")
        print(f"    content: {(msg.get('content') or '')[:220]!r}")
        print("    → this model's chat template likely lacks tool support")
        return None

    fn = calls[0]["function"]
    args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
    print(f"  ✓ tool_calls[0]: {fn['name']}({args})  finish_reason={finish}  {time.time() - t:.1f}s")
    if fn["name"] != "read_file":
        print(f"    ! expected read_file, got {fn['name']}")
    if "path" not in args:
        print("    ! arguments missing required 'path'")
    return {"msg": msg, "calls": calls}


def check_tool_result(model: str, prev: dict) -> bool:
    """Feed the tool result back — the round trip an agent loop repeats."""
    call = prev["calls"][0]
    msgs = [
        {"role": "system", "content": "You are a coding agent. Use the provided tools."},
        {"role": "user", "content": "Read the file src/main.py and tell me what it does."},
        prev["msg"],
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": "print('hello world')",
        },
    ]
    t = time.time()
    r = chat(model, msgs, tools=TOOLS, max_tokens=256)
    txt = (r["choices"][0]["message"].get("content") or "").strip()
    print(f"  reply in {time.time() - t:.1f}s: {txt[:180]!r}")
    if not txt:
        print("  ✗ empty reply after tool result")
        return False
    if "hello" in txt.lower() or "print" in txt.lower():
        print("  ✓ model used the tool result")
    else:
        print("  ! reply did not obviously reference the tool result")
    return True


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"endpoint : {BASE_URL}")
    print(f"model    : {model}")
    print("Note: the first request for a model blocks while its weights load.\n")

    try:
        print("[1/4] /models")
        check_models(model)

        print("\n[2/4] plain completion")
        if not check_plain(model):
            print("\nFAIL: no plain completion")
            return 1

        print("\n[3/4] native tool call")
        prev = check_tool_call(model)
        if prev is None:
            print("\nFAIL: model cannot tool-call — not usable agentically in opencode")
            return 1

        print("\n[4/4] tool result round trip")
        if not check_tool_result(model, prev):
            print("\nFAIL: tool result round trip")
            return 1

    except urllib.error.URLError as e:
        print(f"\nFAIL: cannot reach {BASE_URL} — is ./serve.sh running? ({e})")
        return 1

    print(f"\nPASS — {model} is ready for opencode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
