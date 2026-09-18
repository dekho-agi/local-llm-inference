"""
Vet a model's tool-calling support BEFORE downloading tens of GB of weights.

opencode is only useful with a model mlx-lm can parse tool calls out of, and
that depends entirely on the model's chat template matching one of mlx-lm's
tool parsers. This fetches just the template (a few KB) and runs mlx-lm's own
`_infer_tool_parser` against it, so you learn the answer in seconds.

  python check-tool-parser.py                     # check the shortlist
  python check-tool-parser.py <repo-id> [...]     # check specific repos

A model reporting parser=None cannot emit OpenAI `tool_calls` under mlx-lm.
It will instead leak its raw tool syntax into the message content.
"""

import json
import sys
import urllib.error
import urllib.request

from mlx_lm.tokenizer_utils import _infer_tool_parser

SHORTLIST = [
    "mlx-community/Qwen3-Coder-Next-4bit",
    "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    "mlx-community/gpt-oss-120b-MXFP4-Q8",
    "mlx-community/GLM-4.7-Flash-4bit",
    "mlx-community/GLM-4.5-Air-4bit",
    "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit",
    "mlx-community/Qwen3.8-27B-4bit",
    "mlx-community/Qwen3.6-35B-A3B-4bit",
    "mlx-community/gemma-4-26b-a4b-it-4bit",
]


def fetch(repo: str, path: str) -> str | None:
    url = f"https://huggingface.co/{repo}/raw/main/{path}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read().decode()
    except urllib.error.HTTPError:
        return None
    except Exception:
        return None


def template_for(repo: str) -> str | None:
    """Templates live either in chat_template.jinja or inside tokenizer_config."""
    t = fetch(repo, "chat_template.jinja")
    if t:
        return t
    cfg = fetch(repo, "tokenizer_config.json")
    if cfg:
        try:
            return json.loads(cfg).get("chat_template")
        except json.JSONDecodeError:
            return None
    return None


def main() -> int:
    repos = sys.argv[1:] or SHORTLIST
    print(f"{'model':54s} {'parser':14s} tools?")
    print("-" * 78)
    bad = 0
    for repo in repos:
        tpl = template_for(repo)
        if tpl is None:
            print(f"{repo.replace('mlx-community/',''):54s} {'?':14s} no template found")
            bad += 1
            continue
        parser = _infer_tool_parser(tpl)
        mentions = "tool" in tpl.lower()
        if parser:
            verdict = "OK"
        elif mentions:
            verdict = "LEAKS RAW SYNTAX — template has tools, mlx-lm has no parser"
            bad += 1
        else:
            verdict = "no tool support in template"
            bad += 1
        print(f"{repo.replace('mlx-community/',''):54s} {str(parser):14s} {verdict}")
    print()
    print("A parser of None means opencode cannot drive that model agentically.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
