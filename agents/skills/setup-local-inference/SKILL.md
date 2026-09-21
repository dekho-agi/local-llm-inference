---
name: setup-local-inference
description: Set up a Mac for local LLM inference with MLX and opencode, end to end — detect the machine, create the env, pull and verify models, install the opencode discovery plugin, and explain how the pieces fit. Also use to explain or troubleshoot an existing setup, or to add a model. Covers Apple silicon only; the rtx-* targets are Docker Compose and out of scope.
---

# setup-local-inference

Gets a Mac from nothing to "opencode drives a local model with no network", and
explains the system well enough that the user can operate it alone.

## When to invoke

- "set up local inference", "set up local models", "get opencode working offline"
- Preparing a machine for offline work (a flight, no connectivity)
- Adding a model to an existing setup
- Explaining or debugging a setup that already exists

Apple silicon only. For NVIDIA, use the `rtx-*` Compose stacks instead.

## Rules

Read `AGENTS.md` first. Two failure modes make this worth doing carefully:

- A model that cannot tool-call is useless to opencode, and the failure is
  silent. Gate on `check-tool-parser.py`.
- Weights must be cached before connectivity is lost; `hf download` can exit 0
  on an incomplete download, so gate on `llmctl verify`.

## Steps

1. **Probe.** `./llmctl.sh host` — reports chip, memory, GPU working-set
   ceiling and the matching profile. If `AGENTS.md`
   already describes this machine, use it. No profile fits ⇒ copy the closest
   target dir, adjust defaults and `models.txt`.

2. **Environment.** `conda env create -f apple-m-series/common/environment.yml`.
   Needs `mlx-lm >= 0.31.3`. An unknown-architecture error at load means
   checking the model's `config.json` `architectures` against installed
   `mlx_lm/models/`.

3. **Choose, and gate.** `python apple-m-series/m5-128gb/check-tool-parser.py
   <repo-id>` costs a few KB instead of tens of GB. `parser=None` ⇒ unusable
   for agents. Recommend from `docs/models.md`, and check context cost there
   too — the 80B is 4x cheaper per token than the 30B.

4. **Pull and verify.** `./llmctl.sh pull` then `./llmctl.sh verify`. Pull
   serially; six parallel downloads saturated the link and four failed.
   Throughput swings 2–70 MB/s, so a slow window is not a stall.

5. **Serve.** `./llmctl.sh start` — backgrounds, waits for readiness, so a
   failed load reports at the prompt. One model per session: hot-swapping
   costs ~3x throughput.

6. **Wire opencode.** `./llmctl.sh opencode --install`, then
   `opencode models dekho-local-inference` to confirm without the TUI.

7. **Prove it.** `python smoke-test.py <repo-id>` — four checks, of which the
   native `tool_calls` object is the one that matters. Then the real thing:

   ```bash
   ./llmctl.sh stop --all
   ./llmctl.sh start --offline
   opencode run --model dekho-local-inference/<repo-id> "make a trivial edit"
   ```

   Do not report success until a tool call has executed with
   `HF_HUB_OFFLINE=1`.

## Adding a model later

`check-tool-parser.py` → add to `models.txt` with size and parser in a comment
→ optional label in `llmctl/catalog.json` → `pull` → `verify` →
`llmctl models --ctx N` to confirm fit → restart so the manifest is rewritten.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Model missing from `/models` | Restart `llmctl start`; the manifest is rewritten on start |
| Raw `<\|channel\|>` or XML in replies | No mlx-lm parser for that model |
| Turn truncates mid-tool-call | `MAX_TOKENS` too low; mlx-lm's default is 512 |
| Throughput dropped ~3x | Server has been hot-swapping; restart it |
| Swapping | Over the ~100 GB budget; smaller model or context |
| Works online, fails offline | Repo never cached; `llmctl verify` |
| `gen run` flag errors | `--dry-run`; each runtime names flags differently |
