"""
Pydantic AI demo: tool calling + structured output + dependency injection.

Two agents, same model, same prompt:
  Agent A — no tools, returns plain str (forced to guess)
  Agent B — get_weather tool, RunContext[ServiceConfig] deps, result_type=WeatherResponse

Nemotron uses XML-style tool calls in its text output rather than OpenAI
function-call objects, so Agent B drives an explicit agentic loop: detect
the XML call, invoke the real Python function, feed the result back, then
ask the model to return a WeatherResponse JSON object.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass

from openai import AsyncOpenAI
from pydantic import BaseModel

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
# mlx_lm.server needs no auth; a gateway in front of it might.
API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")
MODEL    = "mlx-community/Qwen2.5-3B-Instruct-4bit"
PROMPT   = "What is the weather in zip code 95832?"

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=300.0)


# ── Dependency: ServiceConfig (injected at runtime) ──────────────────────────
@dataclass
class ServiceConfig:
    weather_api_endpoint: str = "https://mock-weather.internal/api"
    timeout_seconds: int = 5
    source_tag: str = "demo-v2"


# ── Structured output schema ──────────────────────────────────────────────────
class WeatherResponse(BaseModel):
    temp: str
    condition: str
    recommendation: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"</think>", "", text).strip()

def parse_tool_call(text: str):
    """Return (tool_name, {param: value}) if an XML tool call block is found."""
    m = re.search(r"<function=(\w+)>(.*?)</function>", text, re.DOTALL)
    if not m:
        return None, {}
    name = m.group(1)
    params = dict(re.findall(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", m.group(2), re.DOTALL))
    return name, params

def extract_json(text: str) -> dict | None:
    """Pull the first {...} JSON block out of model text."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None
    return None


# ── Tool registry (used by the agentic loop) ─────────────────────────────────
def get_weather(zip_code: str, *, config: ServiceConfig) -> str:
    """Stubbed weather lookup — swap for a real API call in production."""
    print(f"  [Tool] get_weather called — endpoint={config.weather_api_endpoint} tag={config.source_tag}")
    stub = {"95832": "75°F and Sunny"}
    return stub.get(zip_code, "No data for that ZIP code.")


TOOL_SYSTEM = """
You have access to one tool:

  get_weather(zip_code: str) -> str   — returns current weather for a US ZIP code.

To call it, emit ONLY this block (nothing else on that turn):
<tool_call>
<function=get_weather>
<parameter=zip_code>ZIP_CODE_HERE</parameter>
</function>
</tool_call>

After you receive the tool result, respond with a JSON object that matches
this exact schema and nothing else:
{
  "temp": "<temperature string>",
  "condition": "<weather condition>",
  "recommendation": "<clothing or activity tip>"
}
"""


# ── Agent A: no tools, plain string ──────────────────────────────────────────
async def stream_completion(messages: list[dict]) -> str:
    """Stream a chat completion and return the full response text."""
    chunks: list[str] = []
    stream = await client.chat.completions.create(model=MODEL, messages=messages, stream=True)
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            chunks.append(delta)
    return "".join(chunks)


async def run_agent_a() -> str:
    return strip_think(await stream_completion([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": PROMPT},
    ]))


# ── Agent B: tool + dependency injection + structured output ──────────────────
async def run_agent_b(config: ServiceConfig) -> tuple[WeatherResponse, list[dict]]:
    """
    Agentic loop:
      1. Ask model → it may emit an XML tool call.
      2. Detect call → invoke Python function with injected ServiceConfig.
      3. Return result → model produces WeatherResponse JSON.
      4. Parse JSON → validate into WeatherResponse Pydantic model.
    """
    messages = [
        {"role": "system", "content": "You are a weather assistant." + TOOL_SYSTEM},
        {"role": "user",   "content": PROMPT},
    ]
    tool_log: list[dict] = []

    for _ in range(5):
        reply = strip_think(await stream_completion(messages))

        tool_name, params = parse_tool_call(reply)

        if tool_name == "get_weather":
            result = get_weather(params.get("zip_code", ""), config=config)
            tool_log.append({"tool": tool_name, "params": params, "result": result})
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": (
                    f"<tool_response>\n<function={tool_name}>\n{result}\n</function>\n</tool_response>\n\n"
                    "Now respond with the WeatherResponse JSON object only."
                ),
            })
            continue

        # No tool call — try to parse a WeatherResponse from the text
        data = extract_json(reply)
        if data:
            return WeatherResponse(**data), tool_log

        # Model gave prose instead of JSON — ask it to format
        messages.append({"role": "assistant", "content": reply})
        messages.append({
            "role": "user",
            "content": "Please return only the WeatherResponse JSON object now.",
        })

    raise RuntimeError("Agent B did not produce a WeatherResponse within the iteration limit.")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    sep = "=" * 60

    print(sep)
    print(f"PROMPT  : {PROMPT}")
    print(sep)

    # --- Agent A ---
    print("\n── Agent A  (no tools, plain str) ──────────────────────────")
    answer_a = await run_agent_a()
    print(answer_a)

    # --- Agent B ---
    print("\n── Agent B  (tool + ServiceConfig deps + WeatherResponse) ──")
    config = ServiceConfig()                         # runtime dependency
    weather, calls = await run_agent_b(config)       # structured result

    print(f"\n  Type          : {type(weather).__name__}")
    print(f"  temp          : {weather.temp}")
    print(f"  condition     : {weather.condition}")
    print(f"  recommendation: {weather.recommendation}")
    print(f"\n  Full dict     : {weather.model_dump()}")

    # --- Comparison ---
    print(f"\n{sep}")
    print("COMPARISON SUMMARY")
    print(sep)
    print("Agent A → unstructured text, factually ungrounded")
    print(f"Agent B → WeatherResponse object")
    print(f"          temp={weather.temp!r}  condition={weather.condition!r}")
    print(f"          recommendation={weather.recommendation!r}")
    if calls:
        c = calls[0]
        print(f"\n  Tool invoked  : {c['tool']}(zip_code={c['params'].get('zip_code')!r})")
        print(f"  Raw result    : {c['result']!r}")
        print(f"  ServiceConfig : endpoint={config.weather_api_endpoint}  tag={config.source_tag}")
    print(sep)


if __name__ == "__main__":
    asyncio.run(main())
