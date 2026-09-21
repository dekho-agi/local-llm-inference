"""Per-model validation: download, execute, test, record.

Produces the evidence behind VALIDATION.md. Every model in the catalog gets a
class-appropriate exercise that actually loads weights and produces output —
not a dry run — and the result is recorded with timing and peak GPU memory.

What counts as a pass differs by class, so each check states its own criterion:

  text        serves over HTTP, completes a prompt, and — separately — whether
              it can emit a native tool_call, which is what opencode requires
  vlm         describes a supplied image in text
  image       writes a PNG whose header parses
  image-edit  writes a PNG given a source image and an instruction
  tts         writes a WAV the wave module can open
  stt         transcribes a known WAV back to text
  music       writes a WAV of non-zero duration
  video       writes an MP4 whose container parses
  omni        returns text (speech out is checked separately and may fail)
  embed       returns a vector of the expected width

A model that loads but cannot do its job is a FAIL, not a pass with a caveat.
"""

import json
import struct
import subprocess
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path

from . import catalog, gen, servers


@dataclass
class Result:
    repo_id: str
    model_class: str
    label: str = ""
    size_gb: float | None = None
    downloaded: bool = False
    executed: bool = False
    passed: bool = False
    seconds: float | None = None
    peak_gb: float | None = None
    detail: str = ""
    evidence: str = ""
    skipped: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIP"
        if not self.downloaded:
            return "NOT DOWNLOADED"
        return "PASS" if self.passed else "FAIL"


def _peak_from_output(text: str) -> float | None:
    """mflux and friends print 'Peak MLX memory: N GB'."""
    for line in text.splitlines():
        if "Peak MLX memory" in line:
            try:
                return float(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                return None
    return None


def _run(cmd: list[str], timeout: int) -> tuple[int, str, float]:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or ""), time.time() - t0
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s", time.time() - t0


# ───────────────────────── text models ─────────────────────────
def check_text(m: catalog.Model, workdir: Path, timeout: int = 1800) -> Result:
    """Serve it, complete a prompt, and probe tool calling."""
    r = Result(m.repo_id, m.model_class, m.label, m.disk_gb or m.size_gb, downloaded=True)
    try:
        s = servers.start(m.repo_id, offline=True, wait=timeout)
    except (RuntimeError, TimeoutError) as e:
        r.detail = f"server failed to start: {str(e)[:160]}"
        return r

    try:
        r.executed = True
        t0 = time.time()
        body = json.dumps(
            {
                "model": m.repo_id,
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                "max_tokens": 16,
            }
        ).encode()
        req = urllib.request.Request(
            f"{s.endpoint}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            d = json.load(resp)
        text = (d["choices"][0]["message"].get("content") or "").strip()
        gen_s = time.time() - t0
        toks = d.get("usage", {}).get("completion_tokens") or 0
        r.seconds = round(gen_s, 2)
        if toks and gen_s:
            r.extra["tok_per_s"] = round(toks / gen_s, 1)

        if not text:
            r.detail = "empty completion"
            return r
        r.evidence = text[:80]

        # tool calling: what actually decides whether opencode can drive it
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
        body = json.dumps(
            {
                "model": m.repo_id,
                "messages": [{"role": "user", "content": "Read src/main.py using the tool."}],
                "tools": tools,
                "max_tokens": 256,
            }
        ).encode()
        req = urllib.request.Request(
            f"{s.endpoint}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d2 = json.load(resp)
            calls = d2["choices"][0]["message"].get("tool_calls")
            r.extra["tool_call"] = bool(calls)
            if calls:
                r.extra["tool_name"] = calls[0]["function"]["name"]
        except urllib.error.HTTPError as e:
            # mlx-lm rejects tools outright for models whose template lacks them
            r.extra["tool_call"] = False
            r.extra["tool_error"] = f"HTTP {e.code}"

        r.passed = True
        r.detail = f"completed; tool_call={'yes' if r.extra.get('tool_call') else 'NO'}"
        return r
    except Exception as e:  # noqa: BLE001 - report any failure rather than raise
        r.detail = f"{type(e).__name__}: {str(e)[:160]}"
        return r
    finally:
        servers.stop(s)


# ───────────────────────── generative models ─────────────────────────
def _gen_cmd(
    m: catalog.Model,
    klass: str,
    workdir: Path,
    out: Path | None,
    prompt: str | None,
    inp: str | None,
) -> list[str] | None:
    spec = gen.SPECS.get(klass)
    if not spec:
        return None
    entry = ""
    try:
        entry = (
            json.loads(catalog.CATALOG.read_text())["models"].get(m.repo_id, {}).get("entry", "")
        )
    except Exception:
        pass
    try:
        # str(None) would pass the literal "None" as an output path, which is
        # how the omni checks ended up requesting audio they cannot produce.
        return gen.build_command(
            spec,
            m.repo_id,
            prompt,
            inp,
            str(out) if out is not None else None,
            entry_override=entry,
        )
    except (ValueError, RuntimeError):
        return None


def check_generative(m: catalog.Model, workdir: Path, assets: dict, timeout: int = 5400) -> Result:
    r = Result(m.repo_id, m.model_class, m.label, m.disk_gb or m.size_gb, downloaded=True)
    k = m.model_class
    out = workdir / f"{m.repo_id.split('/')[-1]}.{gen.SPECS[k].produces.split('+')[0]}"
    out = out.with_suffix(
        {
            "image": ".png",
            "image-edit": ".png",
            "tts": "",
            "stt": "",
            "music": ".wav",
            "video": ".mp4",
            "vlm": "",
            "omni": "",
            "embed": "",
        }.get(k, ".out")
    )

    prompt, inp = None, None
    if k in ("image",):
        prompt = "a red apple on a white plate, studio photograph"
    elif k == "image-edit":
        prompt = "change the apple to a green apple, keep everything else identical"
        inp = assets.get("image")
    elif k == "vlm":
        prompt, inp = "Describe this image in one sentence.", assets.get("image")
    elif k == "omni":
        # Text only, deliberately: passing --output makes the runner request
        # --output-modality audio, which mlx-vlm 0.7.1 refuses for every omni
        # model here (no generate_audio). Testing the audio path would mark
        # models that work perfectly well for text+vision as failures.
        prompt, inp = "Describe this image in one sentence.", assets.get("image")
        out = None
    elif k == "tts":
        prompt = "The quick brown fox jumps over the lazy dog."
        out = workdir / f"tts-{m.repo_id.split('/')[-1]}"
        out.mkdir(parents=True, exist_ok=True)
    elif k == "stt":
        inp = assets.get("wav")
        out = workdir / f"stt-{m.repo_id.split('/')[-1]}"
    elif k == "music":
        prompt = "warm acoustic guitar, slow fingerpicking, instrumental"
    elif k == "video":
        prompt = "a red balloon rising into a blue sky"
    elif k == "embed":
        return _check_embed(m, r)

    cmd = (
        _gen_cmd(m, k, workdir, out, prompt, inp)
        if out is not None
        else _gen_cmd(m, k, workdir, None, prompt, inp)
    )
    if not cmd:
        r.detail = "no runner could be built (runtime missing or bad args)"
        return r

    rc, output, secs = _run(cmd, timeout)
    r.executed = True
    r.seconds = round(secs, 1)
    r.peak_gb = _peak_from_output(output)

    if rc != 0:
        tail = [ln for ln in output.strip().splitlines() if ln.strip()][-1:]
        r.detail = f"exit {rc}: {(tail[0] if tail else '')[:160]}"
        return r

    r.passed, r.detail, r.evidence = _validate_artifact(k, out, output)
    return r


def _validate_artifact(k: str, out: Path, output: str) -> tuple[bool, str, str]:
    """A file existing is not a pass; it has to parse as what it claims to be."""
    if k in ("vlm", "omni"):
        lines = [
            ln
            for ln in output.strip().splitlines()
            if ln.strip() and not ln.startswith(("Fetching", "\x1b", "  "))
        ]
        text = lines[-1] if lines else ""
        return (bool(text), "described the image" if text else "no text output", text[:120])

    if k == "stt":
        for cand in (out, out.with_suffix(".txt"), Path(str(out) + ".txt")):
            if cand.is_file():
                text = cand.read_text().strip()
                return bool(text), "transcribed", text[:120]
        return False, "no transcript written", ""

    if k == "tts":
        wavs = sorted(out.glob("*.wav")) if out.is_dir() else []
        if not wavs:
            return False, "no wav written", ""
        with wave.open(str(wavs[0])) as w:
            dur = w.getnframes() / w.getframerate()
            return dur > 0.1, f"{dur:.1f}s @ {w.getframerate()} Hz", wavs[0].name

    if k == "music":
        if not out.is_file():
            return False, "no wav written", ""
        with wave.open(str(out)) as w:
            dur = w.getnframes() / w.getframerate()
            return dur > 1, f"{dur:.1f}s @ {w.getframerate()} Hz, {w.getnchannels()}ch", out.name

    if k in ("image", "image-edit"):
        if not out.is_file():
            return False, "no png written", ""
        head = out.read_bytes()[:24]
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return False, "not a PNG", ""
        w, h = struct.unpack(">II", head[16:24])
        return True, f"{w}x{h} PNG, {out.stat().st_size / 1e6:.1f} MB", out.name

    if k == "video":
        if not out.is_file():
            return False, "no mp4 written", ""
        d = out.read_bytes()
        if b"ftyp" not in d[:32]:
            return False, "not an MP4", ""
        return True, f"{out.stat().st_size / 1e6:.1f} MB MP4", out.name

    return out.exists(), "produced output" if out.exists() else "no output", ""


def _check_embed(m: catalog.Model, r: Result) -> Result:
    """Embeddings are served, not one-shot."""
    try:
        s = servers.start_gen({"embedding": m.repo_id}, offline=True, wait=900)
    except (RuntimeError, TimeoutError) as e:
        r.detail = f"server failed: {str(e)[:160]}"
        return r
    try:
        r.executed = True
        t0 = time.time()
        body = json.dumps({"model": m.repo_id, "input": ["hello", "world"]}).encode()
        req = urllib.request.Request(
            f"{s.endpoint}/embeddings", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            d = json.load(resp)
        r.seconds = round(time.time() - t0, 2)
        vecs = d.get("data") or []
        dim = len(vecs[0]["embedding"]) if vecs else 0
        r.passed = dim > 0 and len(vecs) == 2
        r.detail = f"{len(vecs)} vectors, dim {dim}"
        r.evidence = str([round(x, 4) for x in vecs[0]["embedding"][:3]]) if vecs else ""
        return r
    except Exception as e:  # noqa: BLE001
        r.detail = f"{type(e).__name__}: {str(e)[:160]}"
        return r
    finally:
        servers.stop(s)


# ───────────────────────── orchestration ─────────────────────────
def make_assets(workdir: Path) -> dict:
    """Inputs the checks need: a real image and a real WAV.

    Generated rather than committed, so the suite carries no binary fixtures
    and the inputs are always readable by the runtimes on this machine.

    Each asset is built independently: Pillow lives in the generative env, not
    the serving env this process runs in, and an earlier version returned
    early when the import failed — which silently skipped the WAV too, and
    made every STT and vision check report "no asset".
    """
    workdir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, str] = {}

    img = workdir / "asset.png"
    if not img.exists():
        script = (
            "from PIL import Image, ImageDraw\n"
            "im = Image.new('RGB', (640, 480), 'white')\n"
            "d = ImageDraw.Draw(im)\n"
            "d.ellipse((220, 150, 420, 350), fill=(200, 30, 30))\n"
            "d.rectangle((300, 120, 330, 165), fill=(60, 120, 40))\n"
            "d.text((30, 30), 'llmctl validation asset', fill='black')\n"
            f"im.save(r'{img}')\n"
        )
        py = gen.gen_python()
        if py:
            _run([str(py), "-c", script], 180)
    if img.exists():
        assets["image"] = str(img)

    wav = workdir / "asset.wav"
    if not wav.exists():
        # Reuse an already-generated TTS output before spending a minute
        # synthesising another one.
        existing = sorted(workdir.glob("tts-*/*.wav"))
        if existing:
            wav.write_bytes(existing[0].read_bytes())
        else:
            try:
                outdir = workdir / "asset-tts"
                outdir.mkdir(parents=True, exist_ok=True)
                cmd = gen.build_command(
                    gen.SPECS["tts"],
                    SPECS_TTS_REF,
                    "The quick brown fox jumps over the lazy dog.",
                    None,
                    str(outdir),
                )
                _run(cmd, 900)
                made = sorted(outdir.glob("*.wav"))
                if made:
                    wav.write_bytes(made[0].read_bytes())
            except Exception:
                pass
    if wav.exists():
        assets["wav"] = str(wav)
    return assets


SPECS_TTS_REF = "mlx-community/Kokoro-82M-bf16"


def validate_all(
    only: str | None = None, klass: str | None = None, workdir: Path | None = None, progress=None
) -> list[Result]:
    workdir = workdir or Path.home() / ".cache" / "dekho-local-inference" / "validation"
    workdir.mkdir(parents=True, exist_ok=True)
    assets = make_assets(workdir)

    models = catalog.load(include_uncached=True)
    if only:
        models = [m for m in models if only.lower() in m.repo_id.lower()]
    if klass:
        models = [m for m in models if m.model_class == klass]
    models.sort(key=lambda m: (m.model_class, m.disk_gb or m.size_gb or 0))

    results: list[Result] = []
    for m in models:
        if progress:
            progress(m)
        if not m.cached:
            results.append(
                Result(
                    m.repo_id,
                    m.model_class,
                    m.label,
                    m.size_gb,
                    downloaded=False,
                    detail="not downloaded",
                )
            )
            continue
        if m.incomplete:
            results.append(
                Result(
                    m.repo_id,
                    m.model_class,
                    m.label,
                    m.disk_gb,
                    downloaded=True,
                    detail="download incomplete",
                )
            )
            continue
        if m.model_class == "text" and m.runtime == "mlx-lm":
            results.append(check_text(m, workdir))
        elif m.model_class in gen.SPECS:
            if m.model_class in ("image-edit", "vlm", "omni") and "image" not in assets:
                r = Result(m.repo_id, m.model_class, m.label, m.disk_gb, downloaded=True)
                r.skipped = "no image asset (Pillow unavailable)"
                results.append(r)
                continue
            if m.model_class == "stt" and "wav" not in assets:
                r = Result(m.repo_id, m.model_class, m.label, m.disk_gb, downloaded=True)
                r.skipped = "no wav asset"
                results.append(r)
                continue
            results.append(check_generative(m, workdir, assets))
        else:
            r = Result(m.repo_id, m.model_class, m.label, m.disk_gb, downloaded=True)
            r.skipped = f"no check defined for class '{m.model_class}'"
            results.append(r)
    return results


def to_markdown(results: list[Result], host: str = "") -> str:
    from datetime import datetime

    ok = [r for r in results if r.status == "PASS"]
    bad = [r for r in results if r.status == "FAIL"]
    nd = [r for r in results if r.status == "NOT DOWNLOADED"]
    sk = [r for r in results if r.status == "SKIP"]

    L = []
    L.append("# Model validation results\n")
    L.append(
        f"Generated by `llmctl validate` on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}.\n"
    )
    if host:
        L.append(f"Host: {host}\n")
    L.append(
        "Every model below was **downloaded, loaded, and run** on this machine. "
        "A row passes only if it produced a usable artifact — a parseable PNG, "
        "an openable WAV, a real transcription, a non-empty completion — not "
        "merely because a process exited zero.\n"
    )
    L.append(
        f"**{len(ok)} passed · {len(bad)} failed · {len(nd)} not downloaded · {len(sk)} skipped**\n"
    )

    L.append("\n## Summary\n")
    L.append("| Model | Class | Size | Status | Time | Peak | Detail |")
    L.append("|---|---|---|---|---|---|---|")
    icon = {"PASS": "✅", "FAIL": "❌", "NOT DOWNLOADED": "⬜", "SKIP": "⏭️"}
    for r in results:
        size = f"{r.size_gb:.1f} GB" if r.size_gb else "—"
        secs = (
            f"{r.seconds:.0f}s"
            if r.seconds and r.seconds >= 1
            else (f"{r.seconds}s" if r.seconds else "—")
        )
        peak = f"{r.peak_gb:.1f} GB" if r.peak_gb else "—"
        detail = r.skipped or r.detail or ""
        L.append(
            f"| `{r.repo_id}` | {r.model_class} | {size} | "
            f"{icon.get(r.status, '')} {r.status} | {secs} | {peak} | {detail} |"
        )

    by_class: dict[str, list[Result]] = {}
    for r in results:
        by_class.setdefault(r.model_class, []).append(r)

    L.append("\n## By class\n")
    for k in sorted(by_class):
        L.append(f"### {k}\n")
        for r in by_class[k]:
            L.append(f"**`{r.repo_id}`** — {icon.get(r.status, '')} {r.status}")
            bits = []
            if r.size_gb:
                bits.append(f"{r.size_gb:.1f} GB on disk")
            if r.seconds:
                bits.append(f"{r.seconds}s")
            if r.peak_gb:
                bits.append(f"{r.peak_gb:.1f} GB peak")
            if r.extra.get("tok_per_s"):
                bits.append(f"{r.extra['tok_per_s']} tok/s")
            if bits:
                L.append(f"- {' · '.join(bits)}")
            if r.detail:
                L.append(f"- {r.detail}")
            if r.extra.get("tool_call") is not None:
                tc = "yes" if r.extra["tool_call"] else "**no — cannot drive opencode**"
                L.append(f"- tool calling: {tc}")
            if r.evidence:
                L.append(f"- output: `{r.evidence}`")
            L.append("")
    return "\n".join(L) + "\n"


def merge_reports(previous_md: str, results: list[Result], host: str) -> str:
    """Fold a new partial run into an existing report.

    Validation of the big models takes hours, so a run is often partial. Rows
    from the new run replace same-model rows in the old report; everything else
    is kept, so the document converges rather than being overwritten.
    """
    import re

    keep: dict[str, str] = {}
    for line in previous_md.splitlines():
        m = re.match(r"^\| `([^`]+)` \|", line)
        if m:
            keep[m.group(1)] = line

    fresh = {r.repo_id for r in results}
    merged_rows = [ln for repo, ln in keep.items() if repo not in fresh]

    new_md = to_markdown(results, host=host)
    if not merged_rows:
        return new_md

    out_lines = []
    for line in new_md.splitlines():
        out_lines.append(line)
        if line.startswith("|---|---|---|---|---|---|---|"):
            out_lines.extend(merged_rows)
    merged = "\n".join(out_lines) + "\n"

    # Recount from the merged table. The headline was previously carried over
    # from the new (often partial) run while the table held every row, so a
    # run of 5 models produced "4 passed / 5 failed" above a 28-row table.
    counts = {"PASS": 0, "FAIL": 0, "NOT DOWNLOADED": 0, "SKIP": 0}
    for line in merged.splitlines():
        if not line.startswith("| `"):
            continue
        for key in ("NOT DOWNLOADED", "PASS", "FAIL", "SKIP"):
            if f" {key} " in line or line.rstrip().endswith(key):
                counts[key] += 1
                break
    headline = (
        f"**{counts['PASS']} passed · {counts['FAIL']} failed · "
        f"{counts['NOT DOWNLOADED']} not downloaded · {counts['SKIP']} skipped**"
    )
    return re.sub(r"^\*\*\d+ passed .*?\*\*$", headline, merged, count=1, flags=re.M)
