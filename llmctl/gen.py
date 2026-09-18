"""Generative model runners — image, vision, speech, music, video.

These do not go through opencode. opencode speaks /v1/chat/completions with
tool calls, which is the right shape for a coding agent and the wrong shape for
diffusion and audio, so this drives each runtime's own CLI.

Two things make that awkward, and both are handled by data rather than code:

  1. Every runtime names its flags differently — mlx-audio TTS wants
     `--text` and `--output_path` (underscores), STT wants `--audio` and
     `--output-path` (hyphens), mlx-vlm wants `--image`, mflux wants
     `--image-paths`. Each Spec below records the real flag names, verified
     against `--help` rather than assumed.
  2. mflux ships a console script per model family — mflux-generate-krea2,
     mflux-generate-ideogram4, mflux-generate-flux2, mflux-generate-qwen-edit —
     so the entrypoint depends on the model, not just the class. A generative
     catalog entry may carry an "entry" field to say which.

The runtimes live in a separate conda env (DEKHO_GEN_ENV, default
`dekho-apple-gen`) so a dependency conflict here cannot break `llmctl start`.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

GEN_ENV = os.environ.get("DEKHO_GEN_ENV", "dekho-apple-gen")


def gen_python() -> Path | None:
    """Interpreter for the generative env, or None if it is not built."""
    try:
        out = subprocess.run(
            ["conda", "info", "--json"], capture_output=True, text=True, timeout=30
        ).stdout
        for env in json.loads(out).get("envs", []):
            if Path(env).name == GEN_ENV:
                p = Path(env) / "bin" / "python"
                if p.exists():
                    return p
    except Exception:
        pass
    for cand in (
        Path.home() / "miniforge3" / "envs" / GEN_ENV / "bin" / "python",
        Path.home() / "miniconda3" / "envs" / GEN_ENV / "bin" / "python",
    ):
        if cand.exists():
            return cand
    return None


def gen_bin(name: str) -> Path | None:
    py = gen_python()
    if not py:
        return None
    cand = py.parent / name
    return cand if cand.exists() else None


@dataclass
class Spec:
    """How one class of generative model is invoked, with real flag names."""

    klass: str
    label: str
    entry: str = ""  # default console script
    module: str = ""  # or `python -m module`
    model_flag: str = "--model"
    prompt_flag: str = "--prompt"
    input_flag: str = ""  # image/audio input
    output_flag: str = "--output"
    output_is_dir: bool = False
    model_is_path: bool = False  # runtime wants a local dir, not a repo id
    defaults: list[str] = field(default_factory=list)
    produces: str = "file"
    notes: str = ""

    def resolve_entry(self, override: str = "") -> tuple[list[str], str]:
        """Returns (argv prefix, human description of the entrypoint)."""
        py = gen_python()
        if not py:
            raise RuntimeError(f"generative env '{GEN_ENV}' not found — run `llmctl gen setup`")
        script = override or self.entry
        if script:
            exe = gen_bin(script)
            if not exe:
                raise RuntimeError(
                    f"{script} not installed in '{GEN_ENV}' — run `llmctl gen setup`"
                )
            return [str(exe)], script
        if not self.module:
            raise RuntimeError(f"no entrypoint defined for {self.klass}")
        return [str(py), "-m", self.module], f"python -m {self.module}"

    def available(self, override: str = "") -> bool:
        """Entrypoint is present and importable.

        This does NOT mean a given model will load: a runtime can be installed
        and still lack that model's architecture (mlx-vlm is importable but has
        no Wan/LTX module). Use `llmctl gen run --dry-run` and then a real run
        to confirm a model actually works.
        """
        try:
            cmd, _ = self.resolve_entry(override)
        except RuntimeError:
            return False
        if self.module:
            py = gen_python()
            if not py:
                return False
            root = self.module.split(".")[0]
            r = subprocess.run([str(py), "-c", f"import {root}"], capture_output=True)
            return r.returncode == 0
        return True


# Flag names below were read from each runtime's --help, not guessed.
SPECS: dict[str, Spec] = {
    "image": Spec(
        klass="image",
        label="image generation",
        entry="mflux-generate",
        model_flag="--model",
        prompt_flag="--prompt",
        output_flag="--output",
        produces="png",
        notes="mflux; entrypoint varies by model family (see catalog 'entry').",
    ),
    "image-edit": Spec(
        klass="image-edit",
        label="instruction-guided image editing",
        entry="mflux-generate-qwen-edit",
        input_flag="--image-paths",
        output_flag="--output",
        produces="png",
        notes="Pass --model explicitly: the -2511 aliases resolve to the 2509 config.",
    ),
    "vlm": Spec(
        klass="vlm",
        label="vision / image understanding",
        module="mlx_vlm.generate",
        input_flag="--image",
        output_flag="",
        produces="text",
        defaults=["--max-tokens", "512"],
        notes="mlx-vlm. Prints to stdout.",
    ),
    "omni": Spec(
        klass="omni",
        label="omni (text/image/audio in, speech out)",
        module="mlx_vlm.generate",
        input_flag="--image",
        output_flag="--output",
        produces="text+audio",
        defaults=["--max-tokens", "512"],
        notes="Add --output-modality audio for speech out.",
    ),
    "stt": Spec(
        klass="stt",
        label="speech to text",
        module="mlx_audio.stt.generate",
        prompt_flag="",
        input_flag="--audio",
        output_flag="--output-path",
        defaults=["--format", "txt"],
        produces="text",
        notes="parakeet-tdt-0.6b-v3 beats every larger MLX ASR model on WER.",
    ),
    "tts": Spec(
        klass="tts",
        label="text to speech",
        module="mlx_audio.tts.generate",
        prompt_flag="--text",
        output_flag="--output_path",
        output_is_dir=True,
        produces="wav",
        notes=(
            "Note the underscores: --text and --output_path. Kokoro needs "
            "misaki + num2words + phonemizer + espeakng_loader."
        ),
    ),
    "music": Spec(
        klass="music",
        label="music generation",
        module="mlx_audio.music.generate",
        prompt_flag="--caption",
        output_flag="--output",
        produces="wav",
        # --lyrics / --lyrics-file is a *required* mutually exclusive group, so
        # an instrumental request still has to say so explicitly.
        defaults=["--lyrics", "[instrumental]"],
        notes="Takes --caption, not --prompt. --lyrics is required; defaults to [instrumental].",
    ),
    "video": Spec(
        klass="video",
        label="video generation",
        # NOT mlx_vlm: its model dir holds only video_depth_anything, so Wan
        # and LTX fail at load even though the import succeeds. mlx-video
        # provides a console script per family, named in the catalog "entry":
        # mlx_video.wan_2.generate / mlx_video.ltx_2.generate.
        entry="mlx_video.wan_2.generate",
        model_flag="--model-dir",
        model_is_path=True,
        input_flag="--image",
        output_flag="--output-path",
        produces="mp4",
        notes=(
            "Install mlx-video from GIT, not PyPI: `pip install "
            "git+https://github.com/Blaizzy/mlx-video.git` -- the PyPI package "
            "of that name is an unrelated video-I/O library. Do NOT install "
            "mlx-gen alongside the other runtimes: it pins mlx<0.32.0 and "
            "downgrades mlx/mlx-metal to 0.31.2. Slow: ~23 min for 5s."
        ),
    ),
    "embed": Spec(
        klass="embed",
        label="embeddings / reranking",
        module="mlx_vlm.server",
        prompt_flag="",
        output_flag="",
        produces="server",
        notes="Served, not one-shot. Avoid mlx-embeddings + ModernBERT (NaN bug).",
    ),
}

# Back-compat alias for the CLI table.
RUNNERS = SPECS


def env_status() -> dict:
    py = gen_python()
    out: dict = {"env": GEN_ENV, "python": str(py) if py else None, "runners": {}}
    if not py:
        return out
    for name, spec in SPECS.items():
        try:
            _, desc = spec.resolve_entry()
            ok, err = True, ""
        except RuntimeError as e:
            _, desc, ok, err = None, spec.entry or f"python -m {spec.module}", False, str(e)
        out["runners"][name] = {
            "label": spec.label,
            "available": ok,
            "entry": desc,
            "error": err,
        }
    return out


def snapshot_dir(repo_id: str) -> str | None:
    """Local snapshot directory for a cached repo.

    mlx-video takes --model-dir rather than a repo id, so the id has to be
    resolved to the path the weights actually live at.
    """
    import glob

    safe = repo_id.replace("/", "--")
    hits = sorted(
        glob.glob(str(Path.home() / f".cache/huggingface/hub/models--{safe}/snapshots/*/"))
    )
    return hits[-1].rstrip("/") if hits else None


def build_command(
    spec: Spec,
    model: str,
    prompt: str | None = None,
    input_path: str | None = None,
    output: str | None = None,
    entry_override: str = "",
    extra: list[str] | None = None,
) -> list[str]:
    """Assemble argv for a runner, validating what the runtime requires."""
    if spec.prompt_flag and not prompt:
        raise ValueError(f"{spec.klass} needs a prompt")
    if spec.input_flag and spec.klass in ("stt", "image-edit") and not input_path:
        raise ValueError(f"{spec.klass} needs --input")
    if input_path and not Path(input_path).exists():
        raise ValueError(f"input not found: {input_path}")

    cmd, _ = spec.resolve_entry(entry_override)
    model_arg = model
    if spec.model_is_path:
        resolved = snapshot_dir(model)
        if not resolved:
            raise ValueError(
                f"{model} is not in the local cache, and {spec.klass} needs a "
                f"model directory — run `llmctl pull {model}` first"
            )
        model_arg = resolved
    cmd += [spec.model_flag, model_arg]
    if spec.prompt_flag and prompt:
        cmd += [spec.prompt_flag, prompt]
    if spec.input_flag and input_path:
        cmd += [spec.input_flag, input_path]
    if spec.output_flag and output:
        cmd += [spec.output_flag, output]
    cmd += spec.defaults
    if extra:
        cmd += extra
    return cmd
