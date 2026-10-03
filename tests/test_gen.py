"""Generative runner specs and command assembly.

Every flag name here was read from the runtime's own --help and then confirmed
by running it. These tests exist because the runtimes disagree with each other
in ways that are easy to get wrong: mlx-audio TTS uses underscores while its
STT uses hyphens, and music's --lyrics is a required argument even for an
instrumental piece.
"""

import json
import shlex

import pytest

from llmctl.gen import (
    SPECS,
    VIDEO_SIZES,
    Spec,
    build_command,
    catalog_entry,
    estimate_minutes,
    format_command,
    frames_for,
    spec_for,
)


def _spec(klass: str) -> Spec:
    return SPECS[klass]


def test_every_class_has_an_entrypoint():
    for name, spec in SPECS.items():
        assert spec.entry or spec.module, f"{name} has no entry or module"


def test_tts_uses_underscored_flags():
    """mlx_audio.tts.generate wants --text and --output_path, not --prompt."""
    s = _spec("tts")
    assert s.prompt_flag == "--text"
    assert s.output_flag == "--output_path"


def test_stt_uses_hyphenated_output_flag():
    """The same package's STT uses --output-path. They genuinely differ."""
    s = _spec("stt")
    assert s.input_flag == "--audio"
    assert s.output_flag == "--output-path"
    assert s.prompt_flag == ""  # transcription takes no prompt


def test_music_requires_lyrics_even_for_instrumental():
    s = _spec("music")
    assert s.prompt_flag == "--caption"
    assert "--lyrics" in s.defaults


def test_vlm_takes_an_image_and_prints_to_stdout():
    s = _spec("vlm")
    assert s.input_flag == "--image"
    assert s.output_flag == ""


def test_image_edit_uses_plural_image_paths():
    """mflux uses --image-paths, unlike mlx-vlm's --image."""
    assert _spec("image-edit").input_flag == "--image-paths"


def test_prompt_is_required_where_the_runtime_needs_one(monkeypatch):
    monkeypatch.setattr("llmctl.gen.gen_python", lambda: None)
    with pytest.raises(ValueError, match="needs a prompt"):
        build_command(_spec("tts"), "m", prompt=None)


def test_stt_needs_an_input_file(monkeypatch):
    monkeypatch.setattr("llmctl.gen.gen_python", lambda: None)
    with pytest.raises(ValueError, match="needs --input"):
        build_command(_spec("stt"), "m", prompt=None, input_path=None)


def test_missing_input_file_is_rejected(monkeypatch):
    monkeypatch.setattr("llmctl.gen.gen_python", lambda: None)
    with pytest.raises(ValueError, match="input not found"):
        build_command(_spec("vlm"), "m", prompt="p", input_path="/nope/x.png")


def test_missing_env_is_reported_clearly(monkeypatch):
    monkeypatch.setattr("llmctl.gen.gen_python", lambda: None)
    with pytest.raises(RuntimeError, match="gen setup"):
        build_command(_spec("music"), "m", prompt="jazz")


def test_command_shape(monkeypatch, tmp_path):
    """Assembled argv must place model, prompt, input, output and defaults."""
    fake = tmp_path / "python"
    fake.touch()
    monkeypatch.setattr("llmctl.gen.gen_python", lambda: fake)
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    cmd = build_command(_spec("vlm"), "repo/model", prompt="what is this", input_path=str(img))
    assert cmd[:3] == [str(fake), "-m", "mlx_vlm.generate"]
    assert "--model" in cmd and "repo/model" in cmd
    assert "--prompt" in cmd and "what is this" in cmd
    assert "--image" in cmd and str(img) in cmd
    assert "--max-tokens" in cmd  # spec default


def test_entry_override_selects_a_model_specific_script(monkeypatch, tmp_path):
    """mflux ships one script per model family; the catalog picks which."""
    py = tmp_path / "python"
    py.touch()
    script = tmp_path / "mflux-generate-krea2"
    script.touch()
    monkeypatch.setattr("llmctl.gen.gen_python", lambda: py)
    cmd = build_command(
        _spec("image"), "repo/krea", prompt="a cat", entry_override="mflux-generate-krea2"
    )
    assert cmd[0].endswith("mflux-generate-krea2")


def test_unknown_entry_override_is_reported(monkeypatch, tmp_path):
    py = tmp_path / "python"
    py.touch()
    monkeypatch.setattr("llmctl.gen.gen_python", lambda: py)
    with pytest.raises(RuntimeError, match="not installed"):
        build_command(_spec("image"), "m", prompt="p", entry_override="mflux-generate-nonexistent")


# ─────────────────────── per-model overrides ───────────────────────


def test_catalog_overrides_the_video_model_flag():
    """mlx-video's Wan script takes --model-dir, its LTX script --model-repo."""
    s = spec_for(
        _spec("video"), {"entry": "mlx_video.ltx_2.generate", "model_flag": "--model-repo"}
    )
    assert s.entry == "mlx_video.ltx_2.generate"
    assert s.model_flag == "--model-repo"
    assert _spec("video").model_flag == "--model-dir"  # class spec untouched


def test_spec_for_ignores_unrelated_catalog_fields():
    s = spec_for(_spec("video"), {"label": "x", "size_gb": 1.0, "blocked": "why"})
    assert s == _spec("video")


def test_every_video_entry_names_its_model_flag_or_uses_wans():
    """A video model on a non-Wan script must say how it takes the model."""
    from llmctl.paths import CATALOG

    for repo, meta in json.loads(CATALOG.read_text())["models"].items():
        if meta.get("class") != "video" or meta.get("blocked"):
            continue
        if meta.get("entry", "mlx_video.wan_2.generate") != "mlx_video.wan_2.generate":
            assert "model_flag" in meta, f"{repo} runs a non-Wan script with Wan's flag"


def test_ltx_is_blocked_with_a_reason():
    assert catalog_entry("mlx-community/ltx-2.5-mlx-q8").get("blocked")


# ─────────────────────────── video settings ───────────────────────────


@pytest.mark.parametrize("seconds,frames", [(3.4, 81), (5, 121), (1, 25), (0.01, 5)])
def test_frames_are_4n_plus_1(seconds, frames):
    assert frames_for(seconds) == frames
    assert (frames_for(seconds) - 1) % 4 == 0


def test_video_sizes_divide_by_32():
    for w, h in VIDEO_SIZES.values():
        assert w % 32 == 0 and h % 32 == 0


def test_estimate_reproduces_the_measured_run():
    assert estimate_minutes(1280, 704, 81) == pytest.approx(18 + 55 / 60)


def test_format_command_is_pasteable():
    cmd = ["/bin/x", "--prompt", "a cat's hat", "--width", "832", "--flag"]
    text = format_command(cmd)
    assert shlex.split(text.replace("\\\n", " ")) == cmd
    assert text.count("\n") == 3  # one line per flag
