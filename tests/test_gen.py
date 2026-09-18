"""Generative runner specs and command assembly.

Every flag name here was read from the runtime's own --help and then confirmed
by running it. These tests exist because the runtimes disagree with each other
in ways that are easy to get wrong: mlx-audio TTS uses underscores while its
STT uses hyphens, and music's --lyrics is a required argument even for an
instrumental piece.
"""

import pytest

from llmctl.gen import SPECS, Spec, build_command


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
