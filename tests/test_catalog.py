"""Catalog loading and the shipped catalog.json itself."""

import json

from llmctl import catalog
from llmctl.paths import CATALOG

VALID_CLASSES = {
    "text",
    "image",
    "image-edit",
    "vlm",
    "omni",
    "stt",
    "tts",
    "music",
    "video",
    "embed",
}
VALID_RUNTIMES = {"mlx-lm", "mlx-vlm", "mflux", "mlx-audio", "mlx-gen", "other"}


def test_catalog_json_is_valid_and_present():
    data = json.loads(CATALOG.read_text())
    assert data["models"], "catalog should not be empty"


def test_every_catalog_entry_is_well_formed():
    data = json.loads(CATALOG.read_text())["models"]
    for repo_id, meta in data.items():
        assert "/" in repo_id, f"{repo_id} is not a repo id"
        assert meta.get("class") in VALID_CLASSES, f"{repo_id}: bad class"
        assert meta.get("runtime") in VALID_RUNTIMES, f"{repo_id}: bad runtime"
        size = meta.get("size_gb")
        assert size is None or size > 0, f"{repo_id}: bad size"


def test_generative_models_are_not_marked_servable():
    """Only mlx-lm text models can be served by `llmctl start`."""
    data = json.loads(CATALOG.read_text())["models"]
    for repo_id, meta in data.items():
        if meta["class"] != "text":
            assert meta["runtime"] != "mlx-lm", (
                f"{repo_id} is {meta['class']} but claims the mlx-lm runtime"
            )


def test_gpt_oss_tool_calling_limitation_is_documented():
    """The limitation is mlx-lm's, not the model's — the note must say so.

    An earlier version of this test asserted "NO TOOL CALLING", which encoded
    a wrong conclusion: the model emits correct harmony calls, and mlx-lm
    cannot deliver them. See llmctl/tool_parsers/harmony.py.
    """
    data = json.loads(CATALOG.read_text())["models"]
    entry = data["mlx-community/gpt-oss-120b-MXFP4-Q8"]
    assert entry["role"] == "chat-only"
    notes = entry["notes"]
    assert "mlx-lm" in notes, "must attribute the limitation to the runtime"
    assert "harmony" in notes.lower()


def test_find_resolves_exact_and_unique_substring():
    assert catalog.find("mlx-community/Qwen3-Coder-Next-4bit") is not None
    assert catalog.find("Qwen3-Coder-Next") is not None
    assert catalog.find("definitely-not-a-model-xyz") is None


def test_find_rejects_an_ambiguous_substring():
    # "Qwen3-Coder" matches both Next and 30B, so it must not guess
    assert catalog.find("Qwen3-Coder") is None


def test_load_returns_models_and_labels_them():
    ms = catalog.load(include_uncached=True)
    assert ms
    assert all(m.label for m in ms)


def test_incomplete_flag_is_exposed_on_the_model():
    """A mid-download model must not be offered as ready.

    Offering one looks identical to a complete model in a picker and then
    fails on load.
    """
    m = catalog.Model(repo_id="x", cached=True, incomplete=True, runtime="mlx-lm")
    assert not m.ready
    assert not m.servable
    assert catalog.Model(repo_id="y", cached=True, runtime="mlx-lm").ready


def test_a_repo_with_no_weight_files_is_not_ready(tmp_path, monkeypatch):
    """A failed pull leaves the repo dir, snapshot and config.json behind.

    Regression: scan_cache called that 'cached' and therefore ready, so
    pickers offered models that had no weights at all and failed at load.
    """
    from llmctl import catalog as c

    repo = tmp_path / "models--x--y"
    snap = repo / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text('{"max_position_embeddings": 4096}')
    (repo / "blobs").mkdir()

    class FakeRepo:
        repo_id = "x/y"
        repo_type = "model"
        repo_path = repo
        size_on_disk = 1234

    class FakeInfo:
        repos = [FakeRepo()]

    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda *a, **k: FakeInfo())
    got = c.scan_cache()
    assert "x/y" in got
    assert got["x/y"].incomplete, "no safetensors present, must not be ready"
    assert not got["x/y"].ready

    # and with a weight file it becomes ready
    (snap / "model.safetensors").write_bytes(b"\x00" * 16)
    got2 = c.scan_cache()
    assert got2["x/y"].ready
