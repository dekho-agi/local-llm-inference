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


def test_gpt_oss_is_documented_as_lacking_tool_calling():
    """A silent failure worth a regression test: it must stay flagged."""
    data = json.loads(CATALOG.read_text())["models"]
    entry = data["mlx-community/gpt-oss-120b-MXFP4-Q8"]
    assert entry["role"] == "chat-only"
    assert "NO TOOL CALLING" in entry["notes"]


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
