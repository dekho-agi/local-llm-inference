"""The opencode contract: the manifest shape the plugin depends on."""

import json
import os

import pytest

from llmctl import manifest, servers
from llmctl.paths import MANIFEST, SERVERS_DIR, ensure_dirs


@pytest.fixture(autouse=True)
def clean():
    ensure_dirs()
    for f in SERVERS_DIR.glob("*.json"):
        f.unlink()
    MANIFEST.unlink(missing_ok=True)
    yield


def _register(port, model):
    s = servers.Server(port=port, pid=os.getpid(), model=model, context=262144)
    (SERVERS_DIR / f"{port}.json").write_text(json.dumps(s.__dict__))


def test_no_servers_gives_an_empty_but_valid_manifest():
    d = manifest.build()
    assert d["version"] == 2
    assert d["servers"] == []
    # v1 keys stay present so an un-upgraded plugin does not crash
    assert d["endpoint"] is None
    assert d["preloaded"] is None


def test_lowest_port_keeps_the_stable_provider_id():
    """Existing opencode model selections must keep working."""
    _register(8001, "b")
    _register(8000, "a")
    d = manifest.build()
    ids = [s["provider_id"] for s in d["servers"]]
    assert ids[0] == manifest.PRIMARY_ID
    assert ids[1] == f"{manifest.PRIMARY_ID}-8001"


def test_servers_are_ordered_by_port():
    for p in (8002, 8000, 8001):
        _register(p, f"m{p}")
    assert [s["port"] for s in manifest.build()["servers"]] == [8000, 8001, 8002]


def test_v1_compatibility_fields_track_the_primary_server():
    _register(8000, "mlx-community/x")
    d = manifest.build()
    assert d["endpoint"] == "http://127.0.0.1:8000/v1"
    assert d["preloaded"] == "mlx-community/x"
    assert d["max_context_cap"] == 262144


def test_write_is_atomic_and_leaves_no_temp_file():
    _register(8000, "m")
    manifest.write()
    assert MANIFEST.exists()
    assert not MANIFEST.with_suffix(".json.tmp").exists()
    json.loads(MANIFEST.read_text())  # valid JSON


def test_written_manifest_matches_build():
    _register(8000, "m")
    written = manifest.write()
    assert json.loads(MANIFEST.read_text())["servers"] == written["servers"]


def _register_gen(port, roles):
    s = servers.Server(
        port=port,
        pid=os.getpid(),
        model=next(iter(roles.values())),
        kind="gen",
        roles=roles,
    )
    (SERVERS_DIR / f"{port}.json").write_text(json.dumps(s.__dict__))


def test_generative_server_is_never_advertised_to_opencode():
    """Regression: a gen server on the lower port claimed the primary id.

    Its models are TTS/STT/embedding — useless as chat models — and opencode
    would have defaulted to it.
    """
    _register_gen(8000, {"tts": "mlx-community/Kokoro-82M-bf16"})
    _register(8001, "mlx-community/Qwen3-Coder-Next-4bit")
    d = manifest.build()
    ports = [s["port"] for s in d["servers"]]
    assert ports == [8001], "generative server must be excluded"
    assert d["servers"][0]["provider_id"] == manifest.PRIMARY_ID
    assert d["preloaded"] == "mlx-community/Qwen3-Coder-Next-4bit"


def test_a_lone_generative_server_yields_no_providers():
    _register_gen(8000, {"embedding": "mlx-community/Qwen3-Embedding-0.6B-8bit"})
    d = manifest.build()
    assert d["servers"] == []
    assert d["endpoint"] is None


def test_manifest_models_are_only_servable_text_models():
    """Non-LLM runtimes must never reach opencode's model list."""
    _register(8000, "mlx-community/Qwen3-Coder-Next-4bit")
    models = manifest.build()["models"]
    for repo, meta in models.items():
        assert meta["context"], f"{repo} has no context window"
        # a TTS/STT/image model would have no context or a non-mlx-lm runtime
        assert "Kokoro" not in repo
        assert "parakeet" not in repo
        assert "Embedding" not in repo
