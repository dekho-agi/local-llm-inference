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
