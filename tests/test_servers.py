"""Server registry: port allocation, liveness pruning, stale-file handling.

No GPU or model needed — these exercise the bookkeeping that decides whether
`llmctl ps` tells the truth.
"""

import json
import os
import socket

import pytest

from llmctl import servers
from llmctl.paths import SERVERS_DIR, ensure_dirs


@pytest.fixture(autouse=True)
def clean_registry():
    ensure_dirs()
    for f in SERVERS_DIR.glob("*.json"):
        f.unlink()
    yield
    for f in SERVERS_DIR.glob("*.json"):
        f.unlink()


def _register(port, pid, model="m"):
    s = servers.Server(port=port, pid=pid, model=model)
    (SERVERS_DIR / f"{port}.json").write_text(json.dumps(s.__dict__))
    return s


def test_dead_entries_are_pruned_not_reported():
    """A kill -9'd server must not linger as a phantom in ps."""
    _register(8000, 999999)  # a pid that cannot exist
    assert servers.list_servers() == []
    assert not (SERVERS_DIR / "8000.json").exists()


def test_live_entry_is_reported():
    _register(8000, os.getpid())  # our own pid is definitely alive
    live = servers.list_servers()
    assert [s.port for s in live] == [8000]


def test_corrupt_registry_file_is_discarded():
    (SERVERS_DIR / "8000.json").write_text("{not json")
    assert servers.list_servers() == []


def test_registry_file_with_unexpected_fields_is_discarded():
    (SERVERS_DIR / "8001.json").write_text(json.dumps({"bogus": 1}))
    assert servers.list_servers() == []


def test_allocate_skips_ports_we_already_use():
    _register(8000, os.getpid())
    assert servers.allocate_port() >= 8001


def test_allocate_skips_a_port_held_by_anyone():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", servers.BASE_PORT))
        except OSError:
            pytest.skip("base port already in use by something else")
        sock.listen(1)
        assert servers.allocate_port() != servers.BASE_PORT


def test_port_free_reports_a_bound_port_as_taken():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        assert servers.port_free(port) is False
    assert servers.port_free(port) is True


def test_pid_alive():
    assert servers.pid_alive(os.getpid())
    assert not servers.pid_alive(999999)
    assert not servers.pid_alive(0)
    assert not servers.pid_alive(-1)


def test_endpoint_and_get():
    s = _register(8005, os.getpid())
    assert s.endpoint == "http://127.0.0.1:8005/v1"
    assert servers.get(8005).port == 8005
    assert servers.get(9999) is None


def test_stop_on_a_dead_pid_cleans_the_file():
    s = _register(8000, 999999)
    assert servers.stop(s) is True
    assert not (SERVERS_DIR / "8000.json").exists()


def test_uptime_handles_a_missing_timestamp():
    assert servers.Server(port=1, pid=1, model="m").uptime() == "?"
    assert servers.Server(port=1, pid=1, model="m", started_at="nonsense").uptime() == "?"


def test_token_injection_never_leaks_an_absent_file(tmp_path):
    env = {"HF_TOKEN_FILE": str(tmp_path / "nope.txt")}
    servers._inject_token(env)
    assert "HF_TOKEN" not in env


def test_token_injection_reads_and_strips(tmp_path):
    f = tmp_path / "tok.txt"
    f.write_text("  hf_abc123\n")
    env = {"HF_TOKEN_FILE": str(f)}
    servers._inject_token(env)
    assert env["HF_TOKEN"] == "hf_abc123"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "hf_abc123"


def test_existing_token_is_not_overwritten(tmp_path):
    f = tmp_path / "tok.txt"
    f.write_text("from-file")
    env = {"HF_TOKEN": "already-set", "HF_TOKEN_FILE": str(f)}
    servers._inject_token(env)
    assert env["HF_TOKEN"] == "already-set"
