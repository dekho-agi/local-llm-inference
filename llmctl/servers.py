"""Server registry: run several models at once, each on its own port.

One mlx_lm.server can hot-swap models, but swapping costs ~3x throughput until
restart (see apple-m-series/m5-128gb/CONTEXT.md). So when you want two models
concurrently — a 30B and an 8B, say — you run two servers. This tracks them.

Each live server owns <RUN_DIR>/servers/<port>.json. The registry is rebuilt
from those files on every read and dead entries are pruned, so a crashed or
kill -9'd server never lingers as a phantom.
"""

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .paths import LOG_DIR, SERVERS_DIR, ensure_dirs, env_python

BASE_PORT = 8000
PORT_SEARCH_LIMIT = 64


@dataclass
class Server:
    port: int
    pid: int
    model: str
    context: int | None = None
    max_tokens: int = 32768
    prompt_cache_bytes: str = "24G"
    kv_bits: int | None = None
    host: str = "127.0.0.1"
    log: str = ""
    started_at: str = ""
    profile: str = ""

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def alive(self) -> bool:
        return pid_alive(self.pid)

    def uptime(self) -> str:
        if not self.started_at:
            return "?"
        try:
            t0 = datetime.fromisoformat(self.started_at)
        except ValueError:
            return "?"
        secs = int((datetime.now(UTC) - t0).total_seconds())
        if secs < 90:
            return f"{secs}s"
        if secs < 5400:
            return f"{secs // 60}m"
        return f"{secs // 3600}h{(secs % 3600) // 60:02d}m"

    def responds(self, timeout: float = 2.0) -> bool:
        try:
            with urllib.request.urlopen(f"{self.endpoint}/models", timeout=timeout):
                return True
        except Exception:
            return False

    def loaded_models(self, timeout: float = 2.0) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.endpoint}/models", timeout=timeout) as r:
                return [m["id"] for m in json.load(r).get("data", [])]
        except Exception:
            return []


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _path(port: int) -> Path:
    return SERVERS_DIR / f"{port}.json"


def list_servers(prune: bool = True) -> list[Server]:
    ensure_dirs()
    out: list[Server] = []
    for f in sorted(SERVERS_DIR.glob("*.json")):
        try:
            s = Server(**json.loads(f.read_text()))
        except (json.JSONDecodeError, TypeError, OSError):
            f.unlink(missing_ok=True)
            continue
        if s.alive:
            out.append(s)
        elif prune:
            f.unlink(missing_ok=True)
    return out


def get(port: int) -> Server | None:
    for s in list_servers():
        if s.port == port:
            return s
    return None


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def allocate_port(host: str = "127.0.0.1") -> int:
    """First free port at or above BASE_PORT that we are not already using."""
    taken = {s.port for s in list_servers()}
    for port in range(BASE_PORT, BASE_PORT + PORT_SEARCH_LIMIT):
        if port not in taken and port_free(port, host):
            return port
    raise RuntimeError(f"no free port in {BASE_PORT}-{BASE_PORT + PORT_SEARCH_LIMIT - 1}")


def start(
    model: str,
    port: int | None = None,
    context: int | None = None,
    max_tokens: int = 32768,
    prompt_cache_bytes: str = "24G",
    kv_bits: int | None = None,
    host: str = "127.0.0.1",
    profile: str = "",
    offline: bool = False,
    wait: int = 900,
    on_wait=None,
) -> Server:
    """Spawn a detached server and block until it answers, or raise."""
    ensure_dirs()
    port = port or allocate_port(host)
    if not port_free(port, host):
        existing = get(port)
        if existing:
            raise RuntimeError(f"port {port} already serving {existing.model} (pid {existing.pid})")
        raise RuntimeError(f"port {port} is in use by a process we do not manage")

    args = [
        str(env_python()),
        "-m",
        "mlx_lm.server",
        "--model",
        model,
        "--port",
        str(port),
        "--host",
        host,
        "--max-tokens",
        str(max_tokens),
        "--prompt-cache-bytes",
        prompt_cache_bytes,
    ]
    if kv_bits:
        args += ["--kv-bits", str(kv_bits)]

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    _inject_token(env)

    log = LOG_DIR / f"{port}.log"
    with log.open("w") as fh:
        fh.write(f"# llmctl start {model} on :{port}\n")
        fh.flush()
        proc = subprocess.Popen(
            args,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detach: survives the shell that launched it
            env=env,
            cwd=str(Path.home()),
        )

    s = Server(
        port=port,
        pid=proc.pid,
        model=model,
        context=context,
        max_tokens=max_tokens,
        prompt_cache_bytes=prompt_cache_bytes,
        kv_bits=kv_bits,
        host=host,
        log=str(log),
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        profile=profile,
    )
    _path(port).write_text(json.dumps(asdict(s), indent=2) + "\n")

    for i in range(wait):
        if not pid_alive(proc.pid):
            _path(port).unlink(missing_ok=True)
            tail = "\n".join(log.read_text().splitlines()[-30:])
            raise RuntimeError(f"server exited during startup.\n--- log ---\n{tail}")
        if s.responds(timeout=2):
            return s
        if on_wait:
            on_wait(i)
        time.sleep(1)

    raise TimeoutError(f"server on :{port} did not answer within {wait}s; still loading? see {log}")


def stop(s: Server, grace: float = 10.0) -> bool:
    """SIGTERM, then SIGKILL. Returns True if it is gone."""
    if not s.alive:
        _path(s.port).unlink(missing_ok=True)
        return True
    try:
        os.kill(s.pid, signal.SIGTERM)
    except ProcessLookupError:
        _path(s.port).unlink(missing_ok=True)
        return True

    deadline = time.time() + grace
    while time.time() < deadline:
        if not pid_alive(s.pid):
            _path(s.port).unlink(missing_ok=True)
            return True
        time.sleep(0.25)

    try:
        os.kill(s.pid, signal.SIGKILL)
        time.sleep(1)
    except ProcessLookupError:
        pass
    gone = not pid_alive(s.pid)
    if gone:
        _path(s.port).unlink(missing_ok=True)
    return gone


def unmanaged() -> list[int]:
    """PIDs of mlx_lm.server processes we have no registry entry for.

    These come from a foreground `serve.sh`, or a registry wiped by hand.
    """
    known = {s.pid for s in list_servers()}
    out = []
    try:
        res = subprocess.run(["pgrep", "-f", "mlx_lm.server"], capture_output=True, text=True)
        for line in res.stdout.split():
            pid = int(line)
            if pid not in known and pid != os.getpid():
                out.append(pid)
    except Exception:
        pass
    return out


def _inject_token(env: dict) -> None:
    """Resolve the HF token without ever printing it."""
    if env.get("HF_TOKEN"):
        env.setdefault("HUGGING_FACE_HUB_TOKEN", env["HF_TOKEN"])
        return
    p = Path(env.get("HF_TOKEN_FILE", Path.home() / ".dekho" / "hugging-face-token.txt"))
    try:
        tok = p.read_text().strip()
    except OSError:
        return
    if tok:
        env["HF_TOKEN"] = tok
        env["HUGGING_FACE_HUB_TOKEN"] = tok
