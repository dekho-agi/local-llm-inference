"""llmctl CLI. typer for commands/args, rich for tables and prompts."""

import json
import subprocess

import typer
from rich.console import Console
from rich.prompt import Confirm, IntPrompt
from rich.table import Table

from . import catalog, gpu, manifest, servers
from . import host as hostmod
from .paths import LOG_DIR, MANIFEST, RUN_DIR

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local inference toolkit: host detection, model catalog, servers, cache.",
)
console = Console()


# ─────────────────────────── shared helpers ───────────────────────────
def _fit(m: catalog.Model, budget: float, ctx: int | None = None) -> str:
    total = m.total_gb(ctx)
    if total is None:
        return "?"
    if total > budget:
        return "[red]no fit[/]"
    if total > budget * 0.8:
        return "[yellow]tight[/]"
    return "[green]ok[/]"


def _choose_model(models: list[catalog.Model], prompt: str) -> catalog.Model:
    """Numbered picker. Raises typer.Exit if there is nothing to choose."""
    if not models:
        console.print("[red]no models available[/] — run [bold]llmctl pull[/] first")
        raise typer.Exit(1)
    if len(models) == 1:
        console.print(f"only one option: [bold]{models[0].repo_id}[/]")
        return models[0]

    h = hostmod.detect()
    t = Table(box=None, pad_edge=False)
    t.add_column("#", justify="right", style="bold cyan")
    t.add_column("model")
    t.add_column("params")
    t.add_column("disk", justify="right")
    t.add_column("ctx", justify="right")
    t.add_column("KV@ctx", justify="right")
    t.add_column("total", justify="right")
    t.add_column("fit")
    t.add_column("tools")
    for i, m in enumerate(models, 1):
        t.add_row(
            str(i),
            m.label or m.repo_id.split("/")[-1],
            m.params or "—",
            f"{m.disk_gb:.1f}G" if m.disk_gb else "—",
            f"{(m.context or 0) // 1024}k" if m.context else "—",
            f"{m.kv_gb():.1f}G" if m.kv_gb() else "—",
            f"{m.total_gb():.1f}G" if m.total_gb() else "—",
            _fit(m, h.budget_gb),
            "[green]yes[/]" if m.agentic else "[red]no[/]",
        )
    console.print(t)
    idx = IntPrompt.ask(
        prompt, choices=[str(i) for i in range(1, len(models) + 1)], show_choices=False
    )
    return models[idx - 1]


def _choose_server(prompt: str) -> list[servers.Server]:
    live = servers.list_servers()
    if not live:
        console.print("no servers running")
        raise typer.Exit(0)
    if len(live) == 1:
        return live

    t = Table(box=None, pad_edge=False)
    t.add_column("#", justify="right", style="bold cyan")
    t.add_column("port", justify="right")
    t.add_column("model")
    t.add_column("pid", justify="right")
    t.add_column("up", justify="right")
    t.add_column("health")
    for i, s in enumerate(live, 1):
        t.add_row(
            str(i),
            str(s.port),
            s.model.split("/")[-1],
            str(s.pid),
            s.uptime(),
            "[green]ok[/]" if s.responds() else "[yellow]not answering[/]",
        )
    t.add_row("0", "—", "[bold]all of them[/]", "", "", "")
    console.print(t)
    idx = IntPrompt.ask(
        prompt, choices=[str(i) for i in range(0, len(live) + 1)], show_choices=False
    )
    return live if idx == 0 else [live[idx - 1]]


# ─────────────────────────── commands ───────────────────────────
@app.command()
def host():
    """Show the detected machine and what it can hold."""
    h = hostmod.detect()
    t = Table(box=None, pad_edge=False, show_header=False)
    t.add_row("model", h.model)
    t.add_row("chip", f"{h.chip} · {h.cpu_cores} CPU · {h.gpu_cores} GPU cores")
    t.add_row("macOS", h.macos)
    t.add_row("unified memory", f"{h.memory_gb:.0f} GB")
    if h.working_set_gb:
        t.add_row(
            "GPU working set",
            f"[bold]{h.working_set_gb:.1f} GB[/]  (the real ceiling, not total RAM)",
        )
    t.add_row("plan against", f"[bold]{h.budget_gb:.1f} GB[/]  (working set less OS headroom)")
    t.add_row("profile", f"{h.profile}  [dim]{h.profile_dir or ''}[/]")
    console.print(t)
    console.print(
        "\n[dim]mlx-lm raises the wired limit to the working set itself; no sudo sysctl needed.[/]"
    )


@app.command(name="models")
def models_cmd(
    all_: bool = typer.Option(False, "--all", "-a", help="Include models not downloaded."),
    model_class: str | None = typer.Option(
        None,
        "--class",
        "-c",
        help="Filter: text, image, image-edit, vlm, omni, stt, tts, music, video, embed.",
    ),
    ctx: int | None = typer.Option(None, "--ctx", help="Evaluate fit at this context size."),
):
    """List models: what is cached, what it costs, whether it can tool-call."""
    h = hostmod.detect()
    ms = catalog.load(include_uncached=all_)
    if model_class:
        ms = [m for m in ms if m.model_class == model_class]
    if not ms:
        console.print("no models match")
        raise typer.Exit(0)

    t = Table(
        title=f"budget {h.budget_gb:.1f} GB" + (f" · ctx {ctx:,}" if ctx else ""),
        title_justify="left",
        box=None,
        pad_edge=False,
    )
    t.add_column("model")
    t.add_column("class")
    t.add_column("runtime")
    t.add_column("params")
    t.add_column("size", justify="right")
    t.add_column("ctx", justify="right")
    t.add_column("total", justify="right")
    t.add_column("fit")
    t.add_column("tools")
    t.add_column("on disk")
    for m in ms:
        size = m.disk_gb or m.size_gb
        t.add_row(
            m.repo_id.replace("mlx-community/", ""),
            m.model_class,
            m.runtime,
            m.params or "—",
            f"{size:.1f}G" if size else "—",
            f"{(m.context_native or 0) // 1024}k" if m.context_native else "—",
            f"{m.total_gb(ctx):.1f}G" if m.total_gb(ctx) else "—",
            _fit(m, h.budget_gb, ctx) if m.runtime == "mlx-lm" else "—",
            ("[green]yes[/]" if m.agentic else "[red]no[/]") if m.runtime == "mlx-lm" else "—",
            "[green]yes[/]" if m.cached else "[dim]no[/]",
        )
    console.print(t)
    if not all_:
        console.print("[dim]--all to include models in the catalog but not downloaded[/]")


@app.command()
def start(
    model: str | None = typer.Argument(None, help="Repo id or a unique substring."),
    port: int | None = typer.Option(None, "--port", "-p", help="Default: first free from 8000."),
    ctx: int | None = typer.Option(None, "--ctx", help="Context advertised to opencode."),
    max_tokens: int = typer.Option(32768, "--max-tokens"),
    prompt_cache: str = typer.Option("24G", "--prompt-cache", help="e.g. 24G, 8G."),
    kv_bits: int | None = typer.Option(None, "--kv-bits", help="4 or 8. Disables batching."),
    offline: bool = typer.Option(False, "--offline", help="Set HF_HUB_OFFLINE=1."),
):
    """Start a server. With no model, pick one from a list."""
    h = hostmod.detect()
    if model:
        m = catalog.find(model)
        if not m:
            console.print(f"[red]no unique match for[/] {model!r}")
            raise typer.Exit(1)
        if not m.cached:
            console.print(
                f"[red]{m.repo_id} is not downloaded[/] — run [bold]llmctl pull {m.repo_id}[/]"
            )
            raise typer.Exit(1)
    else:
        m = _choose_model(catalog.servable(), "start which model")

    if not m.agentic:
        console.print(
            f"[yellow]note[/] {m.label}: mlx-lm has no tool parser for it, "
            "so opencode cannot drive it agentically (chat only)."
        )

    context = ctx or m.context_native
    if m.context_native and context and context > m.context_native:
        console.print(
            f"[yellow]clamping context[/] {context:,} → {m.context_native:,} "
            "(the model's trained window)"
        )
        context = m.context_native

    total = m.total_gb(context)
    if total and total > h.budget_gb:
        console.print(f"[red]would not fit[/]: {total:.1f} GB needed, {h.budget_gb:.1f} GB budget")
        if not Confirm.ask("start anyway", default=False):
            raise typer.Exit(1)

    live = servers.list_servers()
    if live:
        console.print(
            f"[dim]{len(live)} server(s) already running on "
            f"{', '.join(str(s.port) for s in live)}[/]"
        )

    with console.status(f"loading {m.label} …") as status:

        def tick(i):
            status.update(f"loading {m.label} … {i}s")

        try:
            s = servers.start(
                m.repo_id,
                port=port,
                context=context,
                max_tokens=max_tokens,
                prompt_cache_bytes=prompt_cache,
                kv_bits=kv_bits,
                profile=h.profile,
                offline=offline,
                on_wait=tick,
            )
        except (RuntimeError, TimeoutError) as e:
            console.print(f"[red]failed:[/] {e}")
            raise typer.Exit(1) from e

    manifest.write()
    console.print(f"[green]ready[/]  {m.label}")
    console.print(f"  endpoint   {s.endpoint}")
    console.print(f"  port/pid   {s.port} / {s.pid}")
    console.print(f"  context    {context:,}" if context else "  context    —")
    if total:
        console.print(f"  memory     ~{total:.1f} GB of {h.budget_gb:.1f} GB budget")
    console.print(f"  log        {s.log}")
    console.print("\nnext: [bold]opencode[/] → /models → Dekho Local Inference")


@app.command()
def stop(
    port: int | None = typer.Option(None, "--port", "-p", help="Stop this port."),
    all_: bool = typer.Option(False, "--all", "-a", help="Stop every server."),
):
    """Stop a server. With no options, pick from a list."""
    if port is not None:
        s = servers.get(port)
        if not s:
            console.print(f"no managed server on port {port}")
            raise typer.Exit(1)
        targets = [s]
    elif all_:
        targets = servers.list_servers()
        if not targets:
            console.print("no servers running")
    else:
        targets = _choose_server("stop which server")

    for s in targets:
        ok = servers.stop(s)
        console.print(
            ("[green]stopped[/]" if ok else "[red]failed to stop[/]")
            + f" :{s.port} {s.model.split('/')[-1]} (pid {s.pid})"
        )

    stray = servers.unmanaged()
    if stray:
        console.print(
            f"[yellow]{len(stray)} unmanaged mlx_lm.server process(es)[/]: "
            f"{', '.join(map(str, stray))}"
        )
        if Confirm.ask("kill them too", default=False):
            for pid in stray:
                subprocess.run(["kill", str(pid)])
            console.print("sent SIGTERM")

    manifest.write()


@app.command()
def ps():
    """Show running servers."""
    live = servers.list_servers()
    if not live:
        console.print("no servers running  [dim](llmctl start)[/]")
        stray = servers.unmanaged()
        if stray:
            console.print(
                f"[yellow]but {len(stray)} unmanaged mlx_lm.server "
                f"process(es) exist:[/] {', '.join(map(str, stray))}"
            )
        raise typer.Exit(0)

    t = Table(box=None, pad_edge=False)
    t.add_column("port", justify="right")
    t.add_column("model")
    t.add_column("ctx", justify="right")
    t.add_column("pid", justify="right")
    t.add_column("up", justify="right")
    t.add_column("health")
    t.add_column("endpoint")
    for s in live:
        t.add_row(
            str(s.port),
            s.model.replace("mlx-community/", ""),
            f"{s.context // 1024}k" if s.context else "—",
            str(s.pid),
            s.uptime(),
            "[green]ok[/]" if s.responds() else "[yellow]loading/stuck[/]",
            s.endpoint,
        )
    console.print(t)


@app.command()
def monitor(
    watch: bool = typer.Option(False, "--watch", "-w"),
    interval: float = typer.Option(2.0, "--interval", "-i"),
):
    """GPU utilization and real memory use (RSS is meaningless for MLX)."""
    gpu.show(console, servers.list_servers(), watch=watch, interval=interval)


@app.command()
def pull(
    model: str | None = typer.Argument(None, help="Repo id. Omit to pick from the catalog."),
):
    """Download a model into the local cache so it works offline."""
    if model:
        repo = model
        m = catalog.find(model)
        if m:
            repo = m.repo_id
    else:
        uncached = [
            m for m in catalog.load(include_uncached=True) if not m.cached and m.runtime == "mlx-lm"
        ]
        if not uncached:
            console.print("every catalog model is already downloaded")
            raise typer.Exit(0)
        h = hostmod.detect()
        t = Table(box=None, pad_edge=False)
        t.add_column("#", justify="right", style="bold cyan")
        t.add_column("model")
        t.add_column("params")
        t.add_column("size", justify="right")
        t.add_column("fit")
        t.add_column("role")
        for i, mm in enumerate(uncached, 1):
            t.add_row(
                str(i),
                mm.repo_id.replace("mlx-community/", ""),
                mm.params or "—",
                f"{mm.size_gb:.1f}G" if mm.size_gb else "—",
                _fit(mm, h.budget_gb),
                mm.role or "—",
            )
        console.print(t)
        idx = IntPrompt.ask(
            "pull which model",
            choices=[str(i) for i in range(1, len(uncached) + 1)],
            show_choices=False,
        )
        repo = uncached[idx - 1].repo_id

    console.print(f"downloading [bold]{repo}[/] — resumable, safe to interrupt")
    from .paths import env_python

    rc = subprocess.call(
        [str(env_python()), "-m", "huggingface_hub.commands.huggingface_cli", "download", repo]
    )
    if rc != 0:
        rc = subprocess.call(["hf", "download", repo])
    if rc == 0:
        manifest.write()
        console.print("[green]done[/]")
    else:
        console.print("[red]download failed[/]")
        raise typer.Exit(rc)


@app.command()
def cache(
    prune: bool = typer.Option(False, "--prune", help="Interactively delete a cached model."),
):
    """Show or prune the weight cache."""
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()
    repos = sorted([r for r in info.repos if r.repo_type == "model"], key=lambda r: -r.size_on_disk)
    t = Table(
        title=f"HF cache: {info.size_on_disk / 1e9:.1f} GB across {len(repos)} models",
        title_justify="left",
        box=None,
        pad_edge=False,
    )
    t.add_column("#", justify="right", style="bold cyan")
    t.add_column("model")
    t.add_column("size", justify="right")
    t.add_column("files", justify="right")
    t.add_column("last used")
    for i, r in enumerate(repos, 1):
        t.add_row(
            str(i),
            r.repo_id,
            f"{r.size_on_disk / 1e9:.1f}G",
            str(r.nb_files),
            str(r.last_accessed_str),
        )
    console.print(t)

    orphans = list(LOG_DIR.glob("*.log")) if LOG_DIR.exists() else []
    console.print(f"[dim]state dir {RUN_DIR} · {len(orphans)} log file(s)[/]")

    if not prune:
        console.print("[dim]--prune to delete a model[/]")
        return

    idx = IntPrompt.ask(
        "delete which model (0 to cancel)",
        choices=[str(i) for i in range(0, len(repos) + 1)],
        show_choices=False,
    )
    if idx == 0:
        return
    target = repos[idx - 1]
    running = [s for s in servers.list_servers() if s.model == target.repo_id]
    if running:
        console.print(
            f"[red]refusing[/]: {target.repo_id} is being served on "
            f"{', '.join(str(s.port) for s in running)} — stop it first"
        )
        raise typer.Exit(1)
    if not Confirm.ask(
        f"delete {target.repo_id} ({target.size_on_disk / 1e9:.1f} GB)", default=False
    ):
        return
    strategy = info.delete_revisions(*[rev.commit_hash for rev in target.revisions])
    console.print(f"freeing {strategy.expected_freed_size_str}")
    strategy.execute()
    manifest.write()
    console.print("[green]deleted[/]")


@app.command()
def verify(
    model: str | None = typer.Argument(None, help="Only this model."),
):
    """Check cached weights are complete and note each model's tool parser."""
    import glob
    import struct
    from pathlib import Path as P

    from huggingface_hub import scan_cache_dir

    bad = 0
    t = Table(box=None, pad_edge=False)
    t.add_column("model")
    t.add_column("shards", justify="right")
    t.add_column("tool parser")
    t.add_column("status")

    for repo in sorted(scan_cache_dir().repos, key=lambda r: r.repo_id):
        if repo.repo_type != "model":
            continue
        if model and model not in repo.repo_id:
            continue
        snaps = sorted(glob.glob(str(repo.repo_path) + "/snapshots/*/"))
        if not snaps:
            t.add_row(repo.repo_id, "0", "—", "[red]no snapshot[/]")
            bad += 1
            continue
        snap = snaps[-1]
        shards = sorted(glob.glob(snap + "*.safetensors"))
        idx = glob.glob(snap + "model.safetensors.index.json")
        m = catalog.scan_cache().get(repo.repo_id)
        parser = (m.tool_parser if m else None) or "[dim]none[/]"

        if not shards:
            t.add_row(repo.repo_id, "0", parser, "[yellow]no safetensors[/]")
            continue
        if not idx:
            t.add_row(repo.repo_id, str(len(shards)), parser, "[green]ok (no index)[/]")
            continue
        try:
            exp = json.loads(P(idx[0]).read_text())["metadata"]["total_size"]
            actual = sum(P(f).resolve().stat().st_size for f in shards)
            hdr = 0
            for f in shards:
                with open(P(f).resolve(), "rb") as fh:
                    hdr += 8 + struct.unpack("<Q", fh.read(8))[0]
            # the index counts tensor bytes only; headers are extra
            if actual - hdr == exp:
                t.add_row(repo.repo_id, str(len(shards)), parser, "[green]ok[/]")
            else:
                short = (exp - (actual - hdr)) / 1e9
                t.add_row(
                    repo.repo_id,
                    str(len(shards)),
                    parser,
                    f"[red]incomplete (short {short:.2f} GB)[/]",
                )
                bad += 1
        except FileNotFoundError:
            t.add_row(repo.repo_id, str(len(shards)), parser, "[red]broken symlink[/]")
            bad += 1

    console.print(t)
    partials = list(
        __import__("glob").glob(
            str(__import__("pathlib").Path.home() / ".cache/huggingface/hub/**/*.incomplete"),
            recursive=True,
        )
    )
    if partials:
        mb = sum(P(p).stat().st_size for p in partials) / 1e6
        console.print(f"[yellow]{len(partials)} orphaned partial download(s), {mb:.0f} MB[/]")
    if bad:
        console.print("[red]re-run llmctl pull for the incomplete models[/]")
        raise typer.Exit(1)
    console.print("[green]all cached models verified complete[/]")


@app.command(name="opencode")
def opencode_cmd(
    install: bool = typer.Option(False, "--install", help="Install the discovery plugin."),
):
    """Show or install the opencode integration."""
    from pathlib import Path as P

    plugin_src = P(__file__).resolve().parent.parent / "opencode/plugins/dekho-local-inference.js"
    dest = P.home() / ".config/opencode/plugins/dekho-local-inference.js"

    if install:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(plugin_src)
        console.print(f"[green]installed[/] {dest} → {plugin_src}")
        manifest.write()
    else:
        console.print(
            f"plugin   {'[green]installed[/]' if dest.exists() else '[red]not installed[/]'} "
            f"({dest})"
        )
        console.print(
            f"manifest {'[green]present[/]' if MANIFEST.exists() else '[red]missing[/]'} "
            f"({MANIFEST})"
        )
        if MANIFEST.exists():
            d = json.loads(MANIFEST.read_text())
            for s in d.get("servers", []):
                console.print(
                    f"  provider {s['provider_id']:<32} {s['endpoint']}  "
                    f"{s['model'].split('/')[-1]}"
                )
            console.print(f"  {len(d.get('models', {}))} model(s) offered")
        console.print("[dim]--install to (re)install the plugin[/]")


if __name__ == "__main__":
    app()
