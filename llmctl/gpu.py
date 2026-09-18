"""GPU and memory telemetry.

`ps`/`htop` cannot show MLX memory: weights live in Metal buffers that macOS
does not count in RSS, so a server can report 16 GB RSS while holding 45 GB.
The numbers that are real come from:

  ioreg -c AGXAccelerator   GPU utilization, in-use / allocated memory (no sudo)
  vmmap --summary <pid>     IOAccelerator regions, footprint, footprint peak
  vm_stat                   pageouts, i.e. whether we are over the ceiling
"""

import re
import subprocess
import time


def agx() -> dict[str, int]:
    """Driver-level GPU stats. Empty dict if unavailable."""
    try:
        out = subprocess.run(
            ["ioreg", "-l", "-w", "0", "-r", "-c", "AGXAccelerator"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        return {}
    stats = {}
    for key in (
        "Device Utilization %",
        "Renderer Utilization %",
        "In use system memory",
        "Alloc system memory",
    ):
        m = re.search(rf'"{re.escape(key)}"=(\d+)', out)
        if m:
            stats[key] = int(m.group(1))
    return stats


def proc_memory(pid: int) -> dict[str, str]:
    """Metal buffer residency and footprint for one process."""
    try:
        out = subprocess.run(
            ["vmmap", "--summary", str(pid)], capture_output=True, text=True, timeout=25
        ).stdout
    except Exception:
        return {}
    res = {}
    for line in out.splitlines():
        if line.startswith("Physical footprint:"):
            res["footprint"] = line.split(":", 1)[1].strip()
        elif line.startswith("Physical footprint (peak):"):
            res["peak"] = line.split(":", 1)[1].strip()
        elif "IOAccelerator (graphics)" in line:
            parts = line.split()
            if len(parts) >= 3:
                res["metal"] = parts[2]
    return res


def _parse_size(s: str | None) -> float | None:
    """'41.9G' / '16.3M' / '512K' -> gigabytes."""
    if not s:
        return None
    m = re.fullmatch(r"([\d.]+)\s*([KMGT])?B?", s.strip(), re.I)
    if not m:
        return None
    mult = {"k": 1e-6, "m": 1e-3, "g": 1.0, "t": 1e3}
    return float(m.group(1)) * mult.get((m.group(2) or "g").lower(), 1.0)


def pressure() -> dict[str, float]:
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return {}
    page = 16384
    vals = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip().rstrip(".")
        if not v.isdigit():
            continue
        n = int(v)
        if "Pages free" in k:
            vals["free_gb"] = n * page / 1e9
        elif "Pages active" in k:
            vals["active_gb"] = n * page / 1e9
        elif "Pages wired" in k:
            vals["wired_gb"] = n * page / 1e9
        elif k.strip() == "Pageouts":
            vals["pageouts"] = n
    return vals


def _render(console, servers_list):
    from rich.table import Table

    s = agx()
    t = Table(box=None, pad_edge=False, show_header=False, title="GPU", title_justify="left")
    if s:
        t.add_row("utilization", f"{s.get('Device Utilization %', '?')}%")
        if "In use system memory" in s:
            t.add_row("in use", f"{s['In use system memory'] / 1e9:.2f} GB")
        if "Alloc system memory" in s:
            t.add_row("allocated", f"{s['Alloc system memory'] / 1e9:.2f} GB")
    else:
        t.add_row("", "[yellow]AGXAccelerator not readable[/]")
    console.print(t)

    if not servers_list:
        console.print("\n[dim]no managed servers running[/]")
    for srv in servers_list:
        mem = proc_memory(srv.pid)
        st = Table(
            box=None,
            pad_edge=False,
            show_header=False,
            title=f":{srv.port}  {srv.model.split('/')[-1]}",
            title_justify="left",
        )
        st.add_row("pid / up", f"{srv.pid} · {srv.uptime()}")
        st.add_row("Metal buffers", f"{mem.get('metal', '?')}   [dim]the real residency[/]")
        st.add_row("footprint", f"{mem.get('footprint', '?')}  (peak {mem.get('peak', '?')})")
        # Only warn when the peak is genuinely far above current residency:
        # that gap is retained/fragmented buffers from model swapping, which
        # costs ~3x throughput until restart. A small gap is just normal
        # transient allocation.
        cur_gb, peak_gb = _parse_size(mem.get("footprint")), _parse_size(mem.get("peak"))
        if cur_gb and peak_gb and peak_gb > cur_gb * 1.3:
            st.add_row(
                "",
                f"[yellow]peak {peak_gb:.1f}G vs current {cur_gb:.1f}G[/] "
                "[dim]⇒ has been hot-swapping; restart recovers throughput[/]",
            )
        st.add_row(
            "health", "[green]responding[/]" if srv.responds() else "[yellow]not answering[/]"
        )
        console.print(st)

    p = pressure()
    if p:
        pt = Table(
            box=None, pad_edge=False, show_header=False, title="memory", title_justify="left"
        )
        pt.add_row("wired", f"{p.get('wired_gb', 0):.1f} GB")
        pt.add_row("active", f"{p.get('active_gb', 0):.1f} GB")
        pt.add_row("free", f"{p.get('free_gb', 0):.1f} GB")
        po = p.get("pageouts", 0)
        pt.add_row(
            "pageouts",
            f"{po:.0f}" + ("  [yellow]climbing ⇒ over the ceiling[/]" if po > 100000 else ""),
        )
        console.print(pt)


def show(console, servers_list, watch: bool = False, interval: float = 2.0):
    if not watch:
        _render(console, servers_list)
        return
    try:
        while True:
            console.clear()
            console.print(f"[dim]{time.strftime('%H:%M:%S')} · refresh {interval}s · Ctrl+C[/]")
            _render(console, servers_list)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
