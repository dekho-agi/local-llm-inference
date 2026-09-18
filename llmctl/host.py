"""Host detection: what machine is this, and what can it hold?

The number that matters on Apple silicon is the GPU's recommended working set,
not total RAM. mlx-lm raises the wired limit to it at startup, and weights plus
KV cache plus prompt cache must fit under it.
"""

import platform
import subprocess
from dataclasses import dataclass

from .paths import APPLE_DIR


@dataclass
class Host:
    chip: str
    model: str
    cpu_cores: int
    gpu_cores: int
    memory_gb: float
    working_set_gb: float | None
    macos: str
    profile: str  # which target directory applies
    profile_dir: str | None

    @property
    def budget_gb(self) -> float:
        """What to actually plan against, leaving headroom for the OS."""
        if self.working_set_gb:
            return round(self.working_set_gb - 10, 1)
        return round(self.memory_gb * 0.7, 1)


def _sp(key: str, data: str) -> str:
    for line in data.splitlines():
        if key in line:
            return line.split(":", 1)[1].strip()
    return ""


def detect() -> Host:
    hw = subprocess.run(
        ["system_profiler", "SPHardwareDataType"], capture_output=True, text=True
    ).stdout
    disp = subprocess.run(
        ["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True
    ).stdout

    chip = _sp("Chip", hw) or platform.processor()
    model = _sp("Model Name", hw)
    cores_raw = _sp("Total Number of Cores", hw)
    cpu_cores = int(cores_raw.split()[0]) if cores_raw and cores_raw.split()[0].isdigit() else 0
    mem_raw = _sp("Memory", hw)
    memory_gb = float(mem_raw.split()[0]) if mem_raw and mem_raw.split()[0].isdigit() else 0.0

    gpu_cores = 0
    for line in disp.splitlines():
        if "Total Number of Cores" in line:
            v = line.split(":", 1)[1].strip()
            if v.isdigit():
                gpu_cores = int(v)
            break

    working_set = None
    try:
        import mlx.core as mx

        v = mx.device_info().get("max_recommended_working_set_size")
        if v:
            working_set = round(v / 1e9, 1)
    except Exception:
        pass

    macos = subprocess.run(
        ["sw_vers", "-productVersion"], capture_output=True, text=True
    ).stdout.strip()

    profile, profile_dir = _pick_profile(memory_gb)
    return Host(
        chip=chip,
        model=model,
        cpu_cores=cpu_cores,
        gpu_cores=gpu_cores,
        memory_gb=memory_gb,
        working_set_gb=working_set,
        macos=macos,
        profile=profile,
        profile_dir=profile_dir,
    )


def _pick_profile(memory_gb: float) -> tuple[str, str | None]:
    """Map memory to a target directory, preferring an exact-ish match."""
    if not APPLE_DIR.exists():
        return ("unknown", None)
    candidates = []
    for d in sorted(APPLE_DIR.iterdir()):
        if not d.is_dir() or d.name == "common":
            continue
        # directory names encode their memory, e.g. m5-128gb, m2-16gb
        for part in d.name.split("-"):
            if part.endswith("gb") and part[:-2].isdigit():
                candidates.append((int(part[:-2]), d.name))
                break
    if not candidates:
        return ("unknown", None)
    # Largest profile this machine can satisfy. If it satisfies none, fall
    # back to the smallest rather than the largest — a 8 GB machine must not
    # be handed the 128 GB profile's model list.
    fits = [c for c in candidates if c[0] <= memory_gb + 1]
    chosen = max(fits, key=lambda c: c[0]) if fits else min(candidates, key=lambda c: c[0])
    return (chosen[1], str(APPLE_DIR / chosen[1]))
