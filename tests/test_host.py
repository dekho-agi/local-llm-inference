"""Profile autoselection — maps a machine's memory to a target directory.

The mapping is what makes `llmctl host` correct on a new box without config,
so it is worth pinning. Uses the real directory names in the repo.
"""

import pytest

from llmctl.host import Host, _pick_profile


@pytest.mark.parametrize(
    "memory_gb,expected",
    [
        (128, "m5-128gb"),  # the M5 Max
        (16, "m2-16gb"),  # the M2 Air
        (64, "m2-16gb"),  # between profiles: largest that fits
        (8, "m2-16gb"),  # below every profile: fall back to smallest
    ],
)
def test_profile_selection(memory_gb, expected):
    profile, path = _pick_profile(memory_gb)
    assert profile == expected
    assert path and path.endswith(expected)


def test_budget_prefers_the_gpu_working_set_over_total_ram():
    """The ceiling is the GPU working set, not installed memory."""
    h = Host(
        chip="Apple M5 Max",
        model="MacBook Pro",
        cpu_cores=18,
        gpu_cores=40,
        memory_gb=128.0,
        working_set_gb=115.4,
        macos="26.5.1",
        profile="m5-128gb",
        profile_dir=None,
    )
    assert h.budget_gb == 105.4  # working set less OS headroom
    assert h.budget_gb < h.memory_gb


def test_budget_falls_back_when_the_working_set_is_unknown():
    h = Host(
        chip="?",
        model="?",
        cpu_cores=8,
        gpu_cores=10,
        memory_gb=16.0,
        working_set_gb=None,
        macos="26.0",
        profile="m2-16gb",
        profile_dir=None,
    )
    assert h.budget_gb == pytest.approx(11.2)
