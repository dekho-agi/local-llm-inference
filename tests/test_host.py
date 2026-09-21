"""Profile autoselection — maps a machine's memory to a target directory.

The mapping is what makes `llmctl host` correct on a new box without config,
so it is worth pinning. Uses the real directory names in the repo.
"""

import pytest

from llmctl.host import Host, _pick_profile


@pytest.mark.parametrize(
    "memory_gb,expected,exact",
    [
        (128, "m5-128gb", True),  # the M5 Max
        (16, "m2-16gb", True),  # the M2 Air
        (96, "m5-128gb", False),  # nearest, and flagged inexact
        (36, "m2-16gb", False),  # nearest, and flagged inexact
        (8, "m2-16gb", False),  # below every profile
    ],
)
def test_profile_selection(memory_gb, expected, exact):
    """Nearest profile by memory, with an honest exact flag.

    Regression: this test previously asserted that 64 GB should map to the
    16 GB profile, describing "largest that fits" as correct. It is not — a
    96 GB machine was handed a MacBook Air's 3-model list and ~10 GB budget.
    The test encoded the bug, so CI would have blocked the fix.
    """
    profile, path, is_exact = _pick_profile(memory_gb)
    assert profile == expected
    assert path and path.endswith(expected)
    assert is_exact is exact


def test_inexact_profile_is_flagged_not_silent():
    """A machine between profiles must be told the list is a guess."""
    _, _, exact = _pick_profile(64)
    assert exact is False


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
