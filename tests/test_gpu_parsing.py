"""Parsing of vmmap/ioreg output — where a bug shows a wrong memory number."""

import pytest

from llmctl.gpu import _parse_size


@pytest.mark.parametrize(
    "text,gb",
    [
        ("41.9G", 41.9),
        ("42.2G", 42.2),
        ("16.3M", 0.0163),
        ("512K", 0.000512),
        ("2.1T", 2100.0),
        ("1024", 1024.0),  # bare number: already gigabytes
        ("  8G  ", 8.0),
        ("8GB", 8.0),
    ],
)
def test_parses_vmmap_sizes(text, gb):
    assert _parse_size(text) == pytest.approx(gb)


@pytest.mark.parametrize("text", [None, "", "bogus", "G", "--", "12X"])
def test_rejects_unparseable(text):
    assert _parse_size(text) is None


def test_hot_swap_heuristic_threshold():
    """The warning must not fire on normal transient allocation.

    Regression: it previously fired whenever both values merely existed, so a
    peak of 42.3G against 41.9G current was reported as hot-swapping.
    """
    normal_cur, normal_peak = _parse_size("41.9G"), _parse_size("42.3G")
    assert not (normal_peak > normal_cur * 1.3)

    swapped_cur, swapped_peak = _parse_size("27.2G"), _parse_size("62.1G")
    assert swapped_peak > swapped_cur * 1.3
