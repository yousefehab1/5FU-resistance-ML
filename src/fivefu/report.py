"""Console output helpers."""

from __future__ import annotations

WIDTH = 78


def banner(text: str) -> None:
    """A section header, so a long run is readable when you scroll back."""
    print("\n" + "=" * WIDTH + f"\n{text}\n" + "=" * WIDTH)

