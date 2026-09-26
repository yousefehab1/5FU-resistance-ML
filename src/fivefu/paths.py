"""Where things live.

Scripts take ROOT from project_root(). Set FIVEFU_ROOT to point a whole run at
another copy of the data without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/fivefu/paths.py -> src/fivefu -> src -> repository root
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """The repository root, honouring FIVEFU_ROOT if it is set.

    A FIVEFU_ROOT that does not exist raises rather than falling back. Falling
    back would silently write results into the wrong tree, which is the class of
    failure this whole package is built to avoid.
    """
    override = os.environ.get("FIVEFU_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(
                f"FIVEFU_ROOT is set to {override!r}, which is not a directory. "
                f"Refusing to fall back to {_PACKAGE_ROOT}: that would write "
                f"results into a tree you did not ask for."
            )
        return root
    return _PACKAGE_ROOT


def raw_dir() -> Path:
    return project_root() / "data" / "raw"


def processed_dir() -> Path:
    return project_root() / "data" / "processed"


def models_dir() -> Path:
    return project_root() / "models"


def figures_dir() -> Path:
    return project_root() / "figures"


def golden_dir() -> Path:
    return project_root() / "tests" / "golden"


def config_dir() -> Path:
    return project_root() / "config"
