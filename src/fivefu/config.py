"""Analysis configuration, loaded from config/*.yaml.

Every lookup raises ConfigError when a key is missing; there are no defaults.
A default would be a hidden second copy of the value, so an edit to the YAML
could silently stop having any effect. Only values something actually reads
belong in the config.
"""

from __future__ import annotations

from typing import Any

import yaml

from .paths import config_dir


class ConfigError(KeyError):
    """A config key was missing, or a config file was."""


def _load_file(name: str) -> dict:
    path = config_dir() / name
    if not path.exists():
        raise ConfigError(
            f"Missing config file {path}.\n"
            f"  config/ ships with the repository; it is not generated. If this\n"
            f"  is a fresh checkout the file should be there, so this usually\n"
            f"  means FIVEFU_ROOT points somewhere unintended: {config_dir().parent}"
        )
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} did not parse to a mapping, got {type(loaded).__name__}")
    return loaded


def require(mapping: dict, *keys: str, source: str) -> Any:
    """
    Walk `keys` through nested mappings. Raise on the first one missing.

    Deliberately has no default parameter. See this module's docstring.
    """
    here: Any = mapping
    for depth, key in enumerate(keys):
        if not isinstance(here, dict):
            walked = ".".join(keys[:depth]) or "<root>"
            raise ConfigError(
                f"{source}: cannot look up {key!r} under {walked}, "
                f"which is a {type(here).__name__}, not a mapping"
            )
        if key not in here:
            walked = ".".join(keys[:depth]) or "<root>"
            raise ConfigError(
                f"{source}: no key {'.'.join(keys)!r}. "
                f"{walked} contains {sorted(here)}"
            )
        here = here[key]
    return here


def _reader(name: str):
    """A `require` bound to one config file, so its name is written once and a
    missing-key error always names the file it was actually looked up in."""
    mapping = _load_file(name)

    def read(*keys: str) -> Any:
        return require(mapping, *keys, source=f"config/{name}")

    return read


_analysis = _reader("analysis.yaml")
_cohorts = _reader("cohorts.yaml")
_drugs = _reader("drugs.yaml")
_signatures = _reader("signatures.yaml")


# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

SEED: int = _analysis("seed")
K_GENES: int = _analysis("feature_selection", "k_genes")
N_REPEATS: int = _analysis("cross_validation", "n_repeats")
N_FOLDS: int = _analysis("cross_validation", "n_folds")

# Shared ElasticNet settings. Passed as **ELASTICNET into the estimator by
# fivefu.modeling.elasticnet_pipeline, so adding a key here reaches every
# pipeline at once. 06 and 07 deliberately do not use these.
ELASTICNET: dict = dict(_analysis("elasticnet"))

MAX_GENE_MISSING: float = _analysis("filters", "max_gene_missing")
MIN_LINEAGE_N: int = _analysis("filters", "min_lineage_n")
MIN_DRUG_LINES: int = _analysis("filters", "min_drug_lines")

CEILING_AUC: float = _analysis("thresholds", "ceiling_auc")
CEILING_R: dict = dict(_analysis("thresholds", "ceiling_r"))
MIN_N_TASK: int = _analysis("thresholds", "min_n_task")
MIN_N_STOP: int = _analysis("thresholds", "min_n_stop")
OVERLAP_MIN: int = _analysis("thresholds", "overlap_min")

# ============================================================================
# SCOPE
# ============================================================================

# A set, because every call site tests membership. The scripts all wrote it as
# a set literal; YAML has no set type.
HAEM: frozenset = frozenset(
    _cohorts("haematological_tcga_codes")
)

PRIMARY_DRUG: str = _drugs("primary")
DRUGS: list = list(_drugs("focal"))

# The three-way split. `MODELED_MODULES` is what may be fed to a model;
# `ALL_MODULES` is display only. Using the wrong one changes the feature
# matrix, so they are named to make the wrong one look wrong at the call site.
MODELED_MODULES: list = list(
    _signatures("modelled")
)
REFERENCE_MODULES: list = list(
    _signatures("reference")
)

# Read, not derived. `MODELED_MODULES + REFERENCE_MODULES` puts CellCycle last,
# but 11 and app.py put it fourth, and this list becomes dashboard column
# order. The membership check below is what stops the two halves drifting apart
# now that the concatenation no longer enforces it.
ALL_MODULES: list = list(_signatures("all"))

if set(ALL_MODULES) != set(MODELED_MODULES) | set(REFERENCE_MODULES):
    raise ConfigError(
        f"config/signatures.yaml: `all` holds {sorted(ALL_MODULES)}, but "
        f"`modelled` + `reference` is "
        f"{sorted(set(MODELED_MODULES) | set(REFERENCE_MODULES))}. "
        f"`all` may reorder them; it may not add or drop one."
    )


def describe() -> str:
    """One-screen dump of the live configuration, for run banners."""
    return "\n".join([
        f"  seed={SEED}  k_genes={K_GENES}  cv={N_REPEATS}x{N_FOLDS}",
        f"  elasticnet={ELASTICNET}",
        f"  max_gene_missing={MAX_GENE_MISSING}  min_lineage_n={MIN_LINEAGE_N}"
        f"  min_drug_lines={MIN_DRUG_LINES}",
        f"  ceiling_auc={CEILING_AUC}  ceiling_r={CEILING_R}",
        f"  modelled={MODELED_MODULES}  reference={REFERENCE_MODULES}",
    ])
