"""Per-env CRL hyperparameters from the OGBench paper, resolved from the env name."""

import dataclasses

# Dataset-type token of an OGBench env name (``antmaze-large-navigate-v0`` -> ``navigate``).
DATASET_TYPES = frozenset({"navigate", "stitch", "explore", "play", "noisy"})

# alpha (DDPG+BC coefficient) keyed by (domain, dataset type); unknown combinations raise.
_ALPHA = {
    ("pointmaze", "navigate"): 0.03,
    ("pointmaze", "stitch"): 0.03,
    ("antmaze", "navigate"): 0.1,
    ("antmaze", "stitch"): 0.1,
    ("antmaze", "explore"): 0.003,
    ("humanoidmaze", "navigate"): 0.1,
    ("humanoidmaze", "stitch"): 0.1,
    ("antsoccer", "navigate"): 0.3,
    ("antsoccer", "stitch"): 0.3,
    ("cube", "play"): 3.0,
    ("cube", "noisy"): 0.1,
    ("scene", "play"): 3.0,
    ("scene", "noisy"): 0.1,
    ("puzzle", "play"): 3.0,
    ("puzzle", "noisy"): 0.1,
}

SUPPORTED_DOMAINS = ("pointmaze", "antmaze", "humanoidmaze", "antsoccer", "cube", "scene", "puzzle")

# Actor goal-sampling ratios (p_trajgoal, p_randomgoal) by dataset type; default (1.0, 0.0).
_GOAL_RATIOS = {
    "stitch": (0.5, 0.5),
    "explore": (0.0, 1.0),
}

# Fields this table owns — the ones declared `float | None` on the CRL-family configs.
RESOLVED_FIELDS = ("alpha", "discount", "actor_p_trajgoal", "actor_p_randomgoal")


def crl_env_defaults(env_name: str) -> dict:
    """OGBench CRL hyperparameters for ``env_name``; raises for unsupported domains or dataset types."""
    if env_name.startswith("visual-"):
        raise ValueError(f"{env_name!r}: pixel-based environments are not supported.")
    tokens = env_name.split("-")
    domain = tokens[0]

    if domain == "powderworld":
        raise ValueError(
            f"{env_name!r}: the OGBench paper runs CRL on powderworld with AWR (alpha 3.0) on a "
            "discrete action space, which this offline port does not support. Pass --alpha/--discount/"
            "--actor_p_trajgoal/--actor_p_randomgoal explicitly if you mean to run it anyway."
        )

    dataset_type = next((t for t in tokens[1:] if t in DATASET_TYPES), None)
    if dataset_type is None:
        raise ValueError(
            f"{env_name!r}: no recognizable OGBench dataset type in the name (expected one of "
            f"{sorted(DATASET_TYPES)}). Supported domains: {list(SUPPORTED_DOMAINS)}."
        )
    # Size token, if any (``antmaze-giant-navigate-v0`` -> ``giant``).
    size = tokens[1] if len(tokens) > 1 and tokens[1] != dataset_type else None

    if (domain, dataset_type) not in _ALPHA:
        raise ValueError(
            f"{env_name!r}: no CRL hyperparameters in the OGBench table for domain {domain!r} with "
            f"dataset type {dataset_type!r}. Supported domains: {list(SUPPORTED_DOMAINS)}."
        )

    discount = 0.995 if domain == "humanoidmaze" or (domain in ("antmaze", "pointmaze") and size == "giant") else 0.99
    p_trajgoal, p_randomgoal = _GOAL_RATIOS.get(dataset_type, (1.0, 0.0))
    return {
        "alpha": _ALPHA[(domain, dataset_type)],
        "discount": discount,
        "actor_p_trajgoal": p_trajgoal,
        "actor_p_randomgoal": p_randomgoal,
    }


def _is_crl_family(cfg) -> bool:
    """True for CRL-family configs (duck-typed on the ``alpha`` field to avoid an import cycle)."""
    fields = getattr(cfg, "__dataclass_fields__", {})
    return "alpha" in fields and "low_alpha" not in fields


def resolve_env_defaults(cfg, env_name: str):
    """Fill ``cfg``'s unset per-env fields from the tables; returns ``(resolved_cfg, filled)``."""
    if not _is_crl_family(cfg):
        return cfg, {}
    filled = {}
    missing = [f for f in RESOLVED_FIELDS if getattr(cfg, f, None) is None]
    if missing:
        defaults = crl_env_defaults(env_name)
        filled.update({f: defaults[f] for f in missing})
    if not filled:
        return cfg, {}
    return dataclasses.replace(cfg, **filled), filled
