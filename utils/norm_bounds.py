"""Strict loading/persisting of ETH/UCY min-max normalization bounds.

The visualization and agent-crop pipelines run in a separate process from
training, so they must recover the *exact* training bounds.  There is no
implicit ``[-1, 1]`` fallback: missing artifacts raise :class:`NormBoundsError`.

Artifacts (searched in order inside a run directory):
    1. ``norm_bounds.json`` (preferred, written atomically by the dataset)
    2. ``cor_fm_updated.yml`` / ``cor_fm.yml`` / ``config.yml`` containing the
       four ``*_traj_min/max`` keys.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

REQUIRED_KEYS: tuple[str, ...] = (
    "past_traj_min",
    "past_traj_max",
    "fut_traj_min",
    "fut_traj_max",
)

_SCHEMA = "eth_ucy_norm_bounds_v1"
_YAML_CANDIDATES = ("cor_fm_updated.yml", "cor_fm.yml", "config.yml")
_JSON_NAME = "norm_bounds.json"


class NormBoundsError(RuntimeError):
    """Raised when required normalization bounds cannot be loaded strictly."""


@dataclass(frozen=True)
class NormBounds:
    past_min: float
    past_max: float
    fut_min: float
    fut_max: float

    def __post_init__(self) -> None:
        for name in ("past_min", "past_max", "fut_min", "fut_max"):
            value = float(getattr(self, name))
            if not (value == value) or value in (float("inf"), float("-inf")):
                raise NormBoundsError(f"Non-finite normalization bound {name}={value}")
            object.__setattr__(self, name, value)
        if self.past_max <= self.past_min:
            raise NormBoundsError(
                f"past bounds must be increasing, got ({self.past_min}, {self.past_max})"
            )
        if self.fut_max <= self.fut_min:
            raise NormBoundsError(
                f"future bounds must be increasing, got ({self.fut_min}, {self.fut_max})"
            )

    def as_dict(self) -> dict[str, float]:
        return {
            "past_traj_min": self.past_min,
            "past_traj_max": self.past_max,
            "fut_traj_min": self.fut_min,
            "fut_traj_max": self.fut_max,
        }


def bounds_from_mapping(mapping: Mapping[str, Any], *, source: str = "<mapping>") -> NormBounds:
    missing = [k for k in REQUIRED_KEYS if k not in mapping or mapping[k] is None]
    if missing:
        raise NormBoundsError(
            f"Missing normalization bounds {missing} in {source}. "
            "Refusing to fall back to [-1, 1]; regenerate the training artifact."
        )
    return NormBounds(
        past_min=float(mapping["past_traj_min"]),
        past_max=float(mapping["past_traj_max"]),
        fut_min=float(mapping["fut_traj_min"]),
        fut_max=float(mapping["fut_traj_max"]),
    )


def bounds_from_cfg(cfg: Any, *, source: str = "<cfg>") -> NormBounds:
    """Extract bounds from an in-memory config; never inject defaults."""
    missing = [k for k in REQUIRED_KEYS if not hasattr(cfg, k)]
    if missing:
        raise NormBoundsError(
            f"Config {source} is missing normalization bounds {missing}. "
            "Training must persist them; refusing to default to [-1, 1]."
        )
    return NormBounds(
        past_min=float(getattr(cfg, "past_traj_min")),
        past_max=float(getattr(cfg, "past_traj_max")),
        fut_min=float(getattr(cfg, "fut_traj_min")),
        fut_max=float(getattr(cfg, "fut_traj_max")),
    )


def apply_bounds_to_cfg(cfg: Any, bounds: NormBounds) -> None:
    for key, value in bounds.as_dict().items():
        setattr(cfg, key, value)


def save_norm_bounds(run_dir: str | Path, bounds: NormBounds) -> Path:
    """Atomically write ``norm_bounds.json`` into ``run_dir``."""
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    payload = {"schema": _SCHEMA, **bounds.as_dict()}
    target = run / _JSON_NAME
    tmp = run / f".{_JSON_NAME}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)
    return target


def load_norm_bounds_from_run(run_dir: str | Path) -> NormBounds:
    """Strictly load bounds from a training run directory.

    Raises :class:`NormBoundsError` when no artifact contains the bounds.
    """
    run = Path(run_dir)
    if not run.is_dir():
        raise NormBoundsError(f"Run directory does not exist: {run}")

    json_path = run / _JSON_NAME
    if json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise NormBoundsError(f"Invalid {json_path}: expected a JSON object")
        if payload.get("schema") not in (None, _SCHEMA):
            raise NormBoundsError(
                f"Unsupported norm-bounds schema {payload.get('schema')!r} in {json_path}"
            )
        return bounds_from_mapping(payload, source=str(json_path))

    for name in _YAML_CANDIDATES:
        path = run / name
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        if not isinstance(payload, dict):
            continue
        if all(k in payload and payload[k] is not None for k in REQUIRED_KEYS):
            return bounds_from_mapping(payload, source=str(path))

    raise NormBoundsError(
        f"No normalization bounds found in run directory {run}. "
        f"Expected {_JSON_NAME} or {list(_YAML_CANDIDATES)} with keys {list(REQUIRED_KEYS)}. "
        "Refusing to fall back to [-1, 1]."
    )


def maybe_persist_bounds_from_cfg(cfg: Any, *, run_dir: str | Path | None = None) -> Path | None:
    """Persist bounds when a run/config directory is available.

    Called after training-time bound computation.  Returns the artifact path or
    ``None`` when there is nothing to persist yet.
    """
    target_dir = run_dir if run_dir is not None else getattr(cfg, "cfg_dir", None)
    if target_dir is None or not all(hasattr(cfg, k) for k in REQUIRED_KEYS):
        return None
    path = save_norm_bounds(target_dir, bounds_from_cfg(cfg, source=str(target_dir)))
    return path


def resolve_run_dir_from_checkpoint(ckpt_path: str | Path) -> Path:
    """Resolve ``.../<run>/models/checkpoint_best.pt`` -> ``.../<run>``."""
    ckpt = Path(ckpt_path)
    if ckpt.parent.name == "models":
        return ckpt.parent.parent
    return ckpt.parent
