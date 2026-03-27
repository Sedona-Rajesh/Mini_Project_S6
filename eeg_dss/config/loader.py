"""
eeg_dss/config/loader.py
────────────────────────
Single entry-point for all configuration.  Validates required keys and
exposes a typed-ish namespace so the rest of the codebase never reaches
for raw dicts.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_REQUIRED_SECTIONS = [
    "data",
    "preprocessing",
    "features",
    "alzheimer",
    "depression",
    "training",
    "evaluation",
    "outputs",
    "logging",
]


class Config:
    """Thin wrapper around the YAML dict with dot-access and helpers."""

    def __init__(self, raw: dict[str, Any], config_path: Path) -> None:
        self._raw = raw
        self.config_path = config_path
        self.project_root = _project_root_from_config(config_path)

        # top-level sections become attributes
        for key, value in raw.items():
            setattr(self, key, value)

    # ------------------------------------------------------------------
    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested get:  cfg.get("preprocessing", "bandpass", "l_freq")."""
        node = self._raw
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
        return node

    # ------------------------------------------------------------------
    def output_dir(self, dataset: str, kind: str) -> Path:
        """Return and create the output directory for dataset+kind."""
        p = Path(self._raw["outputs"][dataset][kind])
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    def data_root(self, dataset_key: str) -> Path:
        """Return absolute path to a specific BIDS dataset root."""
        raw_root = Path(self._raw["data"]["raw_root"])
        folder = self._raw["data"][dataset_key]
        return raw_root / folder

    # ------------------------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self._raw.get("random_seed", 42))

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"


def load_config(config_path: str | Path = "configs/config.yaml") -> Config:
    """
    Load YAML configuration, validate required sections, and return a
    ``Config`` instance.

    Parameters
    ----------
    config_path:
        Path to the YAML file.  Relative paths are resolved from cwd.

    Returns
    -------
    Config
    """
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Create configs/config.yaml or pass --config <path>."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got: {type(raw)}")

    missing = [s for s in _REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise ValueError(
            f"Config is missing required sections: {missing}\n"
            f"Config path: {config_path}"
        )

    _resolve_paths(raw, config_path)

    cfg = Config(raw, config_path)
    _setup_logging(cfg)
    logger.info("Config loaded from %s", config_path)
    return cfg


def _project_root_from_config(config_path: Path) -> Path:
    """Infer project root from config location."""
    cfg_dir = config_path.parent
    if cfg_dir.name.lower() == "configs":
        return cfg_dir.parent.resolve()
    return cfg_dir.resolve()


def _resolve_path(path_value: str, base_dir: Path) -> str:
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _resolve_paths(raw: dict[str, Any], config_path: Path) -> None:
    """Resolve path-like config entries from inferred project root."""
    base_dir = _project_root_from_config(config_path)

    data_cfg = raw.get("data", {})
    if isinstance(data_cfg, dict):
        raw_root = data_cfg.get("raw_root")
        if isinstance(raw_root, str):
            data_cfg["raw_root"] = _resolve_path(raw_root, base_dir)
        output_root = data_cfg.get("output_root")
        if isinstance(output_root, str):
            data_cfg["output_root"] = _resolve_path(output_root, base_dir)

    outputs_cfg = raw.get("outputs", {})
    if isinstance(outputs_cfg, dict):
        for dataset_paths in outputs_cfg.values():
            if not isinstance(dataset_paths, dict):
                continue
            for kind, path_value in dataset_paths.items():
                if isinstance(path_value, str):
                    dataset_paths[kind] = _resolve_path(path_value, base_dir)

    logging_cfg = raw.get("logging", {})
    if isinstance(logging_cfg, dict):
        log_file = logging_cfg.get("log_file")
        if isinstance(log_file, str):
            logging_cfg["log_file"] = _resolve_path(log_file, base_dir)


def _setup_logging(cfg: Config) -> None:
    """Configure root logger based on config settings."""
    log_cfg = cfg._raw.get("logging", {})
    level_name = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    log_file = log_cfg.get("log_file")
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
