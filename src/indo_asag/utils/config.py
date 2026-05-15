"""Configuration loading utilities."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


def load_config(path: str = "configs/base.yaml") -> dict:
    """Load YAML config file and return as nested dict.
    
    Args:
        path: Path to YAML config file.
        
    Returns:
        Configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge two config dicts (override takes priority).
    
    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.
        
    Returns:
        Merged configuration dictionary.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
