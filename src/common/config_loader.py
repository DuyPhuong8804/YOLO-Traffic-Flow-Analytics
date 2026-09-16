"""Ham tien ich doc file YAML config dung chung cho toan bo du an."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Doc 1 file YAML va tra ve dict. Neu path la ten file tuong doi,
    tu dong tim trong thu muc configs/ cua project."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = CONFIGS_DIR / path
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
