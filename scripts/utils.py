from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_get(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, out_base: Path, formats: List[str]) -> None:
    for ext in formats:
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)