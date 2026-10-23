"""
Committed default + a named regime's capacity override (week 30).

    from regime_config import committed_with_regime
    cfg, info = committed_with_regime("configs/regime_small_cluster.yaml")   # or None: the committed default

Both YAML files are read from git HEAD (`git show HEAD:<path>`), never from the working copy, so an uncommitted edit of
configs/default.yaml is not used. The regime file may override ONLY the `capacity` block (checked); its keys are merged
over the committed default's capacity block and the result is loaded through dlent_rt.config.load_config (temporary
file, deleted). `info` holds the commit, the blob of configs/default.yaml and the blob of the regime file.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _git(*a) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *a], text=True).strip()


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base.get(k, {}) or {}, v) if isinstance(v, dict) else v
    return out


def committed_with_regime(regime: Optional[str] = None):
    from dlent_rt.config import load_config
    default_txt = _git("show", "HEAD:configs/default.yaml")
    info = {"commit": _git("rev-parse", "HEAD"), "default_blob": _git("rev-parse", "HEAD:configs/default.yaml"),
            "regime": regime or "none (committed default)", "regime_blob": None}
    data = yaml.safe_load(default_txt)
    if regime:
        rel = str(Path(regime).resolve().relative_to(ROOT)) if Path(regime).is_absolute() else regime
        over = yaml.safe_load(_git("show", f"HEAD:{rel}")) or {}
        if set(over) - {"capacity"}:
            raise ValueError(f"{rel} may override only 'capacity', found {sorted(over)}")
        data = _merge(data, over)
        info["regime_blob"] = _git("rev-parse", f"HEAD:{rel}")
        info["regime"] = rel
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "committed_with_regime.yaml"
        p.write_text(yaml.safe_dump(data, sort_keys=False))
        cfg = load_config(p)
    return cfg, info
