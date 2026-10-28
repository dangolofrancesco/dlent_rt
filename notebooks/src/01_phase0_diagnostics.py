# %% [markdown]
# # DLENT-RT — Phase 0 on a configuration
#
# Loads one config file, runs Phase 0 on its history H, and prints what Phase 0 produced
# (`dlent_rt.diagnostics.phase0_summary`). The logic lives in the package; this notebook only shows it.
#
# Config: the environment variable `DLENT_CONFIG` (a path relative to the repository root), default
# `configs/v3_mean_demand.yaml`. Build and run: `DLENT_CONFIG=... python notebooks/src/build.py 01_phase0_diagnostics --execute`.
# No online run.

# %%
import os, sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from dlent_rt.config import load_config
from dlent_rt.data import load_dataset
from dlent_rt.phase0 import run_phase0
from dlent_rt.diagnostics import phase0_summary
warnings.simplefilter("ignore")

CONFIG = os.environ.get("DLENT_CONFIG", "configs/v3_mean_demand.yaml")
cfg = load_config(ROOT / CONFIG)
ds = load_dataset(cfg)
print(f"config: {CONFIG}  |H| = {ds.H.n:,} jobs  |stream| = {ds.stream.n:,}  T = {ds.T / 3600:.1f} h  "
      f"d_max = {ds.d_max_steps / 3600:.2f} h")
print(f"capacity mode {cfg.capacity.mode}: C = {np.round(ds.capacity, 4).tolist()} machines (CPU, RAM)")
if ds.capacity_diag is not None:
    print("mean demand on H:", {k: (np.round(v, 5).tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in ds.capacity_diag.items()})
lr = ds.limit_report
print(f"limits: R_max {lr.r_max_machines} removed {lr.n_removed}; quota {lr.quota} rejected {lr.n_quota_rejected}")

# %% [markdown]
# ## Phase 0 output

# %%
st = run_phase0(ds, cfg)
S = phase0_summary(ds, st, cfg)
rows = [(k, json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in S.items() if k != "flags"]
with pd.option_context("display.max_colwidth", 200, "display.max_rows", 100):
    display(pd.DataFrame(rows, columns=["quantity", "value"]).set_index("quantity"))

# %% [markdown]
# ## Flags

# %%
print("\n".join(f"- {f}" for f in S["flags"]) or "no flags")
