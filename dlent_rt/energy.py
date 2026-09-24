"""
Energy model and grid-data loading.

Converts Google Cluster normalized resource requests into physical kW and kWh,
using a reference machine spec. Loads time-varying electricity prices and
carbon intensity from a separate grid CSV.

References:
  - Reiss & Tumanov (2012), "Heterogeneity and dynamism of clouds at scale:
    Google trace analysis", defines the normalized resource units.
  - Fan et al. (2007), "Power provisioning for a warehouse-sized computer",
    establishes the ~15W/core linear power model for data center servers.
  - Barroso et al. (2019), "The Datacenter as a Computer" (3rd ed.), §5,
    documents PUE and per-component power breakdown in warehouse-scale computing.
  - Google Environmental Report (2019) confirms PUE ~1.10 for Google data centers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EnergyParams:
    """Hardware and conversion constants."""
    spec_cpu_core: float = 64       # cores per 1.0 normalized CPU in the trace
    spec_ram_gb: float = 256        # GB per 1.0 normalized RAM in the trace
    p_core: float = 15.0            # watts per physical core under load
    p_gb: float = 0.35              # watts per GB of RAM
    pue: float = 1.10               # power usage effectiveness (facility overhead)


def compute_power_kw(
    cpu_norm: np.ndarray, ram_norm: np.ndarray,
    scheduling_class: np.ndarray, params: EnergyParams,
) -> np.ndarray:
    """
    Instantaneous power draw in kW per job.

    Maps normalized Google trace units to physical resources, then applies the
    scheduling-class efficiency factor kappa(sc).

    kappa models that latency-sensitive workloads (sc=3) disable CPU power-saving
    features (C-states, DVFS), consuming up to ~35% more power than batch (sc=0).
    Linear interpolation: kappa(sc) = 1.0 + sc * 0.116, giving:
      sc=0 -> 1.000 (batch, energy-optimized)
      sc=1 -> 1.116
      sc=2 -> 1.233
      sc=3 -> 1.349 (latency-critical, power-saving disabled)

    Ref: Lo et al. (2015), "Heracles: improving resource efficiency at scale",
    documents the 20-35% power overhead of latency-critical vs batch workloads.
    """
    cpu_cores = cpu_norm * params.spec_cpu_core
    ram_gb = ram_norm * params.spec_ram_gb
    kappa = 1.0 + scheduling_class * 0.116
    power_w = (params.p_core * cpu_cores * kappa) + (params.p_gb * ram_gb)
    return params.pue * power_w / 1000.0  # W -> kW


def compute_energy_kwh(
    cpu_norm: np.ndarray, ram_norm: np.ndarray,
    scheduling_class: np.ndarray, duration_hours: np.ndarray,
    params: EnergyParams,
) -> np.ndarray:
    """Total energy consumption in kWh = power_kW * duration_hours."""
    power_kw = compute_power_kw(cpu_norm, ram_norm, scheduling_class, params)
    return power_kw * duration_hours


@dataclass
class GridData:
    """Time-varying electricity price and carbon intensity, hourly resolution."""
    timestamps: np.ndarray       # datetime64[ns]
    elec_price: np.ndarray       # $/kWh
    carbon_intensity: np.ndarray # kgCO2/kWh

    def lookup(self, request_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        For each request_time, return (elec_price, carbon_intensity) at that hour.
        Uses floor to the containing hour bucket, then nearest-match.
        """
        req_hours = request_times.astype("datetime64[h]")
        grid_hours = self.timestamps.astype("datetime64[h]")
        idx = np.searchsorted(grid_hours, req_hours, side="right") - 1
        idx = np.clip(idx, 0, len(self.timestamps) - 1)
        return self.elec_price[idx], self.carbon_intensity[idx]

    @property
    def ci_min(self) -> float:
        return float(self.carbon_intensity.min())

    @property
    def ci_max(self) -> float:
        return float(self.carbon_intensity.max())

    @property
    def price_min(self) -> float:
        return float(self.elec_price.min())

    @property
    def price_max(self) -> float:
        return float(self.elec_price.max())


def load_grid_data(path: str) -> GridData:
    """Load hourly grid CSV with columns: timestamp_utc, elec_price_per_kWh,
    carbon_intensity_kgCO2_per_kWh."""
    df = pd.read_csv(path)
    ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp"
    df[ts_col] = pd.to_datetime(df[ts_col]).dt.tz_localize(None)
    df = df.sort_values(ts_col).drop_duplicates(subset=[ts_col])
    return GridData(
        timestamps=df[ts_col].values,
        elec_price=df["elec_price_per_kWh"].values.astype(float),
        carbon_intensity=df["carbon_intensity_kgCO2_per_kWh"].values.astype(float),
    )