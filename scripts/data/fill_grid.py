"""
Fill the hourly price / carbon file to 3 June 2019 00:00 UTC (week 32). The PJM file ends at 1 June 2019 03:00 UTC.

    dlent_venv/bin/python scripts/data/fill_grid.py

Every missing hour h after the last row, up to and including 2019-06-03 00:00 UTC, copies price AND carbon intensity
from the same hour of day on 30 or 31 May: day 1 June -> 30 May, 2 June -> 31 May, 3 June -> 30 May (the two days in
turn). Copied rows get is_filled = True. Input data/grid_data_may_2019.csv (unchanged); output
data/grid_data_may_2019_filled.csv (configs/v3_mean_demand.yaml).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "grid_data_may_2019.csv"
OUT = ROOT / "data" / "grid_data_may_2019_filled.csv"
END = pd.Timestamp("2019-06-03 00:00")
SOURCE_DAYS = (pd.Timestamp("2019-05-30"), pd.Timestamp("2019-05-31"))


def main() -> None:
    g = pd.read_csv(SRC, parse_dates=["timestamp_utc"])
    g["is_filled"] = False
    by_t = g.set_index("timestamp_utc")
    hours = pd.date_range(g.timestamp_utc.max() + pd.Timedelta(hours=1), END, freq="h")
    first_day = pd.Timestamp("2019-06-01")
    rows = []
    for t in hours:
        src = SOURCE_DAYS[(t.normalize() - first_day).days % 2] + pd.Timedelta(hours=t.hour)
        r = by_t.loc[src]
        rows.append({"timestamp_utc": t, "elec_price_per_kWh": r.elec_price_per_kWh,
                     "carbon_intensity_kgCO2_per_kWh": r.carbon_intensity_kgCO2_per_kWh, "is_filled": True})
    out = pd.concat([g, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"{len(rows)} hours filled ({hours[0]} .. {hours[-1]}) -> {OUT}")


if __name__ == "__main__":
    main()
