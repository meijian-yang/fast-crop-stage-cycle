"""Check fast_models.crop_cycle / crop_stages against the original NASA Harvest code
on the sample NDVI files shipped with both repos and, optionally, on gridded series.

Usage: python validate_fast_models.py [--region boone --year 2024 --n 500]
"""
import argparse
import os

import common as C

import numpy as np
import pandas as pd

import run_models as R  # also puts the NASA Harvest repos on sys.path
from fast_models import crop_cycle, crop_stages

CYCLE_KEYS = ["season_start", "greenup", "ndvi_peak", "ndvi_peak_value", "season_end",
              "season_length", "n_seasons"]


def cycle_as_vector(res, year):
    jan1 = pd.Timestamp(f"{year}-01-01")
    v = []
    for k in CYCLE_KEYS:
        x = res[k]
        if isinstance(x, pd.Timestamp) or x is pd.NaT:
            v.append(np.nan if pd.isna(x) else (x - jan1).days + 1)
        else:
            v.append(x)
    return np.array(v, dtype=np.float32)


def compare_series(dates, ndvi, label):
    df = pd.DataFrame({"date": pd.to_datetime(dates), "NDVI": ndvi}).sort_values("date")
    years = range(df["date"].dt.year.min(), df["date"].dt.year.max() + 1)
    cyc_bad = 0
    for y in years:
        ref = R.cycle_vector(df.reset_index(drop=True), y)
        fast = cycle_as_vector(crop_cycle(df["date"], df["NDVI"], y), y)
        cyc_bad += not np.array_equal(ref, fast, equal_nan=True)
    asof = pd.date_range(df["date"].min() + pd.Timedelta(days=7), df["date"].max(), freq="7D")
    ref = []
    for d in asof:
        sub = df[(df["date"] <= d) & (df["date"] > d - pd.Timedelta(days=150))]
        ref.append(R._original_stage_fn()(sub)["Stage"] if len(sub) >= 3 else "Insufficient Data")
    fast = crop_stages(df["date"], df["NDVI"], asof)
    stg_bad = sum(a != b for a, b in zip(ref, fast))
    return cyc_bad, len(years), stg_bad, len(asof)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", help="also test gridded series from this region's 1 km NDVI file")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()

    tot = np.zeros(4, int)
    samples = [
        ("crop-stage-detection", "sample_data/sample_ndvi.csv"),
        ("crop-cycle-detection", "sample_data/example_ndvi.csv"),
    ]
    for repo, rel in samples:
        df = pd.read_csv(os.path.join(C.HARVEST_REPOS, repo, rel))
        groups = df.groupby("field_id") if "field_id" in df else [(rel, df)]
        for name, g in groups:
            r = compare_series(g["date"], g["NDVI"], name)
            tot += r
            print(f"{repo}/{rel} [{name}]: cycle mismatches {r[0]}/{r[1]} years, "
                  f"stage mismatches {r[2]}/{r[3]} dates")

    if a.region:
        d = dict(np.load(C.ndvi_path(a.region, a.year)))
        dates = d["dates"]
        series = []
        for crop in C.CROPS:
            vals, mask = R.usable(d, crop)
            for r_, c_ in zip(*np.nonzero(mask)):
                idx = np.nonzero(np.isfinite(vals[:, r_, c_]))[0]
                if len(idx) >= R.MIN_OBS:
                    series.append((idx, vals[idx, r_, c_]))
        rng = np.random.default_rng(0)
        pick = rng.choice(len(series), min(a.n, len(series)), replace=False)
        part = np.zeros(4, int)
        for i in pick:
            idx, v = series[i]
            part += compare_series(dates[idx], v.astype(np.float64), i)
        tot += part
        print(f"{a.region} {a.year} 1 km, {len(pick)} series: cycle mismatches {part[0]}/{part[1]}, "
              f"stage mismatches {part[2]}/{part[3]}")
    print(f"TOTAL: cycle mismatches {tot[0]}/{tot[1]}, stage mismatches {tot[2]}/{tot[3]}")
