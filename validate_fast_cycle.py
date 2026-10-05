"""Check fast_cycle.cycle_pixel against the original crop-cycle-detection code.

Usage: python validate_fast_cycle.py --region boone --year 2024 [--res 1km|30m] [--n N]
"""
import argparse
import time

import common as C

import numpy as np
import pandas as pd

import fast_cycle as F
import run_models as R  # also puts the NASA Harvest repos on sys.path


def year_bounds(year):
    day = lambda s: np.datetime64(s, "D").astype(np.int64)
    return day(f"{year}-01-01"), day(f"{year}-04-01"), day(f"{year}-10-31")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="boone")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--res", default="1km", choices=["1km", "30m"])
    ap.add_argument("--n", type=int, default=0, help="random sample of N pixels (default all; 3000 at 30m)")
    a = ap.parse_args()

    d = dict(np.load(C.ndvi_path(a.region, a.year, a.res)))
    dates = d["dates"]
    days = dates.astype("datetime64[D]").astype(np.int64)
    series = []
    for crop in C.CROPS:
        vals, mask = R.usable(d, crop)
        for r, c in zip(*np.nonzero(mask)):
            idx = np.nonzero(np.isfinite(vals[:, r, c]))[0]
            if len(idx) >= R.MIN_OBS:
                series.append((idx, vals[idx, r, c].astype(np.float64)))
    if a.res == "30m" and not a.n:
        a.n = 3000
    if a.n:
        rng = np.random.default_rng(0)
        series = [series[i] for i in sorted(rng.choice(len(series), min(a.n, len(series)), replace=False))]

    jan1, apr1, oct31 = year_bounds(a.year)
    F.cycle_pixel(days[series[0][0]], series[0][1], jan1, apr1, oct31)  # compile
    t = time.time()
    fast = np.array([F.cycle_pixel(days[idx], v, jan1, apr1, oct31) for idx, v in series])
    tf = time.time() - t
    t = time.time()
    ref = np.array([R.cycle_vector(pd.DataFrame({"date": pd.to_datetime(dates[idx]), "NDVI": v}), a.year)
                    for idx, v in series], dtype=np.float64)
    tr = time.time() - t

    fast = fast.astype(np.float32).astype(np.float64)  # products are float32, like cycle_vector
    same = (fast == ref) | (np.isnan(fast) & np.isnan(ref))
    bad = np.nonzero(~same.all(axis=1))[0]
    print(f"series {len(series)}, pixels with any difference {len(bad)}")
    for j, name in enumerate(R.CYCLE_BANDS):
        print(f"  {name:16s} mismatches {int((~same[:, j]).sum())}")
    print(f"time per series: original {tr / len(series) * 1000:.1f} ms, fast {tf / len(series) * 1000:.3f} ms"
          f" ({tr / tf:.0f}x)")
    for i in bad[:8]:
        print(f"  series {i}: original {np.round(ref[i], 3)}\n             fast     {np.round(fast[i], 3)}")
