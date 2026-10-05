"""Check fast_stage.stage_weeks against the original crop-stage-detection code.

Usage: python validate_fast_stage.py --region boone --year 2024 [--res 1km|30m] [--n N]
"""
import argparse
import time

import common as C

import numpy as np
import pandas as pd

import fast_stage as F
import run_models as R  # also puts the NASA Harvest repos on sys.path
from crop_stage import smooth_daily_interpolate_ndvi  # noqa: E402


def reference_stages(dates, idx, ndvi, weeks):
    """Original code path (run_models.run_cell, stage part only)."""
    df = pd.DataFrame({"date": pd.to_datetime(dates[idx]), "NDVI": ndvi})
    out = np.zeros(len(weeks), np.uint8)
    for j, wend in enumerate(weeks):
        sub = df[(df["date"] + pd.Timedelta(days=2) <= wend)
                 & (df["date"] > wend - pd.Timedelta(days=R.LOOKBACK_DAYS))]
        if len(sub) < 3:
            continue
        r = R._original_stage_fn()(sub)
        out[j] = R.STAGE_CODE.get(r["Stage"], 0)
    return out


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
    weeks = list(C.nass_weeks(a.year).values())
    wdays = np.array([np.datetime64(w.date(), "D").astype(np.int64) for w in weeks])

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

    # 1) smoothed curves agree numerically
    maxdiff = 0.0
    for idx, v in series[:200]:
        ref = smooth_daily_interpolate_ndvi(pd.DataFrame({"date": pd.to_datetime(dates[idx]), "NDVI": v}))
        fast = F._whittaker(F._pchip_daily(days[idx].astype(float), v), F.WHITTAKER_LAM)
        maxdiff = max(maxdiff, float(np.max(np.abs(ref["NDVI_smooth"].to_numpy() - fast))))
    print(f"max |smoothed NDVI difference| over 200 full series: {maxdiff:.2e}")

    # 2) stage codes agree
    F.stage_weeks(days[series[0][0]], series[0][1], wdays, R.LOOKBACK_DAYS)  # compile
    t = time.time()
    fast = [F.stage_weeks(days[idx], v, wdays, R.LOOKBACK_DAYS) for idx, v in series]
    tf = time.time() - t
    t = time.time()
    ref = [reference_stages(dates, idx, v, weeks) for idx, v in series]
    tr = time.time() - t
    fast, ref = np.array(fast), np.array(ref)
    mism = np.argwhere(fast != ref)
    print(f"series {len(series)}, stage calls {fast.size}, mismatches {len(mism)}")
    print(f"time per series: original {tr / len(series) * 1000:.1f} ms, fast {tf / len(series) * 1000:.3f} ms"
          f" ({tr / tf:.0f}x)")
    for i, j in mism[:10]:
        print(f"  series {i} week {j}: original {ref[i, j]} fast {fast[i, j]}")
