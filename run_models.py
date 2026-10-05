"""Run the NASA Harvest / Agmatix crop-cycle and crop-stage models on the
gridded NDVI series and write GeoTIFF products.

Products (per crop, per grid in {30m, 1km, 9km}):
  cycle_<crop>_<year>.tif         float32, one band per milestone (see CYCLE_BANDS);
                                   dates are day-of-year relative to Jan 1 of <year>
  stage_<crop>_<year>_weekly.tif  uint8, one band per NASS week (ending Sunday);
                                   1-5 = stage A-E, 0 = insufficient data, 255 = no crop
  stage_<crop>_<year>_summary.csv region-wide share of crop pixels in each stage per week

Both models run as numba re-implementations by default: fast_cycle.py (identical
output to crop-cycle-detection, see validate_fast_cycle.py) and fast_stage.py
(identical output to crop-stage-detection, see validate_fast_stage.py).
Pass --cycle-impl original / --stage-impl original to use the original code instead.

Usage:
  python run_models.py --year 2024 --region boone [--grid 9km 1km 30m] [--workers N]
                       [--cycle-impl fast|original] [--stage-impl fast|original]
                       [--cpc-ref cornProgYYwWW.tif]
"""
import argparse
import logging
import os
import sys
import time
import warnings
from multiprocessing import Pool

import common as C  # sets PROJ_DATA before rasterio is used

import numpy as np
import pandas as pd
import rasterio

# The two NASA Harvest repos (see README); HARVEST_REPOS overrides the location.
for _repo in ("crop-cycle-detection", "crop-stage-detection"):
    if not os.path.isdir(os.path.join(C.HARVEST_REPOS, _repo)):
        sys.exit(f"{_repo} not found in {C.HARVEST_REPOS}. Clone it there (see README) "
                 "or set HARVEST_REPOS to the folder that contains it.")
sys.path.insert(0, os.path.join(C.HARVEST_REPOS, "crop-cycle-detection"))
sys.path.insert(0, os.path.join(C.HARVEST_REPOS, "crop-stage-detection", "src"))
from src.crop_cycle import crop_cycle_model  # noqa: E402

import fast_cycle  # noqa: E402
import fast_stage  # noqa: E402

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

MIN_CROP_FRAC = 0.05   # 1 km cell must be >= 5% core crop pixels (5 ha)
MIN_CLEAR_FRAC = 0.5   # >= 50% of the cell's crop pixels clear in a bin (Harvest Landsat rule)
MIN_OBS = 10           # usable 5-day bins needed to run the models on a pixel
LOOKBACK_DAYS = 150    # crop-stage default lookback
BLOCK_PIXELS = 2000    # pixels per worker task
STAGE_CODE = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
CYCLE_BANDS = ["season_start", "greenup", "ndvi_peak", "ndvi_peak_value",
               "season_end", "season_length", "n_seasons"]

_G = {}


def _init(dates, weeks, year, stage_impl, cycle_impl="fast"):
    day = lambda s: np.datetime64(s, "D").astype(np.int64)
    _G.update(dates=dates, weeks=weeks, year=year, stage_impl=stage_impl, cycle_impl=cycle_impl,
              year_bounds=(day(f"{year}-01-01"), day(f"{year}-04-01"), day(f"{year}-10-31")),
              days=dates.astype("datetime64[D]").astype(np.int64),
              wdays=np.array([np.datetime64(w.date(), "D").astype(np.int64) for w in weeks]))


def _doy(ts, year):
    return np.nan if pd.isna(ts) else float((pd.Timestamp(ts) - pd.Timestamp(f"{year}-01-01")).days + 1)


def cycle_vector(df, year):
    cyc = np.full(len(CYCLE_BANDS), np.nan, np.float32)
    try:
        report, _ = crop_cycle_model(df)
    except Exception:
        report = pd.DataFrame()
    if not report.empty:
        pk = pd.to_datetime(report["ndvi_peak_date"])
        main = report[(pk.dt.year == year) & pk.dt.month.between(4, 10)]
        cyc[6] = len(main)
        if not main.empty:
            s = main.loc[main["ndvi_peak_value"].astype(float).idxmax()]
            cyc[:6] = [_doy(s["approx_season_start"], year), _doy(s["greenup_date"], year),
                       _doy(s["ndvi_peak_date"], year), float(s["ndvi_peak_value"]),
                       _doy(s["approx_season_end"], year), float(s["approx_season_length"])]
    return cyc


def _original_stage_fn():
    # imported only when --stage-impl original is used (and by validate_fast_stage.py)
    from crop_stage import run_crop_stage_from_dataframe
    return run_crop_stage_from_dataframe


def stage_original(df, weeks):
    stage = np.zeros(len(weeks), np.uint8)
    for j, wend in enumerate(weeks):
        # a 5-day bin centred on d spans d-2..d+2: only use bins complete by week end
        sub = df[(df["date"] + pd.Timedelta(days=2) <= wend)
                 & (df["date"] > wend - pd.Timedelta(days=LOOKBACK_DAYS))]
        if len(sub) < 3:
            continue
        try:
            r = _original_stage_fn()(sub)
            stage[j] = STAGE_CODE.get(r["Stage"], 0)
        except Exception:
            pass
    return stage


def run_cell(idx, ndvi):
    """One pixel: indices of usable bins and their NDVI -> (cycle vector, stage vector)."""
    dates, weeks, year = _G["dates"], _G["weeks"], _G["year"]
    df = None
    if _G["cycle_impl"] == "original" or _G["stage_impl"] == "original":
        df = pd.DataFrame({"date": pd.to_datetime(dates[idx]), "NDVI": ndvi})
    if _G["cycle_impl"] == "original":
        cyc = cycle_vector(df, year)
    else:
        cyc = fast_cycle.cycle_pixel(_G["days"][idx], ndvi, *_G["year_bounds"]).astype(np.float32)
    if _G["stage_impl"] == "original":
        stage = stage_original(df, weeks)
    else:
        stage = fast_stage.stage_weeks(_G["days"][idx], ndvi, _G["wdays"], LOOKBACK_DAYS)
    return cyc, stage


def run_block(vals):
    """vals: (T, npx) NDVI with NaN for unusable bins -> (cycle (7, npx), stage (W, npx))."""
    npx = vals.shape[1]
    cyc = np.full((len(CYCLE_BANDS), npx), np.nan, np.float32)
    stg = np.full((len(_G["weeks"]), npx), 255, np.uint8)
    for p in range(npx):
        idx = np.nonzero(np.isfinite(vals[:, p]))[0]
        if len(idx) >= MIN_OBS:
            cyc[:, p], stg[:, p] = run_cell(idx, vals[idx, p].astype(np.float64))
    return cyc, stg


def usable(d, crop):
    """-> (T,H,W) NDVI with NaN where a bin is unusable, (H,W) mask of pixels to model."""
    if "label" in d:  # 30 m: one crop per pixel
        mask = d["label"] == C.CROPS[crop]
        return np.where(mask[None], d["ndvi"], np.nan).astype(np.float32), mask
    ndvi, fclr, fcrop = d[f"ndvi_{crop}"], d[f"fclr_{crop}"], d[f"fcrop_{crop}"]
    mask = fcrop >= MIN_CROP_FRAC
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (fclr / np.where(fcrop > 0, fcrop, np.nan) >= MIN_CLEAR_FRAC) & np.isfinite(ndvi) & mask
    return np.where(ok, ndvi, np.nan).astype(np.float32), mask


def grid_info(d):
    """(x0, y0, ncol, nrow, res); older 1 km files store only the first four."""
    g = [float(v) for v in d["grid"]]
    return tuple(g if len(g) == 5 else g + [C.RES])


def grid_transform(d):
    if "transform" in d:  # 9 km CPC grid: x and y pixel sizes differ
        return rasterio.Affine(*d["transform"])
    x0, y0, _, _, res = grid_info(d)
    return rasterio.Affine(res, 0, x0, 0, -res, y0)


def aggregate_9km(d, cpc_ref=None):
    """Aggregate 1 km arrays onto the NASS CPC 9 km grid (window covering the region)."""
    if cpc_ref:
        with rasterio.open(cpc_ref) as ref:
            rt = ref.transform
    else:
        rt = C.CPC_9KM_TRANSFORM
    x0, y0, ncol, nrow, res = (int(v) for v in grid_info(d))
    xs = x0 + (np.arange(ncol) + 0.5) * res
    ys = y0 - (np.arange(nrow) + 0.5) * res
    ci = np.floor((xs - rt.c) / rt.a).astype(int)
    ri = np.floor((ys - rt.f) / rt.e).astype(int)
    c0, r0 = ci.min(), ri.min()
    W9, H9 = ci.max() - c0 + 1, ri.max() - r0 + 1
    R, Cc = np.meshgrid(ri - r0, ci - c0, indexing="ij")
    flat = (R * W9 + Cc).ravel()
    n9 = H9 * W9
    cnt = np.bincount(flat, minlength=n9).astype(np.float32)

    out = {"dates": d["dates"],
           "grid": np.array([rt.c + c0 * rt.a, rt.f + r0 * rt.e, W9, H9, rt.a]),
           "transform": np.array([rt.a, 0, rt.c + c0 * rt.a, 0, rt.e, rt.f + r0 * rt.e])}
    for crop in C.CROPS:
        fcrop = d[f"fcrop_{crop}"]
        out[f"fcrop_{crop}"] = (np.bincount(flat, fcrop.ravel(), n9) / cnt).reshape(H9, W9)
        nd, fc = [], []
        for t in range(len(d["dates"])):
            v, w = d[f"ndvi_{crop}"][t], d[f"fclr_{crop}"][t]
            good = np.isfinite(v) & (w > 0)
            sw = np.bincount(flat, np.where(good, w, 0).ravel(), n9)
            svw = np.bincount(flat, np.where(good, v * w, 0).ravel(), n9)
            with np.errstate(invalid="ignore", divide="ignore"):
                nd.append((svw / sw).reshape(H9, W9))
            fc.append((sw / cnt).reshape(H9, W9))
        out[f"ndvi_{crop}"] = np.stack(nd).astype(np.float32)
        out[f"fclr_{crop}"] = np.stack(fc).astype(np.float32)
    return out


def write_tif(path, arr, transform, nodata, descriptions):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype=arr.dtype, crs=C.CRS, transform=transform,
                       nodata=nodata, compress="deflate", tiled=True) as dst:
        dst.write(arr)
        for i, desc in enumerate(descriptions, 1):
            dst.set_band_description(i, desc)


def run_grid(d, grid, year, workers, region, stage_impl, cycle_impl="fast"):
    dates = d["dates"]
    weeks = C.nass_weeks(year)
    wkeys, wends = list(weeks), list(weeks.values())
    odir = os.path.join(C.OUT, region, grid)
    os.makedirs(odir, exist_ok=True)
    transform = grid_transform(d)
    for crop in C.CROPS:
        vals, mask = usable(d, crop)
        H, W = mask.shape
        rr, cc = np.nonzero(mask)
        pix = vals[:, rr, cc]
        blocks = [pix[:, i:i + BLOCK_PIXELS] for i in range(0, len(rr), BLOCK_PIXELS)]
        print(f"[{grid} {crop}] {len(rr):,} pixels in {len(blocks)} blocks", flush=True)
        t0 = time.time()
        res = []
        with Pool(workers, initializer=_init, initargs=(dates, wends, year, stage_impl, cycle_impl)) as pool:
            for i, r in enumerate(pool.imap(run_block, blocks)):
                res.append(r)
                if (i + 1) % max(1, len(blocks) // 10) == 0:
                    el = time.time() - t0
                    print(f"  {i + 1}/{len(blocks)} blocks, {el / 60:.1f} min elapsed, "
                          f"~{el / (i + 1) * (len(blocks) - i - 1) / 60:.1f} min left", flush=True)
        print(f"[{grid} {crop}] models done in {(time.time() - t0) / 60:.1f} min", flush=True)

        cyc = np.full((len(CYCLE_BANDS), H, W), np.nan, np.float32)
        stg = np.full((len(wends), H, W), 255, np.uint8)
        if res:
            cyc[:, rr, cc] = np.concatenate([r[0] for r in res], axis=1)
            stg[:, rr, cc] = np.concatenate([r[1] for r in res], axis=1)
        write_tif(os.path.join(odir, f"cycle_{crop}_{year}.tif"), cyc, transform, np.nan,
                  [f"{b}{'' if b in ('ndvi_peak_value', 'season_length', 'n_seasons') else '_doy'}"
                   for b in CYCLE_BANDS])
        write_tif(os.path.join(odir, f"stage_{crop}_{year}_weekly.tif"), stg, transform, 255,
                  [f"w{w:02d}_{e:%Y-%m-%d}" for w, e in weeks.items()])

        valid = stg != 255
        summ = pd.DataFrame({"week": wkeys, "week_ending": wends})
        for code, lab in [(0, "insufficient")] + [(v, k) for k, v in STAGE_CODE.items()]:
            summ[lab] = [(stg[j][valid[j]] == code).mean() for j in range(len(wends))]
        summ.to_csv(os.path.join(odir, f"stage_{crop}_{year}_summary.csv"), index=False, float_format="%.3f")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--region", default="boone", choices=list(C.REGIONS))
    ap.add_argument("--grid", nargs="+", default=["9km", "1km"], choices=["9km", "1km", "30m"])
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--stage-impl", default="fast", choices=["fast", "original"])
    ap.add_argument("--cycle-impl", default="fast", choices=["fast", "original"])
    ap.add_argument("--cpc-ref", help="NASS gridded CPC GeoTIFF to take the 9 km grid from "
                                      "(default: built-in grid of cornProg24w20.tif)")
    a = ap.parse_args()
    for g in a.grid:
        if g == "30m":
            d = dict(np.load(C.ndvi_path(a.region, a.year, "30m")))
        else:
            d = dict(np.load(C.ndvi_path(a.region, a.year)))
            if g == "9km":
                d = aggregate_9km(d, a.cpc_ref)
        run_grid(d, g, a.year, a.workers, a.region, a.stage_impl, a.cycle_impl)
