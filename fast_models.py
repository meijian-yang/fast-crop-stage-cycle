"""Date-based entry points to the fast crop cycle and crop stage models.

These work on any NDVI time series: a field, a pixel, a point. They return the
same results as the NASA Harvest / Agmatix code (crop-cycle-detection and
crop-stage-detection) with default parameters; see validate_fast_models.py.

    import pandas as pd
    from fast_models import crop_cycle, crop_stages

    df = pd.read_csv("my_field_ndvi.csv")              # columns: date, NDVI
    crop_cycle(df["date"], df["NDVI"], year=2024)
    # {'season_start': Timestamp('2024-05-28'), 'greenup': ..., 'ndvi_peak': ...,
    #  'ndvi_peak_value': 0.91, 'season_end': ..., 'season_length': 127.0, 'n_seasons': 1}

    crop_stages(df["date"], df["NDVI"], as_of=["2024-06-16", "2024-08-11"])
    # ['B', 'C']

Input rules:
  * dates are used at day precision and must be unique (average same-day
    observations first, e.g. keep the clearest pass as the Harvest fetcher does);
  * missing NDVI values are dropped.
"""
import numpy as np
import pandas as pd

import fast_cycle
import fast_stage

STAGE_LABELS = {0: "Insufficient Data", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
STAGE_DESCRIPTIONS = {
    "A": "Bare soil / planting / emergence",
    "B": "Greenup / rapid growth",
    "C": "Peak maturity",
    "D": "Senescence",
    "E": "Post-harvest / residue / bare soil",
    "Insufficient Data": "Insufficient data for reliable estimation",
}


def _day(x):
    return np.datetime64(pd.Timestamp(x).tz_localize(None).normalize().date(), "D").astype(np.int64)


def _series(dates, ndvi):
    d = pd.to_datetime(pd.Series(list(dates)))
    if d.dt.tz is not None:
        d = d.dt.tz_localize(None)
    days = d.dt.normalize().values.astype("datetime64[D]").astype(np.int64)
    v = np.asarray(ndvi, dtype=np.float64)
    if len(days) != len(v):
        raise ValueError("dates and ndvi must have the same length")
    ok = np.isfinite(v)
    days, v = days[ok], v[ok]
    order = np.argsort(days, kind="stable")
    days, v = days[order], v[order]
    if len(days) and np.any(np.diff(days) == 0):
        raise ValueError("dates must be unique at day precision; combine same-day observations first")
    return days, v


def _date(day):
    return pd.NaT if np.isnan(day) else pd.Timestamp(np.datetime64(int(day), "D"))


def crop_cycle(dates, ndvi, year):
    """Main growing season of `year` from crop-cycle-detection's crop_cycle_model.

    The model finds every season in the series; this returns the one whose NDVI
    peak falls in April-October of `year` with the highest peak, plus how many
    such seasons there were. Give the series some months before and after the
    season so the model can find the winter lows on both sides.

    Returns a dict: season_start, greenup, ndvi_peak, season_end (Timestamps or NaT),
    ndvi_peak_value, season_length (days), n_seasons. All values are NaN/NaT and
    n_seasons is NaN when the model detects no season at all.
    """
    days, v = _series(dates, ndvi)
    out = np.full(7, np.nan)
    if len(days) >= 2:
        out = fast_cycle.cycle_pixel(days, v, _day(f"{year}-01-01"), _day(f"{year}-04-01"),
                                     _day(f"{year}-10-31"))
    jan1 = _day(f"{year}-01-01")
    as_date = lambda doy: _date(np.nan if np.isnan(doy) else jan1 + doy - 1)
    return {
        "season_start": as_date(out[0]),
        "greenup": as_date(out[1]),
        "ndvi_peak": as_date(out[2]),
        "ndvi_peak_value": float(out[3]),
        "season_end": as_date(out[4]),
        "season_length": float(out[5]),
        "n_seasons": float(out[6]),
    }


def crop_stages(dates, ndvi, as_of, lookback_days=150):
    """Crop stage (A-E or 'Insufficient Data') on each date in `as_of`.

    Same as crop-stage-detection's run_crop_stage_from_dataframe applied to the
    observations dated on or before that date and within the previous
    `lookback_days` days (150 is the model's default lookback).
    """
    days, v = _series(dates, ndvi)
    single = isinstance(as_of, (str, pd.Timestamp, np.datetime64)) or not hasattr(as_of, "__len__")
    targets = np.array([_day(a) for a in ([as_of] if single else as_of)], dtype=np.int64)
    if len(days) == 0:
        codes = np.zeros(len(targets), np.uint8)
    else:
        codes = fast_stage.stage_weeks(days, v, targets, lookback_days, 0)
    labels = [STAGE_LABELS[int(c)] for c in codes]
    return labels[0] if single else labels
