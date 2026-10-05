"""Numba re-implementation of nasaharvest/crop-cycle-detection for one pixel.

Derived from src/crop_cycle.py and src/ndvi_preprocessing.py in
https://github.com/nasaharvest/crop-cycle-detection (commit 293f906),
Copyright 2026 Ran Pelta, Agmatix / GrowersTech, licensed under the Apache License,
Version 2.0 (see LICENSE and NOTICE).
Modifications Copyright 2026 Meijian Yang, Columbia University & NASA GISS.
Changes from the original: crop_cycle_model with its default parameters was
re-written as numba-compiled functions on integer day numbers, and the season
selection done in run_models.cycle_vector is folded in. The algorithm and
thresholds are unchanged. Validated against the original by validate_fast_cycle.py.

Steps (crop_cycle_model):
  1. clean_ndvi_timeseries: local linear regression (Gaussian kernel, k=4 days),
     drop observations with signed squared error <= -0.015, then drop 3-point
     downward spikes
  2. daily resample, PCHIP gap-fill, Whittaker smooth (lam=6000)
  3. adaptive peak threshold = clip(p90 of smoothed NDVI, 0.5, 0.7)
  4. scipy argrelmax / argrelmin (order=5, mode='clip'); peaks > threshold,
     troughs < 0.5
  5. trough-to-trough seasons (+ orphan peaks before the first / after the last trough)
  6. season start / end from peak-relative (35%) then fixed (0.35 / 0.40) thresholds,
     green-up as the steepest rise with NDVI in [0, 0.5]
  7. de-duplicate by start then end (keep highest peak), drop seasons < 60 days,
     clip overlapping starts
"""
import numpy as np
from numba import njit

from fast_stage import _pchip_daily, _whittaker, WHITTAKER_LAM

K_OUTLIER = 4.0
DEV_THRESHOLD_LOWER = -0.015
MAX_DAYS_FOR_SPIKES = 30
SPIKE_THRESHOLD = 0.06
MIN_NDVI_FOR_SPIKE = 0.5
START_FIXED = 0.35
END_FIXED = 0.40
START_PCT = 0.35
END_PCT = 0.35
PEAK_PERCENTILE = 90.0
PEAK_CEILING = 0.7
TROUGH_MAX = 0.5
EXTREMA_ORDER = 5
WINDOW_DAYS = 90
MIN_SEASON_LENGTH = 60
GREENUP_MIN = 0.3
GREENUP_MAX = 0.6
GREENUP_NDVI_MAX = 0.5

NAN = np.nan


@njit(cache=True)
def _clean(days, v):
    """clean_ndvi_timeseries -> boolean keep mask."""
    n = len(days)
    x = (days - days[0]).astype(np.float64)
    keep = np.ones(n, np.bool_)
    for p in range(n):
        # local_regression: weighted least squares with pinv, X = [1, x]
        s0 = s1 = s2 = t0 = t1 = 0.0
        for i in range(n):
            w = np.exp(-((x[i] - x[p]) ** 2) / (2.0 * K_OUTLIER * K_OUTLIER))
            s0 += w
            s1 += w * x[i]
            s2 += w * x[i] * x[i]
            t0 += w * v[i]
            t1 += w * x[i] * v[i]
        A = np.empty((2, 2))
        A[0, 0] = s0; A[0, 1] = s1; A[1, 0] = s1; A[1, 1] = s2
        Ai = np.linalg.pinv(A)
        b0 = Ai[0, 0] * t0 + Ai[0, 1] * t1
        b1 = Ai[1, 0] * t0 + Ai[1, 1] * t1
        smooth = b0 + x[p] * b1
        r = v[p] - smooth
        signed = r * r * (1.0 if r > 0 else -1.0)
        if signed <= DEV_THRESHOLD_LOWER:
            keep[p] = False
    # three-point downward spikes on the remaining observations
    idx = np.nonzero(keep)[0]
    spike = np.zeros(n, np.bool_)
    for j in range(1, len(idx) - 1):
        a, b, c = idx[j - 1], idx[j], idx[j + 1]
        if days[c] - days[a] > MAX_DAYS_FOR_SPIKES:
            continue
        if ((v[a] - v[b]) > SPIKE_THRESHOLD and (v[b] - v[c]) < -SPIKE_THRESHOLD
                and v[a] > MIN_NDVI_FOR_SPIKE and v[c] > MIN_NDVI_FOR_SPIKE):
            spike[b] = True
    for i in range(n):
        if spike[i]:
            keep[i] = False
    return keep


@njit(cache=True)
def _is_ext(z, i, want_max):
    n = len(z)
    for s in range(1, EXTREMA_ORDER + 1):
        p = min(i + s, n - 1)
        m = max(i - s, 0)
        if want_max:
            if not (z[i] > z[p] and z[i] > z[m]):
                return False
        else:
            if not (z[i] < z[p] and z[i] < z[m]):
                return False
    return True


@njit(cache=True)
def _round2(x):
    return np.round(x, 2)


@njit(cache=True)
def _stable_desc_order(vals):
    """pandas sort_values(ascending=False) order: ties keep their original order."""
    n = len(vals)
    order = np.arange(n)
    for i in range(1, n):  # insertion sort, stable
        j = i
        while j > 0 and vals[order[j - 1]] < vals[order[j]]:
            order[j - 1], order[j] = order[j], order[j - 1]
            j -= 1
    return order


@njit(cache=True)
def _dedupe(key, pv):
    """Rows kept by: rows with key sorted by pv desc, drop_duplicates(key), then NaN-key rows.
    Returns the new row order (indices)."""
    n = len(key)
    has = np.array([not np.isnan(key[i]) for i in range(n)])
    hidx = np.nonzero(has)[0]
    order = hidx[_stable_desc_order(pv[hidx])]
    out = np.empty(n, np.int64)
    m = 0
    for i in order:
        dup = False
        for j in range(m):
            if key[out[j]] == key[i]:
                dup = True
                break
        if not dup:
            out[m] = i
            m += 1
    for i in range(n):
        if not has[i]:
            out[m] = i
            m += 1
    return out[:m]


@njit(cache=True)
def cycle_pixel(days, ndvi, year_jan1, apr1, oct31):
    """One pixel -> [start_doy, greenup_doy, peak_doy, peak_value, end_doy, length, n_seasons]
    exactly as run_models.cycle_vector(crop_cycle_model(df)). days: sorted unique ints."""
    out = np.full(7, NAN)
    keep = _clean(days, ndvi)
    d = days[keep].astype(np.float64)
    v = ndvi[keep]
    if len(d) < 2:
        return out
    z = _whittaker(_pchip_daily(d, v), WHITTAKER_LAM)
    n = len(z)
    d0 = d[0]

    thr = min(max(np.percentile(z, PEAK_PERCENTILE), 0.5), PEAK_CEILING)
    pk = np.empty(n, np.int64)
    tr = np.empty(n, np.int64)
    npk = ntr = 0
    for i in range(n):
        if _is_ext(z, i, True) and z[i] > thr:
            pk[npk] = i
            npk += 1
        if _is_ext(z, i, False) and z[i] < TROUGH_MAX:
            tr[ntr] = i
            ntr += 1
    if npk == 0 or ntr == 0:
        return out

    # trough-to-trough seasons: (trough_before, trough_after, peak_value, peak_index)
    tb = np.empty(npk + ntr, np.float64)
    ta = np.empty(npk + ntr, np.float64)
    pv = np.empty(npk + ntr, np.float64)
    pi = np.empty(npk + ntr, np.int64)
    ns = 0
    for t in range(ntr - 1):
        best = -1
        for q in range(npk):
            if tr[t] < pk[q] < tr[t + 1]:
                if best < 0 or z[pk[q]] > z[pk[best]]:
                    best = q
        if best >= 0:
            tb[ns] = tr[t]; ta[ns] = tr[t + 1]; pv[ns] = _round2(z[pk[best]]); pi[ns] = pk[best]
            ns += 1
    for q in range(npk):
        if pk[q] < tr[0]:
            tb[ns] = NAN; ta[ns] = tr[0]; pv[ns] = _round2(z[pk[q]]); pi[ns] = pk[q]
            ns += 1
    for q in range(npk):
        if pk[q] > tr[ntr - 1]:  # the original does not round these
            tb[ns] = tr[ntr - 1]; ta[ns] = NAN; pv[ns] = z[pk[q]]; pi[ns] = pk[q]
            ns += 1
    if ns == 0:  # the original fails on pd.concat([]) here; run_models records NaN
        return out
    o = np.argsort(pi[:ns])
    tb, ta, pv, pi = tb[:ns][o], ta[:ns][o], pv[:ns][o], pi[:ns][o]

    # milestones, as indices into the daily series (NaN = NaT)
    st = np.full(ns, NAN)
    en = np.full(ns, NAN)
    gu = np.full(ns, NAN)
    for s in range(ns):
        p = pi[s]
        lo = tb[s] if not np.isnan(tb[s]) else p - WINDOW_DAYS
        for th in (pv[s] * START_PCT, START_FIXED):
            for i in range(p - 1, -1, -1):
                if i < lo:
                    break
                if z[i] <= th:
                    st[s] = i
                    break
            if not np.isnan(st[s]):
                break
        if np.isnan(st[s]) and not np.isnan(tb[s]):
            st[s] = tb[s] + 3
        hi = ta[s] if not np.isnan(ta[s]) else p + WINDOW_DAYS
        for th in (pv[s] * END_PCT, END_FIXED):
            for i in range(p + 1, n):
                if i > hi:
                    break
                if z[i] <= th:
                    en[s] = i
                    break
            if not np.isnan(en[s]):
                break
        if np.isnan(en[s]) and not np.isnan(ta[s]):
            en[s] = ta[s] - 3

        only_2nd = (not np.isnan(ta[s])) and np.isnan(tb[s])
        if not only_2nd and not np.isnan(st[s]):
            # get_dates_for_greenup: window = [start, peak] (inclusive, clipped to series)
            a = max(int(st[s]), 0)
            g_start = NAN
            g_end = NAN
            for i in range(a, min(p, n - 1) + 1):
                if z[i] <= GREENUP_MIN:
                    g_start = i
                if z[i] <= GREENUP_MAX:
                    g_end = i
            if np.isnan(g_start):
                g_start = st[s]
            if not np.isnan(g_end):
                ga = max(int(g_start), 0)
                gb = min(int(g_end), n - 1)
                m = gb - ga + 1
                if m >= 2:
                    seg = z[ga:gb + 1]
                    grad = np.empty(m)  # np.gradient: one-sided at the ends, centred inside
                    grad[0] = seg[1] - seg[0]
                    grad[m - 1] = seg[m - 1] - seg[m - 2]
                    for i in range(1, m - 1):
                        grad[i] = (seg[i + 1] - seg[i - 1]) / 2.0
                    best = -1
                    for i in range(m):
                        if seg[i] >= 0 and seg[i] <= GREENUP_NDVI_MAX and grad[i] > 0:
                            if best < 0 or grad[i] > grad[best]:
                                best = i
                    if best >= 0:
                        gu[s] = ga + best

    # de-duplicate by start, then by end (keep highest peak value)
    o1 = _dedupe(st, pv)
    st, en, gu, pv, pi = st[o1], en[o1], gu[o1], pv[o1], pi[o1]
    o2 = _dedupe(en, pv)
    st, en, gu, pv, pi = st[o2], en[o2], gu[o2], pv[o2], pi[o2]
    ns = len(st)
    ln = en - st
    keepm = np.array([np.isnan(ln[i]) or ln[i] >= MIN_SEASON_LENGTH for i in range(ns)])
    st, en, gu, pv, pi = st[keepm], en[keepm], gu[keepm], pv[keepm], pi[keepm]
    if len(st) == 0:  # empty report -> run_models records NaN, not 0 seasons
        return out
    o3 = np.argsort(pi)
    st, en, gu, pv, pi = st[o3], en[o3], gu[o3], pv[o3], pi[o3]
    ns = len(st)
    for i in range(1, ns):  # _clip_season_overlaps
        if not np.isnan(en[i - 1]) and not np.isnan(st[i]) and st[i] <= en[i - 1]:
            st[i] = en[i - 1] + 3
    ln = en - st

    # run_models.cycle_vector: seasons peaking Apr-Oct of the target year, highest peak
    best = -1
    cnt = 0
    for s in range(ns):
        pday = d0 + pi[s]
        if apr1 <= pday <= oct31:
            cnt += 1
            if best < 0 or pv[s] > pv[best]:
                best = s
    out[6] = cnt
    if best >= 0:
        off = d0 - year_jan1 + 1
        out[0] = st[best] + off
        out[1] = gu[best] + off
        out[2] = pi[best] + off
        out[3] = pv[best]
        out[4] = en[best] + off
        out[5] = ln[best]
    return out
