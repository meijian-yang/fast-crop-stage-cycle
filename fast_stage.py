"""Numba re-implementation of nasaharvest/crop-stage-detection for many weeks at once.

Derived from crop_stage.py in https://github.com/nasaharvest/crop-stage-detection
(commit 2005f53), Copyright 2026 Ran Pelta, Agmatix / GrowersTech, licensed under the
Apache License, Version 2.0 (see LICENSE and NOTICE).
Modifications Copyright 2026 Meijian Yang, Columbia University & NASA GISS.
Changes from the original: the preprocessing (daily resample, PCHIP gap-fill,
Whittaker smoothing) and estimate_stage_adaptive were re-written as numba-compiled
functions with the default parameters fixed, and evaluated for all weeks of a pixel
in one call. The algorithm and thresholds are unchanged.

Reproduces, step for step, what run_models.run_cell does with the original code
for each week ending ``wend``:

  sub = obs with day + lag <= wend and day > wend - lookback    (run_models: lag=2)
  run_crop_stage_from_dataframe(sub)                              (crop_stage.py)
    smooth_daily_interpolate_ndvi: daily resample, PCHIP gap-fill
        (scipy PchipInterpolator), Whittaker smooth (lam=6000, d=2)
    estimate_stage_adaptive with the default parameters

Dates are integer day numbers. Validated against the original by validate_fast_stage.py.
"""
import numpy as np
from numba import njit

UPPER_PERCENTILE = 90.0
MIN_PEAK_NDVI = 0.50
LOWER_THRESHOLD = 0.35
MIN_PEAK_WIDTH = 5
MEASUREMENT_NOISE = 0.005
PROCESS_NOISE_LEVEL = 0.001
PROCESS_NOISE_VELOCITY = 0.0001
MIN_OBSERVATIONS = 30
WHITTAKER_LAM = 6000.0


@njit(cache=True)
def _sign(v):
    return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)


@njit(cache=True)
def _edge_slope(h0, h1, m0, m1):
    # scipy PchipInterpolator._edge_case
    d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if _sign(d) != _sign(m0):
        return 0.0
    if _sign(m0) != _sign(m1) and abs(d) > 3.0 * abs(m0):
        return 3.0 * m0
    return d


@njit(cache=True)
def _pchip_daily(x, y):
    """PCHIP through (x, y) evaluated at every integer day x[0]..x[-1]."""
    n = len(x)
    h = np.empty(n - 1)
    m = np.empty(n - 1)
    for k in range(n - 1):
        h[k] = x[k + 1] - x[k]
        m[k] = (y[k + 1] - y[k]) / h[k]
    d = np.empty(n)
    if n == 2:
        d[0] = m[0]
        d[1] = m[0]
    else:
        # scipy PchipInterpolator._find_derivatives
        for k in range(1, n - 1):
            if _sign(m[k]) != _sign(m[k - 1]) or m[k] == 0 or m[k - 1] == 0:
                d[k] = 0.0
            else:
                w1 = 2 * h[k] + h[k - 1]
                w2 = h[k] + 2 * h[k - 1]
                d[k] = 1.0 / ((w1 / m[k - 1] + w2 / m[k]) / (w1 + w2))
        d[0] = _edge_slope(h[0], h[1], m[0], m[1])
        d[n - 1] = _edge_slope(h[n - 2], h[n - 3], m[n - 2], m[n - 3])

    length = int(x[n - 1] - x[0]) + 1
    out = np.empty(length)
    for k in range(n - 1):
        x0 = x[k]
        hk = h[k]
        i0 = int(x0 - x[0])
        i1 = int(x[k + 1] - x[0])
        out[i0] = y[k]  # observed days keep their value (pandas only fills NaNs)
        for i in range(i0 + 1, i1):
            t = (x[0] + i - x0) / hk
            t2 = t * t
            t3 = t2 * t
            out[i] = ((2 * t3 - 3 * t2 + 1) * y[k] + (t3 - 2 * t2 + t) * hk * d[k]
                      + (-2 * t3 + 3 * t2) * y[k + 1] + (t3 - t2) * hk * d[k + 1])
    out[length - 1] = y[n - 1]
    return out


@njit(cache=True)
def _whittaker(y, lam):
    """Solve (I + lam * D'D) z = y, D = 2nd differences, by banded LDL'."""
    n = len(y)
    # D'D is pentadiagonal; build the three bands of A = I + lam D'D
    a0 = np.empty(n)
    a1 = np.zeros(n)  # a1[i] = A[i, i+1]
    a2 = np.zeros(n)  # a2[i] = A[i, i+2]
    for i in range(n):
        c = 6.0
        if i == 0 or i == n - 1:
            c = 1.0
        elif i == 1 or i == n - 2:
            c = 5.0
        a0[i] = 1.0 + lam * c
    for i in range(n - 1):
        c = -4.0
        if i == 0 or i == n - 2:
            c = -2.0
        a1[i] = lam * c
    for i in range(n - 2):
        a2[i] = lam * 1.0
    if n == 3:  # D'D for a single second difference: [1,-2,1] outer product
        a0[0] = 1 + lam; a0[1] = 1 + 4 * lam; a0[2] = 1 + lam
        a1[0] = -2 * lam; a1[1] = -2 * lam

    # LDL' factorisation: L unit lower with bands l1, l2
    dd = np.empty(n)
    l1 = np.zeros(n)
    l2 = np.zeros(n)
    for i in range(n):
        s = a0[i]
        if i >= 1:
            s -= l1[i - 1] * l1[i - 1] * dd[i - 1]
        if i >= 2:
            s -= l2[i - 2] * l2[i - 2] * dd[i - 2]
        dd[i] = s
        if i + 1 < n:
            s1 = a1[i]
            if i >= 1:
                s1 -= l1[i - 1] * l2[i - 1] * dd[i - 1]
            l1[i] = s1 / dd[i]
        if i + 2 < n:
            l2[i] = a2[i] / dd[i]
    z = np.empty(n)
    for i in range(n):
        s = y[i]
        if i >= 1:
            s -= l1[i - 1] * z[i - 1]
        if i >= 2:
            s -= l2[i - 2] * z[i - 2]
        z[i] = s
    for i in range(n):
        z[i] /= dd[i]
    for i in range(n - 1, -1, -1):
        s = z[i]
        if i + 1 < n:
            s -= l1[i] * z[i + 1]
        if i + 2 < n:
            s -= l2[i] * z[i + 2]
        z[i] = s
    return z


@njit(cache=True)
def _kalman_velocity(v):
    # crop_stage._kalman_velocity with dt = 1 (daily series)
    x0 = v[0]
    x1 = 0.0
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0
    for i in range(1, len(v)):
        xp0 = x0 + x1
        xp1 = x1
        # F P F' + Q with F = [[1,1],[0,1]]
        q00 = p00 + p01 + p10 + p11 + PROCESS_NOISE_LEVEL
        q01 = p01 + p11
        q10 = p10 + p11
        q11 = p11 + PROCESS_NOISE_VELOCITY
        innov = v[i] - xp0
        s = q00 + MEASUREMENT_NOISE
        k0 = q00 / s
        k1 = q10 / s
        x0 = xp0 + k0 * innov
        x1 = xp1 + k1 * innov
        p00 = q00 - k0 * q00
        p01 = q01 - k0 * q01
        p10 = q10 - k1 * q00
        p11 = q11 - k1 * q01
    return x1


@njit(cache=True)
def _stage_code(z):
    """estimate_stage_adaptive -> 0 insufficient, 1..5 = A..E."""
    n = len(z)
    if n < MIN_OBSERVATIONS:
        return 0
    upper = max(np.percentile(z, UPPER_PERCENTILE), MIN_PEAK_NDVI)
    cur = z[n - 1]
    if cur >= upper:
        return 3
    vel = _kalman_velocity(z)
    had_peak = False
    run = 0
    for i in range(n):
        if z[i] >= upper:
            run += 1
        else:
            if run >= MIN_PEAK_WIDTH:
                had_peak = True
            run = 0
    if run >= MIN_PEAK_WIDTH:
        had_peak = True
    if cur >= LOWER_THRESHOLD:
        return 4 if (had_peak and vel <= 0) else 2
    return 5 if had_peak else 1


@njit(cache=True)
def stage_weeks(days, ndvi, week_ends, lookback, lag=2):
    """Stage code per week for one pixel. days must be sorted, unique integers.

    For each date in week_ends, uses observations with day + lag <= date and
    day > date - lookback. lag=2 is for 5-day composites dated at their centre
    (only bins complete by that date); use lag=0 for dated observations.
    """
    out = np.zeros(len(week_ends), np.uint8)
    for j in range(len(week_ends)):
        we = week_ends[j]
        lo = -1
        hi = -1
        for i in range(len(days)):
            if days[i] + lag <= we and days[i] > we - lookback:
                if lo < 0:
                    lo = i
                hi = i
        if lo < 0 or hi - lo + 1 < 3:
            continue
        daily = _pchip_daily(days[lo:hi + 1].astype(np.float64), ndvi[lo:hi + 1])
        if len(daily) < MIN_OBSERVATIONS:
            continue
        out[j] = _stage_code(_whittaker(daily, WHITTAKER_LAM))
    return out
