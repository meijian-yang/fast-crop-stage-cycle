"""Map figures for the 30 m crop cycle / stage products of a region.

Usage: python plot_maps.py [--region boone] [--year 2024]
Writes PNGs to figures/.
"""
import argparse
import json
import os

import common as C  # sets PROJ_DATA before rasterio is used

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from rasterio.features import geometry_mask

# --- palette (dataviz reference instance, light mode) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
BASELINE = "#c3c2b7"
NONCROP = "#ecebe6"  # neutral fill for non-crop land inside the region
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]  # blue 100..700
STAGE_COLORS = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # ordinal blue 250/400/500/600/700, validated
STAGE_NAMES = ["A  Bare soil / planting / emergence", "B  Green-up / rapid growth",
               "C  Peak", "D  Senescence", "E  Post-harvest / residue"]
CROP_NAMES = {"corn": "Corn", "soy": "Soybean"}

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": BASELINE,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

SEQ_CMAP = LinearSegmentedColormap.from_list("blue_seq", SEQ)


def load(path):
    with rasterio.open(path) as r:
        return r.read(), r.transform, r.descriptions


def county(region, transform, shape):
    geom = json.load(open(C.boundary_path(region)))["geometry"]
    inside = ~geometry_mask([geom], out_shape=shape, transform=transform)
    return geom, inside


def extent_km(transform, shape):
    x0, y0 = transform.c, transform.f
    return [x0 / 1e3, (x0 + transform.a * shape[1]) / 1e3, (y0 + transform.e * shape[0]) / 1e3, y0 / 1e3]


def draw_base(ax, geom, inside, transform, shape):
    bg = np.where(inside, 1.0, np.nan)
    ax.imshow(bg, extent=extent_km(transform, shape), cmap=ListedColormap([NONCROP]),
              interpolation="nearest", zorder=0)
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    rings = [np.array(p[0]) / 1e3 for p in polys]  # outer rings
    for ring in rings:
        ax.plot(ring[:, 0], ring[:, 1], color=MUTED, lw=0.8, zorder=3)
    xs = np.concatenate([r[:, 0] for r in rings])
    ys = np.concatenate([r[:, 1] for r in rings])
    pad = 1.0
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - 5.0, ys.max() + pad)  # room below the county for the scale bar
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)  # later imshow calls must not reset the padded limits
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def scale_bar(ax, km=10):
    x0, x1 = ax.get_xlim(); y0, _ = ax.get_ylim()
    xs = x0 + 1.5
    ax.plot([xs, xs + km], [y0 + 1.8] * 2, color=INK2, lw=2, solid_capstyle="butt")
    ax.text(xs + km + 0.8, y0 + 1.8, f"{km} km", ha="left", va="center", fontsize=8, color=INK2)


def doy_to_date(doy, year):
    return pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=float(doy) - 1)


def date_colorbar(fig, ax, im, lo, hi, year, label):
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03, aspect=30)
    span = hi - lo
    step = 7 if span <= 50 else 14
    first = doy_to_date(lo, year)
    ticks = []
    t = first + pd.Timedelta(days=(7 - first.dayofweek) % 7)
    while (t - pd.Timestamp(f"{year}-01-01")).days + 1 <= hi:
        ticks.append((t - pd.Timestamp(f"{year}-01-01")).days + 1)
        t += pd.Timedelta(days=step)
    cb.set_ticks(ticks)
    cb.set_ticklabels([doy_to_date(v, year).strftime("%b %-d") for v in ticks])
    cb.ax.tick_params(labelsize=8, color=BASELINE, length=3)
    cb.outline.set_visible(False)
    if label:
        cb.set_label(label, fontsize=9, color=INK2)


def fig_cycle(region, year, odir):
    metrics = [(0, "Season start", "NDVI begins rising"),
               (2, "NDVI peak", "date of maximum greenness"),
               (4, "Season end", "NDVI back down after senescence / harvest")]
    data = {c: load(os.path.join(C.OUT, region, "30m", f"cycle_{c}_{year}.tif")) for c in C.CROPS}
    _, tr, _ = data["corn"]
    shape = data["corn"][0].shape[1:]
    geom, inside = county(region, tr, shape)

    fig, axes = plt.subplots(2, 3, figsize=(13, 11.2))
    for j, (band, title, sub) in enumerate(metrics):
        for i, crop in enumerate(C.CROPS):
            ax = axes[i, j]
            draw_base(ax, geom, inside, tr, shape)
            a = data[crop][0][band]
            lo, hi = np.percentile(a[np.isfinite(a)], [2, 98])
            lo, hi = np.floor(lo), np.ceil(hi)
            im = ax.imshow(a, extent=extent_km(tr, shape), cmap=SEQ_CMAP, vmin=lo, vmax=hi,
                           interpolation="nearest", zorder=1)
            med = doy_to_date(np.nanmedian(a), year).strftime("%b %-d")
            ax.set_title(f"{CROP_NAMES[crop]}  ·  median {med}", fontsize=10, color=INK2, loc="left", pad=4)
            if i == 0:
                ax.text(0, 1.10, title, transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK)
                ax.text(0, 1.055, sub, transform=ax.transAxes, fontsize=9, color=INK2)
            date_colorbar(fig, ax, im, lo, hi, year, "")
            if i == 1 and j == 0:
                scale_bar(ax)
    fig.suptitle(f"{C.REGIONS[region]['label']} — {year} crop cycle at 30 m", x=0.02, y=0.995, ha="left",
                 fontsize=16, fontweight="bold")
    fig.text(0.02, 0.955, "NASA Harvest / Agmatix crop-cycle model on Sentinel-2 + Landsat 8/9 NDVI. "
             "Crop pixels from USDA CDL; gray = other land cover. Each map has its own date scale.",
             fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.03, wspace=0.06, hspace=0.16)
    path = os.path.join(odir, f"{region}_cycle_30m_{year}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_stage_maps(region, year, odir, weeks=(24, 28, 34, 38)):
    data = {c: load(os.path.join(C.OUT, region, "30m", f"stage_{c}_{year}_weekly.tif")) for c in C.CROPS}
    _, tr, desc = data["corn"]
    shape = data["corn"][0].shape[1:]
    geom, inside = county(region, tr, shape)
    wk = list(C.nass_weeks(year))
    cmap = ListedColormap([BASELINE] + STAGE_COLORS)  # 0 = insufficient data
    norm = BoundaryNorm(np.arange(-0.5, 6.5), cmap.N)

    fig, axes = plt.subplots(2, len(weeks), figsize=(3.4 * len(weeks) + 0.5, 8.6))
    for j, w in enumerate(weeks):
        b = wk.index(w)
        for i, crop in enumerate(C.CROPS):
            ax = axes[i, j]
            draw_base(ax, geom, inside, tr, shape)
            a = data[crop][0][b].astype(float)
            a[a == 255] = np.nan
            ax.imshow(a, extent=extent_km(tr, shape), cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
            if i == 0:
                end = C.nass_weeks(year)[w]
                ax.set_title(f"Week {w}  ·  {end:%b %-d}", fontsize=12, fontweight="bold", color=INK,
                             loc="left", pad=6)
            if j == 0:
                ax.text(-0.06, 0.5, CROP_NAMES[crop], transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=11, fontweight="bold")
    scale_bar(axes[1, 0])
    handles = [Patch(facecolor=c, edgecolor="none", label=n) for c, n in zip(STAGE_COLORS, STAGE_NAMES)]
    handles.append(Patch(facecolor=NONCROP, edgecolor="none", label="Other land cover"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10,
               handlelength=1.4, handleheight=1.0, columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{C.REGIONS[region]['label']} — {year} weekly crop stage at 30 m", x=0.02, y=0.99, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(0.02, 0.935, "NASA Harvest / Agmatix crop-stage model, evaluated each NASS week (ending Sunday) "
             "using only imagery up to that date.", fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.86, bottom=0.12, wspace=0.03, hspace=0.08)
    path = os.path.join(odir, f"{region}_stage_maps_30m_{year}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_stage_timeline(region, year, odir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, crop in zip(axes, C.CROPS):
        s = pd.read_csv(os.path.join(C.OUT, region, "30m", f"stage_{crop}_{year}_summary.csv"),
                        parse_dates=["week_ending"])
        x = s["week_ending"]
        ys = [s[k].to_numpy() * 100 for k in "ABCDE"]
        ax.stackplot(x, ys, colors=STAGE_COLORS, edgecolor=SURFACE, linewidth=1.5)
        # direct labels at each stage's widest week
        cum = np.cumsum(ys, axis=0)
        for k, y in enumerate(ys):
            if y.max() < 25:
                continue
            near = np.nonzero(y >= 0.85 * y.max())[0]
            t = int(near[len(near) // 2])  # middle of the stretch where this stage dominates
            mid = cum[k][t] - y[t] / 2
            ax.text(x.iloc[t], mid, "ABCDE"[k], ha="center", va="center", fontsize=11,
                    fontweight="bold", color=INK if k == 0 else "#ffffff")
        ax.set_title(CROP_NAMES[crop], loc="left", fontsize=12, fontweight="bold", pad=6)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Share of crop pixels")
    handles = [Patch(facecolor=c, edgecolor="none", label=n) for c, n in zip(STAGE_COLORS, STAGE_NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9,
               handlelength=1.4, columnspacing=1.6)
    fig.suptitle(f"{C.REGIONS[region]['label']} — share of 30 m crop pixels in each stage, {year}", x=0.02, y=0.98,
                 ha="left", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.83, bottom=0.2, wspace=0.05)
    path = os.path.join(odir, f"{region}_stage_timeline_{year}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_peak(region, year, odir, band=2):
    data = {c: load(os.path.join(C.OUT, region, "30m", f"cycle_{c}_{year}.tif")) for c in C.CROPS}
    _, tr, _ = data["corn"]
    shape = data["corn"][0].shape[1:]
    geom, inside = county(region, tr, shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7.4))
    for ax, crop in zip(axes, C.CROPS):
        a = data[crop][0][band]
        lo, hi = np.percentile(a[np.isfinite(a)], [2, 98])
        lo, hi = np.floor(lo), np.ceil(hi)
        draw_base(ax, geom, inside, tr, shape)
        im = ax.imshow(a, extent=extent_km(tr, shape), cmap=SEQ_CMAP, vmin=lo, vmax=hi,
                       interpolation="nearest", zorder=1)
        med = doy_to_date(np.nanmedian(a), year).strftime("%b %-d")
        ax.set_title(f"{CROP_NAMES[crop]}  ·  median {med}", loc="left", fontsize=12,
                     fontweight="bold", color=INK, pad=6)
        date_colorbar(fig, ax, im, lo, hi, year, f"{CROP_NAMES[crop]} NDVI peak date")
    scale_bar(axes[0])
    fig.suptitle(f"{C.REGIONS[region]['label']} — {year} NDVI peak date at 30 m", x=0.02, y=0.98, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(0.02, 0.925, "Date of maximum greenness from the NASA Harvest / Agmatix crop-cycle model. "
             "Each map has its own date scale; gray = other land cover.", fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04, wspace=0.06)
    path = os.path.join(odir, f"{region}_peak_30m_{year}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="boone")
    ap.add_argument("--year", type=int, default=2024)
    a = ap.parse_args()
    odir = C.FIGURES
    os.makedirs(odir, exist_ok=True)
    for f in (fig_cycle, fig_stage_maps, fig_stage_timeline, fig_peak):
        print("wrote", f(a.region, a.year, odir))
