"""Shared grid / time-bin definitions for the gridded crop stage + cycle products."""
import json
import os

import pandas as pd
import rasterio
from rasterio.transform import Affine


def _use_bundled_proj():
    """Point PROJ at the database shipped inside the rasterio wheel, if there is one.
    Some conda environments set PROJ_DATA to an older PROJ database, which makes
    EPSG codes fail to resolve; the bundled copy always matches rasterio."""
    bundled = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
    if os.path.isdir(bundled):
        # rasterio re-reads PROJ_DATA whenever it opens a dataset, so set both
        os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = bundled
        from rasterio._env import set_proj_data_search_path
        set_proj_data_search_path(bundled)


_use_bundled_proj()

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "products")
FIGURES = os.path.join(ROOT, "figures")
# Where the two NASA Harvest repos are cloned (see README)
HARVEST_REPOS = os.environ.get("HARVEST_REPOS", os.path.join(ROOT, "external"))

# 1 km grids in NAD83 / CONUS Albers (EPSG:5070, same CRS as the USDA NASS gridded
# Crop Progress & Condition layers). Every region is a window of the Iowa grid,
# so outputs from different regions line up.
CRS = "EPSG:5070"
RES = 1000
REGIONS = {
    # name: upper-left corner, size in 1 km cells, TIGER state / county FIPS for the
    # boundary, and a display name for figures. Add counties with
    #   python gee_ndvi.py add-region --state <name|FIPS> --county <name|FIPS>
    # which writes them to regions.json (merged below).
    # "iowa" defines the parent grid; it is too large for the direct (computePixels) path.
    "iowa": dict(x0=-60_000, y0=2_300_000, ncol=560, nrow=380, statefp="19", countyfp=None,
                 label="Iowa"),
    "boone": dict(x0=150_000, y0=2_137_000, ncol=40, nrow=40, statefp="19", countyfp="015",
                  label="Boone County, Iowa"),
}

REGIONS_FILE = os.path.join(ROOT, "regions.json")
if os.path.exists(REGIONS_FILE):
    with open(REGIONS_FILE) as _f:
        REGIONS.update(json.load(_f))

# USDA NASS gridded Crop Progress & Condition (CPC) 9 km grid, taken from
# cornProg24w20.tif. Pixel width and height differ slightly, and differ a little
# between CPC files; pass --cpc-ref to run_models.py to align to a specific file.
CPC_9KM_TRANSFORM = Affine(8999.255456289853, 0, -2309800.213402343,
                           0, -8995.486488541766, 3185470.286793155)

# CDL codes
CROPS = {"corn": 1, "soy": 5}

BIN_DAYS = 5


def grid(region):
    g = REGIONS[region]
    return g["x0"], g["y0"], g["ncol"], g["nrow"], Affine(RES, 0, g["x0"], 0, -RES, g["y0"])


def time_bins(year):
    """5-day bins from Nov 1 of the previous year to Jan 31 of the next year.

    The padding on either side lets the cycle model find the bounding
    winter troughs of the target season.
    """
    start = pd.Timestamp(f"{year - 1}-11-01")
    end = pd.Timestamp(f"{year + 1}-02-01")
    starts = pd.date_range(start, end - pd.Timedelta(days=BIN_DAYS), freq=f"{BIN_DAYS}D")
    return pd.DataFrame({
        "start": starts,
        "end": starts + pd.Timedelta(days=BIN_DAYS),  # exclusive
        "center": starts + pd.Timedelta(days=BIN_DAYS // 2),
    })


def nass_weeks(year, first=14, last=47):
    """NASS crop progress weeks: ISO week number -> week-ending Sunday."""
    return {w: pd.Timestamp.fromisocalendar(year, w, 7) for w in range(first, last + 1)}


def ndvi_path(region, year, res="1km"):
    return os.path.join(DATA, f"{region}_ndvi_{res}_{year}.npz")


def boundary_path(region):
    return os.path.join(DATA, f"{region}_boundary_5070.geojson")
