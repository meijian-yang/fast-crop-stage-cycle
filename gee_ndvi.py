"""Build NDVI time series for CDL corn / soybean pixels in Google Earth Engine.

Mirrors the NDVI recipe of nasaharvest/crop-stage-detection (src/gee_fetch.py):
  * Sentinel-2 L2A (S2_SR_HARMONIZED), pixels masked with Cloud Score+ cs > 0.6
  * Landsat 8/9 C2 L2, pixels kept where QA_PIXEL is 21824 or 21888 (clear)
  * crop pixels from the USDA Cropland Data Layer, eroded by one native CDL pixel
    (stand-in for the -10 m field buffer)
All clear observations in each 5-day bin are averaged on a 30 m grid.

  direct    1 km grid. Per bin and crop c in {corn, soy}:
              ndvi_c  mean NDVI over clear crop pixels in the cell
              fclr_c  fraction of the cell area that is clear crop pixels
            plus fcrop_c, the fraction of the cell area that is crop.
            -> data/<region>_ndvi_1km_<year>.npz
  direct30  30 m grid: NDVI per pixel and bin, plus a crop label per pixel
            (1 = corn, 5 = soybean, 0 = other).
            -> data/<region>_ndvi_30m_<year>.npz
Both also save the region outline to data/<region>_boundary_5070.geojson.

Both use interactive computePixels requests, which suits county-sized regions
(e.g. Boone County: ~1 min at 1 km, ~4 min at 30 m).

Usage:
  python gee_ndvi.py direct   --project <ee-project> --year 2024 --region boone
  python gee_ndvi.py direct30 --project <ee-project> --year 2024 --region boone
  # add a county as a new region (writes regions.json); names or FIPS codes both work
  python gee_ndvi.py add-region --project <ee-project> --state Iowa --county Story
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import ee
import numpy as np

import common as C

CHUNK = 6  # bins per 1 km computePixels request
MAX_DIRECT_CELLS = 100 * 100  # 1 km cells; larger regions time out or exceed request limits
# 30 m working grid (the pre-2024 CDL grid)
PROJ30_TRANSFORM = [30, 0, -2356095, 0, -30, 3172605]


def proj30():
    return ee.Projection(C.CRS, PROJ30_TRANSFORM)


def grid_transform(region):
    x0, y0, *_ = C.grid(region)
    return [C.RES, 0, x0, 0, -C.RES, y0]


def boundary(region):
    g = C.REGIONS[region]
    if g["countyfp"]:
        fc = ee.FeatureCollection("TIGER/2018/Counties").filter(ee.Filter.And(
            ee.Filter.eq("STATEFP", g["statefp"]), ee.Filter.eq("COUNTYFP", g["countyfp"])))
    else:
        fc = ee.FeatureCollection("TIGER/2018/States").filter(ee.Filter.eq("STATEFP", g["statefp"]))
    return fc.geometry()


def rectangle(region):
    x0, y0, ncol, nrow, _ = C.grid(region)
    x1, y1 = x0 + ncol * C.RES, y0 - nrow * C.RES
    return ee.Geometry.Rectangle([x0, y1, x1, y0], ee.Projection(C.CRS), False)


def crop_cores(year, region):
    """Per crop, the fraction (0-1) of each 30 m proj30() pixel covered by eroded CDL crop.

    The CDL is eroded by one native pixel first (10 m from 2024 on, 30 m before),
    standing in for the -10 m field buffer of the Harvest NDVI fetcher.
    """
    cdl = ee.Image(f"USDA/NASS/CDL/{year}").select("cropland")
    inside = ee.Image.constant(1).clip(boundary(region)).unmask(0)
    cores = {}
    for name, code in C.CROPS.items():
        core = (cdl.eq(code)
                .focal_min(radius=1, kernelType="square", units="pixels")
                .And(inside)
                .reproject(cdl.projection())
                .toFloat()
                .reduceResolution(ee.Reducer.mean(), maxPixels=64)
                .reproject(proj30()))
        cores[name] = core
    return cores


def ndvi_collection(start, end, geom):
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(start, end).filterBounds(geom)
          .linkCollection(cs, ["cs"])
          .filter(ee.Filter.listContains("system:band_names", "cs")))

    def s2_ndvi(img):
        return (img.normalizedDifference(["B8", "B4"]).rename("NDVI")
                .updateMask(img.select("cs").gt(0.6)))

    def ls_ndvi(img):
        sr = img.select(["SR_B4", "SR_B5"]).multiply(0.0000275).add(-0.2)
        clear = img.select("QA_PIXEL").remap([21824, 21888], [1, 1], 0)
        return sr.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI").updateMask(clear)

    col = s2.map(s2_ndvi)
    for ls in ["LC08", "LC09"]:
        col = col.merge(ee.ImageCollection(f"LANDSAT/{ls}/C02/T1_L2")
                        .filterDate(start, end).filterBounds(geom).map(ls_ndvi))
    return col


def to_grid(img, region):
    return (img.reduceResolution(ee.Reducer.mean(), maxPixels=2048)
            .reproject(crs=C.CRS, crsTransform=grid_transform(region)))


def bin_image(start, end, cores, region, k):
    col = ndvi_collection(start, end, rectangle(region))
    empty = ee.Image.constant(0).rename("NDVI").updateMask(0)
    # resample to the 30 m grid so 10 m Sentinel-2 pixels don't blow up the 1 km aggregation
    comp = ee.Image(ee.Algorithms.If(col.size().gt(0), col.mean(), empty)).reproject(proj30())
    clear = comp.mask().gt(0).unmask(0).reproject(proj30())
    bands = []
    for name, core in cores.items():
        ndvi = to_grid(comp.updateMask(core.gt(0.5)), region).rename(f"ndvi_{name}_{k:03d}")
        fclr = to_grid(clear.multiply(core), region).rename(f"fclr_{name}_{k:03d}")
        bands += [ndvi.toFloat(), fclr.unmask(0).toFloat()]
    return ee.Image.cat(bands)


def static_image(cores, region):
    return ee.Image.cat([to_grid(core, region).rename(f"fcrop_{n}") for n, core in cores.items()])


def chunk_images(year, region):
    """Yield (first bin index, image) for each CHUNK of 5-day bins."""
    cores = crop_cores(year, region)
    bins = C.time_bins(year)
    for c0 in range(0, len(bins), CHUNK):
        imgs = [bin_image(b.start.strftime("%Y-%m-%d"), b.end.strftime("%Y-%m-%d"), cores, region, k)
                for k, b in bins.iloc[c0:c0 + CHUNK].iterrows()]
        yield c0, ee.Image.cat(imgs)


def fetch(image, region):
    x0, y0, ncol, nrow, _ = C.grid(region)
    arr = ee.data.computePixels({
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": ncol, "height": nrow},
            "affineTransform": {"scaleX": C.RES, "shearX": 0, "translateX": x0,
                                "shearY": 0, "scaleY": -C.RES, "translateY": y0},
            "crsCode": C.CRS,
        },
    })
    out = {}
    for n in arr.dtype.names:
        a = np.asarray(arr[n], dtype=np.float32)
        out[n] = np.where(np.isfinite(a), a, np.nan)  # masked pixels come back as -inf
    return out


def save(out, year, region):
    bins = C.time_bins(year)
    T = len(bins)
    x0, y0, ncol, nrow, _ = C.grid(region)
    res = {"dates": bins["center"].values.astype("datetime64[D]"),
           "grid": np.array([x0, y0, ncol, nrow, C.RES])}
    for name in C.CROPS:
        res[f"fcrop_{name}"] = out[f"fcrop_{name}"]
        res[f"ndvi_{name}"] = np.stack([out[f"ndvi_{name}_{k:03d}"] for k in range(T)])
        res[f"fclr_{name}"] = np.stack([out[f"fclr_{name}_{k:03d}"] for k in range(T)])
    os.makedirs(C.DATA, exist_ok=True)
    path = C.ndvi_path(region, year)
    np.savez_compressed(path, **res)
    print("wrote", path)


def check_size(region):
    _, _, ncol, nrow, _ = C.grid(region)
    if ncol * nrow > MAX_DIRECT_CELLS:
        raise SystemExit(f"region '{region}' is {ncol}x{nrow} km; direct mode supports up to "
                         f"{MAX_DIRECT_CELLS} 1 km cells. Split it into county-sized regions in common.REGIONS.")


def save_boundary(region):
    geom = boundary(region).transform(ee.Projection(C.CRS), 1).getInfo()
    os.makedirs(C.DATA, exist_ok=True)
    with open(C.boundary_path(region), "w") as f:
        json.dump({"type": "Feature", "properties": {"region": region, "crs": C.CRS},
                   "geometry": geom}, f)


def cmd_direct(year, region):
    check_size(region)
    save_boundary(region)
    jobs = [(-1, static_image(crop_cores(year, region), region))] + list(chunk_images(year, region))
    out = {}

    def run(job):
        c0, img = job
        r = fetch(img, region)
        print("computed", "static" if c0 < 0 else f"chunk {c0}", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(run, jobs):
            out.update(r)
    save(out, year, region)


CHUNK30 = 4  # bins per 30 m request (computePixels responses are capped at 48 MB)


def grid30(region):
    """30 m window on the proj30() grid covering the region's 1 km window."""
    x0, y0, ncol, nrow, _ = C.grid(region)
    px0, py0 = PROJ30_TRANSFORM[2], PROJ30_TRANSFORM[5]
    c0 = int(np.floor((x0 - px0) / 30))
    r0 = int(np.floor((py0 - y0) / 30))
    c1 = int(np.ceil((x0 + ncol * C.RES - px0) / 30))
    r1 = int(np.ceil((py0 - (y0 - nrow * C.RES)) / 30))
    return px0 + c0 * 30, py0 - r0 * 30, c1 - c0, r1 - r0


def fetch30(image, region):
    x0, y0, ncol, nrow = grid30(region)
    arr = ee.data.computePixels({
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": ncol, "height": nrow},
            "affineTransform": {"scaleX": 30, "shearX": 0, "translateX": x0,
                                "shearY": 0, "scaleY": -30, "translateY": y0},
            "crsCode": C.CRS,
        },
    })
    out = {}
    for n in arr.dtype.names:
        a = np.asarray(arr[n], dtype=np.float32)
        out[n] = np.where(np.isfinite(a), a, np.nan)
    return out


def cmd_direct30(year, region):
    check_size(region)
    save_boundary(region)
    cores = crop_cores(year, region)
    label = ee.Image(0)
    for name, code in C.CROPS.items():
        label = label.where(cores[name].gt(0.5), code)
    label = label.reproject(proj30()).rename("label")
    is_crop = label.gt(0)

    bins = C.time_bins(year)
    jobs = [(-1, label.toFloat())]
    for c0 in range(0, len(bins), CHUNK30):
        imgs = []
        for k, b in bins.iloc[c0:c0 + CHUNK30].iterrows():
            col = ndvi_collection(b.start.strftime("%Y-%m-%d"), b.end.strftime("%Y-%m-%d"), rectangle(region))
            empty = ee.Image.constant(0).rename("NDVI").updateMask(0)
            comp = ee.Image(ee.Algorithms.If(col.size().gt(0), col.mean(), empty)).reproject(proj30())
            imgs.append(comp.updateMask(is_crop).toFloat().rename(f"ndvi_{k:03d}"))
        jobs.append((c0, ee.Image.cat(imgs)))

    def run(job):
        c0, img = job
        r = fetch30(img, region)
        print("computed", "label" if c0 < 0 else f"bins {c0}-{c0 + CHUNK30 - 1}", flush=True)
        return r

    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(run, jobs):
            out.update(r)
    x0, y0, ncol, nrow = grid30(region)
    res = {"dates": bins["center"].values.astype("datetime64[D]"),
           "grid": np.array([x0, y0, ncol, nrow, 30]),
           "label": np.nan_to_num(out["label"]).astype(np.uint8),
           "ndvi": np.stack([out[f"ndvi_{k:03d}"] for k in range(len(bins))])}
    os.makedirs(C.DATA, exist_ok=True)
    path = C.ndvi_path(region, year, "30m")
    np.savez_compressed(path, **res)
    print("wrote", path, {c: int((res["label"] == v).sum()) for c, v in C.CROPS.items()})


def cmd_add_region(state, county, name=None):
    """Look up a county in TIGER and save a window on the shared 1 km grid that covers it."""
    states = ee.FeatureCollection("TIGER/2018/States")
    st = state.strip()
    sf = (ee.Filter.eq("STATEFP", st.zfill(2)) if st.isdigit()
          else ee.Filter.Or(ee.Filter.eq("NAME", st), ee.Filter.eq("STUSPS", st.upper())))
    srec = states.filter(sf).first()
    if srec.getInfo() is None:
        raise SystemExit(f"state '{state}' not found in TIGER/2018/States")
    statefp, state_name = srec.get("STATEFP").getInfo(), srec.get("NAME").getInfo()

    co = county.strip()
    cf = ee.Filter.eq("COUNTYFP", co.zfill(3)) if co.isdigit() else ee.Filter.eq("NAME", co)
    counties = ee.FeatureCollection("TIGER/2018/Counties").filter(
        ee.Filter.And(ee.Filter.eq("STATEFP", statefp), cf))
    found = counties.aggregate_array("NAMELSAD").getInfo()
    if len(found) != 1:
        raise SystemExit(f"county '{county}' in {state_name}: expected 1 match, found {found}")
    rec = counties.first()
    countyfp, namelsad = rec.get("COUNTYFP").getInfo(), rec.get("NAMELSAD").getInfo()

    ring = rec.geometry().bounds(1, ee.Projection(C.CRS)).coordinates().getInfo()[0]
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    c0, c1 = int(np.floor(min(xs) / C.RES)), int(np.ceil(max(xs) / C.RES))
    r0, r1 = int(np.floor(min(ys) / C.RES)), int(np.ceil(max(ys) / C.RES))
    entry = dict(x0=c0 * C.RES, y0=r1 * C.RES, ncol=c1 - c0, nrow=r1 - r0,
                 statefp=statefp, countyfp=countyfp, label=f"{namelsad}, {state_name}")
    if name:
        key = name
    elif co.isdigit():
        key = f"{statefp}{countyfp}"
    else:
        key = co.lower().replace(" ", "_")
    if entry["ncol"] * entry["nrow"] > MAX_DIRECT_CELLS:
        print(f"warning: {namelsad} is {entry['ncol']}x{entry['nrow']} km, larger than direct mode supports")

    saved = {}
    if os.path.exists(C.REGIONS_FILE):
        with open(C.REGIONS_FILE) as f:
            saved = json.load(f)
    saved[key] = entry
    with open(C.REGIONS_FILE, "w") as f:
        json.dump(saved, f, indent=2)
    print(f"added region '{key}' to {os.path.basename(C.REGIONS_FILE)}: {entry}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["direct", "direct30", "add-region"])
    ap.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--region", default="boone", help=f"one of {sorted(C.REGIONS)} or one added with add-region")
    ap.add_argument("--state", help="add-region: state name, postal code or FIPS")
    ap.add_argument("--county", help="add-region: county name (e.g. Story) or FIPS")
    ap.add_argument("--name", help="add-region: region key (default: county name in lower case)")
    a = ap.parse_args()
    ee.Initialize(project=a.project)
    if a.cmd == "add-region":
        if not (a.state and a.county):
            ap.error("add-region needs --state and --county")
        cmd_add_region(a.state, a.county, a.name)
    else:
        if a.region not in C.REGIONS:
            ap.error(f"unknown region '{a.region}'; known: {sorted(C.REGIONS)}. Add one with add-region.")
        {"direct": cmd_direct, "direct30": cmd_direct30}[a.cmd](a.year, a.region)
