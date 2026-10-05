# Gridded crop stage and crop cycle products

This package makes **gridded** (30 m, 1 km and 9 km) maps of crop growth stage and crop cycle dates for corn and soybean. It runs the NASA Harvest / Agmatix models below on Sentinel-2 + Landsat NDVI from Google Earth Engine:

- [crop-cycle-detection](https://github.com/nasaharvest/crop-cycle-detection) finds each pixel's season start, green-up, NDVI peak and season end.
- [crop-stage-detection](https://github.com/nasaharvest/crop-stage-detection) assigns a weekly stage from A to E: bare soil, green-up, peak, senescence, post-harvest.

The models were published for single fields. This package applies them to every pixel of a grid and evaluates the stage model at each USDA NASS crop-progress week. It includes numba re-implementations of both models. They give identical results and are fast enough to run 1 million pixels in under a minute.

It has been run and checked for Boone County, Iowa (2023 and 2024) and Story County, Iowa (2024). Other counties can be added with one command (see [Adding a region](#adding-a-region)).

## Setup

Tested with Python 3.13, both with the package versions in `requirements.txt` and in a fresh virtual environment with the latest releases at the time (including pandas 3.0 and numpy 2.5). You also need a Google Cloud project registered for Earth Engine.

```bash
pip install -r requirements.txt
earthengine authenticate

# The two NASA Harvest repos, pinned to the versions this package was tested with
mkdir -p external && cd external
git clone https://github.com/nasaharvest/crop-cycle-detection && git -C crop-cycle-detection checkout 293f906
git clone https://github.com/nasaharvest/crop-stage-detection && git -C crop-stage-detection checkout 2005f53
cd ..
```

If you already have the repos somewhere else, set `HARVEST_REPOS` to the folder that contains both of them instead.

## Run

```bash
# 1. NDVI from Earth Engine (Boone County: about 1 min at 1 km, 4 min at 30 m)
python gee_ndvi.py direct   --project <your-ee-project> --year 2024 --region boone
python gee_ndvi.py direct30 --project <your-ee-project> --year 2024 --region boone

# 2. Both models on every pixel (Boone, all three grids, 1.09M pixels at 30 m: about 1 min on 11 cores)
python run_models.py --year 2024 --region boone --grid 9km 1km 30m

# 3. Figures
python plot_maps.py --year 2024 --region boone

# Optional: confirm the fast models match the original code on your data
python validate_fast_cycle.py --year 2024 --region boone --res 30m
python validate_fast_stage.py --year 2024 --region boone --res 30m
```

Use `--workers N` to set the number of CPU cores for `run_models.py`. It defaults to all but one.

## How it works

```
gee_ndvi.py ──► data/*.npz ──► run_models.py ──► products/<region>/<grid>/*.tif ──► plot_maps.py ──► figures/*.png
                                    │
                                    ├─ cycle: fast_cycle.py (default) or crop-cycle-detection (--cycle-impl original)
                                    └─ stage: fast_stage.py (default) or crop-stage-detection (--stage-impl original)
```

| File | Role |
|---|---|
| `common.py` | grids, regions, 5-day bins, NASS week dates, file paths |
| `gee_ndvi.py` | builds and downloads NDVI from Earth Engine |
| `run_models.py` | runs both models on every pixel and writes the GeoTIFFs |
| `fast_cycle.py` | numba version of the crop-cycle-detection model |
| `fast_stage.py` | numba version of the crop-stage-detection model |
| `validate_fast_cycle.py` | compares `fast_cycle.py` with the original code |
| `validate_fast_stage.py` | compares `fast_stage.py` with the original code |
| `plot_maps.py` | map and timeline figures |

### NDVI

The NDVI recipe follows `gee_fetch.py` in crop-stage-detection:

- **Sentinel-2:** `COPERNICUS/S2_SR_HARMONIZED`, 10 m, B8/B4. Pixels are kept where Cloud Score+ `cs` > 0.6.
- **Landsat 8 and 9:** Collection 2 Level-2, 30 m, SR_B5/SR_B4. Pixels are kept where `QA_PIXEL` is 21824 or 21888 (clear).
- **Crop pixels:** USDA CDL of the same year (corn = 1, soybean = 5), eroded by one CDL pixel to drop field edges. CDL is 10 m from 2024 on and 30 m before.

All clear observations from the three sensors are averaged per 30 m pixel in **5-day bins**. The bins run from Nov 1 of the previous year to Jan 31 of the next year, so the cycle model can find the winter lows on both sides of the season.

- **1 km:** the mean NDVI of each crop's 30 m pixels inside the cell. A bin is used only if at least 50% of the cell's crop pixels are clear, and a cell needs at least 5% crop cover.
- **9 km:** a weighted mean of the 1 km cells on the USDA NASS gridded Crop Progress & Condition grid. Pass `--cpc-ref <file.tif>` to align with a specific CPC file.

Pixels with fewer than 10 usable bins are skipped.

### Differences from the published workflow

- Harvest averages NDVI over field polygons and keeps each satellite pass, preferring Sentinel-2. This package uses grid pixels and 5-day mean composites with no cross-sensor adjustment.
- Sentinel-2 is resampled to 30 m before use.
- The stage model is evaluated at each NASS week-ending Sunday (ISO weeks 14–47), using only bins complete by that date and a 150-day lookback (the model default).

## Outputs

Everything is written to `products/<region>/<grid>/` as GeoTIFFs in EPSG:5070 (NAD83 / CONUS Albers).

**`cycle_<crop>_<year>.tif`** (float32, NaN = not modeled)

| Band | Meaning |
|---|---|
| `season_start_doy` | NDVI starts rising (after planting) |
| `greenup_doy` | fastest green-up |
| `ndvi_peak_doy` | date of peak NDVI |
| `ndvi_peak_value` | peak NDVI |
| `season_end_doy` | NDVI back down after senescence or harvest |
| `season_length` | end minus start, in days |
| `n_seasons` | number of seasons with a peak in Apr–Oct of `<year>` |

Dates are day of year counted from Jan 1 of `<year>`. If a pixel has more than one season, bands 1–6 describe the one with the highest peak.

**`stage_<crop>_<year>_weekly.tif`** (uint8) has one band per NASS week, named like `w30_2024-07-28`. Values: 1–5 = stage A–E, 0 = insufficient data, 255 = not a crop pixel.

**`stage_<crop>_<year>_summary.csv`** gives the share of crop pixels in each stage per week.

## The fast models

Both original models spend most of their time on pandas and sparse-solver overhead for about 150 values per pixel. In a single process on an Apple M4 Pro, the stage model took about 95 ms per pixel (it is run 34 times, once per week) and the cycle model about 11–19 ms. `fast_cycle.py` and `fast_stage.py` re-implement the same steps as compiled numba code:

- **Shared smoothing:** daily resampling, PCHIP gap filling (using scipy's slope formulas) and Whittaker smoothing with λ = 6000, solved as a banded system.
- **Cycle:** the local-regression outlier and spike removal, scipy's `argrelmax`/`argrelmin` edge rules, trough-to-trough seasons, the start / end / green-up rules, duplicate-season removal and overlap clipping.
- **Stage:** the constant-velocity Kalman trend and the threshold and peak rules for A–E, for all 34 weeks in one call.

Each takes about 0.1–0.16 ms per pixel. Their default parameters are fixed; use `--cycle-impl original` or `--stage-impl original` if you need different ones.

Validation on Boone County 2024:

| | Checked | Mismatches |
|---|---|---|
| Cycle, all 7 outputs | all 2,581 series at 1 km and 20,000 random pixels at 30 m | 0 |
| Stage | 87,754 pixel-weeks at 1 km and 102,000 at 30 m | 0 |
| Smoothed NDVI | 200 series | agrees within 1e-12 |
| Products | all 18 files at 9 km, 1 km and 30 m | identical values to the original code |

The full Boone run at 30 m went from about 31 minutes with the original code to under a minute. Re-run the two validation scripts after updating either NASA Harvest repo.

## Adding a region

```bash
python gee_ndvi.py add-region --project <your-ee-project> --state Iowa --county Story
```

This looks up the county in the US Census TIGER boundaries, fits a window on the shared 1 km grid around it, and saves it to `regions.json` under the county's name in lower case (here `story`). After that, use `--region story` with every script. Names, postal codes (`IA`) and FIPS codes (`--state 19 --county 169`) all work, and `--name` sets a different key.

Direct mode supports up to 100 × 100 km, so a whole state needs to be run county by county.

## Known limitations

- **Region size:** only county-sized regions work. Statewide runs need to be tiled.
- **Validation:** the models were validated by their authors on about 53,000 fields in North Carolina, not in Iowa. These products have not yet been compared with NASS crop progress.
- **Stage C:** the stage model sets its peak threshold from the NDVI seen so far, so stage C means "near the highest NDVI so far," not physiological maturity.
- **Season start:** this marks the NDVI rise, which typically comes 2–3 weeks after planting.
- **Snow:** snow passes both cloud masks and shows up as winter NDVI near 0.
- **Statewide 30 m:** the models would take roughly an hour for Iowa, but the NDVI download is county by county (about 4 minutes and 275 MB per county).

## Authors

Meijian Yang, Columbia University & NASA GISS: gridded pipeline, NDVI processing and the numba re-implementations (`fast_cycle.py`, `fast_stage.py`).

The crop cycle and crop stage models are by Ran Pelta (Agmatix / GrowersTech, NASA Harvest).

## Citation and license

Please cite the models:

- Pelta, R. (2026). *Crop Cycle Detection*. NASA Harvest / Agmatix. https://github.com/nasaharvest/crop-cycle-detection
- Pelta, R. (2026). *Crop Stage Detection*. NASA Harvest / Agmatix. https://github.com/nasaharvest/crop-stage-detection

Licensed under Apache 2.0 (see `LICENSE`). `fast_cycle.py` and `fast_stage.py` are modified re-implementations of crop-cycle-detection and crop-stage-detection; see `NOTICE` for attribution.
